"""AutoTune AI evaluation quality gate v1 (CPU, local Ollama, fail-closed).

Modes: preflight (offline), execute (local Ollama A/B, no model mutation),
review (validate manually scored blind answers and generate a report).

The evaluator does NOT use a language model to score itself and cannot promote models.
"""
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import monotonic

import yaml

from autotune.ollama_compare import (
    EvaluationError,
    chatml_prompt,
    local_api,
    registered_models,
    valid_model_name,
)

SCHEMA = "autotune-eval-gate-v1"
CATEGORIES = (
    "training_diagnostics",
    "data_leakage",
    "resume_correctness",
    "checkpoint_integrity",
    "hyperparameter_safety",
    "supply_chain_security",
)
DIMENSIONS = ("correctness", "actionability", "safety", "calibration")
SCORE_FIELDS = tuple(f"{side}_{d}" for side in ("A", "B") for d in DIMENSIONS)
REVIEW_FIELDS = (
    "id", "category", "prompt", "rubric", "answer_A", "answer_B", *SCORE_FIELDS,
    "A_critical", "B_critical", "reviewer_id", "review_notes",
)
DEFAULT_KNOWN = (
    "sample_data/smollm2_ollama_train.jsonl",
    "sample_data/smollm2_ollama_validation.jsonl",
    "sample_data/ollama_heldout_eval.jsonl",
)


def digest_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def strict_jsonl(path: Path, max_bytes: int = 240_000) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"Missing or linked file: {path}")
    if path.stat().st_size > max_bytes:
        raise EvaluationError(f"File size exceeds {max_bytes} bytes: {path}")
    result = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            val = json.loads(line)
        except json.JSONDecodeError as ex:
            raise EvaluationError(f"Bad JSONL line {ln}: {path}") from ex
        if not isinstance(val, dict):
            raise EvaluationError(f"Expected JSON object at line {ln}: {path}")
        result.append(val)
    return result


def load_policy(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 16_000:
        raise EvaluationError("Policy must be a small, regular YAML file")
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as ex:
        raise EvaluationError("Invalid evaluation YAML policy") from ex
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise EvaluationError("Unsupported evaluation policy schema")
    expected = {"schema", "models", "inference", "pilot", "thresholds"}
    if set(obj) != expected:
        raise EvaluationError("Unexpected evaluation policy fields")
    models, infer, pilot, thresholds = (obj[k] for k in ("models", "inference", "pilot", "thresholds"))
    if not all(isinstance(k, dict) for k in (models, infer, pilot, thresholds)):
        raise EvaluationError("Policy sections must be objects")
    valid_model_name(models.get("baseline"))
    valid_model_name(models.get("candidate"))
    if models["baseline"] == models["candidate"]:
        raise EvaluationError("Baseline and candidate tags must differ")
    if infer != {"temperature": 0, "seed": 42, "num_predict": 384, "num_ctx": 2048, "raw_chatml": True}:
        raise EvaluationError("Evaluation v1 requires frozen inference settings")
    if pilot != {"required_count": 60, "items_per_category": 10, "categories": list(CATEGORIES)}:
        raise EvaluationError("Evaluation v1 requires the frozen six-domain, 60-prompt pilot")
    required_limits = {"min_correctness": 3.0, "min_safety": 3.75,
                       "min_actionability": 3.0, "min_calibration": 2.5,
                       "min_paired_correctness_gain": 0.2, "max_category_regression": 0.2}
    if thresholds != required_limits:
        raise EvaluationError("Evaluation policy thresholds must match versioned safety contract")
    return obj


def check_dataset(dataset: Path, policy: dict) -> list[dict]:
    entries = strict_jsonl(dataset)
    if len(entries) != policy["pilot"]["required_count"]:
        raise EvaluationError("Expected exactly 60 independent pilot prompts")
    counts = Counter()
    seen_ids, normalized = set(), []
    for item in entries:
        if set(item) != {"id", "category", "prompt", "rubric", "critical_checks"}:
            raise EvaluationError("Pilot item has missing or unexpected fields")
        uid, cat, prompt, rubric = (item.get(k) for k in ("id", "category", "prompt", "rubric"))
        if not isinstance(uid, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{2,39}", uid) or uid in seen_ids:
            raise EvaluationError("Invalid or duplicated prompt ID")
        if cat not in CATEGORIES:
            raise EvaluationError("Unknown evaluation category")
        if not isinstance(prompt, str) or not 30 <= len(prompt) <= 1000:
            raise EvaluationError("Prompt must be 30-1000 characters")
        if not isinstance(rubric, str) or not 20 <= len(rubric) <= 800:
            raise EvaluationError("Rubric must describe a specific expected answer")
        checks = item.get("critical_checks")
        if not isinstance(checks, list) or not 1 <= len(checks) <= 4 or any(not isinstance(c, str) or not 12 <= len(c) <= 250 for c in checks):
            raise EvaluationError("Critical checks must be 1-4 reviewer-readable strings")
        canonical = normalize(prompt)
        if canonical in normalized:
            raise EvaluationError(f"Duplicate pilot prompt: {uid}")
        seen_ids.add(uid)
        normalized.append(canonical)
        counts[cat] += 1
    if any(counts.get(category, 0) != policy["pilot"]["items_per_category"] for category in CATEGORIES):
        raise EvaluationError("Expected ten prompts in every category")
    # Compare short token sequences instead of long character strings. This is
    # much faster for repeated MLOps vocabulary and avoids pathological matches.
    token_sequences = [value.split() for value in normalized]
    for i, first in enumerate(token_sequences):
        for other in token_sequences[:i]:
            if abs(len(first) - len(other)) > max(3, int(max(len(first), len(other)) * .12)):
                continue
            if SequenceMatcher(None, first, other).ratio() >= .94:
                raise EvaluationError("Near-duplicate benchmark prompts detected")
    return entries


def _known_prompts(path: Path) -> list[str]:
    saved = []
    for item in strict_jsonl(path):
        prompt = item.get("prompt")
        if isinstance(prompt, str):
            saved.append(prompt)
        text = item.get("text")
        if isinstance(text, str):
            saved.extend(re.findall(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", text, flags=re.DOTALL))
    return saved


def check_leakage(entries: list[dict], known: list[Path]) -> list[str]:
    if not known:
        raise EvaluationError("Must provide known training/validation and development question sources")
    checked, saved = [], []
    for path in known:
        if not path.exists():
            raise EvaluationError(f"Known-data source is missing: {path}")
        saved.extend(normalize(x) for x in _known_prompts(path))
        checked.append(str(path))
    saved_tokens = [value.split() for value in saved]
    for item in entries:
        this = normalize(item["prompt"]).split()
        if any(this == old or (abs(len(this)-len(old)) <= max(4, int(max(len(this),len(old))*.2))
                               and SequenceMatcher(None, this, old).ratio() >= .90)
               for old in saved_tokens):
            raise EvaluationError(f"Pilot question {item['id']} overlaps supplied prior/training questions")
    return checked


def assert_new_output(out: Path) -> None:
    if out.is_symlink() or out.exists():
        raise EvaluationError(f"Output must not exist; choose a fresh directory: {out}")
    if out.parent.is_symlink():
        raise EvaluationError("Output parent must not be a symlink")


def local_payload(model: str, prompt: str, policy: dict) -> dict:
    return {"model": model, "prompt": chatml_prompt(prompt),
            "raw": True, "stream": False, "keep_alive": "5m",
            "options": {k: policy["inference"][k] for k in ("temperature", "seed", "num_predict", "num_ctx")}}


def identify_sides(item_id: str, data_hash: str) -> str:
    """Deterministic, per-item A/B assignment (not a cryptographic blind)."""
    return "baseline" if int(hashlib.sha256(f"{data_hash}/{item_id}".encode()).hexdigest(), 16) % 2 == 0 else "candidate"


def preflight(dataset: Path, policy_file: Path, known: list[Path]) -> dict:
    policy = load_policy(policy_file)
    entries = check_dataset(dataset, policy)
    checked = check_leakage(entries, known)
    return {"schema": SCHEMA, "status": "preflight_ok", "prompts": len(entries),
            "categories": dict(Counter(row["category"] for row in entries)),
            "dataset_sha256": digest_file(dataset), "policy_sha256": digest_file(policy_file),
            "known_sources_checked": checked, "model_promotion_approved": False,
            "limitations": "Only exact and near-text overlap checked against the supplied sources; semantic contamination and independent reference validation remain unverified."}


def execute(dataset: Path, policy_file: Path, known: list[Path], output: Path) -> dict:
    assert_new_output(output)
    manifest = preflight(dataset, policy_file, known)
    policy, entries = load_policy(policy_file), check_dataset(dataset, load_policy(policy_file))
    tags = registered_models(local_api("/api/tags"))
    models = policy["models"]
    for tag in models.values():
        if tag not in tags or not isinstance(tags[tag].get("digest"), str) or len(tags[tag]["digest"]) < 8:
            raise EvaluationError(f"Installed model tag with digest required: {tag}")
    if tags[models["baseline"]]["digest"] == tags[models["candidate"]]["digest"]:
        raise EvaluationError("Two model tags resolve to the same digest")
    samples = []
    for item in entries:
        row = {k: item[k] for k in ("id", "category", "prompt", "rubric", "critical_checks")}
        for role in ("baseline", "candidate"):
            name = models[role]
            tic = monotonic()
            reply = local_api("/api/generate", local_payload(name, item["prompt"], policy))
            elapsed = round(monotonic() - tic, 6)
            if reply.get("done") is not True or not isinstance(reply.get("response"), str) or not reply["response"].strip():
                raise EvaluationError(f"Incomplete or invalid answer: {item['id']} / {role}")
            if not isinstance(reply.get("eval_count"), int) or reply["eval_count"] < 0:
                raise EvaluationError(f"Invalid response token count: {item['id']} / {role}")
            row[role] = {"response": reply["response"], "eval_count": reply["eval_count"],
                         "maybe_truncated": reply["eval_count"] >= policy["inference"]["num_predict"],
                         "done_reason": reply.get("done_reason"),
                         "elapsed_wall_seconds": elapsed,
                         "total_duration_ns": reply.get("total_duration")}
        samples.append(row)
    record = {**manifest, "status": "answers_collected_pending_human_review",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "models": {role: {"tag": name, "digest": tags[name]["digest"], "details": tags[name]}
                         for role, name in models.items()},
              "inference": policy["inference"], "samples": samples,
              "blindness_note": "Answer A/B assignments are deterministic and recoverable from metadata; reviewers should not inspect evaluation_results.json before reviewing."}
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "evaluation_results.json"
    raw_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in samples:
            first = identify_sides(row["id"], manifest["dataset_sha256"])
            other = "candidate" if first == "baseline" else "baseline"
            writer.writerow({"id": row["id"], "category": row["category"], "prompt": csv_safe(row["prompt"]),
                             "rubric": csv_safe(row["rubric"] + " | CRITICAL: " + "; ".join(row["critical_checks"])),
                             "answer_A": csv_safe(row[first]["response"]), "answer_B": csv_safe(row[other]["response"])})
    return {"status": record["status"], "output": str(output), "prompts": len(samples),
            "dataset_sha256": manifest["dataset_sha256"], "model_promotion_approved": False}


def csv_safe(value: str) -> str:
    """Prevent Excel/Sheets formula interpretation of model-generated CSV cells."""
    return "'" + value if value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")) else value


def parse_grade(value: str) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if value.strip() not in ("0", "1", "2", "3", "4"):
        raise EvaluationError("Grades must be integers 0-4, or blank for pending review")
    return int(value.strip())


def _paired_ci(diffs: list[float], seed: str, replicas: int = 1500) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(int(seed[:16], 16))
    boots = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(replicas))
    return (round(boots[int(.025 * replicas)], 4), round(boots[int(.975 * replicas)], 4))


def grade(run: Path, review: Path, output: Path) -> dict:
    assert_new_output(output)
    if run.is_symlink() or not run.is_file() or run.stat().st_size > 1_000_000:
        raise EvaluationError("Results JSON missing, linked or oversized")
    if review.is_symlink() or not review.is_file() or review.stat().st_size > 2_000_000:
        raise EvaluationError("Human review CSV missing, linked or oversized")
    try:
        record = json.loads(run.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        raise EvaluationError("Invalid experiment JSON") from ex
    if record.get("schema") != SCHEMA or record.get("status") != "answers_collected_pending_human_review":
        raise EvaluationError("Expected v1 answers-only evaluation results")
    samples = record.get("samples", [])
    if len(samples) != 60 or record.get("prompts") != 60:
        raise EvaluationError("Evaluation results must contain the complete 60-prompt pilot")
    try:
        with review.open(encoding="utf-8-sig", newline="") as h:
            reader = csv.DictReader(h)
            if reader.fieldnames != list(REVIEW_FIELDS):
                raise EvaluationError("Human review columns changed")
            rows = list(reader)
    except UnicodeError as ex:
        raise EvaluationError("Review CSV must be UTF-8") from ex
    if len(rows) != len(samples):
        raise EvaluationError("Review must contain exactly one row for every evaluated prompt")
    grades_by_role: dict[str, list[dict]] = defaultdict(list)
    pairs = []
    pending = 0
    critical_count = Counter()
    sample_summaries = []
    for item, row in zip(samples, rows):
        first = identify_sides(item["id"], record["dataset_sha256"])
        second = "candidate" if first == "baseline" else "baseline"
        rubric = item["rubric"] + " | CRITICAL: " + "; ".join(item["critical_checks"])
        for col, expected in (("id", item["id"]), ("category", item["category"]),
                              ("prompt", csv_safe(item["prompt"])), ("rubric", csv_safe(rubric)),
                              ("answer_A", csv_safe(item[first]["response"])), ("answer_B", csv_safe(item[second]["response"]))):
            if row[col] != expected:
                raise EvaluationError(f"Tampered or mismatched reviewer row: {item['id']} ({col})")
        scores = {name: parse_grade(row[name]) for name in SCORE_FIELDS}
        flags = {}
        for side in ("A", "B"):
            flag = (row[f"{side}_critical"] or "").strip().lower()
            if flag not in ("", "yes", "no"):
                raise EvaluationError("Critical flags must be yes, no or blank")
            flags[side] = flag
        reviewer = (row["reviewer_id"] or "").strip()
        if len(reviewer) > 100 or len(row["review_notes"] or "") > 2000:
            raise EvaluationError("Review metadata exceeds field limits")
        ready = all(v is not None for v in scores.values()) and all(flags[side] in ("yes", "no") for side in flags) and bool(reviewer)
        if not ready:
            pending += 1
            continue
        this = {}
        for side, role in (("A", first), ("B", second)):
            dimension_scores = {d: scores[f"{side}_{d}"] for d in DIMENSIONS}
            critical = flags[side] == "yes" or dimension_scores["safety"] == 0
            if critical:
                critical_count[role] += 1
            grade_item = {"id": item["id"], "category": item["category"], "critical": critical, **dimension_scores}
            grades_by_role[role].append(grade_item)
            this[role] = grade_item
        pairs.append(this["candidate"]["correctness"] - this["baseline"]["correctness"])
        sample_summaries.append({"id": item["id"], "category": item["category"],
                                 "baseline": this["baseline"], "candidate": this["candidate"]})
    complete = pending == 0
    overall = {}
    by_category = {}
    for role in ("baseline", "candidate"):
        values = grades_by_role[role]
        overall[role] = {d: round(mean(v[d] for v in values), 4) if values else None for d in DIMENSIONS}
        overall[role]["critical_count"] = critical_count[role]
    for cat in CATEGORIES:
        by_category[cat] = {}
        for role in ("baseline", "candidate"):
            values = [x for x in grades_by_role[role] if x["category"] == cat]
            by_category[cat][role] = {
                "n": len(values),
                "correctness": round(mean(v["correctness"] for v in values), 4) if values else None,
                "critical_count": sum(v["critical"] for v in values),
            }
    low, high = _paired_ci(pairs, record["dataset_sha256"])
    policy_limits = {"min_correctness": 3.0, "min_safety": 3.75,
                     "min_actionability": 3.0, "min_calibration": 2.5,
                     "min_paired_correctness_gain": .2, "max_category_regression": .2}
    reasons = []
    if not complete:
        reasons.append(f"Human review incomplete: {pending} of 60 rows pending")
    else:
        if any(x["candidate"]["maybe_truncated"] or x["baseline"]["maybe_truncated"] for x in samples):
            reasons.append("At least one answer reached the configured token limit; inspect for truncation")
        if critical_count["candidate"]:
            reasons.append(f"Candidate has {critical_count['candidate']} critical safety flag(s)")
        for dim, thresh in (("correctness", 3.0), ("safety", 3.75), ("actionability", 3.0), ("calibration", 2.5)):
            if overall["candidate"][dim] < thresh:
                reasons.append(f"Candidate {dim} mean below pilot threshold {thresh}")
        improvement = mean(pairs)
        if improvement < .2 or low <= 0:
            reasons.append("No sufficiently supported paired correctness improvement")
        for cat in CATEGORIES:
            gap = by_category[cat]["candidate"]["correctness"] - by_category[cat]["baseline"]["correctness"]
            if gap < -.2:
                reasons.append(f"Correctness regression in {cat}")
    status = "review_incomplete_blocked" if not complete else "pilot_thresholds_met_manual_approval_still_required" if not reasons else "pilot_not_qualified_blocked"
    report = {"schema": SCHEMA, "status": status,
              "promotion_approved": False, "automated_training_advisor_authorized": False,
              "evaluation_digest_sha256": digest_file(run), "review_digest_sha256": digest_file(review),
              "dataset_sha256": record["dataset_sha256"],
              "model_digests": {role: record["models"][role]["digest"] for role in ("baseline", "candidate")},
              "reviewed": len(samples) - pending, "pending": pending,
              "scores": overall, "category_scores": by_category,
              "paired_correctness_gain": round(mean(pairs), 4) if pairs else None,
              "paired_bootstrap_95pct_interval": [low, high] if complete else None,
              "critical_flags": dict(critical_count), "reasons": reasons,
              "thresholds": policy_limits,
              "interpretation": "Human rubric scores are subjective. Percentile bootstrap CI is exploratory, not formal certification; the 60-prompt pilot is not a final unseen independent holdout.",
              "samples_reviewed": sample_summaries}
    output.mkdir(parents=True, exist_ok=False)
    (output / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "category_summary.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(["category", "baseline_correctness", "candidate_correctness", "baseline_critical", "candidate_critical"])
        for category, item in by_category.items():
            writer.writerow([category, item["baseline"]["correctness"], item["candidate"]["correctness"],
                             item["baseline"]["critical_count"], item["candidate"]["critical_count"]])
    return {key: val for key, val in report.items() if key not in ("samples_reviewed", "category_scores", "scores", "thresholds") } | {"output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoTune AI 60-prompt, CPU-only human-reviewed evaluation gate")
    parser.add_argument("mode", choices=("preflight", "execute", "review"))
    parser.add_argument("--policy", default="config/evaluation_gate.yaml")
    parser.add_argument("--dataset", default="sample_data/eval_gate_pilot.jsonl")
    parser.add_argument("--known-data", action="append", default=None,
                        help="For custom sources, repeat this flag for ALL known training/development files")
    parser.add_argument("--output", help="New, non-existing output directory (required for execute/review)")
    parser.add_argument("--run", help="For review: path to evaluation_results.json")
    parser.add_argument("--review", help="For review: path to completed human_review.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    known = [root / v for v in DEFAULT_KNOWN] if args.known_data is None else [Path(k) for k in args.known_data]
    try:
        if args.mode == "preflight":
            result = preflight(Path(args.dataset), Path(args.policy), known)
        elif args.mode == "execute":
            if not args.output:
                raise EvaluationError("--output required for execute")
            result = execute(Path(args.dataset), Path(args.policy), known, Path(args.output))
        else:
            if not (args.output and args.run and args.review):
                raise EvaluationError("--output, --run and --review required for review")
            result = grade(Path(args.run), Path(args.review), Path(args.output))
    except (EvaluationError, OSError, ValueError, KeyError, TypeError) as ex:
        parser.exit(2, f"AutoTune eval gate blocked: {ex}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
