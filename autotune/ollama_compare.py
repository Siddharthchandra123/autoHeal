"""CPU-only, read-only Ollama baseline-vs-adapter evaluation.

Never loads or modifies model weights. It talks exclusively to local Ollama.
Only the user may decide whether a candidate is better / ready for promotion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_LOCAL = "http://127.0.0.1:11434"
MODEL_RE = re.compile(r"^(?:[a-zA-Z0-9][a-zA-Z0-9._-]*/)?[a-zA-Z0-9][a-zA-Z0-9._-]*(?::[a-zA-Z0-9][a-zA-Z0-9._-]*)?$")
USER_MESSAGE_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
SCHEMA = "autotune-ollama-heldout-v0.9.8"


class EvaluationError(RuntimeError):
    pass


def valid_model_name(name: str) -> str:
    if not isinstance(name, str) or not MODEL_RE.fullmatch(name) or ".." in name:
        raise EvaluationError("Invalid Ollama model name; use an exact local Ollama tag")
    return name


def local_api(path: str, payload: dict | None = None, timeout: int = 120) -> dict:
    if path not in ("/api/tags", "/api/generate"):
        raise EvaluationError("Ollama API endpoint not allowlisted")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        OLLAMA_LOCAL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError) as ex:
        raise EvaluationError(f"Local Ollama request failed ({path}): {ex}") from ex
    except (json.JSONDecodeError, UnicodeDecodeError) as ex:
        raise EvaluationError("Local Ollama returned invalid JSON") from ex
    if not isinstance(result, dict):
        raise EvaluationError("Local Ollama response was not an object")
    return result


def read_prompts(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise EvaluationError("Held-out evaluation JSONL file not found or is a symlink")
    if path.stat().st_size > 64_000:
        raise EvaluationError("Held-out file exceeds 64 KB safety cap")
    entries = []
    names = set()
    prompts = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as ex:
            raise EvaluationError(f"Invalid evaluation JSON at line {lineno}") from ex
        if not isinstance(item, dict):
            raise EvaluationError("Evaluation item must be an object")
        identifier, prompt = item.get("id"), item.get("prompt")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", identifier):
            raise EvaluationError(f"Invalid evaluation item ID at line {lineno}")
        if not isinstance(prompt, str) or not 5 <= len(prompt) <= 600:
            raise EvaluationError(f"Invalid evaluation prompt at line {lineno}")
        if identifier in names or prompt.strip().casefold() in prompts:
            raise EvaluationError("Duplicate evaluation ID or prompt")
        names.add(identifier)
        prompts.add(prompt.strip().casefold())
        entries.append({"id": identifier, "prompt": prompt, "rubric": str(item.get("rubric", ""))[:240]})
    if not 2 <= len(entries) <= 64:
        raise EvaluationError("Held-out evaluation must contain 2-64 prompts")
    return entries


def verify_no_exact_leakage(entries: list[dict], known_files: list[Path]) -> list[str]:
    """Checks exact question overlap, not semantic similarity or all possible leakage."""
    existing = [path for path in known_files if path.is_file()]
    known = set()
    for path in existing:
        if path.is_symlink():
            raise EvaluationError("Known training or validation source is a symlink")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as ex:
                raise EvaluationError(f"Invalid known-data JSONL: {path}") from ex
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("prompt"), str):
                known.add(value["prompt"].strip().casefold())
            for question in USER_MESSAGE_RE.findall(str(value.get("text", ""))):
                known.add(question.strip().casefold())
    leaked = [entry["id"] for entry in entries if entry["prompt"].strip().casefold() in known]
    if leaked:
        raise EvaluationError(f"Evaluation prompts overlap supplied known data: {', '.join(leaked)}")
    return [str(path) for path in existing]


def registered_models(tags: dict) -> dict[str, dict]:
    items = tags.get("models")
    if not isinstance(items, list):
        raise EvaluationError("Local Ollama tag response missing 'models'")
    output = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            output[item["name"]] = {
                "digest": item.get("digest"),
                "size": item.get("size"),
                "format": (item.get("details") or {}).get("format") if isinstance(item.get("details"), dict) else None,
            }
    return output


def chatml_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful AI assistant. Answer the question concisely."
        "<|im_end|>\n<|im_start|>user\n"
        + question +
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def generation_payload(model: str, prompt: str) -> dict:
    return {
        "model": model, "prompt": chatml_prompt(prompt),
        "raw": True, "stream": False, "keep_alive": "5m",
        "options": {"temperature": 0, "seed": 42, "num_predict": 128, "num_ctx": 2048},
    }


def run_evaluation(*, baseline: str, candidate: str, dataset: Path, output: Path,
                   known_data: list[Path] | None = None, execute: bool = False) -> dict:
    baseline, candidate = valid_model_name(baseline), valid_model_name(candidate)
    if baseline == candidate:
        raise EvaluationError("Baseline and candidate must be different Ollama tags")
    entries = read_prompts(dataset)
    checked = verify_no_exact_leakage(entries, known_data or [])
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise EvaluationError("Output path is not empty: choose a fresh evaluation folder")
    config = {
        "schema": SCHEMA, "ok": True, "mode": "READ-ONLY OLLAMA COMPARISON PREFLIGHT" if not execute else "OLLAMA COMPARISON",
        "baseline": baseline, "candidate": candidate, "num_prompts": len(entries),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "exact_overlap_sources_checked": checked,
        "settings": {"temperature": 0, "seed": 42, "num_predict": 128, "num_ctx": 2048, "raw_chatml": True},
        "limits": "Tiny held-out set, manual review required; scores are not model-quality or security proof",
        "model_promotion_approved": False,
    }
    if not execute:
        return config
    models = registered_models(local_api("/api/tags"))
    missing = [name for name in (baseline, candidate) if name not in models]
    if missing:
        raise EvaluationError("Missing local Ollama model(s): " + ", ".join(missing))
    config["model_metadata"] = {baseline: models[baseline], candidate: models[candidate]}
    config["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    results = {entry["id"]: {"id": entry["id"], "prompt": entry["prompt"], "rubric": entry["rubric"]} for entry in entries}
    for model, role in [(baseline, "baseline"), (candidate, "candidate")]:
        for entry in entries:
            reply = local_api("/api/generate", generation_payload(model, entry["prompt"]))
            if not reply.get("done") or not isinstance(reply.get("response"), str):
                raise EvaluationError(f"Invalid or incomplete generation from {model} on {entry['id']}")
            results[entry["id"]][role] = {
                "response": reply["response"],
                "eval_count": reply.get("eval_count"),
                "total_duration_ns": reply.get("total_duration"),
                "eval_duration_ns": reply.get("eval_duration"),
                "load_duration_ns": reply.get("load_duration"),
            }
    config["results"] = list(results.values())
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_results.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = ["id", "prompt", "rubric", "baseline_response", "candidate_response", "baseline_eval_count", "candidate_eval_count", "review_winner", "review_notes"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in config["results"]:
            writer.writerow({
                "id": result["id"], "prompt": result["prompt"], "rubric": result["rubric"],
                "baseline_response": result["baseline"]["response"],
                "candidate_response": result["candidate"]["response"],
                "baseline_eval_count": result["baseline"]["eval_count"],
                "candidate_eval_count": result["candidate"]["eval_count"],
                "review_winner": "", "review_notes": "",
            })
    return {key: value for key, value in config.items() if key != "results"} | {"output": str(output), "prompts_completed": len(entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a baseline and fine-tuned SmolLM2 through local Ollama only")
    parser.add_argument("--baseline", default="smollm2:135m-instruct-fp16")
    parser.add_argument("--candidate", default="autotune-smollm2:gguf-v095")
    parser.add_argument("--dataset", default="sample_data/ollama_heldout_eval.jsonl")
    parser.add_argument("--output", default="outputs/ollama-comparison-01")
    parser.add_argument("--known-data", action="append", default=None, help="Training/validation JSONL to check for exact question overlap")
    parser.add_argument("--execute", action="store_true", help="Call the local Ollama API; default is offline preflight")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    defaults = [repo / "sample_data" / "smollm2_ollama_train.jsonl", repo / "sample_data" / "smollm2_ollama_validation.jsonl"]
    try:
        result = run_evaluation(
            baseline=args.baseline, candidate=args.candidate,
            dataset=Path(args.dataset), output=Path(args.output),
            known_data=[Path(p) for p in args.known_data] if args.known_data is not None else defaults,
            execute=args.execute,
        )
    except EvaluationError as ex:
        parser.exit(2, f"AutoTune Ollama comparison stopped: {ex}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
