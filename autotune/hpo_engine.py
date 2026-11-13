import json, math, os, traceback
from datetime import datetime, timezone
from pathlib import Path
import yaml
from .config import config_sha256, load_config
from .hpo_contract import validate_trial_config
from .hpo_search import build_candidates
from .lora_contract import make_plan, verify_manifest
from .lora_gpu import run_lora
from .hf_mlflow import HFMLflowReporter

def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

def _event(path, event, **kwargs):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                            "event": event, **kwargs}, sort_keys=True) + "\n")


def _read_evaluation_evidence(path, min_evaluations):
    """Validate paired train/validation observations for HPO eligibility."""
    rows = []
    previous_step = 0

    if not path.is_file():
        return {
            "qualified": False,
            "evaluation_count": 0,
            "required_evaluations": min_evaluations,
            "reason": "metrics.jsonl is missing",
        }

    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metrics.jsonl line {line_number}") from exc

        if not isinstance(row, dict):
            raise ValueError(f"Metric line {line_number} must be an object")

        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"Invalid evaluation step at line {line_number}")
        if step <= previous_step:
            raise ValueError("Evaluation steps must be strictly increasing")

        for name in ("train_loss", "val_loss"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"Invalid {name} at metrics.jsonl line {line_number}")

        rows.append({
            "step": step,
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
        })
        previous_step = step

    count = len(rows)
    latest = rows[-1] if rows else None

    return {
        "qualified": count >= min_evaluations,
        "evaluation_count": count,
        "required_evaluations": min_evaluations,
        "best_val_loss": min((r["val_loss"] for r in rows), default=None),
        "latest_train_loss": latest["train_loss"] if latest else None,
        "latest_val_loss": latest["val_loss"] if latest else None,
        "latest_generalization_gap": (
            latest["val_loss"] - latest["train_loss"] if latest else None
        ),
        "evaluation_steps": [r["step"] for r in rows],
        "reason": (
            "minimum evaluation evidence satisfied"
            if count >= min_evaluations
            else "minimum evaluation evidence not satisfied"
        ),
    }


def run_hpo(config_path, output, tracking_uri="http://127.0.0.1:5001",
            experiment="AutoTune-AI-MI300X-HPO"):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing non-empty HPO workspace: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = load_config(config_path, mode="hf")
    baseline_plan = make_plan(config_path)
    if baseline_plan["device"] != "gpu" or baseline_plan["quantization"] != "none":
        raise RuntimeError("Phase 2 pilot requires GPU LoRA with quantization: none")
    candidates = build_candidates(base)
    if len(candidates) > int(base["safety"]["max_trials"]):
        raise RuntimeError("Candidate count exceeds safety.max_trials")

    reporter = HFMLflowReporter(tracking_uri, experiment)
    reporter.start_controller(
        config_sha256=config_sha256(base), model=baseline_plan["model"],
        provider="hpo-grid", train_sha256=baseline_plan["train_sha256"],
        validation_sha256=baseline_plan["validation_sha256"],
        max_attempts=len(candidates), fault_injected=False,
        controller_name="AutoTune HPO Controller",
        controller_type="hpo")

    events, state = output / "controller_events.jsonl", output / "controller_state.json"
    trials = []
    _write(state, {"status":"running", "schema":"autotune-hpo-v1",
                   "baseline_config_sha256":config_sha256(base), "trials":[]})
    try:
        for c in candidates:
            i = c["index"]
            trial_dir = output / f"trial-{i:02d}"
            trial_dir.mkdir()
            cfg_path = output / f"trial-{i:02d}.yaml"
            cfg_path.write_text(yaml.safe_dump(c["config"], sort_keys=False), encoding="utf-8")
            cfg, plan = validate_trial_config(c["config"], baseline_plan, cfg_path)
            _event(events, "trial_started", trial=i, patch=c["patch"])
            reporter.start_trial(i, config_sha256=config_sha256(cfg), cfg=cfg, config_path=cfg_path)
            try:
                receipt = run_lora(str(cfg_path), str(trial_dir), allow_cpu=False)
                if not receipt.get("ok"):
                    raise RuntimeError(f"Trainer returned unsuccessful receipt: {receipt}")
                adapter = trial_dir / "final_adapter"
                checkpoint = Path(receipt["best_checkpoint"])
                verify_manifest(adapter, plan, kind="final_adapter")
                verify_manifest(checkpoint, plan, kind="trainer_checkpoint")
                score = float(receipt["best_val_loss"])
                if not math.isfinite(score):
                    raise RuntimeError("Non-finite validation loss")

                min_evaluations = max(3, int(base["monitor"]["min_evaluations"]))
                evidence = _read_evaluation_evidence(
                    trial_dir / "metrics.jsonl", min_evaluations
                )

                if evidence["qualified"]:
                    observed_best = evidence["best_val_loss"]
                    if not math.isclose(
                        score, observed_best, rel_tol=1e-6, abs_tol=1e-6
                    ):
                        raise RuntimeError(
                            "Trainer best_val_loss does not match recorded "
                            "evaluation history"
                        )

                reporter.finish_trial(i, out=trial_dir,
                    receipt_path=trial_dir/"result_receipt.json",
                    checkpoint=checkpoint, score=score, diagnosis=None,
                    fault_injected=False)

                record = {
                    "trial": i,
                    "status": "valid" if evidence["qualified"] else "insufficient_evidence",
                    "patch": c["patch"],
                    "best_val_loss": score,
                    "output_dir": str(trial_dir),
                    "adapter_dir": str(adapter),
                    "best_checkpoint": str(checkpoint),
                    "evaluation_evidence": evidence,
                }

                _event(
                    events,
                    "trial_completed" if evidence["qualified"] else "trial_insufficient_evidence",
                    trial=i,
                    best_val_loss=score,
                    evaluation_count=evidence["evaluation_count"],
                    required_evaluations=evidence["required_evaluations"],
                    latest_generalization_gap=evidence["latest_generalization_gap"],
                )
            except BaseException:
                reporter.fail_trial(i)
                record = {"trial":i, "status":"failed", "patch":c["patch"],
                          "error":traceback.format_exc(), "output_dir":str(trial_dir)}
                _event(events, "trial_failed", trial=i, error=record["error"][-2500:])
            trials.append(record)
            _write(state, {"status":"running", "schema":"autotune-hpo-v1",
                           "baseline_config_sha256":config_sha256(base), "trials":trials})

        valid = [t for t in trials if t["status"] == "valid"]
        if not valid:
            raise RuntimeError("All HPO trials failed; no candidate can be selected")
        best = min(valid, key=lambda t:t["best_val_loss"])
        result = {"ok":True, "schema":"autotune-hpo-v1", "status":"completed",
                  "selection_metric":"best_val_loss", "best_trial":best["trial"],
                  "best_val_loss":best["best_val_loss"], "best_adapter_dir":best["adapter_dir"],
                  "best_checkpoint":best["best_checkpoint"], "promotion_approved":False,
                  "note":"Validate on representative held-out data before production promotion.",
                  "trials":trials}
        result_path = output/"hpo_result.json"
        _write(result_path, result)
        _write(output/"best_trial.json", best)
        reporter.finish_controller(result_path=result_path, best_trial=best["trial"],
                                    best_val_loss=best["best_val_loss"], event_log=events)
        _write(state, {"status":"completed", "schema":"autotune-hpo-v1",
                       "baseline_config_sha256":config_sha256(base),
                       "trials":trials, "best_trial":best["trial"]})
        return result
    except BaseException:
        reporter.fail_controller()
        _write(state, {"status":"failed", "schema":"autotune-hpo-v1",
                       "baseline_config_sha256":config_sha256(base),
                       "trials":trials, "error":traceback.format_exc()})
        raise
