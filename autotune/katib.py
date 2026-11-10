"""Safe Katib Experiment contracts and offline result reconciliation.

Katib owns cross-trial hyperparameter search. AutoTune's self-healing controller
owns only within-trial diagnosis; these loops must never be nested together.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

ALLOWED_PARAMETERS = {
    "learning_rate": {"type": "double", "min": 1e-6, "max": 5e-3},
    "lora_dropout": {"type": "double", "min": 0.0, "max": 0.5},
    "weight_decay": {"type": "double", "min": 0.0, "max": 0.2},
    "lora_r": {"type": "int", "min": 4, "max": 128},
}
_NAME = re.compile(r"^[a-z][a-z0-9-]{0,52}$")


class KatibContractError(ValueError):
    """Raised when an experiment violates AutoTune's safety contract."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise KatibContractError(f"{label} must be a finite number")
    return float(value)


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise KatibContractError("Katib spec must be an object")
    name = spec.get("name", "autotune-hpo")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise KatibContractError("Experiment name must be a DNS-safe lowercase name")
    namespace = spec.get("namespace", "kubeflow")
    if not isinstance(namespace, str) or not _NAME.fullmatch(namespace):
        raise KatibContractError("Namespace must be a DNS-safe lowercase name")
    algorithm = spec.get("algorithm", "random")
    if algorithm not in {"random", "grid", "bayesianoptimization", "tpe"}:
        raise KatibContractError("Unsupported Katib algorithm")
    objective = spec.get("objective", "val_loss")
    if objective not in {"val_loss", "eval_loss"}:
        raise KatibContractError("Objective must be val_loss or eval_loss")
    direction = spec.get("direction", "minimize")
    if direction not in {"minimize", "maximize"}:
        raise KatibContractError("Objective direction must be minimize or maximize")
    if direction != "minimize":
        raise KatibContractError("Current Katib worker supports minimizing loss objectives only")
    max_trials = spec.get("max_trials", 8)
    parallel = spec.get("parallel_trials", 2)
    failed = spec.get("max_failed_trials", 2)
    for value, label, low, high in ((max_trials, "max_trials", 1, 50), (parallel, "parallel_trials", 1, 8), (failed, "max_failed_trials", 0, 10)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise KatibContractError(f"{label} must be an integer between {low} and {high}")
    if parallel > max_trials:
        raise KatibContractError("parallel_trials cannot exceed max_trials")
    params = spec.get("parameters")
    if not isinstance(params, dict) or not params:
        raise KatibContractError("At least one search parameter is required")
    clean = {}
    for key, bounds in params.items():
        if key not in ALLOWED_PARAMETERS or not isinstance(bounds, dict):
            raise KatibContractError(f"Parameter is not allowlisted: {key}")
        rule = ALLOWED_PARAMETERS[key]
        low = _finite_number(bounds.get("min"), f"{key}.min")
        high = _finite_number(bounds.get("max"), f"{key}.max")
        if low < rule["min"] or high > rule["max"] or low >= high:
            raise KatibContractError(f"{key} bounds must satisfy {rule['min']} <= min < max <= {rule['max']}")
        if rule["type"] == "int" and (not low.is_integer() or not high.is_integer()):
            raise KatibContractError(f"{key} bounds must be integers")
        clean[key] = {"type": rule["type"], "min": low, "max": high}
    image = spec.get("image", "registry.example.com/autotune-ai:0.14.0")
    if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9./:_-]{3,200}", image) or image.startswith("-"):
        raise KatibContractError("Container image reference is invalid")
    backend = spec.get("backend", "mock")
    if backend not in {"mock", "hf"}:
        raise KatibContractError("backend must be mock or hf")
    workspace_pvc = spec.get("workspace_pvc")
    if workspace_pvc is not None and (not isinstance(workspace_pvc, str) or not _NAME.fullmatch(workspace_pvc)):
        raise KatibContractError("workspace_pvc must be a DNS-safe claim name")
    if backend == "hf" and not workspace_pvc:
        raise KatibContractError("HF backend requires workspace_pvc for mounted dataset and trial outputs")
    config_path = spec.get("config_path", "/app/config/lora_public_cpu.yaml")
    if (not isinstance(config_path, str) or "\\" in config_path or
            not (config_path.startswith("/app/config/") or config_path.startswith("/workspace/")) or
            ".." in Path(config_path).parts):
        raise KatibContractError("config_path must be inside /app/config or /workspace")
    return {"name": name, "namespace": namespace, "algorithm": algorithm,
            "objective": objective, "direction": direction, "max_trials": max_trials,
            "parallel_trials": parallel, "max_failed_trials": failed,
            "parameters": clean, "image": image, "backend": backend,
            "config_path": config_path, "workspace_pvc": workspace_pvc}


def build_experiment(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = validate_spec(spec)
    params, trial_parameters, args = [], [], ["--backend", cfg["backend"], "--config", cfg["config_path"]]
    for key, bounds in cfg["parameters"].items():
        params.append({"name": key, "parameterType": bounds["type"],
                       "feasibleSpace": {"min": str(bounds["min"]), "max": str(bounds["max"])}})
        camel = key.split("_")[0] + "".join(x.title() for x in key.split("_")[1:])
        trial_parameters.append({"name": camel, "description": key.replace("_", " "), "reference": key})
        args.extend(["--" + key.replace("_", "-"), "${trialParameters." + camel + "}"])
    container = {"name": "training", "image": cfg["image"],
                 "command": ["python", "-m", "autotune.katib_trial"], "args": args,
                 "resources": {"requests": {"cpu": "1", "memory": "2Gi"},
                               "limits": {"cpu": "2", "memory": "4Gi"}}}
    volumes = []
    if cfg["workspace_pvc"]:
        container["env"] = [
            {"name": "AUTOTUNE_TRAIN_FILE", "value": "/workspace/data/train.jsonl"},
            {"name": "AUTOTUNE_VALIDATION_FILE", "value": "/workspace/data/validation.jsonl"},
            {"name": "AUTOTUNE_TRIAL_OUTPUT_DIR", "value": "/workspace/runs/katib"},
        ]
        container["volumeMounts"] = [{"name": "workspace", "mountPath": "/workspace"}]
        volumes = [{"name": "workspace", "persistentVolumeClaim": {"claimName": cfg["workspace_pvc"]}}]
    return {"apiVersion": "kubeflow.org/v1beta1", "kind": "Experiment",
            "metadata": {"name": cfg["name"], "namespace": cfg["namespace"],
                         "labels": {"app.kubernetes.io/part-of": "autotune-ai", "autotune.ai/phase": "8"}},
            "spec": {"parallelTrialCount": cfg["parallel_trials"], "maxTrialCount": cfg["max_trials"],
                     "maxFailedTrialCount": cfg["max_failed_trials"],
                     "objective": {"type": cfg["direction"], "objectiveMetricName": cfg["objective"]},
                     "algorithm": {"algorithmName": cfg["algorithm"]},
                     "parameters": params,
                     "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
                     "trialTemplate": {"primaryContainerName": "training",
                         "trialParameters": trial_parameters,
                         "trialSpec": {"apiVersion": "batch/v1", "kind": "Job",
                             "spec": {"backoffLimit": 0,
                                 "template": {"spec": {"restartPolicy": "Never", "containers": [container], **({"volumes": volumes} if volumes else {})}}}}}}}


def parse_metrics(text: str, metric: str = "val_loss") -> float:
    """Parse a single finite objective from a worker stdout/stderr capture."""
    if metric not in {"val_loss", "eval_loss", "token_f1"}:
        raise KatibContractError("Unsupported metric")
    matches = re.findall(r"(?m)^\s*" + re.escape(metric) + r"\s*=\s*([-+0-9.eE]+)\s*$", text)
    if len(matches) != 1:
        raise KatibContractError(f"Expected exactly one {metric} metric line")
    value = _finite_number(float(matches[0]), metric)
    if metric == "token_f1" and not 0 <= value <= 1:
        raise KatibContractError("token_f1 must be in [0, 1]")
    return value


def summarize_trials(rows: list[dict[str, Any]], metric: str = "val_loss", direction: str = "minimize") -> dict[str, Any]:
    if metric not in {"val_loss", "eval_loss", "token_f1"} or direction not in {"minimize", "maximize"}:
        raise KatibContractError("Invalid objective")
    valid, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise KatibContractError("Trial rows must be objects")
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in seen:
            raise KatibContractError("Trial IDs must be unique non-empty strings")
        seen.add(trial_id)
        status = row.get("status")
        if status != "Succeeded":
            continue
        score = _finite_number(row.get(metric), f"{trial_id}.{metric}")
        if metric == "token_f1" and not 0 <= score <= 1:
            raise KatibContractError("token_f1 must be in [0, 1]")
        valid.append({"trial_id": trial_id, "score": score,
                      "parameters": row.get("parameters", {})})
    ordered = sorted(valid, key=lambda x: x["score"], reverse=direction == "maximize")
    return {"schema": "autotune-katib-summary-v1", "objective": metric, "direction": direction,
            "succeeded_trials": len(valid), "failed_or_incomplete_trials": len(rows) - len(valid),
            "best_trial": ordered[0] if ordered else None,
            "ranking": ordered, "promotion_approved": False,
            "release_status": "HUMAN_REVIEW_REQUIRED" if ordered else "NO_QUALIFIED_MODEL",
            "note": "Katib optimizes the declared objective only. This summary is not an independent holdout evaluation or production approval."}


def write_experiment(spec: dict[str, Any], output: str | Path) -> dict[str, Any]:
    manifest = build_experiment(spec)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {"ok": True, "output": str(path), "experiment": manifest,
            "promotion_approved": False, "mode": "MANIFEST_ONLY"}


def write_summary(rows: list[dict[str, Any]], output: str | Path, metric: str, direction: str) -> dict[str, Any]:
    result = summarize_trials(rows, metric, direction)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
