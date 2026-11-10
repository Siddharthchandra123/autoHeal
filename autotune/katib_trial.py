"""One isolated training attempt per Katib Trial Job.

The default backend is deterministic SYNTHETIC smoke testing. Real Hugging
Face training requires an explicit --backend hf and a prebuilt authorized image.
Katib owns cross-trial search; this worker never invokes the self-healing retry loop.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import yaml

_ALLOWED = {"learning_rate", "lora_dropout", "weight_decay", "lora_r"}


def _safe_float(value: str, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--backend", choices=("mock", "hf"), default="mock")
    for name in sorted(_ALLOWED):
        parser.add_argument("--" + name.replace("_", "-"), type=float, required=False)
    args = parser.parse_args()
    config_path = Path(args.config)
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("Config must be an existing regular file")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("train"), dict):
        raise ValueError("Training config is malformed")
    for name in _ALLOWED:
        value = getattr(args, name)
        if value is None:
            continue
        if name == "lora_r":
            if not value.is_integer() or not 4 <= value <= 128:
                raise ValueError("lora_r must be an integer in [4, 128]")
            config["train"][name] = int(value)
        else:
            value = _safe_float(str(value), name)
            limits = {"learning_rate": (1e-6, 5e-3), "lora_dropout": (0.0, 0.5), "weight_decay": (0.0, 0.2)}
            low, high = limits[name]
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
            config["train"][name] = value
    if args.backend == "hf":
        train_file = os.environ.get("AUTOTUNE_TRAIN_FILE")
        validation_file = os.environ.get("AUTOTUNE_VALIDATION_FILE")
        if train_file or validation_file:
            if not train_file or not validation_file:
                raise ValueError("Both AUTOTUNE_TRAIN_FILE and AUTOTUNE_VALIDATION_FILE are required")
            for candidate in (Path(train_file), Path(validation_file)):
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError("Mounted train/validation files must be regular files")
            config.setdefault("data", {})["train_file"] = train_file
            config["data"]["validation_file"] = validation_file
    if args.backend == "mock":
        if "simulator" not in config:
            config["simulator"] = {"scenario": "healthy"}
        from .mock_trainer import train
        import tempfile
        with tempfile.TemporaryDirectory(prefix="autotune-katib-mock-") as folder:
            result = train(config, Path(folder), 1)
    else:
        output_root = Path(os.environ.get("AUTOTUNE_TRIAL_OUTPUT_DIR", "/workspace/runs/katib"))
        if not output_root.is_absolute() or ".." in output_root.parts or output_root.is_symlink():
            raise ValueError("Trial output directory must be an absolute safe path")
        output_root.mkdir(parents=True, exist_ok=True)
        trial_name = os.environ.get("KATIB_TRIAL_NAME", os.environ.get("HOSTNAME", "trial"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", trial_name):
            raise ValueError("Invalid Katib trial name")
        trial_dir = output_root / trial_name
        if trial_dir.exists():
            raise FileExistsError("Trial output already exists; refusing overwrite")
        trial_dir.mkdir()
        from .hf_trainer import train
        result = train(config, trial_dir, 1)
    score = float(result.best_val_loss)
    if not math.isfinite(score):
        raise ValueError("Trainer returned a non-finite objective")
    # Katib StdOut collector consumes this exact single-line metric contract.
    print(f"val_loss={score:.10f}", flush=True)
    print(f"eval_loss={score:.10f}", flush=True)
    print(f"best_step={int(result.best_step)}", flush=True)
    print(json.dumps({"status": result.state, "reason": result.reason,
                      "best_step": int(result.best_step), "backend": args.backend,
                      "promotion_approved": False}), flush=True)

if __name__ == "__main__":
    main()
