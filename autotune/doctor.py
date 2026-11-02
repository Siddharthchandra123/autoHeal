"""Local environment checks with a machine-readable summary; no secrets logged."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .config import ConfigError, config_sha256, load_config


def inspect(config_path: str = "config/baseline.yaml", mode: str = "mock") -> dict:
    checks = []

    def add(name: str, passed: bool, detail: str):
        checks.append({"name": name, "passed": passed, "detail": detail})

    add("python", sys.version_info >= (3, 11), f"{sys.version_info.major}.{sys.version_info.minor}")
    add("pyyaml", importlib.util.find_spec("yaml") is not None, "PyYAML is required")
    add("pytest", importlib.util.find_spec("pytest") is not None, "pytest is required for verification")
    digest = None
    try:
        config = load_config(config_path, mode=mode)
        digest = config_sha256(config)
        add("configuration", True, f"Valid SHA256={digest[:16]}")
        if mode == "hf":
            for module in ("torch", "transformers", "peft", "datasets"):
                add(module, importlib.util.find_spec(module) is not None, "GPU training dependency")
            for label, filename in (("train_file", config["data"]["train_file"]),
                                    ("validation_file", config["data"]["validation_file"])):
                add(label, Path(filename).is_file(), f"Required dataset: {filename}")
    except (ConfigError, OSError) as exc:
        add("configuration", False, str(exc))
    return {"ok": all(item["passed"] for item in checks), "mode": mode,
            "config_sha256": digest, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoTune AI prerequisites")
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--mode", choices=("mock", "hf"), default="mock")
    args = parser.parse_args()
    result = inspect(args.config, args.mode)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
