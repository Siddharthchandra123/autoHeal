"""Replay an AutoTune trial metrics.jsonl through the Phase 2 diagnostic engine."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import load_config
from .metrics import diagnose


def replay(metrics_path: str | Path, config_path: str | Path) -> dict:
    config = load_config(config_path)
    rows = []
    decisions = []
    with Path(metrics_path).open(encoding="utf8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {number}: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            rows.append(entry)
            result = diagnose(rows, config["monitor"])
            decisions.append({"step": entry.get("step"), **result.as_dict()})
    last = decisions[-1] if decisions else diagnose([], config["monitor"]).as_dict()
    return {"evaluations": len(rows), "last_decision": last,
            "warnings": sum(d["severity"] == "warning" for d in decisions),
            "stops": sum(d["action"] == "stop" for d in decisions),
            "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline metric-event replay / diagnostics")
    parser.add_argument("--file", required=True, help="Path to a trial's metrics.jsonl")
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--output", help="Optional destination for the JSON report")
    args = parser.parse_args()
    result = replay(args.file, args.config)
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf8")
    print(report)


if __name__ == "__main__":
    main()
