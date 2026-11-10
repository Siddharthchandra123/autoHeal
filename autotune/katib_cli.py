"""Offline-first Katib manifest and trial-result CLI."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from .katib import write_experiment, write_summary


def main() -> None:
    p = argparse.ArgumentParser(description="AutoTune AI Phase 8 Katib tools")
    sub = p.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="validate spec and render a Katib Experiment manifest")
    render.add_argument("--spec", default="config/katib_cpu.yaml")
    render.add_argument("--output", default="pipelines/autotune_katib_experiment.yaml")
    summary = sub.add_parser("summarize", help="rank completed trial JSON records")
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", default="outputs/katib/summary.json")
    summary.add_argument("--metric", choices=["val_loss", "eval_loss", "token_f1"], default="val_loss")
    summary.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    args = p.parse_args()
    if args.command == "render":
        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        result = write_experiment(spec, args.output)
    else:
        rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise SystemExit("Input must be a JSON array of trial records")
        result = write_summary(rows, args.output, args.metric, args.direction)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
