"""Optional Unsloth backend, off by default and safe to inspect on CPUs."""
from __future__ import annotations

import argparse
import json
from .unsloth_backend import make_unsloth_plan, run_unsloth


def main() -> None:
    p = argparse.ArgumentParser(description='AutoTune AI optional Unsloth backend; preflight by default')
    p.add_argument('--config', required=True)
    p.add_argument('--output', default='outputs/unsloth-gpu-01')
    p.add_argument('--execute', action='store_true', help='Explicitly run on supported GPU only')
    a = p.parse_args()
    if a.execute:
        result = run_unsloth(a.config, a.output)
    else:
        result = make_unsloth_plan(a.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
