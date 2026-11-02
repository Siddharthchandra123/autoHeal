"""Inspect a Phase 4 advisor proposal without changing files or running training."""
from __future__ import annotations
import argparse
import json
from .advisor import advise
from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description='Offline mock or opt-in live Meta diagnosis, read-only')
    p.add_argument('--config', default='config/baseline.yaml')
    p.add_argument('--provider', choices=['mock', 'meta'], default='mock')
    p.add_argument('--diagnosis', choices=['overfit', 'underfit'], default='overfit')
    opts = p.parse_args(argv)
    config = load_config(opts.config, mode='mock')
    answer = advise(opts.diagnosis, [], config['train'], provider=opts.provider,
                    model=config['meta_agent']['model'])
    print(json.dumps({'mode': 'CPU MOCK; no model training', 'advice': answer.as_dict()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
