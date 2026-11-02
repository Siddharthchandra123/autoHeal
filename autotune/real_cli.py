"""Phase 5: genuine CPU language-model trainer with durable checkpoint selection."""
import argparse
import json
from .real_recovery import run_real


def main():
    parser = argparse.ArgumentParser(description='Train tiny byte LM with real PyTorch gradients and verified weights')
    parser.add_argument('--config', default='config/real_cpu.yaml')
    parser.add_argument('--workspace', default='runs_real')
    parser.add_argument('--resume', help='Existing experiment directory; requires original config and dataset')
    parser.add_argument('--clear-stale-lock', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_real(args.config, args.workspace, resume=args.resume,
                              clear_stale_lock=args.clear_stale_lock), indent=2))


if __name__ == '__main__':
    main()
