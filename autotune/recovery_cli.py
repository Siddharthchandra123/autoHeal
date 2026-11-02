"""Windows-compatible CLI for Phase 3 durable mock recovery."""
import argparse
import json
import sys
from .recovery import RecoveryError, run_durable


def main(argv=None):
    p = argparse.ArgumentParser(description='Durable CPU-only training recovery; no GPU weights are saved')
    p.add_argument('--config', default='config/baseline.yaml')
    p.add_argument('--workspace', default='runs')
    p.add_argument('--resume', help='Existing experiment directory to resume')
    p.add_argument('--abandon-interrupted', action='store_true', help='Operator confirms prior job is stopped; count uncertain attempt against budget')
    p.add_argument('--clear-stale-lock', action='store_true', help='Operator confirms no controller is running; remove a stale lock')
    p.add_argument('--writeback', action='store_true', help='Explicitly generate config/optimized.yaml')
    args = p.parse_args(argv)
    try:
        result = run_durable(args.config, args.workspace, args.resume,
                             write_back=args.writeback,
                             abandon_interrupted=args.abandon_interrupted,
                             clear_stale_lock=args.clear_stale_lock)
    except RecoveryError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({'ok': True, 'experiment': result['experiment'],
                      'status': result['status'], 'attempt_count': len(result['attempts']),
                      'attempt_states': [a['status'] for a in result['attempts']],
                      'best_trial': result['best']['trial'] if result['best'] else None,
                      'best_val_loss': result['best']['best_val_loss'] if result['best'] else None,
                      'checkpoint_sha256': result['best']['checkpoint_sha256'] if result['best'] else None,
                      'note': 'CPU synthetic demo; checkpoint is JSON metadata, not model weights'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
