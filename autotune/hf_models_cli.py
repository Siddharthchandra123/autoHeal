"""AutoTune AI unified CLI for HF causal-LM LoRA and CPU text classification."""
import argparse
import json
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description='HF open-source model training: preflight by default')
    p.add_argument('--config', required=True)
    p.add_argument('--output', default='outputs/hf-model')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--execute', action='store_true', help='Actually download model and fine-tune on CPU')
    g.add_argument('--verify', metavar='DIR', help='Verify integrity of a final artifact directory')
    p.add_argument('--resume-checkpoint', default=None)
    p.add_argument('--pause-after-step', type=int, default=None, help='Controlled checkpoint pause; only for a fresh causal-LM run')
    args = p.parse_args()
    if args.pause_after_step is not None and (not args.execute or args.resume_checkpoint):
        p.error('--pause-after-step requires --execute and a fresh output')
    if args.resume_checkpoint and not args.execute:
        p.error('--resume-checkpoint requires --execute')
    with Path(args.config).open('r', encoding='utf-8') as f:
        peek = yaml.safe_load(f)
    task = peek.get('task', 'causal-lm') if isinstance(peek, dict) else None
    if task == 'text-classification':
        from .hf_classification import classification_plan, run_classification, verify_manifest
        plan = classification_plan(args.config)
        if args.pause_after_step is not None:
            p.error('--pause-after-step currently supports causal-lm only')
        if args.execute:
            result = run_classification(args.config, args.output, resume=args.resume_checkpoint)
        elif args.verify:
            result = verify_manifest(args.verify, plan, kind='model')
        else:
            result = {'ok': True, 'mode': 'OFFLINE PREFLIGHT', 'plan': plan}
    elif task == 'causal-lm':
        from .lora_contract import make_plan
        plan = make_plan(args.config)
        if args.execute:
            from .lora_gpu import run_lora
            result = run_lora(args.config, args.output, allow_cpu=(plan['device'] == 'cpu'),
                              resume_checkpoint=args.resume_checkpoint, pause_after_step=args.pause_after_step)
        elif args.verify:
            from .lora_contract import verify_manifest
            result = verify_manifest(args.verify, plan, kind='final_adapter')
        else:
            result = {'ok': True, 'mode': 'OFFLINE PREFLIGHT', 'plan': plan}
    else:
        p.error('Unsupported task. Available: text-classification, causal-lm')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
