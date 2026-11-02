"""CLI for safe dry-run, explicit Hugging Face LoRA execution, and adapter verification."""
import argparse
import json
from .lora_contract import LoraContractError, make_plan


def main():
    parser = argparse.ArgumentParser(description='Phase 5B PEFT LoRA; default = offline planning only')
    parser.add_argument('--config', default='config/lora_meta_gpu.yaml')
    parser.add_argument('--output', default='outputs/lora-pilot')
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument('--execute', action='store_true', help='Explicitly download model and train')
    controls.add_argument('--verify-adapter', help='Load and run the final adapter from this directory')
    parser.add_argument('--resume-checkpoint', help='Explicit path to verified HF Trainer checkpoint')
    parser.add_argument('--allow-cpu', action='store_true', help='Only for small non-Meta smoke model')
    args = parser.parse_args()
    if args.resume_checkpoint and not args.execute:
        parser.error('--resume-checkpoint requires --execute')
    plan = make_plan(args.config)
    if args.execute:
        from .lora_gpu import run_lora
        result = run_lora(args.config, args.output, resume_checkpoint=args.resume_checkpoint,
                          allow_cpu=args.allow_cpu)
    elif args.verify_adapter:
        from .lora_gpu import reload_adapter
        result = reload_adapter(args.config, args.verify_adapter, allow_cpu=args.allow_cpu)
    else:
        result = {'ok':True, 'mode':'OFFLINE PLAN ONLY; no download or training', 'plan':plan,
                  'next': ('For this small public CPU model, install requirements-lora.txt and use --execute --allow-cpu; execution downloads model files.' if plan['device'] == 'cpu' else 'For this GPU model, verify device and repository access before using --execute.')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
