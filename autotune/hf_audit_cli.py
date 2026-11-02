"""Read-only audit of a completed AutoTune AI Hugging Face LoRA pilot run.

This is deliberately separate from training and does not resume a run. An
optional adapter reload performs fresh inference after all hashes are checked.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .lora_contract import LoraContractError, make_plan, verify_manifest


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise LoraContractError(f'Cannot read valid JSON: {path.name}') from exc
    if not isinstance(value, dict):
        raise LoraContractError(f'Expected JSON object: {path.name}')
    return value


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _best_checkpoint(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LoraContractError('Missing best_checkpoint in result receipt')
    candidate = Path(value)
    # The actual Trainer emits absolute output paths. Relative paths are
    # interpreted relative to the run root, not the current shell directory.
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not _inside(candidate, (root / 'checkpoints').resolve()):
        raise LoraContractError('Best checkpoint must be inside this run/checkpoints')
    if not candidate.is_dir() or candidate.is_symlink():
        raise LoraContractError('Best checkpoint directory missing or unsafe')
    return candidate


def audit_lora_run(config: str | Path, run: str | Path, *, reload: bool = False) -> dict:
    """Verify run provenance, final adapter and full Trainer checkpoint.

    - Offline by default: does not import torch/transformers/datasets/PEFT.
    - `reload=True` intentionally loads pretrained weights and runs inference.
    - Verification does not prove optimizer resume or model quality.
    """
    root = Path(run)
    if root.is_symlink() or not root.is_dir():
        raise LoraContractError('Run directory must be an existing, real directory')
    root = root.resolve()
    plan = make_plan(config)
    recorded_plan = _json_object(root / 'plan.json')
    if recorded_plan != plan:
        raise LoraContractError('Run provenance mismatch (config/model/dataset/paths)')
    state = _json_object(root / 'run_state.json')
    if state.get('status') != 'completed':
        raise LoraContractError('Run is not completed; manual review is required')
    receipt = _json_object(root / 'result_receipt.json')
    if receipt.get('ok') is not True or receipt.get('model') != plan['model']:
        raise LoraContractError('Result receipt is unsuccessful or model differs from plan')
    loss = receipt.get('best_val_loss')
    if isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss):
        raise LoraContractError('Best validation loss missing or nonfinite')
    adapter = root / 'final_adapter'
    if not adapter.is_dir() or adapter.is_symlink():
        raise LoraContractError('Final adapter missing or unsafe')
    if Path(str(receipt.get('adapter_dir', ''))).resolve() != adapter.resolve():
        raise LoraContractError('Adapter path does not match result receipt')
    adapter_manifest = verify_manifest(adapter, plan, kind='final_adapter')
    if receipt.get('adapter_manifest_sha256') != adapter_manifest['files'].get('adapter_model.safetensors'):
        raise LoraContractError('Result receipt does not match adapter weight hash')
    checkpoint = _best_checkpoint(root, receipt.get('best_checkpoint'))
    state_path = state.get('best_checkpoint')
    if not isinstance(state_path, str) or Path(state_path).resolve() != checkpoint:
        raise LoraContractError('Best checkpoint disagrees with completed run state')
    checkpoint_manifest = verify_manifest(checkpoint, plan, kind='trainer_checkpoint')

    result = {
        'ok': True,
        'mode': 'READ-ONLY ARTIFACT AUDIT',
        'model': plan['model'],
        'best_val_loss': float(loss),
        'final_adapter': str(adapter),
        'checkpoint': str(checkpoint),
        'adapter_files_verified': len(adapter_manifest['files']),
        'checkpoint_files_verified': len(checkpoint_manifest['files']),
        'checkpoint_has_training_state': True,
        'actual_resume_executed': False,
        'adapter_reload_executed': False,
        'note': 'SHA-256/provenance check only, not independent quality evaluation or optimizer-resume proof.',
    }
    if reload:
        from .lora_gpu import reload_adapter
        smoke = reload_adapter(str(config), str(adapter), allow_cpu=(plan['device'] == 'cpu'))
        if smoke.get('ok') is not True or smoke.get('adapter_verified') is not True:
            raise LoraContractError('Fresh adapter inference smoke failed')
        result['mode'] = 'ARTIFACT AUDIT + FRESH ADAPTER INFERENCE'
        result['adapter_reload_executed'] = True
        result['generated_token_count'] = smoke['generated_token_count']
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit a completed HF LoRA run before any recovery integration')
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True, help='Completed run directory, e.g. outputs/gpt2-cpu-01')
    parser.add_argument('--reload', action='store_true', help='After audit, reload final adapter and generate 8 tokens')
    args = parser.parse_args(argv)
    try:
        result = audit_lora_run(args.config, args.run, reload=args.reload)
    except (LoraContractError, OSError, ValueError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
