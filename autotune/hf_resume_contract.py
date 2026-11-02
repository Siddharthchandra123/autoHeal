"""Offline-safe, strict contracts for a controlled HF Trainer resume exercise.

No torch or network access is required to call these validators.
"""
from __future__ import annotations

import json
from pathlib import Path

from .lora_contract import LoraContractError, verify_manifest


def validate_pause_step(plan: dict, pause_step: int, resume_checkpoint: str | None = None) -> None:
    if resume_checkpoint is not None:
        raise LoraContractError('Pause can only be requested on a fresh run')
    if isinstance(pause_step, bool) or not isinstance(pause_step, int):
        raise LoraContractError('Pause step must be an integer')
    eval_steps = plan['eval_steps']
    if pause_step < eval_steps or pause_step >= plan['estimated_steps'] or pause_step % eval_steps:
        raise LoraContractError('Pause must align with evaluation/save and precede planned final training step')


def validate_resume_source(run: Path, checkpoint: Path, plan: dict) -> int:
    """Reject completed/blocked runs, unverified state or wrong-step checkpoints.

    Returns step recorded by the HF Trainer checkpoint. Does not load tensors.
    """
    run = Path(run).resolve()
    checkpoint = Path(checkpoint).resolve()
    state_file = run / 'run_state.json'
    receipt_file = run / 'pause_receipt.json'
    try:
        state = json.loads(state_file.read_text(encoding='utf-8'))
        receipt = json.loads(receipt_file.read_text(encoding='utf-8'))
        trainer_state = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as exc:
        raise LoraContractError('Cannot read complete controlled-pause state for resume') from exc
    if state.get('status') != 'paused' or receipt.get('ok') is not True or receipt.get('status') != 'paused':
        raise LoraContractError('Resume requires a verified paused run; never resume completed output in place')
    if state.get('checkpoint') != str(checkpoint) or receipt.get('checkpoint') != str(checkpoint):
        raise LoraContractError('Paused checkpoint path does not match run record')
    step = trainer_state.get('global_step')
    if isinstance(step, bool) or not isinstance(step, int) or step != state.get('step') or step != receipt.get('global_step'):
        raise LoraContractError('Trainer global_step differs from paused run metadata')
    if not 0 < step < plan['estimated_steps']:
        raise LoraContractError('Checkpoint has no remaining training steps')
    if receipt.get('target_step') != plan['estimated_steps']:
        raise LoraContractError('Pausing and resuming must use identical training step plan')
    verify_manifest(checkpoint, plan, kind='trainer_checkpoint')
    return step
