"""Durable on-disk real-PyTorch recovery (CPU-only reference implementation).

Uncertain RUNNING work without a completion receipt is BLOCKED, not silently
repeated. Real weight and optimizer state are verified before selecting a winner.
Separate journal schema from Phase 3 JSON-mock checkpoints prevents confusion.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import math

from .config import load_config, config_sha256
from .real_training import train, dataset_identity, load_verified_checkpoint, RealTrainingError
from .recovery import (_atomic_json, _atomic_yaml, _block, _guard, _hash_file,
                       _json, _new_experiment, _next_config, _save, RecoveryError)

SCHEMA = 'real-torch-v0.6'
RECEIPT_KEYS = {'trial', 'config_sha256', 'dataset_sha256', 'checkpoint_sha256',
                'checkpoint_name', 'state', 'reason', 'best_val_loss', 'best_step'}


def verified_receipt(root: Path, attempt: dict) -> dict:
    trial = attempt['trial']
    trial_dir = root / f'trial-{trial:02d}'
    try:
        receipt = _json(trial_dir / 'result_receipt.json')
        if set(receipt) != RECEIPT_KEYS:
            raise ValueError('Unexpected receipt fields')
        if receipt['trial'] != trial or receipt['config_sha256'] != attempt['config_sha256']:
            raise ValueError('Attempt identity mismatch')
        if receipt['checkpoint_name'] != 'best_checkpoint.pt':
            raise ValueError('Unexpected checkpoint filename')
        if receipt['state'] not in {'completed', 'overfit', 'underfit', 'diverged'}:
            raise ValueError('Unexpected training state')
        if isinstance(receipt['best_step'], bool) or not isinstance(receipt['best_step'], int) or receipt['best_step'] < 1:
            raise ValueError('Invalid step')
        score = receipt['best_val_loss']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            raise ValueError('Invalid loss')
        expected_dataset = dataset_identity(attempt['config'])
        if receipt['dataset_sha256'] != expected_dataset:
            raise ValueError('Dataset changed since attempt')
        checkpoint = trial_dir / 'best_checkpoint.pt'
        loaded = load_verified_checkpoint(checkpoint,
            expected_sha256=receipt['checkpoint_sha256'],
            expected_config_sha256=attempt['config_sha256'], expected_data_sha256=expected_dataset)
        if loaded['epoch'] != receipt['best_step'] or abs(loaded['val_loss'] - score) > 1e-8:
            raise ValueError('Checkpoint loss/epoch mismatch')
        return receipt
    except (ValueError, KeyError, TypeError, OSError, RealTrainingError) as exc:
        raise RecoveryError(f'Invalid real checkpoint/receipt for trial {trial}: {exc}') from exc


def _commit(root: Path, state: dict, attempt: dict):
    receipt = verified_receipt(root, attempt)
    attempt.update({'status': 'completed' if receipt['state'] == 'completed' else 'stopped',
                    'training_state': receipt['state'], 'reason': receipt['reason'],
                    'best_val_loss': receipt['best_val_loss'], 'best_step': receipt['best_step'],
                    'checkpoint_sha256': receipt['checkpoint_sha256']})
    if state['best'] is None or receipt['best_val_loss'] < state['best']['best_val_loss']:
        state['best'] = {'trial': attempt['trial'], 'best_val_loss': receipt['best_val_loss'],
                         'best_step': receipt['best_step'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                         'checkpoint': f"trial-{attempt['trial']:02d}/best_checkpoint.pt"}
        _atomic_yaml(root / 'best_config.yaml', attempt['config'])
    _save(root, state, 'real_attempt_committed', trial=attempt['trial'], status=attempt['status'])


def run_real(config_path='config/real_cpu.yaml', workspace='runs_real', *, resume=None,
             clear_stale_lock=False) -> dict:
    """Run actual CPU tensor training + hash verified durable selection.

    This explicitly does not execute Hugging Face or Meta hosted inference. If
    an attempt is left RUNNING without receipt, human inspection is required.
    """
    config = load_config(config_path, mode='hf')
    if config['model']['name'] != 'autotune/tiny-byte-gru':
        raise RecoveryError('Real CPU mode only supports the tiny byte LM; use HF pilot separately')
    data_hash = dataset_identity(config)
    if resume:
        root = Path(resume).resolve()
        if not (root / 'recovery_state.json').is_file():
            raise RecoveryError('Missing real training recovery journal')
    else:
        root = _new_experiment(config, Path(workspace))
        _atomic_json(root / 'recovery_state.json', {
            'schema': SCHEMA, 'status': 'running', 'block_reason': None,
            'config_sha256': config_sha256(config), 'dataset_sha256': data_hash,
            'max_trials': int(config['safety']['max_trials']), 'attempts': [],
            'best': None, 'revision': 0})
    with _guard(root, clear_stale_lock=clear_stale_lock):
        state = _json(root / 'recovery_state.json')
        if state.get('schema') != SCHEMA or state['config_sha256'] != config_sha256(config):
            raise RecoveryError('Real recovery schema/config mismatch')
        if state['dataset_sha256'] != data_hash:
            raise RecoveryError('Dataset changed; refused unsafe resume')
        if state['attempts'] and state['attempts'][-1]['status'] == 'running':
            previous = state['attempts'][-1]
            if (root / f"trial-{previous['trial']:02d}" / 'result_receipt.json').exists():
                try:
                    _commit(root, state, previous)
                except RecoveryError as exc:
                    _block(root, state, str(exc))
            else:
                _block(root, state, 'Uncertain real training: RUNNING without validated receipt; operator inspection required')
        elif state['status'] == 'blocked':
            raise RecoveryError('Experiment blocked: manual investigation required')
        if state['status'] not in ('completed', 'budget_exhausted'):
            while len(state['attempts']) < state['max_trials']:
                if state['attempts']:
                    previous = state['attempts'][-1]
                    if previous['status'] == 'completed':
                        state['status'] = 'completed'
                        break
                    if previous['status'] != 'stopped':
                        _block(root, state, 'Unexpected prior state: no automatic retry')
                    try:
                        candidate = _next_config(previous, root)
                    except Exception as exc:
                        _block(root, state, f'Policy/advisor error: {exc}')
                    if candidate is None:
                        _block(root, state, 'No approved retry for divergence or unknown diagnosis')
                else:
                    candidate = deepcopy(config)
                trial = len(state['attempts']) + 1
                trial_dir = root / f'trial-{trial:02d}'
                trial_dir.mkdir()
                _atomic_yaml(trial_dir / 'config.yaml', candidate)
                attempt = {'trial': trial, 'status': 'running', 'config': candidate,
                           'config_sha256': config_sha256(candidate),
                           'idempotency_key': hashlib.sha256(f'{root.name}:{trial}:{config_sha256(candidate)}'.encode()).hexdigest()}
                state['attempts'].append(attempt)
                _save(root, state, 'real_attempt_started', trial=trial)
                try:
                    result = train(candidate, trial_dir, trial)
                except Exception as exc:
                    _block(root, state, f'Real trainer failed: {type(exc).__name__}: {exc}')
                if Path(result.checkpoint).resolve() != (trial_dir / 'best_checkpoint.pt').resolve():
                    _block(root, state, 'Unexpected real trainer checkpoint path')
                receipt = {'trial': trial, 'config_sha256': attempt['config_sha256'],
                           'dataset_sha256': data_hash, 'checkpoint_sha256': _hash_file(Path(result.checkpoint)),
                           'checkpoint_name': 'best_checkpoint.pt', 'state': result.state, 'reason': result.reason,
                           'best_val_loss': result.best_val_loss, 'best_step': result.best_step}
                _atomic_json(trial_dir / 'result_receipt.json', receipt)
                try:
                    _commit(root, state, attempt)
                except RecoveryError as exc:
                    _block(root, state, str(exc))
                if attempt['status'] == 'completed':
                    state['status'] = 'completed'
                    break
            else:
                state['status'] = 'budget_exhausted'
        if state['best'] is None:
            _block(root, state, 'No real validated model checkpoint available')
        winning_attempt = state['attempts'][state['best']['trial'] - 1]
        receipt = verified_receipt(root, winning_attempt)
        if receipt['checkpoint_sha256'] != state['best']['checkpoint_sha256']:
            _block(root, state, 'Selected checkpoint differs from validated receipt')
        _atomic_yaml(root / 'active.yaml', winning_attempt['config'])
        _save(root, state, 'real_experiment_finalized')
        return {'ok': True, 'mode': 'REAL PyTorch CPU', 'experiment': str(root), 'status': state['status'],
                'attempt_count': len(state['attempts']), 'best_trial': state['best']['trial'],
                'best_val_loss': state['best']['best_val_loss'],
                'checkpoint_sha256': state['best']['checkpoint_sha256'],
                'checkpoint': state['best']['checkpoint'],
                'note': 'Actual loadable PyTorch model and optimizer weights. Tiny GRU, NOT Meta Llama.'}
