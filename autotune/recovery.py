"""Phase 3: durable, bounded, CPU-mock recovery state machine.

The journal JSON is the authoritative state, not the best-effort event log.
A RUNNING attempt with no validated receipt is *never* retried silently.
The local mock adapter produces simulated checkpoint metadata, not model weights.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import yaml

from .config import load_config, config_sha256
from .mock_trainer import train as mock_train
from .policy import apply_patch, fallback_patch, validate_patch
from .advisor import advise, AdvisorError
from .tracking import event

VERSION = 'phase3-v0.4'
TERMINAL = {'completed', 'blocked', 'budget_exhausted'}

class RecoveryError(RuntimeError):
    """Recoverable operation needs operator action or corrected inputs."""

class SimulatedInterruption(RuntimeError):
    """Test-only fault injection; never used by user-facing CLI."""


def _time():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(obj, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_yaml(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(obj, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _save(root: Path, state: dict, change: str, **fields):
    state['updated_at'] = _time()
    state['revision'] += 1
    _atomic_json(root / 'recovery_state.json', state)
    event(root / 'recovery_events.jsonl', {'revision': state['revision'], 'change': change, **fields})


@contextmanager
def _guard(root: Path, *, clear_stale_lock: bool = False):
    lock = root / '.controller.lock'
    if lock.exists():
        if not clear_stale_lock:
            raise RecoveryError(f'Controller lock exists: {lock}. Ensure no controller is running; use --clear-stale-lock only after verifying this.')
        lock.unlink()
        event(root / 'recovery_events.jsonl', {'change': 'manual_stale_lock_clear', 'warning': 'Operator asserted no controller is active'})
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise RecoveryError('Another controller holds the experiment lock') from e
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'created_at': _time()}, f)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _new_experiment(config: dict, workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for suffix in range(1000):
        name = prefix if suffix == 0 else f'{prefix}-{suffix}'
        root = workspace / name
        try:
            root.mkdir()
            return root
        except FileExistsError:
            continue
    raise RecoveryError('No free experiment name available')


def _validate_receipt(root: Path, attempt: dict) -> dict:
    """Validate receipt, simulated checkpoint metadata, hash, and attempt identity.

    Does NOT claim that a real model adapter checkpoint is loadable.
    """
    trial = attempt['trial']
    trial_dir = root / f'trial-{trial:02d}'
    receipt_path = trial_dir / 'result_receipt.json'
    if not receipt_path.is_file():
        raise RecoveryError(f'No result receipt for trial {trial}')
    try:
        receipt = _json(receipt_path)
        expected_fields = {'trial', 'config_sha256', 'checkpoint_sha256', 'state', 'reason',
                           'best_val_loss', 'best_step', 'checkpoint_name'}
        if set(receipt) != expected_fields:
            raise ValueError('Unexpected receipt schema')
        if receipt['trial'] != trial or receipt['config_sha256'] != attempt['config_sha256']:
            raise ValueError('Receipt trial/config mismatch')
        if receipt['checkpoint_name'] != 'best_checkpoint.json':
            raise ValueError('Unapproved checkpoint path')
        if receipt['state'] not in ('completed', 'overfit', 'underfit', 'diverged'):
            raise ValueError('Unapproved training state')
        if not isinstance(receipt['best_step'], int) or receipt['best_step'] < 1:
            raise ValueError('Invalid best_step')
        loss = receipt['best_val_loss']
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
            raise ValueError('Invalid best loss')
        chk_path = trial_dir / receipt['checkpoint_name']
        if not chk_path.is_file() or chk_path.is_symlink():
            raise ValueError('Checkpoint missing or not a regular file')
        if _hash_file(chk_path) != receipt['checkpoint_sha256']:
            raise ValueError('Checkpoint SHA256 mismatch')
        marker = _json(chk_path)
        if marker.get('note') != 'SIMULATED checkpoint metadata, NOT model weights':
            raise ValueError('Checkpoint is not the expected mock marker')
        if marker.get('best_step') != receipt['best_step'] or marker.get('best_val_loss') != loss:
            raise ValueError('Checkpoint metrics inconsistent with receipt')
        if config_sha256({'train': marker.get('train')}) != config_sha256({'train': attempt['config']['train']}):
            raise ValueError('Checkpoint training configuration mismatch')
        return receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
        raise RecoveryError(f'Invalid trial {trial} recovery receipt: {e}') from e


def _commit_receipt(root: Path, state: dict, attempt: dict):
    receipt = _validate_receipt(root, attempt)
    attempt.update({'status': 'completed' if receipt['state'] == 'completed' else 'stopped',
                    'training_state': receipt['state'], 'reason': receipt['reason'],
                    'best_val_loss': receipt['best_val_loss'], 'best_step': receipt['best_step'],
                    'checkpoint_sha256': receipt['checkpoint_sha256'],
                    'finished_at': _time()})
    if state['best'] is None or receipt['best_val_loss'] < state['best']['best_val_loss']:
        state['best'] = {'trial': attempt['trial'], 'best_val_loss': receipt['best_val_loss'],
                         'best_step': receipt['best_step'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                         'checkpoint': f"trial-{attempt['trial']:02d}/best_checkpoint.json"}
        _atomic_yaml(root / 'best_config.yaml', attempt['config'])
    _save(root, state, 'attempt_committed', trial=attempt['trial'], status=attempt['status'])


def _next_config(previous: dict, root: Path) -> dict | None:
    """Version one policy-approved recommendation per stopped attempt.

    An existing advisor decision is replayed, not regenerated on controller
    restart. Hash and trial/config/checkpoint identity are checked on replay.
    """
    if previous['training_state'] not in {'overfit', 'underfit'}:
        return None
    trial = previous['trial']
    report_path = root / f'trial-{trial:02d}' / 'advisor_decision.json'
    train = previous['config']['train']
    diagnosis = previous['training_state']
    if report_path.exists():
        try:
            record = _json(report_path)
            digest = record.pop('record_sha256')
            if (digest != config_sha256(record) or record['trial'] != trial or
                record['config_sha256'] != previous['config_sha256'] or
                record['checkpoint_sha256'] != previous['checkpoint_sha256'] or
                record['diagnosis'] != diagnosis):
                raise ValueError('Advisor replay identity/hash mismatch')
            patch = validate_patch(record['patch'], train, diagnosis)
            candidate = apply_patch(previous['config'], patch)
            if config_sha256(candidate) != record['output_config_sha256']:
                raise ValueError('Advisor replay output configuration mismatch')
            return candidate
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise RecoveryError(f'Advisor decision replay failed integrity validation: {exc}') from exc
    agent = previous['config']['meta_agent']
    provider = agent.get('provider', 'meta') if agent.get('enabled') else 'off'
    if provider == 'off':
        patch = validate_patch(fallback_patch(diagnosis, train), train, diagnosis)
        source, status, reason = 'deterministic-policy', 'approved', 'Advisor disabled'
    else:
        # Metrics are read locally, with strict numeric allowlisting in advisor.py.
        history_path = root / f'trial-{trial:02d}' / 'metrics.jsonl'
        history = []
        if history_path.is_file():
            for line in history_path.read_text(encoding='utf8').splitlines()[-8:]:
                try:
                    metric = json.loads(line)
                except json.JSONDecodeError:
                    continue
                history.append(metric)
        advice = advise(diagnosis, history, train, provider=provider, model=agent['model'])
        patch = advice.patch
        source, status, reason = advice.source, advice.status, advice.reason
    candidate = apply_patch(previous['config'], patch)
    record = {'trial': trial, 'diagnosis': diagnosis, 'config_sha256': previous['config_sha256'],
              'checkpoint_sha256': previous['checkpoint_sha256'], 'patch': patch,
              'source': source, 'status': status, 'reason': reason,
              'output_config_sha256': config_sha256(candidate)}
    record['record_sha256'] = config_sha256(record)
    _atomic_json(report_path, record)
    event(root / 'recovery_events.jsonl', {'change': 'advisor_decision', 'trial': trial,
                                          'source': source, 'status': status, 'patch': patch})
    return candidate


def _block(root: Path, state: dict, reason: str):
    state['status'] = 'blocked'
    state['block_reason'] = reason
    _save(root, state, 'blocked', reason=reason)
    raise RecoveryError(reason)


def run_durable(config_path: str = 'config/baseline.yaml', workspace: str = 'runs',
                resume: str | None = None, *, write_back: bool = False,
                abandon_interrupted: bool = False, clear_stale_lock: bool = False,
                fault_at: str | None = None) -> dict:
    """Run/resume the durable CPU demonstration.

    Internal ``fault_at`` is for deterministic crash-injection tests only.
    Resume an uncertain RUNNING attempt only by explicit abandon, never automatically.
    """
    config = load_config(config_path, mode='mock')
    if config['meta_agent'].get('enabled') and config['meta_agent'].get('provider', 'meta') not in ('mock', 'meta'):
        raise RecoveryError('Unapproved Meta advisor provider')
    if resume:
        root = Path(resume).resolve()
        if not root.is_dir() or not (root / 'recovery_state.json').is_file():
            raise RecoveryError('Resume path does not contain a Phase 3 recovery state')
    else:
        root = _new_experiment(config, Path(workspace))
        _atomic_json(root / 'recovery_state.json', {
            'schema': VERSION, 'status': 'running', 'block_reason': None,
            'config_sha256': config_sha256(config), 'initial_config': config,
            'max_trials': int(config['safety']['max_trials']), 'attempts': [],
            'best': None, 'revision': 0, 'created_at': _time(), 'updated_at': _time()
        })
    with _guard(root, clear_stale_lock=clear_stale_lock):
        state = _json(root / 'recovery_state.json')
        if state.get('schema') != VERSION:
            raise RecoveryError('Unsupported recovery state schema')
        if config_sha256(config) != state['config_sha256']:
            raise RecoveryError('Input config changed since experiment creation; resume with original baseline')
        if state['status'] in {'completed', 'budget_exhausted'}:
            return {'experiment': str(root), 'best': state['best'], 'attempts': state['attempts'], 'status': state['status']}
        # A prior stop before the state was finalized may leave a valid receipt.
        if state['attempts'] and state['attempts'][-1]['status'] == 'running':
            previous = state['attempts'][-1]
            if (root / f"trial-{previous['trial']:02d}" / 'result_receipt.json').exists():
                try:
                    _commit_receipt(root, state, previous)
                except RecoveryError as exc:
                    _block(root, state, f'Receipt integrity failure: {exc}')
            elif not abandon_interrupted:
                _block(root, state, 'Prior attempt status RUNNING, no receipt. Inspect external jobs; resume only with --abandon-interrupted after confirming the job stopped.')
            else:
                previous['status'] = 'abandoned'
                previous['reason'] = 'Operator explicitly abandoned uncertain attempt'
                previous['finished_at'] = _time()
                state['status'] = 'running'
                state['block_reason'] = None
                _save(root, state, 'operator_abandoned_uncertain_attempt', trial=previous['trial'])
        elif state['status'] == 'blocked':
            # No implicit unblock. Only the uncertain-running case above can unblock.
            raise RecoveryError(f"Experiment blocked: {state.get('block_reason')}")
        while True:
            if state['attempts']:
                prev = state['attempts'][-1]
                if prev['status'] == 'completed':
                    state['status'] = 'completed'
                    break
                if prev['status'] == 'stopped':
                    try:
                        next_config = _next_config(prev, root)
                    except (RecoveryError, AdvisorError, ValueError) as exc:
                        _block(root, state, f"Advisor recovery blocked: {exc}")
                    if next_config is None:
                        _block(root, state, f"No approved autonomous recovery for {prev['training_state']}")
                elif prev['status'] == 'abandoned':
                    next_config = deepcopy(prev['config'])
                else:
                    _block(root, state, f"Unexpected attempt status: {prev['status']}")
            else:
                next_config = deepcopy(config)
            if len(state['attempts']) >= state['max_trials']:
                state['status'] = 'budget_exhausted'
                break
            trial = len(state['attempts']) + 1
            trial_dir = root / f'trial-{trial:02d}'
            trial_dir.mkdir(exist_ok=False)
            _atomic_yaml(trial_dir / 'config.yaml', next_config)
            attempt = {'trial': trial, 'status': 'running', 'config': next_config,
                       'config_sha256': config_sha256(next_config), 'started_at': _time(),
                       'idempotency_key': hashlib.sha256(f"{root.name}:{trial}:{config_sha256(next_config)}".encode()).hexdigest()}
            state['attempts'].append(attempt)
            state['status'] = 'running'
            state['block_reason'] = None
            _save(root, state, 'attempt_started', trial=trial, idempotency_key=attempt['idempotency_key'])
            if fault_at == 'after_state_running':
                raise SimulatedInterruption('after_state_running')
            try:
                result = mock_train(next_config, trial_dir, trial)
            except Exception as exc:
                # No retry of uncertain failures; require operator inspection.
                _block(root, state, f'Trainer raised {type(exc).__name__}: {exc}')
            if fault_at == 'after_training_before_receipt':
                raise SimulatedInterruption('after_training_before_receipt')
            checkpoint = Path(result.checkpoint)
            # The result MUST name the expected artifact, not arbitrary files.
            if checkpoint.resolve() != (trial_dir / 'best_checkpoint.json').resolve():
                _block(root, state, 'Trainer returned unexpected checkpoint location')
            receipt = {'trial': trial, 'config_sha256': attempt['config_sha256'],
                       'checkpoint_sha256': _hash_file(checkpoint),
                       'checkpoint_name': 'best_checkpoint.json',
                       'state': result.state, 'reason': result.reason,
                       'best_val_loss': result.best_val_loss, 'best_step': result.best_step}
            _atomic_json(trial_dir / 'result_receipt.json', receipt)
            if fault_at == 'after_receipt_before_commit':
                raise SimulatedInterruption('after_receipt_before_commit')
            try:
                _commit_receipt(root, state, attempt)
            except RecoveryError as exc:
                _block(root, state, str(exc))
        if state['best'] is None:
            _block(root, state, 'No verified checkpoint available; no model can be selected')
        # Recheck integrity at promotion-selection boundary.
        best_attempt = state['attempts'][state['best']['trial']-1]
        try:
            checked = _validate_receipt(root, best_attempt)
            if checked['checkpoint_sha256'] != state['best']['checkpoint_sha256']:
                raise RecoveryError('Best checkpoint digest differs from recorded selection')
        except RecoveryError as exc:
            _block(root, state, f'Best checkpoint failed final verification: {exc}')
        _atomic_yaml(root / 'active.yaml', best_attempt['config'])
        if write_back:
            _atomic_yaml(Path(config_path).resolve().parent / 'optimized.yaml', best_attempt['config'])
        _save(root, state, 'experiment_terminated', status=state['status'], best_trial=state['best']['trial'])
        return {'experiment': str(root), 'best': state['best'], 'attempts': state['attempts'], 'status': state['status']}
