"""Phase 5E: bounded real HF LoRA trial orchestration, CPU-first.

Each attempt is a NEW run, never an in-place checkpoint resume after a patch.
Only validated YAML numeric hyperparameters can change. The advisor has no
filesystem, shell, or deployment permissions. Existing workspaces are refused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Callable

import yaml

from .advisor import advise
from .config import load_config, config_sha256
from .lora_contract import LoraContractError, make_plan, verify_manifest, _write_json_atomic
from .policy import apply_patch, validate_patch

SCHEMA = 'autotune-hf-selfheal-v0.8.6'  # Controller receipt schema unchanged in Phase 6A


def _audit(path: Path, event: str, **details) -> None:
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **details}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict):
            result.append(row)
    return result[-8:]


def _write_yaml_exclusive(path: Path, data: dict):
    with path.open('x', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())


def preflight(config: str, *, provider: str = 'mock') -> dict:
    if provider not in {'mock', 'meta'}:
        raise LoraContractError('Only mock or meta providers supported')
    cfg = load_config(config, mode='hf')
    plan = make_plan(config)
    if plan['device'] != 'cpu' or plan['quantization'] != 'none':
        raise LoraContractError('Phase 5E currently supports CPU nonquantized LoRA only')
    attempts = int(cfg['safety']['max_trials'])
    if not 1 <= attempts <= 3:
        raise LoraContractError('Phase 5E bounded attempt budget must be 1..3')
    return {'ok': True, 'mode': 'HF SELF-HEAL OFFLINE PREFLIGHT', 'model': plan['model'],
            'device': 'cpu', 'provider': provider, 'max_attempts': attempts,
            'estimated_steps_per_attempt': plan['estimated_steps'],
            'config_sha256': plan['config_sha256'], 'train_sha256': plan['train_sha256'],
            'validation_sha256': plan['validation_sha256'],
            'note': 'No model downloaded or training executed. Fault injection is disabled unless explicitly requested.'}


def execute(config: str, workspace: str, *, provider: str = 'mock',
            fault_inject_overfit: bool = False,
            trainer_fn: Callable | None = None,
            advisor_fn: Callable | None = None, mlflow_uri: str | None = None,
            mlflow_experiment: str = 'AutoTune-AI-CPU', mlflow_reporter=None) -> dict:
    """Execute new, sequential HF attempts; refuse any nonempty existing workspace.

    trainer_fn/advisor_fn exist for isolated offline controller tests only.
    The standard CLI always calls real Hugging Face run_lora.
    """
    info = preflight(config, provider=provider)
    source = load_config(config, mode='hf')
    reference = make_plan(config)
    # Construct optional reporter BEFORE touching the experiment directory.
    # Passing a test reporter does not enable real MLflow network calls.
    if mlflow_reporter is None and mlflow_uri is not None:
        from .hf_mlflow import HFMLflowReporter
        mlflow_reporter = HFMLflowReporter(mlflow_uri, mlflow_experiment)
    root = Path(workspace).resolve()
    if root.is_symlink() or (root.exists() and any(root.iterdir())):
        raise LoraContractError('Workspace is not empty: use a fresh path; never overwrite an existing experiment')
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / '.controller.lock'
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LoraContractError('Experiment lock exists') from exc
    os.close(fd)
    journal = root / 'controller_events.jsonl'
    attempts = []
    cfg = deepcopy(source)
    max_trials = info['max_attempts']
    _write_json_atomic(root / 'source_provenance.json', {'schema':SCHEMA, 'initial_plan':reference,
        'provider':provider, 'max_attempts': max_trials, 'fault_inject_overfit':fault_inject_overfit})
    _write_json_atomic(root / 'controller_state.json', {'schema':SCHEMA, 'status':'running', 'attempt_count':0})
    try:
        if mlflow_reporter is not None:
            parent_id = mlflow_reporter.start_controller(
                config_sha256=reference['config_sha256'], model=reference['model'],
                provider=provider, train_sha256=reference['train_sha256'],
                validation_sha256=reference['validation_sha256'],
                max_attempts=max_trials, fault_injected=fault_inject_overfit)
            _audit(journal, 'mlflow_controller_created', run_id=parent_id)
        if trainer_fn is None:
            from .lora_gpu import run_lora
            trainer_fn = run_lora
        for i in range(1, max_trials + 1):
            name = f'trial-{i:02d}'
            config_path = root / f'{name}.yaml'
            out = root / name
            _write_yaml_exclusive(config_path, cfg)
            plan = make_plan(config_path)
            for field in ('model','revision','train_sha256','validation_sha256','device','quantization'):
                if plan[field] != reference[field]:
                    raise LoraContractError(f'Trial provenance mismatch: {field}')
            _audit(journal, 'attempt_started', attempt=i, config_sha256=plan['config_sha256'])
            if mlflow_reporter is not None:
                trial_id = mlflow_reporter.start_trial(i, config_sha256=plan['config_sha256'],
                                                       cfg=cfg, config_path=config_path)
                _audit(journal, 'mlflow_trial_created', attempt=i, run_id=trial_id)
            _write_json_atomic(root / 'controller_state.json', {'schema':SCHEMA, 'status':'running',
                                                                'active_attempt':i, 'attempt_count':i})
            # No prior checkpoint passed here. Changed LoRA settings require new trial.
            try:
                result = trainer_fn(str(config_path), str(out), allow_cpu=True,
                                    injected_diagnosis='overfit' if (i == 1 and fault_inject_overfit) else None)
            except Exception as exc:
                _audit(journal, 'attempt_failed', attempt=i, error_type=type(exc).__name__)
                raise
            if not isinstance(result, dict) or result.get('ok') is not True:
                raise LoraContractError('Trainer did not return an approved completed result')
            score = result.get('best_val_loss')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise LoraContractError('Trainer produced missing/nonfinite best validation loss')
            # Verify *both* the final adapter and full checkpoint against this trial plan.
            verify_manifest(out / 'final_adapter', plan, kind='final_adapter')
            checkpoint = Path(result.get('best_checkpoint', '')).resolve()
            if not checkpoint.is_relative_to((out / 'checkpoints').resolve()):
                raise LoraContractError('Best checkpoint outside attempt directory')
            verify_manifest(checkpoint, plan, kind='trainer_checkpoint')
            reason = result.get('stop_diagnosis')
            if reason is not None and reason not in {'overfit', 'underfit', 'diverged'}:
                raise LoraContractError('Unrecognized training stop diagnosis')
            injected = bool(result.get('fault_injected', False))
            if injected and not (i == 1 and fault_inject_overfit and reason == 'overfit'):
                raise LoraContractError('Trainer reported unauthorized fault injection')
            receipt = {'attempt':i, 'config_sha256':plan['config_sha256'],
                       'best_val_loss':float(score), 'stop_diagnosis':reason,
                       'fault_injected':injected, 'best_checkpoint':str(checkpoint),
                       'final_adapter':str(out / 'final_adapter')}
            attempts.append(receipt)
            _write_json_atomic(root / f'{name}.receipt.json', receipt)
            _audit(journal, 'attempt_verified', attempt=i, best_val_loss=float(score),
                   stop_diagnosis=reason, fault_injected=injected)
            if mlflow_reporter is not None:
                mlflow_reporter.finish_trial(i, out=out, receipt_path=root / f'{name}.receipt.json',
                    checkpoint=checkpoint, score=float(score), diagnosis=reason, fault_injected=injected)
            if reason is None:
                break  # Healthy completion, no speculative retries.
            if reason not in {'overfit', 'underfit'}:
                _audit(journal, 'blocked_unsupported_diagnosis', attempt=i, diagnosis=reason)
                break
            if i == max_trials:
                _audit(journal, 'retry_budget_exhausted', attempt=i)
                break
            history = _history(out / 'metrics.jsonl')
            if advisor_fn is None:
                advice = advise(reason, history, cfg['train'], provider=provider,
                                model=cfg['meta_agent']['model'])
            else:
                advice = advisor_fn(reason, history, cfg['train'], provider=provider,
                                    model=cfg['meta_agent']['model'])
            if advice.status not in {'approved','fallback'} or advice.diagnosis != reason:
                raise LoraContractError('Advisor response not approved for observed diagnosis')
            patch = validate_patch(advice.patch, cfg['train'], reason)
            proposed = apply_patch(cfg, patch)
            _write_json_atomic(root / f'{name}.advisor_decision.json', {
                'schema':SCHEMA, 'attempt':i, 'source_config_sha256':plan['config_sha256'],
                'diagnosis':reason, 'fault_injected':injected, 'provider':advice.provider,
                'source':advice.source, 'status':advice.status, 'patch':patch,
                'proposed_config_sha256':config_sha256(proposed)})
            _audit(journal, 'config_patch_approved', attempt=i, keys=sorted(patch),
                   fault_injected=injected)
            cfg = proposed
        if not attempts:
            raise LoraContractError('No verified training attempts')
        best = min(attempts, key=lambda a: (a['best_val_loss'], a['attempt']))
        result = {'ok':True, 'schema':SCHEMA, 'status':'completed',
                  'mode':'REAL HF PEFT LoRA CPU; bounded controller',
                  'attempt_count':len(attempts), 'best_trial':best['attempt'],
                  'training_outcome':'finished' if attempts[-1]['stop_diagnosis'] is None else 'stopped_without_further_retry',
                  'promotion_approved':False,
                  'best_val_loss':best['best_val_loss'], 'best_checkpoint':best['best_checkpoint'],
                  'attempts':attempts, 'fault_injection_used':any(a['fault_injected'] for a in attempts),
                  'note':'Validation comparison on toy data, not a model-quality or deployment gate.'}
        _write_json_atomic(root / 'controller_result.json', result)
        _write_json_atomic(root / 'controller_state.json', {'schema':SCHEMA, 'status':'completed',
                            'attempt_count':len(attempts),'best_trial':best['attempt']})
        _audit(journal, 'controller_completed', best_trial=best['attempt'])
        if mlflow_reporter is not None:
            mlflow_reporter.finish_controller(result_path=root / 'controller_result.json',
                best_trial=best['attempt'], best_val_loss=best['best_val_loss'], event_log=journal)
        return result
    except BaseException as exc:
        _write_json_atomic(root / 'controller_state.json', {'schema':SCHEMA, 'status':'blocked',
                         'attempt_count':len(attempts),'error_type':type(exc).__name__,
                         'note':'Do not rerun into this workspace. Inspect journal and create a fresh experiment.'})
        _audit(journal, 'controller_blocked', error_type=type(exc).__name__)
        if mlflow_reporter is not None:
            # Best effort to close remote runs. Preserve the original error and
            # durable local BLOCKED journal even if MLflow is unavailable.
            try:
                for attempt in list(mlflow_reporter.trial_runs):
                    mlflow_reporter.fail_trial(attempt)
                mlflow_reporter.fail_controller()
            except Exception:
                pass
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='CPU-first real HF LoRA recovery orchestrator')
    parser.add_argument('--config', default='config/lora_selfheal_cpu.yaml')
    parser.add_argument('--workspace', default='outputs/hf-selfheal-cpu-01')
    parser.add_argument('--provider', choices=['mock','meta'], default='mock')
    parser.add_argument('--execute', action='store_true', help='Download model and run real HF training')
    parser.add_argument('--mlflow-uri', default=None,
                        help='Optional MLflow tracking server URI (example: http://127.0.0.1:5000)')
    parser.add_argument('--mlflow-experiment', default='AutoTune-AI-CPU',
                        help='Experiment name for optional MLflow tracking')
    parser.add_argument('--fault-inject-overfit', action='store_true',
                        help='CONTROLLED TEST ONLY: force first eval stop, label non-natural diagnosis')
    args = parser.parse_args()
    if args.fault_inject_overfit and not args.execute:
        parser.error('--fault-inject-overfit requires --execute')
    if args.execute:
        result = execute(args.config, args.workspace, provider=args.provider,
                         fault_inject_overfit=args.fault_inject_overfit,
                         mlflow_uri=args.mlflow_uri, mlflow_experiment=args.mlflow_experiment)
    else:
        result = preflight(args.config, provider=args.provider)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
