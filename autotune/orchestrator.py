"""Versioned, bounded train-diagnose-patch-retry controller."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import yaml
from .metrics import diagnose
from .config import load_config, config_sha256
from .mock_trainer import train as mock_train
from .policy import apply_patch, fallback_patch, validate_patch
from .meta_agent import propose
from .tracking import write_json, event


def persist_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf8')
    temporary.replace(path)


def run(config_path: str, mode: str='mock', workspace: str='runs', write_back: bool=True) -> dict:
    config = load_config(config_path, mode=mode)
    if mode not in {'mock','hf'}:
        raise ValueError('mode must be mock or hf')
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    experiment = root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = 0
    while experiment.exists():
        suffix += 1
        experiment = root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{suffix}')
    experiment.mkdir()
    write_json(experiment/'manifest.json', {'mode':mode, 'initial_config':config, 'config_sha256':config_sha256(config), 'schema_contract':'phase2-v0.3'})
    audit = experiment/'events.jsonl'
    max_trials = max(1, min(12, int(config['safety']['max_trials'])))
    best = None
    current = deepcopy(config)
    attempts = []
    for trial in range(1, max_trials+1):
        trial_dir = experiment/f'trial-{trial:02d}'
        persist_yaml(trial_dir/'config.yaml',current)
        event(audit,{'type':'started','trial':trial,'config':current['train']})
        try:
            if mode == 'mock':
                result = mock_train(current,trial_dir,trial)
            else:
                from .hf_trainer import train as hf_train
                result = hf_train(current,trial_dir,trial)
        except Exception as exc:
            event(audit,{'type':'failed','trial':trial,'error':str(exc)})
            raise
        record = {'trial':trial,'state':result.state,'reason':result.reason,
                  'best_val_loss':result.best_val_loss,'best_step':result.best_step,
                  'checkpoint':result.checkpoint,'train':deepcopy(current['train'])}
        attempts.append(record)
        event(audit,{'type':'finished',**record})
        if best is None or result.best_val_loss < best['best_val_loss']:
            best = record
            persist_yaml(experiment/'best_config.yaml',current)
        if result.state == 'completed':
            print('Completed without early stop; retaining global best for promotion review.')
            break
        if trial == max_trials:
            print('Retry budget reached. Exiting.')
            break
        patch = None
        via = 'deterministic-policy'
        if current['meta_agent'].get('enabled'):
            suggestion = propose(result.state,result.history,current['train'],current['meta_agent']['model'])
            try:
                patch = validate_patch(suggestion,current['train'],result.state)
                via = 'meta-muse-spark'
            except (ValueError,TypeError) as exc:
                print('Meta suggestion rejected:', exc)
        if patch is None:
            try:
                patch = validate_patch(fallback_patch(result.state,current['train']),current['train'],result.state)
            except ValueError as exc:
                event(audit,{'type':'halted','reason':str(exc)})
                break
        event(audit,{'type':'configuration_patch','trial':trial,'source':via,'patch':patch})
        current = apply_patch(current,patch)
        persist_yaml(experiment/f'next_config_after_trial_{trial:02d}.yaml',current)
    # Restore the best-performing configuration, not necessarily the last attempt.
    best_cfg = yaml.safe_load((experiment/'best_config.yaml').read_text(encoding='utf8'))
    persist_yaml(experiment/'active.yaml',best_cfg)
    if write_back:
        # Versioned, reviewable source configuration; baseline remains unchanged.
        optimized = Path(config_path).resolve().parent/'optimized.yaml'
        persist_yaml(optimized, best_cfg)
        event(audit, {'type':'writeback','path':str(optimized),'best_trial':best['trial']})
    write_json(experiment/'summary.json', {'best':best, 'attempts':attempts,
              'note':'Selection uses minimum validation loss. Test-set evaluation and human approval are required before promotion.'})
    event(audit,{'type':'finished_experiment','best_trial':best['trial'],'best_val_loss':best['best_val_loss']})
    print(json.dumps({'experiment':str(experiment),'best_trial':best['trial'],
                      'best_val_loss':best['best_val_loss']},indent=2))
    return {'experiment':str(experiment),'best':best,'attempts':attempts}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='config/baseline.yaml')
    p.add_argument('--mode',choices=('mock','hf'),default='mock')
    p.add_argument('--workspace',default='runs')
    p.add_argument('--no-writeback', action='store_true', help='Do not write optimized.yaml beside input config')
    a=p.parse_args()
    run(a.config,a.mode,a.workspace,not a.no_writeback)

if __name__=='__main__':
    main()
