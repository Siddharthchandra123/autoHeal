"""End-to-end Knowledge Engine -> SFT -> sequential model selection orchestrator.

Default mode is preflight. Execution downloads models and performs real training
only when the CLI receives --execute. No automatic release/promotion is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .config import config_sha256
from .lora_contract import LoraContractError, make_plan, _write_json_atomic
from .model_evaluator import evaluate_candidate

BASE = Path(__file__).resolve().parents[1]
REGISTRY = BASE / 'config' / 'knowledge_candidates.json'
BASE_CONFIG = BASE / 'config' / 'lora_public_cpu.yaml'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_registry(path: str | Path = REGISTRY) -> dict:
    body = json.loads(Path(path).read_text(encoding='utf-8'))
    if body.get('schema') != 'autotune-knowledge-candidates-v1' or not isinstance(body.get('candidates'), list):
        raise ValueError('Invalid candidate registry')
    candidates = [x for x in body['candidates'] if x.get('enabled') is True]
    if not candidates:
        raise ValueError('No enabled models in candidate registry')
    for c in candidates:
        if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,80}', c.get('id','')):
            raise ValueError('Invalid candidate id')
        if not re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', c.get('model','')):
            raise ValueError(f"Invalid model ID: {c.get('model')}")
        if c.get('device') not in {'cpu','gpu'} or c.get('quantization') not in {'none','4bit'}:
            raise ValueError('Invalid candidate compute settings')
    return body


def _dataset_paths(dataset_dir: str | Path) -> dict:
    supplied = Path(dataset_dir)
    if supplied.is_symlink():
        raise ValueError('Dataset directory cannot be a symlink')
    root = supplied.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'Dataset directory missing or unsafe: {root}')
    manifest_path = root / 'dataset_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Dataset manifest missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'autotune-knowledge-dataset-v1' or
            manifest.get('source_disjoint') is not True or manifest.get('data_format') != 'messages'):
        raise ValueError('Dataset must be a source-disjoint structured-messages Knowledge Engine dataset')
    files = {name: root / f'{name}.jsonl' for name in ('train','validation','test')}
    if any(not p.is_file() or p.is_symlink() for p in files.values()):
        raise ValueError('Train, validation and test JSONL files are required')
    source_sets, counts = {}, {}
    content_hashes = {}
    declared_counts = manifest.get('counts')
    declared_hashes = manifest.get('file_hashes')
    if not isinstance(declared_counts, dict) or not isinstance(declared_hashes, dict):
        raise ValueError('Dataset manifest must include split counts and file hashes')
    for split, path in files.items():
        sources, digests, rows = set(), set(), 0
        with path.open('r', encoding='utf-8') as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip(): continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('messages'), list) or len(row['messages']) < 2:
                    raise ValueError(f'{path}:{line_no}: invalid structured SFT record')
                messages = row['messages']
                if (any(not isinstance(m, dict) or m.get('role') not in {'system','user','assistant'} or
                        not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages) or
                        messages[-1].get('role') != 'assistant' or not any(m.get('role') == 'user' for m in messages)):
                    raise ValueError(f'{path}:{line_no}: malformed conversation messages')
                source_ids = row.get('source_ids')
                if not isinstance(source_ids, list) or not source_ids or any(not isinstance(x,str) or not x for x in source_ids):
                    raise ValueError(f'{path}:{line_no}: source_ids are required for leakage audit')
                sources.update(source_ids)
                digest = hashlib.sha256(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if digest in digests: raise ValueError(f'{path}: duplicate messages within split')
                digests.add(digest); rows += 1
        if rows < 2: raise ValueError(f'{path}: each split requires at least two examples')
        expected = declared_counts.get(split)
        if not isinstance(expected, int) or expected != rows:
            raise ValueError(f'{path}: manifest row count mismatch')
        expected_hash = declared_hashes.get(split)
        if not isinstance(expected_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_hash):
            raise ValueError(f'{path}: missing/invalid manifest file hash')
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f'{path}: split file hash mismatch')
        source_sets[split], counts[split], content_hashes[split] = sources, rows, digests
    # source_units is the leakage boundary. It can be a source document for
    # generated Q/A or an individual record for a one-file supervised dataset
    # such as Dolly. Older manifests used source_documents for document IDs.
    declared = manifest.get('source_units', manifest.get('source_documents'))
    if not isinstance(declared, dict):
        raise ValueError('Dataset source-unit provenance is required')
    normalized = {k:set(v) for k,v in declared.items() if k in {'train','validation','test'} and isinstance(v,list)}
    if len(normalized) != 3 or any(normalized[k] != source_sets[k] for k in ('train','validation','test')):
        raise ValueError('Dataset source-unit provenance does not match JSONL records')
    split_names = ('train','validation','test')
    for i, left in enumerate(split_names):
        for right in split_names[i+1:]:
            if source_sets[left] & source_sets[right]:
                raise ValueError(f'Source-unit leakage between {left} and {right}')
            if content_hashes[left] & content_hashes[right]:
                raise ValueError(f'Example leakage between {left} and {right}')
    return {'root':root, 'manifest':manifest, 'source_unit_kind':manifest.get('source_unit_kind', 'document'), **files, 'counts':counts}


def _pin_revision(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError('Install requirements-lora.txt to resolve immutable Hub revisions') from exc
    info = HfApi().model_info(model_id)
    sha = getattr(info, 'sha', None)
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise RuntimeError(f'Could not resolve an immutable commit for {model_id}')
    return sha


def _make_config(candidate: dict, defaults: dict, data: dict, revision: str,
                 learning_rate: float, output: Path) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))
    cfg['project'] = f"AutoTune-Knowledge-{candidate['id']}"
    cfg['model']['name'] = candidate['model']
    cfg['model']['max_length'] = int(defaults['max_length'])
    cfg['data']['train_file'] = str(data['train'])
    cfg['data']['validation_file'] = str(data['validation'])
    cfg['train'].update({
        'learning_rate': float(learning_rate), 'num_train_epochs': int(defaults['epochs']),
        'per_device_train_batch_size': int(defaults['batch_size']),
        'gradient_accumulation_steps': int(defaults['gradient_accumulation_steps']),
        'eval_steps': int(defaults['eval_steps']), 'lora_r': int(defaults['lora_r']),
        'lora_alpha': int(defaults['lora_alpha']), 'lora_dropout': float(defaults['lora_dropout']),
    })
    cfg['safety']['max_trials'] = len(defaults['learning_rates'])
    cfg['hf'].update({
        'device':candidate['device'], 'quantization':candidate['quantization'],
        'target_modules':candidate['target_modules'], 'revision':revision,
        'max_steps':int(defaults['max_steps']), 'seed':42,
        'loss_mode':'response_only', 'chat_template_kwargs':candidate.get('chat_template_kwargs',{}),
    })
    path = output / 'configs' / f"{candidate['id']}-lr-{str(learning_rate).replace('.','_')}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return path


def preflight(dataset_dir: str | Path, *, output_dir: str | Path = 'outputs/knowledge-run',
              registry_path: str | Path = REGISTRY) -> dict:
    data = _dataset_paths(dataset_dir)
    registry = _load_registry(registry_path)
    defaults = registry['defaults']
    requirements = {'train':int(defaults.get('minimum_train_rows',100)),
                    'validation':int(defaults.get('minimum_validation_rows',20)),
                    'test':int(defaults.get('minimum_test_rows',20))}
    data_issues = [f"{split}: {data['counts'][split]} rows; minimum {minimum}"
                   for split,minimum in requirements.items() if data['counts'][split] < minimum]
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Preflight output directory is nonempty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    candidate_plans = []
    for candidate in registry['candidates']:
        for lr in defaults['learning_rates']:
            cfg_path = _make_config(candidate, defaults, data, 'main', float(lr), out)
            plan = make_plan(cfg_path)
            candidate_plans.append({'candidate_id':candidate['id'], 'model':candidate['model'],
                'revision':'main (not pinned in offline preflight)', 'learning_rate':float(lr),
                'config':str(cfg_path), 'plan':plan})
    result = {'ok':True, 'mode':'PREFLIGHT_ONLY', 'dataset':str(data['root']),
        'dataset_counts':data['counts'], 'data_quality_gate':{'ready':not data_issues,'requirements':requirements,'issues':data_issues},
        'candidate_count':len(registry['candidates']),
        'trial_count':len(candidate_plans), 'device_policy':'sequential CPU; one model loaded at a time',
        'candidates':candidate_plans,
        'promotion_approved':False,
        'note':'No model weights were downloaded and no training was performed. Execute mode resolves immutable model commits and then runs sequentially.'}
    _write_json_atomic(out / 'preflight.json', result)
    return result


def execute(dataset_dir: str | Path, *, output_dir: str | Path = 'outputs/knowledge-run',
            registry_path: str | Path = REGISTRY, allow_cpu: bool = False,
            max_candidates: int | None = None, allow_small_data: bool = False) -> dict:
    data = _dataset_paths(dataset_dir)
    registry = _load_registry(registry_path)
    candidates = registry['candidates'][:max_candidates] if max_candidates else registry['candidates']
    defaults = registry['defaults']
    requirements = {'train':int(defaults.get('minimum_train_rows',100)),
                    'validation':int(defaults.get('minimum_validation_rows',20)),
                    'test':int(defaults.get('minimum_test_rows',20))}
    data_issues = [f"{split}: {data['counts'][split]} rows; minimum {minimum}"
                   for split,minimum in requirements.items() if data['counts'][split] < minimum]
    if data_issues and not allow_small_data:
        raise ValueError('Automated training data-quality gate blocked execution: ' + '; '.join(data_issues) + '. Use --allow-small-data only for a smoke test, not quality claims.')
    if not allow_cpu and any(x['device'] == 'cpu' for x in candidates):
        raise ValueError('CPU model execution requires explicit --allow-cpu')
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Experiment output directory is nonempty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    journal = out / 'events.jsonl'
    results = []
    validation_file = str(data['validation'])
    test_file = str(data['test'])
    for candidate in candidates:
        candidate_record = {'candidate_id':candidate['id'], 'model':candidate['model'], 'trials':[], 'status':'running'}
        try:
            revision = _pin_revision(candidate['model'])
            candidate_record['revision'] = revision
            _append(journal, {'event':'candidate_started','candidate':candidate['id'],'revision':revision})
            trained = []
            for index, lr in enumerate(defaults['learning_rates'], 1):
                cfg_path = _make_config(candidate, defaults, data, revision, float(lr), out)
                plan = make_plan(cfg_path)
                trial_dir = out / 'runs' / candidate['id'] / f'trial-{index:02d}'
                trial_dir.parent.mkdir(parents=True, exist_ok=True)
                _append(journal, {'event':'training_started','candidate':candidate['id'],'trial':index,
                                  'learning_rate':float(lr),'config_sha256':plan['config_sha256']})
                from .lora_gpu import run_lora
                run = run_lora(str(cfg_path), str(trial_dir), allow_cpu=(candidate['device']=='cpu'))
                if not run.get('ok'):
                    raise RuntimeError('LoRA trainer did not return a completed result')
                # Select LR/trial using validation only. The held-out test split is
                # not touched until one architecture has been selected.
                evaluation = evaluate_candidate(model_id=candidate['model'], revision=revision,
                    adapter_dir=run['adapter_dir'], test_file=validation_file,
                    max_examples=int(defaults['max_test_examples']),
                    chat_template_kwargs=candidate.get('chat_template_kwargs',{}))
                trial = {'trial':index,'learning_rate':float(lr),'best_val_loss':run['best_val_loss'],
                    'config_path':str(cfg_path),'run_dir':str(trial_dir),'run':run,'revision':revision,
                    'validation_evaluation':evaluation,'adapter_dir':run['adapter_dir']}
                trained.append(trial)
                _write_json_atomic(trial_dir / 'knowledge_evaluation.json', evaluation)
                candidate_record['trials'].append(trial)
                _append(journal, {'event':'trial_evaluated','candidate':candidate['id'],'trial':index,
                    'split':'validation','base_f1':evaluation['base']['mean_token_f1'],
                    'candidate_f1':evaluation['candidate']['mean_token_f1'],
                    'delta':evaluation['token_f1_delta']})
            best = max(trained, key=lambda x:(x['validation_evaluation']['candidate']['mean_token_f1'], -x['best_val_loss']))
            base_f1 = best['validation_evaluation']['base']['mean_token_f1']
            candidate_f1 = best['validation_evaluation']['candidate']['mean_token_f1']
            delta = best['validation_evaluation']['token_f1_delta']
            qualifies = candidate_f1 >= float(defaults['minimum_token_f1']) and delta >= float(defaults['minimum_improvement'])
            candidate_record.update({'status':'evaluated','best_trial':best['trial'],
                'validation_base_token_f1':base_f1,'validation_candidate_token_f1':candidate_f1,
                'validation_token_f1_delta':delta,'qualified_on_validation':qualifies,
                'best_adapter_dir':best['adapter_dir'],'best_config_path':best['config_path'],
                'best_run_dir':best['run_dir']})
        except Exception as exc:
            candidate_record.update({'status':'failed','qualified_on_validation':False,
                                     'error_type':type(exc).__name__, 'error':str(exc)[:500]})
            _append(journal, {'event':'candidate_failed','candidate':candidate['id'],'error_type':type(exc).__name__})
        results.append(candidate_record)
        _write_json_atomic(out / f"candidate-{candidate['id']}.json", candidate_record)
    eligible = [r for r in results if r.get('qualified_on_validation')]
    validation_winner = max(eligible, key=lambda x:(x['validation_candidate_token_f1'],x['validation_token_f1_delta'])) if eligible else None
    test_evaluation = None
    final_winner = None
    if validation_winner is not None:
        try:
            # Final test is evaluated once, after all trial/model selection is complete.
            test_evaluation = evaluate_candidate(
                model_id=validation_winner['model'],
                revision=validation_winner['revision'],
                adapter_dir=validation_winner['best_adapter_dir'], test_file=test_file,
                max_examples=int(defaults['max_test_examples']),
                chat_template_kwargs=next((c.get('chat_template_kwargs', {}) for c in candidates if c['id']==validation_winner['candidate_id']), {}))
            test_candidate_f1 = float(test_evaluation['candidate']['mean_token_f1'])
            test_delta = float(test_evaluation['token_f1_delta'])
            if test_candidate_f1 >= float(defaults['minimum_token_f1']) and test_delta >= float(defaults['minimum_improvement']):
                final_winner = validation_winner
            else:
                validation_winner['test_gate'] = {'passed':False,'candidate_token_f1':test_candidate_f1,
                    'token_f1_delta':test_delta,'reason':'Selected candidate did not meet final held-out test thresholds'}
            _append(journal, {'event':'final_test_evaluated','candidate':validation_winner['candidate_id'],
                'split':'test','candidate_f1':test_candidate_f1,'delta':test_delta,
                'passed':final_winner is not None})
        except Exception as exc:
            validation_winner['test_gate'] = {'passed':False,'error_type':type(exc).__name__,'error':str(exc)[:400]}
            _append(journal, {'event':'final_test_failed','candidate':validation_winner['candidate_id'],
                'error_type':type(exc).__name__})
    if validation_winner is not None:
        if final_winner is not None:
            final_winner['test_gate'] = {'passed':True,
                'candidate_token_f1':test_evaluation['candidate']['mean_token_f1'],
                'token_f1_delta':test_evaluation['token_f1_delta']}
            final_winner['final_test_evaluation'] = test_evaluation
        _write_json_atomic(out / f"candidate-{validation_winner['candidate_id']}.json", validation_winner)
        if test_evaluation is not None:
            _write_json_atomic(out / 'final_test_evaluation.json', test_evaluation)
    outcome = {'ok':True, 'schema':'autotune-knowledge-experiment-v1','status':'completed',
        'dataset':str(data['root']), 'dataset_counts':data['counts'],
        'data_quality_gate':{'passed':not data_issues,'smoke_override_used':bool(allow_small_data and data_issues),
                             'requirements':requirements,'issues':data_issues}, 'results':results,
        'test_evaluation':test_evaluation,
        'selection':({'status':'qualified_candidate_found','candidate_id':final_winner['candidate_id'],
                      'model':final_winner['model'],'adapter_dir':final_winner['best_adapter_dir'],
                      'validation_token_f1':final_winner['validation_candidate_token_f1'],
                      'test_token_f1':final_winner['test_gate']['candidate_token_f1'],
                      'test_improvement_over_base':final_winner['test_gate']['token_f1_delta'],
                      'config_path':final_winner['best_config_path'],
                      'run_dir':final_winner['best_run_dir'],
                      'ollama_export_next_step':'python -m autotune.ollama_bridge verify --config <config_path> --run <run_dir>; then use merge and Modelfile commands.'} if final_winner else
                     {'status':'NO_QUALIFIED_MODEL','note':'No candidate passed both validation selection and the single final held-out test gate.'}),
        'promotion_approved':False, 'production_eligible':False,
        'note':'Learning-rate and model selection use validation only. The final test split is evaluated once after selection. Token F1 remains a lexical overlap metric, not a release certification; inspect safety, calibration and domain accuracy before production.'}
    _write_json_atomic(out / 'experiment_result.json', outcome)
    return outcome


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='AutoTune Knowledge -> Autonomous HF fine-tuning')
    parser.add_argument('--dataset', required=True, help='Dataset folder from Knowledge Engine generator')
    parser.add_argument('--output', default='outputs/knowledge-run')
    parser.add_argument('--registry', default=str(REGISTRY))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true', help='Download models and run actual training/evaluation')
    group.add_argument('--preflight', action='store_true', help='Validate dataset and candidate plans only (default)')
    parser.add_argument('--allow-cpu', action='store_true', help='Explicitly authorize sequential CPU model training')
    parser.add_argument('--allow-small-data', action='store_true', help='Override row-count gate for smoke tests only; never implies production quality')
    parser.add_argument('--max-candidates', type=int)
    args = parser.parse_args()
    if args.max_candidates is not None and not 1 <= args.max_candidates <= 10:
        parser.error('--max-candidates must be between 1 and 10')
    try:
        result = execute(args.dataset, output_dir=args.output, registry_path=args.registry,
                         allow_cpu=args.allow_cpu, max_candidates=args.max_candidates,
                         allow_small_data=args.allow_small_data) if args.execute else preflight(
                         args.dataset, output_dir=args.output, registry_path=args.registry)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'ok':False,'error_type':type(exc).__name__,'error':str(exc)},indent=2), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
