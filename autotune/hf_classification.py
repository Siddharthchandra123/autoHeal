"""CPU-first Hugging Face sequence classification task adapter.

Plan and dataset validation work offline. Live training requires optional HF packages.
Model-generated instructions are never evaluated. This module does NOT implement
an autonomous recovery loop; it records metrics and saves auditable checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone

import yaml

from .metrics import MonitorStream


class HFClassificationError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _file(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise HFClassificationError(f'Input file missing or unsafe: {path}')
    return path.resolve()


def _rows(path: Path) -> tuple[list[dict], set[str]]:
    path = _file(path)
    rows, hashes = [], set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HFClassificationError(f'{path}:{line_number}: invalid JSON') from exc
            if not isinstance(record, dict) or not isinstance(record.get('text'), str) or not record['text'].strip():
                raise HFClassificationError(f'{path}:{line_number}: nonempty text required')
            label = record.get('label')
            if isinstance(label, bool) or not isinstance(label, (str, int)) or not str(label).strip():
                raise HFClassificationError(f'{path}:{line_number}: string/integer label required')
            if len(record['text']) > 40000:
                raise HFClassificationError('Row text exceeds configured safety cap')
            canon = {'text': record['text'].strip(), 'label': str(label)}
            digest = hashlib.sha256(canon['text'].encode('utf-8')).hexdigest()
            if digest in hashes:
                raise HFClassificationError(f'{path}: repeated text within split')
            hashes.add(digest)
            rows.append(canon)
    if len(rows) < 2:
        raise HFClassificationError(f'{path}: at least 2 records required')
    return rows, hashes


def _integer(value, name, low, high):
    if isinstance(value, bool) or type(value) is not int or not low <= value <= high:
        raise HFClassificationError(f'{name} must be integer between {low} and {high}')
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise HFClassificationError(f'{name} must be finite between {low} and {high}')
    return float(value)


def classification_plan(config_path: str | Path) -> dict:
    """No network or ML dependencies; explicit classification schema and provenance."""
    path = _file(Path(config_path))
    try:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise HFClassificationError('Invalid YAML') from exc
    if not isinstance(cfg, dict) or set(cfg) != {'task', 'model', 'data', 'train', 'monitor'}:
        raise HFClassificationError('Expected task, model, data, train, monitor sections only')
    if cfg['task'] != 'text-classification':
        raise HFClassificationError('Task must be text-classification')
    model, data, train = cfg['model'], cfg['data'], cfg['train']
    if not all(isinstance(x, dict) for x in (model, data, train, cfg['monitor'])):
        raise HFClassificationError('Model, data, train and monitor must be mappings')
    if set(model) != {'name', 'revision', 'max_length'}:
        raise HFClassificationError('Invalid model fields')
    name, revision = model['name'], model['revision']
    # Repository identifier only: never interpret arbitrary local paths as model code.
    model_pat = r'[a-zA-Z0-9][\w.-]{0,95}(?:/[a-zA-Z0-9][\w.-]{0,95})?'
    if not isinstance(name, str) or not re.fullmatch(model_pat, name) or '..' in name:
        raise HFClassificationError('Use a Hugging Face repository ID, not a filesystem path')
    if not isinstance(revision, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._/-]{0,127}', revision) or '..' in revision:
        raise HFClassificationError('Invalid model revision')
    max_length = _integer(model['max_length'], 'model.max_length', 16, 512)
    if set(data) != {'train_file', 'validation_file'}:
        raise HFClassificationError('Expected train_file and validation_file')
    train_file = _file(Path(data['train_file']))
    val_file = _file(Path(data['validation_file']))
    if train_file == val_file:
        raise HFClassificationError('Train and validation files cannot be identical')
    train_rows, train_hashes = _rows(train_file)
    val_rows, val_hashes = _rows(val_file)
    if train_hashes & val_hashes:
        raise HFClassificationError('Train/validation leakage: overlapping text')
    labels = sorted({row['label'] for row in train_rows})
    if not 2 <= len(labels) <= 100:
        raise HFClassificationError('Requires 2-100 distinct training labels')
    if any(row['label'] not in labels for row in val_rows):
        raise HFClassificationError('Validation includes unseen label')
    expected_train = {'learning_rate', 'weight_decay', 'batch_size', 'max_steps', 'eval_steps', 'seed'}
    if set(train) != expected_train:
        raise HFClassificationError(f'Expected train fields: {sorted(expected_train)}')
    _number(train['learning_rate'], 'train.learning_rate', 1e-7, 0.1)
    _number(train['weight_decay'], 'train.weight_decay', 0, 0.5)
    batch = _integer(train['batch_size'], 'train.batch_size', 1, 32)
    max_steps = _integer(train['max_steps'], 'train.max_steps', 1, 5000)
    eval_steps = _integer(train['eval_steps'], 'train.eval_steps', 1, max_steps)
    _integer(train['seed'], 'train.seed', 0, 2**31 - 1)
    if len(cfg['monitor']) > 20:
        raise HFClassificationError('Monitor mapping too large')
    # Known monitor keys validated by the existing MonitorStream contract.
    allowed = {'min_evaluations', 'patience', 'min_delta', 'generalization_gap', 'underfit_loss_floor',
               'flat_window', 'flat_threshold', 'loss_spike_ratio', 'loss_spike_patience', 'grad_norm_warning'}
    mandatory = {'min_evaluations', 'patience', 'min_delta', 'generalization_gap',
                 'underfit_loss_floor', 'flat_window', 'flat_threshold'}
    if set(cfg['monitor']) - allowed or not mandatory <= set(cfg['monitor']):
        raise HFClassificationError('Unknown or missing monitor settings')
    # Keep the monitoring engine's integer and numeric contracts separate.
    integer_monitor = {'min_evaluations', 'patience', 'flat_window', 'loss_spike_patience'}
    for key, val in cfg['monitor'].items():
        if key in integer_monitor:
            _integer(val, f'monitor.{key}', 1 if key != 'loss_spike_patience' else 2, 10000)
        else:
            _number(val, f'monitor.{key}', 0, 10000)
    if cfg['monitor'].get('loss_spike_ratio', 2.0) <= 1:
        raise HFClassificationError('monitor.loss_spike_ratio must exceed 1')
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode('utf-8')).hexdigest()
    return {'schema': 'autotune-hf-classification-v0.8', 'task': 'text-classification',
            'model': name, 'revision': revision, 'max_length': max_length,
            'config_sha256': cfg_sha, 'train_sha256': _sha(train_file),
            'validation_sha256': _sha(val_file), 'train_file': str(train_file),
            'validation_file': str(val_file), 'train_rows': len(train_rows),
            'validation_rows': len(val_rows), 'labels': labels,
            'train': train, 'monitor': cfg['monitor'], 'device': 'cpu',
            'training_method': 'full-finetune', 'estimate_note':
            'CPU pilot: runtime and memory depend on selected model; no GPU required.'}


def _manifest_files(folder: Path) -> dict:
    if folder.is_symlink() or not folder.is_dir():
        raise HFClassificationError('Missing or unsafe artifact directory')
    files = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise HFClassificationError('Symlink in artifacts')
        if path.is_file() and path.name != 'autotune_manifest.json':
            files[path.relative_to(folder).as_posix()] = _sha(path)
    return files


def save_manifest(folder: Path, plan: dict, *, kind: str):
    files = _manifest_files(folder)
    if kind == 'model' and not (('model.safetensors' in files or 'pytorch_model.bin' in files) and 'config.json' in files):
        raise HFClassificationError('Missing model weights/config')
    if kind == 'checkpoint' and (not {'trainer_state.json', 'optimizer.pt', 'scheduler.pt'} <= files.keys()
            or not any(name.startswith('rng_state') and name.endswith('.pth') for name in files)):
        raise HFClassificationError('Incomplete Trainer checkpoint (including RNG state)')
    if kind not in {'checkpoint', 'model'}:
        raise HFClassificationError('Invalid artifact kind')
    manifest = {'schema': plan['schema'], 'kind': kind, 'model': plan['model'],
                'config_sha256': plan['config_sha256'], 'train_sha256': plan['train_sha256'],
                'validation_sha256': plan['validation_sha256'], 'files': files}
    dest = folder / 'autotune_manifest.json'
    if dest.exists():
        dest.unlink()  # only called after trainer has already completed saving on this owned run
    _atomic_json(dest, manifest)
    return manifest


def verify_manifest(folder: str | Path, plan: dict, kind: str):
    manifest_path = _file(Path(folder) / 'autotune_manifest.json')
    try:
        content = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise HFClassificationError('Malformed manifest') from exc
    for key, expected in [('schema', plan['schema']), ('model', plan['model']),
                          ('config_sha256', plan['config_sha256']), ('train_sha256', plan['train_sha256']),
                          ('validation_sha256', plan['validation_sha256']), ('kind', kind)]:
        if content.get(key) != expected:
            raise HFClassificationError(f'Manifest mismatch: {key}')
    current_files = _manifest_files(Path(folder))
    if content.get('files') != current_files:
        raise HFClassificationError('Artifact integrity mismatch')
    if kind == 'checkpoint' and (not {'trainer_state.json', 'optimizer.pt', 'scheduler.pt'} <= current_files.keys()
            or not any(name.startswith('rng_state') and name.endswith('.pth') for name in current_files)):
        raise HFClassificationError('Checkpoint is missing optimizer, scheduler or RNG state')
    if kind == 'model' and not (('model.safetensors' in current_files or 'pytorch_model.bin' in current_files)
            and 'config.json' in current_files):
        raise HFClassificationError('Model weights/config are missing')
    return {'ok': True, 'kind': kind, 'file_count': len(content['files']), 'model': plan['model']}


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(row) for row in Path(path).read_text('utf-8').splitlines() if row.strip()]


def _event(path: Path, record: dict):
    safe = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in record.items()}
    safe['time_utc'] = datetime.now(timezone.utc).isoformat()
    with path.open('a', encoding='utf8') as handle:
        handle.write(json.dumps(safe, sort_keys=True, allow_nan=False) + '\n')


def run_classification(config_path: str, output: str, *, resume: str | None = None) -> dict:
    plan = classification_plan(config_path)
    out = Path(output).resolve()
    if out.is_symlink():
        raise HFClassificationError('Output symlink refused')
    checkpoint = None
    if resume:
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_relative_to(out / 'checkpoints'):
            raise HFClassificationError('Resume path outside run directory')
        verify_manifest(checkpoint, plan, 'checkpoint')
        if not (out / 'plan.json').is_file() or json.loads((out / 'plan.json').read_text('utf8')) != plan:
            raise HFClassificationError('Resume plan mismatch')
    elif out.exists() and any(out.iterdir()):
        raise HFClassificationError('Nonempty output: select new directory or verified resume')

    # Deferred optional dependencies: offline preflight never downloads weights.
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback)

    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.hf_classification.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise HFClassificationError('Training run already locked') from exc
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        if not (out / 'plan.json').exists():
            _atomic_json(out / 'plan.json', plan)
        _atomic_json(out / 'run_state.json', {'status': 'running'}) if not (out / 'run_state.json').exists() else None
        labels = plan['labels']
        label2id = {name: index for index, name in enumerate(labels)}
        id2label = {index: name for name, index in label2id.items()}
        tokenizer = AutoTokenizer.from_pretrained(plan['model'], revision=plan['revision'], trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(plan['model'], revision=plan['revision'],
            num_labels=len(labels), id2label=id2label, label2id=label2id, trust_remote_code=False)

        def transform(path):
            rows = _read_jsonl(path)
            normalized = [{'text': r['text'], 'labels': label2id[str(r['label'])]} for r in rows]
            ds = Dataset.from_list(normalized)
            return ds.map(lambda batch: tokenizer(batch['text'], truncation=True,
                max_length=plan['max_length']), batched=True, remove_columns=['text'])

        train_set, val_set = transform(plan['train_file']), transform(plan['validation_file'])
        monitor = MonitorStream(out / 'monitor_decisions.jsonl', plan['monitor'])

        class Callback(TrainerCallback):
            def __init__(self):
                self.train_window = []
                self.last_step = 0
                self.stop_reason = None
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    loss = float(logs['loss'])
                    self.train_window.append(loss)
                    _event(out / 'training_logs.jsonl', {'step': int(state.global_step), 'train_loss': loss})
                return control
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and 'eval_loss' in metrics:
                    val = float(metrics['eval_loss'])
                    _event(out / 'metrics.jsonl', {'step': int(state.global_step),
                        'val_loss': val, 'accuracy': float(metrics.get('eval_accuracy', 0))})
                    if self.train_window:
                        row = {'step': int(state.global_step), 'train_loss': sum(self.train_window) / len(self.train_window), 'val_loss': val}
                        decision = monitor.observe(row)
                        _event(out / 'decisions.jsonl', {'step': row['step'], 'diagnosis': decision.state,
                            'action': decision.action})
                        self.train_window.clear()
                        if decision.should_stop:
                            self.stop_reason = decision.state
                            control.should_training_stop = True
                            control.should_save = True
                return control
            def on_save(self, args, state, control, **kwargs):
                ckpt = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                save_manifest(ckpt, plan, kind='checkpoint')
                return control

        params = plan['train']
        args = TrainingArguments(output_dir=str(out / 'checkpoints'), max_steps=params['max_steps'],
            per_device_train_batch_size=params['batch_size'], per_device_eval_batch_size=params['batch_size'],
            learning_rate=params['learning_rate'], weight_decay=params['weight_decay'],
            eval_strategy='steps', save_strategy='steps', logging_strategy='steps',
            eval_steps=params['eval_steps'], save_steps=params['eval_steps'],
            logging_steps=params['eval_steps'], load_best_model_at_end=True,
            metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=3,
            report_to='none', seed=params['seed'], use_cpu=True, save_only_model=False)
        def compute_metrics(pred):
            logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            return {'accuracy': float((np.argmax(logits, axis=-1) == pred.label_ids).mean())}
        callback = Callback()
        trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set,
            processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[callback], compute_metrics=compute_metrics)
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        checkpoint_best = trainer.state.best_model_checkpoint
        if not checkpoint_best:
            raise HFClassificationError('No best checkpoint: check eval_steps and max_steps')
        verify_manifest(checkpoint_best, plan, 'checkpoint')
        model_dir = out / 'best_model'
        if model_dir.exists():
            raise HFClassificationError('Best model directory exists')
        model_dir.mkdir()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        manifest = save_manifest(model_dir, plan, kind='model')
        best_loss = trainer.state.best_metric
        if isinstance(best_loss, bool) or not isinstance(best_loss, (int, float)) or not math.isfinite(best_loss):
            raise HFClassificationError('Missing or invalid best validation loss')
        result = {'ok': True, 'task': 'text-classification', 'device': 'cpu',
                  'model': plan['model'], 'best_val_loss': float(best_loss),
                  'best_checkpoint': str(checkpoint_best), 'model_dir': str(model_dir),
                  'label_mapping': label2id, 'checkpoint_verified': True,
                  'model_sha256': manifest['files'].get('model.safetensors'),
                  'note': 'Model trained on CPU. No autonomous agent retraining or Hub upload.'}
        _atomic_json(out / 'result_receipt.json', result)
        (out / 'run_state.json').unlink(missing_ok=True)
        _atomic_json(out / 'run_state.json', {'status': 'completed'})
        return result
    except BaseException as exc:
        (out / 'run_state.json').unlink(missing_ok=True)
        _atomic_json(out / 'run_state.json', {'status': 'blocked', 'error_type': type(exc).__name__})
        raise
    finally:
        lock.unlink(missing_ok=True)
