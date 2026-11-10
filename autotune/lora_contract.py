"""Stage 5B: offline LoRA preflight and tamper-evident artifact manifests.

The preflight does not import Hugging Face or need network/GPU access.
Manifests detect accidental corruption; they are NOT cryptographic signatures.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .config import config_sha256, load_config


class LoraContractError(ValueError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise LoraContractError(f'{name} must be an integer in [{lo}, {hi}]')
    return value


def _safe_file(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise LoraContractError(f'Missing or unsafe input file: {path}')
    return path.resolve()


def _record_payload(obj: dict, path: Path, line: int) -> tuple[str, str]:
    if isinstance(obj.get('text'), str) and obj['text'].strip():
        return 'text', obj['text'].strip()
    messages = obj.get('messages')
    if isinstance(messages, list) and len(messages) >= 2:
        normalized = []
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in {'system','user','assistant'}:
                raise LoraContractError(f'{path} line {line}: messages must contain system/user/assistant roles')
            content = message.get('content')
            if not isinstance(content, str) or not content.strip():
                raise LoraContractError(f'{path} line {line}: message content must be nonempty')
            normalized.append({'role': message['role'], 'content': content.strip()})
        if normalized[-1]['role'] != 'assistant' or not any(x['role']=='user' for x in normalized):
            raise LoraContractError(f'{path} line {line}: final message must be assistant and a user message is required')
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return 'messages', payload
    raise LoraContractError(f'{path} line {line}: expected nonempty text or structured messages field')


def _sample_digest(path: Path) -> tuple[str, int, set[str]]:
    _safe_file(path)
    samples = set()
    rows = 0
    data_format = None
    with path.open('r', encoding='utf-8') as handle:
        for i, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LoraContractError(f'{path} line {i}: malformed JSON') from exc
            if not isinstance(obj, dict):
                raise LoraContractError(f'{path} line {i}: record must be a JSON object')
            fmt, payload = _record_payload(obj, path, i)
            if data_format is None:
                data_format = fmt
            elif data_format != fmt:
                raise LoraContractError(f'{path}: mixed text/messages formats are not allowed')
            rows += 1
            samples.add(hashlib.sha256(payload.encode('utf-8')).hexdigest())
    if rows < 2:
        raise LoraContractError(f'{path}: requires >= 2 records')
    if len(samples) != rows:
        raise LoraContractError(f'{path}: duplicate examples detected within split')
    return _hash(path), rows, samples


def _sample_format(path: Path) -> str:
    with path.open('r', encoding='utf-8') as handle:
        for i, line in enumerate(handle, 1):
            if line.strip():
                obj = json.loads(line)
                fmt, _ = _record_payload(obj, path, i)
                return fmt
    raise LoraContractError(f'{path}: empty dataset')


def make_plan(config_path: str | Path) -> dict:
    """Validate config and data; NO effects, downloads, or run-directory creation."""
    config_file = _safe_file(Path(config_path))
    cfg = load_config(config_file, mode='hf')
    name = cfg['model']['name']
    if name == 'autotune/tiny-byte-gru':
        raise LoraContractError('Stage 5B requires a Hugging Face causal LM, not the Phase 5A GRU')
    hf = cfg.get('hf', {})
    if not isinstance(hf, dict):
        raise LoraContractError('hf must be a mapping')
    unexpected = set(hf) - {'quantization', 'target_modules', 'revision', 'device', 'max_steps', 'seed', 'loss_mode', 'chat_template_kwargs'}
    if unexpected:
        raise LoraContractError(f'Unknown hf settings: {sorted(unexpected)}')
    quant = hf.get('quantization', 'none')
    if quant not in ('none', '4bit'):
        raise LoraContractError('hf.quantization must be none or 4bit')
    device = hf.get('device', 'gpu')
    if device not in ('gpu', 'cpu'):
        raise LoraContractError('hf.device must be gpu or cpu')
    if quant == '4bit' and device != 'gpu':
        raise LoraContractError('4-bit QLoRA requires supported GPU backend')
    if device == 'cpu' and name.startswith('meta-llama/'):
        raise LoraContractError('Meta Llama training in this pilot requires a compatible GPU')
    modules = hf.get('target_modules', ['q_proj', 'v_proj'])
    if not isinstance(modules, list) or not modules or len(modules) > 8 or any(
            not isinstance(x, str) or not x.isidentifier() for x in modules) or len(set(modules)) != len(modules):
        raise LoraContractError('hf.target_modules must be 1..8 distinct Python-style module names')
    revision = hf.get('revision', 'main')
    if not isinstance(revision, str) or not revision or len(revision) > 128:
        raise LoraContractError('hf.revision must be a nonempty repository revision')
    max_steps = _require_int(hf.get('max_steps', 200), 'hf.max_steps', 1, 100000)
    seed = _require_int(hf.get('seed', 42), 'hf.seed', 0, 2**31 - 1)
    loss_mode = hf.get('loss_mode', 'all_tokens')
    if loss_mode not in ('all_tokens', 'response_only'):
        raise LoraContractError('hf.loss_mode must be all_tokens or response_only')
    template_kwargs = hf.get('chat_template_kwargs', {})
    if not isinstance(template_kwargs, dict) or len(template_kwargs) > 16 or any(not isinstance(k, str) or not isinstance(v, (str, int, float, bool, type(None))) for k,v in template_kwargs.items()):
        raise LoraContractError('hf.chat_template_kwargs must be a small scalar mapping')
    train_path = _safe_file(Path(cfg['data']['train_file']))
    val_path = _safe_file(Path(cfg['data']['validation_file']))
    if train_path == val_path:
        raise LoraContractError('Training and validation paths must differ')
    train_hash, n_train, train_samples = _sample_digest(train_path)
    val_hash, n_valid, valid_samples = _sample_digest(val_path)
    if train_samples & valid_samples:
        raise LoraContractError('Train/validation text leakage: duplicate examples between splits')
    train_format = _sample_format(train_path)
    valid_format = _sample_format(val_path)
    if train_format != valid_format:
        raise LoraContractError('Train/validation data formats must match')
    if loss_mode == 'response_only' and train_format != 'messages':
        raise LoraContractError('response_only loss requires structured messages data')
    batch = int(cfg['train']['per_device_train_batch_size'])
    accumulation = int(cfg['train']['gradient_accumulation_steps'])
    evaluations = int(cfg['train']['eval_steps'])
    epochs = int(cfg['train']['num_train_epochs'])
    # Single-process estimates (GPU single worker); actual Trainer can differ for dropped batches.
    updates_per_epoch = max(1, math.ceil(n_train / (batch * accumulation)))
    estimated_steps = min(max_steps, updates_per_epoch * epochs)
    if estimated_steps < evaluations:
        raise LoraContractError(f'No validation would run: estimated steps={estimated_steps}, eval_steps={evaluations}')
    plan = {
        'schema': 'autotune-lora-plan-v0.7', 'model': name, 'revision': revision,
        'device': device, 'quantization': quant, 'target_modules': modules,
        'max_steps': max_steps, 'seed': seed, 'data_format': train_format,
        'loss_mode': loss_mode, 'chat_template_kwargs': template_kwargs,
        'config_sha256': config_sha256(cfg),
        'train_sha256': train_hash, 'validation_sha256': val_hash,
        'train_rows': n_train, 'validation_rows': n_valid,
        'estimated_steps': estimated_steps,
        'eval_steps': evaluations, 'estimated_evaluations': estimated_steps // evaluations,
        'data_files': {'train': str(train_path), 'validation': str(val_path)}
    }
    return plan


def _write_json_atomic(path: Path, body: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as handle:
            json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _walk_artifacts(directory: Path) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise LoraContractError('Artifact directory must be real, not symlink')
    hashes = {}
    for file in sorted(directory.rglob('*')):
        if file.is_symlink():
            raise LoraContractError(f'Artifact symlink refused: {file}')
        if file.is_file() and file.name != 'autotune_manifest.json':
            relative = file.relative_to(directory).as_posix()
            hashes[relative] = _hash(file)
    if not hashes:
        raise LoraContractError('Artifact directory is empty')
    return hashes


def write_manifest(directory: str | Path, plan: dict, *, kind: str) -> dict:
    """Hash all checkpoint files after HF Trainer has finished saving them."""
    if kind not in ('trainer_checkpoint', 'final_adapter'):
        raise LoraContractError('Invalid manifest kind')
    directory = Path(directory)
    artifacts = _walk_artifacts(directory)
    if kind == 'trainer_checkpoint':
        required = {'adapter_config.json', 'adapter_model.safetensors',
                    'trainer_state.json', 'optimizer.pt', 'scheduler.pt'}
        if not any(k.startswith('rng_state') and k.endswith('.pth') for k in artifacts):
            raise LoraContractError('Full training checkpoint must contain RNG state')
    else:
        required = {'adapter_config.json', 'adapter_model.safetensors'}
    if not required <= set(artifacts):
        raise LoraContractError(f'Missing checkpoint files: {sorted(required - set(artifacts))}')
    body = {'schema': 'autotune-lora-artifact-v0.7', 'kind': kind,
            'config_sha256': plan['config_sha256'],
            'train_sha256': plan['train_sha256'],
            'validation_sha256': plan['validation_sha256'],
            'model': plan['model'], 'revision': plan['revision'], 'files': artifacts}
    _write_json_atomic(directory / 'autotune_manifest.json', body)
    return body


def verify_manifest(directory: str | Path, plan: dict, *, kind: str) -> dict:
    directory = Path(directory)
    try:
        body = json.loads(_safe_file(directory / 'autotune_manifest.json').read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        raise LoraContractError('Missing or malformed artifact manifest') from exc
    if not isinstance(body, dict) or set(body) != {'schema', 'kind', 'config_sha256',
            'train_sha256', 'validation_sha256', 'model', 'revision', 'files'}:
        raise LoraContractError('Malformed artifact manifest contract')
    for field in ('config_sha256', 'train_sha256', 'validation_sha256', 'model', 'revision'):
        if body[field] != plan[field]:
            raise LoraContractError(f'Checkpoint provenance mismatch: {field}')
    if body['schema'] != 'autotune-lora-artifact-v0.7' or body['kind'] != kind:
        raise LoraContractError('Checkpoint artifact schema mismatch')
    actual = _walk_artifacts(directory)
    if body['files'] != actual:
        raise LoraContractError('Artifact hash mismatch: files added, changed or removed')
    if kind == 'trainer_checkpoint':
        required = {'adapter_config.json', 'adapter_model.safetensors',
                    'trainer_state.json', 'optimizer.pt', 'scheduler.pt'}
        if not required <= set(actual) or not any(k.startswith('rng_state') and k.endswith('.pth') for k in actual):
            raise LoraContractError('Checkpoint missing full training state')
    elif kind == 'final_adapter':
        if not {'adapter_config.json', 'adapter_model.safetensors'} <= set(actual):
            raise LoraContractError('Adapter weights or config missing')
    else:
        raise LoraContractError('Unsupported checkpoint kind')
    return body
