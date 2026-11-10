"""AutoTune AI: verified Hugging Face LoRA -> merged Safetensors -> Ollama.

Explicit commands only. No model downloads or subprocess execution during plan.
Full GGUF export is optional, via a locally installed llama.cpp converter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .lora_contract import LoraContractError, make_plan, verify_manifest

SUPPORTED_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'  # legacy default; current plans are model-generic


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _require_directory(path: Path) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise LoraContractError(f'Unsafe or missing directory: {path}')
    resolved = path.resolve()
    for item in resolved.rglob('*'):
        if item.is_symlink():
            raise LoraContractError(f'Symlink not allowed in artifact directory: {item}')
    return resolved


def _require_fresh(path: Path) -> Path:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise LoraContractError('Destination must be absent or an empty, non-symlink directory')
    return path.resolve()


def _files(root: Path) -> dict[str, str]:
    root = _require_directory(root)
    return {p.relative_to(root).as_posix(): _sha(p)
            for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'autotune_ollama_export.json'}


def make_ollama_plan(config_path: str) -> dict:
    plan = make_plan(config_path)
    if not isinstance(plan['model'], str) or '/' not in plan['model']:
        raise LoraContractError('Ollama export requires an explicit Hugging Face model ID')
    if plan['device'] != 'cpu' or plan['quantization'] != 'none':
        raise LoraContractError('Ollama CPU pilot requires CPU LoRA without training quantization')
    if not plan['target_modules']:
        raise LoraContractError('Ollama export requires at least one configured LoRA target module')
    return {'ok': True, 'mode': 'OFFLINE PREFLIGHT - NO MODEL DOWNLOAD',
            'model': plan['model'], 'revision': plan['revision'],
            'train_rows': plan['train_rows'], 'validation_rows': plan['validation_rows'],
            'estimated_steps': plan['estimated_steps'], 'cpu_training': True,
            'ollama_import': 'merged Safetensors or converted GGUF',
            'config_sha256': plan['config_sha256']}


def verify_adapter(config_path: str, run: str) -> dict:
    plan = make_plan(config_path)
    if not isinstance(plan['model'], str) or '/' not in plan['model']:
        raise LoraContractError('Model not supported for this Ollama export')
    run_dir = _require_directory(Path(run))
    adapter = _require_directory(run_dir / 'final_adapter')
    manifest = verify_manifest(adapter, plan, kind='final_adapter')
    # Additional check: PEFT adapter must point to the exact base repo.
    try:
        adapter_config = json.loads((adapter / 'adapter_config.json').read_text('utf-8'))
    except (OSError, ValueError) as exc:
        raise LoraContractError('Invalid adapter_config.json') from exc
    if adapter_config.get('base_model_name_or_path') != plan['model']:
        raise LoraContractError('Adapter base model mismatch')
    return {'ok': True, 'adapter_dir': str(adapter),
            'config_sha256': plan['config_sha256'], 'adapter_files': len(manifest['files']),
            'model': plan['model']}


def _write_manifest(root: Path, payload: dict) -> dict:
    from .lora_contract import _write_json_atomic
    payload['files'] = _files(root)
    _write_json_atomic(root / 'autotune_ollama_export.json', payload)
    return payload


def verify_merge(merged: str) -> dict:
    folder = _require_directory(Path(merged))
    metadata_file = folder / 'autotune_ollama_export.json'
    if not metadata_file.is_file() or metadata_file.is_symlink():
        raise LoraContractError('Missing merge manifest')
    try:
        record = json.loads(metadata_file.read_text('utf-8'))
    except ValueError as exc:
        raise LoraContractError('Invalid merge manifest') from exc
    if record.get('schema') != 'autotune-ollama-export-v0.9.2' or not isinstance(record.get('model'), str) or '/' not in record.get('model',''):
        raise LoraContractError('Wrong merge schema or model')
    if record.get('files') != _files(folder):
        raise LoraContractError('Merged files modified or missing')
    if not any(x.endswith('.safetensors') for x in record['files']):
        raise LoraContractError('Merged model contains no Safetensors files')
    if 'config.json' not in record['files']:
        raise LoraContractError('Merged model missing config.json')
    return {'ok': True, 'merged_dir': str(folder), 'files_verified': len(record['files']),
            'source_config_sha256': record['source_config_sha256']}


def merge_adapter(config_path: str, run: str, destination: str) -> dict:
    proof = verify_adapter(config_path, run)
    output = _require_fresh(Path(destination))
    # All remote-code execution and untrusted pickle loading deliberately disabled.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    output.mkdir(parents=True, exist_ok=True)
    try:
        plan = make_plan(config_path)
        base = AutoModelForCausalLM.from_pretrained(
            plan['model'], revision=plan['revision'],
            torch_dtype=torch.float32, trust_remote_code=False)
        adapted = PeftModel.from_pretrained(base, proof['adapter_dir'], is_trainable=False)
        merged = adapted.merge_and_unload()
        merged.save_pretrained(str(output), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(plan['model'],
            revision=plan['revision'], trust_remote_code=False)
        tokenizer.save_pretrained(str(output))
        record = _write_manifest(output, {
            'schema': 'autotune-ollama-export-v0.9.2',
            'model': plan['model'],
            'revision': plan['revision'],
            'source_config_sha256': proof['config_sha256'],
            'source_adapter': str(Path(proof['adapter_dir']).resolve()),
        })
        verify_merge(str(output))
        return {'ok': True, 'merged_dir': str(output), 'files_hashed': len(record['files'])}
    except BaseException:
        # Never mistake a partially written export for a verified one.
        (output / 'autotune_ollama_export.json').unlink(missing_ok=True)
        raise


def make_modelfile(*, merged: str | None = None, gguf: str | None = None,
                   modelfile: str) -> dict:
    if (merged is None) == (gguf is None):
        raise LoraContractError('Choose exactly one of --merged or --gguf')
    if merged:
        verified = verify_merge(merged)
        source = Path(verified['merged_dir'])
    else:
        source = Path(gguf)
        if source.is_symlink() or not source.is_file() or source.suffix.lower() != '.gguf':
            raise LoraContractError('GGUF must be an existing regular .gguf file')
        if source.stat().st_size < 4096:
            raise LoraContractError('GGUF file is unexpectedly small')
        source = source.resolve()
    target = Path(modelfile)
    if target.exists() or target.is_symlink():
        raise LoraContractError('Refusing to overwrite an existing Modelfile')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Ollama Modelfile accepts slash-separated Windows absolute paths.
    path_str = source.as_posix()
    if '\n' in path_str or '\r' in path_str or '"' in path_str:
        raise LoraContractError('Invalid path for Modelfile')
    body = (f'FROM "{path_str}"\n'
            'PARAMETER temperature 0.3\n'
            'PARAMETER num_ctx 2048\n'
            'PARAMETER stop "<|im_end|>"\n')
    with target.open('x', encoding='utf-8') as f:
        f.write(body)
    return {'ok': True, 'modelfile': str(target.resolve()), 'source': str(source)}


def export_gguf(merged: str, converter: str, output: str) -> dict:
    verify_merge(merged)
    script = Path(converter)
    if script.is_symlink() or not script.is_file() or script.name != 'convert_hf_to_gguf.py':
        raise LoraContractError('Select an existing llama.cpp convert_hf_to_gguf.py script')
    target = Path(output)
    if target.exists() or target.is_symlink() or target.suffix.lower() != '.gguf':
        raise LoraContractError('GGUF output must be a fresh .gguf path')
    target.parent.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(script.resolve()), str(Path(merged).resolve()),
            '--outfile', str(target.resolve()), '--outtype', 'f16']
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode or not target.is_file() or target.stat().st_size < 4096:
        raise LoraContractError(f'GGUF converter failed: {result.stderr[-1500:]}')
    return {'ok': True, 'gguf_path': str(target.resolve()), 'sha256': _sha(target),
            'bytes': target.stat().st_size}


def main() -> None:
    p = argparse.ArgumentParser(description='Verified AutoTune AI HF-to-Ollama export')
    sub = p.add_subparsers(dest='command', required=True)
    for command in ['plan', 'verify', 'merge']:
        sp = sub.add_parser(command)
        sp.add_argument('--config', required=True)
        if command != 'plan':
            sp.add_argument('--run', required=True)
        if command == 'merge':
            sp.add_argument('--output', required=True)
    sp = sub.add_parser('verify-merge')
    sp.add_argument('--merged', required=True)
    sp = sub.add_parser('modelfile')
    group = sp.add_mutually_exclusive_group(required=True)
    group.add_argument('--merged')
    group.add_argument('--gguf')
    sp.add_argument('--output', required=True)
    sp = sub.add_parser('gguf')
    sp.add_argument('--merged', required=True)
    sp.add_argument('--converter', required=True)
    sp.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        if args.command == 'plan':
            result = make_ollama_plan(args.config)
        elif args.command == 'verify':
            result = verify_adapter(args.config, args.run)
        elif args.command == 'merge':
            result = merge_adapter(args.config, args.run, args.output)
        elif args.command == 'verify-merge':
            result = verify_merge(args.merged)
        elif args.command == 'modelfile':
            result = make_modelfile(merged=args.merged, gguf=args.gguf, modelfile=args.output)
        else:
            result = export_gguf(args.merged, args.converter, args.output)
    except (LoraContractError, OSError, ValueError) as exc:
        p.exit(2, json.dumps({'ok': False, 'error': str(exc)}, indent=2) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
