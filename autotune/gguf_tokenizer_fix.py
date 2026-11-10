"""Create a verified, isolated tokenizer-compatibility staging copy for GGUF.

A Transformers/llama.cpp converter mismatch can reject a list-valued
``extra_special_tokens``.  The original model and its manifest are never edited.
Only source-verified special tokens can be migrated to an equivalent legacy
``additional_special_tokens`` list in an untrusted conversion-only copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .lora_contract import LoraContractError
from .ollama_bridge import verify_merge


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dict(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise LoraContractError(f'Cannot read {label}') from exc
    if not isinstance(value, dict):
        raise LoraContractError(f'{label} must contain a JSON object')
    return value


def _special_registry(src: Path, config: dict) -> dict[str, bool]:
    """Read local serialized special-token flags; do not download any metadata.

    Prefer strict reconciliation if the same token appears in both inventories.
    Unknown tokens do not get silently promoted to special tokens.
    """
    registry: dict[str, bool] = {}

    def add(token: str, flag: bool, origin: str) -> None:
        if token in registry and registry[token] != flag:
            raise LoraContractError(f'Conflicting special-token flags for {token!r} in {origin}; manual review required')
        registry[token] = flag

    decoder = config.get('added_tokens_decoder', {})
    if decoder is not None:
        if not isinstance(decoder, dict):
            raise LoraContractError('Malformed added_tokens_decoder; manual review required')
        for index, record in decoder.items():
            if not isinstance(record, dict) or not isinstance(record.get('content'), str) or not isinstance(record.get('special'), bool):
                raise LoraContractError(f'Malformed added_tokens_decoder entry {index}; manual review required')
            add(record['content'], record['special'], 'added_tokens_decoder')

    tokenizer_json = src / 'tokenizer.json'
    if tokenizer_json.is_file():
        tokenizer = _load_dict(tokenizer_json, label='tokenizer.json')
        added = tokenizer.get('added_tokens', [])
        if not isinstance(added, list):
            raise LoraContractError('Malformed tokenizer.json added_tokens; manual review required')
        for record in added:
            if not isinstance(record, dict) or not isinstance(record.get('content'), str) or not isinstance(record.get('special'), bool):
                raise LoraContractError('Malformed tokenizer.json added_tokens entry; manual review required')
            add(record['content'], record['special'], 'tokenizer.json')
    return registry


def _normalize_tokens(src: Path, metadata: dict) -> tuple[dict, list[str]]:
    old = metadata.get('extra_special_tokens')
    if old is None or isinstance(old, dict):
        return metadata, []
    if not isinstance(old, list):
        raise LoraContractError('Unsupported extra_special_tokens type; manual review required')
    additions = metadata.get('additional_special_tokens', [])
    if not isinstance(additions, list):
        raise LoraContractError('additional_special_tokens must be a list; manual review required')
    all_tokens = old + additions
    if (len(all_tokens) > 4096 or
            any(not isinstance(x, str) or not x or len(x) > 256 or '\x00' in x for x in all_tokens)):
        raise LoraContractError('Invalid token metadata; manual review required')

    # The previous patch rejected all tokens not repeated in `additional`.
    # Instead, require the extra token to be flagged `special` in serialized
    # tokenizer information before migrating it into `additional`.
    seen = set(additions)
    unique_added = list(dict.fromkeys(additions))
    missing = [token for token in old if token not in seen]
    if missing:
        registry = _special_registry(src, metadata)
        unverified = [token for token in missing if registry.get(token) is not True]
        if unverified:
            # Display bounded subset for actionable diagnosis, not a silent delete.
            sample = ', '.join(repr(t) for t in unverified[:4])
            raise LoraContractError(f'Extra tokens lack verified special-token definitions: {sample}; manual review required')
        for token in missing:
            if token not in seen:
                unique_added.append(token)
                seen.add(token)

    updated = dict(metadata)
    updated['additional_special_tokens'] = unique_added
    updated['extra_special_tokens'] = {}  # Legacy converter expects mapping here.
    return updated, list(dict.fromkeys(missing))


def prepare_staging(merged: str, staging: str) -> dict:
    src = Path(merged).resolve(strict=True)
    dst = Path(staging).absolute()
    if not src.is_dir():
        raise LoraContractError('Merged source must be a directory')
    if dst.exists() or dst.is_symlink():
        raise LoraContractError('Staging destination already exists; choose a fresh directory')
    if dst == src or src in dst.parents or dst in src.parents:
        raise LoraContractError('Staging directory must be separate from the verified model')
    for entry in src.rglob('*'):
        if entry.is_symlink():
            raise LoraContractError(f'Symlink found in merged source: {entry}')

    verify_merge(str(src))
    config_path = src / 'tokenizer_config.json'
    if not config_path.is_file():
        raise LoraContractError('Missing tokenizer_config.json; do not regenerate weights')
    metadata = _load_dict(config_path, label='tokenizer_config.json')
    updated, migrated = _normalize_tokens(src, metadata)
    changed = isinstance(metadata.get('extra_special_tokens'), list)
    manifest_path = src / 'autotune_ollama_export.json'
    source_hash = _sha256(config_path)
    manifest_hash = _sha256(manifest_path) if manifest_path.exists() else None

    try:
        shutil.copytree(src, dst, symlinks=False)
        staged_config = dst / 'tokenizer_config.json'
        if changed:
            staged_config.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # The source's manifest has been invalidated by the tokenizer change.
        # Do not misrepresent the staging copy as a verified merged model.
        (dst / 'autotune_ollama_export.json').unlink(missing_ok=True)
        # Only tokenizer_config.json may differ from original; all other files,
        # including safetensors, chat templates and vocab, must match exactly.
        for path in src.rglob('*'):
            if not path.is_file() or path.name in ('autotune_ollama_export.json', 'tokenizer_config.json'):
                continue
            relative = path.relative_to(src)
            if _sha256(path) != _sha256(dst / relative):
                raise LoraContractError(f'Staging copy mismatch: {relative}')
        result = {
            'ok': True,
            'mode': 'ISOLATED TOKENIZER METADATA STAGING',
            'verified_merged_source': str(src),
            'staging_dir': str(dst),
            'tokenizer_fix_applied': changed,
            'special_tokens_migrated': migrated,
            'source_tokenizer_sha256': source_hash,
            'staged_tokenizer_sha256': _sha256(staged_config),
            'source_manifest_sha256': manifest_hash,
            'non_tokenizer_files_unchanged': True,
            'weights_untouched': True,
            'warning': 'Staging copy is not a verified merged model. Validate tokenizer and GGUF conversion separately.',
        }
        (dst / 'gguf_staging_provenance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result
    except BaseException:
        if dst.exists() and dst.is_dir():
            shutil.rmtree(dst)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged', required=True, help='Verified AutoTune merged model directory')
    parser.add_argument('--staging', required=True, help='Fresh staging directory for GGUF conversion')
    args = parser.parse_args()
    print(json.dumps(prepare_staging(args.merged, args.staging), indent=2))


if __name__ == '__main__':
    main()
