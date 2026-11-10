import json
from pathlib import Path

import pytest

from autotune import gguf_tokenizer_fix as fix


def _fixture(tmp_path, value):
    src = tmp_path / 'merged'
    src.mkdir()
    (src / 'model.safetensors').write_bytes(b'fake test weights, not real')
    (src / 'config.json').write_text('{}', encoding='utf-8')
    (src / 'autotune_ollama_export.json').write_text('{"verified": true}', encoding='utf-8')
    config = {
        'tokenizer_class': 'GPT2Tokenizer',
        'extra_special_tokens': value,
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>'],
    }
    (src / 'tokenizer_config.json').write_text(json.dumps(config), encoding='utf-8')
    return src


def test_converts_empty_list_in_staging_only(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda x: {'ok': True})
    src = _fixture(tmp_path, [])
    result = fix.prepare_staging(str(src), str(tmp_path / 'gguf_stage'))
    assert result['tokenizer_fix_applied']
    assert json.loads((tmp_path / 'gguf_stage/tokenizer_config.json').read_text())['extra_special_tokens'] == {}
    assert json.loads((src / 'tokenizer_config.json').read_text())['extra_special_tokens'] == []
    assert (src / 'autotune_ollama_export.json').exists()
    assert not (tmp_path / 'gguf_stage/autotune_ollama_export.json').exists()
    assert (tmp_path / 'gguf_stage/model.safetensors').read_bytes() == (src / 'model.safetensors').read_bytes()


def test_preserves_known_special_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda x: {'ok': True})
    src = _fixture(tmp_path, ['<|im_start|>'])
    result = fix.prepare_staging(str(src), str(tmp_path / 'stage'))
    conf = json.loads((tmp_path / 'stage/tokenizer_config.json').read_text())
    assert conf['additional_special_tokens'] == ['<|im_start|>', '<|im_end|>']
    assert result['tokenizer_fix_applied']


def test_dict_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda x: {'ok': True})
    src = _fixture(tmp_path, {})
    result = fix.prepare_staging(str(src), str(tmp_path / 'stage'))
    assert not result['tokenizer_fix_applied']
    assert (src / 'tokenizer_config.json').read_bytes() == (tmp_path / 'stage/tokenizer_config.json').read_bytes()


def test_rejects_unknown_token(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda x: {'ok': True})
    src = _fixture(tmp_path, ['<UNKNOWN>'])
    with pytest.raises(ValueError, match='manual review'):
        fix.prepare_staging(str(src), str(tmp_path / 'stage'))
    assert not (tmp_path / 'stage').exists()


def test_rejects_existing_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda x: {'ok': True})
    src = _fixture(tmp_path, [])
    (tmp_path / 'stage').mkdir()
    with pytest.raises(ValueError, match='already exists'):
        fix.prepare_staging(str(src), str(tmp_path / 'stage'))


def test_rejects_source_manifest_failure(tmp_path, monkeypatch):
    def fail(path): raise ValueError('tampered manifest')
    monkeypatch.setattr(fix, 'verify_merge', fail)
    src = _fixture(tmp_path, [])
    with pytest.raises(ValueError, match='tampered manifest'):
        fix.prepare_staging(str(src), str(tmp_path / 'stage'))


def test_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda x: {'ok': True})
    src = _fixture(tmp_path, [])
    try:
        (src / 'bad').symlink_to(tmp_path / 'other')
    except OSError:
        pytest.skip('symlinks disabled')
    with pytest.raises(ValueError, match='Symlink'):
        fix.prepare_staging(str(src), str(tmp_path / 'stage'))
