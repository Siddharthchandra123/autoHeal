"""v0.9.5: Regression for model-specific special tokens absent in additional_special_tokens."""
import json
import hashlib
import pytest
from autotune import gguf_tokenizer_fix as fix


def _source(tmp_path, extras, decoder=None, tokenizer_json=None):
    src = tmp_path / 'merged'
    src.mkdir()
    (src / 'config.json').write_text('{"model_type":"llama"}')
    (src / 'model.safetensors').write_bytes(b'fake model weights')
    (src / 'autotune_ollama_export.json').write_text('{"verified":true}')
    config = {
        'additional_special_tokens': ['<|im_start|>', '<|im_end|>'],
        'extra_special_tokens': extras,
        'added_tokens_decoder': decoder if decoder is not None else {},
        'chat_template': '{{ message }}',
    }
    (src / 'tokenizer_config.json').write_text(json.dumps(config), encoding='utf-8')
    if tokenizer_json is not None:
        (src / 'tokenizer.json').write_text(json.dumps(tokenizer_json), encoding='utf-8')
    return src


def _decoder(value, flag=True):
    return {'16': {'content': value, 'special': flag}}


def _prep(src, tmp_path, monkeypatch):
    monkeypatch.setattr(fix, 'verify_merge', lambda _: {'ok': True})
    return fix.prepare_staging(str(src), str(tmp_path / 'staged'))


def test_migrates_verified_model_specific_token_preserving_weights(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<jupyter_script>', '<|im_start|>'], _decoder('<jupyter_script>'))
    original = (src / 'tokenizer_config.json').read_bytes()
    original_weights = hashlib.sha256((src / 'model.safetensors').read_bytes()).hexdigest()
    result = _prep(src, tmp_path, monkeypatch)
    staged = json.loads((tmp_path / 'staged/tokenizer_config.json').read_text())
    assert result['ok'] and result['special_tokens_migrated'] == ['<jupyter_script>']
    assert staged['extra_special_tokens'] == {}
    assert staged['additional_special_tokens'] == ['<|im_start|>', '<|im_end|>', '<jupyter_script>']
    assert staged['added_tokens_decoder']['16']['content'] == '<jupyter_script>'
    assert staged['chat_template'] == '{{ message }}'
    assert (src / 'tokenizer_config.json').read_bytes() == original
    assert hashlib.sha256((tmp_path / 'staged/model.safetensors').read_bytes()).hexdigest() == original_weights
    assert not (tmp_path / 'staged/autotune_ollama_export.json').exists()


def test_migrates_using_tokenizer_json_special_flag(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<jupyter_script>'], tokenizer_json={'added_tokens':[{'id':16,'content':'<jupyter_script>','special':True}]})
    result = _prep(src, tmp_path, monkeypatch)
    assert result['special_tokens_migrated'] == ['<jupyter_script>']


def test_rejects_unverified_token(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<unlisted>'])
    monkeypatch.setattr(fix, 'verify_merge', lambda _: {'ok': True})
    with pytest.raises(ValueError, match='manual review'):
        fix.prepare_staging(str(src), str(tmp_path / 'staged'))
    assert not (tmp_path / 'staged').exists()


def test_rejects_non_special_token(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<normal>'], _decoder('<normal>', False))
    monkeypatch.setattr(fix, 'verify_merge', lambda _: {'ok': True})
    with pytest.raises(ValueError, match='manual review'):
        fix.prepare_staging(str(src), str(tmp_path / 'staged'))


def test_rejects_inconsistent_registry(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<special>'], _decoder('<special>', True),
        {'added_tokens':[{'id':16,'content':'<special>','special':False}]})
    monkeypatch.setattr(fix, 'verify_merge', lambda _: {'ok': True})
    with pytest.raises(ValueError, match='Conflicting'):
        fix.prepare_staging(str(src), str(tmp_path / 'staged'))


def test_rejects_invalid_additional_tokens_type(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<jupyter_script>'], _decoder('<jupyter_script>'))
    cfg = src / 'tokenizer_config.json'
    meta = json.loads(cfg.read_text())
    meta['additional_special_tokens'] = 'bad type'
    cfg.write_text(json.dumps(meta))
    monkeypatch.setattr(fix, 'verify_merge', lambda _: {'ok': True})
    with pytest.raises(ValueError, match='manual review'):
        fix.prepare_staging(str(src), str(tmp_path / 'staged'))


def test_known_additional_does_not_need_decoder(tmp_path, monkeypatch):
    src = _source(tmp_path, ['<|im_end|>'])
    result = _prep(src, tmp_path, monkeypatch)
    assert result['special_tokens_migrated'] == []
