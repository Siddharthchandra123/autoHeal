"""Offline security/integration tests for CPU LoRA -> Ollama export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotune.ollama_bridge import (
    SUPPORTED_MODEL, _files, _require_fresh, make_ollama_plan,
    verify_merge, make_modelfile, export_gguf, verify_adapter,
)
from autotune.lora_contract import LoraContractError, make_plan, _write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / 'config/lora_smollm2_ollama_cpu.yaml')


def test_plan_is_cpu_and_model_valid():
    plan = make_ollama_plan(CONFIG)
    assert plan['ok'] is True
    assert plan['model'] == SUPPORTED_MODEL
    assert plan['train_rows'] == 16
    assert plan['estimated_steps'] == 8


def test_planning_has_no_weight_download(monkeypatch):
    # Prevent accidental loading of a large model during offline preflight.
    import sys
    monkeypatch.setitem(sys.modules, 'transformers', None)
    monkeypatch.setitem(sys.modules, 'peft', None)
    assert make_ollama_plan(CONFIG)['ok']


def test_model_generic_export_plan_accepts_registered_model_shape(tmp_path):
    import yaml
    config = yaml.safe_load(Path(CONFIG).read_text())
    config['model']['name'] = 'someone/another-model'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    plan = make_ollama_plan(str(path))
    assert plan['model'] == 'someone/another-model'


def test_wrong_device_rejected(tmp_path):
    import yaml
    config = yaml.safe_load(Path(CONFIG).read_text())
    config['hf']['device'] = 'gpu'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(LoraContractError, match='CPU'):
        make_ollama_plan(str(path))


def test_existing_output_rejected(tmp_path):
    root = tmp_path / 'out'
    root.mkdir()
    (root / 'keep.txt').write_text('do not delete')
    with pytest.raises(LoraContractError, match='Destination'):
        _require_fresh(root)
    assert (root / 'keep.txt').exists()


def _merged_fake(tmp_path):
    out = tmp_path / 'merged'
    out.mkdir()
    (out / 'model.safetensors').write_bytes(b'not-a-real-model' * 500)
    (out / 'config.json').write_text('{"model_type":"llama"}')
    _write_json_atomic(out / 'autotune_ollama_export.json', {
        'schema': 'autotune-ollama-export-v0.9.2',
        'model': SUPPORTED_MODEL, 'source_config_sha256': 'abc', 'files': _files(out),
    })
    return out


def test_fake_export_manifest_verified_without_loading_weights(tmp_path):
    folder = _merged_fake(tmp_path)
    assert verify_merge(str(folder))['files_verified'] == 2


def test_tampered_merge_rejected(tmp_path):
    folder = _merged_fake(tmp_path)
    (folder / 'model.safetensors').write_bytes(b'tampered')
    with pytest.raises(LoraContractError, match='modified'):
        verify_merge(str(folder))


def test_missing_manifest_rejected(tmp_path):
    folder = tmp_path / 'merged'
    folder.mkdir()
    with pytest.raises(LoraContractError, match='manifest'):
        verify_merge(str(folder))


def test_modelfile_for_safetensors(tmp_path):
    merged = _merged_fake(tmp_path)
    path = tmp_path / 'Modelfile'
    response = make_modelfile(merged=str(merged), modelfile=str(path))
    assert response['ok']
    text = path.read_text()
    assert text.startswith('FROM "')
    assert 'temperature 0.3' in text
    with pytest.raises(LoraContractError, match='overwrite'):
        make_modelfile(merged=str(merged), modelfile=str(path))


def test_modelfile_for_gguf(tmp_path):
    gguf = tmp_path / 'model.gguf'
    gguf.write_bytes(b'GGUF' * 2048)
    response = make_modelfile(gguf=str(gguf), modelfile=str(tmp_path / 'Modelfile'))
    assert response['ok']


def test_missing_gguf_refused(tmp_path):
    with pytest.raises(LoraContractError, match='GGUF'):
        make_modelfile(gguf=str(tmp_path / 'none.gguf'), modelfile=str(tmp_path / 'Modelfile'))


def test_mutual_exclusion_sources(tmp_path):
    with pytest.raises(LoraContractError, match='exactly one'):
        make_modelfile(merged='unused', gguf='unused', modelfile=str(tmp_path / 'file'))


def test_converter_requires_real_script(tmp_path):
    merged = _merged_fake(tmp_path)
    with pytest.raises(LoraContractError, match='convert_hf_to_gguf'):
        export_gguf(str(merged), str(tmp_path / 'unknown.py'), str(tmp_path / 'result.gguf'))


def test_missing_adapter_refused(tmp_path):
    with pytest.raises(LoraContractError):
        verify_adapter(CONFIG, str(tmp_path / 'nonexistent'))


def test_split_hashes_and_no_leakage():
    plan = make_plan(CONFIG)
    assert plan['train_sha256'] != plan['validation_sha256']
    assert plan['target_modules'] == ['q_proj', 'v_proj']
