"""No model downloads, hosted APIs or GPU needed for Stage 5B contract tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pytest
import yaml

from autotune.lora_contract import (LoraContractError, make_plan, write_manifest,
                                    verify_manifest, _sample_digest, _walk_artifacts)
from autotune.lora_gpu import _gpu_mode, _log_jsonl, _lock

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return make_plan(ROOT / 'config/lora_meta_gpu.yaml')


def test_meta_preflight_no_hf_download():
    p = plan()
    assert p['model'].startswith('meta-llama/')
    assert p['estimated_evaluations'] >= 2
    assert p['train_sha256'] != p['validation_sha256']


def test_public_cpu_plan():
    p = make_plan(ROOT / 'config/lora_public_cpu.yaml')
    assert p['device'] == 'cpu'
    assert p['model'] == 'sshleifer/tiny-gpt2'


def test_preflight_does_not_mutate_data(tmp_path):
    f = ROOT / 'sample_data/train.jsonl'
    before = f.read_bytes()
    plan()
    assert f.read_bytes() == before


def modify(tmp_path, change):
    raw = yaml.safe_load((ROOT / 'config/lora_meta_gpu.yaml').read_text())
    raw['data']['train_file'] = str(ROOT / 'sample_data/train.jsonl')
    raw['data']['validation_file'] = str(ROOT / 'sample_data/validation.jsonl')
    change(raw)
    loc = tmp_path / 'config.yaml'
    loc.write_text(yaml.safe_dump(raw), encoding='utf8')
    return loc


@pytest.mark.parametrize('field,value', [
    ('quantization', '8bit'),('quantization', None),
    ('device','tpu'),('max_steps',0),('max_steps',100001),('max_steps',True),
    ('seed',-1), ('target_modules', []),('target_modules', ['q_proj','q_proj']),
    ('target_modules',['bad name']),('revision',''),('unrecognized','x')])
def test_invalid_hf_settings(tmp_path, field, value):
    def change(cfg):
        cfg['hf'][field] = value
    with pytest.raises(LoraContractError):
        make_plan(modify(tmp_path, change))


def test_no_eval_steps(tmp_path):
    with pytest.raises(LoraContractError, match='No validation'):
        make_plan(modify(tmp_path, lambda c: c['train'].update(eval_steps=500)))


def test_cpu_meta_rejected(tmp_path):
    with pytest.raises(LoraContractError, match='requires a compatible GPU'):
        make_plan(modify(tmp_path, lambda c: c['hf'].update(device='cpu')))


def test_cpu_4bit_rejected(tmp_path):
    with pytest.raises(LoraContractError):
        make_plan(modify(tmp_path, lambda c: c['hf'].update(device='cpu', quantization='4bit')))


def test_gru_is_not_hf(tmp_path):
    with pytest.raises(LoraContractError):
        make_plan(modify(tmp_path, lambda c: c['model'].update(name='autotune/tiny-byte-gru')))


def test_invalid_jsonl(tmp_path):
    f = tmp_path / 'train.jsonl'
    f.write_text('{this is not JSON}\n', encoding='utf8')
    with pytest.raises(LoraContractError):
        _sample_digest(f)


def test_invalid_empty_text(tmp_path):
    f = tmp_path / 'train.jsonl'
    f.write_text('{"text": " "}\n{"text":"hello"}', encoding='utf8')
    with pytest.raises(LoraContractError, match='nonempty'):
        _sample_digest(f)


def test_duplicate_in_one_split(tmp_path):
    f = tmp_path / 'train.jsonl'
    f.write_text('{"text":"hi"}\n{"text":"hi"}\n', encoding='utf8')
    with pytest.raises(LoraContractError, match='duplicate'):
        _sample_digest(f)


def test_cross_split_leakage(tmp_path):
    train = tmp_path / 'a.jsonl'
    valid = tmp_path / 'b.jsonl'
    train.write_text('{"text":"same"}\n{"text":"train"}\n', encoding='utf8')
    valid.write_text('{"text":"same"}\n{"text":"valid"}\n', encoding='utf8')
    with pytest.raises(LoraContractError, match='leakage'):
        make_plan(modify(tmp_path, lambda c: c['data'].update(train_file=str(train), validation_file=str(valid))))


def fake_artifacts(tmp_path, kind):
    d = tmp_path / kind
    d.mkdir()
    names = ['adapter_config.json', 'adapter_model.safetensors']
    if kind == 'trainer_checkpoint':
        names += ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']
    for name in names:
        (d / name).write_bytes(f'{name}:offline fake bytes'.encode())
    return d


@pytest.mark.parametrize('kind', ['trainer_checkpoint','final_adapter'])
def test_manifest_round_trip(tmp_path, kind):
    p = plan()
    d = fake_artifacts(tmp_path, kind)
    manifest = write_manifest(d, p, kind=kind)
    assert len(manifest['files']) == (6 if kind == 'trainer_checkpoint' else 2)
    assert verify_manifest(d, p, kind=kind) == manifest


@pytest.mark.parametrize('kind', ['trainer_checkpoint','final_adapter'])
def test_manifest_reject_tampering(tmp_path, kind):
    p = plan()
    d = fake_artifacts(tmp_path, kind)
    write_manifest(d, p, kind=kind)
    (d / 'adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(LoraContractError, match='hash mismatch'):
        verify_manifest(d, p, kind=kind)


def test_manifest_reject_extra_files(tmp_path):
    p = plan()
    d = fake_artifacts(tmp_path, 'final_adapter')
    write_manifest(d, p, kind='final_adapter')
    (d / 'untrusted.txt').write_text('unexpected')
    with pytest.raises(LoraContractError, match='hash mismatch'):
        verify_manifest(d, p, kind='final_adapter')


def test_manifest_reject_missing_optimizer(tmp_path):
    p = plan()
    d = fake_artifacts(tmp_path, 'trainer_checkpoint')
    (d / 'optimizer.pt').unlink()
    with pytest.raises(LoraContractError, match='Missing checkpoint'):
        write_manifest(d, p, kind='trainer_checkpoint')


def test_manifest_reject_missing_rng(tmp_path):
    p = plan()
    d = fake_artifacts(tmp_path, 'trainer_checkpoint')
    (d / 'rng_state.pth').unlink()
    with pytest.raises(LoraContractError, match='RNG'):
        write_manifest(d, p, kind='trainer_checkpoint')


def test_manifest_refuses_changed_dataset_provenance(tmp_path):
    p = plan()
    d = fake_artifacts(tmp_path, 'final_adapter')
    write_manifest(d, p, kind='final_adapter')
    altered = deepcopy(p)
    altered['train_sha256'] = '0'*64
    with pytest.raises(LoraContractError, match='provenance mismatch'):
        verify_manifest(d, altered, kind='final_adapter')


def test_manifest_reject_symlink(tmp_path):
    d = fake_artifacts(tmp_path, 'final_adapter')
    try:
        (d / 'alias.txt').symlink_to(d / 'adapter_config.json')
    except (OSError, NotImplementedError):
        pytest.skip('Symlink unavailable')
    with pytest.raises(LoraContractError, match='symlink'):
        _walk_artifacts(d)


def test_run_lock_exclusive(tmp_path):
    lock = _lock(tmp_path)
    try:
        with pytest.raises(LoraContractError, match='locked'):
            _lock(tmp_path)
    finally:
        lock.unlink()


def test_jsonl_normalizes_nan(tmp_path):
    p = tmp_path/'log.jsonl'
    _log_jsonl(p, {'step':1,'val_loss':float('nan')})
    raw = p.read_text('utf8')
    assert 'NaN' not in raw
    row = json.loads(raw)
    assert row['val_loss'] is None and row['nonfinite_metrics']==['val_loss']


def test_gpu_preflight_refuses_missing_gpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(LoraContractError, match='No CUDA/ROCm'):
        _gpu_mode(plan(), explicit_cpu=False)


def test_cpu_requires_explicit_confirmation():
    p = make_plan(ROOT / 'config/lora_public_cpu.yaml')
    with pytest.raises(LoraContractError, match='explicit --allow-cpu'):
        _gpu_mode(p, explicit_cpu=False)


def test_padding_collator_does_not_mask_actual_eos():
    import torch
    from autotune.lora_gpu import _build_collator
    class DummyTokenizer:
        def pad(self, samples, padding, return_tensors):
            return {'input_ids':torch.tensor([[5, 2, 2], [7, 2, 0]]),
                    'attention_mask':torch.tensor([[1, 1, 1], [1, 1, 0]])}
    batch = _build_collator(DummyTokenizer())([{'input_ids':[5,2,2]},{'input_ids':[7,2]}])
    assert batch['labels'].tolist() == [[5,2,2], [7,2,-100]]


def test_gpu_run_refuses_execution_before_directory_created(tmp_path, monkeypatch):
    import torch
    from autotune.lora_gpu import run_lora
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    d = tmp_path/'never_created'
    with pytest.raises(LoraContractError, match='No CUDA/ROCm'):
        run_lora(str(ROOT / 'config/lora_meta_gpu.yaml'), str(d))
    assert not d.exists()


def test_manifest_does_not_claim_signature(tmp_path):
    d = fake_artifacts(tmp_path, 'final_adapter')
    manifest = write_manifest(d, plan(), kind='final_adapter')
    assert 'signature' not in manifest
    assert 'files' in manifest


def test_cpu_plan_cli_shows_cpu_execute_option(capsys, monkeypatch):
    import sys
    from autotune.lora_cli import main
    monkeypatch.setattr(sys, 'argv', ['lora_cli', '--config', str(ROOT / 'config/lora_public_cpu.yaml')])
    main()
    output = json.loads(capsys.readouterr().out)
    assert output['plan']['device'] == 'cpu'
    assert '--allow-cpu' in output['next']
    assert 'GPU' not in output['next']
