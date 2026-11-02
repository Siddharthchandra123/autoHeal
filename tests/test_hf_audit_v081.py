"""Offline artifact-audit tests: no network/model downloads involved."""
import json
from pathlib import Path
import shutil

import pytest
import yaml

from autotune.hf_audit_cli import audit_lora_run, main
from autotune.lora_contract import LoraContractError, make_plan, write_manifest


@pytest.fixture
def run_fixture(tmp_path):
    project = Path(__file__).resolve().parents[1]
    (tmp_path / 'sample_data').mkdir()
    for name in ('train.jsonl', 'validation.jsonl'):
        shutil.copyfile(project / 'sample_data' / name, tmp_path / 'sample_data' / name)
    config = yaml.safe_load((project / 'config/lora_public_cpu.yaml').read_text('utf8'))
    for key in ('train_file', 'validation_file'):
        config['data'][key] = str(tmp_path / config['data'][key])
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), 'utf8')
    plan = make_plan(config_path)
    root = tmp_path / 'out'
    root.mkdir()
    (root / 'plan.json').write_text(json.dumps(plan), 'utf8')
    adapter = root / 'final_adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}', 'utf8')
    (adapter / 'adapter_model.safetensors').write_bytes(b'adapter bytes')
    a = write_manifest(adapter, plan, kind='final_adapter')
    checkpoint = root / 'checkpoints' / 'checkpoint-2'
    checkpoint.mkdir(parents=True)
    for name in ('adapter_config.json','adapter_model.safetensors','trainer_state.json',
                 'optimizer.pt','scheduler.pt','rng_state.pth'):
        (checkpoint / name).write_bytes(f'{name} bytes'.encode())
    write_manifest(checkpoint, plan, kind='trainer_checkpoint')
    receipt = {'ok':True, 'model': plan['model'], 'best_val_loss':10.365,
               'adapter_dir':str(adapter.resolve()), 'best_checkpoint':str(checkpoint.resolve()),
               'adapter_manifest_sha256':a['files']['adapter_model.safetensors']}
    (root / 'result_receipt.json').write_text(json.dumps(receipt),'utf8')
    (root / 'run_state.json').write_text(json.dumps({'status':'completed','best_checkpoint':str(checkpoint.resolve())}),'utf8')
    return config_path, root


def edit_json(path, updater):
    row = json.loads(path.read_text('utf8'))
    updater(row)
    path.write_text(json.dumps(row), 'utf8')


def test_clean_run_verified(run_fixture):
    cfg, root = run_fixture
    result = audit_lora_run(cfg, root)
    assert result['ok'] and result['checkpoint_has_training_state']
    assert result['actual_resume_executed'] is False
    assert result['adapter_reload_executed'] is False


def test_cli_returns_valid_json(run_fixture, capsys):
    cfg, root = run_fixture
    assert main(['--config', str(cfg), '--run', str(root)]) == 0
    assert json.loads(capsys.readouterr().out)['ok'] is True


def test_changed_weights_rejected(run_fixture):
    cfg, root = run_fixture
    (root / 'final_adapter/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(LoraContractError, match='hash mismatch'):
        audit_lora_run(cfg, root)


def test_changed_optimizer_rejected(run_fixture):
    cfg, root = run_fixture
    (root / 'checkpoints/checkpoint-2/optimizer.pt').write_bytes(b'changed')
    with pytest.raises(LoraContractError, match='hash mismatch'):
        audit_lora_run(cfg, root)


def test_run_state_incomplete_rejected(run_fixture):
    cfg, root = run_fixture
    edit_json(root / 'run_state.json', lambda r: r.update(status='running'))
    with pytest.raises(LoraContractError, match='not completed'):
        audit_lora_run(cfg, root)


def test_changed_dataset_rejected(run_fixture):
    cfg, root = run_fixture
    cfgdata = yaml.safe_load(cfg.read_text('utf8'))
    Path(cfgdata['data']['train_file']).write_text('NEW DATA\n', 'utf8')
    with pytest.raises((LoraContractError, ValueError), match='mismatch|dataset|JSON|Invalid'):
        audit_lora_run(cfg, root)


def test_receipt_model_mismatch_rejected(run_fixture):
    cfg, root = run_fixture
    edit_json(root / 'result_receipt.json', lambda r: r.update(model='wrong/model'))
    with pytest.raises(LoraContractError, match='receipt'):
        audit_lora_run(cfg, root)


def test_best_checkpoint_escape_rejected(run_fixture, tmp_path):
    cfg, root = run_fixture
    edit_json(root / 'result_receipt.json', lambda r: r.update(best_checkpoint=str(tmp_path / 'other')))
    with pytest.raises(LoraContractError, match='inside'):
        audit_lora_run(cfg, root)


def test_receipt_adapter_hash_mismatch_rejected(run_fixture):
    cfg, root = run_fixture
    edit_json(root / 'result_receipt.json', lambda r: r.update(adapter_manifest_sha256='bad'))
    with pytest.raises(LoraContractError, match='weight hash'):
        audit_lora_run(cfg, root)


def test_checkpoint_state_disagree_rejected(run_fixture):
    cfg, root = run_fixture
    edit_json(root / 'run_state.json', lambda r: r.update(best_checkpoint='not-right'))
    with pytest.raises(LoraContractError, match='disagrees'):
        audit_lora_run(cfg, root)


def test_negative_nonfinite_loss_rejected(run_fixture):
    cfg, root = run_fixture
    edit_json(root / 'result_receipt.json', lambda r: r.update(best_val_loss='nan'))
    with pytest.raises(LoraContractError, match='validation loss'):
        audit_lora_run(cfg, root)


def test_reload_mocked_only_after_audit(run_fixture, monkeypatch):
    cfg, root = run_fixture
    import autotune.lora_gpu
    called = []
    def fake_reload(c, artifact, *, allow_cpu):
        called.append(allow_cpu)
        return {'ok':True,'adapter_verified':True,'generated_token_count':8}
    monkeypatch.setattr(autotune.lora_gpu, 'reload_adapter', fake_reload)
    result = audit_lora_run(cfg, root, reload=True)
    assert result['adapter_reload_executed'] and result['generated_token_count'] == 8
    assert called == [True]


def test_bad_json_error_code(run_fixture, capsys):
    cfg, root = run_fixture
    (root / 'result_receipt.json').write_text('wrong','utf8')
    assert main(['--config',str(cfg),'--run',str(root)]) == 2
    assert json.loads(capsys.readouterr().out)['ok'] is False
