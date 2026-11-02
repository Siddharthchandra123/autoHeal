"""Offline contract tests for isolated HF optimizer-resume exercise."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autotune.hf_resume_contract import validate_pause_step, validate_resume_source
from autotune.lora_contract import LoraContractError


PLAN = {'eval_steps': 2, 'estimated_steps': 4}


def test_valid_pause():
    validate_pause_step(PLAN, 2)


@pytest.mark.parametrize('step', [-1, 0, 1, 3, 4, 6, True, '2'])
def test_invalid_pauses(step):
    with pytest.raises(LoraContractError):
        validate_pause_step(PLAN, step)


def test_cannot_pause_on_resume():
    with pytest.raises(LoraContractError):
        validate_pause_step(PLAN, 2, 'checkpoint-2')


def make_paused(tmp_path, state='paused', ckpt_step=2):
    root = tmp_path / 'trial'
    ckpt = root / 'checkpoints' / 'checkpoint-2'
    ckpt.mkdir(parents=True)
    (ckpt / 'trainer_state.json').write_text(json.dumps({'global_step': ckpt_step}), 'utf8')
    (root / 'run_state.json').write_text(json.dumps({'status':state, 'checkpoint':str(ckpt.resolve()),'step':2}), 'utf8')
    (root / 'pause_receipt.json').write_text(json.dumps({'ok':True,'status':'paused', 'checkpoint':str(ckpt.resolve()),'global_step':2,'target_step':4}), 'utf8')
    return root, ckpt


def test_paused_resume_contract(tmp_path, monkeypatch):
    import autotune.hf_resume_contract as mod
    root, ckpt = make_paused(tmp_path)
    recorded = []
    monkeypatch.setattr(mod, 'verify_manifest', lambda path, plan, kind: recorded.append((path, kind)))
    assert validate_resume_source(root, ckpt, PLAN) == 2
    assert recorded[0][1] == 'trainer_checkpoint'


@pytest.mark.parametrize('state', ['running','completed','blocked'])
def test_untrusted_states_refused(tmp_path, monkeypatch, state):
    import autotune.hf_resume_contract as mod
    root, ckpt = make_paused(tmp_path, state=state)
    monkeypatch.setattr(mod, 'verify_manifest', lambda *args, **kwargs: None)
    with pytest.raises(LoraContractError, match='verified paused'):
        validate_resume_source(root, ckpt, PLAN)


def test_step_mismatch_refused(tmp_path, monkeypatch):
    import autotune.hf_resume_contract as mod
    root, ckpt = make_paused(tmp_path, ckpt_step=3)
    monkeypatch.setattr(mod, 'verify_manifest', lambda *args, **kwargs: None)
    with pytest.raises(LoraContractError, match='global_step'):
        validate_resume_source(root, ckpt, PLAN)


def test_modified_manifest_refused(tmp_path, monkeypatch):
    import autotune.hf_resume_contract as mod
    root, ckpt = make_paused(tmp_path)
    def reject(*args, **kwargs):
        raise LoraContractError('Artifact hash mismatch')
    monkeypatch.setattr(mod, 'verify_manifest', reject)
    with pytest.raises(LoraContractError, match='hash mismatch'):
        validate_resume_source(root, ckpt, PLAN)


def test_resume_receipt_mismatch_refused(tmp_path, monkeypatch):
    import autotune.hf_resume_contract as mod
    root, ckpt = make_paused(tmp_path)
    receipt_file = root / 'pause_receipt.json'
    receipt = json.loads(receipt_file.read_text('utf8'))
    receipt['checkpoint'] = str(tmp_path / 'different')
    receipt_file.write_text(json.dumps(receipt), 'utf8')
    monkeypatch.setattr(mod, 'verify_manifest', lambda *args, **kwargs: None)
    with pytest.raises(LoraContractError, match='checkpoint path'):
        validate_resume_source(root, ckpt, PLAN)


def test_cli_supports_pause_flag_without_transformers():
    out = subprocess.run([sys.executable, '-m', 'autotune.hf_models_cli', '--help'],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert '--pause-after-step' in out.stdout


def test_resume_config_exists():
    root = Path(__file__).parents[1]
    from autotune.lora_contract import make_plan
    p = make_plan(root / 'config/lora_cpu_resume_probe.yaml')
    assert p['estimated_steps'] == 4
    assert p['eval_steps'] == 2
