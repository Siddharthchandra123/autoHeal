"""Phase 5E isolated orchestration tests: fake Trainer outputs, real SHA manifests.

Tests do NOT assert that Hugging Face fine-tuning was executed in this runtime.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import pytest
import yaml

from autotune.hf_selfheal import execute, preflight
from autotune.lora_contract import LoraContractError, make_plan, write_manifest


@pytest.fixture
def trial_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((root/'config'/'lora_selfheal_cpu.yaml').read_text('utf8'))
    c['data']['train_file'] = str((root/'sample_data'/'train.jsonl').resolve())
    c['data']['validation_file'] = str((root/'sample_data'/'validation.jsonl').resolve())
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(c), encoding='utf8')
    return path


def fake_trainer(scores=(11.0, 10.0), diagnoses=('overfit', None), tamper=False,
                 error=None):
    calls = []
    def trainer(config, output, *, allow_cpu, injected_diagnosis):
        assert allow_cpu is True
        index = len(calls)
        calls.append({'config':config, 'injected_diagnosis':injected_diagnosis})
        if error and index == 0:
            raise RuntimeError('fake trainer failure')
        plan = make_plan(config)
        out = Path(output)
        checkpoint = out/'checkpoints'/'checkpoint-2'
        checkpoint.mkdir(parents=True)
        files = {'adapter_config.json':'{}', 'adapter_model.safetensors':'fake-adapter',
                 'trainer_state.json':'{}','optimizer.pt':'fake-opt',
                 'scheduler.pt':'fake-sched','rng_state.pth':'fake-rng'}
        for name, content in files.items():
            (checkpoint / name).write_text(content, encoding='utf8')
        write_manifest(checkpoint, plan, kind='trainer_checkpoint')
        adapter = out / 'final_adapter'
        adapter.mkdir()
        for k in ('adapter_config.json','adapter_model.safetensors'):
            (adapter/k).write_text(files[k], encoding='utf8')
        write_manifest(adapter,plan,kind='final_adapter')
        if tamper:
            (adapter/'adapter_model.safetensors').write_text('altered', encoding='utf8')
        (out/'metrics.jsonl').write_text(json.dumps({'step':2,'train_loss':10.4,'val_loss':10.3})+'\n','utf8')
        return {'ok':True,'best_val_loss':scores[index], 'stop_diagnosis':diagnoses[index],
                'fault_injected':bool(injected_diagnosis), 'best_checkpoint':str(checkpoint)}
    return trainer, calls


def test_offline_preflight(trial_config):
    p=preflight(str(trial_config))
    assert p['ok'] and p['max_attempts']==2 and p['device']=='cpu'


def test_offline_preflight_no_write(trial_config, tmp_path):
    preflight(str(trial_config))
    assert not (tmp_path/'controller_state.json').exists()


def test_healthy_one_attempt(trial_config, tmp_path):
    trainer, calls=fake_trainer(diagnoses=(None,))
    result=execute(str(trial_config),str(tmp_path/'new'),trainer_fn=trainer)
    assert result['best_trial']==1 and result['attempt_count']==1
    assert len(calls)==1
    assert (tmp_path/'new'/'controller_state.json').exists()


def test_forced_recovery_new_training_attempt(trial_config, tmp_path):
    trainer,calls=fake_trainer()
    root=tmp_path/'run'
    result=execute(str(trial_config),str(root),trainer_fn=trainer,fault_inject_overfit=True)
    assert result['attempt_count']==2 and result['best_trial']==2
    assert calls[0]['injected_diagnosis']=='overfit'
    assert calls[1]['injected_diagnosis'] is None
    assert (root/'trial-01.advisor_decision.json').is_file()
    assert len(list(root.glob('trial-*.yaml')))==2
    first=yaml.safe_load((root/'trial-01.yaml').read_text())
    second=yaml.safe_load((root/'trial-02.yaml').read_text())
    assert second['train']['learning_rate'] < first['train']['learning_rate']
    assert second['train']['lora_dropout'] > first['train']['lora_dropout']
    assert first['data']==second['data']
    assert result['fault_injection_used']


def test_best_remains_first_if_second_worse(trial_config,tmp_path):
    trainer,_=fake_trainer(scores=(9.0,10.0))
    result=execute(str(trial_config),str(tmp_path/'r'),trainer_fn=trainer)
    assert result['best_trial']==1


def test_no_retry_on_unhandled_diagnosis(trial_config,tmp_path):
    trainer,calls=fake_trainer(diagnoses=('diverged',))
    r=execute(str(trial_config),str(tmp_path/'r'),trainer_fn=trainer)
    assert r['attempt_count']==1 and len(calls)==1
    assert not (tmp_path/'r'/'trial-01.advisor_decision.json').exists()


def test_empty_workspace_guard(trial_config,tmp_path):
    root=tmp_path/'old';root.mkdir();(root/'note').write_text('keep')
    trainer,_=fake_trainer()
    with pytest.raises(LoraContractError,match='not empty'):
        execute(str(trial_config),str(root),trainer_fn=trainer)
    assert (root/'note').read_text()=='keep'


def test_manifest_corruption_blocks(trial_config,tmp_path):
    trainer,_=fake_trainer(tamper=True)
    root=tmp_path/'r'
    with pytest.raises(LoraContractError,match='hash mismatch'):
        execute(str(trial_config),str(root),trainer_fn=trainer)
    assert json.loads((root/'controller_state.json').read_text())['status']=='blocked'


def test_failure_blocks_existing_run(trial_config,tmp_path):
    trainer,_=fake_trainer(error=True)
    root=tmp_path/'r'
    with pytest.raises(RuntimeError):
        execute(str(trial_config),str(root),trainer_fn=trainer)
    assert json.loads((root/'controller_state.json').read_text())['status']=='blocked'
    with pytest.raises(LoraContractError,match='not empty'):
        execute(str(trial_config),str(root),trainer_fn=trainer)


def test_retry_budget_one(trial_config,tmp_path):
    c=yaml.safe_load(trial_config.read_text());c['safety']['max_trials']=1
    trial_config.write_text(yaml.safe_dump(c))
    trainer,calls=fake_trainer(diagnoses=('overfit',))
    result=execute(str(trial_config),str(tmp_path/'run'),trainer_fn=trainer)
    assert result['attempt_count']==1 and len(calls)==1
    assert 'retry_budget_exhausted' in (tmp_path/'run'/'controller_events.jsonl').read_text()


def test_beyond_three_trials_rejected(trial_config):
    c=yaml.safe_load(trial_config.read_text());c['safety']['max_trials']=4
    trial_config.write_text(yaml.safe_dump(c))
    with pytest.raises(LoraContractError,match='1..3'):
        preflight(str(trial_config))


def test_provider_disallowed(trial_config):
    with pytest.raises(LoraContractError,match='Only mock or meta'):
        preflight(str(trial_config),provider='unknown')


def test_unsafe_advisor_rejected(trial_config,tmp_path):
    trainer,_=fake_trainer()
    @dataclass
    class BadAdvice:
        status: str='approved'
        diagnosis: str='overfit'
        patch: dict=None
        source: str='fake'
        provider: str='mock'
    def bad(*args,**kwargs):
        return BadAdvice(patch={'learning_rate':99.0})
    root=tmp_path/'r'
    with pytest.raises(ValueError,match='outside approved range'):
        execute(str(trial_config),str(root),trainer_fn=trainer,advisor_fn=bad)
    assert json.loads((root/'controller_state.json').read_text())['status']=='blocked'


def test_real_monitor_stop_respected(trial_config,tmp_path):
    trainer,calls=fake_trainer(diagnoses=('overfit',None))
    r=execute(str(trial_config),str(tmp_path/'r'),trainer_fn=trainer)
    assert r['attempt_count']==2
    assert all(x['injected_diagnosis'] is None for x in calls)


def test_invalid_model_data_not_reused(trial_config,tmp_path):
    trainer,_=fake_trainer()
    src=yaml.safe_load(trial_config.read_text())
    src['hf']['device']='gpu'
    trial_config.write_text(yaml.safe_dump(src))
    with pytest.raises(LoraContractError,match='CPU'):
        execute(str(trial_config),str(tmp_path/'r'),trainer_fn=trainer)
    assert not (tmp_path/'r').exists()
