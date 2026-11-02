"""Phase 6A: offline fake-MLflow contract tests; real MLflow server test pending."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from autotune.hf_mlflow import HFMLflowReporter
from autotune.hf_selfheal import execute
from test_hf_selfheal_v086 import fake_trainer, trial_config


class FakeClient:
    def __init__(self, *, error_on=None):
        self.calls = []
        self.error_on = error_on
        self.experiment = None
        self.run_num = 0

    def _do(self, name, *args, **kwargs):
        if self.error_on == name:
            raise ConnectionError(f'simulated MLflow failure: {name}')
        self.calls.append((name, args, kwargs))

    def get_experiment_by_name(self, name):
        self._do('get_experiment_by_name', name)
        return self.experiment

    def create_experiment(self, name):
        self._do('create_experiment', name)
        self.experiment = SimpleNamespace(experiment_id='e-1')
        return 'e-1'

    def create_run(self, experiment_id, tags=None):
        self._do('create_run', experiment_id, tags=tags)
        self.run_num += 1
        return SimpleNamespace(info=SimpleNamespace(run_id=f'r-{self.run_num}'))

    def log_param(self, *args, **kwargs): self._do('log_param', *args, **kwargs)
    def log_metric(self, *args, **kwargs): self._do('log_metric', *args, **kwargs)
    def set_tag(self, *args, **kwargs): self._do('set_tag', *args, **kwargs)
    def log_artifact(self, *args, **kwargs): self._do('log_artifact', *args, **kwargs)
    def set_terminated(self, *args, **kwargs): self._do('set_terminated', *args, **kwargs)


def reporter(client=None):
    return HFMLflowReporter('http://127.0.0.1:5000', 'AutoTune-AI-CPU', client=client or FakeClient())


def names(c): return [x[0] for x in c.calls]


def test_mlflow_is_optional_no_import(trial_config, tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def no_mlflow(name, *args, **kwargs):
        if name.startswith('mlflow'):
            raise AssertionError('MLflow import attempted in disabled mode')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_mlflow)
    trainer, _ = fake_trainer(diagnoses=(None,))
    result = execute(str(trial_config), str(tmp_path/'r'), trainer_fn=trainer)
    assert result['ok'] and result['attempt_count'] == 1


def test_reporter_rejects_missing_uri():
    with pytest.raises(ValueError): HFMLflowReporter('', 'demo', client=FakeClient())
    with pytest.raises(ValueError): HFMLflowReporter('http://example', '', client=FakeClient())


def test_parent_child_run_lineage(trial_config, tmp_path):
    c = FakeClient(); r = reporter(c)
    trainer, _ = fake_trainer()
    result = execute(str(trial_config), str(tmp_path/'r'), trainer_fn=trainer,
                     mlflow_reporter=r, fault_inject_overfit=True)
    assert result['ok'] and result['attempt_count'] == 2
    assert len([x for x in c.calls if x[0]=='create_run']) == 3
    children = [x for x in c.calls if x[0]=='create_run'][1:]
    assert all(x[2]['tags']['mlflow.parentRunId']=='r-1' for x in children)
    assert len([x for x in c.calls if x[0]=='set_terminated' and x[2].get('status')=='FINISHED']) == 3
    assert 'mlflow_controller_created' in (tmp_path/'r'/'controller_events.jsonl').read_text()
    assert 'mlflow_trial_created' in (tmp_path/'r'/'controller_events.jsonl').read_text()


def test_metrics_and_metadata_but_not_weights(trial_config, tmp_path):
    c = FakeClient(); r = reporter(c)
    trainer, _ = fake_trainer(diagnoses=(None,))
    execute(str(trial_config), str(tmp_path/'r'), trainer_fn=trainer, mlflow_reporter=r)
    metrics = [x for x in c.calls if x[0]=='log_metric']
    assert any(x[1][1]=='train_loss' and x[2]['step']==2 for x in metrics)
    assert any(x[1][1]=='val_loss' and x[2]['step']==2 for x in metrics)
    assert any(x[1][1]=='best_val_loss' for x in metrics)
    artifacts = [x[1][1] for x in c.calls if x[0]=='log_artifact']
    assert all(not x.endswith(('.pt','.pth','.safetensors')) for x in artifacts)
    assert any('autotune_manifest.json' in x for x in artifacts)
    assert any('controller_result.json' in x for x in artifacts)


def test_invalid_metrics_ignored(tmp_path):
    c=FakeClient(); r=reporter(c)
    r.start_controller(config_sha256='a', model='m', provider='mock', train_sha256='b',
        validation_sha256='c',max_attempts=1,fault_injected=False)
    config = tmp_path/'trial.yaml'; config.write_text('train: {}')
    r.start_trial(1, config_sha256='f', cfg={'train':{}}, config_path=config)
    out = tmp_path/'trial'; out.mkdir()
    (out/'metrics.jsonl').write_text(json.dumps({'step':2,'train_loss':None,'val_loss':10.2,
                                           'bad': float('nan'),'other':'text'})+'\n')
    ckpt=out/'checkpoint'; ckpt.mkdir()
    r.finish_trial(1, out=out, receipt_path=out/'receipt.json', checkpoint=ckpt,
                   score=10.2, diagnosis=None, fault_injected=False)
    keys=[x[1][1] for x in c.calls if x[0]=='log_metric']
    assert 'val_loss' in keys and 'train_loss' not in keys and 'bad' not in keys


def test_server_error_blocks_and_preserves_local_journal(trial_config, tmp_path):
    c = FakeClient(error_on='log_metric'); r = reporter(c)
    trainer, _ = fake_trainer(diagnoses=(None,))
    root=tmp_path/'r'
    with pytest.raises(ConnectionError,match='simulated MLflow'):
        execute(str(trial_config),str(root),trainer_fn=trainer,mlflow_reporter=r)
    assert json.loads((root/'controller_state.json').read_text())['status']=='blocked'
    assert 'controller_blocked' in (root/'controller_events.jsonl').read_text()
    assert 'FAILED' in [x[2].get('status') for x in c.calls if x[0]=='set_terminated']


def test_mlflow_is_not_enabled_from_environment_implicitly(trial_config,tmp_path,monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI','http://127.0.0.1:5000')
    trainer, _ = fake_trainer(diagnoses=(None,))
    result = execute(str(trial_config),str(tmp_path/'r'),trainer_fn=trainer)
    assert result['ok']


def test_preflight_never_instantiates_mlflow(trial_config, monkeypatch):
    import autotune.hf_mlflow as m
    def fail(*args,**kwargs): raise AssertionError('unexpected MLflow client')
    monkeypatch.setattr(m, 'HFMLflowReporter', fail)
    from autotune.hf_selfheal import preflight
    assert preflight(str(trial_config))['ok']
