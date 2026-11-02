"""Phase 6B local viewer security, response sanitization and API contract."""
from types import SimpleNamespace as NS
import math
import pytest
from fastapi.testclient import TestClient
from autotune.viewer_api import create_app, check_local_uri, ViewerStore, MAX_METRIC_POINTS

RID = 'a' * 32
PARENT = 'b' * 32


class FakeMLflow:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.exps = [NS(experiment_id='1', name='AutoTune-AI-CPU', lifecycle_stage='active')]
        self.observed = NS(info=NS(run_id=RID, experiment_id='1', status='FINISHED',
                                   start_time=123456, end_time=123789), data=NS(
            metrics={'train_loss': 10.4, 'best_val_loss': 10.36, 'secret': 4.0,
                     'grad_norm': float('nan')},
            tags={'mlflow.runName': 'AutoTune trial-01', 'mlflow.parentRunId': PARENT,
                  'autotune.role': 'trial', 'AWS_SECRET_ACCESS_KEY': 'oops'},
            params={'model': 'sshleifer/tiny-gpt2', 'train.learning_rate': '0.00025',
                    'auth_token': 'DO_NOT_SHOW', 'file_path': 'C:/private'}))

    def search_experiments(self, **kw):
        self.calls.append(('search_experiments',kw))
        if self.fail: raise OSError('secret PRIVATE_PATH: C:/test')
        return self.exps

    def search_runs(self, **kw):
        self.calls.append(('search_runs',kw))
        if self.fail: raise RuntimeError('private API key value')
        return [self.observed]

    def get_run(self, run_id):
        self.calls.append(('get_run',run_id))
        return self.observed

    def get_metric_history(self, run_id, key):
        self.calls.append(('get_metric_history', run_id, key))
        return [NS(step=i, value=i / 10) for i in range(350)] + [NS(step=400, value=float('nan'))]


@pytest.fixture
def api():
    return TestClient(create_app(client=FakeMLflow(), tracking_uri='http://127.0.0.1:5000'))


@pytest.mark.parametrize('url', [
    'http://example.com:5000','http://localhost:5000','https://127.0.0.1:5000',
    'http://127.0.0.1:5000/evil','http://127.0.0.1:5000?token=x',
    'http://user:pw@127.0.0.1:5000','file:///etc/passwd',
    'http://127.0.0.1:0','http://127.0.0.1:65536','http://127.0.0.1:5000/#evil',
])
def test_rejects_untrusted_tracking_uri(url):
    with pytest.raises(ValueError): check_local_uri(url)


def test_loopback_only():
    assert check_local_uri('http://127.0.0.1:5000/')=='http://127.0.0.1:5000'


def test_health_is_honest(api):
    assert api.get('/api/health').json()['mlflow']=='not probed'


def test_experiment_list(api):
    r=api.get('/api/experiments')
    assert r.status_code==200 and r.json()['experiments'][0]['name']=='AutoTune-AI-CPU'


def test_runs_sanitized(api):
    r=api.get('/api/experiments/1/runs').json()['runs'][0]
    assert r['tags']['mlflow.parentRunId']==PARENT
    assert r['params']['model']=='sshleifer/tiny-gpt2'
    assert 'auth_token' not in str(r) and 'DO_NOT_SHOW' not in str(r)
    assert 'AWS_SECRET_ACCESS_KEY' not in str(r)
    assert 'file_path' not in str(r)
    assert 'secret' not in r['metrics'] and 'grad_norm' not in r['metrics']


def test_run_detail(api):
    assert api.get('/api/runs/'+RID).json()['run_id']==RID


def test_metric_history_bounded(api):
    p=api.get(f'/api/runs/{RID}/metrics/train_loss').json()['points']
    assert len(p)==MAX_METRIC_POINTS
    assert p[0]['step']==0 and p[-1]['step']==349
    assert all(math.isfinite(x['value']) for x in p)


def test_invalid_experiment_and_run_ids(api):
    assert api.get('/api/experiments/../../runs').status_code in (400,404)
    assert api.get('/api/experiments/inject%27/runs').status_code==400
    assert api.get('/api/runs/nope').status_code==400
    assert api.get(f'/api/runs/{RID}/metrics/private-token').status_code==400
    assert api.get('/api/experiments/1/runs?limit=999').status_code==422


def test_no_mutation_routes(api):
    assert api.post('/api/experiments').status_code==405
    assert api.delete('/api/runs/'+RID).status_code==405


def test_cors_restrictive(api):
    good=api.options('/api/experiments', headers={'Origin':'http://localhost:3000','Access-Control-Request-Method':'GET'})
    bad=api.options('/api/experiments', headers={'Origin':'http://evil.example','Access-Control-Request-Method':'GET'})
    assert good.headers.get('access-control-allow-origin')=='http://localhost:3000'
    assert bad.status_code==400


def test_host_header_blocked(api):
    assert api.get('/api/health',headers={'Host':'evil.example'}).status_code==400


def test_does_not_leak_tracking_errors():
    fake=FakeMLflow();fake.fail=True
    c=TestClient(create_app(client=fake))
    r=c.get('/api/experiments')
    assert r.status_code==503
    assert 'PRIVATE_PATH' not in r.text


def test_query_limit_passed_to_mlflow():
    fake=FakeMLflow();c=TestClient(create_app(client=fake))
    c.get('/api/experiments/1/runs?limit=9')
    assert ('search_runs', {'experiment_ids':['1'],'max_results':9,
             'order_by':['attributes.start_time DESC']}) in fake.calls
