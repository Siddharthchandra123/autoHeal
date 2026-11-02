"""Phase 5: actual torch training + checkpoint integrity; CPU, offline only."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil

import pytest
import yaml

torch = pytest.importorskip('torch')

from autotune.config import config_sha256, load_config
from autotune.real_training import (
    RealTrainingError, batchify, dataset_identity, load_verified_checkpoint,
    make_model, read_texts, train,
)
from autotune.real_recovery import run_real, verified_receipt, SCHEMA
from autotune.recovery import RecoveryError

BASE = Path(__file__).parents[1]


@pytest.fixture
def case(tmp_path):
    c = deepcopy(load_config(BASE / 'config/real_cpu.yaml', mode='hf'))
    data = tmp_path / 'data'
    data.mkdir()
    for split in ('train', 'validation'):
        target = data / f'{split}.jsonl'
        shutil.copyfile(BASE / 'sample_data' / f'{split}.jsonl', target)
        c['data'][f'{split}_file'] = str(target)
    c['train']['num_train_epochs'] = 3
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(c), encoding='utf-8')
    return c, path, tmp_path


def test_cpu_dataset_fingerprint_is_stable(case):
    c, *_ = case
    assert len(dataset_identity(c)) == 64
    assert dataset_identity(c) == dataset_identity(c)


def test_changed_data_changes_identity(case):
    c, *_ = case
    before = dataset_identity(c)
    with open(c['data']['train_file'], 'a', encoding='utf8') as f:
        f.write('{"text":"Extra training text"}\n')
    assert dataset_identity(c) != before


def test_matching_train_validation_content_rejected(case):
    c, *_ = case
    shutil.copyfile(c['data']['train_file'], c['data']['validation_file'])
    with pytest.raises(RealTrainingError, match='differ'):
        dataset_identity(c)


def test_bad_jsonl_rejected(case):
    c, *_ = case
    p = Path(c['data']['train_file'])
    p.write_text('{bad', encoding='utf8')
    with pytest.raises(RealTrainingError, match='JSONL'):
        read_texts(p)


def test_missing_text_rejected(case):
    c, *_ = case
    p = Path(c['data']['train_file'])
    p.write_text('{"nottext":"xx"}\n{"text":"fine text"}', encoding='utf8')
    with pytest.raises(RealTrainingError, match='text'):
        read_texts(p)


def test_short_dataset_rejected(case):
    c, *_ = case
    p = Path(c['data']['train_file'])
    p.write_text('{"text":"one text"}', encoding='utf8')
    with pytest.raises(RealTrainingError, match='2 examples'):
        read_texts(p)


def test_batchify_alignment_and_padding():
    x, y = batchify(['abcd', 'hello world'], 32)
    assert x.shape == (2, 32)
    assert x[0, 0] == ord('a')
    assert y[0, 0] == ord('b')
    assert y[0, 3] == -100


def test_model_loss_backpropagates():
    m = make_model()
    x, y = batchify(['hello world', 'gradient descent'], 32)
    loss = torch.nn.functional.cross_entropy(m(x).flatten(0, 1), y.flatten(), ignore_index=-100)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters())


def test_checkpoint_actual_weight_and_optimizer_load(case):
    c, _, root = case
    trial_dir = root / 'trial-01'
    result = train(c, trial_dir, 1)
    path = Path(result.checkpoint)
    assert result.best_step >= 1
    assert path.name == 'best_checkpoint.pt'
    assert result.best_val_loss > 0
    obj = load_verified_checkpoint(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   expected_config_sha256=config_sha256(c), expected_data_sha256=dataset_identity(c))
    assert obj['epoch'] == result.best_step
    assert obj['optimizer_state']['state']
    assert obj['model_state']['head.weight'].shape[0] == 256
    assert (trial_dir / 'monitor_decisions.jsonl').exists()
    assert (trial_dir / 'metrics.jsonl').exists()


def test_checkpoint_wrong_config_rejected(case):
    c, _, root = case
    result = train(c, root / 'trial-01', 1)
    p = Path(result.checkpoint)
    with pytest.raises(RealTrainingError, match='identity mismatch'):
        load_verified_checkpoint(p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            expected_config_sha256='0' * 64, expected_data_sha256=dataset_identity(c))


def test_checkpoint_wrong_dataset_rejected(case):
    c, _, root = case
    result = train(c, root / 'trial-01', 1)
    p = Path(result.checkpoint)
    with pytest.raises(RealTrainingError, match='identity mismatch'):
        load_verified_checkpoint(p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            expected_config_sha256=config_sha256(c), expected_data_sha256='0' * 64)


def test_checkpoint_tamper_rejected(case):
    c, _, root = case
    result = train(c, root / 'trial-01', 1)
    p = Path(result.checkpoint)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open('ab') as f:
        f.write(b'TAMPER')
    with pytest.raises(RealTrainingError, match='SHA256'):
        load_verified_checkpoint(p, expected_sha256=digest,
            expected_config_sha256=config_sha256(c), expected_data_sha256=dataset_identity(c))


def test_invalid_model_name_rejected(case):
    c, _, root = case
    c['model']['name'] = 'a-different-model'
    with pytest.raises(RealTrainingError, match='only supports'):
        train(c, root / 'trial-01', 1)


def test_real_controller_completed_and_idempotent_resume(case):
    _, path, root = case
    result = run_real(str(path), str(root / 'runs'))
    assert result['ok'] and result['mode'] == 'REAL PyTorch CPU'
    assert result['status'] == 'completed'
    assert result['attempt_count'] == 1
    runroot = Path(result['experiment'])
    assert verified_receipt(runroot, json.loads((runroot / 'recovery_state.json').read_text())['attempts'][0])
    assert (runroot / 'active.yaml').exists()
    again = run_real(str(path), str(root / 'runs'), resume=str(runroot))
    assert again['attempt_count'] == 1
    assert again['checkpoint_sha256'] == result['checkpoint_sha256']
    assert len(list(runroot.glob('trial-*'))) == 1


def test_tampered_checkpoint_blocks_completed_resume(case):
    _, path, root = case
    result = run_real(str(path), str(root / 'runs'))
    runroot = Path(result['experiment'])
    with (runroot / 'trial-01' / 'best_checkpoint.pt').open('ab') as handle:
        handle.write(b'BAD')
    with pytest.raises(RecoveryError, match='Invalid real checkpoint'):
        run_real(str(path), str(root / 'runs'), resume=str(runroot))


def test_changed_dataset_blocks_resume(case):
    config, path, root = case
    result = run_real(str(path), str(root / 'runs'))
    with open(config['data']['validation_file'], 'a', encoding='utf8') as f:
        f.write('{"text":"A new validation sentence"}\n')
    with pytest.raises(RecoveryError, match='Dataset changed'):
        run_real(str(path), str(root / 'runs'), resume=result['experiment'])


def test_uncertain_running_blocks_resume(case):
    config, path, root = case
    runs = root / 'runs'
    exp = runs / 'manual-uncertain'
    exp.mkdir(parents=True)
    state = {'schema': SCHEMA, 'config_sha256': config_sha256(config),
             'dataset_sha256': dataset_identity(config), 'status': 'running', 'revision': 0,
             'attempts': [{'trial': 1, 'status': 'running'}], 'best': None, 'max_trials': 2}
    (exp / 'recovery_state.json').write_text(json.dumps(state), encoding='utf8')
    with pytest.raises(RecoveryError, match='RUNNING without validated receipt'):
        run_real(str(path), str(runs), resume=str(exp))
    assert json.loads((exp / 'recovery_state.json').read_text())['status'] == 'blocked'


def test_real_recovery_policy_retry_and_budget(case):
    c, path, root = case
    # Deliberately permissive thresholds FORCE underfit classification for testing
    # state transitions; this is NOT evidence of genuine model underfitting.
    c['train']['learning_rate'] = 0.0001
    c['monitor']['flat_threshold'] = 1.0
    c['monitor']['underfit_loss_floor'] = 4.0
    c['safety']['max_trials'] = 2
    path.write_text(yaml.safe_dump(c), encoding='utf8')
    result = run_real(str(path), str(root / 'runs'))
    assert result['status'] == 'budget_exhausted'
    assert result['attempt_count'] == 2
    exp = Path(result['experiment'])
    state = json.loads((exp / 'recovery_state.json').read_text())
    assert [a['status'] for a in state['attempts']] == ['stopped', 'stopped']
    assert (exp / 'trial-01' / 'advisor_decision.json').is_file()
    assert c['train']['learning_rate'] < state['attempts'][1]['config']['train']['learning_rate']
    assert (exp / 'trial-02' / 'best_checkpoint.pt').is_file()
