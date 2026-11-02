import json
from pathlib import Path
import shutil

import pytest
import yaml

from autotune.hf_classification import (HFClassificationError, classification_plan,
                                        save_manifest, verify_manifest)


@pytest.fixture
def fixture_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'sample_data' / 'hf_classification').mkdir(parents=True)
    for key in ('train', 'validation'):
        shutil.copyfile(root / 'sample_data' / 'hf_classification' / f'{key}.jsonl',
                        tmp_path / 'sample_data' / 'hf_classification' / f'{key}.jsonl')
    cfg = yaml.safe_load((root / 'config' / 'hf_bert_tiny_cpu.yaml').read_text('utf8'))
    for key in ('train_file', 'validation_file'):
        cfg['data'][key] = str(tmp_path / cfg['data'][key])
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf8')
    return path


def mutate(path, fn):
    data = yaml.safe_load(path.read_text('utf8'))
    fn(data)
    path.write_text(yaml.safe_dump(data), encoding='utf8')


def test_normal_plan(fixture_config):
    plan = classification_plan(fixture_config)
    assert plan['train_rows'] == 12
    assert plan['validation_rows'] == 4
    assert plan['labels'] == ['negative', 'positive']
    assert plan['task'] == 'text-classification'
    assert plan['device'] == 'cpu'


def test_config_hash_stable(fixture_config):
    assert classification_plan(fixture_config)['config_sha256'] == classification_plan(fixture_config)['config_sha256']


@pytest.mark.parametrize('name', ['../../private', 'foo/../../../etc', 'repo name', 'a/b/c', '', 'evil\\module'])
def test_invalid_model_names(fixture_config, name):
    mutate(fixture_config, lambda c: c['model'].update(name=name))
    with pytest.raises(HFClassificationError):
        classification_plan(fixture_config)


def test_unknown_task(fixture_config):
    mutate(fixture_config, lambda c: c.update(task='image-classification'))
    with pytest.raises(HFClassificationError):
        classification_plan(fixture_config)


def test_bad_learning_rate(fixture_config):
    mutate(fixture_config, lambda c: c['train'].update(learning_rate=12))
    with pytest.raises(HFClassificationError):
        classification_plan(fixture_config)


def test_bool_steps(fixture_config):
    mutate(fixture_config, lambda c: c['train'].update(max_steps=True))
    with pytest.raises(HFClassificationError):
        classification_plan(fixture_config)


def test_eval_without_steps(fixture_config):
    mutate(fixture_config, lambda c: c['train'].update(eval_steps=50))
    with pytest.raises(HFClassificationError):
        classification_plan(fixture_config)


def test_validation_leakage(fixture_config):
    cfg = yaml.safe_load(fixture_config.read_text('utf8'))
    train = Path(cfg['data']['train_file'])
    val = Path(cfg['data']['validation_file'])
    rec = json.loads(train.read_text('utf8').splitlines()[0])
    val.write_text(val.read_text('utf8') + json.dumps(rec) + '\n', 'utf8')
    with pytest.raises(HFClassificationError, match='leakage'):
        classification_plan(fixture_config)


def test_unknown_class_in_validation(fixture_config):
    cfg = yaml.safe_load(fixture_config.read_text('utf8'))
    val = Path(cfg['data']['validation_file'])
    rows = [json.loads(s) for s in val.read_text('utf8').splitlines()]
    rows[0]['label'] = 'neutral'
    val.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', 'utf8')
    with pytest.raises(HFClassificationError, match='unseen label'):
        classification_plan(fixture_config)


def test_data_change_invalidate_plan(fixture_config):
    before = classification_plan(fixture_config)['train_sha256']
    cfg = yaml.safe_load(fixture_config.read_text('utf8'))
    path = Path(cfg['data']['train_file'])
    path.write_text(path.read_text('utf8').replace('simple setup', 'simple configuration'), 'utf8')
    assert classification_plan(fixture_config)['train_sha256'] != before


def test_detect_tampered_model_artifact(fixture_config, tmp_path):
    plan = classification_plan(fixture_config)
    folder = tmp_path / 'out'
    folder.mkdir()
    (folder / 'model.safetensors').write_bytes(b'test-weight-bytes')
    (folder / 'config.json').write_text('{}', 'utf8')
    save_manifest(folder, plan, kind='model')
    assert verify_manifest(folder, plan, kind='model')['ok']
    (folder / 'model.safetensors').write_bytes(b'tampered')
    with pytest.raises(HFClassificationError, match='integrity'):
        verify_manifest(folder, plan, kind='model')


def test_checkpoint_requires_state(fixture_config, tmp_path):
    plan = classification_plan(fixture_config)
    folder = tmp_path / 'checkpoint-3'
    folder.mkdir()
    (folder / 'trainer_state.json').write_text('{}', 'utf8')
    with pytest.raises(HFClassificationError, match='Incomplete'):
        save_manifest(folder, plan, kind='checkpoint')


def test_preflight_no_download(monkeypatch, fixture_config):
    # No HF libraries imported and no remote repository calls during validation.
    import sys
    old = set(sys.modules)
    result = classification_plan(fixture_config)
    assert result['device'] == 'cpu'
    assert not {'transformers', 'datasets', 'peft'} & (set(sys.modules) - old)
