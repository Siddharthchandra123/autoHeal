from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from autotune.config import ConfigError, config_sha256, load_config, validate_config
from autotune.doctor import inspect

BASELINE = yaml.safe_load(Path('config/baseline.yaml').read_text(encoding='utf8'))


def test_config_valid_and_stable_fingerprint():
    assert validate_config(deepcopy(BASELINE))['project'] == 'AutoTune-AI'
    reversed_dict = dict(reversed(list(BASELINE.items())))
    assert config_sha256(BASELINE) == config_sha256(reversed_dict)


def test_rejects_unbounded_trial_budget():
    config = deepcopy(BASELINE)
    config['safety']['max_trials'] = 200
    with pytest.raises(ConfigError, match='max_trials'):
        validate_config(config)


def test_rejects_bad_parameter_and_nan():
    for value in (-1, float('nan'), True, '0.1'):
        config = deepcopy(BASELINE)
        config['train']['learning_rate'] = value
        with pytest.raises(ConfigError):
            validate_config(config)


def test_rejects_shared_train_validation():
    config = deepcopy(BASELINE)
    config['data']['validation_file'] = config['data']['train_file']
    with pytest.raises(ConfigError, match='differ'):
        validate_config(config)


def test_doctor_passes_mock():
    result = inspect('config/baseline.yaml', 'mock')
    assert result['ok'] is True
    assert result['config_sha256']


def test_invalid_config_creates_no_run_dir(tmp_path):
    from autotune.orchestrator import run
    config = deepcopy(BASELINE)
    config['train']['eval_steps'] = 0
    file = tmp_path / 'broken.yaml'
    file.write_text(yaml.safe_dump(config))
    runs = tmp_path / 'runs'
    with pytest.raises(ConfigError):
        run(str(file), 'mock', str(runs))
    assert not runs.exists()


def test_config_loader_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / 'missing.yaml')
