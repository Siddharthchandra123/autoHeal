"""Phase 3 fault-injection and safe recovery tests (CPU synthetic data)."""
import json
from pathlib import Path
import pytest
import yaml
from autotune.recovery import (RecoveryError, SimulatedInterruption,
                               run_durable, _validate_receipt)


def config_file(tmp_path, scenario='overfit', max_trials=3):
    cfg = yaml.safe_load(Path('config/baseline.yaml').read_text(encoding='utf8'))
    cfg['meta_agent']['enabled'] = False
    cfg['simulator']['scenario'] = scenario
    cfg['safety']['max_trials'] = max_trials
    file = tmp_path / 'baseline.yaml'
    file.write_text(yaml.safe_dump(cfg), encoding='utf8')
    return file


def latest_root(tmp_path):
    return next((tmp_path / 'runs').iterdir())


def test_recovery_end_to_end_best_global_checkpoint(tmp_path):
    config = config_file(tmp_path)
    result = run_durable(str(config), str(tmp_path / 'runs'))
    assert result['status'] == 'completed'
    assert len(result['attempts']) == 2
    assert [x['status'] for x in result['attempts']] == ['stopped', 'completed']
    assert result['best']['trial'] == 2
    assert result['best']['best_val_loss'] == pytest.approx(.58)
    root = Path(result['experiment'])
    assert (root / 'active.yaml').exists()
    assert (root / 'recovery_events.jsonl').exists()
    assert not (tmp_path / 'optimized.yaml').exists()  # non-mutating default
    assert not (root / '.controller.lock').exists()
    assert _validate_receipt(root, result['attempts'][1])['best_val_loss'] == pytest.approx(.58)


def test_terminal_resume_does_not_rerun_or_write_extra_events(tmp_path):
    conf = config_file(tmp_path)
    first = run_durable(str(conf), str(tmp_path/'runs'))
    root = Path(first['experiment'])
    events = (root/'recovery_events.jsonl').read_text()
    second = run_durable(str(conf), resume=str(root))
    assert first['best'] == second['best']
    assert (root/'recovery_events.jsonl').read_text() == events


def test_valid_receipt_is_reconciled_without_retraining(tmp_path):
    conf = config_file(tmp_path, scenario='underfit', max_trials=1)
    with pytest.raises(SimulatedInterruption):
        run_durable(str(conf), str(tmp_path/'runs'), fault_at='after_receipt_before_commit')
    root = latest_root(tmp_path)
    lines = (root/'trial-01'/'metrics.jsonl').read_text()
    result = run_durable(str(conf), resume=str(root))
    assert result['status'] == 'budget_exhausted'
    assert len(result['attempts']) == 1
    assert (root/'trial-01'/'metrics.jsonl').read_text() == lines


def test_interrupted_without_receipt_blocks_automatic_retry(tmp_path):
    conf = config_file(tmp_path)
    with pytest.raises(SimulatedInterruption):
        run_durable(str(conf), str(tmp_path/'runs'), fault_at='after_training_before_receipt')
    root = latest_root(tmp_path)
    with pytest.raises(RecoveryError, match='Prior attempt status RUNNING'):
        run_durable(str(conf), resume=str(root))
    ledger = json.loads((root/'recovery_state.json').read_text())
    assert ledger['status'] == 'blocked'
    assert len(ledger['attempts']) == 1
    assert not (root/'trial-02').exists()


def test_explicit_abandon_and_retry_consumes_trial_budget(tmp_path):
    conf = config_file(tmp_path)
    with pytest.raises(SimulatedInterruption):
        run_durable(str(conf), str(tmp_path/'runs'), fault_at='after_state_running')
    root = latest_root(tmp_path)
    result = run_durable(str(conf), resume=str(root), abandon_interrupted=True)
    assert len(result['attempts']) == 3
    assert result['attempts'][0]['status'] == 'abandoned'
    assert result['attempts'][1]['status'] == 'stopped'
    assert result['attempts'][2]['status'] == 'completed'
    assert result['best']['trial'] == 3


def test_digest_tampering_blocks_reconciliation(tmp_path):
    conf = config_file(tmp_path)
    with pytest.raises(SimulatedInterruption):
        run_durable(str(conf), str(tmp_path/'runs'), fault_at='after_receipt_before_commit')
    root = latest_root(tmp_path)
    ckpt = root/'trial-01'/'best_checkpoint.json'
    ckpt.write_text(ckpt.read_text() + ' ', encoding='utf8')
    with pytest.raises(RecoveryError, match='integrity failure'):
        run_durable(str(conf), resume=str(root))
    state = json.loads((root/'recovery_state.json').read_text())
    assert state['status'] == 'blocked'
    assert state['best'] is None
    assert not (root/'active.yaml').exists()


def test_concurrent_lock_refuses_to_start(tmp_path):
    conf = config_file(tmp_path)
    with pytest.raises(SimulatedInterruption):
        run_durable(str(conf), str(tmp_path/'runs'), fault_at='after_state_running')
    root = latest_root(tmp_path)
    (root/'.controller.lock').write_text('simulated previous process')
    with pytest.raises(RecoveryError, match='Controller lock exists'):
        run_durable(str(conf), resume=str(root))
    with pytest.raises(RecoveryError, match='Prior attempt status RUNNING'):
        run_durable(str(conf), resume=str(root), clear_stale_lock=True)
    assert not (root/'.controller.lock').exists()
    assert 'manual_stale_lock_clear' in (root/'recovery_events.jsonl').read_text()


def test_changed_baseline_rejected_on_resume(tmp_path):
    conf = config_file(tmp_path)
    with pytest.raises(SimulatedInterruption):
        run_durable(str(conf), str(tmp_path/'runs'), fault_at='after_state_running')
    root = latest_root(tmp_path)
    y = yaml.safe_load(conf.read_text())
    y['train']['learning_rate'] = .0002
    conf.write_text(yaml.safe_dump(y))
    with pytest.raises(RecoveryError, match='Input config changed'):
        run_durable(str(conf), resume=str(root))


def test_budget_exhaustion_retains_verified_best(tmp_path):
    conf = config_file(tmp_path, max_trials=1)
    result = run_durable(str(conf), str(tmp_path/'runs'))
    assert result['status'] == 'budget_exhausted'
    assert result['attempts'][0]['status'] == 'stopped'
    assert result['best']['trial'] == 1
    assert (Path(result['experiment'])/'active.yaml').exists()


def test_writeback_only_with_explicit_optin(tmp_path):
    conf = config_file(tmp_path)
    result = run_durable(str(conf), str(tmp_path/'runs'), write_back=True)
    assert yaml.safe_load((tmp_path/'optimized.yaml').read_text()) == yaml.safe_load((Path(result['experiment'])/'active.yaml').read_text())
    assert yaml.safe_load(conf.read_text())['train']['lora_dropout'] == .05


def test_corrupt_best_checkpoint_cannot_be_selected(tmp_path, monkeypatch):
    from autotune import recovery
    original = recovery._validate_receipt
    calls = [0]
    def corrupt_after_training(root, attempt):
        calls[0] += 1
        if calls[0] == 3:
            file = root/'trial-02'/'best_checkpoint.json'
            file.write_text(file.read_text()+'tamper')
        return original(root,attempt)
    monkeypatch.setattr(recovery,'_validate_receipt', corrupt_after_training)
    conf = config_file(tmp_path)
    with pytest.raises(RecoveryError, match='Best checkpoint failed final verification'):
        run_durable(str(conf), str(tmp_path/'runs'))
    assert not (latest_root(tmp_path)/'active.yaml').exists()


def test_recovery_cli_help_importable():
    from autotune.recovery_cli import main
    with pytest.raises(SystemExit) as info:
        main(['--help'])
    assert info.value.code == 0
