"""Signal quality, early-stop safety and evidence contract tests."""
import json
import math
from pathlib import Path
import pytest
import yaml
from autotune.metrics import MetricEvent, MonitorStream, diagnose
from autotune.monitor_cli import replay


@pytest.fixture
def settings():
    return dict(min_evaluations=4, patience=2, min_delta=.01,
                generalization_gap=.2, underfit_loss_floor=1.3,
                flat_window=4, flat_threshold=.04,
                loss_spike_ratio=2.0, loss_spike_patience=2,
                grad_norm_warning=1000.0)


def h(pairs):
    return [dict(step=i, train_loss=a, val_loss=b) for i,(a,b) in enumerate(pairs,1)]


def test_metric_event_typed():
    event = MetricEvent.from_mapping({"step": 8, "train_loss": .52, "val_loss": .61,
                                      "grad_norm": 12.0, "timestamp_utc": "2026-11-01T00:00:00Z"})
    assert event.grad_norm == 12.0 and event.step == 8


@pytest.mark.parametrize("row", [{"step": 0, "train_loss": 1., "val_loss": 1.},
                                 {"step": True, "train_loss": 1., "val_loss": 1.},
                                 {"step": 1, "train_loss": "bad", "val_loss": 1.},
                                 {"step": 1, "train_loss": 1.},
                                 {"step": 1, "train_loss": 1., "val_loss": 1., "grad_norm": "bad"}])
def test_invalid_event_rejected(row):
    with pytest.raises(ValueError):
        MetricEvent.from_mapping(row)


def test_missing_metric_is_warning_not_stop(settings):
    dec = diagnose([{"step":1, "train_loss":.5}], settings)
    assert dec.state == "insufficient_data" and dec.severity == "warning"
    assert not dec.should_stop


def test_duplicate_or_reversed_steps_not_stop(settings):
    dec = diagnose(h([(1.,1.),(.9,.9)]) + [{"step": 2, "train_loss": .7, "val_loss": .8}], settings)
    assert dec.state == "insufficient_data" and not dec.should_stop


def test_nonfinite_val_loss_stops(settings):
    dec = diagnose([{"step":1,"train_loss":1.,"val_loss":float("nan")}], settings)
    assert dec.state == "diverged" and dec.should_stop
    assert dec.evidence["metric"] == "val_loss"


def test_nonfinite_gradient_stops(settings):
    dec = diagnose([{"step":1,"train_loss":1.,"val_loss":1., "grad_norm": math.inf}], settings)
    assert dec.state == "diverged" and dec.should_stop


def test_isolated_spike_is_warning(settings):
    hist = h([(1.2,1.2),(1.1,1.1),(1.,1.),(.9,.9),(3.,3.)])
    dec = diagnose(hist,settings)
    assert dec.state == "loss_spike" and dec.action == "continue"


def test_repeated_spikes_stop(settings):
    hist = h([(1.2,1.2),(1.1,1.1),(1.,1.),(2.5,2.5),(6.,6.)])
    dec = diagnose(hist,settings)
    assert dec.state == "diverged" and dec.should_stop


def test_flat_low_losses_are_only_plateau_warning(settings):
    hist = h([(.40,.46),(.399,.459),(.398,.458),(.398,.457)])
    dec = diagnose(hist,settings)
    assert dec.state == "plateau" and not dec.should_stop


def test_grad_norm_warning(settings):
    hist = h([(1.5,1.5),(1.4,1.4),(1.3,1.3),(1.2,1.2)])
    hist[-1]["grad_norm"] = 1400.
    dec = diagnose(hist,settings)
    assert dec.state == "gradient_warning" and dec.action == "continue"


def test_noisy_curve_not_overfit(settings):
    hist = h([(1.7,1.8),(1.6,1.7),(1.45,1.72),(1.3,1.69),(1.15,1.73)])
    assert diagnose(hist,settings).state == "none"


def test_short_warmup_never_stops(settings):
    dec = diagnose(h([(2.,2.),(1.9,1.8)]),settings)
    assert dec.state == "none" and dec.action == "continue"


def test_stream_creates_audit_evidence(tmp_path,settings):
    stream = MonitorStream(tmp_path/"monitor_decisions.jsonl",settings)
    for row in h([(2.,1.8),(1.8,1.7),(1.6,1.65),(1.4,1.6)]):
        stream.observe(row)
    items = [json.loads(l) for l in stream.path.read_text().splitlines()]
    assert len(items) == 4 and items[-1]["decision"]["severity"] == "info"
    assert items[-1]["decision"]["evidence"]["step"] == 4


def test_replay_cli_contract(tmp_path):
    cfg = yaml.safe_load(Path("config/baseline.yaml").read_text())
    config_path = tmp_path/"baseline.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    source = tmp_path/"metrics.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in h([(1.9,1.9),(1.8,1.7),(1.7,1.6),(1.6,1.5)])))
    report = replay(source, config_path)
    assert report["evaluations"] == 4 and report["stops"] == 0
    assert len(report["decisions"]) == 4


def test_replay_invalid_jsonl(tmp_path):
    source = tmp_path/"bad.jsonl"
    source.write_text("{invalid}\n")
    with pytest.raises(ValueError, match="line 1"):
        replay(source, "config/baseline.yaml")


def test_nonfinite_jsonl_replay_remains_critical(tmp_path):
    from autotune.tracking import Tracker
    cfg = yaml.safe_load(Path("config/baseline.yaml").read_text())
    t = Tracker(tmp_path/"trial", 1, cfg)
    t.log(1, float("nan"), 1.4, extra={"grad_norm": math.inf})
    t.finish("diverged", "nonfinite")
    line = (tmp_path/"trial"/"metrics.jsonl").read_text()
    assert "NaN" not in line and "Infinity" not in line
    report = replay(tmp_path/"trial"/"metrics.jsonl", "config/baseline.yaml")
    assert report["stops"] == 1
    assert report["last_decision"]["state"] == "diverged"


def test_integration_creates_diagnostic_stream(tmp_path):
    from autotune.orchestrator import run
    cfg = yaml.safe_load(Path("config/baseline.yaml").read_text())
    path = tmp_path/"baseline.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = run(str(path), "mock", str(tmp_path/"runs"), write_back=False)
    root = Path(result["experiment"])
    decisions = [json.loads(v) for v in (root/"trial-01"/"monitor_decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == 6
    assert decisions[-1]["decision"]["state"] == "overfit"
    assert decisions[-1]["decision"]["action"] == "stop"
    assert (root/"trial-02"/"monitor_decisions.jsonl").exists()
