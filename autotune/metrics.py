"""Phase 2: conservative, auditable metric diagnostics.

No ML inference or mathematical guarantee is implied. Loss-based diagnostics
are heuristics; a task-specific baseline is necessary to interpret underfit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math


@dataclass(frozen=True)
class MetricEvent:
    """One aligned train/validation evaluation, not two independent log steps.

    `step` is a positive, strictly increasing global training step. `timestamp_utc`
    may be absent in synthetic data. Additional metrics remain optional.
    """
    step: int
    train_loss: float
    val_loss: float
    timestamp_utc: str | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    gpu_peak_allocated_gib: float | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> MetricEvent:
        if not isinstance(row, dict):
            raise ValueError("Metric event must be a mapping")
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("step must be a positive integer")
        for name in ("train_loss", "val_loss"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} is missing or non-numeric")
        kwargs: dict[str, Any] = {}
        for name in ("learning_rate", "grad_norm", "gpu_peak_allocated_gib"):
            value = row.get(name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be numeric when supplied")
                kwargs[name] = float(value)
        ts = row.get("timestamp_utc")
        if ts is not None and (not isinstance(ts, str) or not ts.strip()):
            raise ValueError("timestamp_utc must be a nonempty ISO-8601 string")
        return cls(step=step, train_loss=float(row["train_loss"]),
                   val_loss=float(row["val_loss"]), timestamp_utc=ts, **kwargs)


@dataclass(frozen=True)
class Decision:
    state: str
    explanation: str
    severity: str = "info"  # info | warning | critical
    action: str = "continue"  # continue | stop
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return self.action == "stop"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _monitor_params(settings: dict) -> tuple[int, int, float]:
    return (max(3, int(settings["min_evaluations"])),
            max(2, int(settings["patience"])), float(settings["min_delta"]))


def diagnose(history: list[dict], settings: dict) -> Decision:
    """Produce a typed decision for aligned, ordered evaluation observations.

    Invalid/missing observations do not generate an early stop. Clearly
    nonfinite numeric train/validation/gradient values do generate a stop.
    No target metric is estimated from training loss alone.
    """
    if not history:
        return Decision("none", "No evaluation observations yet", evidence={"evaluations": 0})
    events: list[MetricEvent] = []
    for index, row in enumerate(history):
        # The JSONL logger encodes NaN/Inf as null plus a named fault marker.
        if isinstance(row, dict) and row.get("nonfinite_metrics"):
            metrics = row["nonfinite_metrics"]
            if isinstance(metrics, list) and any(k in metrics for k in ("train_loss", "val_loss", "grad_norm")):
                return Decision("diverged", "Non-finite logged training signal",
                                "critical", "stop", {"step": row.get("step"),
                                                       "metrics": metrics})
        try:
            # Legacy Phase 1 history omitted steps; infer index for that API.
            normalized = ({"step": index + 1, **row} if isinstance(row, dict) and "step" not in row else row)
            observation = MetricEvent.from_mapping(normalized)
        except (ValueError, TypeError) as exc:
            return Decision("insufficient_data", f"Invalid or incomplete evaluation #{index+1}: {exc}",
                            "warning", evidence={"record_index": index + 1})
        if events and observation.step <= events[-1].step:
            return Decision("insufficient_data", "Evaluation steps must be strictly increasing",
                            "warning", evidence={"step": observation.step,
                                                  "previous_step": events[-1].step})
        for name in ("train_loss", "val_loss", "grad_norm"):
            value = getattr(observation, name)
            if value is not None and not math.isfinite(value):
                return Decision("diverged", f"Non-finite {name} at step {observation.step}",
                                "critical", "stop", {"step": observation.step, "metric": name})
        events.append(observation)

    min_evals, patience, delta = _monitor_params(settings)
    latest = events[-1]
    if len(events) < min_evals:
        return Decision("none", "Collecting warm-up evaluations", evidence={
            "evaluations": len(events), "min_evaluations": min_evals})

    # A single spike is only an alert. Repeated sharp increases in both losses
    # indicate instability; preserve existing best weights and stop this trial.
    spike_ratio = float(settings.get("loss_spike_ratio", 2.0))
    spike_patience = max(2, int(settings.get("loss_spike_patience", 2)))
    if len(events) >= spike_patience + 1:
        tail = events[-(spike_patience + 1):]
        repeated = all(tail[i].train_loss > max(tail[i-1].train_loss, 1e-12)*spike_ratio and
                       tail[i].val_loss > max(tail[i-1].val_loss, 1e-12)*spike_ratio
                       for i in range(1, len(tail)))
        if repeated:
            return Decision("diverged", "Repeated simultaneous train and validation loss spikes",
                            "critical", "stop", {"step": latest.step, "spike_ratio": spike_ratio,
                                                   "spike_count": spike_patience})

    gap = latest.val_loss - latest.train_loss
    recent = events[-(patience + 1):]
    if len(recent) == patience + 1:
        val_worse = all(recent[i].val_loss-recent[i-1].val_loss > delta
                        for i in range(1, len(recent)))
        train_better = recent[-1].train_loss < recent[0].train_loss - delta
        if val_worse and train_better and gap > float(settings["generalization_gap"]):
            return Decision("overfit", f"Validation loss rose for {patience} evaluations while train loss fell; gap={gap:.3f}",
                            "critical", "stop", {"step": latest.step, "gap": round(gap, 6),
                                                   "patience": patience, "best_val_loss": min(e.val_loss for e in events)})

    w = max(3, int(settings["flat_window"]))
    recent = events[-w:]
    if len(recent) == w:
        train_range = max(e.train_loss for e in recent)-min(e.train_loss for e in recent)
        val_range = max(e.val_loss for e in recent)-min(e.val_loss for e in recent)
        flat = train_range < float(settings["flat_threshold"]) and val_range < float(settings["flat_threshold"])
        if flat:
            evidence = {"step": latest.step, "train_range": round(train_range,6),
                        "val_range": round(val_range,6), "window": w}
            if min(e.train_loss for e in recent) > float(settings["underfit_loss_floor"]) and \
                    min(e.val_loss for e in recent) > float(settings["underfit_loss_floor"]):
                return Decision("underfit", "Train and validation losses remain high and nearly flat",
                                "critical", "stop", evidence)
            return Decision("plateau", "Losses are nearly flat; insufficient evidence of underfitting",
                            "warning", evidence=evidence)

    if len(events) > 1:
        prev = events[-2]
        if (latest.train_loss > max(prev.train_loss, 1e-12)*spike_ratio and
                latest.val_loss > max(prev.val_loss, 1e-12)*spike_ratio):
            return Decision("loss_spike", "A single train/validation loss spike needs confirmation",
                            "warning", evidence={"step": latest.step, "spike_ratio": spike_ratio})

    threshold = float(settings.get("grad_norm_warning", 1000.0))
    if latest.grad_norm is not None and latest.grad_norm > threshold:
        return Decision("gradient_warning", "Gradient norm exceeded the warning threshold",
                        "warning", evidence={"step": latest.step,
                                              "grad_norm": latest.grad_norm, "threshold": threshold})
    return Decision("none", "No stopping condition triggered", evidence={
        "evaluations": len(events), "step": latest.step, "gap": round(gap, 6)})


class MonitorStream:
    """Append-only evidence: every observation produces a diagnostic decision."""
    def __init__(self, path: str | Path, settings: dict):
        self.path = Path(path)
        self.settings = settings
        self.history: list[dict] = []

    def observe(self, row: dict) -> Decision:
        from .tracking import event
        self.history.append(dict(row))
        decision = diagnose(self.history, self.settings)
        event(self.path, {"type": "monitor_decision", "step": row.get("step"),
                          "decision": decision.as_dict()})
        return decision
