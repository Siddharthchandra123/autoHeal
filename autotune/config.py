"""Phase 1: explicit configuration contracts, validation and provenance.

All candidate hyperparameter patches must still go through policy.validate_patch.
Validation alone does not authorize model execution or promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .policy import INTEGERS, LIMITS


class ConfigError(ValueError):
    """Raised when a training configuration violates its static contract."""


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a YAML mapping")
    return value


def _number(value: Any, label: str, low: float, high: float, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ConfigError(f"{label} must be a finite number")
    if not low <= value <= high:
        raise ConfigError(f"{label} outside allowed range [{low}, {high}]")
    if integer and int(value) != value:
        raise ConfigError(f"{label} must be an integer")


def validate_config(config: Any, mode: str = "mock") -> dict:
    """Validate the complete baseline before allocating a run directory.

    This intentionally checks known critical fields and forbids invalid safety
    parameters; a later phase can publish a full JSON Schema for external API use.
    """
    if mode not in {"mock", "hf"}:
        raise ConfigError("mode must be 'mock' or 'hf'")
    c = _mapping(config, "configuration")
    for name in ("project", "model", "data", "train", "monitor", "safety", "meta_agent", "simulator"):
        if name not in c:
            raise ConfigError(f"Missing section: {name}")
    if not isinstance(c["project"], str) or not c["project"].strip():
        raise ConfigError("project must be a non-empty string")
    model = _mapping(c["model"], "model")
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ConfigError("model.name must be non-empty")
    _number(model.get("max_length"), "model.max_length", 32, 32768, integer=True)

    data = _mapping(c["data"], "data")
    for key in ("train_file", "validation_file"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ConfigError(f"data.{key} must be a path string")
    if data["train_file"] == data["validation_file"]:
        raise ConfigError("Train and validation files must differ")

    train = _mapping(c["train"], "train")
    for key, (low, high) in LIMITS.items():
        if key not in train:
            raise ConfigError(f"Missing training parameter: {key}")
        _number(train[key], f"train.{key}", low, high, integer=key in INTEGERS)
    if int(train["lora_r"]) not in {4, 8, 16, 32, 64}:
        raise ConfigError("train.lora_r must be one of 4, 8, 16, 32 or 64")
    for key, low, high in (("per_device_train_batch_size", 1, 1024),
                           ("eval_steps", 1, 100000), ("lora_alpha", 1, 1024)):
        _number(train.get(key), f"train.{key}", low, high, integer=True)

    monitor = _mapping(c["monitor"], "monitor")
    for key in ("min_evaluations", "patience", "flat_window"):
        _number(monitor.get(key), f"monitor.{key}", 1, 10000, integer=True)
    for key in ("min_delta", "generalization_gap", "underfit_loss_floor", "flat_threshold"):
        _number(monitor.get(key), f"monitor.{key}", 0, 10000)
    for key, low, high, integer in (("loss_spike_ratio", 1.01, 100, False),
                                    ("loss_spike_patience", 2, 20, True),
                                    ("grad_norm_warning", 0.01, 1e9, False)):
        if key in monitor:
            _number(monitor[key], f"monitor.{key}", low, high, integer=integer)
    safety = _mapping(c["safety"], "safety")
    _number(safety.get("max_trials"), "safety.max_trials", 1, 12, integer=True)
    agent = _mapping(c["meta_agent"], "meta_agent")
    if not isinstance(agent.get("enabled"), bool):
        raise ConfigError("meta_agent.enabled must be a boolean")
    if not isinstance(agent.get("model"), str) or not agent["model"].strip():
        raise ConfigError("meta_agent.model must be non-empty")
    if agent.get("provider", "meta") not in {"mock", "meta"}:
        raise ConfigError("meta_agent.provider must be mock or meta")
    sim = _mapping(c["simulator"], "simulator")
    if sim.get("scenario") not in {"overfit", "underfit", "healthy"}:
        raise ConfigError("simulator.scenario must be overfit, underfit, or healthy")
    return c


def load_config(path: str | Path, mode: str = "mock") -> dict:
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"Config file not found: {file}")
    try:
        payload = yaml.safe_load(file.read_text(encoding="utf8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc
    return validate_config(payload, mode)


def config_sha256(config: dict) -> str:
    """Content identity independent of YAML whitespace and field ordering."""
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    return hashlib.sha256(raw).hexdigest()
