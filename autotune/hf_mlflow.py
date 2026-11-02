"""Opt-in MLflow lineage for the real Hugging Face self-healing controller.

This reporter logs small configuration/metrics/provenance files only. Actual
adapter tensors and Trainer checkpoint weights are intentionally NOT uploaded.
With tracking disabled, no MLflow import, network call, or state change occurs.
MLflow failure while enabled is explicit: it must not be mistaken for success.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class HFMLflowReporter:
    def __init__(self, tracking_uri: str, experiment: str, *, client: Any = None):
        if not tracking_uri or not isinstance(tracking_uri, str):
            raise ValueError('An explicit, nonempty MLflow tracking URI is required')
        if not experiment or not isinstance(experiment, str):
            raise ValueError('An MLflow experiment name is required')
        if client is None:
            try:
                from mlflow.tracking import MlflowClient
            except ImportError as exc:
                raise RuntimeError('MLflow is optional. Install it with: python -m pip install "mlflow>=3.17.2,<4"') from exc
            client = MlflowClient(tracking_uri=tracking_uri)
        self.client = client
        self.tracking_uri = tracking_uri
        exp = client.get_experiment_by_name(experiment)
        self.experiment_id = exp.experiment_id if exp else client.create_experiment(experiment)
        self.parent_id: str | None = None
        self.trial_runs: dict[int, str] = {}
        self.finished_trials: set[int] = set()
        self.parent_finished = False

    def start_controller(self, *, config_sha256: str, model: str, provider: str,
                         train_sha256: str, validation_sha256: str,
                         max_attempts: int, fault_injected: bool) -> str:
        run = self.client.create_run(self.experiment_id, tags={
            'mlflow.runName': 'AutoTune CPU HF self-healing',
            'autotune.role': 'controller', 'autotune.fault_injection': str(fault_injected).lower(),
            'autotune.promoted': 'false',
        })
        self.parent_id = run.info.run_id
        for key, value in {
            'source_config_sha256': config_sha256, 'model': model,
            'advisor_provider': provider, 'train_sha256': train_sha256,
            'validation_sha256': validation_sha256, 'max_attempts': max_attempts,
        }.items():
            self.client.log_param(self.parent_id, key, str(value))
        return self.parent_id

    def start_trial(self, index: int, *, config_sha256: str, cfg: dict, config_path: Path) -> str:
        if self.parent_id is None or index in self.trial_runs:
            raise RuntimeError('MLflow parent run missing or trial already started')
        run = self.client.create_run(self.experiment_id, tags={
            'mlflow.runName': f'AutoTune trial-{index:02d}',
            'mlflow.parentRunId': self.parent_id, 'autotune.role': 'trial',
            'autotune.attempt': str(index),
        })
        run_id = run.info.run_id
        self.trial_runs[index] = run_id
        self.client.log_param(run_id, 'config_sha256', config_sha256)
        for key, value in cfg.get('train', {}).items():
            if isinstance(value, (str, int, float, bool)):
                self.client.log_param(run_id, f'train.{key}', str(value))
        self._artifact(run_id, config_path, 'configs')
        return run_id

    def _artifact(self, run_id: str, path: Path, subdir: str):
        if path.is_file() and not path.is_symlink():
            self.client.log_artifact(run_id, str(path), artifact_path=subdir)

    def finish_trial(self, index: int, *, out: Path, receipt_path: Path,
                     checkpoint: Path, score: float, diagnosis: str | None,
                     fault_injected: bool) -> None:
        run_id = self.trial_runs[index]
        for metrics_file in (out / 'training_logs.jsonl', out / 'metrics.jsonl'):
            if metrics_file.is_file():
                for line in metrics_file.read_text('utf8').splitlines():
                    row = json.loads(line)
                    step = row.get('step')
                    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                        continue
                    for k, v in row.items():
                        if k in {'step', 'timestamp_utc'} or isinstance(v, bool):
                            continue
                        if isinstance(v, (int, float)) and math.isfinite(float(v)):
                            self.client.log_metric(run_id, k, float(v), step=step)
        self.client.log_metric(run_id, 'best_val_loss', float(score))
        self.client.set_tag(run_id, 'autotune.diagnosis', diagnosis or 'none')
        self.client.set_tag(run_id, 'autotune.fault_injected', str(fault_injected).lower())
        for file, group in (
            (receipt_path, 'receipts'),
            (out / 'plan.json', 'provenance'),
            (out / 'monitor_decisions.jsonl', 'diagnostics'),
            (out / 'final_adapter' / 'autotune_manifest.json', 'manifests/final_adapter'),
            (checkpoint / 'autotune_manifest.json', 'manifests/checkpoint'),
        ):
            self._artifact(run_id, file, group)
        self.client.set_terminated(run_id, status='FINISHED')
        self.finished_trials.add(index)

    def fail_trial(self, index: int) -> None:
        if index in self.trial_runs and index not in self.finished_trials:
            self.client.set_terminated(self.trial_runs[index], status='FAILED')
            self.finished_trials.add(index)

    def finish_controller(self, *, result_path: Path, best_trial: int,
                          best_val_loss: float, event_log: Path) -> None:
        if self.parent_id is None:
            return
        self.client.log_metric(self.parent_id, 'best_val_loss', float(best_val_loss))
        self.client.set_tag(self.parent_id, 'autotune.best_trial', str(best_trial))
        self.client.set_tag(self.parent_id, 'autotune.promoted', 'false')
        self._artifact(self.parent_id, result_path, 'controller')
        self._artifact(self.parent_id, event_log, 'controller')
        self.client.set_terminated(self.parent_id, status='FINISHED')
        self.parent_finished = True

    def fail_controller(self) -> None:
        if self.parent_id is not None and not self.parent_finished:
            self.client.set_terminated(self.parent_id, status='FAILED')
            self.parent_finished = True
