"""Local, read-only MLflow experiment viewer for AutoTune AI Phase 6B.

This module cannot submit training, deploy models, download artifacts or make
arbitrary URL requests. Backend MLflow target is fixed at application startup.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

RUN_ID = re.compile(r"^[a-fA-F0-9]{32}$")
EXPERIMENT_ID = re.compile(r"^\d{1,12}$")
METRIC_NAMES = frozenset({
    'loss', 'eval_loss', 'train_loss', 'val_loss', 'best_val_loss',
    'grad_norm', 'learning_rate', 'eval_accuracy', 'accuracy', 'f1',
})
PARAM_NAMES = frozenset({
    'source_config_sha256', 'config_sha256', 'model', 'advisor_provider',
    'train_sha256', 'validation_sha256', 'max_attempts',
    'train.learning_rate', 'train.lora_dropout', 'train.weight_decay',
    'train.num_train_epochs', 'train.per_device_train_batch_size',
})
TAG_NAMES = frozenset({
    'mlflow.runName', 'mlflow.parentRunId', 'autotune.role',
    'autotune.attempt', 'autotune.diagnosis', 'autotune.best_trial',
    'autotune.fault_injection', 'autotune.fault_injected', 'autotune.promoted',
})
MAX_METRIC_POINTS = 250


from urllib.parse import urlparse

from urllib.parse import urlparse

def check_local_uri(uri: str) -> str:
    """Validate a local or trusted Docker-internal MLflow tracking URI."""
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    if parsed.scheme != "http":
        raise ValueError("Only HTTP tracking URIs are supported")
    if parsed.username or parsed.password:
        raise ValueError("Credentials are not allowed in tracking URI")
    if parsed.hostname not in {"127.0.0.1", "localhost", "mlflow"}:
        raise ValueError(f"Tracking URI host is not allowed: {parsed.hostname}")
    if parsed.port is None or not (1 <= parsed.port <= 65535):
        raise ValueError("Tracking URI must specify a valid port")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("Tracking URI must not contain a path, query, or fragment")
    return uri.rstrip("/")

def _clean_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _subset(values: Any, allowed: frozenset[str]) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    return {str(k): str(v)[:200] for k, v in values.items() if k in allowed and v is not None}


def _read_attr(value: Any, field: str, default=None):
    return getattr(value, field, default)


def _run_data(run: Any) -> dict:
    info, data = run.info, run.data
    metrics = {k: _clean_float(v) for k, v in data.metrics.items()
               if k in METRIC_NAMES and _clean_float(v) is not None}
    return {
        'run_id': str(info.run_id),
        'experiment_id': str(info.experiment_id),
        'status': str(info.status),
        'start_time': _read_attr(info, 'start_time'),
        'end_time': _read_attr(info, 'end_time'),
        'metrics': metrics,
        'params': _subset(data.params, PARAM_NAMES),
        'tags': _subset(data.tags, TAG_NAMES),
    }


class ViewerStore:
    def __init__(self, tracking_uri: str, *, client: Any = None):
        self.tracking_uri = check_local_uri(tracking_uri)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def experiments(self) -> list[dict]:
        results = self.client.search_experiments(max_results=100)
        return [{'experiment_id': str(e.experiment_id), 'name': str(e.name)[:200],
                 'lifecycle_stage': str(getattr(e, 'lifecycle_stage', 'active'))}
                for e in results if EXPERIMENT_ID.fullmatch(str(e.experiment_id))]

    def runs(self, experiment_id: str, limit: int = 50) -> list[dict]:
        if not EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError('Invalid experiment ID')
        limit = max(1, min(int(limit), 100))
        return [_run_data(r) for r in self.client.search_runs(
            experiment_ids=[experiment_id], max_results=limit,
            order_by=['attributes.start_time DESC'])]

    def run(self, run_id: str) -> dict:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError('Invalid run ID')
        return _run_data(self.client.get_run(run_id))

    def history(self, run_id: str, metric: str) -> dict:
        if not RUN_ID.fullmatch(run_id) or metric not in METRIC_NAMES:
            raise ValueError('Invalid run ID or metric')
        points = []
        for p in self.client.get_metric_history(run_id, metric):
            value = _clean_float(getattr(p, 'value', None))
            step = getattr(p, 'step', None)
            if value is not None and type(step) is int and step >= 0:
                points.append({'step': step, 'value': value})
        points.sort(key=lambda p: p['step'])
        # Preserve both ends and a representative subset for large histories.
        if len(points) > MAX_METRIC_POINTS:
            last = len(points) - 1
            points = [points[round(i * last / (MAX_METRIC_POINTS - 1))]
                      for i in range(MAX_METRIC_POINTS)]
        return {'run_id': run_id, 'metric': metric, 'points': points}


def create_app(*, client: Any = None, tracking_uri: str | None = None,
               enable_docs: bool | None = None) -> FastAPI:
    uri = tracking_uri or os.environ.get('AUTOTUNE_MLFLOW_URI', 'http://127.0.0.1:5000')
    store = ViewerStore(uri, client=client)
    # API reference is opt-in for local development; keep it disabled by default.
    if enable_docs is None:
        enable_docs = os.environ.get('AUTOTUNE_ENABLE_DOCS', '').strip() == '1'
    app = FastAPI(
        title='AutoTune AI Local Viewer', version='0.9.1',
        docs_url='/docs' if enable_docs else None,
        redoc_url=None,
        openapi_url='/openapi.json' if enable_docs else None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
    app.add_middleware(CORSMiddleware, allow_origins=[
        'http://127.0.0.1:3000', 'http://localhost:3000'],
        allow_methods=['GET'], allow_headers=['Content-Type'], allow_credentials=False)

    def guarded(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            # Do not leak MLflow internal paths, tokens or request details.
            raise HTTPException(status_code=503, detail='Local MLflow query unavailable') from exc

    @app.get('/api/health')
    def health():
        return {'status': 'ready', 'mode': 'read-only', 'mlflow': 'not probed'}

    @app.get('/api/experiments')
    def experiments():
        return {'experiments': guarded(store.experiments)}

    @app.get('/api/experiments/{experiment_id}/runs')
    def runs(experiment_id: str, limit: int = Query(50, ge=1, le=100)):
        return {'runs': guarded(store.runs, experiment_id, limit)}

    @app.get('/api/runs/{run_id}')
    def run(run_id: str):
        return guarded(store.run, run_id)

    @app.get('/api/runs/{run_id}/metrics/{metric}')
    def metrics(run_id: str, metric: str):
        return guarded(store.history, run_id, metric)

    return app


app = create_app()

