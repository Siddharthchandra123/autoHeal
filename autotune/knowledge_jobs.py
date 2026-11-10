"""Persistent local job wrapper for long-running Knowledge Engine orchestration."""
from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='autotune-knowledge')
_ACTIVE = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(root: Path, job_id: str) -> Path:
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Invalid job ID')
    folder = (root / 'jobs').resolve()
    return folder / f'{job_id}.json'


def get_job(root: Path, job_id: str) -> dict:
    path = _path(root, job_id)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError('Job not found')
    return json.loads(path.read_text(encoding='utf-8'))


def _write_job(path: Path, body: dict) -> None:
    """Write job state atomically so status reads never observe partial JSON."""
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def create_job(root: Path, *, dataset_dir: Path, mode: str, allow_cpu: bool,
               max_candidates: int | None = None, allow_small_data: bool = False) -> dict:
    global _ACTIVE
    if mode not in {'preflight','execute'}:
        raise ValueError('mode must be preflight or execute')
    if mode == 'execute' and not allow_cpu:
        raise ValueError('CPU execution requires explicit allow_cpu=true')
    with _LOCK:
        if _ACTIVE:
            raise RuntimeError('A Knowledge Engine training job is already running; wait for it to finish')
        _ACTIVE = True

    job_id = uuid.uuid4().hex
    try:
        folder = root / 'jobs'
        folder.mkdir(parents=True, exist_ok=True)
        path = _path(root, job_id)
        body = {'job_id':job_id,'mode':mode,'status':'queued','dataset_dir':str(dataset_dir),
                'created_at':_now(),'updated_at':_now(),'result':None,'error':None}
        _write_job(path, body)
    except Exception:
        with _LOCK:
            _ACTIVE = False
        raise

    def worker():
        global _ACTIVE
        try:
            body['status'] = 'running'; body['updated_at'] = _now()
            _write_job(path, body)
            from .knowledge_orchestrator import execute, preflight
            result = (execute(dataset_dir, output_dir=root / 'training_runs' / job_id, allow_cpu=allow_cpu,
                              max_candidates=max_candidates, allow_small_data=allow_small_data) if mode == 'execute' else
                      preflight(dataset_dir, output_dir=root / 'training_runs' / job_id))
            body['status'] = 'completed'; body['result'] = result
        except Exception as exc:
            body['status'] = 'failed'; body['error'] = {'type':type(exc).__name__,'message':str(exc)[:1000]}
        finally:
            body['updated_at'] = _now()
            try:
                _write_job(path, body)
            finally:
                with _LOCK:
                    _ACTIVE = False

    try:
        _POOL.submit(worker)
    except Exception:
        with _LOCK:
            _ACTIVE = False
        raise
    return body
