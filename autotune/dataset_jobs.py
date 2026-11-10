"""Persistent background jobs for long-running Knowledge Engine dataset builds."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autotune-dataset")
_LOCK = threading.RLock()
_ACTIVE: set[str] = set()


def _jobs_dir(datasets_root: Path) -> Path:
    path = datasets_root / ".generation_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(datasets_root: Path, job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid dataset generation job ID")
    base = _jobs_dir(datasets_root).resolve()
    target = (base / f"{job_id}.json").resolve()
    if target.parent != base:
        raise ValueError("Invalid dataset generation job path")
    return target


def _write(path: Path, state: dict[str, Any]) -> None:
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Dataset generation job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _update(path: Path, **values: Any) -> dict[str, Any]:
    with _LOCK:
        state = _read(path)
        state.update(values)
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(path, state)
        return state


def _next_dataset_path(datasets_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).replace(microsecond=0)
    for offset in range(120):
        candidate = datasets_root / f"dataset-{(stamp + timedelta(seconds=offset)).strftime('%Y%m%d-%H%M%S')}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a unique dataset output directory")


def _run_job(datasets_root: Path, path: Path, params: dict[str, Any]) -> None:
    job_id = path.stem
    with _LOCK:
        _ACTIVE.add(job_id)
    try:
        _update(path, status="running", stage="starting", progress={
            "stage": "starting", "processed_chunks": 0, "total_chunks": 0,
            "processed_records": 0, "accepted_pairs": 0, "rejected_pairs": 0,
        })

        def report(progress: dict[str, Any]) -> None:
            current = _read(path)
            _update(path, stage=progress.get("stage", current.get("stage", "running")), progress=progress)

        from .dataset_factory import create_dataset
        output_dir = _next_dataset_path(datasets_root)
        result = create_dataset(output_dir=output_dir, progress_callback=report, **params)
        _update(path, status="completed", stage="completed", result=result,
                error=None, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                progress={**(_read(path).get("progress") or {}), "stage": "completed", "accepted_pairs": sum(result.get("counts", {}).values())})
    except Exception as exc:
        _update(path, status="failed", stage="failed", error={
            "type": type(exc).__name__, "message": str(exc)[:1200]
        }, completed_at_utc=datetime.now(timezone.utc).isoformat())
    finally:
        with _LOCK:
            _ACTIVE.discard(job_id)


def create_job(datasets_root: Path, *, model: str, ollama_url: str = "http://127.0.0.1:11434",
               max_chunks: int = 100, min_pairs: int = 200, seed: int = 42) -> dict[str, Any]:
    datasets_root = datasets_root.resolve()
    datasets_root.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if _ACTIVE:
            raise RuntimeError("A dataset-generation job is already running. Wait for it to finish before starting another.")
        job_id = uuid.uuid4().hex
        path = _job_path(datasets_root, job_id)
        now = datetime.now(timezone.utc).isoformat()
        state = {
            "job_id": job_id, "kind": "dataset_generation", "status": "queued",
            "stage": "queued", "created_at_utc": now, "updated_at_utc": now,
            "completed_at_utc": None, "model": model, "max_chunks": max_chunks,
            "min_pairs": min_pairs, "progress": {
                "stage": "queued", "processed_chunks": 0, "total_chunks": 0,
                "processed_records": 0, "accepted_pairs": 0, "rejected_pairs": 0,
            }, "result": None, "error": None,
        }
        _write(path, state)
        _ACTIVE.add(job_id)  # Reserve immediately so a second click cannot queue another job.
        _EXECUTOR.submit(_run_job, datasets_root, path, {
            "model": model, "ollama_url": ollama_url, "max_chunks": max_chunks,
            "min_pairs": min_pairs, "seed": seed,
        })
        return state


def get_job(datasets_root: Path, job_id: str) -> dict[str, Any]:
    path = _job_path(datasets_root.resolve(), job_id)
    state = _read(path)
    if state.get("status") in {"queued", "running"}:
        with _LOCK:
            is_active = job_id in _ACTIVE
        if not is_active:
            state = _update(path, status="interrupted", stage="interrupted", error={
                "type": "InterruptedError", "message": "The API process restarted before this job completed. Start a new dataset job."
            })
    return state
