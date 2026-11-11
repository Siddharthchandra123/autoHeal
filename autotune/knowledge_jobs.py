"""Persistent local job wrapper for Knowledge Engine orchestration.

Runs the exact CLI module with the API process' Python interpreter. This keeps
CLI and FastAPI execution on the same environment and exposes captured logs to
the dashboard. Local-development service: bind the API to loopback only.
"""

from __future__ import annotations
import time
import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
_WRITE_LOCK = threading.RLock()
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autotune-knowledge")
_ACTIVE = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(root: Path, job_id: str) -> Path:
    if (
        not isinstance(job_id, str)
        or len(job_id) != 32
        or any(c not in "0123456789abcdef" for c in job_id)
    ):
        raise ValueError("Invalid job ID")
    folder = (root / "jobs").resolve()
    return folder / f"{job_id}.json"


def get_job(root: Path, job_id: str) -> dict:
    path = _path(root, job_id)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("Job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(path: Path, body: dict) -> None:
    """Write job state atomically with Windows-safe retries."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(body, indent=2, ensure_ascii=False)

    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")

    with _WRITE_LOCK:
        try:
            with temp.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            last_error = None

            for attempt in range(7):
                try:
                    os.replace(temp, path)
                    return
                except PermissionError as exc:
                    last_error = exc

                    if attempt == 6:
                        raise

                    time.sleep(0.05 * (2**attempt))

            if last_error:
                raise last_error

        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\r\n") + "\n")


def _run_cli(
    *,
    dataset_dir: Path,
    output_dir: Path,
    mode: str,
    allow_cpu: bool,
    max_candidates: int | None,
    allow_small_data: bool,
    log_path: Path,
    on_line=None,
) -> tuple[int, str]:
    """Invoke the same module/entry point used by `python -m` in PowerShell."""
    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        "-m",
        "autotune.knowledge_orchestrator",
        "--dataset",
        str(dataset_dir),
        "--output",
        str(output_dir),
    ]
    if mode == "execute":
        command.append("--execute")
    else:
        command.append("--preflight")
    if allow_cpu:
        command.append("--allow-cpu")
    if allow_small_data:
        command.append("--allow-small-data")
    if max_candidates is not None:
        command.extend(["--max-candidates", str(max_candidates)])

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    _append_log(log_path, "$ " + subprocess.list2cmdline(command))
    proc = subprocess.Popen(
        command,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
        _append_log(log_path, line)
        if on_line is not None:
            on_line(line)
    return proc.wait(), "".join(output_lines)


def create_job(
    root: Path,
    *,
    dataset_dir: Path,
    mode: str,
    allow_cpu: bool,
    max_candidates: int | None = None,
    allow_small_data: bool = False,
) -> dict:
    global _ACTIVE
    if mode not in {"preflight", "execute"}:
        raise ValueError("mode must be preflight or execute")
    if mode == "execute" and not allow_cpu:
        raise ValueError("CPU execution requires explicit allow_cpu=true")
    if max_candidates is not None and not 1 <= max_candidates <= 3:
        raise ValueError("max_candidates must be between 1 and 3")
    with _LOCK:
        if _ACTIVE:
            raise RuntimeError(
                "A Knowledge Engine training job is already running; wait for it to finish"
            )
        _ACTIVE = True

    root = Path(root).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    job_id = uuid.uuid4().hex
    try:
        folder = root / "jobs"
        folder.mkdir(parents=True, exist_ok=True)
        output_dir = root / "training_runs" / job_id
        path = _path(root, job_id)
        log_path = folder / f"{job_id}.log"
        body = {
            "job_id": job_id,
            "mode": mode,
            "status": "queued",
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            "created_at": _now(),
            "updated_at": _now(),
            "python_executable": sys.executable,
            "working_directory": str(Path(__file__).resolve().parent.parent),
            "max_candidates": max_candidates,
            "result": None,
            "error": None,
            "return_code": None,
            "log_tail": [],
        }
        _write_job(path, body)
    except Exception:
        with _LOCK:
            _ACTIVE = False
        raise

    def worker():
        global _ACTIVE
        try:
            body["status"] = "running"
            body["stage"] = "starting_cli"
            body["updated_at"] = _now()
            _write_job(path, body)
            live_lines: list[str] = []

            def publish_line(line: str) -> None:
                live_lines.append(line.rstrip("\r\n"))
                del live_lines[:-80]

                if len(live_lines) % 5 == 0:
                    body["log_tail"] = list(live_lines)
                    body["updated_at"] = _now()

                    try:
                        _write_job(path, body)
                    except OSError as exc:
                        # A dashboard state-write failure must not interrupt
                        # stdout consumption or the training subprocess.
                        warning = (
                            f"[JOB STATE WARNING] Could not persist progress: "
                            f"{type(exc).__name__}: {exc}"
                        )

                        try:
                            _append_log(log_path, warning)
                        except OSError:
                            pass

            code, output = _run_cli(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                mode=mode,
                allow_cpu=allow_cpu,
                max_candidates=max_candidates,
                allow_small_data=allow_small_data,
                log_path=log_path,
                on_line=publish_line,
            )
            body["return_code"] = code
            body["log_tail"] = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-80:]
            if code == 0:
                # CLI prints its final structured result as JSON. Keep it available
                # to the UI while retaining raw logs for diagnosis.
                try:
                    body["result"] = json.loads(output[output.find("{") :])
                except (ValueError, json.JSONDecodeError):
                    body["result"] = {"output_tail": output.splitlines()[-40:]}
                body["status"] = "completed"
                body["stage"] = "completed"
            else:
                body["status"] = "failed"
                body["stage"] = "failed"
                body["error"] = {
                    "type": "TrainingProcessError",
                    "message": "\n".join(body["log_tail"][-30:])[-4000:]
                    or f"CLI exited with code {code}",
                }
        except Exception as exc:
            body["status"] = "failed"
            body["stage"] = "failed"
            body["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        finally:
            body["updated_at"] = _now()
            try:
                body["log_tail"] = (
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                        -80:
                    ]
                    if log_path.exists()
                    else []
                )
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
