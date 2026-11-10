import json
from pathlib import Path

import pytest

from autotune import knowledge_jobs as jobs


def test_atomic_job_state_writer_writes_complete_json(tmp_path):
    path = tmp_path / 'job.json'
    payload = {'job_id': 'a' * 32, 'status': 'running', 'result': None}
    jobs._write_job(path, payload)
    assert json.loads(path.read_text(encoding='utf-8')) == payload
    assert not path.with_name(path.name + '.tmp').exists()


def test_create_job_releases_worker_lock_if_submit_fails(tmp_path, monkeypatch):
    class BrokenPool:
        def submit(self, _worker):
            raise RuntimeError('pool stopped')

    monkeypatch.setattr(jobs, '_POOL', BrokenPool())
    monkeypatch.setattr(jobs, '_ACTIVE', False)
    with pytest.raises(RuntimeError, match='pool stopped'):
        jobs.create_job(tmp_path, dataset_dir=tmp_path / 'dataset', mode='preflight', allow_cpu=False)
    assert jobs._ACTIVE is False


def test_create_job_releases_worker_lock_if_initial_persistence_fails(tmp_path, monkeypatch):
    def fail_write(*_args, **_kwargs):
        raise OSError('disk unavailable')

    monkeypatch.setattr(jobs, '_write_job', fail_write)
    monkeypatch.setattr(jobs, '_ACTIVE', False)
    with pytest.raises(OSError, match='disk unavailable'):
        jobs.create_job(tmp_path, dataset_dir=tmp_path / 'dataset', mode='preflight', allow_cpu=False)
    assert jobs._ACTIVE is False
