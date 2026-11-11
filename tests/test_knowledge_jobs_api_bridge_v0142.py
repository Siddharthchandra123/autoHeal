import json
import sys
from pathlib import Path

from autotune import knowledge_jobs as jobs


def test_cli_bridge_uses_current_python_and_repo_working_directory(tmp_path, monkeypatch):
    seen = {}

    class FakeStdout:
        def __iter__(self):
            return iter(['step 1\n', 'done\n'])

    class FakeProcess:
        stdout = FakeStdout()
        def __init__(self, command, **kwargs):
            seen['command'] = command
            seen['kwargs'] = kwargs
        def wait(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, 'Popen', FakeProcess)
    lines = []
    code, output = jobs._run_cli(
        dataset_dir=tmp_path / 'dataset', output_dir=tmp_path / 'out',
        mode='execute', allow_cpu=True, max_candidates=1,
        allow_small_data=False, log_path=tmp_path / 'job.log', on_line=lines.append,
    )
    assert code == 0
    assert output == 'step 1\ndone\n'
    assert seen['command'][0] == sys.executable
    assert seen['command'][1:3] == ['-m', 'autotune.knowledge_orchestrator']
    assert '--execute' in seen['command'] and '--allow-cpu' in seen['command']
    assert seen['command'][-2:] == ['--max-candidates', '1']
    assert Path(seen['kwargs']['cwd']).resolve() == Path(jobs.__file__).resolve().parent.parent
    assert lines == ['step 1\n', 'done\n']


def test_job_path_rejects_invalid_ids(tmp_path):
    try:
        jobs.get_job(tmp_path, '../bad')
    except ValueError as exc:
        assert 'Invalid job ID' in str(exc)
    else:
        raise AssertionError('invalid job ID accepted')
