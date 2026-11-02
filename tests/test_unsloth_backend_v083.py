"""Offline safety and backend-boundary tests; no Unsloth installation needed."""
from pathlib import Path
import sys

import pytest
import yaml

from autotune.unsloth_backend import (UnslothIntegrationError, assert_gpu_available,
                                       make_unsloth_plan, run_unsloth)

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / 'config/lora_public_cpu.yaml'
GPU = ROOT / 'config/lora_unsloth_gpu.example.yaml'


def test_cpu_preflight_reports_unsupported_without_importing_unsloth():
    before = set(sys.modules)
    report = make_unsloth_plan(CPU)
    assert report['ok'] and report['requires_gpu']
    assert not report['executable_on_configured_device']
    assert not report['live_gpu_training_verified']
    assert 'unsloth' not in set(sys.modules) - before


def test_gpu_config_preflight_is_explicitly_unverified():
    report = make_unsloth_plan(GPU)
    assert report['executable_on_configured_device']
    assert report['live_gpu_training_verified'] is False
    assert report['estimated_steps'] >= 2


def test_gpu_run_refuses_cpu_before_creating_files(tmp_path):
    out = tmp_path / 'should-not-exist'
    with pytest.raises(UnslothIntegrationError, match='GPU'):
        run_unsloth(CPU, out)
    assert not out.exists()


def test_gpu_runtime_preflight_rejects_no_device(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(UnslothIntegrationError, match='No supported'):
        assert_gpu_available({'device': 'gpu'})


def test_unknown_model_invalid_in_existing_contract(tmp_path):
    cfg = yaml.safe_load(GPU.read_text('utf8'))
    cfg['model']['name'] = '../../private'
    p = tmp_path / 'bad.yaml'
    p.write_text(yaml.safe_dump(cfg), encoding='utf8')
    with pytest.raises(UnslothIntegrationError, match="repository ID"):
        make_unsloth_plan(p)


def test_training_requires_explicit_execution_flag():
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'autotune.unsloth_cli', '--config', str(CPU)],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'UNSLOTH OFFLINE PREFLIGHT' in result.stdout
    assert 'not' not in result.stderr.lower()


def test_default_existing_hf_cli_unchanged():
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'autotune.hf_models_cli', '--config', str(CPU)],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'OFFLINE PREFLIGHT' in result.stdout


def test_no_automatic_agent_retry_enabled():
    cfg = yaml.safe_load(GPU.read_text('utf8'))
    assert cfg['meta_agent']['enabled'] is False
    assert cfg['safety']['max_trials'] == 1
