# AutoTune AI Phase 5 - genuine CPU-only PyTorch training, no API keys/network calls.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Create the .venv first; see README.md' }
& $python -c 'import torch; print("PyTorch:",torch.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Install torch; see requirements-real-cpu.txt' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Phase 5 tests failed' }
& $python -m autotune.real_cli --config config/real_cpu.yaml --workspace runs_real
if ($LASTEXITCODE -ne 0) { throw 'Real PyTorch recovery demo failed' }
Write-Host 'Phase 5 CPU demo complete. Inspect runs_real/<experiment>/trial-01/best_checkpoint.pt and recovery_state.json.'
