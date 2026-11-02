# Windows PowerShell; run from extracted Phase 3 project folder.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Create the Python virtual environment first; see README.md' }
& $python -m autotune.doctor --mode mock
if ($LASTEXITCODE -ne 0) { throw 'Doctor failed' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; not running recovery simulation' }
& $python -m autotune.recovery_cli --config config/baseline.yaml --workspace runs
if ($LASTEXITCODE -ne 0) { throw 'Durable recovery demo failed' }
Write-Host 'Phase 3 verification complete. Review the experiment recovery_state.json, recovery_events.jsonl and active.yaml.'
