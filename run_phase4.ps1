# AutoTune AI Phase 4 - Windows PowerShell / CPU-only. No network/API token required.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Create the .venv first; see README.md' }
& $python -m autotune.doctor --mode mock
if ($LASTEXITCODE -ne 0) { throw 'Doctor failed' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $python -m autotune.advisor_cli --provider mock --diagnosis overfit
if ($LASTEXITCODE -ne 0) { throw 'Advisor mock failed' }
& $python -m autotune.recovery_cli --config config/agent_mock.yaml --workspace runs
if ($LASTEXITCODE -ne 0) { throw 'Durable advisor demo failed' }
Write-Host 'Phase 4 CPU-only verification complete. Inspect trial-01/advisor_decision.json and recovery_events.jsonl.'
