# Run from the extracted project folder. Requires Python 3.11, installed requirements.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Create .venv and install requirements first (see README.md).'
}
& $python -m autotune.doctor --mode mock
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; mock run not started' }
& $python -m autotune.orchestrator --mode mock --no-writeback
if ($LASTEXITCODE -ne 0) { throw 'Mock run failed' }
$latest = Get-ChildItem -Path '.\runs' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $latest) { throw 'No experiment folder was produced' }
$metrics = Join-Path $latest.FullName 'trial-01\metrics.jsonl'
$report = Join-Path $latest.FullName 'monitor_replay.json'
& $python -m autotune.monitor_cli --file $metrics --config 'config/baseline.yaml' --output $report
if ($LASTEXITCODE -ne 0) { throw 'Offline monitor replay failed' }
Write-Host "Phase 2 complete. Review: $report"
