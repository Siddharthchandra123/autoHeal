$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m autotune.orchestrator --mode mock
Write-Host "Open runs/ for the latest summary.json and metrics.jsonl."
