$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing .venv Python. Create a virtual environment first." }
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; refusing next step" }
& $Python -m autotune.lora_cli --config config/lora_meta_gpu.yaml
if ($LASTEXITCODE -ne 0) { throw "LoRA plan validation failed" }
Write-Host "Phase 5B dry-run validation complete. No model was downloaded or trained."
