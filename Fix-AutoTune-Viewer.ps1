$ErrorActionPreference = "Stop"
Write-Host "=== AutoTune Viewer API Docker fix (v2) ===" -ForegroundColor Cyan

$source = ".\autotune\viewer_api.py"
if (-not (Test-Path $source)) {
    throw "Run this script from the AutoTune project root; autotune\viewer_api.py was not found."
}
if (-not ((Test-Path ".\docker-compose.yml") -or (Test-Path ".\compose.yaml") -or (Test-Path ".\compose.yml"))) {
    throw "No Docker Compose file found in the current directory."
}

# Read as text and locate the exact top-level function. This handles annotated
# signatures such as: def check_local_uri(uri: str) -> str:
$content = Get-Content -Path $source -Raw
$lines = [System.Collections.Generic.List[string]]::new()
foreach ($line in ($content -split "\r?\n")) { $lines.Add($line) }

$start = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^def\s+check_local_uri\s*\(') {
        $start = $i
        break
    }
}
if ($start -lt 0) {
    throw "Could not find a top-level def check_local_uri(...) in $source. No file was changed."
}

$end = $lines.Count
for ($i = $start + 1; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^(def\s+\w+\s*\(|class\s+\w+\s*[:(]|@)') {
        $end = $i
        break
    }
}

$replacement = @(
'def check_local_uri(uri: str) -> str:'
'    """Validate a local or trusted Docker-internal MLflow tracking URI."""'
'    from urllib.parse import urlparse'
''
'    parsed = urlparse(uri)'
'    if parsed.scheme != "http":'
'        raise ValueError("Only HTTP tracking URIs are supported")'
'    if parsed.username or parsed.password:'
'        raise ValueError("Credentials are not allowed in tracking URI")'
'    if parsed.hostname not in {"127.0.0.1", "localhost", "mlflow"}:'
'        raise ValueError(f"Tracking URI host is not allowed: {parsed.hostname}")'
'    if parsed.port is None or not (1 <= parsed.port <= 65535):'
'        raise ValueError("Tracking URI must specify a valid port")'
'    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:'
'        raise ValueError("Tracking URI must not contain a path, query, or fragment")'
'    return uri.rstrip("/")'
'' 
)

$updated = [System.Collections.Generic.List[string]]::new()
for ($i = 0; $i -lt $start; $i++) { $updated.Add($lines[$i]) }
foreach ($line in $replacement) { $updated.Add($line) }
for ($i = $end; $i -lt $lines.Count; $i++) { $updated.Add($lines[$i]) }

Set-Content -Path $source -Value ($updated -join "`r`n") -Encoding UTF8

$verify = Get-Content -Path $source -Raw
if ($verify -notmatch 'parsed\.hostname not in \{"127\.0\.0\.1", "localhost", "mlflow"\}') {
    throw "Source verification failed: Docker hostname allowlist not found."
}
if ($verify -match 'Viewer only accepts http://127\.0\.0\.1') {
    throw "Source verification failed: old validation message remains."
}
Write-Host "[OK] Updated check_local_uri in $source" -ForegroundColor Green

Write-Host "`n=== Rebuilding only viewer-api ===" -ForegroundColor Cyan
docker compose build --no-cache viewer-api
if ($LASTEXITCODE -ne 0) { throw "Docker build failed. Review the build output." }

Write-Host "`n=== Verifying code in rebuilt image ===" -ForegroundColor Cyan
$inspect = 'from pathlib import Path; s=Path("/app/autotune/viewer_api.py").read_text(); expected="parsed.hostname not in {`"127.0.0.1`", `"localhost`", `"mlflow`"}"; print("Patched allowlist present:", expected in s); print("Old error present:", "Viewer only accepts http://127.0.0.1" in s); assert expected in s and "Viewer only accepts http://127.0.0.1" not in s'
docker compose run --rm --no-deps --entrypoint python viewer-api -c $inspect
if ($LASTEXITCODE -ne 0) { throw "The rebuilt image did not pass source verification. Check Dockerfile.api and Compose build context." }

Write-Host "`n=== Recreating only viewer-api ===" -ForegroundColor Cyan
docker compose up -d --force-recreate viewer-api
if ($LASTEXITCODE -ne 0) { throw "Could not recreate viewer-api." }

Start-Sleep -Seconds 4
docker compose ps viewer-api
docker compose logs --tail 40 viewer-api

Write-Host "`nTest:" -ForegroundColor Yellow
Write-Host "Invoke-RestMethod http://127.0.0.1:8000/api/health"
Write-Host "Invoke-RestMethod http://127.0.0.1:8000/api/experiments"
