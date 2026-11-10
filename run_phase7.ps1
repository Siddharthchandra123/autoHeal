param(
    [switch]$InstallTrainingDependencies,
    [switch]$InstallSemanticSearch,
    [switch]$StartDashboard,
    [switch]$SkipInstall,
    [ValidateRange(1,4096)][long]$MaxUploadMB = 512,
    [ValidateRange(1,1000000000)][long]$MaxCharacters = 100000000
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Respect explicit process environment overrides; otherwise apply v0.13 defaults/arguments.
if (-not $env:AUTOTUNE_MAX_FILE_BYTES) { $env:AUTOTUNE_MAX_FILE_BYTES = [string]($MaxUploadMB * 1024 * 1024) }
if (-not $env:AUTOTUNE_MAX_CHARS) { $env:AUTOTUNE_MAX_CHARS = [string]$MaxCharacters }

$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Python Launcher (py) was not found. Install Python 3.11-3.13 and retry.'
    }
    py -3.13 -m venv (Join-Path $ProjectRoot '.venv-phase7')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv-phase7 with Python 3.13.' }
}

if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & $Python -m pip install -r (Join-Path $ProjectRoot 'requirements-knowledge.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Knowledge Engine dependency installation failed.' }

    if ($InstallSemanticSearch) {
        & $Python -m pip install -r (Join-Path $ProjectRoot 'requirements-knowledge-ml.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Semantic-search dependency installation failed.' }
    }
    if ($InstallTrainingDependencies) {
        Write-Warning 'Hugging Face training dependencies can be large. Check available RAM and disk before proceeding.'
        & $Python -m pip install -r (Join-Path $ProjectRoot 'requirements-knowledge-training.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
    }
}

if ($StartDashboard) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw 'Node.js and npm are required for the optional Next.js dashboard.'
    }
    $WebRoot = Join-Path $ProjectRoot 'web'
    if (-not (Test-Path (Join-Path $WebRoot 'node_modules'))) {
        Push-Location $WebRoot
        try {
            npm install
            if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
        } finally { Pop-Location }
    }
    Start-Process -FilePath 'npm.cmd' -ArgumentList @('run','dev') -WorkingDirectory $WebRoot
    Write-Host 'Dashboard starting at http://127.0.0.1:3000'
}

Write-Host ('Knowledge Engine limits: {0} MB per file; {1:N0} extracted characters per document' -f ([long]$env:AUTOTUNE_MAX_FILE_BYTES / 1MB), [long]$env:AUTOTUNE_MAX_CHARS)
Write-Host 'AutoTune AI Knowledge Engine API: http://127.0.0.1:8010/docs'
Write-Host 'Press Ctrl+C to stop the API. Keep this development service bound to loopback.'
& $Python -m uvicorn autotune.knowledge_engine:app --host 127.0.0.1 --port 8010
