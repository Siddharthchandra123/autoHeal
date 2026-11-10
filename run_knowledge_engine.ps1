param(
    [switch]$InstallTrainingDependencies,
    [switch]$InstallSemanticSearch,
    [switch]$StartDashboard,
    [switch]$SkipInstall
)

# Backward-compatible entry point; full v0.12.0 startup is implemented in run_phase7.ps1.
& (Join-Path $PSScriptRoot 'run_phase7.ps1') @PSBoundParameters
