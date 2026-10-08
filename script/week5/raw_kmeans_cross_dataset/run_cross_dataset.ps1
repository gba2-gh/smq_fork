<#
.SYNOPSIS
    Raw K-means tokens across datasets launcher
    (script/week5/raw_kmeans_cross_dataset/README.md).

.EXAMPLE
    .\script\week5aw_kmeans_cross_datasetun_cross_dataset.ps1 -Preflight
.EXAMPLE
    .\script\week5aw_kmeans_cross_datasetun_cross_dataset.ps1 -DryRun -RunId rkx_dry
.EXAMPLE
    .\script\week5aw_kmeans_cross_datasetun_cross_dataset.ps1 -Smoke -RunId rkx_smoke
.EXAMPLE
    .\script\week5aw_kmeans_cross_datasetun_cross_dataset.ps1 -Hours 8 -RunId rkx_v1
.EXAMPLE
    .\script\week5aw_kmeans_cross_datasetun_cross_dataset.ps1 -Hours 8 -RunId rkx_v1 -Resume
#>

param(
    [switch]$Preflight,
    [switch]$DryRun,
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$ContinueOnGateFailure,
    [switch]$Cpu,
    [string]$RunId,
    [string[]]$Datasets,
    [string]$QicRunId = "qic_v3",
    [string]$ReferenceRun = "ctx_cal_v1",
    [double]$Hours,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

function Resolve-Python {
    if ($PythonExe) { return $PythonExe }
    if ($env:SMQ_PYTHON) { return $env:SMQ_PYTHON }
    $candidates = @(
        "$env:USERPROFILE\AppData\Local\anaconda3\envs\smq\python.exe",
        "$env:USERPROFILE\anaconda3\envs\smq\python.exe",
        "$env:USERPROFILE\miniconda3\envs\smq\python.exe",
        "$env:USERPROFILE\AppData\Local\miniconda3\envs\smq\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "Could not find the 'smq' conda environment. Pass -PythonExe or set `$env:SMQ_PYTHON."
}

$Python = Resolve-Python
Write-Host "Using Python: $Python"

$Cap = [Math]::Min(8, [Environment]::ProcessorCount)
$env:OPENBLAS_NUM_THREADS = "$Cap"
$env:MKL_NUM_THREADS = "$Cap"
$env:OMP_NUM_THREADS = "$Cap"

$PythonArgs = @("-m", "script.week5.raw_kmeans_cross_dataset.run_cross_dataset")
if ($Preflight) {
    $PythonArgs += "--preflight"
}
else {
    if (-not $RunId) { throw "-RunId is required." }
    $PythonArgs += @("--run-id", $RunId, "--qic-run-id", $QicRunId, "--reference-run", $ReferenceRun)
    if ($DryRun) { $PythonArgs += "--dry-run" }
    if ($Smoke) { $PythonArgs += "--smoke" }
    if ($Resume) { $PythonArgs += "--resume" }
    if ($ContinueOnGateFailure) { $PythonArgs += "--continue-on-gate-failure" }
    if ($Datasets) { $PythonArgs += "--datasets"; $PythonArgs += $Datasets }
    if ($Hours) { $PythonArgs += @("--hours", $Hours) }
    if ($Cpu) { $PythonArgs += "--cpu" }
    if (-not ($DryRun -or $Smoke) -and -not $Hours) {
        throw "A full run requires an explicit elapsed-time budget: pass -Hours."
    }
}

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
# Native stderr (library warnings) must not be promoted to a terminating error; the exit code decides.
$ErrorActionPreference = "Continue"
& $Python @PythonArgs
$ExitCode = $LASTEXITCODE
exit $ExitCode
