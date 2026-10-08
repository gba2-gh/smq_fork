<#
.SYNOPSIS
    Time-scale inspection, phase 1 (OT kernel radius) launcher
    (script/week5/timescale_inspection/README.md).

.EXAMPLE
    .\script\week5\timescale_inspection\run_phase1.ps1 -RunId ts1_v1 -Stage screen -DryRun
.EXAMPLE
    .\script\week5\timescale_inspection\run_phase1.ps1 -RunId ts1_v1 -Stage screen -Hours 4
.EXAMPLE
    .\script\week5\timescale_inspection\run_phase1.ps1 -RunId ts1_v1 -Stage confirm beta -Hours 7
#>

param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string[]]$Stage = @("screen"),
    [switch]$DryRun,
    [switch]$NoAlias,
    [string]$SourceRun = "rkx_v2",
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

$PythonArgs = @("-m", "script.week5.timescale_inspection.run_phase1", "--run-id", $RunId, "--source-run", $SourceRun, "--stage") + $Stage
if ($DryRun) { $PythonArgs += "--dry-run" }
if ($NoAlias) { $PythonArgs += "--no-alias" }
if ($Hours) { $PythonArgs += @("--hours", $Hours) }
if (-not $DryRun -and -not $Hours) { throw "A decoding run requires an explicit elapsed-time budget: pass -Hours." }

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PythonArgs
exit $LASTEXITCODE
