<#
.SYNOPSIS
    Time-scale inspection, phase 7 (held-out test and label-free span criteria) launcher
    (script/week5/timescale_inspection/README.md).

.EXAMPLE
    .\script\week5\timescale_inspection\run_phase7.ps1 -RunId ts5_v1 -DryRun
.EXAMPLE
    .\script\week5\timescale_inspection\run_phase7.ps1 -RunId ts5_v1 -Hours 10
#>

param(
    [string]$RunId = "ts5_v1",
    [switch]$DryRun,
    [string]$ContextRun = "rkx_v2",
    [string]$DirectRun = "ts1_v1",
    [string]$Phase5Dir = "phase5",
    [switch]$Cpu,
    [switch]$Smoke,
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

$PythonArgs = @("-m", "script.week5.timescale_inspection.run_phase7", "--run-id", $RunId,
                "--context-run", $ContextRun, "--direct-run", $DirectRun, "--phase5-dir", $Phase5Dir)
if ($Cpu) { $PythonArgs += "--cpu" }
if ($Smoke) { $PythonArgs += "--smoke" }
if ($DryRun) { $PythonArgs += "--dry-run" }
if ($Hours) { $PythonArgs += @("--hours", $Hours) }
if (-not $DryRun -and -not $Smoke -and -not $Hours) { throw "A decoding run requires an explicit elapsed-time budget: pass -Hours." }

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PythonArgs
exit $LASTEXITCODE
