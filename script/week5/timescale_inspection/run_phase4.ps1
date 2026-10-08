<#
.SYNOPSIS
    Time-scale inspection, phase 4 (input representation) launcher
    (script/week5/timescale_inspection/README.md).

.EXAMPLE
    .\script\week5\timescale_inspection\run_phase4.ps1 -RunId ts4_v1 -DryRun
.EXAMPLE
    .\script\week5\timescale_inspection\run_phase4.ps1 -RunId ts4_v1 -Hours 6
#>

param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [switch]$DryRun,
    [string]$Radius = "current",
    [string[]]$Arms,
    [string]$ContextRun = "rkx_v2",
    [string]$DirectRun = "ts1_v1",
    [switch]$Cpu,
    [switch]$NoProbes,
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

$PythonArgs = @("-m", "script.week5.timescale_inspection.run_phase4", "--run-id", $RunId, "--radius", $Radius,
                "--context-run", $ContextRun, "--direct-run", $DirectRun)
if ($Arms) { $PythonArgs += "--arms"; $PythonArgs += $Arms }
if ($Cpu) { $PythonArgs += "--cpu" }
if ($NoProbes) { $PythonArgs += "--no-probes" }
if ($DryRun) { $PythonArgs += "--dry-run" }
if ($Hours) { $PythonArgs += @("--hours", $Hours) }
if (-not $DryRun -and -not $Hours) { throw "A decoding run requires an explicit elapsed-time budget: pass -Hours." }

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PythonArgs
exit $LASTEXITCODE
