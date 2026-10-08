<#
.SYNOPSIS
    Dimensionality-reduction ablation launcher (docs/plans/week5/DIMRED_ABLATION/PLAN.md).

.EXAMPLE
    .\script\week5\dimred_ablation\run_dimred.ps1 -RunId dr_v1 -DryRun
.EXAMPLE
    .\script\week5\dimred_ablation\run_dimred.ps1 -RunId dr_v1 -Hours 10
.EXAMPLE
    .\script\week5\dimred_ablation\run_dimred.ps1 -RunId dr_v1 -Hours 4 -Datasets babel2,babel3,hugadb
#>

param(
    [string]$RunId = "dr_v1",
    [switch]$DryRun,
    [switch]$GatesOnly,
    [string[]]$Stages,
    [string[]]$Datasets,
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

$PythonArgs = @("-m", "script.week5.dimred_ablation.run_dimred", "--run-id", $RunId)
if ($Stages) { $PythonArgs += @("--stages", ($Stages -join ",")) }
if ($Datasets) { $PythonArgs += @("--datasets", ($Datasets -join ",")) }
if ($Cpu) { $PythonArgs += "--cpu" }
if ($Smoke) { $PythonArgs += "--smoke" }
if ($DryRun) { $PythonArgs += "--dry-run" }
if ($GatesOnly) { $PythonArgs += "--gates-only" }
if ($Hours) { $PythonArgs += @("--hours", $Hours) }
if (-not $DryRun -and -not $Smoke -and -not $GatesOnly -and -not $Hours) { throw "A decoding run requires an explicit elapsed-time budget: pass -Hours." }

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PythonArgs
exit $LASTEXITCODE
