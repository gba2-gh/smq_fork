<#
.SYNOPSIS
    Tier 0 MASQ replication launcher (script/week5/masq_experiments/README.md). Stage "hu" = the HuGaDB-focused variants.

.EXAMPLE
    .\script\week5\masq_experiments\run.ps1 -RunId masq_smoke -Stage gates
.EXAMPLE
    .\script\week5\masq_experiments\run.ps1 -RunId masq_v1 -Stage 1 -Hours 6 *>&1 | Tee-Object masq_stage1.log
#>

param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][ValidateSet("gates", "smoke", "1", "2", "hu")][string]$Stage,
    [double]$Hours,
    [switch]$DryRun,
    [switch]$Cpu,
    [string]$TimingsFrom = "masq_smoke",
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
$env:PYTHONIOENCODING = "utf-8"

$PythonArgs = @("-m", "script.week5.masq_experiments.run", "--run-id", $RunId, "--stage", $Stage, "--timings-from", $TimingsFrom)
if ($Cpu) { $PythonArgs += "--cpu" }
if ($DryRun) { $PythonArgs += "--dry-run" }
if ($Hours) { $PythonArgs += @("--hours", $Hours) }
if (($Stage -eq "1" -or $Stage -eq "2" -or $Stage -eq "hu") -and -not $DryRun -and -not $Hours) { throw "Stages 1 and 2 require an explicit elapsed-time budget: pass -Hours." }

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PythonArgs
exit $LASTEXITCODE
