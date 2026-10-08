<#
.SYNOPSIS
    Contextualizer search calibration launcher
    (script/week5/contextualizer_calibration/README.md).

.DESCRIPTION
    Wraps script/week5/contextualizer_calibration/run_calibration.py in the
    smq conda environment. GPU training is allowed; OT stays on the CPU. Numerical
    library threads are capped at min(8, available CPUs). Exit codes are propagated.

.EXAMPLE
    .\script\week5\contextualizer_calibration\run_calibration.ps1 -Preflight
.EXAMPLE
    .\script\week5\contextualizer_calibration\run_calibration.ps1 -DryRun -RunId ctx_cal_dry
.EXAMPLE
    .\script\week5\contextualizer_calibration\run_calibration.ps1 -Smoke -RunId ctx_cal_smoke
.EXAMPLE
    .\script\week5\contextualizer_calibration\run_calibration.ps1 -Probe -RunId ctx_cal_probe
.EXAMPLE
    .\script\week5\contextualizer_calibration\run_calibration.ps1 -Hours 9 -RunId ctx_cal_v1
.EXAMPLE
    .\script\week5\contextualizer_calibration\run_calibration.ps1 -Hours 9 -RunId ctx_cal_v1 -Resume
#>

param(
    [switch]$Preflight,
    [switch]$DryRun,
    [switch]$Smoke,
    [switch]$Probe,
    [switch]$Resume,
    [switch]$IncludeCombinedTargetMatch,
    [switch]$ContinueOnReproMismatch,
    [switch]$Cpu,
    [string]$RunId,
    [string]$QicRunId = "qic_v3",
    [string]$Task2RunId = "task2_v1",
    [double]$Hours,
    [int]$ProbeMicroChunks,
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

$PythonArgs = @("-m", "script.week5.contextualizer_calibration.run_calibration")
if ($Preflight) {
    $PythonArgs += "--preflight"
}
else {
    if (-not $RunId) { throw "-RunId is required." }
    $PythonArgs += @("--run-id", $RunId, "--qic-run-id", $QicRunId, "--task2-run-id", $Task2RunId)
    if ($DryRun) { $PythonArgs += "--dry-run" }
    if ($Smoke) { $PythonArgs += "--smoke" }
    if ($Probe) { $PythonArgs += "--probe" }
    if ($ProbeMicroChunks) { $PythonArgs += @("--probe-micro-chunks", $ProbeMicroChunks) }
    if ($Resume) { $PythonArgs += "--resume" }
    if ($IncludeCombinedTargetMatch) { $PythonArgs += "--include-combined-target-match" }
    if ($ContinueOnReproMismatch) { $PythonArgs += "--continue-on-repro-mismatch" }
    if ($Hours) { $PythonArgs += @("--hours", $Hours) }
    if ($Cpu) { $PythonArgs += "--cpu" }
    if (-not ($DryRun -or $Smoke -or $Probe) -and -not $Hours) {
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
