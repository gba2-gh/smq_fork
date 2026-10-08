<#
.SYNOPSIS
    Task 2 launcher: contextualizer training + OT decoding + scoring on top
    of Task 1's artifacts (script/quantized_input_comparison/README.md).

.DESCRIPTION
    Orchestrates script/quantized_input_comparison/run_task2.py. Defaults to
    CUDA for contextualizer training when available (ordinary deep-learning
    training, like Stage B); OT solves are always CPU, capped at 8 threads,
    per the historical action_transport convention (run_task2.py calls
    ot_arms.set_cpu_thread_cap() itself).

.EXAMPLE
    .\script\quantized_input_comparison\run_task2.ps1 -Preflight
.EXAMPLE
    .\script\quantized_input_comparison\run_task2.ps1 -Smoke -RunId t2_smoke_v1
.EXAMPLE
    .\script\quantized_input_comparison\run_task2.ps1 -Hours 8 -RunId task2_v1
.EXAMPLE
    .\script\quantized_input_comparison\run_task2.ps1 -Hours 8 -RunId task2_v1 -Resume
#>

param(
    [switch]$Preflight,
    [switch]$Smoke,
    [switch]$Resume,
    [string]$RunId,
    [string]$QicRunId = "qic_v3",
    [double]$Hours,
    [switch]$Cpu,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

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

$PythonArgs = @("-m", "script.quantized_input_comparison.run_task2")
if ($Preflight) {
    $PythonArgs += "--preflight"
}
else {
    if (-not $RunId) { throw "-RunId is required for -Smoke or a full run." }
    $PythonArgs += @("--run-id", $RunId, "--qic-run-id", $QicRunId)
    if ($Smoke) { $PythonArgs += "--smoke" }
    if ($Resume) { $PythonArgs += "--resume" }
    if ($Hours) { $PythonArgs += @("--hours", $Hours) }
    if ($Cpu) { $PythonArgs += "--cpu" }
}

Write-Host "Running: $Python $($PythonArgs -join ' ')"
Set-Location $RepoRoot
& $Python @PythonArgs
exit $LASTEXITCODE
