<#
.SYNOPSIS
    Overnight frozen-representation study launcher
    (docs/plans/FROZEN_STUDY_AGENT_INSTRUCTIONS.md).

.DESCRIPTION
    Discovery vs. oracle vs. pooling on frozen SMQ / stage_a_pooled_002 /
    stage_b_003 / stage_c_pooled_001 artifacts. Nothing is trained. This
    runner only saves outputs under results/frozen_study/<RunId>/: no
    report, no verdict, no metric printouts.

    Requires a completed stage_a_pooled_002, stage_b_003, and
    stage_c_pooled_001 on this machine (checked at startup by the runner
    itself).

.EXAMPLE
    .\script\action_transport\run_frozen_study.ps1 -ValidateOnly
.EXAMPLE
    .\script\action_transport\run_frozen_study.ps1 -Smoke -RunId _smoke
.EXAMPLE
    .\script\action_transport\run_frozen_study.ps1 -Hours 8 -RunId frozen_001
.EXAMPLE
    .\script\action_transport\run_frozen_study.ps1 -Hours 4 -RunId frozen_001 -Resume
#>

param(
    [switch]$ValidateOnly,
    [switch]$Smoke,
    [double]$Hours,
    [string]$RunId,
    [switch]$Resume,
    [int]$Workers = 6,
    [int]$ThreadsPerWorker = 5,
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

if (-not $Smoke -and -not $ValidateOnly) {
    if (-not $Hours -or $Hours -le 0.5) { throw "-Hours must be greater than 0.5." }
    if (-not $RunId) { throw "A full run requires -RunId." }
    if ($RunId.EndsWith("_smoke")) { throw "-RunId must not end in '_smoke' unless -Smoke is given." }
}
if ($Smoke -and $RunId -and -not $RunId.EndsWith("_smoke")) {
    throw "-Smoke requires a -RunId ending in '_smoke' (or omit -RunId to use '_smoke')."
}

$Python = Resolve-Python
Write-Host "Using Python: $Python"

$env:OPENBLAS_NUM_THREADS = "$ThreadsPerWorker"
$env:MKL_NUM_THREADS = "$ThreadsPerWorker"
$env:OMP_NUM_THREADS = "$ThreadsPerWorker"

$PythonArgs = @("-m", "script.action_transport.frozen_study")
if ($ValidateOnly) {
    $PythonArgs += "--validate-only"
}
elseif ($Smoke) {
    $PythonArgs += @("--smoke", "--run-id", $(if ($RunId) { $RunId } else { "_smoke" }), "--hours", "1",
        "--workers", [Math]::Min($Workers, 2), "--threads-per-worker", $ThreadsPerWorker)
    Write-Host "Launching a SMOKE run (3 recordings/dataset, rungs [10,20], outer_cap 2, h=2, 2 workers)."
}
else {
    $PythonArgs += @("--hours", $Hours, "--run-id", $RunId, "--workers", $Workers,
        "--threads-per-worker", $ThreadsPerWorker)
    if ($Resume) { $PythonArgs += "--resume" }
    Write-Host "Launching a FULL frozen-representation study: hours=$Hours run_id=$RunId workers=$Workers " `
        "threads_per_worker=$ThreadsPerWorker resume=$Resume"
    Write-Host "This runner only SAVES artifacts. It reports no metric values, no report, no verdict."
}

Push-Location $RepoRoot
try {
    $ErrorActionPreference = "Continue"  # see run_stage_a.ps1 for why (h5py stderr warning)
    & $Python @PythonArgs
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($ExitCode -ne 0) { throw "Python exited with code $ExitCode" }
}
finally {
    Pop-Location
}
