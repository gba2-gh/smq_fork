<#
.SYNOPSIS
    Stage B launcher (docs/plans/THREE_STAGE_MOTION_PLAN.md v1.4, §4).

.DESCRIPTION
    Trains B-code: one masked-code Transformer per (dataset, normalize,
    seed) pooled fitting population, reusing Stage A's cached K=500
    tokenizer. Unlike Stage A's ASOT solver, this is ordinary deep-learning
    training, so it defaults to CUDA when available (a declared choice, not
    Stage A's CPU-default convention) -- pass -Device cpu to force CPU.

.EXAMPLE
    .\script\action_transport\run_stage_b.ps1 -Preflight
.EXAMPLE
    .\script\action_transport\run_stage_b.ps1 -ValidateOnly
.EXAMPLE
    .\script\action_transport\run_stage_b.ps1 -DryRun
.EXAMPLE
    .\script\action_transport\run_stage_b.ps1 -Hours 4 -RunId stage_b_001
.EXAMPLE
    .\script\action_transport\run_stage_b.ps1 -Hours 2 -RunId stage_b_001 -Resume
.EXAMPLE
    # stage_b_003: freeze the optstudy-selected (ALiBi) configuration on stage_b_002's official split
    .\script\action_transport\run_stage_b.ps1 -TrainConfig selected -SplitRun stage_b_002 -Hours 1 -RunId stage_b_003
.EXAMPLE
    .\script\action_transport\run_stage_b.ps1 -Diagnostics -RunId stage_b_003
#>

param(
    [switch]$ValidateOnly,
    [switch]$Preflight,
    [switch]$DryRun,
    [switch]$Diagnostics,
    [double]$Hours,
    [string]$RunId,
    [switch]$Resume,
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "auto",
    [ValidateSet("v1.5", "selected")]
    [string]$TrainConfig = "v1.5",
    [string]$StudyId = "optstudy_002",
    [string]$SplitRun,
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

$PythonArgs = @("-m", "script.action_transport.run_stage_b", "--device", $Device)
if ($ValidateOnly) { $PythonArgs += "--validate-only" }
elseif ($Preflight) {
    $PythonArgs += @("--preflight", "--train-config", $TrainConfig, "--study-id", $StudyId)
    if ($SplitRun) { $PythonArgs += @("--split-run", $SplitRun) }
}
elseif ($DryRun) { $PythonArgs += @("--dry-run", "--train-config", $TrainConfig, "--study-id", $StudyId) }
elseif ($Diagnostics) {
    if (-not $RunId) { throw "-Diagnostics requires -RunId." }
    $PythonArgs += @("--diagnostics", "--run-id", $RunId)
}
else {
    if (-not $Hours -or -not $RunId) { throw "A full run requires both -Hours and -RunId." }
    $PythonArgs += @("--hours", $Hours, "--run-id", $RunId, "--train-config", $TrainConfig, "--study-id", $StudyId)
    if ($Resume) { $PythonArgs += "--resume" }
    if ($SplitRun) { $PythonArgs += @("--split-run", $SplitRun) }
    Write-Host ("Launching a FULL Stage B run: hours=$Hours run_id=$RunId device=$Device resume=$Resume " +
               "train_config=$TrainConfig split_run=$SplitRun")
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
