<#
.SYNOPSIS
    FSQ reconstruction pilot launcher
    (docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md Revision 3).

.DESCRIPTION
    Trains a matched pair of autoencoders (VQ512 vs FSQ512[8,8,8]) on
    HuGaDB from one autoencoder seed, extracts native codes and
    K-means-512 codes from each, and scores both categorical and contextual
    ASOT on all four streams at three downstream seeds. Implements the
    REQUIRED experiment in the plan's §3; LARa, a second autoencoder seed,
    and the released-encoder reference are not auto-scheduled (see
    README.md "Scope of this implementation").

.EXAMPLE
    .\script\fsq_reconstruction\run.ps1 -ValidateOnly
.EXAMPLE
    .\script\fsq_reconstruction\run.ps1 -Smoke
.EXAMPLE
    .\script\fsq_reconstruction\run.ps1 -Hours 30 -RunId fsq_pilot_001
.EXAMPLE
    .\script\fsq_reconstruction\run.ps1 -Hours 30 -RunId fsq_pilot_001 -Resume
#>

param(
    [switch]$ValidateOnly,
    [switch]$Smoke,
    [double]$Hours,
    [string]$RunId,
    [string]$Dataset = "hugadb",
    [int[]]$DownstreamSeeds = @(111, 222, 1538574472),
    [switch]$Resume,
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

if (-not $Smoke -and -not $ValidateOnly) {
    if (-not $Hours -or $Hours -le 3.5) { throw "-Hours must leave at least 3 hours for verification/reporting (§11)." }
    if (-not $RunId) { throw "A full run requires -RunId." }
}

$Python = Resolve-Python
Write-Host "Using Python: $Python"

$Cap = [Math]::Min(8, [Environment]::ProcessorCount)
$env:OPENBLAS_NUM_THREADS = "$Cap"
$env:MKL_NUM_THREADS = "$Cap"
$env:OMP_NUM_THREADS = "$Cap"

$PythonArgs = @("-m", "script.fsq_reconstruction.experiment")
if ($ValidateOnly) {
    $PythonArgs += "--validate-only"
}
elseif ($Smoke) {
    $PythonArgs += @("--smoke", "--run-id", $(if ($RunId) { $RunId } else { "fsq_smoke" }))
    if ($Cpu) { $PythonArgs += "--cpu" }
    Write-Host "Launching a SMOKE run: tiny model, 2 epochs, 1 downstream seed, K=32 alphabet, short "
    "contextualizer/OT budgets. Runs against the real HuGaDB dataset end to end -- this checks that "
    "every stage runs, not that any stage has converged."
}
else {
    $PythonArgs += @("--dataset", $Dataset, "--run-id", $RunId, "--hours", $Hours,
        "--downstream-seeds") + $DownstreamSeeds
    if ($Resume) { $PythonArgs += "--resume" }
    if ($Cpu) { $PythonArgs += "--cpu" }
    Write-Host "Launching the REQUIRED FSQ pilot: dataset=$Dataset hours=$Hours run_id=$RunId " `
        "downstream_seeds=$DownstreamSeeds resume=$Resume cpu=$Cpu"
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
