<#
.SYNOPSIS
    Stage C launcher (docs/plans/THREE_STAGE_MOTION_PLAN.md §5;
    docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md Part 2).

.DESCRIPTION
    Contextual temporal transport on the frozen stage_b_003 embeddings.
    Requires a completed stage_b_003 (the frozen selected/ALiBi
    configuration on the official split, all 12 cells) and a completed
    stage_a_pooled_002 (or later) for the comparison arms and paired
    grouping. Pooled protocol only. Defaults to CPU, matching Stage A's ASOT
    solver convention.

.EXAMPLE
    .\script\action_transport\run_stage_c.ps1 -ValidateOnly
.EXAMPLE
    .\script\action_transport\run_stage_c.ps1 -Preflight -StageBRun stage_b_003 -StageARun stage_a_pooled_002
.EXAMPLE
    .\script\action_transport\run_stage_c.ps1 -DryRun -StageBRun stage_b_003 -StageARun stage_a_pooled_002
.EXAMPLE
    .\script\action_transport\run_stage_c.ps1 -StageBRun stage_b_003 -StageARun stage_a_pooled_002 -Hours 5 -RunId stage_c_pooled_001
.EXAMPLE
    .\script\action_transport\run_stage_c.ps1 -StageBRun stage_b_003 -StageARun stage_a_pooled_002 -Hours 3 -RunId stage_c_pooled_001 -Resume
#>

param(
    [switch]$ValidateOnly,
    [switch]$Preflight,
    [switch]$DryRun,
    [switch]$Report,
    [string]$StageBRun,
    [string]$StageARun,
    [double]$Hours,
    [string]$RunId,
    [switch]$Resume,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
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

if ($Report) {
    if (-not $RunId) { throw "-Report requires -RunId." }
    $PythonArgs = @("-m", "script.action_transport.report_stage_c", "--run-id", $RunId)
}
else {
    $PythonArgs = @("-m", "script.action_transport.run_stage_c", "--device", $Device)
    if ($ValidateOnly) { $PythonArgs += "--validate-only" }
    elseif ($Preflight) {
        if (-not $StageBRun -or -not $StageARun) { throw "-Preflight requires -StageBRun and -StageARun." }
        $PythonArgs += @("--preflight", "--stage-b-run", $StageBRun, "--stage-a-run", $StageARun)
    }
    elseif ($DryRun) {
        if (-not $StageBRun -or -not $StageARun) { throw "-DryRun requires -StageBRun and -StageARun." }
        $PythonArgs += @("--dry-run", "--stage-b-run", $StageBRun, "--stage-a-run", $StageARun)
    }
    else {
        if (-not $Hours -or -not $RunId -or -not $StageBRun -or -not $StageARun) {
            throw "A full run requires -Hours, -RunId, -StageBRun and -StageARun."
        }
        $PythonArgs += @("--hours", $Hours, "--run-id", $RunId, "--stage-b-run", $StageBRun, "--stage-a-run", $StageARun)
        if ($Resume) { $PythonArgs += "--resume" }
        Write-Host "Launching a FULL Stage C run: hours=$Hours run_id=$RunId device=$Device resume=$Resume " `
            "stage_b_run=$StageBRun stage_a_run=$StageARun"
    }
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
