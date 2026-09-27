<#
.SYNOPSIS
    Stage A launcher (docs/plans/STAGE_A_AGENT_INSTRUCTIONS.md v1.3; runner 1.3.1).

.DESCRIPTION
    Thin wrapper around `python -m script.action_transport.run`. Sets
    numerical-library thread caps BEFORE Python starts (they have no effect
    once numpy/BLAS have initialized), resolves the project's Python, and
    requires an explicit -Hours budget for any full run.

    Python resolution order: -PythonExe, then $env:SMQ_PYTHON, then the conda
    env named "smq" under the usual Anaconda/Miniconda locations. It never
    falls back to a bare `python` (the Microsoft Store alias).

.EXAMPLE
    .\script\action_transport\run_stage_a.ps1 -Preflight
.EXAMPLE
    .\script\action_transport\run_stage_a.ps1 -ValidateOnly
.EXAMPLE
    .\script\action_transport\run_stage_a.ps1 -DryRun -Protocol pooled
.EXAMPLE
    .\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 8 -RunId stage_a_pooled_002
.EXAMPLE
    .\script\action_transport\run_stage_a.ps1 -Protocol pooled -Hours 4 -RunId stage_a_pooled_002 -Resume
.EXAMPLE
    .\script\action_transport\run_stage_a.ps1 -Report -RunId stage_a_pooled_002 -CompareTo stage_a_pooled_001
#>

param(
    [switch]$ValidateOnly,
    [switch]$Preflight,
    [switch]$DryRun,
    [switch]$TimingProbe,
    [switch]$Report,
    [ValidateSet("pooled", "subject_disjoint")]
    [string]$Protocol = "pooled",
    [string]$AfterPooledRun,
    [double]$Hours,
    [string]$RunId,
    [string]$CompareTo,
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
        "$env:USERPROFILE\AppData\Local\miniconda3\envs\smq\python.exe",
        "C:\ProgramData\anaconda3\envs\smq\python.exe",
        "C:\ProgramData\miniconda3\envs\smq\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw ("Could not find the 'smq' conda environment. Pass -PythonExe <path\to\python.exe> or set " +
           "`$env:SMQ_PYTHON. (Create it with: conda env create -f environment.yml)")
}

$Python = Resolve-Python
if (-not (Test-Path $Python)) { throw "Python executable not found: $Python" }
Write-Host "Using Python: $Python"

$Cap = [Math]::Min(8, [Environment]::ProcessorCount)
$env:OPENBLAS_NUM_THREADS = "$Cap"
$env:MKL_NUM_THREADS = "$Cap"
$env:OMP_NUM_THREADS = "$Cap"
Write-Host "Numerical thread cap: $Cap"

if ($Report) {
    if (-not $RunId) { throw "-Report requires -RunId" }
    $PythonArgs = @("-m", "script.action_transport.report", "--run-id", $RunId)
    if ($CompareTo) { $PythonArgs += @("--compare-to", $CompareTo) }
}
else {
    $PythonArgs = @("-m", "script.action_transport.run", "--device", $Device)
    if ($ValidateOnly) { $PythonArgs += "--validate-only" }
    elseif ($Preflight) { $PythonArgs += "--preflight" }
    elseif ($DryRun) { $PythonArgs += @("--dry-run", "--protocol", $Protocol) }
    elseif ($TimingProbe) { $PythonArgs += "--timing-probe" }
    else {
        if (-not $Hours -or -not $RunId) { throw "A full run requires both -Hours and -RunId." }
        $PythonArgs += @("--protocol", $Protocol, "--hours", $Hours, "--run-id", $RunId)
        if ($AfterPooledRun) { $PythonArgs += @("--after-pooled-run", $AfterPooledRun) }
        if ($Resume) { $PythonArgs += "--resume" }
        Write-Host "Launching a FULL Stage A run: protocol=$Protocol hours=$Hours run_id=$RunId device=$Device resume=$Resume"
    }
}

Push-Location $RepoRoot
try {
    # PowerShell 5.1 turns native stderr lines (e.g. tslearn's harmless
    # "h5py not installed" warning) into terminating errors under EAP=Stop,
    # even on exit code 0. Use Continue for the native call; $LASTEXITCODE
    # carries the real result.
    $ErrorActionPreference = "Continue"
    & $Python @PythonArgs
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($ExitCode -ne 0) { throw "Python exited with code $ExitCode" }
}
finally {
    Pop-Location
}
