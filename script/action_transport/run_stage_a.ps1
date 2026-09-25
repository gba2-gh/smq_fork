<#
.SYNOPSIS
    Stage A launcher (docs/plans/STAGE_A_AGENT_INSTRUCTIONS.md v1.3).

.DESCRIPTION
    Thin wrapper around `python -m script.action_transport.run`. Sets
    numerical-library thread caps BEFORE Python starts (required: they have
    no effect once numpy/BLAS have already initialized), resolves the
    project's conda environment, and requires an explicit -Hours budget for
    any full run.

.EXAMPLE
    .\run_stage_a.ps1 -ValidateOnly
.EXAMPLE
    .\run_stage_a.ps1 -DryRun -Protocol pooled
.EXAMPLE
    .\run_stage_a.ps1 -TimingProbe
.EXAMPLE
    .\run_stage_a.ps1 -Protocol pooled -Hours 8 -RunId stage_a_pooled_001
.EXAMPLE
    .\run_stage_a.ps1 -Protocol pooled -Hours 8 -RunId stage_a_pooled_001 -Resume
.EXAMPLE
    .\run_stage_a.ps1 -Protocol subject_disjoint -AfterPooledRun stage_a_pooled_001 -Hours 12 -RunId stage_a_sd_001
#>

param(
    [switch]$ValidateOnly,
    [switch]$DryRun,
    [switch]$TimingProbe,
    [ValidateSet("pooled", "subject_disjoint")]
    [string]$Protocol = "pooled",
    [string]$AfterPooledRun,
    [double]$Hours,
    [string]$RunId,
    [switch]$Resume,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PythonExe = "C:\Users\gzaz976\AppData\Local\anaconda3\envs\smq\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Expected the smq conda environment at $PythonExe -- update this path if the environment moved. Do not fall back to a Microsoft Store Python alias."
}
Write-Host "Using Python: $PythonExe"
& $PythonExe --version

$Cap = 8
try {
    $Cap = [Math]::Min(8, [Environment]::ProcessorCount)
} catch {}
$env:OPENBLAS_NUM_THREADS = "$Cap"
$env:MKL_NUM_THREADS = "$Cap"
$env:OMP_NUM_THREADS = "$Cap"
Write-Host "Numerical thread cap: $Cap"

$PythonArgs = @("-m", "script.action_transport.run")
if ($ValidateOnly) { $PythonArgs += "--validate-only" }
elseif ($DryRun) { $PythonArgs += @("--dry-run", "--protocol", $Protocol) }
elseif ($TimingProbe) { $PythonArgs += @("--timing-probe", "--device", $Device) }
else {
    if (-not $Hours -or -not $RunId) {
        throw "A full run requires both -Hours and -RunId."
    }
    $PythonArgs += @("--protocol", $Protocol, "--hours", $Hours, "--run-id", $RunId, "--device", $Device)
    if ($AfterPooledRun) { $PythonArgs += @("--after-pooled-run", $AfterPooledRun) }
    if ($Resume) { $PythonArgs += "--resume" }
    Write-Host "Launching a FULL Stage A run: protocol=$Protocol hours=$Hours run_id=$RunId device=$Device"
}

Push-Location $RepoRoot
try {
    # PowerShell 5.1 wraps a native command's stderr lines as terminating
    # NativeCommandErrors under EAP=Stop even on exit code 0 (e.g. tslearn's
    # harmless "h5py not installed" import warning) -- switch to Continue
    # for the native call itself and check $LASTEXITCODE for the real result.
    $PreviousEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe @PythonArgs
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousEAP
    if ($ExitCode -ne 0) {
        throw "script.action_transport.run exited with code $ExitCode"
    }
}
finally {
    Pop-Location
}
