<#
.SYNOPSIS
    Milestone 1 launcher (script/compositional_dynamics/README.md). Exactly one mode switch is required.

.EXAMPLE
    .\script\compositional_dynamics\run.ps1 -Preflight
.EXAMPLE
    .\script\compositional_dynamics\run.ps1 -Smoke -RunId m1_smoke_01
.EXAMPLE
    .\script\compositional_dynamics\run.ps1 -DryRun
.EXAMPLE
    .\script\compositional_dynamics\run.ps1 -Run -RunId m1_lara_001 -Arms b0_centre,m1_ordered,m1_bag,occupancy -Seeds 111,222,1538574472 -Device cuda -Hours 10
#>

param(
    [switch]$Preflight,
    [switch]$Smoke,
    [switch]$DryRun,
    [switch]$Run,
    [switch]$Parity,
    [string]$RunId,
    [string[]]$Arms = @("b0_centre", "m1_ordered", "m1_bag", "occupancy"),
    [string[]]$Seeds = @("111", "222", "1538574472"),
    [string]$Device = "cuda",
    [string[]]$Splits = @("primary"),
    [double]$Hours,
    [switch]$ReuseCompleted,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$modes = @($Preflight, $Smoke, $DryRun, $Run, $Parity) | Where-Object { $_ }
if ($modes.Count -eq 0) { Get-Help $PSCommandPath -Examples; exit 0 }
if ($modes.Count -gt 1) { throw "Modes are mutually exclusive: choose one of -Preflight -Smoke -DryRun -Run -Parity." }

function Resolve-Python {
    if ($PythonExe) { return $PythonExe }
    if ($env:SMQ_PYTHON) { return $env:SMQ_PYTHON }
    foreach ($c in @("$env:USERPROFILE\AppData\Local\anaconda3\envs\smq\python.exe", "$env:USERPROFILE\anaconda3\envs\smq\python.exe",
                     "$env:USERPROFILE\miniconda3\envs\smq\python.exe", "$env:USERPROFILE\AppData\Local\miniconda3\envs\smq\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    throw "Could not find the 'smq' conda environment. Pass -PythonExe or set `$env:SMQ_PYTHON."
}

$Python = Resolve-Python
Write-Host "Using Python: $Python"
$Cap = [Math]::Min(8, [Environment]::ProcessorCount)
$env:OPENBLAS_NUM_THREADS = "$Cap"; $env:MKL_NUM_THREADS = "$Cap"; $env:OMP_NUM_THREADS = "$Cap"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONPATH = $RepoRoot; $env:PYTHONWARNINGS = "ignore"

$mode = if ($Preflight) { "preflight" } elseif ($Smoke) { "smoke" } elseif ($DryRun) { "dry-run" } elseif ($Parity) { "parity" } else { "run" }
$PyArgs = @("-m", "script.compositional_dynamics.run", $mode, "--arms", ($Arms -join ","), "--seeds", ($Seeds -join ","), "--device", $Device)
if ($RunId) { $PyArgs += @("--run-id", $RunId) }
if ($Hours) { $PyArgs += @("--hours", $Hours) }
if ($Run -or $Smoke) { $PyArgs += @("--splits", ($Splits -join ",")) }
if ($ReuseCompleted) { $PyArgs += "--reuse-completed" }
if ($Run -and -not $Hours) { throw "-Run requires an explicit elapsed-time cap: pass -Hours." }
if (($Run -or $Smoke) -and -not $RunId) { throw "-RunId is required for -Run and -Smoke." }

Set-Location $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PyArgs
exit $LASTEXITCODE
