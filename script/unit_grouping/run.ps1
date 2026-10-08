<#
.SYNOPSIS
    Unit-grouping test launcher (docs/plans/UNIT_GROUPING_TEST_PLAN.md, v2). Exactly one mode switch.
.EXAMPLE
    .\script\unit_grouping\run.ps1 -Smoke -RunId ug_smoke_01
.EXAMPLE
    .\script\unit_grouping\run.ps1 -Run -RunId ug_lara_001 -Splits primary,fold0,fold1,fold2,fold3 -Hours 10 -ReuseCompleted *>&1 | Tee-Object ug.log
#>
param(
    [switch]$Preflight, [switch]$Smoke, [switch]$DryRun, [switch]$Run,
    [string]$RunId, [string[]]$Seeds = @("111", "222", "1538574472"), [string[]]$Splits = @("primary"),
    [double]$Hours, [switch]$ReuseCompleted, [string]$PythonExe
)
$modes = @($Preflight, $Smoke, $DryRun, $Run) | Where-Object { $_ }
if ($modes.Count -eq 0) { Get-Help $PSCommandPath -Examples; exit 0 }
if ($modes.Count -gt 1) { Write-Host "Modes are mutually exclusive."; exit 2 }
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = if ($PythonExe) { $PythonExe } elseif ($env:SMQ_PYTHON) { $env:SMQ_PYTHON } else { "$env:USERPROFILE\AppData\Local\anaconda3\envs\smq\python.exe" }
if (-not (Test-Path $Python)) { Write-Host "smq python not found; pass -PythonExe"; exit 2 }
$Cap = [Math]::Min(8, [Environment]::ProcessorCount)
$env:OPENBLAS_NUM_THREADS = "$Cap"; $env:MKL_NUM_THREADS = "$Cap"; $env:OMP_NUM_THREADS = "$Cap"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"; $env:PYTHONPATH = $RepoRoot
$mode = if ($Preflight) { "preflight" } elseif ($Smoke) { "smoke" } elseif ($DryRun) { "dry-run" } else { "run" }
$PyArgs = @("-m", "script.unit_grouping.run", $mode, "--seeds", ($Seeds -join ","), "--splits", ($Splits -join ","))
if ($RunId) { $PyArgs += @("--run-id", $RunId) }
if ($Hours) { $PyArgs += @("--hours", $Hours) }
if ($ReuseCompleted) { $PyArgs += "--reuse-completed" }
if ($Run -and (-not $Hours -or -not $RunId)) { Write-Host "-Run needs -RunId and -Hours."; exit 2 }
if ($Smoke -and -not $RunId) { Write-Host "-Smoke needs -RunId (containing 'smoke')."; exit 2 }
Set-Location $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PyArgs
exit $LASTEXITCODE
