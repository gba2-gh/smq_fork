<#
.SYNOPSIS
    Self-labelled embedding test launcher (docs/plans/SELF_LABELLED_EMBEDDING_PLAN.md). Exactly one mode switch.
.EXAMPLE
    .\script\self_labelled\run.ps1 -Smoke -RunId sl_smoke_01
.EXAMPLE
    .\script\self_labelled\run.ps1 -Run -RunId sl_pilot_001 -Arms F1,L_full -Seeds 111 -Hours 2 -ReuseCompleted *>&1 | Tee-Object sl_pilot.log
.EXAMPLE
    .\script\self_labelled\run.ps1 -Run -RunId sl_babel2_001 -Hours 12 -ReuseCompleted *>&1 | Tee-Object sl.log
#>
param(
    [switch]$Preflight, [switch]$Smoke, [switch]$DryRun, [switch]$Run,
    [string]$RunId, [string[]]$Arms = @("F0", "F1", "L_full", "L_noeq", "L_noaug"),
    [string[]]$Seeds = @("111", "222", "1538574472", "333", "444"),
    [string[]]$Folds = @("fold0", "fold1", "fold2", "fold3", "fold4"),
    [string]$Device = "cuda", [double]$Hours, [switch]$ReuseCompleted, [string]$PythonExe
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
$PyArgs = @("-m", "script.self_labelled.run", $mode, "--arms", ($Arms -join ","), "--seeds", ($Seeds -join ","), "--folds", ($Folds -join ","), "--device", $Device)
if ($RunId) { $PyArgs += @("--run-id", $RunId) }
if ($Hours) { $PyArgs += @("--hours", $Hours) }
if ($ReuseCompleted) { $PyArgs += "--reuse-completed" }
if ($Run -and (-not $Hours -or -not $RunId)) { Write-Host "-Run needs -RunId and -Hours."; exit 2 }
if ($Smoke -and -not $RunId) { Write-Host "-Smoke needs -RunId (containing 'smoke')."; exit 2 }
Set-Location $RepoRoot
$ErrorActionPreference = "Continue"
& $Python @PyArgs
exit $LASTEXITCODE
