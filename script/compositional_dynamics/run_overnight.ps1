<#
.SYNOPSIS
    One overnight command for Milestone 1 (v2.2): tests, preflight, historical parity, then the full run (primary split first,
    then the four subject-disjoint folds), with a single elapsed-time cap. Secondary cells are skipped first if time runs out.

.EXAMPLE
    .\script\compositional_dynamics\run_overnight.ps1 -RunId m1_lara_001 -Hours 10 *>&1 | Tee-Object m1_overnight.log
#>
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [double]$Hours = 10,
    [string]$Device = "cuda",
    [string]$PythonExe
)
# Native commands write warnings to stderr; with "Stop" and redirection PowerShell would treat those as fatal.
# Exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"
$env:PYTHONWARNINGS = "ignore"
$Start = Get-Date
$Here = $PSScriptRoot
$PyArgs = @(); if ($PythonExe) { $PyArgs = @("-PythonExe", $PythonExe) }

& "$Here\run.ps1" -Preflight @PyArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Preflight failed; nothing was launched."; exit 1 }

# tests record verification.json for the current code hash (required for a conclusive gate)
$Python = if ($PythonExe) { $PythonExe } elseif ($env:SMQ_PYTHON) { $env:SMQ_PYTHON } else { "$env:USERPROFILE\AppData\Local\anaconda3\envs\smq\python.exe" }
$env:PYTHONPATH = (Split-Path -Parent (Split-Path -Parent $Here))
Set-Location $env:PYTHONPATH
& $Python -m script.compositional_dynamics.tests
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed; nothing was launched."; exit 1 }

& "$Here\run.ps1" -Parity @PyArgs     # a mismatch is reported, not fatal: the experiment does not depend on it
$Elapsed = ((Get-Date) - $Start).TotalHours
$Left = [Math]::Max(0.1, $Hours - $Elapsed)
& "$Here\run.ps1" -Run -RunId $RunId -Splits primary,fold0,fold1,fold2,fold3 -Device $Device -Hours $Left -ReuseCompleted @PyArgs
exit $LASTEXITCODE
