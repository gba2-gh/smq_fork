<#
.SYNOPSIS
    Runs Stage 1 and then Stage 2 of the MASQ replication one after another (script/week5/masq_experiments/README.md).

.DESCRIPTION
    Both stages use the same run ID, so Stage 2 reuses Stage 1's cached units. Output of each stage is shown and also written
    to logs\masq_<RunId>_stage<N>.log under the repo root. Stage 2 starts only if Stage 1 ended 'finished' or 'budget_stop'
    (units skipped for lack of time are recorded and can be resumed by re-running); if Stage 1 has failed units (status
    'failed') the chain stops, because a systematic failure such as out-of-memory would repeat in Stage 2. Pass -ContinueOnFailure
    to run Stage 2 anyway. Re-running this script resumes: finished units are cached.

    Only one process may work on a run ID. If a MASQ run for this RunId is already running (for example Stage 1 launched by
    hand), the script refuses to start; with -WaitForRunning it waits for that process to finish, then continues as usual
    (Stage 1 is then a fast cached pass that also retries anything it skipped, followed by Stage 2).

    Logs are UTF-8 (Windows PowerShell 5.1 Tee-Object would write UTF-16).

.EXAMPLE
    .\script\week5\masq_experiments\run_all.ps1
.EXAMPLE
    .\script\week5\masq_experiments\run_all.ps1 -RunId masq_v1 -Hours1 6 -Hours2 6
.EXAMPLE
    .\script\week5\masq_experiments\run_all.ps1 -RunId masq_dry -DryRun
#>

param(
    [string]$RunId = "masq_v1",
    [double]$Hours1 = 6,
    [double]$Hours2 = 6,
    [switch]$DryRun,
    [switch]$ContinueOnFailure,
    [switch]$WaitForRunning,
    [switch]$Cpu,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$RunPs1 = Join-Path $PSScriptRoot "run.ps1"
$ManifestPath = Join-Path $RepoRoot "results\week5\masq_experiments\$RunId\manifest.json"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-RunStatus {
    if (-not (Test-Path $ManifestPath)) { return $null }
    return (Get-Content $ManifestPath -Raw | ConvertFrom-Json)
}

function Get-RunningMasq {
    # python processes working on this run ID (driver or workers)
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "masq_experiments\.run" -and $_.CommandLine -match "--run-id\s+$RunId(\s|$)" }
}

function Invoke-Stage([string]$Stage, [double]$Hours) {
    $log = Join-Path $LogDir "masq_${RunId}_stage$Stage.log"
    $args = @{ RunId = $RunId; Stage = $Stage }
    if ($DryRun) { $args.DryRun = $true } else { $args.Hours = $Hours }
    if ($Cpu) { $args.Cpu = $true }
    if ($PythonExe) { $args.PythonExe = $PythonExe }
    Write-Host ""
    Write-Host "=== MASQ stage $Stage ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')); log: $log ==="
    "" | Out-File -FilePath $log -Encoding utf8
    & $RunPs1 @args *>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $log -Value $line -Encoding UTF8
    }
    $code = $LASTEXITCODE
    Write-Host "=== stage $Stage ended with exit code $code ($(Get-Date -Format 'HH:mm:ss')) ==="
    return $code
}

$started = Get-Date

if (-not $DryRun) {
    $running = @(Get-RunningMasq)
    if ($running.Count -gt 0) {
        if (-not $WaitForRunning) {
            Write-Host "A MASQ run for '$RunId' is already running (process ids: $(($running | ForEach-Object { $_.ProcessId }) -join ', ')). Starting a second one would make two processes write the same units."
            Write-Host "Re-run with -WaitForRunning to wait for it to finish and then continue with Stage 2."
            exit 3
        }
        Write-Host "Waiting for the running MASQ process(es) for '$RunId' to finish ..."
        while (@(Get-RunningMasq).Count -gt 0) { Start-Sleep -Seconds 60 }
        Write-Host "Running process finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); continuing."
    }
}

$code1 = Invoke-Stage "1" $Hours1
if ($DryRun) {
    Invoke-Stage "2" $Hours2 | Out-Null
    exit 0
}
if ($code1 -ne 0) {
    Write-Host "Stage 1 exited with code $code1 (gates failed or the driver crashed); not starting Stage 2."
    exit $code1
}

$m = Get-RunStatus
$status = if ($m) { $m.status } else { "unknown" }
Write-Host "Stage 1 manifest status: $status"
if ($status -eq "failed" -and -not $ContinueOnFailure) {
    Write-Host "Stage 1 has failed units (see not_run in $ManifestPath and the log). Fix the cause, or re-run with -ContinueOnFailure. Not starting Stage 2."
    exit 2
}

$code2 = Invoke-Stage "2" $Hours2
$m = Get-RunStatus
Write-Host ""
Write-Host "Done in $([math]::Round(((Get-Date) - $started).TotalHours, 2)) h. Final manifest status: $($m.status)"
Write-Host "Report: results\week5\masq_experiments\$RunId\REPORT.md"
if ($m.not_run -and ($m.not_run.PSObject.Properties.Count -gt 0)) {
    Write-Host "Units not run:"
    $m.not_run.PSObject.Properties | ForEach-Object { Write-Host "  $($_.Name): $($_.Value)" }
}
exit $code2
