param(
    [string]$Python = "C:\Users\gzaz976\AppData\Local\anaconda3\envs\smq\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$outputRoot = Join-Path $repoRoot "results\segmodel\m1r2"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"

Push-Location $repoRoot
try {
    & $Python -m script.segmodel.m1r2_tests 2>&1 |
        Tee-Object -FilePath (Join-Path $outputRoot "acceptance_tests.log")
    if ($LASTEXITCODE -ne 0) {
        throw "M1r2 acceptance tests failed; full execution was not started."
    }

    $runId = [Guid]::NewGuid().ToString("N")
    $computeSeconds = 16200  # 4.5 h compute, preserving the final 30 min of the 5 h budget.
    $deadlineEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + $computeSeconds
    $stdout = Join-Path $outputRoot "run_stdout.log"
    $stderr = Join-Path $outputRoot "run_stderr.log"
    $arguments = @("-m", "script.segmodel.run_m1r2", "--deadline-epoch",
                   $deadlineEpoch.ToString(), "--run-id", $runId)
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $finished = $process.WaitForExit($computeSeconds * 1000)
    if (-not $finished) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
        "Outer watchdog stopped the active cell at the 4.5-hour compute deadline." |
            Set-Content -Path (Join-Path $outputRoot "deadline_stop.txt")
    }

    & $Python -m script.segmodel.report_m1r2
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment ended, but report generation failed. See the run logs."
    }
    if ($finished -and $process.ExitCode -ne 0) {
        throw "Experiment runner failed with exit code $($process.ExitCode). See run_stderr.log."
    }
    Write-Host "M1r2 finished. Report: $outputRoot\REPORT.md"
}
finally {
    Pop-Location
}
