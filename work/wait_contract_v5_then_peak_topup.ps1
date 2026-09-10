param(
    [string]$BaseDir = "data/contract_v5_residual_24h_v1",
    [string]$TopupDir = "data/contract_v5_residual_peak_topup_v1",
    [int]$TopupDurationSec = 7200,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$base = Join-Path $root $BaseDir
$finishedPath = Join-Path $base "finished.json"
$deadline = (Get-Date).AddHours(15)

while (-not (Test-Path -LiteralPath $finishedPath)) {
    if ((Get-Date) -ge $deadline) {
        throw "Timed out waiting for base collection: $finishedPath"
    }
    Write-Output ("{0:o} waiting for base collection" -f (Get-Date))
    Start-Sleep -Seconds 300
}

$finished = Get-Content -Raw -LiteralPath $finishedPath | ConvertFrom-Json
if (@($finished.forced_pids).Count -gt 0) {
    throw "Base collection forced worker shutdown: $($finished.forced_pids -join ',')"
}
$workerErrors = @(
    Get-ChildItem -LiteralPath (Join-Path $base "logs") -Filter "worker_*.err.log" |
        Where-Object { $_.Length -gt 0 }
)
if ($workerErrors.Count -gt 0) {
    throw "Base collection has nonempty worker stderr: $($workerErrors.Name -join ',')"
}

Write-Output "base collection complete; starting peak residual top-up"
& (Join-Path $PSScriptRoot "run_contract_v5_peak_topup.ps1") `
    -DurationSec $TopupDurationSec `
    -OutputDir $TopupDir `
    -Python $Python
Write-Output "peak residual top-up complete"
