param(
    [int]$DurationSec = 86400,
    [int]$WorkerCount = 8,
    [string]$OutputDir = "data/contract_v5_residual_24h_v1",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_contract_v4_24h.ps1") `
    -DurationSec $DurationSec `
    -WorkerCount $WorkerCount `
    -OutputDir $OutputDir `
    -Python $Python `
    -PfoSupervisor $true `
    -ResidualOnly $true
