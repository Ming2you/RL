param(
    [int]$DurationSec = 7200,
    [int]$WorkerCount = 8,
    [string]$OutputDir = "data/contract_v6_native_pstack_direction_pilot_v1",
    [int]$SeedBase = 960,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$continuousBlocks = @(
    "urban:A", "urban:B", "urban:C", "urban:D", "urban:F",
    "freeway:R_D_W", "freeway:R_F_W", "freeway:R_D_E", "freeway:R_F_E",
    "vsl:FW_W__seg0", "vsl:FW_W__seg1", "vsl:FW_W__seg2",
    "vsl:FW_W__seg4", "vsl:FW_W__seg6", "vsl:FW_W__seg7",
    "vsl:FW_E__seg0", "vsl:FW_E__seg1", "vsl:FW_E__seg2",
    "vsl:FW_E__seg4", "vsl:FW_E__seg6", "vsl:FW_E__seg7"
) -join ","

& (Join-Path $PSScriptRoot "run_contract_v4_24h.ps1") `
    -DurationSec $DurationSec `
    -WorkerCount $WorkerCount `
    -OutputDir $OutputDir `
    -Python $Python `
    -PfoSupervisor $false `
    -PstackAnchor $true `
    -ResidualOnly $true `
    -LocalOnly $false `
    -SeedBase $SeedBase `
    -PerturbStartStep 0 `
    -PerturbBlockCount 4 `
    -PriorityBlocks $continuousBlocks
