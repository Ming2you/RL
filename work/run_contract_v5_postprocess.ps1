param(
    [string]$Data = "data/contract_v5_residual_24h_v1/worker_*.npz",
    [string]$ResultDir = "results/contract_v5_residual_24h_v1",
    [int]$TrainingSteps = 80000,
    [int]$MinimumTransitions = 10000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_contract_v4_postprocess.ps1") `
    -Data $Data `
    -ResultDir $ResultDir `
    -TrainingSteps $TrainingSteps `
    -MinimumTransitions $MinimumTransitions `
    -Python $Python `
    -CheckpointStem "actor_contract_v5_residual_24h_v1" `
    -ActionParameterization "pstack_residual" `
    -BalancedCells $true `
    -PfoSupervisor $true `
    -PstackAnchor $true
