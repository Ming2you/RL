param(
    [string]$Data = "data/contract_v5_residual_24h_v1/worker_*.npz,data/contract_v5_residual_direction_pilot_v1/worker_*.npz",
    [string]$ResultDir = "results/contract_v5_residual_direction_s0",
    [int]$TrainingSteps = 80000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$result = Join-Path $root $ResultDir
$audit = Join-Path $result "dataset_audit.json"
$checkpoint = Join-Path $root "checkpoints/actor_contract_v5_residual_direction_s0.pt"
$evaluation = Join-Path $result "rl_s0_170_190.csv"
New-Item -ItemType Directory -Force -Path $result | Out-Null

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $Arguments"
    }
}

Push-Location $root
try {
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.audit_full_action",
        "--data", $Data,
        "--out", $audit
    )
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.validate_contract_v4_dataset",
        "--audit", $audit,
        "--minimum-transitions", "6000",
        "--out", (Join-Path $result "dataset_gate.json")
    )
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.iql",
        "--data", $Data,
        "--steps", "$TrainingSteps",
        "--gamma", "1.0",
        "--support-weight", "0.3",
        "--channel-dropout", "0.1",
        "--action-parameterization", "pstack_residual",
        "--balanced-cells",
        "--seed", "0",
        "--out", $checkpoint
    )
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.eval_full_action", $checkpoint,
        "--scenarios", "sweet_170_w60,sweet_190_w60",
        "--masks", "RL-FULL",
        "--max-steps", "75",
        "--max-sec", "7200",
        "--pstack-anchor",
        "--out", $evaluation,
        "--trace-dir", (Join-Path $result "traces")
    )
}
finally {
    Pop-Location
}

Write-Output "contract-v5 directional seed-0 training and 170/190 evaluation complete"
