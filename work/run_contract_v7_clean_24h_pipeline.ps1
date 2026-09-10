param(
    [int]$DurationSec = 86400,
    [int]$WorkerCount = 8,
    [string]$RawDataDir = "data/contract_v7_clean_24h_v1",
    [string]$RelabeledDataDir = "data/contract_v7_clean_24h_h12_v1",
    [string]$ResultDir = "results/contract_v7_clean_24h_h12_s0",
    [string]$Checkpoint = "checkpoints/actor_contract_v7_clean_24h_h12_s0.pt",
    [int]$MinimumTransitions = 6000,
    [int]$MinimumLongPositive = 20,
    [int]$TrainingSteps = 40000,
    [int]$SeedBase = 1000,
    [switch]$ResumeAfterH12,
    [switch]$QuarantineH12Errors,
    [string]$Python = "python",
    [string]$TorchLib = "C:\torchlib"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$result = Join-Path $root $ResultDir
$rawPattern = Join-Path $root (Join-Path $RawDataDir "worker_*.npz")
$relabeledPattern = Join-Path $root (Join-Path $RelabeledDataDir "worker_*.npz")
$labelOutput = Join-Path $result "counterfactual_h12.json"
$checkpointPath = Join-Path $root $Checkpoint
$statePath = Join-Path $result "pipeline_state.json"
New-Item -ItemType Directory -Force -Path $result | Out-Null

$existingPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($existingPythonPath) {
    "$TorchLib;$root;$existingPythonPath"
} else {
    "$TorchLib;$root"
}

function Set-PipelineStage {
    param(
        [string]$Stage,
        [string]$Status = "running",
        [string]$Message = ""
    )
    [ordered]@{
        format_version = "contract_v7_clean_24h_pipeline_v1"
        updated_at = (Get-Date).ToString("o")
        stage = $Stage
        status = $Status
        message = $Message
        raw_data_dir = $RawDataDir
        relabeled_data_dir = $RelabeledDataDir
        result_dir = $ResultDir
        checkpoint = $Checkpoint
        duration_sec = $DurationSec
        worker_count = $WorkerCount
        minimum_transitions = $MinimumTransitions
        minimum_long_positive = $MinimumLongPositive
        training_steps = $TrainingSteps
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $statePath
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $Arguments"
    }
}

Push-Location $root
try {
    if (-not $ResumeAfterH12) {
        Set-PipelineStage "collection"
        & (Join-Path $PSScriptRoot "run_contract_v6_native_pstack_pilot.ps1") `
            -DurationSec $DurationSec `
            -WorkerCount $WorkerCount `
            -OutputDir $RawDataDir `
            -SeedBase $SeedBase `
            -Python $Python

        Set-PipelineStage "raw_audit"
        Invoke-CheckedPython @(
            "-B", "-m", "rl_leader.audit_full_action",
            "--data", $rawPattern,
            "--out", (Join-Path $result "raw_dataset_audit.json")
        )
        Invoke-CheckedPython @(
            "-B", "-m", "rl_leader.validate_contract_v4_dataset",
            "--audit", (Join-Path $result "raw_dataset_audit.json"),
            "--minimum-transitions", "$MinimumTransitions",
            "--allow-partial-residual-support",
            "--out", (Join-Path $result "raw_dataset_gate.json")
        )

        Set-PipelineStage "h12_counterfactual"
        Invoke-CheckedPython @(
            "-B", "-m", "rl_leader.generate_long_horizon_labels",
            "--data", $rawPattern,
            "--out", $labelOutput,
            "--workers", "$WorkerCount",
            "--max-rollout-steps", "12",
            "--horizons", "1,3,6,12"
        )
    }
    else {
        if (-not $QuarantineH12Errors) {
            Set-PipelineStage "h12_repair"
            Invoke-CheckedPython @(
                "-B", "-m", "rl_leader.generate_long_horizon_labels",
                "--data", $rawPattern,
                "--out", $labelOutput,
                "--workers", "$WorkerCount",
                "--max-rollout-steps", "12",
                "--horizons", "1,3,6,12",
                "--resume-errors"
            )
        }
    }

    Set-PipelineStage "h12_relabel"
    $relabelArguments = @(
        "-B", "-m", "rl_leader.relabel_long_horizon",
        "--labels", $labelOutput,
        "--data", $rawPattern,
        "--out-dir", (Join-Path $root $RelabeledDataDir),
        "--minimum-positive", "$MinimumLongPositive",
        "--required-labeled-scenarios", "sweet_170_w60,sweet_190_w60",
        "--required-positive-scenarios", "sweet_190_w60",
        "--require-complete-accepted-labels"
    )
    $relabelArguments += if ($QuarantineH12Errors) {
        "--quarantine-label-errors"
    } else {
        "--fail-on-label-errors"
    }
    Invoke-CheckedPython $relabelArguments
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.audit_full_action",
        "--data", $relabeledPattern,
        "--out", (Join-Path $result "relabeled_dataset_audit.json")
    )
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.validate_contract_v4_dataset",
        "--audit", (Join-Path $result "relabeled_dataset_audit.json"),
        "--minimum-transitions", "$MinimumTransitions",
        "--allow-partial-residual-support",
        "--out", (Join-Path $result "relabeled_dataset_gate.json")
    )

    Set-PipelineStage "iql_training"
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.iql",
        "--data", $relabeledPattern,
        "--steps", "$TrainingSteps",
        "--gamma", "1.0",
        "--support-weight", "0.3",
        "--channel-dropout", "0.1",
        "--action-parameterization", "pstack_residual",
        "--balanced-cells",
        "--seed", "0",
        "--out", $checkpointPath
    )

    Set-PipelineStage "evaluation_170_190"
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.eval_full_action", $checkpointPath,
        "--scenarios", "sweet_170_w60,sweet_190_w60",
        "--masks", "RL-FULL",
        "--max-steps", "75",
        "--max-sec", "7200",
        "--pstack-anchor",
        "--out", (Join-Path $result "rl_s0_170_190.csv"),
        "--trace-dir", (Join-Path $result "traces")
    )
    Set-PipelineStage "complete" "complete" "collection, H12 relabel, training, and evaluation completed"
}
catch {
    Set-PipelineStage "failed" "failed" $_.Exception.Message
    throw
}
finally {
    Pop-Location
    $env:PYTHONPATH = $existingPythonPath
}
