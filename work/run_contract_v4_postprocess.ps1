param(
    [string]$Data = "data/contract_v4_24h_v2/worker_*.npz",
    [string]$ResultDir = "results/contract_v4_24h_v2",
    [int]$TrainingSteps = 80000,
    [int]$MinimumTransitions = 10000,
    [string]$Python = "python",
    [string]$CheckpointStem = "actor_contract_v4_24h_v2",
    [string]$ActionParameterization = "absolute",
    [bool]$BalancedCells = $false,
    [bool]$PfoSupervisor = $false,
    [bool]$PstackAnchor = $false
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = "." }
$scenarios = "sweet_155_w60,sweet_170_w60,sweet_170_incident_w60,sweet_170_skew15_w60,sweet_190_w60"
$result = Join-Path $root $ResultDir
$logs = Join-Path $result "logs"
$audit = Join-Path $result "dataset_audit.json"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

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
        "--minimum-transitions", "$MinimumTransitions",
        "--out", (Join-Path $result "dataset_gate.json")
    )

    $checkpoints = @()
    for ($seed = 0; $seed -lt 3; $seed++) {
        $checkpoint = Join-Path $root ("checkpoints/{0}_s{1}.pt" -f $CheckpointStem, $seed)
        $trainingArguments = @(
            "-B", "-m", "rl_leader.iql",
            "--data", $Data,
            "--steps", "$TrainingSteps",
            "--gamma", "1.0",
            "--support-weight", "0.3",
            "--channel-dropout", "0.1",
            "--action-parameterization", $ActionParameterization,
            "--seed", "$seed",
            "--out", $checkpoint
        )
        if ($BalancedCells) { $trainingArguments += "--balanced-cells" }
        Invoke-CheckedPython -Arguments $trainingArguments
        $checkpoints += $checkpoint
    }

    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.analyze_policy_ensemble",
        $checkpoints[0], $checkpoints[1], $checkpoints[2],
        "--data", $Data,
        "--out", (Join-Path $result "ensemble_audit.json")
    )

    $evaluations = @()
    for ($seed = 0; $seed -lt 3; $seed++) {
        $evalOutput = Join-Path $result ("rl_s{0}.csv" -f $seed)
        $stdout = Join-Path $logs ("eval_s{0}.out.log" -f $seed)
        $stderr = Join-Path $logs ("eval_s{0}.err.log" -f $seed)
        $arguments = @(
            "-B", "-m", "rl_leader.eval_full_action", $checkpoints[$seed],
            "--scenarios", $scenarios,
            "--masks", "RL-FULL",
            "--max-steps", "75",
            "--max-sec", "7200",
            "--out", $evalOutput,
            "--trace-dir", (Join-Path $result ("traces_s{0}" -f $seed))
        )
        if ($PfoSupervisor) { $arguments += "--pfo-supervisor" }
        if ($PstackAnchor) { $arguments += "--pstack-anchor" }
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -WorkingDirectory $root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $evaluations += $process
    }
    $evaluations | Wait-Process
    foreach ($process in $evaluations) {
        $process.Refresh()
        if ($process.ExitCode -ne 0) {
            throw "Evaluation process $($process.Id) failed with exit code $($process.ExitCode)"
        }
    }

    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.compare_five_cells",
        "--eval", (Join-Path $result "rl_s*.csv"),
        "--baseline-root", (Join-Path $root "results/five_cell_baselines"),
        "--out", (Join-Path $result "comparison.csv")
    )
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.validate_five_cell_policy",
        "--comparison", (Join-Path $result "comparison.csv"),
        "--phase", (Join-Path $result "phase_comparison.csv"),
        "--out", (Join-Path $result "policy_gate.json")
    )
}
finally {
    Pop-Location
}

Write-Output "contract-v4 audit, training, evaluation, and comparison complete"
