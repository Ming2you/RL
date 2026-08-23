param(
    [string]$TargetedData = "data/targeted_24h_v1/worker_*.npz",
    [string]$ResultDir = "results/five_cell_targeted_v1",
    [int]$TrainingSteps = 80000,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = "." }
$oldData = "data/full_action_v3_response_fixed/*.npz"
$combinedData = "$oldData,$TargetedData"
$scenarios = "sweet_155_w60,sweet_170_w60,sweet_170_incident_w60,sweet_170_skew15_w60,sweet_190_w60"
$result = Join-Path $root $ResultDir
$logs = Join-Path $result "logs"
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
        "--data", $TargetedData,
        "--out", (Join-Path $result "targeted_dataset_audit.json")
    )
    Invoke-CheckedPython @(
        "-B", "-m", "rl_leader.audit_full_action",
        "--data", $combinedData,
        "--out", (Join-Path $result "combined_dataset_audit.json")
    )

    for ($seed = 0; $seed -lt 3; $seed++) {
        $checkpoint = Join-Path $root ("checkpoints/actor_full_iql_targeted24h_s{0}.pt" -f $seed)
        Invoke-CheckedPython @(
            "-B", "-m", "rl_leader.iql",
            "--data", $combinedData,
            "--steps", "$TrainingSteps",
            "--gamma", "1.0",
            "--support-weight", "0.3",
            "--channel-dropout", "0.1",
            "--seed", "$seed",
            "--out", $checkpoint
        )
    }

    $evaluations = @()
    for ($seed = 0; $seed -lt 3; $seed++) {
        $checkpoint = Join-Path $root ("checkpoints/actor_full_iql_targeted24h_s{0}.pt" -f $seed)
        $evalOutput = Join-Path $result ("rl_s{0}.csv" -f $seed)
        $stdout = Join-Path $logs ("eval_s{0}.out.log" -f $seed)
        $stderr = Join-Path $logs ("eval_s{0}.err.log" -f $seed)
        $evalArguments = @(
            "-B", "-m", "rl_leader.eval_full_action", $checkpoint,
            "--scenarios", $scenarios,
            "--masks", "RL-FULL",
            "--max-steps", "75",
            "--max-sec", "7200",
            "--out", $evalOutput
        )
        $process = Start-Process -FilePath $Python -ArgumentList $evalArguments `
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
}
finally {
    Pop-Location
}

Write-Output "targeted audit, training, evaluation, and comparison complete"
