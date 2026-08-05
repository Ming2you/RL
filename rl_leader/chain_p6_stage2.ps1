# Phase 4 stage 2: judge the 6-dim result, then branch.
#   GOOD -> add 2 more seeds (3->5) to harden the statistic, eval, re-judge.
#   BAD  -> data-scaling diagnostic (train frac 0.5 vs 1.0, eval both).
#           still-improving -> launch another collection round, then retrain+eval.
#           saturated       -> STOP and report (fix is a design change, needs human judgment).
# usage: powershell -File rl_leader\chain_p6_stage2.ps1
param(
    [int]$Steps = 40000,
    [int]$MaxSec = 3600,
    [int]$ExtraEpisodes = 30
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
function Stamp { (Get-Date -Format "MM-dd HH:mm") }

Write-Output "$(Stamp) [S2-1] judging 6-dim result"
$j = python rl_leader\judge_p6.py p6s 0,1,2
$j | ForEach-Object { Write-Output "    $_" }
$vline = $j | Select-String -Pattern "^VERDICT=" | Select-Object -First 1
if (-not $vline) { Write-Output "ABORT: no VERDICT line"; exit 1 }
$verdict = ($vline.Line -split "=")[1]
Write-Output "$(Stamp) [S2-1] VERDICT=$verdict"

if ($verdict -eq "INCOMPLETE") {
    Write-Output "STOP: eval results incomplete (truncated or missing). Inspect logs\eval_p6s*."
    exit 1
}

if ($verdict -eq "GOOD") {
    Write-Output "$(Stamp) [S2-G] GOOD -> hardening statistic with seeds 3,4"
    $tp = @()
    foreach ($s in 3, 4) {
        $tp += Start-Process -FilePath python -PassThru -NoNewWindow `
            -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p6/w*.npz", `
                          "--steps", "$Steps", "--seed", "$s", `
                          "--out", "checkpoints/actor_iql_p6_s$s.pt" `
            -RedirectStandardOutput "logs\iql_p6_s$s.log" -RedirectStandardError "logs\iql_p6_s$s.err"
    }
    $tp | Wait-Process
    & "$root\rl_leader\run_seeds_eval.ps1" -MaxSec $MaxSec -Prefix "actor_iql_p6_s" -TagPrefix "p6s" -Seeds 3,4
    Write-Output "$(Stamp) [S2-G] re-judging with 5 seeds"
    python rl_leader\judge_p6.py p6s 0,1,2,3,4 | ForEach-Object { Write-Output "    $_" }
    Write-Output "$(Stamp) STAGE2 COMPLETE (GOOD branch)"
    exit 0
}

# ---- BAD branch: is it data-limited? ----
Write-Output "$(Stamp) [S2-B] BAD -> data-scaling diagnostic (frac 0.5 vs 1.0, seed 0)"
$tp = @()
$tp += Start-Process -FilePath python -PassThru -NoNewWindow `
    -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p6/w*.npz", "--steps", "$Steps", `
                  "--seed", "0", "--frac", "0.5", "--out", "checkpoints/actor_iql_p6_f50.pt" `
    -RedirectStandardOutput "logs\iql_p6_f50.log" -RedirectStandardError "logs\iql_p6_f50.err"
$tp | Wait-Process
if (-not (Test-Path "checkpoints\actor_iql_p6_f50.pt")) { Write-Output "ABORT: frac-0.5 training failed"; exit 1 }

# frac 1.0 is already actor_iql_p6_s0.pt (same seed 0). Eval the 50% model on both cells.
$ev = @()
foreach ($cell in @("skew", "inc")) {
    $ev += Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\eval_guarded.py", "checkpoints/actor_iql_p6_f50.pt", `
                      "--max-sec", "$MaxSec", "--trace-dir", "traces", "--tag", "p6f500", "--cells", $cell `
        -RedirectStandardOutput "logs\eval_p6f500_$cell.log" -RedirectStandardError "logs\eval_p6f500_$cell.err"
}
$ev | Wait-Process
Write-Output "$(Stamp) [S2-B] frac 0.5 result:"
python rl_leader\judge_p6.py p6f50 0 | ForEach-Object { Write-Output "    $_" }
Write-Output "$(Stamp) [S2-B] frac 1.0 result (seed 0):"
python rl_leader\judge_p6.py p6s 0 | ForEach-Object { Write-Output "    $_" }

$scale = python rl_leader\scaling_verdict.py
$scale | ForEach-Object { Write-Output "    $_" }
$sline = $scale | Select-String -Pattern "^SCALING=" | Select-Object -First 1
$scaling = if ($sline) { ($sline.Line -split "=")[1] } else { "UNKNOWN" }
Write-Output "$(Stamp) [S2-B] SCALING=$scaling"

if ($scaling -ne "DATA_LIMITED") {
    Write-Output "STOP: not data-limited (100% data is no better than 50%)."
    Write-Output "      More data will not help - the fix is a design change (action/obs/reward)."
    Write-Output "      Needs human judgment. See logs\eval_p6f500_*.log and traces\trace_p6f500_*.npz."
    exit 0
}

# Round 2 MUST NOT repeat the round-1 truncation flaw (HANDOFF 12.4): 14 workers + 1800s
# guard aborted 100% of episodes at ~57/75 steps, cutting exactly the recovery tail that
# carries the gain (HANDOFF 5). 10 workers + 2700s -> episodes complete, throughput similar.
$R2Workers = 10
$R2MaxEpSec = 2700
Write-Output "$(Stamp) [S2-B] data-limited -> round 2 ($ExtraEpisodes eps x $R2Workers workers, guard ${R2MaxEpSec}s)"
$pids = @()
for ($i = 0; $i -lt $R2Workers; $i++) {
    $tag = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(300 + $i)", `
                      "--episodes", "$ExtraEpisodes", "--max-ep-sec", "$R2MaxEpSec", `
                      "--out", "data/rl_dataset_p6/r2_w$tag.npz" `
        -RedirectStandardOutput "logs\collect_p6r2_w$tag.log" -RedirectStandardError "logs\collect_p6r2_w$tag.err"
    $pids += $p
}
Write-Output ("$(Stamp) [S2-B] round-2 workers: " + ($pids.Id -join ","))
$pids | Wait-Process
Write-Output "$(Stamp) [S2-B] round-2 collection done"
python rl_leader\inspect_p6.py | ForEach-Object { Write-Output "    $_" }

Write-Output "$(Stamp) [S2-B] retraining 3 seeds on enlarged dataset"
$tp = @()
foreach ($s in 0, 1, 2) {
    $tp += Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p6/*w*.npz", "--steps", "$Steps", `
                      "--seed", "$s", "--out", "checkpoints/actor_iql_p6b_s$s.pt" `
        -RedirectStandardOutput "logs\iql_p6b_s$s.log" -RedirectStandardError "logs\iql_p6b_s$s.err"
}
$tp | Wait-Process
& "$root\rl_leader\run_seeds_eval.ps1" -MaxSec $MaxSec -Prefix "actor_iql_p6b_s" -TagPrefix "p6bs"
Write-Output "$(Stamp) [S2-B] re-judging after more data"
python rl_leader\judge_p6.py p6bs 0,1,2 | ForEach-Object { Write-Output "    $_" }
Write-Output "$(Stamp) STAGE2 COMPLETE (BAD->more-data branch)"
