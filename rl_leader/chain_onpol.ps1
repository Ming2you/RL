# Step (2): on-policy collection round (2026-08-04)
#
# Why: SCALING=SATURATED said more data from the SAME distribution is useless. All data so far
# came from hand-written behaviour policies (sweet/uniform/reactive/clamp); the states the
# LEARNED policy actually visits are absent. This is textbook offline-RL distribution shift.
# One round of on-policy collection (DAgger-style) fills exactly that hole.
#
# Training stays offline -> no divergence risk (HANDOFF section 3 rejected env-in-the-loop).
# Cost ~2.2h vs 55h+ for true online RL at the measured 0.5 env-steps/s.
#
# Rolls out the current record policy (hg600) with stratified exploration noise, mixes 30%
# scheduled episodes to keep broad coverage, writes into data/rl_dataset/ as onpol_w*.npz so
# the combined glob "data/rl_dataset/*w*.npz" picks up old + new (both obs13 / act2).
#
# usage: powershell -File rl_leader\chain_onpol.ps1
param(
    [string]$Policy = "checkpoints/actor_hg600_s0.pt",
    [int]$Workers = 10,
    [int]$Episodes = 5,
    [int]$MaxEpSec = 2700,
    [int]$Steps = 40000,
    [string]$Shape = "hinge",
    [int]$ShapeW = 600,
    [int]$MaxSec = 3600,
    [int]$PollSec = 120
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
function Stamp { (Get-Date -Format "MM-dd HH:mm") }

if (-not (Test-Path $Policy)) { Write-Output "ABORT: $Policy not found"; exit 1 }

Write-Output "$(Stamp) [1] on-policy collection: $Workers x $Episodes eps from $Policy"
$pids = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $tg = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(1300 + $i)", `
                      "--episodes", "$Episodes", "--max-ep-sec", "$MaxEpSec", `
                      "--policy", $Policy, "--onpol-frac", "0.7", `
                      "--out", "data/rl_dataset/onpol_w$tg.npz" `
        -RedirectStandardOutput "logs\collect_onpol_w$tg.log" -RedirectStandardError "logs\collect_onpol_w$tg.err"
    $pids += $p
}
($pids.Id -join ",") | Out-File -Encoding ascii "logs\collect_onpol_pids.txt"
while ($true) {
    $alive = 0
    foreach ($pr in $pids) { if (-not $pr.HasExited) { $alive++ } }
    if ($alive -eq 0) { break }
    $t = python rl_leader\inspect_p6.py "data/rl_dataset/onpol_w*.npz" 2>$null | Select-String "^SAMPLES="
    Add-Content "logs\collect_onpol_progress.log" -Value "$(Stamp) alive=$alive $t"
    Start-Sleep -Seconds $PollSec
}
Write-Output "$(Stamp) [1] collection done"
python rl_leader\inspect_p6.py "data/rl_dataset/onpol_w*.npz" | ForEach-Object { Write-Output "    $_" }

Write-Output "$(Stamp) [2] combined dataset (old + on-policy)"
$comb = python rl_leader\inspect_p6.py "data/rl_dataset/*w*.npz"
$comb | ForEach-Object { Write-Output "    $_" }
$nl = $comb | Select-String -Pattern "^SAMPLES=" | Select-Object -First 1
if (-not $nl) { Write-Output "ABORT: no SAMPLES line"; exit 1 }
$n = [int](($nl.Line -split "=")[1])
if ($n -lt 25000) { Write-Output "ABORT: combined only $n samples (expected >27k)"; exit 1 }

Write-Output "$(Stamp) [3] training 3 seeds (shape=$Shape w=$ShapeW) on combined data"
$tp = @()
foreach ($sd in 0, 1, 2) {
    $tp += Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset/*w*.npz", `
                      "--steps", "$Steps", "--seed", "$sd", "--shape", $Shape, `
                      "--shape-w", "$ShapeW", "--out", "checkpoints/actor_onpol_s$sd.pt" `
        -RedirectStandardOutput "logs\iql_onpol_s$sd.log" -RedirectStandardError "logs\iql_onpol_s$sd.err"
}
$tp | Wait-Process
foreach ($sd in 0, 1, 2) {
    if (-not (Test-Path "checkpoints\actor_onpol_s$sd.pt")) { Write-Output "ABORT: seed $sd missing"; exit 1 }
}
Get-Content "logs\iql_onpol_s0.log" | Select-String "shaping=" | ForEach-Object { Write-Output "    $_" }

Write-Output "$(Stamp) [4] evaluating 6 rollouts"
$ev = @()
foreach ($sd in 0, 1, 2) {
    foreach ($cell in @("skew", "inc")) {
        $ev += Start-Process -FilePath python -PassThru -NoNewWindow `
            -ArgumentList "rl_leader\eval_guarded.py", "checkpoints/actor_onpol_s$sd.pt", `
                          "--max-sec", "$MaxSec", "--cells", $cell, "--tag", "onpol$sd", `
                          "--trace-dir", "traces" `
            -RedirectStandardOutput "logs\eval_onpol${sd}_$cell.log" `
            -RedirectStandardError "logs\eval_onpol${sd}_$cell.err"
    }
}
$ev | Wait-Process

Write-Output "$(Stamp) [5] results  (record hinge@600: skew 6234.3 / inc 8072.2)"
python rl_leader\judge_p6.py onpol 0,1,2
Write-Output "$(Stamp) CHAIN ONPOL COMPLETE"
