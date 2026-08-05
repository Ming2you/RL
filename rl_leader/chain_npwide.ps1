# N_P box widening chain (2026-08-05)
#
# Finding: P-Stack's leader searches N_P down to -3315 (leader_intent_N_P_star) and lambda_P
# saturates at 10 on those steps, i.e. the constraint is genuinely binding there. Our action
# box was [0, 2200] - half the lever was cut off. corr(net_inflow, N_P_star) = -0.564 confirms
# the direction is live.
#
# This is a RANGE fix, not a dimension expansion: action stays 2-D [N_P, N_UF]. That matters -
# all four dimension expansions (price / omega / green) were rejected, while the one range fix
# we have evidence for has never been tried on the record configuration.
#
# Gate: probe_np_range.py must show green times actually move in the negative region.
# If the follower is unresponsive to N_P there, widening the box buys nothing.
#
# usage: powershell -File rl_leader\chain_npwide.ps1
param(
    [int]$Workers = 10,
    [int]$Episodes = 26,
    [int]$MaxEpSec = 2700,
    [int]$Steps = 40000,
    [int]$MinSamples = 12000,
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
foreach ($d in @("logs", "traces", "data\rl_dataset_npw")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null }
}
function Stamp { (Get-Date -Format "MM-dd HH:mm") }

Write-Output "$(Stamp) [1] collecting with widened N_P box: $Workers x $Episodes eps"
$pids = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $tg = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(1700 + $i)", `
                      "--episodes", "$Episodes", "--max-ep-sec", "$MaxEpSec", "--np-wide", `
                      "--out", "data/rl_dataset_npw/w$tg.npz" `
        -RedirectStandardOutput "logs\collect_npw_w$tg.log" -RedirectStandardError "logs\collect_npw_w$tg.err"
    $pids += $p
}
($pids.Id -join ",") | Out-File -Encoding ascii "logs\collect_npw_pids.txt"
Write-Output ("$(Stamp) [1] pids: " + ($pids.Id -join ","))
while ($true) {
    $alive = 0
    foreach ($pr in $pids) { if (-not $pr.HasExited) { $alive++ } }
    if ($alive -eq 0) { break }
    $t = python rl_leader\inspect_p6.py "data/rl_dataset_npw/w*.npz" 2>$null | Select-String "^SAMPLES="
    Add-Content "logs\collect_npw_progress.log" -Value "$(Stamp) alive=$alive $t"
    Start-Sleep -Seconds $PollSec
}
Write-Output "$(Stamp) [1] collection done"

$inspect = python rl_leader\inspect_p6.py "data/rl_dataset_npw/w*.npz"
$inspect | ForEach-Object { Write-Output "    $_" }
$nl = $inspect | Select-String -Pattern "^SAMPLES=" | Select-Object -First 1
if (-not $nl) { Write-Output "ABORT: no SAMPLES line"; exit 1 }
$n = [int](($nl.Line -split "=")[1])
if ($n -lt $MinSamples) { Write-Output "ABORT: only $n samples"; exit 1 }
Write-Output "$(Stamp) [2] guard passed: $n samples"

Write-Output "$(Stamp) [3] training 3 seeds (shape=$Shape w=$ShapeW)"
$tp = @()
foreach ($sd in 0, 1, 2) {
    $tp += Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_npw/w*.npz", `
                      "--steps", "$Steps", "--seed", "$sd", "--shape", $Shape, `
                      "--shape-w", "$ShapeW", "--out", "checkpoints/actor_npw_s$sd.pt" `
        -RedirectStandardOutput "logs\iql_npw_s$sd.log" -RedirectStandardError "logs\iql_npw_s$sd.err"
}
$tp | Wait-Process
foreach ($sd in 0, 1, 2) {
    if (-not (Test-Path "checkpoints\actor_npw_s$sd.pt")) { Write-Output "ABORT: seed $sd missing"; exit 1 }
}
Get-Content "logs\iql_npw_s0.log" | Select-String "shaping=" | ForEach-Object { Write-Output "    $_" }

Write-Output "$(Stamp) [4] evaluating 6 rollouts"
$ev = @()
foreach ($sd in 0, 1, 2) {
    foreach ($cell in @("skew", "inc")) {
        $ev += Start-Process -FilePath python -PassThru -NoNewWindow `
            -ArgumentList "rl_leader\eval_guarded.py", "checkpoints/actor_npw_s$sd.pt", `
                          "--max-sec", "$MaxSec", "--cells", $cell, "--tag", "npw$sd", `
                          "--trace-dir", "traces" `
            -RedirectStandardOutput "logs\eval_npw${sd}_$cell.log" `
            -RedirectStandardError "logs\eval_npw${sd}_$cell.err"
    }
}
$ev | Wait-Process
Write-Output "$(Stamp) [5] results  (record hinge@600: skew 6234.3 / inc 8072.2)"
python rl_leader\judge_p6.py npw 0,1,2
foreach ($sd in 0, 1, 2) {
    python rl_leader\analyze_mechanism.py --trace-dir traces --tag "npw$sd"
}
Write-Output "$(Stamp) CHAIN NPWIDE COMPLETE"
