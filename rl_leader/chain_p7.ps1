# Phase 7 chain (2026-08-03): ramp price + urban green price, 11-dim action.
#
# Hypothesis (user): ramp price alone failed because restraining inflow just moves the queue
# to the ramps/urban. P-CENT wins on freeway (-953/-809) and PAYS in urban (+332/+439) using
# urban signals - a channel RL never had. Give the leader BOTH and the trade becomes possible.
#
#   action = [N_P, N_UF, ramp_price x4, green_price x5]
#   split=True  -> Sum(meter) == N_UF hard; ramp price redistributes only
#   green price -> follower.signal_marginal_price, cost += w*g*(p1-ref), ref=prev green
#
# omega dropped: Phase 6H measured its contribution at ~0 (skew +13, inc -54).
#
# Ablations at eval attribute each axis AND their interaction:
#   p7    = both on          p7zr = ramp price off     p7zg = green off      p7bo = both off
#
# usage: powershell -File rl_leader\chain_p7.ps1
param(
    [int]$Workers = 10,
    [int]$Episodes = 26,
    [int]$MaxEpSec = 2700,
    [int]$Steps = 40000,
    [int]$MinSamples = 12000,
    [int]$MaxSec = 3600,
    [int]$PollSec = 120
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
foreach ($d in @("logs", "traces", "data\rl_dataset_p7")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null }
}
function Stamp { (Get-Date -Format "MM-dd HH:mm") }

Write-Output "$(Stamp) [1] collecting: $Workers x $Episodes eps, guard ${MaxEpSec}s, ramp price + green price"
$pids = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $tag = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(900 + $i)", `
                      "--episodes", "$Episodes", "--max-ep-sec", "$MaxEpSec", "--green", `
                      "--out", "data/rl_dataset_p7/w$tag.npz" `
        -RedirectStandardOutput "logs\collect_p7_w$tag.log" -RedirectStandardError "logs\collect_p7_w$tag.err"
    $pids += $p
}
($pids.Id -join ",") | Out-File -Encoding ascii "logs\collect_p7_pids.txt"
Write-Output ("$(Stamp) [1] pids: " + ($pids.Id -join ","))
while ($true) {
    $alive = 0
    foreach ($pr in $pids) { if (-not $pr.HasExited) { $alive++ } }
    if ($alive -eq 0) { break }
    $t = python rl_leader\inspect_p6.py "data/rl_dataset_p7/w*.npz" 2>$null | Select-String "^SAMPLES="
    Add-Content "logs\collect_p7_progress.log" -Value "$(Stamp) alive=$alive $t"
    Start-Sleep -Seconds $PollSec
}
Write-Output "$(Stamp) [1] collection done"

$inspect = python rl_leader\inspect_p6.py "data/rl_dataset_p7/w*.npz"
$inspect | ForEach-Object { Write-Output "    $_" }
$nline = $inspect | Select-String -Pattern "^SAMPLES=" | Select-Object -First 1
if (-not $nline) { Write-Output "ABORT: no SAMPLES line"; exit 1 }
$n = [int](($nline.Line -split "=")[1])
if ($n -lt $MinSamples) { Write-Output "ABORT: only $n samples"; exit 1 }
Write-Output "$(Stamp) [2] guard passed: $n samples"

Write-Output "$(Stamp) [3] training 3 seeds"
$tp = @()
foreach ($s in 0, 1, 2) {
    $tp += Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p7/w*.npz", `
                      "--steps", "$Steps", "--seed", "$s", `
                      "--out", "checkpoints/actor_iql_p7_s$s.pt" `
        -RedirectStandardOutput "logs\iql_p7_s$s.log" -RedirectStandardError "logs\iql_p7_s$s.err"
}
$tp | Wait-Process
foreach ($s in 0, 1, 2) {
    if (-not (Test-Path "checkpoints\actor_iql_p7_s$s.pt")) { Write-Output "ABORT: seed $s missing"; exit 1 }
    Get-Content "logs\iql_p7_s$s.log" -Tail 1 | ForEach-Object { Write-Output "  s${s}: $_" }
}
Write-Output "$(Stamp) [3] training done"

Write-Output "$(Stamp) [4] evaluating: full / ramp-off / green-off / both-off"
foreach ($batch in @(
    @{tag = "p7"; extra = @() },
    @{tag = "p7zr"; extra = @("--zero-price") },
    @{tag = "p7zg"; extra = @("--zero-green") },
    @{tag = "p7bo"; extra = @("--zero-price", "--zero-green") })) {
    $ev = @()
    foreach ($s in 0, 1, 2) {
        foreach ($cell in @("skew", "inc")) {
            $tg = "$($batch.tag)$s"
            $args = @("rl_leader\eval_guarded.py", "checkpoints/actor_iql_p7_s$s.pt",
                "--max-sec", "$MaxSec", "--cells", $cell,
                "--tag", $tg, "--trace-dir", "traces") + $batch.extra
            $ev += Start-Process -FilePath python -PassThru -NoNewWindow -ArgumentList $args `
                -RedirectStandardOutput "logs\eval_${tg}_$cell.log" `
                -RedirectStandardError "logs\eval_${tg}_$cell.err"
        }
    }
    $ev | Wait-Process
    Write-Output "$(Stamp) [4] batch $($batch.tag) done"
}

Write-Output "$(Stamp) [5] results"
foreach ($t in @("p7", "p7zr", "p7zg", "p7bo")) {
    Write-Output "--- $t ---"
    python rl_leader\judge_p6.py $t 0,1,2 | ForEach-Object { Write-Output "    $_" }
}
foreach ($s in 0, 1, 2) {
    Write-Output "$(Stamp) [5] attribution seed $s"
    python rl_leader\analyze_mechanism.py --trace-dir traces --tag "p7$s"
}
Write-Output "$(Stamp) CHAIN P7 COMPLETE"
