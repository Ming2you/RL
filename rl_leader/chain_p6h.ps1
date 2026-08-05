# Phase 6H chain (2026-08-02): HARD budget + omega + intra-merge price.
#
# Design (per user): price must NOT break the budget constraint. Total stays at N_UF;
# price differentiates distribution across ramps (negative-price ramp releases more,
# siblings tighten to compensate).
#
#   split=True        -> Sum(meter) == N_UF enforced (hard)
#   omega action      -> cross-merge split, normalised to the leak-free window
#                        [1-cap/N_UF, cap/N_UF] so no budget is wasted at any N_UF
#   per-ramp price    -> intra-merge split at constant total
#
# Verified by smoke_p6h.py: total deviation exactly 0 across the omega range,
# W-E spans +-746 veh/h, price swaps 1500<->1127 with total unchanged.
#
# What Phase 4/5 got wrong:
#   Phase 4 = hard budget but omega frozen at 0.5 -> redistribution trapped inside each merge
#   Phase 5 = omega free but split=False -> budget demoted to a weak soft anchor, release
#             pinned at cap 6000 during peak, TTT collapsed to no-control level
#
# usage: powershell -File rl_leader\chain_p6h.ps1
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
foreach ($d in @("logs", "traces", "data\rl_dataset_p6h")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null }
}
function Stamp { (Get-Date -Format "MM-dd HH:mm") }

# ---------- [1] collect (no --price-level: hard budget) ----------
Write-Output "$(Stamp) [1] collecting: $Workers x $Episodes eps, guard ${MaxEpSec}s, HARD budget + omega"
$pids = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $tag = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(700 + $i)", `
                      "--episodes", "$Episodes", "--max-ep-sec", "$MaxEpSec", `
                      "--out", "data/rl_dataset_p6h/w$tag.npz" `
        -RedirectStandardOutput "logs\collect_p6h_w$tag.log" -RedirectStandardError "logs\collect_p6h_w$tag.err"
    $pids += $p
}
($pids.Id -join ",") | Out-File -Encoding ascii "logs\collect_p6h_pids.txt"
Write-Output ("$(Stamp) [1] pids: " + ($pids.Id -join ","))
while ($true) {
    $alive = 0
    foreach ($pr in $pids) { if (-not $pr.HasExited) { $alive++ } }
    if ($alive -eq 0) { break }
    $t = python rl_leader\inspect_p6.py "data/rl_dataset_p6h/w*.npz" 2>$null | Select-String "^SAMPLES="
    Add-Content "logs\collect_p6h_progress.log" -Value "$(Stamp) alive=$alive $t"
    Start-Sleep -Seconds $PollSec
}
Write-Output "$(Stamp) [1] collection done"

# ---------- [2] guard ----------
$inspect = python rl_leader\inspect_p6.py "data/rl_dataset_p6h/w*.npz"
$inspect | ForEach-Object { Write-Output "    $_" }
$nline = $inspect | Select-String -Pattern "^SAMPLES=" | Select-Object -First 1
if (-not $nline) { Write-Output "ABORT: no SAMPLES line"; exit 1 }
$n = [int](($nline.Line -split "=")[1])
if ($n -lt $MinSamples) { Write-Output "ABORT: only $n samples (< $MinSamples)"; exit 1 }
Write-Output "$(Stamp) [2] guard passed: $n samples"

# ---------- [3] train ----------
Write-Output "$(Stamp) [3] training 3 seeds"
$tp = @()
foreach ($s in 0, 1, 2) {
    $tp += Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p6h/w*.npz", `
                      "--steps", "$Steps", "--seed", "$s", `
                      "--out", "checkpoints/actor_iql_p6h_s$s.pt" `
        -RedirectStandardOutput "logs\iql_p6h_s$s.log" -RedirectStandardError "logs\iql_p6h_s$s.err"
}
$tp | Wait-Process
foreach ($s in 0, 1, 2) {
    if (-not (Test-Path "checkpoints\actor_iql_p6h_s$s.pt")) { Write-Output "ABORT: seed $s checkpoint missing"; exit 1 }
    Get-Content "logs\iql_p6h_s$s.log" -Tail 1 | ForEach-Object { Write-Output "  s${s}: $_" }
}
Write-Output "$(Stamp) [3] training done"

# ---------- [4] eval + ablations ----------
# full | zero-price (price axis contribution) | fix-omega (cross-merge axis contribution)
Write-Output "$(Stamp) [4] evaluating: full + zero-price + fix-omega"
foreach ($batch in @(
    @{tag = "p6h"; extra = @() },
    @{tag = "p6hzp"; extra = @("--zero-price") },
    @{tag = "p6hfo"; extra = @("--fix-omega") })) {
    $ev = @()
    foreach ($s in 0, 1, 2) {
        foreach ($cell in @("skew", "inc")) {
            $tg = "$($batch.tag)$s"
            $args = @("rl_leader\eval_guarded.py", "checkpoints/actor_iql_p6h_s$s.pt",
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

# ---------- [5] judge + attribution ----------
Write-Output "$(Stamp) [5] results"
foreach ($t in @("p6h", "p6hzp", "p6hfo")) {
    Write-Output "--- $t ---"
    python rl_leader\judge_p6.py $t 0,1,2 | ForEach-Object { Write-Output "    $_" }
}
foreach ($s in 0, 1, 2) {
    Write-Output "$(Stamp) [5] attribution seed $s"
    python rl_leader\analyze_mechanism.py --trace-dir traces --tag "p6h$s"
}
Write-Output "$(Stamp) CHAIN P6H COMPLETE"
