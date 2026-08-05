# Phase 5 chain (2026-08-01): price-LEVEL + omega, unattended overnight.
#
# What changed vs Phase 4 (which failed):
#   1. price now controls TOTAL inflow (follower.metering_price_split=False).
#      Phase 4 measured total variation of 0.0 veh/h across price +-1000; with split off
#      the same sweep moves total 4200..6000 (probe_price_level.py).
#   2. omega_F (W/E link budget split) is exposed. It was frozen at 0.5/0.5 because the RL
#      env bypasses leader.solve; P-Stack uses 0.26..0.98 on this axis.
#   3. behaviour policy is piecewise-constant (hold 8-20 steps) instead of i.i.d. per step.
#      That was the root cause of SCALING=SATURATED: no low-and-flat clamp existed in data.
#
# Stages: collect -> guard -> train 3 seeds -> eval 6 rollouts -> judge -> ablation attribution.
# usage: powershell -File rl_leader\chain_p5.ps1
param(
    [int]$Workers = 10,          # 10 physical cores: 1 worker per core, no HT contention
    [int]$Episodes = 26,
    [int]$MaxEpSec = 2700,       # 75 steps needs ~1875s at 10 workers (Phase 4 used 1800 -> 100% abort)
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
foreach ($d in @("logs", "traces", "data\rl_dataset_p5")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null }
}
function Stamp { (Get-Date -Format "MM-dd HH:mm") }

# ---------- [1] collect ----------
Write-Output "$(Stamp) [1] collecting: $Workers workers x $Episodes eps, guard ${MaxEpSec}s, price-level+omega"
$pids = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $tag = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(500 + $i)", `
                      "--episodes", "$Episodes", "--max-ep-sec", "$MaxEpSec", `
                      "--price-level", "--out", "data/rl_dataset_p5/w$tag.npz" `
        -RedirectStandardOutput "logs\collect_p5_w$tag.log" -RedirectStandardError "logs\collect_p5_w$tag.err"
    $pids += $p
}
($pids.Id -join ",") | Out-File -Encoding ascii "logs\collect_p5_pids.txt"
Write-Output ("$(Stamp) [1] pids: " + ($pids.Id -join ","))
while ($true) {
    $alive = 0
    foreach ($pr in $pids) { if (-not $pr.HasExited) { $alive++ } }
    if ($alive -eq 0) { break }
    $t = python rl_leader\inspect_p6.py "data/rl_dataset_p5/w*.npz" 2>$null | Select-String "^SAMPLES="
    Add-Content "logs\collect_p5_progress.log" -Value "$(Stamp) alive=$alive $t"
    Start-Sleep -Seconds $PollSec
}
Write-Output "$(Stamp) [1] collection done"

# ---------- [2] guard ----------
Write-Output "$(Stamp) [2] inspecting"
$inspect = python rl_leader\inspect_p6.py "data/rl_dataset_p5/w*.npz"
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
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p5/w*.npz", `
                      "--steps", "$Steps", "--seed", "$s", `
                      "--out", "checkpoints/actor_iql_p5_s$s.pt" `
        -RedirectStandardOutput "logs\iql_p5_s$s.log" -RedirectStandardError "logs\iql_p5_s$s.err"
}
$tp | Wait-Process
$missing = 0
foreach ($s in 0, 1, 2) {
    if (-not (Test-Path "checkpoints\actor_iql_p5_s$s.pt")) { $missing++ }
    else { Get-Content "logs\iql_p5_s$s.log" -Tail 1 | ForEach-Object { Write-Output "  s${s}: $_" } }
}
if ($missing -gt 0) { Write-Output "ABORT: $missing checkpoints missing"; exit 1 }
Write-Output "$(Stamp) [3] training done"

# ---------- [4] eval (6 parallel) + ablations (12 more) ----------
# full: both axes live | zp: price zeroed | fo: omega fixed at 0.5 -> attributes each axis.
Write-Output "$(Stamp) [4] evaluating: full + zero-price + fix-omega (18 rollouts, batched)"
foreach ($batch in @(
    @{tag = "p5"; extra = @() },
    @{tag = "p5zp"; extra = @("--zero-price") },
    @{tag = "p5fo"; extra = @("--fix-omega") })) {
    $ev = @()
    foreach ($s in 0, 1, 2) {
        foreach ($cell in @("skew", "inc")) {
            $args = @("rl_leader\eval_guarded.py", "checkpoints/actor_iql_p5_s$s.pt",
                "--max-sec", "$MaxSec", "--price-level", "--cells", $cell,
                "--tag", "$($batch.tag)$s", "--trace-dir", "traces") + $batch.extra
            $ev += Start-Process -FilePath python -PassThru -NoNewWindow -ArgumentList $args `
                -RedirectStandardOutput "logs\eval_$($batch.tag)${s}_$cell.log" `
                -RedirectStandardError "logs\eval_$($batch.tag)${s}_$cell.err"
        }
    }
    $ev | Wait-Process
    Write-Output "$(Stamp) [4] batch $($batch.tag) done"
}

# ---------- [5] judge + attribution ----------
Write-Output "$(Stamp) [5] results"
foreach ($t in @("p5", "p5zp", "p5fo")) {
    Write-Output "--- $t ---"
    python rl_leader\judge_p6.py $t 0,1,2 | ForEach-Object { Write-Output "    $_" }
}
Write-Output "$(Stamp) [5] mechanism attribution (seed 0)"
python rl_leader\analyze_mechanism.py --trace-dir traces --tag p5s0
Write-Output "$(Stamp) CHAIN P5 COMPLETE"
