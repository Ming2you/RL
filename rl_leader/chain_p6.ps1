# Phase 4 chain: wait for collection -> guard on data quality -> train 3 seeds -> eval 6 rollouts.
# Runs unattended. Every stage aborts loudly rather than producing misleading numbers.
# usage: powershell -File rl_leader\chain_p6.ps1 [-MinSamples 12000] [-Steps 40000]
param(
    [int]$MinSamples = 12000,
    [int]$Steps = 40000,
    [int]$PollSec = 120,
    [int]$MaxSec = 3600
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

function Stamp { (Get-Date -Format "MM-dd HH:mm") }

# ---- [1] wait for all collection workers to exit ----
$pidFile = "logs\collect_p6_pids.txt"
if (-not (Test-Path $pidFile)) { Write-Output "ABORT: $pidFile not found"; exit 1 }
$ids = (Get-Content $pidFile) -split ","
Write-Output "$(Stamp) [1] waiting for $($ids.Count) collection workers"
while ($true) {
    $alive = 0
    foreach ($id in $ids) {
        try { $null = Get-Process -Id ([int]$id) -ErrorAction Stop; $alive++ } catch {}
    }
    if ($alive -eq 0) { break }
    Start-Sleep -Seconds $PollSec
}
Write-Output "$(Stamp) [1] collection finished"

# ---- [2] data quality guard ----
Write-Output "$(Stamp) [2] inspecting data/rl_dataset_p6"
$inspect = python rl_leader\inspect_p6.py
$inspect | ForEach-Object { Write-Output "    $_" }
$nline = $inspect | Select-String -Pattern "^SAMPLES=" | Select-Object -First 1
if (-not $nline) { Write-Output "ABORT: inspect produced no SAMPLES= line"; exit 1 }
$n = [int](($nline.Line -split "=")[1])
if ($n -lt $MinSamples) {
    Write-Output "ABORT: only $n samples (< $MinSamples). Not training - inspect data first."
    exit 1
}
Write-Output "$(Stamp) [2] guard passed: $n samples"

# ---- [3] train 3 seeds in parallel ----
Write-Output "$(Stamp) [3] training 3 seeds ($Steps steps)"
$tp = @()
foreach ($s in 0, 1, 2) {
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\iql.py", "--data", "data/rl_dataset_p6/w*.npz", `
                      "--steps", "$Steps", "--seed", "$s", `
                      "--out", "checkpoints/actor_iql_p6_s$s.pt" `
        -RedirectStandardOutput "logs\iql_p6_s$s.log" -RedirectStandardError "logs\iql_p6_s$s.err"
    $tp += $p
}
$tp | Wait-Process
$missing = 0
foreach ($s in 0, 1, 2) {
    if (-not (Test-Path "checkpoints\actor_iql_p6_s$s.pt")) { Write-Output "  MISSING checkpoint seed $s"; $missing++ }
    else { Get-Content "logs\iql_p6_s$s.log" -Tail 1 | ForEach-Object { Write-Output "  s${s}: $_" } }
}
if ($missing -gt 0) { Write-Output "ABORT: $missing checkpoints missing"; exit 1 }
Write-Output "$(Stamp) [3] training done"

# ---- [4] eval 6 rollouts in parallel (collection is over, so no oversubscription) ----
Write-Output "$(Stamp) [4] evaluating (6 parallel rollouts)"
& "$root\rl_leader\run_seeds_eval.ps1" -MaxSec $MaxSec -Prefix "actor_iql_p6_s" -TagPrefix "p6s"
Write-Output "$(Stamp) [4] eval done"

# ---- [5] judge and branch (GOOD -> more seeds / BAD -> diagnose -> maybe more data) ----
& "$root\rl_leader\chain_p6_stage2.ps1" -Steps $Steps -MaxSec $MaxSec
Write-Output "$(Stamp) CHAIN COMPLETE"
