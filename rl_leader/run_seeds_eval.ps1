# 3-seed held-out eval, PARALLEL by (seed x cell) = 6 single-threaded rollouts.
# Replaces run_seeds_eval.sh (original machine only; see HANDOFF section 10).
#
# Safe because (measured 2026-07-31, HANDOFF section 11):
#   - one rollout uses exactly 1.00 core / ~200 MB; 6 concurrent fit in 10 physical cores
#   - the "must run sequentially" warning in HANDOFF 6-1 applied only while 14 collectors
#     were saturating the CPU. Do NOT run this alongside collect_parallel.py workers.
#
# usage:  powershell -File rl_leader\run_seeds_eval.ps1  [-Seeds 0,1,2] [-MaxSec 3600]
param(
    [int[]]$Seeds = @(0, 1, 2),
    [int]$MaxSec = 3600,
    [string]$Prefix = "actor_iql_s",   # checkpoints/<Prefix><seed>.pt
    [string]$TagPrefix = "s"           # trace tag: <TagPrefix><seed>
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # repo root
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
if (-not (Test-Path logs)) { New-Item -ItemType Directory logs | Out-Null }
if (-not (Test-Path traces)) { New-Item -ItemType Directory traces | Out-Null }

$procs = @()
foreach ($s in $Seeds) {
    $ckpt = "checkpoints/$Prefix$s.pt"
    $tag = "$TagPrefix$s"
    if (-not (Test-Path $ckpt)) { Write-Output "SKIP seed ${s}: $ckpt not found"; continue }
    foreach ($cell in @("skew", "inc")) {
        $log = "logs\eval_${tag}_$cell"
        $p = Start-Process -FilePath python -PassThru -NoNewWindow `
            -ArgumentList "rl_leader\eval_guarded.py", $ckpt, "--max-sec", "$MaxSec", `
                          "--trace-dir", "traces", "--tag", $tag, "--cells", $cell `
            -RedirectStandardOutput "$log.log" -RedirectStandardError "$log.err"
        $procs += $p
        Write-Output ("launched seed={0} cell={1} tag={2} pid={3}" -f $s, $cell, $tag, $p.Id)
    }
}
if (-not $procs) { Write-Output "nothing to run"; exit 1 }

$procs | Wait-Process
Write-Output "=== all evals done ==="
foreach ($s in $Seeds) {
    $tag = "$TagPrefix$s"
    foreach ($cell in @("skew", "inc")) {
        $log = "logs\eval_${tag}_$cell.log"
        if (Test-Path $log) {
            Write-Output "--- $tag $cell ---"
            Get-Content $log | Select-String -Pattern "TTT|vs P-|N_UF:|truncat|FAIL" | ForEach-Object { $_.Line }
        }
    }
}

Write-Output "=== mechanism attribution (per seed) ==="
foreach ($s in $Seeds) {
    python rl_leader\analyze_mechanism.py --trace-dir traces --tag "$TagPrefix$s"
}
