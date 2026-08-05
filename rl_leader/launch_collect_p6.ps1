# Phase 4 data collection launcher: 14 detached workers, 6-dim actions.
# Workers are independent processes (Start-Process) - they survive this script exiting.
# Progress: logs\collect_p6_w*.log + incremental npz in data\rl_dataset_p6\
# usage: powershell -File rl_leader\launch_collect_p6.ps1 [-Workers 14] [-Episodes 40]
param(
    [int]$Workers = 14,
    [int]$Episodes = 40,
    [int]$MaxEpSec = 1800
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
if (-not (Test-Path logs)) { New-Item -ItemType Directory logs | Out-Null }
if (-not (Test-Path "data\rl_dataset_p6")) { New-Item -ItemType Directory "data\rl_dataset_p6" | Out-Null }

$pids = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $tag = "{0:d2}" -f $i
    $p = Start-Process -FilePath python -PassThru -NoNewWindow `
        -ArgumentList "rl_leader\collect_parallel.py", "--seed", "$(100 + $i)", `
                      "--episodes", "$Episodes", "--max-ep-sec", "$MaxEpSec", `
                      "--out", "data/rl_dataset_p6/w$tag.npz" `
        -RedirectStandardOutput "logs\collect_p6_w$tag.log" `
        -RedirectStandardError "logs\collect_p6_w$tag.err"
    $pids += $p.Id
    Write-Output ("worker {0} pid={1} -> data/rl_dataset_p6/w{0}.npz" -f $tag, $p.Id)
}
$pids -join "," | Out-File -Encoding ascii "logs\collect_p6_pids.txt"
Write-Output ("launched {0} workers, pids saved to logs\collect_p6_pids.txt" -f $pids.Count)
