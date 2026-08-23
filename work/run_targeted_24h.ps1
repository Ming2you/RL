param(
    [int]$DurationSec = 86400,
    [int]$WorkerCount = 8,
    [string]$OutputDir = "data/targeted_24h_v1",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root $OutputDir
$logs = Join-Path $output "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (Get-ChildItem -Path $output -Filter "worker_*.npz" -ErrorAction SilentlyContinue) {
    throw "Targeted collection output already contains worker datasets: $output"
}

$modeRotations = @(
    "optimizer_local,loose_anchor,optimizer_local,loose_local",
    "loose_anchor,optimizer_local,loose_local,optimizer_local",
    "loose_local,optimizer_local,loose_anchor,optimizer_local",
    "optimizer_local,loose_local,optimizer_local,loose_anchor"
)
$priorityBlocks = "A,B,C,D,F,R_D_W,R_F_W,R_D_E,R_F_E"
$collectionSec = [Math]::Max($DurationSec - 600, 60)
$startedAt = Get-Date
$deadline = $startedAt.AddSeconds($DurationSec)
$processes = @()

for ($index = 0; $index -lt $WorkerCount; $index++) {
    $seed = 500 + $index
    $workerOutput = Join-Path $output ("worker_{0}.npz" -f $seed)
    $stdout = Join-Path $logs ("worker_{0}.out.log" -f $seed)
    $stderr = Join-Path $logs ("worker_{0}.err.log" -f $seed)
    $arguments = @(
        "-B", "-m", "rl_leader.collect_full_action",
        "--scenario-profile", "targeted",
        "--modes", $modeRotations[$index % $modeRotations.Count],
        "--episodes", "1000",
        "--max-steps", "75",
        "--warmup", "5",
        "--t-total", "14400",
        "--max-episode-sec", "5400",
        "--max-total-sec", "$collectionSec",
        "--perturb-start-step", "24",
        "--temporal-rho", "0.95",
        "--budget-perturb-scale", "0.05",
        "--block-perturb-scale", "0.08",
        "--perturb-block-count", "3",
        "--priority-blocks", $priorityBlocks,
        "--priority-probability", "1.0",
        "--dataset-name", "targeted_24h_v1",
        "--seed", "$seed",
        "--out", $workerOutput
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $processes += $process
}

$launch = [ordered]@{
    started_at = $startedAt.ToString("o")
    deadline = $deadline.ToString("o")
    duration_sec = $DurationSec
    collection_sec = $collectionSec
    worker_count = $WorkerCount
    pids = @($processes | ForEach-Object { $_.Id })
}
$launch | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $output "launch.json")

while ((Get-Date) -lt $deadline) {
    $alive = 0
    foreach ($process in $processes) {
        $process.Refresh()
        if (-not $process.HasExited) { $alive++ }
    }
    $datasets = @(Get-ChildItem -Path $output -Filter "worker_*.npz" -ErrorAction SilentlyContinue)
    $transitionHint = @($datasets | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum
    $status = "{0:o} alive={1}/{2} datasets={3} compressed_bytes={4}" -f `
        (Get-Date), $alive, $WorkerCount, $datasets.Count, $transitionHint
    Add-Content -Encoding UTF8 -Path (Join-Path $output "monitor.log") -Value $status
    Write-Output $status
    if ($alive -eq 0) { break }
    $remaining = [Math]::Max(0, ($deadline - (Get-Date)).TotalSeconds)
    Start-Sleep -Seconds ([int][Math]::Min(300, $remaining))
}

$forced = @()
foreach ($process in $processes) {
    $process.Refresh()
    if (-not $process.HasExited) {
        $forced += $process.Id
        Stop-Process -Id $process.Id -Force
    }
}

$finished = [ordered]@{
    finished_at = (Get-Date).ToString("o")
    forced_pids = $forced
    datasets = @(Get-ChildItem -Path $output -Filter "worker_*.npz" | Select-Object Name, Length)
}
$finished | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $output "finished.json")
Write-Output "targeted collection orchestration finished"
