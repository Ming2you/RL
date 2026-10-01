# Launch one counterfactual probe per scenario (training seeds only), hidden, CPUs 0..4.
# Usage: powershell -ExecutionPolicy Bypass -File launch_probe.ps1 -Name p1 -Options options_p1.json
#        -BranchSteps "1,16" [-SeedBase 8101]
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Options,
    [Parameter(Mandatory = $true)][string]$BranchSteps,
    [int]$SeedBase = 8101,
    [switch]$NoControl,
    [string[]]$Only = @(),
    [string]$CheckpointRoot = ""
)
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$out = Join-Path $root "results\sdmpc_rl_machine_b_20260930\probe_$Name"
$logs = Join-Path $root "results\sdmpc_rl_machine_b_20260930\logs"
New-Item -ItemType Directory -Force $out, $logs | Out-Null
foreach ($stop in @("$root\STOP", "$root\results\sdmpc_rl_machine_b_20260930\STOP", "$out\STOP")) {
    if (Test-Path $stop) { throw "STOP present: $stop" }
}
$optionsPath = (Resolve-Path (Join-Path $PSScriptRoot $Options)).Path
$env:PYTHONUTF8 = "1"
$scenarios = @("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")
$records = @()
for ($i = 0; $i -lt 5; $i++) {
    $s = $scenarios[$i]; $seed = $SeedBase + $i; $mask = [int][math]::Pow(2, $i)
    if ($Only.Count -gt 0 -and -not ($Only -contains $s)) { continue }
    $slot = Join-Path $out "${s}_s$seed"
    $argsList = @("-B", "-u", "work/sdmpc_rl_probe_b_20260930/probe.py", "--scenario", $s, "--seed", "$seed",
        "--cpu-mask", "$mask", "--branch-steps", $BranchSteps, "--options", $optionsPath, "--output", $slot)
    if ($NoControl) { $argsList += "--no-control" }
    if ($CheckpointRoot) { $argsList += @("--checkpoint-dir", (Join-Path $CheckpointRoot "${s}_s$seed")) }
    $log = Join-Path $logs "probe_${Name}_${s}_s$seed"
    $p = Start-Process -FilePath "$root\.venv-torch\Scripts\python.exe" -ArgumentList $argsList -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log" -PassThru
    $records += [pscustomobject]@{ scenario = $s; seed = $seed; cpu_mask = $mask; launcher_pid = $p.Id;
        started = (Get-Date).ToString("o"); output = $slot; command = @("$root\.venv-torch\Scripts\python.exe") + $argsList }
}
Copy-Item $optionsPath (Join-Path $out "options.json") -Force
$records | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 (Join-Path $out "launch_$(Get-Date -Format yyyyMMdd_HHmmss).json")
$records | Format-Table scenario, seed, cpu_mask, launcher_pid
