# Launch machine-B canonical evaluation workers (one per scenario, CPU masks set inside the worker), hidden.
# Usage: powershell -ExecutionPolicy Bypass -File launch_all.ps1 -Preflight <path> -Review <path>
#        [-Scenarios s1,s2] [-ResumeScenarios s1]
param(
    [Parameter(Mandatory = $true)][string]$Preflight,
    [Parameter(Mandatory = $true)][string]$Review,
    [string[]]$Scenarios = @("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w"),
    [string[]]$ResumeScenarios = @()
)
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$source = "work/sdmpc_rl_return_eval_b_20260930"
$logs = Join-Path $root "results\sdmpc_rl_machine_b_20260930\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$env:PYTHONUTF8 = "1"
$records = @()
foreach ($s in $Scenarios) {
    $argsList = @("-B", "-u", "$source/worker.py", "--scenario", $s, "--preflight", $Preflight, "--review-receipt", $Review)
    if ($ResumeScenarios -contains $s) { $argsList += "--resume" }
    $log = Join-Path $logs "eval_b2_${s}_$stamp"
    $p = Start-Process -FilePath "$root\.venv-torch\Scripts\python.exe" -ArgumentList $argsList -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log" -PassThru
    $records += [pscustomobject]@{ scenario = $s; launcher_pid = $p.Id; started = (Get-Date).ToString("o");
        command = @("$root\.venv-torch\Scripts\python.exe") + $argsList; stdout = "$log.stdout.log"; stderr = "$log.stderr.log" }
}
$records | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 (Join-Path $logs "eval_b2_launch_$stamp.json")
$records | Format-Table scenario, launcher_pid
