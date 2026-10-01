# For each scenario, wait until the previous probe slot has written N branch files, then launch
# the next probe for that scenario on the same CPU.
# Usage: dispatch_when_done.ps1 -PrevName p4 -PrevSeedBase 8501 -Expected 4 -Name p5 -Options options_p5.json
#        -BranchSteps "1" -SeedBase 8501 -NoControl -CheckpointRoot D:\...\ckpt
param(
    [Parameter(Mandatory = $true)][string]$PrevName,
    [Parameter(Mandatory = $true)][int]$PrevSeedBase,
    [Parameter(Mandatory = $true)][int]$Expected,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Options,
    [Parameter(Mandatory = $true)][string]$BranchSteps,
    [int]$SeedBase = 8501,
    [switch]$NoControl,
    [string]$CheckpointRoot = "",
    [string]$Scenarios = "sweet_155_w,sweet_170_w,sweet_170_incident_w,sweet_170_skew15_w,sweet_190_w"
)
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$all = @("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")
$pending = [System.Collections.ArrayList]@($Scenarios.Split(",") | ForEach-Object { $_.Trim() })
while ($pending.Count -gt 0) {
    foreach ($scenario in @($pending)) {
        $seed = $PrevSeedBase + [array]::IndexOf($all, $scenario)
        $branches = Join-Path $root "results\sdmpc_rl_machine_b_20260930\probe_$PrevName\${scenario}_s$seed\branches"
        $count = 0
        if (Test-Path $branches) { $count = @(Get-ChildItem $branches -Filter *.json).Count }
        if ($count -ge $Expected) {
            Start-Sleep -Seconds 20  # let the previous process publish its status and exit
            $launch = @{ Name = $Name; Options = $Options; BranchSteps = $BranchSteps; SeedBase = $SeedBase; Only = @($scenario) }
            if ($NoControl) { $launch.NoControl = $true }
            if ($CheckpointRoot) { $launch.CheckpointRoot = $CheckpointRoot }
            & "$PSScriptRoot\launch_probe.ps1" @launch
            "dispatched $scenario at $(Get-Date -Format o)"
            $pending.Remove($scenario)
        }
    }
    Start-Sleep -Seconds 30
}
"all dispatched"
