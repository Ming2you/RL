# Wait for each given launcher PID to exit, then launch the named probe for that scenario on its CPU.
# Usage: dispatch_after.ps1 -Pairs "sweet_170_w=13900,sweet_190_w=8328" -Name p3 -Options options_p3.json
#        -BranchSteps "14,16,18" -SeedBase 8401 [-NoControl]
param(
    [Parameter(Mandatory = $true)][string]$Pairs,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Options,
    [Parameter(Mandatory = $true)][string]$BranchSteps,
    [int]$SeedBase = 8401,
    [switch]$NoControl,
    [string]$CheckpointRoot = ""
)
$pending = @{}
foreach ($pair in $Pairs.Split(",")) { $kv = $pair.Trim().Split("="); $pending[$kv[0]] = [int]$kv[1] }
while ($pending.Count -gt 0) {
    foreach ($scenario in @($pending.Keys)) {
        if (-not (Get-Process -Id $pending[$scenario] -ErrorAction SilentlyContinue)) {
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
