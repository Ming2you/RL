param(
    [string]$CollectionDir = "data/contract_v4_24h_v2",
    [string]$ResultDir = "results/contract_v4_24h_v2"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$collection = Join-Path $root $CollectionDir
$finished = Join-Path $collection "finished.json"
$statusLog = Join-Path $collection "postprocess_watcher.log"

while (-not (Test-Path -LiteralPath $finished)) {
    Add-Content -Encoding UTF8 -Path $statusLog -Value `
        ("{0:o} waiting_for_collection" -f (Get-Date))
    Start-Sleep -Seconds 300
}

Add-Content -Encoding UTF8 -Path $statusLog -Value `
    ("{0:o} starting_postprocess" -f (Get-Date))

& (Join-Path $PSScriptRoot "run_contract_v4_postprocess.ps1") `
    -Data (Join-Path $CollectionDir "worker_*.npz") `
    -ResultDir $ResultDir `
    -TrainingSteps 80000 `
    -MinimumTransitions 10000

$exitCode = $LASTEXITCODE
Add-Content -Encoding UTF8 -Path $statusLog -Value `
    ("{0:o} postprocess_exit={1}" -f (Get-Date), $exitCode)
if ($exitCode -ne 0) {
    exit $exitCode
}
