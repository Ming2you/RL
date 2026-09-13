param(
    [string]$Config = "work\five_cell_shared_v1.json"
)
$ErrorActionPreference = "Stop"
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoPath
$configPath = (Resolve-Path -LiteralPath $Config).Path
$plan = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoPath $plan.output_dir))
if (-not $outputPath.StartsWith($repoPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must stay within this repository"
}
[void][System.IO.Directory]::CreateDirectory($outputPath)
$pythonPath = (Get-Command python).Source
foreach ($name in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
    [System.Environment]::SetEnvironmentVariable($name, "1", "Process")
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$stdoutPath = Join-Path $outputPath "runner_$stamp.stdout.log"
$stderrPath = Join-Path $outputPath "runner_$stamp.stderr.log"
$arguments = @("-u", "-B", "-m", "work.run_five_cell_shared", "--config", ('"' + $configPath + '"'))
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    python = $pythonPath
    config = $configPath
    stdout = $stdoutPath
    stderr = $stderrPath
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath "launch.json") -Encoding UTF8
$record | ConvertTo-Json
