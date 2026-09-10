param(
    [string]$Config = "work\sequential_ddqn_170_incident_v1.json",
    [ValidateSet("rl_leader.run_sequential_response_ddqn", "work.run_response_cql_ablation", "work.run_response_reference_coverage", "work.run_response_continuation_cycle", "work.run_response_onpolicy_refit", "work.run_response_value_head_ablation", "work.run_response_nonlinear_coverage", "work.run_response_cost_refinement", "work.run_response_multistep_ablation")]
    [string]$RunnerModule = "rl_leader.run_sequential_response_ddqn",
    [switch]$ProbeOnly
)

$ErrorActionPreference = "Stop"
if ($ProbeOnly -and $RunnerModule -ne "work.run_response_continuation_cycle") {
    throw "ProbeOnly is supported only by the continuation cycle runner"
}
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoPath
$configPath = (Resolve-Path -LiteralPath $Config).Path
$runConfig = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoPath $runConfig.output_dir))
if (-not $outputPath.StartsWith($repoPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must stay within the RL repository"
}
[void][System.IO.Directory]::CreateDirectory($outputPath)
$lockPath = Join-Path $outputPath "runner.lock"
$lockHandle = [System.IO.File]::Open($lockPath, "OpenOrCreate", "ReadWrite", "None")
$priorLockHandles = [System.Collections.Generic.List[System.IDisposable]]::new()
try {
    $priorDirectories = @($runConfig.start_only_after_output_dir) + @($runConfig.additional_lock_dirs)
    foreach ($priorDirectory in $priorDirectories) {
        if (-not $priorDirectory) { continue }
        $priorPath = [System.IO.Path]::GetFullPath((Join-Path $repoPath $priorDirectory))
        if (-not $priorPath.StartsWith($repoPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Previous output directory must stay within the RL repository"
        }
        $priorLockHandles.Add([System.IO.File]::Open((Join-Path $priorPath "runner.lock"), "OpenOrCreate", "ReadWrite", "None"))
    }
    foreach ($name in @("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
        [System.Environment]::SetEnvironmentVariable($name, "1", "Process")
    }
    [System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")
    $pythonPath = Join-Path $repoPath ".venv-torch\Scripts\python.exe"
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $stdoutPath = Join-Path $outputPath "runner_$stamp.stdout.log"
    $stderrPath = Join-Path $outputPath "runner_$stamp.stderr.log"
    $statusPath = Join-Path $outputPath "process.json"
    $arguments = @("-u", "-B", "-m", $RunnerModule, "--config", ('"' + $configPath + '"'))
    if ($ProbeOnly) { $arguments += "--probe-only" }
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $record = [ordered]@{
        pid = $process.Id
        wrapper_pid = $PID
        started_at = (Get-Date).ToString("o")
        config = $configPath
        module = $RunnerModule
        probe_only = [bool]$ProbeOnly
        stdout = $stdoutPath
        stderr = $stderrPath
        state = "running"
    }
    $record | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    $process.WaitForExit()
    $process.Refresh()
    $record["exit_code"] = $process.ExitCode
    $record["finished_at"] = (Get-Date).ToString("o")
    $record["state"] = if ($process.ExitCode -eq 0) { "exited" } else { "failed" }
    $record | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
} finally {
    foreach ($handle in $priorLockHandles) { $handle.Dispose() }
    $lockHandle.Dispose()
}
