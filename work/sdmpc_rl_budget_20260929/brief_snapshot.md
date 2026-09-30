# Task: immutable baseline runtime snapshot

Read this first. Implement source/input snapshot preparation for the newly
authorized budget-RL pilot. Do not implement RL/controller changes.

Source root: C:/Users/alsrj/Documents/Numerical Simulation
Destination repo: C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL
Actual entry: work/sdmpc_externality_ablation_20260923/run_scenario.py,
--variant upper --externality on. Current branch codex/sdmpc-rl-budget-20260929.

Write scope: work/sdmpc_rl_budget_20260929/freeze_runtime.py,
work/sdmpc_rl_budget_20260929/test_freeze_runtime.py,
work/sdmpc_rl_budget_20260929/snapshot_report.md and generated snapshot under
artifacts/sdmpc_budget_baseline_20260929/. Do not edit other files or git commit.

Create an explicit, byte-preserving source/input snapshot with manifest SHA256,
file relative names and source Git revision/dirty provenance. Preserve only
runtime sources and necessary JSON/YAML fixtures, not all past output trees,
third-party deps, archives or large result logs. Inspect dependency/bootstrap
chain yourself; source runtime.load -> profile_current.bootstrap -> run_cell
loads work/sdmpc_matrix_14400_20260912/historical_tree; source fixtures referenced
by runtime.load and bootstrap must be included. Include scenario protocols for
sweet_170_incident_w, plus other protocol folders if tiny. Never modify source
project. Reject existing differing destination files. Record file hashes before
and after copying; preserve original absolute provenance in original JSON.

Inspect original source tree instructions as applicable. Snapshot creation alone
does not execute simulation. Avoid .pyc/numba writes to original source. Use
standard-library code for freezing; use apply_patch for code edits. Existing
.venv-torch/Scripts/python.exe works but scipy/pytest aren't installed there;
parent is preparing separate dependencies. Run unittest tests of path validation,
copy/hash verification and mismatch refusal. Report exact commands/outcomes and
any missing runtime dependency. This is immutable baseline data, not a replacement
for existing RL src. Parent will implement isolated module bootstrap separately.

Return concise DONE/BLOCKED, changed paths, snapshot size/count and test result.
