# Task: close the three wrapper review findings

Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL. Read wrapper_review.md in this
directory and implement its three P2 required remedies. All initial probes,
parity and order-isolation sessions have now completed; no traffic run is active.

Write only budget_controller.py, budget_env.py, budget_runtime.py,
test_budget_controller.py and new test_budget_contract.py in this directory,
plus wrapper_fix_report.md. Do not edit frozen snapshot, old RL, TD3 or runner.
No commit. Use apply_patch. Do not change control selection, observation values,
budget ranges, physical constraints, guard, gamma/reward or solver behavior.

1. Export detached per-candidate failure/achieved/requested/feasibility/TTT,
stationarity/local-QP/price-history diagnostics before env.step prepares next
reference and clears them. Preserve selected archive-point identity separately.
Do not serialize full predicted TrafficState trajectories. Keep executed check
separate from rejected candidate check. Add plant log diagnostics/state to the
returned trace for audit, without double-counting plant timing. Include complete
phase wall/CPU metadata if straightforward; leave training timings to runner.

2. Environment checkpoints must reject mismatched reward divisor, cfg, lower
options, source-snapshot identity and observation version/normalization contract
before mixing restored simulator with a new lower controller. Same-contract
restore must remain exact and must not invoke PFO twice. A saved profile hash
alone is insufficient. Observe budget_runtime/runner outer pins but give Env its
own robust contract boundary as review requests.

3. budget_runtime.boot must verify frozen snapshot before imports and retain its
identity in returned rt; provide verify method for callers on completion/failure.
Only immutable snapshot root/source is now needed for real runs. Original source
root was used earlier for read-only diagnostic but need not remain allowed if
you require a snapshot manifest. Do not edit original source/manifest. run_budget
already verifies all payload and implementation pins before every step and at
completion. Parent will connect final verification to probe/check_wrapper scripts.

Regression tests cheap/synthetic: fallback retains nonzero incoming dual and
failed candidate evidence; audit detached across next preparation; reward/config/
options/source mismatch refused; same contract checkpoint admitted; source hash
tamper rejected before importing src. Avoid complex general frameworks.
Use .venv-torch/Scripts/python.exe -B with .deps-budget and this directory on
sys.path. Requires require_escalated for installed numpy/scipy package ACLs.
Do not rerun expensive traffic sims: parent will run one end-to-end smoke/resume.
Record tests/commands and changed files, return short DONE/BLOCKED.
