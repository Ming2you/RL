# Carry environment and runner regression tests

Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL, feature branch dirty by design.
Read docs/rl_budget_carry_plan_20260929.md and new version's budget_env.py,
run_budget.py, compare_runs.py. Coordinator owns those production files. Another
agent owns budget_controller.py and test_carry_controller.py, still implementing.

Write ONLY work/sdmpc_rl_carry_20260929/test_carry_environment.py,
test_carry_run_contracts.py and this scratch folder/env_test_report.md.
No production edits, commits, git index edits or numerical simulations.

Adapt useful fake-runtime regression coverage from v1 test_budget_contract.py and
test_run_contracts.py instead of inventing heavy fixtures. New reference sources:
previous, pfo_initial, pfo_recovery. Fallback source reference_fallback. Guards
physical or h3; action anchor previous EXECUTED budget; failed requests preserved.

Test: regular prepare calls no PFO, initial prepare calls once, InvalidReference
alone triggers PFO recovery, unexpected exceptions propagate, failed recovery
fails closed. Observation identifies reference not PFO, contains carried budget
and source. Exact checkpoint restoration without PFO call; contract rejects
guard/config/options/source/schema mismatches before replacing state. True terminal
and -interval TTT/100 unchanged. Fake Coordinates now needs lower/upper arrays.

Runner compare is now only center/rl, separate carry formats, 20 updates/interval,
batch32. Test completed validation permits higher H3 TTT in physical mode but
rejects it in h3 mode, refuses invalid executed budgets/physics, source/cost/PFO
call accounting mismatches, reused/partial runs, NaNs and profile/model drift.

Run the two tests with isolated .deps-budget/.venv-torch and report exact command,
counts and production issues. You may request escalation for Windows dependency
ACLs. If controller is not yet written, work on tests independently then run when
it appears. Return concise report path and file list. Do not duplicate controller
tests or run a full simulation.
