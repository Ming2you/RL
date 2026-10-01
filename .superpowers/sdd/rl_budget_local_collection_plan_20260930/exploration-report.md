# Exploration sidecar: DONE

Implemented only:
- `work/sdmpc_rl_local_20260930/exploration.py`
- `work/sdmpc_rl_local_20260930/test_exploration.py`
- This report.

## Public API and integration

`LocalBudgetPolicy(behavior, seed, capacity)` exposes the specified `choose`,
`commit`, `state_dict`, and `load_state_dict` methods. Runtime dependencies are
NumPy and the standard library only. No environment/controller/model imports.

- `choose(anchor)` returns an independent float32 action of shape `(2,)` and a
  plain JSON audit. It uses the actual first anchor as base, the specified local
  mean-reversion formula, signed NP, and NUF projection. Carry emits exact zeros
  without advancing its RNG.
- Audit fields distinguish `desired_offset`, `desired_budget_raw`,
  `desired_budget`, `raw_action`, final `action`, `raw_request`,
  `projected_request`, and `requested_offset`. Request arithmetic promotes the
  final float32 action to float64 before multiplying by `[50, 1000]`, matching
  the existing physical transform.
- `offset_limited` and `rate_limited` are two-element boolean lists;
  `desired_nuf_projected` and `request_nuf_projected` are booleans. Both budget
  projection and float32 rounding are represented explicitly.
- `commit(row)` validates exact anchor/action/first-request equality, finite
  executed budget, allowed string flags, boolean recovery consistency, and the
  physical guard. It returns the complete audit, including `base_before`,
  `base_after`, `realized_offset_before`, `realized_offset_after`,
  `executed_budget`, and `executed_offset` (relative to the pre-execution base).
- Only `reference_fallback` and `pfo_recovery` trigger rebasing, after execution.
  Both reasons are retained when applicable. `pfo_initial` alone does not rebase.
  No TTT, price, Q, or diagnostic comparison field is accessed.
- Choice audits have `committed=False` and null post-execution fields. Commit
  audits have `committed=True`, `rebased`, and ordered `rebase_reasons`. The
  parent may replace/merge `row['exploration_audit']` with the commit audit.
- Checkpoint format is `local-budget-policy-v1`; `count` counts completed
  commits and audit `step` is zero-based. State includes spec/formula, base,
  realized offset, last execution, pending choice, and RNG. Loading validates
  a separate candidate before mutation, reconstructs pending audits, and
  replays the seeded local RNG to validate count/state consistency. Load cost
  is linear in count, appropriate for the planned 75-interval trajectories.
  Snapshots and returned arrays/records are independent copies.

## Synthetic verification

Final result: **121 passed in 0.29s**, exit 0.

Command, from the RL workspace:

```powershell
& '.venv-torch/Scripts/python.exe' -B -c 'import os, sys; os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"; sys.path.insert(0, ".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--noconftest", "work/sdmpc_rl_local_20260930/test_exploration.py"]))'
```

Coverage includes exact formula/float32 arithmetic, zero-noise mean reversion,
carry RNG invariance, both NUF boundaries without fictitious offsets, signed NP,
rate limits, fallback/recovery and initial-reference behavior, request/anchor
consistency, pending and committed JSON resume, independent copies, invalid
inputs/spec/state, overflow rejection, and transactional failure behavior.
Regression tests reproduced and fixed NumPy-array enum values being accepted
through scalar truth conversion. Final tests include all three enum fields.

The dependency ACL required an elevated test invocation; bytecode, pytest cache,
plugin autoload, and conftest discovery were disabled. No installs, old-file
edits, model loads, simulations, production tests, commits, or model overrides.
Collector integration/production execution remains outside this sidecar task.
