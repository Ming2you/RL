# Task 1: pure exploration policy

Implement only work/sdmpc_rl_local_20260930/exploration.py and
work/sdmpc_rl_local_20260930/test_exploration.py. Report to this ledger directory
exploration-report.md. No other edits, commits, model loads or simulations.
Read docs/rl_budget_local_collection_plan_20260930.md for the exact formula.
Use apply_patch, NumPy + stdlib only, and run synthetic tests. Runtime is
.venv-torch/Scripts/python.exe with .deps-budget on sys.path for pytest/NumPy.
Escalation may be needed for ACL; never install anything.

Public interface for concurrent parent implementation (do not rename):

```
class LocalBudgetPolicy:
    def __init__(self, behavior, seed, capacity): ...
    def choose(self, anchor): ...  # returns (float32 action[2], plain JSON record)
    def commit(self, row): ...    # returns plain JSON record, completes pending choice
    def state_dict(self): ...     # copied, JSON-safe snapshot including RNG/state
    def load_state_dict(self, state): ...
```

- behavior is exactly `carry` or `local`. Seed is a nonnegative integer, capacity
  is positive finite. Anchor/arrays finite length2. Do not constrain signed NP.
- Base initialized from actual first anchor, realized offset zero. For local,
  target offset=clip(.8*realized+[10,100]*rng.normal(size=2),[-50,-500],[50,500]).
  Desired budget=base+offset with only NUF clipped to [0,capacity]. Raw incremental
  action=(desired-anchor)/[50,1000], clipped [-1,1], cast float32. Actual predicted
  request must be computed from that final float32 action using float64 budget
  arithmetic and NUF projection, so logs do not confuse desired and realizable.
- For carry, return float32 zeros, no RNG advance, desired/requested=the projected
  anchor. Still track base/actual offsets for diagnostics. No Q or env access.
- choose twice without commit, or commit without choose, must fail. A checkpoint
  may represent pending choice; loading it must reproduce commit correctly.
- commit validates row action_anchor and action_requested against pending choice,
  B_requested[0] against projected request, finite B_executed and flags. Existing
  row fields: `selection_source` in {lower_solution,reference_fallback},
  `reference_source` in {previous,pfo_initial,pfo_recovery},
  `reference_recovery` bool consistent with pfo_recovery, `guard_mode=physical`.
  Rebase AFTER execution iff selection_source=reference_fallback or
  reference_source=pfo_recovery (including reference_recovery=True). Rebase to
  actual B_executed and reset offset0. Otherwise realized=B_executed-base.
  Never look at any TTT/price/Q field; do not rebase just for pfo_initial.
- Return records with explicit base_before/base_after, desired/realized offsets,
  desired_budget, projected_request, raw_action, final action, rate/projection
  flags, and rebase reasons. choose record/commit record may be merged by parent
  under row['exploration_audit']; commit should return complete audit record.
- state contains format/spec, base, realized_offset, pending, RNG, count. Deep
  copies; reject mismatched behavior/seed/capacity/formula/shape/nonfinite and
  internally inconsistent snapshots. Loading validation must not partially mutate
  a valid current instance on failure. Serialize after commits in collector.
- Keep this small and direct, not a generic policy/validation framework. The
  existing physical transform is authoritative and must remain untouched.

Synthetic tests: formula exactness, zero-noise mean reversion, no carry RNG draws,
both NUF boundaries/no fictitious positive offset, NP signed/rate limits,
physical fallback/recovery rebase (not TTT), commit request/anchor consistency,
deterministic continuation and pending restore, independent state copies,
invalid inputs/spec/state and failure atomicity. No production tests here.
