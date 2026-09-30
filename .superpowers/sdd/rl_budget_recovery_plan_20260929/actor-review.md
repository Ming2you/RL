SPEC: FAIL
QUALITY: FAIL

1. **[P1] Freeze hidden actor weights in the controlled repair arms.**
   [actor_repair.py:24](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/actor_repair.py:24)
   freezes only the critics. Both the inherited optimizer and the fresh Adam at
   line 30 include every actor parameter, and `actor_step` backpropagates and steps
   that optimizer. The hidden layers therefore remain trainable in all three
   treatments, contrary to Task 1's explicit "Keep hidden weights ... fixed"
   requirement in
   [the plan:20](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/docs/rl_budget_recovery_plan_20260929.md:20).
   Consequently, the completed exports do not establish the specified fixed-hidden
   comparison: representation changes can contribute to the measured recovery and
   treatment ranking. Freeze every non-head actor parameter in each arm while
   preserving the continuation arm's head optimizer history. Add an invariant
   check on the fitted learner's hidden tensors across multiple updates, including
   a nonzero head and populated optimizer history; the existing
   [test_actor_repair.py:31](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/test_actor_repair.py:31)
   checks only critic immutability and the untouched base actor. Retain the existing
   exports as prior evidence; compliant replacement exports need a new version
   before they can satisfy this gate.

2. **[P2] Honor an existing STOP before resetting or restoring the environment.**
   [evaluate.py:120](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/evaluate.py:120)
   enters `env.restore` or `env.reset` and writes a checkpoint before the first
   `stopped(output)` check at line 138. With STOP already present in a checked
   directory, a fresh invocation still executes five physical warmup intervals and
   the initial PFO solve through `BudgetEnv.reset`; a resumed invocation still
   reconstructs and evaluates its reference. Thus Task 4's stop mechanism permits
   new numerical work even when the stop request predates process startup. Check
   STOP before either initialization branch and return a paused status without
   replacing an existing checkpoint. Cover fresh and resume startup with a stub
   environment that rejects any reset/restore call while STOP is present; the
   focused checkpoint-validation tests do not exercise this condition.
