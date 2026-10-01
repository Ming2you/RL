SPEC: FAIL - full-run reconciliation has the two acceptance gaps below.
QUALITY: FAIL - correct these gaps before accepting the reconciliation output.

1. [P2] Validate the failed-policy run before using its TTT denominator.
   [analyze_sdmpc_recovery_20260929.py:69](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/analyze_sdmpc_recovery_20260929.py:69)
   reads only `episode_00_summary.json`, then publishes its TTT and
   `improvement_vs_failed_rl_pct`. Unlike the carry-center path at line 46,
   this path does not verify completion, summary/trace accounting, scenario,
   canonical profile, source/runtime contract, or the old model identity. A
   stale or mismatched summary therefore changes the reported improvement
   while the output can still say `status="reconciled"`; a zero denominator
   instead aborts the analysis. The old model's SHA check does not authenticate
   its evaluation summaries. Load this input through `load_completed_run`
   with mode `rl`, seeds `[None]`, the expected scenario/source/runtime and
   `model_sha256=BASE_HASH`; compare its physical/profile/schema/warmup
   contract with the other runs and require a positive finite denominator.

2. [P2] Reconcile terminal inventory before declaring the runs reconciled.
   [analyze_sdmpc_recovery_20260929.py:75](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/analyze_sdmpc_recovery_20260929.py:75)
   copies `summary["terminal_inventory"]` into the comparison, while the
   diagnostics called at line 77 independently copy `trace[-1]["inventory"]`.
   The delegated `validate_episode` checks TTT, coverage and decision times,
   but never checks either inventory value. A final trace inventory that
   disagrees with the summary, or is negative/nonfinite, can therefore survive
   all reconciliation checks and produce contradictory inventory values in
   the same accepted output. This leaves plan task 5's inventory reconciliation
   unimplemented. Check finite, nonnegative terminal inventory and equality
   between each accepted summary and its final trace before writing the result.

No additional actionable findings in actor selection/export loading, frozen
base-model/physical imports, or evaluator resume/STOP integration. These findings
concern the new analysis acceptance path; they do not establish that any current
result is numerically wrong or require interrupting the active evaluations.

Review was limited to the plan, actor-rereview1.md, critic-review.md,
final-review.diff, current integration code and directly called contracts.
All eight current files match final-review.diff. Read-only hashing confirmed
the selected actor is
`82bae10a4c27b2723797be71ae41a1076aad72c8a5f86aec25cbd352fbc3f0ff`,
the old model remains
`3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`,
and all 12 pinned prior implementation files plus 151 frozen payload files match.
All five evaluation settings bind the same actor, fit completion and evaluator,
canonical profiles, and physical guard mode. The v2 source hashes also match
the completed fit and two-interval serialized-parity record.

Accepted existing evidence: regression_v2.xml reports 523 tests, zero failures,
errors or skips, in 57.884 seconds; actor_fit_v2 records 375 balanced head-only
updates per arm and selected `continue`; smoke_v2 records parity PASS. The
regression report contains no recovery-analyzer tests, and the two-interval
smoke does not exercise full-run reconciliation. No tests, simulations, analysis
runs, worker stops, source edits, commits or scheduler changes were performed.
Only this report was created. Numerical superiority remains unproven until
complete results pass reconciliation.
