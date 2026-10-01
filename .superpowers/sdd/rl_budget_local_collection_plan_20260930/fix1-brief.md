# Collector fix round 1

Read implementation-review.md in this directory first. Resolve R1-R5 only in
new local_runtime.py/collect.py/validate.py/run_wave.py/test_collection.py (and
at most one focused new test/helper if essential) under work/sdmpc_rl_local_20260930.
Do NOT change exploration.py/test_exploration.py (approved), old sources, physical
contracts, the collection formula, outcomes/thresholds or any completed output.
No production collector/model/simulator execution. No commit/install/OS changes.
Use apply_patch. Report to fix1-report.md here, with before/after regression
evidence and exact test commands. Previous scripts are preserved in before-fix1/.

Root causes/findings are statically identified. First reproduce them with focused
synthetic tests, then scoped fixes; the143passingtests are not evidence against
these missing cases. Reuse old frozen helpers where valid, but old validators
cannot be weakened or altered. Keep implementation direct, no generic scheduler.

R1: enforce the DECLARED canonical cohort/output root, not arbitrary duplicate
roots. This is simpler than a general experiment registry. Standalone smoke and
coordinator children must share ownership/maximum5numericalworker enforcement.
Address orphaned startup/import windows: a launch intent/reservation is durable
before spawning, and unknown/live identity is fail-closed rather than proof of
death. Preserve pending/completed cohort identity across coordinator loss. A
startup child must not slip past a new coordinator just because it has not yet
acquired its output lock. Never use potentially destructive process probes.
Tests may monkeypatch a canonical root to tmpdir; do not expose a production
CLI bypass. Preserve normal prescribed output tree carry|local/scenario.

R2: add new prefix/boundary validation for clock, remaining horizon, normalized
anchor and previous requested/executed budgets and physical raw/projection.
Use actual schema fields and source semantics. Validate before env.step and bind
saved observation to the checkpoint boundary, including zero and terminal steps.
Retain the admitted complete75validator unchanged. Synthetic fixture now needs
real clock/memory fields, not a two-dimensional observation that stubs all of this.

R3: inventory must be recomputed from physical state using the FROZEN accounting
definition, not the trace's claimed total. Bind checkpoint simulator state,
previous control and accounting to the final trace. Resolve the exact frozen
inventory function/required fields before coding. Tests need no simulator steps.
Do not introduce approximate inventory or change physical accounting.

R4: direct compare_pairs must authenticate scenario/behavior/seed/mask for each
slot, regardless of coordinator preflight. Test swapped intact directories.

R5: durable per-session timing must include checkpoint serialization and orderly
exit bookkeeping within a precisely stated measurable scope. Final timing-ledger
publication itself may be excluded explicitly; do not imply it is measured.
Record failed/partial sessions and flag UNKNOWN missing elapsed time after abrupt
loss, rather than reporting a lower bound as complete wall time. Resume may use
valid data when time is unknown, but all timing summaries must disclose this.
Use controlled-clock tests with nonzero checkpoint write cost. Completion-last
and preservation/non-overwrite behavior must remain coherent with session files.

After focused regressions pass, run combined local143+added tests. No re-running
old immutable phase tests is needed. Parent does a scoped independent re-review
before any actual physical smoke or collection. Report limitations honestly.
