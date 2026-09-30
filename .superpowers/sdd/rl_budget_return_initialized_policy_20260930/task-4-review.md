# Task 4 Independent Review

Date: 2026-09-30

**SPEC: PASS**

**QUALITY: APPROVED**

No actionable findings by severity (P0/P1/P2/P3) were identified in the scoped
canonical-evaluation adapters. No changes are requested before the proposed
GitHub handoff. This review is not a physical-evaluation admission receipt and
does not authorize or initiate an evaluation.

## Scope and method

The binding reference was `task-4-brief.md`; `task-4-report.md` and the complete
`task-4.diff` were reviewed against the ten current Python files under
`work/sdmpc_rl_return_eval_20260930/`. Every diff section matches its current
source file, and every file SHA256 matches the final retained test source map.
Directly reused definitions were traced only where needed to resolve Task 4's
inference, physical/accounting, checkpoint, restore, locking, and timing contracts.
The MC results note was read for the fixed-policy context.

This was static review and read-only inspection of retained evidence. No suite,
authentication command, optimizer, environment, simulation, or evaluation was
run. No actual physical checkpoint or model was deserialized by this review.
No code, receipt, commit, or push was produced. The sole authored file is this
review. Paths and line numbers below are relative to the RL repository root.

## Contract assessment

| Area | Assessment and concrete references |
| --- | --- |
| Exact model | `policy.py:20-39,42-77` checks the pinned MC model/completion, settings/output/source/data identities, MC format and spec, done phase, 250 MC updates, ancestor counters, continuation, unchanged-policy flags, and actor tensor identity. These references are under `work/sdmpc_rl_return_eval_20260930/`. The fixed actor implementation is resolved through `work/sdmpc_rl_return_wave_20260930/actor.py:7-43`: strict float32 keys/shapes/bounds, meta construction, no gradients, 2367/64/64/2 ReLU/tanh inference. |
| Inference isolation | `work/sdmpc_rl_return_eval_20260930/support.py:40-54,89-107` authenticates reused code and enforces the import boundary. `policy.py:4-5` extracts only the actor definition. The inherited boundary at `work/sdmpc_rl_return_wave_20260930/wave_support.py:184-193` excludes generic learner/data/TD3/runtime collisions. The legacy loader's function-local TD3 import (`work/sdmpc_rl_multi_20260929/compare_runs.py:115`) stays in the isolated child launched by Task 4 `preflight.py:9-16`; `baselines.py:36-60` blocks physical calls, tensor loads, and optimizer construction there. |
| Canonical baselines and profiles | `work/sdmpc_rl_return_eval_20260930/baselines.py:8-33,49-62` uses the actual completed carry centers and compares scenario, null seed/model, profile, schema, source/runtime/environment contracts, zero actions, carry anchors, and projected requests. The reused `work/sdmpc_rl_multi_20260929/compare_runs.py:21-151` reconciles warmup plus all 75 intervals, area totals, timeline/terminal flags, no learning/Q, initial/recovery PFO, reference counts, one lower candidate, and decision wall/CPU sums. Task 4 `checks.py:52-87` adds executed-control feasibility, budgets, inventory, timing, and queue evidence. Current baseline evidence is hash-bound by preflight; training-wave returns are not substituted as baselines. |
| Canonical physical path | `work/sdmpc_rl_return_eval_20260930/worker.py:94-129,142-162` constructs the unchanged snapshot-backed `BudgetEnv` with `training_seed=None` and physical guard, checks schema/config/options/profile and matched warmup, and applies actor actions without exploration, learning, Q selection, or an added preview/search. `checks.py:13-49,90-108` explicitly adapts canonical metadata rather than impersonating training data. The float64 residual projection, signed NP, bounded NUF, and previous-executed anchor remain in `work/sdmpc_rl_multi_20260929/budget_controller.py:11-22,34-57,187-205`. |
| Transition/accounting validation | Task 4 `checks.py:90-108` binds the canonical validator into the unchanged checkpoint check. The reused definitions at `work/sdmpc_rl_nuf_retention_20260930/validate.py:15-131` verify observation memory, transition chaining, reward/cumulative/area accounting, true terminals, carry projection, physical state boundary, no Q/learning, PFO counts, and timing sums. The complete-sequence helper at `work/sdmpc_rl_value_audit_20260930/projection_audit.py:62-99` has training wording in its error text but imposes no training-only metadata requirement. |
| Admission and runtime | `work/sdmpc_rl_return_eval_20260930/preflight.py:35-64` requires exact source/spec/preflight/model/physical-contract hashes and named parent/reviewer admission. `worker.py:58-98` performs admission before physical startup and retains the kernel single-writer lock and process identity. `support.py:99-107` preserves all four STOP scopes. Thread limits are inherited from `work/sdmpc_rl_return_wave_20260930/wave_support.py:36-50`. There is no dispatcher or dependency installer. |
| Resume and finalization | `work/sdmpc_rl_return_eval_20260930/worker.py:66-83,106-118,169-204` refuses uncertain reset/step operations and contradictory partial exports, validates the checkpoint before restore, and checks the exact returned observation. Before a finalization marker exists, a retained terminal checkpoint follows the inherited k=80 restore branch without plant/PFO/reference reconstruction; afterward, `finalize` publishes from retained exports without physical boot or simulator deserialization. Retained outputs are compared, not silently replaced (`worker.py:18-51`). Completed duplicates are refused by the reused `work/sdmpc_rl_return_wave_20260930/worker.py:12-33`; immutable checkpoints and the atomic pointer remain at `wave_support.py:120-148`. |
| Honest computation accounting | `work/sdmpc_rl_return_eval_20260930/checks.py:114-143` reports actor, forecast, observation, PFO, reference, lower solve, guard, decision, plant, and session scopes. `support.py:79-84,129-141` explicitly bounds session timing and preserves unknown interrupted durations. `work/sdmpc_rl_return_wave_20260930/restore_accounting.py:27-53,56-134` measures the entire restore call, preserves inherited decision timing, and labels reference reconstruction counts as source-derived. Restore cost is already included once in session totals. |
| Readout and acceptance | `work/sdmpc_rl_return_eval_20260930/readout.py:11-48,51-117` checks completed output/checkpoint/diagnostic/timing bindings without loading physical checkpoints, retains ordinary absent/partial/invalid scenario outcomes, computes matched raw/percent and interval-window differences, and requires all five individual improvements above the specified strict threshold. It distinguishes incomplete evidence from a complete negative result. Passing means only `eligible_for_separate_reproduction`; `goal_achieved` and automatic reproduction dispatch remain false. |

## Retained evidence

The following evidence was inspected, not regenerated:

- `work/sdmpc_rl_return_eval_20260930/evidence/0d0f12c3/evidence.json`
  records 49 tests, zero failures/errors/skips, 55.567 seconds, zero actual
  physical steps and optimizer constructions/updates. Its two retained SHA maps
  each contain 1,473 entries and have no differences. This checks the retained
  preservation record; it is not a new authentication of all 1,473 input files.
- The matching `preflight.json` records 375/375 bitexact retained Task 2 actor
  actions and the five actual canonical carry baselines:
  3103.0110715680044, 3935.903236508048, 5546.224352256691,
  4250.876599300032, and 6604.2970168093225. Each baseline has 75 controlled
  intervals, 14,400 seconds, its own canonical profile hash, and reconciled
  warmup/accounting evidence.
- Source inspection of `test_eval.py:50-62,88-210,213-309` covers the explicit
  canonical-versus-old-contract distinction, admission, STOP/lock, operation
  orphan refusal, resume and completed duplicates, both terminal publication
  paths, retained-output refusal, partial/full readout, all-five acceptance,
  threshold boundaries, import collisions, and identity mismatch cases.
- Final source-map digest:
  `9a9de385546732cd158295fdd03486cc928864930d2671b8b9e5f45c077314b2`.
- Spec digest:
  `c4d3f93ba94da908ffa3f79fbb2dfc1c22f0e608730078679ca37b50e96c10c2`.
- The three retained artifact SHA256 values were independently checked against
  the report: preflight
  `14335232c3358611c8605c9aa2db65057803d845d9646e1c326dbe347621a366`;
  evidence
  `df4cc24aacea783bad11872319b9673c8a0e501ecb23a7be50fce021f3ba20b3`;
  tests XML
  `c58e91c723e3317427ecc8c1e404699218b6be24f5664b86a058177fe09f3f7e`.

## Limits and handoff concerns

`results/sdmpc_rl_balanced_goal_20260930/return_canonical_v1` is absent, as
confirmed by a directory existence check. Canonical candidate TTT, all-five
acceptance, and actual evaluator runtime remain unknown. Passing synthetic
tests and retained actor parity do not establish physical performance.

No new actual restore or canonical physical execution was attempted. Confidence
in the wrapper comes from static contract resolution and retained synthetic
tests, with the unchanged restore implementation and prior evidence reused as
required by the brief. Queue exposure and peak inventory describe retained
interval endpoints, not exact substep exposure or extrema. Session timing has
the explicit exclusions above; parallel worker sums are not elapsed wall time.

There is no review blocker for the requested code/evidence handoff. A future
physical launch still requires the separate parent-authored exact admission
receipt and parent worker-budget control. Neither this review nor the retained
test result is evidence of an achieved performance goal or stochastic
generalization.
