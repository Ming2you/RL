# Scoped Fix1 Re-review

Reviewed 2026-09-30 against `fix1-brief.md`, `implementation-review.md`,
`fix1-report.md`, and `fix1.diff`, in that order. Scope is R1-R5 and concrete
Important breakage in this fix. Dependencies were read only to establish source
semantics. Findings below are static deductions, not executed reproductions.

**SPEC: FAIL.** R1 and R3 are not fully addressed; R2, R4, and R5 are addressed.

**QUALITY: FAIL.** The new configuration comparison rejects valid physical
checkpoints, and completed cohort identity still depends on a later ownership
transaction. Resolve these before physical smoke or collection.

## Important Findings

### F1 [P1] Compare the complete physical configuration, including runtime attributes

**New regression in the R3 fix:**
[validate.py:44](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/validate.py:44).

`settings["physical_config"]` is produced with the frozen runtime's
`to_plain_dict` at
[collect.py:88](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:88).
That serializer deliberately includes non-private runtime-added dataclass
attributes. The new boundary comparison instead uses `plain(env["sim"].cfg)`.
Its imported implementation calls `dataclasses.asdict`, which omits those
attributes. These representations are not equivalent for the actual pinned
configuration.

Concrete evidence: the frozen configuration has
`network.terminal_zero_gradient = true`, which is not a declared `NetworkConfig`
field. The runtime restores that extra attribute onto the dataclass. Relevant
dependency locations are:

- [budget_runtime.py:19](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/budget_runtime.py:19): `plain` uses `asdict`.
- [historical_config.py:233](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_config.py:233): complete serializer; runtime extras are restored at line 273 and `terminal_zero_gradient` is assigned at line 308.
- [runtime.py:16](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_slide_alignment_20260922/runtime.py:16): loads this physical configuration through `restore_historical_config`.
- [factory_config.json:1584](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/outputs/extended_matrix_no_slsqp_20260920/attempt_1/SDMPC6/sweet_190_skew15_w/factory_config.json:1584): pinned configuration contains the extra field.

Consequently, a valid physical prefix cannot resume: `validate_checkpoint`
calls this comparison before restore
([collect.py:43](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:43)).
An uninterrupted run reaches it after all 75 intervals and fails before
completion publication
([collect.py:149](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:149)).
The synthetic fixture uses a plain dictionary for `sim.cfg`, so it cannot expose
this mismatch
([test_collection.py:62](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:62)).

**Action:** use the frozen complete-config serialization semantics in the new
wrapper, without modifying the frozen serializer or weakening the config hash.
Add a focused synthetic boundary case with a nested dataclass and a runtime-added
attribute: an intact boundary must pass, and changing that attribute must fail.
No simulator steps are needed.

### F2 [P2] Persist completed cohort identity without relying on a later scan

**R1 remains incomplete in the new finalization path:**
[collect.py:183](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:183),
[local_runtime.py:187](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:187).

The worker calls `release_worker` before writing `completion.json` at
[collect.py:193](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:193).
`release_worker` therefore records `exited`, even on successful completion.
A standalone worker has no later coordinator release. The same gap exists if
the coordinator is lost after child publication but before polling/releasing it.
Only a subsequent successful ownership transaction discovers the marker and
persists `completed`.

Concrete counterexample: complete a standalone slot, preserve its directory by
moving it away, then start that slot again at its canonical path. No marker now
exists at that path, and the retained ledger entry says `exited`. The new
reservation checks at
[local_runtime.py:144](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:144)
reject only a present completion marker, a `completed` record, or an occupied
reservation, so another fresh episode is admitted. This bypasses the intended
cohort identity retention even though the new root check works.

The supplied regression masks this ordering defect: it explicitly calls
`ensure_cohort_idle` between completion and moving the output
([test_collection.py:603](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:603)).
That extra transaction promotes the record before the assertion.

**Action:** persist a monotone record of the finished slot during worker
finalization, with coherent recovery if completion publication is interrupted.
Keep completion-last publication and validated terminal-checkpoint recovery;
do not require an unrelated future scan to prevent recollection. Add the same
move-and-retry regression without the intervening idle scan, and cover
coordinator loss before its post-exit release. Retain checkpointed-slot identity
as well, so preserving/moving its files cannot authorize a fresh episode.

## R1-R5 Disposition

| Finding | Verdict | File/line evidence and assessment |
| --- | --- | --- |
| R1 | **NOT ADDRESSED** | Canonical roots and masks are enforced at [local_runtime.py:49](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:49); shared reservations/cap at line 141; query-only, fail-closed identity handling at lines 64 and 98; durable intent precedes spawn at [run_wave.py:58](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/run_wave.py:58). Those portions are addressed. Finished-slot retention still fails as described in F2. |
| R2 | **ADDRESSED** | [validate.py:13](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/validate.py:13) checks clock, remaining horizon, normalized anchor and requested/executed memory, including terminal zeros. Lines 94-102 validate prefix history and the frozen raw/projected transform; lines 58-73 bind saved budget/observation boundaries, including zero steps. [collect.py:95](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:95) validates before restore; line 122 checks live observations before step. The complete-75 validator remains in use at `validate.py:129`. F1 independently blocks otherwise-valid physical resumes. |
| R3 | **NOT ADDRESSED** | [frozen_inventory.py:14](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/frozen_inventory.py:14) reuses the actual frozen inventory functions and state methods; required accounting fields are checked at line 35. [validate.py:105](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/validate.py:105) recomputes every row's inventory; lines 46-68 bind simulator state, previous control, area/total accounting and budgets. The accounting definition is correct, but F1 makes the new boundary validation unusable with the pinned physical config. |
| R4 | **ADDRESSED** | [validate.py:215](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/validate.py:215) independently checks each paired slot's scenario, behavior, demand seed, exploration seed and CPU mask before comparison. Direct intact-directory swap cases are present at [test_collection.py:372](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:372). |
| R5 | **ADDRESSED** | [collect.py:62](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:62) starts the session clock; lines 174-193 close the environment, release ownership, sample elapsed, publish session/timing records, then publish completion. [local_runtime.py:202](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:202) preserves missing-session UNKNOWN with null total and an explicit known subtotal. Scope exclusions are stated at line 31. [validate.py:194](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/validate.py:194) reconciles timing; paired UNKNOWN is propagated at line 239. Controlled write-cost and loss cases are present at `test_collection.py:392`, `:535`, `:555`, and `:589`. |

## Evidence and Limits

The reported **189 PASS in 33.54 s, exit 0** is acknowledged and was not rerun.
The fixture's dictionary config and extra idle scan explain why those passes do
not establish F1/F2 closure. No additional Important finding was identified in
this diff; this is not an audit of unrelated pre-existing code.

The six reviewed source files match the destination Git blob IDs in `fix1.diff`.
Diff SHA-256:
`6de8d4b8fa713ae7ef2a52bae100f076c8fb3020a762de11dfaca20dc0a34032`.
The exploration and exploration-test SHA-256 values match those in the fix
report. The diff does not alter the exploration formula, frozen physical code,
screen thresholds, sequential interval loop, or carry-before-local coordinator
ordering. STOP/ABORT checks, fixed one-thread masks, the shared five-reservation
cap, and the no-learning/no-model training-only contract remain present. Actual
physical compatibility and concurrent process behavior were not executed here.

Only this report was written. No tests, collector/simulator/production processes,
model loads, installs, commits, model-setting changes, or changes to completed or
checkpointed outputs were performed. The targeted frozen physical-config read
above inspected configuration metadata; no model or evaluation trajectories or
observations were loaded. No production admission is established by this review.
