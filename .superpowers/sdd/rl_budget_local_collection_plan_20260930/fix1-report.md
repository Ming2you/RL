# Collector Fix Round 1 Report

Status: R1-R5 implemented and synthetically verified; ready for the parent's
independent re-review. NOT production admission. No physical smoke or collection
was performed.

Final combined result: **189 PASS in 33.54 s, exit 0**: the original 121
exploration + 22 collector cases, plus 46 collector regression cases.

## Scope and preservation

Read `fix1-brief.md` first, then `implementation-review.md`, the fixed collection
plan, and the relevant frozen source definitions. Modified only:

- `work/sdmpc_rl_local_20260930/local_runtime.py`
- `work/sdmpc_rl_local_20260930/collect.py`
- `work/sdmpc_rl_local_20260930/validate.py`
- `work/sdmpc_rl_local_20260930/run_wave.py`
- `work/sdmpc_rl_local_20260930/test_collection.py`
- This report.

Used the one permitted focused helper addition:
`work/sdmpc_rl_local_20260930/frozen_inventory.py`. It reuses the exact frozen
accounting functions without importing their solver/simulator dependency graph
or booting a runtime. No generic scheduler or configurable cohort registry was
introduced.

`exploration.py` and `test_exploration.py` remain line-for-line identical to
`implementation-review.diff` (254 and 386 lines respectively). SHA-256:

- exploration.py: `ac75013514583ac83b9965d9e8f57b619b187a5b8511a5e0e801e972301a6feb`
- test_exploration.py: `8d2e8f0cd4370adab22a85a4519f92494efcb35bcb993455cd1d5002817c90a7`

The old multi, value-audit, recovery, and frozen-snapshot source directories have
no tracked diff. No old file, physical contract, exploration formula, threshold,
completed result, or preserved `before-fix1/` script was edited. No install,
commit, model override, OS configuration change, destructive process probe, or
external message was performed.

## Reproduction before implementation fixes

First upgraded only the synthetic fixture/tests. The fixture now has the real
clock and normalized remaining-horizon, action-anchor, previous-requested, and
previous-executed field names. It has physical inventory components, previous
control, area accounting, and checkpoint budget fields. Its raw budget is kept
separate from its capacity-projected request.

With all four collector implementation modules still unchanged, the initial
focused run produced **17 FAIL, 22 deselected in 14.93 s, exit 1**:

| Finding | Before-fix observation |
| --- | --- |
| R1 | A completed carry trajectory could be collected again under another root; expected rejection did not occur. |
| R2 | Eight independent observation-field corruptions and a B_raw corruption all reached the synthetic step sentinel on resume. |
| R3 | Four terminal checkpoint corruptions (inventory, simulator state, previous control, and freeway/urban TTT split) resumed and completed instead of failing. |
| R4 | Direct compare_pairs accepted intact directories swapped by behavior or by scenario. |
| R5 | A controlled two-session clock reported 380 seconds while 402 seconds had elapsed, with checkpoint writes costing 3 seconds, status writes 2 seconds, and close 5 seconds. |

Initial reproduction command, from the RL workspace:

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','-k','fix1','--tb=short']))"
```

The first sandboxed attempt could not read the existing `.deps-budget/pytest`
package and raised `AttributeError: module 'pytest' has no attribute 'main'`.
The authorized test command then ran with elevated filesystem access to the
existing dependencies. Nothing was installed. The test count above is from the
executed regression suite, not that dependency-access failure.

## Changes and evidence

### R1: canonical cohort ownership

Both run entry points enforce the declared root
`results/sdmpc_rl_balanced_goal_20260930/local_budget_v1`, retaining the prescribed
`carry|local/scenario` layout and fixed scenario masks. There is no production
root-override flag. Tests monkeypatch the module constant to a temporary root.

A root-level locked, atomically replaced and fsynced reservation ledger records
the slot, demand/exploration seeds, CPU mask, unique reservation token, owner and
worker PID/creation identity/command. A coordinator persists its launch intent
before Popen, then registers the returned PID. The child claims that same token
once. Standalone workers use the same ledger before boot. Active workers and
pending launches share the maximum-five count; completed slot identities remain
reserved against recollection. Per-output locks remain in place.

Recovery treats absent child identity and unknown/live processes as occupied.
Only proven exit or PID reuse permits automatic reclamation. Windows probes use
query-only OpenProcess/GetProcessTimes/GetExitCodeProcess and CloseHandle, never
signals or termination. A new coordinator checks the ledger before dispatching.

Regression coverage includes alternate roots, standalone overlap, five mixed
active/reserved slots, duplicate claims, durable intent visible inside mocked
Popen, coordinator loss before child registration, registered live/unknown/dead
identities, PID reuse, and completed identity after moving the original output.

### R2: prefix and boundary validation

New validation checks clock, remaining horizon, normalized anchor and both
previous budget memories for every transition and its next observation. It
checks the carried anchor against the preceding execution and reuses the frozen
`budget_controller.residual_budget` for raw and projected requests. Saved zero,
prefix, and terminal observations are bound to the checkpoint's budget boundary;
terminal observations must be entirely zero. Live observations are also checked
before step. Resume validates the checkpoint before restore or any new step.

The old complete-75 validator is unchanged. The revised synthetic fixture now
exercises that actual validator instead of stubbing it. Added cases cover
missing clock schema, zero/one/75-step boundaries, corrupted checkpoint budget
fields, live initial-observation rejection, raw and projected requests, and all
eight clock/memory observation components.

### R3: frozen physical inventory and terminal state

Resolved the accounting source before coding:

- Frozen `run_cell.py:71` exposes `physical_inventory` as `inventory`.
- Frozen `historical_tree/src/controllers/sensitivity_dmpc.py:199` defines
  physical_inventory as total freeway + total urban + off-ramp storage occupancy
  + actual freeway-buffer vehicle counts.
- Frozen `historical_tree/src/models/state.py` supplies the TrafficState methods,
  including effective-lane core occupancy, ramp/origin queues, movement queues,
  storage occupancy, and the fallback legacy-queue semantics.
- Frozen `historical_tree/src/simulation/player_cost_accounting.py:14` counts
  actual upstream/downstream freeway buffers using nominal lanes and segment
  length. Urban arrival/release reservations and duplicate legacy views are not
  added as extra inventory.

The helper loads the unchanged standard-library-only state definitions under a
separate module name and selects the two pure accounting function definitions
with AST, compiling their unchanged bodies. It does not copy an approximate
formula, boot the runtime, import physical solver modules, or advance a plant.
Required state fields are enforced. The physical configuration saved in settings
must match the already frozen configuration hash.

Every trace inventory is recomputed from its plant_state. Checkpoint simulator
state, previous control, physical configuration, warmup accounting, total and
area TTT, last requested/executed budgets and final trace are reconciled before
completion, including terminal resumes where the observation is all zeros.
Synthetic component accounting is 63 vehicles, remains 63 after changing only
excluded duplicate/reservation views, and becomes 54 after the specified lane
and freeway-buffer changes.

Final review also found a zero-step accounting edge case: NaN total/freeway/urban
TTT bypassed the old subtraction comparison. Three focused tests reproduced it
before its repair (**3 FAIL, 65 deselected in 1.62 s**). Boundary validation now
requires finite nonnegative accounting and reconciles area and warmup totals.

### R4: paired-slot authentication

compare_pairs independently compares each loaded slot's scenario, behavior,
demand seed, exploration seed and CPU mask against the prescribed slot before
forming pairs or computing the unchanged screen. Both intact-directory swap
regressions now reject without relying on coordinator preflight.

### R5: durable session timing

Each reservation is also a session identity. Immutable per-session start/end
files are separate from interval checkpoints; checkpoint and reservation history
retain expected session IDs. An abrupt loss before even publishing a session
start therefore cannot silently disappear from a resumed timing history.
Orderly prefix, failure, stop and completion exits record their duration after
checkpoint/data serialization, environment close, status and ownership exit
bookkeeping. Prior session records are preserved.

Precisely measured scope: from the worker's timer sample after acquiring its
output lock and reservation, before publishing the session start, through
boot/reset/restore, steps, every checkpoint and data-output write, close, status
and ownership exit bookkeeping. Final session/timing-ledger and completion-marker
publication, output-lock release, interpreter startup/teardown, coordinator time
and offline analysis are explicitly excluded. Their duration is NOT implied to
be measured. Summary/trace/experience serialization is included.

`timing.json` carries KNOWN/UNKNOWN, elapsed_wall_seconds, known elapsed subtotal,
missing-session IDs, session hashes and this scope. Missing elapsed time yields
`elapsed_wall_seconds: null`; the known subtotal is explicitly only a lower bound.
Valid checkpoint data may resume with UNKNOWN timing. summary.json records the
status/scope and references timing.json; load_completed authenticates both and
returns the merged timing fields. Paired output also discloses UNKNOWN.

Completion hashes the timing ledger, which authenticates the individual session
files, and remains the last publication. Tests cover controlled uninterrupted
and prefix/resumed clocks, failure work after a checkpoint, missing end or entire
session records, loss before session-start publication, timing-file tampering,
completion-last ordering and non-overwrite on a completed-run retry.

## Verification commands and results

R2-R4 focused check after their first implementation: **15 PASS in 14.53 s**.
The first integrated ownership test exposed Windows fsync rejecting a read-only
handle (37 FAIL, 2 PASS); changing only that handle to read/write corrected it.
The then-current collector suite passed **39 cases in 26.87 s**. Extended focused
checks passed **37 cases in 21.01 s**, then **43 in 22.65 s**. The intermediate
combined suite passed **186 in 32.78 s** before adding the three zero-boundary
accounting cases described above.

Commands for these focused checks used the same interpreter/import setup:

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','-k','fix1 and not duplicate and not session_time','--tb=short','--show-capture=no']))"
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','--tb=short','--show-capture=no']))"
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','-k','fix1','--tb=short','--show-capture=no']))"
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','-k','zero_boundary_nonfinite','--tb=short','--show-capture=no']))"
```

Final combined verification, after all implementation and test edits:

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_exploration.py','work/sdmpc_rl_local_20260930/test_collection.py','--tb=short','--show-capture=no']))"
```

**189 passed in 33.54 s, exit 0.** No immutable old-phase test suite was rerun.

## Limits and handoff

All trajectories and checkpoints used for tests are synthetic and temporary.
FakeEnv replaces the physical environment, boot and physical identity/preflight
are stubbed, and the unrelated queue-exposure validator remains stubbed. The
complete-sequence validator and exact inventory accounting are exercised on the
synthetic inputs. No actual collector subprocess, environment boot/reset/step,
simulation, model load/inference, or historical model/trajectory data load ran.

An intent left without a provably exited child remains blocked even if the
coordinator died before actually spawning it. This is deliberately fail-closed;
there is no automatic force-clear or alternate-root escape. Diagnosis of that
ambiguous state is outside this bounded fix. A failure before the initial durable
checkpoint is likewise preserved for diagnosis, not silently restarted.

The new configuration/session metadata is for this not-yet-admitted collector
version; old pre-fix checkpoints are not silently migrated. Physical checkpoint
compatibility, real multiprocess launch timing and actual numerical behavior
remain for the parent's independent re-review and subsequent admitted smoke.
The original training thresholds, outcome interpretation and lack of policy
admission remain unchanged.

## Round 2: F1/F2 Repairs

Read `fix2-brief.md` first, then the complete F1/F2 findings in `fix1-review.md`.
Both findings were reproduced and repaired. This section supersedes the round-1
closure claims for the configuration comparison and moved-slot identity.
Status: ready for the parent's scoped independent re-review, NOT physical
admission. No remaining implementation blocker for these two defects.

Scope stayed within the same six collector modules/helper/tests and this report.
`before-fix2/` was preserved. No exploration, old source, completed output,
physical contract, screen threshold, or collection formula was changed. No new
helper, scheduler, migration mechanism, or production escape flag was added.

### Before-fix reproductions

Added tests before changing the implementation. Result: **5 FAIL, 68 deselected
in 2.97 s, exit 1**.

- F1: a synthetic nested dataclass configuration with the runtime-added
  `network.terminal_zero_gradient = True` was rejected at validate_boundary,
  despite matching its complete serialized configuration and frozen-style hash.
  This fixture requires no environment boot or simulator step.
- F2: immediately moving a slot's directory, without an intervening ownership
  scan, allowed a fresh episode after each of four states: started initialization
  that failed before a checkpoint, checkpointed prefix, standalone completion,
  and child completion with no coordinator post-exit release. Each expected
  rejection failed because the pre-fix code admitted the fresh synthetic run.

Exact reproduction command (RL workspace):

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','-k','fix2','--tb=short','--show-capture=no']))"
```

### F1: complete configuration serialization

The existing frozen helper now also selects and compiles the unchanged
`to_plain_dict` definition from the frozen
`work/sdmpc_matrix_14400_20260912/historical_config.py`. Its original recursive
dataclass, runtime-extra, mapping, sequence and set semantics are retained.
It does not import or initialize the physical runtime. validate_boundary uses
this complete representation when comparing simulator configuration with
settings. The existing configuration hash and physical_config hash check are
unchanged.

The nested-dataclass regression now accepts the intact configuration, rejects a
changed runtime-added attribute, and also rejects changing the copied settings
configuration without changing the pinned hash.

### F2: monotone cohort progress and recovery

The canonical ledger now retains a run ID, `cohort_phase` and checkpoint step
count independently of the process occupancy state. The phases progress from
`started` to `checkpointed` to `finished`; releasing a process does not erase or
downgrade that history. The run ID is assigned with the durable reservation,
before boot or reset, and reused by every resumed session.

A slot with retained history cannot start a fresh episode after its files are
moved. Resume requires its checkpoint and matching run identity. After the
ordinary checkpoint validators pass, the collector also checks the retained
step count before restore or any new step. An older prefix or different run
cannot replace that cohort history. Progress is persisted after each successful
checkpoint write; a checkpoint saved just before interruption of its progress
update can still advance the recorded count on valid resume.

During finalization, the worker itself durably records `finished` after closing
the environment and before sampling session elapsed time. Completion publication
remains last. Neither standalone completion nor a coordinator's later poll is
needed to preserve finished identity. `finished` permits recovery from the same
validated terminal checkpoint when the completion marker was not published;
that recovery executes zero further intervals. Coordinator and standalone
admission use the same existing resume flag and ledger checks.

Removed the intervening ensure_cohort_idle call from the original moved-completion
regression. Two existing helper-level tests now express the stricter semantics:
the worker-cap test fills the released capacity with an unstarted slot, and the
lost-resume-session test explicitly requests resume of its retained checkpoint.
Their original worker-limit, claim-once and UNKNOWN-time checks remain in place.

Added recovery coverage proves:

- Interruption at session-end publication or completion-marker publication leaves
  `finished` and step 75 immediately in the ledger; the same terminal checkpoint
  resumes without calling step, preserving the run ID and prior session files.
- Missing session elapsed time remains UNKNOWN; a measured earlier session
  remains KNOWN when only completion-marker publication was interrupted.
- A relocated prefix cannot authorize fresh collection, but restoring its files
  to the canonical slot permits continuation under the same run ID.
- An earlier checkpoint and an unrelated run ID are rejected before another
  interval, without overwriting the supplied checkpoint.
- A controlled 7-second finished-ledger write is included in measured session
  time, and completion.json is still the final publication.

### Round-2 verification

The initial five reproductions passed after implementation: **5 PASS, 68
deselected in 3.73 s, exit 0**, using the reproduction command above.

Covering collector suite: **79 PASS in 64.58 s, exit 0**. This is the prior 68
collector cases plus 11 round-2 cases, with no old immutable phase suite rerun.

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py','--tb=short','--show-capture=no']))"
```

Final combined suite, after all round-2 implementation/test edits:
**200 PASS in 66.13 s, exit 0** (121 unchanged exploration + 79 collector cases).

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_exploration.py','work/sdmpc_rl_local_20260930/test_collection.py','--tb=short','--show-capture=no']))"
```

Tests used the existing dependencies with the already required filesystem-access
elevation. Exploration file SHA-256 values still match the values recorded
above. The old multi/value-audit/recovery/frozen-snapshot source directories still
have no tracked diff. No actual collector subprocess, physical environment boot,
numerical simulation, model/data load, install, commit, OS change, or model
override was performed. Fixtures and limits remain as documented in round 1.

The additional cohort metadata is not silently migrated onto pre-fix ledgers;
no production collection exists under the reviewed version. Missing checkpoints
after a started attempt and unresolved launch identities remain fail-closed for
diagnosis. Real physical compatibility and concurrent process behavior remain
unexecuted and belong to the parent's re-review and subsequently admitted smoke.
