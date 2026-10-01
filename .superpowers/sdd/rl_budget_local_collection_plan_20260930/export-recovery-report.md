# Terminal Export Recovery And Bounded Continuation

Status: implemented and synthetically verified; ready for parent review.
No concrete blocker identified. No actual export, coordinator resume, simulator
or model run/load, environment construction/reset/restore/step, PFO solve,
recollection, migration, installation, commit, or model override was performed.
Only new files under `work/sdmpc_rl_export_recovery_20260930` and this report were
written. The frozen 236-test suite was not run.

## Failure And Parent Evidence

The brief was read first. Its SHA256 is
`33e18ea48fa8b88ac85c3ecd2fd95812c18eb9fe449a8cf68d435eeb325f6864`.
Read-only inspection independently confirmed the persisted failed status and
traceback: immutable v2 `collect.py:155` calls `save(trace.json, trace)`, then
`local_runtime.py:42` rejects the trace in `json.dumps(..., allow_nan=False)`:

```text
ValueError: Out of range float values are not JSON compliant: inf
```

The coordinator failure is exactly
`RuntimeError: Child failed/stopped: carry/sweet_155_w, exit=1`.
Coordinator PID 46348 has creation identity 134351863068826283 in the retained
ownership ledger. Carry155's terminal reservation token is
`e4f2fc1ec62f4be1adfb0a5a1b9d4123`, run ID
`2962784e595d4ee8b1272dda7762dbc6`, worker PID 50024, creation identity
134351863088160545. Old 1-step and 2-step records remain in that ledger.
These are persisted identities, not a claim that this task performed fresh
real-process liveness probes. Execution rechecks all identities under locks.

Parent-provided read-only validation evidence, explicitly distinguished from
the synthetic tests in this task:

| Carry Scenario | Saved Transitions | Positive Infinity Leaves | Parent Checkpoint Validation |
| --- | ---: | ---: | --- |
| sweet_155_w | 75 | 4 | Passed |
| sweet_170_w | 34 | 7 | Passed |
| sweet_170_incident_w | 48 | 0 | Passed |
| sweet_170_skew15_w | 51 | 0 | Passed |
| sweet_190_w | 42 | 5 | Passed |

All **250 saved transitions validate**, so no additional data repair is needed.
Every reported nonfinite trace leaf belongs to the same two candidate
stationarity shapes. Later terminal exports for 170 and 190 require no extra
allowlist or physical repair. Their current prefixes must be resumed normally.

For carry155, the parent also reports successful actual `validate_checkpoint`,
a finite summary, TTT **3210.204341**, zero-NUF requests **0**, terminal inventory
**412.5094381**, fallback count **3**, PFO calls **1**, lower candidate solves
**75**. These are perturbed training results, not canonical evaluation results.
At control_step 14, `selection_source='reference_fallback'`, the candidate has
`converged=false`, and the executed control has `physical_control_valid=true`
and `budget_feasible=true`. The four original +infinity leaves are:

```text
14/candidates/0/rows/2/primal_stationarity
14/candidates/0/rows/3/primal_stationarity
14/candidates/0/rows/5/primal_stationarity
14/candidates/0/stationarity
```

These are rejected-candidate diagnostics. Infinity remains evidence of failed
or unavailable stationarity checking, never zero or convergence proof. The
original failed candidate, fallback selection, convergence flag and raw
numerical checkpoint must remain. Source interpretation follows the brief's
`budget_controller.py:115-184` and frozen `sensitivity_dmpc.py:419,516` references.
No user STOP is known; the old attempt's internal ABORT and historical logs
remain evidence and must not be removed.

## Delivered Code

- `export_recovery.py` (336 lines): narrow adapter using immutable v2 helpers.
- `supervise.py` (200 lines): bounded companion invoking the original runner.
- `conftest.py`, `test_diagnostics.py`, `test_export.py`, `test_supervisor.py`:
  isolated static data, fake process probes/handles, and focused covering tests.
- `run_tests.py`: runs only these new tests with existing dependencies, records
  exact source hashes, and retains JUnit evidence and synthetic fixtures.
- `preservation-evidence.json`: every before/after hash for the 214 preserved files.

The exporter only accepts a canonical v2 behavior/scenario slot. It checks
user STOP at every cohort scope, acquires the original cohort and slot locks,
proves coordinator/launcher/actual-worker death using creation identities,
rejects unknown or incomplete identities, authenticates the original plan,
source/runtime/contract/settings, and matches the exact serialization failure
and its traceback. An exited ledger label alone is insufficient.

It boots only to register frozen checkpoint classes at future execution time;
it never constructs an environment. It requires the saved 75-transition, k=80
checkpoint and retained 75-step cohort record. The original checkpoint,
boundary, complete per-transition/sequence, queue-exposure and policy replay
validators run on raw and exported traces. Settings, transition content and
policy state reconcile; summaries reconcile exactly. The original
`load_completed` additionally validates a staged normal completion set.

Only Python +infinity floats at typed list-index paths matching
`trace[i].candidates[j].stationarity` or
`trace[i].candidates[j].rows[k].primal_stationarity` become
`{"nonfinite_float": "+inf"}`. Each path, original value/type and replacement
is recorded. NaN, negative infinity, nonfinite values at any other trace path,
physical state/action/reward/accounting nonfinites, malformed structures and
pre-tagged objects are rejected. Finite values and all other trace structure
are preserved. The raw checkpoint and training transitions are unchanged.

Publication uses strict JSON, fsynced staging files and exclusive no-overwrite
publication. Existing conflicting output files are refused. Schema and timing
files are reused byte-for-byte. The exact original failed status is archived
under `.export-recovery/original-status.json` before current status changes.
The existing locked `release_worker(..., finished=True)` marks the actual
retained token/run ID/75-step record finished. No replacement reservation is
created. The canonical completion marker is published last.

The five normal `outputs_sha256` entries remain exactly `trace.json`,
`summary.json`, `experience.pt`, `observation_schema.json`, `timing.json`.
An additional `export_recovery` completion field links `export-receipt.json`
and its SHA256, so no unknown output-manifest entry is introduced. The receipt
contains both helper hashes, original checkpoint/settings/status/timing/schema/
session/log/process hashes, source identity, exact known failure, changed paths,
output hashes, experience/summary digests, retained identity, and measured
additional export time. It explicitly disclaims convergence and physical or
training-data changes.

The supervisor has an exclusive `.export-supervisor` lock, durable process and
attempt/status records, append-only stdout/stderr logs, at most 10 repair
attempts, each canonical slot once, and at most 11 original-runner launches.
It calls only `run_wave.py --output <canonical root> --resume` and the focused
export helper. A known terminal failure is exported before the next resume.
Completed slots are skipped by the original runner. Launched handles are
waited on; known live numerical identities continue to be awaited after a
launcher exits. UNKNOWN blocks continuation. Arbitrary failures, missing
checkpoints, user STOP, unproven stopped siblings, ambiguous startup and
duplicate supervisors do not trigger retries. Old ABORT/logs are preserved.
The original maximum five numerical workers and one thread per worker remain;
export is sequential with a dead cohort, not an additional numerical worker.
No solver, policy, collection setting, model or migration behavior is added.

## Verification

Final command, executed from the RL workspace:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_export_recovery_20260930/run_tests.py
```

**100 passed in 19.81 seconds**, exit code 0: 14 diagnostic tests, 60 export
tests, 26 supervisor tests. Final evidence:

- `work/sdmpc_rl_export_recovery_20260930/test-evidence/ecb45969/tests.xml`
- `work/sdmpc_rl_export_recovery_20260930/test-evidence/ecb45969/evidence.json`
- Synthetic fixtures: `work/sdmpc_rl_export_recovery_20260930/_t/ecb45969`

The JUnit file SHA256 is
`b67a07c99a0dfd089b404ccd7bdd8a97e959491c0f6c2620394b5bc2429e5ecc`.
The evidence JSON binds all seven Python files to the final test run and checks
those hashes remain unchanged during execution. Python is 3.12.14, MSC v.1944,
64-bit AMD64, using the existing `.venv-torch` and `.deps-budget` dependencies.

| Requirement | Covering Evidence |
| --- | --- |
| Exact Inf tags, finite identity, no widened shapes | `test_diagnostics.py`, 14 cases |
| Raw and tagged validator/replay/summary equivalence; original loader acceptance | `test_export_original_validators_retained_identity_and_complete_retry` |
| True-terminal-only; nonfinite state/action/reward/TTT and invalid policy | `test_reject_invalid_checkpoint`, 8 cases |
| Missing/changed artifacts, source/runtime; input mutation during validation | `test_missing_artifacts_fail_before_publication`, `test_changed_artifacts_rejected`, `test_source_and_runtime_authentication_fail_closed`, `test_input_change_during_validation_preserved_and_rejected` |
| Raw checkpoint, original failed status, sessions/schema/timing preserved | Successful export and retained-history tests |
| Original locked 75-step cohort identity retained, prior 2-step session retained | `test_retains_previous_sessions_and_75_step_record` |
| No real boot/environment/collector/wave or extra physical step | Static fixture construction; real numerical entrypoints fail immediately if called; all subprocess handles are fake |
| Later carry170, carry190 and local terminal repair | `test_later_slots_use_same_allowlist`, with 7/5/4 diagnostic leaves |
| Conflicts, complete retry, tampering, four publication interruptions, completion last | `test_conflicting_output_never_overwritten`, `test_complete_retry_detects_tampering`, `test_interruption_preserves_stage_and_blocks_retry`, final-publication guard tests |
| Live/UNKNOWN/reused identities, STOP, canonical paths and original locks | Identity/STOP/lock cases in `test_export.py` |
| Exact failure gate, original completed-slot skip, no ABORT changes | `test_inspect_only_exact_failure_and_skip_original_runner` |
| Duplicate supervisor, durable bounds, ten distinct repairs, wrong errors, failed export, no retry | Supervisor rejection/bound cases |
| Launch wait, actual-child draining after launcher exit, STOP after draining, bookkeeping-failure wait | Supervisor process-boundary cases |

The final suite uses the unchanged original physical accounting, boundary,
rows, sequence, replay, summarize, `load_completed`, lock, session-timing and
cohort-release implementations. Source/runtime admission is substituted with
isolated fixture identities, and class-registration boot is a counting stub.
Supervisor successful numerical completion is simulated, not executed. Actual
production pickle deserialization and continuation are not claimed by these
tests; actual-checkpoint validation evidence above comes from the parent.

Development evidence remains available: initial test collection failed before
the adapter existed; fixture-root lookup and Windows path-length setup issues
were then corrected. Intermediate runs reported 47 pass/15 fail, 50 pass/12
fail, and 14 pass/48 fixture errors before 84, 95 and finally 100 passing cases.
The first plain `-m pytest` command found no pytest; the existing dependency
directory required sandbox read escalation. Approved test executions used the
already installed packages. No installation, ACL change, actual run or actual
result write occurred.

## Preservation Hashes

Independent read-only hashing before and after work found **214 files before,
214 after, zero changed, missing or added files** in the v1/v2 source and
cohort-result trees, excluding bytecode caches. Full per-file evidence is in
`work/sdmpc_rl_export_recovery_20260930/preservation-evidence.json`.

Final helper SHA256:

```text
export_recovery.py  9e8a50a6afdd9aab960c87f31bb441ccb5559555876a7e49373cd31e5f9c5355
supervise.py        51d889872e10e6d5bd2203fb1eb987e1c32aaa7bb183e07a0fad4af6b2e7e895
```

Actual carry155 input SHA256, unchanged:

```text
checkpoint.pt           a26d09c9e41b5b55deccccfc1d6b40a0e8dd8d12d474a7f56657bcb3ee92f388
settings.json           6f9a41a30c7abfbb65e4a1292f9468c01e67226140b1bd06c9a29f152aec26a1
status.json             93fe0cc199b7285ccdb73bb28ef1cb6b4d7f42fc23edcaf1b8988bb7346934fa
timing.json             69eae078b2d140b5ea720419a134670a2f05d5992366da0081a86b9c23543be2
observation_schema.json aa16d39ef63b9d37dd51fbf11171210669a7e75a5ad7e88e99eaabd5208920d6
stderr.log              2f125341127ff6347df0a22a46cdf0f67ec842ba3fa20a420873694ceefebe17
```

## Timing And Review Limits

Original carry155 `timing.json` remains KNOWN with worker-session elapsed time
**838.5200439000037 seconds**. Old paired summaries retain their defined locked
worker-session scope. Export work is additional nonzero measured cost; it is
not rewritten into old sessions or silently omitted from an end-to-end claim.
The parent must report export cost separately from old worker-session totals.

`export_elapsed_wall_seconds` measures the successful helper invocation from
entry before cohort/slot locks through class-registration boot, authentication,
validation/replay, serialization, staged-loader verification, publication of
normal outputs and retained-cohort finish bookkeeping. It excludes receipt and
completion publication, lock release, interpreter startup/teardown and
supervisor time. The exact scope is stored in every receipt. This is not a
full end-to-end runtime measurement; supervisor attempts also retain wall-clock
start/end records. Failed or interrupted helper attempts must not be treated
as zero cost or folded into a successful receipt's interval.

Interruption deliberately leaves staging and any already published outputs
intact and blocks automatic retry/continuation. A fully completed retry verifies
receipt/input/output/helper hashes, current export status, retained finish
identity and the original completed loader, without rewriting files. Partial
staging or interrupted supervisor state requires review; this adapter does not
provide cleanup or partial-transaction repair. The publication scheme uses a
final canonical completion marker and fsynced files; it is not an atomic
multi-file filesystem transaction or a tested power-loss guarantee.

Real process death/source/STOP checks are repeated at execution. A newly live
or UNKNOWN identity, new STOP, changed source/artifact, or conflicting output
is a concrete execution blocker by design. None has been asserted away based
on this task's synthetic evidence.

## Parent Handoff

These commands are supplied for review and were **not executed**. From the RL
workspace, export only the already terminal carry155 checkpoint:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_export_recovery_20260930/export_recovery.py --slot carry/sweet_155_w --execute
```

Then continue the same cohort under the bounded companion:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_export_recovery_20260930/supervise.py --execute
```

The supervisor can also perform the initial known carry155 export itself when
started first. Both entrypoints require explicit `--execute`; no output-root
override, migration or model override is offered. Review the carry155 receipt
and its linked completion hash, confirm the original loader's acceptance and
unchanged training/physical summary, then proceed with the bounded continuation.
Do not run carry155 through the collector again or call terminal reset/step.

## Fix Round 1: Accepted F1-F3

Status: F1-F3 implemented and covered; ready for the parent's scoped re-review.
No concrete blocker remains from this fix round. The independent review's
findings were accepted as written. Earlier readiness language above describes
the pre-review implementation; this appendix records the fixes to its gaps.
The preserved `before-export-fix1` files were not changed.

Only the new companion, its focused tests/test runner, and this report changed.
No frozen v1/v2 source or result edit, actual export/resume, simulator or model
load/run, environment reset/step, PFO, migration, recollection, installation,
commit or model override occurred. The parent's 06:12 KST read-only observation
of no RL numerical workers is recorded as parent evidence; it does not replace
the helper's fresh creation-identity checks at future execution.

### Reproductions Before Fixing

Added `test_fix1.py` first, then ran this focused command against the unchanged
pre-fix companion implementations from the RL workspace:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_export_recovery_20260930/run_tests.py -k fix1
```

Result: **9 failed, 1 passed, 100 deselected in 12.01 seconds**, exit code 1.
Evidence: `work/sdmpc_rl_export_recovery_20260930/test-evidence/111670ab/`.

- F1: both direct and Windows-redirector synthetic launches finalized all ten
  completed slots with a new coordinator PID/attempt and no new reservations.
  Real companion `launched()` failed at `prove_dead()` with
  `OSError: Coordinator creation identity UNKNOWN`.
- F2: changing only the retained terminal token or run ID produced no error
  through either `export()`'s completed retry or `verify_complete()`.
  All four expected rejection tests failed.
- F3: missing receipts on non-primary repaired slots and on completed cohorts,
  plus changed receipts on completed cohorts, produced no error through real
  `inspect()`. The changed non-primary receipt case already rejected correctly
  and supplied the one passing control case.

After the fixes, the same focused command reported **10 passed, 100 deselected
in 13.59 seconds**, exit code 0; evidence directory `test-evidence/2d883ab6/`.

### Scoped Changes

F1: added `coordinator_bootstrap.py` (124 lines), solely for identity binding
and delegation to unchanged v2. `supervise.launched()` now records both the
explicit bootstrap command and exact original runner command in its durable
attempt. The bootstrap verifies the live owner/launcher/actual-process
creation identities and parent relationship with immutable v2 `verify_claim`.
It publishes an exclusive `.coordinator.json` binding. The supervisor checks
that binding while the process is live, saves its hash and actual identity to
both attempt and state, and only then acknowledges startup. The bootstrap
requires agreement between both durable records before it calls the unchanged
`run_wave.main()` with the exact original `--output ... --resume` arguments and
original runner `sys.argv`. It does not dispatch numerical workers itself.

After the launched handle exits, the supervisor records the actual runner's
new process/attempt snapshot. `prove_dead()` can authenticate that snapshot
against the acknowledged binding, helper hashes, commands and launcher/actual
creation identities independently of reservations. This supports zero-worker
finalization, retains direct/redirector distinctions, and never infers death
from completion. Missing/changed binding, missing acknowledgement, different
process, live/UNKNOWN identity or STOP still blocks acceptance. Historical
coordinators remain provable through their original retained reservation owners.
Original cohort/slot locks and bounded continuation limits are unchanged.

Bootstrap invocation is explicit in attempt provenance:

```text
<same interpreter> -B -u <companion>/coordinator_bootstrap.py --attempt <canonical cohort>/.export-supervisor/attempts/<id>.json
```

Both the bootstrap and supervisor use bounded startup handshake waits. They do
not select jobs, change collection, add retry classes, or change the original
runner. Bootstrap source is included in `helper_sha256` alongside both existing
helpers. No independent manual bootstrap execution is required or was performed.

F2: `verify_complete()` now checks canonical slot, receipt slot/token/run ID,
original-loader-authenticated settings, and the unique retained terminal record.
That record must remain the current record for the slot, with the same run ID,
75 steps and finished phase. Original ownership bookkeeping changing `exited`
to `completed` remains valid. Replacing the token/run ID or appending a different
current reservation does not. Every completed exporter acceptance uses this
shared check; the former phase/count-only retry check was removed.

F3: `inspect()` validates all existing child completions before either its
completed-cohort return or incomplete-cohort continuation path. A repaired
completion is identified by its `export_recovery` metadata, not receipt-file
existence. Every linked receipt must pass target existence, hash, original
input/output provenance and F2 retained-identity checks. Earlier non-primary
repairs receive the same checks. Ordinary completions without that metadata
continue through the original loader.

### Final Verification

Expanded the regressions to 33 cases and adapted existing fake launches to the
explicit bootstrap handshake. No original test case was removed. Final command:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_export_recovery_20260930/run_tests.py
```

**133 passed in 63.48 seconds**, exit code 0: 14 diagnostic, 60 export,
26 existing supervisor, and 33 fix-round cases. The frozen test suites were not
run. Full evidence is in:

```text
work/sdmpc_rl_export_recovery_20260930/test-evidence/e6fe4f20/tests.xml
work/sdmpc_rl_export_recovery_20260930/test-evidence/e6fe4f20/evidence.json
work/sdmpc_rl_export_recovery_20260930/test-evidence/e6fe4f20/preservation.json
```

Final JUnit SHA256:
`c7ef231339091df07d07856f8778e2c4fa85e95e7a63e51722b7ba3ff7d055ae`.
The evidence JSON pins all nine Python files and verifies that their hashes did
not change during the run. Synthetic fixtures are retained under `_t/e6fe4f20`.

| Accepted Finding | New Covering Evidence |
| --- | --- |
| F1 new coordinator, all ten complete, no new reservation | `test_fix1_f1_new_coordinator_no_reservations`, direct and redirector cases; real `launched`, bootstrap registration, `prove_dead`, `inspect`, original completion loaders and pair comparison; reservation bytes unchanged |
| F1 invalid/missing binding and completion cannot waive identity | Seven `test_fix1_f1_invalid_binding_never_proves_completion` cases: missing/changed binding, missing ACK, different process, live, UNKNOWN, STOP |
| F1 registration and unchanged delegation | Five invalid-registration cases plus `test_fix1_bootstrap_delegates_only_after_durable_ack`; parent/creation/UNKNOWN/command/STOP rejection; ACK requires both durable records; exact original argv and return value verified |
| F2 receipt and retained identity | Four token/run-ID replacement cases through retry and direct verifier; slot/receipt-run-ID/appended-replacement cases; legitimate `exited` to `completed` bookkeeping accepted |
| F3 every supervisor path | Ten real-`inspect` cases: missing/changed receipt, changed raw input, changed retained token, changed retained run ID, each on a non-primary repaired slot and on whole-cohort completion |

All ten-slot completions in these tests are static synthetic artifacts. The
original loaders and paired reconciliation run on them; no physical collection
runs. Actual `Popen` is replaced with fake handles, and the bootstrap delegation
test substitutes only the runner entrypoint with a recording callback. The
F1 launch/final-inspection integration does not replace `launched()` or
`inspect()`. Direct and redirector relationship checks use the immutable
identity verifier against isolated synthetic process probes.

Final executable helper SHA256:

```text
coordinator_bootstrap.py  9aead7b555b2bb192932a718ac29c07c28b1a4c8aa3673b3bbd7466f871622e3
export_recovery.py        f79513942cd892e346dd68a188ecd2221e862b55558170c3a35d90ed8f80a4b3
supervise.py              a119ee8a87b9c280d9a887f253f02308331543f04eb9331cc474eb8ca945e5a9
```

Read-only before/after hashing again found **214 frozen v1/v2 source/result
files before and after, zero changed, missing or added files**. Per-file hashes
are in the final run's `preservation.json`. All seven `before-export-fix1`
snapshot hashes still match the previously reported pre-fix test-source pins.

Remaining limits are unchanged: no actual bootstrap process or numerical
continuation was launched in this round; production validation remains the
parent's next reviewed action. Interrupted or unacknowledged bootstrap startup
fails closed, as do interrupted export staging and unknown process state.
Previously issued receipts from different helper hashes are not silently
re-admitted; no actual pre-fix export exists according to the parent evidence.
Export diagnostics, immutable raw checkpoints, original failed status evidence,
old ABORT/logs, timing scopes and the required separate export-cost accounting
are unchanged by F1-F3. Public exporter/supervisor review commands above remain
the same. Parent scoped re-review precedes any actual export or resume.
