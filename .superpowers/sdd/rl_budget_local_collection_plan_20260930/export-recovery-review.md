# Independent Export Recovery Review

Reviewed 2026-09-30. Read the brief first, followed by the report and diff. Scope: the new export adapter and bounded supervisor, their tests as existing evidence, and their concrete calls into immutable helpers. This is a static review, not execution approval or a production validation result.

## SPEC Verdict: FAIL

The diagnostic transformation and initial export validation follow the requested output-only boundary, but bounded continuation has a valid completion path it cannot accept (F1). Completed-export acceptance also leaves retained-identity and receipt-provenance gaps (F2, F3). Resolve these in the new helpers before treating the combined repair/continuation workflow as ready. No immutable v1/v2 change is requested.

## QUALITY Verdict: FAIL

The implementation is compact and reuses the original validators, locks, timing reconciliation and cohort-release helper. The recorded 100 passing tests are authentic to the reviewed files, but the supervisor success fixtures bypass the process/attempt transition responsible for F1. Retry and completed-cohort tests also omit F2/F3. These are actionable behavioral gaps, not requests for broader refactoring.

## Findings

### F1 [P1] Preserve coordinator identity when a final resume starts no workers

Location: [export_recovery.py:130](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/export_recovery.py:130), called by [supervise.py:115](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:115) and [supervise.py:19](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:19).

After the last incomplete slot is successfully exported, the supervisor still launches the original runner to publish the cohort comparison/completion. The original [run_wave.py:54](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/run_wave.py:54) skips every completed slot, while [run_wave.py:142](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/run_wave.py:142) records a new coordinator attempt. No worker reservation is created for this attempt. For a fresh coordinator PID, `prove_dead()` therefore finds no matching reservation owner and raises `Coordinator creation identity UNKNOWN`, even after successful cohort finalization. This happens before the supervisor records the launch as exited; its state becomes failed and subsequent invocation is rejected as interrupted.

Action: durably bind the actual launched coordinator's creation identity to its attempt independently of worker reservations, including the Windows redirector case, and use that authenticated identity in the death proof. Retain STOP/live/UNKNOWN rejection and original locks; do not waive missing identity because completion exists. Add a synthetic all-ten-slots-complete finalization case with a genuinely new coordinator record, no new reservations, and the real new-helper `launched`/`inspect` paths.

Existing evidence misses this: [test_supervisor.py:65](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/test_supervisor.py:65) replaces final inspection, its fake wave at line 87 changes only the old process timestamp, and [test_supervisor.py:129](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/test_supervisor.py:129) replaces both inspection and launch for the ten-repair case. Finding established by control flow; no reproducer was run.

### F2 [P2] Bind a completed retry to the receipt's retained token and run ID

Location: [export_recovery.py:216](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/export_recovery.py:216), with [export_recovery.py:175](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/export_recovery.py:175).

The completed branch verifies artifact hashes, then checks only `cohort_phase == "finished"` and `control_steps == 75` on the latest slot reservation. Neither that branch nor `verify_complete()` compares the retained token/run ID with `receipt["token"]`, `receipt["run_id"]` and the authenticated settings. Changing only the terminal reservation's token or run ID, while leaving process identities and phase/count intact, passes these checks. The initial-export identity check at line 240 is bypassed on a completed retry. `cohort_before_sha256` is recorded but does not close this gap.

Action: reconcile the receipt's slot/token/run ID with the appropriate retained terminal record and completion/settings on every completed-export acceptance path. Accommodate legitimate later ownership bookkeeping without accepting a replacement identity. Add focused synthetic completed-retry cases for a changed token and changed run ID. The existing tampering cases at [test_export.py:198](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/test_export.py:198) cover files, not the retained reservation identity.

### F3 [P2] Verify linked export receipts on every supervisor completion path

Location: [supervise.py:23](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:23) and [supervise.py:40](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:40).

When the root completion exists, `inspect()` returns after the original pair comparison and never calls `verify_complete()` for repaired slots. On the incomplete-cohort path, it decides whether to verify an exported slot from receipt-file existence; a missing receipt is rejected only for the primary failed slot. Thus deletion or corruption of a linked receipt can be accepted after cohort completion, and deletion can be accepted for an earlier repaired, non-primary slot before another resume. The original [validate.py:175](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/validate.py:175) intentionally checks only the five normal outputs and does not validate the additional `export_recovery` link. Child-completion hashes do not authenticate a missing or changed receipt target. This loses the required helper/input provenance and separately measured export cost while allowing success or continuation.

Action: identify repaired completions from their `export_recovery` metadata and require receipt/hash/input/retained-identity verification for all such slots before either returning completed or launching another wave. A missing target must fail closed. Keep ordinary original completions valid through the original loader. Add synthetic missing/changed-receipt cases for both a non-primary repaired slot and a completed cohort, without replacing `inspect()`.

## Requirements Supported By Static Inspection

- `tag_diagnostics()` admits only Python positive-infinity floats at the two exact, typed candidate paths. It preserves other JSON values, rejects pre-tagged objects, NaN, negative infinity and nonfinite leaves elsewhere, and records path/value/type. It does not change convergence flags or claim convergence.
- The initial exporter requires 75 transitions and `k == 80`, invokes original checkpoint/boundary/policy checks on raw and transformed traces, runs complete row/sequence validation, checks unchanged summaries/experience, and calls original `load_completed()` on the staged five-output set. It invokes boot for frozen-class availability but never constructs a `BudgetEnv` or calls reset, restore or step itself.
- Canonical-root admission, original cohort/slot locks, STOP checks, creation-identity death checks, source/runtime/settings admission and exact traceback matching are present. The inspected actual carry155 traceback matches that boundary. Historical attempt ABORT is distinct from user STOP.
- Publication preserves raw checkpoint/settings/schema/timing/sessions/logs, archives failed status before replacement, refuses existing conflicting outputs, preserves interrupted staging, calls original `release_worker(..., finished=True)` and publishes canonical completion last. F2 qualifies the claimed completed-retry identity guarantee.
- The supervisor delegates collection to immutable `run_wave.py --resume`, caps repairs at ten distinct canonical slots and wave launches at eleven, holds an exclusive supervisor lock and appends logs. The original runner retains the five-worker/one-thread limits and completed-slot skip. No migration, recollection, model override or new numerical behavior is introduced by the wrapper.
- The receipt defines a nonzero additional export interval and explicit exclusions. Original worker-session timing remains unchanged; it is not an end-to-end runtime total. Parent readout must account separately for export cost and failed/interrupted attempts. F3 must be fixed to ensure that provenance survives supervisor acceptance.

## Evidence And Limits

Read-only checks confirmed that all seven Python files exactly match the supplied diff and their recorded test-evidence hashes. The existing JUnit artifact records 100 tests, zero failures/errors/skips, and 19.804 seconds; its SHA256 matches `evidence.json`. This review did not rerun those tests or the frozen suite.

| Reviewed Artifact | SHA256 |
| --- | --- |
| `export_recovery.py` | `9e8a50a6afdd9aab960c87f31bb441ccb5559555876a7e49373cd31e5f9c5355` |
| `supervise.py` | `51d889872e10e6d5bd2203fb1eb987e1c32aaa7bb183e07a0fad4af6b2e7e895` |
| `export-recovery.diff` | `5558236191409ce3b08ec714917c15e9f8fe3abf1e28894807ac39a1cdb05663` |
| `tests.xml` | `b67a07c99a0dfd089b404ccd7bdd8a97e959491c0f6c2620394b5bc2429e5ecc` |

The six reported actual carry155 input hashes (checkpoint, settings, failed status, timing, schema and stderr) still match. Read-only reservation metadata retains latest counts 75/34/48/51/42, totaling 250, plus the historical one-/two-step records. These are metadata/hash observations, not fresh numerical checkpoint validation. The preservation artifact reports 214 before/after files with no mismatches or missing files; the full 214-file tree was not independently rehashed here.

No test, adapter, supervisor, collector or validator was executed. No checkpoint was deserialized; no physical runtime/model was loaded; no process liveness was probed; no numerical process, export, resume, reset, step, migration or recollection was started. Actual physical/TTT/checkpoint acceptance remains the parent's prior evidence and future execution checks, not a claim of this review. Synthetic fixture subtrees that denied read access were not escalated or opened. No source or result file was modified. The only review output is this ledger file.
