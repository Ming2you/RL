# Export Fix Round 1: Scoped Re-Review

Reviewed 2026-09-30. Read `export-fix1-brief.md` first, then the prior F1-F3 findings, appended Fix Round 1 report, fix diff and affected implementations/tests. Scope is F1-F3 and breakage introduced by this diff. Unrelated immutable code was not reopened for review.

## SPEC Verdict: PASS

F1, F2 and F3 are addressed within the reviewed scope. Process UNKNOWN continues to fail closed, and metadata-linked repaired completions require receipt authentication before the supervisor's continuation or completed return. No outstanding actionable SPEC finding was identified in this fix diff.

## QUALITY Verdict: PASS

The fixes are confined to the companion and focused tests. The small bootstrap binds process identity and delegates to the unchanged runner; retained-identity reconciliation is shared through `verify_complete()`, and receipt validation precedes both inspection branches. Existing evidence now exercises the previously missing integration paths. No new actionable regression was found by static review.

## F1: Addressed

The supervisor records the bootstrap command, original runner command, owner identity, helper hashes and source in its durable attempt at [supervise.py:95](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:95). Registration uses immutable `verify_claim()` to validate live creation identities and direct/Windows-redirector parent relationships at [coordinator_bootstrap.py:45](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/coordinator_bootstrap.py:45). The parent independently verifies the live binding before writing its hash and actual identity to both durable records at [supervise.py:118](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:118).

Delegation waits for agreement between the attempt and supervisor state, then calls unchanged `run_wave.main()` with the original argv at [coordinator_bootstrap.py:106](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/coordinator_bootstrap.py:106). After launch exit, the new coordinator process/attempt is bound to that identity. [coordinator_bootstrap.py:67](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/coordinator_bootstrap.py:67) validates the stored binding, acknowledgement, hashes and runner process, then requires both launcher and actual coordinator death independently of worker reservations. [export_recovery.py:132](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/export_recovery.py:132) uses this proof while retaining historical reservation-owner admission. Completion-file existence does not substitute for process proof.

Covering evidence: [test_fix1.py:48](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/test_fix1.py:48) exercises real new-helper `launched()`/`inspect()` with a new coordinator, ten completed slots, no new reservations and both launch modes. Cases at lines 101, 227 and 247 cover invalid/missing binding, missing acknowledgement, live/UNKNOWN/STOP, registration identity and delegation only after durable acknowledgement. Handles and process probes remain synthetic.

## F2: Addressed

[export_recovery.py:196](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/export_recovery.py:196) now reconciles original-loader-authenticated settings with canonical slot, receipt slot/run ID, the unique receipt token and the latest retained slot record. That record must have the same run ID, 75 steps and finished phase. Completed exporter retry uses this shared verifier at line 228; supervisor acceptance also calls it. Legitimate `exited` to `completed` bookkeeping remains admissible without allowing a replacement token/run ID.

Covering evidence: [test_fix1.py:128](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/test_fix1.py:128) covers token/run-ID changes through retry and direct verification; cases at lines 142 and 149 cover legitimate bookkeeping, receipt slot/run-ID mismatch and an appended replacement reservation.

## F3: Addressed

[supervise.py:23](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/supervise.py:23) validates every existing child completion and identifies repairs by `export_recovery` metadata. It authenticates each linked receipt, original inputs/outputs and retained identity before either the root-completed branch or incomplete-cohort branch. Missing receipt targets therefore fail instead of being treated as ordinary completions. Original completions without repair metadata still use the original loader. Fresh exports are additionally checked at line 205 before the next wave.

Covering evidence: [test_fix1.py:187](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_export_recovery_20260930/test_fix1.py:187) uses real `inspect()` for missing/changed receipts, changed raw input, replaced token and replaced run ID, both for an earlier non-primary repair and for whole-cohort completion.

## Preservation And Evidence

- All seven preserved pre-fix files match the exact source hashes from the prior reviewed 100-test evidence. Reconstructing the seven changed/new files from the supplied diff matches their current contents. All nine current Python files match the final evidence hashes.
- Existing JUnit records confirm the focused pre-fix result of nine failures/one pass, followed by ten focused passes. Final evidence records 133 tests, zero failures/errors/skips and 63.472 seconds. The final JUnit SHA256 matches its evidence manifest. These are inspected prior results, not tests rerun during this review.
- Independently rehashed all 214 files enumerated in the final preservation record: every current hash matches its recorded before/after hash. This verifies the listed frozen v1/v2 source and result files, including preserved raw checkpoints; it does not independently validate checkpoint numerics or enumerate unlisted additions.
- The diff leaves diagnostic tagging, raw numerical infinities, physical/training validation, terminal checks, output publication, historical evidence and timing scopes unchanged. Helper provenance now includes the bootstrap. Original worker limits, repair bounds, STOP/lock gating and separate export-cost accounting remain in place. No convergence or end-to-end runtime claim is added.

| Artifact | SHA256 |
| --- | --- |
| `export-fix1.diff` | `8f7db0157911b28ded5b1e63229a5465c60cf7fd5c3a5d7868523a2f93d00feb` |
| `coordinator_bootstrap.py` | `9aead7b555b2bb192932a718ac29c07c28b1a4c8aa3673b3bbd7466f871622e3` |
| `export_recovery.py` | `f79513942cd892e346dd68a188ecd2221e862b55558170c3a35d90ed8f80a4b3` |
| `supervise.py` | `a119ee8a87b9c280d9a887f253f02308331543f04eb9331cc474eb8ca945e5a9` |
| Final `tests.xml` | `c7ef231339091df07d07856f8778e2c4fa85e95e7a63e51722b7ba3ff7d055ae` |

## Limits

This PASS is a scoped static review supported by existing synthetic evidence, not a claim that production export/continuation succeeded. Real cross-process startup timing, Windows filesystem/process behavior and production checkpoint acceptance remain unexecuted here. Missing/unknown identities, incomplete acknowledgement or interrupted staging must continue to block automatic continuation at execution.

No reported tests or project entrypoints were run; no checkpoint was deserialized; no production bootstrap, physical/model load, export, resume or numerical process was started. No code/results were modified and no commit was made. The only file written by this re-review is this ledger review.
