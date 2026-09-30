# Bounded Terminal-Quota SPEC + QUALITY Review

## Findings and Verdicts

No actionable bugs identified in the supplied script/tests and necessary dependency contracts. No priority/path/line defect entries are warranted.

- **SPEC: PASS.** The implementation matches the bounded terminal-quota brief.
- **QUALITY: PASS.** No blocking correctness, preservation, or output-lifecycle defect found by static review.
- **Review disposition: admitted for the parent's specified production ablation.** This is implementation admission only, not a result for the hypothesis or admission of a policy.

## Scope and Evidence

Reviewed `work/sdmpc_rl_value_audit_20260930/terminal_quota_brief.md`, the implementation report, and the supplied `terminal-quota-review.diff`. The diff contains only the new 425-line script and 646-line test file; an in-memory line comparison confirmed both current files exactly match its added contents.

Dependency inspection was limited to the reused loading, replay, TD target/update, metrics, fingerprint/RNG, provenance, locking, and JSON-writer contracts in `terminal_fit.py`, recovery `common.py` / `critic_diagnostic.py`, multi `td3.py` / `run_budget.py` / `budget_runtime.py`, and the projection audit's manifest/completion producer. No broader audit was performed.

Verified SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| `terminal_quota.py` | `837c237e91872bd7fed53929dcbbf536aa60dbe5f8ce54ccb4404bfe54c35c3e` |
| `test_terminal_quota.py` | `e397fbef28cc3bf3cdc2eeaa650e5dba188ed60a70566483cc1b9bdeeb6e8353` |
| Supplied review diff | `4929211fa8d0bb34f304351bcf38e39b3522946606ae7abd7cda8641e9130437` |
| Brief | `dcff1a3f26c6e99415837f989eea1c6919af44441a20119ba9fd4b0f3dc0566b` |

The script and test hashes agree with the implementation report.

## SPEC Checks

All source links below refer to the reviewed workspace files.

| Requirement | Assessment and evidence |
| --- | --- |
| Paired sampling; exactly 20% from each scenario | PASS. [terminal_quota.py:113](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:113) draws eight with-replacement indices from each 150-row group, copies them, and replaces only column zero with row 74 or 149. Seven paired draws and incidental terminals remain. Replacement has its own local RNG. |
| Paired noise and isolated evaluation | PASS. [terminal_quota.py:213](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:213) creates separate local sampling/training/evaluation streams. [terminal_quota.py:262](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:262) passes the same noise tensor to both arms and checks it was not mutated. Evaluation uses one fixed tensor for all logs and arms, reproducibly across seeds. |
| Full independent initialization; frozen old actor | PASS. [terminal_quota.py:133](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:133) deep-copies the complete original learner, checks its serialized identity, retains critic Adam and critic targets, and loads the final online actor into the frozen actor target. Both actors have gradients disabled and are in evaluation mode. |
| Unchanged Bellman target and optimizer | PASS. [terminal_quota.py:202](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:202) checks the original settings. [critic_diagnostic.py:111](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/critic_diagnostic.py:111) uses reward plus gamma=1 continuation, masked by true termination, with the existing clipped smoothing and twin-target minimum. Its critic step uses the copied Adam and tau=.005 every second diagnostic update; no actor step occurs. |
| Replay and old-state immutability | PASS. [terminal_quota.py:208](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:208) invokes both required validators and works on copied tensors. Final arm checks cover actor/target/actor optimizer, replay and unchanged learner controls. [terminal_quota.py:349](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:349) preserves global RNG across authentication, loading and fitting; original learner/model fingerprints are rechecked after final provenance verification. |
| Fixed production budget and worker count | PASS. [terminal_quota.py:28](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:28) fixes seeds 6529/6530, 3750 updates and all five required log points. Seeds and arms execute sequentially; the existing helper enforces one numerical worker. [terminal_quota.py:414](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:414) accepts only output plus standard help, with no abbreviation or budget/model override. Short budgets are internal test arguments. |
| Authentication and hashes | PASS. [terminal_quota.py:56](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:56) requires a completed projection audit, the exact base hash, exact 73-path input manifest, matching source hashes and source/runtime contract. It hashes the exact completion bytes parsed and checks them again. Imported diagnostic sources, checkpoint/runtime identities and completion/input identities are captured before load, after load and after fitting. The projection producer uses the same manifest and completion structure. |
| STOP and non-overwrite | PASS. [terminal_quota.py:314](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:314) rejects nonempty output before acquisition and rechecks inside the exclusive lock, allowing only its lock file. STOP covers repository, goal root and output before load/write, inside the lock, between steps, before Adam, and before completion. STOP unwinds without subsequent diagnostic writes. |
| Nonfinite and failure handling | PASS. [terminal_quota.py:150](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:150) checks parameters, gradients and Adam state around every step; the pre-step hook rejects missing/nonfinite gradients before Adam mutation. The reused helper checks loss. JSON is validated with `allow_nan=False` before the existing atomic writer. [terminal_quota.py:400](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:400) publishes completion last, after verification; stopped or failed work does not reach it. No learned-weight export exists. |
| Complete diagnostics and accounting | PASS. [terminal_quota.py:239](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:239) records the original scenario/time-slice/episode diagnostics, all ten exact terminal targets with Q1/Q2/min-Q errors, sample and actual terminal counts, and index/noise hashes. Metadata retains initial/final frozen and critic fingerprints, original state identities, update counts, elapsed time and PID. |
| Faithful predeclared criterion | PASS. [terminal_quota.py:302](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_quota.py:302) requires quota final min-Q MAE strictly lower in each seed, pooled MAE at most half of uniform, and each quota final MAE strictly below 5.701837813854217. Averaging the two seed MAEs correctly pools their equal ten-terminal sets. Full curves and both critics remain available. The criterion is evaluated only after the fixed work finishes; the caveat preserves the no-deployment/no-generalization interpretation and instruction against extending an unsuccessful budget. |

## QUALITY and Verification Limits

The supplied tests cover pairing, independent RNGs, common noise, logging-independent results, initial copy/optimizer ownership, Bellman masking, target cadence, diagnostic detail, original/frozen-state guards, STOP/lock races, nonempty output, source/input/completion changes, nonfinite failure paths, JSON validation, criterion outcomes and the CLI boundary. The synthetic fixtures stub production loading, snapshot access, physical boot and weight serialization. The real manifest verifier is exercised against temporary synthetic files.

The implementation report records **64 passed in 16.38s, exit code 0**. This is supplied execution evidence, not a test result produced by this reviewer. As requested, no tests were rerun, no model or production diagnostic was executed, no simulator was run, and no model override was requested or applied. Independent checks here were read-only source/contract inspection, file hashing and exact comparison with the supplied diff.

The documented lifecycle is consistent: `completion.json` is authoritative; `status.json` remains `verifying` after success, and a STOP may leave earlier partial JSON. A pre-existing lock-only output is intentionally refused. These behaviors do not contradict the brief.

Actual production authentication, full-budget execution and the empirical criterion remain for the parent. Static admission does not assert that those runs have succeeded. The only file written by this review is this review document.
