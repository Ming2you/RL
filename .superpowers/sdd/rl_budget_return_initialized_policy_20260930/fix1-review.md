# Task 1 Fix 1 Independent Re-review

Date: 2026-09-30. Scope: R1/P2 from `task-1-review.md` and direct regressions of `fix1.diff` only. Reviewed `fix1-brief.md`, the scoped diff, the appended implementation report, affected code, and retained test evidence. No broad new audit was performed.

**Addressed: R1. Open: none. SPEC verdict: PASS. QUALITY verdict: PASS.** These verdicts resolve the original review's sole blocker for the exact revised source identified below, carrying forward its unaffected assessment. They do not establish actual training success or canonical traffic acceptance.

## R1 Resolution

The STOP handler at [run.py:93](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/run.py:93) now calls `persist()` only when `learner.phase != "done"`. During resumed final publication, it therefore preserves the existing durable checkpoint and `latest.json`, including their binding to the immutable `model_final.pt`. It no longer creates a differently hashed checkpoint solely because the current process/session identity has changed.

The focused regression at [test_return_init.py:392](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/test_return_init.py:392) covers the accepted failure precisely: three distinct PID/creation/session identities, a previously completed learner checkpoint, real hard-link publication, STOP immediately after linking, and a second resume after clearing STOP. It forbids all resumed learner updates and checks unchanged pointer bytes, candidate/checkpoint hashes, exactly one checkpoint, equal learner state and counters, valid completion output hashes, and distinct stopped/completed session records.

## Direct Regression Check

- A newly reached `done` phase is checkpointed by the phase-change branch at [run.py:65](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/run.py:65) before the next STOP check. A resumed `done` phase comes from the existing authenticated checkpoint. Skipping another checkpoint in the STOP handler therefore does not discard uncheckpointed completed updates at the reachable STOP checks.
- STOP before or after final linking leaves the same durable `done` checkpoint available for subsequent publication. Repeated publication attempts retain the existing candidate hash check and completed-run rejection.
- Non-`done` phases, including `gate_failed`, retain their previous STOP checkpoint behavior. The gate-failure path publishes no candidate, so it does not acquire the candidate/pointer conflict addressed by R1.
- Current process identity remains recorded in session start/end files; the checkpoint retains its original creator identity. Source/data checks, update counts, optimizer/RNG state, continuation declaration, and the scientific specification are unaffected by the change.

No direct regression or additional open finding was identified within this scope.

## Retained Evidence

The supplied evidence is authoritative; no successful test suite, focused test, validator, or optimizer run was repeated during this re-review. Evidence JSON hashes and their bound JUnit hashes were checked read-only.

| Evidence run | Verified result |
| --- | --- |
| `a68aaa12` | Original runner hash; the new regression fails with `FileExistsError: Existing final candidate differs`. |
| `c2e7d593` | Fixed runner hash; the same regression passes. |
| `1dba6b73` | Final fixed source; 39 passed, zero failures/errors/skips, including the R1 regression. |

The final evidence records zero actual-data optimizer updates and zero environment runs. The regression source confirms that its completed learner is prepared synthetically before resumed updates are forbidden.

Both diff hunks were reconciled against `before-fix1`; applying them produces the current complete `run.py` and `test_return_init.py` by line content. The other five original Python files match the snapshot byte-for-byte. Both reused helper hashes are unchanged. All nine current source/helper hashes match `test-evidence/1dba6b73/evidence.json`; its SPEC fields and digest match the original evidence.

| Reviewed identity | SHA256 |
| --- | --- |
| `fix1-brief.md` | `4cc9fa2572e40655a59c1d29aa863ad638311c258e044005ec4838e5d6812dfe` |
| `fix1.diff` | `6c62691ced33bf4f9fd4a2c62053144c1b2496efdd243e6354cf042284d5395c` |
| Appended `task-1-report.md` | `92970d7360e0f914612ce664e721ca8b80f90f3b31bf833ba256480c40b33a8b` |
| Fixed `run.py` | `f1019cc867c074810493024e40716cf801cde4987c172dbbf85bde1c966ef5d6` |
| Fixed `test_return_init.py` | `13a1689aeaa708c1538b29f9555bd6cd209f98ef3d959a7570391c2b671ec016` |
| Final `1dba6b73/evidence.json` | `b065c410968c2ac80c36bb914ad67bad7149579044a2ad1f9fafbdeed2bf3484` |
| Final `1dba6b73/tests.xml` | `8f7d36ecdb50078c2b32844aa152bd3319803b28af6123cfb8fca265e3187e92` |
| Unchanged serialized SPEC | `933e29c1b316530ceff1605d527532e482a658c12fa6c13fd24a93074ef6593e` |

Evidence paths are under `work/sdmpc_rl_return_init_20260930/test-evidence/`. The complete exact source identity remains the nine-entry `source_sha256` object in the final evidence.

Only `fix1-review.md` was written during this re-review. No code edits, new tests/repros, actual training, simulation, environment execution, approval receipt, or dispatch occurred. The actual `return_init_v1` output directory remains absent. Parent retains ownership of any later exact-hash approval receipt and actual-run dispatch.
