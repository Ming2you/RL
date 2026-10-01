# Task 1 Independent SPEC + QUALITY Review

Date: 2026-09-30. Scope: the Task 1 brief, implementation report, supplied diff, and corresponding offline data/learner/runner implementation and evidence.

**SPEC verdict: FAIL. QUALITY verdict: FAIL.** One P2 finding blocks approval because the required STOP/resume behavior can strand the final candidate. Do not issue an approval receipt for this source revision.

## Ordered Findings

### 1. [P2] STOP during resumed final publication makes the next resume fail

Location: [run.py:93](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/run.py:93), specifically the unconditional `persist()` in the STOP handler. Related code: [process-bearing checkpoint payload at line 47](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/run.py:47) and [candidate hash comparison/publication at line 80](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/run.py:80).

A valid interrupted run can already have a `done` checkpoint A, with all 1000/250/10 updates complete, but no completion record. On resume in a new process, the runner restores A and skips the update loop. It links `model_final.pt` to A, then checks STOP at line 86. If STOP is present at that check, the handler creates checkpoint B and advances `latest.json`. B contains the new session/process identity, so its bytes/hash differ from A even though the learner state is unchanged. `model_final.pt` remains linked to A.

After STOP is cleared, the next `--resume` restores B, encounters the existing A candidate, and raises `FileExistsError: Existing final candidate differs` at line 82. Further resumes repeat the failure. This is a valid finalization interruption, with no source/data mutation and no extra optimizer update. Recovery now requires changing/removing an existing artifact or changing the frozen source, rather than the promised in-place resume. The candidate weights are not shown to be wrong; the final candidate and checkpoint pointer become inconsistent.

Required before PASS: make final publication and STOP recovery preserve a consistent immutable candidate/checkpoint binding, and add coverage for STOP after linking during resumption of a `done` checkpoint with a different process identity. This review applies no fix.

New targeted evidence: executed the exact `execute()` function AST extracted from the reviewed `run.py`, with an in-memory filesystem, a fixed completed learner, distinct session identities, and an injected STOP at the post-link check. Persistence used standard-library pickle as a stand-in for checkpoint serialization; no Torch/NTFS integration claim is made by this repro. The production payload explicitly includes the changing process identity, which establishes the hash divergence independently of the serializer. Observed output:

```text
first_resume_outcome: stopped
completion_exists: False
candidate_equals_latest: False
second_resume: FileExistsError: Existing final candidate differs
optimizer_updates: 0; filesystem_writes: 0; environment_runs: 0
```

The existing [runner STOP test at line 365](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_init_20260930/test_return_init.py:365) stops at Phi update 2. The seven deterministic learner-resume positions exercise learner state restoration, not this final publication interruption. Their successful results remain valid and do not cover this failure.

## Specification Assessment

- Data/provenance: root completion/comparison pins, isolated existing validators and carry-reference authentication, training-only profile checks, and the ten selected trajectories agree with the brief. Only 375 carry plus 375 NUF-local transitions enter the loader, 150 per scenario. Warmup is excluded from returns; sequential observations, rewards, terminals, and execution-to-next-anchor continuity are checked. Current and next anchors come from float64 traces, and terminal next anchors are zero.
- Action/continuation semantics: one memoryless state-conditioned actor and one shared Phi with twin residual critics. Scenario labels only select batches/diagnostic groups. Replay, actor, and carry targets use float32 increments followed by float64 physical request projection and float32 normalized network inputs. NP remains signed, NUF is clamped to capacity 6000, and zero action carries the previous executed anchor. Execution is not substituted for the critic action.
- Learner: the specified width-64/two-ReLU models, zero final heads, seed 7200, independent saved RNG state, 1000 Phi updates, 250 critic updates, and 10 gated actor updates are implemented. Every optimizer batch contains exactly eight rows per scenario, with the prescribed carry/local and forced-terminal quotas. Carry labels are trajectory returns; local targets use detached sequential residual TD with gamma 1 and terminal masking before next-network evaluation. Target copies and the 125 Polyak updates follow the specified schedule.
- Proposal and diagnostics: Phi is frozen and hashed; critics/targets freeze before the actor phase; no target actor or subsequent changing-policy TD is used. The pooled/per-scenario numerical gate has the required thresholds and no automatic extension. Diagnostics cover per-row/per-scenario horizons, both heads/minQ, terminal errors, behavior-return caveats, own-target residuals, projected-action sensitivities, aliases, saturation, and hashes. The exported continuation is explicitly carry, not Q of the proposed actor; no traffic-improvement claim is made.
- Runtime: source/spec/data/review binding, finite checks, strict finite JSON output, immutable hash-bound phase checkpoints, process identities, single-thread setup, and normal partial-run/completed-run rejection are present. The finding above breaks the mandatory STOP/resume contract at final publication and is the reason SPEC does not pass.

## Quality Assessment

The implementation is compact and separates data, learning, serialization/runtime, and dispatch responsibilities. Existing serialization and process-identity helpers are reused. No additional data/learner correctness defect was identified in the reviewed source. Checkpoint phase/counter validation, frozen-weight hashes, deterministic RNG restoration, and the successful existing tests support the ordinary training and resume paths. The final publication failure is a concrete recovery defect and a missing boundary test, so QUALITY does not pass.

## Evidence And Review Limits

The supplied diff contains seven new Python files; all seven reconstructed additions match the current files exactly by line content. All nine source/helper SHA256 values match the authoritative evidence's `source_sha256` object. All 353 files in its preserved-file manifest were hash-checked read-only and still match. The actual `results/sdmpc_rl_balanced_goal_20260930/return_init_v1` output does not exist.

The authoritative final evidence records **38 passed, 0 failures, 0 errors, 0 skips**, plus isolated real-data authentication and read-only validation of 750 rows. Those successful same-code tests and validators were not rerun. The authentication receipt records five carry references, five new local episodes, no new environment runs, and no raw environment checkpoint loads.

Reviewed artifact hashes:

| Artifact | SHA256 |
| --- | --- |
| `task-1-brief.md` | `f7a4520ab353b494535d6b3464dfd75e29cc787de05f727b7fa40aa7108790f5` |
| `task-1-report.md` | `24781d97671663f01dfc23692487192d1115ba72c1b5b174e027858d32eced75` |
| `task-1.diff` | `18bc66394d6d14032ee25fd9a3f4749f96e1be84bc55fbc9f92fdb32b468dfd3` |
| `test-evidence/319ebe03/evidence.json` | `66a74946012489bb230259c54b2bdc138d9aa7ebd17d0a897f6db28404d90e9a` |
| `test-evidence/319ebe03/authentication.json` | `dcd544490e4d3a8c4117e3f617a7aa2cad5483bfc3de0709ee83e18c87092487` |
| `test-evidence/319ebe03/tests.xml` | `8d6a9d59a2f3f203938f295074e8774c3c40f57eba2fd086e1345c18b8aa773c` |
| Serialized production SPEC | `933e29c1b316530ceff1605d527532e482a658c12fa6c13fd24a93074ef6593e` |
| Reviewed `run.py` | `944b9e7d027eda1e39c5672c935f4044e80fb2ad3201487507609c0dd7cd0bd9` |

The evidence paths above are under `work/sdmpc_rl_return_init_20260930/`; brief/report/diff are beside this review. The SPEC digest is the implementation's serialized specification digest, not the Markdown brief's file hash.

An initial attempt to import the real runner for the new repro stopped during Torch initialization because sandbox access to the existing `.deps-budget/colorama/__init__.py` was denied. No update or test suite ran. The successful replacement used the exact runner function with standard-library in-memory doubles, avoiding dependency access, installations, ACL changes, and filesystem fixtures. This limitation does not change the authoritative 38-test result.

No real-data optimization, environment boot/reset/step, canonical evaluation, recollection, dependency installation, commit/push, ACL/power change, code fix, or approval receipt was performed. No additional thread or numerical worker was started. Only this review report was written. Actual numerical-fit success and canonical traffic acceptance remain unmeasured and outside this review. Parent retains dispatch and exact-hash receipt ownership; approval must wait for resolution and review of finding 1.
