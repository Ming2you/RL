# Task 1 Implementation Report

Status: DONE (implementation and tests). Independent review and the one actual training job remain with the parent. No real-data optimization, environment boot/reset/step, commit, dependency installation, ACL change, or traffic-improvement claim occurred.

## Implemented Contract

- `work/sdmpc_rl_return_init_20260930/` is the only new code location. The actual output root is fixed to `results/sdmpc_rl_balanced_goal_20260930/return_init_v1`; it does not exist after this implementation.
- Authentication invokes the existing frozen NUF validators and their isolated original-reference validators. The child forbids boot/reset/step/collection and non-experience Torch deserialization, reconciles the full comparison, and checks identities before/after. Original validators also inspect the old local references for causal provenance; only the selected 375 carry plus 375 NUF-local rows are loaded into the learner.
- The training loader requires the ten admitted training trajectories, 750 transitions, 150/scenario, the unchanged 2367-observation contract, true terminal, sequential continuity, correct reward, and no canonical profile. Exact float64 current and next anchors come from trace, with zero terminal next anchors. Warmup never enters return sums.
- The actor is a memoryless 2367-64-64-2 ReLU network with zero final layer and `[0.2,0.1]*tanh` output. Previous execution remains the only anchor, including after fallback. NP remains signed; only NUF is clamped to `[0,6000]`. Replay, actor-gradient and zero-carry target actions use float32 actions, then float64 physical transformation, division by `[1000,10000]`, then network float32 conversion.
- One shared Phi and two residual critics use two width-64 ReLU hidden layers. Seed 7200 uses a dedicated NumPy Generator and separately saved Torch initialization RNG. The learner does not consume the caller's global RNG streams; updates need no Torch randomness or target noise.
- Phi has exactly 1000 Adam updates at 3e-4. Each minibatch has eight carry rows/scenario: one forced terminal plus seven uniform-with-replacement draws. Phi is then frozen and hashed; residual heads remain zero and targets are exact copies.
- Critics have exactly 250 Adam updates at 3e-4. Each minibatch has four carry/four local per scenario, each subgroup one forced terminal plus three uniform draws. Carry targets are `Gcarry-Phi`; local targets use the specified residual TD with next zero-carry projected requests. Full targets are detached, and terminal/carry rows are masked before all next-network evaluation. Polyak tau .005 runs after each second critic update, for 125 target updates.
- Pre-actor diagnostics include every scenario/horizon row and aggregated horizon bands, Phi and Q training/behavior discrepancies, both heads/minQ, terminal exact errors, own-target residuals, projected-action gradients and probe values, exact projection aliases and float32 network-input aliases, saturation counts, and zero/action/request/output hashes. Local behavior-return discrepancies explicitly carry the continuation caveat.
- The numerical gate requires pooled carry Phi MSE to be at most half its own initialization and every scenario MSE to strictly decrease. A failed gate publishes a numerical-fit failure with no candidate or actor updates; update counts are never extended.
- Only after the gate, exactly ten actor-only Adam steps maximize Q1 at 3e-4 using the same balanced carry/local quotas. Phi, critics and targets are hash-checked and frozen. There is no target actor and no subsequent critic TD. The final checkpoint explicitly declares critic continuation `carry`, not Q of the new actor.
- The runner fixes source/spec/data/review identity before updates, uses one numerical thread, records actual PID and process creation time, checks STOP at repo/goal/output, and saves immutable phase/periodic checkpoints with atomic hash-bound pointers. Checkpoints contain all model/optimizer/RNG/counter/metric/identity state. The final `model_final.pt` is an atomic hard link to the final immutable checkpoint, and only it is admitted. Intermediate checkpoints are for resumption only.
- Fresh runs refuse partial outputs; resume refuses changed identities, hashes, phase/counter contracts or completed runs. Failed updates preserve the last finite authenticated checkpoint and partial files. JSON is finite and strict. No broad process framework or environment runtime is imported into the learner; the existing serialization helper and stdlib-only process identity helper are reused.

## Verification

Exact PowerShell test command, from the RL repo:

```powershell
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_init_20260930/run_tests.py'
```

Final result: **38 passed, zero failures/errors/skips**, pytest time 3.537 seconds. The wrapper additionally ran isolated real-data authentication, read-only 750-row loading/schema/anchor checks, and before/after preservation checks. Optimizer tests use small synthetic arrays and reduced phase counts to exercise boundaries; no real-data fit gate has been measured. The production 1000/250/10 constants are explicitly tested and have no CLI overrides.

Final evidence:

- `work/sdmpc_rl_return_init_20260930/test-evidence/319ebe03/evidence.json`
- `work/sdmpc_rl_return_init_20260930/test-evidence/319ebe03/authentication.json`
- `work/sdmpc_rl_return_init_20260930/test-evidence/319ebe03/tests.xml`
- Evidence SHA256: `66a74946012489bb230259c54b2bdc138d9aa7ebd17d0a897f6db28404d90e9a`
- Authentication SHA256: `dcd544490e4d3a8c4117e3f617a7aa2cad5483bfc3de0709ee83e18c87092487`
- JUnit SHA256: `8d6a9d59a2f3f203938f295074e8774c3c40f57eba2fd086e1345c18b8aa773c`
- Specification SHA256: `933e29c1b316530ceff1605d527532e482a658c12fa6c13fd24a93074ef6593e`

Coverage includes algebra/projection and precision, signed NP, post-fallback zero carry, terminal and carry next-network masking (poisoned next arrays), every minibatch's exact scenario/behavior quotas, warmup-excluded returns, evaluation leakage, data/root/checkpoint hash failures, initialization and phase freezing, Polyak schedule, independent RNG, exact complete-state equality after resumption at seven phase positions, failed numerical gates, finite JSON, partial-file preservation, STOP/resume, source mutation refusal, review identity and no-overwrite. No environment modules appear in the learner/test process.

The initial test attempt `cc0d80fe` is retained: 26 passed and 12 fixture-setup errors, all caused by a missing parent for pytest's explicit basetemp. Creating that parent in the test launcher fixed the setup; the final run passed. Both attempts authenticated the same root hashes and preserved all checked artifacts. Existing `.deps-budget/pytest` and test fixture directories required normal escalation for ACL access; no ACLs were changed.

## Preserved Inputs

The final evidence contains exact before/after hashes for **353 old files**, including the frozen 151-file physical snapshot, relevant old source files, both cohorts' outputs and checkpoint bytes, root completion/comparison/plan/status/process records and ownership ledgers. All matched. Raw environment checkpoint bytes were hashed only, not deserialized. Pre-existing dirty repository changes were left alone.

Authenticated NUF root hashes:

- completion: `2d5c7a881fe5cb818686b13dd83f2c98eb08059e4b698cbe3a1be8757fc0c117`
- comparison: `60f7b670e262348d3d24548809b2aafc26bc7c2a7e7c8f17340fbd17c2803801`

No actual `return_init_v1` output was created. Zero real-data training updates, zero new environments, zero commits. The parent retains completion readout, documentation and dispatch ownership.

## Changed Paths

All changed paths, including every retained synthetic checkpoint/fixture and evidence file, are enumerated in `work/sdmpc_rl_return_init_20260930/changed-paths.json` (67 paths: 65 hashed files plus the inventory and this report). No old file is in this list.

New source files and their final tested SHA256:

| File under work/sdmpc_rl_return_init_20260930 | SHA256 |
| --- | --- |
| authenticate.py | 747ca1b8e1e2f8787e3fe368f0a190f2c9874cba9c04e85df073269b336063a5 |
| data.py | 10a8e894cd6c54a77fe26ed96bd314026e9390a6674b7ad64960fb82c9102cdb |
| learner.py | b667bcf8accbcc5ed105a4c2069017e1d29f73274826d5052590fb804a912ebe |
| run.py | 944b9e7d027eda1e39c5672c935f4044e80fb2ad3201487507609c0dd7cd0bd9 |
| run_tests.py | 7b395f1352bf0a6070c73ce8bcd5605c385f5b400b8565583940524d280949c2 |
| runtime.py | 8663b77b76abeb3777c827470491bfb14ab332f2523ee044e7348ecdd9af4975 |
| test_return_init.py | 6da5b11b1d4cf7b1f9dbd983e923e9082f7817f3c637131d03806daef1aa4a42 |

Additional new paths are `changed-paths.json`, both `test-evidence/{319ebe03,cc0d80fe}/` directories, retained `_t/319ebe03/` synthetic fixtures, and `.superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-1-report.md`.

## Actual Job Handoff

This command is **not executed**. Parent dispatches it once, only after independent review of the tested source. The runner requires an independent review JSON receipt at the path below with `status: "approved"`, a nonempty `reviewer`, the exact `source_sha256` object from final evidence (including the two reused helpers), and `spec_sha256` from final evidence. This is the brief's independent-review prerequisite, not an implementation self-approval. No approval receipt was fabricated.

```powershell
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_init_20260930/run.py' --review '.superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-1-review.json' --output 'results/sdmpc_rl_balanced_goal_20260930/return_init_v1'
```

Resume an interrupted admitted job with the same command plus `--resume`, after the parent resolves/removes any STOP. Do not use another output path or rerun a completed job. Exit codes are 0 completed candidate, 2 numerical-fit failure, 3 STOP; other exceptions are failures. Outputs include immutable checkpoints, latest.json, settings.json, session start/end identities, status.json, metrics.json, and completion.json; only success adds model_final.pt.

Concerns/limits: independent review remains pending; actual Phi fit and finite real-data training outcomes are unmeasured. The parent must account for other jobs within the global eight-worker limit before dispatch; this runner adds one single-thread numerical job and performs validation subprocesses sequentially. Windows process identity and same-volume hard-link final export are intentional local-runtime assumptions (tested here). Checkpoint recovery uses the last completed atomic pointer and preserves orphan files, so an abrupt crash may replay updates since the last 25-update/phase checkpoint. The actor is only a proposal: bounded increments do not establish bounded episode drift, action ranking, canonical improvement, or goal completion.

## Fix Round 1: Accepted R1 Publication STOP/Resume

Status: scoped R1 fix implemented and tested; pending the parent's R1-only independent re-review. This section does not approve the implementation or supersede the independent review's approval authority. No actual training job or approval receipt was issued.

The accepted defect was reproduced before changing `run.py`, using the real runner, Torch serialization, filesystem checkpoints, and hard-link publication. The regression injects three distinct PID/creation/session identities: checkpoint A uses 101/10100/publication-101, first resume uses 202/20200/publication-202, and second resume uses 303/30300/publication-303. These are deterministic test identities, not three spawned OS workers. STOP is created immediately after the real candidate link. The synthetic completed learner is prepared before resumption, and all subsequent learner updates are forbidden.

Before the fix, the first resume stopped without completion, retaining the candidate hash `49507d5179dc7841dd87c3665b59593de37424d4768794f5ff88b4252d193fcd` but advancing latest to `6a3b1e6b78221f75397dab63cf5c836f5c044d2594d9db47c15cce86834b77a8`. After clearing STOP, the second resume raised exactly `FileExistsError: Existing final candidate differs`. Resumed optimizer updates: zero. This failed reproduction is retained under `test-evidence/a68aaa12/`, with detailed binding evidence in `_t/a68aaa12/test_final_publication_stop_re0/r1-publication.json`.

The only production change is in `run.py:93`: the STOP handler calls `persist()` only when the learner phase is not `done`. A done checkpoint is already durable when publication starts; preserving it keeps the immutable candidate and latest pointer bound to the same bytes. Current process identity still goes into the separate session start/end records. Non-done checkpointing, learner phases, scientific parameters, data, validation and export semantics are unchanged. No runtime helper edit was necessary.

The regression at `test_return_init.py:392` now passes. It verifies successful second resumption, zero resumed optimizer updates, unchanged latest-pointer bytes, unchanged candidate/checkpoint hashes, exactly one retained checkpoint, exact equality of the completed learner state, unchanged counters, valid completion output hashes, and distinct stopped/completed session records. In the focused passing run, original checkpoint/latest/candidate all retained SHA256 `85ec78bbfb878eaa020af8e0337e14a6eb08905b07634f58d364d332dde51dfb`.

### Exact Tests And Retained Evidence

From the RL repo, the focused command was run once before the fix and once after it:

```powershell
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_init_20260930/run_tests.py' -k final_publication_stop_resume_changed_process
```

- Before fix, `a68aaa12`: 1 failed, 38 deselected, 0.411 seconds; the expected final-candidate mismatch above.
- After fix, `c2e7d593`: 1 passed, 38 deselected, 0.345 seconds; second resume completed with no updates.

The final full suite was run exactly once after the focused pass:

```powershell
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_init_20260930/run_tests.py'
```

Final result, `1dba6b73`: **39 passed, zero failures/errors/skips, 3.504 seconds**. The unchanged test launcher also repeated isolated read-only authentication and 750-row data validation. Each run preserved all 353 checked old source/artifact hashes. All optimizer work in tests used synthetic fixtures; actual-data optimizer updates and environment runs were both zero. Existing dependency/fixture ACL access used normal escalation, without installation or ACL changes.

Evidence files are under `work/sdmpc_rl_return_init_20260930/test-evidence/`:

| Evidence | SHA256 |
| --- | --- |
| a68aaa12/evidence.json | 24ff86507025405be9c945604d60996545c44d3af01112bd21f421f1ad13c64c |
| a68aaa12/tests.xml | ea6831a7e3e597efa6ae3864938e48537ccf9955089fb3a2a9d67991333d1675 |
| c2e7d593/evidence.json | 237de79ffedd6fa6bae141eb4afacd6fad5547507e233228604d206df076252a |
| c2e7d593/tests.xml | 8af35d08f6ca60f36561f9d9c91deef16dccd2ff4e954baa44c3b3f96879c026 |
| 1dba6b73/evidence.json | b065c410968c2ac80c36bb914ad67bad7149579044a2ad1f9fafbdeed2bf3484 |
| 1dba6b73/tests.xml | 8f7d36ecdb50078c2b32844aa152bd3319803b28af6123cfb8fca265e3187e92 |
| 1dba6b73/authentication.json | dcd544490e4d3a8c4117e3f617a7aa2cad5483bfc3de0709ee83e18c87092487 |

### Final Source Identity And Scope

Only these existing files were manually edited in fix round 1:

- `work/sdmpc_rl_return_init_20260930/run.py`
- `work/sdmpc_rl_return_init_20260930/test_return_init.py`
- `.superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-1-report.md` (this appended section only)

Final tested hashes superseding the original table's two entries:

| Source | SHA256 |
| --- | --- |
| run.py | f1019cc867c074810493024e40716cf801cde4987c172dbbf85bde1c966ef5d6 |
| test_return_init.py | 13a1689aeaa708c1538b29f9555bd6cd209f98ef3d959a7570391c2b671ec016 |

`authenticate.py`, `data.py`, `learner.py`, `run_tests.py`, and `runtime.py` match the parent's `before-fix1` snapshot byte-for-byte. The two reused helpers are unchanged. All nine final source/helper hashes are recorded in `test-evidence/1dba6b73/evidence.json`. The scientific SPEC digest remains `933e29c1b316530ceff1605d527532e482a658c12fa6c13fd24a93074ef6593e`.

Generated test artifacts are retained in `test-evidence/a68aaa12/`, `test-evidence/c2e7d593/`, `test-evidence/1dba6b73/`, and the corresponding `_t/` directories, including the failed reproduction. Prior evidence, `changed-paths.json` (the original implementation inventory), and source-history snapshots were not rewritten. The actual `results/sdmpc_rl_balanced_goal_20260930/return_init_v1` directory remains absent. No actual optimization/simulation, commit, self-approval, or dispatch occurred. The parent owns the scoped diff, R1-only re-review, and any later exact-hash approval receipt and actual-run command already documented above.
