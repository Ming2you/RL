# Independent Pre-Pilot Review

Date: 2026-09-29. Final narrow closure review of the remaining dynamic-config provenance finding, after both owners declared source stability and the refreshed consolidated suite completed. No known P1/P2 acceptance or lifecycle blockers remain in the reviewed scope for the bounded finite pilot.

PILOT_REVIEW: PASS

SPEC: PASS

QUALITY: PASS

This is admission for two training episodes and four frozen evaluations under the reviewed, pinned contracts. It is not traffic-performance acceptance, a speedup claim, solver convergence certification, or evidence of generalization. The parent retains responsibility for building the final gate and launching the bounded pilot; this reviewer launched neither.

## Config Finding Closed

- Environment portion CLOSED: [budget_env.py:235](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:235) now uses the existing frozen runtime's `to_plain_dict()` for config/options under contract version 2. [budget_env.py:267](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:267) applies the same lossless representation to saved-simulator and current-runtime config before live state replacement. Controller and general runtime file hashes remain unchanged; no solver or global `plain()` change was needed. Repaired environment SHA256: `cf39a3207b6bfa28aeaa16d4c64be3f2452bc5ce7a2569ac08f83a251443a569`.
- Independently ran four focused, in-memory tests using the actual frozen `historical_config.to_plain_dict()` in place of the fixture serializer: dynamic receiving config/options/runtime drift refusal, dynamic saved-simulator drift refusal, detached same-config exact restore, and terminal/legacy-version rejection. All four passed in 0.154 s; no traffic solves or files were created. The owner separately reports 22 wrapper tests passed.
- Runner portion CLOSED: [run_budget.py:160](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_budget.py:160) and [run_budget.py:161](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_budget.py:161) now hash `rt['rc'].to_plain_dict(cfg/options)` directly. Inspected [test_run_budget.py:237](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/test_run_budget.py:237) and [test_run_budget.py:254](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/test_run_budget.py:254): actual settings construction uses the real frozen serializer; dynamic network, nested MPC, and lower-option changes alter the appropriate hash; resume/policy drift is rejected before metadata/learner mutation. Owner reports 25 focused tests passed. No generic serializer or physical-controller behavior was changed.
- New [smoke_resume_config_contract.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_budget_20260929/smoke_resume_config_contract.json) inspected: `passed=true`, serialized resume, 2362 features, step 6, reward `-0.20650868151393298`, nonterminal, one actual interval only. SHA256 `a23d894549f5b4568baef5e8ed24c842dc98c2af1915358a4cd9303a16ad085d`. Parent confirms it ran under the new contract with exact physical next-step parity.
- Independently parsed refreshed [preflight_tests.xml](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_budget_20260929/preflight_tests.xml): 135 cases, zero errors/failures/skips, 14.770 s. Confirmed all five new environment/runner dynamic-config regression cases appear exactly once. SHA256 `46838d9b2b3fd50c71a942f526e8ef5184eeef438cc6f67511aefcb72c15ddbe`. The consolidated suite and traffic smoke were supplied by the parent, not rerun by this reviewer.
- The preflight builder and its test now require `smoke_resume_config_contract.json`; builder SHA256 `ac0c81b81e511a81c7a6e4d52805d34002b04f18b19df471e8a68b9bf0d4d3dc`. Confirmed old smoke is unchanged and the archived `preflight_tests_before_dynamic_config_fix.xml` retains the original 130-test artifact hash.
- Prior repaired native step-30 parity, candidate isolation, and four influence diagnostics remain supporting evidence, not full-run performance claims. This narrow repair only changes config provenance/admission, not physical budget/PFO guard, candidate-price selection, requested-action replay, or timing semantics. Previously closed lifecycle findings remain closed; no repair-induced regression was found in this scope.

## Current Source Identity

All seven scoped files parse. Reverified the frozen baseline: 151 files, 3,975,269 bytes; manifest SHA256 `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`. Baseline, controller, general runtime, comparator and pilot hashes remain unchanged from the preceding review. The only reviewed production changes for this closure are environment config serialization/versioning, the two runner hash expressions, and the gate's new smoke filename.

| File | Current SHA256 |
|---|---|
| budget_controller.py | `fa0aa996f4b3e43024160a9ea81ec4fab02dd40264c6b020dd67d35477548e98` |
| budget_env.py | `cf39a3207b6bfa28aeaa16d4c64be3f2452bc5ce7a2569ac08f83a251443a569` |
| budget_runtime.py | `6ae552bad2c092af9d2ab55b0c81e34558754861217fe427788820689800e9a6` |
| run_budget.py | `6db11e238c1258ff654ef16657313e38e3d96a0c24c3d65d1d7213c6338ad950` |
| compare_runs.py | `88327e316ed024494a8d73ab471f5d372437445898d47f7763e4393c1902aa0b` |
| run_pilot.py | `a515e5a13f69234c6a196ca06140643b616bcd9eb0cb90a2d09ef2e31295584c` |
| build_preflight.py | `ac0c81b81e511a81c7a6e4d52805d34002b04f18b19df471e8a68b9bf0d4d3dc` |

SHA256 of `json.dumps(run_budget.pins(DEFAULT_SNAPSHOT), sort_keys=True, allow_nan=False).encode()` is `ebb5f7169c4daeea65a0c78e9f3a4c42d62fcf37c58d3b3a5e9d2e0a6c596e1a` (13 non-test task Python files plus snapshot identity). Use this reviewed source identity when building the final evidence gate; a later source change needs corresponding review.

## Nonblocking Follow-Ups

- PFO hidden memory is checkpointed but not fully actor-observed. This is explicitly partially observed; history/recurrent validation remains future work, not a launch blocker.
- Queue-capacity durations are disclosed endpoint-sampled estimates. Summed queue-seconds are not exact substep elapsed exposure; exact exposure is broader validation work.
- Mechanical review-time source/evidence binding would harden the gate workflow. Preserve final-review-then-pin stability. Measured diagnostic/concurrent timings do not establish isolated speedup or generalization.

Only this report was modified by the reviewer. No implementation, baseline, results, full-suite rerun, traffic simulation, training, gate artifact, or commit was created or changed by this closure review.

## Appendix A: Pre-Repair Review

The following findings, evidence and source pins are preserved history of the earlier HOLD, not the current verdict. The version-1 config finding is closed above; historical line references describe the then-reviewed implementation.

### Original Acceptance Blocker (Version 1)

**[P2] Checkpoint/config identity still omits runtime-added physical configuration.**

Locations: [budget_env.py:235](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:235), [budget_env.py:266](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:266), [budget_runtime.py:19](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_runtime.py:19), and the shared runner hash at [run_budget.py:160](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_budget.py:160).

`contract()` and the saved-simulator checks serialize config through `plain()` -> `dataclasses.asdict()`. This drops public attributes added after dataclass construction. The frozen baseline deliberately adds `network.terminal_zero_gradient` at [historical_config.py:308](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_config.py:308); the plant reads it at [metanet.py:520](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_tree/src/models/metanet.py:520). The baseline also has dynamically added MPC options. These are behavioral configuration, not incidental metadata.

Independently reproduced without a traffic solve, using the real `BudgetEnv.checkpoint/contract/restore` methods and the existing synthetic environment fixture: save with `terminal_zero_gradient=True`, receive with `False`, then restore. The serialized contract has no such field; the two contracts compare equal; restore succeeds and leaves `sim.cfg.network.terminal_zero_gradient=True` but `controller.lower.cfg.network.terminal_zero_gradient=False`. This defeats fail-closed resume identity and permits the restored plant/PFO and new lower controller to disagree about physics. The runner's config hash also ignores the field.

Required repair: use a lossless configuration representation consistently for environment contracts, saved-simulator/current-runtime validation, and runner policy/config hashing. The frozen runtime already exposes [to_plain_dict at historical_config.py:233](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_config.py:233), which preserves public runtime-added attributes. Add refusal tests for both receiving-config and saved-simulator dynamic-field drift before replacing live state, plus a runner hash sensitivity test. Same-config serialized next-step parity does not test incompatible-config rejection.

## Closed Findings

- Candidate audit retention: resolved at [budget_controller.py:120](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_controller.py:120), [budget_controller.py:163](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_controller.py:163), and [budget_controller.py:180](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_controller.py:180). Returned diagnostics retain detached candidate requests, residuals, inner price/local rows, achieved budgets, predicted TTT, and candidate/archive/reference selection identities across the next prepare.
- Runtime source verification: resolved at [budget_runtime.py:44](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_runtime.py:44) and [budget_runtime.py:67](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_runtime.py:67). Full payload verification precedes frozen imports; the returned verifier also pins manifest identity for later checks.
- Checkpoint repair is partial: reward divisor, declared config/options, snapshot and observation implementation/order now participate in admission. Full PFO state and prepared reference are retained; restoring does not solve PFO twice. The dynamic-field omission above is the remaining issue, not a failure of the supplied same-config smoke.
- Runner lifecycle: resolved at [run_pilot.py:65](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_pilot.py:65), [run_pilot.py:101](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_pilot.py:101), and [run_pilot.py:138](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_pilot.py:138). Both skipped and exited children require validated completions. Paused zero-exit children cannot advance the stage; siblings are signaled and drained. All five child artifacts, including seed-6103 validation and the two-episode training/model provenance, are checked before final completion. No additional lifecycle blocker found.
- Comparison identity/accounting: resolved at [compare_runs.py:21](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/compare_runs.py:21), [compare_runs.py:69](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/compare_runs.py:69), and [compare_runs.py:153](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/compare_runs.py:153). Controller roles and distinct runs, finite costs, 75 consecutive interval identities, 14,400-second terminal, cumulative/regional TTT, reward, guard and timing accounting are validated before comparison. Endpoint queue-seconds are separately validated and disclosed.
- Runner policy/runtime repair: [run_budget.py:64](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_budget.py:64) rejects runtime drift without overwriting old resume metadata; [run_budget.py:81](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_budget.py:81) binds exported policies to source/runtime/observation/reward contracts and records distinct training profiles. The dynamic-config hash omission is covered by the single open finding above. Requested-action replay, true-terminal semantics, terminal-checkpoint finalization, and frozen evaluation remain intact by inspection and the supplied focused tests.

## Semantics And Disclosures

- No new physical-budget, PFO H3 guard, selected-price commit, or requested-action replay defect was found. Signed NP remains horizon inbound-minus-outbound service in vehicles; NUF remains the sum of meter commands in vehicles/hour. Only NUF is clipped. Candidates start from copied incoming prices; fallback preserves those prices. Reward uses actual plant interval TTT, not predicted or fixed-tail labels.
- The actor does not observe all PFO hidden memory. Full memory is checkpointed and `run_budget.observation_schema()` explicitly says Markov sufficiency is unproven. Accepted for this bounded, explicitly partially observed pilot; history/recurrent validation remains necessary before stronger claims.
- [budget_env.py:221](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:221) now discloses the 90% threshold, physical capacities, seconds/vehicles units, interval duration, and endpoint-sampled method. This can understate or overstate within-interval exposure. Per-queue duration and summed queue-seconds are estimates, not exact substep elapsed exposure; this is not a pilot blocker with the current disclosure.
- Forecast/observation/PFO/reference/lower/guard/actor timings are separate; plant time is separate. The saved step-30 original 41.6171421 s and wrapper 43.5665366 s are diagnostic samples, not speedup evidence. Concurrent pilot CPU-affinity runs are not isolated speed benchmarks. No generalization, convergence certification, or traffic-performance acceptance follows from these diagnostics.

## Evidence And Remaining Admission

- Independently verified the complete frozen payload: 151 files, 3,975,269 bytes; manifest SHA256 `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`. Baseline was not edited.
- Inspected `parity_step30_repaired.json`: `passed=true`, step 30, three candidates, matched saved state rather than a full run. SHA256 `58b557c78e2c03cc2134af9033f090ea71060f0a5adb261b6ef617c3c0cb3a43`.
- Inspected `smoke_resume.json` and its producer: actual serialized torch checkpoint, independent environment, exact next observation/reward/control/committed dual, requested action replay, 2362-dimensional observation. SHA256 `85aa31e5b70c17fb79e518ebaebc6c8d3ebb7e8403138c000825bd947b12b752`.
- Independently parsed the final [preflight_tests.xml](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_budget_20260929/preflight_tests.xml): 130 test cases, zero errors/failures/skips, recorded time 15.659 s (parent rounded wall time 15.67 s). SHA256 `3f0b3d42161ff3ad99e4ed9ae4c016a7f8e7e0473b4b95da9a79c9cded0791bb`. Linnaeus's fix appendix and the focused lifecycle/runner test implementations were inspected. Suites were not rerun by this reviewer; the passing suite lacks the dynamic-config refusal case independently reproduced above.
- Read the isolation result (`passed=true`) and all four step-5/10/30/50 influence summaries (`completed`, `action_influence_observed=true`). These are diagnostics, not replay labels or proof of full-run improvement.
- All seven scoped Python files parse successfully. Rechecked their hashes after the owner-stable notification. Only small in-memory contract/gate checks and read-only source/evidence/hash inspection were performed here; no traffic simulations or training were launched.
- [build_preflight.py:18](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/build_preflight.py:18) requires successful diagnostics, four influence results, nonempty clean unit results and an independent pass review; [run_pilot.py:31](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/run_pilot.py:31) rejects source drift from the built gate. Executed the actual builder against final evidence with its output writer disabled: it reached and rejected this HOLD review; nothing was written. Final gate creation appropriately remains with the parent after a successful re-review, not before. The open blocker must be fixed and regression/source pins refreshed first.
- Optional hardening, not a second pilot blocker: bind review-time source hashes mechanically into the builder, and revalidate the recorded evidence hashes at launch. Present evidence hashes provide inspectable provenance, not proof that evidence predates no later edits. The parent's specified final-review-then-pin workflow must preserve source/evidence stability. Exact substep exposure and history/recurrent validation are further validation work, not claims established here.

## Historical Source Pins

These identify the version containing the open finding; they are not final admission pins.

| File | SHA256 |
|---|---|
| budget_controller.py | `fa0aa996f4b3e43024160a9ea81ec4fab02dd40264c6b020dd67d35477548e98` |
| budget_env.py | `21fd45f48aad45bab24d7594025e998cd56c500ecfc866807ca664d368e2d05b` |
| budget_runtime.py | `6ae552bad2c092af9d2ab55b0c81e34558754861217fe427788820689800e9a6` |
| run_budget.py | `fbc14be2df36060062c713fd18f0439a175ae2e24d4e2287bb6ba6b06e2b44fb` |
| compare_runs.py | `88327e316ed024494a8d73ab471f5d372437445898d47f7763e4393c1902aa0b` |
| run_pilot.py | `a515e5a13f69234c6a196ca06140643b616bcd9eb0cb90a2d09ef2e31295584c` |
| build_preflight.py | `23bb274c7746752dabc689991c265fc8073b7edefb157c694215c32633a4cc03` |

SHA256 of `json.dumps(run_budget.pins(DEFAULT_SNAPSHOT), sort_keys=True, allow_nan=False).encode()` is `1a8c66586598b80f5124a2f94d2183259ecff36e97448a462d784747e5f9238b` (13 non-test task Python files plus snapshot identity).

Only this report was written during this scoped re-review. No implementation, baseline, result, or commit was changed.

## Appendix B: HOLD History

### 2026-09-29: Initial Final Admission Withheld

Historical verdict: HOLD; SPEC and QUALITY both NEEDS_CHANGES. The sole remaining P2 was lossy dynamic-config serialization in environment restore admission and runner config provenance. An in-memory reproduction accepted a `terminal_zero_gradient` mismatch and left restored plant and new lower-controller config inconsistent. The 130-test suite, repaired native parity, and original serialized-step smoke passed but did not cover incompatible dynamic-config rejection.

Historical source-pins digest: `1a8c66586598b80f5124a2f94d2183259ecff36e97448a462d784747e5f9238b`. Historical environment SHA256: `21fd45f48aad45bab24d7594025e998cd56c500ecfc866807ca664d368e2d05b`; runner SHA256: `fbc14be2df36060062c713fd18f0439a175ae2e24d4e2287bb6ba6b06e2b44fb`. Original smoke SHA256: `85aa31e5b70c17fb79e518ebaebc6c8d3ebb7e8403138c000825bd947b12b752`; original 130-test JUnit SHA256: `3f0b3d42161ff3ad99e4ed9ae4c016a7f8e7e0473b4b95da9a79c9cded0791bb`.

The parent accepted the finding and assigned narrow repairs to the environment and runner owners using the existing frozen runtime's `to_plain_dict()`. All runs were reported idle. The parent will preserve the old smoke as versioned evidence and supply a new actual serialized-step smoke under the repaired contract.

Agreed closure review is limited to the lossless config contract/hash repair, focused receiving/saved-config rejection and hash-sensitivity tests, and the new serialized-step smoke. Previously closed findings will not be reopened absent a repair-induced regression. Broader suggestions remain nonblocking. The current verdict at the top of this document remains authoritative and will change only after stable repair evidence is reviewed.
