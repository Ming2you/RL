# Final Bounded Pilot Review

Date: 2026-09-29. Independent implementation and numerical-evidence review.

**PILOT_REVIEW: PASS**

**CODE_REVIEW: PASS**

**CONTROLLER_SPEC: PASS**

**CONTROLLER_QUALITY: PASS**

**NUMERICAL_ADMISSION: PASS FOR THE BOUNDED PILOT ONLY**

## Verdict And Scope

No blocking finding remains for the exact implementation and evidence identities
listed below. The three sampled-state admission gates are satisfied. This is
permission to proceed through the existing preflight gate to the finite pilot,
not a finding that training, full-run evaluation, performance acceptance or
scientific validation has already succeeded.

The original two P2 findings remain preserved in
[implementation_review.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/implementation_review.md).
Its Round 1 closure applies to the repaired source pins reviewed here. The
earlier HOLD and pending-numerical statements in that historical report are not
erased; this separate final report resolves numerical admission for these pins.

Admitted scope is exactly:

1. Two full sequential training episodes from scratch, demand seeds 6201 and
   6202, with the new carry action/observation contract and no reused v1 policy
   or replay.
2. One normal-reset canonical zero-action carry-center full-run evaluation.
3. One normal-reset canonical frozen-RL full-run evaluation of the resulting
   final model, without evaluation exploration or gradient updates.
4. Physical-only performance guard with strict physical and executed-budget
   checks retained, PFO on initialization/reset or required recovery, and
   previous-executed-budget anchoring. This is not strictly PFO-free.
5. The existing 75 controlled intervals / 14400-second episode contract,
   five warmup intervals, true-terminal handling and actual interval TTT reward.
   Use 20 TD3 updates per collected transition after replay reaches batch 32:
   880 updates after the first 75 transitions and 2380 after 150 transitions,
   assuming both episodes complete.
6. The frozen orchestrator's maximum of two concurrent one-core children,
   STOP/checkpoint/drain behavior, and termination after these two training
   episodes and two evaluations. No legacy DDQN resumption, recurring
   automation, additional collection or automatic experiment extension.

Before launching, build a fresh gate from this exact report and the evidence
below. The runner must still verify current production/snapshot pins and every
gate-bound evidence hash. A later source or evidence change invalidates this
review's applicability and requires renewed admission. No gate or numerical
worker was launched by this reviewer.

## Review Method

- Read the binding plan, preserved implementation review and scoped repair
  closure. Reconfirmed that the repaired admission helper enforces expected
  state, mode and action identity and reconciles every summary with its detailed
  executed-control record. Reconfirmed that completed children must match the
  physical pilot guard in both settings locations, with all skipped children
  validated before any sibling launch.
- Independently recomputed all 11 production hashes and called the read-only
  pin verifier, which verified the 151-file frozen runtime against its manifest.
  Every smoke/probe source pin matches the current repaired implementation.
  The comparator controller hash in all three probe settings matches the
  preserved v1 controller.
- Called the production `validate_probe` helper directly against the already
  completed records for steps 5, 30 and 50. All three passed. This performed
  JSON validation only: no simulator boot, probe, training or test execution.
  JSON bytes were hashed before and after reading.
- Independently reconciled carry anchors and incoming prices against the frozen
  saved fixtures, the action-to-budget arithmetic, NUF-only clipping, anchor
  offsets, requested-versus-executed budgets, candidate/fallback price
  commitment, H3 diagnostics, timing decomposition and finite reached states.
  Physical-control changes also correspond to changed actual reached-state
  records. This is an audit of recorded numerical evidence, not a new physical
  simulation or an independent recomputation of the plant model.
- Parsed the fresh JUnit XML and inspected the actual smoke's saved audits and
  pinned smoke assertions. No unchanged unit suite or traffic run was rerun.
  Implementer reports inform test coverage; they do not substitute for the
  detailed admission evidence.

## Code And Regression Integration

Both original P2 findings are closed at the source hashes below:

- State/evidence binding: exactly the expected two modes and five labeled
  actions at each expected state; exactly ten detailed records per state;
  timeline, settings, requested/executed controls, checks, reference/PFO
  semantics, summaries and influence reconciled. Distinct influential states
  are counted as a set, not by folder or duplicated completion claim.
- Pilot guard: physical plan guard required in completed-child settings and the
  coordinator contract. All existing train/center/RL completions are checked
  before any new sibling can launch. New commands explicitly pass physical
  guard; newly exited children use the same validator.

The unchanged controller retains the reviewed carried anchor, signed NP,
NUF-only clipping, current-state witness validation, fail-closed recovery,
rejected-request preservation and candidate/fallback dual ownership. Controller
specification and quality remain PASS within the original review scope.

Fresh JUnit reports **130 declared and actual test cases, 47.192 seconds,
zero failures, errors or skips**. Counts reconcile to controller 54,
environment 16, run contracts 15, runner 10, pilot 16 and preflight 19.
The previously reviewed regression sources cover the two P2 failures, exact
pause/terminal resume, real-Torch 2380-update synthetic-transition scheduling,
frozen evaluation without updates, STOP/drain and skipped-child provenance.
The real-Torch regression is not a completed traffic-training pilot.
The reviewer did not independently rerun these tests.

## Actual Smoke

The fresh pinned smoke is PASS: two actual intervals with serialized
restore at initialization and at the ordinary carried step, 2367 observation
features, and final environment step 7.

The pinned producer asserts exact observation, reward, terminal and executed
control parity across restore at both steps, plus explicit committed-dual
equality at the first step and carried-anchor equality before the second.
There is no separate explicit full-precision committed-dual equality assertion
after the second step; do not extend the smoke's claim beyond these assertions
and the reviewed checkpoint regressions.

Saved audits reconcile:

| Controlled Step | Reference | PFO Calls | Actual Interval TTT | Cumulative TTT |
|---|---|---:|---:|---:|
| 5 | pfo_initial | 1 | 20.650868151393297 | 130.3543410535561 |
| 6 | previous | 0 | 20.61315248655373 | 150.96749354010984 |

Both executed physical controls and executed budgets pass their checks, use
physical guard and are nonterminal. The second anchor equals the first executed
budget, and its incoming prices equal the preceding committed prices. Cumulative
TTT reconciles with cumulative freeway/urban totals; each interval reconciles
with its interval plant log. The first reward is -0.20650868151393298,
equal to negative interval TTT / 100. This smoke is not a full episode.

## Matched Numerical Gates

Each state has ten completed actual one-interval branches:
`v1_pfo_h3` and `carry_physical`, each with zero,
NP +/-0.25 and NUF +/-0.25 actions. The original times are 900, 5400 and
9000 seconds, with reached times 1080, 5580 and 9180 seconds respectively.

All **30/30** detailed executions record valid physical control and strict
executed-budget feasibility, with achieved budgets no greater than executed
budgets. Selected/executed physical points and the detailed execution checks
agree. All summaries are derived consistently from the detailed records.

| Saved Step | V1 Mean Decision Seconds | Carry Mean Decision Seconds | Carry PFO Calls / 5 | Carry Fallbacks / 5 | Nonzero Carry Actions Changing Execution |
|---|---:|---:|---:|---:|---|
| 5 | 29.8652 | 29.3976 | 5 | 1 | np_plus, np_minus, nuf_minus |
| 30 | 27.1293 | 19.2854 | 0 | 0 | nuf_minus |
| 50 | 27.7322 | 20.9673 | 0 | 0 | np_plus, np_minus, nuf_plus, nuf_minus |

The initial state's five PFO calls are one per independent initialization,
not five sequential ordinary-step calls. All ten noninitial carry branches
use `previous` references, zero PFO calls and zero PFO time. All fifteen v1
branches use current PFO references and one PFO call each.

Nonzero executable influence is present at **three distinct states**, exceeding
the required two. Influence is not inferred from changed budgets alone: it is
verified from executed coordinate differences above 1e-9 and changed reached
states. Even excluding the initial fallback branch, accepted nonzero lower
responses establish initial-state influence.

Independent anchor checks match the saved fixtures: initial witness budget
[-112.87566638955153, 6000.0], step-30 prior executed budget
[-235.3233457606126, 5615.086523], and step-50 prior executed budget
[-166.88041923193384, 5019.8465432922185]. Noninitial anchors are not silently
replaced by the freshly evaluated current witness budget.

## Observations And Limits

- Initial NP tightening requests [-125.37566638955153, 6000.0], which the lower
  response cannot satisfy. The record retains that failed request and candidate,
  then executes the valid reference budget [-112.87566638955153, 6000.0] with
  incoming prices. It is a genuine fallback, not a relabeled successful request.
- Initial NUF increase clips 6250 to the 6000 capacity and aliases the zero
  action. NP remains signed and unclipped. None of the ten noninitial carry
  requests is clipped. At step 30, zero, NP increase, NP decrease and NUF
  increase all execute the same physical control despite different requests.
- The step-30 influencing NUF decrease increases actual interval TTT by
  0.02023555678943012 relative to carry zero. Executable influence therefore
  does not imply benefit. At step 50 all four nonzero actions affect execution,
  but only NP increase improves interval TTT relative to carry zero.
- Eight carry lower-selection diagnostics indicate an H3 rejection would have
  occurred: three initial, one at step 30 and four at step 50. One of the eight
  is the separately rejected initial NP request; seven are executed lower
  responses permitted by physical-only guard. These diagnostics compare
  predicted H3 TTT, not a full-run performance guarantee.
- All fifteen carry records report `converged=false`. Their strict execution
  checks pass; this admits bounded feasible execution, not solver convergence
  or optimality. No convergence gate was part of the approved admission plan.
- Mean scalar rollouts are v1/carry 145.8/145.8 at step 5, 108.8/86.0 at
  step 30 and 120.4/96.6 at step 50. Decision timings include preparation/PFO,
  lower solve and guard work as recorded by the pinned producer. They exclude
  the actual plant step and are not end-to-end episode times.
- Probe CPU masks are 1, 4 and 16, with three concurrent one-core processes.
  These are matched-state descriptive measurements, not isolated repeated
  timing trials. Reference policy, anchor and guard differ; a causal or
  generalized speedup claim is unsupported.
- The 30 branches and two-interval smoke are not full-run or learned-policy
  results. No long-horizon TTT gain, generalization, legacy 5% contract,
  trained-policy quality or recovery frequency across unseen states is
  established. Probe responses must not become replay/Q targets.
- After the admitted finite pilot, reconcile all four trajectories, 75 controls
  per trajectory, terminal time/inventory, TTT, clipping, requested failures,
  fallbacks, recovery/PFO counts, reached states, Q diagnostics, optimizer
  updates, frozen-evaluation behavior and runtime. Compare against v1 only
  descriptively under the shared plant contract, then stop and report.

## Exact Production Identities

Production directory: [carry implementation](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929).
All SHA256 values below are full file-byte hashes.

| File | SHA256 |
|---|---|
| [budget_controller.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_controller.py) | `0402511ae8e965e315b4b9808e23bbdc9c2fd274666a983b2a3303e6b916f9c2` |
| [budget_env.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_env.py) | `6ccb4ef400106e29282797ec5d69b7a8dbeee5a1310bd7d6f97b7c7e0c7ee2c4` |
| [budget_runtime.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_runtime.py) | `6ae552bad2c092af9d2ab55b0c81e34558754861217fe427788820689800e9a6` |
| [build_preflight.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/build_preflight.py) | `a9377bd446599f773974e86618a513cc47ce81b39aebd70f330f461b8a36f082` |
| [compare_runs.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/compare_runs.py) | `ad44ddb2794ee232a025412379930e8275b4a5948b44d54296a3ddf63a56e485` |
| [freeze_runtime.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/freeze_runtime.py) | `8a13455b1d87a117eec7f97786f7996786bf49c0d8aaef6e2a3ef8c1cca608ab` |
| [probe_carry.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/probe_carry.py) | `58f5a05cf197250dba7f4ff0afa4323bb7dfd6e6369c5f064ed7da73fd160c52` |
| [run_budget.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/run_budget.py) | `11f1d07d1c02eecd945ebf9923394a5e9daf115f1086a0ceaa047af41823d193` |
| [run_pilot.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/run_pilot.py) | `d462083afed597faf6e1021e4a59396603a15f370ffff47d5bf79ac8e1fc3eef` |
| [smoke_budget.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/smoke_budget.py) | `3b324a61ae8f1d929179b285f7c8ff8265009f7e03be84a8aa53a3cbc5525659` |
| [td3.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/td3.py) | `746b9c6c45aba37cf9dc169274cb64a22ec51fcb1b2c5dc06f2d7dd3104ee1b3` |

Frozen runtime manifest:
[manifest.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/manifest.json)
SHA256 `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
Verified frozen contents: 151 files, 3,975,269 bytes.
The three fixture hashes are additionally listed for inspectability; they are
already covered by that verified snapshot manifest.

| File | SHA256 |
|---|---|
| [outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/plant_004.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/plant_004.json) | `41d175385fc7c21726700a3002c6f32427953a4202f914f07e20f3418abb70f9` |
| [outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/plant_029.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/plant_029.json) | `a05ed5683d5e5ac86c19b2d9a91040e6a87262d3da446ed9e2e8f0c1f57be7ac` |
| [outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/plant_049.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/plant_049.json) | `dcafb774fcc07fd685ce73d820e8b1264d4c44f5c6a15608292b78a6b79d9a03` |

## Exact Admission Evidence Identities

Evidence directory: [admission_v1](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1).
The following 38 artifacts comprise smoke, JUnit and all 36 probe
settings/completion/detail files. The preflight gate additionally binds this
final report itself, for 39 evidence paths with one review. Status/log files
are not used as numerical admission authority.

| File | SHA256 |
|---|---|
| [smoke_resume.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/smoke_resume.json) | `bc32a0d6c7083a72a4772fab3facdb24526980a113b726f5e952487c39c48d7a` |
| [probe_step5/completion.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/completion.json) | `c4cadab77775294b274db0a27d3290e59d6df42d7f2fbeffb16959225a079662` |
| [probe_step5/settings.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/settings.json) | `42f9c3dd83b5bf010401582630c5fd2ae58fe5c231eacdb9fcd55a1707e10a3d` |
| [probe_step5/carry_physical_np_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/carry_physical_np_minus.json) | `3abca3d379bd6b1fa59412e309d25ae197c5ddd8f4fcd7286477d35a15ffd58d` |
| [probe_step5/carry_physical_np_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/carry_physical_np_plus.json) | `0e928f575817e04b7aaf091d537763bf9fb8c5f4a75dcf446fad83c2f067d3dd` |
| [probe_step5/carry_physical_nuf_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/carry_physical_nuf_minus.json) | `56bb0927cbeb84af6567932595a025bdb691735181622519bd76a3efd9dd1db0` |
| [probe_step5/carry_physical_nuf_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/carry_physical_nuf_plus.json) | `a8c34337bd786ba8c2379606932c844860f36d8d6a69a0357dbfafe59ca6e762` |
| [probe_step5/carry_physical_zero.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/carry_physical_zero.json) | `2f890f65434d1e90e080940e8207ddd57823145260f0f708fd90879c85571d8b` |
| [probe_step5/v1_pfo_h3_np_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/v1_pfo_h3_np_minus.json) | `7493a1253cc7c72f4d7b6278c523d56951639aa71d6acf7edc33e43a0e0b7022` |
| [probe_step5/v1_pfo_h3_np_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/v1_pfo_h3_np_plus.json) | `53c03d7d0dee353fb16e021047e7853ce9677578bc7a32e97b6d9d29785bd63e` |
| [probe_step5/v1_pfo_h3_nuf_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/v1_pfo_h3_nuf_minus.json) | `485db0f45a7db47870741b700fa1eb691824313f158eb26ab85fafb8e3c9c593` |
| [probe_step5/v1_pfo_h3_nuf_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/v1_pfo_h3_nuf_plus.json) | `038bc86b26f5e56ff4ed2eca0b7258a6e7b5c7c27109551cb4ff2e083669f7a9` |
| [probe_step5/v1_pfo_h3_zero.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step5/v1_pfo_h3_zero.json) | `cdfd09bc28c7cfea982664bd95c6a3e27f4a01e2db8c9af6f160f396c6f2f288` |
| [probe_step30/completion.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/completion.json) | `bd9dd01560b3e679c96a03f19ef44aab80264fefacb504a6ee6ab8f8d29ef930` |
| [probe_step30/settings.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/settings.json) | `4b1814d6d27a6f156578825bc5d9ae8d86d6d618492e4995e44c41c2b3a2695e` |
| [probe_step30/carry_physical_np_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/carry_physical_np_minus.json) | `cdc7453d902326a2873aabd4827806f6bd65c7e18f9520fa091e3042ddd71a3c` |
| [probe_step30/carry_physical_np_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/carry_physical_np_plus.json) | `1f0aa89157f99a55a06423ddcf460aa50d51b196df9087991fad3d1fb6924aa9` |
| [probe_step30/carry_physical_nuf_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/carry_physical_nuf_minus.json) | `a155008a04a739d2e2deb56c3eb2b4702f3979b5ad4d4398de92c87ddbcab4e2` |
| [probe_step30/carry_physical_nuf_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/carry_physical_nuf_plus.json) | `aa8b71c2f999117222a78e07b36b509ced5000fec4c5117beedf8c2b93144cc6` |
| [probe_step30/carry_physical_zero.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/carry_physical_zero.json) | `211e70340994455d5cd9f674f6044a3d32e7fa0b662af9c961990cd0ee2ea880` |
| [probe_step30/v1_pfo_h3_np_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/v1_pfo_h3_np_minus.json) | `235864108229c97d896ee7e68c5966eb017495f285b0c794b66eced3af849ccf` |
| [probe_step30/v1_pfo_h3_np_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/v1_pfo_h3_np_plus.json) | `a42ae1be0186028fc9d452fbedf3b655093c44a91d347da09714e7ea40d1c3d7` |
| [probe_step30/v1_pfo_h3_nuf_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/v1_pfo_h3_nuf_minus.json) | `c5b1bcd460baf457f3124c319fcc4dd421550dce181da8ffb56e632aa081493a` |
| [probe_step30/v1_pfo_h3_nuf_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/v1_pfo_h3_nuf_plus.json) | `e43b21293a78c68034eae3f6181f06a0e9c392ba87b232b1a3b032c4a3e319e5` |
| [probe_step30/v1_pfo_h3_zero.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step30/v1_pfo_h3_zero.json) | `f5fec9ad6b001eab78fa3e22bcf56614518d4f0e798f54eb3b56c97052db9fbb` |
| [probe_step50/completion.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/completion.json) | `eb86862df1bdc4c8049437387726c15712ba624abebe62334bf338803733d6e4` |
| [probe_step50/settings.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/settings.json) | `10437b5dd980d97f507813202e20a3fe7c3ae4e1ec0ae486b6e47bcc6345d0dc` |
| [probe_step50/carry_physical_np_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/carry_physical_np_minus.json) | `640afa317b62c4c314a329e3a9d0f5bb64ca5a6ffb80310e84b55a63dedf261d` |
| [probe_step50/carry_physical_np_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/carry_physical_np_plus.json) | `2da2dd6be6a8645570c1ba95251c653007bac648d0777458075aa79091dae167` |
| [probe_step50/carry_physical_nuf_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/carry_physical_nuf_minus.json) | `b33ce87f7f62d0f0a127afe4a8d55c40751824173b40bfcb81d040201a632d57` |
| [probe_step50/carry_physical_nuf_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/carry_physical_nuf_plus.json) | `200f02cee4c6cd5a1a268eb86fa16c0acf092a8330c3c57d2d7dbf9797a9ca88` |
| [probe_step50/carry_physical_zero.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/carry_physical_zero.json) | `fbb7964fe290daf86c82a19c6b7d6916e53f3a88be596676f1e3b0a4ea3cce87` |
| [probe_step50/v1_pfo_h3_np_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/v1_pfo_h3_np_minus.json) | `d1e5ff5902b35e79edd97761ab93f2d29d886a14ce6b07734ab79232deec8b1c` |
| [probe_step50/v1_pfo_h3_np_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/v1_pfo_h3_np_plus.json) | `f1470d2ee75168f69e38f7c84cbb62f7cfeaab3a455745911d46e757086fff8a` |
| [probe_step50/v1_pfo_h3_nuf_minus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/v1_pfo_h3_nuf_minus.json) | `4a212cf8f01548c5f86b3ac5385900163e336430f4286afbf97b593249d92c29` |
| [probe_step50/v1_pfo_h3_nuf_plus.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/v1_pfo_h3_nuf_plus.json) | `9c637056f88ce82264b44161c4a96050c43a842f78f46ca3e3c4a271f2aeb328` |
| [probe_step50/v1_pfo_h3_zero.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/probe_step50/v1_pfo_h3_zero.json) | `53d9f246e6089ea75acd05ca14bf1b80662e33c2de5693c8556b34c70c62bfaa` |
| [preflight_tests.xml](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/preflight_tests.xml) | `bbe799860b7523e8f0f058e2ae4928df21e14c3e4a2cdef6e0afb45388f4056e` |

## Review And Test Provenance

These context hashes identify the reviewed plan, historical review, reports,
v1 comparator and current regression sources. They are provenance references,
not additional artifacts silently injected into the gate's evidence set.
Later pilot progress must not rewrite this final review.

| File | SHA256 |
|---|---|
| [docs/rl_budget_carry_plan_20260929.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/docs/rl_budget_carry_plan_20260929.md) | `01980d39ecc8daaacc422e8a7e8219ce9503b91b321241d3bb60cbd5729aaf38` |
| [work/sdmpc_rl_carry_20260929/implementation_review.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/implementation_review.md) | `7f5f76517b3b55ab3abe5a2293318d63c531222837dc75396f894de306ae1a98` |
| [work/sdmpc_rl_budget_20260929/budget_controller.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_controller.py) | `fa0aa996f4b3e43024160a9ea81ec4fab02dd40264c6b020dd67d35477548e98` |
| [work/sdmpc_rl_carry_20260929/test_carry_controller.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/test_carry_controller.py) | `78a6aa18c80eceb4938775b4bd82a898b1c0d70e3f137639e050836faf99e3cd` |
| [work/sdmpc_rl_carry_20260929/test_carry_environment.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/test_carry_environment.py) | `1625918062c962dba2fb188bfe2d152a1b8543b6d83710e404af7ae08026fe9d` |
| [work/sdmpc_rl_carry_20260929/test_carry_pilot.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/test_carry_pilot.py) | `69a377d61278a9bcc996f3dcff482ae317d08c6e73bbdf96421decead1cb8497` |
| [work/sdmpc_rl_carry_20260929/test_carry_preflight.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/test_carry_preflight.py) | `85a9117d599a6dc5dea2745b58f027315c8339c4295ac0b4ce3c61704973a918` |
| [work/sdmpc_rl_carry_20260929/test_carry_run_contracts.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/test_carry_run_contracts.py) | `ec817b0626159466aa620cc6d9255ec4783e663f1f8e990913490600f2928ce9` |
| [work/sdmpc_rl_carry_20260929/test_carry_runner.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/test_carry_runner.py) | `d787ada0b0887fe3641d369f1ca09fefcb75f14a6a32e88898901593da5b4854` |
| [.superpowers/sdd/rl_budget_carry_plan_20260929/review_brief.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_carry_plan_20260929/review_brief.md) | `447ae82e7c906b91b87c6f42cf6c34d27d9ed1668301b5842cb6a03fa0c2ba0d` |
| [.superpowers/sdd/rl_budget_carry_plan_20260929/controller_report.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_carry_plan_20260929/controller_report.md) | `4406d378ab008ad3f7fd7e8f6ee69548145c7e3ab9080f0b5445adeff95f843a` |
| [.superpowers/sdd/rl_budget_carry_plan_20260929/env_test_report.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_carry_plan_20260929/env_test_report.md) | `cde04916cae6fc4870c71a7a8c5b8f864c472c32547cff4fba295b12534cb5be` |
| [.superpowers/sdd/rl_budget_carry_plan_20260929/review_fix_round1_report.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_carry_plan_20260929/review_fix_round1_report.md) | `64ba6b974e6f78bb61d5eef013ca6776612c44fa716aba4819eb8510ded573bc` |

## Finalization

This is a final prelaunch report, not a living run log. Preserve these bytes
once included in the gate; record later pilot outcomes in a separate artifact,
not an addendum here. The report's own file hash is to be computed externally
by the gate, avoiding a self-referential hash.

Only this new report was written during the final numerical-evidence review.
The original findings and closure were left intact. No production, test,
frozen-runtime, evidence, git-index or branch changes were made by this review,
and no tests, traffic probes, training or evaluation were rerun.

