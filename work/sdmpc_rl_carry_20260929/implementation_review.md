# Independent Carry Implementation Review

Date: 2026-09-29. Reviewed the binding plan, controller brief/report, all eleven
v2 production Python files, relevant frozen lower/runtime code, and changes
against frozen v1. Findings below are established by source inspection; no new
tests, traffic simulations, training, or gate generation were performed.

## Findings

### [P2] Admission counts probe folders instead of distinct sampled states

Location: [build_preflight.py:25](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/build_preflight.py:25), especially lines 28-34.

The builder never checks the completion's `step` or `settings.step` against the
expected 5/30/50. It increments `influence_states` from one Boolean per folder.
Consequently, a successful influential step-30 completion copied into
`probe_step50/completion.json` also passes the source/PFO checks and is counted
as a second influential state. The resulting gate admits training without
evidence of influence at two distinct states or any evidence from state 50.
This directly defeats plan tasks 3-4; matching source hashes do not identify
the sampled state. The six admission tests do not catch this: their accepted
fixtures omit both step fields and probe records entirely.

Required closure: validate both recorded step identities against each expected
state and count distinct validated states. Validate the expected two modes and
five action identities per state when deriving influence/PFO/validity results;
bind the detailed evidence used for that decision. A duplicated or mislabeled
completion must be rejected before an admission artifact is written.

### [P2] Resumed pilot children need not use the declared physical guard

Location: [run_pilot.py:13](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/run_pilot.py:13), with [compare_runs.py:102](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/compare_runs.py:102).

`validate_child` passes mode, seeds, source/runtime pins and model identity, but
never compares the child's guard with `plan["guard_mode"]`. The shared loader
explicitly accepts either `physical` or `h3`. Thus a complete carry-center run
created with the supported `--guard-mode h3` option is accepted and skipped on
pilot resume even though the plan declares physical-only execution. With an
otherwise normal physical training/RL run, the mismatch is discovered only in
the final comparison, after collection. If all preexisting children use `h3`,
their contracts agree and the pilot can report completion under a false
physical-only plan. The guard changes executable actions, so this is material
to the proposed experiment, not just a label mismatch.

Required closure: make child acceptance enforce the plan's physical guard,
including its coordinator contract, for both completed/skipped and newly exited
children. Keep the general comparator's ability to inspect `h3` runs if useful;
the bounded pilot must reject them. Cover this refusal before launching siblings.

## Verdict

- `CODE_REVIEW: HOLD` / integration quality needs the two admission repairs.
- `CONTROLLER_SPEC: PASS` and `CONTROLLER_QUALITY: PASS` for the finalized source.
- No additional controller, environment, TD3 integration, or ordinary
  checkpoint/STOP lifecycle blocker was found by inspection.
- `SMOKE_REVIEW: PASS` for the supplied two-interval serialized smoke below.
- `NUMERICAL_ADMISSION: PENDING`. This report is not final pilot admission.

Neither finding demonstrates a defect in an ordinary fresh physical run. They
block reliable enforcement of this pilot's required evidence and resume
contracts. Both can be repaired without changing physical dynamics, lower
optimization, action semantics, or the frozen v1 artifacts.

## Controller And Integration Assessment

- [budget_controller.py:34](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_controller.py:34): current bounds/grid projection precedes original execution validation. The anchor is copied from the prior executed budget, initialized from the first valid witness. NP stays signed; only NUF is clipped; scales remain `[50, 1000]`.
- [budget_controller.py:62](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_controller.py:62): lower solves receive copied request/seed/prices; `finally` restores persistent prices and budget. Final-candidate success commits its prices, archive recovery retains incoming prices, and reference fallback retains the failed request separately while executing the validated witness budget. Audits are detached. The actor never commands physical actuators directly.
- [budget_controller.py:124](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_controller.py:124): physical and budget checks remain mandatory in both guard modes. H3 rejection is conditional and its pre-fallback delta remains visible. The actual frozen budget policy is `upper`, meaning `G <= B` without extra margin; physical-only does not mean budget-free.
- [budget_env.py:163](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_env.py:163): ordinary steps validate projected previous control; only initial preparation or `InvalidReference` recovery calls PFO. Failed recovery propagates. Observation includes reference source/control/budget/TTT, carried budget, previous request/execution/slack, and time. No shared-state mutation defect was found: frozen lower `begin` copies state/forecast/previous, PFO receives a state copy, and probe plant execution uses a state copy.
- [budget_env.py:254](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/budget_env.py:254): lossless dynamic config/options contracts are preserved from repaired v1. Checkpoints retain simulator, PFO memory, carried budget/prices, prepared reference and observation schema; restore reconstructs preparation without another PFO solve. Changed run/policy/checkpoint and observation contracts reject v1 reuse.
- [run_budget.py:216](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/run_budget.py:216): replay stores the requested action with actual sequential `-interval_TTT/100`, gamma 1 and true terminal only. Twenty updates are attempted per new transition, with the unchanged learner withholding updates until 32 samples. The declared two episodes imply 880 updates after episode one and 2380 total; these are code-derived expectations, not measured results. Evaluation does not add transitions or update weights.
- Runner locks, atomic checkpoint replacement, terminal-checkpoint finalization, completed-child refusal, paused-child handling and sibling draining are retained from v1. The pilot launches at most two single-core jobs, seeds 6201/6202, one canonical carry-center run and one frozen canonical RL run. No recurring workflow or legacy-goal claim is introduced.
- Decision timing includes forecast/observation, reset/recovery PFO, projection/validation, actor, lower and guard costs. Plant and learning costs are separately recorded. Comparison reconciles 75 controls, 14400 seconds, cumulative/regional TTT, reward, reference/PFO identity and timing; matched probes remain diagnostics rather than replay labels.

## Evidence And Remaining Work

- Independently parsed all 11 production files with `ast.parse` and verified the
  complete frozen snapshot: 151 files, 3,975,269 bytes, manifest SHA256
  `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
- `budget_runtime.py`, `td3.py` and `freeze_runtime.py` are byte-identical to v1.
  The six other v1 production hashes checked match its final review. No v1 file
  or result was modified by this review.
- Inspected the finalized controller tests and implementer's report of **54
  passing fake-runtime cases**, plus the six admission tests reported passed.
  Also read [env_test_report.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_carry_plan_20260929/env_test_report.md)
  and all four associated test sources: environment 16, run contracts 15,
  runner 10, pilot 12. That report records **53 passed in 35.697 seconds**, with
  zero failures/errors/skips. Combined reported count is **113**. No unchanged
  suite was rerun; consolidated JUnit verification remains pending.
- The new runner tests use real Torch/TD3 with synthetic transitions and compare
  all learner/replay tensors, action sequences and exploration RNG on paused
  and terminal resume. They assert 2380 updates, 150 transitions, two true
  terminals, and no learner mutation during frozen evaluation. Pilot tests
  mock subprocesses and cover two-worker staging, STOP, draining, duplicate
  completion, model provenance and evidence hashes. These sources substantiate
  the reported scope without claiming physical performance. No additional
  blocker was found in this test review.
- The new tests do not close the two P2 findings. Admission fixtures lack state
  identity/record coverage. Pilot child fixtures always use physical guard;
  the run-contract drift test changes only the top-level guard and therefore
  tests internal inconsistency, not a consistently configured `h3` child
  presented to a physical-only plan.
- Independently inspected [smoke_resume.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/smoke_resume.json),
  SHA256 `9fa78405a52357339dc90dc7dc8cf274d78af9dadc69bad2bbfcf007907e4c5e`.
  `passed=true`, serialized resume, 2367 observation features, environment step
  7 after actual steps 5 and 6. Rechecked its complete source pins against the
  current eleven production files and verified snapshot identity. Both audits
  have valid physical/budget execution, physical guard, no fallback, and no
  terminal. Reference/PFO sequence is `pfo_initial/1`, then `previous/0`.
- Smoke interval TTT is 20.650868151393297 then 20.61315248655373; the first
  reward is -0.20650868151393298 and the cumulative difference reconciles.
  The ordinary-step anchor `[-100.37566638955153, 5750]` equals the prior
  executed request, differing from both prior achieved budget and current
  witness `[-107.16535887174751, 5639.450869584549]`. This directly exercises
  the intended carried-budget distinction.
- The pinned smoke producer asserts exact restored observations, rewards,
  terminal flags and controls at both steps, plus explicit committed-dual
  equality at the first step. The second step has no separate double-precision
  dual assertion; its next-observation equality includes the observed dual.
  Parity evidence is the successful producer assertions; restored counterparts
  are not separately stored in the artifact and were not recomputed here.
- After admission repairs, inspect the consolidated regression results and
  actual matched-state records at 5/30/50 before final admission. Preserve this
  smoke artifact under its recorded pins; later source changes need explicit
  evidence/version handling. Matched probes and training have not launched.
- Preserve the disclosed partial observability of PFO memory, endpoint-only
  queue exposure, and descriptive concurrent timing. None supports a
  generalization, causal speedup, or legacy performance claim.
- Plan task 6 still needs final clipping/reached-state/Q diagnostics and the
  descriptive comparison with frozen v1 native/center under the same plant
  contract. `compare_runs.py` currently covers carry-center versus carry-RL;
  the existing v1 analysis script is hardwired to v1 format/seeds. The saved
  traces and learner checkpoint provide inputs for later read-only analysis.

## Reviewed Source Hashes

| File | SHA256 |
|---|---|
| budget_controller.py | `0402511ae8e965e315b4b9808e23bbdc9c2fd274666a983b2a3303e6b916f9c2` |
| budget_env.py | `6ccb4ef400106e29282797ec5d69b7a8dbeee5a1310bd7d6f97b7c7e0c7ee2c4` |
| budget_runtime.py | `6ae552bad2c092af9d2ab55b0c81e34558754861217fe427788820689800e9a6` |
| build_preflight.py | `80a0856946f80637bcb2e562196c4aaaa5e1084583e8480644f3577b08543610` |
| compare_runs.py | `ad44ddb2794ee232a025412379930e8275b4a5948b44d54296a3ddf63a56e485` |
| freeze_runtime.py | `8a13455b1d87a117eec7f97786f7996786bf49c0d8aaef6e2a3ef8c1cca608ab` |
| probe_carry.py | `58f5a05cf197250dba7f4ff0afa4323bb7dfd6e6369c5f064ed7da73fd160c52` |
| run_budget.py | `11f1d07d1c02eecd945ebf9923394a5e9daf115f1086a0ceaa047af41823d193` |
| run_pilot.py | `019b0ef40244346b7aa2d2dcd563a0cbabb26f96a0978177f1118a65e3b79e3c` |
| smoke_budget.py | `3b324a61ae8f1d929179b285f7c8ff8265009f7e03be84a8aa53a3cbc5525659` |
| td3.py | `746b9c6c45aba37cf9dc169274cb64a22ec51fcb1b2c5dc06f2d7dd3104ee1b3` |

Only this report was written. Production, tests, frozen artifacts, git index and
branch were left untouched. No numerical worker was started.

## Initial Review Finalized

2026-09-29: This initial review is complete. Both original P2 findings and the
reviewed source-hash table above are preserved unchanged. `CODE_REVIEW: HOLD`
remains in force; controller spec/quality and the reviewed smoke remain PASS
within their recorded scope. No final pilot admission is granted.

The earlier pending consolidated-JUnit status is now resolved for this original
version. Independently parsed [preflight_tests.xml](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/preflight_tests.xml):
113 declared and actual test cases, zero failures/errors/skips, 39.581 seconds.
Suite counts reconcile to controller 54, environment 16, run contracts 15,
runner 10, pilot 12, and preflight 6. Artifact SHA256:
`54e94575cf53c8a2da6fdb5a6c2285fbe1d5f538cee50ea003fb932dad8b8a4c`.
No suite was rerun by this reviewer. Passing original tests do not close the
two uncovered admission findings.

The user assigned repairs to Heisenberg, limited to `build_preflight.py`,
`run_pilot.py` and covering tests. A precise fix report will trigger scoped
re-review. Preserve the original smoke/JUnit artifacts and their identities;
the planned fresh admission subdirectory will hold smoke, JUnit and matched
probes for the repaired source pins. Those future artifacts have not been
reviewed or admitted here. The user confirms no training has launched.

## Round 1 Scoped Closure

2026-09-29. Reviewed [review_fix_round1_report.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_carry_plan_20260929/review_fix_round1_report.md)
and the four authorized files against the two original P2 findings. This
closure supersedes the earlier HOLD for the repaired source hashes below;
the original findings, initial verdict and evidence remain preserved history.

**CODE_REVIEW: PASS**

**NUMERICAL_ADMISSION: PENDING**

Both original P2 findings are closed. No additional blocking defect was found
in this scoped review. Existing controller spec/quality conclusions are
unchanged. This is code readiness only, not final pilot admission.

### P2 State And Evidence Identity: Closed

[build_preflight.py:33](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/build_preflight.py:33)
now checks completion and settings step identities against each expected
5/30/50 state, requires the separate settings file to agree, and enforces the
exact two modes and five labeled actions. Every expected summary identity must
appear once, with exactly the ten corresponding detailed JSON files. Detailed
records must match settings, mode/action identity and original/reached times.
The original copied-step-30-at-step-50 case therefore rejects directly.

Detailed physical/budget execution checks are reconciled with executed points,
selection points, executed/achieved budgets and selected TTT. Reference source,
PFO counts and guard semantics are checked per mode/state. Influence is derived
from nonzero actions' executed physical points versus the same-mode zero action
at the producer's 1e-9 threshold; summaries and completion claims must agree.
[build_preflight.py:139](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/build_preflight.py:139)
counts distinct validated states in a set, requiring at least two.

The reader at
[build_preflight.py:125](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/build_preflight.py:125)
hashes JSON before and after reading and binds those validated bytes. With one
review, the manifest includes 39 paths: 30 details, three settings, three
completions, smoke, JUnit and review. Existing launch-time hash verification
then checks those detailed artifacts as well.

Inspected all 19 preflight tests, including duplicate state identity, missing
or mislabeled summaries/details, action/mode mismatch, falsely claimed
influence, multiple changed actions at only one state, invalid detailed
execution, PFO/guard inconsistencies, exact evidence membership, and mutation
during JSON validation. They assert refusal without writing an admission gate.

### P2 Physical Pilot Guard: Closed

[run_pilot.py:13](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/run_pilot.py:13)
now requires the physical plan guard in both completed-child settings and the
embedded coordinator contract after the common run validation. Consistently
configured `h3` train, center and RL children cannot pass this acceptance path.

[run_pilot.py:56](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_carry_20260929/run_pilot.py:56)
validates all existing completed children across stages before writing
plan/process metadata or launching any sibling. Thus an incompatible center or
later-stage RL completion is rejected even when training remains unfinished.
New child commands explicitly pass the plan guard; newly exited children use
the same validator before progression. STOP and draining behavior is retained.

Inspected all 16 pilot tests. The new cases construct internally consistent
`h3` children for every role, check both settings locations, verify zero launch
and unchanged child files on resume refusal, assert explicit physical guard
arguments, and reject a newly exited wrong-guard child before the RL stage.

### Verification And Admission Boundary

- Independently parsed the four scoped files with `ast.parse`; no modules or
  tests were executed by that check. Test-method counts are preflight 19 and
  pilot 16. The fix report records 76 unittest cases plus 54 pytest controller
  cases passed (130 total). Those suites were not rerun; fresh consolidated
  JUnit evidence has not been reviewed in this closure.
- Rehashed all eleven production files: the two repairs match the fix report,
  and the other nine match the original reviewed hashes. No broader source
  re-review was performed. Rehashed the original smoke and 113-test JUnit;
  both retain the identities recorded above.
- Fresh smoke, matched probes at 5/30/50 and consolidated JUnit must be reviewed
  under the repaired source pins in the new admission subdirectory. The
  preserved original smoke is not relabeled as evidence for these new pins.
  No numerical acceptance, training authorization or final admission marker
  is issued here.

| Round-1 Scoped File | SHA256 |
|---|---|
| `build_preflight.py` | `a9377bd446599f773974e86618a513cc47ce81b39aebd70f330f461b8a36f082` |
| `run_pilot.py` | `d462083afed597faf6e1021e4a59396603a15f370ffff47d5bf79ac8e1fc3eef` |
| `test_carry_preflight.py` | `85a9117d599a6dc5dea2745b58f027315c8339c4295ac0b4ce3c61704973a918` |
| `test_carry_pilot.py` | `69a377d61278a9bcc996f3dcff482ae317d08c6e73bbdf96421decead1cb8497` |

Only this closure was appended by the reviewer. No production/test edits,
test reruns, evidence rewrites, traffic runs or git mutations were performed.

### Fresh Round-1 Evidence Reviewed

2026-09-29: The newly supplied `admission_v1` artifacts resolve this closure's
pending smoke and consolidated-JUnit evidence checks. No tests or traffic runs
were rerun by the reviewer.

- [preflight_tests.xml](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/preflight_tests.xml):
  130 declared and actual cases, zero failures/errors/skips, 47.192 seconds.
  Confirmed the copied-state, detailed-evidence and consistent-H3/prelaunch
  refusal regressions are present. SHA256
  `bbe799860b7523e8f0f058e2ae4928df21e14c3e4a2cdef6e0afb45388f4056e`.
- [smoke_resume.json](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_carry_20260929/admission_v1/smoke_resume.json):
  PASS, serialized resume, two actual intervals, 2367 features, PFO 1 then 0,
  physical guard and valid physical/budget execution at both steps. The
  ordinary-step anchor equals the prior executed budget and differs from its
  current reference. All eleven production pins and the snapshot manifest
  identity match; the two repaired pins match the closure table. SHA256
  `bc32a0d6c7083a72a4772fab3facdb24526980a113b726f5e952487c39c48d7a`.

`CODE_REVIEW: PASS` remains current. Numerical pilot admission remains pending
matched-state probe results at 5/30/50 and final evidence review. No final
admission marker or training authorization is issued by this addendum.
