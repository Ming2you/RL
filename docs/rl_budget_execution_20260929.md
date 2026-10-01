# Budget RL execution record

## Scope and authorization

The user authorized steps 1-6 on 2026-09-29. This is a new S-DMPC budget
experiment, not authorization to resume legacy DDQN or scheduled automations.
Design: `docs/rl_budget_leader_design_20260929.md`.

## Global constraints

- Preserve legacy code, models, results, STOP files and paused automations.
- Preserve the source project at `C:/Users/alsrj/Documents/Numerical Simulation`.
- Freeze the actual local 2026-09-23 externality-ON upper-budget implementation,
  including uncommitted files, historical plant and frozen scenario inputs.
- Nine players, four freeway groups, H3, at most six lower iterations.
- No physical-control output from RL, no learned prices, no solver differentiation.
- Residual action in [-1,1]^2: NP + 50*a0 veh; NUF + 1000*a1 veh/h,
  clipping only NUF to [0,total_ramp_capacity].
- Retain original physical/control/budget checks and PFO H3 TTT guard.
- Commit only selected dual state; PFO fallback retains incoming prices.
- Train on requested actions and actual interval -TTT/C, gamma=1.
- True terminal is 14,400 seconds (5 warmup + 75 control intervals).
  Truncation is not terminal. Invalid reference is not a zero-cost success.
- Do not pool old DDQN replay or reuse its baseline/acceptance threshold.
- At most eight total simulation workers; initially single-process lower solves.
- Tests and budget-effect gates precede training. No arbitrary long collection.
- Full evaluations have frozen weights/preprocessing, no exploration, identical
  scenario/forecast/initial state/constraints and measured end-to-end runtime.

## Task 1: Freeze source and inputs

Located actual baseline in the separate local Numerical Simulation project.
GitHub S-DMPC branch ends at 2026-09-22 and does not contain the required wrapper.
Create a byte-preserving, hash-verified runtime snapshot and bootstrap smoke test.
Record source Git revision plus dirty/untracked provenance, not revision alone.

## Task 2: Freeze comparison contract

First scenario: sweet_170_incident_w from the new S-DMPC frozen protocols.
Record runtime versions, source/input hashes, model and controller contracts.
Separate train, validation and held-out evaluation; do not claim cross-scenario
generalization from one deterministic scenario or identical forecast seeds.

## Task 3: Separate execution wrapper

Separate reference preparation, request generation, side-effect-isolated budget
evaluation and selected-path commit. Existing candidate generation must reproduce
baseline requests, physical controls, TTT, residuals and price history. Zero
residual must equal current mapped PFO budget. Test failure/order isolation.

## Task 4: Action influence gate

At free-flow, buildup, incident and recovery states, compare center and signed
budget-axis requests from identical snapshots. Log requested/achieved/executed
budgets, slack, binding, physical response, guard reasons and multi-interval TTT.
If no noncentral action changes accepted physical control, diagnose before TD3.

## Task 5: Bounded TD3 pilot

Use a small actor/twin critics and replay of actual sequential transitions with
exploration during collection. Preserve checkpoints and episode/terminal state.
No fixed-tail labels. Freeze normalization from physical scales/training only.
Do not continue blind training if the action influence gate fails.

## Task 6: Full-run comparison

Compare original up-to-three candidates, center-only lower solve and RL one-budget
policy through normal reset, all 75 control intervals and 14,400 seconds. Include
PFO, lower solve, guard and actor costs. Report residual inventory, queue exposure,
fallback, convergence separately from traffic performance; preserve failures.

## Progress

- 2026-09-29: Created branch codex/sdmpc-rl-budget-20260929 from c40eb9d.
- No related Python runners found at initial process check.
- Task 1 complete: 151 frozen files, 3,975,269 bytes; snapshot manifest
  07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005.
  Original project and legacy paused runs remain unchanged.
- Task 2: isolated numpy 2.3.5 / scipy 1.16.3 / existing torch 2.14.0+cpu.
  Historical source-project dependency versions differed; old performance numbers
  are reference only. New pilot train demand seeds 6101/6102 use disclosed +/-2%
  flow multipliers; validation uses 6103. Canonical evaluation uses the unchanged
  stored forecast, seed42. This is a development benchmark, not an untouched test.
- Task 3: real native-wrapper parity passed at saved steps 5 and 30, including
  requests, controls, TTT, original residuals and every inner price/gradient row.
  Actual center/NP-plus evaluation-order and persistent-price isolation passed.
  Reset/checkpoint observation parity passed (2362 physical-scaled features).
  Three review issues in audit retention/resume admission/source verification are
  being fixed before long runs. Hidden PFO coupling memory is checkpointed but
  not entirely observed by the MLP: Markov sufficiency is not established.
- Task 4 complete (bounded influence gate, not performance acceptance): saved
  steps 5/10/30/50, five actions per state, three actual intervals per branch.
  All four states have noncentral actions that change physical control. Examples:
  step30 NUF-minus tail -0.3049288 veh*h vs center; step50 NUF-minus -0.4710990.
  These fixed-continuation diagnostics are not optimal Q labels and are not
  added to replay. Full-run traffic improvement remains unknown.
- Task 5 preparation: TD3 synthetic unit suite passed 63 tests. Real training
  remains held until runner review and wrapper fixes pass.
- Task 6 pending. Legacy 5% target/absolute TTT cutoff is not transplanted here.

## Pre-pilot review and execution admission

- Wrapper independent review: three P2 repairs for detached candidate diagnostics,
  checkpoint contracts and snapshot verification before import. No physical solver
  or candidate-selection change requested.
- Runner independent review: HOLD pending repairs for paused-child completion,
  comparison role identity, exported model provenance, dependency-stable resume,
  and finite/consecutive trajectory reconciliation. TD3 equations and requested-
  action replay were not found defective. See the versioned review reports.
- Final admission requires regression tests, repaired native parity, serialized
  actual-step resume smoke, influence/isolation evidence and independent re-review.
  `build_preflight.py` hashes evidence and pins implementation before launch.
- Planned pilot: two sequential training episodes (6101, 6102), frozen validation
  (6103), and native/center/RL canonical evaluations. At most three one-core child
  processes, no recurring automation. Root or child STOP prevents stage advancement.
- Host CPU: Intel Core i7-12700K, 12 physical / 20 logical cores (queried locally).
  Concurrent distinct-affinity timings are measured operating costs, not a clean
  isolated speedup benchmark. The logical-core class/topology was not established.
- No training or full evaluation has started as of this pre-pilot review entry.

### Repaired preflight evidence

- Consolidated regression: 130 tests passed, zero failures/skips, 15.67 seconds.
  JUnit: `results/sdmpc_rl_budget_20260929/preflight_tests.xml`.
- `smoke_resume.json`: Torch serialization and restoration into another environment
  reproduced the next actual interval's observation, reward, control and committed
  prices exactly. 2362 features; requested action [0.25,-0.25] preserved in replay.
- `parity_step30_repaired.json`: native budget requests, every inner gradient/price
  row, selected controls, residuals and TTT still match the preserved original.
  Recorded wall times 41.617/43.567 seconds are parity-test costs, not speedup proof.
- Wrapper and runner fix reports record repaired admission/provenance/lifecycle
  contracts. Independent final re-review precedes pilot launch.
- Queue near-capacity duration uses explicitly labeled 180-second endpoint sampling
  at 90% of ramp/boundary capacities. It is an estimate, not exact substep exposure;
  summed values across queues are queue-seconds, not elapsed time.

### Pilot commands (after final review)

From the RL repository root, using the isolated dependency directory:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_budget_20260929/build_preflight.py --evidence results/sdmpc_rl_budget_20260929 --review work/sdmpc_rl_budget_20260929/final_pilot_review.md --output results/sdmpc_rl_budget_20260929/preflight.json
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_budget_20260929/run_pilot.py --output results/sdmpc_rl_budget_20260929/pilot_v1 --gate results/sdmpc_rl_budget_20260929/preflight.json
```

Resume only with `--resume`, unchanged pinned source/dependencies and no live
related runner. Root/child STOP files are honored; do not remove a user STOP.
Completed children are validated and skipped, never recollected. The pilot has
two training episodes and four evaluations, then exits regardless of improvement.
It does not reinstate the old indefinitely repeating DDQN authorization.

## Pilot v1 launch

- Final review found and closed a further configuration-provenance gap: dataclass
  serialization omitted runtime-added physical/MPC settings. Contracts now reuse
  the frozen runtime's lossless `to_plain_dict`; incompatible checkpoint version 1
  is refused. Global serialization and physical solver behavior were not changed.
- Final regression: 135 passed, zero failures/skips, 14.77 seconds. The previous
  130-case JUnit remains archived as `preflight_tests_before_dynamic_config_fix.xml`.
- `smoke_resume_config_contract.json` passed actual serialized next-step parity
  under the corrected contract. Prior smoke and all probe results are preserved.
- Independent final review: SPEC / QUALITY / PILOT_REVIEW PASS. Admission is for
  this bounded pilot, not performance or convergence acceptance.
- Launched `pilot_v1` using the commands above. Orchestrator PID 29380; initial
  children train 43056, native 15772, center 43644. Process identity and start time
  are in `pilot_v1/process.json` and the per-child process records.
- Initial controlled intervals completed in all three children without errors.
  Production Python files, source snapshot, gate and experiment settings are frozen
  for the duration of the run. Legacy jobs/schedules remain untouched and paused.

### Admission report addendum

The independent reviewer appended one documentation-only bullet after the gate
was built, reporting a successful read-only gate dry-run. The review was already
PASS before launch; its scope, findings, verdict and source pins did not change.
All other evidence hashes still match. Do not rewrite the active gate to hide this:

- Gate-recorded review hash: `33cf605ad33098e8a672f9cb0ae2a4fb7ec0eb50408616671ee83cf2870e9acb`.
- Final review hash: `6d991988af22c32fdb0a5a1db66a0e9530ae5d4a10294d7877127e7f06150eff`.
- Reviewer confirmed the exact documentation-only change and no further edits.
  Active configuration/source pins and all underlying diagnostic/test results are
  unchanged. This note preserves the provenance discrepancy explicitly.

Resolved without modifying the gate or experiment: preserved the review including
the extra bullet as `final_pilot_review_with_dry_run_addendum.md`, then removed
only that post-admission bullet from `final_pilot_review.md`. Its SHA256 now exactly
matches the gate-recorded `33cf605a...`; the addendum copy matches `6d991988...`.
Both versions and this history remain available. All admission evidence is again
byte-identical to the active gate's recorded hashes.

## Completed training and intermediate diagnostics

Both training episodes finished: 150 actual sequential transitions, two true
terminals, 119 critic updates and 59 delayed actor updates. Frozen model SHA256:
`56d1eec1c0716e15d4ee33e54136b9c003cf6d10e6ad341324815d5103b15785`.
Canonical RL and held-out-seed validation are running from this same model without
updates. Training TTT is not a canonical comparison: demand is perturbed and
actions are exploratory.

Read-only diagnostics are in `pilot_v1/training_diagnostics.json`, generated by
`work/analyze_sdmpc_budget_pilot_20260929.py` outside the pinned runner directory.
Replay actions/rewards/terminals were reconciled against the completed traces.

- Fallback increased from 19/75 to 49/75 training intervals. Budget infeasibility
  and H3 TTT rejection overlap; their reason counts must not be added as distinct
  fallback events. NUF clipping occurred in 29/75 and 48/75 intervals.
- At logged training states, the final actor varies little: NP action range
  [-0.7642,-0.6964], NUF [0.8570,0.9053]. This is not yet evidence about its
  closed-loop evaluation state distribution.
- Mean Q1/Q2 are -0.9418/-0.9622, versus realized behavior return-to-go -26.8799.
  Exploratory future actions changed during training, so behavior returns are NOT
  current-policy Q ground truth. Long-horizon value propagation is unestablished;
  these numbers do not prove a calibrated value error or algorithmic failure.
- Only 119 updates have occurred. This is a bounded integration/learning pilot,
  not a converged policy or a test of the ultimate capacity of TD3.

## Discussed alternative: previous executed budget reference

The user asked whether per-step PFO could be removed in favor of learning changes
to the previous budget. This is a proposed separate experiment, not an approved
change to the active pilot. The current model, source pins and settings remain
unchanged.

PFO currently supplies the mapped achieved-budget reference, lower-solver starting
control/linearization point, and H3 TTT reference/fallback. Replacing only the
budget reference while retaining those PFO calls would not remove their cost.
In completed canonical evaluations, mean PFO time was 3.5414 seconds/step for
native and 3.3445 for center (6.65% and 13.11% of their respective decision wall
time). These are concurrent-run measurements, not an isolated speedup guarantee.
Changing the initial point can also change lower-solver cost and TTT.

A candidate design is previous EXECUTED budget plus a bounded actor increment,
with previous executed control projected to current bounds as the initial point.
Do not accumulate a rejected requested budget as if it had been executed. NP is
state-dependent signed H3 net service, not a conserved inventory; prior feasibility
does not imply current feasibility after demand/incident/state changes. Define
initialization, current-state validation, feasible recovery and budget drift limits.
Using PFO only on reset/recovery is an explicit variant, not strictly PFO-free.
Removing the H3 performance guard is distinct from removing physical constraints.
The PFO-specific observation and action contract would change, requiring retraining.
Whether this improves learning or full-run performance remains untested.

## Pilot v1 final result

The parent exited successfully with `PILOT_COMPLETE`; both training episodes and
all four evaluations finished. Tasks 1-6 are complete as a bounded implementation
and evaluation pilot, NOT as performance acceptance. No subsequent experiment,
legacy DDQN loop, or recurring automation was started.

All canonical rows start from the same normal reset, include identical warmup,
run all 75 controls through 14400 seconds, and use profile SHA256
`74778a836d4bbe736c8c1f533164c612b0b5ef916313b1a9617866e964a984cb`.
The reference is this frozen native S-DMPC, not the legacy P-Stack/DDQN contract.

| Mode | Total TTT (veh*h) | Improvement vs native | Decision mean / median (s) | PFO fallback |
|---|---:|---:|---:|---:|
| Native, up to 3 budgets | 5508.864413550166 | Reference | 53.2367 / 51.6104 | 0/75 |
| Center, 1 PFO-achieved budget | 5547.959809795623 | -0.7097% | 25.5202 / 25.3992 | 4/75 |
| Frozen TD3, 1 proposed budget | 5623.784268350441 | -2.0861% | 33.3828 / 34.1031 | 66/75 |

Center is a lower solve at the PFO-achieved budget, not direct PFO-only execution.
Frozen TD3 is also 1.3667% worse than center. Its freeway TTT is 4097.1098 versus
native 3803.3174, while urban TTT is 1526.6745 versus 1705.5470. Net loss is
114.9199 veh*h; the regional difference is descriptive, not an identified cause.
Terminal inventories are native 421.3660, center 437.6813, RL 438.5515 vehicles.
Sampled near-capacity totals are 4500/3600/4680 queue-seconds respectively; these
are endpoint estimates across queues, not exact substep durations.

Validation seed 6103 finished with TTT 5522.278099382, fallback 65/75 and unchanged
119 learner updates. Its demand profile and warmup differ, so its raw TTT must
not be compared against the canonical native row as an improvement claim.

Canonical RL actions vary narrowly: NP [-0.7655,-0.6960], NUF [0.8567,0.9061].
NUF was clipped in 73/75 steps. There were 56 budget-infeasibility and 63 H3 TTT
rejection events, overlapping within 66 fallback steps. Only nine lower responses
were executed. These observations motivate checking action feasibility/clipping
and value learning before any large collection campaign. They do not isolate
PFO anchoring as the cause of the performance loss.

All candidate solves reached the six-iteration limit, and all three canonical
runs retain `converged_count=0`. Physical/execution gates passing is not proof of
optimizer convergence. The policy has only 150 transitions and 119 critic updates;
neither convergence nor generalization nor nonlinear-price learning is claimed.

### Runtime diagnosis: budget search removed, lower solve retained

Measured per control interval, including PFO, actor, lower response and guard:

| Component / count | Native | Center | RL |
|---|---:|---:|---:|
| PFO mean wall seconds | 3.5414 | 3.3445 | 3.2880 |
| Lower mean wall seconds | 49.5081 | 22.0302 | 29.6073 |
| Actor mean wall seconds | Not an actor | Not an actor | 0.000391 |
| Guard mean wall seconds | 0.0207 | 0.0072 | 0.3466 |
| Total lower solves / episode | 224 | 75 | 75 |
| Scalar traffic rollouts / interval | 229.03 | 123.24 | 172.73 |
| Tangent rollouts / interval | 14.00 | 5.25 | 6.08 |
| FD columns / interval | 69.77 | 32.51 | 40.81 |
| Local QP calls / interval | 161.28 | 54 | 54 |
| Quantization trials / candidate solve | 1.18 | 1.00 | 6.68 |

Mean decision time fell about 37.3% from native to RL, not to one third. A single
budget still incurs six inner iterations, nine cheap local surrogate QPs per
iteration, global prediction/derivative refresh, line search and feasibility
restoration. The scalar rollout, sensitivity and repair work dominate, not actor
inference. Candidate caches are retained within an interval, so the native three
solves are not three independent equal-cost copies.

Compared with center, RL has the same 54 local QPs but more rollout and repair
work. Its many failed budgets trigger additional post-quantization restoration
trials before the guard falls back; those computations have already been paid.
This is consistent with the observed 29.61 versus 22.03 seconds in the lower
phase. Different closed-loop states and concurrent CPU placement also affect
timing; these results do not causally isolate each extra second to the budget.
The final selected-action multiplier diagnostic is NOT a substantial bottleneck:
its RL total is only 0.7973 seconds for all 75 steps, with zero added scalar or
tangent rollouts due to reuse. Do not prioritize removing it for speed.

PFO accounts for only 9.85% of RL decision wall time in this run. Removing it may
help, but cannot on these measurements explain away the dominant lower cost;
a new warm start may also alter that cost. A PFO-free variant needs its own
reference/observation/guard contract and retraining, as discussed above.

### Verification and retained evidence

- `pilot_v1/comparison.json` was independently recomputed and exactly matched
  using the same absolute run paths. An initial read-only assertion used relative
  paths and differed in the serialized folder field; no experiment file changed.
- All five child completion contracts, runtime versions, 151 frozen source files,
  production source pins and the exported model hash were revalidated successfully.
- The 135-test preflight, parity/isolation probes and serialized resume smoke remain
  the admission evidence. No production code changed during or after the pilot.
- Parent/child processes exited; stdout reports both final children exit 0. No
  numerical worker is intentionally left running by this pilot.
- Reproducibility records: `pilot_v1/plan.json`, per-child settings/checkpoints,
  completed traces/summaries, training diagnostics, model/hash and comparison.
  Full traces and checkpoints remain local under ignored `results/`; this document
  does not imply that large artifacts have been uploaded to GitHub.

Next work is a separately versioned decision, not an automatically launched run:
test a feasible previous-executed-budget reference, separate H3 performance guard
from physical validity, and diagnose bounded action execution and critic training
before adding large data batches. Any follower performance optimization requires
matched-state parity and full-run validation, not merely fewer iterations.

### Carry-reference follow-up completed

See `docs/rl_budget_carry_plan_20260929.md` for implementation,130-test admission,
matched-state evidence and the completed finite carry-reference pilot. Ordinary
per-step PFO was removed (one initial call retained), previous executed budgets
became the action anchor, and H3 performance rejection was disabled while
physical/budget validity remained mandatory. Training was restarted, not resumed
across the changed contract. Existing v1 sources/results remain preserved.

Canonical carry-center TTT5546.224352, mean13.0439s/step. Frozen carry TD3
TTT18646.151687, mean21.4863s/step after150transitions/2380critic updates.
The learned checkpoint is rejected: nearconstant[-1,-1] actions,68/75physical
reference fallbacks and37final intervals with all ramp commands0. The center
timing improvement is structural, not a learned gain. Full traces, summaries,
checkpoints and reconciled diagnostics are local in
`results/sdmpc_rl_carry_20260929/pilot_v1/`.

The finite follow-up and all workers finished; no next experiment/automation was
started. Next work requires a focused critic/actor and recovery/action-contract
diagnostic, not another blind collection campaign. No legacy DDQN5%claim applies.
