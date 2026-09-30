# Previous-executed-budget pilot v2

## Authorization and scope

The user authorized continuing after discussing removal of per-step PFO and the
cost of difficult lower solves. Preserve pilot v1, its source hashes and results.
This is a bounded development experiment, not revival of the paused legacy DDQN
loop, scheduled automation or a claim about the legacy 5% contract.

## Hypotheses

1. Carrying the previous executed budget and projected previous control can avoid
   ordinary per-step PFO without making current-state execution invalid.
2. The v1 H3 performance guard and clipping can mask nominal actions; physical
   feasibility and short-horizon performance rejection must be measured separately.
3. More critic updates on a small real sequential batch may be more informative
   than another long collection campaign. This is not assumed to fix learning.

## Global constraints

- Work on the existing feature branch in a separate `work/sdmpc_rl_carry_20260929`
  namespace. Do not edit frozen snapshot or v1 production files/results.
- Reuse frozen physical runtime, demand, incident, warmup, nine players, H3,
  six inner iterations, physical constraints, price updates and TTT accounting.
- Strict execution checks remain. A failed requested budget must remain in the
  record and replay; never relabel it as successfully executed.
- New reference policy: PFO on the initial controlled step or when the projected
  previous control fails current-state physical validation. Otherwise no PFO.
  Report this as PFO-on-reset/recovery, not strictly PFO-free.
- The actor anchor is the prior executed budget, initialized from the first valid
  reference witness. Actions keep the original scales [50 veh, 1000 veh/h].
  NUF is bounded to [0, ramp capacity]. Signed NP must not be clipped to zero.
  Track clipping and divergence from current witness explicitly; do not silently
  replace the requested budget or loosen physical constraints.
- Project the previous physical control to current coordinate bounds/grid before
  checking it. A failed reference recovery fails closed, not as a zero-cost episode.
- Two explicit guard modes: physical/budget validity only; or additionally H3 TTT
  not worse than the current valid reference. Guard mode is part of contracts.
- The fallback uses the current validated reference's achieved budget and incoming
  prices; a successful lower response commits its own requested budget and prices.
- Observation must include reference identity, carried budget, current reference
  controls/budget/TTT, previous execution and time. Do not reuse the v1 policy or
  replay across the changed action/observation contract.
- True sequential -interval TTT/100 reward, gamma=1, true terminal only. Preserve
  complete checkpoint/RNG/contracts, STOP handling and no duplicate collection.
- No more than eight numerical workers in total; use at most three one-core jobs
  for this bounded pilot. No new recurring automation or automatic open-ended run.
- Include all PFO/recovery/projection/validation/actor/lower costs in timing. Timing
  on different states is descriptive; use matched-state probes to isolate changes.

## Tasks and gates

1. Implement and review the separate controller/reference/action contract, with
   fake-runtime tests for rejected requests, guards, signed NP and safe fallback.
2. Implement observation, environment checkpoint/resume and finite runner contracts.
   Verify actual next-step parity across serialization and unchanged physical TTT.
3. Run matched-state probes at saved control states 5, 30 and 50 with zero action
   and +/-0.25 on each axis. Compare PFO/current reference with carried/previous
   reference; record valid responses, requested/executed budgets, H3 rejections,
   lower solver effort and measured time. These are diagnostics, not Q labels.
4. Admit training only if all executions remain physically valid, the new path
   avoids ordinary PFO at the sampled noninitial states, and at least two sampled
   states have a nonzero action causing a distinct executable physical response.
   Otherwise preserve evidence, diagnose the failed gate, and do not launch a
   large collection run or silently change the contract mid-experiment.
5. After gate/review, run two full sequential training episodes from scratch with
   declared demand seeds 6201/6202, a zero-action carry-center full run, and one
   frozen canonical RL evaluation. The carry-center separates learning from the
   new reference/guard structure. Use
   physical-only performance guard mode (budget checks retained). Use 20 TD3
   gradient updates per new transition once batch size 32 is reached. Record this
   change explicitly; v1-versus-v2 is not a single-factor causal ablation.
6. Reconcile 75 controls / 14400s, TTT, fallbacks, clipping, reached states, terminal
   inventory, Q diagnostics and runtime. Compare with v1 canonical native/center
   reference descriptively under the same plant contract. Stop this finite pilot
   after reporting; further runs require a separately justified next experiment.

## Progress

- Plan created before implementation. Pilot v1 is complete; no numerical worker
  is expected to remain active. Production changes in this plan have not started.
- Implementation checkpoint: separate controller/environment/runner/probe modules
  created. V1 source preservation revalidated. Admission-unit checks: 6 passed.
  Controller tests, environment/runner regression and independent code review are
  in progress. No real traffic simulation or training launched for v2 yet.
- Initial integrated evidence: 113 unit tests passed and actual two-interval
  serialized resume matched exactly (2367 features; initial PFO1, ordinary PFO0).
  Independent controller spec/quality passed. Two P2 admission issues were found:
  distinct sampled-state evidence was not enforced; resumed child guard was not
  required to match the pilot. Both repaired with focused regressions; original
  evidence preserved at the results root, not reused under changed pins.
- Fresh `results/sdmpc_rl_carry_20260929/admission_v1/` evidence: 130 tests passed
  (47.19s, no errors/skips), actual two-interval serialized smoke passed again.
  Scoped independent re-review precedes the matched-state batch. No training yet.
- Scoped re-review closed both findings: CODE_REVIEW PASS. Started three one-core
  matched probes using the following command (steps/masks 5/1, 30/4, 50/16):

```powershell
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_carry_20260929/probe_carry.py --output results/sdmpc_rl_carry_20260929/admission_v1/probe_step30 --step 30 --cpu-mask 4
```

Each checks ten independent one-interval branches (two reference modes x five
actions), from the identical saved state for that step. Source is pinned. These
results are not added to replay and cannot establish long-horizon improvement.

### Matched-state findings

All three probes completed (30 actual single-interval branches). Every execution
was physically valid and satisfied its executed budget; requested failures remain
recorded separately. Detailed-record validation passed under current source pins.
Nonzero executable action influence exists at all three states. At step30 four
nominal actions still produce the same physical response: action aliasing remains.

| Saved step | PFO-reference mean decision (s) | Carry mean decision (s) | Carry PFO calls / 5 branches | Carry fallback |
|---|---:|---:|---:|---:|
| 5 | 29.8652 | 29.3976 | 5 (independent initializations) | 1/5 |
| 30 | 27.1293 | 19.2854 | 0 | 0/5 |
| 50 | 27.7322 | 20.9673 | 0 | 0/5 |

Initial-step behavior still legitimately uses PFO. Mean scalar predictions for
step30 fell from108.8 to86.0 and step50 from120.4 to96.6. This supports testing
the carried reference, but seed/anchor/guard all differ and concurrent timing is
not an isolated causal speedup estimate. No long-horizon gain or trained-policy
claim is made. All pre-admission workers exited. Final independent evidence
review precedes the finite learner pilot.

### Bounded pilot launch

Final independent `PILOT_REVIEW: PASS`; review SHA256
`e854db2d033105bb385e963a3f4bf5b6320f61fa205e55c809903cc98eba6d36`.
Admission gate binds 39 evidence files, 130 tests and three influential states.
No STOP existed and the output directory was new. Commands:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_carry_20260929/build_preflight.py --evidence results/sdmpc_rl_carry_20260929/admission_v1 --review work/sdmpc_rl_carry_20260929/final_pilot_review.md --output results/sdmpc_rl_carry_20260929/admission_v1/preflight.json
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_carry_20260929/run_pilot.py --output results/sdmpc_rl_carry_20260929/pilot_v1 --gate results/sdmpc_rl_carry_20260929/admission_v1/preflight.json
```

Parent PID33268; initial child launchers train35292 (CPU mask1), center47184
(mask4). Complete command identities/start time are in process records. The
Windows launchers each spawn one numerical runtime child; this is two numerical
workers. Stage2 evaluates the frozen model only after training and center finish.
Production source, admission evidence and configuration are now immutable during
the pilot. Checkpoints and STOP files remain authoritative. Legacy automation and
DDQN experiments remain paused and untouched.

### First completed comparison arm

Carry-center (zero actor increment, no learning) completed the canonical 75-control,
14400-second run and passed the full completed-run contract validator:

- TTT 5546.224352256691 veh*h (native v1 5508.8644: 0.6782% higher).
- Mean decision 13.0439s, median12.5595s, total978.2934s; measured with concurrent
  training, not an isolated timing benchmark.
- PFO1 initial call, no recovery calls; reference fallback3/75.
- Terminal inventory413.6461 vehicles; no convergence certification.
- Compared with v1's PFO-reference single-budget center, TTT5547.9598 and total
  decision1914.0122s, this is nearly identical TTT (-0.0313%) at lower measured
  time (-48.8878%). This result is not attributable to RL learning.

Training/frozen-RL outcomes remain pending at this entry. Do not conflate this
zero-action arm with the learned policy or with the multi-budget native baseline.

### Final outcome: pilot complete, learned policy rejected

All four full trajectories finished. Parent session92361 exited0 after
`CHILD_FINISHED train 0`, `CHILD_FINISHED rl 0`, `PILOT_COMPLETE`.
Final parent status is `completed`; a command-line process inventory found no
remaining carry/budget pilot Python worker. No next experiment or automation
was launched. Tasks1-6 of this finite plan are complete, not a performance goal.

Read-only reconciliation:

```powershell
.venv-torch/Scripts/python.exe -B work/analyze_sdmpc_carry_pilot_20260929.py --pilot results/sdmpc_rl_carry_20260929/pilot_v1 --output results/sdmpc_rl_carry_20260929/pilot_v1/diagnostics.json
```

Output `CARRY_DIAGNOSTICS_SAVED 150 2380`; source/snapshot/runtime pins,
model hash, replay-to-trace actions/rewards/terminal flags and episode TTT return
sums passed. The analysis helper has separate synthetic return-boundary and
zero-budget streak checks. Production code/configuration/evidence remained
unchanged. Admission remains130tests PASS plus actual serialized resume and
30 matched-state branches.

Frozen model SHA256:
`3fde81f528c85dd2ab21850e23a637d922030a46512c6228ac9a1a75364daeee`.
Preserve `pilot_v1/train/model_final.pt`, both episode models, checkpoints,
settings, model_hash.json, traces, summaries, comparison.json and diagnostics.json.
Large results remain local under ignored results/, not uploaded to GitHub.

#### Canonical evaluation

All rows below use the same physical170-incident development scenario/profile,
normal reset, warmup and75controls/14400s. Native and PFO-center are preserved
v1 descriptive references; coordinator/reference/guard contracts differ from
carry v2. This is not the older DDQN/P-Stack5730.7929 acceptance contract.

| Policy | Full TTT (veh*h) | Mean decision (s) | Median (s) | Fallback |
|---|---:|---:|---:|---:|
| Native S-DMPC, up to3 budgets, v1 | 5508.864414 | 53.2367 | 51.6104 | 0/75 |
| PFO-reference center, one budget, v1 | 5547.959810 | 25.5202 | 25.3992 | 4/75 |
| Carry-center, zero increments | 5546.224352 | 13.0439 | 12.5595 | 3/75 |
| Frozen carry TD3 | 18646.151687 | 21.4863 | 21.5153 | 68/75 |

Carry RL is236.1954% worse than carry-center and238.4754% worse than native.
Do not adopt this checkpoint. Center is nearly identical in TTT to PFO-center
at48.8878% lower measured decision time, but this is not an RL learning gain
or an isolated causal timing estimate. Timing includes all actual reference,
initial/recovery PFO, actor, lower and guard work; training/plant costs are
separate. Every carry episode called PFO exactly once initially and never for
recovery. This is not strictly PFO-free. All lower runs have converged_count0;
physical/budget validation is not optimization convergence or traffic stability.

#### Observed failure path

- Final actor saturates nearly at[-1,-1] on all75canonical decisions and on
  all150logged training observations. Canonical NP range[-0.9999903,-0.9999517],
  NUF range[-0.9999998,-0.9999979]. It does not recover the ramp budget when
  queues accumulate.
- Only7canonical lower solutions were accepted;68steps failed requested-budget
  feasibility and executed the projected prior control reference. These are
  physical/budget rejections, not H3 performance rejections. H3 guard is disabled;
  `h3_guard_would_reject` is diagnostic only (32steps).
- Accepted steps17,18,19,33,37,38 reduced NUF approximately6000->5000->4000->
  3000->2000->1000->0; accepted step39 retained0. Step indices are zero-based.
  Fallback steps between these retain/reproject prior physical controls, not a
  freshly optimized recovery policy. NUF and all actual ramp commands are0 for
  the final37consecutive control intervals (steps38-74). Center never reaches0.
- Final inventory is10689.2746 versus413.6461 vehicles for center. Urban TTT is
  15186.9636 versus1810.5905 (+13376.3731); freeway TTT is3459.1881 versus
  3735.6339 (-276.4458). This is principally persistent urban loss, not just
  an uncounted terminal artifact. The objective already includes the within-run
  urban loss; no reward/TTT accounting discrepancy was found.
- Physical feasibility of a held reference does not imply acceptable discharge
  or recovery. PFO recovery is triggered by invalid physical reference, not by
  growing queues or deteriorating network performance, so it never activates here.
- RL lower work averages21.0131s (97.7977% of decision time), actor0.000355s,
  PFO0.04666s. Scalar predictions139.51/step and quantization trials8.12 compare
  with center65.56 and1.32; both still use54local QPs/step. Removing routine PFO
  did not remove the expensive follower solve and failed-request repair work.

These traces identify the executed failure mechanism, but do not causally isolate
actor update ratio, data coverage, guard removal and incremental anchoring.
Those were changed together; additional controlled tests are required.

#### Training and Q diagnostics

Training consists of150transitions, two true terminals,2380critic and1190actor
updates. Seed6201 TTT7928.6825, terminal2526.07, NUFzero15steps; seed6202
TTT32452.3314, terminal16457.30, NUFzero65steps. Perturbed exploratory training
episodes are not canonical policy evaluations. Neither successful recovery
coverage nor adequate learning is established by the update count.

Minimum-critic one-step absolute Bellman residual: mean0.4397, median0.1812,
max28.7615 in scaled reward units. Realized exploratory behavior return mean
is-133.6196; Q1/Q2 means are-12.9100/-12.9516. The discrepancy is a warning,
NOT a direct current-policy calibration error: behavior uses changing exploratory
future actions. Terminal residuals have no continuation-policy ambiguity.
Bootstrapped consistency alone does not certify long-horizon value correctness.
This pilot provides no evidence of nonlinear-price learning or generalization;
the actor outputs two budgets, with the existing lower price mechanism retained.

#### Next experiment, not launched

1. Diagnose critic/actor instability without new traffic collection first: freeze
   these150transitions and test a preregistered one-variable update/actor-delay
   ablation. Report terminal residuals and action saturation; do not treat offline
   replay fit or behavior-return agreement as policy improvement. The hypothesis
   that20updates/transition amplify unsupported action gradients is not yet proven.
2. Separate the incremental-action issue from the fallback issue. At serialized
   on-policy states before budget closure, compare matched sequential branches
   with retained budget, controlled budget increases and the current decrease.
   Continue their future actions explicitly through the remaining horizon;
   do not assign irreversible one-step labels. First capture/replay exact state,
   dual, previous controls and budget memory, not just an approximate traffic state.
3. Test one scoped PFO-free recovery design: a current-state lower optimization
   at the carried budget, rather than indefinitely holding the same prior physical
   controls after an infeasible request. First verify that an executable recovery
   exists and measure its extra solver cost. Do not assert it fixes zero budget:
   that may require a separately declared budget-recovery rule or action contract.
4. Admit a new small training/evaluation version only after the relevant diagnostic
   supports its change. Keep the canonical physical/TTT contract and zero-action
   comparison, report every fallback/recovery cost, and reject persistent all-ramp
   closure unless paired full-horizon evidence supports it. No arbitrary24h batch,
   hidden H3 fallback, unreported bound or automatic repeat of this failed model.
