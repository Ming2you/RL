# Balanced-policy recovery: completed, controller still rejected

## Decision

The bounded continuation in `rl_budget_recovery_plan_20260929.md` is complete.
All five canonical 14400-second evaluations finished and passed reconciliation.
One shared actor was selected before seeing evaluation TTT and used unchanged in
every scenario. No evaluation data entered training.

Actor/critic directional mismatch was recoverable with additional output-head
fitting. The catastrophic ramp-closure pattern disappeared. However, every
scenario still has higher TTT than the matched carry-budget center. This actor
is a diagnostic ablation, **not an accepted controller** or a replacement for
the preserved baseline. No P-Stack, nonlinear-price or generalization claim is
made. The current learner is the shared TD3 budget policy, not the paused legacy
DDQN experiment.

## Full-run results

TTT is in simulator vehicle-hours; lower is better. The comparator is the
zero-action **previous-executed-budget carry center**, not P-Stack and not an
unconditionally fixed numeric budget. Its physical recovery behavior is retained.
Warmup, demand, incident, follower constraints and accounting are matched within
each scenario. All rows cover five warmup intervals, 75 controlled intervals and
14400 simulated seconds, without exploration or a performance fallback gate.

| Scenario | Carry-center TTT | Previous failed RL TTT | Repaired actor TTT | Increase vs center |
| --- | ---: | ---: | ---: | ---: |
| 155 | 3103.011072 | 30692.368191 | 3283.155247 | 5.8055% |
| 170 | 3935.903237 | 33126.802572 | 4403.029230 | 11.8683% |
| 170-incident | 5546.224352 | 33125.635550 | 5808.423445 | 4.7275% |
| 170-skew | 4250.876599 | 33123.575802 | 4577.708381 | 7.6886% |
| 190 | 6604.297017 | 36340.971289 | 7255.832502 | 9.8653% |

The 80.0-89.3% decrease relative to the failed RL policy is recovery from an
extreme failure, not evidence of improvement over a useful controller.

## What was fitted

The existing balanced replay contains 750 transitions: 150 per scenario, from
two complete exploratory episodes per scenario. No new collection was used.
Each actor minibatch contains eight states per scenario, for 40 states total.
Three paired treatments received 375 updates each, or 3000 sampled states per
scenario per treatment, with replacement. These are not new plant transitions.

Both critics and the actor's hidden layers were frozen. Only the actor output
head was trainable. All arms used the same hidden weights, state-sampling
schedule and learning rate, differing only in head/optimizer initialization.
No TD3 joint update or target-network update was called during actor fitting.

| Treatment | Final equal-scenario Q1 | Gap to nine-point grid maximum |
| --- | ---: | ---: |
| Original actor and original Adam history | -6.80659914 | 0.00001148 |
| Original actor and fresh Adam | -6.80663996 | 0.00005264 |
| Zero output head and fresh Adam | -6.80680666 | 0.00021887 |

Initial Q1 was -6.84406281 and the grid gap was 0.03747549. All three treatments
made the NUF action positive on all 750 replay states. The first treatment won
the preregistered frozen-Q1 criterion and alone entered full traffic evaluation.
This is not evidence that its Q1 ranking is correct in the simulator.

An earlier `actor_fit_v1` incorrectly allowed hidden-layer updates. Review
caught this before full evaluation. That artifact and its smoke are retained as
non-admitted evidence. The corrected head-only implementation, invariants,
metadata and tests produced `actor_fit_v2`, which is the only evaluated version.
The initial STOP-handling review finding was also fixed before admission.

## Closed-loop diagnosis

1. The old policy's 69 consecutive all-ramp-zero intervals per scenario became
   **zero** all-ramp-zero intervals in every repaired evaluation. Final inventory
   is 410.8-419.3 vehicles instead of roughly 15600-18000. No ramp floor or hidden
   performance guard was introduced: there are zero reference fallbacks and one
   initialization PFO per repaired run, followed by 74 previous-budget references.
2. Both requested action dimensions still have absolute value at least 0.95 in
   all 375 evaluated decision states. Mean action is approximately `[1,0.9995]`.
   This is an almost constant boundary policy, not demonstrated state-dependent
   coordination. The head-only experiment tests recovery, not optimal capacity
   for learning a differentiated policy.
3. Requested NUF increments are clipped in all 75 steps of every scenario;
   executed NUF is exactly 6000 throughout. NP increases by approximately 50 per
   step, from an initial reference near -112.88 to a final budget near 3637.12
   (-112.95 to 3637.04 for skew). Thus nominal actor variation in the positive
   NUF region need not create different executable budget requests. This
   projection equivalence needs to be audited before trusting continuous-action
   gradients. It does not, by itself, prove an incorrect transition mapping.
4. There are 47,63,71,54,69 distinct exact physical-control dictionaries across
   the five repaired trajectories. These counts are across different states,
   not state-matched response equivalence classes. Follower variation must not
   be confused with learned variation in the leader's saturated requests.
5. No run reports solver convergence, although executed controls pass the
   existing physical/budget checks. The same limitation is present in the
   preserved center. Feasibility is not an optimality certificate.

The first 15 controlled intervals are slightly better than the center, by
0.67-3.52 vehicle-hours. The largest excess loss occurs in controls 30-44 and,
in several scenarios, 45-59. For incident, the five consecutive 15-step TTT
differences are -3.522681, +10.902612, +205.343823, +49.033874, +0.441464.
The corresponding controlled intervals are 900-3600, 3600-6300, 6300-9000,
9000-11700 and 11700-14400 seconds. This is a full-trajectory comparison of two
different reached-state paths, not a same-state causal action label.

### Value calibration on the repaired continuation

The evaluator retained the original critics, stored each observation, and logged
both Q values. Reconciliation reloaded the same actor and critics and reproduced
all 375 requested actions and twin-Q outputs from the archived observations.

| Scenario | Mean min(Q1,Q2) | Mean realized reward-to-go | Mean Q minus return |
| --- | ---: | ---: | ---: |
| 155 | -2.2018 | -14.5291 | 12.3273 |
| 170 | -2.4812 | -20.2116 | 17.7303 |
| 170-incident | -2.7643 | -27.3750 | 24.6107 |
| 170-skew | -2.5188 | -21.0796 | 18.5608 |
| 190 | -3.1625 | -35.9135 | 32.7511 |

Units are reward units, with reward = -interval_TTT/100 and gamma=1. These are
operational prediction discrepancies under the repaired continuation. The old
critic was not refitted to that continuation, and its smoothed TD3 target policy
differs from deterministic evaluation. These numbers alone therefore do not
establish a Bellman error or the ordering of unexecuted actions.

## Independent critic diagnostic

An independent, reviewed critic-only experiment used the same 750 training
transitions, fixed the original final actor, and initialized both target actors
to that final actor. Two arms used identical sampled batches and noise, original
critic/optimizer initialization, 3750 additional critic updates, and 1875 target
updates. Target tau was 0.005 versus 1.0; actor updates were disabled. Each arm
had 30000 draws per scenario. Neither diagnostic critic was exported as policy.

| Condition | Full-replay twin TD loss | Terminal min-Q absolute error |
| --- | ---: | ---: |
| Initial, both arms | 1.1840 | 5.7018 |
| 3750 updates, tau=0.005 | 60.9532 | 41.5798 |
| 3750 updates, tau=1.0 | 860.9204 | 147.4217 |

Terminal errors have an exact one-interval target with no continuation ambiguity.
Exploratory behavior reward-to-go was logged separately and is not mislabeled
as current-policy truth. More updates and faster target propagation did not fix
this controlled replay experiment; simple update-count insufficiency is not an
adequate explanation of its outcome. This does not rule out insufficient state
or action coverage, function-approximation problems, or other target schedules.
The paired diagnostic took 33.93 seconds, not hours of new simulation.

## Measured runtime

| Scenario | Center mean decision s | Repaired mean decision s | Repaired full elapsed s |
| --- | ---: | ---: | ---: |
| 155 | 11.1032 | 11.8680 | 960.72 |
| 170 | 13.7684 | 14.6024 | 1175.80 |
| 170-incident | 13.5721 | 15.4350 | 1229.18 |
| 170-skew | 12.4620 | 13.1156 | 1055.93 |
| 190 | 16.3198 | 16.0579 | 1277.32 |

Actor plus read-only twin-Q inference takes 1.20-1.63 milliseconds per decision.
The lower solve takes 11.65-15.83 seconds, about 98% of decision time. Removing
ordinary budget search does not remove this follower cost. Initialization PFO,
reference preparation and guards are included; reset/full elapsed accounting is
retained separately. These are concurrent one-core jobs, not isolated speed
benchmarks. Five numerical workers were used, below the eight-worker limit.

## Next bounded experiment, not launched

Keep one shared policy/value model and equal five-scenario sampling. Do not
increase data collection duration or deploy this actor merely because its
frozen-critic objective improved.

1. Test the observation and terminal target contract directly: remaining
   horizon, previous executed budget, physical projection and true termination.
   Fit only exact terminal targets first, with a separate terminal diagnostic
   set. If that small task cannot fit, isolate preprocessing/optimization before
   adding bootstrap targets or actor updates. Current replay has only two
   terminal transitions per scenario, so report that coverage limit explicitly.
2. Audit Q invariance for different nominal actions that produce identical
   projected requests at the same state, especially positive NUF at its cap.
   Compare raw-action and effective-request encodings in a versioned diagnostic,
   preserving the physical action contract. Do not silently change the action
   semantics or add a performance guard to completed runs.
3. If the preceding checks pass, collect a small synchronized training-only wave
   under a frozen common policy, one complete episode per scenario. Use its
   realized returns only as values of the specified continuation, not as labels
   that declare an action intrinsically good/bad under all future policies.
   Compare a bounded critic calibration phase with bootstrapped training before
   allowing the actor to maximize either critic. Keep canonical evaluations out
   of fitting and distinguish policy evaluation from policy improvement.
4. Admit a new actor only after terminal calibration, time-resolved value error,
   executable-action sensitivity and saturation diagnostics pass predefined
   checks. Then repeat all five canonical full runs with one frozen model and
   report each scenario, not only an aggregate. Matched branch tests may test
   local rankings but must disclose their fixed continuation and cannot replace
   full-run evaluation.

The measured evidence supports separating actor tracking, value calibration and
projection-aware action learning. It does not establish that the underlying
leader/follower idea is impossible or that adding more data alone will solve it.

## Verification and preserved evidence

- Shared base model SHA-256:
  `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`.
- Selected actor `actor_fit_v2/continue.pt`, SHA-256:
  `82bae10a4c27b2723797be71ae41a1076aad72c8a5f86aec25cbd352fbc3f0ff`.
- Original 151-file physical snapshot manifest SHA-256:
  `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
- Reconciliation script SHA-256:
  `2075fd16c7b198016b223066f53a546eb15f0626af667d3dd4389f52b5977226`.
- Combined old/new regression: 523 tests passed, 57.89s. The final analyzer fixes
  have a separate eight-test passing suite. Independent reviews and scoped
  re-reviews passed after their findings were repaired.
- Real two-interval save/resume smoke reproduced observations, actions, controls,
  rewards, Qs, budgets, TTT and dual state exactly. This tests recovery, not
  full-run performance or generalization.
- Full evaluations exec63728,27237,15994,37443,65604 all exited 0. Reconciliation
  exec99996 exited 0 with `RECOVERY_ANALYSIS_PASS`. A subsequent process inventory
  found no Python worker for this recovery experiment. Checkpoint hashes remain
  unchanged. No paused legacy runner or automation was restarted.
- The analyzer authenticates both preserved reference full runs, canonical
  scenario/profile/source/runtime identity, summary/trace TTT and terminal
  inventory, selected actor identity, and the evaluation observation archives.
- No commit, push, unbounded collection or new scheduled job was performed.

Code: `work/sdmpc_rl_recovery_20260929/` and
`work/analyze_sdmpc_recovery_20260929.py`. Results root:
`results/sdmpc_rl_recovery_20260929/`, including `actor_fit_v2/completion.json`,
`critic_temporal_v1/diagnostics.jsonl`, `smoke_v2/parity.json`,
`regression_v2.xml`, and `evaluation_v2/analysis.json`. Evaluation directories
contain settings, full traces, summaries, observations, resumable checkpoints and
completion records. Keep `actor_fit_v1`/`smoke_v1` as non-admitted history.

Read-only reconciliation can be repeated with a new output path; it refuses to
overwrite existing evidence:

```powershell
.venv-torch/Scripts/python.exe -B work/analyze_sdmpc_recovery_20260929.py --evaluations results/sdmpc_rl_recovery_20260929/evaluation_v2 --output results/sdmpc_rl_recovery_20260929/evaluation_v2/analysis_recheck.json
```
