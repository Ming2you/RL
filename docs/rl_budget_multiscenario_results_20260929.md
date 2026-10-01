# Balanced shared-policy pilot: completed, performance rejected

## Decision

The requested balanced five-scenario learning path is implemented and exercised
end to end. One shared actor and two shared TD3 critics were trained; each
scenario contributed exactly 20% of every minibatch. All 20 full trajectories
completed, including 10 exploratory collection episodes and 10 canonical
evaluation episodes. All 22 scheduled jobs completed without rerunning a finished
collection. The finite pilot and subsequent read-only diagnostics have exited.

The trained controller is **not accepted**: all five canonical full-run TTTs are
substantially worse than the matched carry-budget center. Balanced sampling did
not resolve actor saturation or long-horizon value miscalibration. This is an
implementation/diagnostic milestone, not a traffic-performance achievement.
No legacy DDQN 5% goal, P-Stack improvement, generalization, or nonlinear-price
causation is claimed. No new automation or follow-on collection was launched.

## Frozen scope and identities

- Code: `work/sdmpc_rl_multi_20260929/`.
- Results: `results/sdmpc_rl_multi_20260929/pilot_v1/`.
- Admission: `results/sdmpc_rl_multi_20260929/admission_v1/`.
- Plan: `docs/rl_budget_multiscenario_plan_20260929.md`.
- Source snapshot: 151 files, manifest SHA-256
  `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
- Admission gate SHA-256:
  `c482fbba86ceefeb1c5938fc733f8fb73ee4760efe5d26463642b5aa5b8c797e`.
- Final shared checkpoint: `pilot_v1/train_round1/model_final.pt`, SHA-256
  `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`.
- Round0 checkpoint SHA-256:
  `97eb5679948f9dbc4afb5f04bd77ddb5e7b31c3e36a684726560cc2e0b35a190`.
- Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, Torch 2.14.0+cpu, PyYAML 6.0.3.

The physical carry controller/environment/runtime and snapshot freezer are
byte-identical to the preceding carry pilot. The actor selects two requested
budget increments, not nonlinear price coefficients or direct physical controls:
`B_request = B_previous_executed + [50,1000] * action`. NUF is clipped to its
physical range; signed NP is retained. Initialization/recovery PFO remains,
but no ordinary per-step PFO is run. The guard checks physical/budget feasibility,
not performance. No hidden ramp floor was added. Existing follower, demand,
incident, warmup and TTT contracts were preserved per scenario.

Every evaluation uses the original reset, five warmup intervals, all 75 control
steps and 14400 seconds. No exploration, forced intervention, or H3 performance
guard is used. Scenario IDs select replay strata but are not network inputs;
all scenarios share the same 2367-feature observation contract. This is a
state-conditioned action-value function, not a state-independent scalar or an
average of independently trained models. Markov sufficiency remains unproven.

## Balanced collection and learning

| Scenario | Round0 seed | Round1 seed | Stored transitions | Terminal transitions | Cumulative training draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| sweet_155_w | 6301 | 6401 | 150 | 2 | 6000 |
| sweet_170_w | 6302 | 6402 | 150 | 2 | 6000 |
| sweet_170_incident_w | 6303 | 6403 | 150 | 2 | 6000 |
| sweet_170_skew15_w | 6304 | 6404 | 150 | 2 | 6000 |
| sweet_190_w | 6305 | 6405 | 150 | 2 | 6000 |

There are 750 collected sequential transitions. The 30000 training draws are
sampling with replacement, not 30000 newly simulated transitions. Each update
uses eight draws per scenario in one batch of 40. There were 375 central updates
per round, 750 critic updates and 375 delayed actor updates total. Workers never
train private models. A complete five-scenario wave uses one frozen common policy
before the next central training phase. Evaluation data never enters replay.

Reward is actual negative interval TTT divided by 100, gamma=1, with bootstrap
disabled only on true termination. Round0 uses 32 uniform actions per scenario,
then Gaussian exploration; round1 uses the first shared model plus Gaussian
exploration. The final model is evaluated without noise. Collection has the
documented demand perturbations; its TTT is not a canonical performance score.

The failed preceding pilot used 20 updates/new transition; this pilot used 1.
Balance and update frequency therefore changed together. This is not a
single-factor ablation and cannot isolate the causal effect of scenario balance.
Equal sample counts also do not establish equal gradient magnitudes.

## Canonical full-run comparison

The comparator below is **zero-action carry-budget center**, not P-Stack. Both
methods use the same scenario, physical controller and full-run TTT accounting.
TTT uses the simulator's vehicle-hour accounting; lower is better.

| Scenario | Carry-center TTT | Shared RL TTT | RL / center | Center mean decision s | RL mean decision s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 155 | 3103.011072 | 30692.368191 | 9.8912 | 11.1032 | 8.1035 |
| 170 | 3935.903237 | 33126.802572 | 8.4166 | 13.7684 | 8.8565 |
| 170-incident | 5546.224352 | 33125.635550 | 5.9726 | 13.5721 | 7.9282 |
| 170-skew | 4250.876599 | 33123.575802 | 7.7922 | 12.4620 | 7.8719 |
| 190 | 6604.297017 | 36340.971289 | 5.5026 | 16.3198 | 8.6495 |

The 170-incident center exactly reproduces the preserved earlier TTT
5546.224352256691. No evaluation is a shortened prefix. Every RL row uses the
same final checkpoint hash, with no evaluation learning.

The 22-job pilot took about 69.39 minutes from recorded parent start to completion
file creation. Central training itself took 6.3425s and 6.5095s. At most five
single-core numerical workers were scheduled, below the eight-worker budget.
Timings include initialization/recovery PFO, reference preparation, actor,
follower solve and guards as recorded by the existing timing contract; RL actor
timing also includes read-only twin-Q diagnostics. Full elapsed/reset accounting
is separately retained in completion files. These were concurrent single-core
runs, not isolated timing trials. No preview-free speedup is claimed. Faster
decisions in a collapsed traffic regime are not a useful controller improvement.

## Observed failure mechanism

1. Both actor outputs are saturated in all 375 canonical decision states:
   approximately `[+0.999997,-0.999955]`; both dimensions have 100% absolute value
   at least 0.95. The common policy has not learned useful state differentiation.
2. NUF falls from 6000 to approximately 5000,4000,3000,2000,1000,0.728,then 0.
   It reaches exactly 0 at control index 6, the seventh controlled interval.
   All five scenarios then have 69 consecutive intervals with all ramp commands 0,
   totaling 12420 seconds. This closure pattern is read from executed controls,
   not merely inferred from nominal actor outputs.
3. Final network inventory is 15576.55,16612.46,16607.34,16602.48,17955.35 vehicles
   in scenario order, versus approximately 414-422 for the centers. Urban TTT is
   the dominant loss. Each RL evaluation has zero reference fallbacks and one
   initial PFO. Thus the failure here is executed bad requests, not a fallback
   holding an old request as in the preceding single-scenario failure.
4. Physical/budget checks passed, but `converged_count=0` in all center and RL
   evaluations. Feasibility is not solver convergence or good network control.

### Value calibration

For each canonical trajectory, compare min(Q1,Q2) logged before execution with
the realized reward-to-go under that same frozen deterministic policy. This
avoids treating an exploratory behavior return as the current policy's value.

| Scenario | Mean predicted Q | Mean realized return | Mean Q minus return |
| --- | ---: | ---: | ---: |
| 155 | -8.8586 | -209.9459 | 201.0873 |
| 170 | -9.3745 | -226.6172 | 217.2427 |
| 170-incident | -9.3553 | -226.6039 | 217.2486 |
| 170-skew | -9.3720 | -226.5457 | 217.1737 |
| 190 | -10.1314 | -248.5208 | 238.3894 |

These are reward units (TTT/100), not TTT itself. Optimism is severe on the
full-horizon trajectory. TD3's smoothed target policy is not exactly this
deterministic evaluation policy, so this is an operational calibration diagnostic,
not a claim about an exact population Bellman error.

### Actor versus critic mismatch

Read-only probes of the final model on 750 stored replay states show that Q1's
best action on the nine-point `{-1,0,1}^2` grid is `[+1,+1]` in all 750 states.
Q2 chooses it in 743 states and `[+1,0]` in 7. Yet the actor stays near `[+1,-1]`.
Q1's local derivative with respect to either action is positive in all 750 states.
The final actor therefore fails to track even its own critic's local NUF
direction; it is incorrect to say the final critic consistently prefers closure.

Measured mean NUF tanh slope is 0.0000975-0.0001279 across scenario replay groups.
Q1's mean NUF action derivative 0.0177-0.0195 becomes a mean derivative of roughly
0.0000020-0.0000027 at the actor preactivation. This is direct evidence of strong
saturation attenuation, not proof that it is the only cause. Shared hidden
parameters, optimizer history and early critic gradients can also matter.
No actor-reset rescue or counterfactual traffic benefit has been demonstrated.

The learned grid preference is not evidence that `[+1,+1]` truly improves TTT.
Value calibration and actor optimization are distinct problems; fixing only one
must not be advertised as a traffic-performance solution.

## Focused next experiments, not launched

Preserve balanced collection and 20% replay strata. Do not collect another arbitrary
24 hours or repeat this configuration without testing a specific failure.

1. Freeze the final critic and compare continued actor fitting with a reset
   output head on the same balanced replay, without plant rollouts. Log Q1
   objective, action direction, saturation and output gradients. This tests the
   actor-recovery hypothesis independently of critic learning. Increased predicted
   Q alone is not acceptance.
2. Audit time-resolved Q targets and action rankings before actor updates.
   Compare a critic-only fitting phase against the existing schedule, retaining
   true terminals and separate exploratory/current-policy return diagnostics.
   Do not reinterpret fixed-continuation returns as unconditional optimal labels.
3. Use a small, matched reached-state diagnostic to test whether NUF reopening
   improves subsequent traffic after early closure, recording identical future
   continuation and its limitations. An apparently better short branch must
   still pass all five ungated full runs before acceptance.
4. Only after those tests, choose a versioned actor/critic or action-coordinate
   change. Keep demand, follower constraints and TTT accounting fixed. Any
   anti-collapse guard must be disclosed and separately evaluated, not silently
   inserted into this completed experiment.

## Verification and reproduction

- 486 synthetic tests PASS, plus five real serialized reset/carried-state smokes.
- Independent learner and lifecycle review findings were repaired and re-reviewed;
  final admission attestation binds exact source/test/runtime/evidence hashes.
- Main pilot exec21637 exited 0 with `MULTI_PILOT_COMPLETE`.
- Read-only reconciliation exec88197 exited 0 with `MULTI_DIAGNOSTICS_PASS`.
- Value surface and actor-gradient probes exited 0; neither trained or saved a
  modified checkpoint. The original model hash remains unchanged.
- No process for this multi-scenario pilot remains. An unrelated calibration
  process was observed and left untouched. Paused legacy loops/automations remain
  untouched. No commit, push or scheduler change was requested or performed.

Full artifacts: `comparison.json`, `diagnostics.json`, `value_probe.json`,
`actor_gradient_probe.json`, each episode's summary/trace, both training rounds'
metrics/diagnostics/checkpoints, and `plan.json` under the result directory.
The source-bound read-only analysis commands are:

```powershell
.venv-torch/Scripts/python.exe -B work/analyze_sdmpc_multi_20260929.py --pilot results/sdmpc_rl_multi_20260929/pilot_v1 --output results/sdmpc_rl_multi_20260929/pilot_v1/diagnostics_recheck.json
.venv-torch/Scripts/python.exe -B work/probe_sdmpc_multi_values_20260929.py --pilot results/sdmpc_rl_multi_20260929/pilot_v1 --output results/sdmpc_rl_multi_20260929/pilot_v1/value_probe_recheck.json
.venv-torch/Scripts/python.exe -B work/probe_sdmpc_multi_actor_gradients_20260929.py --pilot results/sdmpc_rl_multi_20260929/pilot_v1 --output results/sdmpc_rl_multi_20260929/pilot_v1/gradient_probe_recheck.json
```

Analysis scripts refuse to overwrite existing evidence. Do not restart the
completed pilot or mutate its source, configuration, checkpoint or admission.
