# P13: refit the shared neural policy with observed on-policy outcomes

> Complete and authenticated. The fresh pilot improved three profiles and
> worsened 155 and 190; no admission. Latest continuation:
> [P14 late-window comparison](rl_continuation_p14_20261002.md).

## Decision before fitting

P12 completed all ten candidate branches at 2026-10-02 11:44:01 KST.
Full authentication on this continuation passed source/model pins, all carry
controls, 600 new terminal transitions and exact bounded-policy action/memory
replay. No worker or applicable STOP remains.

Matched carry-relative TTT changes (%; negative is better):

| Policy | 155/9101 | 170/9102 | incident/9103 | skew15/9104 | 190/9105 |
|---|---:|---:|---:|---:|---:|
| P11 raw neural | +5.9587 | -5.1999 | +1.9351 | -9.3918 | -1.6683 |
| P12 NUF nonnegative | +5.8936 | -9.2119 | -3.6431 | -6.1571 | -1.8509 |
| P12 fitting bounds | +1.0064 | -2.7632 | -0.3287 | -7.8665 | -7.4394 |

Both bounds improve four profiles but still worsen 155. Removing NUF tightening
alone helped incident and 170; imposing the additional bounds reduced the 155
loss and improved 190, while losing some gains elsewhere. This paired comparison
does not identify which of the additional NP/NUF bounds caused each effect.
No candidate passes admission. No canonical evaluation has been performed.

Refit one shared neural policy using all allowed measured training outcomes,
then apply the fixed P12 `fitting_bounds` rule. That rule has the smaller worst
profile loss (+1.01% versus +5.89%) in this diagnostic. It is shared across all
scenarios and is chosen before P13 fitting or prospective scores. No mixture
of scenario-specific winners is permitted.

## Fixed fitting protocol

- Use the original 40 P10 fitting trajectories, ten unique P11 trajectories
  (carry plus raw neural on each 910x profile), and ten new P12 trajectories.
  Exclude P12's imported carry copies to avoid double counting. This gives
  60 full trajectories and 900 recorded action labels at decisions 16-30.
- The three fitting profiles per scenario are the original two P10 seeds plus
  9101/9102/9103/9104/9105 respectively. These 910x profiles were always training
  profiles and now become fitting data. They must not be reported as independent
  validation of P13. P10's reserved third-column holdout stays excluded, as do
  all canonical observations and results.
- Fit the raw requested action labels without projecting or relabeling them.
  Weights use their actual complete-rollout returns. The fixed output bounds
  apply during deployment; the training loss alone cannot certify their effect.
- Keep the P11 fitting recipe: return weight
  `clip(exp(((carry_TTT - TTT)/carry_TTT)/0.03), 0.1, 10)`, normalized to mean 1
  within scenario; exactly 20 examples per scenario in each 100-example batch;
  2000 AdamW updates, seed 11001, learning rate 0.0003, weight decay 0.001,
  gradient norm limit 1. Reinitialize from the same zero output layer rather
  than selecting a checkpoint. No hyperparameter sweep or early stopping.
- Inputs are the same 2367 observations plus two latched anchors. Fit input
  normalization on these fitting rows only, standard-deviation floor 0.05,
  clip normalized inputs to +/-10. Same two 64-unit tanh layers and tanh output.
  No scenario/seed/behavior IDs, outcomes or future observations enter the model.
- Before decision 16 use carry; learn decisions 16-30; after 30 return both
  latched budgets. Apply the unchanged P12 fitting bounds during 16-30 only.
  Preserve solver six iterations, physical feasibility guard, all constraints,
  +/-2% demand distribution, and complete 75-interval TTT accounting.
- Fitting gate: authenticated provenance and terminal accounting, finite loss
  and weights, final weighted fitting MSE below 80% of the zero-action value,
  exact artifact reload, action/history/bound checks and one physical interval.
  Freeze final source/model/spec before the prospective pilot.

## Prospective pilot and continuation

Use fresh training seeds 155/9201, 170/9202, incident/9203, skew15/9204,
190/9205. Generate a complete carry and authenticate its restored control before
each candidate branch. Record true-terminal experience through decision 75.
No existing completed branch is rerun. Maximum four numerical workers, one
numerical thread each, CPU masks 1/2/4/8; preserve every source/spec version.

This pilot has one new seed per scenario and cannot meet admission by itself.
Only a promising single shared policy proceeds to additional independent seeds.
Admission remains at least five seeds per scenario, no >=+10% degradation, and
mean deltas <= -2%, -9%, -6%, +1%, -3% for 155/170/incident/skew15/190.
Only passing policies are preregistered for canonical evaluation. Final success
still requires all five canonical gains and a fresh-folder repeat of the same
frozen policy. Never use canonical outcomes for training or relax these gates.

Source: `work/sdmpc_rl_p13_20261002`; output: `results/sdmpc_rl_p13_20261002`.
Honor STOP before/during fitting and rollouts. No automatic commit/push or OS changes.

## Fitting result

The declared refit completed 2000 actual optimizer updates with one Torch
thread, exactly 40,000 sampled examples per scenario (20% each). All 60 unique
trajectory inputs passed provenance and terminal-experience checks. The weighted
action MSE fell from 0.3676554561 to 0.0234319847, passing the predeclared fitting
gate. This is action-fitting evidence; prospective TTT improvement remains untested.

Model SHA-256:
`53ba39c306c2e830c3ab9a228cf0c8b95822b41abfc193d7c9c77d44214ca617`.
The raw neural spec is `fit_v1/neural_policy.json`; the single deployed spec
`fit_v1/policy.json` wraps it with the fixed `fitting_bounds` rule. The original
pre-fit protocol is preserved as `results/sdmpc_rl_p13_20261002/protocol.md` and
matches the training plan hash. All fitting sources remain frozen.

## Preflight and dispatch

Preflight passed identical model reload, zero-action prefix, sequential-history
validation, both-budget return, finite float32 actions, fixed output bounds,
bad-observation rejection and one actual physical interval using an existing
fitting-profile checkpoint. The 151-file physical manifest remains
`07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
Deployed policy spec SHA-256 is
`8ce29ace7515cd8d70bad23005510bddcc45f40eeec92ccd4712c40894c27b2a`.

Started the hidden pilot queue at 2026-10-02 14:06:38 KST, launcher PID 6208,
after confirming no Python process remained and no applicable STOP existed.
Output is `results/sdmpc_rl_p13_20261002/pilot1`. Jobs are 170/9202,
incident/9203, 190/9205, skew15/9204, then 155/9201 when a worker slot is free.
Each job generates its fresh carry, authenticates its k16 carry continuation,
then runs the same frozen bounded neural policy. TTT always includes the full
75-interval accounting. The queue inherits all prior source pins and adds the
new source, manifest, neural model and both specs.

At 14:07:49 KST coordinator PID 1748 and four numerical runtime children
12376/17228/3228/7328 matched the recorded commands, parents and start times.
The numerical masks were 1/2/4/8 and each had over 70 seconds CPU time;
the queue and all four worker error logs were empty. All 71 source/artifact
pins still matched. Evidence: `pilot1/launch_identity.json`. PIDs are only
historical identifiers and must be matched with command/start time on each check.

## Next heartbeat

1. Read this record, then check STOP and actual process PID/parent/start/command
   against `pilot1/plan.json` and `events.jsonl`. A venv shim and its runtime
   child count as one logical worker. Observe at most four active numerical
   runtimes, masks 1/2/4/8, with all numerical thread environment values one.
2. Inspect `queue.stdout.log`, `queue.stderr.log`, `pilot1/logs/*.log` and
   `completion.json` if present. Do not read active worker `status.json`.
   Do not launch a second queue or rerun completed work. Keep P9's intentional
   old-wave STOP and all earlier failed/recovered versions intact.
3. After all five jobs exit 0, run the existing venv Python with
   `work/sdmpc_rl_p13_20261002/pilot_analysis.py`. It checks all source/model
   pins, five full carries, exact restored controls, terminal transitions and
   fresh bounded-neural action/memory replay before writing `analysis.json`.
4. Interpret all five fresh profiles together. If promising, declare an
   independent seed expansion for the unchanged shared policy. Otherwise
   diagnose the complete outcomes and declare a bounded next learning step.
   Preserve the original fitting/holdout/canonical separation and admission.

Sources/specs must remain unchanged while the pilot runs. STOP at the repo,
P13 root/fit/pilot/slot or established global result roots takes precedence.
The coordinator propagates STOP and stops dispatch; workers check between
physical intervals. No canonical run is authorized by fitting loss alone.

## Final authenticated result

All five jobs exited 0 at 2026-10-02 16:14:00 KST. At the 17:59 continuation,
no Python process or applicable STOP remained. The full `pilot_analysis.py`
passed five full carries, exact restored carry controls, 600 terminal transitions,
source/model pins and fresh bounded-neural action/memory replay. Its one-shot
`pilot1/analysis.json` is preserved.

Carry-relative TTT changes (%; negative is better):

| 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---:|---:|---:|---:|---:|
| +2.6797 | -2.5854 | -1.7746 | -7.8223 | +2.5124 |

No >=+10% collapse occurred. The shared policy fails the gain gate and has only
one new seed per scenario. P13 has no canonical registration/evaluation and no
claim of general improvement. Its loss cannot be directly compared with P11's
loss because the fitting examples differ.

`diagnostic_action_phases.json` records the paired interval costs and executed
trajectories. Every profile continued negative NP requests at decisions 26-30.
For 155 and 190, most extra TTT accumulated after that late tightening, during
31-45 (+71.34 and +123.56 veh-h respectively). This motivates P14's paired
early-return and late-NP-hold comparison on the same training profiles, keeping
the model frozen. It does not by itself establish a causal benefit of either change.
