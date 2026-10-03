# P11: shared neural policy from measured training returns

> P11 pilot is complete and authenticated. It improves three training profiles
> and worsens two, so it is not admitted. Latest continuation:
> [P12 action-bound experiment](rl_continuation_p12_20261002.md).

## Entry and fixed scope

P10 recovery finished on 2026-10-02 at 03:54:21 KST. Full authentication at
05:59 KST passed all 60 branches, 3600 transitions and 15 exact carry controls.
All 48 recovered branches remain byte-identical to wave1. P9's earlier carry
and legacy reproduction certification remains valid; its source pins match.
There is no STOP and no worker remains from either wave.

P10 mean TTT deltas versus matched carry (%; three seeds each):

| Policy | 155 | 170 | incident | skew15 | 190 |
|---|---:|---:|---:|---:|---:|
| B1 both return | +0.803 | -4.066 | -0.349 | +0.426 | -2.552 |
| Long both return | +0.209 | -6.024 | -1.367 | +0.812 | -1.281 |
| Retain NUF during binding | +0.209 | -4.840 | -1.664 | +0.354 | -1.449 |

No P10 completed branch degraded by 10% or more. All three arms miss the
unchanged admission gains and have fewer than five seeds per scenario.
Longer binding and NUF retention have different effects across profiles;
there is no supported universal fixed-window winner. None is canonically
registered. P10 performed no neural optimizer updates.

## Bounded learning experiment, declared before fitting

Fit one shared neural action policy using **return-weighted behavioral cloning**.
This is an offline supervised policy update using measured complete-rollout
returns. It does not differentiate a learned critic, estimate an optimal Q
function, or establish improved control from training loss alone.

Use only the first two P10 fitting seeds, with all four behaviors per profile:
155: 8701/8501; 170: 8702/8502; incident: 8703/8503;
skew15: 8804/8504; 190: 8705/8505. The five third-column P10 seeds remain
held out and are not loaded by fitting or used to tune hyperparameters.
No canonical observations, outcomes, scenario IDs or seed IDs enter the model.

The neural part acts on decisions 16-30. Before 16, use zero action. At 16,
latch both observed budget anchors; after 30, restore both latched anchors
with the existing action scales/clipping. This limits neural intervention to
the window covered by all P10 candidate actions. It is part of the proposed
policy, with the physical guard, solver and environment unchanged.

Input: all 2367 existing observation features plus the two normalized latched
anchors. Reconstruct the same anchor features for every behavior, including
carry, from its decision-16 observation. Do not feed behavior-specific memory
flags, future observations, policy tags, total TTT or return weights as inputs.
Training labels are the recorded 2-D requested actions at decisions 16-30:
600 labels from 40 full trajectories. Return weights use complete episode TTT,
including the authenticated carry prefix; no truncated return or bootstrap.

Trajectory weight is `clip(exp(((carry_TTT - TTT)/carry_TTT)/0.03), 0.1, 10)`.
Normalize weights to mean 1 within each scenario. Fit mean weighted action MSE
with exactly 20 samples per scenario in each 100-sample minibatch. Sample
uniformly within each scenario, with replacement, using seed 11001.

Freeze these choices: input centering/scaling fitted only on fitting rows,
per-feature standard deviation floor 0.05, normalized inputs clipped to ±10;
two hidden layers of 64 units with tanh; tanh 2-D output; zero final layer
initialization; AdamW learning rate 0.0003, weight decay 0.001, gradient norm
limit 1; exactly 2000 updates; CPU and one numerical thread. No sweep, early
stopping, checkpoint selection or holdout-driven refit. Save the final model.

Implementation gate before rollout: all provenance/shape/action/terminal checks
pass; output and loss finite; final fitting weighted MSE below 80% of the
zero-action predictor's MSE; artifact reload gives identical actions; actor
history and post-window restoration tests pass. This gate checks that fitting
worked, not that the policy improves traffic. If it fails, preserve the failure
and diagnose before any rollout.

## Prospective pilot and unchanged admission

Freeze the single model, source and spec before five complete training-profile
pilots: 155/9101, 170/9102, incident/9103, skew15/9104, 190/9105.
These seeds are outside P10 fitting and holdout sets. Generate each carry with
the unchanged ±2% training-demand distribution and cache its k16 checkpoint.
Run the exact carry restore control and then the neural policy through decision
75. Include all 75 intervals and warmup in TTT. Record true-terminal experience.
Use at most four numerical workers with one numerical thread each.

This five-profile pilot is not canonical evaluation and cannot satisfy the
five-seed-per-scenario gate. Do not claim success from it or switch models by
scenario. Analyze its complete results; only a promising shared candidate
receives further independent training-distribution validation. Canonical
registration still requires ≥5 seeds/scenario, zero ≥+10% runs, mean deltas
≤ -2%, -9%, -6%, +1%, -3% in scenario order 155/170/incident/skew15/190.
The ultimate five canonical improvements and identical-policy fresh-folder
reproduction requirements remain unchanged.

Sources live in `work/sdmpc_rl_p11_20261002`; results and model artifacts live
in `results/sdmpc_rl_p11_20261002`. Preserve P9, P10 and recovery source/specs.
Honor STOP before fit, during updates and during rollout. Do not modify OS
settings or commit/push automatically. Continue the existing heartbeat.

## Fit and preflight results

`fit_v1/completion.json` reports 2000 actual neural optimizer updates, with
40,000 sampled labels per scenario (exactly 20% each). The final weighted
fitting MSE is 0.0369947217, compared with 0.4382175505 for zero action.
The final layer changed from its zero initialization. Fitting passed its
predeclared implementation gate. These errors measure action fitting only;
they are not TTT improvement estimates.

Only the 40 allowed fitting trajectories were exposed through the separate
`work/sdmpc_rl_p11_20261002/fit_manifest.json`. The five holdout trajectory
sets and canonical data were not loaded by fitting. The training plan records
input/metadata hashes, source hashes, normalization and the fixed optimizer
configuration. Its original protocol text is preserved verbatim as
`results/sdmpc_rl_p11_20261002/fit_v1/protocol.md`; the hash matches the
pre-fit plan. No holdout error was used to select this checkpoint.

- Frozen model: `results/sdmpc_rl_p11_20261002/fit_v1/model.pt`.
- SHA-256: `c5f38d4c5faff04fa6eb6fbf6c0169ac8643ea016bd27e88542fb578fe8178e2`.
- Shared policy spec: `fit_v1/policy.json`, SHA-256
  `288e37682720eea156694d60561d9a65f405670dd458bfa084fff2371d7efcd3`.
- `preflight.json` passed reload identity, fresh sequential state, zero-action
  prefix, both-budget restoration after 30, action bounds/dtype/nonfinite checks,
  and one actual physical training interval. The same frozen 151-file physical
  source and one Torch thread were confirmed. This is not a performance run.

## Active pilot and continuation entry point

P11 pilot launched hidden at **2026-10-02 06:12:16 KST**. Launcher PID 16740,
coordinator PID 12548. Initial numerical workers and verified CPU masks:

| Profile | Numerical PID | Mask |
|---|---:|---:|
| 170 / 9102 | 11484 | 1 |
| incident / 9103 | 14164 | 2 |
| 190 / 9105 | 15340 | 4 |
| skew15 / 9104 | 14956 | 8 |

All started at 06:12:16 and their command/start/parent identities were verified.
155/9101 is queued next. Four workers use CPU, with numerical thread environment
limits 1 set before import. No startup stderr error was present. Each begins
by generating its new full 75-interval carry and k16 checkpoint, then executes
the carry restore control and shared neural branch. Every score must include
the true terminal and complete carry prefix. Do not score partial carry logs.

Active output: `results/sdmpc_rl_p11_20261002/pilot1`.
Queue logs: `results/sdmpc_rl_p11_20261002/queue.stdout.log` and `queue.stderr.log`.
Worker logs/events/plan/completion live in `pilot1`; cache lives in `pilot1/cache`.
Read process identities, logs, completion and STOP; do not read active worker
`status.json`. This queue and fit refuse reuse of their existing output folders.
Preserve all pinned P11 and inherited P9/P10/recovery sources/specs while active.

Once `pilot1/completion.json` reports completed, authenticate with:

```powershell
& C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe -B work/sdmpc_rl_p11_20261002/pilot_analysis.py
```

The analyzer is one-shot and creates `pilot1/analysis.json`. If that file already
exists, inspect its status and hashes rather than rerunning and overwriting it.
It verifies all five complete carries, exact restore controls, model/source
hashes, experience provenance, 60-row continuations and fresh neural action
replay. A passing pilot still needs sufficient independent seed coverage and
the unchanged gain/safety gates before any canonical registration. If the
pilot is weak, use its training-profile traces to diagnose and declare the next
bounded experiment. Never substitute the P10 holdout or canonical set for fitting.

## 2026-10-02 09:59 KST: pilot complete, not admitted

Pilot completion: 08:12:38 KST, all five worker exits 0. No P11 Python process
remains. Full `pilot_analysis.py` authentication passed: five whole carries,
five exact restore controls, 600 continuation transitions, fresh neural replay,
all policy/source hashes, full TTT accounting and unchanged physical controls.
No STOP was found. `pilot1/analysis.json` is the durable result; do not overwrite.

| Scenario / seed | Neural TTT | Delta versus carry |
|---|---:|---:|
| 155 / 9101 | 3253.148793 | +5.9587% |
| 170 / 9102 | 4460.125773 | -5.1999% |
| incident / 9103 | 5751.508253 | +1.9351% |
| skew15 / 9104 | 4299.324957 | -9.3918% |
| 190 / 9105 | 6706.743582 | -1.6683% |

This confirms three improved and two worse training profiles, without a ≥+10%
collapse. The actor's fitting loss was not evidence that its requested actions
would improve every physical trajectory. One seed per scenario is insufficient
for admission, and the 155/incident losses already reject promotion.

Failure investigation found action extrapolation outside the fitting support.
All 600 fitting NUF labels are nonnegative; all 400 decision-16-25 NP labels
are nonpositive. P11 nevertheless produces negative NUF in 12 of the 15 learned
decisions on each failed profile, and it raises NP well before the teacher's
return window. On 155 it later has to tighten NP again to return to its latched
anchor. The measured cost difference accumulates mainly after decision 30.
This identifies a testable hypothesis, not a demonstrated causal explanation.

Evidence: `diagnostic_action_phases.json` and `fitting_action_bounds.json`.
Keep this model frozen. P12 predeclares two shared action-bound comparisons on
these same five training profiles, reusing authenticated baselines. It does
not add neural optimizer updates or alter the physical feasibility guard.
