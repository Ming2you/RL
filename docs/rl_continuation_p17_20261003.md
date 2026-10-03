# P17: fit one shared policy from the best observed training trajectories

> Superseded before fitting by [P17 v2](rl_continuation_p17_v2_20261003.md).
> The label audit found 20 early NUF labels outside this draft actor's bounds.
> No optimizer update or fresh rollout was performed with this version. Preserve
> its sources/protocol and audit; continue with the versioned correction.

## Authenticated P16 outcome

P16 finished at 2026-10-03 07:27:20 KST, all five jobs exiting 0. At the
09:59 continuation no Python process or applicable STOP remained. Full analysis
authenticated 600 new and 300 reused terminal transitions, source/cache/control
hashes, full 75-interval TTT accounting and exact fresh actor/memory replay.

| P16 policy | 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---|---:|---:|---:|---:|---:|
| np_neutral | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| np_relax_100 | -0.3192% | +0.6407% | -3.2130% | -5.7481% | -3.2178% |

Positive NP behavior provides a small 155 improvement and gains in three other
profiles, without a fallback in these five branches. It fails the admission
thresholds and worsens 170. Neutral NP reproduces carry TTT on all five. These
are paired training diagnostics, not fresh validation or canonical evidence.
P16 contains zero neural optimizer updates. Preserve the completed report at
`results/sdmpc_rl_p16_20261003/wave1/analysis.json`.

## Predeclared fitting protocol

P11/P13 fitted mixtures containing losing trajectories. P17 instead fits the
best complete observed trajectory for each allowed training profile, including
carry as a candidate. This is supervised behavioral cloning using measured
returns to select labels. It does not estimate a Q function or perform an online
policy-gradient update, and selected trajectory scores are not policy scores.

Audit the 80 allowed trajectory references in P15's support audit plus the
15 P15 candidates and 10 P16 candidates. Exclude reused carry copies. These
105 records span exactly four fitting seeds per scenario:

| Scenario | Fitting seeds | Fresh pilot seed |
|---|---|---:|
| 155 | 8701, 8501, 9101, 9201 | 9301 |
| 170 | 8702, 8502, 9102, 9202 | 9302 |
| incident | 8703, 8503, 9103, 9203 | 9303 |
| skew15 | 8804, 8504, 9104, 9204 | 9304 |
| 190 | 8705, 8505, 9105, 9205 | 9305 |

For each profile select the smallest full TTT. Treat values within 1e-9 veh-h
of the minimum as tied; prefer carry, then policy name and experience path.
Authenticate source pins, report hashes, terminal accounting and all referenced
artifacts. Keep all 105 records and a separate explicit list of 20 selections.
P10's reserved third seed and every canonical record remain excluded. No policy
identity or scenario/seed identifier enters the neural input or execution.

Fit all 60 observed actions per selected trajectory, decisions 16-75: 1200
labels, 240 per scenario. This includes observed recovery/hold behavior after
30, which previously differed between demonstrations and the fixed actor tail.
Normalize only these fitting observations plus the two decision-16 anchors.
Use 2369 inputs, two 64-unit tanh hidden layers, a two-unit tanh action output
initialized to zero, seed 11001, 2000 AdamW updates, learning rate 0.0003,
weight decay 0.001 and gradient norm limit 1. Standard deviation floor 0.05,
normalized input clip [-10,10]. Every minibatch has 20 samples per scenario
(100 total), uniform across its 240 labels; no return weighting or retuning.
Thus each scenario contributes exactly 40000 sampled labels.

## One shared deployed actor

Before decision 16 emit zero. Latch the two budget anchors at 16. Use the same
network on observed state/history during 16-75; learn the recovery speed as well
as the earlier actions. Input, anchor memory, numerical precision and sequential
decision checks follow the previous actor contract.

Apply these fixed observation-based bounds to the network's requested action:

- During 16-30, NP may tighten within [-1,0]. A positive NP action is capped by
  the headroom to the latched NP anchor plus 100, using the same scale as P16.
- During 16-30, NUF may move only toward its latched anchor, at a rate chosen
  by the network or zero. If below the anchor, clamp to [0, min(1,gap/1000)];
  if above, clamp to [max(-1,gap/1000),0].
- After 30, both NP and NUF may move only toward their latched anchors or
  hold. The same sign/gap clamp uses action scales 50/1000. This permits both
  carry and observed return speeds, while preventing further late tightening
  away from the anchor. It intentionally replaces P13's forced full-speed return.

An all-zero network reproduces requested carry actions across all 75 decisions,
including after a fallback. Physical action projection, six solver iterations,
guard, fallback, simulator, horizon and reward accounting remain unchanged.
These are joint data/actor changes; do not attribute an outcome to either one
alone. A fitting-loss decrease is not evidence of traffic improvement.

## Gates, fresh pilot and continuation

Create immutable new sources in `work/sdmpc_rl_p17_20261003` and results in
`results/sdmpc_rl_p17_20261003`. Preserve this protocol before fitting. Release
one model only if parameters are finite, reload is exact, loss falls below 80%
of the zero-network loss, membership/label counts match and STOP is absent.
Do not change the fit seed, update count or architecture to pass this gate.

Preflight must check reload, sequential/finite float32 inputs, zero-network
carry identity, neutral/positive/negative bounds, overshoot recovery, no forced
tail return, all fitting-label action ranges, UTF-8 metadata handling, and an
actual interval on an old fitting profile. Pin sources/model/spec before pilot.

Then run only the five preregistered fresh 930x training profiles with the same
frozen model. Each profile generates a full carry, authenticates its k16 restored
carry continuation exactly, and evaluates the actor to true terminal. Keep
U(0.98,1.02), five warmup plus 75 control intervals and 14400 s horizon. Use at
most four numerical workers, each one numerical thread, masks 1/2/4/8, hidden
windows. Inspect PID/parent/start/command, logs, completion and STOP; do not read
active worker status files or duplicate a running/completed wave. No auto
commit/push, dependency installation or OS change.

After full authentication, expand a promising fixed candidate to at least five
seeds per scenario. Admission remains zero >=10% degradation and means <= -2%,
-9%, -6%, +1%, -3% for 155/170/incident/skew15/190. If weak, preserve this
result and design the next bounded experiment from allowed training data only.
No canonical registration until admission passes; no holdout/canonical tuning.
Success still needs strict gains on all five canonicals and fresh-folder
reproduction by exactly the same frozen shared policy.
