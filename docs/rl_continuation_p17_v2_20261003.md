# P17 v2: fit one shared policy from the best observed training trajectories

> Pilot completed and authenticated on 2026-10-03. The frozen model improves
> all five fresh 930x training profiles. This is one seed per scenario and does
> not pass admission yet. See the final result below and
> [published evidence](rl_results_20261003/README.md). No numerical job remains.

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
- During 16-30, clamp NUF to the global minimum/maximum of the selected
  20 trajectories' NUF action labels in those decisions, including zero in the
  interval. Compute and freeze this single shared envelope before fitting. Do
  not impose a budget-anchor ceiling in this early phase; the unchanged physical
  capacity projection still applies. The envelope stays inside [-1,1].
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

Create immutable new sources in `work/sdmpc_rl_p17_v2_20261003` and results in
`results/sdmpc_rl_p17_v2_20261003`. Preserve this protocol before fitting. Release
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

## Pre-fit correction from the label-support audit

No P17 v1 model was fitted and no fresh pilot was run. Its pre-fit audit found
20 of 1200 selected labels outside the proposed early NUF anchor-return bound,
all in three 910x fitting profiles. Some successful demonstrations request a
small NUF increase at the anchor; the selected skew15 trajectory also includes
small negative NUF actions. This contradicts the assumption that all selected
labels fit v1's early bounds. Preserve the original sources/protocol and
`results/sdmpc_rl_p17_20261003/selected_action_support_v1.json`.

P17 v2 changes only that early NUF bound to the global envelope of the selected
training labels. Selection, full-sequence labels, architecture, optimizer,
update count, seed, equal scenario sampling, late return/hold constraint, fresh
930x pilot seeds and admission criteria remain as declared. Freeze the exact
envelope in the manifest, training plan and actor spec before the pilot. Check
that all 1200 selected labels now fit the declared support within 1e-5. This is
a pre-fit correction using training data, not feedback from a pilot/holdout.

## Fitting and support results

The audit authenticated all 105 allowed trajectory records and selected exactly
20, one per profile. Their 1200 labels are all within v2's deployed support to
1e-5. The frozen early NUF action envelope is [-0.05645276978611946, 1.0],
derived only from selected decisions 16-30. Evidence: `fit_manifest.json` and
`results/sdmpc_rl_p17_v2_20261003/selected_action_support.json`.

The single declared fit completed 2000 optimizer updates. Action MSE decreased
from 0.10792497545480728 to 0.006706832908093929; exact reload passed. Each
scenario contributed 40000 sampled labels, and Torch used one numerical thread.
The neural model SHA-256 is
`b0381c55d999ec4c72512f91e7849cc79eb725baf085bf4242759e238a81eae8`.
No reserved holdout or canonical record was loaded. This confirms supervised
model fitting only; traffic performance remains untested until the fresh pilot.

## Preflight and pilot dispatch

Preflight passed all 1200 label-support checks, 75 zero-network carry-action
checks, reload/sequential/finite-input tests, early envelope and late sign/gap
bounds, UTF-8 worker handling, exact old-profile carry interval reproduction
and one fitted-policy physical interval. The frozen physical snapshot remains
`07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
No fresh 930x profile was used in preflight.

The hidden pilot started at 2026-10-03 10:17:09 KST, launcher PID 5460,
coordinator PID 4604. No Python process, applicable STOP or existing 930x result
artifact was present before dispatch. At 10:19:08 four numerical runtime
children 1676/14628/19512/18756 matched the queue commands, parent identities
14128/12152/13904/16048 and start times. Masks were 1/2/4/8; each had accumulated
over 116 CPU seconds. All numerical thread environment values are one and error
logs are empty. The 155 job is queued until a slot is available.

`pilot1/launch_artifact_audit.json` verifies all 111 source/artifact pins, model,
protocol, preflight and fitting-receipt hashes, and disjoint fitting/pilot seeds.
`pilot1/launch_identity.json` records observed process identities. Recheck PID,
parent, start and command on later heartbeats; historical PIDs may be reused.
Final outcomes are pending. No canonical candidate is registered.

### Next heartbeat

1. Read this record and inspect repo/global/P17 v2 root/fit/pilot/slot STOP,
   actual process identities, `pilot1/plan.json`, `events.jsonl`, root pilot
   stdout/stderr, `pilot1/logs/*.log`, branch files and completion. Leave active
   worker `status.json` alone. Do not duplicate this wave or its completed work.
2. After all five jobs exit 0, run the existing venv Python with
   `work/sdmpc_rl_p17_v2_20261003/pilot_analysis.py` once. It checks complete
   carry generation, exact restored carry, policy/source identities, full
   accounting, terminal experience and exact fresh action/memory replay before
   writing `pilot1/analysis.json`.
3. Judge the frozen shared model on all five fresh training profiles. Expand
   only if promising, using a declared seed batch and unchanged five-seed/gain
   gates. Otherwise preserve this fit/pilot and declare the next bounded
   experiment. A fitting-loss decrease alone is insufficient.

## Final authenticated pilot result, 2026-10-03

All five jobs finished with exit code 0 at 12:19:32 KST. During the user's
13:05 KST request to push results, no Python process was running, all error
logs were empty and no applicable STOP was present. The analyzer authenticated
full carries, exact restored carries, source/model identities, 600 terminal
transitions, complete TTT accounting and exact fresh action/memory replay.

| Scenario / seed | Carry TTT | Shared policy TTT | Delta | Policy fallbacks |
|---|---:|---:|---:|---:|
| 155 / 9301 | 3193.855237 | 3121.111618 | -2.2776% | 9 |
| 170 / 9302 | 4258.070035 | 4047.627886 | -4.9422% | 4 |
| incident / 9303 | 6160.857920 | 5841.352527 | -5.1861% | 6 |
| skew15 / 9304 | 4274.524335 | 4233.771529 | -0.9534% | 1 |
| 190 / 9305 | 7065.447374 | 6881.517144 | -2.6032% | 1 |

All five training profiles improve under one frozen observation/history policy.
This supports further seed validation but does not establish canonical gains,
five-seed robustness or admission. Current 170/incident/190 gains are below their
mean targets. The 155 fallback count of nine deserves inspection across the next
seeds. Do not tune this frozen candidate on the pilot before completing its
declared seed evaluation.

Next heartbeat: the pilot and analysis are already complete; do not rerun either.
Keep the exact model/spec/source frozen and preregister a bounded additional four
training seeds per scenario. Generate each same-PC carry and restore control,
then evaluate the shared policy over the full horizon. Combine only independent
seeds for its five-seed gate. No canonical registration until all unchanged gates
pass. The user's push request authorizes this publication; it does not enable
automatic commits or pushes in future heartbeats.
