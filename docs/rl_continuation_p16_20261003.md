# P16: collect neutral and positive NP behavior

> Complete and authenticated. Continue with
> [P17 v2](rl_continuation_p17_v2_20261003.md), a shared-model fit and fresh
> training pilot. Preserve this wave and its source/spec versions.

## Decision from authenticated P15

At the 2026-10-03 05:59 KST continuation, P15 recovery had completed all five
jobs with exit code 0. No Python process or applicable STOP was present.
`analyze_recovery.py` authenticated the 53 preserved files, 84 source/artifact
pins, all 15 complete candidate branches (900 transitions), five reused carry
controls (300 transitions), cache hashes, full accounting and exact fresh
action/memory replay. Original `wave1` remains incomplete and preserved.

TTT changes relative to each paired carry (%; negative improves):

| Policy | 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---|---:|---:|---:|---:|---:|
| P13 | +2.6797 | -2.5854 | -1.7746 | -7.8223 | +2.5124 |
| P15 half NP | +2.4200 | +5.8318 | +3.1851 | -5.2096 | -6.3712 |
| P15 half NP + restore NUF | +2.4200 | +5.8318 | +4.7457 | -5.2096 | -3.9661 |
| P15 restore NUF | +2.6797 | -4.0757 | -3.3624 | -7.8223 | +2.3438 |

No run degraded by 10%, but no candidate meets the gain thresholds. Halving NP
tightening benefits 190 and harms 170/incident; faster NUF return helps
170/incident but leaves 155 unimproved. These are measured effects on one
training profile per scenario, not evidence of seed robustness. No canonical
candidate is registered. P15 performs zero neural optimizer updates.

The fitting/diagnostic support audit through P14 contains 20 allowed training
profiles. Carry is best on three of four 155 profiles; the best observed 155
gain is only 0.8742%. P15's weaker tightening still loses on 155. The next
bounded experiment therefore collects behavior in the untested neutral/positive
NP direction. It does not assume that this direction will improve traffic.

## Protocol declared before dispatch

Freeze P13's neural parameters and its NUF bounds:

- Model SHA-256: `53ba39c306c2e830c3ab9a228cf0c8b95822b41abfc193d7c9c77d44214ca617`.
- Base spec SHA-256: `8ce29ace7515cd8d70bad23005510bddcc45f40eeec92ccd4712c40894c27b2a`.

Use the two shared policies below during decisions 16-30. Let `a` be P13's
bounded action, `b` the current NP budget observation in units of 1000, and
`b16` its latched decision-16 value.

1. `np_neutral`: set the NP action to zero; preserve P13's NUF action. This
   carries the executed NP budget during the window, including any fallback
   changes, and isolates removal of the learned NP intervention.
2. `np_relax_100`: set NP action to
   `min(abs(a[0]), clip((b16 + 0.1 - b) * 20, 0, 1))`; preserve NUF. This
   mirrors negative NP actions toward relaxation, with requested NP headroom
   bounded by 100 above the latched budget (subject to observation precision).
   If the current budget is already above that ceiling, request zero NP change.

Both use exactly P13 before decision 16 and after decision 30, including return
of both budgets to the latched anchors. The former experimental restriction
against positive NP before 26 is changed deliberately in P16; the original
[-1, 1] action range, action scales 50/1000, solver six iterations, physical
projection, execution checks and fallback are unchanged. No controller, plant
or objective implementation is altered. No scenario/seed identifier or future
outcome enters either policy. Neural memory and normalization are unchanged.

Run exactly ten new terminal continuations on paired training seeds 9201-9205
(155/170/incident/skew15/190), reusing authenticated P13 carry checkpoints and
control artifacts. Each outcome includes its carry prefix and all 75 control
intervals, with five warmup intervals and the same 14400 s horizon. Demand stays
U(0.98, 1.02). P10 reserved holdout and canonical runs are excluded. These 920x
profiles are diagnostic training data; repeated use is not independent validation.

## Checks and dispatch rules

Sources: `work/sdmpc_rl_p16_20261003`.
Results: `results/sdmpc_rl_p16_20261003/wave1`.

Before launch, require no existing wave, inspect process identity and STOP,
authenticate P13 and P15 sources/analysis, and preserve this protocol in the
result root. Preflight checks the override formulas, positive action and ceiling
cases, exact action/memory equality outside the window, two identical reloads,
sequential/finite float32 input enforcement, UTF-8 cache handling and a physical
interval. Pin new sources/specs plus inherited runtime/model files.

Use at most four numerical workers, one numerical thread each, masks 1/2/4/8,
hidden windows. Preserve all previous sources and outcomes. Honor STOP in repo,
global result roots, P16 root/wave/slots and the reused P13 root/pilot/slots.
Do not read active worker `status.json`. Do not recompute completed branches,
change running source/specs, commit/push, or alter OS settings.

After completion, run `work/sdmpc_rl_p16_20261003/analyze_wave.py`. Require full
75-interval accounting, 600 new and 300 reused transitions, source/cache/control
hashes and exact policy replay. Compare both candidates with P13 and carry on
each profile. Preserve failures. No improvement or neural training completion
is claimed before these checks.

## Following decision

Use the authenticated new returns to choose the next bounded shared-model
update or additional behavior collection. A model update may use these allowed
training profiles but must then be evaluated on fresh declared training seeds;
sample each scenario equally. Do not combine per-scenario winning policies for
execution. A promising fixed candidate requires at least five seeds per scenario,
zero >=10% degradation and mean TTT deltas <= -2%, -9%, -6%, +1%, -3% for
155/170/incident/skew15/190 before canonical preregistration. Thresholds are
unchanged. Final success requires strict improvement on all five canonicals and
fresh-folder reproduction using exactly the same frozen shared policy.

## Preflight and launch

Preflight passed 450 unchanged action/memory comparisons outside 16-30 and 150
override-formula checks inside the window. It also passed NP sign/headroom and
NUF-preservation cases, identical reloads, input/sequential checks, UTF-8 cache
handling, exact reproduction of all 19 deterministic carry-row fields for
170/9202 decision 16, and an actual positive-NP physical interval. Both-budget
return still handles an NP budget above the anchor. The 151-file physical
snapshot identity remains
`07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
Receipt: `results/sdmpc_rl_p16_20261003/preflight.json`.

The hidden queue started at 2026-10-03 06:13:52 KST, launcher PID 17092,
coordinator PID 16500. Before launch there was no Python process or applicable
STOP. At 06:14:27 the four numerical runtime children 5848/19272/16736/20032
matched the recorded worker commands and parent/start identities, with CPU
masks 1/2/4/8 and over 33 CPU seconds each. Their venv shim parents were
21352/20244/14688/20028. Error logs were empty. These are historical identities;
always recheck them before acting. The 155 job waits for a free slot.

At 06:16:38 all four workers had loaded the expected cached carries and each
had accumulated over 160 CPU seconds. Coordinator/worker PID, parent, command,
start time and masks matched. `wave1/launch_identity.json` preserves that check.
`wave1/launch_artifact_audit.json` verifies all 78 source/artifact pins, ten
cache files against the authenticated P15 cache hashes, 15 exact imported
control files, numerical thread environment and protocol/preflight hashes.
All error logs were empty. Final outcomes remain pending.

### Next heartbeat

1. Inspect STOP, `wave1/plan.json`, `events.jsonl`, process identity, queue and
   worker logs, completed branch files and `completion.json`. Leave active
   `status.json` files alone. Do not start another wave while this one runs.
2. After all five jobs complete with exit code 0, use the existing venv Python
   to run `work/sdmpc_rl_p16_20261003/analyze_wave.py` once. Preserve its report.
3. Compare full outcomes; follow the shared-model/independent-seed decision
   above. No admission or canonical registration has occurred yet.

## Final outcome, 2026-10-03

All five jobs completed successfully at 07:27:20 KST. At the 09:59 heartbeat,
no Python process or applicable STOP was present and all error logs were empty.
`analyze_wave.py` authenticated 600 candidate and 300 reused control transitions,
source/cache/control hashes, full accounting and exact fresh policy replay.

| Policy | 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---|---:|---:|---:|---:|---:|
| np_neutral | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| np_relax_100 | -0.3192% | +0.6407% | -3.2130% | -5.7481% | -3.2178% |

Relaxation yields the first small improvement on the 155/9201 diagnostic
profile and helps three other profiles, but worsens 170 and fails admission.
All five relaxation branches have zero fallbacks. Neutral NP reproduces carry
TTT on every paired profile. No >=10% degradation occurred. P16 did not update
neural weights. Final report: `results/sdmpc_rl_p16_20261003/wave1/analysis.json`.

The next shared learner selects the best observed complete trajectory per
allowed fitting profile and learns recovery/hold actions as well as the early
window. P17 v1's pre-fit support audit exposed 20 NUF label incompatibilities;
it was preserved without fitting. P17 v2 fixes that restriction using a single
global envelope from selected training labels, then runs the declared fresh
930x training pilot. No canonical candidate has been registered.
