# P12: test action bounds learned from P11's fitting data

> Complete and authenticated. Both shared candidates improve four training
> profiles but worsen 155; neither is admitted. Latest continuation:
> [P13 shared-model refit](rl_continuation_p13_20261002.md).

## Verified starting point

P11 pilot completed at 2026-10-02 08:12:38 KST; all five workers exited 0.
At the 09:59 continuation, the full analyzer authenticated five 75-interval
carries, five exact restored controls, five neural branches, 600 recorded
transitions, all policy/source hashes and fresh neural action/memory replay.
No P11 worker remains and no STOP is present. P9/P10 certification is preserved.

P11 prospective training deltas (negative improves TTT): 155/9101 +5.9587%,
170/9102 -5.1999%, incident/9103 +1.9351%, skew15/9104 -9.3918%,
190/9105 -1.6683%. No ≥+10% collapse occurred, but the shared policy fails the
gain requirements. It is not eligible for canonical registration.

## Diagnosis

The observed action-fitting error did not establish control performance.
Inspection of the 600 P10 fitting labels found:

- Zero negative NUF actions among all 600 labels.
- Zero positive NP actions in the 400 labels from decisions 16-25.
- No positive NP/NUF action requested a budget above its decision-16 latched
  anchor (maximum NP floating error 3.73e-7 veh).

The P11 neural policy violated this support. It requested negative NUF actions
in 12 of 15 learned-window decisions for both 155 and incident, reaching about
-0.17 (170 veh/h per decision). It also raised NP before decision 26; 155's
NP output became positive from decision 18 and eventually approached +1.
The later fixed return consequently tightened NP again. Most TTT loss occurred
after decision 30: +146.70 veh-h during 31-45 for 155, and +112.61 veh-h after
30 for incident. These are temporal associations, not proof of the causal
effect of either output error.

Evidence: P11 `diagnostic_action_phases.json` and `fitting_action_bounds.json`.
Only fitting and P11 training-profile traces were inspected. P10's reserved
third-column holdout and canonical data remain excluded.

## Paired experiment, declared before dispatch

Keep P11's neural weights fixed at SHA-256
`c5f38d4c5faff04fa6eb6fbf6c0169ac8643ea016bd27e88542fb578fe8178e2`.
Compare two shared observation/history policies on the same five 910x training
profiles. This is an action-bound diagnosis, not another neural optimizer run.
Do not choose different candidates for different scenarios.

1. `nuf_nonnegative`: on decisions 16-30, clip the neural NUF action below at
   zero. Keep its NP output unchanged. This isolates removal of NUF tightening.
2. `fitting_bounds`: on 16-25, also clip NP above at zero; on 26-30 cap any
   positive NP action at the nonnegative gap to the latched NP anchor divided
   by 50. On 16-30 cap positive NUF at the nonnegative gap to the latched NUF
   anchor divided by 1000. Thus capacity increases cannot overshoot that anchor.

Before 16 and after 30, both policies are exactly the P11 carry/return policy.
Bounds affect proposed actions only. Keep the physical guard, projections,
solver six iterations, carry/recovery implementation, ±2% profiles, full
75-interval TTT and terminal handling unchanged. Both bounds must leave every
fitting label unchanged within floating tolerance before any rollout.

Reuse all five authenticated P11 carries, k16 checkpoints and completed carry
controls. Copy the durable control/experience files with byte-hash provenance
to a new P12 output; do not rerun complete baselines or raw P11. Execute exactly
ten new 60-step continuations (two per profile), each with the full carry prefix
in TTT. The P11 neural results remain the paired reference. Keep every source,
spec and model version immutable during the run. Maximum four numerical workers,
one numerical thread each. Honor global, result-root, wave and slot STOP files.

Inspect full paired outcomes before deciding whether either bound is useful.
Neither a one-seed improvement nor a training-label test establishes success.
Admission remains ≥5 seeds per scenario, no ≥+10% degradation, and mean deltas
≤ -2%, -9%, -6%, +1%, -3% for 155/170/incident/skew15/190. Only qualifying shared
policies may be canonically registered and reproduced in fresh output folders.
If both are weak, preserve this causal comparison and use allowed training data
to declare the next bounded shared-model update; never tune on canonical results.

Source: `work/sdmpc_rl_p12_20261002`. Output root:
`results/sdmpc_rl_p12_20261002`. No automatic commit/push or OS changes.

## Dispatch failure and versioned repair

The first queue started at 2026-10-02 10:10:57 KST and finished as `failed`
at 10:11:07. All four dispatched workers exited 1 before calling the numerical
collector; the fifth profile had not been dispatched. Each traceback identified
the same missing UTF-8 encoding in the P12 wrapper's read of output `carry.json`.
The cached-carry import adds `reused_from`, whose Korean path cannot be decoded
with the Windows cp949 default. The earlier actor/physical preflight did not
exercise this wrapper. No new candidate branch or experience file was completed;
the only branch files in `wave1` are the five imported P11 carry controls.

Preserved the failed source, specs, wave, logs and completion record. Created
`work/sdmpc_rl_p12_recovery_20261002` with an explicit UTF-8 read. The worker
diff is exactly that encoding argument; the bounded policy and two specs are
byte-identical to the original versions. `preflight_recovery.py` reproduced
the cp949 error on all four failed metadata files and verified the repaired
wrapper passes the correct TTT and arguments to the collector. Syntax checks
passed. The original 600-label and physical preflight remains applicable and
is hash-linked by `preflight_recovery.json`.

The recovery queue was dispatched at 2026-10-02 10:16:34 KST (launcher PID 1092)
after confirming that no Python process remained and no applicable STOP was
present. Its new output is `results/sdmpc_rl_p12_20261002/wave2_recovery`.
It authenticates the failed version's source pins and adds all recovery source
pins, then imports the same five P11 controls and runs the original ten planned
candidate branches. The old wave is not resumed. Admission and policy behavior
are unchanged; no optimizer update is performed in P12.

At 10:17:53 KST the live coordinator was PID 6376 (parent launcher 1092).
The four numerical runtime children were 1076/6624/16604/16748, respectively
170/incident/190/skew15, with observed masks 1/2/4/8 and CPU time increasing
past 76 seconds. Their venv parents were 16440/12284/8036/9812. All matched
the recorded start time, command and queue parent. Thread environment values
were all one, error logs were empty, and no completion record existed.
Evidence is in `wave2_recovery/launch_identity.json` and the root's
`recovery_launch_check.json`. Recheck identities at the next continuation;
these PIDs must never be treated as permanent identities.

## Continuation checks

- Identify the live coordinator from `wave2_recovery/plan.json` and workers
  from `events.jsonl`, matching command, parent and creation time. A venv shim
  and its runtime child are one logical worker, not two numerical workers.
- Read `queue_recovery.stdout.log`, `queue_recovery.stderr.log` and
  `wave2_recovery/logs/*.log`; inspect `completion.json` if it exists. Do not
  read active worker `status.json`, which can interfere with atomic replacement.
- Honor STOP at repo, P12 root, both P12 waves and slots, P11 root/pilot/slots,
  and the established global result roots. Leave P9's intentional wave1 STOP
  intact. Do not rerun a wave that exists or any completed candidate.
- Maximum four numerical workers, masks 1/2/4/8, one numerical thread each.
  Source/spec/model files are pinned and must remain unchanged during execution.
- Once all five jobs finish with exit 0, run
  `work/sdmpc_rl_p12_recovery_20261002/analyze_wave.py` with the existing venv
  Python. It authenticates full accounting, controls, experience, source hashes
  and fresh bounded-neural replay, then writes one preserved `analysis.json`.
- Compare both shared bounds across all five paired training profiles. Continue
  with a declared bounded experiment and sufficient independent training seeds;
  no canonical evaluation until the unchanged admission gate is met.

## Final authenticated result

`wave2_recovery/completion.json` reports five worker exits 0 and completion at
2026-10-02 11:44:01 KST. At the 13:59 continuation no Python process remained
and no applicable STOP was present. The full analyzer authenticated all pinned
sources and weights, imported controls, carry accounting, 600 new terminal
transitions and exact fresh actor/memory replay. `analysis.json` is preserved.

Carry-relative TTT changes (%; negative is better):

| Shared candidate | 155 | 170 | incident | skew15 | 190 |
|---|---:|---:|---:|---:|---:|
| NUF nonnegative | +5.8936 | -9.2119 | -3.6431 | -6.1571 | -1.8509 |
| Fitting bounds | +1.0064 | -2.7632 | -0.3287 | -7.8665 | -7.4394 |

No >=+10% degradation occurred. Both still fail the 155 gain requirement and
have only one paired training seed per scenario. No canonical registration or
evaluation occurred. The stricter bound reduces the worst-profile loss and
improves 190, while the simpler bound performs better on 170 and incident.
The extra NP/NUF bounds were bundled, so their separate causal effects are not
established. P13 retains the stricter shared rule and refits the neural model
using the newly observed outcomes, followed by fresh 920x training profiles.
