# P14: isolate the effect of late neural NP tightening

> Complete and authenticated. Both variants were worse than P13 on every paired
> profile and are rejected for seed expansion. Latest continuation:
> [P15 NP-strength/NUF-recovery comparison](rl_continuation_p15_20261002.md).

## Authenticated starting point

P13's five prospective training profiles finished with exit 0. The full analyzer
passed five 75-interval carries, exact restored controls, 600 terminal transitions,
all source/model hashes and fresh bounded-neural action/memory replay. At the
17:59 continuation no Python process or applicable STOP remained.

P13 carry-relative TTT changes (%; negative is better):

| 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---:|---:|---:|---:|---:|
| +2.6797 | -2.5854 | -1.7746 | -7.8223 | +2.5124 |

No >=+10% collapse occurred, but 155 and 190 worsened and the gain/seed gates
are not met. The lower fitting loss did not establish better control. No P13
candidate is registered for canonical evaluation.

## Diagnosis and experiment declared before dispatch

All five neural branches continued negative NP actions at every decision 26-30.
The 155 branch used approximately -0.91 to -0.99 in that period; its losses were
+8.96 veh-h during 26-30 and +71.34 during 31-45. The 190 branch lost +37.18
and +123.56 veh-h in those periods. Earlier 16-25 loss was only +2.65 for 155
and -0.82 for 190. NP then returned at the maximum allowed rate from decision
31. These are temporal associations, not proof that ending earlier improves TTT.
Evidence: `results/sdmpc_rl_p13_20261002/diagnostic_action_phases.json`.

Keep the exact frozen P13 model, normalization and P12 fitting bounds. Model
SHA-256 is `53ba39c306c2e830c3ab9a228cf0c8b95822b41abfc193d7c9c77d44214ca617`;
base spec SHA-256 is `8ce29ace7515cd8d70bad23005510bddcc45f40eeec92ccd4712c40894c27b2a`.
Compare exactly two shared observation/history policies on the same five 920x
training profiles, with no new optimizer update or scenario-specific switch:

1. `return_both_w25`: identical to P13 through decision 25; at 26-30 return
   both observed budgets toward the decision-16 latched anchors, using the
   original action scales and clipping. After 30 it is again exactly P13.
2. `hold_np_after25`: identical to P13 except at 26-30 replace a negative NP
   action with zero. Keep P13's NUF action and any positive NP action. After
   30 keep P13's original both-budget return. This isolates stopping additional
   late NP tightening from bringing the entire recovery window forward.

All changes are to proposed actions. Preserve the physical guard, solver six
iterations, physical limits, +/-2% demand distribution, full 75-interval TTT
and true-terminal handling. Prefix actions and observations before 16 are carry.
Policy memory and latched anchors are identical to P13; no future information,
scenario/seed identifiers or outcomes enter policy execution.

## Verification and bounded compute

Before dispatch, verify exact equality with P13 on its recorded observations
outside decisions 26-30, formula equality inside that window, identical memory,
repeat-load determinism, sequential and dtype checks, and one physical interval.
Retain the explicit UTF-8 metadata read that repaired the P12 startup error.

Authenticate and reuse all five completed P13 carries and k16 checkpoints.
Copy the carry-control branch and experience files with byte hashes into a new
wave. Run only ten new complete 60-step continuations (two per profile), each
including the full carry prefix in its reported TTT. Do not rerun carry or P13.
Use at most four numerical workers, each with one numerical thread and masks
1/2/4/8. Preserve all source/spec/model versions. Honor STOP at repo, result,
wave and slot roots. Do not read active `status.json`.

After completion, authenticate all terminal records and fresh policy replay,
then compare the whole paired table. A promising shared policy must receive
independent seed validation; these five reused profiles are diagnostic training
data, not new independent evidence. If neither candidate is useful, preserve
the comparison and use it to design the next bounded shared-model update.

Admission stays >=5 seeds per scenario, no >=+10% degradation, and mean deltas
<= -2%, -9%, -6%, +1%, -3% for 155/170/incident/skew15/190. Canonical data and
P10's reserved holdout remain outside fitting/tuning. Only passing candidates
are preregistered. Final success still requires all five canonical gains and
the same frozen policy reproducing them in fresh folders.

Source: `work/sdmpc_rl_p14_20261002`; results: `results/sdmpc_rl_p14_20261002`.
This is a paired diagnosis around a trained neural model, with zero optimizer
updates in P14. No automatic commit/push or operating-system changes.

## Preflight repair

The first preflight passed 550 unchanged-action checks, 50 override-formula
checks and the UTF-8 worker regression, then reached the actual physical
interval. Its comparison incorrectly accessed `source` on the raw environment
row, although that field belongs to the compact saved-row schema. It failed
with `KeyError: source` before issuing a passed receipt or dispatching a wave.
Preserved `preflight.py` and `preflight_v1_failure.json`; `preflight_v2.py`
uses the existing `probe.compact` conversion before comparing saved-row fields.
No model, policy rule, spec or physical code was changed by this repair.

## Passed preflight and dispatch

`preflight_v2.py` passed all 550 comparisons outside decisions 26-30 and all
50 override-formula comparisons inside that window, with identical memory.
Reload, sequential input, dtype/finite checks, UTF-8 worker metadata and the
actual first physical interval passed. Its TTT and compact source matched P13.
The original physical manifest remains unchanged. `preflight.json` pins the
six source files and both specs; do not rerun it or modify those files.

The queue started at 2026-10-02 18:10:27 KST (launcher PID 7564), after confirming
no Python process remained and no applicable STOP existed. Output:
`results/sdmpc_rl_p14_20261002/wave1`. It imports the five exact P13 carry-control
branches and experience records and dispatches only the ten new candidate
branches. The 920x profiles are paired diagnostic training profiles here.
No neural optimizer update occurs in P14 and no canonical trial is registered.

At 18:12:15 KST coordinator 14412 and four numerical children
16516/9212/12712/15996 matched the queue's start, command and parent identities.
Their venv parents were 8092/9048/6612/6236; masks were 1/2/4/8 and CPU time
exceeded 105 seconds each. All numerical thread environment values were one.
The four worker and queue error logs were empty, all 79 source/artifact hashes
and 15 imported control-file hashes matched, and no completion file existed.
Evidence: `wave1/launch_identity.json`. Recheck live identities next time;
historical PIDs alone never authorize process actions.

## Next heartbeat

- Identify the actual coordinator and workers using `wave1/plan.json`,
  `events.jsonl`, and live PID/parent/start/command. A venv shim plus runtime
  child is one logical worker. Limit numerical runtimes to four, masks 1/2/4/8,
  all numerical thread environment values one.
- Inspect `queue.stdout.log`, `queue.stderr.log`, `wave1/logs/*.log`, STOP and
  `completion.json` if present. Do not read active `status.json`; do not duplicate
  the wave or completed branches. Honor global, P14 root/wave/slot and source
  P13 root/pilot/slot STOP. Preserve the intentional old P9 STOP.
- After five worker exits 0, run the existing venv Python with
  `work/sdmpc_rl_p14_20261002/analyze_wave.py`. It checks full accounting, source
  pins, byte-identical imported controls, cache hashes, terminal experience and
  fresh late-window actor replay, then writes one preserved `analysis.json`.
- Compare both shared policies across the entire paired table. If promising,
  preregister additional independent training seeds for the same frozen policy;
  otherwise declare the next bounded shared-learning step. Keep the unchanged
  admission, canonical separation and final fresh-folder reproduction gates.

## Final authenticated result

All five jobs exited 0 at 2026-10-02 19:24:54 KST. The full analyzer passed all
source/cache/model pins, exact imported controls, terminal TTT accounting and
fresh action/memory replay for 600 new transitions. No >=+10% collapse occurred.

Carry-relative TTT changes (%; negative is better):

| Variant | 155 | 170 | incident | skew15 | 190 |
|---|---:|---:|---:|---:|---:|
| Hold NP after 25 | +5.4256 | -0.5777 | -1.6838 | -6.3956 | +3.1811 |
| Return both after 25 | +5.4256 | -0.3049 | +0.8811 | -6.3956 | +2.7295 |

Every cell is worse than the matched P13 reference. Neither candidate advances
to independent seeds or canonical registration. The paired interventions do not
support the hypothesis that ending late NP tightening would fix P13's losses.

The equal TTTs on 155 and skew15 were audited separately in `equal_ttt_audit.json`.
The two candidates have different action and requested/executed budget records,
but identical achieved flows, sources, interval costs, inventory and all queues.
This verifies that the equal costs are consistent with the saved trajectories;
different budget requests need not change achieved physical flows.

P15 audits the 80 unique allowed trajectories and collects new paired actions
using three fixed variants of NP strength and NUF recovery, keeping P13's
neural model frozen. Existing results remain preserved and are not rerun.
