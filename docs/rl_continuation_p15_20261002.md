# P15: collect paired evidence on NP strength and NUF recovery

> Complete and authenticated on 2026-10-03. Recovery preserved 11 candidates
> and finished the four missing branches. No candidate passed admission.
> Continue with [P16](rl_continuation_p16_20261003.md), which collects neutral
> and positive NP behavior. Keep the incomplete original `wave1` intact.

## Authenticated decision

P14 completed at 2026-10-02 19:24:54 KST with all five jobs exiting 0.
At the 21:59 continuation, no Python process or applicable STOP remained.
Full authentication passed source/model/cache hashes, exact imported carry
controls, 600 new terminal transitions and fresh action/memory replay.

Carry-relative TTT changes (%; negative improves):

| Policy | 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---|---:|---:|---:|---:|---:|
| P13 original | +2.6797 | -2.5854 | -1.7746 | -7.8223 | +2.5124 |
| P14 hold NP after 25 | +5.4256 | -0.5777 | -1.6838 | -6.3956 | +3.1811 |
| P14 return both after 25 | +5.4256 | -0.3049 | +0.8811 | -6.3956 | +2.7295 |

Both late-window changes are worse than P13 on all five profiles. Reject both
for further seed expansion. The paired intervention does not support the earlier
hypothesis that late NP tightening caused the P13 losses. It does not establish
that stronger tightening is generally beneficial. No admission or canonical trial.

An audit of the 80 unique allowed fitting/diagnostic trajectories across 20
profiles finds carry is best on three of four 155 profiles. On the remaining
155 profile the best measured gain is only 0.8742%. The best observed incident
gain across these four profiles is 3.6431%. These examples provide weak support
for the required mean gains of 2% and 6%, respectively. More cloning of the same
actions alone is not established as sufficient; collect bounded new behavior.
P10's reserved holdout and all canonical data remain excluded from this audit.

## Factorial comparison, declared before dispatch

Freeze P13's exact neural model and fitting bounds, including both-budget return
after decision 30. Model SHA-256:
`53ba39c306c2e830c3ab9a228cf0c8b95822b41abfc193d7c9c77d44214ca617`.
Base spec SHA-256:
`8ce29ace7515cd8d70bad23005510bddcc45f40eeec92ccd4712c40894c27b2a`.

Compare a two-by-two combination of NP tightening strength and NUF recovery
speed during decisions 16-30. The full-strength/learned-NUF cell is the completed
P13 reference, so run only these three new shared policies:

1. `half_np`: multiply negative NP actions by 0.5. Preserve positive NP actions
   and the learned bounded NUF action. This changes tightening strength only.
2. `restore_nuf`: preserve NP; replace the NUF action with the positive gap
   to its decision-16 latched anchor, divided by 1000 and clipped to [0, 1].
   This requests the fastest permitted NUF return after a fallback reduces it.
3. `half_np_restore_nuf`: apply both changes, to measure their interaction.

Outside 16-30 all three are exactly P13. The same rule applies to all scenarios;
no scenario/seed identifier, future outcome or per-scenario winner enters the
policy. Model weights, normalization, latched memory and action scales stay
fixed. Bounds remain within the original action limits and anchor ceilings.
The physical guard and solver still decide feasible execution; do not alter them.

This experiment performs zero neural optimizer updates. Its purpose is to
measure different requested actions and collect their actual complete returns
before another shared-model update. Do not claim that a smaller action or faster
NUF return must improve TTT. All three full outcomes must be authenticated.

## Bounded execution and checks

Use the five paired 920x training profiles and reuse authenticated P13 carries,
k16 checkpoints and carry-control records with byte hashes. Execute exactly
15 new 60-interval continuations and include the original carry prefix in full
75-interval TTT. Do not rerun P13, P14, or completed carry controls.
Maintain +/-2% demand, solver six iterations, physical constraints, true-terminal
experience, maximum four numerical workers, one numerical thread each, CPU
masks 1/2/4/8. Preserve source/spec versions; honor STOP before and during work.

Preflight must check exact action/memory equality with P13 outside 16-30,
each override formula inside the window, positive NP recovery preservation,
NUF bounds, reload and sequential-input checks, UTF-8 cached metadata reading,
and one actual physical interval. Use the existing compact-row conversion for
saved-versus-live row comparisons. Freeze preflight sources before dispatch.

After completion, authenticate source/cache/control hashes, terminal accounting
and fresh policy replay, then compare the whole factorial table. A promising
single shared candidate proceeds to preregistered independent training seeds.
If these actions are weak, use the authenticated diagnostics to declare the next
bounded exploration or shared-model update, preserving all failed outcomes.

Admission remains >=5 seeds per scenario, no >=+10% degradation, mean deltas
<= -2%, -9%, -6%, +1%, -3% for 155/170/incident/skew15/190. Only admitted
candidates are canonically preregistered. Final success still requires strict
five-scenario improvement and fresh-folder reproduction with the same frozen
policy. No threshold relaxation, holdout fitting, or canonical tuning.

Source: `work/sdmpc_rl_p15_20261002`; results: `results/sdmpc_rl_p15_20261002`.
No automatic commit/push or operating-system changes.

## Audit, preflight and dispatch

`audit_support.py` authenticated all 80 unique trajectory references, 20 paired
profiles, source pins and branch/experience hashes. The preserved
`support_audit.json` includes their explicit provenance and the best measured
training behavior per profile. Reserved holdout and canonical inputs are absent.
The audit is descriptive and is not an admitted policy or an oracle score claim.

Preflight passed 675 unchanged action/memory comparisons outside 16-30 and
225 override-formula checks inside the window, plus positive NP preservation,
NUF lower/upper clipping, reload, sequential/dtype checks, the UTF-8 worker
regression and one actual physical interval. The 151-file physical manifest
remains `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
`preflight.json` pins the six source files and three specs. Preserve them.

Started the hidden queue at 2026-10-02 22:09:14 KST (launcher PID 6696), after
checking that no Python process remained and no applicable STOP was present.
Output is `results/sdmpc_rl_p15_20261002/wave1`. The queue copies five exact P13
carry controls with their experience files and runs only the 15 new branches.
It maintains four numerical worker slots with one numerical thread each.

## Next heartbeat

1. Read this record, inspect STOP, then match actual PID/parent/start/command
   against `wave1/plan.json` and `events.jsonl`. A venv shim and its runtime child
   count as one numerical worker. Observe at most four numerical runtimes with
   masks 1/2/4/8 and numerical thread environment values one.
2. Read queue and `wave1/logs/*.log` stdout/stderr, completed branch names and
   `completion.json` if present. Do not read active worker `status.json` or
   duplicate an existing wave/completed branch. Honor repo, global result roots,
   P15 root/wave/slot and source P13 root/pilot/slot STOP. Preserve old P9 STOP.
3. After all five jobs exit 0, run the existing venv Python with
   `work/sdmpc_rl_p15_20261002/analyze_wave.py`. It authenticates complete carry
   accounting, 900 new terminal transitions, 300 reused control transitions,
   source/cache/control hashes and exact fresh strength-policy action/memory
   replay before writing a single preserved `analysis.json`.
4. Compare all three shared variants with P13 on all five profiles. Advance a
   promising fixed policy to declared independent seeds; otherwise declare the
   next bounded exploration or shared-model update from the allowed evidence.
   No canonical registration until the unchanged admission gates pass.

## Interrupted-wave audit, 2026-10-03

At the 01:59 KST heartbeat, neither the P15 coordinator nor any Python worker
was running. `wave1/completion.json` was absent. The event log records successful
exits for skew15, 170 and incident, but no exit for 190 or 155. All stderr logs
are empty and no applicable STOP was found. The last recorded 190 progress is
decision 40 of `restore_nuf`; 155 logged only its cached carry load. The OS boot
time was 2026-10-02 02:13:16 KST, before P15 started, so a new reboot was not
observed. The cause of process disappearance is unknown; do not label it a
numerical failure, user stop or completed experiment.

`work/sdmpc_rl_p15_recovery_20261003/audit_partial.py` authenticated all 80
original source/model pins, P13 cache hashes, imported controls, carry metadata,
the complete branch set and all corresponding terminal experience. Fresh actor
replay matched actions and memory exactly. The preserved receipt is
`results/sdmpc_rl_p15_20261002/recovery_audit_20261003.json`:

- 11 completed candidate branches and five reused carry controls (960 validated
  transitions). Their branch/experience/carry artifacts are pinned for byte-copy.
- Four missing branches: 190/9205 `restore_nuf`; 155/9201 `half_np`,
  `half_np_restore_nuf`, and `restore_nuf`.
- No orphan candidate experience or partial experience file was present.
- Full P15 conclusions and the next experiment wait until all 15 candidates
  are complete. No candidate is canonically eligible.

Recovery preserves `wave1`, the exact original worker/policy/specs and all
completed outputs. New orchestration lives in
`work/sdmpc_rl_p15_recovery_20261003`. The recovery queue copies authenticated
files into `wave2_recovery`, dispatches only the two incomplete scenario jobs,
and uses the unchanged worker's completed-branch skip behavior. The 190 job
runs one missing branch; the 155 job runs three. The incomplete branch resumes
from its original k16 checkpoint because no durable within-branch checkpoint
exists. No complete candidate or carry control is recomputed.

Python 3.12.14, Torch 2.14.0+cpu, NumPy 2.3.5 and SciPy 1.16.3 still match.
`preflight_recovery.py` checks a saved carry interval against the current physical
runtime before dispatch. All recovery sources are added to the pinned plan;
model parameters, evaluation criteria and the original P15 experiment remain
unchanged. Maximum two numerical workers are needed for this recovery, each
with one numerical thread, within the four-worker cap.

The recovery preflight passed: all 19 saved compact-row fields other than timing
matched exactly for the 190/9205 carry decision 16, including achieved flows,
fallback source, inventories, queues and interval/total TTT. The physical
151-file snapshot identity and numerical versions match the original run.
Receipt: `recovery_preflight_20261003.json`.

Started the hidden recovery queue at 2026-10-03 02:08:11 KST (launcher PID
13772), after again confirming no Python process remained and no applicable
STOP was present. This is the same declared P15 comparison with unchanged
options/model/source. Only incomplete branches are dispatched. The full paired
analysis remains pending until these branches finish.

At 02:09:29 KST coordinator 13412 and two numerical runtime children
9968/21288 matched the recorded command, start and parent identities; their
venv parents were 19240/17520. CPU masks were 1/2 and numerical thread settings
were one. Both had accumulated more than 55 CPU seconds. Error logs were empty,
all 84 source/artifact pins matched, and all 53 recovered files were byte-identical
to `wave1`. Evidence: `wave2_recovery/launch_identity.json`. Recheck identities
on subsequent heartbeats; these historical PIDs are not permanent identifiers.

### Recovery continuation

- Read `wave2_recovery/plan.json` and `events.jsonl`; match actual PID, parent,
  start and command. Inspect `queue_recovery.stdout.log`,
  `queue_recovery.stderr.log`, `wave2_recovery/logs/*.log`, completion and STOP.
  Do not inspect active worker `status.json` or launch another existing wave.
- Honor STOP in the repo/global roots, P15 root, both P15 waves and their slots,
  and source P13 root/pilot/slots. Do not remove the historical P9 STOP.
- Once recovery reports completion with successful remaining jobs, run
  `work/sdmpc_rl_p15_recovery_20261003/analyze_recovery.py` with the existing
  venv Python. It verifies the byte-preserved old artifacts, then invokes the
  original P15 full analyzer against `wave2_recovery`. Its final report covers
  all 15 candidates, 900 candidate transitions and 300 reused control transitions.
- Continue the original paired comparison and unchanged admission workflow.
  Recovery itself is not new learning, seed validation or success evidence.

## Final authenticated P15 result, 2026-10-03

Recovery completed at 03:15:14 KST with all five job records successful (three
reused original jobs, two recovery jobs). At the 05:59 continuation, no Python
process or applicable STOP remained. The full recovery analyzer passed: 53
preserved artifacts are byte-identical, 84 source/artifact pins match, all 15
candidate branches and five imported controls are complete, and fresh policy
action/memory replay matches. Terminal accounting covers 900 new candidate and
300 reused control transitions. Final evidence is
`results/sdmpc_rl_p15_20261002/wave2_recovery/analysis.json`.

| Policy | 155/9201 | 170/9202 | incident/9203 | skew15/9204 | 190/9205 |
|---|---:|---:|---:|---:|---:|
| P13 | +2.6797 | -2.5854 | -1.7746 | -7.8223 | +2.5124 |
| half_np | +2.4200 | +5.8318 | +3.1851 | -5.2096 | -6.3712 |
| half_np_restore_nuf | +2.4200 | +5.8318 | +4.7457 | -5.2096 | -3.9661 |
| restore_nuf | +2.6797 | -4.0757 | -3.3624 | -7.8223 | +2.3438 |

Values are full TTT changes versus carry in percent. No >=10% collapse occurred,
but every shared candidate misses the gain gate. Half-strength NP trades a large
190 gain for worse 170/incident outcomes. Faster NUF return improves 170/incident
relative to P13 while 155 remains unimproved. The two half-strength variants
have byte-identical experience on 155, 170 and skew15; this is preserved paired
evidence, not additional independent seeds. No canonical registration and no new
optimizer updates. Follow the P16 protocol for the next bounded action-direction
comparison; P10 reserved holdout and canonical results stay excluded.
