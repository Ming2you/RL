# Local collection v2 execution record

LATEST: COMPLETE, no running worker/supervisor. All750transitions reconciled;
predeclared screenFAILED170. Do NOT resume this completed cohort or train an
actor from the failed screen. Read rl_budget_local_collection_results_20260930.md
and the SDD ledger/rebase-diagnosis-brief.md for next bounded investigation.

Historical resume record as of2026-09-30 06:30KST follows:
Carry155 output-only recovery and completed-output verification both passed.
Original75transition checkpoint/settings/schema/timing/logs are unchanged.
Four positive-inf rejected-candidate stationarity diagnostics have explicit
JSON tags; raw numerical inf remains in the checkpoint. No physical interval
was repeated. The failed status is archived before its completed replacement.
Remaining carry170/incident/skew/190 resume from34/48/51/42, then the original
runner collects the five paired local trajectories. No policy improvement yet.

## Completed bounded continuation

- Launch06:29:31KST, launcher2856, actualsupervisor25164, creation FILETIME
  134351909711276110. Command:
  `.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_export_recovery_20260930/supervise.py --execute`.
- Parent process/command/log record:
  `local_budget_v2/export-supervisor-parent-20260930_062931.json`.
- Supervisor durable state: `local_budget_v2/.export-supervisor/state.json`;
  attempt314820db745b49d0b5608d0fac51f954. Actualcoordinator44524 delegates to
  unchanged v2; originalwaveattempt5575201be550434999246c6137ecdd98.
- CarryworkeractualPIDs170=50604,incident=50756,skew=23760,190=5872. Check live
  creation identities before acting; these recorded PIDs are not timeless.
- At most5 one-thread numericalworkers. No unrelated numericalprocess was
  present at launch. All numerical sources/configurations remain frozen.
  Do not independently launch run_wave or another supervisor while this is live.
- Companion133testsPASS63.48s; independent scoped SPEC/QUALITY PASS, all3review
  findings closed. All214frozen source/result inputs matched before execution.
- Carry155 receiptSHA:
  `8e54beb7fcfb8e7ec322aed45b18ef1096ae7497e6adc83f00e8f6228ee5a83e`.
  Original TTT3210.2043413811693, inventory412.5094380679876 unchanged. This is a
  perturbed training profile, not the canonical155baseline/evaluation.
- Original defined worker-session time838.5200439000037s. Additional successful
  export scope5.575445799971931s; excludes process startup/teardown, receipt and
  final completion publication. Do not claim these as full end-to-end runtime.
  Supervisor attempt durations and logs preserve continuation/export overhead.

The prior05:28KST stop was an output-only diagnostic JSON failure. Its250saved
intervals remain the continuation prefix; no recollection or remigration.

Historical start record below: approximately05:14KST RUNNING, training only.
The main five-scenario shared-policy goal remains ACTIVE and NOT achieved.
No new learned policy, canonical evaluation, or traffic improvement is claimed.

## Historical initial execution

- Source: `work/sdmpc_rl_local_20260930_v2` (immutable while running).
- Output: `results/sdmpc_rl_balanced_goal_20260930/local_budget_v2`.
- Start:2026-09-30T05:11:46.7999076+09:00, hidden launch PID3100.
- Actual coordinator PID46348; attempt `b96cfc7cd0164c15a0c26e02ebdff860`.
- Command: `.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_local_20260930_v2/run_wave.py --resume`.
- Wave1 collects five carry trajectories; wave2 then collects the five paired
  local exploratory trajectories. Same demand seeds6801..6805, behavior seeds
  6901..6905, physical contract and exploration formula. No learner is loaded.

| Scenario | Actual Worker PID | Windows Launcher PID | CPU Mask |
| --- | ---: | ---: | ---: |
| sweet_155_w | 50024 | 39104 | 1 |
| sweet_170_w | 1736 | 40364 | 4 |
| sweet_170_incident_w | 24112 | 50300 | 16 |
| sweet_170_skew15_w | 49320 | 42108 | 64 |
| sweet_190_w | 3036 | 42128 | 256 |

All five workers advanced actual intervals with empty stderr. Carry155 stdout
starts at `3/75`, then4,5,6,7: its preserved first two intervals were not repeated.
Windows process queries independently confirmed all five affinity masks.
Creation identities and current progress are in the per-slot process/session
files and `.ownership/reservations.json`; these PIDs are historical, not a reason
to assume a process is still alive later.

A separate project, `C:/Users/alsrj/Documents/Numerical Simulation`, was running
one pinned single-core simulation (PID49668,mask8) and a serial waiting queue
(PID48984). Read-only inspection confirmed sequential dispatch, not a parallel
pool. Neither was changed or messaged. With this cohort the observed numerical
worker count is six, within the total eight-worker budget. Recheck before future
dispatches; Windows redirector processes are not extra numerical workers.

## Admission and preservation

The v1 local implementation passed200synthetic tests and independent review.
Actual standalone carry155 saved interval1 (PID28800) and resumed interval2
(PID45436), both exit0. Its defined locked-worker time was66.75636630004738s.
The initial parallel wave failed before any child boot due to the Windows venv
redirector PID differing from the real interpreter PID. All its processes exited.

V2 separates launcher and worker identities while preserving token claim-once,
creation-time checks, unknown/live fail-closed behavior and the five-worker cap.
The v2 suite passed236tests in78.46s, including the real stdlib-only process
probe. Independent SPEC and QUALITY review passed. Reports and diffs are under
`.superpowers/sdd/rl_budget_local_collection_plan_20260930/`.

The parent then ran, in order (all exit0):

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/migrate_prefix.py
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/migrate_prefix.py --execute
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/migrate_prefix.py
```

The first call verified the real frozen checkpoint by loading/restoring without
reset, step or PFO. The second published the migration; the third recognized and
verified the committed migration without another publication. Exact equality
held for physical contract, full saved state, two transitions, observation,
policy/RNG and session bytes. Only settings.identity changed in the new copy.
Boundary k7,1260s; run ID2962784e595d4ee8b1272dda7762dbc6 retained.

Migration receipt `.migration/committed.json` SHA256:
`f1b895d659c930708782e3b4fd3e99b21f3ca5ff6540e315880de8ab1dbf2a43`.
Total remains750transitions:2 reused+748 additional,150per scenario.
All54 original v1 source/result files were rehashed after migration/restart:
zero length or hash mismatches. Original source, failed logs and checkpoint are
preserved, including byte-exact provenance in the v2 migration staging record.

Timing is the explicitly defined aggregate locked-worker session scope.
Failed coordinator/interpreter startup cost and migration validation are outside
that scope and are not represented as zero. No full evaluation runtime exists
for a newly improved policy yet.

## Next actions

1. Inspect current process creation/commands, status, logs, STOP and completion;
   do not launch another coordinator while this one or its workers are alive.
2. Wait for both five-episode waves and the authenticated paired comparison.
   Do not interpret partial TTT, collection completion or a passing coverage
   screen as achievement of the learned-policy goal.
3. Reconcile750transitions,150per scenario, paired profiles, interval rewards,
   terminal inventories, budget drift/zero requests, physical fallbacks, actual
   response diversity and timing. Apply the unchanged predeclared screen.
4. If justified by that evidence, specify/test a shared learner revision before
   training. `rl_budget_next_learner_notes_20260930.md` is conditional guidance,
   not a preapproved algorithm or automatic actor admission.
5. Evaluate one frozen learned checkpoint on all five canonical full runs;
   apply the strict per-scenario improvement and separate reproduction gate in
   `rl_budget_balanced_goal_20260930.md`. Otherwise diagnose and iterate.

After a proven stopped/failed coordinator, use normal v2 `run_wave.py --resume`
only after checking live children, locks and STOP. Do not remigrate once progress
has advanced, modify source/configuration, clear ambiguous reservations, delete
STOP files, recollect completed episodes or revive paused legacy experiments.
The three-hour heartbeat remains authorized; unchanged checks should stay quiet.

## Terminal export diagnosis

The immutable collector calls strict json.dumps(allow_nan=False) before publishing
trace.json. Carry155 reached14400s and all physical/sequence/policy validators
passed, but four positive infinities in candidate stationarity diagnostics at
control_step14 stopped serialization. A full read-only trace scan found no other
nonfinite leaves. The selected execution at that interval was reference_fallback,
converged=false, physical_control_valid=true and budget_feasible=true. Thus the
nonfinite candidate must remain explicit as rejected/nonconverged evidence;
it must not be changed to zero or interpreted as an invalid executed reward.

Read-only revalidation of the saved75step checkpoint passed. Training-only
summary: TTT3210.2043413811693, terminalinventory412.5094380679876, zeroNUF0,
3physicalfallbacks,1initialPFO/noPFOrecovery,75lowerbudgetsolves,4unique executed
budgets and35physical control vectors. Decision wall783.9111446s,CPU749.15625s;
aggregate worker-session time838.5200439s including its preserved prefix.
These are the perturbed155training profile, not canonical goal evaluation.
Do not compare this TTT directly with the canonical3103.0110715680044 baseline.

The repair will export only allowlisted positive-inf candidate diagnostics as
explicit JSON tags and retain raw values in the unchanged checkpoint. Physical
and learning fields stay exact; any other nonfinite data remains an error.
Additional exporter time will be measured separately. A bounded companion may
then resume the same immutable v2 runner, skipping completed slots, and repair
only this known terminal export failure. It is not yet admitted or executed.
