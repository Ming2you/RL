# Five-scenario continuation on this PC

> Latest continuation: [P17 v2 record](rl_continuation_p17_v2_20261003.md).
> P9-P17 v2 are complete and authenticated. The frozen P17 model improves all
> five fresh 930x training profiles: -2.2776%, -4.9422%, -5.1861%, -0.9534%,
> -2.6032% for 155/170/incident/skew15/190. It has one seed per scenario;
> admission and final canonical success remain unproven. No numerical job runs.
> Next: keep the model frozen and preregister four more training seeds per
> scenario. Inspect PID/logs/completion/STOP first; the existing pilot and its
> analysis are complete and must not be repeated. Results and models are saved
> in [the publication snapshot](rl_results_20261003/README.md).
> Preserve earlier versions and completed branches; never duplicate an active wave.

## Objective and fixed acceptance criteria

The user asked on 2026-10-01 to continue training until all five scenarios improve.
Work in `C:/Users/alsrj/Documents/끼/RL-sdmpc`, branch
`codex/sdmpc-rl-budget-20260929`, starting commit `ef3c4d5`.
Keep the parent checkout's unrelated uncommitted work intact.

Success requires one shared frozen observation-conditioned policy, identical across
155, 170, incident, skew15 and 190. Each complete canonical run must reduce TTT
relative to the authenticated carry center on this PC by more than
`max(1e-6, 1e-8 * center_TTT)`. Freeze source, model and configuration, then repeat
all five in new output directories. Both waves must pass. No improvement is guaranteed.

Keep the physical snapshot, 6 lower iterations, 5 warmup plus 75 control intervals,
14400 s horizon, physical feasibility guard and accounting unchanged. Keep the
existing U(0.98, 1.02) training profiles. Equal scenario sampling (20% each) applies
to shared-model training. Do not train on canonical observations/outcomes or select
candidates by repeated canonical trials. Register qualifying candidates in advance.

Training admission remains: at least five seeds per scenario, zero runs at or
above +10% TTT degradation, mean deltas <= -2% (155), -9% (170), -6% (incident),
+1% (skew15), -3% (190). The first P9 wave is diagnostic and cannot satisfy this
five-seed gate by itself. Structured-policy experiments are evidence for the next
shared learner; do not report them as completed neural training.

## Reusable environment and preservation

- Python: `C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe`.
- Python 3.12.14, torch 2.14.0+cpu, numpy 2.3.5, scipy 1.16.3, PyYAML 6.0.3.
- Machine fingerprint matches the historical machine B exactly: AMD64 Family 23
  Model 113 Stepping 0, Windows-10-10.0.19045-SP0, six CPUs, AVX2.
- Read existing raw data from
  `C:/Users/alsrj/Desktop/RL/results/sdmpc_rl_machine_b_20260930`.
- Read cached carries/checkpoints from
  `D:/RL_data/sdmpc_rl_machine_b_20260930/ckpt`.
- All five chosen carry and k16 checkpoint hashes match the Git archive inventory.
  All 18 Python files in `work/sdmpc_rl_multi_20260929` match the prior checkout.
  Evidence: `results/sdmpc_rl_p9_20261001/preflight_hashes.json`.
- C: had about 4.5 GB free at preflight. Avoid duplicate raw archive restoration
  and large replay/checkpoint copies. Initial P9 outputs are small JSON files.
- No Python worker was running at initial process inspection. No new dependency
  installation, git commit, push or power-setting change is part of this stage.

## P8 comparison confound discovered

The training branches called B1 used `probe.Option(after='return')`, which restores
**both NP and NUF** to the two-element pre-window anchor. The registered
`PerimeterActor` returns **NP only** and always emits NUF action zero. Fallback can
lower the carried NUF budget. Therefore the published B1 training table is not a
matched comparison of window length alone against P8's observation-based actor.
The previous single-slot equality check (190 seed 8605) does not establish equality
on other slots after a fallback changes NUF.

For 190 seed 8705, the old B1 option and P8 16-27 actor agree through decision 25.
At decision 26, the old option starts returning both budgets: NUF rises from
3915.26 to 4915.26. The P8 actor continues binding and keeps NUF at 3915.26; it later
falls to 3452.58 and stays there even after NP return. For skew15 seed 8804, old B1
returns NUF toward 6000 while P8 keeps it near 4519.39. This is a causal hypothesis,
not proof that restoring NUF will cure gridlock or meet the objective.

The raw `row['control_step']` is zero-based (0..74). Actor/CLI decisions are
one-based (1..75). New diagnostic CSV uses `decision_step = control_step + 1`.
Do not modify the frozen environment to rename its historical field.

Read-only trace extraction:

```powershell
& C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe -B work/sdmpc_rl_p9_20261001/prepare.py --data C:/Users/alsrj/Desktop/RL/results/sdmpc_rl_machine_b_20260930
```

Output: `results/sdmpc_rl_p9_20261001/diagnosis/paired_trace.csv` (240 rows) and
`inputs.json` with input hashes. Do not rerun `prepare.py` during an active wave;
the runner pins its option/spec/source files.

## P9 first wave: predeclared comparison

All four candidates use the unchanged original float32 observation decoder and
action limits. New code lives in `work/sdmpc_rl_p9_20261001`; registered P8 code is
unchanged. The recovery rule only proposes actions and never overrides the
physical guard or selects a control by its predicted TTT.

1. `b1_np_only`: exact registered B1 behavior, 16-25, NP-only return.
2. `long_np_only`: exact P8 16-27 behavior; reproduction control.
3. `long_both_return`: same 16-27 window, return both latched anchors.
4. `long_both_fallback_return`: same as 3, but after a fallback caused during
   binding, permanently switch to return on the next decision. Carry fallback
   before the window is ignored. Observations alone drive this rule.

Slots: 155/8701, 170/8702, incident/8703, skew15/8804, 190/8705. The two known
collapse seeds are included deliberately. Every slot also runs a zero-action
branch from checkpoint 16; its final TTT must exactly reproduce the cached carry.
Check the legacy long candidate reproduces available P8 results exactly. Inspect
complete trajectories and NUF budgets, not just final TTT. A failed reproduction
invalidates policy conclusions until diagnosed.

Use up to four workers (one Torch/BLAS thread each), low priority, hidden windows.
Queue starts the two collapse slots first, then 170, incident, and 155. The queue
records command, PID, start time, immutable specs/source hashes and completion.
Never read an actively written worker `status.json`; use logs/completed branches.
Honor STOP at repository, balanced-goal, machine-result, wave and slot roots.
Do not duplicate running/completed jobs or silently remove a STOP file.

## Validation and next actions

- Seven unit tests pass: exact legacy action equality, clipped two-budget return,
  pre-window fallback exclusion, immediate persistent recovery, fresh-episode
  determinism, invalid/nonsequential input rejection, and behavior-specific hashes.
- A one-interval smoke run is in `results/sdmpc_rl_p9_20261001/smoke_v1`.
  Its partial TTT and printed percent changes are **not** full-run performance.
- Full run location: `results/sdmpc_rl_p9_20261001/wave2` (launch details appended
  after startup). Do not assume a process is active from a stale PID alone.

When the wave finishes, authenticate completion and zero-action reproduction,
compare all four policies with each matched carry and historical P8. Summarize
fallback timing, NUF recovery, queues and TTT. If recovery removes collapse with
useful gains, extend the shared candidate to the other preserved training seeds
(>=5 per scenario) and then design shared-model learning from training-only data.
If it fails, record the falsified hypothesis and use the traces to choose a
different bounded experiment; do not repeat an unchanged failed setup.

Before any canonical evaluation, adapt/authenticate the evaluator for the new
actor identity and register the candidate with validation evidence. Preserve
existing registry entries and attempt counts. Canonical results remain outside
training. Do not claim success before both frozen five-scenario waves pass.

An hourly heartbeat in this chat should inspect process identity and durable
progress, continue analysis/training after each completed wave, and avoid duplicate
dispatch. Notify in Korean only for meaningful verified changes, actionable
failures, required user input or success. User pause/stop takes priority. The
computer and Codex app must remain running for local scheduled continuation.

## Launch log

- Heartbeat `rl-5` is ACTIVE, hourly, attached to this chat
  `01a0f654-78db-7a33-ad77-bb2f602e5b47`.
- Smoke completed: restored carry and all four candidate first intervals match
  their original expected TTT exactly. This does not replace full carry replay.
- The initial `wave1` was stopped by the assistant before policy comparisons to
  move NumPy import after thread limits are configured. Its STOP is an internal
  diagnostic stop and remains intact; never resume or score wave1. New work goes
  to wave2, using the same predeclared candidates/seeds and fresh source pins.
- Wave2 started 2026-10-01 16:38:36 KST. Launcher PID 8860, coordinator PID 13420.
  Four numerical workers loaded their authenticated cached carries without errors:
  190/8705, skew15/8804, 170/8702, incident/8703. 155/8701 remains queued.
  Verified worker CPU masks 1, 2, 4, 8; thread limits are now in the process
  environment before any numerical import. PID values are historical identifiers,
  not evidence of continued liveness.
- Queue stdout/stderr: `results/sdmpc_rl_p9_20261001/queue_wave2.*.log`.
  Worker logs and process events: `results/sdmpc_rl_p9_20261001/wave2/logs/` and
  `events.jsonl`. Wave2 source/spec files are pinned: do not edit them while active.
- After `wave2/completion.json` reports completed, run:

```powershell
& C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe -B work/sdmpc_rl_p9_20261001/analyze.py --wave results/sdmpc_rl_p9_20261001/wave2 --historical C:/Users/alsrj/Desktop/RL/results/sdmpc_rl_machine_b_20260930
```

The analyzer refuses partial waves, authenticates source hashes, all 60 remaining
intervals per branch and TTT sums, and requires exact carry and available P8
reproduction. Outputs are `analysis.json` and `analysis.csv`. If it refuses,
diagnose the cause before making any performance claim.

## 17:36 KST heartbeat: first completed training branches

The wave is still running with the original four numerical workers and 155 queued.
PID, parent PID, command line and creation time match the launch; affinity masks
remain 1/2/4/8. No active-root STOP, stderr error, or source/spec hash change was
found. No duplicate job was launched. All four dispatched slots have completed
zero-action restore controls with exact cached-carry TTT reproduction.

Authenticated complete B1 NP-only branches so far (negative delta is better):

| Training slot | Registered B1 behavior, NP-only return | Historical B1 probe, both-budget return |
| --- | ---: | ---: |
| skew15 seed 8804 | +15.0062% | +2.1819% |
| 170 seed 8702 | +1.7476% | +1.7807% |
| incident seed 8703 | -1.7460% | -1.7918% |
| 190 seed 8705 (completed during this check) | +28.1504% | -1.8444% |

The skew15 NP-only branch is a completed run above the predeclared +10% collapse
threshold. Its final NUF budget is 4523.1195 versus 5999.9820 in the historical
both-return probe. Thus the prior statement that B1 has no training collapses
does not apply to the registered actor implementation. These historical/current
branches also differ in float32 decoding, so they alone do not quantify the
isolated causal NUF effect. The ongoing long-NP versus long-both comparison uses
the same decoder and will test that effect. No claim about the recovery candidates
or canonical performance is yet supported.

Evidence: `results/sdmpc_rl_p9_20261001/heartbeat_1735.json` authenticates unchanged
sources, complete 60-row continuations, TTT identities and prefix-plus-interval
accounting, carry reproduction and input/output hashes. Whole-wave completion
and the full analyzer remain pending. Continue the same wave unchanged.

The 190 B1 NP-only branch also finished and passed the same per-branch accounting
and carry checks (`heartbeat_1737_190.json`). Its final NUF is 3780.7746, versus
5999.8154 in the historical both-return probe. Both known P8 failure slots now
show that the shorter registered B1 actor can collapse as well. Long-NP and
recovery candidates are still being evaluated; do not attribute this to window
length alone or relax the zero-collapse admission gate.

## 18:36-18:38 KST heartbeat: NUF recovery effect verified

170 and skew15 slots exited successfully. The queue dispatched 155/8701 at
18:32:42 on CPU 2 (launcher PID 1556, numerical worker PID 9912). Incident
finished its last branch during this check. The 190 slot is evaluating its
fallback-return branch; 155 is still running. No duplicate experiment was started.
Process command lines/start times match the queue. No active STOP, stderr failure
or pinned-source change; C: has about 4.68 GB free.

The legacy 16-27 NP-only policy reproduces the prior P8 TTT **exactly** on all
four earlier slots. Both-return and NP-only branches have identical actions,
budgets, queues and TTT through decisions 16-27 (timing measurements excluded).
Their first action difference is at decision 28, when NUF recovery begins.
This is a matched intervention on recovery behavior, unlike the earlier
historical B1 probe versus actor comparison.

Completed training deltas versus each slot's carry (%; negative is better):

| Slot | Long NP-only | Long both return | Long both, return on first binding fallback |
| --- | ---: | ---: | ---: |
| 190/8705 | +60.2354 | -0.6678 | pending |
| skew15/8804 | +15.6212 | +5.1199 | -0.5558 |
| 170/8702 | +3.5149 | +3.4233 | -0.2200 |
| incident/8703 | +0.4275 | +0.3827 | -1.2683 |
| 155/8701 | pending | pending | pending |

For the known 190 failure, restoring NUF removes the collapse in this paired
run and lowers TTT by 4170.2474 veh-h relative to NP-only return. It also reduces
the skew15 loss, but by itself still loses to carry there. Earlier fallback
return helps skew15 on this seed. Gains on 170 and incident remain far below
the predeclared mean targets; no candidate is admitted to canonical evaluation.
These are diagnostic structured policies, not newly trained neural models.

Evidence: `heartbeat_1835.json` authenticates completed branches, source hashes,
TTT accounting, carry checks and P8 reproduction. `paired_recovery_check_1837.json`
records prefix equality and first action difference for all four matched pairs.
The incident fallback branch also finished with TTT 5988.026117148412 and passed
60-row/timeline checks (SHA-256
`2ac02ea6a90e18a2c0ba619801782a87365b54e19609dc5eb0dd11632f30310c`).

Continue wave2 unchanged. After full authentication, assess the shared fallback
candidate across all five before deciding the next bounded experiment. A corrected
10-interval B1 with both-budget return and the same observation decoder is a
useful missing comparison if the current candidates remain too weak. The old
B1 table cannot substitute for testing that exact candidate. Do not expand a
weak long-window candidate solely because it fixed the known collapse.

## 19:37 KST heartbeat: 23 of 25 branches complete

The 190 slot exited successfully at 19:00:53. Only 155/8701 remains active
(launcher 1556, numerical PID 9912, start 18:32:42, CPU mask 4), with the original
queue 13420 still alive. Current process command/start identities match the
record. There are no active STOP files, stderr errors or source/spec changes.
No overlapping experiment was launched; wave2's final two branches are pending.

All five carry controls now reproduce their cached TTT exactly. All five
`long_np_only` branches also reproduce the historical P8 results exactly.
`heartbeat_1936.json` authenticates the 23 completed full continuations, including
their 60 sequential rows, TTT accounting, output hashes and unchanged source pins.

New completed results (training deltas, negative is better):

- 190/8705 `long_both_fallback_return`: TTT 6929.900592182065,
  **+1.2059%**, versus -0.6678% for waiting until decision 28 to return both.
  Early return removes the large collapse but does not improve this seed over
  carry; it is not a five-scenario solution. Do not generalize the skew15 benefit.
- 155/8701 `b1_np_only`: **-0.1434%**.
- 155/8701 `long_np_only`: **-0.3422%**, exact historical P8 reproduction.
- 155 both-return and fallback-return are still running and must not be scored
  from their partial logs.

The whole-wave analyzer remains deferred until completion. None of these
candidates is canonically admitted: safety repair alone does not meet the
170/incident gain requirements or the five-seed training gate. Preserve this
wave and finish its two pending branches before selecting the next bounded
comparison. The useful next missing arm remains the observation-based
10-interval B1 with both-budget return, not the historical simulator-state probe.
