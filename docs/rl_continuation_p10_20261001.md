# P10: retain NUF capacity while learning from paired training continuations

> Complete and authenticated: `results/sdmpc_rl_p10_20261001/wave2_recovery`.
> The 60 branches and 3600 transitions pass all checks. No policy qualifies
> for canonical admission. Continue at [P11](rl_continuation_p11_20261002.md).
> Preserve both P10 waves and their pinned sources; do not rerun their queues.

## Entry point and goal

Continue the user's five-scenario shared-policy objective from
`rl_continuation_p9_20261001.md`. Repository:
`C:/Users/alsrj/Documents/끼/RL-sdmpc`.
Keep the existing `rl-5` heartbeat and all success/admission rules unchanged.
The original canonical five-scenario target has not been achieved.

P9 completed at 20:01:08 KST, all five workers exited 0, and its full analyzer
passed at this continuation. All five carry controls and all five historical P8
NP-only policies reproduced exactly. See
`results/sdmpc_rl_p9_20261001/wave2/analysis.json` and `analysis.csv`.

P9 final training deltas (%; negative is better):

| Scenario/seed | B1 NP-only | Long NP-only | Long both return | Early fallback return, both |
| --- | ---: | ---: | ---: | ---: |
| 155/8701 | -0.143 | -0.342 | -0.397 | -1.836 |
| 170/8702 | +1.748 | +3.515 | +3.423 | -0.220 |
| incident/8703 | -1.746 | +0.428 | +0.383 | -1.268 |
| skew15/8804 | +15.006 | +15.621 | +5.120 | -0.556 |
| 190/8705 | +28.150 | +60.235 | -0.668 | +1.206 |

Restoring NUF fixes the large 190 collapse in this paired run, but restoration
alone does not provide sufficient 170/incident gains. Immediate return after the
first binding fallback helps four slots slightly but loses 190. No P9 policy
qualifies for canonical evaluation; this remains one seed per scenario.

## Predeclared P10 hypothesis and policy arms

Fallback can lower the carried NUF cap while NP continues to tighten. Waiting
until the binding window ends leaves that cap depressed for several decisions.
Test whether keeping the latched pre-window NUF cap during binding improves the
physical trajectory. This changes budget proposals only: solver, physical guard,
projection and scoring stay frozen. No performance-based control override.

Three observation/history policies, all shared across the five scenarios:

1. `b1_both_w25`: NP bind -50 below achieved on decisions 16-25; restore both
   original budget anchors afterwards. This corrects the missing B1 comparison.
2. `long_both_w30`: same behavior through decision 25 but bind through 30, then
   restore both anchors. Comparison with 1 isolates window length.
3. `retain_nuf_w30`: same as 2, additionally steer NUF toward its pre-window
   latched anchor during binding. Comparison with 2 isolates when NUF recovers.

All use the same float32 observation decoder, NP action limit 50 veh, NUF action
limit 1000 veh/h, latched return, and no early fallback abort. Original P8/P9
sources and completed outputs are unchanged. New code is in
`work/sdmpc_rl_p10_20261001` and has distinct module names to avoid import clashes.

## Balanced training collection

Three seeds per scenario, with existing verified carry/checkpoint caches:

| Scenario | Seeds |
| --- | --- |
| 155 | 8701, 8501, 8801 |
| 170 | 8702, 8502, 8802 |
| incident | 8703, 8503, 8803 |
| skew15 | 8804, 8504, 8704 |
| 190 | 8705, 8505, 8805 |

Run one carry control and three policy branches on each slot: 60 complete
continuations in total. Each starts before decision 16 and ends at decision 75
(14400 s); total TTT includes the authenticated carry prefix and warmup. The
cache is read from `D:/RL_data/sdmpc_rl_machine_b_20260930/ckpt`; every chosen
carry and checkpoint hash must match the Git archive inventory before dispatch.

This collection additionally saves float32 observations, actions, next observations,
reward, true-terminal flags and the policy's four deterministic memory features.
P9 stored only compact rows, so the carry reruns on its five slots collect new
baseline experience and verify that recording has not changed physical results.
Do not rerun any completed P10 slot without a specific recovery diagnosis.

Expected training data: 3600 transitions, exactly 720 per scenario. They are
60-step continuations with the true terminal, not full-reset 75-step replay.
Do not invent the first 15 transitions or bootstrap past the terminal. Retain
the profile, behavior-policy and source identities. Rewards are -interval TTT/100,
gamma=1. Entire rollout return labels apply to the recorded continuation; they
are not optimal Q labels. Carry and policy continuations are training-only.

The first two seed columns are prospective fitting data; the third column is a
training-distribution holdout for any later value/policy fit. Do not silently
train on that holdout after looking at its errors. Keep scenario sampling at 20%
per neural training minibatch. P10 itself performs no optimizer updates.

## Validation and acceptance

- P10's seven tests pass, covering exact legacy equivalence for both window
  lengths, NUF-only intervention during binding, latched and clipped recovery,
  deterministic fresh episodes, invalid settings, actual collector serialization
  using a synthetic 60-step environment, corruption rejection and STOP behavior.
- The real preflight restores 190/8705 and executes one controlled interval.
  All three policies propose [-1, 0] initially and reproduce P9's first-interval
  TTT exactly (439.93403129146316). Physics snapshot verification and one Torch
  thread pass. This partial interval is not a performance score.
- After full completion, `p10_analysis.py` checks source pins, all 60 rows per
  continuation, TTT sums, carry reproduction, array hashes, sequential observations
  and memory, action/reward alignment, true terminal, training-only provenance,
  and exact replay through a fresh actor. Never score partial outputs.
- At least five training seeds per scenario and the existing mean-delta gates
  (-2% 155, -9% 170, -6% incident, +1% skew15, -3% 190) plus zero >=+10% collapses
  are still required before canonical registration. P10's three seeds alone
  cannot admit a policy. Report per-scenario results and failures, not only mean.
- Do not choose scenario-specific checkpoints or a post-hoc mixture. If an arm
  shows useful repeatable response, extend independent training validation and
  use the recorded data for a single calibrated shared learner. If all are weak,
  analyze state/control differences and revise the hypothesis; do not blindly
  run more actor gradients against an uncalibrated critic.

## Execution and recovery

Python: `C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe` (the verified
existing environment). Launch hidden, low-priority workers, at most four, each
pinned to one logical CPU and with numerical thread limits set before import.
Do not change power settings, install dependencies or commit/push automatically.

Output: `results/sdmpc_rl_p10_20261001/wave1`.
The plan pins sources, all specs, cached artifacts and prerequisite evidence.
Queue events include PID, command and creation time. Use stdout/stderr and
completed branch files to inspect progress; do not read active `status.json`.
Honor repository, balanced-goal, machine-result, wave and slot STOP files.
Keep the old internally stopped P9 wave1 untouched.

Each completed branch has `branches/*.json` plus matching
`experience/<policy>.npz` and `.json`. A missing companion file, orphan experience,
failed worker, source mismatch or incomplete timeline must be diagnosed before
reuse. Preserve failed outputs. Never overwrite an existing wave or experience.

When `wave1/completion.json` reports completed:

```powershell
& C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe -B work/sdmpc_rl_p10_20261001/p10_analysis.py --wave results/sdmpc_rl_p10_20261001/wave1
```

Next: authenticate, summarize per-scenario/seed results and NUF/queue responses,
then continue toward training a shared policy and the unchanged five-scenario
canonical/reproduction gates. Do not stop the continuation merely because this
collection finishes. Notify only for meaningful verified findings or action needed.

## Launch record

- 2026-10-01 20:47:55 KST: wave1 launched hidden. Launcher PID 14728,
  coordinator PID 16744. Process identities are historical; verify liveness and
  command/start time at each continuation.
- First slots: 190/8705, skew15/8804, 170/8702, incident/8703. All loaded their
  cached carries successfully. Numerical worker CPU masks are 1/2/4/8; no startup
  stderr error. The queue holds the remaining eleven slots.
- Cache preflight passed for every selected carry and checkpoint against the
  archive inventory. All P9 pinned sources remained unchanged. P10 plan records
  its source/spec/cache pins and P9-analysis/P10-preflight hashes.
- Queue logs: `results/sdmpc_rl_p10_20261001/queue.stdout.log` and
  `queue.stderr.log`. Worker logs/events live inside `wave1`.
- Expected output is 60 branches and 3600 transitions, equally distributed over
  five scenarios. No optimizer update or canonical run was launched.
- Source files in both P9 and P10 are now pinned by the active queue. Make any
  future implementation change in a new version after stopping/draining the
  affected run; do not change pinned code/specs during collection.

## 21:39 KST heartbeat: collection and replay checks pass

The original four workers and queue are active with matching command lines,
parent PIDs and creation times. Numerical affinities remain 1/2/4/8. No active
STOP, stderr error or source/spec hash change. No duplicate dispatch or code
change was made; the first four slots continue through their policy branches.

Six complete branches, 360 transitions, are authenticated in
`results/sdmpc_rl_p10_20261001/heartbeat_2137.json`. All four carry controls
reproduce exactly. The completed corrected-B1 branches are 170/8702 +1.7807%
and skew15/8804 +2.2178%. These do not meet improvement gates and are not a
scenario-wide result.

The actual saved experience passes file hashes, float32 shapes, sequential
observations and policy memory, action/reward alignment, true-terminal handling,
training-profile provenance, and fresh-actor replay with identical actions.
This confirms the collector works on real trajectories as well as synthetic
tests. No optimizer update, holdout fitting or canonical evaluation has occurred.
Continue the predeclared collection unchanged; the whole-wave analyzer remains
pending. No user action is needed for this routine check.

## 22:39 KST heartbeat: two slots complete, 14 branches authenticated

170/8702 and skew15/8804 exited successfully. Their queue slots advanced to
155/8701 (numerical PID 10208, start 22:26:53, mask 4) and 170/8502
(PID 7828, start 22:29:13, mask 2). Original 190/8705 and incident/8703
workers remain active as PIDs 16280 and 7028, masks 1 and 8. Coordinator 16744
and launcher 14728 retain their original command/start identities. There are
still exactly four numerical workers; no extra dispatch was made.

`results/sdmpc_rl_p10_20261001/heartbeat_2239.json` authenticates 14 completed
branches and 840 transitions. Four carry controls reproduce exactly. Source
and spec hashes, experience hashes, timeline/accounting, train-only provenance,
terminal handling, observation/memory sequence and fresh-actor action replay
all pass. No active STOP or stderr error was found; whole-wave completion is
still pending. C: has approximately 4.7 GB free (decimal).

Completed training deltas (% versus paired carry; one seed per row):

| Scenario / seed | B1 both, end 25 | Long both, end 30 | Retain NUF, end 30 |
|---|---:|---:|---:|
| 170 / 8702 | +1.7807 | +0.7656 | +3.1191 |
| skew15 / 8804 | +2.2178 | +8.3031 | +2.8670 |
| incident / 8703 | -1.7918 | -2.3372 | pending |
| 190 / 8705 | -1.8028 | -1.4389 | pending |

Retaining NUF reduces the long-window loss on the completed skew15 seed but
increases it on the completed 170 seed. None of these partial-wave findings
establishes a shared improvement or justifies canonical admission. Finish the
unchanged collection before deciding the next experiment. No optimizer update,
holdout fitting or canonical evaluation occurred.

Process inspection returned Normal priority (class 32) for the numerical
processes, despite the intended low-priority launch recorded above. Their
single-CPU affinities are confirmed; no process priority or OS setting was
changed during this check. Do not describe low priority as verified.

## 23:40 KST heartbeat: 25 branches authenticated

The first incident/8703 and 190/8705 slots also exited with code 0. The queue
advanced to incident/8503 (numerical PID 9348, start 22:43:48, mask 8) and
190/8505 (PID 6004, start 22:53:13, mask 1). Together with 155/8701 (10208,
mask 4) and 170/8502 (7828, mask 2), four numerical workers remain active.
Their command/start/parent identities match queue events; coordinator 16744
is unchanged. No duplicate work, active STOP, stderr error or source/spec
change was found. C: has approximately 4.64 GB free (decimal).

`results/sdmpc_rl_p10_20261001/heartbeat_2340.json` authenticates 25 branches
and 1500 transitions. All eight completed carry controls reproduce exactly.
The 14 prior branch and experience hashes are unchanged, so their successful
actor replay checks are reused; all newly completed candidate experience was
replayed with a fresh actor. Provenance, sequences, actions, rewards, terminal
handling and TTT accounting pass for every completed branch.

New candidate deltas (% versus paired carry):

| Scenario / seed | B1 both, end 25 | Long both, end 30 | Retain NUF, end 30 |
|---|---:|---:|---:|
| 155 / 8701 | -0.1106 | -0.8742 | pending |
| 170 / 8502 | -3.1529 | -7.4762 | pending |
| incident / 8503 | -0.5621 | pending | pending |
| incident / 8703 | previously verified | previously verified | -3.2271 |
| 190 / 8705 | previously verified | previously verified | -0.9508 |

The 170 long-window result changes from +0.7656% on seed 8702 to -7.4762%
on seed 8502. This remains seed-sensitive evidence, below the required
five-seed coverage and insufficient for admission. NUF retention improves
incident/8703 versus the other two arms but gives less improvement on
190/8705. Preserve the full planned collection; do not select a per-scenario
mixture or infer success from these partial results. No optimizer update,
canonical evaluation, process-priority change or OS-setting change occurred.

## 2026-10-02 00:40 KST heartbeat: 35 branches authenticated

Seven slots have exited successfully. Current numerical workers are 190/8505
(PID 6004, mask 1), skew15/8504 (9940, mask 4, start 10-01 23:48:14),
155/8501 (21456, mask 2, start 10-02 00:05:20) and 170/8802
(992, mask 8, start 10-02 00:39:10). Command/start/parent identities match
the original queue. Coordinator 16744 remains active; no extra job was launched.
No active STOP, stderr error, source/spec change or full-wave completion exists.
C: has approximately 4.59 GB free (decimal).

`results/sdmpc_rl_p10_20261001/heartbeat_20261002_0040.json` authenticates
35 branches and 2100 transitions, including ten exact carry reproductions.
The previous 25 branch/experience pairs are unchanged. All new candidate
experience passes fresh-actor replay; every completed branch passes provenance,
timeline, accounting, terminal, shape and sequence checks.

New completed candidate deltas (% versus paired carry):

- 155/8701 retain NUF: -0.8742%, equal TTT to long both.
- 170/8502 retain NUF: -6.2771%.
- incident/8503 long both and retain NUF: both -3.3830%.
- 190/8505 B1 both: -1.6603%; long both: +3.4492%.
- skew15/8504 B1 both: -3.1357%.
- 155/8501 B1 both: +2.9903%.

These completed training results continue to show seed sensitivity. No collapse
has appeared in the completed P10 branches, but the wave and seed coverage are
incomplete. Do not treat equal TTT alone as proof of equal trajectories, or use
partial logs as scores. Continue the immutable collection. Whole-wave analysis,
neural fitting and canonical admission remain pending; no user action is needed.

## 2026-10-02 01:41 KST heartbeat: 43 branches authenticated

Ten slots have completed with exit code 0. The four current numerical workers
are 170/8802 (PID 992, mask 8), incident/8803 (6068, mask 1, start 01:02:16),
190/8805 (2464, mask 2, start 01:22:31), and skew15/8704
(20732, mask 4, start 01:22:36). Their command/start/parent identities and
affinities match the queue; coordinator 16744 is unchanged. The last slot,
155/8801, remains queued. No active STOP, stderr error or source/spec change
was found. No duplicate dispatch or system/process setting change was made.

`results/sdmpc_rl_p10_20261001/heartbeat_20261002_0141.json` authenticates
43 branches, 2580 transitions and 12 exact carry reproductions. The prior
35 branch/experience pairs retain their hashes; new candidate records pass
fresh-actor replay. All completed records pass the existing provenance,
timeline, accounting, terminal and sequence checks. Whole-wave completion is
still pending. C: has approximately 4.58 GB free (decimal).

New completed training deltas (% versus paired carry):

- 190/8505 retain NUF: +2.5629%.
- skew15/8504 long both: -4.7722%; retain NUF: -2.7625%.
- 155/8501 long both and retain NUF: both +1.2376%.
- 170/8802 B1 both: -10.8272%.

Seed 8802 is the predeclared held-out 170 seed. Its completed score is recorded
for verification only and must not enter fitting or be used to revise the
learner after inspecting its errors. The first two seeds of every scenario
are now collected, but neural fitting remains deferred until full collection
authentication. No candidate has met admission; finish the existing queue and
preserve the split. No user action is needed for this routine check.

## 2026-10-02 02:20 KST: recovery after accidental shutdown

The user reported turning the computer off and requested resumption. Process
inspection found no Python process running. The old wave has no completion
file, no STOP, and 48 durable complete branches (2880 transitions), including
14 carry controls. Source/spec pins and all selected cached carry/k16 hashes
are unchanged. The frozen physical snapshot's 151 files verify against the
same manifest hash. No completed output was lost or invalid in this audit.

All 48 branches were reauthenticated for provenance, exact carry reproduction,
timeline/accounting, experience hashes/shapes/sequences/terminal handling and
fresh-actor action/memory replay. Their JSON and experience files were copied
byte-for-byte into a new collection directory. Original wave1 data and partial
logs remain intact. No orphan experience files were found.

Active output: `results/sdmpc_rl_p10_20261001/wave2_recovery`.
New coordinator source: `work/sdmpc_rl_p10_recovery_20261002/recover_queue.py`.
The worker, policies, specs, physical solver and cached checkpoint code are
unchanged. The new plan pins all original sources plus the recovery source,
the original plan hash, cache hashes and every imported file hash. It retains
all 15 planned profiles and the original fit/holdout split. Imported branches
are reused data, not additional independent samples or repeated measurements.

Only the following 12 missing branches are dispatched:

| Scenario / seed | Remaining branches |
|---|---|
| 170 / 8802 | retain NUF |
| incident / 8803 | long both, retain NUF |
| 190 / 8805 | B1 both, long both, retain NUF |
| skew15 / 8704 | long both, retain NUF |
| 155 / 8801 | carry, B1 both, long both, retain NUF |

Interrupted branches had no durable intermediate simulator/experience checkpoint.
They restart before decision 16 using the authenticated k16 checkpoint and run
through true terminal decision 75; complete branches are skipped by the original
worker. The cached carry prefix still contributes to whole-episode TTT. No
partial score is accepted and no completed branch is recomputed.

Recovery checks passed: existing output refusal, overlapping coordinator lock
refusal, source STOP detection, incomplete branch rejection, missing experience
rejection, copied hash equality and exclusion of every complete branch from the
pending schedule. Evidence: `wave2_recovery/recovery_checks.json` and `plan.json`.

Launch: 02:20:20 KST, launcher PID 6176, coordinator PID 6012. Initial numerical
workers, all started 02:20:21, are 170/8802 PID 16056 (mask 1), incident/8803
1980 (mask 2), 190/8805 7732 (mask 4), and skew15/8704 15600 (mask 8).
Command/start/parent identity and actual CPU affinity were checked after launch.
All four loaded their cached carries and are using CPU; startup stderr is empty.
155/8801 remains queued until a worker is free. Numerical thread limits are set
to 1 before process import. No OS settings, automatic commit or push changed.

Both old and new queue locks are held during recovery. STOP in the repository,
named result roots, either wave, or any planned slot prevents further dispatch
and propagates to the active recovery wave. This recovery launch is one-shot;
if it is interrupted again, authenticate its durable results and prepare another
new output rather than overwriting its launch receipt or running it twice.

Continue monitoring `wave2_recovery/run_started.json`, `events.jsonl`, `logs/`,
and `queue.stderr.log`; do not read active worker `status.json`. When its
`completion.json` reports completed, run:

```powershell
& C:/Users/alsrj/Desktop/RL/.venv-torch/Scripts/python.exe -B work/sdmpc_rl_p10_20261001/p10_analysis.py --wave results/sdmpc_rl_p10_20261001/wave2_recovery
```

The old wave1 completion remains absent by design. Authenticate the assembled
60-branch collection once, then continue the bounded analysis/shared-model
learning workflow under the unchanged admission gates. P10 remains a training
data collection and structured-policy experiment; no optimizer update has run.

## 2026-10-02 05:59 KST: full collection certified

Recovery completed at 03:54:21 KST, with all five recovery workers exiting 0.
No P10 process remains. The full analyzer passed all 60 branches, 3600
transitions, 15 exact carry controls, fresh-actor replay, training provenance
and true-terminal accounting. All recovered files and all P9/P10/recovery
source hashes remain unchanged; no STOP exists. Evidence is in
`wave2_recovery/analysis.json` and `analysis.csv`.

All three candidate means miss the fixed admission criteria; none has a
completed ≥+10% collapse. See the complete mean table and next bounded
experiment in `rl_continuation_p11_20261002.md`. These 60 are unique branches:
the 48 copied during recovery are not extra samples.

Additional fitting-profile trace checks explain identical TTT for the two
long-window arms on 155/8701, 155/8501 and incident/8503. Requested actions
and one executed budget differ, but achieved flows, cumulative TTT, urban
queues and selection source match at every decision. Most later differing
requests select the same `requested_budget_infeasible` reference fallback.
Increasing the NUF request therefore did not change the observed traffic
outcome on these profiles. This does not establish that all simulator states
are identical. Evidence: P11's `p10_equal_trajectory_details.json`.
