# Active goal: balanced shared-policy improvement

## Latest continuation pointer

The terminal/projection/quota audits below are already complete; do not repeat
them. The subsequent balanced local-budget collection is also COMPLETE with750
preserved transitions, but its predeclared state-coverage screenFAILED170.
The old collection is stopped; no new learner has been admitted. Read
`rl_budget_local_collection_results_20260930.md` and
`.superpowers/sdd/rl_budget_local_collection_plan_20260930/progress.md` for the
completed fallback-rebase/physical-response diagnosis. The current next task is
`rl_budget_nuf_retention_probe_20260930.md` and its own SDD progress ledger:
NUF-retention collection COMPLETE/PASS375transitions, two pairedtrainingTTTs
improved and threeworsened; nolearnedpolicy yet. Allworkers exited. Read
rl_budget_nuf_retention_results_20260930.md. The NEW next stage is
rl_budget_return_initialized_policy_20260930.md and its own SDDprogressledger:
Task1actualtrainingCOMPLETE (1000Phi/250critic/10actor,fitgatePASS); modelhash
7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904.
Read rl_budget_return_init_results_20260930.md and
rl_budget_return_wave_results_20260930.md; Task2actualfive-profilewave COMPLETE/
PASS375transitions, allourworkersdrained. NUFcapall375; nocanonicalimprovement
claim. Task3actualCOMPLETE250critic-onlyon-policyMCupdates, unchangedPhi/actor,
source/outputfrozen. See rl_budget_return_mc_results_20260930.md. Task4next:
canonical5scenarioevaluationofsameactor innewreturn_canonical_v1 afteradapter
tests/review. Noadditionalactorgradientstepyet. See currentSDDledger. Do not repeat completed
collection/audits/training. Keep the goal
ACTIVE; completed collection is not shared-policy improvement.

## User authorization and current scope

On 2026-09-30 the user explicitly requested setting a goal and continuing to
work until improvement. This authorizes the diagnosis, scoped implementation,
testing/review, bounded balanced collection, shared-model training, full-run
evaluation and reproduction loop below. A completed pilot is not completion of
this goal. Do not stop at an unchanged failed result when a useful next diagnostic
is available. Do not promise that a successful policy is guaranteed.

This goal follows `rl_budget_recovery_results_20260930.md`. It concerns the
current previous-executed-budget shared TD3 work, not the paused legacy DDQN
170-incident/P-Stack 5% experiments. The old automation's identifier is reused,
but its old prompt and objective are replaced. Do not resume old runners,
automations, STOP-marked output directories, or unrelated projects.

Workspace: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.
Goal thread: `01a01d6b-4af5-7ea1-b55c-e91b1ec51dcc`.
Heartbeat automation id: `ddqn`; name: `다섯 시나리오 공유 RL 개선 계속`.
Frequency: every three hours in this thread. It is a checkpoint and recovery
mechanism, not permission to launch an overlapping experiment every three hours.
The app goal is active with no user-specified token budget. User STOP/pause or
later changes take priority. Local scheduled work requires the computer and app
to remain running; do not change power settings automatically.

## Success contract

Use one frozen shared state-conditioned policy/value model for all five
scenarios. Twin critics within that shared learner are allowed; independently
specialized scenario checkpoints or post-hoc mixtures are not this objective.
Keep each scenario at 20% of every training minibatch. Equal sampling is not a
claim of equal gradient magnitude or out-of-sample generalization.

| Scenario | Matched carry-center full-run TTT |
| --- | ---: |
| sweet_155_w | 3103.0110715680044 |
| sweet_170_w | 3935.903236508048 |
| sweet_170_incident_w | 5546.224352256691 |
| sweet_170_skew15_w | 4250.876599300032 |
| sweet_190_w | 6604.2970168093225 |

The first success target is improvement in **every** row, not just the mean.
For each row, center TTT minus candidate TTT must exceed
`max(1e-6, 1e-8 * center_TTT)` vehicle-hours, so reconciliation-level numerical
differences do not pass. Report actual percentages; do not call a small decrease
a 5% achievement or silently reinstate the old 5% P-Stack target. The comparator
is the zero-action carry center with its existing physical recovery behavior,
not P-Stack and not an unconditionally fixed numerical budget.

All acceptance runs must:

- Start from the original canonical reset with five warmup intervals, execute
  all 75 control intervals and reach 14400 seconds. Do not score a prefix.
- Preserve scenario profile, demand, incident, follower constraints, physical
  source/runtime contract and TTT accounting against the authenticated center.
- Use the same model hash across scenarios, no exploration or learning during
  evaluation, no intervention gate, forced first action or performance guard.
- Retain and disclose the existing physical feasibility guard and initial/recovery
  PFO. Include all reference, actor, lower-solver, guard and reset costs in their
  appropriate measured timing fields. Do not advertise millisecond inference as
  end-to-end control latency.
- Keep evaluation observations and outcomes outside training replay. Training
  demand perturbations and exploratory trajectories are not canonical scores.
- Validate completion/source/settings/runtime/model identity, TTT timeline,
  terminal inventory and complete action/control records before comparing.

When all five pass, freeze and hash the checkpoint, configuration and source.
Reproduce all five full evaluations in new output directories. Require both
complete five-scenario comparisons to pass using the same frozen candidate.
Only then complete the goal, pause the heartbeat and stop launching experiments.
Repeat deterministic runs verify reproducibility, not generalization. A claim
about P-Stack or nonlinear-price causation requires separate matched evidence.

## Preserved starting evidence

- Physical snapshot: `artifacts/sdmpc_budget_baseline_20260929/source`, 151 files;
  manifest SHA-256
  `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.
- Balanced pilot: `results/sdmpc_rl_multi_20260929/pilot_v1`.
  Base checkpoint `train_round1/model_final.pt`, SHA-256
  `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`.
  Replay has 750 sequential transitions, 150 and two true terminals per scenario.
- Recovery evidence: `results/sdmpc_rl_recovery_20260929`.
  Selected actor-only export `actor_fit_v2/continue.pt`, SHA-256
  `82bae10a4c27b2723797be71ae41a1076aad72c8a5f86aec25cbd352fbc3f0ff`.
  This is not a jointly trained/resumable replacement TD3 checkpoint.
- Recovery full-run TTTs in scenario order:
  3283.155247293268, 4403.029229973646, 5808.4234447405615,
  4577.708380613563, 7255.832502258501. All are worse than the centers.
  `evaluation_v2/analysis.json` authenticated all five completed runs.
- Output-head fitting corrected the old actor/critic directional mismatch and
  removed sustained all-ramp closure. The new actor still saturates both actions;
  NUF requests clip to 6000 throughout and NP increases approximately 50 per step.
- More critic updates on the same replay did not solve calibration: terminal
  min-Q MAE rose from 5.7018 to 41.5798 with tau=0.005, and 147.4217 with tau=1.
  Neither critic-only diagnostic was exported as a policy.
- Previous implementation/evaluation reviewed; 523 regression tests plus eight
  analyzer tests passed. This is not automatic admission of new code.

Read those results, not the old legacy audit, as the starting instruction for
this goal. Existing completed code/configuration/artifacts remain immutable.
Use new versioned code/output directories for changes.

## Next bounded work

1. Verify the terminal/observation contract and run a small exact-terminal-target
   fit diagnostic. Check true terminal flags, reward scaling, remaining horizon,
   normalization and previous executed budget. Separate training fit from a
   diagnostic holdout; only two terminal states per scenario are presently
   available. Do not call that sample sufficient for generalization.
2. Audit nominal-action Q predictions for state-matched actions that collapse to
   the same projected executable budget. Test the positive-NUF cap case before
   choosing a representation correction. A numerically different request is not
   necessarily a different follower response.
3. Based on those tests, predeclare one falsifiable next hypothesis. Use a small
   synchronized training-only five-scenario wave if new state coverage is needed.
   Preserve balanced sampling and one central model. A frozen-policy return is
   a value for that continuation, not an unconditional optimal-action label.
4. Test/review scoped changes before training/evaluation. Compare terminal and
   time-resolved value calibration, saturation, projected budget diversity and
   executed controls. Do not select/deploy solely from predicted Q or loss.
5. Evaluate the admitted frozen candidate on all five scenarios. Reconcile
   per-scenario TTT, inventory, interval losses, reached states and full runtime.
   On failure, record what the experiment falsified and repeat with a revised
   diagnosis. Do not repeat an unchanged configuration or collect 24 hours by
   default. On a passing candidate, follow the reproduction gate above.

## Execution and stopping rules

Before any new run, inspect current status/completion/logs, process PID plus
creation time and command, locks, and relevant STOP files. A stale process.json
is not proof that a job is active. Do not create duplicates, rewrite an active
experiment, or rerun completed collection. Reproduction of a qualifying candidate
is an explicit exception, always in a separate directory.

Use at most eight numerical workers in total for this goal; account for internal
Torch/BLAS threads and subprocesses. Preserve unrelated processes. Launch hidden
background windows only when needed, with process identity and logs recorded.
Keep checkpoints, completed data, configuration and source hashes. Resume a failed
authorized run only after identifying its last durable progress and cause.

The goal-level output root is `results/sdmpc_rl_balanced_goal_20260930`.
Honor `RL/STOP`, `results/sdmpc_rl_balanced_goal_20260930/STOP`, and the active
experiment's STOP files. Do not delete/ignore STOP automatically. If a user stop
is present, stop new dispatches, preserve progress and pause the continuation.
Keep old experiment STOP markers intact; they are not implicit permission to
reopen old experiments. The latest explicit user instruction always governs.

Stay quiet on unchanged/non-actionable scheduled checks. Notify in Korean for a
meaningful verified result, actionable failure, required user input, or confirmed
goal achievement. Genuine external blockers must be reported, not covered with
repeated fake progress or relaxed acceptance conditions. No automatic commits,
pushes, dependency installation or operating-system changes are authorized by
this goal alone.

## Continuation ledger

### 2026-09-30: goal registered and heartbeat replaced

- Created the explicit app goal; status active, no token budget requested.
- Updated existing heartbeat id `ddqn` to the new balanced-policy prompt, name
  and ACTIVE status, keeping its three-hour cadence and this same target thread.
  This replaced its old paused legacy objective rather than adding a duplicate.
- Preflight found no related Python worker and no STOP under the current multi
  pilot/recovery roots or repository root. No paused unrelated automation changed.
- No new simulation or training has been launched by this setup step. Next action
  is the terminal/observation diagnostic above, not rerunning completed recovery.
- Append each new hypothesis, version, command, process identity, source/model
  hash, test/review evidence, outcome and next action here or link a versioned
  experiment report from here. Keep active process details current.

### 2026-09-30: terminal and projection audits completed

- See `rl_budget_value_audit_results_20260930.md` for commands, process IDs,
  source/completion hashes, tests/review and measured results. Both actual jobs
  exited successfully; neither produced a deployable policy or a traffic score.
- The 750 training transitions match collection accounting and provenance.
  Terminal-only fitting reduces train error but leaves round/holdout limitations.
  Distinct nominal actions with identical capped budgets receive different Q.
- Next bounded hypothesis is terminal underexposure during TD. Implement the
  predeclared two-seed, uniform-versus-terminal-quota ablation described in
  `work/sdmpc_rl_value_audit_20260930/terminal_quota_brief.md`, test/review, then
  run once in a new directory under the goal root. Do not rerun completed audits.
- Current goal turn is progress, not blocked. No five-scenario TTT improvement
  or success has been claimed. Continue from the diagnostic outcome.

### 2026-09-30: terminal-quota hypothesis supported, collection revision next

- Terminal-quota ablation completed once, PID31648 exit0,301.1386s. Both seeds
  passed the predeclared diagnostic criterion: terminal min-Q MAE41.58->4.50
  and41.31->2.89 versus matched uniform TD. No policy or TTT gain is implied.
  Full provenance/curves/limitations are in the value audit results document.
- All three value_audit_v1 diagnostics are completed and preserved. In the quota
  output completion.json overrides the stale `verifying` status. Do not rerun.
- Implement and review `rl_budget_local_collection_plan_20260930.md` next,
  then run its small paired training-only five-scenario wave. It addresses
  incremental-noise budget drift and missing near-center sequential states.
  No local_budget_v1 collector has started yet. Existing saturated actors are
  not admitted for another collection/evaluation wave.
- Goal remains ACTIVE, not achieved or blocked. No full-run improvement claimed.
  Three-hour heartbeat remains active. No worker should remain from this stage.
- Final phase integration review passed; combined158testsPASS35.44s. All agents
  and numerical sessions closed. Heartbeat readback confirms ACTIVE/every3hours
  on the correct thread. Next continuation entry point is the local collection
  plan. This is a completed diagnostic phase, not completion of the active goal.

### 2026-09-30: local collection implementation in review

- New files under `work/sdmpc_rl_local_20260930` implement pure budget-level
  mean-reverting exploration, a resumable training-only collector, provenance
  reconciliation and a two-stage five-worker runner. Old sources are untouched.
- Progress/review recovery map:
  `.superpowers/sdd/rl_budget_local_collection_plan_20260930/progress.md`.
  Read that ledger before repeating implementation or dispatching any run.
- Exploration121 and collector22 synthetic tests pass; combined143PASS8.47s.
  Read-only physical/runtime/schema identity preflight passed. No production
  collection has run. Independent task review is pending; do not launch before
  the ledger records admission. First real carry155 prefix will be resumed in
  place into the full wave, not discarded or repeated.

### 2026-09-30: actual prefix passed, Windows parallel startup repair

- Local collector fixes passed200synthetic tests and independent scoped review.
  Actual carry155 saved interval1 then restored and continued interval2; both
  processes exited0. No interval was recollected. Locked-worker time66.7563663s.
- First parallel coordinator (launcher37600, actual27044,04:31:19KST) failed
  before any of its five children booted the environment: Windows venv Popen
  reports the redirector PID, not the child interpreter's PID. A separate tiny
  nonnumerical subprocess reproduced this exact mismatch. All workers exited.
- Preserve v1 code and results unchanged. A new bounded repair task implements
  separate launcher/worker identity and explicit migration of the two-interval
  prefix into v2. Do not restart the failed v1 wave or discard its checkpoint.
  Current recovery map is the same local-collection SDD progress.md, with
  windows-recovery-brief.md and windows-recovery-report.md. Source/root suffixes
  are sdmpc_rl_local_20260930_v2 and local_budget_v2, not yet admitted or launched.
- No shared learned-policy evaluation or improvement has occurred in this
  collection stage. Goal remains active; this is a diagnosed local runtime
  defect, not an external blocker. Three-hour heartbeat authorization unchanged.

### 2026-09-30: v2 verified and five-scenario collection running

- Full execution record: `rl_budget_local_collection_status_20260930.md`.
  V2 passed236tests and independent review; actual checkpoint migration/restore
  and repeat receipt verification passed without recollecting the two intervals.
  All54v1source/result files remain byte-exact. Both source versions now frozen.
- Active output is `local_budget_v2`, not failed v1. Hidden coordinator launched
  at05:11:46KST (launcher3100,actual46348); allfivecarryworkers advanced without
  stderr. Carry155 startsat3/75. Five local exploration episodes follow.
- Observed total numerical workers6 (our5 plus a verified unrelated serial
  single-core simulation), within8. Other project/processes were left untouched.
- Next is completed paired collection analysis and learner specification, not
  another migration, duplicate runner or automatic actor export. Goal remains
  ACTIVE and unachieved. No new shared-policy TTT gain is claimed.

### 2026-09-30: terminal export recovery, preserve250intervals

- By05:28KST the v2 coordinator and allworkers exited after carry155 reached75
  steps but strict JSON export rejected four positive-inf candidate stationarity
  diagnostics. Other workers saved prefixes before draining. Total250real
  intervals remain in checkpoints; no loss/recollection is authorized or needed.
- Read-only checkpoint validation passed. Inf is in a rejected candidate at
  control_step14; the physical fallback executed feasibly. No reward/state/action
  nonfinite value was found. Do not misreport this as solver convergence.
- Current task is an output-only companion, outside immutable v1/v2 sources:
  `work/sdmpc_rl_export_recovery_20260930`, requirements/recovery map in the same
  SDD ledger's export-recovery-brief.md/progress.md. It will preserve raw Inf,
  use explicit tagged JSON diagnostics, reject unrelated nonfinite data, and
  resume the same cohort after review. Not yet admitted/executed; do not launch
  a duplicate runner, remigrate or repeat the completed155physical trajectory.
- Goal remains active. Latest turn is evidence-based diagnosis/progress, not a
  genuine external blocker or a new policy performance result.
