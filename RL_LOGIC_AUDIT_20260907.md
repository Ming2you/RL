# Sequential DDQN audit and corrected pilot

## Current status: USER PAUSED (2026-09-09 23:11 KST)

The user explicitly asked to pause this RL work and turn off its scheduled
runs to work on something else. This instruction OVERRIDES every earlier
continuing authorization below. Do not launch, train, collect, evaluate or
automatically resume this project until the user explicitly requests resume.
An already queued heartbeat is not a new user authorization to restart.

Automation `ddqn` (the only configured Codex automation found) was changed
from ACTIVE to PAUSED with the automation API; its saved status was verified.
No matching RL Windows scheduled task or remaining experiment/worker process
was found after shutdown. No unrelated scheduled task was changed.

Cooperative STOP files were added at:
- `results/response_dqn_170_incident/sequential_multistep_v1/STOP`
- `results/response_dqn_170_incident/sequential_td_audit_v1/STOP`

The second path is a shared predecessor STOP honored by the newer runners.
Keep BOTH until explicit resume. The active runner observed STOP, checkpointed
its in-flight intervals and exited normally: wrapper45392/child14332, exit0,
finished2026-09-09T23:11:52.0172896+09:00; status`paused`.
No force-kill, configuration change, checkpoint deletion or result overwrite.

Both policies completed training (3members x24kupdates each) and paused their
full evaluations after48of75transitions, last control_step47, terminalfalse.
These are incomplete trajectories, NOT accepted full-run TTT results:
- `one_step`: cumulativeTTT5186.874183055881; measured wall3011.180915seconds.
- `greedy_five_step`: cumulativeTTT5186.788407053982; wall3032.165600seconds.

Both `evaluation/checkpoint.pkl` files remain available. SHA256:
- one_step: `baaea49b73810ee78734d5ad5e71f57887bacd659cfec79809ff7cc3b0365ba2`
- greedy_five_step: `b0b5bee6e38d3fd72a0658e744bee5f451db169d7015e55c2e18921c8f44d56a`

All6trained model files still match the frozen `model_hashes.json` record.
On explicit resume, inspect current state first, clear only user-authorized
STOPs, reuse the completed models and resume remaining evaluation intervals
through the existing locked launcher and unchanged `work/response_multistep_v1.json`.
Do not recollect or retrain completed work. Re-enable scheduling only as
requested by the user. The5% goal is unmet; it is paused, not completed.

## 2026-09-10 GitHub handoff export (not a resume)

The user requested a GitHub backup of accumulated research data, project
memory and next steps. The USER PAUSED instruction above remains in force.
No STOP was removed, no automation was enabled, and no RL workload restarted.
`RL_HANDOFF_20260910.md` is the concise current entry point; historical IQL
handoffs now point there so incompatible scope/metric claims are not reused.

The complete local `data/results/models/checkpoints` inventory was packaged
as3039byte-preserved files:2,206,836,922original bytes,734,164,552ZIP bytes,
18parts of at most40MiB in `artifacts/research_snapshot_20260910`.
ZIP SHA256 `e2598c0580c6a07a8bff4e770c2855dc219b6cb3a7264ff37ad34ce236c67970`.
All3039members were independently read from the ZIP and compared with the
local originals; part, ZIP and member hashes passed. No raw files were deleted.
The three new snapshot-tool tests passed. No training/simulation was required.
Personal app memory/settings/credentials, the venv and paper draft are excluded.
Project decisions and evidence are summarized in the handoff and this audit.

The raw existing mixed line endings in source/configuration and previously
tracked data are preserved with `.gitattributes`; this intentionally avoids
Git autocrlf invalidating the177runtime source pins. The code behavior and
active experiment configuration were not changed by this packaging step.
The raw data roots remain local and ignored for future untracked-file additions;
their complete contents are stored in the split archive, not omitted.

Remote and local main were equal at preflight (origin `Ming2you/RL`, commit
071ea03). Commit/push verification is recorded by Git history and the user's
close-out message; this precommit entry alone does not claim upload success.

Final staging checks confirmed all177Git-index runtime blobs exactly match
the existing experiment SHA256pins and local bytes. Existing tracked sources
were explicitly restaged without line-ending normalization to achieve this;
the resulting historical CRLF-only diffs are intentional byte preservation.
All267Python source modules parsed successfully. The18archive parts are each
at most40MiB. Snapshot restore tests passed3/3; the earlier235RL regression
result is historical and was not rerun or represented as a new training run.

## Decision

Continue simulator-assisted, response-aware DDQN, using real sequential
transitions. Do not restart the proposed r31 fixed-tail label collection as the
main policy-learning loop. Fixed-tail labels remain useful diagnostic evidence
about a specified continuation policy, not estimates of optimal recovery.

The follower MPCs still execute traffic controls. The current action catalog is
a small set of anchor-relative coordination operators, including linear,
quadratic and cross terms. This is an MPC-anchored response selector, not yet a
standalone RL leader with no native MPC computation.

## Findings

1. `response_ddqn_recovery._evaluate_first_action`, with `recovery_depth=0`,
   follows P-Stack after the first action. `recovery_state_to_replay` marks all
   labels terminal and copies the current observation into the successor field.
   This estimates a fixed-continuation truncated return/advantage. Later learned
   recovery actions cannot revise that target through a Bellman update.
2. Twenty rollout steps do not cover the remaining episode at step 18.
   Treating those labels as terminal is an intentional regression encoding, but
   cannot establish their ordering under full 14,400-second TTT.
3. The DDQN learner itself correctly uses online-network argmax and
   target-network evaluation. `--no-bootstrap` disables event-group resampling,
   not the Bellman successor term. All-terminal replay was what removed that
   term in recent label-only runs.
4. The replay merger previously checked shapes and catalog but allowed different
   reward meanings to mix. Interval negative TTT, terminal return-to-go and
   anchor-relative advantage do not share the same learning objective.
5. Epsilon exploration under the existing LCB policy was restricted to actions
   already satisfying behavior support. An unsupported nonlinear candidate
   could never gain support through that exploration path.
6. The r31 command used eight response-preview workers but only one tail-branch
   worker. Its expensive nineteen-step continuations were serial. Increasing
   preview workers did not parallelize those tails.
7. Numerical catalog IDs changed across experiments (old 103 = new 69). Stable
   keys and catalog fingerprints are necessary. Matching plant/controller state,
   scenario, experiment contract, and saved observation is necessary for reuse.
8. The previous best used a one-step intervention gate and runtime response
   previews. Its TTT is evidence for that specific controller, not proof that an
   unrestricted preview-free DDQN achieves the same result or speed.

## Implemented corrections

- Raw collection declares `interval_negative_ttt` and `environment_terminal`.
- Collection can continue a saved environment without resetting; absolute
  control-step indices, successor observations, masks, and response features must
  agree across restart boundaries.
- Explicit simulator exploration may sample unsupported but executable actions.
  The new greedy policy treats P-Stack/action 0 as an ordinary competing Q value,
  with no LCB advantage guard. Exploitation still requires behavior support.
- Replay merge rejects incompatible reward/done meanings, MC discount factors,
  recovery horizons/depths, experiment contracts, and episode horizons. Missing
  legacy metadata is accepted as raw only without transformed-label clues.
- `--require-sequential-td` rejects fixed labels and requires one-step rewards,
  nonterminal rows, actual environment termination, and gamma=1 for this finite
  total-TTT objective. `--no-group-bootstrap` clarifies the old option's meaning.
- The new runner stores environment, RNG state, accumulated replay, payload
  identity, wall time, and progress after each completed transition. Resume
  uses the saved environment; trace segments preserve interrupted attempts.

## Pilot protocol

Configuration: `work/sequential_ddqn_170_incident_v1.json`.
Runner: `python -B -m rl_leader.run_sequential_response_ddqn --config
work/sequential_ddqn_170_incident_v1.json` (one command line).

1. Validate the snapshot against the saved best trace and environment contract.
2. Construct a ten-action fixed catalog by stable keys: P-Stack identity, the
   previous best residual, four linear operators, one quadratic, two cross, and
   one mixed operator. This is a small direction-checking experiment, not
   exhaustive coverage of nonlinear potentials or control owners.
3. Round 0 runs two trajectories from the pre-action step-18 snapshot to the
   actual step-74 terminal. Their first actions cover best residual and anchor;
   every subsequent state permits fresh action selection with epsilon=0.2.
4. Use two actors with four preview workers per actor. Native numeric libraries
   use one thread. Release preview pools between collection/training/evaluation.
5. Train three DDQN members on the accumulated raw replay (gamma=1, 6,000
   gradient steps, 128x128 hidden units). Successor actions remain in the target.
   Every member uses the complete tiny pilot dataset; seed spread is not a
   calibrated confidence interval.
6. Evaluate the frozen policy from step 18 through the real terminal with
   epsilon=0, no first-action forcing and no LCB guard. The earlier prefix is
   fixed, so report this as continuation-policy evaluation.
7. Round 1 collects with the new policy plus exploration. One actor starts at the
   beginning of the control episode to cover pre-18 states; the other continues
   from step 18. Retrain on cumulative transitions and re-evaluate.
8. Run the final policy from the normal environment reset to 14,400 seconds with
   no intervention-step gate and epsilon=0. Report this separately from suffix
   evaluation and exploration episodes. Include response-preview computation.

Training is batched between simulator episodes. DDQN is off-policy even though
new data includes states reached by the latest policy. This is not strict
single-dataset offline RL, nor a gradient update after every simulator step.

## Evidence and acceptance criteria

Reference P-Stack total TTT: 5730.792964723197.
Previously verified best gated total TTT: 5682.100819335536 (0.849658% reduction).
Five-percent target: 5444.2533164870365. The target is not achieved yet.

- The toy delayed-recovery test must learn root action value near -2 although
  its P-Stack continuation would return -101; this exercises the actual learner.
- The real two-step resume smoke must reproduce reference states 19 and 20
  after best-action-at-18 then anchor-at-19, leaving both rows nonterminal.
- Every complete collected trajectory must have exactly one terminal row, at
  the end, with contiguous successor states, masks and response features.
- Prefix TTT minus the sum of interval rewards must reconcile to simulator TTT.
- Each experiment reports executed action counts, terminal inventory, scope and
  measured wall time. No improvement claim from training loss or partial TTT.
- Check direction after this pilot. Extend data only if action coverage/value
  calibration and closed-loop outcomes justify it. A two-round pilot cannot
  establish generalization or a statistically reliable 5% gain.

Validation completed on 2026-09-07: 75 relevant unit/regression tests passed.
The real simulator resume smoke passed both state-hash checks in about 230
seconds of measured episode work. Its two rows have `done=[0,0]`; this is
verification data, not a full-run performance result.

## Remaining risks and alternatives

Global action support is not state-local support. A mask verifies executable
responses, not value accuracy. Fixed finite observations may still alias some
controller or network history. Early-state and held-out demand/incident coverage
must be tested before generalization claims.

For gamma=1, the sum of interval negative TTT equals the negative finite-horizon
objective; the observation contains time. End-of-episode inventory is reported
as a diagnostic, not silently added to one controller's reward. If clearing
remaining queues is required, extend the evaluation horizon for every controller.

If sequential DDQN has useful responses but poor Q calibration, consider
recurrent Q inputs, limited n-step targets with correct truncation, or a learned
response model. For a future continuous coordination policy, SAC/TD3 is a
separate research comparison requiring new data and feasibility handling; the
follower's discrete response map does not make continuous prices automatically
easy to optimize. Runtime preview removal needs a separately trained input
contract or response predictor, not zeroing preview features of this model.

## Operation

Output: `results/response_dqn_170_incident/sequential_td_audit_v1/`.
`status.json` records the stage. Each actor/evaluation has `progress.json`,
`checkpoint.pkl`, `replay.npz`, trace files, and a final `summary.json`.
Creating `STOP` in the output directory pauses at the next saved transition;
removing it and rerunning the same command resumes the same configuration.
Training interruption preserves any written members in their attempt directory.
`work/start_sequential_response_ddqn.ps1` runs the pipeline with persistent logs,
an exclusive launcher lock and `process.json` containing PID and exit status.
There is no repeated 24-hour collection quota in this pilot.

Launch verified on 2026-09-07 at 15:44 KST: wrapper PID 14496, Python launcher
PID 47448 (read `process.json` for the current run, not these historical PIDs).
Both round-0 actors persisted their first control-step-18 transition with
`done=0`, ten distinct valid response groups and the expected best/anchor
actions. Collection is in progress; no new full-run improvement result exists
at that checkpoint. A thread heartbeat checks meaningful changes hourly,
including completion/error evidence and the next small experiment decision.

### Round-0 collection and shutdown recovery (16:54 KST)

Both exploratory suffix episodes reached control step 74 and saved 57 rows
each. The 114-row merged replay passes the interval-TTT and true-terminal
contracts: 112 nonterminal rows, two terminal rows, contiguous successors, and
matching successor masks/features. Combined action counts are
`[87, 4, 5, 2, 3, 4, 2, 2, 4, 1]` in pilot catalog order.

| Episode | Total TTT including fixed prefix | Reduction vs P-Stack | Terminal inventory | Episode wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Actor 00, first action best residual | 5689.930090 | 0.713041% | 431.751464 | 2699.678955 |
| Actor 01, first action anchor | 5795.420767 | -1.127729% | 430.718327 | 2698.133461 |

These are epsilon=0.2 collection trajectories, not learned-policy evaluations.
They do not improve on the prior gated best and do not meet the 5% goal. All
ten actions have at least one executed sample, but this is very sparse support,
not evidence of adequate state-local coverage or reliable Q values.

The first launcher stalled after receiving both completed actor results.
Actor workers had cached nested response pools and depended on ordinary
`atexit` cleanup. Python multiprocessing joins a worker's active children
before those callbacks, so the outer pool waited for workers that themselves
waited for still-open preview workers. The actor entry point now closes and
joins preview pools in `finally`, on success, STOP, and errors. Main-process
evaluation also joins its preview workers before advancing phases.

Validation: 78 relevant tests pass, including a real nested-process regression
that exercises normal completion and interruption. Completed replay files were
validated before stopping only the identified stalled runner tree. Restarting
the same configuration reused both episodes and advanced to training without
new simulator steps. Restart: 16:54 KST, wrapper PID 5068, launcher PID 30576;
consult `process.json` for current process identity.

Round-0 training completed at 16:54:35 KST: all three 6,000-update model
checkpoints and the ensemble manifest exist. Reloaded checkpoints match the
pilot catalog; training losses and Q outputs on all 114 stored states are
finite. The manifest confirms gamma=1, 112 Bellman-bootstrapped rows and no
event-group resampling. Training loss is not a performance acceptance metric.
The runner advanced to epsilon=0 continuation evaluation with eight preview
workers; the learned-policy TTT remains pending. No new best result is claimed.

Preserved replay SHA-256 checksums:

- Actor 00: `862B57425FE515CDB5DC71E4EBB4A610BB65F79CC4C5B56950502B050F92DA2D`
- Actor 01: `FC80B5DB68235FF7A19C8C929D11929936AF59A89A10016A60F61228426E7CDF`

### Round-0 learned-policy evaluation and calibration audit (17:49 KST)

The completed epsilon=0 continuation policy produced **TTT 6078.606529**, a
**6.069205% deterioration** versus P-Stack, not an improvement. This includes
the fixed pre-step-18 prefix; only steps 18-74 used the learned policy. It is
not the final ungated-from-reset test. Terminal inventory was 435.543182 versus
430.850061 for P-Stack. Measured suffix evaluation wall time was 3255.905443 s
(54.27 min), including response previews. Comparing that suffix time directly
to P-Stack's full-run time would mix timing scopes.

Sources: `results/response_dqn_170_incident/sequential_td_audit_v1/round_00/`
`evaluation/summary.json`, `evaluation/replay.npz`, `evaluation/trace.jsonl`,
`training_replay.npz`, and `model/ensemble_manifest.json` plus its checkpoints.
The machine-readable diagnostic is `round_00/calibration_audit_v2.json` under
the same root. It includes source SHA-256 hashes, per-step errors and action
coverage. `calibration_audit_v1.json` is preserved as the earlier diagnostic.

Verified observations:

- All 57 transitions satisfy the real-terminal and successor contracts. The
  experiment hash, scenario and horizon match training and the P-Stack
  baseline. Interval rewards reconcile to total TTT including the prefix.
- Reloaded ensemble mean-Q argmax reproduces all 57 recorded selections and
  trace Q values. All ten response groups remained distinct and no validity
  gate failure was logged. A catalog-key mismatch or alias collapse does not
  explain this observed run.
- Training/evaluation action counts respectively: anchor 87/18, action 2
  (`linear_first_negative`) 5/7, action 5 (`linear_second_positive`) 4/11,
  action 9 (`combo_first_pos_quad_first_dec`) 1/21. Other candidates were not
  selected during evaluation. Action 9's only training execution was step 73,
  but the greedy policy first used it at step 27 and repeatedly from step 37.
- At step 18, ensemble mean Q implied remaining TTT 1900.436211, while the
  actual frozen-policy continuation incurred 3966.229860. The initial state
  exactly matches a training observation, so state novelty alone cannot
  explain the root value error. Subsequent reached states also diverge.
- Q implied less remaining TTT than this continuation incurred at all 57
  states. Mean underprediction was 738.873608 TTT units. This comparison is
  policy-specific, not proof that every possible future recovery is bad.
- Independently of that policy-specific comparison, 11 selected mean-Q values
  were positive and 22 exceeded their own immediate reward after scaling.
  With `r=-interval_TTT <= 0` and gamma=1, any true return obeys `Q <= r <= 0`.
  These bound violations are genuine value-estimation errors; a different
  future recovery cannot make them valid. The bound alone does not identify
  their cause. Mean absolute frozen-ensemble-policy one-step TD residual was
  0.464306 in scaled Q units despite small training loss.

Working hypothesis: globally admitting a candidate after one sampled execution
allows unsupported state-action Q extrapolation and optimistic successor
maximization. Step-73 support for action 9 does not establish its value at
step 27. The coverage and calibration evidence supports this hypothesis, but
a controlled follow-up is still needed before assigning causality to it or
to nonlinear pricing itself.

The planned round-1 collection automatically started after evaluation and
worker cleanup completed. One actor starts at reset, the other at step 18;
both use the new frozen policy with epsilon=0.2. Keep this configuration
unchanged to test whether reached-state/action coverage corrects the error.
Do not convert the bad observed continuation into an irrevocable terminal
label: subsequent Bellman updates must still allow a better recovery policy.
After its retraining and final full-run, repeat this audit, comparing value
bound violations, temporal support, and TTT. If coverage increases but values
remain invalid or TTT does not improve, investigate the bootstrap/value
estimator with a controlled change instead of simply increasing collection
hours or relying on training loss.

Reproduction (run from the RL repository; choose a new output filename):

```powershell
.venv-torch\Scripts\python.exe -B -m work.audit_sequential_td_evaluation `
  --evaluation-dir results/response_dqn_170_incident/sequential_td_audit_v1/round_00/evaluation `
  --model-dir results/response_dqn_170_incident/sequential_td_audit_v1/round_00/model `
  --baseline results/response_dqn_170_incident/pstack_rl_contract_v2/summary.json `
  --out results/response_dqn_170_incident/sequential_td_audit_v1/round_00/calibration_audit_recheck.json
```

This diagnostic reads frozen artifacts only; it does not run a simulator,
alter the active runner, or install realized returns as training labels.
The previous gated best remains 5682.100819 TTT (0.849658% reduction).
The 5% goal is not achieved.

### Round-1 evaluation and next controlled learning comparison (19:48 KST)

Round-1 collection and training completed. The two epsilon=0.2 collection
episodes produced TTT 6956.192212 (from reset, 75 rows) and 6124.669049
(step-18 suffix, 57 rows including the fixed prefix). These are exploratory
data, not policy acceptance tests. Cumulative training now contains 246 rows
and 242 nonterminal Bellman targets. Its three models each completed 6,000
updates. The subsequent epsilon=0 suffix evaluation produced **6287.663314
TTT, 9.717160% worse than P-Stack**, with terminal inventory 431.383857 and
2742.296205 seconds of suffix wall time.

The same diagnostic ran successfully on `round_01/evaluation`; the output is
`round_01/calibration_audit_v1.json` under the pilot output root. Source and
selection parity checks still pass. The initial Q-implied remaining TTT was
1511.453915 versus the realized 4175.286659. Positive selected-Q rows decreased
from 11 to zero and immediate-reward-bound violations from 22 to three, but
mean remaining-TTT underprediction increased from 738.873608 to 773.669634.
Thus fewer simple bound violations did not imply better control or calibration.

Action 9's training support increased from 1 to 29 and the new policy stopped
selecting it. Instead, action 3 (`linear_first_positive`) with only three
training executions was selected 19 times. Overall evaluation counts were
`[6, 0, 22, 19, 6, 0, 1, 0, 3, 0]` versus training counts
`[126, 4, 16, 3, 6, 46, 3, 6, 7, 29]`. This is consistent with optimism moving
to a different weakly observed action, not a problem confined to nonlinear
prices. It does not prove that a finite number of on-policy collection rounds
cannot work. It motivates testing the estimator, holding the data fixed.

The existing vanilla-DDQN full-run from reset is running and must finish;
do not interrupt it or overlap it with another training/simulation job. Its
outcome is still pending. Current source additions affect training only:
`--conservative-alpha` defaults to zero, preserving vanilla DDQN. A masked
discrete CQL(H)-style penalty can optionally augment the existing Huber
Bellman loss:

```text
loss = DDQN_Huber_loss
       + alpha * mean(logsumexp(Q(s, valid_responses)) - Q(s, observed_action))
```

Invalid/duplicate response groups are excluded from the regularizer. It does
not change the physical action catalog, reward, terminal logic, runtime
selection or fallback behavior. Bellman bootstrapping and exploration remain
available; observed bad continuations are not converted into terminal labels.
This is an experimental adaptation of CQL regularization, not a claim that
the paper's lower-bound guarantees automatically apply to this gamma=1
finite-horizon neural implementation.

Implementation validation: 82 focused tests passed, including mask/gradient
checks, candidate permutation and Q-shift invariance, alpha validation,
old-checkpoint loading, disabled-penalty behavior, and a real DDQN toy where
a later recovery still makes the earlier action beneficial. Re-running the
complete round-1 diagnostic after the training-only change reproduced the
previous diagnostic dictionary exactly, confirming recorded vanilla inference
is unchanged. No new real-data CQL training has been launched yet.

Next experiment specification: `work/response_cql_ablation_v1.json`. Once the
current runner completes, first reconcile and audit its full-run result. If
it meets the target, reproduce it instead of launching an unnecessary ablation.
Otherwise reuse the current alpha=0 model as the control and train alpha=0.1
and alpha=1.0 on the **same 246-row replay**, same seeds, network and update
budget. Do not add newly completed evaluation data to this comparison; that
would confound the regularization change. The pinned replay SHA-256 is
`E5FEEFE0D1487E6CB15801BB862C4436F7700391FBA9880FD05257836934AD5E`.
Use the existing training CLI with `--conservative-alpha`; the specification
records all common arguments and versioned output directories. Honor STOP
in both the old and new experiment directories and avoid overwriting partial
models. Train one ensemble at a time after previous compute has stopped.

Inspect finite values and action ranks on frozen states first. Such screening
does not measure a new policy's TTT. Evaluate viable distinct candidates from
reset with epsilon=0, no forced action or LCB guard, retaining the same anchor
and response-preview accounting. Use at most eight total preview workers,
recording per-run allocations; compare runtime only at matched allocations.
Run the completed-policy calibration audit only with the policy that actually
generated that evaluation. Decide further collection from this controlled
comparison instead of repeating the same unregularized estimator indefinitely.

Primary methodological sources: Kumar et al., [Conservative Q-Learning](https://arxiv.org/html/2006.04779v3)
(Equation 4, discrete logsumexp implementation), and Fujimoto et al.,
[Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900)
(batch-RL extrapolation error). These motivate the hypothesis; they do not
establish a traffic-control improvement in our experiment.

### Vanilla full-run completed; CQL comparison launched (20:59 KST)

The vanilla sequential-DDQN pilot exited normally at 20:41:57 KST. Its final
epsilon=0 run from reset completed all 75 control steps and produced
**TTT 7179.354752, 25.276812% worse than P-Stack**. Terminal inventory was
427.931659 and full-run wall time including response previews was
3708.449277 seconds (61.81 minutes). The model is not promoted. Its complete
audit is `sequential_td_audit_v1/ungated_full_run/calibration_audit_v1.json`
under `results/response_dqn_170_incident/`. All contract/selection checks pass;
mean remaining-TTT underprediction is 1331.582505, with three immediate-reward
bound violations and no positive selected mean-Q values. Improved simple
value bounds did not prevent severe closed-loop deterioration.

Both planned CQL variants were then trained on the pinned 246-row replay,
without adding evaluation trajectories. Each uses three members, 6,000
updates and the same seeds/recipe as the vanilla control. They differ only
in conservative alpha (0.1 or 1.0). Model checkpoints and manifests are in
`sequential_cql_audit_v1/alpha_010/model` and `alpha_100/model` beneath the
same results parent. `model_hashes.json` pins the six checkpoint hashes.

Frozen-state screening found finite but distinct policies. On the 75 saved
vanilla full-run states, alpha=0.1 selected anchor 68 times and alpha=1.0
selected it 67 times; both still selected other responses. Neither selected
the actions with only three training samples there. However, alpha=1.0 still
had 15 positive selected Q values on that state set. These are diagnostic
observations, not their own rollout results or evidence of a 5% improvement.
Both variants therefore proceed to the planned controlled closed-loop test;
larger regularization is not presumed better.

**Current supervised output:**
`results/response_dqn_170_incident/sequential_cql_audit_v1/`.
The old pilot is finished; do not restart its completed collection/training.
Use this new directory's `process.json`, `status.json`, `screening.json`,
`alpha_010/evaluation/progress.json` and `alpha_100/evaluation/progress.json`
for subsequent heartbeats. The coordinator writes `comparison.json` and a
calibration audit for each completed variant. `ablation_complete` without a
verified target is another decision point, not the end of the authorized loop.

`work/run_response_cql_ablation.py` checks the frozen data hash, source
experiment completion, training recipes and model hashes, then evaluates
the two variants from reset. Each actor uses four preview workers, eight in
total. It reuses exact episode checkpoints and closes nested pools explicitly.
STOP in either the old pilot directory or the new comparison directory
pauses at a saved transition. It does not alter reward/terminal semantics,
force a first action, use an LCB guard, or change follower constraints.
If a candidate meets 5%, status becomes `goal_candidate_requires_confirmation`;
the heartbeat must freeze/reproduce it before claiming success.

The persistent launcher now accepts an allowlisted `-RunnerModule` argument,
defaulting to the old runner. For this comparison it also holds the previous
experiment's launcher lock for the entire run, preventing a simultaneous
restart of the old pilot. The current launch was verified at 20:59:43 KST,
wrapper PID 9396 and Python launcher PID 26236. These are historical launch
identifiers; read current `process.json` before acting on any process.

Validation: all 89 focused tests passed, including nested-process shutdown,
training recipe mismatch rejection, primary/previous STOP handling, and CQL
tests. Both real ensembles passed dataset/configuration and finite-Q checks.
The PowerShell launcher parsed without errors. All foreground training/test
processes completed before launching the paired simulation job.

Resume command after clearing a user-requested STOP, from the RL repository:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_cql_ablation_v1.json `
  -RunnerModule work.run_response_cql_ablation
```

For unattended launch, start that PowerShell command with `Start-Process
-WindowStyle Hidden` as before. Do not launch a second copy while its locks
are held. The frozen spec's original `prepared_not_launched` field is planning
metadata; runtime stage is in the output `status.json`.

### CQL results and reference-coverage experiment (23:08 KST)

The paired CQL job exited normally at 22:15:43 KST. Both complete 75-step
evaluations used four preview workers each and the original baseline contract:

| Policy | Full-run TTT | Reduction vs P-Stack | Terminal inventory | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| CQL alpha=0.1 | 6058.283827 | -5.714582% | 434.570752 | 4553.923732 |
| CQL alpha=1.0 | 6307.976104 | -10.071610% | 433.668670 | 4480.777910 |

The less regularized variant substantially reduced the vanilla DDQN's
deterioration but did not beat P-Stack. Both completed audits report zero
positive selected mean-Q values and zero immediate-reward-bound violations.
Mean remaining-TTT underprediction was 295.544546 for alpha=0.1 and -3.116181
for alpha=1.0. Near-zero average error can hide offsetting errors and bad
action ranking; it is not evidence of good control. Both audits and all
source hashes are preserved under `sequential_cql_audit_v1/alpha_*/`.

For alpha=0.1, aligning each recorded interval reward to
`pstack_rl_contract_v2/trace.jsonl` decomposes the 327.490863 excess TTT as:

- Steps 0-17: -1.948084 (slightly better than P-Stack within this prefix).
- Steps 18-26: +117.197333.
- Steps 27-40: +190.270864.
- Steps 41-74: +21.970750.

About 94% accumulated in steps 18-40, before its first nonlinear action at
step 43. Do not attribute the total loss to the later nonlinear actions.
The early apparently beneficial interventions changed the subsequent state;
same nominal anchor choices later do not imply the same physical trajectory.
These timing observations do not isolate the causal contribution of each
earlier intervention.

The 246-row training replay has just 18 pre-step-18 rows, of which eight use
anchor. The only step-0 observation selected action 5. A scan of 98 replay
NPZ files under the 170-incident data/results roots found no full all-anchor
sequential replay with the current ten-action catalog fingerprint. Existing
P-Stack and known-best traces establish benchmark performance but do not
contain the full current-catalog successor response features needed by this
learner. Reconstructing reference replay is new data construction, not another
attempt to improve or retune those benchmark controllers.

**Current supervised output:**
`results/response_dqn_170_incident/sequential_cql_reference_v1/`.
The CQL alpha comparison is complete; do not relaunch it. The new input plan
is `work/response_reference_coverage_v1.json`, executed by
`work/run_response_reference_coverage.py`. Launch verified at 23:08:25 KST,
wrapper PID 40392 and Python launcher PID 22672; consult current `process.json`
instead of relying on these historical IDs.

This experiment reuses the original 246 rows and the completed alpha=0.1
policy's 75 rows (321 existing transitions), whose file hashes are pinned.
Only two additional reference episodes are collected, in parallel with four
preview workers each:

1. Native P-Stack from reset (75 rows), expected TTT 5730.792964723197.
2. The known-best action at the verified step-18 state, then native P-Stack
   continuation through the real terminal (57 rows), expected TTT
   5682.100819335536 including the fixed prefix.

Reference totals must reproduce within 0.0001 TTT, and each interval reward
must match the existing trace. Action IDs must be exactly anchor throughout,
or action 1 once then anchor for the second reference. A mismatch blocks
training for diagnosis rather than silently treating a changed controller as
the reference. Each reference still stores true sequential rewards, successor
features and terminal flags. It does not install an irreversible terminal
value label or claim the demonstrated action is universally optimal.

The pipeline then compares two training datasets with alpha=0.1, identical
seeds, network, 6,000-update budget and all other training settings:

- `policy_only`: original data plus reached states of the current best
  ungated policy, 321 rows.
- `with_references`: the same 321 rows plus the two verified reference
  trajectories, 453 rows.

This isolates the contribution of adding reference coverage from simply
adding the current policy's reached-state data. It does not separate the
effects of the two reference sources from each other. Both learned policies
are then evaluated from reset with epsilon=0, no forced action or LCB guard.
Their reference behavior during collection is not an evaluation step gate.
Training ensembles run sequentially after collection; paired evaluations
use at most eight total preview workers. Complete evaluations are audited by
the same policy-calibration script before any promotion.

Read `status.json`, `references/pstack/progress.json`,
`references/known_best_suffix/progress.json`, later `policy_only/evaluation/`
and `with_references/evaluation/`. The new coordinator reuses the existing
comparison runner with per-variant pinned datasets and model paths; immutable
`evaluation_spec.json` records the constructed comparison. Completed model
attempts are reused and partial attempts are preserved in separate directories.
All merged replay arrays are checked against their declared source arrays on
resume. The launcher holds locks for this run and both predecessor runs.
STOP in any of the three experiment directories is respected.

Validation: 92 focused tests passed, including reference total/interval/action
parity checks, true-terminal rejection, merged replay identity and all previous
collector/CQL regressions. A real-data validation-only run confirmed the 321
reused rows and the eight-worker budget. No 5% success is claimed. If this
controlled comparison also fails, inspect estimator/ranking and action-space
limitations rather than treating another unrestricted collection quota as the
default next step.

Resume using the hidden persistent launcher after an authorized STOP is cleared:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_reference_coverage_v1.json `
  -RunnerModule work.run_response_reference_coverage
```

### Reference results and terminal-boundary diagnostic (2026-09-08)

The reference-coverage runner exited normally at 01:16:08 KST. Both learned
variants completed all 75 control steps with the unchanged experiment contract:

| Policy | Full-run TTT | Reduction vs P-Stack | Terminal inventory | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Policy data only, 321 rows | 6112.272994 | -6.656671% | 430.013522 | 3975.316100 |
| With references, 453 rows | 6320.590308 | -10.291723% | 425.675904 | 3960.061029 |

The two reference collections reproduced their existing interval traces and
totals exactly. However, adding those references did not improve learned control.
Both learned evaluations had zero positive selected Q values and zero
immediate-reward-bound violations. Mean remaining-TTT underprediction was
469.014153 and 452.769431 respectively. The policy-only model used anchor 53
times and nonlinear actions 19 times; the reference model used anchor 69 times,
the known-best residual three times (52, 60, 68), and nonlinear actions three
times (2, 13, 17). Those uses are not evidence of nonlinear-price benefit.

Paired interval accounting against P-Stack gives excess TTT by time band:

| Policy | Steps 0-17 | Steps 18-26 | Steps 27-40 | Steps 41-74 |
| --- | ---: | ---: | ---: | ---: |
| Policy data only | -3.041186 | 164.731973 | 211.892302 | 7.896940 |
| With references | -3.142889 | 199.929973 | 345.297690 | 47.712569 |

These are trajectory differences, not isolated intervention effects. In
particular, early locally beneficial changes can precede later losses.

Inspection confirmed that observation `time.phase` is absolute normalized
simulation time, not a repeating signal phase. All true last-step training
observations have phase 0.9875; the finite horizon is observable. No time-feature
or terminal-flag change is justified by that check.

There is a directly testable fitting problem even on the training data. The
453-row replay contains only seven true terminal transitions (1.545%). Their
exact scaled Q targets are their immediate negative-TTT rewards, approximately
-0.212. The existing reference ensemble predicts values ranging from +0.338 to
-4.985 for their actual observed actions. Its mean absolute terminal error is
2.090830 scaled units, versus 0.335324 mean nonterminal frozen-policy TD error.
This does not require assumptions about an unobserved future recovery policy:
there is no future transition after the real environment terminal.

**Current versioned experiment:**
`results/response_dqn_170_incident/sequential_terminal_strata_v1/`.
Input: `work/response_terminal_strata_v1.json`. The reference experiment is
complete and must not be relaunched. Three new members have already finished
training on the identical 453-row replay, with the same seeds, 6,000 updates,
network, learning rate, gamma=1, CQL alpha=0.1 and target interval. The only
training intervention is allocating 25% of each 32-row batch to true terminal
transitions and 75% to nonterminal transitions. Terminal rows may be sampled
with replacement. Nonterminal targets still bootstrap over successor actions;
no trajectory return or irreversible bad-action label is installed.

This intentionally reweights both TD fitting and the CQL penalty. It is a
terminal-stratified sampling experiment, not an unbiased implementation of
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), and does not
inherit its reported results. The optional `terminal_batch_fraction=0` default
preserves the original RNG sampling path and old checkpoints load unchanged.
No reward, demand, incident, follower constraint or controller contract changed.

Read-only diagnostics are in `control_terminal_fit.json` and
`terminal_025/terminal_fit.json`, generated by
`work/audit_response_terminal_fit.py` with source/model/dataset hashes. Terminal
MAE fell from 2.090830 to 0.037361 (98.21% lower); all-row frozen-ensemble TD MAE
fell from 0.362451 to 0.164696 (54.56% lower). Nonterminal MAE fell to 0.166695.
The predeclared fit gate required terminal MAE below half the control and
all-row TD MAE below twice the control. `fit_gate.json` records that both passed.
This gate was checked before launch; training-set fit is not TTT evidence.

The next evaluation uses the existing comparison runner with one learned
variant, `terminal_025`, and eight preview workers. It starts from normal reset,
epsilon=0, no forced action, no step gate and no LCB fallback. All preview and
anchor costs are included. Its historical control used four preview workers
concurrently with another actor, so raw wall-time differences are not an
allocation-matched speed benchmark. `status.json` / `process.json` are runtime
authority; follow `terminal_025/evaluation/progress.json`. STOP in this output
or any of its three predecessors is honored. The wrapper locks all four roots.

Validation: 98 focused tests passed, including the six new terminal-sampling
tests. Tests cover unchanged uniform RNG draws, exact stratum membership,
preservation of selected training membership, absent-stratum rejection, old
checkpoint loading and recovery through later action choices. All training and
test foreground processes completed before the simulation launch.

Resume with the hidden persistent launcher, only when no existing runner holds
the locks and the user has authorized clearing any STOP:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_terminal_strata_v1.json `
  -RunnerModule work.run_response_cql_ablation
```

Launch verified at 02:06:09 KST on 2026-09-08: wrapper PID 20320, Python
launcher PID 36236. At 02:09 KST, two transitions were checkpointed with
`terminal=false`, eight process-based preview workers and no new execution
error. All four exclusive locks were held and all four STOP files absent.
Use current process/status files rather than these historical PIDs.

### Separate response-equivalence defect to address next

An independent read-only subagent audit found that the current response-memory
fingerprint hashes nominal prices and reset-only diagnostics. Therefore the
reported ten distinct response groups cannot be interpreted as ten distinct
physical controls. Saved physical vectors have only 4-7 groups per state in the
reference-policy evaluation and 4-8 in the policy-only evaluation, using the
same 1e-6 tolerance. Every state has physical duplicates. For example, at the
reference policy's step 74, actions [2, 6, 7, 8] share one physical vector and
[3, 9] share another. Different genuinely persistent follower memory may still
justify a split, so physical equality alone is not permission to merge them.
The parent independently reproduced these physical counts from both frozen
replays, checked the raw/anchor-relative feature decomposition, and preserved
per-step group counts and source hashes in
`sequential_terminal_strata_v1/physical_alias_audit.json`.

The strict `RLLeaderEnv._follower_runtime_fingerprint` is appropriate for
snapshot/context integrity but too broad for the equivalence test currently
using it. Nominal price fields installed by coordination and scratch fields
such as `_seg13_diag`, `_seg_traj`, and `_wu._repair_diagnostics` contaminate
it. Anchor trials also fingerprint the active optimizer follower instead of
the canonical adopted follower state used by the subsequent decision. A
no-simulation controlled mutation of identical saved followers produced ten
hashes solely from different catalog price inputs, despite equal controls.

Keep the terminal-sampling ablation's existing masks unchanged to isolate its
effect. Do not claim response deduplication is correct or nonlinear causation
from this evaluation. The next separate implementation must identify the
canonical state actually carried into the next solve, exclude only provably
overwritten inputs/scratch state, retain genuine warm-start memory, and add
regressions for nominal-only mutations versus causal memory changes. Preserve
the strict snapshot fingerprint. Report physical groups separately from
continuation-equivalence groups and save enough per-candidate continuation
evidence to audit splits. Old replay lacks that evidence and must not be
silently relabeled as corrected-equivalence data.

#### Canonical continuation follow-up (02:58 KST)

The follow-up read-only trace qualifies the broad fingerprint finding above:
**do not remove every price field or every scratch-looking field.** The
canonical key must describe the post-commit environment, not just whichever
follower `_active_controller` currently references. Native anchor commitment
installs the prepared optimizer controller and adopts its selected follower
memory into the RL follower (`env.py`, `step_prepared_optimizer_anchor`). A
residual commitment retains the existing native controller's pricing/history;
the prepared native trial is discarded. The next anchor synchronizes only
the explicit `_copy_follower_runtime_state` list before cloning that native
controller. Consequently, equal executed controls do not guarantee equal
subsequent native anchors.

Retain at least the following audited causal state in an explicit, versioned
payload, restricted initially to the current controller configuration:

- Full next `TrafficState`, time/step and executed previous control, including
  budgets and the `wu_b3_release_ratio_*` diagnostics used by price refresh.
- RL-carried `_prev_coupling`, `_lambda_P`, `_lambda_UF`, the `_np_*` predictor
  and corrector fields in the runtime copy list, and inner Wu off-ramp memory
  `_last_offramp_flow` / `_has_last_offramp_flow`. Preserve `None` versus zero.
- `_phase_resolved_active_signals`: the main solve resets it, but native
  `local_green_costs` probes can consume its previous value before that reset.
- Retained native follower prices, references, trust settings, directives,
  certificates and pricing runtime flags. The inner price-refresh predicate
  can return without updating even when the outer refresh interval is one.
- Native `_beta_prev_total_veh`, `_regret_window`, pending incumbent prediction,
  forced-step counter and entry-update guard. The current configuration enables
  `regret_guard_steps=3`; this is existing native P-Stack behavior shared with
  the benchmark, not an added learned-policy LCB fallback. Preserve additional
  beta-estimator history if it influences the selected action.
- Both FAR hysteresis flags. They are not unconditionally recomputed.

Old residual prices, certificates and directives are cleared/replaced by the
next coordination `apply` and are not copied to the native follower. Those
obsolete residual inputs can be excluded. `_seg_traj`, `_seg13_diag` and
`_wu._repair_diagnostics` reset before their next consumers; `_wu._omega_f`,
`_nuf_solve_cache` and `_link_share_ctx` are also reset/recomputed at decision
entry. Static configuration-derived segment models and candidate traces are
not causal dynamic memory. Keep the existing strict fingerprint for integrity.

Parent source inspection confirmed the runtime-copy list (`env.py:861`),
native decision-entry path (`env.py:927`), regret carry-over
(`stackelberg_wu_metered.py:302`) and refresh early return / release-diagnostic
consumer (`stackelberg_wu_metered.py:1218`). Require matching validity and
immediate reward as well as the next-state key before merging. Do not round
state fields to make hashes equal. Unsupported controller modes must refuse
this equivalence shortcut until audited; for example SPSA can make a refresh
counter causal via its random seed.

Required regressions before enabling a new mask: obsolete residual-only and
reset-only mutations preserve the key; every retained causal memory mutation
changes it; native pricing/regret differences remain distinct despite equal
controls; phase-active sets are consumed before reset; budgets/release
diagnostics/hidden plant buffers/FAR flags remain distinguished; branch labels
alone do not split otherwise equal canonical state; checkpoint round-trips
and dictionary ordering preserve the key. Save the retained payload or its
fieldwise digests so every split is inspectable.

The terminal-sampling full-run remains active and unmodified. This follow-up
is design evidence for the next version, not a claim that its mask has already
been repaired or that missing memory explains a measured fraction of TTT loss.

### Terminal-stratified result and corrected-equivalence cycle (04:00 KST)

The terminal-stratified runner completed at 03:17:21 KST, exit code zero.
Its ungated full-run TTT is **5909.832493062337**, or **3.12416675% worse**
than P-Stack. Terminal inventory is 428.685949; measured wall time is
4266.487935 seconds with eight preview workers. This improves over the matched
453-row uniform-sampling control (TTT 6320.590308), but does not meet the
P-Stack baseline or the 5% target. The model chose anchor 72 times, linear
action 2 at steps 17/19, and nonlinear action 8 at step 2. All 75 steps passed
validity, with no positive selected Q or immediate-reward-bound violation.
Mean remaining-TTT underprediction was 461.859923; frozen-policy TD MAE was
0.698415 scaled units. Better terminal fitting did not solve full-horizon
calibration or action ranking.

Its excess TTT against baseline is -3.056931 at steps 0-17, +83.409800 at
18-26, +102.082775 at 27-40 and -3.396115 at 41-74. Again these are reached
trajectory differences, not per-action causal labels. Do not rerun the
completed terminal-stratified experiment.

Next version: `work/response_continuation_cycle_v1.json`, output
`results/response_dqn_170_incident/sequential_continuation_v1/`. The dedicated
continuation identity is being implemented/reviewed in
`rl_leader/response_continuation_state.py`. Main integration is in the existing
collector, model/replay metadata, and `work/run_response_continuation_cycle.py`.
The new mask mode is `post_commit_continuation_v1`; old data/checkpoints default
to `legacy_follower_runtime_v1`. Do not silently recompute old replay masks or
mix the two modes. Models carry the mask mode, and runtime selection refuses a
model/mask mismatch. The original strict snapshot fingerprint is unchanged.

The post-commit identity includes actual interval reward and terminal status;
the collector adds validity. Each selected action's committed identity must
match its preview before a row is recorded. Unsupported identity states cause
an audit failure, not a silently masked "infeasible" candidate. New traces
separately record physical-control groups, continuation groups, and candidate
component identities. The learner still uses the same observation schema,
physical response features, true one-step rewards and Bellman successor targets.

Planned acceptance gates before the fresh learning cycle:

1. Unit/independent review of the new identity and integration. Retain causal
   state and conservatively preserve unknown serializable runtime fields;
   reject unsupported modes/object types instead of using repr-based hashes.
2. Matched-state probes at verified baseline steps 17 and 18. Their observations
   already exactly match the frozen reference replays. Compare every candidate's
   physical preview against old recorded features, execute all ten candidates
   independently, and verify preview/commit identity equality. For every newly
   merged group, also execute the next native anchor from each member and
   require equal resulting identity and controls. If no merge is verifiable,
   stop at a diagnostic phase instead of blindly collecting more data.
3. Collect two fresh reference trajectories first: all-anchor from reset (75
   transitions) and known-best step-18 residual then anchor (57). Require their
   old interval traces, action sequences and full TTT totals to reproduce before
   spending the exploration budget.
4. Collect two additional epsilon=0.25 response-space exploration trajectories:
   one from reset and one from step 18 with the known-best first intervention.
   These are random masked behavior policies, not falsely described as rollouts
   of a newly learned policy. Total new replay is at most 264 transitions.
5. Train three members on only the fresh, compatible replay using the existing
   6,000-update, alpha=0.1, gamma=1, terminal-fraction=0.25 recipe, then evaluate
   the frozen ensemble from reset without a gate/forced action/LCB fallback.

Each two-actor stage uses four preview workers per actor; the final single
evaluation may use eight. The persistent wrapper will hold the new root and
all four predecessor locks. STOP in any of those roots is respected. Completed
probes, episodes and model directories are reused; source code/input hashes are
pinned. Source changes after a failed probe need a reviewed versioned output,
not silent adoption of incompatible checkpoints. Corrected data construction
and mask semantics necessarily differ from the old experiment, so a resulting
TTT change would not isolate the quantitative causal effect of deduplication.

The new cycle is not yet recorded as launched in this entry. Check subsequent
launch notes and current `process.json` / `status.json` before taking action.
Foreground probe command, from the repository, before persistent collection:

```powershell
.venv-torch/Scripts/python.exe -B -m work.run_response_continuation_cycle `
  --config work/response_continuation_cycle_v1.json --probe-only
```

After probes and review pass, use the hidden persistent launcher with:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_continuation_cycle_v1.json `
  -RunnerModule work.run_response_continuation_cycle
```

## Continuation parity diagnosis and v3 (2026-09-08 09:31 KST)

The v1 probe-only wrapper (PID 27552, child 35680) failed at 04:32:21
with `preview/commit identity mismatch: action 1`. No collection or model
training started. Its config and output remain unchanged.

The v2 diagnostic probe (wrapper 46664, child 30992, 04:38:27-04:41:41,
exit 0) preserved individual preview/commit component and field digests.
Its status is `probe_commit_mismatch_requires_diagnosis`, NOT a passed probe.
All physical preview features exactly matched the old reference replay.
Step 17 groups were `[0], [1], [2], [3,9], [4], [5], [6,7], [8]`, with
commit mismatches at actions 1,4,6,8. Step 18 groups were `[0], [1],
[2,6,7,8,9], [3], [4], [5]`, with mismatches at actions 1,3.
The only differing fields were these previous-control diagnostic copies:

- `leader_base_accumulation`
- `leader_state_accumulation_base`
- `leader_boundary_leg_excluded_veh`

No plant, physical control, follower/native-controller memory, reward or
validity difference appeared in those saved comparisons. The numerical
magnitude of the report differences was not saved; floating-point summation
order is a hypothesis, not a measured conclusion. Source search found these
keys emitted in `leader.py:objective_terms` and copied in
`stackelberg_mpc.py:_make_fallback_evaluation`, with no later control consumer.
The previous-diagnostics price-refresh consumer in
`stackelberg_wu_metered.py` reads only `wu_b3_release_ratio_*`, which remains
included. This justifies excluding exactly the three report copies, not
rounding physical state or dropping diagnostics wholesale.

At 09:28-09:31, after confirming no matching live runner, the identity helper
was changed accordingly and a mutation regression added. The probe now also
persists both branches when next-anchor validation fails, then blocks
collection with `probe_next_anchor_mismatch_requires_diagnosis`. Regression
tests cover evidence preservation and the blocking gate. Subagent Hubble's
requested additional read-only review was unavailable due to its usage-limit
error; the parent performed the consumer trace above. Do not describe that
extra review as completed.

New immutable plan: `work/response_continuation_cycle_v3.json`, root
`results/response_dqn_170_incident/sequential_continuation_v3`. It keeps the
same planned 264-transition ceiling and two-by-four worker allocation, and
adds the v2 lock/STOP to all predecessors. Neither v1 nor v2 contains fresh
corrected replay to reuse; their diagnostic probes remain preserved. Run the
regression suite and then this locking, probe-only command before collection:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_continuation_cycle_v3.json `
  -RunnerModule work.run_response_continuation_cycle -ProbeOnly
```

Only a fully passed probe may proceed using the same command without
`-ProbeOnly`; it reuses successful probes and saves resumable episodes before
training/evaluation. A passing probe is not evidence of better TTT. No new
full-run result or 5% achievement exists at the time of this entry. See later
launch/result notes and process/status files for current state.

### v3 regression and probe launch (09:33 KST)

All 146 focused regression tests passed in 139.230 seconds. This included
continuation state/cycle/integration, terminal sampling, reference coverage,
CQL runner/objective, process-pool cleanup, collection contracts, sequential
learning/recovery, response DQN and MC-conversion regression suites.

The locking probe-only command above started at 09:33:20 KST, wrapper PID
41348 and child PID 25280. Logs have prefix `runner_20260908_093320_270` in
the v3 root. The initial phase was `probing_continuation_equivalence`.
SHA256 pins at launch:

- Plan: `b4bd6f23f7bd520c35adb2946b2f4d5591192f115f6993daf17b7be5923dfc57`
- Identity helper: `b6cf70a11fd9768db95a836e70f64cb8975f043ccd591abfaaae5ff4ffe12a3f`
- Cycle runner: `ff733609add90c0f220e5ae24e676019658a7113dfd5479a3633a3e980b90900`

The root's `implementation_hashes.json` pins all 14 implementation files.
Do not edit these during this experiment or overwrite the previous probes.

### v3 probe passed and cycle resumed (09:38 KST)

The v3 probe-only wrapper exited 0 at 09:36:39 with `probe_complete`.
Both states exactly reproduce the old physical previews; all 20 independent
preview/commit checks pass. At step 17, the ten nominal candidates have seven
continuation groups: `[0], [1], [2], [3,9], [4], [5], [6,7,8]`.
At step 18 there are six: `[0], [1], [2,6,7,8,9], [3], [4], [5]`.
All seven representative-to-alias pairs (3 at step 17, 4 at step 18) also
match in next-native-anchor identity and physical controls. This is a bounded
parity check, not a proof for every future state or a TTT improvement.

At 09:38:13 the same v3 plan was resumed without `-ProbeOnly` using a hidden
persistent PowerShell wrapper, PID 40008. It holds the v3 and predecessor
locks. Passed probes are reused, then two reference episodes are collected
and verified before the two exploration episodes. After 264 compatible raw
sequential transitions at most, the runner trains three members and performs
one ungated full evaluation. Check current `process.json`, `status.json` and
`collection/*/progress.json` for live child identity/progress. The hourly
`ddqn` heartbeat remains active and follows this audit's newer output root.
No second runner or duplicate scheduler was created. This cycle's completion
will still require the full acceptance/diagnostic review below.

Initial collection smoke check at approximately 09:41 KST: wrapper 40008 /
child 36652 are running; log prefix `runner_20260908_093814_132`.
`pstack_reference` has two committed transitions through step 1, cumulative
TTT 158.77109107001834, saved wall time 166.349765 seconds.
`known_best_reference` has two through step 19, cumulative TTT including its
fixed prefix 2507.392240918869, wall time 194.572284 seconds. Both saved
`checkpoint.pkl`, `replay.npz`, and `progress.json`; neither is terminal.
There is no fresh training or full-evaluation result yet. Do not mistake the
fixed-prefix reference for ungated learned-policy performance. The two
actors continue within the shared eight-preview-worker budget. Resume this
root, not old completed collections, on the next actionable heartbeat.

### v3 completed reference parity (2026-09-08 11:03 KST)

Both fresh references completed, passed the runner's parity gate, and were
independently rechecked read-only with `verify_reference` against the original
interval traces, full TTT totals and prescribed action sequences. The replay
contract hash and `post_commit_continuation_v1` mode also match. Each episode
has exactly one true terminal transition at control step 74.

- `pstack_reference`: 75 transitions, TTT 5730.792964723197 (delta 0),
  terminal inventory 430.8500610965373, measured wall 4020.772538 seconds.
  Continuation-group count min/mean/max: 5 / 6.053333 / 8; all 75 rows merge
  at least one of the ten nominal candidates.
- `known_best_reference`: fixed-prefix continuation, 57 transitions from
  step 18 with prescribed first action 1, TTT including prefix
  5682.100819335536 (delta 0), inventory 430.1801541627784, measured suffix
  wall 2629.998049 seconds. Groups: 5 / 5.947368 / 7; all 57 rows merge
  at least one nominal candidate. This is NOT an ungated learned result.

Timings include the collection previews: two concurrent actors with four
preview workers each. The suffix timing excludes replaying its saved prefix;
do not report it as a full-run runtime or a preview-free inference benchmark.
These measurements validate the new data/mask construction on two references,
not a performance benefit of the new policy or nonlinear pricing.

At the 11:01 heartbeat, wrapper 40008 / child 36652 remained live, with no
STOP file or actionable failure. The two exploration episodes had already
started automatically after reference verification: `explore_full` had 13
transitions through step 12, and `explore_after_best` had 14 through step 31.
Keep the active config/source pins unchanged. The same runner will merge the
four compatible episodes, train three members, then evaluate ungated. Do not
repeat the completed 132 reference transitions or start another runner.

### User-requested recovery after shutdown (2026-09-08 12:12 KST)

The user reported the computer had shut down and requested continuation.
The previous wrapper 40008 / child 36652 recorded exit 1 at 12:06:10,
with `BrokenProcessPool` during the corrected model's full evaluation.
The log establishes abrupt worker termination; it does not independently
establish the operating-system-level cause. No matching live RL or spawned
worker process remained, and no applicable STOP file was present.

Collection and training had already finished before the interruption:

- All four collection episodes are complete: 75 + 57 + 75 + 57 = 264 rows.
- Exploration from reset: TTT 5745.099378554329, inventory 431.3654480634343,
  wall 3889.577979 seconds. Epsilon is 0.25, NOT a frozen-policy evaluation.
- Exploration after the fixed step-18 prefix: TTT 5728.648480627664,
  inventory 430.295136669961, suffix wall 2810.352039 seconds; also NOT an
  ungated learned result.
- Three trained members each completed 6,000 updates on the frozen batch.
  Training action counts are `[228,8,8,7,6,5,2,0,0,0]`; zero representative
  counts for nominal actions 7-9 must be interpreted with the new alias groups.
  Full action/response coverage and Q calibration remain to be audited after
  the actual learned-policy evaluation completes.

Read-only preflight verified all 20 pinned source/input hashes, all three
model hashes, the 264-row training hash and training recipe, all completed
episode chains, and both reference parity checks. The partial evaluation
checkpoint contains 13 transitions through control step 12, next step 13,
TTT so far 1242.2331736820793 and saved wall 902.129900 seconds. Its observed
state, replay arrays, environment contract and equivalence mode agree. Replay
arrays were compared at their persisted dtypes: a preliminary raw float64
list versus float32 replay comparison was too strict, not data corruption.
No simulator, policy, reward or evaluation configuration was changed.

The interrupted process/status, checkpoint, replay and progress were copied
without overwriting originals into `sequential_continuation_v3/resume_20260908_1208/`,
alongside `preflight.json`. The checkpoint SHA256 is
`7b06e25c528078a0afe73da49e67b23994c7479c66b12881e5ac1a561a7b43ad`;
the evaluation payload signature is
`73b0957304cc073c7ff8c29c4c98288bc491be785b1a85597e0fb18710c8895b`.

Resumed the same locking wrapper at 12:11:56 with wrapper PID 18840,
child 20372 and log prefix `runner_20260908_121156_878`:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_continuation_cycle_v3.json `
  -RunnerModule work.run_response_continuation_cycle
```

The runner reused all four summaries and the completed ensemble, then entered
`evaluating` with eight preview workers. New trace segments preserve the
old trace. It resumes from the exact checkpoint, not a newly forced action
or a newly selected prefix. Runtime after resumption accumulates saved active
time plus new active time, excludes downtime, and can miss uncheckpointed work
lost on termination. Label it resumed measured runtime, not an uninterrupted
speed benchmark; a threshold candidate still requires a frozen, clean full
reproduction. No 5% result has been established.

On the user's preceding request, the existing `ddqn` follow-up schedule was
changed from hourly to once every three hours; the training runner itself
remains continuous. Preserve that schedule and do not create a duplicate.

Post-resume smoke verification passed: the evaluation advanced to 15 saved
transitions through step 14. Every array in the original 13-transition replay
prefix remains exactly unchanged, and all three model hashes still match.
The fresh trace is `trace_resume_001.jsonl`; the old trace is preserved.
No collection episode or training member was rerun. The current runner is
healthy at this check and continues the remaining full evaluation.

## v3 result and reached-state refit (2026-09-08 14:25 KST)

The resumed v3 wrapper exited 0 at 12:55:16; its full evaluation completed
all 75 control steps with the required scenario/contract, epsilon=0, no
forced action or LCB fallback. TTT is **6104.21938307732**, **6.5161387% worse**
than P-Stack, not a goal candidate. Terminal inventory is 436.0796793506895
(P-Stack 430.8500610965373). Measured resumed active runtime is 3493.295324
seconds including native-anchor and candidate previews, with the downtime/
lost-uncheckpointed-work caveat in the recovery entry. Do not present this
as a clean uninterrupted speed measurement. Do not rerun the completed v3.

The audit reconciles rewards/TTT, all 75 greedy actions and trace segments,
contract and model hashes. Actions: anchor 71 times, action 5 at steps 0/3,
and known-best artifact action 1 at steps 16/18. Action 1 changes only
`urban.D.g_offset`, `urban.F.g_green`, `freeway.R_D_W.g_vsl`, and
`freeway.R_F_W.g_meter`; action 5 changes only `freeway.R_F_W.g_vsl`.
There were **zero selected quadratic/cross residuals** in this evaluation.
The catalog's generic `hybrid` label for action 1 does not imply a nonlinear
price action. This result establishes neither nonlinear-price learning nor
nonlinear-price causation.

Excess interval TTT versus P-Stack, from the reconciled float32 replays:
steps 0-15: -4.507149; 16-18: +8.071762; 19-26: +152.293579;
27-40: +206.672310; 41-74: +10.895987. These are trajectory differences,
not causal counterfactual labels for the four interventions. Initial predicted
remaining TTT is 3899.367778 versus realized 5994.516094; mean remaining-TTT
underprediction is 792.982817. Frozen-policy TD MAE is 1.810628 scaled,
with 20 positive selected-Q rows and 22 immediate-reward-bound violations.
All validity gates pass. Continuation groups min/mean/max are 5/5.866667/7
on evaluation and 5/5.821970/8 on the 264-row training batch. Median nearest
same-step standardized observation RMS is 0.500125; initial states at steps
0 and 3 are exactly in training, but steps 16/18 have RMS 0.936865/1.259900.

Independent read-only feature/Bellman review by subagent Locke found:

- Constant response columns (including previously all-zero offsets/deltas)
  have fallback scale 1 and retain exactly initialized first-layer weights.
  Newly reached raw offsets can therefore drive arbitrary Q variation. At
  step 38, B/D offsets become 105/75 fallback-scaled units. Zeroing only
  training-constant response coordinates changes selected Q from -33.4472
  to -3.7239, but changes **none** of the 75 frozen greedy choices. This
  proves calibration sensitivity, not the cause of the TTT loss.
- Training terminal MAE is only 0.019105 on four terminal states, versus
  2.412376 on the newly reached terminal (Q +2.197793, exact target -0.214583).
  Terminal time/phase matches training. Some reached B-queue observations
  exceed 30 standardized units; smallest positive observation standard
  deviation is 0.006371, not a numerical near-zero-scale explosion.
- No reward sign, true-terminal-mask, replay successor, or this ensemble's
  support mismatch was found. Per-member DDQN and ensemble greedy operators
  differ on some states by design. Keep that distinction in diagnostics.

### Next falsifiable experiment

New plan `work/response_onpolicy_refit_v1.json`, output
`results/response_dqn_170_incident/sequential_onpolicy_refit_v1`. No runner
has been recorded as launched at the time of this entry. The new runner is
`work.run_response_onpolicy_refit`; check later launch/status notes first.

Reuse the 264 existing rows plus all 75 actual sequential transitions from
the completed v3 policy: 339 rows, **zero new collection episodes**. Preserve
negative interval-TTT rewards, true terminals and Bellman successor targets;
do not replace them with realized-return labels. Refit the same three seeds,
6,000 updates/member, CQL alpha .1, gamma 1 and terminal batch fraction .25.
An opt-in exact training-constant observation/response feature mask is being
implemented with matching training/inference handling, persisted masks and
legacy-compatible defaults. Newly varying coordinates in the expanded batch
remain learnable. Do not clip active coordinates or alter physical controls.

Before spending a full simulator evaluation, compare old/new Q on exactly the
same last-policy states. Prespecified fit gate: terminal behavior-Q MAE must
fall by at least 50%, and mean ensemble-greedy one-step behavior TD MAE must
not exceed 1.25 times the control. These are in-sample calibration checks,
NOT held-out policy performance or a runtime fallback guard. Failure blocks
automatic full evaluation and requires another diagnosis; success launches
one unchanged-contract, normal-reset, ungated 75-step evaluation. The feature
mask alone has not established a policy improvement, and this refit is not
an isolated estimate of the quantitative effect of either change.

The plan pins both source datasets, the control summary/manifest/checkpoints,
training/environment specs, and implementation sources. It holds the new
root plus v3 and all predecessor locks/STOP aliases. Training uses three
single-thread workers; the single subsequent evaluation uses eight preview
workers. Existing checkpoints/results remain intact. Subagent Curie is
reviewing orchestration; Locke owns the scoped feature-mask patch/tests.

### Refit implementation/review and launch (14:40 KST)

Locke implemented the opt-in persisted exact-constant masks in
`response_dqn.py` and the training CLI. Default and old checkpoints retain
the original unmasked behavior. Matched-seed unit training verifies identical
training losses/weights/in-support predictions, while masked-coordinate
perturbations no longer affect predictions. Active coordinates remain
unclipped; newly varying coordinates are reactivated on refitting.

Curie's orchestration review identified incomplete source pinning and STOP
not reaching a running trainer. Both were fixed before launch: the plan now
pins all 166 runtime Python files under `src` (excluding tests) and
`rl_leader`, plus work helpers/auditor and the launcher, 172 files total.
STOP paths propagate as operational CLI arguments to members, checked before
setup, at most every 100 gradient updates, and before returning. Completed
member outputs are preserved; an interrupted incomplete ensemble is retained
and any retry uses a new attempt directory. No new optimizer-resume format was
introduced. STOP paths do not enter the model's learning hyperparameters.

Final scoped independent review found no blocking issue. The parent's full
focused regression run passed **168 tests in 145.164 seconds**, including
feature masks, cooperative STOP, the refit gate, old checkpoint behavior,
continuation identities, terminal sampling, replay contracts and DDQN tests.
No applicable STOP file existed and the new root did not yet exist at preflight.

Started the hidden locking wrapper at 14:40:17 KST, wrapper PID 30692:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_onpolicy_refit_v1.json `
  -RunnerModule work.run_response_onpolicy_refit
```

Current source/config must now stay unchanged. Follow this new root's
`process.json`, `status.json`, `fit_comparison.json`, and
`onpolicy/evaluation/progress.json`; do not relaunch v3. The same-state
calibration gate is a pre-evaluation resource check, not a runtime fallback.
If it fails, diagnose instead of launching a full run automatically. If it
passes, the existing evaluation runner proceeds without any intervention
gate. The 5% acceptance and clean reproduction requirement remain unchanged.

### Refit calibration passed; evaluation running (14:43 KST)

New root `sequential_onpolicy_refit_v1`: wrapper 30692 / child 32168,
started 14:40:18, log prefix `runner_20260908_144018_384`. Plan SHA256:
`2dbb3bafa2ff6101ec4cf8d24431057abfbd6f80f78e2f533d10786570b28ed5`.
The three 6,000-update members completed on 339 rows, with five true terminal
rows and 334 continuing Bellman rows. Behavior counts:
`[299,10,8,7,6,7,2,0,0,0]`. No new simulation collection was run.

The prespecified same-state calibration gate passed:

- Latest-policy 75-state terminal behavior-Q MAE: 2.412376 -> 0.034209.
- Latest-policy behavior TD MAE: 1.810628 -> 0.144326.
- Positive behavior-Q rows on those states: 20 -> 0; immediate-reward-bound
  violations: 22 -> 1. The residual terminal overestimate remains small,
  not exactly zero.
- On the original 264 rows, TD MAE mildly worsened 0.134926 -> 0.151851,
  and terminal MAE 0.019105 -> 0.043847. Do not imply every fit metric improved.

These are in-sample errors after including the 75 transitions in training.
They do not establish a TTT improvement or justify a claim of generalization.
The fitted mask disables 63/238 observation columns and 8/60 response columns.
A read-only check on the first trained member found exactly zero Q difference
on all 339 valid training state/action pairs with its constant masks enabled
versus disabled. No active-feature clipping or runtime fallback was added.

The runner automatically entered `evaluating` with eight preview workers.
Initial smoke check: one saved transition through step 0, cumulative TTT
130.42489548385547, measured wall 100.679624 seconds, no terminal/error.
Follow `onpolicy/evaluation/progress.json`; keep this experiment immutable.
The full result and 5% test remain pending. Review it on the existing
three-hour heartbeat; do not relaunch training or the completed v3 run.

### Refit full result and value-head comparison (2026-09-08)

The latest completed root is `sequential_onpolicy_refit_v1`, wrapper 30692 /
child 32168, exit code 0 at 15:38:49 KST. Its ungated normal-reset evaluation
completed all 75 decisions / 14,400 seconds under the unchanged acceptance
contract. TTT is **6140.1674526914285**, **7.1434178566% worse** than P-Stack;
terminal inventory 430.88953745200564, measured uninterrupted wall time
3480.2337458 seconds with eight preview workers, including anchor/previews.
Signature `a937b4b6c4376d138776ba5a7f780381c3cd3af82f870fdc986c3dc0d2b77000`.
This is not a target candidate and must not be retrained or rerun in place.

Actions were `[69,4,0,0,0,2,0,0,0,0]`: action 5 at steps 0 and 3; action 1
at 7, 9, 10 and 24. No quadratic/cross action was selected. Response-group
counts were min/mean/max 5/5.88/7, with zero validity failures. Excess interval
TTT versus anchor by inclusive control-step bands: 0..6 +0.091164,
7..15 +21.977852, 16..18 +43.938324, 19..26 +138.872177,
27..40 +193.239933, 41..74 +11.255085. These trace differences do not isolate
the causal effect of an individual intervention.

The previous in-sample calibration gains did not transfer to the new policy:
33 selected Q values are positive despite negative interval-TTT rewards;
34 exceed their observed immediate reward. Initial predicted remaining TTT
4631.422424 versus realized policy return 6030.464140; mean underprediction
875.609556, ensemble-greedy TD MAE 1.084151. Terminal Q is +2.951029 versus
the exact scaled target -0.213583, error 3.164612, ensemble std .135929.
The terminal nearest same-step state RMS distance is .274920.

Curie's read-only review corrects an important interpretation: step 7, the
first departure from the preceding policy, is an EXACT existing state with
two behavior samples, actions 2 and 0, but not action 1. Its action-1 Q
-41.7125 exceeds anchor -42.2206 although both are already negative.
The failure includes state-action support/ranking, not simply unseen states
or positive Q. Sign constraints cannot by themselves guarantee this decision
is corrected. Do not describe the proposed head as a proven root-cause cure.

The user's computer-off/resume request was checked against persisted files
and read-only Win32 process identities at 17:28 KST: no RL Python or launcher
was live. The completed result is intact; resume the next version, not old
collection. The existing `ddqn` heartbeat remains ACTIVE every THREE hours.

New controlled plan: `work/response_value_head_v1.json`, output
`results/response_dqn_170_incident/sequential_value_head_v1`, runner
`work.run_response_value_head_ablation`. Reuse the 339 training rows plus all
75 latest actual sequential transitions: exactly 414 rows, no new collection,
same row multiplicities, true terminals and Bellman successor targets.
Compare paired-seed `free_q` versus `finite_horizon_cost_v1`, keeping three
members, 6,000 updates/member, masks, gamma 1, CQL .1, terminal fraction .25,
and all other training conditions fixed. Both receive full ungated evaluation;
no calibration-based selection gate is used. Evaluation runs two actors with
four preview workers each, total eight; training is three single-thread
members per variant, variants trained sequentially.

The opt-in head is `Q_scaled = -remaining_intervals * softplus(logit)`.
The raw `time.phase` grid determines the remaining horizon, including the
current decision. Validate float32 grid tolerance then ROUND, never truncate:
80 total intervals, warmup leaves 75, final decision 1, terminal successor 0.
Persist this time/value contract and apply the head consistently in current,
online-successor, target-successor and inference/CQL calculations. It enforces
Q <= 0, terminal-successor Q = 0, and Bellman targets <= immediate reward;
it does NOT enforce fitted Q <= immediate reward or correct action ranking.
Softplus outputs are already reward-scaled per-interval costs. This also
changes optimization geometry, so compare loss/convergence as well as TTT.
Zero positive-Q counts are structural, not evidence of learning success.

Input hashes and all runtime source hashes are pinned; STOP aliases and
exclusive locks span every predecessor. Completed models/replays are reused
on recovery, never overwritten. No new run was launched when this entry was
written; follow the subsequent launch/status entry. If both heads still fail,
inspect action-level ranking/support before another refit or large collection.
The authorized 5% loop and clean full-run reproduction requirement continue.

### Value-head implementation and preflight (17:37 KST)

Locke completed the scoped core/CLI patch and 12 head tests; Curie's independent
read-only review found no blocking issue in the core or orchestration. Parent
regressions passed 186 tests in 142.523 seconds, then the expanded orchestration
and recipe tests passed 12 tests in .043 seconds (one additional distinct
happy-path test, 187 distinct tests total). The final orchestration check
verifies identical data/seeds, STOP propagation, and evaluating both variants
even with deliberately poor calibration. Old free-Q checkpoint compatibility,
late recovery learning, true terminals, masks, continuation parity and resume
tests passed. Added explicit 0..74 / 75-row policy-source validation.

Read-only merged-data preflight: 414 rows, six true terminals and 408 continuing
rows, counts `[368,14,8,7,6,9,2,0,0,0]`. Rewards range -240.050629 to
-20.718054. Cost contract derives raw phase column 0, N=80, first remaining75,
last remaining1 and all six terminal successors0. Source inventory covers all
166 runtime Python files plus seven work/launcher files (173 pins total).
All nine input hashes match; no applicable STOP exists; new root is absent.

Launch command for this version, through the existing hidden locking wrapper:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_value_head_v1.json `
  -RunnerModule work.run_response_value_head_ablation
```

Keep its pinned configuration and runtime sources immutable after launch.
Follow its `process.json`, `status.json`, `fit_comparison.json`, and each of
`free_q/evaluation/progress.json` and
`finite_horizon_cost/evaluation/progress.json`. The existing three-hour
heartbeat is unchanged. Await the following process identity/smoke record.

### Value-head comparison launched (17:38 KST)

Hidden wrapper PID **34420**, child runner PID **29132**, started
2026-09-08T17:37:48.4932282+09:00. Log prefix
`runner_20260908_173748_472`. Process state is running; initial status is
training `free_q` on 414 transitions with zero new simulator collection
episodes. Both stdout/stderr were initially empty, with no startup error.
The wrapper holds all version/predecessor locks. Keep this experiment's
sources/config fixed; do not start another runner. Final results and the 5%
acceptance check are pending. Read later smoke notes before taking action.

### Both heads trained; full evaluations started (17:40 KST)

Both three-member ensembles completed 6,000 updates/member and passed recipe,
checkpoint-head, finite-Q and cost-contract checks. `status.json` is now
`evaluating`, variants `free_q` and `finite_horizon_cost`, four preview workers
per actor. No new collection was performed. Both evaluations are normal-reset,
epsilon-zero, unforced and ungated; neither was selected by calibration.

Plan SHA256 `25a7c25912a4334e2526f339ce28f3d3d645eb8261e2d5b7884a08532b8dd7e8`;
414-row replay SHA256
`42081d2858ecc97e6aa1c3af3e7a6b67ac195b5f4f16d7414b9077e75ceeb640`.
All six checkpoint hashes are in `model_hashes.json`, whose SHA256 is
`7b23ae2f83bf3907815dc427189c84a85e269490585c62f24c4a9f1ce21d352d`.

Frozen 414-row ensemble-greedy behavior TD MAE is .115679 for free Q versus
.155303 for cost Q; terminal behavior MAE .014785 versus .040969. On the
latest 75 trajectory rows, TD MAE is .104666 versus .182594, terminal MAE
.010925 versus .009588. Both have zero positive behavior Q in-sample; cost
head also has exact zero terminal-successor Q. Its structural checks passed,
but the fit metrics do not uniformly favor it and do not establish TTT gain.
Preserve the two-arm test rather than adjusting training during evaluation.

Startup stderr contains the same pre-existing component geometry calibration
warnings as the prior completed experiment; no traceback was observed in
startup inspection. The first saved simulator transitions were still pending
at this entry, so no full-run result is claimed. Follow both progress files,
then `comparison.json` and per-variant `calibration_audit.json`. Do not launch
an overlapping experiment or rerun completed training. The target remains
unmet; use the existing three-hour heartbeat for completed-result diagnosis.

### First saved transitions verified (17:41 KST)

Wrapper 34420 and child 29132 remain live with matching start times. Both
full evaluations have saved one transition through control step 0, terminal
false, cumulative TTT 130.42489548385547. Measured initial wall time is
104.896658 seconds for `free_q` and 105.249204 for `finite_horizon_cost`.
This verifies that both trained policies entered the actual collector and
checkpointed successfully; these one-step values are not performance evidence.
Leave the current two evaluations running, total eight preview workers.

### Value-head full comparison reconciled (2026-09-08 20:13 KST)

`sequential_value_head_v1` completed, wrapper34420/child29132 exited0 at
18:46:53. Independent re-execution of the read-only calibration audit verified
both full traces, checkpoint greedy choices, rewards, 75 steps0..74, true final
terminal, 14,400 seconds, scenario and unchanged acceptance contract. No STOP
file, traceback or surviving Python experiment process was found at review.

- `free_q`: TTT5884.0284638055855, **2.6738969637% worse** than P-Stack;
  terminal inventory430.85376632564453, wall4082.056899700001 seconds.
- `finite_horizon_cost`: TTT5695.7051958025695, **0.6122672575% better**;
  terminal inventory431.97053383860225, wall3841.0280577999984 seconds.
  This is below P-Stack but still 251.451879315533 above the 5% target.

Both used four preview workers concurrently, including anchor and response
previews in measured wall time. Both are ungated normal-reset learned-policy
evaluations, not preview-free inference. This single paired experiment supports
the head change on this training batch/scenario, not generalization or proof
that the sign bound alone caused the gain; horizon scaling also changes fitting.
No threshold reproduction is due because neither reaches 5%.

Both action histograms are `[71,2,0,0,0,2,0,0,0,0]`, but intervention timing
differs: both use action5 at0/3 and action1 at9; free Q uses action1 again at18,
cost Q at11. No quadratic/cross/combo candidate is selected. The gain is NOT
evidence that nonlinear coordination prices improved performance. Excess TTT
by bands0..6,7..15,16..18,19..26,27..40,41..74 is respectively:
free Q `[.091164,-3.205070,5.056152,65.155380,83.652822,2.485014]`;
cost Q `[.091164,-4.795616,7.877823,29.799561,-66.678913,-1.381741]`.
These bands are descriptive, not independent intervention effects.

Cost Q's initial predicted remainingTTT5822.860336 versus realized5586.001883
is much better calibrated than free Q4877.435684 versus5774.325068. Mean
remaining-TTT underprediction is44.565702 versus411.048287; frozen-policy TD
MAE.236690 versus.638305. Cost Q has0positive selected Q and1immediate-reward
bound violation, free Q9positive and10violations. Terminal cost Q-.162764
versus exacttarget-.211705 (error.048942, ensemble std.005271); free Q1.375390
versus-.212599 (error1.587988). Cost trajectory terminal same-step nearest
state RMS.134577. Both have0validity failures; distinctresponse min/mean/max
cost5/5.826667/7, free5/5.893333/7. Full inventories remain reported; do not
conceal cost Q's slightly higher terminal inventory than baseline.

### Next falsifiable experiment: targeted nonlinear response coverage

Read-only action/response analysis identifies a more specific support limit:
cost-policy action6 is a distinct executable representative only at steps1/6;
action9 only at7; actions7/8 never are representatives on this trajectory.
At step1, pure-quadratic action6 is physically different from every linear
candidate (response-feature RMS versus anchor32.895290), but exact-state
training behavior is only anchor (three rows in the414 batch). At step7,
combo action9 also differs physically from every linear candidate (anchor
RMS35.887787), with zero global behavior support. The existing support filter
therefore prevents choosing9 anywhere. Step6/action6 is already observed and
physically equals anchor, so it is not the proposed collection target.

New plan `work/response_nonlinear_coverage_v1.json`, root
`results/response_dqn_170_incident/sequential_nonlinear_coverage_v1`, runner
`work.run_response_nonlinear_coverage`. Construct exactly two branches from
verified cost-policy prefixes: step1 force6 once; step7 force9 once. Then
allow the SAME frozen cost-head policy to choose every later action, epsilon0,
until the real terminal. These are exploratory fixed-prefix continuations,
NOT acceptance evaluations. Their paired full-TTT differences describe the
first intervention and subsequent closed-loop recourse under this frozen
policy, not optimal action values. Retain all actual interval-negative-TTT
sequential transitions with successor actions and true terminals; do not
convert the branch returns into fixed labels or terminalize early decisions.

The prefix helper replays only the selected actions, not a candidate search,
and checks every prefix observation, state fingerprint, reward, full encoded
post-commit continuation identity and contract against the actual source
trace. It checkpoints each committed prefix interval and requested snapshot;
STOP and recovery reuse completed work. Any discrepancy fails closed with
diagnostics, before branch collection. First branch masks/features and exact
first action must also match the source after actual response previews.

Controlled training arms: refit-only uses existing414 plus both latest75-step
evaluations=564 rows; targeted uses those same564 plus74+68 branch rows=706.
Both use the finite-horizon cost head, same three seeds and6,000 updates,
gamma1, CQL.1, terminal fraction.25 and masks. This separates new targeted
coverage from simply incorporating the150 newly completed evaluation rows.
It does not isolate the effects of individual nonlinear coefficients.
Evaluate BOTH new policies from normal reset ungated, without calibration
selection or runtime fallback. Two4-worker branches, then sequential3-worker
trainers, then two4-worker full evaluations, never overlapping these stages.

No new run has launched when this entry was written. All prior results and
checkpoints are preserved; parent implements runner/plan, Locke owns the
matched-prefix helper/tests, Curie performs independent read-only review.
Follow the later launch/status note. The5% target and three-hour supervision
continue; do not stop at this0.61% improvement or blindly repeat training.

### Coverage preflight/review (20:35 KST)

Curie's independent read-only review found no blocker in the runner, plan or
matched-prefix helper. The first head departure is at identical model inputs
at step11: cost Q1-Q0=+1.74410, free Q1-Q0=-.02066. At free's step18, free
Q1-Q0=+.02829 while cost ranks anchor above1 by1.33238. This adds action-rank
evidence but is not proof that equal observations imply equal hidden states;
the new prefix helper checks full continuation identities for that reason.

Read-only preflight confirms564 rows,8true terminals, support
`[510,18,8,7,6,13,2,0,0,0]`, and the same80-interval value contract. Neither
targeted state-action pair occurs in this expanded baseline. All13 input
hashes match; all175 runtime/runner source pins are present; no STOP exists,
no new root exists yet, and disk has over126GiB free. The paired refit test
will include action9 support unlocking and sampling-distribution changes;
do not describe it as isolated proof of nonlinear-price superiority.

Parent regressions passed192 tests in149.608 seconds; the extended coverage
orchestration suite then passed6 tests in.042 seconds, including one additional
test of564-vs706 dataset routing, same cost-head recipe/seeds,2x4worker stage
allocation and evaluating both policies even with poor calibration. Prefix
helper unit tests are still the remaining launch prerequisite at this entry.
No experiment has started; wait for the later verified launch record.

### Matched-prefix tests passed; coverage prelaunch review

Locke completed the two assigned files only. Parent verification passed all16
prefix/coverage tests in1.302 seconds, bringing the distinct focused regression
coverage to203 tests (192-test run plus one added orchestration test and ten
prefix tests). Tests cover STOP during a commit, checkpoint/target-file crash
ordering, no repeated completed prefix steps, incompatible inputs and hidden
state rejection, and canonical full-identity comparison against a frozen real
fixture. Curie's independent static review found no blocking issue. These
tests do not replace the actual source-prefix parity check during startup.

Use the hidden locking launcher for the versioned coverage experiment:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_nonlinear_coverage_v1.json `
  -RunnerModule work.run_response_nonlinear_coverage
```

After launch keep all pinned runtime sources and this configuration unchanged.
Expected stages: exact matched-prefix capture through step7; two fixed-prefix
exploration tails with4previewworkers each; paired cost-head refits564/706;
two ungated full evaluations with4previewworkers each. Read any later launch,
failure or progress entry before acting. First inspect `prefix/failure_*.json`
if startup fails; never bypass state/continuation parity to force collection.
Do not relaunch any previous completed collection or training.

### Coverage experiment launched (20:35 KST); first prefix verified

Hidden wrapper15072 / child36608 started at
2026-09-08T20:35:39.6642212+09:00, module
`work.run_response_nonlinear_coverage`, log prefix
`runner_20260908_203539_632`. Plan SHA256:
`045094027cd7de9ad7ef565c08241f7997b63b784b5beabb7800348a98187a06`.
Process is running; status is `capturing_matched_prefix`. The actual prefix
through the first committed action passed observation, reward, state and full
continuation parity. `prefix/step_001.pkl` was saved at20:36:23 with the
corresponding resumable prefix checkpoint. No failure report or traceback
was present at20:36:51; stderr contains prior component calibration warnings.
Step7 and branch collection were still pending at that inspection. Keep this
configuration/source fixed; read later smoke notes before acting.

### Both matched prefixes verified; two exploratory tails running (20:43 KST)

All seven replayed prefix commits passed the actual raw observation, reward,
state-fingerprint and full post-commit continuation checks against the frozen
cost-policy trace. Both `prefix/step_001.pkl` and `prefix/step_007.pkl` exist.
No failure report or traceback was found. Status at20:43:21 is
`collecting_targeted_branches`,2episodes,4previewworkers per actor. This is
the planned two-tail collection, not another full-run acceptance result.

Follow `collection/quadratic_step01/progress.json` and
`collection/combo_step07/progress.json`, then their
`matched_branch_diagnostic.json` files. They should hold74 and68 actual
sequential rows respectively at true completion. Subsequent stages refit564
and706 rows and evaluate both policies ungated. Preserve the fixed prefix
snapshots and all completed branch rows on recovery; do not duplicate this
runner. First saved tail transitions were still pending at the above check.
The 0.612267% improvement is the previous cost-head full run only; 5% remains
unmet. Continue on the existing three-hour heartbeat after this stage changes.

### First targeted transitions saved

Both tails now have their first actual sequential transition/checkpoint:
quadratic branch through step1, cumulativeTTT158.80630906191806,
wall137.916336 seconds; combo branch through step7,
cumulativeTTT616.1538847495422, wall142.716539 seconds. Both are nonterminal.
These prefix-inclusive partial costs are not comparable policy-performance
results. Snapshot hashes: step001
`f78ce0f837f87f758de61161184caaa284fa5427c1c484273bb470c244b24424`, step007
`5481c177ed43838b425fd86d22c29d082efa2efdbcfd34f1e9284b717e7ed2bd`.
Wrapper15072/child36608 remain live with matching start times. Leave the
two4-worker tails running; retain checkpoints and continue the planned refits
and full evaluations. No new target achievement is claimed.

### Nonlinear coverage result reconciled (2026-09-08 23:13 KST)

`sequential_nonlinear_coverage_v1` finished at23:05:32, wrapper15072/child36608,
exit0. No surviving Python experiment, applicable STOP or traceback was found.
Independent read-only re-audit verified both full evaluations against checkpoint
greedy choices, raw rewards, steps0..74, true terminal, 14,400seconds, scenario
and unchanged acceptance contract. Both remain ungated and include all previews.

- Refit-only564: TTT5694.030775216803, **0.6414852139% better** than P-Stack;
  inventory431.24797575891125, wall3927.0792545seconds,4previewworkers.
- Targeted706: TTT5698.656123148699, **0.5607747789% better**;
  inventory429.4306890818408, wall4001.6555436seconds,4previewworkers.

Neither reaches5%; the small refit-only improvement over the previous0.612267%
is not a substantive solution. The added142 targeted transitions did not improve
the paired learned endpoint. Histograms: refit-only
`[70,2,0,0,1,2,0,0,0,0]`; targeted`[73,1,0,0,1,0,0,0,0,0]`.
Refit uses5at0/3,1at9/11,4at19. Targeted uses1at18 and4at30, otherwiseanchor.
Neither selected quadratic/cross/combo actions. Distinctresponse min/mean/max:
refit5/5.826667/7; targeted5/5.4/8. Both have0validity failures,0positive
selected Q and1immediate-reward bound violation at the final decision.

Refit/targeted frozen-policy TD MAE:.186796/.193998; mean remaining-TTT
underprediction29.514885/41.111438. Initial predicted/realized remainingTTT:
refit5889.451218/5584.327442; targeted6146.780523/5588.952772. Terminal Q errors
.036828/.058721, nearest same-step observation RMS.088184/.126859. Excess
TTT bands0..6,7..15,16..18,19..26,27..40,41..74:
refit`[.091164,-4.795616,7.877823,29.799561,-67.131044,-2.604050]`;
targeted`[0,0,.017166,-20.862061,-10.466579,-.825359]`. No individual-action
causal claim follows from these interval differences.

Both fixed-prefix branches were separately revalidated for exact initial
observation/response/mask, forced representative, prefix TTT, scenario/contract,
74or68 raw one-step rows and true terminal:

- Step1 forcequadratic6, then frozen cost-policy recourse: TTT5662.790913673756,
  **1.1866080570% better** than P-Stack,32.914282128813 lower than its matched
  sourcecost policy. Inventory420.35390368597643, suffixwall4543.7705239seconds.
- Step7 forcecombo9, then frozen cost-policy recourse: TTT5848.623136909051,
  **2.0560884490% worse** than P-Stack,152.917941106482 above matched source.
  Inventory432.85543713424187, suffixwall3691.3331692seconds.

These are exploratory fixed-prefix interventions with later learned-policy
recourse, not ungated learned results or optimal action-value labels. The
quadratic intervention found a useful path; the current trained policy does
not implement it. This does not establish nonlinear-price superiority generally.

### Value propagation and fitting diagnosis

The706-row targeted model loses the entry path: at the original source state0,
mean Q5-Q0=-.278978, while the564-row refit has+.200169. At the matched state1,
targeted Q6-Q0=-.349529. Quadraticfirst Q6=-60.966517; its ensemble-greedy TD
target is-61.018869, yet the recorded older-policy tail return is-55.323660.
This suggests inaccurate successor values/ranking rather than just a missing
first transition; that recorded return remains diagnostic, not an optimal or
fixed training target. Targeted greedy choices on those frozen branch states
are72anchors and2action4, unlike the older recourse with10action2 choices.

Exact model-input/action collision check on706 rows:651unique groups,
30repeated groups,0conflicting reward/next-observation/done outcomes. This
does not prove full Markov sufficiency or rule out hidden continuation effects.

Direct-Q loss derivatives under the actual terminal sampling weights provide
mixed evidence about CQL. At state0/action5, all three members' local CQL
derivatives favor raisingQ5, as do TD derivatives. Per-member target-minus-Q5
is[.730412,.384552,1.753162] on the five same-input action5rows. Therefore
"CQL directly suppresses entry action5" is refuted. At state1/action6,
CQL favors loweringQ6 for all three members; per-member TD residuals are
[.777851,.208687,-1.143593]. These independent-Q derivatives are NOT actual
Adam parameter-update attribution and do not guarantee a margin improves;
shared parameters and bootstrapped successor values can still mediate effects.

Training losses were still declining at6,000updates: first/last1000-update
means by seed are .206279/.095768, .227333/.086555, .231976/.100159.
Declining loss motivates more optimization, not a claim of convergence or
better TTT. Curie's independent review supports separating optimization depth
from CQL strength rather than merely repeating data collection or blamingCQL.

### Next controlled optimization-depth and CQL experiment

Plan `work/response_cost_refinement_v1.json`, root
`results/response_dqn_170_incident/sequential_cost_refinement_v1`, runner
`work.run_response_cost_refinement`. Reuse EXACTLY the existing706-row replay;
do not merge the150 newly completed evaluation rows or collect new episodes.
Reuse the completed targeted alpha.1/6k model and full result as short control.
Train paired-seed cost-head endpoints at24,000updates/member:
`long_cql_010` alpha.1 and `long_cql_001` alpha.01, three members each.
All other architecture, masks, sampling, optimizer, reward, terminal, catalog
and support settings remain unchanged. No alpha0arm or checkpoint selection.

Long.1 versus existing short.1 tests more optimization on the same objective;
long.01 versus long.1 tests regularization at equal updates. Equal updates do
not imply equal convergence. The long.1 run must exactly reproduce the stored
first6,000loss values, paired seeds, normalization, inputs and recipe (except
gradient count). This is a correctness check, NOT a performance gate or proof
of intermediate6k weight equality; no intermediate weight file is captured.
Record old/final path margins, bootstrap/recorded-return diagnostics and
1,000-update loss blocks. Evaluate BOTH predeclared24k endpoints ungated,
independent of calibration quality. No fixed-return labels or runtime guards.

Train each ensemble with3single-thread workers, sequentially; then two full
evaluations with4previewworkers each, total8. Source and input hashes, all
predecessor locks/STOP aliases and completed-checkpoint reuse are preserved.
Preflight at23:28:28:12input hashes match,176sourcepins cover all runtime,
noSTOP, no new output root. Plan SHA256:
`4eede4592ecc30bb24f41e52580811c0278a49e347698b813c9a723bb8645e61`.
Five focused runner tests pass; full regressions and independent final review
are in progress. No new experiment has launched at this entry. Follow the
later launch/status record; the5% acceptance and three-hour loop continue.

### 2026-09-09 restart: cost refinement ready to launch

The user's latest request explicitly resumes the work after the interruption.
At11:07KST the process inventory contained no RL Python process or experimental
wrapper. The newest completed root remains `sequential_nonlinear_coverage_v1`:
its process exited0 at23:05:32 onSep8; status is `ablation_complete`. No
`sequential_cost_refinement_v1` output root existed at11:11:29 onSep9.
Therefore resume the previously prepared refinement plan, not old collection.

Fresh preflight verified all12 pinned input hashes, all176 source files,
zero applicable STOP files, and117.89GiB free disk. The plan SHA256 remains
`4eede4592ecc30bb24f41e52580811c0278a49e347698b813c9a723bb8645e61`.
The previous unfinished test-session result and reviewer were unavailable
after restart, so neither was counted as passed. Fresh parent regressions
passed206 tests in150.784seconds using `.venv-torch/Scripts/python.exe -B -m
unittest` with all `src/tests/test_response*.py` and
`src/tests/test_sequential_response*.py` modules (OMP/MKL/OpenBLAS/NumExpr1).
Linnaeus independently reviewed the new runner, plan, tests and required
helpers: no actionable findings. Actual6k loss-prefix parity remains a
runtime prerequisite, not something established by static review.

Launch command (hidden, with existing exclusive predecessor locks):

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_cost_refinement_v1.json `
  -RunnerModule work.run_response_cost_refinement
```

Training is sequential across the two variants,3single-thread members each;
then both endpoints receive ungated full75-step evaluations,4previewworkers
each and8total. No new simulator collection, changed reward, physical
contract, guard or acceptance threshold. This entry records prelaunch only;
follow the later process identity and actual startup confirmation. Existing
heartbeat `ddqn` is ACTIVE at3-hour intervals; no duplicate or schedule edit.

### Cost refinement launched and training verified (Sep9 11:13KST)

Hidden wrapper13392 launched at11:12:38.439659; `process.json` records
child16268, start2026-09-09T11:12:39.1273458+09:00, module
`work.run_response_cost_refinement`, logs `runner_20260909_111239_105`.
The venv child delegates to runtimePython22072; these are one process tree,
not duplicate runners. The current stage is `training`, variant
`long_cql_010`,706transitions,24,000updates/member,0new collection episodes.
The plan,176implementation hashes and short-control diagnostics were saved.

Training subprocess23044/runtime20320 has exactly3worker children:
29072,30656,30716. All three had accumulated about25CPU-seconds at inspection,
with about289MiB working set each. The actual command has3workers,
1thread/member, CQL.1, gamma1, cost head, raw sequential TD,24kupdates and all
STOP aliases. Stderr was empty at startup. No completion or loss-prefix
agreement is claimed yet; do not launch another runner or change pinned code.

Next inspect this root's `process.json`, `status.json`, logs and
`long_cql_010/loss_prefix_verification.json`. Once both ensembles finish,
the existing runner evaluates both endpoints with4previewworkers each.
Any failure requires diagnosis and a documented resume, without discarding
completed checkpoints. A completed comparison still does not end the5% loop.
Best accepted-scope full result remains5694.030775216803 (0.641485% improvement,
not5%; not a separately reproduced threshold candidate). Three-hour heartbeat
supervision remains active and unchanged.

### Cost refinement complete: longer fitting did not improve full-run TTT

The Sep9 follow-up found the runner exited0 at12:29:56.8291144KST. Status is
`ablation_complete`, not a goal candidate. Both fresh independent re-audits
(`long_cql_*/heartbeat_reaudit_20260909.json`) passed baseline contract,
frozen-model greedy selection, trace/replay reward agreement and full75-step
normal-reset accounting through14400sec. Both use epsilon0, no forced first
action, no intervention gate or LCBguard, and4previewworkers each. No active
experiment or STOP was found. Previous finished collection was not repeated.

Results (original P-Stack5730.792964723197):

| Variant | Total TTT | Improvement | Terminal inventory | Full-run wall seconds |
| --- | ---: | ---: | ---: | ---: |
| 24k CQL.1 | 5729.759968814113 | 0.018025% | 431.519403496223 | 4246.874698 |
| 24k CQL.01 | 5755.017596738613 | -0.422710% | 431.341602717957 | 4373.291832 |

The .1 control EXACTLY matched all three stored first6000loss histories,
seeds, inputs, normalizers and recipes except totalupdatecount. This verifies
the intended optimization-depth comparison, not intermediate weight equality.
Neither24k endpoint beats the previous best full5694.030775216803 (0.641485%).
No threshold reproduction is warranted. All previews/anchor computations
remain in walltime; these are not preview-free policies.

CQL.1 action counts `[71,1,0,0,1,2,0,0,0,0]`: nonanchors at18/1,30/4,
32/5,44/5. CQL.01 counts `[57,2,6,3,2,4,1,0,0,0]`: nonanchors at0/5,
3/5,6/6,9/1,11/1,19/4,30/4,32/2,34/3,35/2,36/3,37/2,38/2,39/2,
40/2,41/3,42/5,44/5. The sole pure-quadratic choice is atstep6, not the
previous useful matched state1. Nominal nonlinear use does not establish a
new physical response or nonlinear-price causation.

The two policies' excess TTT versus P-Stack by intervals0-6,7-15,16-18,
19-26,27-40,41-74 is respectively:
`.1: [0,0,.017160,-20.862075,11.318761,8.493132]` and
`.01: [.091165,-4.795614,7.877817,29.799546,-24.534141,15.785872]`.
These trajectory differences do not causally attribute each band to an
individual action without matched-state alternatives.

Training one-step TD MAE falls to .098511/.083604, but fresh reached-state
frozen-policy TD MAE is .102008/.238338 (.1/.01). Remaining-TTT underprediction
is positive at75/75 states in BOTH, mean204.511649/277.602134. Initial predicted
remaining TTT5385.892995/5069.047165 versus realized5620.056601/5645.314264.
Terminal Q error is+.085651/+.043653 scaled; both violate the exact immediate
reward bound at the final step. Endpoint standardized nearest terminal-state
RMS .180293/.159726 and ensemble std .010104/.013998. Validity failures0,
positiveQ0; distinct continuation groups min/mean/max5/5.96/8 and5/5.706667/7.
Thus longer fitting and weaker CQL do not resolve reached-state calibration.
Underprediction against one frozen recourse policy is not proof that every
future policy incurs that loss, or that one-step DDQN is logically invalid.

### Next: controlled greedy-path backup experiment, no new collection

Plan `work/response_multistep_v1.json`, runner
`work.run_response_multistep_ablation`, root
`results/response_dqn_170_incident/sequential_multistep_v1`.
Hypothesis: short-horizon bootstrapping and missing reached-state transitions
contribute to the observed long-horizon miscalibration. This is a falsifiable
mechanism test, not a proven diagnosis or a promise of5% improvement.

Reuse706 existing rows plus the two just-completed75-step evaluations=856.
Read-only feasibility confirms856unique(episode,step) keys,12episodes,
844exact next-observation/response/mask links and12true terminals. No new
simulator collection and no return relabeling. Train two paired3-member
cost-head ensembles, seeds20260917-19,24k updates, CQL.01, all remaining
settings unchanged: `one_step` horizon1 versus `greedy_five_step` horizon<=5.
Both share the SAME856rows. Newone-step versus old706describes data change;
new5step versus new1step isolates backup rule at equal updates, not equal
compute or equal convergence. Both final endpoints must be evaluated ungated
regardless of fit. Report measured training and full evaluation runtime.

The optional capped greedy-path target uses actual recorded rewards only
while the recorded successor action equals CURRENT ONLINE member's
supported/masked greedy action. At the first disagreement, data boundary or
five-transition limit, use online argmax evaluated by the target network.
At a true terminal use zero successor value. Never join unrelated episodes,
ambiguous keys or inconsistent next-state chains. Recompute targets at every
update; no old policy's total return becomes an immutable action label.
Gamma1 and the original interval-negative-TTT/true-terminal replay stay intact.
Default horizon1 must preserve old training/inference behavior exactly.

This uses the greedy-action cutoff principle described by
[Sutton et al. (2014), Section1](https://proceedings.mlr.press/v32/sutton14.pdf),
not the paper's proposed PQ algorithm or a full lambda-return implementation.
Multi-step backup is a testable DQN design choice, not a universal cure;
[Hernandez-Garcia and Sutton (2019)](https://arxiv.org/abs/1901.07510) study
its sensitivity to backup length, correction and target-network timing.
No neural-network convergence guarantee is inferred. For the old .01members,
600-620 of856unweighted rows permit length5greedy-consistent paths at their
frozen endpoints, so the proposed cutoff is not invariably just1step.

Parent owns orchestration/plan/runner tests; Bernoulli owns the optional core
backup and its tests. Linnaeus reviews independently. At16:49KST plan input
hashes10/10match, STOPcount0, no output root; core helper remains in progress.
PlanSHA256 `373b1e5e616a089e7bf660abcf469c6df7421da4005ff894ecaa01f7d3007ad3`.
No experiment has launched at this entry. Require core tests, old-loss parity,
regressions and final review before launch. All178listed runtime files must
exist and be pinned at launch. Existing8-worker budget, predecessor locks,
STOP aliases and three-hour supervision remain unchanged.

### Multistep implementation and prelaunch verification complete (Sep9)

Bernoulli implemented the optional backup in `rl_leader/response_dqn.py`
(no extra helper module) and `rl_leader/train_response_dqn.py`, with23focused
tests passing. Parent added6orchestration tests and the controlled runner.
The final plan has177runtime source files; the obsolete prospective helper
pin was removed BEFORE launch. Final planSHA256:
`e045d687afa3122613772bbf7a431d5d55e15231275c757f76a6ce44eacdf875`.

Parent full regression run passed235tests in149.380seconds, using all
`test_response*.py` and `test_sequential_response*.py` modules under unittest.
Additional actual706-row checks reproduced EXACTLY the first8stored losses
for all6old checkpoints (3seeds times2CQL strengths), with horizon1. The two
old frozen ensembles still reproduce their entire75-step recorded greedy
action/trace-Q selections under the modified loader/inference code. This is
an8-update parity check, not another24000-update retraining or weight proof.
Actual856-row horizon5 cost-head training completed8updates with finite
losses and844verified links, without saving a candidate or collecting data.

Linnaeus' orchestration review found only the initially missing launcher
allowlist entry, which parent fixed. Final independent core review could not
finish because that reviewer became unavailable; do not count it as passed.
Parent reviewed the implemented core and focused tests directly: supported
ONLINE argmax, TARGET boundary evaluation, cutoff before nongreedy behavior,
zero true-terminal successor, retained nonterminal-gap bootstrap, immutable
raw rewards, strict episode-key/next-state checks and unchanged horizon1 math.
No blocking issue remains in that parent review. An independent final core
review is still a review limitation, not a reason to fabricate approval.

The runner remains unlaunched at this entry; read the later process record.
Launch through the existing hidden locking wrapper:

```powershell
./work/start_sequential_response_ddqn.ps1 `
  -Config work/response_multistep_v1.json `
  -RunnerModule work.run_response_multistep_ablation
```

Do not resume old completed collection. Keep the final source/config fixed
after launch. Train sequential ensembles with3single-thread workers, then
evaluate2actors with4previewworkers each. Dataset856 is frozen for both arms;
the next full-run results, not these unit/smoke tests, determine performance.

### Multistep experiment launched (Sep9 22:16KST)

Hidden wrapper45392 / child14332 started at
2026-09-09T22:16:29.1835227+09:00, module
`work.run_response_multistep_ablation`, log prefix
`runner_20260909_221629_119`. RuntimePython30104 belongs to the same venv
process tree. Actual training process47320/runtime48496 spawned exactly3
members14628,48824,45900; each accumulated about39CPU-seconds at inspection.
This is one experiment, not duplicate training runners.

Startup verified and saved856rows,12true terminals,844exact sequential links,
action counts `[762,24,24,10,11,20,4,0,0,1]`. Frozen merged replaySHA256:
`e3893d1ffd2491660a466b2821cc686d3fad5fb742c46a6e4b5844deb9733d21`.
All10input hashes matched,177source files were pinned, and noSTOP existed.
Current phase is `training`, variant`one_step`,horizon1,24k updates/member.
Stderr was empty. No model completion or new TTT is claimed at this point.

Keep current code/config fixed. Next inspect this root's process/status/logs,
then `one_step` and `greedy_five_step` model manifests, training-call timings,
fit diagnostics and both full evaluation summaries/calibration audits.
The runner automatically proceeds from training to two full2x4evaluations.
On completion reconcile full TTT, action/response identities, reached states,
Q calibration, band losses, nonlinear effect and terminal inventory/runtime.
If still above5444.2533164870365, continue with a falsifiable diagnostic and
next versioned experiment; do not repeat completed collection. Best full
result remains5694.030775216803 (0.641485% improvement). Existing three-hour
heartbeat and5% independently-reproduced acceptance rule remain active.

## Continuing authorization and 5% stopping rule

On 2026-09-07 the user explicitly requested continued iterations until at least
5% TTT improvement. Keep the running two-round pilot unchanged. Its completion
is an intermediate checkpoint, not the end of the authorized experiment loop.
The existing three-hour thread heartbeat supervises subsequent experiments without
creating a duplicate runner or a second scheduled task.

Acceptance requires a frozen learned policy evaluated from the normal reset
through all 75 control steps and 14,400 simulation seconds in
`sweet_170_incident_w60`, with the unchanged baseline experiment contract
`95694fc1e5bb06da621e784a7e4d4bad56135b6d75ac360bb3a42b2d18501831`.
P-Stack TTT is 5730.792964723197; accepted RL TTT must be no greater than
5444.2533164870365. Preserve baseline demand, incident, warmup, follower
feasibility, TTT accounting and physical constraints. Do not lower the target
or change evaluation conditions to obtain a nominal gain.

Use epsilon=0 without an intervention-step gate, forced first action, or LCB
fallback guard. Native P-Stack coordination remains a selectable anchor in
this response-aware formulation; its computation and all response previews
must remain included in runtime reporting. This is not preview-free inference.
Report nonlinear action usage and terminal inventory alongside TTT. Do not
claim nonlinear pricing caused an improvement without an appropriate ablation.

After each full result, inspect reached-state coverage, action/response aliasing,
Q calibration, policy action counts and interval losses. If the threshold is
not met, select a concrete failure hypothesis, run a small diagnostic or focused
collection batch, test any code fixes, train a versioned model, and evaluate
again. Preserve previous checkpoints/results; avoid repeating an unchanged
failed configuration or extending collection solely because more time passed.
Keep the existing eight-worker compute budget and prevent overlapping runners.

Once a candidate meets the threshold, freeze its checkpoint, record hashes and
configuration, and rerun the same full evaluation to verify reproducibility
before declaring this scenario's goal achieved. A repeated deterministic run
does not establish held-out-scenario generalization. Then stop launching new
experiments and report the evidence. Until that point, continue the measured
loop unless the user pauses/stops it or a genuine external blocker requires
user input. Honor STOP files and retain resumable checkpoints throughout.

## References

- van Hasselt, Guez and Silver, [Deep Reinforcement Learning with Double
  Q-learning](https://arxiv.org/abs/1509.06461).
- Farama, [Handling Time Limits](https://gymnasium.farama.org/main/tutorials/handling_time_limits/):
  distinguish a genuine terminal state from a collection-time truncation when
  deciding whether to retain the successor value.
