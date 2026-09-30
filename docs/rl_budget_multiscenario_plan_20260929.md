# Balanced five-scenario shared action-value pilot

## Goal and interpretation

The user explicitly confirmed equal training on155,170,170-incident,170-skew
and190, rather than five seeds of170-incident. Learn one shared state-conditioned
action-value model and one shared actor. TD3 retains its two shared critics for
clipped-double-Q estimation; this is not five independently trained policies,
an average of five checkpoint weights, or one state-independent scalar value.

Scenario IDs, in fixed order:
`sweet_155_w`, `sweet_170_w`, `sweet_170_incident_w`, `sweet_170_skew15_w`,
`sweet_190_w`. Use the existing frozen protocol for each ID.

## Global constraints

- Preserve all prior code, checkpoints, data and paused legacy jobs. New code:
  `work/sdmpc_rl_multi_20260929`; new evidence/results:
  `results/sdmpc_rl_multi_20260929`. No commits, pushes or new automations requested.
- Reuse the frozen151-file physical snapshot and unchanged carried-budget
  controller, observations and follower constraints. Initial/recovery PFO remains;
  no routine per-step PFO. Guard is physical/budget validity, not H3 performance.
  A prior-control fallback is not a guaranteed congestion-recovery policy.
- Each full episode: original reset, five warmup intervals,75controls,14400s.
  Actual sequential reward=-intervalTTT/100, gamma1, true terminal only. Training
  demand perturbation is explicitly tagged; canonical evaluation has no exploration,
  perturbation, intervention gate, forced actions or unreported performance guard.
- One actor and two shared TD3 critics. Equal per-scenario replay capacity and
  exactly20% of every training minibatch per scenario. Batch40 means8each.
  Do not update before every scenario has sufficient data. Balance collection
  counts as well as gradient sampling; never favor a quicker-finishing worker.
- Collector workers do not train private learners. One frozen policy per collection
  round; all five complete real sequential trajectories before central balanced
  updates. The next round uses the updated common model. This is iterative
  simulator-assisted off-policy TD3, not counterfactual-label supervision.
- Two bounded rounds, one full perturbed episode per scenario per round:
  10episodes/750transitions total. Seeds6301..6305 then6401..6405 in scenario order.
  Initial actor zero;32uniform random actions per scenario in round0, then Gaussian
  exploration std0.3; round1 uses the shared frozen actor plus std0.3 noise.
- Use one central TD3 update per newly admitted transition (375per complete round),
  not the failed pilot's20updates/transition. This conservative pilot change is
  explicit: balanced data and update frequency are not a single-factor ablation.
  Critic/actor diagnostics must not imply causal attribution or improvement.
- At most five one-core numerical collectors/evaluators at once, and one CPU-thread
  learner after collection, within the user's total eight-worker budget. Pin BLAS
  and Torch threads. Do not run duplicate children or modify active source/config.
- Preserve source/runtime/profile/model hashes and exact resume contracts. STOP
  files are authoritative. Reuse completed matching child outputs; never silently
  repeat completed collection. No evaluation transitions enter training replay.
- Compare the same final frozen model across all five canonical scenarios against
  zero-action carry baselines. Report per-scenario TTT, inventory, fallbacks,
  saturation, zero-ramp duration, interval losses and complete decision runtime.
  An unweighted mean of relative improvements is descriptive; it cannot hide a
  catastrophic individual scenario. No legacy DDQN5%claim or generalization claim.

## Tasks

1. Verify scenario protocols/config/schema compatibility; write shared contract,
   manifest and balanced learner with deterministic checkpoint/sampling tests.
2. Implement resumable frozen-policy collectors/evaluators, centralized round
   training and a finite five-worker orchestrator. Keep physical modules unchanged.
3. Test sample balance, no cross-episode bootstrap, replay provenance, source/model
   and scenario mismatches, partial resume/STOP, policy ownership and canonical
   evaluation separation. Independently review learner and integrated lifecycle.
4. Run a small actual multi-scenario smoke including serialized restore. Admit full
   runs only with valid executions, common observation schema and equal sampling.
   Bind gate to current source, tests, smoke and review evidence.
5. Execute the two balanced rounds, required canonical carry baselines, and five
   final shared-policy full runs. Reuse validated existing170-incident center if
   equivalence can be established; otherwise record why a fresh benchmark is needed.
6. Reconcile all scenarios and150transitions/scenario in replay, checkpoint hash,
   gradient sample shares, Q diagnostics, action/control response and measured cost.
   Document whether shared learning succeeded as an implementation and whether
   performance improved as a separate empirical question. Stop this bounded pilot
   after reporting; do not blindly repeat failed settings or restart legacy loops.

## Admission notes

The preceding single-scenario carry pilot failed with nearconstant[-1,-1] actor
outputs and37final intervals of zero ramp commands. Balanced training is a
falsifiable coverage intervention, not a guarantee that action integration,
critic extrapolation or a physically valid but ineffective fallback is solved.
Do not add an undisclosed ramp lower bound or change TTT to make this pilot pass.

## Progress (chronological)

- Goal created and five-scenario interpretation confirmed by user.
- Prior carry pilot fully stopped; no active experiment configuration is edited.
- Implementation and admission are pending. No new simulator worker launched.
- Five protocol/config contracts verified with the frozen runtime; initial-state
  hashes agree, demand-profile hashes remain scenario-specific. The four physical
  carry modules are byte-identical to the preceding version.
- Balanced learner implemented and strict restore review fixes in progress;
  fix-round1 focused211tests passed. Integrated lifecycle tests/code review pending.
- Canonical evaluation rows now log Q1/Q2 for the actual requested action before
  the environment step. This permits comparison with realized returns under the
  same frozen deterministic policy. Inference timing includes this read-only Q
  diagnostic; no evaluation transition is admitted to replay. Exploration-return
  comparisons remain explicitly different from current-policy calibration.
- Scenario IDs label replay strata, not network inputs. State/forecast/remaining
  time remain the existing observation contract; hidden future conditions and
  PFO memory mean full Markov sufficiency remains unproven. Equal sample counts
  do not imply equal gradient magnitudes when raw costs differ.
- All five centers will be re-evaluated as a timing/comparison wave under the
  same new runner contract (including model/scenario provenance and value-audit
  logging); the prior170-incident center remains an independent preserved
  reproducibility reference, not newly collected training data. No old collection
  is rerun on resume; completed matching workers are validated and skipped.
- Admission implementation is complete:486synthetic tests passed in52.69s;
  independent learner and integrated lifecycle code reviews passed. All five
  actual two-interval serialized smokes passed with2367common features, valid
  physical/budget execution and PFO counts1initial/0carried. Smoke processes
  exited0; these are not full-run performance evidence. Final source/evidence
  attestation is pending before the one finite pilot may start.
- Final independent attestation and machine admission PASS. The finite pilot is
  now running in results/sdmpc_rl_multi_20260929/pilot_v1; parentPID18160 and
  exec session21637. Source/configuration is frozen for its complete execution.
  Results and shared-learning/performance conclusions remain pending.
- Tasks1-6 COMPLETE: all22jobs exited successfully, balanced750transition replay
  and750updates verified, five canonical comparisons and read-only diagnostics
  reconciled. Outcome: shared balanced learning works mechanically, but this
  trained policy is rejected for severe TTT losses in allfive scenarios.
  See docs/rl_budget_multiscenario_results_20260929.md for evidence, actor/critic
  mismatch and saturation diagnosis, runtime, limitations and focused next tests.
  No pilot worker remains; no subsequent collection or automation was launched.
