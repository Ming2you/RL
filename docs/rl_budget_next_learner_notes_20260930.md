# Conditional learner design notes

Status: design notes only, not an admitted learner, exported model, or new run.
The current authorized collection and its screen remain specified in
`rl_budget_local_collection_plan_20260930.md`. Finish that small balanced wave
and inspect the actual results before choosing or implementing this revision.

## Evidence to carry forward

- The completed terminal-quota audit improved terminal calibration relative to
  matched uniform TD, but did not establish accurate earlier values or improve
  traffic TTT. Do not deploy its diagnostic critics or reuse the saturated actor.
- The original incremental action representation allows different nominal
  actions to request exactly the same projected budget. A revised critic should
  represent the projected requested budget consistently, while preserving the
  nominal action in provenance. Executed budget is an outcome, not a substitute
  action input that is available before the follower runs.
- Balanced scenario counts alone did not prevent missing useful states. The
  planned mean-reverting collection tests local coverage without spending an
  arbitrary additional day under a collapsed policy.

## Candidate revision, subject to the collection readout

Consider a fresh shared critic and a zero-initialized budget-level actor within
the collected local range. Convert its desired budget through the unchanged
previous-executed-budget anchor, rate limits, and capacity projection. Carry any
policy base/rebase memory explicitly in the learner observation and checkpoint;
do not hide it from the value model. This is a candidate for testing, not proof
that absolute targets are better than incremental actions.

Exactly zero incremental action holds the current executed budget. A zero
budget-level offset instead requests the stored base budget. These are equal
on the unperturbed carry trajectory, but need not be equal after intervention.
Specify the continuation policy explicitly before defining its value targets.

For a carry training trajectory, the observed remaining sum of interval rewards
is a sample return under that trajectory's carry continuation. It may initialize
or diagnose the corresponding value, but is not Q-star and does not label nearby
actions. Local exploratory trajectories have a different continuation. Preserve
their real sequential transitions and use explicitly defined TD targets rather
than declaring their realized returns to be the value of a new target policy.
For all bootstrapping, gamma stays one and only true termination masks the
continuation. Time/remaining horizon stays in the observation.

The mathematical distinction is standard: Q-pi conditions on an initial action
and subsequent policy pi; finite-horizon value depends on remaining time. See
[OpenAI Spinning Up's RL introduction](https://raw.githubusercontent.com/openai/spinningup/master/docs/spinningup/rl_intro.rst).
Twin critics and delayed actor updates can reduce approximation-driven policy
errors, but are not guarantees of safe improvement on this offline batch. See
[Fujimoto et al., ICML 2018](https://proceedings.mlr.press/v80/fujimoto18a.html).

## Admission before another full evaluation

1. Authenticate the collection and report its predeclared coverage screen,
   including negative results. No canonical evaluation traces enter training.
2. Freeze a small learner specification with equal 20 percent scenario sampling,
   terminal exposure, projection-equivalent critic inputs, reproducible RNG,
   explicit actor initialization/constraints, and no performance fallback.
3. Test the representation and terminal targets; inspect errors across the
   horizon before admitting actor updates. Own-target TD residual alone is not
   an independent calibration measurement. Observed behavior returns are not
   target-policy ground truth off that behavior's continuation.
4. Export one frozen candidate and run the same checkpoint on all five canonical
   environments. Count physical fallbacks, initial/recovery PFO and every solve
   in runtime. Do not call better training loss an improved traffic policy.
5. Apply the existing per-scenario improvement and separate reproduction rule
   from `rl_budget_balanced_goal_20260930.md`; otherwise diagnose and iterate.

No new hyperparameters or model family are approved by these notes. In
particular, this does not authorize automatic training on a failed coverage
screen or revive the paused legacy DDQN/P-Stack experiment.
