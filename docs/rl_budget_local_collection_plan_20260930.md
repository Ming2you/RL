# Next bounded stage: near-center training coverage

Status: v2 COMPLETE750transitions; no worker running. Paired screenFAILED170
terminalinventory. Read rl_budget_local_collection_results_20260930.md and
the SDD rebase-diagnosis-brief.md before further work. Do not resume/recollect,
remigrate, or modify frozen v1/v2/companion sources. This is the
next authorized stage of `rl_budget_balanced_goal_20260930.md`, not a new goal.
Read `rl_budget_value_audit_results_20260930.md` first. Do not repeat its finished
diagnostics or export their diagnostic critic copies.

Current recovery entry point is
`.superpowers/sdd/rl_budget_local_collection_plan_20260930/progress.md` and its
`windows-recovery-brief.md`. Active replacement source/root have suffix `_v2`
and `local_budget_v2`. Two existing intervals were explicitly verified and
migrated, not recollected. The collection total is still750, not752. See
`rl_budget_local_collection_status_20260930.md` for process evidence and resume
instructions. Do not rerun migration after ordinary collection advances.
The current recovery task is `export-recovery-brief.md` in the same SDD directory;
its companion code is `work/sdmpc_rl_export_recovery_20260930` (completed).

## Falsifiable hypothesis

Current balanced replay is balanced by scenario, but its second round has zero
NUF requests in 68-69 of 75 intervals per scenario. Incremental action noise
can cause a budget random walk. A small budget-level mean-reverting exploration
wave around a training carry-center can add local state/action coverage with
less cumulative drift, without changing the physical follower or reward.

The terminal-quota ablation supports increasing boundary exposure, but still
has large errors and does not justify deploying the saturated actor. Address
the missing state distribution next, rather than perform another arbitrary
critic-update sweep or long collection under that actor.

## Fixed collection scope

- New versioned implementation directory `work/sdmpc_rl_local_20260930` and
  result root `results/sdmpc_rl_balanced_goal_20260930/local_budget_v1`.
- Freeze the existing physical snapshot and previous-executed-budget interface,
  action scales [50,1000], reward -interval_TTT/100, gamma=1, five warmup and
  75 actual sequential intervals to 14400 seconds, physical feasibility guard,
  initial/recovery PFO, and all demand/accounting contracts.
- Two TRAINING trajectories per scenario: zero-action carry behavior and a
  local exploratory behavior, paired on the same perturbed training profile.
  Use training demand seeds 6801 through 6805 in the established scenario order;
  verify they do not identify an already completed collection under this new
  contract. Never use canonical evaluation profiles or observations for fitting.
- Use distinct behavior/run identities even though paired demand seeds match.
  The old trainer's (scenario,seed) uniqueness assumption cannot be reused
  blindly: preserve explicit scenario, demand seed, behavior, run ID, profile,
  source/runtime and experience hashes. Do not weaken old validators in place.
- No online learning during these trajectories. They are collection runs, not
  evaluations of an improved RL policy. Ten complete runs at most, 750 new
  transitions total, 150 per scenario. At most five one-thread collectors at a
  time (below the total eight-worker budget); run the carry wave then the local
  wave, with durable per-interval checkpoints and no duplicate work.

## Local exploration rule

This rule is a data-collection policy, not a performance fallback or evaluator:

1. Initialize a local base budget from the environment's actual first action
   anchor, not from saved canonical evaluation data. Local offset starts zero.
2. Draw two independent standard-normal values from a local RNG with seeds
   6901 through 6905 by scenario. Target offset is
   `clip(0.8 * previous_realized_offset + [10,100] * noise,
   [-50,-500], [50,500])`, in physical NP/NUF budget units.
3. Add offset to the base and apply the unchanged NUF capacity projection.
   Convert that desired budget to the existing incremental action using the
   current ACTUAL previous-executed-budget anchor and [50,1000] scales, then
   clip the action to [-1,1]. The environment remains the sole physical executor.
4. After execution, record actual offset `executed_budget - base`. On an
   explicitly logged physical fallback/reference recovery, rebase to that
   actual executed budget and reset offset zero. Resolve exact existing trace
   flag names in implementation; do not use a TTT comparison to trigger rebase.
5. Keep requested/realized offsets, projection, action-rate limiting, actual
   execution and rebase cause in the trace. A clipped positive NUF request at
   capacity must not accumulate fictitious positive exploration state.

Tests must prove the rule from actual anchors, zero-noise mean reversion, both
capacity boundaries, action-rate limits, guard rebasing, deterministic resume,
and absence of extra PFO calls, candidate previews or evaluation-state access.
Independent review must precede production collection. Existing completed
code/configurations remain immutable; reuse read-only helpers where possible.

## Readout before actor training

Authenticate complete paired profiles/trajectories, reward, terminal flags,
inventory, source/runtime and STOP/process identity as in the previous audits.
Compare each exploratory run to its SAME-PROFILE training carry run, not the
canonical center numbers. Report TTT, interval losses, inventories, requested
and actual budget drift, zero NUF requests, physical fallbacks, response/control
diversity, and full measured runtime. Do not label a continuation's observed
return as unconditional action truth or call the training comparison acceptance.

Predeclared collection-design screen (not a success target): in every scenario,
exploration should have no more than five additional zero-NUF-request intervals
and terminal inventory no greater than 1.25 times its paired training center
(with denominator floor one). Report the actual numbers even if the screen
fails. If it fails, retain the data and diagnose the rule before collecting more;
do not train the actor blindly on another collapsed batch. If it passes, this
only supports near-center coverage. It says nothing about improved optimal Q.

After this screen, implement the smallest justified shared learner revision:
preserve exactly 20% scenario sampling; incorporate terminal exposure with
full error reporting; enforce budget-projection consistency in critic inputs;
address initialization/actor extrapolation using the new sequential data.
Specify and test that learner before training or exporting it. Evaluate one
frozen admitted candidate over all five CANONICAL full runs, then return to the
main goal's diagnosis/reproduction loop. Do not silently convert this into five
specialized policies or an average-only target. Goal remains active throughout.
