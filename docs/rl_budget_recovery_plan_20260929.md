# Balanced-policy recovery experiment

## Scope

Continue the user's five-scenario shared-policy work after the completed balanced
pilot. Preserve all previous data, checkpoints, source and admission artifacts.
Do not restart legacy DDQN loops or automations. Use at most eight numerical
workers, with one CPU thread per new job. No active related process was found.

The prior policy is rejected. It closes all ramps for 69 final control intervals
in every scenario. Q1 now locally favors increasing both actions on all replay
states, but the actor remains near [+1,-1]. Tanh attenuation is measured; its
causal contribution and critic temporal propagation still need controlled tests.

## Tasks and decision rules

1. Use only the existing 750 training transitions and frozen final critics.
   Compare 375 balanced actor updates under three conditions: original actor and
   optimizer; original actor with fresh optimizer; output head reset to zero with
   fresh optimizer. Keep hidden weights, critic, samples, learning rate and update
   count fixed. Each batch has eight states per scenario. Save metrics and separate
   actor-only exports, not counterfeit resumable TD3 training checkpoints.
2. Independently probe critic-only temporal propagation on the same replay with
   fixed actor: existing target rate versus hard target synchronization, holding
   other choices constant. Report terminal errors separately from exploratory
   reward-to-go discrepancies. This is a diagnostic, not a trained policy or a
   proof of correct off-policy values. No evaluation trajectory enters fitting.
3. Review the focused code/tests and diagnostic evidence. If an actor treatment
   reduces mismatch with its frozen Q1 without nonfinite behavior, select it using
   pre-evaluation metrics only: highest equal-scenario Q1 objective after the fixed
   budget, with initial deterministic variant ordering as tie breaker. Otherwise
   retain the failure evidence and formulate the next specific test.
4. Evaluate the selected common actor as a clearly named actor-repair ablation on
   the five canonical full runs. Retain original critic estimates for calibration;
   their continuation-policy interpretation changes after actor repair. Use
   unchanged physical modules, original reset/75 controls/14400s, no exploration,
   no performance guard or silent ramp floor. Reuse validated carry-center results.
   Bind new evaluator and export identities, save per-step resumable checkpoints,
   respect STOP and do not duplicate a completed scenario.
5. Reconcile TTT, control coverage, inventory, measured runtime and common model
   identity across all five scenarios. Record whether actor fitting was repaired
   separately from whether traffic improved. Do not claim P-Stack improvement,
   nonlinear-price learning, out-of-sample generalization, or reliable Q ranking
   merely because an offline loss or predicted value improved.

New code: work/sdmpc_rl_recovery_20260929. New results:
results/sdmpc_rl_recovery_20260929. Prior implementation is imported read-only.
No new 24-hour collection, commits or pushes are part of this bounded experiment.

## Completion

Completed 2026-09-30. All five actor_fit_v2 full evaluations and source-bound
reconciliation passed. Actor fitting recovered the frozen critic's direction,
but all five TTTs remain 4.73-11.87% worse than the matched carry center. The
controller is not accepted. Original and diagnostic checkpoints are preserved;
no further worker, collection or scheduler was launched. Results and the next
bounded diagnostic plan are in `rl_budget_recovery_results_20260930.md`.
