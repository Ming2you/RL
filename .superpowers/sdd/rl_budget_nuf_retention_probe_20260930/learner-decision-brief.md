# Conditional learner decision sidecar

Read-only technical recommendation, no implementation or numerical training.
The current four collection workers must finish before learner admission.
Own only learner-decision-report.md in this directory; no other writes.

Read docs/rl_budget_next_learner_notes_20260930.md and the completed value audit
results, original work/sdmpc_rl_multi_20260929/td3.py, and the new frozen
work/sdmpc_rl_nuf_retention_20260930/exploration.py. Inspect further local
interfaces only where needed. Do not repeat diagnostics or launch environments.

We have one shared 2367-observation twin-Q learner with equal20% scenario samples
and real finite-horizon transitions, reward=-intervalTTT/100, gamma1, true terminal.
Original TD3 after750updates saturated actor; terminal-quota helped terminal error
but not earlier calibration. Distinct nominal actions can map to identical NUF
capped budgets. New collection keeps initialNUF explorationbase, NP still rebases
on physical fallback. Same-seed170 probe restored healthy trajectory; remaining4
are active. Conditional successful data comprises5carry+5newlocal trajectories,
150transitions/scenario, no canonical evaluation data.

Recommend ONE minimal next learner experiment, bounded/offline-first then later
on-policy continuation, with an explicit falsifiable calibration/admission gate.
Focus: how to avoid huge finite-horizon baseline value error driving unsupported
actor extrapolation with only two trajectories/scenario. Compare plain fresh
terminal-quota projected-budget TD3 vs a fixed state-only value control variate
Phi with residual TD Q=Phi+A. Phi may be fit to behavior MC as numerical initializer,
never ground truth for another continuation or Q*. Do not prescribe elaborate
framework changes, arbitrary long collection, or nearest-time labels as optimal.
Budget-level zero actor + explicit base/rebase memory is a candidate, not imposed.
Specify exact actor/request/critic input semantics and what available evidence
can and cannot independently validate; identify one decisive limitation.

Report <=120lines with local code references, mathematical target, concrete small
experiment/gates, and suggested tests. No full five-scenario scoring, implementation,
dependencies, commit/push, or edits to frozen data/source. External theory if used
must be checked via primary sources, otherwise keep conclusions local/conditional.
