# Local collection failed-screen diagnosis

Read this first. Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL.
Current task is read-only analysis, not implementation or new simulation.
Read docs/rl_budget_local_collection_results_20260930.md for exact observations,
authority hashes and limitations. Goal is balanced single shared value/policy,
all5canonical TTTs strictly improved; currently no qualifying policy.

750actual training transitions in local_budget_v2 are complete; predeclared
screen fails only170 (inventory7.1652timescarry; localTTT11945.9393 vs4602.4115).
Allworkersandboundedexportcompanionexited. Do not restart completed collection.
Oldsource/results remain immutable, including taggeddiagnosticInf and receipts.

1. Verify completed paired trace/summary/comparison hashes read-only. Use
   standard parsers; no torch/checkpoint boot, source execution, processchanges,
   testsuites, installs, commits or canonical evaluation data fitting.
2. Produce a compact source-backed decomposition for all5: requested/achieved/
   executed budget, original vs moving base displacement, rebaseevents,
   rate/offsetclipping, actual ramp closures/control discontinuities, time
   distribution of paired intervalTTTdelta and terminalinventory. Explain what
   is demonstrated and what requires causal experiment. In170, near-zero
   D_ramps happen14 BEFORE fallback15 resets NUFbase2840.8. Do not attribute
   all loss to rebase without accounting for this order.
3. Inspect only relevant immutable policy/controller/env/validator source.
   Recommend the smallest falsifiable next probe: test retaining initial NUF
   exploration center (not NP, which is signed/state dependent) while preserving
   actual-anchor conversion, physical fallback, rates, gamma/reward/horizon and
   unchanged RNG draws. Assess whether alternative conclusions are supported.
   No request to implement or run this candidate; no learner admission yet.
4. Specify an exact small follow-on protocol, acceptance/falsification and what
   data can be reused. No arbitrary24hcollection, no5independentpolicies, no
   treating behavior continuation as optimal-actionlabels. Consider one focused
   diagnostic first versus fivebalancednewlocal-onlyruns with frozenexisting
   carryreferences. Keep totalnumericalworkerbudget<=8. Avoid redesigning the
   physical follower or broad infrastructure for this analysis.

Write full report <=150lines at
.superpowers/sdd/rl_budget_local_collection_plan_20260930/rebase-diagnosis-report.md.
Optional bounded computed evidence JSON may be written in the same SDD folder;
report exactcommands/provenance. Use apply_patch for prose/code ifneeded.
Do not modify any existing artifact. Return concise status/keyfinding/limits.
