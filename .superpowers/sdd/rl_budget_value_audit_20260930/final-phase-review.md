# Final Value-Audit Integration Review

Date: 2026-09-30

Integration verdict: PASS for the completed value-audit phase.

No actionable findings. No priority/path/line defect entries are warranted.
This verdict does not admit a policy, establish traffic improvement, complete
the all-five shared-model goal, or approve an unimplemented collector.

## Scope and Verification

Reviewed `progress.md`, `final-phase-review.diff`, the terminal-fit brief in
`work/sdmpc_rl_value_audit_20260930/terminal-brief.md`, the ledger's
`projection-brief.md`, and `work/sdmpc_rl_value_audit_20260930/terminal_quota_brief.md`.
Used the three independent task reviews for their implementation assessments;
this review concentrated on shared contracts, actual results, provenance, and
the conclusions in `docs/rl_budget_value_audit_results_20260930.md`.

All eight files in the supplied diff exactly match their current contents by
line: three scripts, three test files, the results report, and the next-stage
plan. The unrelated dirty branch was excluded. Diff SHA-256:
`4cad4544567b7e80dc3553368db00520509a099af68243025aee8c4d554b87e1`.

Accepted the recorded focused results of 46, 48, and 64 synthetic tests and the
parent's newly recorded combined regression: **158 PASS in 35.44 s, exec96004,
exit 0**. These are supplied execution results, not tests run by this reviewer.
The parent reports no remaining numerical process. No model loading, inference,
numerical experiment, test rerun, simulation, implementation edit, commit, or
model override was performed. Only this review report was written.

## Cross-Component Evidence

- **Shared immutable base:** all three diagnostics bind the preserved final
  model hash `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`.
  Read-only hashing matched all 73 projection inputs, all 151 physical snapshot
  files and their manifest, all 12 pinned implementation files, the diagnostic
  source manifests, and all four recorded runtime entrypoints. Terminal-fit and
  quota before/after identity objects match. Quota binds the exact projection
  completion, its input manifest, and its source/runtime contract.
- **Compatible replay and initialization:** projection completion covers all
  five scenarios in both rounds, 750 transitions and ten true terminals.
  Terminal fit and quota record the same original learner fingerprint. Every
  quota arm/seed's initial ten terminal rows, targets, and predictions exactly
  match terminal fit's original-terminal record, including min-Q MAE
  5.701837813854217. The terminal and continuing-TD diagnostics therefore refer
  to the same stored boundary conditions.
- **Actual bounded execution:** both terminal folds and arms record 1000
  updates. Quota has both required seeds and every predeclared log point;
  each arm has 3750 critic updates, 1875 target updates, zero actor updates,
  and 30000 draws per scenario. Paired initial-state/noise hashes and frozen
  actor/target/optimizer fingerprints agree. Original learner/model and global
  RNG preservation are recorded. The three output directories contain JSON
  records and lock files, with no exported learned weights.
- **Completion and source identity:** all three completions say `completed`.
  Their hashes and executed script hashes match the results document. Quota's
  settings and metrics hashes also match; projection's aliases hash matches its
  completion. Quota intentionally retains `status.json: verifying`; the report
  correctly treats completion as authoritative, not as a reason to rerun.

## Conclusions Against Results

References below use repository-relative paths and one-based line numbers.

- `docs/rl_budget_value_audit_results_20260930.md:40`: all four fit table rows
  match the saved metrics. Only fresh round 0 meets both critics' <=0.01 train
  maximum-error criterion. The report correctly limits the capacity inference,
  discloses prior exposure of the original critics to both rounds, and does not
  turn the five-point holdout into a generalization claim.
- `docs/rl_budget_value_audit_results_20260930.md:89`: the saved alias rows
  reproduce 59 states, 177 groups, and the reported twin-Q mean/max spreads.
  Controller and environment source support the same-request equivalence:
  nominal action is projected before physical execution, and subsequent memory
  uses requested/executed budgets. The report explicitly excludes a measured
  counterfactual follower rollout, optimal ranking, or dominant-cause claim.
- `docs/rl_budget_value_audit_results_20260930.md:117`: all five training TTT,
  zero-request, terminal-inventory, and mean-action readbacks agree with the
  saved summaries/traces. Their evaluation flags are false, with distinct
  perturbed training seeds across rounds. The first-round settings confirm
  32 initial uniform steps and noise standard deviation 0.3. The report does
  not compare these unmatched training profiles to canonical acceptance scores.
- `docs/rl_budget_value_audit_results_20260930.md:191`: final uniform/quota
  MAEs are 41.579805/4.500399 and 41.308354/2.893314. All three predeclared
  criteria pass. The pooled reductions are 91.079892% versus uniform and
  35.163770% versus initialization, matching the report. The worse seed-6529
  maximum error, nonmonotonic quota curves, and all four early/middle/late
  residual rows are retained accurately. The reported 127.5-130.2 behavior gap
  range corresponds to min-Q. Own-target TD residuals and behavior-return
  discrepancies are correctly distinguished from error against true policy Q.
- `docs/rl_budget_local_collection_plan_20260930.md:3`: the local collection
  stage is explicitly **UNIMPLEMENTED**, with no result attributed to it. This
  review checks that boundary and its alignment with the goal, not its future
  implementation. `docs/rl_budget_balanced_goal_20260930.md:29` still requires
  one frozen shared model, 20% sampling per scenario, improvement in every one
  of the five canonical full runs, and reproduction with that same candidate.
  The results report keeps the goal active and makes no TTT-improvement claim.

## Residual Risks and Evidence Limits

- Ten stored terminals, two sampling seeds, and one frozen saturated actor
  cannot establish broad calibration, representation sufficiency, policy
  quality, or generalization. The quota result supports the bounded sampling
  hypothesis; it remains insufficient for deployment or goal completion.
- The supplemental geometry numbers at
  `docs/rl_budget_value_audit_results_20260930.md:60` are not present in a
  standalone saved geometry artifact in the reviewed output directories.
  Their exact distance/rank/SVD values were not independently revalidated
  under the no-model-loading/no-experiment constraint. No contradiction was
  found, but those figures are outside this review's verified result set and
  are not needed for the three principal diagnostic conclusions.
- Runtime hashing covers the recorded entrypoints, not every dependency
  binary. In-memory immutability relies on the completed runs' fingerprints
  and the independently reviewed guards; this review checked their consistency
  and current file identities without reconstructing the numerical execution.
- The future collection and learner revisions still require their own
  implementation, tests, review, and measured results. Nothing in this review
  substitutes for the all-five full-run acceptance and reproduction gates.
