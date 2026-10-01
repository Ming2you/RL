# Balanced shared RL: value audit results

Status: active goal NOT achieved. These are training-data diagnostics, not new
traffic evaluations or a replacement deployable policy. See
`rl_budget_balanced_goal_20260930.md` for the five-scenario success contract.

## Evidence identity

- Base: `results/sdmpc_rl_multi_20260929/pilot_v1/train_round1/model_final.pt`,
  SHA-256 `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`.
- Replay: 750 sequential transitions, two complete 75-step exploratory episodes
  per scenario, ten true terminals. No evaluation observations were consumed.
- Diagnostic code: `work/sdmpc_rl_value_audit_20260930`.
- Output: `results/sdmpc_rl_balanced_goal_20260930/value_audit_v1`.
- Prior source, model, replay and results preserved. No policy weights exported.
- Terminal fit: 46 synthetic tests passed in 4.66 s; independent specification
  and quality review passed before execution.
- Projection audit: initial review identified four defects. Fixes added input
  manifests, collection provenance binding, lock-held STOP/output checks and
  float32 alias deduplication. 48 tests passed in 22.78 s, then independent
  scoped re-review passed. Production run completed afterward.

## Terminal-only fit

Command (repo root, local CPU runtime):

```powershell
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_value_audit_20260930/terminal_fit.py --output results/sdmpc_rl_balanced_goal_20260930/value_audit_v1/terminal_fit
```

PID 44240, exit 0. Started 2026-09-30 02:21:09.594 KST, ended
02:21:37.369 KST, approximately 27.775 s. One numerical thread. Each fit used
one terminal from each scenario per update, 1000 updates, equal 20% sampling.
The other collection round was a diagnostic holdout, then the folds reversed.
Targets were the actual immediate rewards `-interval_TTT/100`, with no bootstrap.
Both critics trained; the original actor, optimizer and original learner stayed
unchanged. Two arms used copied original critics or fresh critics; both used
fresh Adam. These diagnostic copies were not exported.

| Train round | Critics | Train min-Q MAE | Holdout min-Q MAE | Train Q1 max error | Train Q2 max error |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Original | 0.007843 | 1.089069 | 0.018877 | 0.007366 |
| 0 | Fresh | 0.000402 | 1.565513 | 0.000657 | 0.000940 |
| 1 | Original | 0.025830 | 1.043060 | 0.055979 | 0.067630 |
| 1 | Fresh | 0.157259 | 3.250172 | 0.362521 | 0.403691 |

Original min-Q MAE across all ten terminals was 5.701837813854217 reward units.
Only the fresh round-0 fit met the descriptive maximum-error <=0.01 criterion
for BOTH critics. Do not report that every fit passed that criterion.

Train errors fell substantially without bootstrapping. This establishes that
the present architecture can fit at least one five-terminal fold accurately;
it does not establish that capacity/optimization is adequate everywhere. The
round-1 residual and diagnostic holdout errors remain material. The original
critics had already seen both rounds during TD3 training, so their holdout is
only held out of these EXTRA supervised updates. Five holdout points cannot
establish generalization. Changing targets, sample distribution and Adam state
together prevents assigning the original failure solely to bootstrapping.

A read-only geometry probe found ten distinct terminal inputs (minimum pairwise
L2 distance 1.0082534083450432). Each five-row augmented input matrix had rank
five; smallest singular values were 1.2820072720207085 and 0.5735807086447285.
Exact duplicate inputs with conflicting terminal targets do not explain these
particular fitting residuals. This does not prove Markov sufficiency.
These supplemental figures came from a read-only console probe; unlike the
three saved diagnostics, they were not independently revalidated in the final
integration review and are not used by the next-stage admission criteria.

Completion confirms input/source/runtime, original learner/model and global RNG
immutability. Completion SHA-256:
`b7baba1b26cff5d7a460c4ecc6ff0d561c1d1ede80cf7ea213113582395142d2`.
Executed source SHA-256:
`07251732bd66b27c5315f580ab4de1f79bdd4f1943a1ec5c33bdd5e821443ba2`.

## Replay and projection consistency

```powershell
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_value_audit_20260930/projection_audit.py --output results/sdmpc_rl_balanced_goal_20260930/value_audit_v1/projection
```

PID 39672, exit 0, elapsed 10.1953 s. All 750 saved transitions matched their
authenticated collection traces exactly at the stored dtype. Checks covered:

- Terminal flags only at each episode's final control interval, with zero
  terminal next observation and exact nonterminal observation chaining.
- Control indices, simulation clock, remaining horizon, previous requested
  and executed budgets, action anchors and projected requests.
- Reward equal to negative actual interval TTT divided by 100.
- Collection order and provenance matching the frozen final model, physical
  configuration/options/source/runtime contract and 73 input files unchanged.

The same saved training state can admit several nominal positive NUF actions
that clip to exactly the same request. Found 59 such states and 177 groups
(three NP settings per state). All 177 groups had twin-Q spread above 1e-6.

| Quantity | Q1 | Q2 |
| --- | ---: | ---: |
| Mean within-alias spread | 0.002508418 | 0.006914283 |
| Maximum within-alias spread | 0.011645973 | 0.021507502 |

Each group contains distinct float32 action inputs whose physical budget
transform is identical. This violates action equivalence in the learned value
representation. The measured spread is much smaller than the observed terminal
calibration error, so this evidence does NOT establish the dominant cause of
the policy's poor TTT. No counterfactual follower rollout was executed; equality
is justified by the physical controller consuming the projected request, not
the otherwise unused nominal action. Different projected requests can still
produce identical follower decisions; that larger quotient was not measured.

Completion SHA-256:
`b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3`.
Executed source SHA-256:
`ab3787b94c8bbc4a1e76a9d04b942159a01c6753b292b6d6f5d9c21a256b86db`.

## Next single-factor experiment

The saved exploratory collection also has a severe reached-state imbalance.
Summary/trace readback (training profiles, NOT canonical evaluation scores):

| Scenario | Round-0 TTT | Round-1 TTT | Round-1 NUF request zero /75 | Round-1 terminal inventory |
| --- | ---: | ---: | ---: | ---: |
| 155 | 7016.750 | 30904.752 | 69 | 15724.198 |
| 170 | 16104.856 | 32593.523 | 68 | 16442.593 |
| 170 incident | 15512.497 | 32863.935 | 69 | 16491.151 |
| 170 skew | 12206.697 | 32744.191 | 69 | 16458.341 |
| 190 | 27230.374 | 36676.693 | 69 | 18153.082 |

Every round uses authenticated but perturbed training demand profiles. These
TTTs must not be divided by canonical center scores to claim matched policy
degradation. They do show that uniform SCENARIO counts alone do not provide
balanced coverage of useful and failed traffic states. A zero requested NUF
is not asserted to equal zero actual metering without a separate control audit.
Round-1 mean nominal NUF actions range -0.815 to -0.921. Collecting more under
that same saturated actor would extend a predominantly failed region rather
than supply missing near-center state/action comparisons.

There is also a collection-design issue worth isolating next. Actions are
budget INCREMENTS from the previous executed budget, not absolute budget
levels. Without clipping/recovery, zero-mean independent action noise gives
`B_t = B_0 + sum(scale * action_i)`, a budget random walk rather than local
exploration around the carry center. Upper clipping further makes positive
and negative NUF disturbances asymmetric near 6000. The original first-round
collector used 32 uniform [-1,1] steps followed by Gaussian action noise with
standard deviation .3. Its first 32 unbounded NUF increments alone have a
nominal standard deviation `1000 * sqrt(32/3)`, approximately 3266 budget units.
Actual clipping/recovery changes that distribution; this calculation is not
a prediction of TTT. It explains why centered nominal noise is not evidence
of near-center state coverage.

A future small training-only wave should compare local budget-level or
mean-reverting exploration with the existing incremental random walk, under
matched perturbed demand. Preserve the external previous-executed-budget
action interface and physical constraints. Log requested AND executed offsets,
guard rebasing, cumulative drift, terminal inventory and all intervals. Do not
reuse canonical evaluation states as training data. Do not treat an observed
return under a chosen continuation as an unconditional optimal-action label.

Hypothesis: sparse terminal exposure contributes to boundary-value drift under
the continuing TD objective. Terminal rows are only 10/750 (1.333%) of replay.
Compare original uniform replay with a quota arm that replaces one of eight
draws per scenario by a terminal draw, retaining the other seven unchanged.
All five scenarios still supply exactly 20% of every batch. Keep checkpoint,
optimizer state, reward, gamma, actor, target-update cadence and smoothing noise
matched. Two fixed seeds, 3750 updates per arm, one CPU thread, no policy export.

Predeclared support criterion: quota final min-Q terminal MAE below uniform in
both seeds, pooled MAE at least 50% lower than uniform, and below the initial
5.701837813854217 in both seeds. This is NOT policy admission. Behavior returns
provide descriptive trajectory discrepancies, not fixed-policy ground truth.
Do not deploy the old saturated actor even if this diagnostic passes.

Implementation requirements: `work/sdmpc_rl_value_audit_20260930/terminal_quota_brief.md`.
Tests and independent review must precede production execution. No expensive
new traffic batch has been selected at this stage; choose the next bounded
learning/collection change from the diagnostic result, keeping the already
authorized overall five-scenario goal active. New collection must address
near-center sequential state coverage, rather than reuse the saturated actor.

## Terminal-quota ablation: completed

The predeclared experiment above subsequently passed 64 synthetic tests in
16.38 s and independent specification/quality review. Executed once:

```powershell
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_value_audit_20260930/terminal_quota.py --output results/sdmpc_rl_balanced_goal_20260930/value_audit_v1/terminal_quota_v1
```

Console redirected to the sibling `terminal_quota_v1.console.log`. PID 31648,
start 2026-09-30 02:53:12.752 KST, end 02:58:13.891 KST, exit 0, elapsed
301.1386 s. One CPU numerical worker, four sequential arm/seed fits, no new
plant transitions, actor updates or weights export. Each arm/seed completed
3750 critic and 1875 target updates, drawing 30000 samples per scenario.

| Seed | Uniform terminal min-Q MAE | Quota terminal min-Q MAE | Quota maximum terminal error |
| --- | ---: | ---: | ---: |
| 6529 | 41.579805 | 4.500399 | 10.770208 |
| 6530 | 41.308354 | 2.893314 | 7.668707 |
| Pooled mean | 41.444080 | 3.696857 | Not pooled |

All three predeclared criteria passed. The pooled error is 91.08% lower than
uniform after the same update budget, but only 35.16% lower than the INITIAL
5.701838. Neither percentage is a traffic/TTT improvement. Original model,
learner, global RNG, input/source/runtime hashes, and frozen arm actor states
remained unchanged. Actual terminal draws were 370-427 per scenario under
uniform, versus 4077-4131 under quota, with the same 30000 total draws each.

This supports insufficient boundary exposure as a contributing cause under
this replay and fixed actor. It does not make the critic calibrated enough to
deploy. For example, quota seed6529 still predicts -18.594698 for the round-1
155 terminal whose exact reward is -7.824490. Its maximum terminal error is
worse than the original maximum 8.640768, despite better mean error. Curves are
nonmonotonic; no checkpoint was chosen by inspecting them.

Mean TD residual loss across equal-sized scenario/time groups at final update:

| Seed/arm | Early 0-24 | Middle 25-49 | Late 50-74 |
| --- | ---: | ---: | ---: |
| 6529 uniform | 1.2593 | 1.2651 | 180.2742 |
| 6529 quota | 6.4859 | 9.0614 | 30.9332 |
| 6530 uniform | 1.5446 | 1.5359 | 175.3442 |
| 6530 quota | 5.2893 | 5.7028 | 37.0923 |

Boundary emphasis trades higher early/middle residual for lower late residual.
These are each arm's OWN learned-target residuals, not errors against a common
true value function. The observed-behavior return gap remains large (mean
127.5-130.2 across all rows, in reward units), but behavior continuation differs
from the fixed target actor, so this cannot be interpreted as a Q-error label.

Source SHA-256:
`837c237e91872bd7fed53929dcbbf536aa60dbe5f8ce54ccb4404bfe54c35c3e`.
Completion SHA-256:
`05207dd42b86f284321396449272728b921367c6fce66d83c9041afbd7fa738b`.
Settings SHA-256:
`f90fa669d54cc0c49ab9cb88b19cd40f06048f413f33ece996a4ca82972ffbf7`.
Metrics SHA-256:
`6c6d61ca508e64feb5cc9b147fda0efd75a6b9d5396c476c76d8d691622fd8e5`.

Use `completion.json` as authoritative. The diagnostic intentionally leaves
`status.json` at `verifying` after success so completion is its final write.
The process has exited; do not mistake that status for an active worker or rerun
this completed experiment. Preserve all curves and both seeds, not only averages.

Next: implement/test the bounded training-data collection in
`rl_budget_local_collection_plan_20260930.md`. Keep terminal quota as a supported
candidate component, not a complete remedy. No new frozen policy qualifies for
five-scenario acceptance yet; the app goal and three-hour heartbeat remain active.

## Phase closure, not goal completion

Combined regression: all 158 tests passed in 35.44 s, exit 0. Independent final
integration review found no actionable issue in the three diagnostics, their
cross-component identities, saved results and principal conclusions. See
`.superpowers/sdd/rl_budget_value_audit_20260930/final-phase-review.md` for scope
and residual limitations. This does not admit the unimplemented next collector.
All diagnostic/test processes and subagents are finished. No commit or push
was made; existing staged and unrelated changes remain intact. The next entry
point is the local collection plan, not another run of these completed scripts.
