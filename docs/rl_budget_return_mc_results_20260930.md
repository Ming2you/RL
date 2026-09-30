# Observed-continuation MC fit

Completed once on 2026-09-30. This is critic-only TRAINING fit, not an improved
physical policy, heldout calibration, or a canonical traffic evaluation.

## Execution

- Source: `work/sdmpc_rl_return_mc_20260930`, now frozen.
- Output: `results/sdmpc_rl_balanced_goal_20260930/return_mc_v1`.
- Reviewed by Bacon: SPEC PASS / QUALITY APPROVED, no findings. Retained scoped
  suite: 46 passed, 27.637 seconds; 745 pre-existing artifacts preserved.
- Hidden launch at 12:25:45 KST, launcher43944, actual37492,
  creationFILETIME134352123452018260, sessiona32527de3b144742ba95f1658adf7449.
  Parent command/log/receipt in the current SDD directory, `mc-parent-20260930_122545.json`.
- Actual job completed250MC updates. Process drain verified with CIM, no stderr.
  Locked-session elapsed44.9066693s excludes preceding authentication/load and
  interpreter/parent overhead; not traffic-control runtime.
- Model SHA256 `820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6`.
- Completion SHA256 `a815f65f315ce1a73cce8a3754b9edb3fac9dbed6a8fd0aece9b9238ea532675`.
- Metrics SHA256 `7588ad61950cff3a28077557fddce4a69fb7d8d1590fb6ae334290755b9ff7a0`.
- All completion-bound output hashes verified after exit. Actor and Phi hashes
  unchanged. Final critics and targets both
  `ef38c8267b13bdc967528deec7d128ea01d6e5b7834638ec573cfbb3a9196e2b`.

## Measured Fit

Exactly375 actual pi1 transitions,75/scenario, seeds7301..7305. Each minibatch40
contains8/scenario, including1forcedterminal. Total2000samples/scenario,250forced
terminals/scenario; all sampled terminal counts266/270/285/275/277. Actor/Phi
bit-exact, critic Adam continued from step250 to500; gamma1, actual finite-horizon
returns, reward minus intervalTTT/100, warmup excluded. No TD bootstrap.

| Scenario | Q1 before MSE | Q1 after MSE | After terminal abs error |
| --- | ---: | ---: | ---: |
| 155 | 86.208715 | 53.476632 | 1.642196 |
| 170 | 114.582409 | 62.628597 | 3.050896 |
| 170 incident | 262.074901 | 199.869116 | 2.023139 |
| 170 skew15 | 104.084032 | 56.821008 | 2.563472 |
| 190 | 539.924837 | 451.760150 | 8.723227 |

Pooled Q1 MSE221.374979->164.911100; MAE10.868688->8.291093. Terminal Q1
MAE8.474237->3.600586; Q2 terminal MAE8.468608->1.885908. Before values compare
carry-continuation Q with new pi1 returns, hence continuation discrepancy rather
than independently calibrated pi1 errors. After values are in-sample errors.
For190, first25decisions (remaininghorizon51..75) Q1 MAE34.053080, positive bias;
250MC steps do not establish accurate long-horizon Q. No extra epochs admitted.

All375 observed NUF requests6000; cap clips every positive actor increment. This
wave supplies no NUF action variation and cannot identify a NUF action gradient.
NP requests range-114.854440..21.636619 across differing states/anchors; this
alone does not identify causal NP rankings. No nonlinear-price result is implied.

## Next Decision

Do not take another actor-gradient step from this fit yet. The existing small
shared actor has passed real five-profile trajectory health checks but has never
been measured on the matched canonical five-scenario acceptance profiles. Admit
Task4: one canonical full-run evaluation of the same frozen actor/model on all5,
using unchanged physics/physical guard and initial/recovery PFO, no exploration,
learning, Q selection, or performance guard. Reuse completed canonical carry
centers; do not recollect them. This directly tests whether the actual policy,
not its loss, improves any/all required cases before another proposal.

Only after exact source/adapter tests and independent review launch the new
versioned evaluator. The active goal remains unmet. A passing all5candidate must
be separately reproduced; a failed one triggers a falsifiable next diagnosis,
not identical reruns or arbitrary24hourcollection.
