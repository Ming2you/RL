# RL 5% P-Stack Improvement Target Status

> Historical status. Current ungated best is 5694.030775216803 (+0.641485%),
> not the gated result below. Work and scheduling are USER PAUSED.
> See [2026-09-10 handoff](RL_HANDOFF_20260910.md) for current evidence and next steps.

Updated: 2026-09-03

## Goal

For `sweet_170_incident_w60`, achieve at least 5% lower 14400 s total TTT than
the P-Stack baseline, or establish a reproducible blocker under the current
response-mediated marginal-price residual framework.

Baseline:

- P-Stack total TTT: `5730.792965 veh-h`
- 5% target total TTT: `5444.253316 veh-h`
- Required gain: `286.539648 veh-h`

## Current Best

The best valid full-run result remains:

- Artifact: `results/rl_phase0_implementation_20260902/cached_tail_170_incident_h12_steps21_24_seed03_v1/summary.json`
- Forced residual steps: `21,24`
- Total TTT: `5692.178526 veh-h`
- Gain over P-Stack: `38.614439 veh-h`
- Percent improvement: `0.673806%`

This beats P-Stack, but it is far from the 5% target. The remaining gap to the
target is about `247.925210 veh-h`.

## Experiments Run Toward The 5% Target

### 1. Step 21 H12 residual expansion

Artifact:
`results/rl_phase0_implementation_20260902/cached_sampler_step21_seed2026090306_h120_h12x40_mag1_reuseh1_v1/summary.json`

- H1 candidates: `120`
- H12 candidates: `40`
- H12-positive candidates: `12`
- Best H12 gain: `24.041369 veh-h`

This did not beat the previous step-21 H12 best:

- Previous best artifact:
  `results/rl_phase0_implementation_20260902/cached_sampler_step21_seed2026090302_h80_h12x16_reuseh1_v1/summary.json`
- Previous best H12 gain: `24.757120 veh-h`
- Previous best residual:
  `freeway.R_F_W.g_meter=0.75`, `urban.B.g_green=-0.25`,
  `vsl.FW_W__seg4.g_vsl=-0.25`

Conclusion: increasing the sampled residual magnitude to `1.0` and adding a
targeted step-21 neighborhood did not unlock a larger long-horizon response.

### 2. Step 21 best residual full-tail replay

Artifact:
`results/rl_phase0_implementation_20260902/cached_step21_upper_bound_sample_best_v2.json`

- Candidate: `random:seed2026098221_draw3:23327fda39be`
- Total TTT: `5693.030988 veh-h`
- Gain over P-Stack: `37.761977 veh-h`
- Percent improvement: `0.658931%`

This reproduces the previous cached-tail result and confirms that the best
single step-21 residual is real, but it is still not near 5%.

### 3. Step 21 direct P-CENT one-step upper-bound check

Artifact:
`results/rl_phase0_implementation_20260902/cached_step21_upper_bound_direct_pcent_v2.json`

- Intervention: direct centralized P-CENT physical control for one step,
  followed by native P-Stack to the end of the simulation
- Total TTT: `5745.226837 veh-h`
- Gap versus P-Stack: `+14.433872 veh-h`
- Percent change: `+0.251865%`

Conclusion: at the same step-21 state, directly applying one P-CENT physical
action does not expose a hidden large gain. It worsens total TTT.

### 4. Multi-step residual combinations already tested

Representative full-run results:

- `21 + 24`: `5692.178526 veh-h`, `0.673806%` improvement
- `21 only`: `5693.030988 veh-h`, `0.658931%` improvement
- `21 + 22`: `5711.021835 veh-h`, `0.344998%` improvement
- `21 + 22 + 24`: `5707.690211 veh-h`, `0.403134%` improvement
- late step `26` H3-positive candidate: `5735.455168 veh-h`, `0.081354%` worse

Conclusion: adding adjacent or late short-horizon-positive interventions often
creates delayed losses. The best result is mostly a single step-21 response,
with a tiny additional contribution from step 24.

## DDQN Status

The best saved DDQN exact-policy full-run is:

- Artifact family: `results/response_dqn_170_incident/*_eval_v1/summary.json`
- Total TTT: `5727.850138 veh-h`
- Gain over P-Stack: `2.942827 veh-h`
- Percent improvement: `0.051351%`
- Non-anchor actions: `1 / 75`

Important caveat: the current response-aware DDQN evaluation still previews
candidate follower responses online before selecting an action. The neural
network forward pass is cheap, but the current evaluation mode is dominated by
follower MPC response preview cost. A deployment-speed claim requires replacing
or caching that preview stage with a learned response/mask predictor.

## Root-Cause Assessment

The 5% failure is unlikely to be explained by training time alone.

Evidence:

- The best discovered executable residual response improves only about
  `0.67%`, while the target requires `5%`.
- Larger residual sampling and targeted nonlinear neighborhoods did not improve
  the best H12 label.
- Short-horizon-positive steps frequently become long-horizon losses.
- Direct one-step P-CENT physical control at the most useful state worsened the
  full-run TTT.
- The DDQN catalog is currently narrower than the sampler space: the structured
  DDQN catalog restricts magnitudes to `<= 0.5`, while the best discovered
  residual uses `0.75`. This explains why DDQN can underperform the hand-picked
  residual, but not the much larger 5% gap.

Current blocker:

The existing leader action is mediated by follower MPCs. Many nominal price
vectors collapse to the same executable follower response, and most executable
responses around the congested period either match P-Stack, create small gains,
or create delayed recovery losses. Under this current marginal-price residual
framework, the discovered reachable response set does not contain enough
independent long-horizon-positive actions to support a 5% gain.

## Recommended Next Direction

Stop brute-force residual/DDQN retraining under the current catalog as the main
path. Continue only as diagnostic support.

Priority changes:

1. Move the deployable policy from nominal price-vector selection to executable
   response selection.
2. Train or approximate a follower response/mask predictor so DDQN inference
   does not require online candidate MPC preview.
3. Expand the action contract to include the known best residual scale
   (`0.75`) and state-local response templates before any further DDQN training.
4. Add an early-pruning evaluator for full-tail branches:
   abort a branch once it can no longer beat the current best or 5% target.
5. If 5% remains the hard target, change the control authority:
   allow the leader to constrain or regularize follower objectives more directly,
   not only send marginal-price residuals that followers may project away.

Practical next experiment:

- Build a response-level selector that treats each unique realized follower
  response as the action.
- Use P-Stack as the fallback anchor.
- Seed the response catalog with the known step-21/step-24 winners.
- Evaluate whether response-level control can produce more than the current
  `0.67%` without changing follower feasibility constraints.
