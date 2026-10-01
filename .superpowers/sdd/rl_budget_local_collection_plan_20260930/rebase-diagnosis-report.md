# Bounded Rebase Diagnosis (2026-09-30)

Decision: propose one real 170 NUF-center-retention rollout first; do not implement or launch it in this diagnosis. Reuse a successful unchanged candidate run as the 170 member of five balanced local-only outputs. No model training until all five are complete and admission is separately satisfied.
The completed collection remains a failed collection-design screen, with no qualifying shared policy. Trace arithmetic proves moving-center drift; it does not identify how much TTT loss that drift caused.

## Authority and Scope

Repository: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`. Data root: `results/sdmpc_rl_balanced_goal_20260930/local_budget_v2`.
Root completion SHA256: `5b6e645f6f6ca64c4f593c52d19ee3ab5bda7ae3c3d31f423d6fae2d2f3ef4a3`.
Comparison SHA256: `c294c1f5b2100daeaccb4c927357a8bc73d44391f34641fd37b4ba5b2555c4cd`.
PASS: 77 hash comparisons: root completion/comparison, 10 child completions, all 50 child manifest outputs, eight relevant implementation files, frozen manifest, and six relevant frozen sources. Binary experience files were hashed only, never deserialized.
PASS: 750 JSON intervals reconcile against summaries/comparison, including paired profile hashes/seeds/warmup, carried anchors, TTT sums and area totals, rewards, terminal flags, physical validity and no-learning flags. All ten terminal inventories independently reconcile with frozen accounting arithmetic.
Eight export receipts match child output hashes and retain checkpoint-authoritative=true, convergence-claim=false, physical-or-training-change=false. Diagnostic Inf tags and all prior artifacts remain untouched. This is not a rerun of the original checkpoint validator.
Parent-confirmed supervisor completion and zero Python workers are accepted; no process changes or restart. No simulator, model/checkpoint load, source-module execution, test suite, install, commit, canonical fitting, or framework implementation.
Exact successful PowerShell computation commands, per-file hashes, unrounded statistics and source provenance are in `rebase-diagnosis-evidence.json` (`commands`, `hashes`, `extra.source_hashes`). Commands use `Get-Content -Raw | ConvertFrom-Json`, `Get-FileHash -Algorithm SHA256`, and arithmetic only; working directory is the repository above.

## Budget and Drift Decomposition

All steps below are zero-based control steps, 75 per run; five 180s warmup intervals precede control at 900s; final time is 14400s. C=carry, L=local. Carry is previous-executed-budget zero action, not P-Stack or a standalone PFO leader.
Q=`B_requested[0]`, G=`G_achieved`, X=`B_executed`. G is the follower execution-check budget coordinate; do not equate it with realized plant throughput. Plant release diagnostics were inspected separately. NP is signed/state dependent (veh); NUF is an upper constraint (veh/h), `G <= B`, not a required flow.
Means over all 75 intervals, in Q / G / X order; full ranges and per-coordinate displacement are retained in evidence.

| Scenario | Mode | NP mean Q / G / X | NUF mean Q / G / X |
| --- | --- | ---: | ---: |
| 155 | C | -36.958 / -163.096 / -35.733 | 5646.886 / 5592.432 / 5640.798 |
| 155 | L | -4.779 / -161.623 / -2.155 | 5691.712 / 5506.066 / 5690.150 |
| 170 | C | -48.813 / -176.775 / -47.758 | 5999.998 / 5847.307 / 5999.998 |
| 170 | L | 35.483 / -127.136 / 38.810 | 3746.902 / 3651.578 / 3713.328 |
| 170-incident | C | -18.245 / -168.976 / -16.736 | 5999.898 / 5777.582 / 5999.896 |
| 170-incident | L | -39.535 / -167.243 / -37.575 | 5375.633 / 5113.867 / 5365.151 |
| 170-skew | C | -44.925 / -166.772 / -43.883 | 5999.873 / 5685.562 / 5999.871 |
| 170-skew | L | -26.830 / -167.168 / -25.475 | 5626.926 / 5509.762 / 5628.272 |
| 190 | C | -22.996 / -173.863 / -21.579 | 5999.888 / 5587.533 / 5999.886 |
| 190 | L | -47.252 / -177.818 / -45.317 | 5787.349 / 5405.445 / 5781.383 |

Local displacement pairs below are NP / NUF. All initial NUF centers are 6000. Q and X happen to share the same maximum absolute displacement from the initial anchor in these five local traces; this does not imply Q=X per interval.

| Scenario | Initial NP | max abs(Q-initial) | max abs(Q-moving base) | max abs(X-moving base) | max abs(base-initial) | Final base NP / NUF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 155 | -112.069 | 170.973 / 653.606 | 39.368 / 242.160 | 61.887 / 242.160 | 131.606 / 411.446 | 19.537 / 5588.554 |
| 170 | -114.814 | 246.997 / 3349.333 | 41.338 / 500.000 | 84.833 / 2520.354 | 208.451 / 3159.199 | 93.637 / 3249.984 |
| 170-incident | -110.259 | 116.928 / 1062.065 | 38.461 / 466.290 | 52.984 / 467.994 | 101.842 / 914.579 | -8.417 / 5085.421 |
| 170-skew | -110.183 | 127.116 / 769.611 | 35.524 / 443.692 | 93.412 / 443.692 | 102.329 / 328.517 | -7.854 / 5674.080 |
| 190 | -110.148 | 109.220 / 536.415 | 35.626 / 368.272 | 69.792 / 368.272 | 83.371 / 168.143 | -26.777 / 5831.857 |

Fallback/rebase events below include unchanged-coordinate resets; NP values for every event are in evidence. There were no PFO recoveries in any run. Local step -> NUF base-after:
- 155: 1->5999.97, 3->5999.97, 6->5937.44, 9->5937.44, 10->5937.44, 11->5937.44, 14->5759.24, 16->5588.55.
- 170: 0->6000.00, 1->6000.00, 9->5758.25, 12->5361.15, 15->2840.80, 16->2840.80, 26->3249.98.
- 170-incident: 0->6000.00, 1->6000.00, 11->5819.43, 13->5553.41, 14->5553.41, 16->5085.42.
- 170-skew: 3->5914.25, 4->5914.25, 9->5671.48, 15->5674.08.
- 190: 0->6000.00, 1->6000.00, 4->5958.00, 7->5877.16, 9->5877.16, 10->5877.16, 14->5831.86.
Carry fallback steps: 155 {1,14,16}; 170 {1,14}; incident {1,3,14,15}; skew {1,14}; 190 {1,4,14}. Their maximum initial NUF drift is respectively 456.585, 0.002885, 0.104905, 0.158388, 0.115159.
Rate clipping: zero NP and zero NUF events across all ten runs. Local offset clipping: only 170 NUF at steps 72,73; all other coordinates/runs zero. Local desired NUF projection counts (155/170/incident/skew/190) are 7/4/5/1/4; final request projection counts 4/0/0/0/1. No zero NUF requests. Offset bounds apply to the moving center, not the initial center.

## Physical Response and Timing

Only local 170 closes ramps: both D metering settings <=1e-6 at steps 14-16; actual release is ~8.46e-16/1.11e-15 veh per interval, then resumes at 17. All other local/carry ramps have positive interval releases. This is a 540s closure starting at 3420s, not a zero-budget request.
Largest adjacent applied ramp-setting jump (veh/h) and green-time jump (s), with step; these are descriptive discontinuities, not physical-gate violations:

| Scenario | Max ramp C / L (step) | Max green C / L (step) |
| --- | ---: | ---: |
| 155 | 231.143 (15) / 357.527 (40) | 24.600 (18) / 25.841 (17) |
| 170 | 720.000 (42) / 1267.987 (14) | 42.000 (40) / 56.473 (28) |
| 170-incident | 555.000 (50) / 585.358 (47) | 36.996 (50) / 38.400 (17) |
| 170-skew | 655.849 (39) / 319.136 (22) | 57.600 (39) / 43.200 (39) |
| 190 | 514.515 (40) / 540.000 (38) | 46.728 (15) / 50.400 (56) |

Paired interval TTT sums, local minus carry (positive=worse), in veh*h. Bin boundaries are physical times [900,3600), [3600,5580), [5580,9900), [9900,14400]; equal warmup cancels.

| Scenario | Steps 0-14 | 15-25 | 26-49 | 50-74 | Total delta | Terminal inventory C / L (veh) | L/C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 155 | -0.797 | -0.723 | 54.666 | -1.082 | 52.065 | 412.509 / 415.840 | 1.00807 |
| 170 | 1.533 | 225.999 | 3095.274 | 4020.721 | 7343.528 | 413.976 / 2966.222 | 7.16520 |
| 170-incident | -2.200 | 8.394 | -101.724 | 22.588 | -72.942 | 417.623 / 419.321 | 1.00407 |
| 170-skew | 0.546 | -1.546 | -25.102 | 31.520 | 5.418 | 423.151 / 417.502 | 0.98665 |
| 190 | 0.134 | 5.494 | 206.213 | 60.535 | 272.376 | 426.669 / 423.756 | 0.99317 |

170 has 96.90% of net excess TTT at steps 26-74; 54.75% at 50-74. Peak interval excess is 209.818 at step 48. Excess inventory at ends of steps 14/25/49/74 is 20.830/1000.415/4143.148/2552.246 veh.
170 terminal inventory is predominantly urban movement queues: C=38.389, L=2464.770 veh; their difference accounts for 2426.381 of 2552.246 excess veh. Ramp queues are 3.224/113.202. All ten component decompositions are in evidence; legacy boundary/urban queue views and storage availability were not double-counted as vehicles.
The incident case improves TTT despite 914.579 NUF base drift; 190 has only 168.143 base drift but +272.376 TTT. Rebase count/magnitude alone cannot explain the five outcomes.

## Mechanism Versus Causality

1. Local 170 fallbacks 9 and 12 ratchet the NUF center 6000 -> 5758.248869 -> 5361.154284. Physical fallback freshly evaluates carried controls; its budget need not equal the prior request.
2. At 14, Q=X NUF=4961.061339 but G NUF=2840.800684. D settings fall from 1265.449/1267.987 to near zero; F settings remain 1418.322/1422.479. A/B/C greens are 20/92. This feasible lower solution precedes the large base reset; its predicted three-step TTT exceeds the reference by 5.332466.
3. At 15, Q NUF=5081.775497. The selected lower candidate fails NP by +7.443873 (NUF has slack -2213.952543); physical fallback repeats step-14 controls, X=G NUF=2840.800684, and both exploration coordinates rebase. Step 16 rebases NP again with NUF unchanged. Step 26 raises the NUF center to 3249.983841; it stays there thereafter.
4. Proven mechanism: fallback updates the center and resets offsets, so a +/-500 moving-center rule does not bound episode displacement. The low center persists and constrains subsequent requests. Unproven: the fraction of loss caused by this feedback versus earlier coupled NP/NUF follower response and later state propagation.
5. Unsupported alternatives: treating G as a target that must equal Q, declaring zero-request collapse, blaming rate clipping, concluding all fallback is harmful, freezing signed NP at its initial value, turning on an H3 guard, or claiming a new learner will repair this. Each either contradicts evidence or changes another mechanism.

## Minimal Falsifiable Follow-On

Proposed treatment only: retain initial NUF exploration center `base[1]=B0_NUF` through fallback/recovery. Preserve current NP rebasing. After every commit, compute NUF realized offset as `X[1]-B0_NUF`; NP keeps the existing reset-on-rebase rule. This preserves the meaning of realized offset when its center is retained. Resetting NUF offset to zero while retaining its center would be a different treatment and would break the existing offset/execution consistency invariant.
Keep mean reversion 0.8, noise scales [10,100], offset limits [50,500], float32 action conversion, scales/rates [50,1000], NUF capacity projection, actual previous-executed action anchor, physical reference/fallback, follower settings/duals, gamma=1, reward=-intervalTTT/100, H=3, and full 75-control-step horizon unchanged.
Retain every PCG64 `normal(size=2)` draw in the same order and the same training demand seed. NP numerical trajectories may change downstream because the plant changes; its update rule must not change.
With B0_NUF=6000, desired NUF lies in [5500,6000] after projection. The actual rate-limited request can temporarily lie below 5500 after a low physical fallback; X and achieved throughput are not artificially clamped to that band.
First divergence is step 10, not 15: existing requests/controls/states through step 9 should reproduce. Using the saved step-10 noise and actual anchor, consistent retained-center arithmetic gives Q_NUF=5552.771855 vs old 5504.421625 (+48.350230). This is only action arithmetic at the shared pre-intervention history, not a hypothetical network rollout.

1. Predeclare the treatment and diagnostics, preserve old outputs, and use a new candidate output root. Run real local 170 only, training seed 6802 and exploration seed 6902, from the same initial state/warmup through step 74. One numerical worker, one thread. Do not resume an old post-divergence physical checkpoint or relabel old transitions as treatment.
2. Integrity gate: same frozen physical/config/profile identities, preserved RNG vectors, pre-intervention numerical agreement at existing 1e-8 accounting tolerance (ignore wall-clock fields), actual anchor chain, valid physical execution, unchanged terminal/reward/accounting. Failure invalidates attribution; do not infer that center retention failed.
3. Primary sufficiency test: unchanged coverage screen, zero-NUF requests <= carry+5 (=5) and terminal inventory <=1.25*413.97611790563116 = 517.470147382039 veh. Also require total TTT < old local 11945.939292655728 to support the proposed loss-reduction mechanism. Report TTT relative to frozen carry 4602.4114708736415 without adding a policy-win claim.
4. Falsify sufficiency if a valid treatment run still fails the inventory screen; TTT not lower falsifies the stated loss-reduction prediction for this seed. Screen pass with worse TTT is insufficient for this proposed advance. Continued D closure would show that fixed center does not prevent that response; closure removal alone is not success.
5. If accepted, keep this exact 170 rollout as one of the eventual five. Run only the remaining local 155/incident/skew/190 using seed pairs 6801/6901, 6803/6903, 6804/6904, 6805/6905 and identical frozen candidate code. No treatment retuning after seeing 170. These four may use four single-thread numerical workers; total concurrent numerical workers across shared work must remain <=8.
6. This is 75 new transitions first, then 300 only if justified: five equal 75-transition candidate strata. Reuse all five completed carry references and all old local traces as immutable comparisons; no carry recollection and no arbitrary 24h extension. A changed candidate after 170 invalidates reuse of that run as its stratum.
7. No model/value/actor training before all five candidate trajectories finish and all five coverage/integrity screens pass; passing collection screens is still not automatic learner admission. Preserve all old 750 transitions with behavior/version provenance; they are actual behavior transitions, never optimal-action labels. Any future admitted learner minibatch remains 20% per scenario with one shared state-conditioned value/policy.
This is a valid minimal probe of retaining the NUF center and its downstream closed-loop consequences, conditional on reproduction and invariant checks. It does not isolate the step-15 reset alone, hold later NP actions fixed, estimate across-seed effects, or demonstrate all-five canonical TTT improvement. No broader follower redesign is needed to answer this question.

## Source Anchors

`work/sdmpc_rl_local_20260930_v2/exploration.py:70,117,134,172,228`: offsets, RNG, commit/rebase and realized-offset consistency.
`work/sdmpc_rl_multi_20260929/budget_controller.py:34,54,124,143,168`: fresh reference, actual anchor, lower selection, physical fallback and Q/X/G audit.
`work/sdmpc_rl_multi_20260929/budget_env.py:120,134,163,191,238`: training perturbation, warmup, recovery, plant execution, finite horizon and reward.
`work/sdmpc_rl_local_20260930_v2/validate.py:76,134,161,215`: trace/summary/manifest contracts and predeclared screen; read only, not invoked.
artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_externality_ablation_20260923/fixed_policy.py:20,24,38 and `exception_controller.py:16,56`: upper constraints and execution checks.
artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_tree/src/controllers/sensitivity_dmpc.py:199, `models/state.py:956,983,994,1013`, and `simulation/player_cost_accounting.py:14`: terminal inventory definition.

Outcome: diagnosis complete; experiment proposed only. The shared five-scenario improvement goal remains active and unachieved.
