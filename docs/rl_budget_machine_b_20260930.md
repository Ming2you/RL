# Machine B continuation (2026-09-30)

The balanced shared-policy goal (`rl_budget_balanced_goal_20260930.md`) continues on a second PC
("machine B"). Everything below was measured on machine B. Machine-A records stay immutable history.

## Machine and environment

- Machine B: AMD Ryzen 5 3500X (6 CPUs, AVX2, no AVX-512), 16 GB RAM, Windows 10 19045.
  Repository `C:\Users\alsrj\Desktop\RL`, branch `codex/sdmpc-rl-budget-20260929` at `384dd65`.
- Data restored from `artifacts/sdmpc_research_handoff_20260930` (1,627 files, `verify-local` PASS).
- `.venv-torch` built from the Codex runtime Python 3.12.14 (same build string as machine A) with
  torch 2.14.0+cpu, numpy 2.3.5, scipy 1.16.3, PyYAML 6.0.3 (pytest added for tests only).
  `runtime_versions()` equals the pinned contract exactly. `.deps-budget` on machine A held only
  numpy 2.3.5 and pytest, so it is not needed.
- Machine B is roughly 1.8-2x slower per interval than machine A with five parallel workers
  (carry center 1576-2193 s vs 882-1273 s).

## Cross-machine numerics

1. Actor: the frozen `return_mc_v1` actor reproduces 177/375 recorded Task-2 actions bit-exactly;
   198 differ by at most 4 float32 ULP (max |diff| 9.3e-10). No MKL/oneDNN/ATen setting changed this.
2. Physics: the unchanged zero-action carry center re-run on machine B
   (`results/sdmpc_rl_machine_b_20260930/center_repro_v1`) diverges from machine A after 16-20
   control intervals, starting at 1e-12..1e-10 veh-h and amplifying:

| Scenario | Machine B carry TTT | Machine A carry TTT | B - A | First divergence |
| --- | ---: | ---: | ---: | ---: |
| sweet_155_w | 3101.4778926950567 | 3103.0110715680044 | -1.533 (-0.049%) | step 17 |
| sweet_170_w | 3934.6182377626324 | 3935.903236508048 | -1.285 (-0.033%) | step 19 (+19.0 at step 41) |
| sweet_170_incident_w | 5546.224358313904 | 5546.224352256691 | +6e-6 (0.000%) | step 20 (stays ~1e-10) |
| sweet_170_skew15_w | 4244.34789709013 | 4250.876599300032 | -6.529 (-0.154%) | step 16 |
| sweet_190_w | 6593.685996544476 | 6604.2970168093225 | -10.611 (-0.161%) | step 17 |

The lower S-DMPC never reports convergence (6 Jacobi iterations); round-off-level input changes
flip its discrete outcomes and the difference grows through congestion. Consequences:

- Machine-A centers are NOT a valid matched baseline on machine B; with them a zero-change policy
  would "improve" four scenarios. Machine-B evaluations use the machine-B centers above.
- The acceptance rule (each scenario improves by more than max(1e-6, 1e-8*baseline)) is far below
  this chaotic noise floor (0.03-0.16% from a 1e-10 perturbation). A candidate whose systematic
  effect is not clearly larger than the noise passes or fails by chance. Candidates should be
  screened on training profiles for effects that exceed the measured noise before a canonical run.

## Machine-B canonical evaluator

`work/sdmpc_rl_return_eval_b_20260930` is a copy of the frozen Task-4 evaluator with only:

- cross-machine actor parity within 16 ULP (reported, not bit-exact); within-run actor checks
  stay bit-exact; the preflight is bound to machine B by a CPU/platform fingerprint and the digest
  of the 375 actions computed on this machine (re-checked by the worker);
- CPU masks `(1,2,4,8,16)` (machine A's `(1,4,16,64,256)` address CPUs machine B lacks);
- machine-B centers and baseline TTTs, guarded to live under the machine-B root;
- evaluator module `policy.py` renamed to `canonical_policy.py`: the frozen snapshot imports its
  own `policy` module by bare name at boot, so the Task-4 evaluator could never have booted
  (it was only ever tested with boot blocked). A regression test now forbids such collisions.

Tests: 59 passed (`evidence/9f7515aa`). Independent read-only review findings F1-F4 addressed.
First launch `return_canonical_b1` failed at boot with zero physical steps (collision above);
its slots are preserved. Active output: `results/sdmpc_rl_machine_b_20260930/return_canonical_b2`.

## Evidence already visible before any new learning

Training-profile carry trajectories (local_budget_v2) show that after the step-14/15 fallback the
carried budget is held at roughly NP -20..+3 and NUF 5540..6000, while the achieved G during
congestion (steps 20-50) is NP -200..-470 and NUF 4000..5900. The upper budgets are therefore
slack through congestion; only budget tightening below the achieved level can change the lower
solution there. Paired exploration runs (local_budget_v2, nuf_retention_v1) never bound NP in steps
26-55, yet their full-run TTT differed from carry by -7.8%..+5.3%, consistent with chaotic
amplification of early differences rather than a demonstrated systematic lever.

## Canonical evaluation of `return_mc_v1` on machine B: FAILED in all five

Evaluator `work/sdmpc_rl_return_eval_b_20260930` (59 tests, preflight `evidence/9f7515aa`,
receipt `results/sdmpc_rl_machine_b_20260930/receipts/eval_b2_parent_review.json`), output
`results/sdmpc_rl_machine_b_20260930/return_canonical_b2`, readout status `failed_acceptance`
(`return_canonical_b2_readout.json`). All five ran 75 intervals / 14400 s with the physical guard.

| Scenario | Candidate TTT | Machine-B carry TTT | Improvement |
| --- | ---: | ---: | ---: |
| sweet_155_w | 3272.9055344510407 | 3101.4778926950567 | -171.428 (-5.53%) |
| sweet_170_w | 4498.589110674243 | 3934.6182377626324 | -563.971 (-14.33%) |
| sweet_170_incident_w | 6048.639832711838 | 5546.224358313904 | -502.415 (-9.06%) |
| sweet_170_skew15_w | 4419.073567365916 | 4244.34789709013 | -174.726 (-4.12%) |
| sweet_190_w | 6699.71697017048 | 6593.685996544476 | -106.031 (-1.61%) |

These losses are one-directional and 10-100x the chaotic noise floor, so they are a systematic
effect of the policy, not chance. Mechanism (155 trace vs center): the actor raises NP by ~0.2 per
interval while NP binds in free flow (steps 1-14; early TTT -2.0), the step-15 fallback then
resets the anchor to a looser reference NP (-43 vs -52), and from congestion onset (steps 18-26)
the lower solution evacuates the urban region far less (achieved NP -66..-225 vs -133..-257)
although NP is slack at that moment: budget history acts through the carried dual prices. Urban
accumulation then costs up to +12 veh-h per interval through the peak.

Hypothesis for the next candidate: the useful lever is the opposite one, perimeter tightening
(lower NP than achieved) around congestion onset, and possibly early. Probe P1 tests it.

## Probe P1 (training profiles only)

`work/sdmpc_rl_probe_b_20260930/probe.py`, options `options_p1.json`, output
`results/sdmpc_rl_machine_b_20260930/probe_p1/<scenario>_s<seed>`, seeds 8101..8105. One carry run
per scenario with in-memory checkpoints at control steps 1 and 16, then branches continued with
carry to the terminal: early NP tightening/relaxation (-5 / +5 per step for 14 steps), NP binding
50 below achieved for 10 steps and 100 below for 20 steps (then return), NUF binding 500 below
achieved for 10 steps, a tiny NUF shift, and a zero-action restore control (determinism).
Smoke evidence: restore + carry reproduced carry intervals bit-exactly; a 2 veh/h NUF shift at
step 2 changed nothing at all (TTT identical), so small budget changes are absorbed by the lower
solver and chaos comes from arithmetic-level differences, not from tiny budget moves.

## Probe results so far (training profiles; positive = worse than carry)

P1 (seeds 8101-8105; one carry per scenario, branches restored bit-exactly: all five zero-action
controls reproduced carry TTT exactly):

| Option | 155 | 170 | incident | skew15 | 190 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NP bind 50 below achieved, steps 16-25, return | -7.46% | -6.23% | -3.09% | -0.94% | +1.54% |
| NP bind 100 below achieved, steps 16-35, return | -7.30% | -5.87% | -4.98% | +2.59% | (pending) |
| early NP relax +5/step, steps 1-14 | -3.55% | +0.34% | -6.71% | -0.15% | -0.40% |
| early NP tighten -5/step, steps 1-14 | +3.32% | +8.85% | -3.77% | +0.27% | +1.49% |
| NUF bind 500 below achieved, steps 16-25 | +32.97% | +46.58% | +6.61% | +36.09% | +22.76% |
| NUF -2 veh/h at step 16 (noise probe) | 0 | 0 | -1.72% | +2.15% | -4.90% |

P2 (170, seed 8202, early NP relax from step 1): +5/step -1.78%, +10/step -3.52%,
"margin" (NP kept 30 above last achieved for 20 steps) -4.00%.
P3 (seeds 8401-8405, NP-bind dose/timing): 155 seed 8401, bind 50 steps 16-25: **+4.18%** (the same
option was -7.46% on seed 8101).

Findings:

- The chaotic noise floor is large: a 2 veh/h NUF change that the lower solver does not absorb moves
  the full-run TTT by up to +/-5%. Single-seed differences below ~5% are not evidence.
- Large systematic effects exist in the harmful direction: NUF tightening (+6.6..+46.6%), and the
  `return_mc_v1` actor on canonical profiles (+1.6..+14.3%).
- Early fallbacks are `requested_budget_infeasible`: carry keeps the initial PFO NP budget, which
  becomes infeasible as demand loads (fallbacks at steps 2 and 15). Early relaxation removes them;
  early tightening multiplies them (every other step).
- Outcomes hinge on urban congestion: improvements coincide with 150-830 lower peak urban queue and
  lower urban TTT at a smaller freeway cost; the losing cases show higher urban queue peaks. The
  mild 155 seed-8401 profile (urban queue peak 298) is hurt by unneeded NP binding.
- Next (P4, seeds 8501-8505, branch checkpoints cached on D:): stateless perimeter feedback on the
  urban queue (bind NP 50 below achieved while the urban queue exceeds a threshold, otherwise keep NP
  30 above achieved to avoid infeasibility fallbacks), versus the fixed-time NP bind.

### P3 (seeds 8401-8405): NP-bind dose and timing (return to the pre-option anchor afterwards)

| Option (start step, delta, window) | 155 s8401 | 170 s8402 | incident s8403 | skew15 s8404 | 190 s8405 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16, 50, 10 | +4.18% | -9.56% | -3.46% | +2.55% | -1.87% |
| 16, 50, 15 | +2.49% | -9.23% | -0.15% | +5.81% | -2.33% |
| 16, 30, 10 | -1.04% | +0.84% | -1.89% | -1.03% | (pending) |
| 14, 50, 10 | +4.68% | -9.46% | -0.63% | +1.63% | (pending) |
| 18, 50, 10 | -1.27% | -7.06% | +0.68% | -0.06% | (pending) |

Across P1+P3, NP binding at congestion onset is a reproducible, large gain only for 170 (-6..-10% on
two seeds and every delta >= 50) and a moderate one for incident (-3.1/-3.5% at 16/50/10). For 155,
skew15 and 190 the sign flips between seeds; the measured noise floor (+/-2-5%) is as large as the
effects. Weaker binding (delta 30) is mildly positive except for 170.

P4 so far: all-queue feedback from step 1 (on 150/off 100, delta 50, margin 30) gave -1.36% on 190
seed 8605 but with 37 fallbacks (carry: 3), and +2.65% on 155 seed 8501. The all-queue trigger
includes the boundary_in gate queues that NP gating itself fills, so the gating latches on.
An independent review of the new actor/evaluator found no decoding errors and asked for machine
binding, spec pre-registration, a lock and exclusive writes, all implemented in
`work/sdmpc_rl_perimeter_b_20261001` (actor, canonical_eval, readout, register). The actor now also
offers the frozen model's protected accumulation N_P (gate and on-ramp queues excluded) as trigger,
verified against the plant state within 2e-5 veh.

### P4-P6 and the first canonical attempt (2026-10-01, machine B)

- All-queue perimeter feedback (P4, bind NP while the urban queue incl. gate queues exceeds a
  threshold) is unsafe: +55% and +70% collapses on 190 seed 8505, +8.6% skew, +2.5..+5.7% 155; the
  trigger includes the gate queues that gating fills, so it latches. Rejected.
- Early "keep feasible" relaxation combined with binding (P5 C1/C2) was worse than binding alone on
  the same seeds (155 +6.0 vs +3.0, 170 +1.2 vs -3.8, 190 +7.4/-0.2 vs -1.7/-10.3). Relaxation alone
  (C4) ranged -7.3..+2.3%. Protected-accumulation feedback (C3, N_P on 400/off 300) ranged -7.9..+2.2%.
- The null actor reproduced carry bit-exactly on every training slot, and on canonical
  sweet_170_skew15_w it reproduced the machine-B center TTT 4244.34789709013 exactly (the most
  chaotic scenario), so the new evaluator's canonical path is validated.
- Candidate B1 (hold to step 15, bind NP 50 below achieved for steps 16-25, return to the anchor
  latched at step 16), registered as entry 2 before the run, training evidence 155 2/4, 170 3/4,
  incident 3/3, skew15 2/4, 190 3/4 improved. The actor matches the probe option bit-exactly until
  step 36 on 190 seed 8605 (float32-decoding differences then diverge chaotically).
- Canonical attempt 1 (results/sdmpc_rl_machine_b_20260930/canonical_b1_run1): sweet_170_incident_w
  5683.356414679126 vs center 5546.224358313904 = -2.47%, FAILED. The remaining scenarios are run
  to completion for full reporting.

### Canonical attempt 1 result: FAILED (readout `failed_acceptance`)

Root `results/sdmpc_rl_machine_b_20260930/canonical_b1_run1`, registry entry 2 (spec
8bb199ce...deb8, evaluator sources b2d4e9b2...9a7d), machine B, all five 75 intervals / 14400 s.

| Scenario | B1 TTT | Machine-B center | Improvement | Pass |
| --- | ---: | ---: | ---: | --- |
| sweet_155_w | 3047.6458727837307 | 3101.4778926950567 | +53.83 (+1.74%) | yes |
| sweet_170_w | 4181.56812235315 | 3934.6182377626324 | -246.95 (-6.28%) | no |
| sweet_170_incident_w | 5683.356414679126 | 5546.224358313904 | -137.13 (-2.47%) | no |
| sweet_170_skew15_w | 4147.720078222679 | 4244.34789709013 | +96.63 (+2.28%) | yes |
| sweet_190_w | 7441.973139930229 | 6593.685996544476 | -848.29 (-12.87%) | no |

### Why canonical results disagree with the training evidence

The training profiles are the canonical `forecast.json` with three demand multipliers drawn from
U(0.98, 1.02) (freeway_mainline, urban_boundary, ramp_arrival); the canonical profile is exactly
the multiplier-1.0 point of the same construction (`budget_runtime.protocol` vs
`BudgetEnv(training_seed=...)`). Regressing the machine-B carry TTT of all training carries on the
mean multiplier and evaluating at 1.0:

| Scenario | Training carries | Predicted at 1.0 | Canonical carry | Gap |
| --- | --- | ---: | ---: | ---: |
| 155 | 2967-3293 (n=5) | 3159 +/- 127 | 3101.5 | -1.8% |
| 170 | 4208-4587 (n=6) | 4316 +/- 146 | 3934.6 | -8.8% |
| incident | 5772-6082 (n=4) | 5877 +/- 85 | 5546.2 | -5.6% |
| skew15 | 4069-4418 (n=5) | 4164 +/- 115 | 4244.3 | +1.9% |
| 190 | 6444-7171 (n=4) | 6741 +/- 225 | 6593.7 | -2.2% |

For 170 and incident the canonical carry is better than EVERY training carry, including seeds whose
three multipliers are all below 1 (170 seed 8502: 0.9986/0.987/0.9947 gives 4300 = +9.3%). The
numerical (cross-machine, 1e-10) perturbation left these canonical carries almost unchanged
(-0.03%, +0.0000%), so they are not fragile lucky trajectories: the exact canonical demand sits in
a narrow favourable basin of the baseline controller, and +/-0.5..2% demand changes leave it.
B1's canonical 170 (4181.6) and incident (5683.4) are better than all training carries of those
scenarios but not better than these special centers. B1 won where the canonical carry is typical
(155, skew15) and lost where it is anomalously good (170, incident) and on 190.

Consequence: policies developed on the +/-2% training distribution are evaluated on a baseline point
the training distribution never shows. The all-five strict criterion is therefore limited by (i) the
+/-3-5% chaotic spread of single runs and (ii) this train/canonical baseline shift, not only by the
policy. Next steps need a user decision (training distribution near the canonical point, a
statistical multi-realization criterion, or stopping); see the chat report of 2026-10-01.

### Final training-profile evidence for B1 (after P6/P7; all runs complete, no process left)

| Scenario | B1 change vs matched carry, per training seed | Mean | Improved |
| --- | --- | ---: | ---: |
| 155 | -7.5, +4.2, +3.0, -0.1, -0.4 | -0.2% | 3/5 |
| 170 | -6.2, -9.6, -3.8, +1.8, -10.8 | -5.7% | 4/5 |
| incident | -3.1, -3.5, -0.6, -1.8, +1.3 | -1.5% | 4/5 |
| skew15 | -0.9, +2.5, -3.2, +2.1, +2.2 | +0.6% | 2/5 |
| 190 | +1.5, -1.9, -1.7, -10.3, -1.8, -5.0 | -3.2% | 5/6 |

Stateless approximations of B1's return (V2 relax to achieved+200, V3 relax to -40) collapsed on
190 seed 8705 (+22.8% / +22.3%) where B1's latched return gave -1.8%; the canonical candidate
therefore used the latched return. Weak binding (delta 30, V5) was +13.3% (155), +6.2% (170),
+10.2% (skew), -2.8% (incident), -4.2% (190): unsafe.

Status at 2026-10-01 07:30: goal NOT achieved (canonical attempt 1 failed, 2/5). All compute idle,
awaiting the user's decision on the next direction (see the report in chat).

The user chose to keep developing on the training profiles and to keep the lower level at
6 Jacobi iterations (30 iterations had neither converged nor changed performance materially).

### P8 (2026-10-01): longer and later binding windows (training profiles, branch at step 16)

Queue `queue_runner.py --name p8`, 16 training slots (seeds 8501-8505, 8605, 8701-8705, 8801-8805),
one job per CPU, carries and step-16 states from the D: checkpoint cache. Every option holds carry
to step 15, binds NP below the achieved value inside the window and then returns to the anchor
latched at the first binding step (actor specs `p8a`..`p8f` in `work/sdmpc_rl_perimeter_b_20261001/specs`).
With delta >= 50 the per-interval action limit (50 veh) saturates, so only the window matters:
p8c (delta 100, steps 16-30) reproduced p8b (delta 75, steps 16-30) TTT exactly on both slots it
ran, so p8c and p8d (delta 50, steps 16-35, the same window as p8a) were dropped from the queue.
Change vs the matched carry in % (negative = better); B1 = the canonical candidate (probe option
NP bind 50, steps 16-25, return).

| Scenario, seed | B1 16-25 | 16-27 (p8f) | 16-30 (p8b) | 16-35 (p8a) | 18-32 (p8e) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 155 s8501 | +2.99 |  | +1.20 | +1.19 |  |
| 155 s8701 | -0.11 | -0.34 | -0.76 | -1.07 | -0.82 |
| 155 s8801 | -0.36 | +0.54 | -0.40 | -0.54 | +0.95 |
| 170 s8502 | -3.81 |  | -7.74 | -6.41 |  |
| 170 s8702 | +1.78 | +3.51 | +1.14 | +2.75 | +3.92 |
| 170 s8802 | -10.84 | -9.72 | -11.15 |  | -11.64 |
| incident s8503 | -0.56 |  | -3.29 | -1.73 |  |
| incident s8703 | -1.79 | +0.43 | -2.32 | -2.03 | +1.59 |
| incident s8803 | +1.31 | +0.10 | +1.60 |  | -4.65 |
| skew15 s8504 | -3.17 |  | -0.57 | -3.76 |  |
| skew15 s8704 | +2.15 | +3.33 | +2.96 | +3.41 | -1.07 |
| skew15 s8804 | +2.18 | +15.62 | +15.55 |  | +3.59 |
| 190 s8505 | -1.66 |  | +8.00 | +5.34 |  |
| 190 s8605 | -10.34 |  | -4.37 | -6.51 |  |
| 190 s8705 | -1.84 | +60.24 | +48.70 | +48.70 | -3.16 |
| 190 s8805 | -5.01 | -2.22 | -3.74 |  | +4.64 |

Findings:

- Every window longer than B1's 10 intervals collapsed on at least one slot: 190 s8705 +48.7%
  (16-30, 16-35) and +60.2% (16-27), skew15 s8804 +15.6% (16-27, 16-30). The collapses are urban
  gridlock: on 190 s8705 urban TTT +3189..+3522 veh-h, 8-9 fallbacks after step 16 (B1: 2), peak
  urban queue 3575-4013 veh (carry 2943). The ramp-queue peak reaches its 720 veh cap in carry and
  B1 as well, so the peak alone does not separate the collapses.
- Longer windows deepen the 170 and incident gains on some slots (170 s8502 -7.7%, s8802 -11.2%;
  incident s8503 -3.3%) but not consistently (170 s8702 +1.1..+3.9%, incident s8803 +1.6%).
- A later start (18-32) avoided the s8705 collapse (-3.2%) but lost on 190 s8805 (+4.6%), skew15
  s8804 (+3.6%) and 170 s8702 (+3.9%).
- B1 remains the only window without a collapse on all 26 training slots evaluated (worst +4.2%,
  155 s8401) and it improved 190 on all four P8 seeds. No P8 window dominates it.
- The canonical B1 runs fell back to the reference control inside the binding window in all five
  scenarios (step 16 and some of 22-25; the centers fall back at 2 and 15, plus 17 for 170 and
  incident, from `selection_source` in `center_repro_v1/*/episode_00_trace.json`): late in the
  window the bound request becomes infeasible. How much of the binding effect acts through these
  fallbacks is not known yet; the trace comparison is a next step.

Data organisation (2026-10-01 15:30): every machine-B result is exported to
`docs/machine_b_results/` (`centers`, `canonical`, `carries` with the demand multipliers,
`branches` with every completed branch and its source hash) by
`work/sdmpc_rl_probe_b_20260930/export_results.py`, and the raw files are archived in
`artifacts/sdmpc_machine_b_20261001` (1,127 files, 735 MB raw, 78 MB ZIP in two parts, `verify` PASS)
by `work/sdmpc_machine_b_archive_20261001.py`.

Status at 2026-10-01 15:30: goal NOT achieved (canonical attempts: `return_mc_v1` 0/5, B1 2/5).
All compute idle. Goals and next tasks: `RL_HANDOFF_20261001_machine_b.md`.
