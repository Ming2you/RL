# NUF-retention exploration readout

All five complete, authenticated, predeclared integrity/coverage screens PASS.
Output: results/sdmpc_rl_balanced_goal_20260930/nuf_retention_v1.
375 new actual transitions (75/scenario), five old carry references reused, zero
reference intervals recollected. No actor, Q, training, canonical score, or goal
achievement is implied. The immutable old750 transitions remain preserved.

## Paired training profiles

| Scenario | Carry TTT | New exploration TTT | Change vs carry | Terminal vehicles | Distinct controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 155 | 3210.204341 | 3310.913197 | +3.1371% | 421.642488 | 43 |
| 170 | 4602.411471 | 4553.111049 | -1.0712% | 418.717764 | 67 |
| 170 incident | 5570.093987 | 5862.840970 | +5.2557% | 407.703518 | 69 |
| 170 skew | 4618.714187 | 4259.431213 | -7.7789% | 417.181817 | 58 |
| 190 | 6589.493773 | 6643.434200 | +0.8186% | 422.011326 | 54 |

Lower TTT is better. Each pair shares its training-only perturbed profile;
none uses the canonical acceptance profile. Two rows improved, three worsened.
NUF retention is therefore not a universally beneficial traffic policy.

All 375 requests are distinct within their own75-step episode; zero NUF requests
and simultaneous D-ramp closures are absent. Fallback counts are7/6/6/4/7, with
one initial PFO and no recovery PFO per scenario. All requested NUF bases stay
6000, but executed budgets remain physical outcomes. Scenario170 has maximum
requested/executed NUF drift500.000001/560.643081, including float32 conversion.
The old170 failure (TTT11945.939293, terminal2966.221803) was corrected with only
the NUF-center treatment;75noisevectors matched, physical prefix0..9 matched,
first action difference10. This is one same-seed closed-loop causal probe, not
proof of generalization or of the isolated effect of a particular later step.

## Interval and runtime evidence

TTT differences vs paired carry in control-step bands0..24 /25..49 /50..74:

- 155: -1.983590 / +100.380284 / +2.312162.
- 170: +1.452509 / -100.127686 / +49.374755.
- Incident: +2.955703 / +268.319608 / +21.471672.
- Skew: +6.513394 / -360.393335 / -5.403033.
- 190: +5.014009 / +291.382881 / -242.456463.

Most differences arise in congestion/discharge, not the free-flow prefix.
Peak inventories are2171.200/3365.743/4713.743/3246.027/5186.941vehicles.
Healthy terminal inventory alone cannot rule out large intermediate losses.

Decision wall seconds:1022.932/1048.243/984.097/980.295/1161.571.
Locked worker-session seconds:1079.546/1107.483/1039.847/1037.211/1216.444.
These scopes include serialized checkpoints as documented by each timing.json;
worker-session excludes interpreter/coordinator/final completion publication.
Parallel worker sums are not campaign elapsed time. No inference-only timing or
preview-free learned-control claim is made; no learned model exists here.

## Provenance and next decision

Completion SHA256:2d5c7a881fe5cb818686b13dd83f2c98eb08059e4b698cbe3a1be8757fc0c117.
Comparison SHA256:60f7b670e262348d3d24548809b2aafc26bc7c2a7e7c8f17340fbd17c2803801.
Runner/policy hashes, command, actual process creation identities, review72tests
and pilot evidence are in the probe plan's SDD progress.md. Both coordinators
and all five workers exited. Fresh CIM after completion returned no Python jobs.

Admit this data for the next bounded learner specification, not an actor by
default. Use carry returns only for their recorded continuation, projected
requested budgets as critic actions, equal20%scenario minibatches, one shared
actor/twinQ, then a small new-profile on-policy wave. Avoid a fresh-Q calibration
barrier whose arbitrary update count confounds slow target propagation with data
quality. Details are in rl_budget_return_initialized_policy_20260930.md.
