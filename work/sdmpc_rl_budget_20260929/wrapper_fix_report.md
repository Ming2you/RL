# Wrapper Fix Report

Status: DONE. Date: 2026-09-29.

Implemented the three required P2 remedies in `wrapper_review.md`, plus the
parent-authorized endpoint queue-duration estimate. Implementation files are
stable and ready for the parent's real smoke/resume and independent re-review.
No traffic simulations, commits, staging, frozen-source edits, or runner/TD3
edits were performed by this task.

## Changed Files

- `budget_controller.py`
- `budget_env.py`
- `budget_runtime.py`
- `test_budget_controller.py`
- `test_budget_contract.py` (new)
- `wrapper_fix_report.md` (new)

## Audit Interfaces

`audit['candidates']` is a detached list preserving every lower response's
requested budget, status/reason, feasibility/convergence, residual, stationarity,
stationarity point, inner price history (`rows`), local-QP diagnostics
(`local_rows`), multiplier evidence and other existing scalar/vector metadata.
Each entry adds `candidate_index`, `achieved`, `predicted_TTT`, and
`response_check`. The response's `evaluation` and `control` objects are excluded;
predicted TrafficState trajectories are not exported. Shared derivative evidence
is retained in `audit['derivatives']`.

`audit['execution_check']` remains the executed-control check. Separate
`lower_selection_check` and `lower_selection_identity` preserve the lower point
considered before the PFO guard, even when that point is rejected.
`selected_identity` identifies the executed candidate, archive point or PFO
reference. Archive identity includes its candidate index, source label and point;
the original failed candidate's final point remains in its candidate record.
The entire controller audit is deep-copied before next-reference preparation.

`audit['plant_log']` retains the actual interval log including plant diagnostics;
`audit['plant_state']` is the detached actual endpoint state. Existing queue and
inventory fields remain available. Candidate checks reuse cached final-point
evaluations; added cache hits and audit overhead are reporting costs, not new
solver iterations or changes to selection.

## Checkpoint Boundary

New checkpoints store `contract`: reward divisor, full cfg, lower options,
source snapshot identity, and observation version/implementation fingerprint,
ordered names, links, buffer keys and delay bins. The observation implementation
fingerprint covers its normalization expressions and numeric-leaf traversal.
Old checkpoints without this contract are intentionally refused.

Restore compares the receiving contract before closing/replacing live state or
constructing a new lower controller. It also checks the saved simulator cfg and
runtime cfg, and reconstructs the current observation schema from saved inputs
without a solve. Same-contract serialized restore preserves the prepared
reference, forecast, PFO memory, prices, budget memory and timings. The existing
reference reconstruction remains; restore never calls the PFO solver again.
Observation values and normalization expressions themselves were not changed.

## Runtime Verification

`boot()` accepts a completed snapshot root or its `source` directory. It verifies
the entire manifest payload before importing frozen runtime code. Original-source
roots without a snapshot manifest are refused.

- `rt['snapshot_identity']`: serializable `manifest_sha256`, `file_count`, `total_bytes`.
- `rt['verify']()`: rechecks all payload files and the pinned manifest identity;
  callable from the parent's completion/failure/finally paths.
- `verify_frozen_source(root, expected=None)`: read-only verifier without booting.

Actual frozen snapshot verification passed: 151 files, 3,975,269 bytes;
manifest SHA256 `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.

## Timing And Queue Comparison Schema

PFO, reference, lower and guard timings remain disjoint. Forecast and observation
wall/CPU metadata are now included. `step(action, mode='rl', actor_seconds=0.,
actor_cpu_seconds=None)` accepts actor CPU time measured by the caller.
`decision_wall_seconds` sums forecast, observation, PFO, reference, lower, guard
and actor wall components. `controller_cpu_seconds` sums those CPU components
excluding actor; `decision_cpu_seconds` adds actor CPU when supplied and is
otherwise `None`. This is a component sum, not end-to-end environment/runner
time. Plant wall/CPU are separately measured once around `sim.step`; they are
not included in decision totals. Next preparation/observation belongs to the next
decision. Training/checkpoint timing remains the runner's responsibility.

`audit['queue_near_capacity_estimate']` has this schema:

```text
method: 'interval_endpoint_sampled'
exact_substep_exposure: false
threshold_fraction: 0.9
interval_seconds: actual plant elapsed seconds (normally 180)
duration_units: 's'
capacity_units: 'veh'
capacity_definition:
  ramp_queue: 'cfg.network.ramp_queue_max_veh'
  boundary_queue: 'cfg.network.boundary_queue_max_veh'
ramp_queue: {queue_id: {queue_veh, capacity_veh, near_capacity_seconds}}
boundary_queue: {queue_id: {queue_veh, capacity_veh, near_capacity_seconds}}
```

For each queue, estimated seconds equal the interval duration times the indicator
that endpoint queue occupancy is at least 0.9 times its stated capacity. Sum
across intervals for each queue's estimated duration. Summing across different
queues yields queue-seconds, not elapsed time. The field names and actor CPU
interface were sent to Linnaeus for runner/comparison integration.

This is explicitly an endpoint estimate, not exact substep exposure. The frozen
`src/simulation/coupling.py:237` builds aggregate diagnostics and returns those,
not its local `urban_rows`/`freeway_rows` histories. Its freeway aggregator
(`:44`) sums/averages diagnostic values; `src/models/urban_queue_model.py:1125`
retains selected sums and endpoint values. Thus the returned plant log cannot
recover each queue's threshold-crossing duration. Endpoints can miss or overstate
within-interval exposure. No frozen plant instrumentation was added.

The estimate covers ramp and boundary queues only. No near-capacity duration was
invented for mainline-origin queues or urban movement queues: mainline-origin
queues have no inspected finite storage cap, while the movement queue clip is
a numerical guard (`urban_queue_model.py:532`, 1e9 vehicles), not a physical
capacity. The distinct `movement_storage_capacity` diagnostic normalization is
not silently treated as that queue's hard capacity.

## Validation

All commands ran from the RL repository root with `-B`; installed package ACL
access used `require_escalated`. The initial regression run reproduced missing
candidate metadata, audit aliasing, missing checkpoint contracts and source
tampering reaching imports. Final synthetic suite: **19 tests PASS**, 0.230 s.

```powershell
.\.venv-torch\Scripts\python.exe -B -c "import sys, unittest; sys.path[:0] = ['.deps-budget', 'work/sdmpc_rl_budget_20260929']; suite = unittest.defaultTestLoader.loadTestsFromNames(['test_budget_controller', 'test_budget_contract']); result = unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())"

.\.venv-torch\Scripts\python.exe -B -c "import sys; sys.path[:0] = ['.deps-budget', 'work/sdmpc_rl_budget_20260929']; from budget_runtime import DEFAULT_SNAPSHOT, verify_frozen_source; print(verify_frozen_source(DEFAULT_SNAPSHOT))"

git diff --check -- work/sdmpc_rl_budget_20260929/budget_controller.py work/sdmpc_rl_budget_20260929/budget_env.py work/sdmpc_rl_budget_20260929/budget_runtime.py work/sdmpc_rl_budget_20260929/test_budget_controller.py
```

Coverage includes nonzero-dual fallback, failed and feasible candidate evidence,
archive identity and guard-rejected archive evidence, audit detachment after the
next preparation, actual-interval synthetic plant trace retention, disjoint
timing sums, queue threshold/units/duration, reward/cfg/options/source/observation
mismatch refusal before replacement, saved-simulator cfg mismatch, legacy
checkpoint refusal, exact serialized same-contract observation restore without
PFO, terminal restore, pre-import source tampering, both snapshot root forms and
completion manifest/payload tampering. Whitespace diff check passed.

Control selection, physical budget ranges, constraints, TTT guard, reward,
discount semantics and frozen solver code were not changed. Real traffic/resume
parity is intentionally left to the parent's smoke and scoped independent review.

## Final Review Follow-Up: Dynamic Config Identity

The independent `final_pilot_review.md` identified a further P2 omission:
`plain()` uses `dataclasses.asdict`, which drops runtime-added public dataclass
attributes such as `network.terminal_zero_gradient`. The focused repair changes
only `budget_env.py`, `test_budget_contract.py`, and this report.

Environment contracts now use the existing frozen
`rt['rc'].to_plain_dict` serializer for both cfg and lower options. Saved-simulator
and current-runtime cfg validation uses that same representation. The existing
`plain()` normalization is applied only after lossless config expansion.
Environment contract version is now 2; version 1 identities are refused because
they cannot attest to dynamic config. No `config_plain`/`config_digest` public
helper was added. `budget_runtime.py`, global `plain()`, protocol validation,
physical TrafficState serialization, plant/solver behavior, and the frozen
snapshot were unchanged in this follow-up.

Three new synthetic tests first reproduced the omission, then passed with the
repair. They cover dynamic receiving network attributes, nested MPC attributes,
lower options, a separately replaced runtime cfg, and saved-simulator config
drift. Refusal occurs before live state is replaced or a new lower controller is
constructed. They also verify detached dynamic config metadata and exact
same-config serialized observation restore without another PFO call. Fixtures
stub the frozen serializer's declared-field plus public-dynamic-attribute
semantics, as requested.

Final focused command (same full wrapper command recorded above):
`loadTestsFromNames(['test_budget_controller', 'test_budget_contract'])`.
Result: **22 tests PASS**, 0.300 s. Scoped `git diff --check` also passed.
No traffic simulations or commits were run. Wrapper implementation is stable
for the parent's complete-suite and real smoke reruns, followed by re-review.

Coordinated with Linnaeus: runner configuration and options hashes use
`digest(rt['rc'].to_plain_dict(value))`, reusing the same serializer without a
duplicate abstraction. Runner files and runner hash tests remain Linnaeus's scope.
