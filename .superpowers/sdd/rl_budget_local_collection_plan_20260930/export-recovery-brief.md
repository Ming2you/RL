# Bounded terminal export recovery

Read this first. This task fixes an OUTPUT boundary, not the physical simulator,
RL reward, rollout, collection settings or source. Do not modify the frozen v1
or v2 code. Do not migrate or recollect data. Work only in a new small directory
`work/sdmpc_rl_export_recovery_20260930` and write export-recovery-report.md here.
No actual result edits/production export/run/physics/model loads/install/commit
by subagent. Synthetic tests may use isolated fake fixtures. Use apply_patch.

## Reproduced actual failure

The v2 wave ran five carry workers from05:11:46KST. At155terminal it failed at
collect.py:155 -> local_runtime.save -> json.dumps(...allow_nan=False), before
trace.json publication. Status error: ValueError: Out of range float values are
not JSON compliant: inf. All coordinator/children drained and exited. No user
STOP is known; old attempt ABORT is internal and must be preserved.

Saved control counts:155=75,170=34,incident=48,skew=51,190=42; total250real
transitions. Carry155totalTTT3210.204341 (perturbed TRAINING, not canonical).
The completed physical trajectory already passed validate_rows/boundary and
policy replay before serialization failed. All checkpoints remain authoritative.
Do not run155again, including terminal env.step/reset. Other four resume prefixes.

Read-only boot+checkpoint inspection found exactly4positive infinity leaves,
all in trace control_step14:
- candidates/0/rows/2/primal_stationarity
- candidates/0/rows/3/primal_stationarity
- candidates/0/rows/5/primal_stationarity
- candidates/0/stationarity

No other nonfinite leaf was found in that trace. Frozen solver uses positive
infinity for failed/unavailable stationarity checks; it is NOT convergence proof.
budget_controller selects execution through explicit physical/budget feasibility
and retains candidate audit separately. Preserve these diagnostics and never
change inf to zero or claim solver convergence. Actual source references:
work/sdmpc_rl_multi_20260929/budget_controller.py:115-184;
frozen historical_tree/src/controllers/sensitivity_dmpc.py:419,516.

## Small companion export helper

Import immutable v2 helpers read-only. Only operate at canonical v2 slots, after
proving coordinator/all numerical children dead (creation identities), no
STOP, and holding cohort/slot locks. Reject unknown/live identities. Authenticate
source/runtime/settings and validate the terminal checkpoint with existing
physical/per-transition/policy validators. No restore/reset/step/PFO is needed
after boot registers checkpoint classes. Require75true-terminal transitions.

Export strict valid JSON without changing the raw checkpoint or training
experience. Allow ONLY positive infinity at the two structurally exact candidate
stationarity paths: trace index/candidates/index/stationarity, or
trace index/candidates/index/rows/index/primal_stationarity. Encode each as an
explicit tagged diagnostic object, e.g. {"nonfinite_float":"+inf"}, and record
the exact changed paths/value/type in an export receipt. Reject NaN, -inf and
any nonfinite value elsewhere, including physical state/actions/rewards/TTT.
Finite values and all other trace structure must remain exactly unchanged.
The original checkpoint keeps the authoritative numerical inf values.

Re-run original validators/replay/summarize on the exported trace; prove the
same experience/settings/summary and physical outcomes. Existing load_completed
ignores candidate stationarity in numerical reconciliation, but this must be
tested, not assumed. Publish the normal trace.json,summary.json,experience.pt,
observation_schema.json,timing.json completion set without changing settings,
checkpoint, sessions or historical logs. Reuse exact schema/timing files.
No preexisting conflicting outputs may be overwritten; preserve staged data
on interruption and fail closed, with verifiable idempotent complete retry.

Record helper source hash, original checkpoint/settings/status/timing hashes,
known serialization failure, transformed diagnostic paths, output hashes,
new export elapsed time and its precise scope in a sidecar export receipt.
Keep the original failed status as evidence. Link receipt/hash from an additional
completion field (not an unknown entry in the fixed outputs_sha256 set).
Mark retained75step cohort identity finished using the existing locked helper
after proving dead workers, then publish completion LAST. New current status
may say export-completed, but archive its original failed status first. All
physical source/configuration/TTT/true terminals remain unchanged.

Do not rewrite old session timing. Post-run export work is a separate measured
cost, not zero and not silently included/excluded from a full runtime claim.
Document that paired old summaries retain their defined worker-session scope;
the parent must include the additional export cost separately in readout.

## Bounded continuation wrapper

Provide a small supervisor using the immutable v2 run_wave.py --resume. It must
wait on the launched coordinator/children, not start duplicates. On failure,
only the exact known terminal diagnostic-JSON failure above is auto-recoverable:
validate/export its saved75step checkpoint, then resume the same cohort. Do not
retry arbitrary failures, missing checkpoint, user STOP or UNKNOWN process state.
Preserve old attempt ABORT/logs. Never remigrate; completed slots are skipped by
the original runner. At most10terminal export repairs, each slot once; use an
exclusive supervisor lock, durable process/attempt/status and append logs. The
original runner keeps maximum5numerical workers, one thread each, within total8.
No PFO/model/evaluation or new numerical behavior in the wrapper.

Synthetic tests must cover allowlisted Inf tags, finite identity, forbidden
nonfinite physical data, missing/changed artifacts/source, terminal only,
rawcheckpoint/session preservation, original load_completed acceptance, actual
cohort75step retention, no extra step on export, complete retry/interruption,
wrong errors/live/UNKNOWN/STOP/duplicate supervisor, and bounded resume/skip.
Keep this a small export adapter, not a general scheduler or another collector
version/migration framework. Do not rerun the frozen236test suite; run covering
new tests and report exact evidence. The parent will review before any actual
export or coordinator resume.
