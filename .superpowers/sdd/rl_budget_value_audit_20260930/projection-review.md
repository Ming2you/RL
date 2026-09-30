# Projection/replay audit review

SPEC: FAIL

QUALITY: FAIL - changes required before production execution.

Scope: the supplied `projection-review.diff`, containing only `projection_audit.py` and `test_projection_audit.py`, against the scoped brief and stated base `c40eb9de19ee780d08a7e3e91b40c6cdd4807a92`. Dependencies were read to establish behavior; findings below belong to the new audit and its tests.

## Actionable findings

### 1. [P1] Recheck every consumed input identity before declaring completion

Location: [projection_audit.py:174](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/projection_audit.py:174), final verification block at lines 174-178; collection identities are only appended at line 170.

The final checks cover the frozen/implementation source pins, final model, runtime versions, and two diagnostic sources. They never rehash the consumed collection files, the round-0 predecessor model, or the pilot completion record. In particular, `run["identity"]` is saved without being revalidated, and it does not include the trace used at line 158. A collection experience, trace, settings file, or predecessor model can change after loading while this audit still writes `status="completed"` and prints `PROJECTION_AUDIT_PASS`. This violates the explicit before/after identity requirement and leaves the recorded result associated with stale input identities.

Action: capture an input manifest before consuming the artifacts and recheck it immediately before publishing completion. Include both models, the pilot completion record, and every consumed collection file: completion, settings, experience, trace, summary, observation schema, and runtime versions. Persist the manifest with the result. Add an isolated orchestration regression that changes a fixture input after loading and verifies that completion is refused; do not invoke the physical runtime in that test.

Dependency evidence: [train_round.py:65](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:65) records only completion/settings/experience hashes in each returned identity; `load_collections` performs no later verification on behalf of its caller.

### 2. [P2] Bind admitted collection provenance to the authenticated final model

Location: [projection_audit.py:147](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/projection_audit.py:147), collection admission at lines 147-151 and replay comparison at lines 163-165.

`load_collections` establishes internal consistency, expected seeds/scenarios, and matching contracts. The audit then compares transition tensors, but discards `run["provenance"]` and never compares it with `model["training_profiles"]`. Thus a consistently repackaged collection with the same transition values but a different run ID and experience hash passes the audit, even though it is not the preserved collection admitted by the hash-authenticated model. Exact replay values establish membership; they do not authenticate the collection's historical identity.

Action: compare each round's ordered collection provenance with the corresponding entries in the authenticated final model's `training_profiles`, including scenario, seed, profile hash, run ID, and experience hash, before using the collections. Add a fixture case with unchanged transitions and consistently changed collection provenance that must be rejected. This check is separate from finding 1: an unchanged replacement throughout the audit must also fail admission.

Dependency evidence: [train_round.py:66](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:66) already returns these provenance fields; the existing training validator compares them to `training_profiles` at [train_round.py:154](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:154). The shared base loader does not perform this comparison.

### 3. [P2] Revalidate overwrite and STOP guards while holding the output lock

Location: [projection_audit.py:119](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/projection_audit.py:119), preflight and lock acquisition at lines 119-126.

The directory-emptiness check occurs before `exclusive_run`. Two invocations can both pass it; one can finish while the other is descheduled before lock acquisition. The delayed invocation then acquires the released lock and overwrites the first invocation's artifacts, because nothing inside the lock checks that the directory is still unused. The shared `save` helper replaces existing destinations. A STOP created between preflight and lock acquisition likewise does not prevent the initial writes and runtime boot.

Action: under the acquired lock, recheck STOP and reject any existing output entry other than the lock file owned by that context before writing `process.json` or loading the runtime. Retain the early checks for fast refusal. Add a controlled lock-context fixture that introduces an existing completion file or STOP immediately before yielding and verify that the audit refuses work and preserves existing files.

Dependency evidence: [run_budget.py:25](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_budget.py:25) only creates/acquires `runner.lock`; it does not enforce output emptiness.

### 4. [P2] Require distinct nominal actions after float32 conversion

Location: [projection_audit.py:99](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/projection_audit.py:99), conversion and equivalence check at lines 99-103.

Candidate uniqueness is checked only before conversion. The post-conversion check proves that requests agree, but allows all candidate actions to become the same float32 action. Static counterexample, not executed: with `anchor=[0., 5000.00012]` and `capacity=6000.`, the upper-clipping threshold is approximately `0.99999988`; the generated second components are approximately `0.99999998`, `0.99999999`, and `1.0`, all of which round to float32 `1.0`. The audit records three copies of one critic input as an alias group with zero spread, inflating alias coverage and biasing the mean spread. Distinct representable clipped actions do exist below `1.0` in this example.

Action: construct/deduplicate candidates in the actual float32 critic-input domain, group their actual `residual_budget` outputs by exact equality, and emit only groups with at least two distinct float32 actions. Generate representable clipping-boundary candidates so the conversion does not silently discard available comparisons. Extend [test_projection_audit.py:57](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/test_projection_audit.py:57) with this boundary case and assert distinct logged actions as well as equal projected requests.

## Evidence and scope limits

The supplied evidence reports **13 focused tests passing in 1.35 s** using the `.deps-budget` pytest path. That result is accepted as supplied and was not rerun. The tests cover sequence validation, an ordinary upper-cap alias case, twin-critic logging/nonmutation, STOP path detection, and schema lookup; they do not exercise `main()` admission/final verification, lock ordering, or float32 candidate collapse.

Static inspection supports the intended replay-only path: all five scenarios and both rounds are iterated, the shared base loader requires 150 replay entries per scenario, and each saved transition is compared exactly after conversion to the final replay dtype. The audit checks clock, horizon, terminals, memory, reward, and projection, logs both critics and their spreads, and states the limits of the experiment. No training update, simulation step, policy export, or canonical evaluation-observation path is called by the audit.

No production diagnostic, tests, model inference, simulation, training, or commits were executed during this review. No implementation or preserved artifact was modified; only this review report was written. The numerical counterexample above is source-based reasoning, not a claim about an executed pilot result.
