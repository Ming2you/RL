# Independent implementation review

Reviewed 2026-09-30 against both task briefs/reports, the supplied
`implementation-review.diff`, and
`docs/rl_budget_local_collection_plan_20260930.md`. Findings below are static
source deductions, not executed fault-injection results. Paths are relative to
the RL workspace; line numbers identify the reviewed source.

| Task | SPEC | QUALITY | Production disposition |
| --- | --- | --- | --- |
| Exploration | PASS | PASS | Pure component approved within its brief; no actionable finding. |
| Collector / paired readout | FAIL | FAIL | Fix the findings below and review the changes before the first carry prefix smoke. |

The parent reports combined **143 PASS in 8.47 s, exit 0**, with source unchanged
since the diff. Earlier reports give 121 exploration and 22 collector passes.
These results are acknowledged, not independently rerun. They do not cover the
specific gaps below. No production collection or policy admission is established.

## Collector findings

### R1 [P1] Bind collection ownership and duplicate prevention to the fixed cohort

**Locations:** `work/sdmpc_rl_local_20260930/run_wave.py:104`,
`run_wave.py:154`, and `work/sdmpc_rl_local_20260930/collect.py:17`.

Every coordinator/child lock and completed-output check is scoped to the supplied
output directory. Both CLIs accept another directory; neither checks the fixed
cohort elsewhere. Running two coordinators with different `--output` roots admits
the same prescribed scenarios, seeds and behaviors twice and permits two sets
of five physical collectors. Likewise, a completed scenario/behavior can be
recollected at another path. New UUID run IDs do not make that new authorized
coverage. This violates the maximum-five collectors, ten-episode bound, and the
instruction to reject already completed collection under this contract.

**Required change:** Enforce the declared collection root, or maintain a shared
cohort reservation keyed by contract/scenario/demand seed/behavior independently
of output path. Coordinate standalone smoke workers with the same ownership and
worker limit. Retain reservations/completion identity across coordinator loss.
**Regression to add:** Two distinct output roots and a standalone child must not
admit duplicate cohort work; an already completed key must remain rejected after
changing only the destination. Existing per-directory lock tests do not test this.

### R2 [P1] Apply observation and budget-history checks before resuming a prefix

**Locations:** `work/sdmpc_rl_local_20260930/collect.py:38` and
`work/sdmpc_rl_local_20260930/validate.py:49`.

Checkpoint validation always passes `complete=False`, so it skips the only
validator that binds observations to clock, remaining horizon, action anchor,
previous requested/executed budgets and `B_raw`. The remaining checks cover
shape/finiteness, adjacent observations, actions/rewards and the final simulator
clock/TTT. For example, change only the first stored observation's clock or
remaining-horizon component in a one-step checkpoint. Its last next-observation,
policy state, trace and simulator remain unchanged: resume accepts it and can
perform the remaining 74 expensive intervals before full validation rejects it.
Behavior replay reads trace anchors, so it cannot detect this observation fault.

The reused old `projection_audit.py:62` requires exactly 75 transitions; its
clock/memory checks at lines 79-93 are not a prefix validator. Calling it only
at completion therefore does not satisfy strict sequential resume validation.

**Required change:** Add equivalent prefix-compatible validation in the new
wrapper, including the saved boundary observation and previous-executed anchor;
retain the old complete-episode validator unchanged. Reject inconsistent history
before another physical step. **Regression to add:** Independently corrupt clock,
remaining horizon, budget memory, anchor and raw projection in a short checkpoint
and verify rejection before `step`, including the initial and terminal boundaries.

### R3 [P1] Reconcile terminal inventory with physical state before issuing the screen

**Locations:** `work/sdmpc_rl_local_20260930/validate.py:64`,
`validate.py:115`, `validate.py:147`, and
`work/sdmpc_rl_local_20260930/collect.py:39`.

The inventory screen trusts `trace[-1]['inventory']`. The completed loader only
checks that the copied summary value is finite/nonnegative and equals the summary
reconstructed from that same trace. Neither the new validators nor the imported
sequence/queue validators reconcile inventory with `plant_state` or the saved
simulator. This has a direct resume path: after the final interval checkpoint but
before completion publication, replace its terminal trace inventory with zero.
Policy replay and simulator clock/TTT checks still agree; resume at `k == 80`
executes no further interval, builds fresh matching output hashes, and accepts
the fabricated inventory in the paired screen. Hashing the resulting files does
not detect an inconsistency already present in the resumed checkpoint.

The old environment records both `plant_state` and inventory at
`work/sdmpc_rl_multi_20260929/budget_env.py:221`. Its terminal restore returns an
all-zero observation at line 320, so observation equality does not authenticate
terminal physical state or inventory.

**Required change:** Reconcile inventory using the frozen accounting definition
and saved physical state, and bind the checkpoint simulator boundary to the last
trace state/control/accounting before completion. **Regression to add:** A
terminal-checkpoint inventory/state mismatch must fail before a completion marker
or passing screen is emitted. No plant simulation is needed for that regression.

### R4 [P2] Bind each paired-readout slot to its expected scenario and behavior

**Location:** `work/sdmpc_rl_local_20260930/validate.py:137`.

`compare_pairs` infers the two roles and scenario from directory names but never
checks those expected values against loaded settings. `load_completed` checks
membership in the allowed sets, not equality to the requested slot. Swapping two
complete scenario pairs preserves all hashes, run-ID uniqueness and within-pair
profile equality, yet the outer `scenario` labels are wrong. Swapping carry/local
directories reverses the comparator and can change the screen outcome without
altering any authenticated artifact. The returned summaries can contradict the
outer labels while the result still says `reconciled` and 150/scenario.

**Scope/mitigation:** `run_wave.validate_job` checks these identities on the normal
coordinator path. This finding concerns the paired-readout function itself,
including independent post-run reconciliation; it is not a claim that the normal
coordinator skips its existing check.

**Required change:** Validate scenario, behavior and prescribed seeds/mask against
each requested slot inside the readout, or share the same job validator without a
circular import. **Regression to add:** Unmodified completed directories swapped
by scenario or behavior must be rejected by `compare_pairs` directly.

### R5 [P2] Preserve completed session time across checkpointed resumes

**Locations:** `work/sdmpc_rl_local_20260930/collect.py:103`,
`collect.py:128`, and `work/sdmpc_rl_local_20260930/validate.py:80`.

The saved elapsed counter is sampled before `checkpoint_save` serializes and
replaces the checkpoint. A prefix then writes status/log output and returns
without persisting its final session duration. Resume takes the earlier counter
as `elapsed_before`, permanently omitting the preceding session's final checkpoint
write and exit bookkeeping. Repeated short prefixes accumulate that omission;
failed work after the last checkpoint is also absent. This contradicts the stated
aggregate-worker-session scope, which explicitly includes checkpoints, and the
plan's full measured runtime requirement.

**Required change:** Keep durable per-session timing alongside the interval
checkpoint and reconcile it on resume/completion. Distinguish unknown time after
an abrupt process loss rather than presenting the checkpoint counter as full
elapsed time. **Regression to add:** With a controlled clock and nonzero checkpoint
serialization cost, resumed and uninterrupted timing accounting must include all
completed session work within the declared scope.

## Exploration and cross-interface assessment

- The local formula, signed NP, NUF-only projection, rate limits and second
  projection after float32 action rounding match the authoritative
  `budget_controller.py:11` transform. The audit distinguishes desired and actual
  projected requests. Carry draws no noise and emits exact float32 zeros.
- Commit checks the actual anchor/action/request and rebases only after physical
  fallback or PFO recovery. Initial PFO alone and TTT/price/Q diagnostics do not
  trigger rebasing. State copies, pending choices, seeded RNG reconstruction and
  failure-atomic loading follow the exploration brief. The collector checkpoints
  after commit and reconstructs the committed behavior state from the trace.
- The gate file independently hashes to
  `b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3`.
  Its metadata agrees with the old physical guard, previous-executed anchor,
  [50,1000] scales, reward divisor 100, gamma 1, schema and five-scenario order.
  The new identity wrapper pins old sources/runtime and imported validator
  sources without loading a learner. No new action-transform mismatch was found.
- New training records deliberately use `sdmpc-local-training-v1`; the old
  `train_round.load_collections` and `(scenario, seed)` uniqueness rules cannot
  consume the paired behavior cohort unchanged. There is no automatic learner
  export here. A later learner needs a separately reviewed provenance adapter.
- Within one cooperating coordinator, the carry/local barrier, per-child locks,
  STOP checks, per-attempt ABORT draining, completion-last publication and
  prefix-not-complete behavior are present. R1 is the missing cohort-wide guard.
  The orphan preflight only probes currently held child locks; it does not consult
  persisted PID/creation/command identity or cover a spawned child still importing
  before lock acquisition. Include that startup window when repairing ownership.
- The five physical-control fields match the frozen `ControlAction` actuators;
  budget labels and diagnostics are correctly excluded from diversity counts.
  The numerical screen uses the prescribed +5 zero-NUF and 1.25*max(1,carry)
  inventory thresholds and requires all five rows. Completion explicitly disclaims
  canonical evaluation, policy admission and goal achievement. R3/R4 concern the
  evidence feeding that screen, not the threshold formula.

## Evidence and limits

All seven new Python files match the supplied diff line-for-line. Diff SHA-256:
`8a793d81600fe08526bd947f4bfb9956ab5a5c9538ef5270af256312e57edde5`.
Read-only inspection covered the necessary old runtime, controller, environment,
checkpoint/lock helpers, comparison/sequence validators, trainer input contract,
and frozen control definition. It did not inspect the unrelated dirty branch.

The collector suite replaces the physical environment, identity/preflight, schema,
sequence validator and queue validator (`test_collection.py:74`). Its resume test
compares exploration audits, not full physical checkpoint equivalence. Its paired
test is a valid-fixture happy path. Reported passes therefore do not establish
the missing boundary/identity/accounting checks or real physical reproducibility.

No tests, model loads, environment boots, simulations, network operations, source
edits, commits, model overrides or collector launches were performed here. Only
this review file was written. Runtime execution and physical smoke remain
unverified; the parent should resolve these findings, then perform the authorized
first carry-155 prefix and continue that same episode after review. The original
all-five canonical improvement and reproduction goal remains active and unmet.
