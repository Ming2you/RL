# Task 1 Independent SPEC and QUALITY Review

Reviewed 2026-09-30. Binding brief read first, followed by the exact plan,
implementation report, supplied diff, all 17 candidate Python files, and only
the frozen interfaces needed to assess those changes.

**SPEC: CHANGES REQUIRED.** The treatment and scientific gates conform, but the
new worker-exit wait does not preserve cohort STOP/failure propagation.

**QUALITY: CHANGES REQUIRED.** One actionable P2 finding below. This is not a
clean review and does not admit the real pilot. No other actionable finding
was identified within the requested scope.

## Finding

### R1 [P2] Propagate STOP and launcher failure before waiting for actual worker exit

Location: `work/sdmpc_rl_nuf_retention_20260930/run_wave.py:113` (call before the
exit-code check at lines 116-117), with the blocking loop at lines 53-71 and
ABORT publication deferred to lines 124-127.

If a Windows redirector exits while its claimed numerical worker remains live,
`run_stage` enters `wait_worker_exit` before acting on the launcher's nonzero
exit code. That loop polls only process identities. It neither checks cohort
STOP markers nor returns to poll the other children. Until the actual worker
exits, the coordinator cannot enter its `finally` block to publish the shared
ABORT. A STOP in another slot is particularly affected: workers check their own
slot/ancestor STOP paths and their attempt ABORT, not sibling-slot STOP paths
(`local_runtime.py:281-291`, `collect.py:132`). The orphaned worker and its
siblings can therefore keep executing subsequent intervals after a known
launcher failure or sibling STOP. A hung worker can leave that propagation
blocked indefinitely. This is introduced by the new blocking wait, not a
request to redesign the frozen process machinery.

**Read-only confirmation:** executed only the two AST-extracted function
definitions, `run_stage` and `wait_worker_exit`, with in-memory paths, reads,
writes, process handles, identities, and clock. No candidate-runtime import,
real process spawn, file write, simulator, or test-suite invocation occurred.
The first fake launcher returned exit 1 while its actual worker stayed live;
a sibling STOP became pending on the first fake wait tick. All three wait
ticks observed `abort_published=false`, with no further STOP check or sibling
poll. Only after the fixture made the worker dead at tick 3 was ABORT published
and the second launcher drained. The fixture deliberately bounded the wait;
production has no such bound.

**Required change:** preserve actual-worker draining and creation-identity
checks, but publish the attempt ABORT immediately on a known launcher failure.
Keep all-slot STOP and other-child failure checks responsive while waiting for
known live workers. Once aborting, continue draining every launcher and actual
worker; do not release live/UNKNOWN ownership or enable retries.

**Targeted regression needed:** an exited redirector with a still-live claimed
worker, followed by a sibling-slot STOP or another child's failure. Assert that
ABORT is published before the live worker exits, all siblings are drained, and
ownership remains retained until actual exit. Existing
`test_workflow.py:152-167` checks eventual worker exit/UNKNOWN handling in
isolation; it does not cover coordinator STOP/failure propagation during that
wait. The 69 existing tests were not rerun.

## SPEC Checks Otherwise Satisfied

- `exploration.py:172-190`: only NUF base retention and the consistent realized
  offset change. NP rebases/resets on the original events. Mean reversion,
  bounds, float32 conversion, rate/projector math, PCG64 draw ordering, and
  executed-action anchors remain the v2 implementation. The saved-trace test
  covers equality through step 9 and the step-10 request 5552.771855 versus
  5504.421625, without treating hypothetical later rows as physical experience.
- `collect.py:80-168`, `validate.py:78-132`: normal reset or retained full
  checkpoint restore; one action into unchanged BudgetEnv per interval; actual
  physical execution, anchor chain, reward/accounting, 75 intervals, true final
  action/terminal and 14400 seconds. No extra preview/PFO, model load, training,
  performance guard, canonical evaluation, or terminal export retry was added.
- `references.py:49-138`, `reference_check.py:9-28`: pinned external carry/local
  completions, original-validator authentication in a fresh interpreter, eight
  repaired receipts and their linked inputs, paired profile/seeds/config/source
  binding, and distinct new run IDs. No raw reference environment deserialization.
- `diagnostics.py:26-65`: source-pinned pure tagger admits only the original typed
  stationarity +Inf paths, retains raw checkpoint values, rejects unknown
  nonfinite trace values, and reconciles the diagnostic receipt and finite core
  experience. `validate.py:197-221` requires the complete audited output set.
- `probe.py:37-110`, `run_wave.py:143-221`: exact all-75 noise checks, physical
  and native-transition prefix through step 9 at absolute 1e-8, first action
  divergence at 10, explicit pilot thresholds, archived invalid attribution,
  then only four remaining slots with unchanged candidate identity. Completed
  pilot collection is skipped; external carries are never dispatched.
- Five retained local slots, monotonic progress, source/run binding, locks,
  creation identities, failed/UNKNOWN retry refusal and duplicate refusal are
  present. R1 is the exception to preserved STOP/draining behavior. Stage widths
  are one and four single-thread workers. The report correctly leaves accounting
  for unrelated numerical work under the global <=8 ceiling to the parent.
- Worker timing retains the admitted session scope and includes diagnostic
  serialization; reference verification and overlapping coordinator/readout
  timing are separately described. Readouts include TTT, inventories, budget
  drift/headroom, fallback/closure/control diversity and Q absence, with no
  learner admission or policy-improvement claim.

## Evidence Checked Read-Only

- Supplied diff SHA256:
  `2294dfbe1bc2d08ae619cca58b6735e4d5719f07588c12d00892b29f29170de2`.
  All 17 current Python Git blob IDs match its new-file entries. All 17 file
  hashes also match the recorded tested sources; all 18 candidate identity
  source entries match current files.
- Evidence `work/sdmpc_rl_nuf_retention_20260930/test-evidence/887f768a/evidence.json`:
  `5889180247aeda265acb71ca3ab8ebe749b0cf2037500c31c5c7588c106d89bf`.
  JUnit `tests.xml`:
  `e4b636590bc078023750a612da1ed3872c4f8691dac4b4b3841ea5089fa2173d`.
  Parsed 69 cases, 0 failures, 0 errors, 0 skipped, 45.495 seconds; counts are
  collection 21, policy 11, probe 17, workflow 20. XML records the successful
  original-validator test for ten reference runs and eight repaired receipts.
- Rehashed all 202 preserved source/result entries: zero mismatches. Root
  completion remains `5b6e645f6f6ca64c4f593c52d19ee3ab5bda7ae3c3d31f423d6fae2d2f3ef4a3`;
  comparison remains `c294c1f5b2100daeaccb4c927357a8bc73d44391f34641fd37b4ba5b2555c4cd`.
- Rehashed the pinned snapshot manifest and 151 payload files, 12 frozen
  implementation sources, physical-contract gate and its two admitted validator
  sources: zero mismatches. Both copied inventory/launch helpers remain
  byte-identical to v2.
- No pytest suite, production entrypoint, reference-authentication subprocess,
  physical run, model load or training was executed during this review. The
  isolated AST check above is additional review evidence, not a rerun of the
  recorded tests. Existing mocks establish bookkeeping, not a live numerical
  outcome. Current runtime package loading and actual Windows child execution
  were not re-exercised.
- `results/sdmpc_rl_balanced_goal_20260930/nuf_retention_v1` is absent. Only this
  review ledger was written; candidate code, frozen sources/results and unrelated
  dirty work were left intact. No commit.

Parent disposition: fix R1 in the candidate, add the focused regression, refresh
the changed-source evidence/diff, and obtain a clean independent review before
authorizing the single 170 pilot. No pilot performance conclusion is available.
