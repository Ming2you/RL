# Integrated shared-Q review

Reviewed: 2026-09-29. Consolidated initial code review, including completed Task 2
synthetic execution evidence. Final admission remains a separate review.

CODE_REVIEW: FAIL / changes required

SPEC: FAIL / changes required

QUALITY: FAIL / changes required

MULTI_REVIEW: PENDING / final admission not issued

The normal collection/training path implements one shared actor and twin critics,
five equal collection groups, and centralized balanced updates. Six integration
findings remain: one P1 and five P2. They concern admission, resumed state,
completed-output validation, and STOP handling. They do not repeat Task 1 R1-R3.

## Scope and evidence

Read the plan, Task 2 test brief, conceptual integration-version diff, current
production modules, shared learner interface, current synthetic test definitions,
and read-only analysis script. The directory-version diff is not a deletion plan;
the prior carry directory remains present. Recomputed hashes confirm that
budget_controller.py, budget_env.py, budget_runtime.py, and freeze_runtime.py are
identical between carry and multi versions.

Task 1's [separate fix-round re-review](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-1-rereview-round1.md)
records independent SPEC/QUALITY PASS for R1-R3 and accepts the implementer's
211 passing learner tests. That execution evidence was not rerun here.

The completed [Task 2 test report](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-2-test-report.md)
reports **140 collected: 133 passed, 7 failed in 33.99 seconds**, exit code 1.
Its seven retained failing regressions reproduce four defect categories. The
reported production hashes match this review's hashes, rechecked after receipt.
These are accepted execution results from the test owner, corroborated by source
inspection and the focused probes below; this reviewer did not rerun that suite.
No actual five-scenario smoke evidence has been supplied.

| Consolidated finding | Test-report category | Reported failing variants |
| --- | --- | --- |
| I3: restored trainer contract | D1 | seed, sample_counts, replay (3) |
| I4: completed model provenance | D2 | provenance (1) |
| I5: STOP during launch | D4 | stage, child (2) |
| I6: completed experience before siblings | D3 | malformed experience (1) |

I1-I2 are additional admission findings. I1 also incorporates the coordinator's
cached-gate concern; the empty-evidence acceptance was independently reproduced.
These six IDs are the consolidated scope for one fix dispatch.

Two focused, in-memory probe commands completed successfully in the existing
.venv-torch interpreter, one at a time, with bytecode disabled and numerical
thread environment variables set to one. No traffic/runtime boot, actual child
process, gradient update, source edit, or persistent test artifact was involved.
The only persistent write from this review is this report. An initial attempt to
import synthetic fixture helpers stopped before probing because sandbox-visible
pytest lacked `mark`; the successful retry loaded only fixture definitions with
AST, without importing pytest or executing tests. No reported suite was repeated.

## Findings

### I1 [P1] Revalidate required evidence when consuming the admission gate

Location: [build_preflight.py:34](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/build_preflight.py:34), especially lines 36-40; consumed at [run_pilot.py:190](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_pilot.py:190).

`verify_gate` accepts a ready flag, current source pins, and an empty
`evidence_sha256` dictionary. The empty `any(...)` check passes, so this can start
the pilot with no test XML, review, or smoke results. The in-memory probe returned
`accepted: true` for exactly that object. A nonempty arbitrary evidence list also
does not establish that the required eight evidence roles exist or were valid.

Even for a builder-generated gate, consumption only hashes the evidence files:
it does not repeat the current test-source comparison at lines 58-59. Editing a
test after gate creation leaves both production pins and the saved record's hash
unchanged, so the existing gate remains accepted against different tests.

Require a versioned gate schema and the complete typed evidence set, and run the
same source/test/XML/smoke/review validation on consumption. Reject empty,
incomplete, or arbitrary evidence lists and changed current test sources before
any child starts. Add negative tests for these cases; the current lifecycle test
fixture itself uses the accepted empty-evidence gate at test_multi_pilot.py:22.

### I2 [P2] Bind review approval to the source and executions it reviewed

Location: [build_preflight.py:65](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/build_preflight.py:65), with gate construction at line 70.

The review check searches only for a success substring. It does not check which
source pins, test-source hashes, test XML, or smoke artifacts the reviewer
approved. A previous successful review can therefore be combined with freshly
generated evidence for changed code and produce a new admission gate. Hashing
that old review into the new gate prevents later editing of the review, but does
not establish that it reviewed this version or these executions. A quoted or
superseded success line is also sufficient.

Use an unambiguous final verdict with a machine-checkable attestation of the
reviewed source and evidence identities, and compare those identities with the
candidate gate inputs. Test stale-source approval, stale-evidence approval, and
conflicting/superseded verdicts. Keep code-review approval distinct from final
admission approval.

### I3 [P2] Reconcile restored trainer replay with the admitted collections

Location: [train_round.py:151](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:151), particularly lines 155-162.

The trainer first builds replay from validated predecessor/current collections,
then replaces it with `ck['learner']`. After replacement it checks only the update
count against `initial_updates + completed` and the metrics length. The general
TD3 loader intentionally permits other valid seeds and arbitrary valid replay
contents; it cannot enforce this experiment's round-specific provenance.

A real-TD3 in-memory resume probe reached the next `update(40)` call with each of
these independent changes to an otherwise matching zero-update round checkpoint:
(a) seed 99 instead of 6300; (b) 74 transitions in one scenario instead of 75,
including loss of that episode's terminal; and (c) unchanged 75-per-group counts
but a reward changed to -999 and the final terminal changed to false. The update
was intercepted before executing any gradients. The latter corruption can retain
all final counts while changing the learned targets and bootstrapping across the
round boundary. Replacement with same-shaped evaluation data is likewise not
excluded by this restore boundary.

The integration owner's D1 additionally reproduces the sampling-counter case
with one real TD3 update and valid optimizer history: changing each scenario's
cumulative samples from 8 to 7 reaches another update. This confirms that the
missing check is the trainer's exact batch/round contract, not Task 1's generic
optimizer-history validation.

Before any resumed update, compare restored replay, in its stored float32 form,
with the exact validated predecessor plus current collections. Enforce seed,
round-specific counts and cumulative sampling totals, and validate every saved
metric's phase/sample record. Preserve the valid no-readdition continuation path.
This is an integration ownership requirement, separate from Task 1 optimizer and
generic counter validation. Reported failing regression anchor:
[test_multi_runner.py:400](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_multi_runner.py:400).

### I4 [P2] Match completed model provenance to the trainer's actual inputs

Location: [run_pilot.py:50](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_pilot.py:50), through line 58; permissive metadata validation at [run_budget.py:95](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_budget.py:95).

Completed-training validation reconstructs the actual collection identities, but
never compares their provenance to `model['training_profiles']` or compares
`model['training_run_id']` with `settings['run_id']`. The policy validator checks
only nonempty fields, unique scenario/seed pairs, and equal profile counts. A
different collector hash, profile hash, seed schedule, or training run can be
accepted as long as those structural properties and learner counts remain valid.

A metadata-only probe accepted both an unrelated training_run_id and an unrelated
experience_sha256 in an otherwise matching model/summary. The probe stubbed the
core learner loader to isolate this orchestration check; no claim about core
restore execution is made by that probe. The integration owner's D2 independently
confirms acceptance of the wrong experience hash using a real completed two-round
TD3 checkpoint and the actual learner loader. Compare the exact ordered
predecessor provenance plus this round's
five validated provenance rows, require the training run identity, and reconcile
the model replay against those inputs. Merely matching the file hash recorded in
the completion is not an input-provenance check. Reported failing regression anchor:
[test_multi_pilot.py:251](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_multi_pilot.py:251).

### I5 [P2] Recheck stage and child STOP files during launch and draining

Location: [run_pilot.py:123](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_pilot.py:123), particularly line 124; polling at line 144.

The preflight loop checks all STOP scopes once, but the launch loop subsequently
checks only the pilot root. Creating a stage STOP or the first child's STOP after
the first launch still starts the remaining four workers. Controlled fake-Popen
probes reproduced five launches for both locations; only one should have been
started. A child's STOP is not propagated to the root until that child exits, so
siblings can keep doing numerical work while it finishes its current interval.

Recheck all relevant STOP scopes before each launch and on each polling cycle;
promote any detected stop to the shared root and drain already-started children.
Retain cooperative checkpoint/exit behavior and the five-worker cap. Both STOP
locations also fail in the integration owner's D4. Reported regression anchor:
[test_multi_pilot.py:181](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_multi_pilot.py:181).

### I6 [P2] Validate completed experience payloads before launching siblings

Location: [run_pilot.py:73](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_pilot.py:73), used for skip acceptance at lines 120-128; hash-only experience check at [compare_runs.py:144](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/compare_runs.py:144).

The shared completed-run validator verifies the experience file hash but never
loads or validates its payload. A completion with a matching hash for a malformed
payload is accepted for skipping, and missing siblings are launched. Payload
settings/contract, transition count, adjacency, requested actions, rewards, and
terminal flags are checked only later by `train_round.load_collections`.

An in-memory probe accepted a completed collector whose declared matching payload
contained 74 transitions; `torch.load` was never called by `validate_job`. Passing
the same payload to `verify_experience` correctly rejected it. Thus training is
eventually blocked, but only after unnecessary sibling collection, contrary to
the explicit reject-before-launch requirement. Share the existing experience
validation at the completed-collector acceptance boundary and retain the trainer
check. The integration owner's D3 additionally confirms that all four missing
siblings launch and the stage returns success. Reported regression anchor:
[test_multi_pilot.py:227](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_multi_pilot.py:227).

## Positive assessment and limits

- No private per-scenario training call was found in run_budget.py. Local TD3
  instances serve frozen inference; experience is persisted only in collect mode.
  Round 0 uses 32 uniform actions then Gaussian 0.3 exploration; round 1 loads the
  shared model and uses Gaussian 0.3 exploration. Canonical evaluations have no
  training demand seed and follow the no-exploration branch.
- Five distinct collection directories, scenario IDs, scheduled seeds, source and
  runtime identities, model hashes, common contracts/schema, complete traces and
  sequential experience are checked on admission to a fresh training round.
  The shared learner draws eight samples per scenario per 40-sample update and
  masks continuation at true terminals. New episodes are not intentionally joined.
- The fixed job schedule contains ten 75-transition collections, two isolated
  375-update training stages, five center runs, and five final-model evaluations.
  Normal scheduling launches at most five numerical children; inherited BLAS
  settings, runtime affinity, and TD3's one-thread setting are present. No normal
  path to more than five numerical workers or concurrent learner/collector work
  was found. Real execution/affinity evidence is still pending.
- Source/runtime/profile/model drift is checked on ordinary episode resume.
  The checkpoint retains environment and exploration RNG state; model inference
  is reloaded from the immutable policy. Output locks, validated completion skips,
  cooperative failure draining, and terminal finalization without a new step are
  present, subject to the findings above.
- Full-run trace validation covers 75 decisions, the 14,400-second endpoint, true
  terminal timing, reward/TTT and area accounting, physical/budget validity,
  disabled H3 gating, carry reference/PFO semantics, and decision-time components.
  The unchanged physical modules preserve initial/recovery PFO and carry behavior.
- Smoke explicitly claims two actual intervals with serialized reset and carried
  restore, not a full episode. Training Q diagnostics explicitly distinguish
  exploratory behavior returns from current-policy Q ground truth. The analysis
  script uses realized canonical returns under the same frozen policy, notes TD3
  target smoothing, reports per-scenario diagnostics and the worst relative
  improvement, and disclaims generalization. No unsupported performance or causal
  improvement claim was found in these normal output paths. These are implemented
  diagnostics, not yet observed outcomes.

## Evidence required before final admission

1. Scoped fixes and independent re-review of I1-I6, without changing the retained
   physical modules or reopening Task 1 R1-R3 absent a new regression.
2. A post-fix integration report and one full passing synthetic test run bound to
   settled source and test hashes, resolving the current seven failures and adding
   admission regressions for I1-I2. Include actual shared TD3
   round updates, deterministic episode/trainer continuation, malformed restore
   and provenance rejection, terminal handling, evaluation separation, completion
   reuse, STOP at all scopes, launch failure/draining, and the five-worker bound.
   Prove required suites actually ran; a minimum count alone is insufficient.
3. Five distinct real scenario smokes against those same settled sources, with
   canonical profile identity, one common observation schema, physical/budget
   validity, initial PFO then ordinary carried reference, and serialized restore
   parity. Record runtime versions so test/smoke evidence can be reconciled with
   the intended pilot runtime. Synthetic rows do not satisfy this requirement.
4. A final independent admission review explicitly bound to the current source,
   test and smoke evidence, followed by gate construction and verification. This
   document supplies no final admission approval.

After admission, the bounded full pilot still must reconcile 150 transitions and
two true episode terminals per scenario, 750 shared updates, 6,000 cumulative
sample draws per scenario, ten canonical evaluation runs using the same final
model, full TTT/inventory/fallback/saturation/zero-ramp/interval/runtime diagnostics,
and per-scenario outcomes. Record why a fresh 170-incident center is needed if
the existing center cannot satisfy equivalence. These are later execution and
reporting obligations, not evidence already supplied by this code review.

## Reviewed source identities

Paths below are relative to the RL repository for compactness. SHA256 values:

```text
work/sdmpc_rl_multi_20260929/run_budget.py
7524B335458753C15D7894E872F6BCC80056B7CC4C3756CF7728ABDEB48E985E
work/sdmpc_rl_multi_20260929/train_round.py
23B7A49D94D60F2F37F8A525180F442781FDC99463B893B6095FB25483F940A0
work/sdmpc_rl_multi_20260929/compare_runs.py
8295D41C2A1E4CDE787D5C81C291A9068070F02EFBCE5B16C8E7E9C5BD7C8906
work/sdmpc_rl_multi_20260929/run_pilot.py
9FD8CFA8766AB938CB1F3E289C02A801A981203715D0CD63BC08C3E583FB72DC
work/sdmpc_rl_multi_20260929/build_preflight.py
5D33DAC912DF086C2C3AE633DB93C99D68E693F90B721A794F60390D1408CC6C
work/sdmpc_rl_multi_20260929/run_tests.py
8B0453ADD40E8512736DFC141756BD6651080939E00FCB5620464001657BF5A6
work/sdmpc_rl_multi_20260929/smoke_budget.py
466517E55B2A8000BEDC7B4CBD58BAF3E9A3166F63CAA200E0A80E25F74A3D1D
work/sdmpc_rl_multi_20260929/td3.py
E27665133CCB3FAC2B1769DA1C93DF73A60C73572FA4CCEDE24A1296292E2E8E
work/analyze_sdmpc_multi_20260929.py
F7E15C205B429A97DA267E3A7698E4714048716251C6AD3FA9FED691FF28888B
```

No production/test code was modified, no traffic was run, no agent was spawned,
and no commit was made.
