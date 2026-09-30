# TD3, Sequential Runner, and Comparison Review

## Dynamic Configuration Repair and Re-Review Handoff

2026-09-29: Addressed the runner portion of the sole blocker in
`final_pilot_review.md`. This follow-up supersedes the earlier production hashes
and focused-test count below. The independent final review remains HOLD until
its owner re-reviews the coordinated repairs; no gate or launch was performed.

**SPEC: PASS for the scoped runner hash repair; ready for re-review.**
**QUALITY: PASS for focused synthetic verification: 25 passed in 13.44s**,
exit code 0, plus scoped `git diff --check`.

The only production change made in this follow-up is two expressions in
`run_budget.py`: config and lower-option hashes now use
`digest(rt['rc'].to_plain_dict(...))`. This reuses the existing frozen serializer
already used for protocol validation and preserves public runtime-added
attributes recursively. No generic serializer/helper was added. Aristotle
confirmed the same serializer is used by his version-2 environment contract and
saved-simulator/current-runtime checks; his reported 22 passing wrapper tests
were not rerun here.

Added two regressions to `test_run_budget.py`, exercising actual runner settings
construction and the real frozen `historical_config.to_plain_dict` without
booting its traffic runtime. Before the repair, both tests failed: dynamic-field
changes kept the old hash and resume accepted the drift. After repair:

- Identical configuration produces identical hashes.
- Changing `network.terminal_zero_gradient`, nested
  `mpc.leader_rollout_box_walk`, or a public runtime-added lower-option flag changes
  the corresponding config/options hash while preserving the unrelated hash.
- Dynamic config drift refuses resume without modifying the original metadata,
  and rejects a policy before mutating the learner.
- The complete focused comparison/runner/pilot suite still passes, including
  requested-action replay, frozen evaluation, resume boundaries, and lifecycle
  acceptance checks.

`run_budget.py` SHA256:
`6db11e238c1258ff654ef16657313e38e3d96a0c24c3d65d1d7213c6338ad950`.
Git blob: `01acd051be61d339abb9e81b13181f18e68a5593`.
`compare_runs.py` and `run_pilot.py` remain unchanged from the final review,
verified by their SHA256 values. Only `run_budget.py`, its owned test file, and
this report were edited. No active simulation, training, snapshot/legacy edit,
commit, or new infrastructure was introduced. The verification command in the
previous follow-up below now runs the 25-test focused suite.

## Scoped Fix Follow-Up

2026-09-29: Implemented the authorized fixes. All three production runner files
are stable and ready for the parent's consolidated pytest/JUnit, final review,
and subsequent source-pin freeze. No simulation or pilot was launched.

**Current SPEC: PASS for the scoped runner/comparison contracts.** All five
correctness findings below are addressed. The comparison now includes final
freeway/urban TTT and endpoint-sampled near-capacity estimates supplied by the
environment owner. These estimates are explicitly not exact substep exposure.

**Current QUALITY: PASS for focused synthetic verification.** Final result:
**23 passed in 19.48s**, exit code 0; scoped `git diff --check` passed. Parent
reports actual smoke/parity and the separate 107-test suite passed. Those parent
checks were not rerun here, and this follow-up does not replace final review or
authorize a long run.

Changes:

- `run_budget.py`: versioned run/checkpoint/policy metadata; persisted run and
  profile identities; reject runtime drift before writing resume metadata; bind
  models to source/runtime/config/options/protocol/observation-normalization and
  reward contracts; retain training run/seed/profile provenance while allowing
  distinct validation/evaluation demand. Checkpoint contracts and episode
  boundaries are checked before restore. Requested-action replay and frozen
  evaluation behavior remain covered by integration tests using the real TD3.
- `compare_runs.py`: shared completed-run acceptance validates modes, distinct
  folders/run IDs, model hashes, source/runtime contracts, episode summaries,
  consecutive step/time identities, finite costs, regional and cumulative TTT,
  rewards, terminal flags, guard results, and timing components. Native/center/RL
  remain frozen-original evaluations. Queue durations are aggregated per queue;
  cross-queue totals are labeled queue-seconds. Concurrent one-CPU timings are
  explicitly not an isolated speedup benchmark.
- `run_pilot.py`: validate exited and skipped children, including two training
  episodes and their model plus the seed-6103 validation. A paused or unverified
  zero-exit child cannot advance the stage or complete the pilot. Pauses/failures
  signal and drain siblings. The original bounded two-stage workflow, maximum
  three children, masks 1/4/16, STOP semantics, and checkpoint resume are preserved.

Focused tests: updated `test_compare_runs.py`; added `test_run_contracts.py`,
`test_run_budget.py`, and `test_run_pilot.py`. Coverage includes every blocker,
missing/invalid completion artifacts, source/runtime/model mismatches, metadata
preservation, requested-action replay through synthetic fallback, exact learner
continuation after an odd critic update and terminal-finalization interruption,
frozen evaluation, bounded orchestration, paused validation/resume, failed child
and launch-exception cleanup, exposure estimation, and regional TTT reporting.
The traffic environment and child subprocesses are mocked; real TD3 updates and
Torch serialization run only on one-dimensional synthetic observations. Test
outputs live in temporary directories and are cleaned up.

The first integration run exposed test-only whole-module-cache restoration, which
caused duplicate Torch registration. The test now restores only the mocked
environment module. No production fix was needed after the readiness update.
Pytest required escalated package access because the sandbox could not load its
installed entry point. No bytecode, pytest cache, implementation changes outside
the three owned files, old-run/snapshot modifications, or commits were made.

Final production Git blob IDs:

- `run_budget.py`: `0c60eb2a11197e653e15b53a379a84ec153b64ec`
- `compare_runs.py`: `998c2e361bb87406b72e0a61289d1ec682191f8b`
- `run_pilot.py`: `cd67f4473fdb402ec9f15efadad89cee85158880`

Verification command from the RL repository, with the installed-package access
needed on this Windows host:

```powershell
.\.venv-torch\Scripts\python.exe -B -c "import os,sys; os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; sys.path[:0]=['.deps-budget','work/sdmpc_rl_budget_20260929']; import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--confcutdir=work/sdmpc_rl_budget_20260929','work/sdmpc_rl_budget_20260929/test_compare_runs.py','work/sdmpc_rl_budget_20260929/test_run_contracts.py','work/sdmpc_rl_budget_20260929/test_run_budget.py','work/sdmpc_rl_budget_20260929/test_run_pilot.py']))"
```

## Original Review (Superseded by the Fix Follow-Up)

Date: 2026-09-29. Scope: Tasks 5-6 in `docs/rl_budget_execution_20260929.md`,
`brief_td3.md`, and `results/sdmpc_rl_budget_20260929/review_td3_runner.diff`.
Follow-up scope includes obvious lifecycle/acceptance blockers in `run_pilot.py`.

**SPEC: FAIL for the combined Tasks 5-6 deliverable.** No TD3 equation or replay
contract blocker found. Pilot child acceptance, evaluation identity, model
provenance, exact-resume runtime checks, and completion reconciliation need
correction. Task 6 also lacks the requested queue-exposure measurement.

**QUALITY: CHANGES REQUIRED.** The reported 63 learner tests exercise meaningful
synthetic behavior; the two comparison tests do not cover the defects below.
There is no runner boundary test in this diff. **Runner verdict: HOLD before long
runs until finding 1 is fixed.** The parent reports that probes/isolation are now
complete; they were not rerun here. Findings 3-4 affect training artifact/resume
preparation; findings 2 and 5 must be fixed before relying on the full-run
comparison. Wrapper audit, environment-contract, and boot-manifest repairs remain
with the separately assigned owner.

## Correctness Findings

### 1. [P1] A paused validation child can make the pilot report completion

Location: `run_pilot.py:69-80`, `run_pilot.py:95-100`; related
`run_budget.py:137-140`.

The orchestrator treats return code 0 as a completed child. The child runner also
returns 0 when a STOP file pauses it, including `output/validation/STOP`, which
does not set the pilot-level STOP. If validation pauses while canonical RL
finishes, the final comparison checks only native, center, and canonical RL.
The orchestrator then writes `finite_pilot_complete=True` and `evaluation_runs=4`
without a completed validation episode. A child-local STOP already present on
resume follows this same path. Existence-only skipping at lines 51-52 likewise
does not validate the skipped child's completion contract.

Confirmed by executing the actual orchestrator and comparison functions with
in-memory I/O and fake child processes obeying the runner's paused-exit contract:
`validation_completion_exists: False`, `validation_status: paused`, followed by
`PILOT_COMPLETE` and `finite_pilot_complete: True`. No subprocess was launched.

Require each exited or skipped child to have a parsed, matching completed artifact
with the expected mode, source contract, episode count, and seed/profile identity.
Require two completed training episodes plus their model, and explicitly require
the completed seed-6103 validation before pilot completion. A zero-exit paused
child must pause the pilot and signal/drain siblings; it must not advance the
stage or count as a finished evaluation.

Aside from this defect, the inspected plan has at most three children at a time,
uses distinct one-core masks, drains a stage before the next, signals siblings
when a child fails, and uses `finally` to wait for active children after ordinary
orchestrator exceptions. No additional lifecycle blocker is claimed. Actual
process kill/restart behavior was not exercised.

### 2. [P1] Comparison labels are not checked against recorded controller modes

Location: `compare_runs.py:9-14`, `compare_runs.py:38-46`.

The mode comes entirely from the caller's dictionary key. Neither
`completion.json.mode` nor `settings.json.mode` is checked, and an RL model hash
is not required for the RL entry. Passing one completed native folder for all
three CLI arguments succeeds and labels it native, center, and RL. Swapping
native and RL folders can similarly invert the reported improvement.

Confirmed with the actual `compare()` function and an in-memory JSON reader:
`same_native_folder_in_all_roles: reconciled ['native', 'center', 'rl']`.
Require each requested role to match both recorded modes, reject reused run
identities, and require the RL model identity. Cover duplicate and swapped inputs.

### 3. [P2] Exported policies cannot establish their training source/preprocessing contract

Location: `run_budget.py:121-125`, `run_budget.py:188-196`.

Both episode and final policy files contain only learner state and observation
names. Loading checks names and TD3 architecture/hyperparameters, but cannot
check the training snapshot, environment implementation, physical scaling, reward
contract, or training profile identities. A policy trained under changed scaling
or environment semantics with identical names is accepted. Evaluation's source
pins describe the current evaluator, and its model SHA identifies bytes; neither
establishes the training contract. The sibling `model_hash.json` is not read and
also lacks the source/preprocessing contract.

Include a versioned training contract in exported policies and validate the
relevant source, observation/scaling, and reward contracts before evaluation.
Record training profile identities/seeds without requiring them to equal the
intentionally distinct evaluation profile. This should be added before creating
the pilot artifacts that will feed Task 6.

### 4. [P2] Resume permits dependency drift and erases the previous version record

Location: `run_budget.py:85-96`; `compare_runs.py:23`.

Resume compares settings/source pins, then unconditionally overwrites
`runtime_versions.json`. Dependency versions are outside the compared settings
and checkpoint. Upgrading NumPy, SciPy, or Torch between pause and resume is
therefore accepted, potentially changing solver/optimizer behavior while the
result still appears to be one exact continuation. The original version record
is lost. Comparison likewise ignores runtime-version files, so three runs with
different numerical runtimes can be reconciled under identical source pins.

Validate a persisted runtime contract before restore or metadata replacement,
and include it in comparison compatibility. Preserve attempt-specific runtime
metadata if changed-runtime continuation is deliberately treated as a new run.

### 5. [P2] Completion reconciliation does not validate the trajectory or finite costs

Location: `compare_runs.py:15-30`.

The check verifies 75 rows, terminal flags, a summary duration, and one total-cost
sum. It does not validate consecutive step/control-step identities, the 180-second
timeline through 14,400 seconds, per-row cumulative TTT, or finite numerical
values. A repeated/mixed trace can pass while the summary supplies the claimed
duration. NaN also defeats `abs(mismatch) > tolerance` and guard comparisons.

In-memory checks against the actual function produced:

- `repeated_step_and_timestamp: reconciled` after replacing every step/time with
  the first interval's identity while retaining the original flags/costs.
- `nonfinite_ttt: reconciled nan` after setting summary TTT to NaN.
- `settings_completion_pin_mismatch: reconciled` after changing only settings pins.

Reconcile episode identity, timestamps, finite nonnegative interval costs,
cumulative totals, and settings/completion contracts before computing ratios.
Require positive finite denominator TTT. These are artifact-validation defects;
the probes do not establish that normal plant execution itself emits bad costs.

## Additional Specification Gap

`compare_runs.py:32-40` reports maximum queue sizes and maximum spread only, and
explicitly says subinterval queue-capacity duration is not measured. These maxima
do not measure exposure or time near capacity: short and sustained queue peaks
can produce identical results. This disclosure is appropriate, but Task 6's
queue-exposure requirement remains unmet. Add the required measurement before
claiming Task 6 complete, coordinating any additional trace fields with the
environment owner. This gap alone need not block a bounded pilot.

## Checks Without a Finding

- Requested-action replay: `run_budget.py:143-156` passes the original bounded
  residual action to both the environment and `learner.add`. The inspected
  residual mapping does not mutate it; fallback executed budgets are separate
  audit fields. `td3.py:114-123` copies all transition inputs.
- Targets: `td3.py:125-174` uses gamma exactly 1, target actor plus clipped local
  Gaussian noise, bounded target actions, the twin-target minimum, and bootstrap
  removal only for true termination. Targets have no gradients. Actor and target
  updates are delayed by two critic updates, with the specified Polyak factor.
- Failure/terminal interface: the environment returns true termination only at
  step 80 / 14,400 seconds. Invalid reference preparation raises; runner exceptions
  produce `failure.json` with `terminated_as_success=False` and do not add a
  fabricated zero-cost terminal. STOP returns from a saved interval boundary.
- Learner resume: state includes all four networks, both Adam states, update
  count, ordered bounded replay, and independent NumPy/Torch RNGs. The reported
  synthetic round trips cover both sides of the delayed-update boundary. Runner
  separately saves/restores exploration RNG and the environment checkpoint.
- Episode boundaries, by static inspection: a terminal checkpoint contains the
  current terminal trace but excludes that episode from `completed`. Resuming
  skips interval execution and regenerates its summary/model once. A subsequent
  episode checkpoint includes prior summaries and the new episode index. No
  duplicate replay/update defect was found in this ordering. Actual environment
  serialization/restore equivalence is not established by this review.
- Frozen evaluation: exploration, replay insertion, and updates are restricted
  to train mode. `act` is deterministic and gradient-free; this MLP has no dropout
  or batch normalization requiring eval-mode behavior. Physical observation
  scales are fixed. Evaluation with a validation seed perturbs demand without
  learning; the comparison correctly excludes that run from the frozen-original
  benchmark. No cross-scenario generalization claim is made.
- Costs: successful transitions use actual plant interval TTT / -100. Warmup
  cost is included in total TTT and reconciled separately. Decision wall time
  includes PFO, reference preparation, actor, lower solve, and guard; terminal
  inventory, fallback, candidate solves, and convergence are kept separate.
  Elapsed wall time starts after bootstrap and only preserves checkpointed time
  across interrupted attempts, so it is not a complete failed-attempt compute
  ledger. Do not use it alone for total training-cost claims.
- Source checks hash the snapshot manifest and current non-test task Python
  files, verify snapshot contents, compare on resume, and recheck at intervals.
  Model and runtime contract gaps are findings 3-4 above.
- Compute: the default pilot is two sequential 75-transition episodes (150
  transitions, 119 updates); an evaluation is one 75-transition episode. The
  inspected bootstrap enforces one logical CPU and numerical thread limits.
  Each child runner does not launch parallel simulation workers. The new pilot
  orchestrator caps its own children at three, within the global limit of eight;
  unrelated experiments still require parent coordination. No worker or training
  process was started here.

## Evidence and Review Limits

The five reviewed working files matched the supplied diff's Git blob IDs:
`td3.py` fc553e1, `run_budget.py` 89aaf27, `compare_runs.py` f9b0ebd,
`test_td3.py` da957c7, and `test_compare_runs.py` fbc2ae4. The added orchestrator
was reviewed at blob b593f9c3a6b83bbb3f941407479d5049a1ff4013; runner and comparison
hashes were rechecked and unchanged after this follow-up.

Read the learner report and test implementations; the reported 63 learner and
two comparison test results were not rerun. Additional probes used Python
3.12.14, the actual comparison/orchestrator functions, synthetic in-memory data,
mocked I/O, and fake child processes; no files or simulator state were created by
those probes. Runtime and environment were read only for interfaces, timing, terminal, and checkpoint
contracts. No expensive simulation, candidate-order test, implementation edit,
legacy-run edit, snapshot edit, or commit was performed. This review document is
the only file written.
