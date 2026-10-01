# Terminal Quota Implementation Report

Status: DONE. Implementation and synthetic verification are complete. Production
execution remains with the parent after independent review. No empirical result
for the terminal-quota hypothesis is claimed here.

## Ownership and Scope

Only these three files were created, relative to the RL workspace:

- `work/sdmpc_rl_value_audit_20260930/terminal_quota.py`
- `work/sdmpc_rl_value_audit_20260930/test_terminal_quota.py`
- `.superpowers/sdd/rl_budget_value_audit_20260930/terminal-quota-report.md`

The complete requirements were read first from
`work/sdmpc_rl_value_audit_20260930/terminal_quota_brief.md`.
Existing sources, completed artifacts, the dirty index, and unrelated working
files were not edited, staged, reverted, or committed. No production diagnostic,
simulator episode, evaluation input collection, policy export, installation,
collection infrastructure, or model override was performed.

The parent's later NUF request-coverage readback did not change this ablation.
The implementation has no policy admission or collection decision logic.

Implementation SHA-256:

- `terminal_quota.py`: `837c237e91872bd7fed53929dcbbf536aa60dbe5f8ce54ccb4404bfe54c35c3e`
- `test_terminal_quota.py`: `e397fbef28cc3bf3cdc2eeaa650e5dba188ed60a70566483cc1b9bdeeb6e8353`

## Fixed Experiment

The production CLI accepts only `--output` (and standard `--help`), with argument
abbreviation disabled. There is no seed, model, budget, tau, resume, overwrite,
or selection switch. Internal callable update/log arguments support synthetic
tests; the CLI does not forward any budget override.

- Seeds: 6529 then 6530, sequentially in one process.
- Arms: `uniform` and `terminal_quota`.
- Budget: 3750 online critic updates per arm per seed.
- Logging: updates 0, 375, 750, 1500, and 3750.
- One CPU numerical worker: Torch intra/inter-op threads and OMP/MKL/OpenBLAS
  thread settings are one, using the existing terminal-fit helper.
- Every minibatch has eight samples from each of five scenarios, total 40.
- Uniform draws use replacement across all 150 stored rows per scenario.
- Quota starts with the same indices and replaces only the first draw of each
  scenario with a uniform draw from terminal rows 74 and 149. The other seven
  indices remain identical. Incidental terminals are retained.
- Uniform RNG: NumPy default generator seeded by the experiment seed.
- Replacement RNG: independent NumPy generator seeded by `seed + 100000`.
- Training smoothing noise: local CPU Torch generator seeded by the experiment
  seed, scale 0.2, clipping at +/-0.5. Each paired update passes the very same
  noise tensor to both arms, with an immutability check between uses.
- Evaluation smoothing noise: a separate fixed CPU stream with seed 86529,
  reused across all logs, arms and both seeds. Logging cannot consume the
  training streams.

Each arm is a deep copy of the final original learner, including critic Adam,
critic targets, replay and local RNG states. The full copied state is checked
against the original before freezing. Both actor modules then contain the final
online actor weights and are frozen in evaluation mode. The actor optimizer is
copied and never stepped. The original target actor's lag is intentionally
replaced, as required; the original learner itself is untouched.

The existing `critic_diagnostic.critic_step` implements the critic update and
target cadence, using its existing `target_values`: gamma=1, the true terminal
mask, and critic target tau=0.005 every second update. No old trainer or physical
simulator is duplicated. Scheduled counts are 3750 online and 1875 target critic
updates, 30000 draws per scenario, and 150000 total samples per arm per seed.
Actual terminal counts are measured, not inferred from the minimum quota.
The original learner's internal training counters remain unchanged; diagnostic
update/sample counts are recorded separately.

## Authentication and Preservation

`terminal_fit/common.load_base` supplies the authenticated original model and
learner. The stored replay is validated by both `terminal_fit.extract_terminals`
and `critic_diagnostic.replay_tensors`, including finite CPU float32 shapes,
the five groups of 150, true terminals at 74/149, and within-episode continuity.

Before loading the base, the runner requires the completed projection audit at
`results/sdmpc_rl_balanced_goal_20260930/value_audit_v1/projection/completion.json`.
It requires `status=completed`, the exact authenticated base hash, and the exact
73 expected training-file paths with matching content hashes. The manifest
includes the base completion, both trained checkpoints, and seven files from
each of ten stored training collections. Missing, extra, substituted, or changed
inputs are rejected. These files are hashed only; collection replay is not
loaded or simulated by this diagnostic.

The projection completion is pinned from the exact bytes parsed, and rehashed
after reading. Its two recorded diagnostic source hashes and its source/runtime
contract must match. The terminal-fit identity helper verifies the baseline
source pins and records checkpoint, source, runtime-version, executable, Torch
and NumPy entry-point hashes. Additional pins cover this runner and its brief,
terminal-fit/common, critic diagnostic, and the imported TD3, run-budget,
budget-runtime and freeze-runtime sources. The small explicit input-path list
avoids importing the projection audit's physical-runtime dependencies.

Full identity capture is performed before base loading, immediately afterward,
and after all updates. Any identity difference rejects completion. Complete
original learner fingerprints include module modes and parameter gradient
controls; the complete loaded model is fingerprinted as well. Final checks occur
after the final provenance verification. Global Python/NumPy/Torch CPU RNG state
is preserved across loading, fitting, and final verification. Arm actor,
actor-target, actor-optimizer, replay, and unchanged learner controls are checked
at the end of each seed. The working replay tensors are separately fingerprinted.

## Finite Checks and Output Lifecycle

Finite parameter, existing-gradient and Adam-state checks surround every critic
step. An optimizer pre-step hook rejects missing/nonfinite critic gradients
after backward and before Adam can mutate parameters. The existing helper also
rejects nonfinite loss. Target parameters and Adam moments are checked after
updates. The hook is removed even on failure.

STOP is checked at the repository, balanced-goal root and output. Checks occur
before acquiring the lock, inside it, before base loading and writes, during
identity capture, before/between/after arm steps, immediately before Adam, and
before completion. Acquiring the existing exclusive `runner.lock` is the only
allowed initial output write. A nonempty output is rejected both before and
inside the lock; the inner check permits only the newly acquired lock file.
An existing empty directory is accepted. A pre-existing lock-only directory is
conservatively rejected before acquisition, so there is no resume path.

All JSON goes through the existing atomic writer after `allow_nan=False`
validation. The output files are:

- `settings.json`: fixed settings, initial source/input/runtime identities,
  original fingerprints, command, PID, and interpretation caveats.
- `metrics.json`: both seeds' complete curves, actual sample/terminal counts,
  original diagnostic summaries, all ten terminal targets and predictions,
  initialization/frozen/critic fingerprints, and stream hashes.
- `status.json`: the last running/verifying phase, or a failure reason.
- `completion.json`: the authoritative successful completion, written last
  after identity and immutability checks, with final metadata and the criterion.
- `runner.lock`: the existing exclusive-lock helper's lock file.

There are no serialized weights or policy artifacts. Completion is the final
diagnostic write, with no later diagnostic validation or status-file write.
On STOP, the callable returns `status=stopped` and the CLI exits 2; it performs
no further output writes, including status updates. Thus persisted status/curves
can describe an earlier phase after STOP. On success `status.json` remains
`verifying`; consumers must use `completion.json` as the authoritative result.
Ordinary exceptions write failed status only when STOP is absent and re-raise.
A stopped/failed run never publishes a new completion.

Every logged checkpoint retains original diagnostics by scenario, time slice,
and episode, including terminal absolute errors, observed behavior-return
discrepancy, and TD loss/residual summaries. Added exact-terminal rows retain
scenario, collection round, replay index, immediate target, Q1/Q2/min-Q,
signed and absolute errors. Aggregate and scenario terminal MAE/max errors remain
available even when the criterion passes. Index stream hashes are per arm;
training noise hashes are shared; fixed evaluation noise has its own hash.

## Predeclared Interpretation

The diagnostic reports support for the sampling hypothesis only if final min-Q
terminal MAE is lower than uniform in both seeds, pooled MAE is at least 50%
lower, and quota final MAE is below 5.701837813854217 in both seeds. Pooling gives
equal weight to both seeds, each containing all ten stored terminals.

All budgets and logs finish independently of metric values. No selection,
early stopping, export, or policy admission uses the result. If the criterion
is unmet, the report says not to keep extending this same update budget.
Observed behavior returns are not fixed-policy Q truth. Exact rewards identify
only the ten stored terminal state-actions. The old saturated actor remains
frozen; there is no traffic, holdout, deployment, or generalization claim.

## Synthetic Verification

Exact final command, run from the RL workspace in PowerShell:

```powershell
.venv-torch/Scripts/python.exe -B -c "import os, sys; os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'); sys.path.insert(0, '.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'work/sdmpc_rl_value_audit_20260930/test_terminal_quota.py']))"
```

Final result: `64 passed in 16.38s`, exit code 0. The process finished. An earlier
version passed 61 tests in 14.88s; the final suite adds three checks for original
state mutation and RNG preservation during final verification.

The first sandboxed command failed before collection because Windows ACLs denied
access to `.deps-budget/pytest/__init__.py`, producing an incomplete namespace
import without `pytest.main`. A read-only probe confirmed `PermissionError`.
The exact synthetic command was then run with approved elevated sandbox access.
No runtime installation or ACL modification was made. `-B` and disabled pytest
cache avoid generated files beside the owned sources. All generated fixture
files reside in pytest temporary directories.

Tests use a tiny hidden-width-8 CPU learner with synthetic three-dimensional
observations and five schema-compatible replay groups. Only three diagnostic
updates are run per arm/seed in runner tests. Both original Adam histories are
populated using synthetic replay. Autouse stubs forbid production loaders,
snapshot access, `torch.load`, `torch.save`, and physical-runtime boot. The real
projection-manifest verifier is exercised against 73 synthetic temporary files.

Verified coverage:

- Equal scenario counts, exact pairing of the remaining seven indices, both
  terminal choices, incidental terminals, independent RNGs and reproducibility.
- Shared training-noise object and values, separate fixed evaluation noise,
  logging-independent final critic/target/optimizer states and stream hashes.
- Full independent copies with existing Adam state and critic targets; frozen
  final online actor in both actor modules; original gamma and terminal mask.
- Critic-only updates and exact tau/cadence; no joint TD3 or actor optimizer step.
- Original learner/model/replay/RNG preservation; actor, target, actor optimizer
  and original state mutation rejection, including late verification mutations.
- Exact per-terminal predictions/errors and scenario aggregates; preserved
  original scenario/time-slice/episode diagnostics.
- STOP at all three roots, inside-lock races, between arm steps, during load,
  final status and final identity; no writes after STOP is observed.
- Nonempty output rejection, empty-directory acceptance and real lock exclusion.
- Source/input/completion changes during load and updates; wrong projection
  status/base/manifest/contract; completion change during parsing verification.
- Nonfinite initial/post-step parameters, gradients before Adam, target state,
  optimizer moments, and diagnostic/loss overflow; no completion/weights on failure.
- Atomic JSON NaN rejection, predeclared interpretation pass/fail conditions,
  and the fixed CLI settings with prohibited overrides rejected.

## Concerns and Handoff

No unresolved synthetic-test failure or implementation blocker remains.
The production replay, runtime authentication and full update budget were not
executed, as explicitly required. Parent independent review and production
execution are outstanding, and no hypothesis outcome can be inferred from
these synthetic tests. Reviewers should note the documented completion-marker
semantics and strict refusal of pre-existing lock-only output directories.
