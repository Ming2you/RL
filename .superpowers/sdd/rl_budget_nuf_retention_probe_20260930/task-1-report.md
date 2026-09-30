# Task 1: NUF-retention candidate

Status: implementation and scoped verification complete; ready for independent parent review.
No treatment episode, actual simulator reset/step/run, model training, migration, install,
commit, push, ACL/power change, or old-source/result edit was performed. The new numerical
root `results/sdmpc_rl_balanced_goal_20260930/nuf_retention_v1` is still absent.
Writes are confined to `work/sdmpc_rl_nuf_retention_20260930` and this report.

## Implemented surface

- Mechanical copies from immutable `work/sdmpc_rl_local_20260930_v2`: `exploration.py`,
  `local_runtime.py`, `collect.py`, `run_wave.py`, `validate.py`, `frozen_inventory.py`,
  `launch_identity.py`. The last two remain byte-identical. No migration modules,
  historical checkpoints, or historical results were copied into the candidate.
- `exploration.py`: initial actual NUF center retained on every commit; NUF realized
  offset is actual executed NUF minus that center. NP rebase/reset remains unchanged.
  Mean reversion, PCG64 draw order, scales, float32 conversion, request projection,
  and actual previous-executed action anchors retain the v2 implementation.
  Policy format is `nuf-retention-policy-v1`; collection format is
  `sdmpc-nuf-retention-training-v1`.
- `local_runtime.py` / `collect.py`: canonical five-local-slot root, no migration,
  no carry dispatch, retained run IDs/progress and full environment checkpoints.
  Settings bind treatment, source/runtime/physical contract, profile, seeds, actual
  new run ID and both old paired references including their artifact hashes and IDs.
  Collector requires the runner's bound reservation. Failed/unknown status cannot
  be arbitrarily resumed. A stopped/checkpointed episode resumes its retained state.
- `run_wave.py`: explicit `--stage pilot` (170 only) and `--stage remaining`
  (155/incident/skew/190 only). The second stage authenticates and recomputes the
  completed pilot's gate against the unchanged plan/source identity. Completed 170
  is validated and reused, never relaunched. Existing cohort/slot locks, STOP paths,
  launch intent, redirector claim checks and draining survive. Coordinator and
  launcher records now include actual creation identity; live/UNKNOWN retained
  processes block continuation. Draining waits for actual workers as well as launchers.
- `references.py` / `reference_check.py`: pinned read-only v2 authentication in a
  fresh interpreter using the original `supervise.inspect`, original completed
  loader, and repaired-export verifier. Boot/collection and raw environment checkpoint
  deserialization are explicitly disabled. Only `experience.pt` ordinary arrays
  are loaded. Raw checkpoint files are hashed where provenance requires it.
- `diagnostics.py`: reuses only the source-pinned pure `tag_diagnostics` function.
  Raw checkpoints retain +Inf. Export tags only candidate stationarity and candidate-row
  primal-stationarity +Inf at the original typed list-index paths. Unknown nonfinite
  values fail. `diagnostic-audit.json` binds changed paths, helper, raw checkpoint,
  raw/export trace, settings and experience hashes/digests. Completed validation
  reconciles that receipt, unchanged core experience, full physical/terminal/accounting
  checks, policy replay, timing and retained slot/run identity. No export-retry adapter.
- `probe.py`: all 75 noise vectors must match old local exactly. Pilot actual actions,
  physical control/state, native observations/next observations, rewards and TTT must
  match through zero-based step 9 at the existing absolute `1e-8` tolerance; first
  action difference must be 10. Timers and changed exploration memory are excluded.
  Invalid attribution is archived without a performance claim. The pilot requires
  zero NUF requests <=5, inventory <=517.470147382039 and TTT <11945.939292655728;
  comparison with carry TTT 4602.4114708736415 is included. The explicit decimal
  inventory threshold is used after checking its carry-derived equivalent at `1e-8`
  (their binary products differ by one ULP). Other scenarios use the unchanged
  carry+5 / 1.25*max(1, carry inventory) screen.
- Readout includes per-interval TTT/inventory, headroom/base/budget drift, fallback
  events, D-ramp closure, physical control diversity, Q absence and measured timing.
  Original worker-session timing scope is preserved; diagnostic serialization is
  inside it. Reference verification and coordinator elapsed time have separate,
  explicit scopes; coordinator waits overlap workers and must not be added to them.
  No end-to-end timing, millisecond control, canonical improvement, or learner admission
  is claimed. Future minibatches are explicitly 20% per scenario.

## Verification and evidence

Final command, from repository root, using existing dependencies with approved read access:

```powershell
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_tests.py'
```

Result: **69 passed, 0 failed, 0 skipped, 45.49 seconds**.
Only the four new scoped test files ran; neither frozen 236/133 suite was run.
Evidence directory: `work/sdmpc_rl_nuf_retention_20260930/test-evidence/887f768a`.

- `evidence.json` SHA256: `5889180247aeda265acb71ca3ab8ebe749b0cf2037500c31c5c7588c106d89bf`.
- `tests.xml` SHA256: `e4b636590bc078023750a612da1ed3872c4f8691dac4b4b3841ea5089fa2173d`.
- Evidence records exact command, runtime/candidate identity, all 17 new Python file
  hashes, test counts, and **202 identical before/after hashes** covering old sources,
  root/child completions, all old experience/trace/summary/schema/timing outputs,
  settings/checkpoints, repaired receipts and their linked inputs.
- Old root completion: `5b6e645f6f6ca64c4f593c52d19ee3ab5bda7ae3c3d31f423d6fae2d2f3ef4a3`.
- Old comparison: `c294c1f5b2100daeaccb4c927357a8bc73d44391f34641fd37b4ba5b2555c4cd`.
- Pinned diagnostic helper: `f79513942cd892e346dd68a188ecd2221e862b55558170c3a35d90ed8f80a4b3`.
- New policy: `5338c784c997e3427ef2c1b39ee7e572f0d6fc96940f200acf445a87f5741291`.
- New collector: `7e5b7958413676de01e4d109f3caedf1d00c857d8fbe465b12221e923cf5acf7`.
- New runner: `45b2a8f02ab17e88a36aec786f1bb7088b927f57d6ecabeca2adb8ab8b5c58a6`.
- Unchanged copied inventory helper: `9d8204d860df1a673f0e6f59941cb4a0ace35aa04fedcd6f79fa36de8201feaa`.
- Unchanged copied redirector helper: `5cc1a94037b349072ff7ff37e9ceb756c3205b84cb8c1940b4305a5f1514e160`.

Coverage includes retained NUF/unchanged NP, realized-offset consistency, low-executed
rate-limited recovery, both cap boundaries, zero-noise reversion, pending/committed
RNG roundtrips, old-format rejection, saved170 action equality through 9 and step10
request **5552.771855 vs old 5504.421625** (absolute regression tolerance `1e-6`, no
relative tolerance), all75 exact noise vectors, physical/native-prefix mutations,
strict gate boundaries, five independent retained slots, monotonic progress,
successful-pilot reuse, failed-pilot blocking, changed candidate/profile rejection,
STOP, spawn uncertainty, actual redirector identity/draining, true terminal without
an extra step, diagnostic path/core-experience tampering, forbidden nonfinite values,
serialization timing, and saved real physical config including its dynamic
`terminal_zero_gradient` attribute.

The first test iterations exposed fixture omissions (required static-state fields,
temporary-directory sequence-validator path, fixture output parent), a dataclass
constructor that would import topology, and the explicit threshold's one-ULP edge.
They were corrected and retained evidence remains in the earlier run directories.
No physical acceptance check was weakened. The initial sandbox-only reference check
reported package metadata as null and failed runtime identity; with existing-dependency
read access, the original validators passed unchanged for all ten runs and eight receipts.

## Limits and parent handoff

Synthetic collector tests use `FakeEnv`; workflow dispatch/process cases use mocks.
They prove bookkeeping, serialization and gating, not the physical outcome or a live
end-to-end numerical launch. The independent parent review and actual 170 rollout
remain necessary. No causal/performance conclusion is available yet.
The runner enforces one/four workers in its stages and proves the old cohort idle;
the shared global <=8-worker ceiling also requires the parent to account for unrelated
numerical work outside this cohort. UNKNOWN ownership or arbitrary failed work stops
for diagnosis; there is no automatic recovery/retry path. Do not edit candidate Python
sources after starting the pilot: their identity is frozen for remaining-stage reuse.

Parent commands after independent review, from
`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL` (not executed here):

```powershell
# Optional read-only original-reference verification; no numerical boot.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/reference_check.py'

# Stage 1: only local sweet_170_w, seeds 6802/6902, normal reset, 75 intervals.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_wave.py' --stage pilot

# Only for an interrupted/stopped/checkpointed pilot, after honoring STOP/ownership.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_wave.py' --stage pilot --resume

# Read-only integrity/sufficiency readout; exit 2 means blocked/invalid.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/probe.py'

# Stage 2: allowed only if unchanged completed pilot passes every gate; never repeats 170.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_wave.py' --stage remaining --resume

# Final five-scenario readout. This does not admit or run any learner.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/probe.py' --all
```

Persistent STOP markers are never removed by this workflow. A negative or invalid pilot
stays archived and blocks expansion. The old 750 transitions retain their original
behavior/version provenance and are never relabelled as treatment experience.

## FixRound1

Accepted R1 addressed; ready for the parent's scoped re-review. Changes are limited
to `run_wave.py` and its covering `test_workflow.py`, plus this appended report and
generated scoped-test evidence. All other 15 candidate Python files are byte-identical
to `before-fix1`; the reviewed 17-file snapshot itself remains unchanged. No real pilot,
production collection, training, source freeze, old-source/result edit, or commit ran.
The scientific treatment, physical interfaces, gates, seeds and profiles are unchanged.

### Reproduction and fix

Three bounded synthetic regressions were added before changing the runner. In the
clean reproduction (`test-evidence/c204bb75`), all three failed as expected:

- Nonzero launcher exit: ABORT appeared at fake tick 3, after the first actual worker
  was dead and already released, instead of tick 0 while it was still live.
- Successful exited launcher followed by a sibling-slot STOP: ABORT appeared at fake
  tick 4, again after actual death/release, instead of tick 1.
- Successful exited launcher followed by another child's failure: the bounded fixture
  eventually allowed both actual workers to die; the old path removed both children
  and raised without publishing ABORT. The assertion recorded no ABORT observation.

The initial reproduction in `30d2ef63` also demonstrated the first two failures; its
third case hit the fixture's finite-wait assertion. The fixture escape condition was
corrected to let all three reviewed paths finish and retain inspectable evidence.
These ticks are test events, not real elapsed-time or responsiveness measurements.

The fix extracts the unchanged creation-identity checks into the nonblocking
`worker_exited` probe. Ordinary polling leaves a successful exited launcher active
while its actual worker is alive, then continues polling other children and all-slot
STOP on the next cycle. A known nonzero exit raises immediately with that child still
active, so the existing `finally` block publishes attempt ABORT before waiting on it.
The final drain still waits for every known launcher and actual worker, retains
live/UNKNOWN ownership, and propagates UNKNOWN after attempting the sibling drains.
No early release, forced termination, retry, or new process framework was added.

Each new regression now asserts ABORT while the first actual worker is still live,
no release before that point, both launchers drained, release only after the matching
actual identities are dead, no successful-output validation on the aborted path,
and preservation of the sibling STOP marker.

### Commands and evidence

Commands executed from the repository root with read access to existing dependencies:

```powershell
# Pre-fix: 3 expected failures; post-fix: 3 passed, 69 deselected.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_tests.py' -k 'fix1_abort_precedes_actual_worker_death'

# Targeted workflow coverage: 23 passed, 49 deselected, 0.87s.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_tests.py' -k 'test_workflow'

# Refreshed candidate suite: 72 passed, 0 failed/errors/skipped, 50.63s.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_nuf_retention_20260930/run_tests.py'
```

Evidence paths below are relative to `work/sdmpc_rl_nuf_retention_20260930/`:

- Clean pre-fix reproduction: `test-evidence/c204bb75/evidence.json`, SHA256
  `91689c53f7716be2b9a4c5b278c80c5dbe6c5258ba133f37f09fbf0ec0d99c66`.
- Post-fix three-case pass: `test-evidence/4016313e`.
- Targeted workflow pass: `test-evidence/55dccf3c/evidence.json`, SHA256
  `9f817ef820993c79042ffd094d4a138e90b69debd96e2f144f253bd09b434731`.
- Final refreshed evidence: `test-evidence/bbd349b3/evidence.json`, SHA256
  `9d2189bedf735bde76ed48f26e03b861f456a2a8c4653ced1f0a0e3741b76084`.
- Final JUnit: `test-evidence/bbd349b3/tests.xml`, SHA256
  `0b939500049f497c20b231900ff066c282c1c70608d907505c6ddf9656694c77`.
- Fixed runner SHA256:
  `a9fdf4cc390eba170e79b8439cf518b1959795a800f053bb223a0bff1f18f9af`.
- Covering workflow tests SHA256:
  `bd1b2db2a295a5957a8fa5c50cc6658b4ccf6d25e3a50a987713a089281d1cd5`.

The final evidence refreshes all 17 candidate-file hashes and the candidate source
identity. All 202 preserved old source/result hashes match before/after and the
previous reviewed evidence. A final rehash found zero drift from the tested files
and zero changes to `before-fix1`. No frozen suite was run. The production
`nuf_retention_v1` root remains absent.

Limits: process handles, process identity states, clock ticks and child dispatch in
the new regressions are synthetic; no real redirector/worker was launched. The
ordinary coordinator polling cadence remains five seconds, and actual workers honor
ABORT at their existing interval/checkpoint boundaries. After ABORT, final draining
can still wait for a known live worker; UNKNOWN remains fail-closed. This is the same
scientific candidate with corrected orchestration and a refreshed source identity,
not a source freeze or permission to launch. Parent scoped re-review precedes any
real 170 pilot; the earlier parent-only numerical commands remain unexecuted.
