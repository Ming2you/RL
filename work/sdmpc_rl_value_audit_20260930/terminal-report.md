# Terminal-fit sidecar implementation report

## Status and scope

Implemented and verified with **46 passing synthetic tests** on 2026-09-30.
The production diagnostic was not executed. No production learner was loaded,
no canonical evaluation data was loaded, and no simulation, policy export,
installation, commit, or source edit outside this task's three files occurred.
Production fit quality and production input immutability remain unmeasured until
the parent runs the admitted diagnostic.

Changed files, relative to the existing `RL` repository:

- `work/sdmpc_rl_value_audit_20260930/terminal_fit.py`
- `work/sdmpc_rl_value_audit_20260930/test_terminal_fit.py`
- `work/sdmpc_rl_value_audit_20260930/terminal-report.md`

Read `terminal-brief.md` first, then inspected the read-only recovery `common.py`,
the multi-scenario TD3, runtime/locking/atomic-write helpers, and the prior critic
diagnostic and its synthetic tests. Preserved the existing dirty repository,
including staged and unstaged work. Parent-owned projection files that appeared
during implementation were left untouched. Replay/observation/projection audits
are outside this sidecar's ownership; replay validation here is limited to the
schema, finiteness, sample count, and terminal positions required for extraction.

## Implemented experiment

The CLI has exactly one required argument, `--output <new directory>`. Its budget
is fixed at **1000 supervised updates per arm per fold**, with metrics at
**0, 50, 250, and 1000**. Smaller budgets are injectable through Python functions
for synthetic tests only; the CLI has no update-budget or seed switch.

The loader comes from `work/sdmpc_rl_recovery_20260929/common.py`, including its
fixed baseline path/hash, runtime verification, source pins, and baseline
completion/replay checks. The expected baseline SHA-256 remains:

`3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`

The extractor requires exactly the five existing scenarios, 150 replay rows per
scenario, CPU float32 numeric columns of the expected shapes, finite values,
boolean termination flags, and true-terminal positions exactly `[74, 149]`.
It copies only the ten terminal observation/action/reward tuples. Row indices
are zero-based and retain their scenario and collection round.

| Fold | Training rows | Diagnostic holdout rows |
| --- | --- | --- |
| Round 0 training | Row 74 from each scenario | Row 149 from each scenario |
| Round 1 training | Row 149 from each scenario | Row 74 from each scenario |

Both splits contain five rows, exactly one per scenario. Each fullbatch update
uses all five training rows, with equal 20% scenario contribution. Targets are
the exact stored immediate rewards. The loss is the sum of the two critic MSEs.
There is no bootstrapping, reward-to-go computation, smoothing noise, reward
normalization/clipping, augmentation, actor update, or target-network update.

Each fold starts both arms independently:

1. A deep copy of the original trained twin critics with a fresh Adam optimizer.
2. A deep copy of fresh critics from the same TD3 observation dimension and hidden
   size, initialized with seed **8100**, with a fresh Adam optimizer.

Both arms use learning rate **0.0003**. The fresh initialization template is shared
only as an immutable starting point: each fold gets an independent copy and fresh
Adam state. Every critic layer/parameter is trainable. No learner update method
or actor/target forward path participates in fitting.

## Metrics and interpretation

Before any extra supervised updates, `metrics.json` records original-critic
errors on all ten terminals. Every checkpoint then records both train and holdout
metrics for each arm and fold:

- Per-row scenario, round, exact replay index, and immediate-reward target.
- Predictions, signed errors, and absolute errors for Q1, Q2, and min(Q1, Q2).
- Aggregate MSE, MAE, and maximum absolute error for all three heads.
- Exact training rows, cumulative training draws by scenario, and total draws.

Error summaries use float64 arithmetic on the original float32 predictions and
targets. Training remains CPU float32. Each completed arm/fold reports whether
each critic's final **training** maximum absolute error is <= **0.01 reward
units**. These flags are descriptive; failure to meet the threshold does not
prevent a successfully executed experiment from reporting completion.

The original critic previously saw both folds during joint TD3 training. Its
diagnostic holdout excludes only the extra supervised updates implemented here.
The fresh critic's holdout is untrained within that fold of this diagnostic, but
there is only one sample per scenario and no out-of-sample generalization claim.
Both caveats are written into the JSON, including the arm-level metrics.

Holdout errors are never used for updates, early stopping, arm selection, or
policy admission. This experiment does not establish traffic-control performance.

## Lifecycle, provenance, and immutability

The runner requires a nonexistent output path, creates it exclusively, and holds
the existing `exclusive_run` OS lock throughout the run and final verification.
Existing empty directories, populated directories, and files are rejected.

STOP is checked at all three exact locations:

- `<output>/STOP`
- `results/sdmpc_rl_balanced_goal_20260930/STOP`
- Repository-root `STOP`

Checks occur before output creation and input hashing/loading, at fold/arm
boundaries, before and after each update, and before the completion write. STOP
is never removed. If STOP already exists on entry, the CLI returns a JSON stopped
status on stdout, exits with code 2, and creates/writes nothing. In particular,
an occupied output containing STOP remains untouched. A STOP arriving after the
run starts writes a stopped status and partial metrics, with actual completed
update/draw counts and explicitly numbered metric checkpoints. It does not write
`completion.json`. Read-only final identity checks can still run after STOP.

Artifacts for a successful run:

| File | Contents |
| --- | --- |
| `settings.json` | Command, PID, UTC start, working/output paths, STOP paths, exact settings, original architecture/specification, initial hashes, caveats |
| `metrics.json` | Original ten-terminal baseline and per-fold/per-arm checkpoint metrics |
| `status.json` | Final status, reason, UTC end, per-arm progress, post-run identities, immutability results |
| `completion.json` | Written only after successful fitting, verification, and final STOP checks |
| `runner.lock` | Existing helper's OS lock marker; not a model artifact |

All data artifacts are JSON and use the existing atomic temporary-write/replace
helper, preceded by strict `allow_nan=False` validation. No diagnostic weights,
optimizer checkpoints, pickle files, or policy exports are written. Nonfinite
predictions, loss, gradients, parameters, Adam moments, or JSON metrics fail
closed and cannot produce a completion claim.

Before and after execution, provenance checks include the baseline model and its
completion file, sidecar/common/brief sources, existing baseline source pins and
frozen manifest validation, runtime versions, and hashes of the Python executable
and Torch/NumPy entrypoints (including the Torch extension). Runtime metadata has
its own SHA-256. These are entrypoint/version identities, not a claim to hash
every installed dependency binary. Mixed helper-module locations are rejected.

Original learner state fingerprints include actor/critic and target parameters,
optimizer states, replay, counters, learner RNG states, parameter gradients,
requires-grad flags, and module training modes. The original deserialized model
state is checked separately. Global Python, NumPy, and CPU Torch RNG states are
preserved, including failure paths. CPU Torch intra-op/inter-op and the
OMP/MKL/OpenBLAS environment thread limits are one. No original parameter or
optimizer is used for fitting.

## Verification performed

Existing local runtime:

- `.venv-torch/Scripts/python.exe`: Python 3.12.14
- Torch 2.14.0+cpu
- NumPy 2.3.5
- pytest 8.4.2 from `.deps-budget`

The initial sandboxed runtime probe encountered an ACL `PermissionError` reading
`.deps-budget/pytest/__init__.py`. The same read-only probe and the scoped test
run succeeded with tool escalation, as anticipated by the brief. No packages or
permissions were modified. This was an environment-access issue, not a failed
test.

Exact runtime probe, run from `RL`:

```powershell
& '.\.venv-torch\Scripts\python.exe' -B -c "import sys; sys.path.insert(0, '.deps-budget'); import torch, numpy, pytest; print({'python':sys.version, 'torch':torch.__version__, 'numpy':numpy.__version__, 'pytest':pytest.__version__})"
```

Exact test command, run from `RL` using the existing runtime with tool escalation:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& '.\.venv-torch\Scripts\python.exe' -B -c "import sys; sys.path.insert(0, '.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', '--noconftest', '-o', 'addopts=', '--basetemp=work/sdmpc_rl_value_audit_20260930/.terminal-test-tmp', 'work/sdmpc_rl_value_audit_20260930/test_terminal_fit.py']))"
```

Result: **46 passed in 4.66 seconds**, exit code **0**. One focused suite run;
no failed tests. Test-created temporary files were confined to the explicitly
named directory under this task and removed after verification. Bytecode and
pytest cache writes were disabled.

The tests use three-dimensional synthetic observations, hidden size eight,
synthetic replay, and one-to-three-step fitting budgets. A fixture prohibits
production loading, actual source-pin traversal, `torch.load`, and `torch.save`.
Runner tests inject synthetic in-memory learners and synthetic input/source
identity files. The real provenance helper is tested only against a synthetic
input file, with production source pins substituted. No production data is read
or fitted by the suite.

Coverage includes terminal-only copying; 16 malformed replay/terminal cases;
disjoint reverse folds; exact balance and immediate rewards; all critic layers
and fresh Adam state; exact Q/min-Q metrics; deterministic seed-8100 initialization
and repeated folds; original learner/optimizer/replay/gradient/mode/RNG
preservation; unchanged training when only holdout targets or logging frequency
change; explicit holdout labels; JSON-only artifact names; all three early and
loop STOP locations; STOP during loading and before completion; overwrite
rejection; actual lock contention; source/model drift; nonfinite prediction,
loss, gradient, parameter, Adam and JSON guards; failure RNG preservation; and
the fixed production CLI contract.

## Handoff and concerns

The parent can invoke the script with the local Python runtime and a new output
directory after admitting the production diagnostic. That invocation has not
been run here. The script itself sets dependency paths and one-thread limits.

There are no outstanding implementation/test failures. Practical limitations:

- Actual baseline loading, frozen-source compatibility, production 1000-update
  numerics, terminal error values, and production before/after hashes remain
  unverified by design. Synthetic passing tests do not imply a <=0.01 fit result.
- The existing runtime/lock helper is Windows-specific. Local dependency ACLs
  may require the same tool escalation used for the tests.
- A run interrupted by STOP retains its output and cannot resume into that
  occupied directory. The next admitted attempt needs a new directory.
- Warm-start holdout has prior TD3 exposure; fresh holdout has only five samples.
  Neither is a policy-admission or generalization result.

No learner changes or recommendations for the parent's separate replay,
observation, or budget-projection audits are made by this sidecar.
