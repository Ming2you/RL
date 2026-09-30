# Carry environment, runner, and pilot test report

Read `docs/rl_budget_carry_plan_20260929.md`, the environment test brief, carry
production modules, and the v1 regression tests. Implemented the original test
scope plus the explicitly authorized runner/pilot test extension.

## Result

**53 tests passed in 35.697 seconds, exit code 0.** No failures, errors, or skips.
Subtests exercise additional contract mutations within those 53 test methods.
No production issues were exposed by the covered cases.

| File | Tests |
| --- | ---: |
| `work/sdmpc_rl_carry_20260929/test_carry_environment.py` | 16 |
| `work/sdmpc_rl_carry_20260929/test_carry_run_contracts.py` | 15 |
| `work/sdmpc_rl_carry_20260929/test_carry_runner.py` | 10 |
| `work/sdmpc_rl_carry_20260929/test_carry_pilot.py` | 12 |

Those four files and this report are the only files written for this task.
No production edits, commits, index edits, numerical traffic simulations, or
real pilot subprocess launches were performed. Temporary synthetic run records
and Torch checkpoints were created under temporary directories and cleaned up.

## Coverage

- Environment: initial preparation calls PFO once; ordinary preparation never
  calls PFO; only `InvalidReference` invokes recovery; unexpected errors and
  failed initial/recovery references fail closed. PFO receives a copied state.
- Observation and transitions: reference controls, reference source one-hot,
  carried action anchor, reference budget/TTT, and previous request are observed.
  A failed request remains distinct from its fallback execution in the audit and
  next observation. Rewards remain negative interval TTT divided by 100; only
  control 75 terminates, with a zero terminal observation and 14400s duration.
- Checkpoint: pickle round trips restore observation, hidden warm memory, duals,
  anchor, and the actual next-step observation/reward/audit/state exactly, without
  PFO, for all three reference sources. Guard, configuration, public dynamic
  configuration/options, source identity, schema/normalization, demand, missing
  prepared reference, and legacy-contract mismatches reject before live state
  replacement. Terminal restoration performs no reference preparation.
- Completed-run validation: center/RL only; separate carry formats; 20 updates
  per interval and batch 32. Physical guard permits higher selected H3 TTT while
  h3 guard rejects it. Failed physical/budget execution checks, source/PFO-call
  inconsistencies, recovery/fallback/candidate summary counts, timing and TTT
  accounting errors, nonfinite costs, reused/partial runs, and provenance drift
  reject. Queue-exposure accounting remains covered.
- Real Torch runner: two synthetic episodes with seeds 6201/6202 produce 150
  transitions and exactly **2380 = 119 * 20** updates. Replay retains requested
  actions and only transitions 74/149 are terminal. Pausing at k=37 after the
  first 20 updates, and interruption at the first episode's terminal checkpoint,
  both resume to the exact uninterrupted learner state, action sequence, and
  exploration RNG state without duplicate collection. Frozen canonical evaluation
  leaves the entire learner/replay state unchanged. Parent STOP, duplicate output,
  resume boundaries, policy contracts, and dynamic config hashes are covered.
- Pilot: subprocesses are entirely mocked. Train6201/6202 plus center launch with
  CPU masks 1/4, followed by RL with mask 1 only after both finish; maximum live
  children is two. Tests preserve STOP/resume, skipped-completion validation,
  zero-exit missing completion rejection, failed-child handling, and launch-error
  sibling draining/stream closure. Admission evidence hashes are checked with
  matching and mutated files; other synthetic gates use the authorized empty map.

## Exact Verification Command

Working directory: the `RL` repository root. Run in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/.deps-budget;$PWD/work/sdmpc_rl_carry_20260929"; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; & '.venv-torch/Scripts/python.exe' -B -c 'from pathlib import Path; import numpy, torch, unittest; root = Path.cwd(); assert Path(numpy.__file__).is_relative_to(root / ".deps-budget"), numpy.__file__; assert Path(torch.__file__).is_relative_to(root / ".venv-torch"), torch.__file__; print("NumPy:", numpy.__version__, numpy.__file__, flush=True); print("Torch:", torch.__version__, torch.__file__, flush=True); suite = unittest.defaultTestLoader.loadTestsFromNames(["test_carry_environment", "test_carry_run_contracts", "test_carry_runner", "test_carry_pilot"]); result = unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

Confirmed imports: NumPy 2.3.5 from `.deps-budget/numpy/__init__.py`, Torch
2.14.0+cpu from `.venv-torch/Lib/site-packages/torch/__init__.py`; Python 3.12.14.

The sandbox initially denied access to `.deps-budget/numpy/__init__.py`
(`PermissionError: [WinError 5]`), allowing Python to fall back to bundled NumPy.
The brief-authorized dependency ACL escalation succeeded. The final command above
asserted both import locations before running all tests; the final result is from
that isolated-dependency run. No ACLs or dependency files were changed.

Physical performance, numerical solver behavior, and admission-probe outcomes
remain outside this synthetic test task and belong to the coordinator's work.
