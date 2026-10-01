# Multi-scenario synthetic integration/lifecycle test report

Date: 2026-09-29. Repository: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

## Status

Implementation is complete in the three assigned new test files. Final run:
**133 passed, 7 failed in 33.99 seconds**, 140 tests collected, pytest exit code 1.
The seven failures reproduce four production defect categories below. Failing
regressions are intentionally retained without xfail, skips, or production changes.

## Ownership and isolation

Only these files were created/edited by this task:

- `work/sdmpc_rl_multi_20260929/test_multi_contracts.py`
- `work/sdmpc_rl_multi_20260929/test_multi_pilot.py`
- `work/sdmpc_rl_multi_20260929/test_multi_runner.py`
- `.superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-2-test-report.md`

No production or old experiment files were edited. No commits, agents, installs,
traffic simulations, or actual child processes were run. Existing carry test
files were read as examples, never imported. One test verifies all relevant
same-named modules resolve to the new experiment directory and no `src` modules
are loaded. Tests use pytest temporary directories. Pytest cache and bytecode
writes are disabled.

## Coverage

`test_multi_contracts.py`:

- Real serialized 75-transition experience artifacts from all five exported
  scenarios; both seed schedules, round/model tags, identity and hash checks.
- Rejection of missing/duplicate folders, scenarios and run IDs; wrong seeds,
  round, policy seed, model, source, runtime, schema, partial/missing artifacts,
  experience hash/settings/contract, transition discontinuities, changed actions,
  rewards and terminals, nonfinite observations, wrong length, and evaluation data.
- Full episode timeline, true terminal only, TTT sums and area accounting, reward,
  PFO initialization/recovery/carry, lower solve count, decision wall/CPU components,
  physical and budget validity, disabled H3, and no local learning.
- Two finite Q estimates required for collection/RL; center Q must be absent/None.
- Canonical unseeded evaluation; five distinct smoke records with matching source,
  schema, serialized-resume evidence and physical carry audits. Canonical forecast
  reads are explicitly mocked at the read boundary and their real digest is used.
- Preflight main/verify_gate: XML pass/skip/failure/count, exact source/test/XML
  hashes, successful test exit, review marker, smoke evidence, and post-gate drift.
  These are synthetic gate fixtures, not claims of actual physical smoke success.

`test_multi_pilot.py`:

- Complete six-stage, 22-job pilot using a controlled fake Popen/poll/wait lifecycle;
  five numerical workers maximum, one-bit CPU masks and single-thread environments,
  no overlap between learner and collectors/evaluators, both seed schedules, and
  all evaluations using the same final model without training seeds.
- All completed-run/trainer validation remains real. Synthetic trainer completions
  contain actual in-process TD3 updates, optimizer state, metrics and replay; fake
  processes never execute CLI commands or launch a solver.
- Nonzero exit, paused/zero-exit incomplete worker, malformed completion, launch
  exception, STOP before startup and during launches, active output locks, draining,
  log closure, and no promotion to later stages or pilot completion after failure.
- Validation before skipping a completed collector or launching siblings; checkpoint
  resume command; completed trainer seed, cumulative sampling counts and provenance;
  gate/plan drift and duplicate completion rejection.

`test_multi_runner.py`:

- Real runner main and real TD3; only physical environment/runtime boundaries are
  replaced with a stochastic synthetic environment. All requested actions, rewards,
  next observations and terminals are compared exactly with the saved experiences.
- Round 0: first 32 actions exactly uniform, remaining actions zero/frozen actor plus
  Gaussian sigma 0.3. Round 1: loaded nonzero frozen actor plus Gaussian sigma 0.3.
  `add` and `update` are forbidden during collection/evaluation. Trace Q estimates
  are checked against both real critics on the current observation/requested action.
- Inference and collection resume at the uniform/Gaussian boundary; exact environment
  RNG, exploration RNG, actions, replay experience and loaded immutable policy;
  checkpoints contain no learner state. Terminal-finalization interruption does not
  duplicate a transition. Startup failure preserves/checks existing run identity.
- Source/runtime/scenario/profile/model drift rejection before any new step; STOP at
  child/stage/parent yields a checkpoint without transitions or completion.
- Real centralized 375-update round 0 and round 1: changed actor weights, eight samples
  per scenario per update, 75/150 replay entries per scenario, terminal positions
  74/149, continuity within episodes but distinct episode boundaries, finite Q
  diagnostics, and successful real completed-training validation.
- Real interrupted training at update 25 matches uninterrupted final actor/critics,
  targets, optimizer states, RNGs, replay, counters, metrics and provenance exactly.
- Trainer checkpoint/settings/progress/counter corruption, changed collection hashes
  and wrong predecessor round are tested before further gradient updates.

## Confirmed defects

### D1: Trainer resume accepts a changed seed, sampling counts and replay size

Production: `work/sdmpc_rl_multi_20260929/train_round.py:156` and `:161`.

Regression: `test_training_resume_rejects_corrupt_checkpoint_before_update[seed]`,
`[sample_counts]` and `[replay]` in `test_multi_runner.py:400`.

Starting from a valid paused round-0 checkpoint with unchanged settings, replace the
learner seed 6300 with 99, or remove one complete replay row from one scenario
(74 instead of 75 entries). A third variant performs one real update, preserves
its valid weights, optimizer history, metrics and update count, then changes each
scenario's sample count from 8 to 7. Resume loads each case and reaches `learner.update(40)`.
The regression sentinel fails with:

```text
Failed: Corrupt trainer checkpoint reached a new gradient update
```

Resume compares settings, completed count, metrics length and learner update count,
but does not check the restored learner seed, expected per-scenario replay counts,
or exact cumulative sampling counts.
The fresh pre-resume learner's replay check occurs before checkpoint state overwrites
it. The sampling-counter case uses real one-update optimizer history, avoiding
rejection for unrelated malformed optimizer state. Required behavior: validate restored state against the trainer's
seed, expected replay and cumulative sampling contract before another update.

### D2: Completed trainer does not reconcile policy provenance with its collectors

Production: `work/sdmpc_rl_multi_20260929/run_pilot.py:50` and `:55`;
`work/sdmpc_rl_multi_20260929/run_budget.py:89`.

Regression:
`test_completed_trainer_checks_actual_checkpoint_identity_counters_and_provenance[provenance]`
in `test_multi_pilot.py:251`.

Build a valid completed two-round pilot, change the final policy's last
`training_profiles[*].experience_sha256` to `wrong-collector`, and update the
completion's model file hash. `validate_job` returns success:

```text
Failed: DID NOT RAISE <class 'ValueError'>
```

The validator reloads actual collections and checks their identities in settings,
but policy provenance validation only establishes balanced scenarios, unique seeds
and nonempty fields. It never compares policy provenance with those actual collector
provenance records or the predecessor's records. Required behavior: reconcile the
exported policy's full ordered provenance with the predecessor plus current runs.

### D3: Corrupt completed collection experience is skipped before launching siblings

Production: `work/sdmpc_rl_multi_20260929/run_pilot.py:67` and `:121`.

Regression: `test_completed_collection_experience_is_validated_before_sibling_launch`
in `test_multi_pilot.py:227`.

Provide a completed fifth collector with 75 valid trace rows but only 74 experience
transitions, and a matching experience file hash in completion. Run stage 0 with
four missing siblings. All four fake siblings launch and the stage returns success:

```text
Failed: DID NOT RAISE <class 'ValueError'>
CHILD_FINISHED collect_round0/sweet_155_w 0
CHILD_FINISHED collect_round0/sweet_170_w 0
CHILD_FINISHED collect_round0/sweet_170_incident_w 0
CHILD_FINISHED collect_round0/sweet_170_skew15_w 0
```

`validate_job` checks completion, trace and experience file hash, but does not call
`verify_experience` before skip/launch. The central training admission eventually
rejects this payload, after avoidable numerical work. Required behavior: validate
the serialized experience's full contract and trajectory before any sibling launch.

### D4: New stage/child STOP during launch does not block remaining siblings

Production: `work/sdmpc_rl_multi_20260929/run_pilot.py:123`.

Regression:
`test_stop_appearing_during_launch_prevents_new_siblings_and_drains[stage]` and
`[child]` in `test_multi_pilot.py:181`.

The first fake Popen creates a STOP file in its stage or own output folder.
The coordinator still launches all five workers before polling. Both tests fail:

```text
AssertionError: STOP must prevent launching another numerical worker
assert 5 == 1
```

Startup checks all STOP locations, but the launch loop checks only the parent
output STOP. Parent STOP during launch passes; preexisting parent/stage/child STOP
all pass. Required behavior: recheck active-stage/child STOP locations before each
new launch and propagate the stop while draining already active workers.

## Execution record

All suite executions use one Python process for exactly these three test files.
Working directory: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& '.venv-torch/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --confcutdir=work/sdmpc_rl_multi_20260929 --tb=short work/sdmpc_rl_multi_20260929/test_multi_contracts.py work/sdmpc_rl_multi_20260929/test_multi_pilot.py work/sdmpc_rl_multi_20260929/test_multi_runner.py
```

1. Sandboxed attempt failed before collection: `No module named pytest.__main__;
   'pytest' is a package and cannot be directly executed`. Read-only inspection
   confirmed access denied for `.deps-budget/pytest/__init__.py` and `__main__.py`.
   The same command ran successfully with approved dependency ACL escalation.
2. First executable run: `10 failed, 115 passed in 28.75s`, exit 1. Several lifecycle
   tests stopped at a test-fixture incompatibility with TD3 optimizer-history checks.
   Fixture repair uses real TD3 updates and cooperative fake-child STOP handling.
3. Second executable run: `6 failed, 134 passed in 31.39s`, exit 1. All failures are
   D1-D4 above. Added Q-audit negative coverage since the first run.
4. Final run: `7 failed, 133 passed in 33.99s`, exit 1. The sampling-counter case now
   isolates the missing trainer validation with valid optimizer history and also
   reproduces D1. Full final failure summary:

```text
FAILED test_multi_pilot.py::test_stop_appearing_during_launch_prevents_new_siblings_and_drains[stage]
FAILED test_multi_pilot.py::test_stop_appearing_during_launch_prevents_new_siblings_and_drains[child]
FAILED test_multi_pilot.py::test_completed_collection_experience_is_validated_before_sibling_launch
FAILED test_multi_pilot.py::test_completed_trainer_checks_actual_checkpoint_identity_counters_and_provenance[provenance]
FAILED test_multi_runner.py::test_training_resume_rejects_corrupt_checkpoint_before_update[seed]
FAILED test_multi_runner.py::test_training_resume_rejects_corrupt_checkpoint_before_update[sample_counts]
FAILED test_multi_runner.py::test_training_resume_rejects_corrupt_checkpoint_before_update[replay]
7 failed, 133 passed in 33.99s
```

No full old/new repository suite, physical smoke, gate admission, or traffic run was
attempted. Synthetic evidence fixtures are not usable admission artifacts. No test
was weakened to accommodate production behavior; fixture repairs preserve the
binding requirements and leave the confirmed production regressions failing.

## Source identity

SHA-256 immediately before final run; production files were only read by this task:

```text
7524B335458753C15D7894E872F6BCC80056B7CC4C3756CF7728ABDEB48E985E run_budget.py
23B7A49D94D60F2F37F8A525180F442781FDC99463B893B6095FB25483F940A0 train_round.py
9FD8CFA8766AB938CB1F3E289C02A801A981203715D0CD63BC08C3E583FB72DC run_pilot.py
8295D41C2A1E4CDE787D5C81C291A9068070F02EFBCE5B16C8E7E9C5BD7C8906 compare_runs.py
5D33DAC912DF086C2C3AE633DB93C99D68E693F90B721A794F60390D1408CC6C build_preflight.py
E27665133CCB3FAC2B1769DA1C93DF73A60C73572FA4CCEDE24A1296292E2E8E td3.py
947B4EF201137FB7AEBA6F6DC40F84BFACFF3441FC2D963279C1466BCE3979A5 test_multi_contracts.py
3F1B2C7807716DB9915EC6E7A2267193FA65B24E12A0A013EBD5C8C78A941783 test_multi_pilot.py
F69A63649FDF23B962D2310E3B290576DA7768DEE3C2512A48490B36C1F299BB test_multi_runner.py
```
