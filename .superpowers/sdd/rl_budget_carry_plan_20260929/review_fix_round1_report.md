# Carry review fix round 1

Read both P2 findings in `work/sdmpc_rl_carry_20260929/implementation_review.md`
verbatim, including their required closure sections:

- "[P2] Admission counts probe folders instead of distinct sampled states"
- "[P2] Resumed pilot children need not use the declared physical guard"

Implemented both requested repairs. Independent review closure and new numerical
admission evidence remain the coordinator/reviewer's responsibility.

## Files changed

Only these four authorized files and this new report were written in this round:

- `work/sdmpc_rl_carry_20260929/build_preflight.py`
- `work/sdmpc_rl_carry_20260929/run_pilot.py`
- `work/sdmpc_rl_carry_20260929/test_carry_preflight.py`
- `work/sdmpc_rl_carry_20260929/test_carry_pilot.py`

The previous environment report, implementation review, existing smoke and JUnit
evidence, and all other production sources were preserved. No numerical traffic
run, real child process, commit, or index edit was performed. Test run files and
synthetic gates existed only in temporary directories. No independent approval
was authored; the preflight tests supply a synthetic reviewer response in memory
without writing an approval marker to disk.

## Admission repair

`build_preflight.py:33` validates each expected state 5/30/50 against both the
completion step and embedded settings step, the separate settings file, and each
detailed record's settings and original/reached state times. The expected modes
are exactly `v1_pfo_h3` and `carry_physical`; the five actions are exactly zero,
NP +/-0.25, and NUF +/-0.25 with their declared labels and vectors.

Each state must have exactly ten distinct summary identities and the ten matching
detailed JSON files. Missing, extra, duplicated, or mislabeled records reject
before output is written. Detailed audit execution checks must be valid and must
agree with the executed control point, selection point, budget, achieved budget,
and TTT. Signed NP remains supported and the original upper-budget condition is
retained. Carry records must declare physical guard and the correct reference:
initial PFO at step 5, previous reference with zero PFO calls/time at steps 30/50.
The v1 records must declare current PFO and satisfy their H3 performance guard.

Influence is recomputed from nonzero actions' executed physical points relative
to their same-mode zero-action control, using the producer's 1e-9 tolerance.
Distinct validated step numbers are counted in a set. All completion/summary
claims are cross-checked against values derived from the detailed records;
summary booleans cannot substitute for valid controls or PFO evidence.

The admission manifest now includes the SHA256 of all 30 detailed records, three
probe settings files, three completions, smoke, tests, and review: 39 evidence
paths for one review. JSON reads verify the same hash before and after reading,
so the manifest cannot silently bind bytes different from the validated input.

## Pilot repair

`run_pilot.py:13` keeps the common completed-run validation and additionally
requires the physical pilot guard in both child settings and the embedded
coordinator contract. Consistent h3 train, center, and RL runs reject.

`run_pilot.py:56` validates every already-completed child, across all stages,
before any sibling launch or run metadata rewrite. This includes an h3 center
or RL completion encountered while training still needs to run on resume.
Existing child records are preserved on refusal. Newly exited children use the
same acceptance path. Commands explicitly pass `--guard-mode` with the plan's
value. Existing STOP, skipped-completion, staged concurrency, and launch-failure
draining behavior remains covered.

## Regression results

Before the production fixes, the two focused reproduction methods failed as
expected: copied step-30 completion was accepted at step 50, and h3 train/center/
RL children were accepted by a physical plan (four failed assertions across two
test methods, exit code 1).

Final verification: **130 cases passed across two commands**, no failures,
errors, or skips:

| Suite | Count | Time | Exit |
| --- | ---: | ---: | ---: |
| Carry unittest discovery | 76 | 34.639s | 0 |
| Pytest controller cases | 54 | 0.23s | 0 |

The 76 unittest methods comprise environment 16, pilot 16, preflight 19,
completed-run contracts 15, and real-Torch runner 10. Focused admission/pilot
coverage is 35 methods. Additional mutations are exercised as subtests.

New tests cover state identity, duplicate/missing/mislabeled summary and detail
records, false influence claims, multiple changed actions at only one state,
invalid detailed execution despite a successful summary, nonfinite/control-audit
mismatches, actual PFO/reference/guard violations, exact evidence hash membership,
and concurrent evidence mutation. Pilot regressions cover consistent h3 children,
both settings locations, prelaunch sibling refusal, explicit guard arguments,
and wrong-guard children exiting after launch.

Real Torch coverage again verified 2380 updates, requested-action replay, exact
paused/terminal resume, and frozen evaluation without learner changes. The 54
controller cases are pytest-style and therefore were run separately from the
unittest discovery command.

## Exact commands

All commands used the `RL` repository root as the working directory. Final
commands used the brief-authorized dependency ACL escalation, with no ACL changes.

Initial reproduction, before production changes:

```powershell
$env:PYTHONPATH = 'work/sdmpc_rl_carry_20260929'; $env:PYTHONDONTWRITEBYTECODE = '1'; & '.venv-torch/Scripts/python.exe' -B -m unittest -v test_carry_preflight.PreflightTests.test_rejects_copied_step30_completion_at_step50 test_carry_pilot.PilotTests.test_consistent_h3_children_rejected_by_physical_plan
```

Final unittest verification:

```powershell
$env:PYTHONPATH = "$PWD/.deps-budget;$PWD/work/sdmpc_rl_carry_20260929"; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; & '.venv-torch/Scripts/python.exe' -B -c 'from pathlib import Path; import numpy, torch, unittest; root = Path.cwd(); assert Path(numpy.__file__).is_relative_to(root / ".deps-budget"), numpy.__file__; assert Path(torch.__file__).is_relative_to(root / ".venv-torch"), torch.__file__; print("NumPy:", numpy.__version__, numpy.__file__, flush=True); print("Torch:", torch.__version__, torch.__file__, flush=True); suite = unittest.defaultTestLoader.discover("work/sdmpc_rl_carry_20260929", pattern="test_carry_*.py"); result = unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

Final pytest verification, with persistent cache disabled:

```powershell
$env:PYTHONPATH = "$PWD/.deps-budget;$PWD/work/sdmpc_rl_carry_20260929"; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; & '.venv-torch/Scripts/python.exe' -B -c 'from pathlib import Path; import numpy, pytest; root = Path.cwd(); assert Path(numpy.__file__).is_relative_to(root / ".deps-budget"), numpy.__file__; assert Path(pytest.__file__).is_relative_to(root / ".deps-budget"), pytest.__file__; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "work/sdmpc_rl_carry_20260929/test_carry_controller.py"]))'
```

Verified NumPy 2.3.5 from `.deps-budget` and Torch 2.14.0+cpu from `.venv-torch`.
The pytest command also asserts that pytest comes from `.deps-budget`.

## Source hashes and evidence preservation

| Production file | Reviewed SHA256 | Round-1 SHA256 |
| --- | --- | --- |
| `build_preflight.py` | `80a0856946f80637bcb2e562196c4aaaa5e1084583e8480644f3577b08543610` | `a9377bd446599f773974e86618a513cc47ce81b39aebd70f330f461b8a36f082` |
| `run_pilot.py` | `019b0ef40244346b7aa2d2dcd563a0cbabb26f96a0978177f1118a65e3b79e3c` | `d462083afed597faf6e1021e4a59396603a15f370ffff47d5bf79ac8e1fc3eef` |

Rehashed all eleven production files: the other nine still match the independent
review's recorded hashes. Preserved existing evidence:

- `results/sdmpc_rl_carry_20260929/smoke_resume.json`:
  `9fa78405a52357339dc90dc7dc8cf274d78af9dadc69bad2bbfcf007907e4c5e`
- `results/sdmpc_rl_carry_20260929/preflight_tests.xml`:
  `54e94575cf53c8a2da6fdb5a6c2285fbe1d5f538cee50ea003fb932dad8b8a4c`

These production changes intentionally change source pins. The coordinator must
generate final smoke/probe/JUnit admission evidence in a new subdirectory under
the new pins. No existing numerical evidence was relabeled, replaced, or treated
as admission for the changed source. Independent reviewer closure is still pending.
