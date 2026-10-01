# Task 2: Frozen Shared-Actor Training-Wave Worker

Status: implemented and scoped checks pass; ready for parent review before any actual prefix or wave dispatch. No real simulation, reset, PFO/lower solve, optimization, or training was performed. No commit was made. The actual `results/sdmpc_rl_balanced_goal_20260930/return_policy_wave_v1` directory did not exist at final implementation verification. Task 1 model/source/data and frozen physical inputs were preserved.

## Changed Paths

Only this report and the new `work/sdmpc_rl_return_wave_20260930/` directory were written:

- `actor.py`: complete strict float32 2367-64-ReLU-64-ReLU-2 actor loading, admission authentication, and inference under `no_grad`.
- `wave_support.py`: frozen serialization/import helpers, fixed slots/seeds, source/model hashes, STOP handling, immutable checkpoints, actual worker identity, and locked-session timing.
- `checks.py`: frozen observation/accounting/sequence/actuator checks, exact actor replay, checkpoint boundary checks, diagnostic export receipts, and focused summaries.
- `worker.py`: one single-thread slot; `--scenario`, `--resume`, and `--max-new-steps`. No dispatcher, reservations coordinator, learner, or automatic next wave.
- `readout.py`: read-only acceptance of exactly five complete unique scenarios and diagnostic health screens.
- `parity.py`: isolated original-Actor parity and read-only physical import/profile/control authentication.
- `conftest.py`, `test_wave.py`, `run_tests.py`: synthetic environment and scoped test/evidence runner.
- `test-evidence/`: retained test reports; final accepted evidence is `a64504a6/`. Earlier `e83f662e/` records the temporary-directory setup failure; `2c134acb/` records the earlier 54-test pass.
- `_t/`: retained synthetic fixture runs and checkpoints. These are explicitly synthetic and are outside the actual results wave root.

No Task 1, frozen helper, physical snapshot, parent documentation, launch, or budget source was edited.

## Admission And Imports

Pinned candidate: `results/sdmpc_rl_balanced_goal_20260930/return_init_v1/model_final.pt`, SHA-256 `7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904`.

Pinned Task 1 completion SHA-256: `34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757`.

Pinned Task 1 specification digest: `933e29c1b316530ceff1605d527532e482a658c12fa6c13fd24a93074ef6593e`.

Admission reconciles completion/settings/outputs, recorded sources and data hashes, physical snapshot/runtime/contract, phase `done`, counts `phi=1000, critic=250, actor=10, polyak=125`, finite state, loss lengths, and the recorded numerical gate. It does not reinterpret Task 1's actual fit evidence or claim traffic improvement. Critic continuation remains `carry`.

The environment process never imports Task 1 `runtime.py`, `learner.py`, or `data.py`, nor the old `td3`/`train_round` modules. The adapter constructs its architecture on the meta device, assigns every admitted tensor strictly, disables gradients, and uses the stored float32 bounds `[0.2, 0.1]` multiplied by `tanh`. Native observation/action float32 arrays are retained. No random replacement model is used.

Small unchanged pure definitions are compiled from source-pinned frozen files into a local helper namespace: observation/boundary/row validation, inherited sequence validation, environment-contract verification, and `Coordinates` actuator validation. Frozen inventory helpers and the pinned diagnostic tagger are reused. This avoids importing the old collector's policy, learner dependencies, settings, seeds, or completed-run identities. It does not patch or globally rename physical modules.

## Worker And Recovery Contract

The five fixed scenario slots each hold the original Windows kernel byte lock for the worker session. Their fixed canonical output path is `results/sdmpc_rl_balanced_goal_20260930/return_policy_wave_v1/<scenario>`. Five distinct slots bound the wave to at most five workers. Parent retains global budget-eight admission, unrelated-job accounting, hidden dispatch, launcher/CIM records, and actual-worker drain responsibility.

| Scenario | Training Seed | CPU Mask | Expected Profile SHA-256 |
| --- | ---: | ---: | --- |
| sweet_155_w | 7301 | 1 | 13ee7d327102df2869895c0e7c1a738d56127739cc4668e973ae728904109cd1 |
| sweet_170_w | 7302 | 4 | 6d79f730e86d1c6a39bedaed2202aec00747e80105a88e37fdb77c0242e7973c |
| sweet_170_incident_w | 7303 | 16 | 6024fcc498a0e62d079a6874cbf443c170fc339087e18c004d7a8bceaf98601a |
| sweet_170_skew15_w | 7304 | 64 | 3da6dadce56fa7283a943550ab44757372a7219ab57a91dd79473b51b004b8ab |
| sweet_190_w | 7305 | 256 | 9b2636ed113efef3f6fc5f42e24424a298cf96bcf61f2e20ef8ff2ec06815629 |

Original `BudgetEnv` constructs demand, resets/restores, projects actions, advances physics, and applies the physical guard. Reward, gamma, warmup, horizon, previous-executed anchor, config/options, followers, initial/recovery PFO, and one candidate per step are unchanged. Settings include full physical config/options serialized with frozen `to_plain_dict`, source/model/contract/run identity, profile/seed, and warmup TTT. Mode is `training_wave`; evaluation/exploration/improvement claims are false. `policy_q=None`, `learning=[]`, and zero learning time are enforced.

Each actual worker records its PID, creation FILETIME, parent PID, executable and original command in `process.json` and immutable session-start records under the slot lock. These are actual interpreter identities, which can differ from the Windows venv launcher PID. Session end is not evidence of process exit: parent must still wait for the actual worker.

STOP is honored at repository, goal, wave root and slot levels, including before reset/restore and between intervals. Markers are never removed. A completed slot refuses both normal start and resume. Nonempty partial/orphan slots require explicit resume and a bound full checkpoint; identity mismatches fail closed.

Each actual interval writes an immutable `checkpoints/<uuid>.pt` with the complete `BudgetEnv.checkpoint()`, observation, raw trace, experience, schema, settings/model identity, session IDs, current timing and progress. An atomically replaced `latest.json` binds its hash and progress. Invalid payloads remain raw orphan evidence without becoming the current pointer. No orphan is deleted or silently promoted.

Resume validates the full retained observation/action/reward/terminal chain, physical accounting, configuration/options/normalization, boundary state and clock, and exact admitted actor action for every retained observation before original `env.restore`. Restored observation equality is required. No reset or already-checkpointed interval is recollected. The existing prepared reference/PFO state is retained in the full checkpoint.

The small `operations/` journal distinguishes a published interval checkpoint from an uncertain in-flight reset/step. If a process dies during physics or after an interval but before publishing its checkpoint pointer, resume refuses to repeat that uncertain operation. Parent must diagnose the preserved evidence; this implementation neither guesses that physics did not run nor automatically promotes an orphan. This is a deliberate recovery limit, not an automatic restart path.

## Completion, Diagnostics And Readout

Completion requires all 75 true-terminal controlled intervals and a 14400-second terminal clock. Checks cover reward/area TTT, physical inventory, observation memory, anchor/request/execution chain, actuator constraints, budgets, single candidate, physical guard, PFO counts, finite physical/learning values and decision timing components.

Only the pinned tagger may translate known rejected-candidate `+Inf` stationarity paths for JSON. Raw checkpoints retain the original trace. `diagnostic-audit.json` binds the raw trace digest, changed paths, tagger hash, checkpoint pointer/hash, settings and experience hashes. NaN, negative infinity, pre-tagged raw data, and nonfinite physical/experience values fail. No convergence, nonlinear-price, or current-policy Q claim is added.

Final files are `settings.json`, `latest.json`, raw immutable checkpoint(s), `trace.json`, `experience.pt`, `observation_schema.json`, `diagnostic-audit.json`, `summary.json`, `timing.json`, and a hash-bound `completion.json`, plus operation/session/process/status evidence.

`readout.py` authenticates every slot, profile, source/model/settings/schema, full 75-row experience sequence, summary and receipt. It hashes raw physical checkpoints but never deserializes them or boots the simulator. Missing/extra completed trajectories, duplicate run IDs, mixed models, wrong profiles/seeds, canonical evaluation flags, or integrity failures are rejected.

Summaries include TTT and area totals; interval TTT/inventory arrays; terminal/peak inventory; nominal action saturation (exact and 99%); cap/zero-NUF counts; request/execution/control diversity and drift; fallbacks and D-ramp closures; PFO/recovery/candidate counts; endpoint queue exposure; actor/full-decision/plant wall and CPU timing; and locked worker-session timing.

The predeclared health screens are each scenario's zero-NUF requests <=5 and terminal inventory <=550. A failure is recorded after completion and does not trigger recollection, another wave, or early physical stopping. Readout exits successfully on an integral completed wave even when `all_health_gates_pass` is false; parent must inspect that field. No paired carry exists for these new profiles, so no percentage comparison is emitted.

Timing scope runs from before locked session-start publication through authentication, boot, reset/restore, inference, steps, validation, checkpoints, data export and close. It excludes final end/timing/summary/completion publication, lock release, interpreter startup/teardown, parent work and offline analysis. Missing interrupted session ends yield `UNKNOWN`, `elapsed_wall_seconds=null`, and a known lower bound. Actor time is one component of decision latency. Readout does not sum parallel worker sessions as wave elapsed wall time; parent owns that measurement.

## Verification And Limits

Final command, run from the repository root:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_wave_20260930/run_tests.py
```

Final result: **70 passed, 0 failed, 0 errors, 0 skipped**, 42.23 seconds of pytest execution. The evidence command includes prior read-only authentication/parity separately from this test duration. Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, Torch 2.14.0+cpu, PyYAML 6.0.3.

The suite covers strict actor tensors/bounds/dtypes/nonfinites, spec/phase/counters/gate/continuation failures, import collisions, changed hashes, all STOP levels, kernel duplicate refusal, completed refusal, partial prefix resume, source/model/settings mutations, exact retained action/observation/accounting/terminal constraints, actuator violations, checkpoint boundary/options/normalization failures, unknown interrupted-session timing, restored-observation mismatch, uncertain in-flight refusal, retained raw orphans, diagnostic +Inf boundaries, health threshold equality/inequality, all five health gates, and missing/duplicate/mixed-policy/wrong-profile readout rejection.

The end-to-end synthetic test executes a one-interval 155 prefix, resumes it and completes the other four synthetic scenarios: 375 synthetic steps, five resets, one restore. The prefix checkpoint hash and first transition are preserved. A test guard rejects physical-checkpoint deserialization by the completed-wave readout.

Read-only evidence:

- All **1500 saved current/next observations**, shape 1500x2367, produce **bit-identical** original-Actor and adapter outputs in separate interpreters.
- Observation bytes SHA-256: `5b2c6595427ef1cd7a6a5fd8acd2658802f1e02a4f89153228567143e3b28723`.
- Action bytes SHA-256: `8f08b8146edface8c07ebab80b9522ff123684500e0ec28ac0a2cc7f8c381e09`.
- An isolated read-only boot/profile-construction check resolves the actual frozen physical `runtime.py`, matches all five new profile hashes, and has no learner/data/td3/train_round modules loaded. It calls no reset/step/solve.
- The frozen pure actuator validator accepts **750 preserved real predecessor control rows** without a rollout.
- **351 preserved files** have identical before/after SHA-256 values, including Task 1 outputs/checkpoints/model/sources and admitted data/physical inputs. Full manifest is in final `evidence.json`.

The initial sandbox check could list `.deps-budget` but could not read several frozen package files/metadata, so it failed closed on runtime identity. Authorized checks were rerun through sandbox escalation with the existing environment. No dependencies were installed, no ACLs changed, and no runtime requirement was relaxed. A missing `_t` parent directory in the test runner and a package-relative import in read-only control reconstruction were fixed within the owned directory before final verification.

Limits: no actual Task 2 integration interval or physical wave was run; no actual TTT, health, live checkpoint serialization/resumption, latency or improvement result is claimed. The real prefix is the parent's next integration check after review. No earlier learner/probe/full-source suite was rerun, no learner was fitted, and no Task 3 decision was taken.

## Evidence And Source Manifest

Final evidence folder: `work/sdmpc_rl_return_wave_20260930/test-evidence/a64504a6/`.

| File | SHA-256 |
| --- | --- |
| evidence.json | eae903be76dec2285f0af725e26a75a8e4f1b5c2a127ac939e47bd4c234b279a |
| parity-authentication.json | 342259500f95a29abceedbcf03efb766d62ebf8339713dacd90022d96746d7e2 |
| tests.xml | bd5339490cff57f2298a5d089dd046bc5482957888ddc683a9a98a0e84b55391 |

All following paths are under `work/sdmpc_rl_return_wave_20260930/`; these are the final tested source hashes, also recorded in the evidence:

| Source | SHA-256 |
| --- | --- |
| actor.py | fd0c9156d004dac97cdb6e828f6f7f0922b811eb59425e01810eaf10f5a7312f |
| checks.py | bd672f57b050676919a8e2f4dc87e0e50d1d51d1643854e2af2584d7b9c38f27 |
| conftest.py | d50be148f29f7f1ee7719cf8754905b87a44475e795f2fd170cdd12b2fa9dbf6 |
| parity.py | ed0bcb9e880fc0af413e8dc0c8b7a8e76ddfb1082a9025c73c909999b41d1f18 |
| readout.py | 3c1daac9f1f26abb9f65428b01ae55f4a475f7661899776cd341b0016a210bdb |
| run_tests.py | e4c7758eeb7061458910da73a04edc82f5886d8788d842d15e064f90a2c65617 |
| test_wave.py | 359aa238f1db126830bbedcfe627e7d62e4cd1cd2fcd725126ec0a78d1cecaae |
| wave_support.py | 9dded90afa56572927ea81fc26faa8f2af72f78e6c9cb0eeb63624dc93ed95a1 |
| worker.py | f0adb61ffe9d762611e018302edfb00ebd8fc3e5ecc612084c206dae02f7c655 |

## Parent Integration Commands

Commands below are documented only, **not executed during implementation**. Working directory is the `RL` repository. Parent owns review, budget checks, hidden launch/redirection, process enumeration and wait/drain records. Use the same existing Python environment as the admitted predecessor.

Read-only parity/authentication can be rerun first:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_wave_20260930/parity.py
```

Actual one-interval 155 integration prefix after parent review:

```powershell
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_return_wave_20260930/worker.py --scenario sweet_155_w --max-new-steps 1
```

After parent validates that prefix and its worker has exited, these are the five worker command lines for the bounded wave. Resume 155 in place; never delete, reset or recollect its prefix. Parent may dispatch these concurrently within its five-worker/global-eight admission and must record/wait every actual worker, not merely the redirector launchers.

```powershell
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_return_wave_20260930/worker.py --scenario sweet_155_w --resume
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_return_wave_20260930/worker.py --scenario sweet_170_w
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_return_wave_20260930/worker.py --scenario sweet_170_incident_w
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_return_wave_20260930/worker.py --scenario sweet_170_skew15_w
.venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_return_wave_20260930/worker.py --scenario sweet_190_w
```

Read-only completed-wave readout after all actual workers drain:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_wave_20260930/readout.py
```

Readout writes JSON only to stdout and does not modify completed artifacts. Parent may retain stdout in its own reviewed readout destination. Worker exit 0 means a valid checkpointed prefix or completion; exit 2 means STOP; exceptions fail nonzero. Inspect `latest.json`/`completion.json` to distinguish prefix from completion. Parent reviews actual integrity/health/timing and decides Task 3 separately.

## Fix Round 1: Partial Restore Accounting And Method-Spy Coverage

Status on 2026-09-30: the unaffected accounting/testing scope in `task2-fix1-brief.md` is implemented and tested. **R1 remains unresolved and Task 2 is not self-approved.** The parent's requested user interpretation of the inherited reference-reconstruction exception is still pending. No actual prefix or wave was dispatched. Earlier integration commands in this report remain documentation only and are not authorization to dispatch before that decision and parent review.

Qualification of the original prepared-reference claim: the raw checkpoint retains the prepared reference, but the unchanged frozen `BudgetEnv.restore` creates a new controller and calls `prepare_reference` once on a successful nonterminal (`k < 80`) restore. It does not simply reuse the prepared lower evaluation/cache. The pinned terminal (`k == 80`) branch does not call `prepare_reference`. This inherited reconstruction does not repeat plant intervals or PFO; it is the extra-reference-preview conflict identified by the independent review. Accounting does not resolve whether it is allowed. No restoration/physics redesign or frozen-source edit was made.

### Scoped Changes

All Python paths below are under `work/sdmpc_rl_return_wave_20260930/`.

- Added `restore_accounting.py`: wall and process CPU measurements bracket the entire unchanged `env.restore` call. The scope includes its validation, copies, observer work and reference reconstruction. These are **not isolated reference-only measurements**.
- A durable `sessions/<id>.restore-start.json` binds session-start hash, run/settings identity, source checkpoint path/hash/progress, pinned source hashes, `checkpoint_k`, saved reference/observation timing, and the source-derived expected reconstruction count (1 nonterminal / 0 terminal). A corresponding restore-end record binds the start hash, call outcome, measured wall/CPU and inherited timing after a successful return.
- `worker.py` records resume intent, calls that wrapper, and binds restoration-file hashes in the enclosing session-end record. A returned call must leave inherited reference/observation timing unchanged. Neither checkpoints' saved timing nor trace decision timing is increased by the wrapper.
- `wave_support.py` validates restoration records and checkpoint hashes, then includes `restoration` under the existing session timing. `checks.py` and `readout.py` clarify its scope. Completed readout still never deserializes raw physical checkpoints. Existing timing/summary/completion hashing binds the nested provenance.
- Restore wall time is already within the worker session total. It is reported as a component and is **not added again**, assigned to the preceding/next decision, or treated as wave elapsed wall time. Record-publication/verification overhead remains within the wider session scope, outside the measured restore call.
- A missing restore end keeps duration and reconstruction completion `UNKNOWN`. An interrupted resume session with no restore-start record also retains unknown restore scope. A raised call can have measured duration but unknown reconstruction completion; its expected branch count is not advertised as a completed/measured preview count. Known components do not turn a missing enclosing session end into known session elapsed time.
- Added `test_restore.py`, registered in `run_tests.py`; extended `test_wave.py` for resume/readout accounting and tamper rejection. `conftest.py` now copies saved timing in its synthetic restore to match the existing frozen contract. `actor.py` and `parity.py` are unchanged from `before-task2-fix1/`. No coordinator or dispatcher was added.

### Verification

Commands executed from the `RL` repository using the existing environment (authorized sandbox escalation only for existing dependency access):

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_wave_20260930/run_tests.py -k 'real_frozen or interrupted_restore or restore_not_started or restore_provenance or changed_inherited'
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_wave_20260930/run_tests.py
```

- Focused run `f2bec8e8`: **25 passed, 70 deselected**, 1.67 seconds.
- Full run `d45d6eca`: **95 passed**, no failures/errors/skips, 49.959 seconds.
- The focused tests compile the unchanged, hash-pinned `BudgetEnv` and `BudgetController` class definitions into an isolated namespace and execute their real `restore` / `__init__` / `prepare_reference` bodies. They use the real frozen `Observation` class, synthetic in-memory state/control/config, and spy coordinates, mask, lower begin/evaluate/execution-check dependencies. The real physical lower runtime is never called by these spies; reset/plant-step/PFO/lower-solve sentinels fail if invoked.
- Successful cases cover `k=5` (`pfo_initial`), `k=6` (`previous` and `pfo_recovery`), `k=79`, and terminal `k=80`. Each nonterminal case observes exactly `lower.begin`, `lower.evaluate`, `lower.execution_check` once; terminal observes none. They verify exact observation/schema, prepared boundary, reference source/control/point/budget, previous-executed-budget anchor (initial reference-budget anchor at `k=5`), dual/previous-action memory, simulator clock/state, copied PFO memory, and unchanged saved decision timing. Terminal has no active prepared controller/anchor, matching the frozen implementation.
- Deterministic clocks establish honest whole-call accounting (synthetic 3 wall / 2 CPU seconds inside an enclosing 100-second session, never 103). These numbers are test inputs, not measured traffic-runtime latency. Failure/lost-publication tests retain unknown reconstruction/duration where appropriate. Tamper tests reject session/checkpoint/source/settings/branch/hash/timing changes, negative or nonnumeric durations, and restore wall duration exceeding its enclosing session.
- Synthetic prefix/resume retains the first trace row unchanged. Five-scenario synthetic completion/readout includes per-scenario restore provenance; changing a restore-end record is rejected. Readout is exercised with raw-checkpoint deserialization forbidden.
- Read-only parity remains bit-identical on **1,500 observations**. Frozen pure actuator validation accepts **750 preserved predecessor control rows**. Import/profile construction performs no reset, step or solve, and loads no learner/data/td3/train_round modules.
- **351 preserved files** have identical before/after hashes, including Task 1 model/source/data and frozen physical inputs. Tested worker-source hashes were also unchanged through each run. The actual `results/sdmpc_rl_balanced_goal_20260930/return_policy_wave_v1` directory remained absent after verification.

No actual physical simulation, reference preview, PFO/optimization, training update, scientific-policy change, dependency install, ACL change, parent-document edit, or commit occurred. Actual physical serialization/resume effects, traffic metrics and real restore latency remain unverified. Parent owns the pending interpretation, review, dispatch/budget and actual-fit evidence.

### Fix-Round Evidence And Hashes

Evidence files are under `work/sdmpc_rl_return_wave_20260930/test-evidence/`. Both runs retain full source and 351-file preservation manifests in `evidence.json`; synthetic pytest artifacts are under the owned `_t/` directory.

| Evidence | SHA-256 |
| --- | --- |
| f2bec8e8/evidence.json | 448ccb2e1fc44971ec370f922f1766302e7ab26354b3f5a72622d54ee30e2c12 |
| f2bec8e8/tests.xml | 95ca6ef6bb220a71768f9c8faafdf28d1593c7f69ac464acaabbe06c4466d504 |
| d45d6eca/evidence.json | 0f40c58486743975c5ac54ed7155e4c96e814c2adddac48c1f3169d69f3bdae6 |
| d45d6eca/tests.xml | e0c79175a9da5bee9daa3935b49032c8a955b8c0524557861121ad6cdaa6bb35 |
| Both runs: parity-authentication.json | bbfa1c334195fda97b1e804c298f2ba1576a113f8323d73c3ed76e6a0bc6961a |

Final tested source manifest (same in both runs):

| Source | SHA-256 |
| --- | --- |
| actor.py (unchanged) | fd0c9156d004dac97cdb6e828f6f7f0922b811eb59425e01810eaf10f5a7312f |
| checks.py | eb197441edb80876ebbf0a3ec0438b9f0f9f294662d4bc37b979b89e570e0450 |
| conftest.py | bb93f70e1843438526fcb5a48c6b2bcefd908b41e01bd30dce672681a0507c47 |
| parity.py (unchanged) | ed0bcb9e880fc0af413e8dc0c8b7a8e76ddfb1082a9025c73c909999b41d1f18 |
| readout.py | 3a3c24a12ecdffb74cb79d64d337267be0a2973db70fe5692475172fe23320e8 |
| restore_accounting.py (new) | e0f654f3b6cd6dac26c5a89f3c792f6e4142c13e94e081368eb9c0e60f0e67d9 |
| run_tests.py | 0009ca2b6b92f8585031113e266a105f4fbfb19f98ccf5b22553640fa88705c6 |
| test_restore.py (new) | 6ba1d96b4ccc12101617e02d5942bfce270c6f278b5fbb1a13c27fd59d19ac20 |
| test_wave.py | ed74494de9d4b2725dbc8a81fd685a48702060a9d88f354d634a92d1bf1eb90d |
| wave_support.py | b9af68ebbe4e0b8992bd1ee672b01b0b87b1829d29635cd487f58e3d44e21418 |
| worker.py | 395948daea7d0b472da03686a21d992246dbfe24f2c4081f45c95f1ea2056709 |

Unchanged frozen method-source pins: `work/sdmpc_rl_multi_20260929/budget_env.py` = `6ccb4ef400106e29282797ec5d69b7a8dbeee5a1310bd7d6f97b7c7e0c7ee2c4`; `work/sdmpc_rl_multi_20260929/budget_controller.py` = `0402511ae8e965e315b4b9808e23bbdc9c2fd274666a983b2a3303e6b916f9c2`.

Changed scope: seven existing Python files plus the two new Python files above, generated evidence/synthetic fixtures under the owned wave folder, and this appended report only. Parent-owned brief/review/docs/launch/budget files were not changed. Passing source tests/accounting is **partial progress only**, not resolution of R1 or approval for a physical run.
