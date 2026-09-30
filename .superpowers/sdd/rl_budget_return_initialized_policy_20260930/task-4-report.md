# Task 4: Fixed-policy canonical evaluator

## Status and scope

Implemented for independent review. Final focused suite: **49 passed**, zero
failures/errors/skips, 55.567 seconds. Actual read-only authentication reconciled
all five completed carry centers and proved **375/375 bitexact actor actions**.
All **1,473 retained input files** have identical before/after SHA256 maps.

**Actual canonical evaluation outputs are ABSENT.**
`results/sdmpc_rl_balanced_goal_20260930/return_canonical_v1` does not exist.
No actual boot/reset/step/restore, physical checkpoint deserialization, optimizer
construction/update, training, simulation, reproduction, dispatch, install,
commit, or push was performed. No actual parent approval receipt was authored.
Synthetic test receipts are explicitly labeled and confined to the new test
fixtures; they cannot admit the real source or input manifest.

Work was limited to the new evaluator directory and this report. Existing dirty
work and completed code/data were not edited. The brief and the MC results note
were read; no entire plan/history was read. The parent retains global worker
budget, physical input inventory, independent review, and launch responsibility.

## Changed paths

All following paths are under `work/sdmpc_rl_return_eval_20260930/`:

| File | Purpose |
| --- | --- |
| `support.py` | Exact contract, pinned dependencies, STOP scopes, preservation maps, checkpoint/lock/session reuse |
| `policy.py` | Strict MC archive authentication, reused meta-only actor, 375-observation action parity |
| `checks.py` | Explicit canonical metadata/environment adapters; physical/accounting/trace/checkpoint validation; complete timing and coverage summaries |
| `baselines.py` | Isolated read-only legacy center loader, additional control/accounting checks, actual center hashes |
| `preflight.py` | Read-only preflight, source/spec/input binding, parent-review admission gate |
| `worker.py` | One locked canonical scenario, exact checkpoint resume, immutable evidence and stable finalization |
| `readout.py` | Read-only partial/full readout; all-five matched comparisons and strict acceptance |
| `conftest.py` | Synthetic ledger fixture and actual physical/optimizer blockers |
| `test_eval.py` | Focused contract, mismatch, STOP/lock/resume/publication/readout tests |
| `run_tests.py` | Guarded tests, actual authentication/parity, before/after preservation evidence |
| `evidence/` | Retained preflight, test XML, source/input maps and run results, including earlier debugging runs |
| `_t/` | Retained synthetic-only fixtures/checkpoints and explicitly synthetic review data |

Also added this named `task-4-report.md`. No old runner was modified.

## Implementation and reuse

The physical process constructs only the frozen 2367/64 ReLU/64 ReLU/2 tanh
actor with bounds `[0.2, 0.1]`, using the unchanged Task 2 meta-construction
definition. It never imports Task 1 generic runtime/learner/data, Task 3
learner/data, or TD3. No critic inference or optimizer exists on this path.
`policy_q=None` is explicit in settings, trace, and summary. The authenticated
archive retains its learning components as data without constructing learners.

The unchanged Task 2 helpers provide actor inference, operation markers,
kernel-backed single-writer locking, immutable interval checkpoints with atomic
latest pointer, strict physical observations/accounting/controls, diagnostic
Inf tagging, and entire-call restore accounting. Selected unchanged definitions
are compiled into explicit private namespaces with the new model/format and
validator bindings. Old modules' globals are not rebound to canonical metadata.
The original training-only validators still reject canonical settings and
profiles; focused tests show the new explicit canonical contract accepts them.

Canonical environment construction is exactly
`BudgetEnv(rt, scenario=..., training_seed=None, guard_mode='physical')` against
the unchanged frozen snapshot. Original reset, five warmup intervals, 75
controlled intervals, 180-second control interval, 14,400-second terminal,
gamma 1 and `-interval_ttt/100` remain. Actions retain native float32 and use the
existing float64 anchor-plus-`[50,1000]*action` projection. The anchor carries
the previous executed budget. No independent policy-base memory was added.
Signed NP, NUF in `[0,6000]`, initial/recovery PFO, previous-control reference,
one lower candidate, and the physical feasibility guard are preserved.

The legacy `compare_runs.load_completed_run(folder, 'center', [None])` executes
only in an isolated, hidden, read-only process. Its TD3 import does not enter
the actor process. In that child all physical entry points, optimizer
construction and tensor deserialization are blocked. It authenticates all five
centers' complete timelines, terminals, warmup/interval/area totals, profiles,
source/runtime/environment/schema, no-learning/no-Q behavior, reference/PFO
accounting, controls, budget projection and fallback feasibility. It hashes
current retained center files, including raw checkpoint bytes without loading
them. Canonical profile contents are only read and hashed, never simulated.

| Scenario | Reconciled canonical carry TTT |
| --- | ---: |
| sweet_155_w | 3103.0110715680044 |
| sweet_170_w | 3935.903236508048 |
| sweet_170_incident_w | 5546.224352256691 |
| sweet_170_skew15_w | 4250.876599300032 |
| sweet_190_w | 6604.2970168093225 |

Admission requires a parent-authored receipt binding the exact source-map
digest, spec digest, preflight file SHA, model SHA and frozen physical contract.
It must declare `approved_for_physical_evaluation` and identify both reviewer
and parent admission. Admission runs before physical startup. There is no
receipt generator or automatic next-stage queue.

Each scenario has its own one-thread worker. STOP markers at REPO, GOAL, new
root and scenario remain in place. Worker identity records PID, creation time,
parent PID, executable and original command. Uncheckpointed operation markers
refuse resume; raw orphan evidence remains. A contradictory final export beside
a partial checkpoint is rejected before boot. Completed duplicates are refused.

Nonterminal resume calls the unchanged complete `env.restore` once, checks
exact observation equality, and measures the entire call's wall/process CPU.
Saved interval timings stay unchanged. Reference reconstruction is reported as
source-derived, not a measured preview count. No plant interval or PFO is
repeated. A stop after step 75 but before export resumes the k=80 boundary with
zero reference reconstruction and no new plant/PFO. Once `finalization.json`
exists, resume performs publication only, without boot or simulator unpickle.
Retained tensors/trace/audit/timing/summary are compared and preserved, never
silently overwritten. Synthetic tests verify exact retained hashes across STOP.

Readout hashes raw checkpoint bytes but never deserializes physical state. It
keeps all five rows, including absent, partial, invalid and non-improving rows.
Partial progress comes from the authenticated latest pointer and operations;
it is not labeled a complete performance measurement. Complete comparisons
report positive improvement as baseline minus candidate, raw and percent,
interval differences and windows 1..25, 26..50, 51..75, inventory, requests,
executed budgets/physical controls, fallback/PFO counts and timing components.
All five must individually exceed `max(1e-6, 1e-8*baseline_TTT)`. Success is only
`eligible_for_separate_reproduction`; `goal_achieved` always remains false.

## Exact identities

Final source-map digest:
`9a9de385546732cd158295fdd03486cc928864930d2671b8b9e5f45c077314b2`

Evaluator spec digest (full structured spec retained in preflight/evidence):
`c4d3f93ba94da908ffa3f79fbb2dfc1c22f0e608730078679ca37b50e96c10c2`

| Identity | SHA256 |
| --- | --- |
| MC model | `820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6` |
| MC completion | `a815f65f315ce1a73cce8a3754b9edb3fac9dbed6a8fd0aece9b9238ea532675` |
| MC model spec | `3610bb43c22430a3fb9ba95f7f14175b2b29b68a2000d6bfe1af04ec4c8ff811` |
| Actor tensors | `9956ba4f6f5d64c7a65bf05bdab47984efe041e15358dafebbb9bfd731d44877` |
| Physical contract | `d6d14c4c18a7f8cfc88969bf0e9b8231ba5c452600593db49b125c1e5730eb70` |
| Projection completion | `b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3` |
| Final preflight | `14335232c3358611c8605c9aa2db65057803d845d9646e1c326dbe347621a366` |
| Final test evidence | `df4cc24aacea783bad11872319b9673c8a0e501ecb23a7be50fce021f3ba20b3` |
| Final test XML | `c58e91c723e3317427ecc8c1e404699218b6be24f5664b86a058177fe09f3f7e` |

Final per-file source SHA256:

```text
a9301be0564741cf95d5c986931c5964670b8f87081b79164a44295e67b604c2  baselines.py
f6810ec3991f6aab78e4072cae26d9dfe3f08dbc8c8298a31ffc3b35b9d7d046  checks.py
50ca612d6346ff85aa0d2c9abe2902c25faeeff62073938984b7d74522455c33  conftest.py
5af616017d71f0e284cd32198b3d4355503ce79a042b9020f4cc89c2529a0f0d  policy.py
b28e0ad48c0d13d2929e1686f4c667111e6ca0727e65799360953386abba3eb4  preflight.py
bcac1d51fa278bc96b1fc7369833663aa21493ee1b054760061a4b5e9de321a0  readout.py
b2c21948323f7f3e5e7e4329efab5b237093c88e165a7343d981e734f9d02081  run_tests.py
36fba2e310a564c0a23d520ff0e9cf85b80b571bd97ef717f470856e11ac0ce0  support.py
ad146cd0391d3d4f7e867de882e9d8e7eb9d286c07301117eec15d555d91ec0e  test_eval.py
9f0b4b9d6f9c66014eb804807a674a46cbf9b7ecae94d8fd408f01fc2a3b3be3  worker.py
```

## Verification evidence

Final artifacts:

- `work/sdmpc_rl_return_eval_20260930/evidence/0d0f12c3/preflight.json`
- `work/sdmpc_rl_return_eval_20260930/evidence/0d0f12c3/evidence.json`
- `work/sdmpc_rl_return_eval_20260930/evidence/0d0f12c3/tests.xml`

The evidence contains the executed command, exact full source map/spec, runtime
versions, test result, and separate complete before/after SHA maps. Preflight
contains center evidence hashes and summaries, actual profile/physical contract,
and per-scenario parity evidence. No model/data identity rests only on a literal
TTT comparison. Runtime: Python 3.12.14, Torch 2.14.0+cpu, NumPy 2.3.5,
SciPy 1.16.3, PyYAML 6.0.3, using the existing `.venv-torch` interpreter.

Coverage includes MC format/spec/phase/counters/ancestor/continuation/actor
mismatch; meta actor without RNG draws or gradients; float32 input/output;
canonical new-versus-old validation; evaluation-only/no-Q/no-update flags;
physical guard/one-candidate/accounting failures; center source/runtime/env,
profile/TTT/schema/seed failures; receipt/source/spec/file-hash changes;
four STOP scopes; kernel writer lock; boundary resume; completed duplicate and
uncertain-operation refusal; both after-75 publication paths; retained-output
refusal; failed-checkpoint orphan preservation without pointer publication;
partial readout without simulator loads; all-five acceptance, single failure,
missing/extra/duplicate cases, and threshold neighbors.

The test harness blocks actual physical methods, optimizer construction/update,
and actual simulator checkpoint loads. Only the fixed MC archive and Task 2
experience arrays may be deserialized from actual data; all executable test
trajectories use the small synthetic ledger. Actual canonical profiles are read
only for authentication hashes. No automatic prefix simulation was run.

Prior successful Task 2 suite evidence was reused, not rerun:
`work/sdmpc_rl_return_wave_20260930/test-evidence/d45d6eca/evidence.json`,
SHA `0f40c58486743975c5ac54ed7155e4c96e814c2adddac48c1f3169d69f3bdae6`:
95 tests passed, 49.959 seconds, 351 preserved files, zero physical steps and
optimizer updates. The brief supplies the independently proven actual restore
context. The MC note supplies the prior 46-test fit-suite evidence; no fit suite
or whole-history rerun was needed.

Retained development evidence: `9bf6a45c` records 19 passed/21 fixture path-length
failures before synthetic paths were shortened; `13836dc6` records the first
40-test pass. Initial adapter checks exposed and fixed the isolated legacy
import order and episode-level baseline metadata normalization. The bundled
base Python lacked Torch; the existing `.venv-torch` was selected without
installation. Reading an existing dependency required sandbox escalation; the
guarded read-only/test commands succeeded with it. No automatic approval review
rejected an action.

## Parent commands

Run from the RL workspace. These commands are for the parent; no physical launch
was executed by this implementation task.

Scoped verification:

```powershell
& .venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_eval_20260930/run_tests.py
```

Optional fresh read-only preflight to a new, nonexisting path:

```powershell
& .venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_eval_20260930/preflight.py --output work/sdmpc_rl_return_eval_20260930/evidence/parent-preflight.json
```

The exact tested admission file is `evidence/0d0f12c3/preflight.json` under the
new source root. Use the same admitted file for all five workers. The parent
must independently review the source and author its receipt; the worker's
required receipt fields are `format=sdmpc-fixed-policy-canonical-v1-parent-review`,
`decision=approved_for_physical_evaluation`, `source_sha256`, `spec_sha256`,
`preflight_sha256`, `model_sha256`, `physical_contract_sha256`, nonempty `reviewer`
and nonempty `parent_admission`. Required hashes are above. A different source,
spec or preflight needs a matching new review; there is no bypass flag.

After independent review/admission, an example single hidden worker launch:

```powershell
$task4Root = (Get-Location).Path
$task4Source = 'work/sdmpc_rl_return_eval_20260930'
$task4Preflight = "$task4Source/evidence/0d0f12c3/preflight.json"
$task4Review = '.superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-4-parent-review.json'
$task4Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$task4Scenario = 'sweet_155_w'
$task4Log = ".superpowers/sdd/rl_budget_return_initialized_policy_20260930/task4-$task4Scenario-$task4Stamp"
$env:PYTHONUTF8 = '1'
$task4Arguments = @('-B', '-u', "$task4Source/worker.py", '--scenario', $task4Scenario, '--preflight', $task4Preflight, '--review-receipt', $task4Review)
Start-Process -FilePath "$task4Root/.venv-torch/Scripts/python.exe" -ArgumentList $task4Arguments -WorkingDirectory $task4Root -WindowStyle Hidden -RedirectStandardOutput "$task4Root/$task4Log.stdout.log" -RedirectStandardError "$task4Root/$task4Log.stderr.log" -PassThru
```

This starts only one chosen scenario. Parent dispatches each of the five exact
scenario names above under the global eight-worker budget, retains launcher
identity and verifies actual worker PID/creation/command and process drain.
Logs are outside the fresh scenario slot. Resume only an admitted, retained,
uncompleted slot by appending `--resume`; the runner refuses uncertain inflight
operations and completed duplicates. `--max-new-steps N` is available for a
parent-requested interruption, but no actual prefix test is requested here.

Read-only status/full readout (stdout; parent may retain it outside run slots):

```powershell
& .venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_eval_20260930/readout.py --preflight work/sdmpc_rl_return_eval_20260930/evidence/0d0f12c3/preflight.json --partial
& .venv-torch/Scripts/python.exe -B work/sdmpc_rl_return_eval_20260930/readout.py --preflight work/sdmpc_rl_return_eval_20260930/evidence/0d0f12c3/preflight.json
```

Full readout returns nonzero for incomplete/invalid evidence and retains every
scenario's status/error. A complete negative performance result has
`failed_acceptance` in JSON and is a valid measurement, not a collection failure.
No readout path dispatches reproduction or feeds evaluation data into training.

## Residual concerns and limits

- Actual canonical performance is unknown; this is implemented/tested code, not
  physical evaluation or an achieved goal. Independent review and parent
  admission are still required before any physical execution.
- No fresh actual restore test was run. Unchanged Task 2 restore code and prior
  evidence are reused; the new wrapper is tested with synthetic boundary state.
- Session timing excludes pre-lock review/input authentication and publication,
  interpreter and parent overhead, as explicitly stated in the retained timing
  scope. Whole restore wall/CPU includes all work in `env.restore` and is already
  charged once inside session time. Do not add it again. An interrupted session
  has unknown duration; known sums are only lower bounds.
- Actor inference is only one measured decision component. Forecast,
  observation, PFO, reference, lower solve, guard, plant and session scopes are
  retained separately. Parallel worker sums do not measure elapsed wall time.
- Peak inventory and queue exposure are based on retained controlled-interval
  endpoints, not exact within-interval extrema/exposure. No preview-free or
  nonlinear-price causation claim is made.
- Passing all five only establishes eligibility for separate reproduction.
  Even deterministic repeated all-five success would not establish stochastic
  generalization. Failure never drops a scenario or changes physics/profiles.
- Preserve the evidence and orphan files. Never overwrite this completed model,
  old centers, old training wave, or a completed canonical slot to retry.

