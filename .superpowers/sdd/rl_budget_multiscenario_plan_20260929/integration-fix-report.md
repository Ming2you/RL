# Consolidated integration fix round 1

Date: 2026-09-29. Repository: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

## Result and scope

I1-I6 are implemented. Initial fix-round focused execution: **199 passed in 56.82 seconds**,
exit code 0. This includes all original seven failing regressions. Actual pytest
execution-evidence hooks also reconciled **199 XML identities / 199 collected /
199 passed**. There are no xfails or skipped tests. The subsequent I2 smoke-runtime
amendment below records the latest result: **205 passed in 49.19 seconds**.

This is an implementation/test report, not an independent admission approval.
No actual smoke, traffic, full source-bound admission run, commit, agent, install,
or real subprocess child was run. No real PASS attestation or pilot gate was
issued. Complete fake approval/evidence artifacts exist only in pytest temporary
fixtures and explicitly identify their reviewer/report as synthetic.

Changed files, relative to RL:

- `work/sdmpc_rl_multi_20260929/train_round.py`
- `work/sdmpc_rl_multi_20260929/run_pilot.py`
- `work/sdmpc_rl_multi_20260929/compare_runs.py`
- `work/sdmpc_rl_multi_20260929/build_preflight.py`
- `work/sdmpc_rl_multi_20260929/run_tests.py`
- `work/sdmpc_rl_multi_20260929/test_multi_contracts.py`
- `work/sdmpc_rl_multi_20260929/test_multi_pilot.py`
- `work/sdmpc_rl_multi_20260929/test_multi_runner.py`
- This report.

`run_budget.py` did not need changes. TD3, its tests, all physical modules, the
analysis helper, old experiments and actual experiment data were not edited.
The original task-2 test report remains a historical record of the pre-fix failures.

## Changes by finding

### I1: Typed admission and reusable consumer validation

`build_preflight.py:58` implements `validate_evidence`, shared by the builder and
`verify_gate` (`:122`). The new gate format is `sdmpc-multi-admission-v2`.
Exactly nine named evidence roles are required. References have absolute paths
and lowercase SHA-256 hashes; different roles must refer to different files.
Legacy, empty, incomplete and arbitrary evidence maps are rejected.

Both creation and consumption recheck current production pins, current runtime,
the complete current `test_*.py` hash map, XML hash, test exit status, exact test
execution identities, five role-specific smoke scenarios, canonical forecast
digests, common schema, serialized resume and physical/carry audits, reviewer
attestation and human report hash. A later test edit invalidates a cached gate.

`run_tests.py:11` records real pytest execution identities in XML properties and
in the source-bound record. An identity is the absolute test file path followed
by `::` and the test/class/parameter suffix. A test counts as passed only after
setup, call and teardown all pass. Admission reconciles XML identities with both
collected and passed lists, rejects duplicate identities, and requires every
current test file to have executed. The six required test filenames are also
explicitly required, so deleting a suite cannot produce a smaller admissible set.
The minimum count of 60 remains an additional check, not the suite-coverage check.

### I2: Independent final reviewer attestation

The builder no longer searches prose for `MULTI_REVIEW: PASS`. It requires a
separate JSON attestation, passed as `--attestation`, with a final admission
decision and exact reviewed source, test, runtime and evidence identities. The
attestation binds the human report hash as well as all test/smoke hashes.
Code-review-only, stale, rejected, superseded, or conflicting extra verdict fields
cannot admit a pilot. Neither production builder nor consumer creates approval.
The complete accepted schema and reviewer workflow are documented below.

### I3: Exact restored training state

`train_round.py:89` adds shared `reconcile_training`; `training_inputs` (`:109`)
reconstructs admitted inputs before a checkpoint can replace the learner.
Round 1 first validates the actual completed predecessor output, including its
own collections and provenance, then loads that immutable model and appends the
five current collections to form the expected replay.

Every replay column is compared with `torch.equal` in its stored dtype and shape:
float32 observations, requested actions, rewards and next observations, plus
boolean true-terminal flags. Equal-sized evaluation-like or otherwise altered
data is rejected. Seed must be 6300; per-scenario replay counts must be 75 or 150;
updates must equal `375 * round + completed`; cumulative sample counts must equal
eight times the total updates for every scenario.

Checkpoints now explicitly retain `training_run_id` and the full ordered
`training_profiles`; both must equal the admitted inputs/settings on restore.
`validate_metric` (`:75`) checks each saved update's absolute update number, exactly
eight samples per scenario, finite losses, and delayed actor-loss phase. In
particular, round 1 starts at absolute update 376 and includes actor loss on its
first update. Validation precedes any resumed gradient update.

The resumed learner replaces the newly reconstructed expected-input learner;
experience is not added again after restoration. Real round-0 and round-1
continuation tests prove bitwise equality of final weights, optimizers, RNGs,
replay, counters and metrics. TD3 math was not changed.

### I4: Completed training matches actual inputs

`train_round.py:133` adds `validate_training_output`, used by the pilot's
`validate_training`. It reconstructs current inputs and the admitted predecessor,
then compares model replay against all of them using the same reconciliation as
resume. It also requires exact ordered provenance, seed schedule, profile hashes,
collector experience hashes, training run identity, update/sample totals and
every metric's absolute phase. Completion hashes alone cannot substitute for
these checks.

### I5: STOP throughout launch and polling

`run_pilot.py:82` adds `propagate_stop`, used at startup, before every launch and
before/after every child poll. It checks root, stage and child output scopes for
all plan jobs, propagates a detected STOP to the root, and lets existing children
checkpoint and drain. Root STOP creation uses exclusive file creation, preserving
a preexisting user STOP even if it appears concurrently. Child/stage STOP contents
are never overwritten.

The five-worker assignments and cap are unchanged. The orchestration parent sets
OMP/MKL/OpenBLAS thread environment variables to 1 before importing TD3/Torch.

### I6: Experience admission before completion reuse

`compare_runs.py:150` now loads and calls `verify_experience` for every completed
collector. Completion reuse, newly completed child validation, comparison and
pilot startup therefore reject malformed payloads before sibling launch. The
trainer retains its independent `verify_experience` call after loading each
collection; its separate admission check was not removed.

## Focused coverage and execution

The original 140 tests were retained, with valid admission fixtures upgraded to
the complete new schema. Added 59 cases cover legacy/empty/incomplete/arbitrary
gate roles, test edits after gate creation, runtime/source changes, missing or
unexecuted suites, XML identity corruption, stale source/test/XML/smoke/report
approval, code-review versus final approval, superseded/rejected/conflicting
verdicts, reviewer smoke-runtime declarations, equal-sized replay substitutions,
all replay columns and terminal flags, ordered provenance/run identities, metric
update/sample/phase/loss drift, round-1 exact continuation, STOP during polling,
user STOP preservation, and parent thread limits before Torch import.

Real gradient updates remain synthetic-only: five artificial experience groups
feed the real TD3 learner in-process. All physical runtime/environment boundaries
and Popen children remain controlled fakes.

Working directory for both runs: RL. Existing `.venv-torch` and `.deps-budget`
were used with approved dependency ACL escalation, no package changes.

Common environment:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
```

First command:

```powershell
& '.venv-torch/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --confcutdir=work/sdmpc_rl_multi_20260929 --tb=short work/sdmpc_rl_multi_20260929/test_multi_contracts.py work/sdmpc_rl_multi_20260929/test_multi_pilot.py work/sdmpc_rl_multi_20260929/test_multi_runner.py
```

Result: `1 failed, 198 passed in 53.25s`, exit 1. The sole failure was a misplaced
test block: existing plan-drift assertions had been appended to the new paused
thread-budget test. They were moved back to their original completed-pilot test.
No production requirement or assertion was weakened.

Final command, additionally checking the actual execution-evidence hooks:

```powershell
$integrationScript = @'
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
here = Path('work/sdmpc_rl_multi_20260929').resolve()
sys.path.insert(0, str(here))
from run_tests import ExecutionEvidence
execution = ExecutionEvidence()
with tempfile.TemporaryDirectory(prefix='multi-fix-xml-') as directory:
    xml = Path(directory) / 'focused.xml'
    code = pytest.main(['-q', '-p', 'no:cacheprovider', '--confcutdir=' + str(here), '--tb=short', '--junitxml=' + str(xml), *[str(here / name) for name in ('test_multi_contracts.py', 'test_multi_pilot.py', 'test_multi_runner.py')]], plugins=[execution])
    cases = ET.parse(xml).getroot().findall('.//testcase')
    identities = [case.find('./properties/property[@name="nodeid"]').get('value') for case in cases]
    assert sorted(identities) == sorted(execution.collected)
    if code == 0:
        assert sorted(identities) == sorted(execution.passed())
    print('EXECUTION_IDENTITY_PARITY', len(identities), 'collected', len(execution.passed()), 'passed')
raise SystemExit(int(code))
'@
& '.venv-torch/Scripts/python.exe' -B -c $integrationScript
```

Final output:

```text
199 passed in 56.82s
EXECUTION_IDENTITY_PARITY 199 collected 199 passed
```

Exit code 0. One numerical Python process, exactly the three focused test files.
The temporary XML was removed by its temporary-directory context. No source-bound
full-suite admission evidence was generated by this run.

## Admission and attestation schema

All objects below reject extra top-level fields. SHA-256 strings are lowercase
64-character hexadecimal. Artifact references are `{ "path": "<absolute path>",
"sha256": "<file SHA-256>" }` and must point to distinct files.

Gate (`sdmpc-multi-admission-v2`) fields:

| Field | Required value or meaning |
| --- | --- |
| `format` | `sdmpc-multi-admission-v2` |
| `ready_for_bounded_pilot` | Boolean `true` |
| `source_pins` | Exact current `run_budget.pins(DEFAULT_SNAPSHOT)` object |
| `runtime_versions` | Exact current `run_budget.runtime_versions()` object |
| `scenarios` | Ordered `list(td3.SCENARIOS)` |
| `evidence` | Exactly the nine roles below, each an artifact reference |
| `tests` | Integer matching validated XML test count |
| `observation_dim` | Integer matching the common smoke schema |

Exact evidence roles:

```text
test_xml
test_record
review_report
review_attestation
smoke:sweet_155_w
smoke:sweet_170_w
smoke:sweet_170_incident_w
smoke:sweet_170_skew15_w
smoke:sweet_190_w
```

Test record (`sdmpc-multi-test-evidence-v2`) fields are exactly `format`,
`exit_code` (integer 0), `source_pins`, `runtime_versions`, `test_sha256`,
`xml_sha256`, `collected_nodeids`, and `passed_nodeids`. `test_sha256` maps every
current absolute `test_*.py` path to its file hash. The two node-ID lists must
match every XML `testcase`'s single `properties/property name="nodeid"` value,
with no duplicate IDs and no missing test-file coverage. The existing runner
produces this record; it is not reviewer-authored.

Independent final attestation (`sdmpc-multi-final-review-v1`) fields:

| Field | Accepted value or meaning |
| --- | --- |
| `format` | `sdmpc-multi-final-review-v1` |
| `review_kind` | `final_admission`; `code_review` is rejected |
| `status` | `final`; superseded or pending decisions are rejected |
| `decision` | `approved`; every other verdict is rejected |
| `reviewer` | Nonempty independent reviewer identity string |
| `source_pins` | Exact source object reviewed, equal to current source and test record |
| `runtime_versions` | Exact runtime reviewed, equal to current runtime and test record |
| `test_sha256` | Exact complete map of reviewed test-source hashes |
| `evidence_sha256` | Exactly eight role-to-file-hash entries: every gate evidence role except `review_attestation` |
| `smoke_runtime_versions` | Exactly five scenario keys, each mapping to the `runtime_versions` object recorded in that real smoke JSON; all must equal the admitted runtime |

The runtime-version object's keys are `python`, `numpy`, `scipy`, `torch`, and
`PyYAML`. The source-pin object is produced by `pins`: snapshot manifest hash,
frozen file count and the implementation-file hash map. Equality is against the
full current objects, not a subset. The human report can contain historical
verdicts, but only the separate, exact, final JSON decision is authoritative.

The attestation does not hash itself. The gate hashes the attestation; the
attestation hashes all eight other evidence roles, including its human report.
This avoids a circular hash dependency while binding all reviewed inputs.

## Reviewer creation and gate commands

These are future coordinator/reviewer steps, not actions performed in this task.
After source freeze, the coordinator must run the full source-bound synthetic
suite with `run_tests.py --output <fresh-evidence-directory>`, then obtain five
real canonical smokes. An independent reviewer must inspect those exact sources,
tests, executions, runtime identities and human report before authoring the final
JSON object above. Approval must not be copied from this implementation report.

The I2 amendment below makes the smoke producer serialize `runtime_versions()`
directly after boot. The independent reviewer's `smoke_runtime_versions` map must
copy those recorded objects from the five reviewed smoke JSON files. Admission
compares each recorded object with the admitted runtime and the attestation;
missing metadata is rejected rather than inferred from an external runtime.

The following creation command persists a complete reviewer-authored JSON payload
without generating a verdict or filling in current hashes. Replace the placeholder
with the independent reviewer's complete exact object. The output is created
exclusively so an existing decision cannot be silently overwritten:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$reviewerJson = @'
<complete independent reviewer-authored JSON attestation using the exact schema above>
'@
$writeAttestation = @'
import json
import sys
from pathlib import Path
payload = json.load(sys.stdin)
with Path(sys.argv[1]).open('x', encoding='utf-8') as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
'@
$reviewerJson | & '.venv-torch/Scripts/python.exe' -B -c $writeAttestation '<absolute final-attestation.json path>'
```

Once the reviewer has issued that file, gate construction is:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& '.venv-torch/Scripts/python.exe' -B work/sdmpc_rl_multi_20260929/build_preflight.py --evidence '<fresh-evidence-directory>' --review '<absolute independent-human-review.md path>' --attestation '<absolute final-attestation.json path>' --output '<fresh-gate.json path>'
```

The builder validates all candidate evidence before writing the gate, then calls
the same consumer verification. Pilot startup calls consumer verification again.
Neither command performs review or invents approval.

## Compatibility and remaining limits

- Old gates, old test records, and trainer checkpoints lacking explicit ordered
  provenance/run identity are deliberately rejected. No actual experiment artifacts
  were migrated or changed. Generate fresh admission evidence after source freeze.
- A round-1 predecessor must be `model_final.pt` in its completed training output,
  with settings, metrics, completion and source collection artifacts still available.
  A detached policy file alone is insufficient to prove its training inputs.
- Runtime/profile/physical evidence remains pending until five real smokes and
  independent admission review. This run proves synthetic integration behavior only.
- Attestations are explicit, hash-bound local reviewer records, not cryptographic
  signatures or a reviewer-identity authentication system. Trust and access control
  for reviewer-authored local files remain the coordinator's responsibility.
- STOP remains cooperative: already running numerical work exits at the child's
  existing checkpoint boundary; workers are not forcibly terminated.

## Final file hashes

SHA-256 for changed code/tests after the passing run:

```text
C994F8D1BD10C065E6AD1E608CB2FEC0DF4016A47E255BCFE6388041105E6998 train_round.py
C2E184F02B93DA06475D1AC905F4A888B68BFF80E6E01E7F85019678135AE09F run_pilot.py
A659700DBA89F08948E5817B4866E7256E36A1AE9E87CB9D1E64D6E8B746C738 compare_runs.py
BD5EEACDC82CE8C015926EDCECA7E87E3E3836AB2E3405ABE6DDE338FA4970EE build_preflight.py
5BAF8CC7F833540A9FDD9009CB3EBF0E8F3018283B02FBEBAD96A39054F249E5 run_tests.py
800BCDF84EE20A2E35CBB071ABFE4E92C37D88D76E094488AB548AC50020F9F9 test_multi_contracts.py
B52E3471EA1D8A3DBA0CD3C400E5E4C755453FEC53579968DD03B2AC46507FDF test_multi_pilot.py
AD248A9DEE999A2432CC5C0612F89E14348C28136CC3607DF2A770FB564185C5 test_multi_runner.py
```

Read-only before/after checks matched for all protected files:

```text
E27665133CCB3FAC2B1769DA1C93DF73A60C73572FA4CCEDE24A1296292E2E8E td3.py
18FBD3BC93534E1ED1DCF9218A3C6D114B0D760C0DEE794F217FB392B689A3D0 test_td3.py
6CCB4EF400106E29282797EC5D69B7A8DBEEE5A1310BD7D6F97B7C7E0C7EE2C4 budget_env.py
0402511AE8E965E315B4B9808E23BBDC9C2FD274666A983B2A3303E6B916F9C2 budget_controller.py
6AE552BAD2C092AF9D2AB55B0C81E34558754861217FE427788820689800E9A6 budget_runtime.py
8A13455B1D87A117EEC7F97786F7996786BF49C0D8AAEF6E2A3EF8C1CCA608AB freeze_runtime.py
```

## I2 amendment: recorded smoke runtime

Requested before scoped re-review, 2026-09-29. Ownership was explicitly extended
to `work/sdmpc_rl_multi_20260929/smoke_budget.py` for this evidence-only amendment.

Changed in this amendment:

- `smoke_budget.py`: call `run_budget.runtime_versions()` immediately after
  `boot(...)` and persist that returned object as `runtime_versions` in the smoke
  JSON. No physical execution, solver, or smoke parity logic changed.
- `build_preflight.py`: `validate_evidence` requires each smoke's recorded runtime
  to equal the admitted runtime, then compares the attestation's complete
  `smoke_runtime_versions` map directly with the five recorded smoke objects.
  Builder and consumer share these checks.
- `test_multi_contracts.py`: valid fake smokes now include recorded runtime
  metadata. Six new cases cover missing runtime, wrong runtime (even when the
  reviewer agrees with that wrong runtime), and reviewer/recorded-runtime mismatch,
  each at both builder and consumer boundaries. Tests update file hashes and
  attestation bindings so failures specifically exercise runtime reconciliation.
- This report: appended results and schema clarification; superseded the prior
  external-runtime inference instructions above.

Schema clarification: every smoke JSON now requires this object, populated by
the producer after boot:

```json
{
  "runtime_versions": {
    "python": "<sys.version>",
    "numpy": "<installed numpy version>",
    "scipy": "<installed scipy version>",
    "torch": "<installed torch version>",
    "PyYAML": "<installed PyYAML version>"
  }
}
```

For each scenario, admission requires exact equality among the recorded smoke
object, the admitted runtime, and `attestation.smoke_runtime_versions[scenario]`.
The gate's admitted runtime already matches the current runtime, source-bound
test record, and attestation's top-level runtime. Smoke hashes bind the added
metadata to the reviewed artifacts. Gate/test/attestation format identifiers and
the nine evidence roles are unchanged. Old smoke files without this metadata
cannot satisfy admission; no old artifact was edited or backfilled.

Validation reused the exact final execution-evidence command and environment in
the Focused coverage and execution section above. One numerical Python process
ran only the three `test_multi_*.py` files using existing dependencies and approved
dependency ACL escalation. Output:

```text
205 passed in 49.19s
EXECUTION_IDENTITY_PARITY 205 collected 205 passed
```

Exit code 0. Temporary XML was removed by its temporary-directory context.
No real smoke, traffic, full admission, commit, agent, or package installation was
run. Protected TD3/tests and physical-module SHA-256 values still exactly match
the before/after values recorded above. No other production files were changed.

Post-amendment SHA-256 values supersede the corresponding earlier entries:

```text
E389FAA343082330FAD571E7EF1A4D2EC212B4FAD33B7E877D6A3EDEB0DFB714 smoke_budget.py
064D14CE37696F2565402D9D5956222A28C0BD65EE974A58CED334EBC0C66D24 build_preflight.py
211F93FB19CC62B693B4CCFE4002B2E67A412278F5A57E763625961D7C080ED3 test_multi_contracts.py
```

Source is ready for scoped re-review. Full source-bound test evidence, five real
smokes and the independent final admission attestation remain future steps.
