# Integration re-review: fix round 1

Reviewed: 2026-09-29. Scope: I1-I6 in integration-review.md and regressions
introduced by their fixes, including the final recorded-smoke-runtime amendment.

CODE_REVIEW: PASS / scoped fix-round review

SPEC: PASS / I1-I6 addressed

QUALITY: PASS / no actionable remaining finding or new regression found in scope

MULTI_REVIEW: PENDING / final admission not issued

## Finding dispositions

### I1: ADDRESSED

[build_preflight.py:58](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/build_preflight.py:58)
requires the exact nine named evidence roles, absolute paths, valid SHA-256
references, and distinct files. Both the builder and
[verify_gate:125](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/build_preflight.py:125)
call the same evidence validator. Consumption now rechecks the current source,
runtime, complete current test-file hash map, XML identity/hash/pass evidence,
five role-specific canonical smoke results, and review attestation. Empty,
partial, arbitrary and legacy cached gates no longer bypass these requirements.

The test record's collected/passed identities must match the unique XML identities,
with coverage of every current test file and all six explicitly required suites.
[run_tests.py:11](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_tests.py:11)
records absolute identities and only marks a case passed after setup, call, and
teardown pass. Current-test edits invalidate previously generated gates.
Regression coverage is at test_multi_contracts.py:421 and :463; the actual
execution-hook output and fresh full-suite evidence are reconciled below.

### I2: ADDRESSED

[build_preflight.py:107](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/build_preflight.py:107)
requires a separate final-admission attestation with exact fields, final status,
approved decision, reviewer identity, current source/runtime/test identities, and
all eight other evidence hashes, including the human report. The gate hashes the
attestation itself, avoiding a circular dependency. A success phrase in prose
cannot authorize admission; code-review, superseded, rejected, conflicting and
stale attestations are rejected. Production code does not generate approval.

The final amendment is present in both producer and consumer:
[smoke_budget.py:18](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/smoke_budget.py:18)
reads runtime_versions immediately after boot and persists that object at line 66.
[build_preflight.py:99](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/build_preflight.py:99)
requires each recorded smoke runtime to equal the admitted runtime and compares
the attestation's five-scenario runtime map with those recorded objects. Missing
runtime metadata is rejected rather than supplied by the reviewer. Tests at
test_multi_contracts.py:511 and :533 cover stale approval and missing/wrong/mismatched
smoke runtime, including both builder and cached-gate consumption boundaries.

### I3: ADDRESSED

[train_round.py:89](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:89)
reconciles seed 6300, exact round-specific replay counts, update totals, cumulative
eight-per-update sampling totals, every stored replay column, and each metric's
absolute update/actor phase and finite losses. Replay comparison checks dtype,
shape and tensor equality against reconstructed admitted inputs, including true
terminal flags, requested actions, rewards and observations.

[Resume:228](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:228)
also checks training_run_id and full ordered provenance, then reconciles the
restored learner before checkpointing or another gradient update. It replaces the
temporary reconstructed learner; it does not append the collections again after
restore. The original three failing resume variants remain covered at
test_multi_runner.py:401; equal-size replay/provenance/metric corruptions are
covered at :474. The round-1 continuation test at :565 checks the delayed-actor
boundary at absolute update 376 and exact final learner/metric equality.

### I4: ADDRESSED

[train_round.py:133](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/train_round.py:133)
implements the completed-output validator now called by
[run_pilot.py:33](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_pilot.py:33).
It reconstructs collections and predecessor inputs, requires the model's exact
training run/round/provenance, restores the model and applies the same replay,
metric and sampling reconciliation as resume. Completion metadata is compared
with that validated learner.

Round 1 additionally validates the completed round-0 output and its actual
collection artifacts before using its model. The validation recursion ends at
round 0; it performs no training updates. Full ordered provenance comparison
includes scheduled seeds, profile hashes and experience hashes. The original
provenance regression is retained at test_multi_pilot.py:254; replay/provenance/run
identity/metric mutations with refreshed file hashes are covered at
test_multi_runner.py:530. The new requirement to retain the completed predecessor
directory is explicit and compatible with the pilot's existing paths.

### I5: ADDRESSED

[run_pilot.py:82](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/run_pilot.py:82)
checks parent, stage and child STOP scopes for all relevant plan jobs. It runs at
startup, before each new launch, and before/after every poll. Detection propagates
STOP to the root while already-started children drain cooperatively. Exclusive
STOP-file creation at line 76 preserves a user STOP that already exists or
appears concurrently. Job stages, child assignments and five-worker bounds are
unchanged; the parent sets numerical thread variables before importing TD3.

The original stage/child-during-launch regressions remain at
test_multi_pilot.py:184. Added polling and user-STOP preservation checks are at
:296; parent import-time thread settings are checked at :318. No new launch or
draining regression was found by inspection.

### I6: ADDRESSED

[compare_runs.py:146](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/compare_runs.py:146)
now deserializes each completed collection's experience and calls verify_experience
before returning acceptance. Pilot startup and stage prevalidation consequently
check the payload before skipping the collector or launching missing siblings.
The trainer retains its independent admission check. Evaluation modes do not enter
this experience-loading branch. The original 74-transition completed-payload
regression remains at test_multi_pilot.py:230. Function-local imports avoid a
module-initialization cycle, and the shared verifier does not recurse into the
completed-run loader.

## Regression and evidence assessment

Read integration-fix-report.md through its final runtime amendment, the fix brief,
both conceptual directory-version diffs, live changed modules, and the relevant
retained and added test cases. Reconstructed the seven reviewed production files
from each diff against the retained carry baseline and compared the resulting
fix-round versions with live files: all seven matched after line-ending/final
newline normalization. The actual round-1 changes are confined to validation,
evidence recording and STOP handling; run_budget.py is unchanged.

Live SHA-256 values match the fix report, including the amendment's superseding
hashes. TD3, test_td3.py, all four physical/freeze modules, and the analysis helper
match their previously reviewed hashes. No physical or TD3 update math changed.
The report's **205 focused tests passed in 49.19 seconds**, including all seven
original failing variants, is accepted as implementer execution evidence.

During re-review the coordinator supplied a fresh source-bound full run:

```text
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_multi_20260929/run_tests.py --output results/sdmpc_rl_multi_20260929/admission_v1
486 passed in 52.69 seconds; exit 0 (coordinator-reported execution)
```

Independently read and reconciled its tests.xml and test_source_pins.json without
executing tests: all 486 XML cases have unique identities and no failure/error/skip;
collected IDs, passed IDs and XML IDs match exactly. All six current test-file
hashes, the complete production/frozen source pins, XML hash, and runtime versions
match. Per-file execution counts:

| File | Passed identities |
| --- | ---: |
| test_td3.py | 211 |
| test_carry_controller.py | 54 |
| test_carry_environment.py | 16 |
| test_multi_contracts.py | 123 |
| test_multi_pilot.py | 26 |
| test_multi_runner.py | 56 |
| Total | 486 |

The first sandboxed metadata comparison could not see NumPy/SciPy/PyYAML versions
and returned null values. A read-only metadata/evidence reconciliation with
approved dependency access matched the recorded runtime: Python 3.12.14, NumPy
2.3.5, SciPy 1.16.3, Torch 2.14.0+cpu, PyYAML 6.0.3. No suite or numerical probe
was rerun. The recorded runtime objects include the full Python build string.

Full-run evidence SHA-256:

```text
tests.xml
80872e1ac9b6a6090693fc25ef392680b2ad879f436a1024703dbdc67c2a0402
test_source_pins.json
0392ebbd56042f4bb27c8d703b52ebe1173f3e0f0ec365be1fd52a183d17440b
```

## Remaining admission work

This review closes I1-I6 at the code/spec/quality level. It does not approve a
pilot gate and does not assert physical smoke success or full-episode performance.
Five actual canonical scenario smokes with matching settled source, recorded
runtime, schema, physical/carry audits and serialized restore evidence remain
pending. After inspecting those results together with the source-bound tests,
an independent final admission report and hash-bound JSON attestation must be
issued. No final approval JSON or admission gate was created by this re-review.

The new v2 gate/test records and explicit trainer checkpoint provenance reject
legacy evidence intentionally. Preserve completed predecessor and collection
artifacts for round-1 reconciliation. STOP continues to take effect cooperatively
at worker checkpoint boundaries. These documented constraints are not new defects.

## Reviewed fix identities

File names refer to work/sdmpc_rl_multi_20260929; full source/test identities are
also bound by the verified test_source_pins.json above.

```text
C994F8D1BD10C065E6AD1E608CB2FEC0DF4016A47E255BCFE6388041105E6998 train_round.py
C2E184F02B93DA06475D1AC905F4A888B68BFF80E6E01E7F85019678135AE09F run_pilot.py
A659700DBA89F08948E5817B4866E7256E36A1AE9E87CB9D1E64D6E8B746C738 compare_runs.py
064D14CE37696F2565402D9D5956222A28C0BD65EE974A58CED334EBC0C66D24 build_preflight.py
5BAF8CC7F833540A9FDD9009CB3EBF0E8F3018283B02FBEBAD96A39054F249E5 run_tests.py
E389FAA343082330FAD571E7EF1A4D2EC212B4FAD33B7E877D6A3EDEB0DFB714 smoke_budget.py
211F93FB19CC62B693B4CCFE4002B2E67A412278F5A57E763625961D7C080ED3 test_multi_contracts.py
B52E3471EA1D8A3DBA0CD3C400E5E4C755453FEC53579968DD03B2AC46507FDF test_multi_pilot.py
AD248A9DEE999A2432CC5C0612F89E14348C28136CC3607DF2A770FB564185C5 test_multi_runner.py
```

Only this re-review report was written. Source, tests, previous review records,
and experiment artifacts were not edited. No traffic, test suite, agent, commit,
or production job was launched by this reviewer.
