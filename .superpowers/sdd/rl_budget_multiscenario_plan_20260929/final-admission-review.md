# Final admission review

Date: 2026-09-29. Reviewer: Codex independent integration reviewer.

MULTI_REVIEW: PASS

Review kind: final_admission. Status: final. Decision: approved.

Approve admission to the specified bounded, balanced five-scenario shared-TD3
pilot against the exact source, runtime, tests and smoke evidence bound by the
accompanying final-admission-attestation.json. No admission-blocking finding
remains in the reviewed evidence.

This is the final evidence review following the scoped I1-I6 re-review, not a
restart of the code review. The unchanged source-bound test record connects this
decision to [integration-rereview-round1.md](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/.superpowers/sdd/rl_budget_multiscenario_plan_20260929/integration-rereview-round1.md),
which records all six integration findings ADDRESSED and CODE_REVIEW/SPEC/QUALITY
PASS. Task 1's independent strict-restore approval remains unchanged.

## Evidence reviewed

All seven execution artifacts are in:

`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/results/sdmpc_rl_multi_20260929/admission_v1`

- tests.xml and test_source_pins.json.
- smoke_sweet_155_w.json.
- smoke_sweet_170_w.json.
- smoke_sweet_170_incident_w.json.
- smoke_sweet_170_skew15_w.json.
- smoke_sweet_190_w.json.

The coordinator reported the full test command exiting 0 with 486 passing tests
in 52.69 seconds and all five actual smoke commands exiting 0 with the
SMOKE_RESUME_REPLAY_PASS message and observation dimension 2367. This reviewer
independently inspected the stored artifacts, source bindings and numerical
audits. No test suite, simulator, collector, trainer or smoke was rerun.

## Tests and source identity

The XML contains exactly 486 unique test identities with no failure, error or
skipped testcase. Every identity matches the record's collected and passed lists;
the record has integer exit_code 0. Every current test file is represented:

| Test file | Passed identities |
| --- | ---: |
| test_td3.py | 211 |
| test_carry_controller.py | 54 |
| test_carry_environment.py | 16 |
| test_multi_contracts.py | 123 |
| test_multi_pilot.py | 26 |
| test_multi_runner.py | 56 |
| Total | 486 |

Recomputed current production pins, all six test hashes and the XML hash match
the test record exactly. Both test artifacts retain the hashes verified during
the scoped re-review. All five smokes carry those same current production pins.
Frozen snapshot verification also succeeds for all 151 files, totaling 3,975,269
bytes. Manifest SHA-256:

```text
07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005
```

Current runtime metadata, the test record and each actual smoke's own recorded
post-boot runtime object match exactly:

```text
python: 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]
numpy: 2.3.5
scipy: 1.16.3
torch: 2.14.0+cpu
PyYAML: 6.0.3
```

Dependency metadata was read with approved access because the sandbox otherwise
hides some installed package metadata. The verification remained read-only.

## Five actual smokes

Each file has its correct distinct scenario identity and canonical profile digest,
independently recomputed from that scenario's frozen forecast.json. Each reports
passed=true, serialized_resume=true, final step 7, and the explicitly bounded
two-actual-intervals scope. All five observation schemas are identical and contain
2,367 unique feature names. Common schema digest:

```text
f350b9d825e45d4eaf9d977363d3813a3ed20d545249a7c41d2589341c3ca78d
```

| Scenario | Canonical profile SHA-256 | Resume / physical / budget |
| --- | --- | --- |
| sweet_155_w | 1eaf3d70d7cb02b640f700a6f89ce3d50cbf7ca77bfac3889a02c7f3f7ebc8e9 | PASS |
| sweet_170_w | 79d46c15323cc102f3ddccc4f1b10db1bf788dca00e31b8f2aebb57fa7b04e06 | PASS |
| sweet_170_incident_w | 74778a836d4bbe736c8c1f533164c612b0b5ef916313b1a9617866e964a984cb | PASS |
| sweet_170_skew15_w | e8aabfa9ec04ecc78692b2c8234faf98dc57e2f7f609876d0b84a122c7d7db8a | PASS |
| sweet_190_w | 5ce6e183482ec830a45aa504bdbfc5b9632b84bd98cfd5a543b1de45f00437e1 | PASS |

For every scenario, both stored audits were checked:

- Steps 5 and 6, control steps 0 and 1, end times 1080 and 1260 seconds, and no
  termination or truncation.
- Requested actions [0.25, -0.25] then [0, 0].
- Exactly one initial PFO with reference_source=pfo_initial, followed by zero PFO
  calls with reference_source=previous; no recovery and no enabled H3 guard.
- Physical-control and budget validity true for both executed controls, one lower
  candidate, and selection_source=lower_solution.
- Finite stored budget certificates; achieved minus budget matches the reported
  residual, band/scaled excesses are zero, and executed budgets/achieved flows
  agree with their corresponding execution-check values.
- Finite nonnegative interval/total/area TTT; area totals reconcile, the second
  interval accounts for the cumulative TTT change, and the recorded first reward
  equals minus interval TTT divided by 100. Stored plant times match audit times.
- Exactly one replay entry in that smoke's scenario and zero in the other four.

The pinned smoke producer emits the success artifact only after serialized reset
and carried-state restore comparisons: observations, rewards, termination and
controls agree with the uninterrupted path, with dual/anchor checks at the
specified boundaries. The inspected artifacts and reported successful executions
support those assertions. The reviewer did not independently rerun serialization
or the physical solver.

The early interval TTT values match across four scenarios because their frozen
forecasts share this initial segment; the first difference from sweet_155_w for
170, 170-incident and 190 is forecast row 21. The skew profile differs from row 0.
Distinct full-profile hashes and scenario-specific files were verified. These
short smokes do not establish incident-window behavior or full-horizon performance.

## Artifact bindings

The attestation binds exactly eight evidence roles: the finalized bytes of this
human report plus the seven execution artifacts below. It also contains the exact
current production pins, complete current test hash map, current runtime object,
and the five runtime objects copied from the actual reviewed smoke files.

```text
test_xml
80872e1ac9b6a6090693fc25ef392680b2ad879f436a1024703dbdc67c2a0402
test_record
0392ebbd56042f4bb27c8d703b52ebe1173f3e0f0ec365be1fd52a183d17440b
smoke:sweet_155_w
b68748b64ed4f2726c981dca4977e99d0aacb129dd19c1a3640265faa47eda41
smoke:sweet_170_w
5e25a6e99f3fc9cf9eba6e321e3ccdd42d8dc48dbdca54baed94d2124753a49e
smoke:sweet_170_incident_w
3c3d9a1e65103f3ccf57af15b004d0af1b118d25f6fdf8987a6d982aba008c0e
smoke:sweet_170_skew15_w
4b44ae3dd2c4d0abb72dd4c96e0ec8c07d761d77c1208557bbd6cc9de1477814
smoke:sweet_190_w
739cc4aaa2eff4fff8118506b305c1e82d00e6f9e1e28022624961cedd5a0692
```

## Decision scope

Admission is supported for the existing finite plan: two balanced collection and
central-training rounds, followed by canonical carry-center and final shared-model
evaluation across the five specified scenarios, under the existing resource,
source/resume and STOP contracts. This approval is not a claim of completed full
runs, improved traffic performance, calibrated Q values, or generalization.
Those conclusions require the later full-pilot evidence and per-scenario report.

No production/test file or existing execution artifact was modified. Only this
final report and its exact-schema attestation are produced. No pilot or gate is
created by this review. The report is finalized before attestation hashing and
will not be edited afterward in this review.
