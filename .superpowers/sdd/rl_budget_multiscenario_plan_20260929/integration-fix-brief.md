# Integrated fix round 1

Fix all six findings I1-I6 verbatim in integration-review.md beside this brief.
The original test findings are in task-2-test-report.md. Own the new multi-version
run_budget.py, train_round.py, run_pilot.py, compare_runs.py, build_preflight.py,
run_tests.py only as needed, and your three test_multi_*.py files. Do not edit
td3.py/test_td3.py (already independently approved), physical modules, old code,
analysis helper or actual experiment data. No traffic, commits or agents.

I3/I4 must compare actual float32 replay columns/terminal flags/actions/rewards
against admitted predecessor plus current collections, not only shapes/counts.
Use a narrowly shared helper if it removes duplicate reconciliation. Check exact
seed6300, update/sample totals, per-metric update phase/sample counts, full ordered
training provenance and training_run_id at resume and completed admission.
Keep legitimate interrupted training exactly reproducible and avoid duplicate
collection/data insertion. Keep learner's actual math unchanged.

I1/I2: implement one versioned typed admission schema and reusable evidence
validation called by both builder and consumer. Require the exact expected roles,
not arbitrary/empty hash maps. Recheck current test files, XML, source and allfive
actual smoke identities on consume. Final approval must be a separate
machine-checkable reviewer attestation binding source/test/evidence identities,
not a searched PASS substring. A JSON attestation alongside the final human
review is appropriate; bind its report hash too if used. The code must NOT create
its own PASS attestation. The independent reviewer will issue it only after fresh
tests and real smokes. Define/document the exact schema/creation command in your
report so the coordinator/reviewer can supply valid evidence. Update synthetic
tests to create complete valid fake evidence rather than bypass validation.

I5: detect any relevant parent/stage/child STOP before each new launch and at every
poll, propagate to root and drain existing workers cooperatively. Do not overwrite
a preexisting user STOP. I6: completed collectors must have their experience payload
validated before skip/sibling launch, reusing verify_experience and preserving the
trainer's independent check.

Add focused negative tests for I1/I2 and equal-sized replay/provenance corruption,
metrics/sample/phase drift and mid-drain STOP where existing140tests do not cover.
Keep all existing meaningful tests; do not weaken assertions or skip failures.
Also set the orchestration parent's numerical thread environment to1 before
importing Torch, consistent with the declared compute budget, without changing
child assignments or worker count.

Run your three synthetic integration test files in one numerical process, using
the existing .venv-torch and .deps-budget with escalation if ACL requires it.
Do not run source-bound full admission or real smoke yet. Append a detailed
integration-fix-report.md here with exact code changes, test commands/results,
attestation schema and any remaining concerns. Return concise status and paths.
