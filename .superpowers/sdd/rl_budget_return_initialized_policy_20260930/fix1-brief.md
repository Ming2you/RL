# Task1 fix round1

Accepted review R1/P2 from task-1-review.md: STOP after linking final candidate
during a resumed done checkpoint creates a new process-bearing checkpoint and
latest pointer that no longer matches the immutable candidate. Subsequent resume
fails. Fix this scoped publication/STOP recovery defect and add exact regression
with changed process/session identity. Preserve immutable artifacts and no-retrain
completed phases; no scientific/hyperparameter/data/source-history changes.

Own only candidate run.py, focused tests and task-1-report.md appendedfixsection.
Additional runtime helper edit only if necessary and explain it. Original7files
snapshotted in before-fix1. Reproduce before fixing, run focused and final suite
once, retain evidence/finalsourcehashes. No actual optimization/simulation/commit.
Parent will generate scoped diff and request R1-only re-review; do not selfapprove.
