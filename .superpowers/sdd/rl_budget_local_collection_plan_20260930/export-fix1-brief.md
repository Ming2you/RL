# Export companion fix round 1

Read export-recovery-review.md F1-F3 first; those are the full open findings.
Fix only these in the new companion and its focused tests. No v1/v2/source/result
edits, real exports/resumes, physics, model runs, migration, recollection,
install or commit. Before-fix files are in before-export-fix1. Use apply_patch.

F1 is accepted: preserve actual coordinator creation identity even when an
all-completed finalization launches zero numerical workers/reservations. Bind
this independently and durably to the supervisor attempt, including Windows
venv redirectors. A tiny process-identity bootstrap that then delegates to the
unchanged original runner is allowed if needed; its code/command/identity must
be explicit in provenance and hashes. Do not waive missing/unknown identity or
infer death merely from a completion file. Do not rewrite the immutable runner.
Test the actual new launched/inspect integration with new coordinator identity
and no new reservations, plus invalid/missing bindings. Preserve STOP and locks.

F2 is accepted: all completed repair paths must bind receipt slot/token/run ID
to the retained75step record and authenticated settings, allowing legitimate
later bookkeeping but not replacing identity. Add changed-token/runID cases.

F3 is accepted: metadata-linked repaired completions require receipt existence,
hash/input/retained-identity verification before ANY continuation or completed
return, including earlier non-primary repaired slots. Ordinary original
completions remain admitted by original loader. Test missing/changed receipts
for non-primary repaired slots and whole-cohort completion through real inspect.

Reproduce focused failures first; then run covering new tests and the100+new
companion suite, not frozen suites. Append exact evidence/commands/limits to
export-recovery-report.md under fix round1. No broader infrastructure work.
Parent will do scoped re-review before actual export or numerical resume.
