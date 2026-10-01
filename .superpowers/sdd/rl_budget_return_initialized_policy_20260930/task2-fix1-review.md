# Task 2 Fix 1: R1-Only Independent Re-Review

SPEC verdict: **PASS** against the clarified Task 2 brief.

QUALITY verdict: **PASS** for R1 and direct fix regressions.

Addressed: **R1**. Open findings: **none within this scope**.

## R1 Disposition

1. **Contract ambiguity resolved.** The clarification at the top of `task-2-brief.md` and the plan's Task 2 clarification explicitly permit inherited nonterminal reference reconstruction through unchanged `BudgetEnv.restore`, while prohibiting additional per-decision RL previews and repeated plant intervals, candidate solves or PFO. This is the parent's conservative default after an unanswered optional preference question, not claimed user approval. The appended report's earlier pending/unresolved language records the pre-clarification state and does not leave an approval blocker under the current instructions.

2. **Whole-restore accounting addressed.** [restore_accounting.py:27](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_wave_20260930/restore_accounting.py:27) brackets exactly one unchanged `env.restore` call with wall/process-CPU measurements. Labels explicitly include validation, copies and observer work; they do not claim isolated reference-only latency. Reconstruction counts are declared from the pinned branch, not presented as measured preview counts. Saved reference/observation timing is checked unchanged. The wider worker-session duration already includes restore, and neither session totals nor decision timings receive an additional charge. The wrapper adds no reset, plant-step, lower-solve, PFO or per-decision preview call.

3. **Frozen-method coverage addressed.** [test_restore.py:139](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_wave_20260930/test_restore.py:139) executes the hash-pinned real restore/controller method bodies and real `Observation` with synthetic state and lower spies. Cases cover initial, ordinary, recovery and late nonterminal checkpoints plus terminal restore. Assertions establish one reconstruction branch for nonterminal restore, none at terminal, exact observation/boundary preservation and unchanged saved timings; reset/step/PFO/solve sentinels prohibit physical execution. Deterministic clocks verify a 3-second restore remains a component of a 100-second session, not a 103-second total. This closes the original FakeEnv-only coverage gap without claiming a real physics test.

4. **Provenance and interruption handling addressed.** [restore_accounting.py:56](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_wave_20260930/restore_accounting.py:56) binds restore records to session/settings/source/checkpoint identity; enclosing session hashes and timing/summary/completion reconciliation retain that provenance. Missing restore-end publication preserves UNKNOWN duration/completion; a raised call can retain measured duration while reconstruction completion remains UNKNOWN. Missing enclosing session ends remain UNKNOWN independently. Tests cover these cases, inherited-timing mutation and record tampering. Completed readout validates JSON and checkpoint hashes without deserializing physical checkpoints. Direct changes preserve the existing exact-observation check and worker control flow.

## Evidence And Limits

Reviewed `task2-fix1.diff`, the appended report, clarified brief/plan, implementation and retained test evidence. All nine changed/new Python files reconstruct exactly from the supplied fix diff; all eleven current source hashes match `d45d6eca/evidence.json`. Both frozen method-source files still match their pins. Focused/full evidence-file hashes match the report.

Accepted retained evidence: focused `f2bec8e8` has 25 passing tests; full `d45d6eca` has 95 passing tests, zero failures/errors/skips, and the 351-file before/after preservation manifest. No successful suite, parity check or new reproduction was run during this re-review.

Actual physical serialization/resume, traffic outcomes and real restore latency remain unverified integration limits, not reopened R1 findings. The actual wave root was absent when checked. This review makes no dispatch decision or claim of user approval. Only this review file was written; no fixes or actual run were performed.
