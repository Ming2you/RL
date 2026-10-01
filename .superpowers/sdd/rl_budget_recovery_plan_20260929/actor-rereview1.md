SPEC: PASS
QUALITY: PASS

P1 resolved. [actor_repair.py:29](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/actor_repair.py:29)
freezes the actor before enabling only the output head. Clearing gradients to
None prevents Adam from advancing frozen parameters, including parameters with
existing optimizer history; the continuation arm retains its head history.
End-of-fit checks enforce exact hidden-weight and critic equality. The new
[multi-update test:66](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/test_actor_repair.py:66)
covers all three arms with populated Adam history and verifies that the head
changes while hidden weights remain fixed. Exports declare head-only training,
and the evaluator requires that declaration. The completed actor_fit_v2 source
hashes and selected continue export hash match the current files; the v1 selected
export retains its original hash.

P2 resolved. [evaluate.py:96](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/evaluate.py:96)
returns on an existing STOP before boot, model loading, reset, restore, or
checkpoint replacement. The actor fit also checks STOP before loading its base.
The [fresh/resume tests:22](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_recovery_20260929/test_evaluate.py:22)
reject numerical initialization and verify unchanged checkpoint bytes on resume.

No new load-bearing findings in the fixes. Current files match actor-fix1.diff.
Validation uses the supplied 18 PASS / 2.60s result; no tests or numerical runs
were launched during review. This is a scoped code PASS: smoke_v2 remains
pending, and smoke_v1 parity does not validate the v2 export/evaluator pair.
