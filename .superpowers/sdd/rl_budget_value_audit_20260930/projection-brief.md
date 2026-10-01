# Projection/replay audit review requirements

Review new projection_audit.py and test_projection_audit.py only. This read-only
diagnostic authenticates the preserved balanced pilot collections and final
model before comparing observations, clock, true terminals, reward accounting,
previous requested/executed budget memory, physical action projection and exact
final replay membership. It may boot the frozen runtime to derive capacity and
verify configuration, but must never simulate, train, export policy weights or
modify old artifacts. Replay-only, no canonical evaluation observations.

Same-state nominal actions that map to exactly equal clipped budget requests
are grouped by the actual residual_budget transform, including post-float32
equality checks. Log both critics' predictions and spread, do not label this an
optimality or full physical follower-response experiment. Every scenario and
both rounds must be included. Check all source/model/runtime identities before
and after; strict STOP/overwrite/exclusive-run protection and inspectable outputs.

13 focused tests PASS in1.35s, command:
`.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_value_audit_20260930/test_projection_audit.py']))"`.
The initial direct python -m pytest failed because pytest is in .deps-budget;
no dependency was installed or changed. No production diagnostic has run yet.

The diff is projection-review.diff in this same directory. Report SPEC and
QUALITY verdicts, only actionable scoped findings with references, to
projection-review.md. Do not rerun the same tests or change implementation. Any
source dependency needed for reasoning may be read. Base HEAD c40eb9de19ee780d08a7e3e91b40c6cdd4807a92;
dirty existing changes are unrelated and must be preserved. No commits.
