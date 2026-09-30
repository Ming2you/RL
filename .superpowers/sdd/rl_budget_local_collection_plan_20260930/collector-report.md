# Collector implementation report

Status: implementation ready for independent review, NOT production admission.
Owned new files: local_runtime.py, collect.py, validate.py, run_wave.py,
test_collection.py under work/sdmpc_rl_local_20260930. Old sources untouched.

22 synthetic tests PASS7.86s, exit0 using:
`.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','work/sdmpc_rl_local_20260930/test_collection.py']))"`

FakeEnv replaces the physical simulator. Both carry/local paths run75synthetic
steps, including pausedprefix/resume versus uninterrupted comparison, deterministic
policy audits, completed loaders and a complete synthetic10-run paired readout.
Other tests cover changed settings/boundary, STOP/aborts, source drift,9contract
mutations, physical control diversity excluding budgets/metadata, changed hashes,
job mapping, orphaned Windows lock, worker-failure draining and completed skips.
The admitted complete-sequence validator is stubbed in these wrapper tests;
its own existing48tests cover its data logic. No physical source was executed.

Separate read-only real preflight executed local_runtime.identity(): PASS,
correct authenticated gate,7pinnednew/imported sources and expectedruntime
Python3.12.14/NumPy2.3.5/SciPy1.16.3/Torch2.14.0+cpu/PyYAML6.0.3.
No env boot/reset/step, no model load/inference and no collection was run.

Design: new source identity wraps existing frozen runtime instead of editing it.
Metadata uses distinct training-only format and behavior IDs; old validators
not weakened. Full-sequence validator reused with trusted training records only.
Checkpointed episodes restore the exact RNG/policy reconstructed from trace.
Five one-thread child processes, masks reused from prior admitted5-scenariopilot.
Per-attempt abort drains siblings without poisoning persistent user STOP files.
Prefix limit is an operational checkpoint boundary, excluded from policy settings
so a smoke prefix continues the same run without changing its control policy.

Timing is explicitly aggregate worker sessions until completion assembly;
includes boot/reset/restore/steps/checkpoints, not offline analysis/final file
publication. Physical decision phase sums retain the existing environment's
scope; behavior commit/logging is reflected in aggregate wall time. No latency
or speedup claim from these synthetic tests. Failed pre-initial-checkpoint runs
are not silently reset; they require diagnosis. No production workers active.

Remaining: independent implementation review, combined tests, first actual carry
prefix smoke, resume into the prescribed carry then local waves, final paired
reconciliation and explicit limits. The overall RL improvement goal is unmet.
