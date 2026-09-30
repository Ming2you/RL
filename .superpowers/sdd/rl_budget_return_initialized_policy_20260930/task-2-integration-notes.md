# Parent integration notes, not dispatch authorization

Task2 stays conditional on Task1 actual output/readout. Do not implement in the
frozen Task1 source folder after training: use a separate versioned wave folder.

Existing BudgetEnv has2367observation entries. reset/restore/step
and its physical contract remain unchanged. Frozen original
work/sdmpc_rl_multi_20260929/run_budget.py offers exclusive_run (Windows kernel
byte lock), checkpoint_save, file_hash, observation_schema; budget_runtime has
boot/read/plain/digest. Exact config serialization must use frozen to_plain_dict.

Important import boundary: new learner.py imports generic runtime.py; physical
BudgetEnv.boot also imports a different frozen module named runtime. Do NOT import
the learner package into an environment process under that generic module name.
The candidate is a dict checkpoint with learner/models/actor containing
net.0/2/4 weights+bias and bounds buffer. A tiny separately source-pinned inference
adapter can instantiate the exact2367-64-64-2ReLU+bounds*tanh architecture from
these weights, no optimizer/data/learner import. Test bit-identical outputs to
the admitted Learner.Actor in isolated subprocesses on fixed test observations.
Do not monkeypatch frozen physical modules or rename them globally at runtime.

The latest frozen local collector handles +Inf diagnostics correctly, but its
LocalBudgetPolicy replay, settings seeds and cohort canonical paths are specific
to NUF-retention and cannot simply be relabelled as a learned-policy run.
Reuse physics/accounting functions that actually fit, not the old load_completed
or canonical_worker identities. New verifier must validate the frozen actor's
act(observation) for each recorded action and declare critic continuation carry.

Useful frozen validate.py helpers: validate_observation, physical_config,
validate_boundary, and validate_rows. validate_rows requires policy_q=None;
for an actor-only physical rollout keep policy_q=None and perform Q diagnostics
offline later, so no misleading current-policy Q or omitted inferencecost.
It validates the real action/anchor/request chain, physical state inventory,
reward/accounting/terminal sequence, one solve, physical guard, PFO/timing.
Check its inherited sequence_validator before deciding reuse. summarize is NOT
generic: it reads exploration_audit/base; write a focused learnedpolicy summary.

diagnostics.py's tagger is pinned and only admits known rejected-candidate+Inf
stationarity paths, raw checkpoint retained. A new wrapper can reuse its pure
tagging function, with a new receipt binding checkpoint/settings/experience/trace.
Never swallow other nonfinites or overwrite the completed parent's artifacts.

Prefer one compact resumable worker plus readout and a bounded launch strategy;
avoid cloning a large new framework. If using parent-driven hidden dispatch,
each actual worker must own its kernel slot lock and log pid+creation time,
source/model/contract/settings; check global STOP and its slot STOP each interval.
Parent must record all launchers/actual workers and wait for them to drain before
any new dispatch. Windows venv launcher and actual worker PIDs differ. A dead
launcher is not sufficient proof of worker exit. Account five numerical workers,
not ten redirector/interpreter process entries. Partial completion never restarts
from reset and completed slots must be skipped/authenticated.
