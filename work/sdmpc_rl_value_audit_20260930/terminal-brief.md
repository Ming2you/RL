# Terminal-fit diagnostic task

## Scope and ownership

Implement `terminal_fit.py` and `test_terminal_fit.py` in this directory only.
Write full report to `terminal-report.md`. No commits, source changes elsewhere,
simulation, policy exports or actual production diagnostic execution. Run focused
synthetic tests. Parent will run the admitted diagnostic. Existing runtime is
`.venv-torch/Scripts/python.exe`, local dependencies `.deps-budget` (ACL may need
tool escalation). One Torch/BLAS thread. No installations.

The parent independently implements replay/observation and budget-projection
auditing. Do not duplicate that work. This task tests whether the network can fit
exact terminal targets without bootstrapping, not whether it controls traffic.

## Input and experiment

Import read-only `work/sdmpc_rl_recovery_20260929/common.py` with its existing
`load_base`, `BASE`, `BASE_HASH`, `REPO`, `TD3`, `SCENARIOS`, `torch`, `save`,
`file_hash`, `runtime_versions`, `exclusive_run` helpers. Inspect these first.
The preserved final learner has 150 replay rows per scenario and true terminals
at indices74 and149, one per collection round. Do not load canonical evaluation
data. Check source/model hashes before and after; report input immutability.

Use two folds: train on each scenario's round0 terminal, diagnostic-holdout on
round1; then reverse. Five training samples and five holdout samples per fold,
exactly one per scenario. For each fold compare:

1. copied original trained critics, fresh Adam;
2. freshly initialized critics from same-size TD3 architecture, fresh Adam,
   seed8100, same initialization in both folds.

Both arms use original learning_rate=0.0003, fixed1000 supervised updates,
fullbatch5 with equal20% scenario contribution, loss=sum of twin-critic MSE
against the stored immediate reward. No bootstrap/noise/target/actor updates,
no reward normalization/clipping or data augmentation. Record metrics at0,50,
250,1000. Before diagnostic, report original critics' errors on all10terminals.
Holdout is never used for updates, stopping or arm selection. Critic copies only,
do not mutate original learner/optimizers/replay/RNG. No learned weights export.

Each metric should include per-scenario predictions/target/errors for bothQs and
minQ, aggregate MSE/MAE/max error, training draws by scenario and exact row indices.
Criteria are descriptive: final per-critic train max abs error <=0.01 reward
units. Holdout errors are diagnostic only, not a policy admission gate.
Explicitly note that the original critic has previously seen both folds during
joint TD3 training; its holdout excludes only these extra supervised updates.
Fresh-critic holdout is untrained in this diagnostic, but there is only one
sample per scenario and no out-of-sample generalization claim.

CLI `--output <new directory>`. Reject occupied outputs; exclusive lock; STOP in
output, goal root `results/sdmpc_rl_balanced_goal_20260930`, and repository root
before expensive loading and during fitting. Honor STOP without completion
claims; do not remove STOP. Record command, pid, start/end, input/source/runtime
hashes and settings. Write settings/status/metrics/completion atomically using
existing helper where appropriate, JSON only, no pickle diagnostic model.

## Verification

Focused tests for terminal-only extraction, true-terminal positions and finite
shape validation, training/holdout separation, exact scenario balance, exact
immediate reward targets (not reward-to-go/bootstrap), all critic layers eligible
for training, original state/RNG unchanged, deterministic fresh initialization
and folds, honest warm-start holdout labeling, no model export, early/loop STOP,
overwrite and finite result guards. Prefer small synthetic networks and a small
injected update budget in tests; production CLI has fixed1000. Do not load or fit
production data in tests. Report exact commands and counts in terminal-report.md.

Return only completion status, changed paths, test summary and concerns.
