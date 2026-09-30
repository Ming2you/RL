# Bounded TD3 Learner Report

Date: 2026-09-29

Status: implemented and verified with synthetic unit tests. Final result:
**63 passed in 4.09s**, exit code 0.

## Changed Paths

Only these files were created, relative to the RL repository:

- `work/sdmpc_rl_budget_20260929/td3.py`
- `work/sdmpc_rl_budget_20260929/test_td3.py`
- `work/sdmpc_rl_budget_20260929/td3_report.md`

No runtime, environment, wrapper, snapshot, runner, or legacy files were edited.
No commit was created. No traffic simulations or training episodes were launched.
The action-influence gate remains a prerequisite for real learner execution;
these synthetic tests do not establish that gate.

Coordination update from the runner owner: the disjoint `run_budget.py` is
implemented against the brief's learner API, and the probe gate showed real
physical response and short-tail differences. That probe evidence was not used
as labels or test data here. Actual training remains held pending test and
wrapper-review clearance. TD3 unit validation is complete; wrapper review is
outside this task and was not performed here.

## Implementation

- API: `TD3(observation_dim, seed, hidden=64)`, `act`, `add`, `update`,
  `state_dict`, `load_state_dict`, and `spec`; exposes `updates` and `replay`
  with `len(learner.replay)`.
- CPU float32, one Torch intra-op thread, no inter-op thread changes.
- Two hidden ReLU layers of width 64 by default, two tanh actor outputs,
  zero final actor weights and bias, and two independent critics.
- Adam learning rate `3e-4`, gamma exactly `1.0`, Polyak tau `0.005`, target
  Gaussian noise standard deviation `0.2`, noise clipping `0.5`, action
  clipping to `[-1, 1]`, and actor/target updates every two critic updates.
- Bellman targets use the target actor and elementwise minimum of target
  critics. Bootstrap is zero only for true termination. Targets have no
  gradient graph. The actor maximizes the first online critic.
- Rewards are accepted as already scaled interval rewards, represented in
  float32, and receive no clipping or additional scaling. Requested actions
  are stored and used in critic regression.
- Replay is a FIFO deque capped at 10,000 transitions, sampled uniformly
  with replacement. Inputs are copied. Validation rejects incorrect shapes,
  non-real/nonfinite values, float32 overflow, out-of-range actions, and
  non-boolean termination flags. Python bool and NumPy bool scalars are valid.
- `act` is deterministic and gradient-free. An undersized replay makes
  `update` return `{}` without advancing updates or RNGs. Successful updates
  return `critic_loss` and `updates`, plus `actor_loss` on delayed actor steps.
- Checkpoints include format, observation dimension, hyperparameters, all
  online and target parameters, both Adam states, update count, complete
  ordered replay, and local NumPy/Torch RNG states. Export/import own their
  storage. Replay uses packed CPU tensors and supports
  `torch.load(..., weights_only=True)`. Restore requires matching architecture
  and hyperparameters; the constructor seed may differ and is restored.
- Initialization preserves the global Torch RNG; updates use local RNGs.
  No evaluation-feature masking, solver differentiation, or external
  environment dependencies were introduced.

The equations were independently implemented after consulting the
[author's TD3 reference](https://github.com/sfujim/TD3/blob/master/TD3.py),
with the architecture and constants required by `brief_td3.md`.

## Verification

Environment: Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5, pytest 8.4.2.
Execution used `require_escalated` for the installed Windows package ACLs.
Only the TD3 synthetic test file ran. The prepared `.deps-budget` and task
directory were inserted on `sys.path`; no legacy RL `src` modules were imported.
Bytecode, pytest cache, and external pytest plugin autoload were disabled.

Command, run from the RL repository root:

```powershell
.\.venv-torch\Scripts\python.exe -B -c "import os, sys; os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'; sys.path[:0] = ['.deps-budget', 'work/sdmpc_rl_budget_20260929']; import pytest; raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', '--confcutdir=work/sdmpc_rl_budget_20260929', 'work/sdmpc_rl_budget_20260929/test_td3.py']))"
```

Coverage includes zero actor initialization; bounded deterministic inference;
target actor selection; noise scaling and both clipping stages; elementwise
twin minimum; continuing versus terminal bootstrap through both the target
helper and actual update; unclipped rewards and requested-action regression;
critic learning; delayed actor/target updates and exact Polyak interpolation;
absence of target gradients; validation and replay copy isolation; FIFO
capacity; empty checkpoints; checkpoint compatibility validation; and global
RNG/inter-op-thread isolation.

Checkpoint continuation is checked after 0, 1, 2, and 3 updates, across three
further updates each, using an in-memory Torch serialization round trip.
Losses, all model/optimizer/replay/RNG states, and actions match exactly despite
intervening draws from global RNGs. A separate capacity-wrap checkpoint case
verifies exact continuation after eviction and a subsequent replay insertion.
All verification is synthetic and within the recorded CPU environment.
