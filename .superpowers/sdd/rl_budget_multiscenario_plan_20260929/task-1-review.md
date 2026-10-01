# Task 1 review

Reviewed: 2026-09-29. Scope: Task 1 learner and tests only.

## Verdicts

- **SPEC: FAIL / changes required.** Equal scenario replay/sampling, unchanged TD3 equations, and valid checkpoint continuation satisfy the brief. Strict checkpoint hyperparameter, completeness, and counter validation have the three P2 gaps below.
- **QUALITY: FAIL / changes required.** The implementation is focused and readable, and the behavioral tests are substantial. The restore boundary silently accepts inconsistent training state; the negative checkpoint tests omit the reproduced cases. Resolve R1-R3 and add focused rejection tests before approval.

All findings concern the explicitly requested strict checkpoint contract. R1 and the optimizer-loading part of R2 inherit permissive behavior from the prior core; they are unmet Task 1 requirements, not newly changed TD3 equations. These findings do not depend on requiring transactional recovery for arbitrary corrupt network payloads.

## Findings

### R1 [P2] Validate effective optimizer hyperparameters, not only spec metadata

Location: [td3.py:313](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:313), both optimizer loads at lines 313-314; the separate spec comparison is at lines 252-255.

The loader checks `spec` and then imports optimizer parameter groups without checking their effective settings. Starting with a valid checkpoint after two updates and changing only `critic_optimizer.param_groups[0].lr` to `0.0` is accepted. The restored learner still reports `spec()['learning_rate'] == 0.0003`, while its actual critic learning rate is zero. A subsequent update increments the counters but leaves every critic parameter unchanged. This violates exact hyperparameter checking and silently changes the required learning behavior.

Action: validate both saved optimizer parameter groups against the learner's required Adam configuration, including learning rate and other settings that change the update math, before installing state. Reject conflicts even when the spec metadata is unchanged. Extend the incompatible-checkpoint tests at [test_td3.py:481](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:481) with parameter-group mutations for each optimizer.

### R2 [P2] Require complete optimizer history consistent with the update counter

Location: [td3.py:313](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:313), with unchecked update-counter installation at line 318.

A checkpoint with `updates == 2` is accepted after replacing `critic_optimizer['state']` with an empty dict. The next update lazily reinitializes Adam history; the resulting critic parameters differ from a valid restore of the same checkpoint. Separately, changing only `updates` from 2 to 3 is accepted even though every saved critic Adam step is 2. The next call consequently executes an actor/target update at count 4, whereas the intact checkpoint would perform a critic-only update at count 3. Successful restore therefore does not establish completeness or a coherent continuation phase.

Action: validate expected optimizer state entries, moment shapes, and step counters against the saved network parameters and TD3 update count. Critic steps must match `updates`; actor steps must match `updates // policy_delay`. Preserve legitimate empty optimizer state before its first scheduled step. Reject missing history and step/count disagreement before installing the payload. Add focused negative cases alongside [test_td3.py:564](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:564); the existing valid round-trip test does not exercise these corruptions.

### R3 [P2] Bound cumulative sampling counters by the restored replay history

Location: [td3.py:266](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:266).

The counter check uses `updates * replay_capacity` regardless of the actual replay sizes. A checkpoint with only eight stored transitions in each group and two completed updates accepts 4,000 samples per scenario. That history is impossible through the public API: replay lengths never shrink, and every update requires each group to contain at least its per-scenario batch contribution. With eight stored entries, two updates can have sampled at most 16 entries per scenario, even with replacement. Accepting this payload defeats the required invalid-counter rejection and restores false cumulative sampling totals.

Action: after validating replay groups, also require the equal cumulative count to be no greater than `updates * min(restored_group_lengths)`, retaining the existing integer, equality, and lower-bound checks. This also rejects positive update histories with empty groups. Add a case with an impossible count below the current capacity-based upper bound to [test_td3.py:564](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:564).

## Requirements verified by inspection and existing reported evidence

| Area | Assessment |
| --- | --- |
| Public API and input validation | Correct fixed ordered scenario tuple, required scenario argument, copied float32 numeric inputs, original-precision action bounds, finite checks, and boolean-only termination. |
| Replay balance | Independent FIFO deques each capped at 2,000; majority overflow cannot evict minority transitions. Counts reflect stored entries. |
| Warmup and sampling | Positive integer batch size divisible by five; every group must reach its quota. Default batch draws exactly eight per group with replacement using the local NumPy RNG. |
| Shared gradients | One combined twin-critic loss and one critic optimizer step. Delayed actor uses the same balanced observations. Scenario IDs never enter network inputs. |
| TD3 math | Direct comparison with the carry core shows unchanged initialization, architecture, target minimum, terminal masking, noise/clipping, losses, optimizers, delay, and Polyak equations. CPU float32 and one Torch thread are preserved. |
| Valid checkpoints | Export deep-copies all networks, targets, optimizer state, replay, RNGs, and counters. Restore copies replay inputs and optimizer/RNG payloads. No valid-checkpoint aliasing or RNG continuation defect found. |
| Invalid scenarios and replay | Exact ordered scenario metadata and keyed group sets are checked. Replay column names, tensor type/device/dtype/layout, shape, capacity, finiteness, and action bounds are validated. |
| Test quality | Tests inspect actual sampled critic/actor batches, eviction, terminal targets, optimizer steps, exact continuation, source mutation, and direct numerical parity with the carry implementation. Negative optimizer/history cases in R1-R3 are missing. |

Evidence anchors: [warmup and actual sampling tests:159](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:159), [eviction test:341](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:341), [checkpoint continuation:427](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:427), [global RNG isolation:603](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:603), [carry-core math parity:622](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:622).

## Review verification

Read the Task 1 brief, implementer report, supplied diff, full new learner/test source, and prior carry learner. Recomputed the direct carry-to-multi diff; the changes match the supplied diff. The implementer's reported **122 passed in 4.51 seconds** was accepted as prior evidence and was **not rerun**.

One additional in-memory numerical process investigated the concrete restore concerns above. It imported only the learner and its numerical dependencies, used synthetic transitions, disabled bytecode, and wrote no source, test, or checkpoint files. It used the existing Torch environment with approved sandbox escalation for dependency access. Exit code: 0. Output:

```text
optimizer_lr_mismatch: accepted=True, spec_lr=0.0003, effective_lr=0.0,
  critic_unchanged_after_update=True
missing_optimizer_state: accepted=True, updates=2, critic_state_entries=0
missing_optimizer_state_continuation: critics_match=False
samples_exceed_possible_history: accepted=True, updates=2,
  samples_per_group=4000, replay_counts=8 in each of the five groups
updates_disagree_with_optimizer: accepted=True, stored_updates=3,
  saved_critic_steps=[2.0], actor_updated_next=True
```

Exact probe command, run from the RL repository:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
$reviewProbe = @'
from copy import deepcopy
import json
import sys
import torch
sys.path.insert(0, 'work/sdmpc_rl_multi_20260929')
from td3 import TD3, SCENARIOS
learner = TD3(3, seed=19, hidden=8)
for scenario in SCENARIOS:
    for i in range(8):
        learner.add([i / 10, 1, -1], [0.2, -0.3], -i - 1, [i / 10 + 0.1, 1, -1], i % 3 == 0, scenario)
learner.update()
learner.update()
valid = learner.state_dict()
def restore(state):
    result = TD3(3, seed=999, hidden=8)
    result.load_state_dict(state)
    return result
wrong_lr = deepcopy(valid)
wrong_lr['critic_optimizer']['param_groups'][0]['lr'] = 0.0
restored = restore(wrong_lr)
params = [p.detach().clone() for p in restored.critics.parameters()]
restored.update()
print(json.dumps({'case': 'optimizer_lr_mismatch', 'accepted': True, 'spec_lr': restored.spec()['learning_rate'], 'effective_lr': restored.critic_optimizer.param_groups[0]['lr'], 'critic_unchanged_after_update': all(torch.equal(a, b) for a, b in zip(params, restored.critics.parameters()))}))
missing_moments = deepcopy(valid)
missing_moments['critic_optimizer']['state'] = {}
restored = restore(missing_moments)
reference = restore(valid)
print(json.dumps({'case': 'missing_optimizer_state', 'accepted': True, 'updates': restored.updates, 'critic_state_entries': len(restored.critic_optimizer.state)}))
restored.update()
reference.update()
print(json.dumps({'case': 'missing_optimizer_state_continuation', 'critics_match': all(torch.equal(a, b) for a, b in zip(reference.critics.parameters(), restored.critics.parameters()))}))
impossible_samples = deepcopy(valid)
impossible_samples['sample_counts'] = dict.fromkeys(SCENARIOS, 4000)
restored = restore(impossible_samples)
print(json.dumps({'case': 'samples_exceed_possible_history', 'accepted': True, 'updates': restored.updates, 'samples_per_group': restored.sample_counts[SCENARIOS[0]], 'replay_counts': restored.replay_counts()}))
wrong_updates = deepcopy(valid)
wrong_updates['updates'] = 3
restored = restore(wrong_updates)
metrics = restored.update()
print(json.dumps({'case': 'updates_disagree_with_optimizer', 'accepted': True, 'stored_updates': 3, 'saved_critic_steps': sorted(set(float(s['step']) for s in valid['critic_optimizer']['state'].values())), 'actor_updated_next': 'actor_loss' in metrics}))
'@
& '.venv-torch/Scripts/python.exe' -B -c $reviewProbe
```

Reviewed source SHA256:

```text
work/sdmpc_rl_multi_20260929/td3.py
1E4F1D277574362460B8EE33003089AFCA97E8B755814D363058FD6CC839D31F
work/sdmpc_rl_multi_20260929/test_td3.py
1879CF8FE8057C34DD69E3EDDDACCC44A7E778212B38B43FC5E60CFC02EA39AD
work/sdmpc_rl_carry_20260929/td3.py
746B9C6C45ABA37CF9DC169274CB64A22EC51FCB1B2C5DC06F2D7DD3104EE1B3
```

The carry hash matches the implementer's preservation record. Only this review report was written. No runner/collector files were reviewed or edited, no traffic simulation ran, and no commit was made.
