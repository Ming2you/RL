# Task 1: balanced shared TD3 learner

Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL.
Own ONLY work/sdmpc_rl_multi_20260929/td3.py and test_td3.py plus this task's
report file. Start from the exact prior work/sdmpc_rl_carry_20260929/td3.py core;
never edit prior experiment files. No commits, no traffic simulation.

Implement a single shared TD3 actor and two shared critics with equal scenario
replay and gradient sampling. Fixed scenario order:
sweet_155_w, sweet_170_w, sweet_170_incident_w, sweet_170_skew15_w, sweet_190_w.

Public API:
- TD3(observation_dim, seed, hidden=64), fixed SCENARIOS constant.
- act(obs) identical bounded NumPy2vector output.
- add(obs, requested_action, reward, next_obs, terminated, scenario): required
  valid scenario ID; copied numeric arrays, true boolean termination, same strict
  bounds and finite validation as prior core. Each scenario has independent deque
  capacity2000; same capacity means balanced eviction, no minority eviction.
- update(batch_size=40): reject invalid/nondivisible-by5sizes. Return{} until
  every group has at least batch_size/5transitions. Sample exactly8per group for
  default batch with replacement; use dedicated checkpointed RNG. One combined
  loss/optimizer step, not five separate models or optimizers. Metrics include
  sampled_per_scenario for this update; retain cumulative sample counts.
- replay_counts(): dict scenario->stored count; total_transition_count() returns
  sum of those counts. No old global .replay API is required by the new runner.
- state_dict/load_state_dict: new format sdmpc-multi-td3-v1, exact hyperparameter
  and scenario-order checks, complete independent replay copies/RNG/optimizers/
  networks/targets/update counter/sampling counters. Reject old checkpoints,
  unknown/missing/duplicate scenarios, malformed replay shapes, invalid counters.
  Replay payload is dict keyed by scenario, each column layout like prior core:
  observations,actions,rewards,next_observations,terminated CPU tensors.

Keep original TD3 math unchanged: gamma1, lr3e-4, tau.005, smoothing.2clip.5,
delay2, two64ReLUlayers, zero actor final layer, float32CPU, one Torch thread.
Preserve deterministic RNG initialization/restoration and independent checkpoint
snapshots. Scenario metadata is only for stratified replay, not actor/critic input.

Tests: equal sampling even when groups have very unequal amounts; all-five warmup;
no update from one scenario alone; strict input/default40/divisibility validation;
equal bounded eviction; unchanged true-terminal target/no cross-episode bootstrap;
exact save/load next action/update/counters parity and independent checkpoint copy;
source input mutation does not corrupt replay; old/wrong-scenario restore rejected.
Port relevant existing core tests; avoid tests that merely assert metadata strings.

Use .venv-torch/Scripts/python.exe and .deps-budget as needed. Numerical tests may
require sandbox escalation due to local dependency ACL; do not install packages or
change ACL. No more than one numerical test process. Write full report to
.superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-1-report.md with changed
files, exact commands/results and concerns. Return only status/test summary/paths.
