# Task: small tested TD3 learner, not a simulator or runner

Repository C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL, branch
codex/sdmpc-rl-budget-20260929. Preserve legacy code and runs.

Write scope only: work/sdmpc_rl_budget_20260929/td3.py,
work/sdmpc_rl_budget_20260929/test_td3.py,
work/sdmpc_rl_budget_20260929/td3_report.md. Use apply_patch; do not git commit.

Implement ordinary two-dimensional continuous TD3, not old DDQN or fixed-return
label learning. Standard reference equations are in the author's public code
https://github.com/sfujim/TD3/blob/master/TD3.py. Prefer independent concise
implementation of equations; no external framework required. Existing torch is
installed in .venv-torch. Tests need require_escalated due Windows package ACL.
Parent prepared .deps-budget numpy2.3.5/scipy1.16.3/pytest8.4.2. Insert .deps-budget
and this task directory on sys.path when running tests, never import old RL src.

API: TD3(observation_dim:int, seed:int, hidden:int=64), .act(observation)->np.ndarray
shape(2,), deterministic and no gradients; .add(obs, requested_action, reward,
next_obs, terminated:bool) stores copies in replay; .update(batch_size=32)->dict
or empty dict when fewer than batch_size samples; .state_dict()->dict and
.load_state_dict(state)->None exact optimizer/target/update-count/replay/RNG resume.
Expose .updates and .replay (length) for runner; .spec() metadata hyperparameters.
CPU and torch.set_num_threads(1). Do not alter inter-op threads after work started.

Actor MLP64x64 with tanh2 output and zero final linear weights/bias. Twin independent
critics Q(s,a). gamma=1 exactly, r is already actual negative interval TTT /100.
Learning rate3e-4, target polyak tau=.005, target-policy Gaussian noise std.2 clip.5,
action clamp[-1,1], actor/targets update every2 critic updates. Use target actor and
min of target critics, zero bootstrap only on true terminated. No reward clipping,
failure terminal relabel, CQL, response previews, or differentiating through solver.
Bound replay capacity10000, no constant-feature mask based on evaluation data.
Validate finite inputs/dimensions/action range and genuinely boolean terminated
(not truthy strings); never mutate caller arrays. Save optimizer/torch/NumPy local
RNG state. Saved replay should be full for exact continuation; serialization is
handled by parent via torch.save of state_dict. Include format and observation_dim.

Tests: zero actor; bounded deterministic act; correct target bootstrap on continuing
vs true terminal (factor target construction into testable helper if useful);
actor and target delay; twin min; no target gradient; input/replay copy validation;
save/load reproduces next update and action exactly with RNG and optimizers.
Report tests and changed paths in td3_report.md. Do not start simulation/training
episodes; only synthetic unit tests. User authorized this task as step5 preparation,
real learner must not run until action-influence gate passes.
