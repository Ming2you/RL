# Critic temporal diagnostic

Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL. Read this first.
Own ONLY work/sdmpc_rl_recovery_20260929/critic_diagnostic.py and
test_critic_diagnostic.py plus critic-report.md in this brief's directory.
Do not change prior versioned source, other new files, datasets, models, runtime,
git index, schedules or running experiments. No plant rollouts. No commit.

Use frozen final model
results/sdmpc_rl_multi_20260929/pilot_v1/train_round1/model_final.pt hash
3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650.
750 training transitions,150 per scenario, two true terminals per scenario.
Existing work/sdmpc_rl_multi_20260929/td3.py and helper modules read-only.

Implement a small standalone reproducible critic-only diagnostic. Two clones from
same final model, actor fixed at final actor, initialize target actor to this fixed
actor in BOTH conditions. Keep critic/optimizer initial states the same. Use3750
critic updates (10 times pilot round budget), exactly8samples/scenario in every
40batch with identical random samples/noise between arms. Preserve TD3 gamma1,
target smoothing noise. Arm A target tau=.005, arm B target tau=1; update critic
targets every2critic updates in both arms. No actor/actor-target updates. Log at
0,375,750,1500,3750 including per-scenario terminal error and behavior return
gap (explicitly not current-policy truth), total TD loss, sample counts and
time-slice diagnostics. No policy exported, no data collected. Freeze and hash
input, record own source/runtime/seed/parameters, refuse output overwrite, check
STOP file between batches, save partial diagnostics then exit cleanly. OneCPUthread.
Do not run full3750experiment until coordinator requests; run focused synthetic
tests only. Tests must cover true terminals, unchanged actor, exactbalance,
same arm samples/noise, target schedule and no input mutation. Keep implementation
small; no new framework/admission system. Use existing .venv-torch with .deps-budget
path, escalated local ACL if needed. Write report with spec/quality self-review,
test result, files, concerns; no broad claims of causal superiority from this.
