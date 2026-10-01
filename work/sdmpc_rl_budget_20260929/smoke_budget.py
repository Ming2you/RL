"""One real interval, serialized resume parity, and requested-action replay check."""
import argparse
import io
from pathlib import Path
from budget_runtime import boot, DEFAULT_SNAPSHOT, save, plain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-mask", type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rt = boot(DEFAULT_SNAPSHOT, args.cpu_mask)
    from budget_env import BudgetEnv
    from td3 import TD3
    import numpy as np
    import torch
    env = BudgetEnv(rt)
    other = BudgetEnv(rt)
    try:
        obs = env.reset()
        buffer = io.BytesIO()
        torch.save(env.checkpoint(), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
        np.testing.assert_array_equal(obs, other.restore(checkpoint))
        action = np.array([0.25, -0.25], dtype=np.float32)
        next_obs, reward, terminal, audit = env.step(action)
        restored_obs, restored_reward, restored_terminal, restored_audit = other.step(action)
        np.testing.assert_array_equal(next_obs, restored_obs)
        assert reward == restored_reward and terminal == restored_terminal and not terminal
        assert plain(audit["control"]) == plain(restored_audit["control"])
        np.testing.assert_array_equal(audit["committed_dual"], restored_audit["committed_dual"])
        assert len(audit["candidates"]) == 1
        learner = TD3(len(obs), 6100)
        learner.add(obs, action, reward, next_obs, bool(terminal))
        np.testing.assert_array_equal(learner.replay[0][1], action)
        assert abs(audit["interval_ttt"] / 100. + reward) < 1e-12
        rt["verify"]()
        save(args.output, dict(passed=True, snapshot_identity=rt["snapshot_identity"],
             observation_dim=len(obs), serialized_resume=True, step=env.k,
             requested_action=action, reward=reward, terminated=terminal,
             scope="one_actual_interval_not_full_run", audit=audit))
        print("SMOKE_RESUME_REPLAY_PASS", len(obs), reward, flush=True)
    finally:
        env.close()
        other.close()
        rt["verify"]()


if __name__ == "__main__":
    main()
