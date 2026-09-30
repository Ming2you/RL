"""One real interval, serialized resume parity, and requested-action replay check."""
import argparse
import io
from pathlib import Path
from budget_runtime import boot, DEFAULT_SNAPSHOT, save, plain
from run_budget import pins, verify_pins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-mask", type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rt = boot(DEFAULT_SNAPSHOT, args.cpu_mask)
    source_pins = pins(DEFAULT_SNAPSHOT)
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
        assert audit["pfo_calls"] == 1 and audit["reference_source"] == "pfo_initial"
        assert not audit["h3_guard_enabled"]
        assert env.controller.reference["source"] == "previous" and env.reference_timing["pfo_calls"] == 0
        # Restore an ordinary carried-budget step as well, not just reset/PFO state.
        buffer = io.BytesIO()
        torch.save(env.checkpoint(), buffer)
        buffer.seek(0)
        checkpoint2 = torch.load(buffer, map_location="cpu", weights_only=False)
        np.testing.assert_array_equal(next_obs, other.restore(checkpoint2))
        np.testing.assert_array_equal(env.controller.action_anchor, other.controller.action_anchor)
        second_obs, second_reward, second_terminal, second_audit = env.step([0., 0.])
        restored_second, other_reward, other_terminal, other_audit = other.step([0., 0.])
        np.testing.assert_array_equal(second_obs, restored_second)
        assert second_reward == other_reward and second_terminal == other_terminal
        assert plain(second_audit["control"]) == plain(other_audit["control"])
        assert second_audit["pfo_calls"] == 0 and second_audit["reference_source"] == "previous"
        learner = TD3(len(obs), 6100)
        learner.add(obs, action, reward, next_obs, bool(terminal))
        np.testing.assert_array_equal(learner.replay[0][1], action)
        assert abs(audit["interval_ttt"] / 100. + reward) < 1e-12
        verify_pins(DEFAULT_SNAPSHOT, source_pins)
        save(args.output, dict(passed=True, snapshot_identity=rt["snapshot_identity"],
             source_pins=source_pins,
             observation_dim=len(obs), serialized_resume=True, step=env.k,
             requested_action=action, reward=reward, terminated=terminal,
             scope="two_actual_intervals_with_serialized_resume_not_full_run", audit=audit,
             carried_step_audit=second_audit))
        print("SMOKE_RESUME_REPLAY_PASS", len(obs), reward, flush=True)
    finally:
        env.close()
        other.close()
        rt["verify"]()


if __name__ == "__main__":
    main()
