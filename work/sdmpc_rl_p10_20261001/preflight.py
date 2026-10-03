"""One real training interval: authenticate import boundary and unchanged first action."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_multi_20260929"))
from budget_runtime import boot, DEFAULT_SNAPSHOT


def main():
    rt = boot(DEFAULT_SNAPSHOT, 16)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    from retention_actor import RetentionActor
    torch.set_num_threads(1)
    cache = Path("D:/RL_data/sdmpc_rl_machine_b_20260930/ckpt/sweet_190_w_s8705/k16.pt")
    env = BudgetEnv(rt, scenario="sweet_190_w", training_seed=8705, guard_mode="physical")
    try:
        state = torch.load(cache, map_location="cpu", weights_only=False)
        obs = env.restore(state)
        options = json.loads((HERE / "options.json").read_text())
        actions = []
        for option in options:
            actor = RetentionActor.load(REPO / option["spec_path"], env.observer.names)
            assert actor.sha256 == option["spec_sha256"]
            action = actor.act(obs)
            np.testing.assert_array_equal(action, np.array([-1., 0.], np.float32))
            actions.append(action.tolist())
        next_obs, reward, terminal, row = env.step(np.array(actions[0], np.float32), "rl", 0., actor_cpu_seconds=0.)
        old = json.loads((REPO / "results/sdmpc_rl_p9_20261001/wave2/sweet_190_w_s8705/branches/k16_p9_actor_b1_np_only.json").read_text())
        assert row["control_step"] == 15 and row["total_ttt"] == old["rows"][0]["total_ttt"]
        assert not terminal and next_obs.dtype == np.float32 and next_obs.shape == (2367,)
        assert reward == -row["interval_ttt"] / 100.
        rt["verify"]()
        report = dict(status="passed", scope="one_training_interval_only_not_performance", actions=actions,
                      first_interval_ttt=row["total_ttt"], actual_physical_steps=1,
                      checkpoint_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                      torch_threads=torch.get_num_threads(), snapshot=rt["snapshot_identity"])
        out = REPO / "results/sdmpc_rl_p10_20261001"
        out.mkdir(parents=True, exist_ok=True)
        (out / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
