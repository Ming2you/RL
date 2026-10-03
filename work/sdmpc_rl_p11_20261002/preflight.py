"""P11 actor history/reload tests and one real interval, never a score."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/sdmpc_rl_p11_20261002"
sys.path.insert(0, str(REPO / "work/sdmpc_rl_multi_20260929"))
from budget_runtime import boot, DEFAULT_SNAPSHOT


def main():
    if (OUT / "preflight.json").exists():
        raise FileExistsError("Preserve prior preflight")
    if any((p / "STOP").exists() for p in (REPO,OUT,OUT / "fit_v1")):
        raise RuntimeError("STOP")
    rt = boot(DEFAULT_SNAPSHOT, 1)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    from neural_policy import NeuralPolicy
    torch.set_num_threads(1)
    spec = OUT / "fit_v1/policy.json"
    completion = json.loads((OUT / "fit_v1/completion.json").read_text())
    if completion["status"] != "fit_passed":
        raise ValueError("Fitting gate did not pass")
    env = BudgetEnv(rt, scenario="sweet_170_w", training_seed=8702, guard_mode="physical")
    try:
        cache = Path("D:/RL_data/sdmpc_rl_machine_b_20260930/ckpt/sweet_170_w_s8702/k16.pt")
        obs = env.restore(torch.load(cache, map_location="cpu", weights_only=False))
        names = env.observer.names
        a, b = NeuralPolicy.load(spec,names), NeuralPolicy.load(spec,names)
        ridx = list(names).index("memory/remaining/0")
        ia = [list(names).index("memory/action_anchor/0"),list(names).index("memory/action_anchor/1")]
        for step in range(1,76):
            o = obs.copy(); o[ridx] = (76-step)/75.
            if step > 16:
                o[ia] -= np.array([.1,.1],np.float32)
            x, y = a.act(o), b.act(o)
            np.testing.assert_array_equal(x,y)
            if step < 16:
                np.testing.assert_array_equal(x,np.zeros(2,np.float32))
            if step > 30:
                expected = np.clip((obs[ia].astype(float)-o[ia].astype(float))*[20.,10.],-1.,1.).astype(np.float32)
                np.testing.assert_array_equal(x,expected)
            assert x.dtype == np.float32 and np.isfinite(x).all() and np.max(np.abs(x))<=1.
        for bad in (obs.astype(np.float64),np.full_like(obs,np.nan)):
            try: NeuralPolicy.load(spec,names).act(bad)
            except ValueError: pass
            else: raise AssertionError("Bad observation accepted")
        fresh = NeuralPolicy.load(spec,names)
        action = fresh.act(obs)
        try: fresh.act(obs)
        except ValueError: pass
        else: raise AssertionError("Repeated decision accepted")
        next_obs, reward, terminal, row = env.step(action,"rl",0.,actor_cpu_seconds=0.)
        assert row["control_step"]==15 and not terminal and next_obs.shape==(2367,)
        assert reward == -row["interval_ttt"]/100.
        rt["verify"]()
        report = dict(status="passed", scope="actor_contract_and_one_training_interval_only",
            spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(), model_sha256=completion["model_sha256"],
            checkpoint_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(), action=action.tolist(),
            tests=["identical reload actions", "carry prefix", "latched both-budget return", "sequential history",
                   "finite float32 bounded actions", "nonfinite/dtype rejection", "actual physical interval"],
            torch_threads=torch.get_num_threads(), snapshot=rt["snapshot_identity"])
        with (OUT / "preflight.json").open("x",encoding="utf-8") as f:
            json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
