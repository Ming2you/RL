# BC 데이터 수집(Phase 1, 2026-07-22) — optimizer-leader teacher rollout → (obs, budget-action)
"""optimizer-leader(P-Stack)를 시나리오에서 굴리며 매 스텝 (env._observe, 선택 budget→action)을
남긴다. RL 정책의 behavioral-cloning 사전학습 데이터. teacher 궤적 = 진짜 optimizer 궤적.

관측 정의는 env._observe와 동일(배포 시 정책이 보는 것과 일치).
usage: python rl_leader/collect_bc_data.py <scenario> <out.npz> [T_total]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_leader.env import RLLeaderEnv
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController


def collect(scenario: str, out: str, T_total: float = 14400.0):
    env = RLLeaderEnv(scenario_name=scenario, T_total=T_total)
    teacher = StackelbergWuMeteredController(env.cfg)   # optimizer-leader
    env.reset()
    X, Y, meta = [], [], []
    done = False
    while not done:
        forecast = env._forecast()
        prev = env.previous if env.previous is not None else env._fixed_prev()
        control = teacher.decide(env.sim.state.copy(), forecast, prev)
        obs = env._observe()
        a = env.budget_to_action(control.N_P_star, control.N_UF_star)
        X.append(obs); Y.append(a)
        meta.append((env.step_idx, float(control.N_P_star), float(control.N_UF_star)))
        obs2, ttt, done = env.step_with_control(control)
        if env.step_idx % 10 == 0:
            print(f"  [{scenario}] step {env.step_idx}/{env.n_steps} "
                  f"N_P={control.N_P_star:.1f} N_UF={control.N_UF_star:.1f} cum_ttt={env.sim.total_ttt:.0f}",
                  flush=True)
    X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, Y=Y, meta=np.asarray(meta, dtype=np.float32), scenario=scenario)
    print(f"saved {out}  X={X.shape} Y={Y.shape}")


if __name__ == "__main__":
    scen = sys.argv[1] if len(sys.argv) > 1 else "sweet_170_w60"
    out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "rl_leader" / "bc_data" / f"{scen}.npz")
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 14400.0
    collect(scen, out, T)
