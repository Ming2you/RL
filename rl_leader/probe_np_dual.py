# N_P dual 되쓰기 효과(2026-08-05) — λ_P가 살아나나, 그러면 N_P가 레버가 되나
"""env가 컨트롤러 레이어를 우회해 _lambda_P가 0에 고정돼 있었다(probe_np_multistate: 8상태
전부 λ_P=0). _commit_np_dual로 되쓰기를 복제했을 때:
  (a) λ_P가 0을 벗어나 실제로 누적되나
  (b) 그러면 N_P 스윕에 follower(green)가 반응하나
np_dual OFF/ON을 같은 시나리오로 굴려 비교한다.

usage: python rl_leader/probe_np_dual.py [max_step] [every]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, SIGNALS
from src.controllers.leader import LeaderAction

MAXST = int(sys.argv[1]) if len(sys.argv) > 1 else 40
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SCEN = "sweet_190_skew15_w60"
NUF, NP0 = 5254.0, 1161.0
NPS = [-3000.0, 0.0, 2200.0]


def run(np_dual):
    env = RLLeaderEnv(scenario_name=SCEN, np_dual=np_dual)
    env.reset()
    a = env.budget_to_action(NP0, NUF)
    lam_traj, spreads = [], []
    while env.step_idx < MAXST:
        if (env.step_idx - env.warmup) % EVERY == 0:
            st, prev, fc = env.sim.state, env.previous, env._forecast()
            greens = []
            for npv in NPS:
                c = env.follower.solve(st.copy(), LeaderAction(npv, NUF), fc, prev).control
                greens.append(sum(float(c.green_times.get(f"{s}_p1", 0.0)) for s in SIGNALS))
            spreads.append((env.step_idx, float(env.follower._lambda_P),
                            max(greens) - min(greens), greens))
        env.step(a)
        lam_traj.append(float(env.follower._lambda_P))
    return lam_traj, spreads


for mode in (False, True):
    lam, sp = run(mode)
    lam = np.array(lam)
    print(f"\n=== np_dual = {mode} ===")
    print(f"  λ_P 궤적: min={lam.min():.3f} max={lam.max():.3f} mean={lam.mean():.3f} "
          f"| 0이 아닌 스텝 {int((lam > 1e-9).sum())}/{len(lam)}")
    print(f"  {'step':>5} {'λ_P':>8} {'green변동':>10}   green(-3000/0/+2200)")
    for stp, lp, spread, gs in sp:
        print(f"  {stp:5d} {lp:8.3f} {spread:10.1f}   " + "/".join(f"{g:.1f}" for g in gs))
    live = sum(1 for _, _, s, _ in sp if s > 0.5)
    print(f"  → N_P가 green을 움직인 지점: {live}/{len(sp)}")
