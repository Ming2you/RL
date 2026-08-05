# N_P가 '어떤 상태에서든' 죽어 있나(2026-08-05) — 단일 상태 probe의 일반화
"""probe_np_range.py는 step14에서 N_P −3500~+2200이 green을 전혀 안 움직이고 λ_P=0임을 보였다.
그러나 P-Stack 궤적에선 λ_P가 11% 스텝에서 10(포화)이므로 binding 상태는 존재한다.
여러 스텝에서 반복해 'N_P가 어디서든 죽었나 / 특정 국면에서만 사나'를 가른다.

각 probe 지점에서 N_P ∈ {-3000, 0, +2200} 3점만 풀어 비용을 아낀다.

usage: python rl_leader/probe_np_multistate.py [every] [max_step]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS, SIGNALS
from src.controllers.leader import LeaderAction

EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAXST = int(sys.argv[2]) if len(sys.argv) > 2 else 45
SCEN = "sweet_190_skew15_w60"
NUF = 5254.0
NPS = [-3000.0, 0.0, 2200.0]


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    env.reset()
    a = env.budget_to_action(1161.0, NUF)
    print(f"=== {SCEN}, N_UF={NUF:.0f} 고정, N_P 3점 비교 ===")
    print(f"{'step':>5} {'t(s)':>6} {'rho_max':>8} | {'λ_P(-3000/0/+2200)':>22} | "
          f"{'green합':>22} | {'green 변동':>10}")
    live = 0
    tot = 0
    while env.step_idx < MAXST:
        if (env.step_idx - env.warmup) % EVERY == 0:
            st, prev, fc = env.sim.state, env.previous, env._forecast()
            lams, greens = [], []
            for npv in NPS:
                c = env.follower.solve(st.copy(), LeaderAction(npv, NUF), fc, prev).control
                d = getattr(c, "diagnostics", None) or {}
                lams.append(float(d.get("wu_faithful_lambda_P", float("nan"))))
                greens.append(sum(float(c.green_times.get(f"{s}_p1", 0.0)) for s in SIGNALS))
            spread = max(greens) - min(greens)
            tot += 1
            if spread > 0.5 or (np.nanmax(lams) - np.nanmin(lams)) > 1e-9:
                live += 1
            o = env._observe()
            print(f"{env.step_idx:5d} {env.step_idx*env.dt:6.0f} {float(o[5]):8.2f} | "
                  + "/".join(f"{x:6.2f}" for x in lams)
                  + " | " + "/".join(f"{g:7.1f}" for g in greens)
                  + f" | {spread:10.1f}")
        env.step(a)
    print(f"\n  N_P가 무언가를 바꾼 지점: {live}/{tot}")
    print("판정: 0/N 이면 N_P는 이 env에서 완전히 죽은 차원 — 액션 2차원 중 1차원이 무용.")


if __name__ == "__main__":
    main()
