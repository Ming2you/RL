# 천장 sweep(2026-07-23) — 고정 N_UF budget에서 P-Stack follower TTT (RL 상한 측정)
"""RLLeaderEnv(리더 탐색 우회, 고정 budget→follower→plant)로 N_UF를 손으로 조여가며
windowed TTT를 잰다. 이게 budget-RL이 달성 가능한 천장. tighter N_UF가 P-Stack(6379)을
PFO(6299)·P-CENT(5757) 쪽으로 내리면 천장 실재 → RL 학습기 재설계 가치 있음.

usage: python rl_leader/ceiling_sweep.py [scenario]
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv


def rollout_fixed(scen: str, n_p: float, n_uf: float):
    env = RLLeaderEnv(scenario_name=scen)
    env.reset()
    warm = float(env.sim.total_ttt)             # warmup(0..4) 누적
    a = env.budget_to_action(n_p, n_uf)         # 고정 액션
    done = False
    rel = []
    while not done:
        _, _, done, info = env.step(a)
        rel.append(info.get("N_UF", n_uf))
    win = float(env.sim.total_ttt) - warm
    return win, float(np.mean(rel))


def main(scen="sweet_190_skew15_w60"):
    NP = 1161.0   # P-Stack skew peak 평균값으로 고정, N_UF만 sweep
    print(f"=== 천장 sweep: {scen} (N_P 고정={NP:.0f}, N_UF sweep) ===", flush=True)
    print(f"기준: P-Stack 6379 | PFO 6299 | P-CENT 5757 | NC 6882", flush=True)
    print(f"{'N_UF':>7} {'windowed_TTT':>13} {'실현N_UF':>10}", flush=True)
    for n_uf in [6000, 5254, 4500, 3500, 2500, 1500, 800]:
        win, act = rollout_fixed(scen, NP, n_uf)
        print(f"{n_uf:7.0f} {win:13.1f} {act:10.1f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sweet_190_skew15_w60")
