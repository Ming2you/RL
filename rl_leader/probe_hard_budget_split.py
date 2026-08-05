# 하드 budget + ω + 가격 도달집합(2026-08-02) — "총량 유지, 램프 간 차등" 설계 검증
"""사용자 설계: 가격은 budget 제약을 깨지 않고, 그 안에서 램프별 배분에 차등을 준다.
   → split=True(총량 하드) + ω 액션(merge 간) + 가격(merge 내). 이 조합은 아직 미검증이다.
     Phase 4 = split=True + ω 고정(0.5) → merge 간 축이 죽어 있었다.
     Phase 5 = split=False → budget이 soft가 되어 총량 통제를 잃었다.

측정: (a) ω를 움직여도 총량이 N_UF로 보존되나(링크 cap 간섭 확인)
      (b) 도달 가능한 (W,E) 조합
      (c) 같은 총량에서 가격이 merge 내부를 재분배하나

usage: python rl_leader/probe_hard_budget_split.py [n_uf] [warm]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS, LINKS
from src.controllers.leader import LeaderAction

NUF = float(sys.argv[1]) if len(sys.argv) > 1 else 5254.0
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 10
NP0 = 1161.0
SCEN = "sweet_190_skew15_w60"


def solve(env, omega_w, prices):
    f = env.follower
    f.metering_price_split = True          # ★총량 하드 유지
    f._wu._omega_f = {LINKS[0]: omega_w, LINKS[1]: 1.0 - omega_w}
    if prices is not None:
        f.metering_marginal_price = {rp: prices[i] for i, rp in enumerate(RAMPS)}
        f.metering_marginal_price_ref = {
            rp: float(env.previous.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
        f.metering_marginal_price_trust_frac = 0.20
    try:
        ctrl = f.solve(env.sim.state.copy(), LeaderAction(NP0, NUF),
                       env._forecast(), env.previous).control
    finally:
        f.metering_marginal_price = None
    m = {rp: float(ctrl.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
    W = m["R_D_W"] + m["R_F_W"]
    E = m["R_D_E"] + m["R_F_E"]
    return m, W, E


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    env.reset()
    a = env.budget_to_action(NP0, NUF)
    while env.step_idx < WARM:
        env.step(a)
    cap = 3000.0
    lo, hi = max(0.0, 1.0 - cap / NUF), min(1.0, cap / NUF)
    print(f"=== {SCEN} step{env.step_idx}, N_UF={NUF:.0f} 고정, split=True(총량 하드) ===")
    print(f"  링크 cap={cap:.0f} → 총량 보존되는 ω 구간 = [{lo:.3f}, {hi:.3f}]\n")
    print(f"{'ω_W':>6} | {'W':>7} {'E':>7} | {'총량':>7} | {'N_UF 대비':>9} | 램프별")
    for w in [0.20, 0.30, lo, 0.45, 0.50, 0.55, hi, 0.70, 0.80]:
        w = float(np.clip(w, 0.01, 0.99))
        m, W, E = solve(env, w, None)
        tot = W + E
        det = " ".join(f"{r.replace('R_','')}={m[r]:6.0f}" for r in RAMPS)
        print(f"{w:6.3f} | {W:7.0f} {E:7.0f} | {tot:7.0f} | {tot-NUF:+9.0f} | {det}")

    print(f"\n=== 같은 ω(0.5)에서 가격의 merge 내부 재분배 ===")
    for pv in [0.0, 300.0, -300.0]:
        pr = [pv, -pv, 0.0, 0.0]     # 서쪽 merge 내부만 차등
        m, W, E = solve(env, 0.5, pr)
        det = " ".join(f"{r.replace('R_','')}={m[r]:6.0f}" for r in RAMPS)
        print(f"  p(R_D_W)={pv:+6.0f} | W={W:6.0f} E={E:6.0f} 총량={W+E:6.0f} | {det}")


if __name__ == "__main__":
    main()
