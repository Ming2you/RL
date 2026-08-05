# 레벨가격 파라미터화 탐색(2026-08-01) — 래칫 붕괴를 어떻게 끊나
"""probe_price_stability.py 실측: split=False + 균일가격 g=+150이면 Σmeter가
[4200,3000,2100,1500,0,0,...]로 0까지 붕괴한다(TTT +2957 악화).

원인: 가격비용 = w·g·(meter − ref)에서 ref = 직전 스텝 metering(env.py가 그렇게 주입).
기준점이 따라 내려오므로 "지금보다 더 줄여라"가 매 스텝 반복 → 수렴점 없는 래칫.
budget anchor는 w=T_c_h≈0.05로 사실상 무력(wu_faithful_follower.py:2992 주석).

후보:
  prev    — 현행(대조군). 붕괴 예상.
  budget  — ref를 budget 함의 수준(ω·N_UF/램프수)에 고정. 가격이 그 주변 편차만 유발 → 균형점 존재.
  anchorW — ref=budget + budget penalty weight 상향(anchor를 실제로 물게).

usage: python rl_leader/probe_price_anchor.py <variant> [steps] [price]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS
from src.controllers.leader import LeaderAction

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "budget"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
PRICE = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
SCEN = "sweet_190_skew15_w60"
NP0, NUF0 = 1161.0, 5254.0
LINK_OF = {"R_D_W": "FW_W", "R_F_W": "FW_W", "R_D_E": "FW_E", "R_F_E": "FW_E"}


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    f = env.follower
    f.metering_price_split = False
    if VARIANT == "anchorW":
        f.metering_budget_penalty_weight = float(
            getattr(f, "metering_budget_penalty_weight", 0.05)) * 20.0
    w0 = getattr(f, "metering_budget_penalty_weight", None)
    env.reset()
    print(f"=== variant={VARIANT} g={PRICE:+.0f} steps={STEPS} "
          f"budget_penalty_w={w0} ===", flush=True)

    tot, ttt, t0 = [], 0.0, time.time()
    prev = env.previous
    for k in range(STEPS):
        la = LeaderAction(NP0, NUF0)
        om = f._wu._omega_f
        if VARIANT == "prev":
            ref = {rp: float(prev.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
        else:   # budget 함의 수준에 고정 — 링크예산을 램프 수로 균등 분배
            ref = {rp: float(om.get(LINK_OF[rp], 0.5)) * NUF0 / 2.0 for rp in RAMPS}
        f.metering_marginal_price = {rp: PRICE for rp in RAMPS}
        f.metering_marginal_price_ref = ref
        f.metering_marginal_price_trust_frac = 0.20
        try:
            ctrl = f.solve(env.sim.state.copy(), la, env._forecast(), prev).control
        finally:
            f.metering_marginal_price = None
        log = env.sim.step(ctrl, env._forecast()[0], env.step_idx)
        env.previous = ctrl.copy(); prev = env.previous; env.step_idx += 1
        s = sum(float(ctrl.ramp_metering.get(rp, 0.0)) for rp in RAMPS)
        tot.append(s); ttt += float(log.urban_ttt + log.freeway_ttt)

    tot = np.array(tot)
    tv = float(np.abs(np.diff(tot)).sum())
    tail = tot[len(tot) // 2:]
    print(f"  궤적: {np.round(tot, 0).tolist()}", flush=True)
    print(f"  mean={tot.mean():7.1f} std={tot.std():7.1f} min={tot.min():.0f} max={tot.max():.0f}",
          flush=True)
    print(f"  후반부 mean={tail.mean():7.1f} std={tail.std():6.1f}  TV={tv:.1f}", flush=True)
    print(f"  누적 TTT={ttt:.1f}   소요={time.time()-t0:.0f}s  스텝당={((time.time()-t0)/STEPS):.1f}s",
          flush=True)
    collapsed = tail.mean() < 1000.0
    stable = tail.std() < 400.0
    print(f"  → 붕괴={collapsed}  후반안정={stable}  "
          f"{'OK' if (not collapsed and stable) else 'REJECT'}", flush=True)


if __name__ == "__main__":
    main()
