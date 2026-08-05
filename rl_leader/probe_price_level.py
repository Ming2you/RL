# 가격이 '총량'을 움직일 수 있는가 — split 모드 ON/OFF 비교(2026-08-01)
"""Phase 4의 가격이 무력했던 이유가 metering_price_split=True 때문인지 직접 잰다.

split=True (현행): 총량 Σmeter ≡ ω·N_UF 하드 고정, 가격은 merge 내부 배분 채점만.
split=False       : priced_metering 분기 활성 — budget은 soft anchor, 가격이 방류 수준 유도
                    (wu_faithful_follower.py:3048-3058, 주석 "가격이 방류 수준을 유도").

동일 상태에서 균일 가격 g를 전 램프에 걸고 총 방류량 응답곡선을 잰다.
가격이 총량을 못 바꾸면 split=True에서 평평, split=False에서 기울기가 나와야 한다.

usage: python rl_leader/probe_price_level.py [scenario] [warm_steps]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS
from src.controllers.leader import LeaderAction

SCEN = sys.argv[1] if len(sys.argv) > 1 else "sweet_190_skew15_w60"
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NP0, NUF0 = 1161.0, 5254.0
GRID = [-1000.0, -500.0, -250.0, 0.0, 250.0, 500.0, 1000.0]


def total_release(env, prices, split):
    f = env.follower
    f.metering_price_split = split
    f.metering_marginal_price = prices
    f.metering_marginal_price_ref = {
        rp: float(env.previous.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
    f.metering_marginal_price_trust_frac = 0.20
    try:
        ctrl = f.solve(env.sim.state.copy(), LeaderAction(NP0, NUF0),
                       env._forecast(), env.previous).control
    finally:
        f.metering_marginal_price = None
        f.metering_price_split = True
    m = {rp: float(ctrl.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
    return sum(m.values()), m


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    env.reset()
    a = env.budget_to_action(NP0, NUF0)
    while env.step_idx < WARM:
        env.step(a)
    print(f"=== {SCEN}, step {env.step_idx} (t={env.step_idx*env.dt:.0f}s), "
          f"budget N_UF={NUF0:.0f} 고정 ===", flush=True)
    print(f"{'g(veh/h)':>10} | {'split=True 총량':>16} | {'split=False 총량':>17} | 램프별(split=False)", flush=True)
    rows = []
    for g in GRID:
        pr = {rp: g for rp in RAMPS}
        t_split, _ = total_release(env, pr, True)
        t_lvl, m_lvl = total_release(env, pr, False)
        rows.append((g, t_split, t_lvl))
        detail = " ".join(f"{rp.replace('R_','')}={m_lvl[rp]:6.0f}" for rp in RAMPS)
        print(f"{g:10.0f} | {t_split:16.1f} | {t_lvl:17.1f} | {detail}", flush=True)

    ts = np.array([r[1] for r in rows]); tl = np.array([r[2] for r in rows])
    print(f"\nsplit=True  총량 범위 {ts.min():.1f}~{ts.max():.1f}  (변동폭 {ts.max()-ts.min():.1f})", flush=True)
    print(f"split=False 총량 범위 {tl.min():.1f}~{tl.max():.1f}  (변동폭 {tl.max()-tl.min():.1f})", flush=True)
    print("\n판정:", flush=True)
    if ts.max() - ts.min() < 1.0 and tl.max() - tl.min() > 50.0:
        print("  ★ 가격의 총량 채널은 split=False에서만 열린다 — Phase 4가 무력했던 구조적 이유 확정.", flush=True)
    elif tl.max() - tl.min() < 50.0:
        print("  split=False에서도 총량이 안 움직인다 — 다른 곳에서 막고 있다(추가 조사 필요).", flush=True)
    else:
        print("  split=True에서도 총량이 움직인다 — 전제 재검토 필요.", flush=True)


if __name__ == "__main__":
    main()
