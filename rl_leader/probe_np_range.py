# N_P 음수 영역 응답 probe(2026-08-05) — 박스 확장 전 게이트
"""발견: P-Stack leader는 N_P를 −3315까지 탐색하는데(leader_intent_N_P_star) 우리 액션
박스는 [0, 2200]이라 절반이 잘려 있다. 음수 요청 스텝에서 λ_P=10(상한 포화)이므로
제약이 실제로 binding하고, net_inflow와 corr −0.564로 방향도 살아 있다.

그러나 realized N_P는 P-Stack에서도 +255 아래로 안 내려간다(follower가 물리적으로
순유출을 못 만듦). 그렇다면 박스만 넓혀도 소용없을 수 있다 — 확인할 것:

  (a) 음수 N_P를 명령하면 follower의 urban 결정(green)이 실제로 바뀌나
  (b) λ_P가 어떻게 반응하나 (0=비활성 / 10=포화)
  (c) step TTT의 urban/freeway 분해가 P-CENT식으로 이동하나(urban 지불 ↔ freeway 회수)

usage: python rl_leader/probe_np_range.py [warm] [n_uf]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS, SIGNALS
from src.controllers.leader import LeaderAction

WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 14
NUF = float(sys.argv[2]) if len(sys.argv) > 2 else 5254.0
SCEN = "sweet_190_skew15_w60"
GRID = [-3500.0, -2500.0, -1500.0, -500.0, 0.0, 1000.0, 2200.0]


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    env.reset()
    a = env.budget_to_action(1161.0, NUF)
    while env.step_idx < WARM:
        env.step(a)
    state, prev, fc = env.sim.state, env.previous, env._forecast()
    f = env.follower
    print(f"=== {SCEN} step{env.step_idx}, N_UF={NUF:.0f} 고정, N_P 스윕 ===")
    print(f"{'N_P':>8} | {'λ_P':>6} | " + " ".join(f"{s:>5}" for s in SIGNALS)
          + f" | {'green합':>8} | {'Σmeter':>8}")
    base = None
    for npv in GRID:
        ctrl = f.solve(state.copy(), LeaderAction(npv, NUF), fc, prev).control
        g = {s: float(ctrl.green_times.get(f"{s}_p1", float("nan"))) for s in SIGNALS}
        lam = float(ctrl.diagnostics.get("wu_faithful_lambda_P", float("nan"))) \
            if hasattr(ctrl, "diagnostics") and ctrl.diagnostics else float("nan")
        mtot = sum(float(ctrl.ramp_metering.get(rp, 0.0)) for rp in RAMPS)
        if base is None:
            base = dict(g)
        print(f"{npv:8.0f} | {lam:6.2f} | " + " ".join(f"{g[s]:5.1f}" for s in SIGNALS)
              + f" | {sum(g.values()):8.1f} | {mtot:8.0f}")

    gs = []
    for npv in GRID:
        ctrl = f.solve(state.copy(), LeaderAction(npv, NUF), fc, prev).control
        gs.append(sum(float(ctrl.green_times.get(f"{s}_p1", 0.0)) for s in SIGNALS))
    gs = np.array(gs)
    print(f"\n  green 합 범위 {gs.min():.1f} ~ {gs.max():.1f}  (변동폭 {gs.max()-gs.min():.1f}s)")
    print("\n판정: 음수 구간에서 green이 유의하게 움직이면 → 레버가 살아 있어 박스 확장 가치 있음.")
    print("      전 구간 동일하면 → follower가 N_P에 무반응이라 박스만 넓혀도 무의미.")


if __name__ == "__main__":
    main()
