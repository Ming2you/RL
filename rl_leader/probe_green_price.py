# urban green 가격 응답 probe(2026-08-03) — 수집 전 게이트
"""§12.6·12.7: P-CENT는 freeway에서 이기고 urban에서 지불한다(green sd가 P-Stack의 3.5~10배).
RL엔 per-signal urban 채널이 없어 그 거래를 못 한다. green 가격을 열기 전에 확인할 것:

 (1) 가격이 green_times를 실제로 움직이나 — 유효 스케일은?
 (2) bang-bang인가 점진적인가 (코드에 폭주 사례 기록: sweet_155 C 56→92)
 (3) per-signal 차등이 되나 (한 신호만 밀면 그 신호만 움직이나)
 (4) 총 green 예산 제약이 있나 (cycle 120s 안에서 상쇄되나)

비용식: cost += weight · g_ext · (p1 − ref)   (wu_faithful_follower.py:879-883)
양수 가격 → p1을 ref보다 줄이는 게 유리. green 범위 [20,92], baseline_move_box=False(자유).

usage: python rl_leader/probe_green_price.py [warm]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv
from src.controllers.leader import LeaderAction

SIGNALS = ["A", "B", "C", "D", "F"]
SCEN = "sweet_190_skew15_w60"
NP0, NUF0 = 1161.0, 5254.0
WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def solve_green(env, prices, trust=None):
    f = env.follower
    prev = env.previous
    ref = {s: float(prev.green_times.get(f"{s}_p1", 56.0)) for s in SIGNALS}
    f.signal_marginal_price = prices
    f.signal_marginal_price_ref = ref
    f.signal_marginal_price_trust_sec = trust
    try:
        ctrl = f.solve(env.sim.state.copy(), LeaderAction(NP0, NUF0),
                       env._forecast(), prev).control
    finally:
        f.signal_marginal_price = None
    g = {s: float(ctrl.green_times.get(f"{s}_p1", float("nan"))) for s in SIGNALS}
    return g, ref


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    env.reset()
    a = env.budget_to_action(NP0, NUF0)
    while env.step_idx < WARM:
        env.step(a)
    base, ref = solve_green(env, None)
    print(f"=== {SCEN} step{env.step_idx}, budget 고정 ===")
    print(f"  기준 ref(직전 green): " + " ".join(f"{s}={ref[s]:5.1f}" for s in SIGNALS))
    print(f"  가격無 green        : " + " ".join(f"{s}={base[s]:5.1f}" for s in SIGNALS))
    print(f"  합계 {sum(base.values()):.1f}s (cycle {120.0*len(SIGNALS):.0f}s 중)\n")

    print("--- [1] 균일 가격 스윕 (전 신호 동일 g) ---")
    print(f"{'g':>8} | " + " ".join(f"{s:>6}" for s in SIGNALS) + " |    합계")
    for g in [-0.5, -0.1, -0.02, 0.0, 0.02, 0.1, 0.5]:
        gr, _ = solve_green(env, {s: g for s in SIGNALS})
        print(f"{g:8.2f} | " + " ".join(f"{gr[s]:6.1f}" for s in SIGNALS)
              + f" | {sum(gr.values()):7.1f}")

    print("\n--- [2] per-signal 차등 (A만 +0.1, 나머지 0) ---")
    gr, _ = solve_green(env, {"A": 0.1, "B": 0.0, "C": 0.0, "D": 0.0, "F": 0.0})
    d = {s: gr[s] - base[s] for s in SIGNALS}
    print("  Δ: " + " ".join(f"{s}={d[s]:+6.1f}" for s in SIGNALS)
          + f" | 합계Δ={sum(d.values()):+.1f}")

    print("\n--- [3] trust region 효과 (균일 +0.1) ---")
    for tr in [None, 6.0, 18.0]:
        gr, _ = solve_green(env, {s: 0.1 for s in SIGNALS}, trust=tr)
        mv = max(abs(gr[s] - ref[s]) for s in SIGNALS)
        print(f"  trust={str(tr):>5} → " + " ".join(f"{s}={gr[s]:5.1f}" for s in SIGNALS)
              + f" | 최대이동 {mv:.1f}s")

    print("\n판정: 중간 g에서 green이 점진 이동하면 학습 가능. "
          "모든 g에서 20/92 극단이면 bang-bang이라 액션 해상도가 없다.")


if __name__ == "__main__":
    main()
