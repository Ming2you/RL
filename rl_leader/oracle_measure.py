# 오라클 budget 측정(2026-07-23) — 위상별 N_UF 스케줄로 foresight 천장 측정
"""sweep(고정 budget) 다음 단계. 리더가 근시안이 아니라 foresight가 있으면
budget 정책이 P-Stack을 얼마나 넘나 = 오라클 천장.

설계: 고수요 창(plateau+rampdown)엔 tight, 저수요 구간엔 loose(불필요한 ramp큐 방지).
constant sweep 최선과 비교해 위상 적응이 추가 이득을 주는지 본다.
N_P는 P-Stack peak 평균(1161)으로 고정, N_UF만 위상별로 변주.

usage: python rl_leader/oracle_measure.py [scenario]
"""
from __future__ import annotations
import sys, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv

BASE = {"P-Stack": 6379, "PFO": 6299, "P-CENT": 5757, "NC": 6882}  # skew windowed 기준선


def rollout_sched(scen, n_p, nuf_fn):
    env = RLLeaderEnv(scenario_name=scen)
    env.reset()
    warm = float(env.sim.total_ttt)
    done = False
    while not done:
        t = env.step_idx * env.dt
        _, _, done, info = env.step(env.budget_to_action(n_p, nuf_fn(t)))
    return float(env.sim.total_ttt) - warm


def best_const(log: Path):
    best = None
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s*(\d{3,})\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            nuf, ttt = float(m.group(1)), float(m.group(2))
            if best is None or ttt < best[1]:
                best = (nuf, ttt)
    return best


def main(scen="sweet_190_skew15_w60"):
    NP = 1161.0
    T_LO, T_HI = 1260.0, 5220.0          # 고수요 창(plateau+rampdown)
    LOOSE = 6000.0
    bc = best_const(ROOT / "rl_leader" / "logs" / "ceiling_final.log")
    print(f"=== 오라클 측정(위상별 N_UF): {scen} ===", flush=True)
    print(f"기준선: " + " | ".join(f"{k} {v}" for k, v in BASE.items()), flush=True)
    if bc:
        print(f"sweep 최선 상수: N_UF={bc[0]:.0f} → {bc[1]:.1f}", flush=True)
    print(f"\n위상: t∈[{T_LO:.0f},{T_HI:.0f}] = tight, 그 외 = loose({LOOSE:.0f})", flush=True)
    print(f"{'N_UF_tight':>10} {'windowed_TTT':>13} {'vs P-Stack':>11}", flush=True)
    results = []
    for nuf_t in [4500, 3500, 2500, 1500, 800]:
        fn = (lambda v: (lambda t: v if T_LO <= t < T_HI else LOOSE))(nuf_t)
        ttt = rollout_sched(scen, NP, fn)
        print(f"{nuf_t:10.0f} {ttt:13.1f} {ttt-BASE['P-Stack']:+11.1f}", flush=True)
        results.append((nuf_t, ttt))
    best = min(results, key=lambda x: x[1])
    print(f"\n오라클(위상) 최선: N_UF_tight={best[0]:.0f} → {best[1]:.1f}", flush=True)
    if bc:
        print(f"  vs 상수최선 {bc[1]:.1f} (Δ={best[1]-bc[1]:+.1f}) | vs P-Stack 6379 (Δ={best[1]-6379:+.1f})", flush=True)
    print(f"  → PFO 6299 / P-CENT 5757 대비 위치로 스칼라-budget 천장 판정", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sweet_190_skew15_w60")
