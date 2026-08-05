# hinge/far 항 호출 probe(2026-08-04) — ① 재수집 전 게이트
"""진짜 hinge/far를 수집 때 로깅하려면 먼저 확인해야 한다:
  (a) 호출 규약이 맞나 (전례: _feasible_nuf_capacity가 0.0을 조용히 반환)
  (b) 값이 상태에 따라 실제로 변하나 (상수면 Φ로 무용)
  (c) 스텝당 비용이 얼마나 되나 (매 스텝 호출 → 14시간 수집에 곱해진다)
  (d) 현재 쓰는 근사(obs[6]×max(0,obs[5]-1))와 얼마나 상관되나
      — 상관이 매우 높으면 진짜 항을 써도 이득이 작다는 뜻이라 14시간을 아낀다.

usage: python rl_leader/probe_hinge_far.py [steps]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SCEN = "sweet_190_skew15_w60"
NP0, NUF0 = 1161.0, 5254.0


def main():
    env = RLLeaderEnv(scenario_name=SCEN)
    env.reset()
    stack = env._stack

    from src.controllers.stackelberg_mpc import leader_hinge_cost
    print("import leader_hinge_cost: OK", flush=True)
    print("has _mfd_far_cost_to_go:", hasattr(stack, "_mfd_far_cost_to_go"), flush=True)

    a = env.budget_to_action(NP0, NUF0)
    rows, t_h, t_f = [], 0.0, 0.0
    for k in range(STEPS):
        s = env.sim.state
        fc = env._forecast()
        t0 = time.time()
        try:
            hinge = float(leader_hinge_cost(env.cfg, [s], fc, force=True))
        except Exception as e:
            print(f"  hinge 호출 실패: {type(e).__name__}: {e}"); hinge = float("nan")
        t_h += time.time() - t0
        t0 = time.time()
        try:
            far = float(stack._mfd_far_cost_to_go(s))
        except Exception as e:
            print(f"  far 호출 실패: {type(e).__name__}: {e}"); far = float("nan")
        t_f += time.time() - t0
        o = env._observe()
        proxy = float(o[6]) * max(0.0, float(o[5]) - 1.0)
        rows.append((k, hinge, far, proxy, float(o[6]), float(o[5])))
        env.step(a)

    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    names = ["hinge", "far", "proxy", "over/10", "rho_max"]
    print(f"\n{'스텝':>5} " + " ".join(f"{n:>12}" for n in names), flush=True)
    for i in range(0, len(rows), max(1, len(rows) // 12)):
        print(f"{rows[i][0]:>5} " + " ".join(f"{arr[i,j]:12.4f}" for j in range(5)), flush=True)

    print(f"\n비용: hinge {1000*t_h/STEPS:.2f} ms/step   far {1000*t_f/STEPS:.2f} ms/step "
          f"(env 스텝은 ~15000 ms이므로 무시 가능해야 함)", flush=True)
    for j, n in enumerate(names):
        v = arr[:, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            print(f"  {n:8}: 전부 NaN"); continue
        print(f"  {n:8}: min={v.min():10.4f} max={v.max():10.4f} sd={v.std():10.4f}"
              f"  {'★상수 — Φ로 무용' if v.std() < 1e-9 else ''}", flush=True)

    ok = np.isfinite(arr).all(axis=0)
    if ok[0] and ok[2] and arr[:, 0].std() > 1e-9 and arr[:, 2].std() > 1e-9:
        print(f"\n  corr(hinge, proxy) = {np.corrcoef(arr[:,0], arr[:,2])[0,1]:+.4f}", flush=True)
    if ok[1] and ok[2] and arr[:, 1].std() > 1e-9 and arr[:, 2].std() > 1e-9:
        print(f"  corr(far,   proxy) = {np.corrcoef(arr[:,1], arr[:,2])[0,1]:+.4f}", flush=True)
    if ok[0] and ok[1] and arr[:, 0].std() > 1e-9 and arr[:, 1].std() > 1e-9:
        print(f"  corr(hinge, far)   = {np.corrcoef(arr[:,0], arr[:,1])[0,1]:+.4f}", flush=True)
    print("\n판정: 값이 변하고 proxy와의 상관이 낮으면(<0.9) 진짜 항을 로깅할 가치가 있다.", flush=True)


if __name__ == "__main__":
    main()
