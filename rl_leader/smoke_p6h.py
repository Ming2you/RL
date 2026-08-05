# Phase 6 스모크(2026-08-02) — 하드 budget + ω창정규화 + 가격 배분
"""검증: 총량이 항상 N_UF로 보존되면서, ω가 merge 간을·가격이 merge 내를 재분배하나.
Phase 5 실패(총량 통제 상실)의 정확한 반대 조건을 확인한다.

usage: python rl_leader/smoke_p6h.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS, omega_window
from rl_leader import collect_parallel as CP

SCEN = "sweet_190_skew15_w60"
NP0 = 1161.0
fail = 0

env = RLLeaderEnv(scenario_name=SCEN, price_action=True, price_level=False, omega_action=True)
print(f"=== [1] 구성 ===", flush=True)
print(f"  action_dim={env.action_dim} obs_dim={env.obs_dim} price_mag={env.price_mag} "
      f"split={env.follower.metering_price_split} cap_link={env.cap_link:.0f}", flush=True)
if env.action_dim != 7 or env.follower.metering_price_split is not True:
    print("  X 구성 오류"); fail += 1
else:
    print("  O", flush=True)

print("=== [2] ω 가용 창 (총량 보존 조건) ===", flush=True)
for nuf in [3500.0, 4500.0, 5254.0, 6000.0]:
    lo, hi = omega_window(nuf, env.cap_link)
    print(f"  N_UF={nuf:5.0f} → ω ∈ [{lo:.3f}, {hi:.3f}]  폭={hi-lo:.3f}", flush=True)

env.reset()
for _ in range(10):
    env.step(env.compose_action(NP0, 5254.0, [0.0] * 4, 0.5))

print("=== [3] 총량 보존 + ω 재분배 (N_UF=5254) ===", flush=True)
tots, asym = [], []
for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
    _, _, _, info = env.step(env.compose_action(NP0, 5254.0, [0.0] * 4, u))
    rel = info["ramp_release"]
    W, E = rel["R_D_W"] + rel["R_F_W"], rel["R_D_E"] + rel["R_F_E"]
    tots.append(W + E); asym.append(W - E)
    print(f"  u={u:.2f} (ω={info['omega_w']:.3f}) → W={W:6.0f} E={E:6.0f} "
          f"총량={W+E:6.0f} (N_UF 대비 {W+E-5254:+.0f})  W−E={W-E:+7.0f}", flush=True)
if max(abs(t - 5254.0) for t in tots) > 60.0:
    print("  X 총량이 보존되지 않는다"); fail += 1
elif max(asym) - min(asym) < 200.0:
    print("  X ω가 재분배를 못 만든다"); fail += 1
else:
    print(f"  O 총량 편차 max={max(abs(t-5254.0) for t in tots):.0f}, "
          f"W−E 폭={max(asym)-min(asym):.0f}", flush=True)

print("=== [4] 가격 merge 내부 재분배 (총량 불변이어야) ===", flush=True)
t2 = []
for g in [-env.price_mag, 0.0, env.price_mag]:
    _, _, _, info = env.step(env.compose_action(NP0, 5254.0, [g, -g, 0.0, 0.0], 0.5))
    rel = info["ramp_release"]
    t2.append(sum(rel.values()))
    print(f"  p(D_W)={g:+6.0f} → D_W={rel['R_D_W']:6.0f} F_W={rel['R_F_W']:6.0f} "
          f"| 총량={sum(rel.values()):6.0f}", flush=True)
if max(t2) - min(t2) > 60.0:
    print("  X 가격이 총량을 바꾼다(하드 budget 위반)"); fail += 1
else:
    print("  O 가격은 배분만 바꾼다", flush=True)

print("=== [5] 스케줄러 ω 위치 ===", flush=True)
om = CP.make_omega_schedule(np.random.default_rng(0), 75)
print(f"  범위 {om.min():.2f}~{om.max():.2f} (정규화 위치여야 0~1)", flush=True)
if om.min() < 0.0 or om.max() > 1.0:
    print("  X 범위 이탈"); fail += 1

print(f"\nRESULT: {'ALL PASS' if fail == 0 else str(fail) + ' FAILURES'}", flush=True)
sys.exit(0 if fail == 0 else 1)
