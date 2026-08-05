# Phase 5 스모크(2026-08-01) — price_level + omega 액션이 실제로 작동하나
"""수집 발사 전 게이트:
 (1) 차원: price_level+omega면 action_dim=7, obs_dim=23
 (2) ω 주입이 실제로 W/E 총방류 비대칭을 만드나 (Phase 4에선 항상 0.0이었다)
 (3) 가격이 총량을 움직이나 (split=False라야 함)
 (4) 스케줄러가 piecewise-constant를 만드나 (SATURATED 원인 수정 확인)

usage: python rl_leader/smoke_p5.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS, PRICE_LEVEL_MAG, OMEGA_LO, OMEGA_HI
from rl_leader import collect_parallel as CP

SCEN = "sweet_190_skew15_w60"
NP0, NUF0 = 1161.0, 5254.0
WARM = 10
fail = 0

print("=== [1] 차원 ===", flush=True)
env = RLLeaderEnv(scenario_name=SCEN, price_action=True, price_level=True, omega_action=True)
print(f"  action_dim={env.action_dim} obs_dim={env.obs_dim} price_mag={env.price_mag} "
      f"split={env.follower.metering_price_split}", flush=True)
if env.action_dim != 7 or env.obs_dim != 23 or env.follower.metering_price_split is not False:
    print("  X 차원/모드 오류"); fail += 1
else:
    print("  O", flush=True)

env.reset()
a0 = env.compose_action(NP0, NUF0, [0.0] * 4, 0.5)
for _ in range(WARM):
    env.step(a0)

print("=== [2] ω 주입 → W/E 비대칭 (Phase 4에선 항상 0.000) ===", flush=True)
rows = []
for w in [0.25, 0.40, 0.50, 0.60, 0.75]:
    _, _, _, info = env.step(env.compose_action(NP0, NUF0, [0.0] * 4, w))
    rel = info["ramp_release"]
    W = rel["R_D_W"] + rel["R_F_W"]
    E = rel["R_D_E"] + rel["R_F_E"]
    rows.append(W - E)
    print(f"  ω_W={w:.2f} → W={W:7.1f} E={E:7.1f}  W−E={W-E:+8.1f}  (info ω={info['omega_w']:.3f})",
          flush=True)
if max(abs(x) for x in rows) < 50.0:
    print("  X ω가 비대칭을 못 만든다"); fail += 1
else:
    print(f"  O 비대칭 범위 {min(rows):+.1f} ~ {max(rows):+.1f}", flush=True)

print("=== [3] 가격 → 총량 (split=False) ===", flush=True)
tots = []
for g in [-PRICE_LEVEL_MAG, 0.0, PRICE_LEVEL_MAG]:
    _, _, _, info = env.step(env.compose_action(NP0, NUF0, [g] * 4, 0.5))
    t = sum(info["ramp_release"].values())
    tots.append(t)
    print(f"  g={g:+7.0f} → 총량={t:7.1f}", flush=True)
if max(tots) - min(tots) < 100.0:
    print("  X 가격이 총량을 못 바꾼다(split이 안 꺼졌나?)"); fail += 1
else:
    print(f"  O 총량 변동폭 {max(tots)-min(tots):.1f} veh/h", flush=True)

print("=== [4] 스케줄러 piecewise-constant ===", flush=True)
rng = np.random.default_rng(0)
for mode in ["clamp", "hold_wide", "reactive"]:
    _, nuf = CP.make_budget_schedule(mode, rng, 75)
    runs, cur = [], 1
    for i in range(1, len(nuf)):
        if abs(nuf[i] - nuf[i - 1]) < 1e-9:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    print(f"  {mode:10} 유지길이 min={min(runs)} max={max(runs)} 평균={np.mean(runs):.1f} "
          f"| N_UF {nuf.min():.0f}~{nuf.max():.0f}", flush=True)
    if np.mean(runs) < 6.0:
        print(f"  X {mode} 유지가 너무 짧다"); fail += 1
pr = CP.make_price_schedule("hold", rng, 75, PRICE_LEVEL_MAG)
om = CP.make_omega_schedule(rng, 75)
print(f"  price(hold) 범위 {pr.min():.0f}~{pr.max():.0f} | omega 범위 {om.min():.2f}~{om.max():.2f}",
      flush=True)
if om.min() < OMEGA_LO - 1e-6 or om.max() > OMEGA_HI + 1e-6:
    print("  X omega 범위 이탈"); fail += 1

print(f"\nRESULT: {'ALL PASS' if fail == 0 else str(fail) + ' FAILURES'}", flush=True)
sys.exit(0 if fail == 0 else 1)
