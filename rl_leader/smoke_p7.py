# Phase 7 스모크(2026-08-03) — 램프 가격 + urban green 가격 11차원
"""게이트: (1) 차원 11 (2) 총량 하드 보존 (3) 램프 가격이 배분만 (4) green 가격이 green을 움직임
        (5) green 0이면 가격 없는 것과 동일(중립성) (6) 스케줄러 hold

usage: python rl_leader/smoke_p7.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS, SIGNALS, GREEN_PRICE_MAG
from rl_leader import collect_parallel as CP

SCEN = "sweet_190_skew15_w60"
NP0, NUF0 = 1161.0, 5254.0
fail = 0

env = RLLeaderEnv(scenario_name=SCEN, price_action=True, omega_action=False, green_action=True)
print("=== [1] 구성 ===", flush=True)
print(f"  action_dim={env.action_dim} obs_dim={env.obs_dim} "
      f"price_mag={env.price_mag} green_mag={env.green_mag} "
      f"split={env.follower.metering_price_split}", flush=True)
if env.action_dim != 11:
    print("  X action_dim이 11이 아니다"); fail += 1
else:
    print("  O", flush=True)

env.reset()
for _ in range(10):
    env.step(env.compose_action(NP0, NUF0, [0.0] * 4, None, [0.0] * 5))

print("=== [2][3] 총량 하드 + 램프 가격은 배분만 ===", flush=True)
tots = []
for g in [-env.price_mag, 0.0, env.price_mag]:
    _, _, _, info = env.step(env.compose_action(NP0, NUF0, [g, -g, 0.0, 0.0], None, [0.0] * 5))
    rel = info["ramp_release"]
    tots.append(sum(rel.values()))
    print(f"  ramp p(D_W)={g:+7.1f} → D_W={rel['R_D_W']:6.0f} F_W={rel['R_F_W']:6.0f} "
          f"| 총량={sum(rel.values()):6.0f}", flush=True)
if max(tots) - min(tots) > 60.0:
    print("  X 램프 가격이 총량을 바꾼다"); fail += 1
else:
    print("  O 총량 불변", flush=True)

print("=== [4] green 가격이 green을 움직이나 ===", flush=True)
base = None
moved = 0
for gp in [0.0, GREEN_PRICE_MAG, -GREEN_PRICE_MAG]:
    _, _, _, info = env.step(env.compose_action(NP0, NUF0, [0.0] * 4, None, [gp] * 5))
    gt = info["green_times"]
    if base is None:
        base = dict(gt)
    d = {s: gt[s] - base[s] for s in SIGNALS}
    nz = sum(1 for s in SIGNALS if abs(d[s]) > 0.5)
    moved = max(moved, nz)
    print(f"  green g={gp:+5.2f} → " + " ".join(f"{s}={gt[s]:5.1f}" for s in SIGNALS)
          + f" | 움직인 신호 {nz}개, 합계={sum(gt.values()):6.1f}", flush=True)
if moved == 0:
    print("  X green 가격이 아무것도 안 움직인다"); fail += 1
else:
    print(f"  O 최대 {moved}개 신호 반응", flush=True)

print("=== [5] green 0 중립성 (가격 채널 휴면과 동일해야) ===", flush=True)
envA = RLLeaderEnv(scenario_name=SCEN, price_action=True, green_action=False)
envB = RLLeaderEnv(scenario_name=SCEN, price_action=True, green_action=True)
envA.reset(); envB.reset()
mx = 0.0
for _ in range(4):
    _, _, _, ia = envA.step(envA.compose_action(NP0, NUF0, [0.0] * 4))
    _, _, _, ib = envB.step(envB.compose_action(NP0, NUF0, [0.0] * 4, None, [0.0] * 5))
    mx = max(mx, abs(ia["step_ttt"] - ib["step_ttt"]))
print(f"  4스텝 max|Δstep_ttt| = {mx:.3e}", flush=True)
if mx > 1e-5:
    print("  X green=0인데 궤적이 다르다"); fail += 1
else:
    print("  O 중립", flush=True)

print("=== [6] 스케줄러 ===", flush=True)
rng = np.random.default_rng(0)
for gm in ["zero", "uniform", "diff"]:
    sch = CP.make_green_schedule(gm, rng, 75, GREEN_PRICE_MAG)
    runs, cur = [], 1
    for i in range(1, len(sch)):
        cur = cur + 1 if np.allclose(sch[i], sch[i - 1]) else 1
        if not np.allclose(sch[i], sch[i - 1]):
            runs.append(cur)
    runs.append(cur)
    print(f"  {gm:8} shape={sch.shape} 범위 {sch.min():+.2f}~{sch.max():+.2f} "
          f"유지평균={np.mean(runs):.1f}", flush=True)

print(f"\nRESULT: {'ALL PASS' if fail == 0 else str(fail) + ' FAILURES'}", flush=True)
sys.exit(0 if fail == 0 else 1)
