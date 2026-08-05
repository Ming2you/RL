# split=False 다스텝 안정성 검증(2026-08-01) — 레짐 플래핑/과소방류 병리가 RL 경로에도 나오나
"""metering_price_split=False를 켜면 가격이 총 방류량을 정한다(probe_price_level.py 확인:
변동폭 1800 veh/h). 그러나 이 모드는 P-Stack leader 탐색에서 병리가 관측돼 꺼졌다
(wu_faithful_follower.py:3028-3032 — incumbent↔후보 교대 커밋 → Σmeter TV 1.74배, 과소방류 −800).

RL leader는 탐색을 하지 않으므로 그 병리가 없을 것으로 예상되나, 10시간 수집 전에 실측한다.
동일 시나리오를 split ON/OFF로 각각 N스텝 굴려 (a) Σmeter 총변동(TV) (b) 누적 TTT (c) 스텝 소요를 비교.

usage: python rl_leader/probe_price_stability.py [steps] [price]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS

SCEN = "sweet_190_skew15_w60"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
PRICE = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
NP0, NUF0 = 1161.0, 5254.0


def run(split_mode, price):
    env = RLLeaderEnv(scenario_name=SCEN, price_action=True, price_mag=500.0)
    env.follower.metering_price_split = split_mode
    env.reset()
    a = env.compose_action(NP0, NUF0, [price] * 4)
    tot, ttt, t0 = [], 0.0, time.time()
    for _ in range(STEPS):
        _, r, done, info = env.step(a)
        tot.append(sum(info["ramp_release"].values()))
        ttt += info["step_ttt"]
        if done:
            break
    tot = np.array(tot)
    tv = float(np.abs(np.diff(tot)).sum())          # 총변동
    return dict(tot=tot, tv=tv, ttt=ttt, sec=time.time() - t0,
                mean=float(tot.mean()), std=float(tot.std()))


print(f"=== split ON/OFF 안정성 비교: {SCEN}, {STEPS}스텝, 균일가격 g={PRICE:+.0f} ===", flush=True)
res = {}
for name, split in [("split=True(현행)", True), ("split=False(레벨가격)", False)]:
    r = run(split, PRICE)
    res[name] = r
    print(f"\n--- {name} ---", flush=True)
    print(f"  Σmeter mean={r['mean']:7.1f} std={r['std']:6.1f} "
          f"min={r['tot'].min():.0f} max={r['tot'].max():.0f}", flush=True)
    print(f"  총변동(TV)={r['tv']:8.1f}  (스텝당 {r['tv']/max(len(r['tot'])-1,1):.1f})", flush=True)
    print(f"  누적 step TTT={r['ttt']:.1f}   소요={r['sec']:.0f}s", flush=True)
    print(f"  궤적: {np.round(r['tot'][:12], 0).tolist()}", flush=True)

a_, b_ = res["split=True(현행)"], res["split=False(레벨가격)"]
print("\n판정:", flush=True)
tv_ratio = b_["tv"] / max(a_["tv"], 1e-9)
print(f"  TV 비율(OFF/ON) = {tv_ratio:.2f}  (병리 기준 1.74배 이상이면 플래핑 의심)", flush=True)
print(f"  총량 차이 = {b_['mean']-a_['mean']:+.1f} veh/h  (과소방류 병리는 −800 규모였음)", flush=True)
print(f"  TTT 차이 = {b_['ttt']-a_['ttt']:+.1f} (음수면 레벨가격이 유리)", flush=True)
if tv_ratio < 1.74 and abs(b_["mean"] - a_["mean"]) < 800:
    print("  → 병리 미재현. RL 경로에서 split=False 사용 가능.", flush=True)
else:
    print("  → 병리 징후 있음. 수집 전 추가 조사 필요.", flush=True)
