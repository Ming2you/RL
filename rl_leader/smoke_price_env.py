# Phase 4 스모크(2026-07-31) — 수집 발사 전 필수 게이트 3종
"""(1) zero-price 중립성: price_action=True + 가격 0 이 budget-only와 동일 궤적인가
   (주입(ref/trust 세팅)이 solve 자체를 바꾸면 기존 27.7k와 MDP가 달라져 위험).
(2) 가격 응답: +500(R_D_W)이 merge 내 재분배(sister 램프 반대부호)를 일으키는가(probe §2.3 재현).
(3) 6차원 랜덤 + 음수 N_P 스텝 완주.

usage: python rl_leader/smoke_price_env.py
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, RAMPS

SCEN = "sweet_170_incident_w60"
NP0, NUF0 = 1161.0, 5254.0
K = 8
fail = 0

print("=== [1] zero-price 중립성 (K=%d steps) ===" % K, flush=True)
env_a = RLLeaderEnv(scenario_name=SCEN)                       # budget-only
env_b = RLLeaderEnv(scenario_name=SCEN, price_action=True)    # price, 가격 0
assert env_a.obs_dim == 13 and env_a.action_dim == 2, (env_a.obs_dim, env_a.action_dim)
assert env_b.obs_dim == 23 and env_b.action_dim == 6, (env_b.obs_dim, env_b.action_dim)
oa, ob = env_a.reset(), env_b.reset()
d0 = float(np.abs(ob[:13] - oa).max())
print(f"  obs_dim a/b = {env_a.obs_dim}/{env_b.obs_dim}  reset obs[:13] max|Δ| = {d0:.2e}", flush=True)
if d0 > 1e-6:
    print("  ✗ reset 관측 불일치"); fail += 1
max_dttt, max_dnuf, max_dobs = 0.0, 0.0, 0.0
for i in range(K):
    aa = env_a.budget_to_action(NP0, NUF0)
    ab = env_b.compose_action(NP0, NUF0, None)     # 가격 0
    oa, ra, da, ia = env_a.step(aa)
    ob, rb, db, ib = env_b.step(ab)
    max_dttt = max(max_dttt, abs(ia["step_ttt"] - ib["step_ttt"]))
    max_dnuf = max(max_dnuf, abs(ia["N_UF"] - ib["N_UF"]))
    max_dobs = max(max_dobs, float(np.abs(ob[:13] - oa).max()))
print(f"  {K}스텝 max|Δ|: step_ttt={max_dttt:.3e}  N_UF={max_dnuf:.3e}  obs[:13]={max_dobs:.3e}", flush=True)
if max(max_dttt, max_dnuf, max_dobs) > 1e-5:
    print("  ✗ zero-price가 budget-only와 다르게 굴러감 — 주입 자체가 dynamics를 바꿈"); fail += 1
else:
    print("  ✓ 중립성 성립 (기존 데이터 분포와 앵커 일치)", flush=True)

print("=== [2] 가격 응답 (probe 재현: R_D_W +500) ===", flush=True)
env_c = RLLeaderEnv(scenario_name=SCEN, price_action=True)
env_c.reset()
for i in range(K):    # env_b와 동일 상태로 전진(위에서 env_b도 K스텝 소진)
    env_c.step(env_c.compose_action(NP0, NUF0, None))
# 같은 상태에서: env_b는 가격 0, env_c는 R_D_W=+500
_, _, _, ib = env_b.step(env_b.compose_action(NP0, NUF0, None))
_, _, _, ic = env_c.step(env_c.compose_action(NP0, NUF0, [500.0, 0.0, 0.0, 0.0]))
rel0, rel1 = ib["ramp_release"], ic["ramp_release"]
d = {rp: rel1[rp] - rel0[rp] for rp in RAMPS}
print("  Δrelease: " + "  ".join(f"{rp}={d[rp]:+.1f}" for rp in RAMPS), flush=True)
tot = sum(d.values())
if abs(d["R_D_W"]) < 1.0:
    print("  ✗ 자기응답 없음 — 가격이 impotent (이 상태에선 이상)"); fail += 1
else:
    sis_ok = d["R_D_W"] * d["R_F_W"] <= 0
    print(f"  ✓ 자기응답 {d['R_D_W']:+.1f}, sister(R_F_W) {d['R_F_W']:+.1f} "
          f"({'반대부호=재분배' if sis_ok else '동부호?'}), 총합Δ={tot:+.1f} (0 근처=총량보존)", flush=True)

print("=== [3] 6차원 랜덤 + 음수 N_P ===", flush=True)
rng = np.random.default_rng(0)
ok = True
for i in range(3):
    act = rng.uniform(-1, 1, size=6)
    _, r, _, info = env_c.step(act)
    print(f"  step: N_P={info['N_P']:+7.1f} N_UF={info['N_UF']:6.1f} "
          f"prices=[{','.join(f'{info['prices'][rp]:+.0f}' for rp in RAMPS)}] r={r:.1f}", flush=True)
_, _, _, info = env_c.step(env_c.compose_action(-500.0, NUF0, None))
print(f"  음수 N_P: 요청 -500 → info N_P={info['N_P']:+.1f}", flush=True)
if abs(info["N_P"] - (-500.0)) > 1e-6:
    print("  ✗ N_P 매핑 오류"); fail += 1

print(f"\nRESULT: {'ALL PASS' if fail == 0 else f'{fail} FAILURES'}", flush=True)
sys.exit(0 if fail == 0 else 1)
