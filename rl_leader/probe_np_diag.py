# N_P dual 루프가 어디서 끊기나(2026-08-05)
"""_commit_np_dual을 넣었는데 λ_P가 여전히 0이다. 끊기는 지점을 특정한다:
  1. follower의 dual 게이트 플래그 상태 (use_dual_np / np_price_enabled)
  2. solve가 실제로 leader를 받았나 (dual_active)
  3. diagnostics에 np 관련 키가 나오나 (projected_target / sum_nin / lambda_next)
  4. _np_last_real_q / _np_prev_accum 이 채워지나

usage: python rl_leader/probe_np_diag.py [warm]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv
from src.controllers.leader import LeaderAction

WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 12
env = RLLeaderEnv(scenario_name="sweet_190_skew15_w60", np_dual=True)
f = env.follower
print("[1] follower 게이트 플래그")
for k in ("use_dual_np", "np_price_enabled", "lambda_np_cap", "_lambda_P"):
    print(f"    {k:22} = {getattr(f, k, '없음')}")
print(f"    solve 시그니처       = {f.solve.__doc__ is not None and 'doc있음' or ''}")
import inspect as _i
print(f"    solve params         = {list(_i.signature(f.solve).parameters)}")

env.reset()
a = env.budget_to_action(1161.0, 5254.0)
for _ in range(WARM):
    env.step(a)

print(f"\n[2] {WARM}스텝 후 follower 내부 상태")
for k in ("_lambda_P", "_np_corrector_pending", "_np_last_real_q",
          "_np_prev_accum", "_np_last_sum_nin", "_np_bias_ratio"):
    print(f"    {k:24} = {getattr(f, k, '없음')}")

print("\n[3] solve 1회의 diagnostics (np/lambda 관련)")
ctrl = f.solve(env.sim.state.copy(), LeaderAction(0.0, 5254.0), env._forecast(),
               env.previous).control
d = getattr(ctrl, "diagnostics", None) or {}
hits = {k: v for k, v in d.items() if "np" in k.lower() or "lambda" in k.lower()}
if hits:
    for k, v in sorted(hits.items()):
        print(f"    {k:44} = {v}")
else:
    print("    ★np/lambda 관련 키가 하나도 없다 — dual 블록 자체가 안 돈다")

print("\n[4] mpc 설정")
m = env.cfg.mpc
for k in ("np_candidate_lambda", "np_primal_dual_iters", "np_bias_correction",
          "np_dual_deadband_frac", "np_deadband_violation_override"):
    print(f"    {k:32} = {getattr(m, k, '없음')}")
print(f"    leader.N_P_crit_veh              = {getattr(env.cfg.leader, 'N_P_crit_veh', '없음')}")
