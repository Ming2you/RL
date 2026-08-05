# far를 관측에서 근사할 수 있나(2026-08-04) — 재수집 회피 게이트
"""probe_hinge_far 결과: corr(hinge, 근사)=0.97이라 진짜 hinge 로깅은 무가치.
그러나 corr(far, 근사)=0.69로 far는 다른 정보를 담는다.

far를 쓰려면 재수집(14h)이 필요하다 — 단, obs 13차원의 선형/단순 조합으로 far가
잘 설명되면 재수집 없이 지금 데이터로 바로 Φ=far근사를 시험할 수 있다.
여러 시나리오·수요에서 모아 회귀한다.

usage: python rl_leader/probe_far_approx.py [n_ep] [steps]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, make_random_scenario

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
NAMES = ["uveh", "fveh", "rampq", "originq", "rho_mean", "rho_max", "over",
         "v_mean", "inflow", "ramp_arr", "prevNP", "prevNUF", "phase"]

rng = np.random.default_rng(7)
X, Y = [], []
for ep in range(N_EP):
    scen = make_random_scenario(rng)
    env = RLLeaderEnv(scenario_dict=scen)
    env.reset()
    stack = env._stack
    n_p = float(rng.uniform(400, 1800))
    n_uf = float(rng.uniform(4000, 6000))
    a = env.budget_to_action(n_p, n_uf)
    for k in range(STEPS):
        o = env._observe()
        try:
            far = float(stack._mfd_far_cost_to_go(env.sim.state))
        except Exception:
            far = float("nan")
        if np.isfinite(far):
            X.append(o.tolist()); Y.append(far)
        env.step(a)
    print(f"  ep{ep} d={scen.get('urban_scale',0):.2f} 수집 {len(Y)}", flush=True)

X = np.array(X, float); Y = np.array(Y, float)
print(f"\n표본 {len(Y)}  far range {Y.min():.1f}~{Y.max():.1f} sd {Y.std():.1f}")

print("\n[개별 상관]")
for j, nm in enumerate(NAMES):
    if X[:, j].std() > 1e-9:
        print(f"  {nm:9} corr={np.corrcoef(X[:,j], Y)[0,1]:+.3f}")

# 선형 회귀 (표준화)
Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
A = np.column_stack([np.ones(len(Y)), Xs])
beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
pred = A @ beta
r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
print(f"\n[선형회귀 13차원] R² = {r2:.4f}")
order = np.argsort(-np.abs(beta[1:]))
for j in order[:6]:
    print(f"  {NAMES[j]:9} beta={beta[j+1]:+8.2f}")

# 간단 후보들
cands = {
    "fveh": X[:, 1],
    "fveh+rampq": X[:, 1] + X[:, 2],
    "over*rho_max": X[:, 6] * X[:, 5],
    "fveh*rho_mean": X[:, 1] * X[:, 4],
    "accum^2": X[:, 1] ** 2,
}
print("\n[단순 후보와의 상관]")
for nm, v in cands.items():
    if v.std() > 1e-9:
        print(f"  {nm:14} corr={np.corrcoef(v, Y)[0,1]:+.3f}")
print("\n판정: R²가 높고(>0.9) 단순 후보 상관이 높으면 재수집 없이 far근사로 시험 가능.")
