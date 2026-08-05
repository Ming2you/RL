# clamp의 에피소드 수준 가치(2026-08-04) — 신용할당 가설의 전제 검증
"""HANDOFF §12.9: 데이터의 한계 통계가 "더 풀수록 좋다"(corr(peak N_UF, step TTT) = −0.612)를
가리키는데, 이는 즉시 램프큐 감소만 보이고 지연된 breakdown 비용이 안 잡히기 때문이라고 해석했다.

그 해석이 맞다면 **에피소드 전체 TTT**로 보면 부호가 뒤집혀야 한다(조이는 게 이득).
뒤집히지 않으면 신용할당 문제가 아니라 **시뮬레이터가 실제로 "budget-only 리더에겐
조이는 게 손해"라고 답하는 것**이고, γ·n-step·shaping 어느 것도 소용없다.

수요·stressor가 TTT를 지배하므로 반드시 통제한다.

usage: python rl_leader/analyze_clamp_value.py [glob]
"""
from __future__ import annotations
import glob as globmod
import sys

import numpy as np

PAT = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\TRLAB\Desktop\찐찐막\RL\data\rl_dataset_p7\w*.npz"

rows = []
for f in sorted(globmod.glob(PAT)):
    d = np.load(f, allow_pickle=True)
    if d["obs"].shape[0] == 0:
        continue
    obs, ep, nuf = d["obs"], d["ep"], d["budget"][:, 1]
    rew, meta = d["rew"], list(d["meta"])
    phase = obs[:, 12]
    for m in meta:
        e = m["ep"]
        sel = ep == e
        if sel.sum() < 60 or m.get("aborted"):
            continue
        ph, v, r = phase[sel], nuf[sel], rew[sel]
        pk = (ph >= 0.10) & (ph <= 0.45)
        if pk.sum() < 10:
            continue
        inb = v[pk] < 5000.0                      # peak에서 '조인' 스텝
        best = cur = 0
        for x in inb:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        rows.append(dict(
            ttt=float(-r.sum()),                  # 에피소드 총 TTT(보상 = −step TTT)
            peak_nuf=float(v[pk].mean()),
            clamp_len=int(best),
            demand=float(m.get("demand", 0.0)),
            stressor=str(m.get("stressor", "none")),
            mode=str(m.get("mode", "")),
        ))

if not rows:
    print("에피소드 없음"); sys.exit(1)
ttt = np.array([r["ttt"] for r in rows])
pn = np.array([r["peak_nuf"] for r in rows])
cl = np.array([r["clamp_len"] for r in rows], float)
dm = np.array([r["demand"] for r in rows])
st = np.array([r["stressor"] for r in rows])
print(f"에피소드 {len(rows)}개  TTT {ttt.min():.0f}~{ttt.max():.0f}  "
      f"peak_N_UF {pn.min():.0f}~{pn.max():.0f}  clamp_len 0~{cl.max():.0f}")

# 통제 회귀: TTT ~ 1 + demand + demand^2 + stressor더미 + peak_nuf (+ clamp_len)
def fit(extra_cols, names):
    X = [np.ones_like(ttt), dm, dm ** 2,
         (st == "skew").astype(float), (st == "incident").astype(float)] + extra_cols
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, ttt, rcond=None)
    resid = ttt - X @ beta
    dof = max(len(ttt) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    base = ["const", "demand", "demand^2", "skew", "incident"]
    for i, nm in enumerate(base + names):
        print(f"    {nm:12} beta={beta[i]:+10.3f}  se={se[i]:8.3f}  t={beta[i]/max(se[i],1e-9):+6.2f}")
    return beta, se

print("\n[1] peak 평균 N_UF의 효과 (양수 beta = 많이 풀수록 TTT 증가 = 조이는 게 이득)")
fit([pn], ["peak_N_UF"])

print("\n[2] clamp 지속길이의 효과 (음수 beta = 오래 조일수록 TTT 감소 = 이득)")
fit([cl], ["clamp_len"])

print("\n[3] 둘 다")
fit([pn, cl], ["peak_N_UF", "clamp_len"])

# 층화 비교: 수요·stressor 매칭 안에서 clamp 상/하위 비교
print("\n[4] 층화 비교 (수요 3분위 × stressor 안에서 clamp_len 상위25% vs 하위25%)")
qs = np.quantile(dm, [1 / 3, 2 / 3])
for s in ["none", "skew", "incident"]:
    for qi, (lo, hi) in enumerate([(-1e9, qs[0]), (qs[0], qs[1]), (qs[1], 1e9)]):
        m = (st == s) & (dm >= lo) & (dm < hi)
        if m.sum() < 8:
            continue
        c = cl[m]
        hiq, loq = np.quantile(c, 0.75), np.quantile(c, 0.25)
        a, b = ttt[m][c >= hiq], ttt[m][c <= loq]
        if len(a) < 2 or len(b) < 2:
            continue
        print(f"    {s:9} d분위{qi+1}  n={m.sum():3d} | 긴clamp {a.mean():8.0f} (n={len(a)})  "
              f"짧은clamp {b.mean():8.0f} (n={len(b)})  차이 {a.mean()-b.mean():+8.0f}")

print("\n판정: [1] peak_N_UF beta가 유의하게 양수이고 [2] clamp_len이 음수면 "
      "→ 에피소드 수준에선 조이는 게 이득 = 신용할당 문제(수정 가치 있음).")
print("      부호가 반대거나 무의미하면 → 시뮬레이터가 '조이면 손해'라고 답하는 것 "
      "= 신용할당 수정으로 못 고침.")
