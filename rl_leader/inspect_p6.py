# Phase 4 데이터 점검(2026-07-31) — 체이닝 가드용. 마지막 줄 SAMPLES=<n>을 스크립트가 읽는다.
"""수집된 data/rl_dataset_p6/를 요약: 샘플/에피소드/abort율, 차원, 가격·budget 커버리지.
usage: python rl_leader/inspect_p6.py [glob]
"""
from __future__ import annotations
import glob as globmod
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "rl_dataset_p6" / "w*.npz")

O, A, P, B, D = [], [], [], [], []
meta, bad = [], []
files = sorted(globmod.glob(PAT))
for f in files:
    try:
        d = np.load(f, allow_pickle=True)
        if int(d["obs"].shape[0]) == 0:
            continue
        O.append(d["obs"]); A.append(d["act"]); D.append(d["done"])
        if "prices" in d.files:
            P.append(d["prices"])
        if "budget" in d.files:
            B.append(d["budget"])
        meta += list(d["meta"])
    except Exception as e:
        bad.append(f"{Path(f).name}: {type(e).__name__}")

if not O:
    print(f"files={len(files)} readable=0 bad={bad}")
    print("SAMPLES=0")
    sys.exit(0)

obs = np.concatenate(O); act = np.concatenate(A); done = np.concatenate(D)
n = obs.shape[0]
print(f"files={len(files)} readable={len(O)} bad={bad if bad else 'none'}")
print(f"obs_dim={obs.shape[1]} act_dim={act.shape[1]} done_flags={int(done.sum())}")
if meta:
    ab = sum(1 for m in meta if m.get("aborted"))
    from collections import Counter
    print(f"episodes={len(meta)} aborted={ab} ({100*ab/len(meta):.0f}%)")
    print("budget_modes=" + str(dict(Counter(m.get("mode") for m in meta))))
    print("price_modes=" + str(dict(Counter(m.get("price_mode") for m in meta))))
    print("stressors=" + str(dict(Counter(m.get("stressor") for m in meta))))
if B:
    b = np.concatenate(B)
    print(f"N_P  range=[{b[:,0].min():.0f}, {b[:,0].max():.0f}] neg_frac={100*(b[:,0]<0).mean():.1f}%")
    print(f"N_UF range=[{b[:,1].min():.0f}, {b[:,1].max():.0f}]")
if P:
    p = np.concatenate(P)
    nz = (np.abs(p).max(1) > 1e-9)
    print(f"prices range=[{p.min():.0f}, {p.max():.0f}] nonzero_frac={100*nz.mean():.1f}% "
          f"per_ramp_std={np.round(p.std(0), 0).tolist()}")
print(f"SAMPLES={n}")
