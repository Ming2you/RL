# Phase 4 결과 판정(2026-07-31) — 체이닝 분기용. 마지막 줄 VERDICT=GOOD|BAD|INCOMPLETE.
"""eval 로그(logs/eval_<tag><seed>_<cell>.log)에서 windowed TTT를 긁어 budget-only 3시드
기준선과 비교한다. 판정 규칙을 코드에 박아 사후 해석 여지를 없앤다.

기준선(Phase 2, 27.7k budget-only 3시드):  skew 6324.8 ± 116.0 / inc 8196.0 ± 76.8
GOOD = 한 셀 이상에서 기준선을 그 셀 std보다 크게 개선 AND 다른 셀이 std보다 크게 악화되지 않음.

usage: python rl_leader/judge_p6.py [tag_prefix=p6s] [seeds=0,1,2]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = {"skew": (6324.8, 116.0), "inc": (8196.0, 76.8)}   # budget-only 3시드 mean, std
PSTACK = {"skew": 6378.9, "inc": 8386.2}
PCENT = {"skew": 5757.4, "inc": 8016.3}

tag_prefix = sys.argv[1] if len(sys.argv) > 1 else "p6s"
seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2"])]

res = {"skew": [], "inc": []}
trunc = 0
for s in seeds:
    for cell in ["skew", "inc"]:
        p = ROOT / "logs" / f"eval_{tag_prefix}{s}_{cell}.log"
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "시간가드 절단" in txt or "truncated" in txt.lower():
            trunc += 1
            continue
        m = re.search(r"windowed TTT = ([\d.]+)", txt)
        if m:
            res[cell].append(float(m.group(1)))

print(f"tag={tag_prefix} seeds={seeds} truncated={trunc}")
ok = True
deltas = {}
for cell in ["skew", "inc"]:
    v = np.array(res[cell])
    if len(v) == 0:
        print(f"  {cell}: 결과 없음")
        ok = False
        continue
    b, sd = BASE[cell]
    d = b - v.mean()      # 양수 = 개선
    deltas[cell] = d
    print(f"  {cell}: n={len(v)} mean={v.mean():.1f} std={v.std(ddof=1) if len(v) > 1 else 0:.1f} "
          f"range=[{v.min():.1f},{v.max():.1f}] | vs budget-only {b:.1f}: {-d:+.1f} "
          f"| vs P-Stack {PSTACK[cell]:.0f}: {v.mean()-PSTACK[cell]:+.1f} "
          f"| vs P-CENT {PCENT[cell]:.0f}: {v.mean()-PCENT[cell]:+.1f}")

if not ok or len(deltas) < 2:
    print("VERDICT=INCOMPLETE")
    sys.exit(0)

improved = any(deltas[c] > BASE[c][1] for c in deltas)
regressed = any(deltas[c] < -BASE[c][1] for c in deltas)
print(f"  판정근거: improved={improved} (한 셀 이상 +std 초과 개선), "
      f"regressed={regressed} (어느 셀도 -std 초과 악화 없어야 함)")
print("VERDICT=" + ("GOOD" if (improved and not regressed) else "BAD"))
