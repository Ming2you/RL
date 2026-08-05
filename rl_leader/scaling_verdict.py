# 데이터-스케일링 판정(2026-07-31) — 체이닝 분기용. 마지막 줄 SCALING=DATA_LIMITED|SATURATED|UNKNOWN.
"""같은 시드(0)로 학습한 frac=0.5 모델과 frac=1.0 모델의 held-out TTT를 비교한다.
100%가 50%보다 의미 있게 좋으면 곡선이 아직 오르는 중 = 데이터 부족 → 더 모으면 이득.
차이가 없거나 역전이면 포화 = 더 모아도 소용없고 설계를 바꿔야 한다.

임계는 budget-only 3시드 std(skew 116 / inc 77)의 절반 — 시드 노이즈보다 작은 차이는 신호로 안 본다.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESH = {"skew": 58.0, "inc": 38.5}


def read(tag, cell):
    p = ROOT / "logs" / f"eval_{tag}_{cell}.log"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if "시간가드 절단" in txt:
        return None
    m = re.search(r"windowed TTT = ([\d.]+)", txt)
    return float(m.group(1)) if m else None


gains, missing = {}, []
for cell in ["skew", "inc"]:
    half, full = read("p6f500", cell), read("p6s0", cell)
    if half is None or full is None:
        missing.append(cell)
        continue
    gains[cell] = half - full        # 양수 = 100%가 더 좋음(TTT 낮음) = 데이터 더 필요
    print(f"  {cell}: frac0.5={half:.1f}  frac1.0={full:.1f}  gain={half-full:+.1f} "
          f"(임계 {THRESH[cell]:.0f})")

if missing or not gains:
    print(f"  누락: {missing}")
    print("SCALING=UNKNOWN")
    sys.exit(0)

data_limited = any(gains[c] > THRESH[c] for c in gains)
print(f"  판정근거: 한 셀 이상에서 100%가 50%보다 임계 초과 개선 = {data_limited}")
print("SCALING=" + ("DATA_LIMITED" if data_limited else "SATURATED"))
