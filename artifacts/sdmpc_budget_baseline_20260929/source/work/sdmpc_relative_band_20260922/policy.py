"""후보별 양측 ±5% band와 budget 변화에 따른 폭의 미분."""
import numpy as np
FRACTION=.05


def halfwidth(budget):
    return FRACTION*np.abs(np.asarray(budget,dtype=float))


def budget_gradient(dual,budget,scales):
    dual,budget,scales=map(np.asarray,(dual,budget,scales))
    # h+=(G-B-.05|B|)/s, h-=(-G+B-.05|B|)/s.
    # B=0에서는 |B|의미분이없다.0은방향힌트용subgradient선택일뿐이다.
    return (dual[1]-dual[0]-FRACTION*np.sign(budget)*(dual[0]+dual[1]))/scales


def decode_grid_value(normalized,scale,allowed):
    # 허용값과같은정규화좌표만정확한원속도로복원한다.근처연속값을반올림하지않는다.
    for speed in allowed:
        if float(normalized)==float(speed)/scale:return float(speed)
    return float(normalized)*scale
