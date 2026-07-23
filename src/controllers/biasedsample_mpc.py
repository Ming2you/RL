# box 상단 편향 Halton 샘플링(2026-07-21, 사용자 설계) — 선택점 분포 기반 후보 효율화
"""Top-biased low-discrepancy leader sampling.

raw-data 관찰(b9 8셀): 리더 선택점 best_N_UF/N_P가 box 상단 0.85~0.94에 몰린다
(하단 0~0.4는 거의 미사용). 원본은 Halton을 [0,1]에 **균등**하게 뿌려 하단에 후보를
낭비한다. 이 사본은 Halton fraction u를 u^BIAS_POW(<1)로 warp해 상단에 집중시킨다 —
box는 그대로(위기 대응 여력 유지), 후보 밀도만 실측 분포에 맞춘다.

warp: u -> u^p (p<1). p=0.5면 median 0.5->0.71, p=0.35면 0.5->0.79. 안전망으로
warp 안 한 균등 소수(잔여)도 섞어 하단을 완전히 버리지 않는다.

BIAS_SAMPLE=1 + BIAS_POW=<p>일 때만 활성. 미설정 시 원본과 완전 동일(비트동일).
"""
from __future__ import annotations

import types

from src.controllers.leader import LeaderAction


def _biased_low_discrepancy_samples(self, count, np_lower, np_upper, nuf_lower, nuf_upper):
    count = max(0, int(count))
    np_span = max(np_upper - np_lower, 0.0)
    nuf_span = max(nuf_upper - nuf_lower, 0.0)
    p = float(getattr(self.cfg.mpc, "leader_bias_sample_pow", 1.0) or 1.0)

    def vdc(index, base):
        value, denom, n = 0.0, 1.0, max(0, int(index))
        while n:
            n, rem = divmod(n, base)
            denom *= base
            value += rem / denom
        return value

    # 안전망: 후보의 1/5는 warp 없이 균등(하단 커버). 나머지 4/5는 상단 warp.
    n_uniform = max(1, count // 5)
    actions = []
    for i in range(1, count + 1):
        u_np = vdc(i, 2)
        u_nuf = vdc(i, 3)
        if i > n_uniform and p < 1.0:
            u_np = u_np ** p      # 상단(1)으로 몰기
            u_nuf = u_nuf ** p
        actions.append(LeaderAction(
            float(np_lower + u_np * np_span),
            float(nuf_lower + u_nuf * nuf_span),
        ))
    return actions


def enable_biased_sampling(controller) -> None:
    controller._orig_low_discrepancy = controller._continuous_low_discrepancy_samples
    controller._continuous_low_discrepancy_samples = types.MethodType(
        _biased_low_discrepancy_samples, controller
    )
