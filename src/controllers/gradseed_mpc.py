# proxy-gradient로 리더 예산(N_P,N_UF) 탐색을 유도하는 시드 확장(2026-07-21, 사용자 설계)
"""Gradient-guided leader budget search — 기존 flagship 인스턴스에 바인딩하는 방식.

원본 StackelbergMPC는 coarse 스텝에서 (N_P, N_UF) 박스에 Halton 저불일치 샘플을
**균등**하게 뿌린다. 사용자 관찰: 리더가 이미 proxy 목적함수를 계산하므로 그 gradient로
후보를 하강 방향에 몰아주면 적은 평가로 최적 예산에 접근할 수 있다.

핵심 — gradient는 **proxy**의 유한차분(full rollout 아님). full rollout 유한차분은
P-CENT SLSQP를 죽인 병목이지만 proxy는 싸서(follower 근사만) 2D gradient가 4회
proxy 호출로 끝난다. 비평활(capacity drop)은 gradient를 거칠게 하지만 ① 시드일 뿐
(전역 채점은 여전히 full rollout) ② 원본 Halton 시드도 함께 유지(안전망)라 무해.

`enable_gradseed(controller)`가 인스턴스 두 메서드를 오버라이드 — flagship 생성 후
속성 세팅을 건드리지 않는다. cfg.mpc.leader_gradseed_enabled=True로만 활성.
"""
from __future__ import annotations

import types

from src.controllers.leader import LeaderAction


def _gradseed_continuous_leader_search(self, state, forecast, previous, *args, **kwargs):
    # _continuous_seed_actions가 state/forecast를 안 받으므로 여기서 인스턴스에 실어둔다.
    self._gs_state = state
    self._gs_forecast = forecast
    return self._orig_continuous_leader_search(state, forecast, previous, *args, **kwargs)


def _gradseed_seed_actions(self, previous, bounds, np_lower, np_upper,
                           nuf_lower, nuf_upper, clipped):
    base = self._orig_continuous_seed_actions(
        previous, bounds, np_lower, np_upper, nuf_lower, nuf_upper, clipped
    )
    if not bool(getattr(self.cfg.mpc, "leader_gradseed_enabled", False)):
        return base
    np_span = max(np_upper - np_lower, 1.0e-9)
    nuf_span = max(nuf_upper - nuf_lower, 1.0e-9)
    h_np = 0.02 * np_span
    h_nuf = 0.02 * nuf_span
    c = clipped(float(previous.N_P_star), float(previous.N_UF_star))
    state = getattr(self, "_gs_state", None)
    forecast = getattr(self, "_gs_forecast", None)
    if state is None or forecast is None:
        return base

    def f(a):
        return float(self._proxy_score_candidate(0, a, state, forecast, previous)["objective"])

    fpx = f(clipped(c.N_P_star + h_np, c.N_UF_star))
    fmx = f(clipped(c.N_P_star - h_np, c.N_UF_star))
    fpy = f(clipped(c.N_P_star, c.N_UF_star + h_nuf))
    fmy = f(clipped(c.N_P_star, c.N_UF_star - h_nuf))
    g_np = (fpx - fmx) / (2.0 * h_np)
    g_nuf = (fpy - fmy) / (2.0 * h_nuf)
    norm = (g_np * g_np + g_nuf * g_nuf) ** 0.5
    if norm <= 1.0e-12:
        return base
    dx, dy = -g_np / norm, -g_nuf / norm
    ray = [
        clipped(c.N_P_star + dx * frac * np_span, c.N_UF_star + dy * frac * nuf_span)
        for frac in (0.1, 0.25, 0.5, 0.9)
    ]
    return self._unique_leader_actions(ray + base)


def enable_gradseed(controller) -> None:
    """flagship 인스턴스(F1Stackelberg...)에 proxy-gradient 시드를 바인딩한다."""
    controller._orig_continuous_leader_search = controller._continuous_leader_search
    controller._orig_continuous_seed_actions = controller._continuous_seed_actions
    controller._continuous_leader_search = types.MethodType(
        _gradseed_continuous_leader_search, controller
    )
    controller._continuous_seed_actions = types.MethodType(
        _gradseed_seed_actions, controller
    )
    controller.cfg.mpc.leader_gradseed_enabled = True
