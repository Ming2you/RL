# Joint controller(2026-07-06, per-actuator scalar price 한계 보완) — 신규 모드, 기존 무수정
"""reports/price_channel_arc_report_20260706.md §4.1 / 2026-07-06 notes §6.1의 설계 원칙 구현.

두 baseline의 objective를 모두 보존한다(둘 중 하나를 default로 가정하지 않음):
  - B2TR-style: `WuFaithfulFollower` own-TTS + green marginal price/trust (hinge 없음)
  - F1-style  : `F1WuFaithfulFollower` own-TTS + ρ_crit hinge + 동일 가격 구조
joint 탐색은 **mixin이 탐색만 교체**하고 objective는 상속으로 결정되므로 두 모드가
같은 코드로 지원된다(`_solve_with`가 각 계열의 `_solve_freeway_agent_local`을 호출).

freeway joint(1차): 기존 구조 분석 —
  `_solve_with(RM벡터)`는 내부에서 VSL 전 후보를 탐색한 최적응답을 반환하므로
  min_RM min_VSL = RM×VSL 격자의 전역 최솟값과 동일(RM-VSL cross term은 follower
  탐색에서 이미 포착; 잃었던 것은 B3 가격 투영). 남은 빈 곳 = **RM끼리의 결합**:
  자율/priced 분기의 per-ramp 좌표하강은 ramp×ramp 상호작용을 놓칠 수 있고, equality
  분기의 simplex는 성기다. 처방 = RM **full cross-grid**(분율 곱집합):
  - leader 없음(PFO/incumbent): full grid 그대로 — 격자가 좌표하강의 방문집합을
    포함하므로 joint 비용 ≤ 기존 비용이 보장된다.
  - equality: 각 grid 조합을 Σ=budget으로 비례 스케일(cap clip) — 성긴 7점 simplex를
    25개 비율 후보로 세분화. N_UF 계약(등식) 유지.
  - cap: Σ ≤ budget 조합만 허용(전무 시 비례 투영 fallback).

urban joint는 controller 측(`offset_joint_enabled`, stackelberg_wu_metered.py)의 패턴
directive + 2단계 green 변형으로 구현되어 이 파일의 컨트롤러 서브클래스가 켠다.
"""
from __future__ import annotations

import itertools
from typing import Dict, Mapping, Optional

import numpy as np

from src.controllers.f1_wu_faithful_follower import (
    F1StackelbergWuMeteredController,
    F1WuFaithfulFollower,
)
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState


class _JointFreewaySearchMixin:
    """freeway agent의 metering 탐색을 RM full cross-grid로 교체(objective는 상속)."""

    # 조합 폭발 가드: 소유 ramp가 이보다 많으면 부모 탐색으로 폴백.
    joint_max_ramps_per_link: int = 3

    def _solve_freeway_agent_metered(
        self,
        link: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        demand: DemandStep,
        snapshot: ControlAction,
        leader: Optional[object] = None,
    ):
        net = self.cfg.network
        owned = [r for r in net.ramps if net.ramp_to_freeway.get(r) == link]
        caps = {r: float(net.ramp_capacity_veh_h[r]) for r in owned}
        if not owned or len(owned) > self.joint_max_ramps_per_link:
            return super()._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, leader,
            )
        # metering 가격이 하달된 구성(아카이브된 B3 계열)은 부모의 priced 분기 유지.
        if getattr(self, "metering_marginal_price", None) is not None:
            return super()._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, leader,
            )

        evals_total = 0

        def _solve_with(meter: Mapping[str, float]):
            probe_prev = ControlAction(
                ramp_metering=dict(snapshot.ramp_metering),
                vsl=dict(snapshot.vsl),
                green_times=dict(snapshot.green_times),
                offsets=dict(snapshot.offsets),
                inflow_outflow_allocation={},
            )
            probe_prev.ramp_metering.update({r: float(v) for r, v in meter.items()})
            vsl_dict, cost, e = self._solve_freeway_agent_local(
                link, state, coupling, demand, probe_prev,
            )
            return vsl_dict, cost, e

        # ---- RM full cross-grid(분율 곱집합 + 현재값) ----
        per_ramp_values: Dict[str, list] = {}
        for r in owned:
            vals = {float(f) * caps[r] for f in self.ramp_metering_fractions}
            vals.add(float(np.clip(snapshot.ramp_metering.get(r, caps[r]), 0.0, caps[r])))
            per_ramp_values[r] = sorted(vals)
        combos = [
            {r: v for r, v in zip(owned, values)}
            for values in itertools.product(*[per_ramp_values[r] for r in owned])
        ]

        n_uf_star = float(getattr(leader, "N_UF_star", 0.0)) if leader is not None else 0.0
        if leader is not None and n_uf_star > 0.0:
            omega_f = float(self._wu._omega_f.get(link, 0.0))
            cap_sum = sum(caps[r] for r in owned)
            budget = float(np.clip(omega_f * n_uf_star, 0.0, cap_sum))
            nuf_mode = str(getattr(
                self.cfg.mpc, "wu_faithful_nuf_coordination_mode", "equality"
            ))
            if nuf_mode == "cap":
                kept = [c for c in combos if sum(c.values()) <= budget + 1.0e-9]
                if not kept:
                    # 전무 시 비례 투영 fallback(기존 cap 분기와 동일 철학).
                    kept = []
                    for c in combos:
                        total = sum(c.values())
                        scale = budget / max(total, 1.0e-9)
                        kept.append({
                            r: float(np.clip(v * scale, 0.0, caps[r]))
                            for r, v in c.items()
                        })
                combos = kept
            else:
                # equality: 각 조합을 Σ=budget으로 비례 스케일(cap clip) — 성긴 simplex를
                # 격자 비율들로 세분화. clip으로 Σ가 budget보다 약간 작아질 수 있음(근사).
                scaled = []
                for c in combos:
                    total = sum(c.values())
                    scale = budget / max(total, 1.0e-9)
                    scaled.append({
                        r: float(np.clip(v * scale, 0.0, caps[r]))
                        for r, v in c.items()
                    })
                combos = scaled
        # 중복 제거(스케일 후 동일 조합 정리).
        seen = set()
        unique_combos = []
        for c in combos:
            key = tuple(round(c[r], 6) for r in owned)
            if key not in seen:
                seen.add(key)
                unique_combos.append(c)

        best_meter: Dict[str, float] = {}
        best_vsl: Dict[str, float] = {}
        best_cost = float("inf")
        for meter in unique_combos:
            vsl_dict, cost, e = _solve_with(meter)
            evals_total += e
            if cost < best_cost:
                best_cost, best_vsl, best_meter = cost, vsl_dict, dict(meter)
        return best_vsl, best_meter, evals_total


class JointWuFaithfulFollower(_JointFreewaySearchMixin, WuFaithfulFollower):
    """Joint-B2TR objective: own-TTS(hinge 없음) + RM full cross-grid."""


class JointF1WuFaithfulFollower(_JointFreewaySearchMixin, F1WuFaithfulFollower):
    """Joint-F1 objective: own-TTS + ρ_crit hinge + RM full cross-grid."""


class JointB2TRController(StackelbergWuMeteredController):
    """Joint-B2TR 모드: B2TR 가격 구조 그대로 + joint follower(탐색만 교체)."""

    def _make_follower_solver(self, cfg: ExperimentConfig):
        return JointWuFaithfulFollower(cfg)


class JointF1Controller(F1StackelbergWuMeteredController):
    """Joint-F1 모드: F1RHO(ρ hinge) 구조 그대로 + joint follower."""

    def _make_follower_solver(self, cfg: ExperimentConfig):
        follower = JointF1WuFaithfulFollower(cfg)
        follower.f1_spillback_weight = 0.0  # F1RHO 규약(spill hinge는 위상적 무력 판정)
        return follower
