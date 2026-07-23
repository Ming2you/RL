# F1(2026-07-06, 사용자 제안): 안전 페널티를 follower objective로 이관 — 원본 무수정 사본
"""F1 아키텍처 실험판(reports/price_channel_arc_report_20260706.md §7, 07-05 notes §14).

설계 원리 "안전은 권한을 따라 이사해야 한다": 절벽/스필백 정보가 leader의 미분(가격)을
통과하면 3국면 붕괴(약함→쓰레기→0)로 소실된다 — 대신 follower의 **후보 단위 비선형
평가**(국소 정책 비교) 안에 hinge 페널티로 직접 넣는다.
  urban follower  : own-TTS + w_u·Σ max(0, 점유 − 0.5·cap)  [선형·차량수·veh·h]
  freeway follower: own-TTS + w_f·Σ max(0, ρ − ρ_crit)·L·lanes

**원본 보존**: `local_signal_plant.py`/`wu_faithful_follower.py`는 무수정. 이 파일은
(1) 세 urban rollout의 F1 사본(spill hinge 추가), (2) `F1WuFaithfulFollower` — urban은
probe 규약(모듈 이름 일시 교체)으로 F1 rollout을 주입해 부모 로직 재사용, freeway는
`_solve_freeway_agent_local` 사본에 ρ_crit hinge 추가, (3) `F1StackelbergWuMeteredController`.
가중치 0이면 부모와 비트 동일(휴면 게이트).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import src.controllers.wu_faithful_follower as wff
from src.controllers.local_signal_plant import (
    LocalSignalModel,
    _allocate_receiving_counts,
)
from src.controllers.local_freeway_plant import freeway_substep_local
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandStep
from src.models.metanet import effective_rho_crit
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl


# ---------------------------------------------------------------------------
# F1 urban rollout 사본 3종 — 원본과의 차이는 spill_thresh/spill_weight hinge뿐.
# 페널티: 매 substep, 추적 중인 링크의 여유공간 s_eff[link]가 thresh(=(1−frac)·cap)
# 아래로 내려간 부족분(=점유의 frac·cap 초과분)을 선형으로 비용에 가산.
# ---------------------------------------------------------------------------

def _spill_pen(
    s_eff: Mapping[str, float],
    spill_thresh: Optional[Mapping[str, float]],
    spill_weight: float,
    dt_h: float,
) -> float:
    if spill_thresh is None or spill_weight <= 0.0:
        return 0.0
    pen = 0.0
    for link, thr in spill_thresh.items():
        space = s_eff.get(link)
        if space is not None and space < thr:
            pen += (thr - space)
    return spill_weight * pen * dt_h


def f1_rollout_local_tts(
    model: LocalSignalModel,
    q0: Mapping[str, float],
    arr_movement: Mapping[str, float],
    s_eff0: Mapping[str, float],
    green_p1: float,
    green_p2: float,
    substeps: int,
    dt_h: float,
    s_eff_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    spill_thresh: Optional[Mapping[str, float]] = None,
    spill_weight: float = 0.0,
) -> float:
    """rollout_local_tts 사본 + spill hinge(원본: local_signal_plant.py:128)."""
    net = model.cfg.network
    cycle = max(net.cycle_length, 1.0e-9)
    green = {"p1": float(green_p1), "p2": float(green_p2)}
    q: Dict[str, float] = {m: max(0.0, float(q0.get(m, 0.0))) for m in model.movements}
    own_origin_links = {model.origin_of[m] for m in model.movements if model.origin_of[m]}
    s_eff: Dict[str, float] = {link: max(0.0, float(v)) for link, v in s_eff0.items()}

    cost = 0.0
    for sub in range(substeps):
        if s_eff_by_substep is not None:
            for recv, prof in s_eff_by_substep.items():
                if recv in s_eff:
                    s_eff[recv] = max(0.0, float(prof[sub]))
        for m in model.movements:
            q[m] += arr_movement.get(m, 0.0) * dt_h
        intended_by_link: Dict[str, Dict[str, float]] = {}
        no_link_intended: Dict[str, float] = {}
        for m in model.movements:
            pid = model.phase_of[m]
            green_fraction = green[pid] / cycle
            intended = min(q[m], dt_h * green_fraction * model.cap_flow_of[m])
            if intended < 0.0:
                intended = 0.0
            recv = model.receiving_of[m]
            if recv and recv in s_eff:
                intended_by_link.setdefault(recv, {})[m] = intended
            else:
                no_link_intended[m] = intended
        actual: Dict[str, float] = dict(no_link_intended)
        for link, intended in intended_by_link.items():
            actual.update(_allocate_receiving_counts(
                model.receiving_space_rule, intended, s_eff.get(link, 0.0),
            ))
        for m in model.movements:
            departed = min(q[m], max(0.0, actual.get(m, 0.0)))
            if departed <= 0.0:
                continue
            q[m] -= departed
            recv = model.receiving_of[m]
            if recv in own_origin_links and recv in s_eff:
                s_eff[recv] = max(0.0, s_eff[recv] - departed)
        cost += sum(q.values()) * dt_h
        cost += _spill_pen(s_eff, spill_thresh, spill_weight, dt_h)  # F1
    return float(cost)


def f1_rollout_local_tts_phased(
    model: LocalSignalModel,
    q0: Mapping[str, float],
    arr_by_substep: Mapping[str, List[float]],
    gf_by_substep: Mapping[str, List[float]],
    s_eff0: Mapping[str, float],
    substeps: int,
    dt_h: float,
    s_eff_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    spill_thresh: Optional[Mapping[str, float]] = None,
    spill_weight: float = 0.0,
) -> float:
    """rollout_local_tts_phased 사본 + spill hinge(원본: local_signal_plant.py:206)."""
    q: Dict[str, float] = {m: max(0.0, float(q0.get(m, 0.0))) for m in model.movements}
    own_origin_links = {model.origin_of[m] for m in model.movements if model.origin_of[m]}
    s_eff: Dict[str, float] = {link: max(0.0, float(v)) for link, v in s_eff0.items()}

    cost = 0.0
    for sub in range(substeps):
        if s_eff_by_substep is not None:
            for recv, prof in s_eff_by_substep.items():
                if recv in s_eff:
                    s_eff[recv] = max(0.0, float(prof[sub]))
        for m in model.movements:
            prof = arr_by_substep.get(m)
            if prof:
                q[m] += prof[sub] * dt_h
        intended_by_link: Dict[str, Dict[str, float]] = {}
        no_link_intended: Dict[str, float] = {}
        for m in model.movements:
            gf = gf_by_substep[m][sub]
            intended = min(q[m], dt_h * gf * model.cap_flow_of[m])
            if intended < 0.0:
                intended = 0.0
            recv = model.receiving_of[m]
            if recv and recv in s_eff:
                intended_by_link.setdefault(recv, {})[m] = intended
            else:
                no_link_intended[m] = intended
        actual: Dict[str, float] = dict(no_link_intended)
        for link, intended in intended_by_link.items():
            actual.update(_allocate_receiving_counts(
                model.receiving_space_rule, intended, s_eff.get(link, 0.0),
            ))
        for m in model.movements:
            departed = min(q[m], max(0.0, actual.get(m, 0.0)))
            if departed <= 0.0:
                continue
            q[m] -= departed
            recv = model.receiving_of[m]
            if recv in own_origin_links and recv in s_eff:
                s_eff[recv] = max(0.0, s_eff[recv] - departed)
        cost += sum(q.values()) * dt_h
        cost += _spill_pen(s_eff, spill_thresh, spill_weight, dt_h)  # F1
    return float(cost)


def f1_rollout_local_tts_ramp_aware(
    model: LocalSignalModel,
    q0: Mapping[str, float],
    arr_movement: Mapping[str, float],
    s_eff0: Mapping[str, float],
    offramp_inflow: Mapping[str, float],
    offramp_occ0: Mapping[str, float],
    ramp_queue0: Mapping[str, float],
    reservoir_drain: Mapping[str, float],
    freeway_congestion: Mapping[str, float],
    ramp_metering_weight: float,
    green_p1: float,
    green_p2: float,
    substeps: int,
    dt_h: float,
    arr_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    gf_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    s_eff_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    spill_thresh: Optional[Mapping[str, float]] = None,
    spill_weight: float = 0.0,
) -> float:
    """rollout_local_tts_ramp_aware 사본 + spill hinge(원본: local_signal_plant.py:272)."""
    net = model.cfg.network
    cycle = max(net.cycle_length, 1.0e-9)
    green = {"p1": float(green_p1), "p2": float(green_p2)}

    def _gf(m: str, sub: int) -> float:
        if gf_by_substep is not None:
            prof = gf_by_substep.get(m)
            if prof is not None:
                return float(prof[sub])
        return green[model.phase_of[m]] / cycle

    ramp_movement_set = {m for mv in model.onramp_movements.values() for m in mv}
    queue_movements = [m for m in model.movements if model.kind_of[m] != "off_ramp"]
    q: Dict[str, float] = {m: max(0.0, float(q0.get(m, 0.0))) for m in queue_movements}
    occ: Dict[str, float] = {
        orr: max(0.0, float(offramp_occ0.get(orr, 0.0))) for orr in model.offramp_movements
    }
    res: Dict[str, float] = {
        ramp: max(0.0, float(ramp_queue0.get(ramp, 0.0))) for ramp in model.onramp_movements
    }
    own_origin_links = {model.origin_of[m] for m in queue_movements if model.origin_of[m]}
    s_eff: Dict[str, float] = {link: max(0.0, float(v)) for link, v in s_eff0.items()}

    cost = 0.0
    for sub in range(substeps):
        if s_eff_by_substep is not None:
            for recv, prof in s_eff_by_substep.items():
                if recv in s_eff:
                    s_eff[recv] = max(0.0, float(prof[sub]))
        for m in queue_movements:
            if arr_by_substep is not None:
                prof = arr_by_substep.get(m)
                if prof is not None:
                    q[m] += float(prof[sub]) * dt_h
                    continue
            q[m] += arr_movement.get(m, 0.0) * dt_h

        for ramp in model.onramp_movements:
            res[ramp] = max(0.0, res.get(ramp, 0.0) - reservoir_drain.get(ramp, 0.0) * dt_h)

        for ramp, movements in model.onramp_movements.items():
            requests: Dict[str, float] = {}
            for m in movements:
                green_fraction = _gf(m, sub)
                requests[m] = min(
                    max(0.0, q[m]), dt_h * green_fraction * model.cap_flow_of[m]
                )
            requested_total = sum(requests.values())
            ramp_space = max(0.0, model.ramp_queue_max - res.get(ramp, 0.0))
            scale = 1.0 if requested_total <= ramp_space else ramp_space / max(requested_total, 1.0e-9)
            released_total = 0.0
            for m, requested in requests.items():
                actual = min(max(0.0, q[m]), requested * scale)
                q[m] = max(0.0, q[m] - actual)
                released_total += actual
            res[ramp] = min(model.ramp_queue_max, res.get(ramp, 0.0) + released_total)
            cost += (
                released_total * freeway_congestion.get(ramp, 0.0) * ramp_metering_weight
            )

        for off_ramp, movements in model.offramp_movements.items():
            occupancy = occ.get(off_ramp, 0.0)
            if occupancy <= 0.0:
                continue
            released_total = 0.0
            for m in movements:
                beta = model.beta_of[m]
                if beta <= 0.0:
                    continue
                green_fraction = _gf(m, sub)
                intended = min(beta * occupancy, dt_h * green_fraction * model.cap_flow_of[m])
                recv = model.receiving_of[m]
                if recv and recv in s_eff:
                    actual = min(intended, s_eff[recv])
                else:
                    actual = intended
                if actual <= 0.0:
                    continue
                released_total += actual
                if recv in own_origin_links and recv in s_eff:
                    s_eff[recv] = max(0.0, s_eff[recv] - actual)
            occ[off_ramp] = max(0.0, occupancy - released_total)

        for off_ramp in model.offramp_movements:
            inflow = max(0.0, float(offramp_inflow.get(off_ramp, 0.0))) * dt_h
            cap = model.offramp_storage_cap.get(off_ramp, 0.0)
            occ[off_ramp] = min(cap, occ.get(off_ramp, 0.0) + inflow)

        intended_by_link: Dict[str, Dict[str, float]] = {}
        no_link_intended: Dict[str, float] = {}
        for m in queue_movements:
            if m in ramp_movement_set:
                continue
            green_fraction = _gf(m, sub)
            intended = min(max(0.0, q[m]), dt_h * green_fraction * model.cap_flow_of[m])
            recv = model.receiving_of[m]
            if recv and recv in s_eff:
                intended_by_link.setdefault(recv, {})[m] = intended
            else:
                no_link_intended[m] = intended
        actual: Dict[str, float] = dict(no_link_intended)
        for link, intended in intended_by_link.items():
            actual.update(_allocate_receiving_counts(
                model.receiving_space_rule, intended, s_eff.get(link, 0.0),
            ))
        for m in queue_movements:
            if m in ramp_movement_set:
                continue
            departed = min(q[m], max(0.0, actual.get(m, 0.0)))
            if departed <= 0.0:
                continue
            q[m] -= departed
            recv = model.receiving_of[m]
            if recv in own_origin_links and recv in s_eff:
                s_eff[recv] = max(0.0, s_eff[recv] - departed)

        cost += (sum(q.values()) + sum(occ.values()) + sum(res.values())) * dt_h
        cost += _spill_pen(s_eff, spill_thresh, spill_weight, dt_h)  # F1
    return float(cost)


# ---------------------------------------------------------------------------
# F1 follower / controller
# ---------------------------------------------------------------------------

class F1WuFaithfulFollower(WuFaithfulFollower):
    """F1: urban 0.5cap spill hinge + freeway ρ_crit hinge를 own objective에 추가."""

    def __init__(self, cfg: ExperimentConfig, authority: str = "proposed"):
        super().__init__(cfg, authority)
        # 점유가 frac·cap을 넘으면 발화(사용자 스펙 0.5). 여유공간 관점 문턱 = (1−frac)·cap.
        self.f1_spillback_frac: float = 0.5
        self.f1_spillback_weight: float = 1.0   # 선형·차량수·veh·h — 1이 1차 정확값
        self.f1_rho_weight: float = 1.0
        # 가중치 0이면 부모와 비트 동일(휴면).

    # ---- urban: F1 rollout 주입(probe 규약 — wff 모듈 이름 일시 교체, try/finally) ----

    def _solve_urban_agent_local(self, signal, state, coupling, arr_movement,
                                 s_eff_frozen, reservoir_drain, freeway_congestion,
                                 previous, leader=None, lambda_p=0.0,
                                 forecast_arrivals=None, horizon_h=1.0, demand=None,
                                 candidates_override=None, committed_prev=None):
        # committed_prev(2026-07-18): base의 BASELINE-BOX green 앵커용 신규 kwarg.
        # 이 오버라이드가 시그니처에 없으면 base 호출부(L3804, committed_prev=previous)에서
        # TypeError — 플래그십(F1) 런 전멸. spillback 경로(자체 구현)는 BASELINE_BOX를
        # 쓰지 않으므로(플래그십은 해당 플래그 미설정) 전달만 하고 소비하지 않는다.
        if self.f1_spillback_weight <= 0.0:
            return super()._solve_urban_agent_local(
                signal, state, coupling, arr_movement, s_eff_frozen, reservoir_drain,
                freeway_congestion, previous, leader, lambda_p, forecast_arrivals,
                horizon_h, demand, candidates_override, committed_prev,
            )
        net = self.cfg.network
        thr_frac = 1.0 - float(self.f1_spillback_frac)
        model = self._local_models[signal]
        spill_thresh: Dict[str, float] = {}
        for m in model.movements:
            recv = model.receiving_of[m]
            if recv:
                cap = float(net.urban_link_storage_veh.get(recv, 0.0))
                if cap > 0.0:
                    spill_thresh[recv] = thr_frac * cap
        w = float(self.f1_spillback_weight)

        def _plain(*a, **k):
            return f1_rollout_local_tts(*a, spill_thresh=spill_thresh, spill_weight=w, **k)

        def _phased(*a, **k):
            return f1_rollout_local_tts_phased(*a, spill_thresh=spill_thresh, spill_weight=w, **k)

        def _ramp(*a, **k):
            return f1_rollout_local_tts_ramp_aware(*a, spill_thresh=spill_thresh, spill_weight=w, **k)

        originals = (
            wff.rollout_local_tts,
            wff.rollout_local_tts_phased,
            wff.rollout_local_tts_ramp_aware,
        )
        wff.rollout_local_tts = _plain
        wff.rollout_local_tts_phased = _phased
        wff.rollout_local_tts_ramp_aware = _ramp
        try:
            return super()._solve_urban_agent_local(
                signal, state, coupling, arr_movement, s_eff_frozen, reservoir_drain,
                freeway_congestion, previous, leader, lambda_p, forecast_arrivals,
                horizon_h, demand, candidates_override,
            )
        finally:
            (wff.rollout_local_tts,
             wff.rollout_local_tts_phased,
             wff.rollout_local_tts_ramp_aware) = originals

    # ---- freeway: 사본 + ρ_crit hinge(원본: wu_faithful_follower._solve_freeway_agent_local) ----

    def _solve_freeway_agent_local(
        self,
        link: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        demand: DemandStep,
        previous: ControlAction,
        vsl_override: Optional[Sequence[float]] = None,
    ):
        if self.f1_rho_weight <= 0.0:
            return super()._solve_freeway_agent_local(
                link, state, coupling, demand, previous, vsl_override=vsl_override,
            )
        net = self.cfg.network
        sim = self.cfg.simulation
        ff = self.cfg.freeway_follower
        model = self._local_freeway_models[link]
        horizon = max(1, ff.freeway_prediction_horizon_steps or self.cfg.mpc.horizon_steps)
        dt_h = sim.T_f_h
        vsl_max = max(ff.vsl_set)
        smooth_w = ff.vsl_smoothness_weight
        n_seg = model.n_seg
        prev_vec = [segment_vsl(previous, link, i, self.cfg) for i in range(n_seg)]
        # F1 hinge 상수.
        rho_crit = float(net.rho_crit)
        seg_veh = float(net.freeway_segment_length_km) * float(net.freeway_lanes)
        w_rho = float(self.f1_rho_weight)

        candidates = (
            self._wu._relaxed_freeway_segment_candidates(link, n_seg, state, coupling, previous, demand)
            if self.cfg.mpc.relaxed_quantized_controls
            else self._wu._freeway_segment_candidates(link, n_seg, previous)
        )
        vsl_sequences = self._freeway_vsl_sequence_candidates(
            link, n_seg, previous, candidates, horizon,
        )
        # JOINT h_local probe: vsl를 단일 값으로 고정 채점(고정 (meter,vsl) own-TTS).
        if vsl_override is not None:
            fixed_vec = [float(v) for v in vsl_override][:n_seg]
            if len(fixed_vec) < n_seg:
                fixed_vec += [float(fixed_vec[-1] if fixed_vec else vsl_max)] * (n_seg - len(fixed_vec))
            vsl_sequences = [[list(fixed_vec) for _ in range(horizon)]]
        # PRICE-TR: VSL 가격 활성 시 trust region 밖 후보 제외(base와 동일 규약).
        elif self.vsl_marginal_price and self.vsl_marginal_price_trust_kmh is not None:
            trust_v = float(self.vsl_marginal_price_trust_kmh)
            kept = []
            for seq in vsl_sequences:
                fv = seq[0] if seq else []
                ok = True
                for i, v in enumerate(fv):
                    ref = self.vsl_marginal_price_ref.get(f"{link}__seg{i}")
                    if ref is not None and abs(float(v) - float(ref)) > trust_v + 1.0e-9:
                        ok = False
                        break
                if ok:
                    kept.append(seq)
            if kept:
                vsl_sequences = kept

        rhos0 = list(state.freeway_density.get(link, []))
        speeds0 = list(state.freeway_speed.get(link, []))
        lanes0 = list(state.freeway_effective_lanes.get(link, [])) or [
            float(net.freeway_lanes) for _ in range(n_seg)
        ]
        if len(lanes0) != n_seg:
            lanes0 = [float(net.freeway_lanes) for _ in range(n_seg)]
        origin_q0 = max(0.0, float(state.mainline_origin_queue.get(link, 0.0)))
        ramp_q0 = {r: max(0.0, float(state.ramp_queue.get(r, 0.0))) for r in model.owned_ramps}
        occ0: Dict[str, float] = {}
        for off_ramp in model.owned_offramps:
            cap = model.offramp_storage_cap.get(off_ramp, 0.0)
            storage = net.off_ramp_storage_link.get(off_ramp, "")
            avail = float(state.urban_link_storage.get(storage, cap))
            occ0[off_ramp] = max(0.0, cap - avail)
        recv_links: set = set()
        for off_ramp in model.owned_offramps:
            for _signal, movement in self._wu._offramp_drain_flow.get(off_ramp, []):
                rl = str(self._wu._specs[movement].get("receiving_link", ""))
                if rl:
                    recv_links.add(rl)
        recv_occ0: Dict[str, float] = {}
        for rl in recv_links:
            cap = float(net.urban_link_storage_veh.get(rl, 0.0))
            avail = float(state.urban_link_storage.get(rl, cap))
            recv_occ0[rl] = max(0.0, cap - avail)
        urban_exit = float(net.boundary_out_capacity_veh_h)

        best_vec, best_obj = list(prev_vec), float("inf")
        best_offramp_flow: Dict[str, float] = {o: 0.0 for o in model.owned_offramps}
        evals = 0
        for sequence in vsl_sequences:
            candidate_control = ControlAction(
                ramp_metering=dict(previous.ramp_metering),
                vsl=dict(previous.vsl),
                green_times=dict(previous.green_times),
                offsets=dict(previous.offsets),
                inflow_outflow_allocation={},
            )
            first_vec = sequence[0] if sequence else list(prev_vec)
            for i, v in enumerate(first_vec):
                candidate_control.vsl[f"{link}__seg{i}"] = float(v)
            rhos = list(rhos0)
            speeds = list(speeds0)
            prev_lanes = list(lanes0)
            origin_q = origin_q0
            ramp_q = dict(ramp_q0)
            occ = dict(occ0)
            recv_occ = dict(recv_occ0)
            blocked_q = {r: 0.0 for r in model.owned_ramps}
            cost = 0.0
            first_offramp_flow = dict(best_offramp_flow)
            first_substep = True
            for horizon_idx in range(horizon):
                current_vec = sequence[min(horizon_idx, len(sequence) - 1)]
                for i, v in enumerate(current_vec):
                    candidate_control.vsl[f"{link}__seg{i}"] = float(v)
                for _ in range(sim.K_cf):
                    ramp_release = self._local_ramp_release(link, rhos, ramp_q, candidate_control, demand)
                    for ramp, rel in ramp_release.items():
                        ramp_q[ramp] = max(0.0, ramp_q.get(ramp, 0.0) - max(0.0, rel) * dt_h)
                    for ramp in model.owned_ramps:
                        approach = max(0.0, float(coupling.get(f"u_on_{ramp}", 0.0)))
                        if self.count_blocked_ramp_inflow:
                            q = max(0.0, ramp_q.get(ramp, 0.0))
                            space = max(0.0, net.ramp_queue_max_veh - q)
                            arrival = approach * dt_h
                            adm1 = min(blocked_q[ramp], space)
                            adm2 = min(arrival, space - adm1)
                            ramp_q[ramp] = min(net.ramp_queue_max_veh, q + adm1 + adm2)
                            blocked_q[ramp] = blocked_q[ramp] - adm1 + (arrival - adm2)
                        else:
                            ramp_q[ramp] = min(
                                net.ramp_queue_max_veh,
                                max(0.0, ramp_q.get(ramp, 0.0)) + approach * dt_h,
                            )
                    storage_avail = {
                        o: max(0.0, model.offramp_storage_cap.get(o, 0.0) - occ.get(o, 0.0))
                        for o in model.owned_offramps
                    }
                    offramp_capacity = self._local_offramp_capacity(link, storage_avail)
                    rhos, speeds, prev_lanes, origin_q, offramp_flow, veh_count = freeway_substep_local(
                        model, rhos, speeds, prev_lanes, occ, origin_q,
                        ramp_release, offramp_capacity, candidate_control, demand,
                    )
                    for off_ramp in model.owned_offramps:
                        cap = model.offramp_storage_cap.get(off_ramp, 0.0)
                        if cap <= 0.0:
                            continue
                        inflow = max(0.0, float(offramp_flow.get(off_ramp, 0.0)))
                        drain, recv_intake = self._local_offramp_drain(
                            off_ramp, occ.get(off_ramp, 0.0), recv_occ, candidate_control, dt_h,
                        )
                        occupied = min(cap, max(0.0, occ.get(off_ramp, 0.0) + (inflow - drain) * dt_h))
                        occ[off_ramp] = occupied
                        for recv_link, intake in recv_intake.items():
                            recv_cap = float(net.urban_link_storage_veh.get(recv_link, 0.0))
                            if recv_cap <= 0.0:
                                continue
                            relief = urban_exit if urban_exit > 0.0 else float("inf")
                            ro = min(recv_cap, max(0.0, recv_occ.get(recv_link, 0.0) + (intake - relief) * dt_h))
                            recv_occ[recv_link] = ro
                    if first_substep:
                        first_offramp_flow = {o: max(0.0, float(offramp_flow.get(o, 0.0))) for o in best_offramp_flow}
                        first_substep = False
                    link_vehicles = sum(veh_count)
                    link_ramp_queue = sum(max(0.0, ramp_q.get(r, 0.0)) for r in model.owned_ramps)
                    link_offramp_storage = sum(occ.get(o, 0.0) for o in model.owned_offramps)
                    link_blocked_queue = sum(blocked_q.values())
                    cost += (
                        link_vehicles + link_ramp_queue + link_offramp_storage + link_blocked_queue
                    ) * dt_h
                    # F1 ρ_crit hinge: 자기 본선 예측 밀도의 임계 초과 차량수를 선형 가산 —
                    # 절벽 정보가 미분 없이 후보 단위로 평가된다(후보가 절벽을 넘기면 통째로 비쌈).
                    # two_branch면 segment별 임계를 후보 VSL(first_vec)이 옮긴 ρ_crit(VSL)로 — VSL이
                    # 임계 올린 segment의 고밀도는 초과 아님(정합성). OFF면 nominal → 비트 동일.
                    excess = sum(
                        max(0.0, float(r) - effective_rho_crit(
                            net, first_vec[i] if i < len(first_vec) else net.v_free))
                        for i, r in enumerate(rhos)
                    )
                    if excess > 0.0:
                        cost += w_rho * excess * seg_veh * dt_h
            smooth = sum(abs(first_vec[i] - prev_vec[i]) for i in range(min(n_seg, len(first_vec))))
            for prev_step, next_step in zip(sequence, sequence[1:]):
                smooth += sum(
                    abs(next_step[i] - prev_step[i])
                    for i in range(min(len(prev_step), len(next_step), n_seg))
                )
            # PRICE-TR: VSL 가격 활성이면 smoothness 마찰 0(trust가 보폭 제약, base와 동일).
            if not (self.price_smoothness_disabled and self.vsl_marginal_price):
                cost += smooth_w * smooth
            if self.vsl_marginal_price:
                for i, value in enumerate(first_vec):
                    key = f"{link}__seg{i}"
                    g_vsl = self.vsl_marginal_price.get(key)
                    if g_vsl is not None and i < len(prev_vec):
                        ref = float(self.vsl_marginal_price_ref.get(key, float(prev_vec[i])))
                        cost += self.vsl_marginal_price_weight * float(g_vsl) * (
                            float(value) - ref
                        )
            # JOINT vsl×metering cross항(설정 시에만) — base와 동일 규약(link-binding VSL×ramp meter).
            if self.vsl_meter_cross_price:
                vsl_bind = float(min(first_vec)) if first_vec else vsl_max
                for ramp in model.owned_ramps:
                    h_c = self.vsl_meter_cross_price.get(ramp)
                    if h_c is None:
                        continue
                    m_ref, v_ref = self.vsl_meter_cross_ref.get(ramp, (0.0, vsl_bind))
                    m_now = float(previous.ramp_metering.get(ramp, float(m_ref)))
                    cost += self.vsl_meter_cross_weight * float(h_c) * (
                        (m_now - float(m_ref)) * (vsl_bind - float(v_ref))
                    )
            evals += 1
            if cost < best_obj:
                best_obj, best_vec = cost, list(first_vec)
                best_offramp_flow = dict(first_offramp_flow)
        if best_obj < float("inf"):
            for off_ramp, flow in best_offramp_flow.items():
                self._wu._last_offramp_flow[off_ramp] = float(flow)
            self._wu._last_offramp_flow[link] = float(sum(best_offramp_flow.values()))
            self._wu._has_last_offramp_flow = True
        else:
            for off_ramp in best_offramp_flow:
                self._wu._last_offramp_flow[off_ramp] = 0.0
            self._wu._last_offramp_flow[link] = 0.0
        vsl_dict: Dict[str, float] = {f"{link}__seg{i}": float(v) for i, v in enumerate(best_vec)}
        vsl_dict[link] = float(min(best_vec)) if best_vec else vsl_max
        return vsl_dict, best_obj, evals


class F1StackelbergWuMeteredController(StackelbergWuMeteredController):
    """P-Stack(B2TR 기본 구성) + F1 follower."""

    def _make_follower_solver(self, cfg: ExperimentConfig):
        return F1WuFaithfulFollower(cfg)
