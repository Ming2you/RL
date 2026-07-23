# controller-독립 free-flow delay reference (spec 16.11) — β 흡수 마르코프 체인으로 계산
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ExperimentConfig
from src.models.urban_queue_model import movement_specs


@dataclass
class FreeFlowReference:
    """시나리오·seed당 1회 계산해 여섯 controller가 공유하는 자유류 기준 TTT[veh·h]."""

    total_ttt: float
    urban_ttt: float
    freeway_ttt: float
    expected_urban_journey_h: Dict[str, float]
    expected_freeway_journey_h: float


def _urban_link_free_time_h(cfg: ExperimentConfig) -> float:
    """내부/경계 directed 링크의 자유류 통과시간[h] — 220대×6m / 50km/h (plant 정합)."""
    net = cfg.network
    length_km = float(net.grid_link_storage_veh) * net.urban_avg_vehicle_length_m / 1000.0
    return length_km / max(net.urban_avg_speed_km_h, 1.0e-9)


def _freeway_link_free_time_h(cfg: ExperimentConfig) -> float:
    net = cfg.network
    return net.freeway_segments_per_link * net.freeway_segment_length_km / max(net.v_free, 1.0e-9)


def expected_journey_times(cfg: ExperimentConfig) -> Dict[str, float]:
    """approach source별 기대 자유류 여정시간[h] (urban 구간만).

    상태 = approach source(게이트 in링크/내부 링크/off-ramp storage). 한 단계 전이는
    "movement 서비스 후 receiving link를 자유류로 통과해 다음 교차로 도착"이며 시간은
    receiving link의 자유류 통과시간이다. 흡수 = boundary_out 링크 끝 도착(sink) 또는
    on_ramp 전이(w_r 핸드오프, urban 구간 종료). β가 전이 확률이라 기대시간은
    E[T(source)] = Σ_movement β × (t_link(receiving) + E[T(next_source)]) 를
    고정점 반복으로 푼다(흡수 체인이라 수렴)."""
    net = cfg.network
    specs = movement_specs(cfg)
    t_link_h = _urban_link_free_time_h(cfg)
    out_links = set(net.boundary_out_links)

    # source key 규약은 urban_queue_model.approach_routing과 동일하다.
    transitions: Dict[str, list[tuple[float, str | None, float]]] = {}
    for movement, spec in specs.items():
        kind = str(spec.get("kind", ""))
        if kind == "off_ramp":
            source = net.off_ramp_storage_link.get(str(spec.get("origin", "")), "")
        else:
            source = str(spec.get("origin", ""))
        beta = float(spec.get("beta", 0.0))
        if str(spec.get("ramp", "")):
            # on_ramp 전이: 교차로 통과 즉시 w_r 핸드오프 — urban 여정 종료(추가 링크시간 0).
            transitions.setdefault(source, []).append((beta, None, 0.0))
            continue
        receiving = str(spec.get("receiving_link", ""))
        if str(spec.get("destination", "")) in out_links:
            # sink: 유출 링크를 자유류로 통과한 뒤 시스템 이탈.
            transitions.setdefault(source, []).append((beta, None, t_link_h))
        else:
            transitions.setdefault(source, []).append((beta, receiving, t_link_h))

    expected: Dict[str, float] = {source: 0.0 for source in transitions}
    # 흡수 마르코프 체인 기대시간 고정점 반복 — 14링크 망에서 수십 회면 충분히 수렴.
    for _ in range(200):
        delta = 0.0
        for source, branches in transitions.items():
            value = 0.0
            for beta, next_source, t_step in branches:
                value += beta * (t_step + (expected.get(next_source, 0.0) if next_source else 0.0))
            delta = max(delta, abs(value - expected[source]))
            expected[source] = value
        if delta < 1.0e-12:
            break
    return expected


def compute_free_flow_reference(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
) -> FreeFlowReference:
    """수요 적분 × 기대 자유류 여정시간으로 TTT reference를 계산한다.

    TTT = Σ(진입 차량 × 체류시간)이므로 자유류 기준 TTT_ref ≈ Σ_입구 Σ_interval
    arrival(t)×min(journey, horizon−t). no-control 시뮬레이션이 아니라 분석적
    reference(spec 16.11: free-flow travel time 기반, controller 독립)다."""
    net = cfg.network
    sim = cfg.simulation
    profile = DemandProfile(cfg, scenario)
    journeys = expected_journey_times(cfg)
    t_fw_h = _freeway_link_free_time_h(cfg)
    t_link_h = _urban_link_free_time_h(cfg)
    off_ratio_by_link = {
        link: min(1.0, sum(
            ratio for off_ramp, ratio in net.off_ramp_split_ratio.items()
            if net.off_ramp_from_freeway.get(off_ramp) == link
        ))
        for link in net.freeway_links
    }
    # off-ramp 복귀 차량의 urban 여정 기대시간(off-ramp storage 통과 포함).
    off_journeys = {
        off_ramp: t_link_h + journeys.get(storage, 0.0)
        for off_ramp, storage in net.off_ramp_storage_link.items()
    }

    dt_h = sim.T_c_h
    steps = max(1, int(round(sim.T_total / max(sim.control_interval, 1.0e-9))))
    urban_ttt = 0.0
    freeway_ttt = 0.0
    for step in range(steps):
        t_sec = step * sim.control_interval
        remaining_h = (sim.T_total - t_sec) / 3600.0
        demand = profile.at(t_sec)
        # 게이트 진입: urban 여정.
        for gate in net.boundary_in_links:
            veh = max(0.0, demand.urban_boundary.get(gate, 0.0)) * dt_h
            urban_ttt += veh * min(journeys.get(gate, 0.0), remaining_h)
        # 외생 on-ramp 수요: x_on 대기 없이 자유류면 즉시 w_r→본선 — urban 체류 ≈ 0.
        # freeway 본선 진입: 본선 자유류 통과(off-ramp 분기 가중 평균).
        for link in net.freeway_links:
            veh = max(0.0, demand.freeway_mainline.get(link, 0.0)) * dt_h
            off_ratio = off_ratio_by_link[link]
            fw_time = min(t_fw_h, remaining_h)
            freeway_ttt += veh * fw_time
            # off-ramp로 빠진 비율은 urban으로 복귀해 잔여 여정을 수행한다.
            mean_off_journey = sum(off_journeys.values()) / max(len(off_journeys), 1)
            urban_ttt += veh * off_ratio * min(mean_off_journey, max(0.0, remaining_h - fw_time))
        # ramp 외생 수요는 본선 합류 후 잔여 본선 통과(대략 절반 지점 합류 → 절반 통과).
        for ramp in net.ramps:
            veh = max(0.0, demand.ramp_arrival.get(ramp, 0.0)) * dt_h
            freeway_ttt += veh * min(0.5 * t_fw_h, remaining_h)

    return FreeFlowReference(
        total_ttt=float(urban_ttt + freeway_ttt),
        urban_ttt=float(urban_ttt),
        freeway_ttt=float(freeway_ttt),
        expected_urban_journey_h={k: float(v) for k, v in journeys.items()},
        expected_freeway_journey_h=float(t_fw_h),
    )
