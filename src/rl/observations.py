from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.models.urban_queue_model import movement_storage_capacity
from src.rl.action_space import LeaderTargetAction
from src.rl.agents import RLAgentSpec


@dataclass(frozen=True)
class RLObservation:
    """DDQN 입력으로 바로 쓸 수 있는 feature vector와 locality metadata."""

    agent_id: str
    family: str
    features: np.ndarray
    feature_names: Tuple[str, ...]
    owned_links: Tuple[str, ...] = ()
    owned_ramps: Tuple[str, ...] = ()
    owned_movements: Tuple[str, ...] = ()
    owned_signals: Tuple[str, ...] = ()
    connected_coupling_links: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "family": self.family,
            "features": self.features.copy(),
            "feature_names": self.feature_names,
            "owned_links": self.owned_links,
            "owned_ramps": self.owned_ramps,
            "owned_movements": self.owned_movements,
            "owned_signals": self.owned_signals,
            "connected_coupling_links": self.connected_coupling_links,
        }


def build_leader_observation(
    state: TrafficState,
    cfg: ExperimentConfig,
    demand: DemandStep,
    previous_control: ControlAction | None,
) -> RLObservation:
    """leader는 compact global summary만 관찰한다."""

    previous = previous_control if previous_control is not None else ControlAction.fixed(cfg)
    net = cfg.network
    freeway_cap = max(
        len(net.freeway_links)
        * net.freeway_segments_per_link
        * net.freeway_segment_length_km
        * net.freeway_lanes
        * net.rho_max,
        1.0e-9,
    )
    urban_cap = _urban_capacity_veh(cfg)
    ramp_cap = max(net.ramp_queue_max_veh * max(len(net.ramps), 1), 1.0e-9)
    boundary_cap = max(cfg.leader.mfd_boundary_queue_capacity_veh, 1.0e-9)
    demand_cap = max(net.freeway_capacity_veh_h * max(len(net.freeway_links), 1), 1.0e-9)

    storage_ratios = _urban_storage_occupancy_ratios(state, cfg, tuple(net.urban_link_storage_veh))
    features = np.asarray(
        [
            state.protected_accumulation_veh(net) / urban_cap,
            state.total_freeway_vehicles(net) / freeway_cap,
            _sum_positive(state.ramp_queue.values()) / ramp_cap,
            _mean_positive(state.boundary_queue.values()) / boundary_cap,
            _max_positive(state.boundary_queue.values()) / boundary_cap,
            _mean(storage_ratios),
            _max(storage_ratios),
            previous.N_P_star / max(abs(cfg.leader.N_P_star_range[1]), abs(cfg.leader.N_P_star_range[0]), 1.0),
            previous.N_UF_star / max(abs(cfg.leader.N_UF_star_range[1]), 1.0),
            _sum_positive(demand.freeway_mainline.values()) / demand_cap,
            _sum_positive(demand.ramp_arrival.values()) / max(net.total_ramp_capacity, 1.0e-9),
        ],
        dtype=float,
    )
    return RLObservation(
        agent_id="leader",
        family="leader",
        features=np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        feature_names=(
            "protected_accumulation_ratio",
            "freeway_vehicle_ratio",
            "ramp_queue_ratio",
            "boundary_queue_mean_ratio",
            "boundary_queue_max_ratio",
            "urban_storage_mean_ratio",
            "urban_storage_max_ratio",
            "previous_N_P_star_ratio",
            "previous_N_UF_star_ratio",
            "freeway_demand_ratio",
            "ramp_demand_ratio",
        ),
    )


def build_follower_observations(
    state: TrafficState,
    cfg: ExperimentConfig,
    agents: Mapping[str, RLAgentSpec],
    leader_action: LeaderTargetAction,
    previous_control: ControlAction | None,
    demand: DemandStep,
) -> Dict[str, RLObservation]:
    """leader target을 공유 feature로 붙이되, follower traffic state는 local set만 쓴다."""

    previous = previous_control if previous_control is not None else ControlAction.fixed(cfg)
    observations: Dict[str, RLObservation] = {}
    for agent_id, agent in agents.items():
        if agent.is_freeway:
            observations[agent_id] = _build_freeway_observation(
                state,
                cfg,
                agent,
                leader_action,
                previous,
                demand,
            )
        elif agent.is_urban:
            observations[agent_id] = _build_urban_observation(
                state,
                cfg,
                agent,
                leader_action,
                previous,
                demand,
            )
    return observations


def _build_freeway_observation(
    state: TrafficState,
    cfg: ExperimentConfig,
    agent: RLAgentSpec,
    leader_action: LeaderTargetAction,
    previous_control: ControlAction,
    demand: DemandStep,
) -> RLObservation:
    net = cfg.network
    link = str(agent.link)
    segment_index = int(agent.segment_index or 0)
    density = _segment_value(state.freeway_density, link, segment_index)
    speed = _segment_value(state.freeway_speed, link, segment_index)
    flow = _segment_value(state.freeway_flow, link, segment_index)
    upstream_density = _segment_value(state.freeway_density, link, segment_index - 1)
    downstream_density = _segment_value(state.freeway_density, link, segment_index + 1)
    ramp_queues = [state.ramp_queue.get(ramp, 0.0) for ramp in agent.ramps]
    ramp_demands = [demand.ramp_arrival.get(ramp, 0.0) for ramp in agent.ramps]
    connected_movement_queues = [
        state.urban_movement_queue.get(movement, 0.0)
        for movement in agent.connected_movements
    ]
    offramp_storage_ratios = _urban_storage_occupancy_ratios(state, cfg, agent.urban_links)
    previous_vsl = segment_vsl(previous_control, link, segment_index, cfg)
    previous_metering = [
        previous_control.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp])
        / max(net.ramp_capacity_veh_h[ramp], 1.0e-9)
        for ramp in agent.ramps
    ]

    # Spec 19 locality: freeway actor에는 자기 segment/ramp와 직접 연결 요약만 넣는다.
    features = np.asarray(
        [
            density / max(net.rho_max, 1.0e-9),
            speed / max(net.v_free, 1.0e-9),
            flow / max(net.freeway_capacity_veh_h, 1.0e-9),
            upstream_density / max(net.rho_max, 1.0e-9),
            downstream_density / max(net.rho_max, 1.0e-9),
            _mean_positive(ramp_queues) / max(net.ramp_queue_max_veh, 1.0e-9),
            _max_positive(ramp_queues) / max(net.ramp_queue_max_veh, 1.0e-9),
            _mean_positive(ramp_demands) / max(net.total_ramp_capacity, 1.0e-9),
            _sum_positive(connected_movement_queues) / max(_movement_capacity_sum(cfg, agent.connected_movements), 1.0e-9),
            _mean(offramp_storage_ratios),
            previous_vsl / max(max(cfg.freeway_follower.vsl_set), 1.0e-9),
            _mean(previous_metering),
            leader_action.N_UF_star / max(abs(cfg.leader.N_UF_star_range[1]), 1.0),
            leader_action.N_P_star / max(abs(cfg.leader.N_P_star_range[1]), abs(cfg.leader.N_P_star_range[0]), 1.0),
        ],
        dtype=float,
    )
    return RLObservation(
        agent_id=agent.agent_id,
        family=agent.family,
        features=np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        feature_names=(
            "segment_density_ratio",
            "segment_speed_ratio",
            "segment_flow_ratio",
            "upstream_density_ratio",
            "downstream_density_ratio",
            "assigned_ramp_queue_mean_ratio",
            "assigned_ramp_queue_max_ratio",
            "assigned_ramp_demand_mean_ratio",
            "connected_movement_queue_ratio",
            "connected_offramp_storage_ratio",
            "previous_local_vsl_ratio",
            "previous_local_ramp_metering_ratio",
            "leader_N_UF_star_ratio",
            "leader_N_P_star_ratio",
        ),
        owned_links=(link,),
        owned_ramps=agent.ramps,
        owned_movements=(),
        connected_coupling_links=agent.connected_movements + agent.urban_links,
    )


def _build_urban_observation(
    state: TrafficState,
    cfg: ExperimentConfig,
    agent: RLAgentSpec,
    leader_action: LeaderTargetAction,
    previous_control: ControlAction,
    demand: DemandStep,
) -> RLObservation:
    net = cfg.network
    signal = str(agent.signal)
    movement_queues = [
        state.urban_movement_queue.get(movement, 0.0)
        for movement in agent.movements
    ]
    movement_caps = [
        movement_storage_capacity(cfg, movement, net.urban_movements[movement])
        for movement in agent.movements
    ]
    storage_ratios = _urban_storage_occupancy_ratios(state, cfg, agent.urban_links)
    boundary_queues = [state.boundary_queue.get(link, 0.0) for link in agent.boundary_links]
    boundary_demands = [demand.urban_boundary.get(link, 0.0) for link in agent.boundary_links]
    ramp_queues = [state.ramp_queue.get(ramp, 0.0) for ramp in agent.ramps]
    offramp_storage_ratios = _urban_storage_occupancy_ratios(
        state,
        cfg,
        tuple(net.off_ramp_storage_link.get(off_ramp, "") for off_ramp in agent.off_ramps),
    )
    effective_green = max(net.effective_green_total, 1.0e-9)
    previous_green = previous_control.green_times.get(f"{signal}_p1", 0.5 * effective_green)
    previous_offset = previous_control.offsets.get(signal, 0.0)

    # Spec 19 locality: signal actor에는 자기 movement/link/ramp coupling 요약만 넣는다.
    features = np.asarray(
        [
            _safe_sum_ratio(movement_queues, movement_caps),
            _safe_max_ratio(movement_queues, movement_caps),
            _mean(storage_ratios),
            _max(storage_ratios),
            _mean_positive(boundary_queues) / max(cfg.leader.mfd_boundary_queue_capacity_veh, 1.0e-9),
            _max_positive(boundary_queues) / max(cfg.leader.mfd_boundary_queue_capacity_veh, 1.0e-9),
            _mean_positive(boundary_demands) / max(net.movement_capacity_veh_h, 1.0e-9),
            _mean_positive(ramp_queues) / max(net.ramp_queue_max_veh, 1.0e-9),
            _mean(offramp_storage_ratios),
            previous_green / effective_green,
            previous_offset / max(net.cycle_length, 1.0e-9),
            leader_action.N_P_star / max(abs(cfg.leader.N_P_star_range[1]), abs(cfg.leader.N_P_star_range[0]), 1.0),
            leader_action.N_UF_star / max(abs(cfg.leader.N_UF_star_range[1]), 1.0),
        ],
        dtype=float,
    )
    return RLObservation(
        agent_id=agent.agent_id,
        family=agent.family,
        features=np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        feature_names=(
            "local_movement_queue_sum_ratio",
            "local_movement_queue_max_ratio",
            "connected_storage_mean_ratio",
            "connected_storage_max_ratio",
            "boundary_queue_mean_ratio",
            "boundary_queue_max_ratio",
            "boundary_demand_mean_ratio",
            "connected_ramp_queue_ratio",
            "connected_offramp_storage_ratio",
            "previous_green_p1_ratio",
            "previous_offset_ratio",
            "leader_N_P_star_ratio",
            "leader_N_UF_star_ratio",
        ),
        owned_links=agent.urban_links,
        owned_ramps=agent.ramps,
        owned_movements=agent.movements,
        owned_signals=(signal,),
        connected_coupling_links=agent.boundary_links + agent.off_ramps,
    )


def _segment_value(values: Mapping[str, list[float]], link: str, index: int) -> float:
    row = values.get(link, [])
    if index < 0 or index >= len(row):
        return 0.0
    return float(row[index])


def _sum_positive(values) -> float:
    return float(sum(max(0.0, float(value)) for value in values))


def _mean_positive(values) -> float:
    values = [max(0.0, float(value)) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _max_positive(values) -> float:
    values = [max(0.0, float(value)) for value in values]
    return float(max(values)) if values else 0.0


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _max(values) -> float:
    values = [float(value) for value in values]
    return float(max(values)) if values else 0.0


def _movement_capacity_sum(cfg: ExperimentConfig, movements: Tuple[str, ...]) -> float:
    return float(sum(movement_storage_capacity(cfg, movement) for movement in movements))


def _safe_sum_ratio(values, caps) -> float:
    return _sum_positive(values) / max(_sum_positive(caps), 1.0e-9)


def _safe_max_ratio(values, caps) -> float:
    ratios = [
        max(0.0, float(value)) / max(float(cap), 1.0e-9)
        for value, cap in zip(values, caps)
    ]
    return _max(ratios)


def _urban_storage_occupancy_ratios(
    state: TrafficState,
    cfg: ExperimentConfig,
    links: Tuple[str, ...],
) -> Tuple[float, ...]:
    ratios = []
    for link in links:
        if not link:
            continue
        cap = float(cfg.network.urban_link_storage_veh.get(link, 0.0))
        if cap <= 0.0:
            continue
        occupancy = max(0.0, cap - state.urban_link_storage.get(link, cap))
        ratios.append(float(occupancy / cap))
    return tuple(ratios)


def _urban_capacity_veh(cfg: ExperimentConfig) -> float:
    movement_cap = sum(
        movement_storage_capacity(cfg, movement, spec)
        for movement, spec in cfg.network.urban_movements.items()
    )
    link_cap = sum(float(value) for value in cfg.network.urban_link_storage_veh.values())
    return float(max(movement_cap + link_cap, 1.0e-9))
