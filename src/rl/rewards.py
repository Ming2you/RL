from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from src.models.state import ExperimentConfig, TrafficState
from src.models.urban_queue_model import movement_storage_capacity
from src.rl.agents import RLAgentSpec


@dataclass(frozen=True)
class RewardBreakdown:
    """reward와 해석 가능한 cost 항을 함께 보관한다."""

    reward: float
    cost: float
    terms: Dict[str, float]


def leader_reward(
    state: TrafficState,
    cfg: ExperimentConfig,
    global_step_ttt: float,
) -> RewardBreakdown:
    """Spec 19 leader reward: global TTT + density excess + urban half-cap TTS."""

    density_excess_tts = freeway_density_excess_tts(state, cfg)
    urban_halfcap_tts = urban_storage_halfcap_tts(state, cfg)
    cost = float(global_step_ttt + density_excess_tts + urban_halfcap_tts)
    return RewardBreakdown(
        reward=-cost,
        cost=cost,
        terms={
            "global_step_ttt": float(global_step_ttt),
            "freeway_density_excess_tts": float(density_excess_tts),
            "urban_storage_halfcap_tts": float(urban_halfcap_tts),
        },
    )


def follower_rewards(
    state: TrafficState,
    cfg: ExperimentConfig,
    agents: Mapping[str, RLAgentSpec],
) -> Dict[str, RewardBreakdown]:
    rewards: Dict[str, RewardBreakdown] = {}
    for agent_id, agent in agents.items():
        if agent.is_freeway:
            rewards[agent_id] = freeway_follower_reward(state, cfg, agent)
        elif agent.is_urban:
            rewards[agent_id] = urban_follower_reward(state, cfg, agent)
    return rewards


def freeway_follower_reward(
    state: TrafficState,
    cfg: ExperimentConfig,
    agent: RLAgentSpec,
) -> RewardBreakdown:
    """freeway follower는 자기 segment/ramp/off-ramp local TTS만 비용화한다."""

    dt_h = float(cfg.simulation.control_interval_h)
    net = cfg.network
    link = str(agent.link)
    idx = int(agent.segment_index or 0)
    state.ensure_freeway_lane_profile(net)
    density_values = state.freeway_density.get(link, [])
    lane_values = state.freeway_effective_lanes.get(link, [])
    density = float(density_values[idx]) if 0 <= idx < len(density_values) else 0.0
    lanes = float(lane_values[idx]) if 0 <= idx < len(lane_values) else float(net.freeway_lanes)
    segment_veh = max(0.0, density) * net.freeway_segment_length_km * max(lanes, 0.0)
    ramp_queue_veh = sum(max(0.0, state.ramp_queue.get(ramp, 0.0)) for ramp in agent.ramps)
    origin_queue_veh = max(0.0, state.mainline_origin_queue.get(link, 0.0)) if idx == 0 else 0.0
    offramp_storage_veh = sum(
        _storage_occupancy_veh(state, cfg, storage_link)
        for storage_link in agent.urban_links
    )
    local_tts = float((segment_veh + ramp_queue_veh + origin_queue_veh + offramp_storage_veh) * dt_h)
    return RewardBreakdown(
        reward=-local_tts,
        cost=local_tts,
        terms={
            "segment_tts": float(segment_veh * dt_h),
            "ramp_queue_tts": float(ramp_queue_veh * dt_h),
            "origin_queue_tts": float(origin_queue_veh * dt_h),
            "offramp_storage_tts": float(offramp_storage_veh * dt_h),
        },
    )


def urban_follower_reward(
    state: TrafficState,
    cfg: ExperimentConfig,
    agent: RLAgentSpec,
) -> RewardBreakdown:
    """urban follower는 자기 movement/link/boundary local TTS만 비용화한다."""

    dt_h = float(cfg.simulation.control_interval_h)
    movement_queue_veh = sum(
        max(0.0, state.urban_movement_queue.get(movement, 0.0))
        for movement in agent.movements
    )
    storage_veh = sum(
        _storage_occupancy_veh(state, cfg, link)
        for link in agent.urban_links
    )
    boundary_queue_veh = sum(
        max(0.0, state.boundary_queue.get(link, 0.0))
        for link in agent.boundary_links
    )
    local_tts = float((movement_queue_veh + storage_veh + boundary_queue_veh) * dt_h)
    return RewardBreakdown(
        reward=-local_tts,
        cost=local_tts,
        terms={
            "movement_queue_tts": float(movement_queue_veh * dt_h),
            "storage_tts": float(storage_veh * dt_h),
            "boundary_queue_tts": float(boundary_queue_veh * dt_h),
        },
    )


def freeway_density_excess_tts(state: TrafficState, cfg: ExperimentConfig) -> float:
    """rho가 rho_crit를 넘는 segment vehicle-hours를 합산한다."""

    dt_h = float(cfg.simulation.control_interval_h)
    net = cfg.network
    state.ensure_freeway_lane_profile(net)
    total = 0.0
    for link, densities in state.freeway_density.items():
        lanes = state.freeway_effective_lanes.get(link, [])
        for idx, rho in enumerate(densities):
            lane_count = lanes[idx] if idx < len(lanes) else net.freeway_lanes
            total += (
                net.freeway_segment_length_km
                * max(0.0, float(lane_count))
                * max(float(rho) - net.rho_crit, 0.0)
                * dt_h
            )
    return float(total)


def urban_storage_halfcap_tts(state: TrafficState, cfg: ExperimentConfig) -> float:
    """urban movement/link/boundary queue의 50% cap 초과분을 vehicle-hours로 환산한다."""

    dt_h = float(cfg.simulation.control_interval_h)
    total = 0.0
    for movement, spec in cfg.network.urban_movements.items():
        queue = max(0.0, state.urban_movement_queue.get(movement, 0.0))
        cap = max(movement_storage_capacity(cfg, movement, spec), 1.0e-9)
        total += max(queue - 0.5 * cap, 0.0) * dt_h
    for link, cap_value in cfg.network.urban_link_storage_veh.items():
        cap = max(float(cap_value), 1.0e-9)
        occupancy = _storage_occupancy_veh(state, cfg, link)
        total += max(occupancy - 0.5 * cap, 0.0) * dt_h
    boundary_cap = max(float(cfg.leader.mfd_boundary_queue_capacity_veh), 1.0e-9)
    for queue in state.boundary_queue.values():
        total += max(max(0.0, float(queue)) - 0.5 * boundary_cap, 0.0) * dt_h
    return float(total)


def _storage_occupancy_veh(state: TrafficState, cfg: ExperimentConfig, link: str) -> float:
    cap = cfg.network.urban_link_storage_veh.get(link)
    if cap is None:
        return 0.0
    return float(max(0.0, float(cap) - state.urban_link_storage.get(link, float(cap))))
