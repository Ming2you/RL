from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.models.state import ControlAction, ExperimentConfig, segment_vsl
from src.rl.agents import RLAgentSpec


@dataclass(frozen=True)
class LeaderTargetAction:
    """leader DDQN action index가 가리키는 물리 target."""

    index: int
    N_P_star: float
    N_UF_star: float
    normalized: Tuple[float, float]

    def as_dict(self) -> Dict[str, float]:
        return {
            "index": float(self.index),
            "N_P_star": float(self.N_P_star),
            "N_UF_star": float(self.N_UF_star),
            "normalized_N_P_star": float(self.normalized[0]),
            "normalized_N_UF_star": float(self.normalized[1]),
        }


@dataclass(frozen=True)
class FreewayLocalAction:
    """segment actor가 자기 segment VSL과 배정 ramp metering만 갱신한다."""

    index: int
    vsl_km_h: float
    ramp_rate_factor: float | None
    vsl: Dict[str, float]
    ramp_metering: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "vsl_km_h": self.vsl_km_h,
            "ramp_rate_factor": self.ramp_rate_factor,
            "vsl": dict(self.vsl),
            "ramp_metering": dict(self.ramp_metering),
        }


@dataclass(frozen=True)
class UrbanLocalAction:
    """intersection actor가 자기 signal의 green split과 offset만 갱신한다."""

    index: int
    green_p1_sec: float
    green_p2_sec: float
    offset_delta_sec: float
    green_times: Dict[str, float]
    offsets: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "green_p1_sec": self.green_p1_sec,
            "green_p2_sec": self.green_p2_sec,
            "offset_delta_sec": self.offset_delta_sec,
            "green_times": dict(self.green_times),
            "offsets": dict(self.offsets),
        }


class LeaderDiscreteActionSpace:
    """`(N_P_star, N_UF_star)`를 작은 DDQN용 격자로 펼친다."""

    DEFAULT_DDQN_N_P_RANGE = (-100.0, 1000.0)

    def __init__(self, cfg: ExperimentConfig, n_p_bins: int = 5, n_uf_bins: int = 5):
        self.cfg = cfg
        self.n_p_bins = max(1, int(n_p_bins))
        self.n_uf_bins = max(1, int(n_uf_bins))
        self.actions = self._build_actions(cfg, n_p_bins, n_uf_bins)

    @property
    def size(self) -> int:
        return len(self.actions)

    def neutral_index(self) -> int:
        midpoint = np.asarray([0.0, 0.0], dtype=float)
        distances = [
            float(np.linalg.norm(np.asarray(action.normalized, dtype=float) - midpoint))
            for action in self.actions
        ]
        return int(np.argmin(distances))

    def map_index(self, index: int) -> LeaderTargetAction:
        return self.actions[_checked_index(index, len(self.actions))]

    @staticmethod
    def _build_actions(
        cfg: ExperimentConfig,
        n_p_bins: int,
        n_uf_bins: int,
    ) -> Tuple[LeaderTargetAction, ...]:
        n_p_bins = max(1, int(n_p_bins))
        n_uf_bins = max(1, int(n_uf_bins))
        n_p_physical_low, n_p_physical_high = [float(v) for v in cfg.leader.N_P_star_range]
        n_p_low, n_p_high = _ddqn_leader_np_range(cfg)
        n_uf_low, n_uf_high = [float(v) for v in cfg.leader.N_UF_star_range]
        n_p_values = np.linspace(n_p_low, n_p_high, n_p_bins)
        n_uf_values = np.linspace(n_uf_low, n_uf_high, n_uf_bins)
        actions = []
        index = 0
        for n_p in n_p_values:
            for n_uf in n_uf_values:
                actions.append(
                    LeaderTargetAction(
                        index=index,
                        N_P_star=float(np.clip(n_p, n_p_physical_low, n_p_physical_high)),
                        N_UF_star=float(np.clip(n_uf, n_uf_low, n_uf_high)),
                        normalized=(
                            _normalize_to_unit(n_p, n_p_low, n_p_high),
                            _normalize_to_unit(n_uf, n_uf_low, n_uf_high),
                        ),
                    )
                )
                index += 1
        return tuple(actions)


class FreewaySegmentActionSpace:
    """segment별 local VSL/RM discrete action을 물리 bound로 사영한다."""

    def __init__(self, cfg: ExperimentConfig, agent: RLAgentSpec):
        if not agent.is_freeway:
            raise ValueError("FreewaySegmentActionSpace requires a freeway agent.")
        self.cfg = cfg
        self.agent = agent
        self.actions = self._build_actions(cfg, agent)

    @property
    def size(self) -> int:
        return len(self.actions)

    def neutral_index(self) -> int:
        max_vsl = max(action.vsl_km_h for action in self.actions)
        candidates = [
            idx
            for idx, action in enumerate(self.actions)
            if action.vsl_km_h == max_vsl
            and (action.ramp_rate_factor is None or action.ramp_rate_factor == max(_ramp_factors(self.cfg)))
        ]
        return int(candidates[0]) if candidates else 0

    def map_index(self, index: int) -> FreewayLocalAction:
        return self.actions[_checked_index(index, len(self.actions))]

    @staticmethod
    def _build_actions(
        cfg: ExperimentConfig,
        agent: RLAgentSpec,
    ) -> Tuple[FreewayLocalAction, ...]:
        vsl_values = _bounded_vsl_values(cfg)
        ramp_factors = _ramp_factors(cfg) if agent.ramps else (None,)
        actions = []
        action_index = 0
        for vsl in vsl_values:
            for factor in ramp_factors:
                vsl_map = {f"{agent.link}__seg{agent.segment_index}": float(vsl)}
                ramp_metering = {}
                if factor is not None:
                    for ramp in agent.ramps:
                        cap = float(cfg.network.ramp_capacity_veh_h[ramp])
                        ramp_metering[ramp] = float(np.clip(factor, 0.0, 1.0) * cap)
                actions.append(
                    FreewayLocalAction(
                        index=action_index,
                        vsl_km_h=float(vsl),
                        ramp_rate_factor=None if factor is None else float(factor),
                        vsl=vsl_map,
                        ramp_metering=ramp_metering,
                    )
                )
                action_index += 1
        return tuple(actions)


class UrbanIntersectionActionSpace:
    """signal별 green split과 offset delta를 작은 discrete set으로 둔다."""

    def __init__(self, cfg: ExperimentConfig, agent: RLAgentSpec):
        if not agent.is_urban:
            raise ValueError("UrbanIntersectionActionSpace requires an urban agent.")
        self.cfg = cfg
        self.agent = agent
        self.green_pairs = _green_split_candidates(cfg)
        self.offset_deltas = _offset_delta_candidates(cfg)
        self.size = len(self.green_pairs) * len(self.offset_deltas)

    def neutral_index(self) -> int:
        green_idx = len(self.green_pairs) // 2
        zero_delta_idx = min(
            range(len(self.offset_deltas)),
            key=lambda idx: abs(self.offset_deltas[idx]),
        )
        return green_idx * len(self.offset_deltas) + zero_delta_idx

    def map_index(
        self,
        index: int,
        previous_control: ControlAction | None = None,
    ) -> UrbanLocalAction:
        index = _checked_index(index, self.size)
        green_idx = index // len(self.offset_deltas)
        delta_idx = index % len(self.offset_deltas)
        p1, p2 = self.green_pairs[green_idx]
        delta = self.offset_deltas[delta_idx]
        signal = str(self.agent.signal)
        previous_offset = 0.0
        if previous_control is not None:
            previous_offset = float(previous_control.offsets.get(signal, 0.0))
        offset = (previous_offset + delta) % max(float(self.cfg.network.cycle_length), 1.0e-9)
        return UrbanLocalAction(
            index=index,
            green_p1_sec=float(p1),
            green_p2_sec=float(p2),
            offset_delta_sec=float(delta),
            green_times={f"{signal}_p1": float(p1), f"{signal}_p2": float(p2)},
            offsets={signal: float(offset)},
        )


def build_follower_action_spaces(
    cfg: ExperimentConfig,
    agents: Mapping[str, RLAgentSpec],
) -> Dict[str, FreewaySegmentActionSpace | UrbanIntersectionActionSpace]:
    spaces: Dict[str, FreewaySegmentActionSpace | UrbanIntersectionActionSpace] = {}
    for agent_id, agent in agents.items():
        if agent.is_freeway:
            spaces[agent_id] = FreewaySegmentActionSpace(cfg, agent)
        elif agent.is_urban:
            spaces[agent_id] = UrbanIntersectionActionSpace(cfg, agent)
    return spaces


def compose_control_action(
    cfg: ExperimentConfig,
    leader_action: LeaderTargetAction,
    follower_action_indices: Mapping[str, int],
    agents: Mapping[str, RLAgentSpec],
    follower_action_spaces: Mapping[str, FreewaySegmentActionSpace | UrbanIntersectionActionSpace],
    previous_control: ControlAction | None,
) -> tuple[ControlAction, Dict[str, Dict[str, Any]]]:
    """leader target 이후 local follower action을 하나의 ControlAction으로 합친다.

    inflow_outflow_allocation은 이 milestone에서 학습 action으로 열지 않고,
    기존 fixed action의 안전한 perimeter allocation을 유지한다.
    """

    previous = previous_control.copy() if previous_control is not None else ControlAction.fixed(cfg)
    control = ControlAction.fixed(cfg)
    control.N_P_star = float(leader_action.N_P_star)
    control.N_UF_star = float(leader_action.N_UF_star)
    physical_actions: Dict[str, Dict[str, Any]] = {}
    projection_count = 0
    max_requested_vsl_delta = 0.0
    max_applied_vsl_delta = 0.0

    for agent_id, agent in agents.items():
        action_space = follower_action_spaces[agent_id]
        action_index = int(follower_action_indices.get(agent_id, action_space.neutral_index()))
        if agent.is_freeway:
            action = action_space.map_index(action_index)
            projected_vsl, vsl_diag = _project_freeway_vsl(cfg, agent, action, previous)
            control.vsl.update(projected_vsl)
            control.ramp_metering.update(action.ramp_metering)
            projection_count += int(vsl_diag["vsl_projected"])
            max_requested_vsl_delta = max(max_requested_vsl_delta, vsl_diag["requested_vsl_delta"])
            max_applied_vsl_delta = max(max_applied_vsl_delta, vsl_diag["applied_vsl_delta"])
            action_details = action.as_dict()
            action_details.update(vsl_diag)
            action_details["vsl"] = dict(projected_vsl)
            action_details["raw_vsl"] = dict(action.vsl)
            physical_actions[agent_id] = action_details
        elif agent.is_urban:
            action = action_space.map_index(action_index, previous)
            control.green_times.update(action.green_times)
            control.offsets.update(action.offsets)
            physical_actions[agent_id] = action.as_dict()

    control.diagnostics.update({
        "rl_action_projection_applied": float(projection_count > 0),
        "rl_projected_vsl_action_count": float(projection_count),
        "rl_max_requested_vsl_delta": float(max_requested_vsl_delta),
        "rl_max_applied_vsl_delta": float(max_applied_vsl_delta),
        "rl_action_fallback_used": 0.0,
        "rl_inflow_outflow_allocation_fixed_placeholder": 1.0,
        "rl_emits_e_control": 0.0,
    })
    return control, physical_actions


def centralized_action_space_metadata(
    agents: Mapping[str, RLAgentSpec],
    follower_action_spaces: Mapping[str, FreewaySegmentActionSpace | UrbanIntersectionActionSpace],
) -> Dict[str, Any]:
    """full centralized DDQN baseline은 factorized metadata로만 둔다."""

    return {
        "kind": "factorized_centralized_placeholder",
        "full_joint_action_count": None,
        "joint_ddqn_materialized": False,
        "emits_e_control": False,
        "heads": {
            agent_id: follower_action_spaces[agent_id].size
            for agent_id, agent in agents.items()
            if agent.is_freeway or agent.is_urban
        },
    }


def _project_freeway_vsl(
    cfg: ExperimentConfig,
    agent: RLAgentSpec,
    action: FreewayLocalAction,
    previous: ControlAction,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """이전 control 기준 VSL step 제한과 절대 VSL bound를 동시에 적용한다."""

    link = str(agent.link)
    segment_index = int(agent.segment_index or 0)
    key = f"{link}__seg{segment_index}"
    requested_vsl = float(action.vsl.get(key, action.vsl_km_h))
    previous_vsl = segment_vsl(previous, link, segment_index, cfg)
    vsl_low = float(cfg.freeway_follower.vsl_min_km_h)
    vsl_high = float(cfg.freeway_follower.vsl_max_km_h)
    max_step = max(0.0, float(cfg.freeway_follower.max_vsl_step))
    requested_bounded = float(np.clip(requested_vsl, vsl_low, vsl_high))
    step_low = max(vsl_low, previous_vsl - max_step)
    step_high = min(vsl_high, previous_vsl + max_step)
    applied_vsl = float(np.clip(requested_bounded, step_low, step_high))
    requested_delta = abs(requested_vsl - previous_vsl)
    applied_delta = abs(applied_vsl - previous_vsl)
    projected = abs(applied_vsl - requested_vsl) > 1.0e-9
    return (
        {key: applied_vsl},
        {
            "requested_vsl_km_h": float(requested_vsl),
            "bounded_requested_vsl_km_h": float(requested_bounded),
            "previous_vsl_km_h": float(previous_vsl),
            "applied_vsl_km_h": float(applied_vsl),
            "requested_vsl_delta": float(requested_delta),
            "applied_vsl_delta": float(applied_delta),
            "vsl_step_bound": float(max_step),
            "vsl_projected": float(projected),
        },
    )


def _checked_index(index: int, size: int) -> int:
    if size <= 0:
        raise ValueError("action space must not be empty.")
    index = int(index)
    if index < 0 or index >= size:
        raise IndexError(f"action index {index} outside [0, {size}).")
    return index


def _normalize_to_unit(value: float, low: float, high: float) -> float:
    if abs(high - low) <= 1.0e-9:
        return 0.0
    return float(2.0 * (float(value) - low) / (high - low) - 1.0)


def _ddqn_leader_np_range(cfg: ExperimentConfig) -> Tuple[float, float]:
    """Use a compact pilot grid for N_P targets while preserving physical bounds."""

    physical_low, physical_high = [float(v) for v in cfg.leader.N_P_star_range]
    default_low, default_high = LeaderDiscreteActionSpace.DEFAULT_DDQN_N_P_RANGE
    low = max(physical_low, float(default_low))
    high = min(physical_high, float(default_high))
    if low >= high:
        return physical_low, physical_high
    return low, high


def _bounded_vsl_values(cfg: ExperimentConfig) -> Tuple[float, ...]:
    low = float(cfg.freeway_follower.vsl_min_km_h)
    high = float(cfg.freeway_follower.vsl_max_km_h)
    values = sorted({
        float(np.clip(value, low, high))
        for value in cfg.freeway_follower.vsl_set
    })
    return tuple(values) if values else (float(np.clip(max(cfg.freeway_follower.vsl_set), low, high)),)


def _ramp_factors(cfg: ExperimentConfig) -> Tuple[float, ...]:
    low = float(cfg.freeway_follower.ramp_metering_rate_min)
    high = float(cfg.freeway_follower.ramp_metering_rate_max)
    mid = 0.5 * (low + high)
    return tuple(sorted({float(np.clip(v, low, high)) for v in (low, mid, high)}))


def _green_split_candidates(cfg: ExperimentConfig) -> Tuple[Tuple[float, float], ...]:
    net = cfg.network
    effective = float(net.effective_green_total)
    lower = max(float(net.green_min), effective - float(net.green_max))
    upper = min(float(net.green_max), effective - float(net.green_min))
    midpoint = 0.5 * (lower + upper)
    p1_values = sorted({float(np.clip(v, lower, upper)) for v in (lower, midpoint, upper)})
    return tuple((p1, effective - p1) for p1 in p1_values)


def _offset_delta_candidates(cfg: ExperimentConfig) -> Tuple[float, ...]:
    step = float(cfg.urban_follower.max_offset_step)
    return (-step, 0.0, step)
