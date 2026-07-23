from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from src.controllers.freeway_follower import FreewayFollowerResult
from src.controllers.grid_parallel import build_chunk_payloads, evaluate_grid_items
from src.controllers.inflow_outflow_allocation import (
    AllocationResult,
    BALANCE_INFLOW_KINDS,
    BALANCE_OUTFLOW_KINDS,
    INFLOW_KINDS,
    InflowOutflowAllocationModule,
    OUTFLOW_KINDS,
)
from src.controllers.leader import LeaderAction
from src.controllers.nash_solver import NashResult, _relax_map
from src.controllers.relaxed_quantization import (
    accumulate_repair_diagnostics,
    merge_repair_diagnostics,
    repair_vsl_value,
)
from src.controllers.simplified_inflow_outflow_allocation import (
    SimplifiedInflowOutflowAllocationModule,
)
from src.controllers.spillback_constraints import (
    assess_offramp_spillback,
    assess_onramp_spillback,
    ramp_arrivals_over_horizon,
)
from src.controllers.structured_grid import (
    GridControlCandidate,
    sensitivity_direction_candidates,
    sensitivity_probe_candidates,
    structured_grid_candidates,
)
from src.controllers.urban_follower import UrbanFollower
from src.models.demand import DemandStep, merge_freeway_lane_loss
from src.models.metanet import effective_lane_profile
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    _movement_capacity_flow,
    _phase_green_fraction,
    boundary_group_key,
    ensure_urban_state,
    estimate_onramp_green_release_flows,
    estimate_onramp_reservoir_inflow,
    movement_storage_capacity,
    movement_specs,
    safe_balance_index,
)


@dataclass(frozen=True)
class AgentSpec:
    id: str
    kind: str
    signal: str = ""
    link: str = ""
    movements: tuple[str, ...] = ()
    ramps: tuple[str, ...] = ()
    off_ramps: tuple[str, ...] = ()
    neighbors: tuple[str, ...] = ()
    segment_index: int = -1


@dataclass
class AgentSolve:
    agent_id: str
    objective: float
    ramp_metering: Dict[str, float] = field(default_factory=dict)
    vsl: Dict[str, float] = field(default_factory=dict)
    green_times: Dict[str, float] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)
    allocation: Dict[str, float] = field(default_factory=dict)
    infeasibility: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)


def _distributed_grid_process_chunk(payload: dict) -> list[tuple[GridControlCandidate, float, ControlAction, Dict[str, float]]]:
    coordinator = DistributedCoordinator(payload["cfg"], ablation=payload["ablation"])
    return [
        coordinator._rollout_grid_objective(
            payload["state"],
            candidate,
            payload["forecast"],
            payload["leader"],
            incumbent_obj=payload["incumbent_obj"],
            precheck_diag=precheck_diag,
        )
        for candidate, precheck_diag in payload["items"]
    ]


def _freeway_agent_id(link: str, segment_index: int | None = None) -> str:
    suffix = link.split("_")[-1] if "_" in link else link
    if segment_index is None:
        return f"F_{suffix}"
    return f"F_{suffix}{segment_index}"


def _urban_agent_id(signal: str) -> str:
    return f"U_{signal}"


def _urban_signal_for_movement(spec: Mapping[str, object], signals: Iterable[str]) -> str:
    signal_set = set(signals)
    phase = str(spec.get("phase", ""))
    if "_" in phase:
        owner = phase.split("_", 1)[0]
        if owner in signal_set:
            return owner
    signal = str(spec.get("signal", ""))
    return signal if signal in signal_set else ""


def _configured_segment_index(mapping: object, key: str, fallback: int, n_segments: int) -> int:
    if isinstance(mapping, Mapping) and key in mapping:
        return int(np.clip(float(mapping[key]), 0.0, float(n_segments - 1)))
    return int(np.clip(float(fallback), 0.0, float(n_segments - 1)))


def build_agent_specs(cfg: ExperimentConfig) -> tuple[list[AgentSpec], list[AgentSpec]]:
    """현재 topology에서 Wu식 urban/freeway agent 분할을 자동 유도한다."""
    net = cfg.network
    specs = movement_specs(cfg)
    movement_owner = {
        movement: _urban_signal_for_movement(spec, net.signals)
        for movement, spec in specs.items()
    }
    urban_agents: list[AgentSpec] = []
    for signal in net.signals:
        movements = tuple(
            movement
            for movement, spec in specs.items()
            if movement_owner.get(movement) == signal
        )
        ramps = tuple(
            ramp for ramp, ramp_movements in net.on_ramp_to_movement.items()
            if any(movement in movements for movement in ramp_movements)
        )
        off_ramps = tuple(
            off_ramp for off_ramp, ramp_movements in net.off_ramp_to_movement.items()
            if any(movement in movements for movement in ramp_movements)
        )
        neighbors = sorted({
            _freeway_agent_id(
                net.ramp_to_freeway[ramp],
                _configured_segment_index(
                    getattr(net, "ramp_merge_segment_index", {}),
                    ramp,
                    net.freeway_segments_per_link // 2,
                    net.freeway_segments_per_link,
                ),
            )
            for ramp in ramps
        } | {
            _freeway_agent_id(
                net.off_ramp_from_freeway[off_ramp],
                _configured_segment_index(
                    getattr(net, "off_ramp_segment_index", {}),
                    off_ramp,
                    net.freeway_segments_per_link - 1,
                    net.freeway_segments_per_link,
                ),
            )
            for off_ramp in off_ramps
        })
        urban_agents.append(AgentSpec(
            id=_urban_agent_id(signal),
            kind="urban",
            signal=signal,
            movements=movements,
            ramps=ramps,
            off_ramps=off_ramps,
            neighbors=tuple(neighbors),
        ))

    urban_by_ramp = {
        ramp: _urban_agent_id(movement_owner[ramp_movements[0]])
        for ramp, ramp_movements in net.on_ramp_to_movement.items()
        if ramp_movements and ramp_movements[0] in specs and movement_owner.get(ramp_movements[0])
    }
    urban_by_offramp = {
        off_ramp: _urban_agent_id(movement_owner[ramp_movements[0]])
        for off_ramp, ramp_movements in net.off_ramp_to_movement.items()
        if ramp_movements and ramp_movements[0] in specs and movement_owner.get(ramp_movements[0])
    }
    freeway_agents: list[AgentSpec] = []
    for link in net.freeway_links:
        for segment_index in range(net.freeway_segments_per_link):
            ramps = tuple(
                ramp for ramp in net.ramps
                if net.ramp_to_freeway.get(ramp) == link
                and _configured_segment_index(
                    getattr(net, "ramp_merge_segment_index", {}),
                    ramp,
                    net.freeway_segments_per_link // 2,
                    net.freeway_segments_per_link,
                ) == segment_index
            )
            off_ramps = tuple(
                off_ramp
                for off_ramp in net.off_ramps
                if net.off_ramp_from_freeway.get(off_ramp) == link
                and _configured_segment_index(
                    getattr(net, "off_ramp_segment_index", {}),
                    off_ramp,
                    net.freeway_segments_per_link - 1,
                    net.freeway_segments_per_link,
                ) == segment_index
            )
            neighbors = sorted({
                urban_by_ramp[ramp]
                for ramp in ramps
                if ramp in urban_by_ramp
            } | {
                urban_by_offramp[off_ramp]
                for off_ramp in off_ramps
                if off_ramp in urban_by_offramp
            })
            freeway_agents.append(AgentSpec(
                id=_freeway_agent_id(link, segment_index),
                kind="freeway",
                link=link,
                ramps=ramps,
                off_ramps=off_ramps,
                neighbors=tuple(neighbors),
                segment_index=segment_index,
            ))
    return urban_agents, freeway_agents


def _project_to_target(target: float, upper: Mapping[str, float], weights: Mapping[str, float]) -> Dict[str, float]:
    release = {key: 0.0 for key in upper}
    remaining = float(np.clip(target, 0.0, sum(max(v, 0.0) for v in upper.values())))
    active = {key for key, value in upper.items() if value > 1.0e-9}
    while remaining > 1.0e-9 and active:
        w_sum = sum(max(weights.get(key, 1.0), 1.0e-9) for key in active)
        if w_sum <= 1.0e-9:
            break
        changed = False
        for key in list(active):
            proposed = remaining * max(weights.get(key, 1.0), 1.0e-9) / w_sum
            spare = max(0.0, upper[key] - release[key])
            if proposed >= spare - 1.0e-9:
                release[key] += spare
                remaining -= spare
                active.remove(key)
                changed = True
        if not changed:
            for key in active:
                release[key] += remaining * max(weights.get(key, 1.0), 1.0e-9) / w_sum
            remaining = 0.0
    return {key: float(min(max(value, 0.0), upper[key])) for key, value in release.items()}


def _project_to_bounded_target(
    target: float,
    lower: Mapping[str, float],
    upper: Mapping[str, float],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    keys = tuple(upper)
    bounded_lower = {
        key: float(np.clip(lower.get(key, 0.0), 0.0, max(0.0, upper[key])))
        for key in keys
    }
    bounded_upper = {key: float(max(bounded_lower[key], upper[key])) for key in keys}
    total_lower = sum(bounded_lower.values())
    total_upper = sum(bounded_upper.values())
    clipped_target = float(np.clip(target, total_lower, total_upper))
    residual_upper = {
        key: max(0.0, bounded_upper[key] - bounded_lower[key])
        for key in keys
    }
    residual = _project_to_target(
        clipped_target - total_lower,
        residual_upper,
        weights,
    )
    return {
        key: float(np.clip(bounded_lower[key] + residual.get(key, 0.0), bounded_lower[key], bounded_upper[key]))
        for key in keys
    }


ABLATION_MODES = (
    "FULL_COUPLING",
    "NO_U_TO_F_INFO",
    "NO_F_TO_U_INFO",
    "NO_CROSS_NETWORK_INFO",
    "LOCAL_ONLY_COUPLING_PLAYERS",
    "FIXED_URBAN_COUPLING_PLAYERS",
    "FIXED_FREEWAY_COUPLING_PLAYERS",
    "FIXED_ALL_COUPLING_PLAYERS",
    "WU_GREEN_VSL_ONLY_TTT",
)


class DistributedCoordinator:
    """Wu §IV-D 형태의 agent별 follower coordinator.

    이 1차 구현은 기존 follower 휴리스틱을 재사용하되, 적용 변수는 agent 소유 변수로
    제한하고 coupling variable 변화량으로 반복 종료를 판단한다.

    Stage 3 ablation(plan §10~§11): physical 결합·차량 이동은 plant에 그대로 두고,
    여기서 strategic 정보 교환(u→f 예측 방출, f→u 압력)만 차단하거나 coupling player
    (U_D/U_F, merge·off-ramp freeway agent)의 결정을 고정 정책으로 대체한다.
    잔여 player와 leader는 변경된 game 기준으로 매 호출 재최적화된다."""

    def __init__(self, cfg: ExperimentConfig, ablation: str = "FULL_COUPLING"):
        if ablation not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {ablation}")
        self.cfg = cfg
        self.ablation = ablation
        self.urban_agents, self.freeway_agents = build_agent_specs(cfg)
        self.urban_follower = UrbanFollower(cfg)
        self.allocation_module = InflowOutflowAllocationModule(cfg)
        self.simplified_allocation_module = SimplifiedInflowOutflowAllocationModule(cfg)
        self._repair_diagnostics: Dict[str, float] = {}
        self._specs = movement_specs(cfg)
        self._phase_movements: Dict[str, Dict[str, list[str]]] = {}
        for signal in cfg.network.signals:
            self._phase_movements[signal] = {
                phase_id: [
                    movement
                    for movement, spec in self._specs.items()
                    if spec.get("phase") == f"{signal}_{phase_id}"
                ]
                for phase_id in ("p1", "p2")
            }
        self._upstream_leaving_map = self._build_upstream_leaving_map()
        # coupling player 식별(plan §9.2): ramp/off-ramp 결합을 가진 agent — topology에서 자동.
        self.coupling_urban_ids = {a.id for a in self.urban_agents if a.ramps or a.off_ramps}
        self.coupling_freeway_ids = {a.id for a in self.freeway_agents if a.ramps or a.off_ramps}

    def _green_vsl_only_ttt_mode(self) -> bool:
        return self.ablation == "WU_GREEN_VSL_ONLY_TTT"

    def _no_metering_control(self, ramps: Optional[Iterable[str]] = None) -> Dict[str, float]:
        net = self.cfg.network
        selected = net.ramps if ramps is None else tuple(ramps)
        return {ramp: net.ramp_capacity_veh_h[ramp] for ramp in selected}

    def _zero_offsets(self) -> Dict[str, float]:
        return {signal: 0.0 for signal in self.cfg.network.signals}

    def _apply_green_vsl_only_authority(self, control: ControlAction) -> ControlAction:
        if not self._green_vsl_only_ttt_mode():
            return control
        control.N_P_star = 0.0
        control.N_UF_star = 0.0
        control.ramp_metering = self._no_metering_control()
        control.offsets = self._zero_offsets()
        control.inflow_outflow_allocation = {}
        control.diagnostics["wu_green_vsl_only_ttt_authority"] = 1.0
        return control

    def _leaderless_default_control(self) -> ControlAction:
        control = ControlAction.fixed(self.cfg)
        control.N_P_star = 0.0
        control.N_UF_star = 0.0
        # PFO/WU guard candidates must not smuggle in allocation authority.
        control.inflow_outflow_allocation = {}
        return control

    def _stackelberg_allocation_plan(
        self,
        state: TrafficState,
        leader: Optional[LeaderAction],
        forecast: list[DemandStep],
    ) -> Optional[AllocationResult]:
        """실험용 Stackelberg allocation ablation을 기본 direct path와 분리한다."""
        if leader is None:
            return None
        mode = str(getattr(self.cfg.mpc, "stackelberg_allocation_mode", "direct"))
        if mode == "direct":
            return None
        if mode == "pso":
            return self.allocation_module.solve(state, leader, forecast)
        if mode == "simplified":
            return self.simplified_allocation_module.solve(state, leader, forecast)
        raise ValueError(f"Unknown stackelberg_allocation_mode: {mode}")

    def _full_controller_guard_candidates(
        self,
        current: ControlAction,
    ) -> list[tuple[str, ControlAction]]:
        guards = [
            ("previous", current.copy()),
            ("no_control", ControlAction.uncontrolled(self.cfg)),
            ("default", self._leaderless_default_control()),
        ]
        out: list[tuple[str, ControlAction]] = []
        for label, control in guards:
            control.N_P_star = 0.0
            control.N_UF_star = 0.0
            out.append((label, self._apply_green_vsl_only_authority(control)))
        return out

    def _build_upstream_leaving_map(self) -> Dict[str, list[tuple[str, str, float]]]:
        """Wu `_upstream_leaving_map`와 같은 urban-to-urban phase coupling 지도.

        하류 phase pressure는 특정 internal movement 하나가 아니라 같은 incoming
        approach에서 같은 phase에 서는 모든 turn split의 합을 본다. 그래야 상류 green
        release가 downstream approach 전체 도착압으로 전달된다.
        """
        net = self.cfg.network
        signal_set = set(net.signals)
        producers_by_link: Dict[str, list[tuple[str, str]]] = {}
        for up_movement, up_spec in self._specs.items():
            dest = str(up_spec.get("destination", ""))
            up_signal = str(up_spec.get("signal", ""))
            if dest and up_signal in signal_set:
                producers_by_link.setdefault(dest, []).append((up_signal, up_movement))

        upstream_map: Dict[str, list[tuple[str, str, float]]] = {}
        for signal in net.signals:
            for phase_id, movements in self._phase_movements[signal].items():
                entries: list[tuple[str, str, float]] = []
                beta_by_origin: Dict[str, float] = {}
                for movement in movements:
                    spec = self._specs[movement]
                    origin = str(spec.get("origin", ""))
                    if not origin:
                        continue
                    beta_by_origin[origin] = beta_by_origin.get(origin, 0.0) + float(spec.get("beta", 0.0))
                for origin, beta in beta_by_origin.items():
                    for up_signal, up_movement in producers_by_link.get(origin, []):
                        entries.append((up_signal, up_movement, beta))
                upstream_map[f"{signal}_{phase_id}"] = entries
        return upstream_map

    def _signal_leaving_rate(
        self,
        movement: str,
        control: ControlAction,
        state: TrafficState,
        demand: DemandStep,
    ) -> float:
        """상류 movement green release rate[veh/h]를 downstream phase pressure로 보낸다."""
        spec = self._specs[movement]
        green_fraction = _phase_green_fraction(control, self.cfg, spec)
        cap_flow = _movement_capacity_flow(control, self.cfg, movement, spec)
        dt_h = max(self.cfg.simulation.T_c_h, 1.0e-9)
        available_flow = max(0.0, state.urban_movement_queue.get(movement, 0.0)) / dt_h
        kind = str(spec.get("kind", ""))
        beta = float(spec.get("beta", 1.0))
        if kind == "boundary_in":
            origin = str(spec.get("origin", ""))
            available_flow += max(0.0, demand.urban_boundary.get(origin, 0.0)) * beta
        elif kind == "on_ramp":
            ramp = str(spec.get("ramp", ""))
            available_flow += max(0.0, demand.ramp_arrival.get(ramp, 0.0)) * beta
        return float(min(green_fraction * cap_flow, available_flow))

    def _block_u_to_f(self, agent: AgentSpec) -> bool:
        """이 freeway agent가 urban 예측 정보(u_on 등)를 보면 안 되는가."""
        if self.ablation in {"NO_U_TO_F_INFO", "NO_CROSS_NETWORK_INFO"}:
            return True
        return self.ablation == "LOCAL_ONLY_COUPLING_PLAYERS" and agent.id in self.coupling_freeway_ids

    def _block_f_to_u(self, agent: AgentSpec) -> bool:
        """이 urban agent가 freeway 예측 압력/예측 off-ramp 정보를 보면 안 되는가."""
        if self.ablation in {"NO_F_TO_U_INFO", "NO_CROSS_NETWORK_INFO"}:
            return True
        return self.ablation == "LOCAL_ONLY_COUPLING_PLAYERS" and agent.id in self.coupling_urban_ids

    def _urban_player_fixed(self, agent: AgentSpec) -> bool:
        return (
            self.ablation in {"FIXED_URBAN_COUPLING_PLAYERS", "FIXED_ALL_COUPLING_PLAYERS"}
            and agent.id in self.coupling_urban_ids
        )

    def _freeway_player_fixed(self, agent: AgentSpec) -> bool:
        return (
            self.ablation in {"FIXED_FREEWAY_COUPLING_PLAYERS", "FIXED_ALL_COUPLING_PLAYERS"}
            and agent.id in self.coupling_freeway_ids
        )

    def _forecast_offramp_arrivals(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        link: str,
    ) -> float:
        """이 freeway link에서 갈라지는 off-ramp의 horizon 누적 예측 도착량[veh].

        off-ramp 도착 = diverge segment 도달 유량 × split. 현재 link 끝 유량을 기준으로
        forecast 본선 수요 비율만큼 horizon에 걸쳐 누적한다(boundary forecast가 본선
        수요를 바꾸면 off-ramp 예측 유입도 같이 변하게 — myopic이 아님)."""
        net = self.cfg.network
        dt_h = self.cfg.simulation.T_c_h
        horizon = max(1, self.cfg.mpc.horizon_steps)
        steps = forecast[: horizon]
        flows = state.freeway_flow.get(link, [])
        base_flow = float(flows[-1]) if flows else 0.0
        base_mainline = max(1.0e-9, float(forecast[0].freeway_mainline.get(link, 0.0)))
        total = 0.0
        for off_ramp in net.off_ramps:
            if net.off_ramp_from_freeway.get(off_ramp) != link:
                continue
            split = net.off_ramp_split_ratio.get(off_ramp, 0.0)
            for step in steps:
                # 본선 수요 비율로 도달 유량을 스케일 — forecast가 커지면 off-ramp 예측↑.
                scale = max(0.0, float(step.freeway_mainline.get(link, 0.0))) / base_mainline
                total += max(0.0, base_flow * scale * split) * dt_h
        return float(total)

    def _forecast_offramp_arrivals_by_ramp(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        link: str,
    ) -> Dict[str, float]:
        """link 집계와 같은 회계로 off-ramp별 horizon arrival[veh]을 만든다."""
        net = self.cfg.network
        dt_h = self.cfg.simulation.T_c_h
        horizon = max(1, self.cfg.mpc.horizon_steps)
        steps = forecast[: horizon]
        flows = state.freeway_flow.get(link, [])
        base_flow = float(flows[-1]) if flows else 0.0
        base_mainline = max(1.0e-9, float(forecast[0].freeway_mainline.get(link, 0.0)))
        out: Dict[str, float] = {}
        for off_ramp in net.off_ramps:
            if net.off_ramp_from_freeway.get(off_ramp) != link:
                continue
            split = net.off_ramp_split_ratio.get(off_ramp, 0.0)
            total = 0.0
            for step in steps:
                scale = max(0.0, float(step.freeway_mainline.get(link, 0.0))) / base_mainline
                total += max(0.0, base_flow * scale * split) * dt_h
            out[off_ramp] = float(total)
        return out

    def _freeway_neighbor_pressure(
        self,
        agent: AgentSpec,
        state: TrafficState,
        coupling: Mapping[str, float],
        lane_profile: Mapping[str, list[float]],
    ) -> float:
        """인접 segment 상태를 VSL/metring 판단에 넣는 freeway-to-freeway coupling pressure."""
        net = self.cfg.network
        rhos = state.freeway_density.get(agent.link, [])
        speeds = state.freeway_speed.get(agent.link, [])
        flows = state.freeway_flow.get(agent.link, [])
        lanes = lane_profile.get(agent.link, [net.freeway_lanes for _ in rhos])
        if agent.segment_index < 0 or not rhos:
            return 0.0
        pressure = 0.0
        for idx in (agent.segment_index - 1, agent.segment_index + 1):
            if idx < 0 or idx >= len(rhos):
                continue
            rho = float(coupling.get(f"rho_{agent.link}_seg{idx}", rhos[idx]))
            speed = float(coupling.get(
                f"speed_{agent.link}_seg{idx}",
                speeds[idx] if idx < len(speeds) else net.v_free,
            ))
            flow = float(coupling.get(
                f"flow_{agent.link}_seg{idx}",
                flows[idx] if idx < len(flows) else 0.0,
            ))
            lane_loss = max(
                0.0,
                float(coupling.get(
                    f"lane_loss_{agent.link}_seg{idx}",
                    net.freeway_lanes - float(lanes[idx] if idx < len(lanes) else net.freeway_lanes),
                )),
            )
            density_pressure = max(0.0, rho - net.rho_crit)
            speed_pressure = max(0.0, (net.v_free - speed) / max(net.v_free, 1.0e-9))
            flow_pressure = max(0.0, flow / max(net.freeway_capacity_veh_h, 1.0e-9) - 1.0)
            pressure += density_pressure + 0.25 * net.rho_crit * speed_pressure + 0.25 * net.rho_crit * flow_pressure
            pressure += 0.5 * lane_loss
        return float(max(0.0, pressure))

    def _response_is_better(
        self,
        candidate_obj: float,
        candidate_diag: Mapping[str, float],
        best_obj: float,
        best_diag: Mapping[str, float],
    ) -> bool:
        if not best_diag and not np.isfinite(best_obj):
            return True
        candidate_violation = self._candidate_constraint_violation(candidate_diag)
        best_violation = self._candidate_constraint_violation(best_diag)
        candidate_feasible = candidate_violation <= 1.0e-9
        best_feasible = best_violation <= 1.0e-9
        if candidate_feasible and not best_feasible:
            return True
        if best_feasible and not candidate_feasible:
            return False
        if not candidate_feasible and not best_feasible:
            if candidate_violation < best_violation - 1.0e-9:
                return True
            if candidate_violation > best_violation + 1.0e-9:
                return False
        if self._leader_balance_tiebreak_active(candidate_diag, best_diag):
            tol = 1.0e-3 * max(abs(float(candidate_obj)), abs(float(best_obj)), 1.0)
            if abs(float(candidate_obj) - float(best_obj)) <= tol:
                candidate_balance = float(candidate_diag.get(
                    "distributed_grid_leader_balance_tiebreak_score",
                    np.inf,
                ))
                best_balance = float(best_diag.get(
                    "distributed_grid_leader_balance_tiebreak_score",
                    np.inf,
                ))
                if candidate_balance < best_balance - 1.0e-12:
                    return True
                if candidate_balance > best_balance + 1.0e-12:
                    return False
        return candidate_obj < best_obj - 1.0e-12

    def _candidate_constraint_violation(self, diagnostics: Mapping[str, float]) -> float:
        spillback = float(diagnostics.get("distributed_response_total_spillback_violation_veh", 0.0))
        leader_feasible = float(diagnostics.get(
            "distributed_grid_leader_total_constraint_violation",
            0.0,
        ))
        return float(max(0.0, spillback) + max(0.0, leader_feasible))

    def _leader_balance_tiebreak_active(
        self,
        candidate_diag: Mapping[str, float],
        best_diag: Mapping[str, float],
    ) -> bool:
        return bool(
            candidate_diag.get("distributed_grid_leader_balance_tiebreak_active", 0.0)
            or best_diag.get("distributed_grid_leader_balance_tiebreak_active", 0.0)
        )

    def _grid_authority(self) -> str:
        return "wu" if self._green_vsl_only_ttt_mode() else "proposed"

    def _prepare_grid_control(
        self,
        control: ControlAction,
        leader: Optional[LeaderAction],
    ) -> ControlAction:
        out = control.copy()
        out.N_P_star = float(leader.N_P_star) if leader is not None else 0.0
        out.N_UF_star = float(leader.N_UF_star) if leader is not None else 0.0
        return self._apply_green_vsl_only_authority(out)

    def _rollout_grid_objective(
        self,
        state: TrafficState,
        candidate: GridControlCandidate,
        forecast: list[DemandStep],
        leader: Optional[LeaderAction],
        incumbent_obj: float = np.inf,
        precheck_diag: Optional[Mapping[str, float]] = None,
    ) -> tuple[GridControlCandidate, float, ControlAction, Dict[str, float]]:
        from src.simulation.coupling import run_coupled_interval

        control = self._prepare_grid_control(candidate.control, leader)
        s = state.copy()
        total_ttt = 0.0
        freeway_ttt = 0.0
        urban_ttt = 0.0
        horizon = max(1, min(len(forecast), self.cfg.mpc.horizon_steps))
        horizon_h = self.cfg.simulation.T_c_h * horizon
        early_terminated = False
        completed_steps = 0
        for demand in forecast[:horizon]:
            result = run_coupled_interval(s, control, demand, self.cfg)
            urban_ttt += float(result.urban_ttt)
            freeway_ttt += float(result.freeway_ttt)
            total_ttt += float(result.urban_ttt + result.freeway_ttt)
            s.time_sec += self.cfg.simulation.control_interval
            completed_steps += 1
            if np.isfinite(incumbent_obj) and total_ttt > incumbent_obj + 1.0e-12:
                early_terminated = True
                break

        if precheck_diag is None:
            proxy_obj, proxy_diag = self._response_tts_objective(
                state,
                control,
                forecast,
                residual=0.0,
                proxy_objective=0.0,
            )
        else:
            proxy_diag = dict(precheck_diag)
            proxy_obj = float(proxy_diag.get("distributed_response_objective_tts", 0.0))
        spillback_violation = float(proxy_diag.get("distributed_response_total_spillback_violation_veh", 0.0))
        spillback_penalty = (
            self.cfg.freeway_follower.ramp_queue_penalty
            * spillback_violation
            * self.cfg.simulation.T_c_h
            * horizon
        )
        objective = float(total_ttt + spillback_penalty)
        if early_terminated:
            objective = float(max(objective, incumbent_obj + 1.0e-9))
        diag = dict(proxy_diag)
        diag.update({
            "distributed_grid_search_active": 1.0,
            "distributed_grid_rollout_objective": float(objective),
            "distributed_grid_rollout_ttt": float(total_ttt),
            "distributed_grid_leader_constraint_penalty": 0.0,
            "distributed_grid_leader_constraint_penalty_disabled": 1.0,
            "distributed_grid_rollout_freeway_ttt": float(freeway_ttt),
            "distributed_grid_rollout_urban_ttt": float(urban_ttt),
            "distributed_grid_rollout_horizon_steps": float(horizon),
            "distributed_grid_rollout_completed_steps": float(completed_steps),
            "distributed_grid_early_terminated": float(early_terminated),
            "distributed_grid_proxy_objective_tts": float(proxy_obj),
            "distributed_grid_selected_stage_coarse": float(candidate.stage == "coarse"),
            "distributed_grid_selected_stage_fine": float(candidate.stage == "fine"),
            "distributed_grid_selected_stage_sensitivity_probe": float(candidate.stage == "sensitivity_probe"),
            "distributed_grid_selected_stage_sensitivity_direction": float(candidate.stage == "sensitivity_direction"),
            "distributed_grid_candidate_label_hash": float(abs(hash(candidate.label)) % 1000000),
            "distributed_grid_selected_target_net_inflow_candidate": float(
                "target_net_inflow" in candidate.label
            ),
            "distributed_response_rollout_active": 1.0,
            "distributed_response_rollout_ttt": float(total_ttt),
            "distributed_response_rollout_freeway_ttt": float(freeway_ttt),
            "distributed_response_rollout_urban_ttt": float(urban_ttt),
            "distributed_response_terminal_rollout_vehicles": float(
                s.total_urban_vehicles(self.cfg.network) + s.total_freeway_vehicles(self.cfg.network)
            ),
        })
        return candidate, objective, control, diag

    def _leader_direct_feasible_set_diagnostics(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: list[DemandStep],
        leader: Optional[LeaderAction],
    ) -> Dict[str, float]:
        if leader is None:
            return {}
        ensure_urban_state(state, self.cfg)
        net = self.cfg.network
        _demand, horizon_h, steps = self._response_horizon_demand(forecast)
        movement_arrivals = self._movement_forecast_arrivals_veh(steps)
        service: Dict[str, float] = {}
        available_by_movement: Dict[str, float] = {}
        raw_onramp_by_ramp: Dict[str, float] = {}
        onramp_by_movement = {
            movement: ramp
            for ramp, movements in net.on_ramp_to_movement.items()
            for movement in movements
        }

        for movement, spec in self._specs.items():
            available = max(0.0, state.urban_movement_queue.get(movement, 0.0)) + max(
                0.0,
                movement_arrivals.get(movement, 0.0),
            )
            cap_veh = horizon_h * _phase_green_fraction(control, self.cfg, spec) * _movement_capacity_flow(
                control,
                self.cfg,
                movement,
                spec,
            )
            served = min(available, max(0.0, cap_veh))
            service[movement] = float(served)
            available_by_movement[movement] = float(available)
            if str(spec.get("kind", "")) == "on_ramp":
                ramp = onramp_by_movement.get(movement, "")
                if ramp:
                    raw_onramp_by_ramp[ramp] = raw_onramp_by_ramp.get(ramp, 0.0) + served

        for ramp, raw_total in raw_onramp_by_ramp.items():
            if raw_total <= 1.0e-9:
                continue
            ramp_space = max(0.0, float(net.ramp_queue_max_veh) - max(0.0, state.ramp_queue.get(ramp, 0.0)))
            scale = min(1.0, ramp_space / raw_total)
            for movement in net.on_ramp_to_movement.get(ramp, []):
                if movement in service:
                    service[movement] *= scale

        remaining: Dict[str, float] = {
            movement: max(0.0, available_by_movement.get(movement, 0.0) - service.get(movement, 0.0))
            for movement in self._specs
        }
        inflow_veh = sum(
            service.get(movement, 0.0)
            for movement, spec in self._specs.items()
            if str(spec.get("kind", "")) in INFLOW_KINDS
        )
        outflow_veh = sum(
            service.get(movement, 0.0)
            for movement, spec in self._specs.items()
            if str(spec.get("kind", "")) in OUTFLOW_KINDS
        )
        projected_net_inflow_veh = inflow_veh - outflow_veh
        projected_net_inflow_rate = projected_net_inflow_veh / max(horizon_h, 1.0e-9)
        target_net_inflow_veh = float(leader.N_P_star)
        target_net_inflow_rate = target_net_inflow_veh / max(horizon_h, 1.0e-9)
        residual_veh = projected_net_inflow_veh - target_net_inflow_veh
        eps_veh = float(self.cfg.urban_follower.eps_U)
        net_violation_veh = max(0.0, abs(residual_veh) - eps_veh)

        def grouped_densities(kinds: set[str]) -> list[float]:
            queues: Dict[str, float] = {}
            caps: Dict[str, float] = {}
            for movement, spec in self._specs.items():
                if str(spec.get("kind", "")) not in kinds:
                    continue
                key = boundary_group_key(spec)
                queues[key] = queues.get(key, 0.0) + remaining.get(movement, 0.0)
                caps[key] = caps.get(key, 0.0) + max(
                    movement_storage_capacity(self.cfg, movement, spec),
                    1.0e-9,
                )
            return [queues[key] / max(caps[key], 1.0e-9) for key in sorted(queues)]

        b_in = safe_balance_index(grouped_densities(BALANCE_INFLOW_KINDS))
        b_out = safe_balance_index(grouped_densities(BALANCE_OUTFLOW_KINDS))
        balance_score = b_in * b_in + b_out * b_out
        storage_violation = 0.0
        for movement, spec in self._specs.items():
            cap = max(movement_storage_capacity(self.cfg, movement, spec), 1.0e-9)
            current_violation = max(0.0, max(0.0, state.urban_movement_queue.get(movement, 0.0)) - cap)
            projected_violation = max(0.0, remaining.get(movement, 0.0) - cap)
            storage_violation += max(0.0, projected_violation - current_violation)
        total_violation = net_violation_veh + storage_violation
        allocation_active = float(
            bool(control.inflow_outflow_allocation)
            or control.diagnostics.get("stackelberg_simplified_allocation_active", 0.0) > 0.5
        )
        return {
            "distributed_grid_leader_direct_feasible_set_active": 1.0,
            "distributed_grid_leader_allocation_module_active": float(allocation_active),
            "distributed_grid_leader_allocation_module_disabled": float(1.0 - allocation_active),
            "distributed_grid_leader_net_inflow_target_rate_veh_h": float(target_net_inflow_rate),
            "distributed_grid_leader_projected_net_inflow_rate_veh_h": float(projected_net_inflow_rate),
            "distributed_grid_leader_net_inflow_target_veh": float(target_net_inflow_veh),
            "distributed_grid_leader_projected_net_inflow_veh": float(projected_net_inflow_veh),
            "distributed_grid_leader_net_inflow_residual_veh": float(residual_veh),
            "distributed_grid_leader_net_inflow_abs_residual_veh": float(abs(residual_veh)),
            "distributed_grid_leader_net_inflow_eps_veh": float(eps_veh),
            "distributed_grid_leader_net_inflow_violation_veh": float(net_violation_veh),
            "distributed_grid_leader_storage_violation_veh": float(storage_violation),
            "distributed_grid_leader_total_constraint_violation": float(total_violation),
            "distributed_grid_leader_balance_B_in": float(b_in),
            "distributed_grid_leader_balance_B_out": float(b_out),
            "distributed_grid_leader_balance_tiebreak_score": float(balance_score),
            "distributed_grid_leader_balance_tiebreak_active": 1.0,
            "distributed_grid_leader_projected_inflow_veh": float(inflow_veh),
            "distributed_grid_leader_projected_outflow_veh": float(outflow_veh),
        }

    def _grid_feasibility_precheck(
        self,
        state: TrafficState,
        candidate: GridControlCandidate,
        forecast: list[DemandStep],
        leader: Optional[LeaderAction],
    ) -> tuple[bool, Dict[str, float]]:
        control = self._prepare_grid_control(candidate.control, leader)
        _obj, diag = self._response_tts_objective(
            state,
            control,
            forecast,
            residual=0.0,
            proxy_objective=0.0,
        )
        violation = float(diag.get("distributed_response_total_spillback_violation_veh", 0.0))
        leader_diag = self._leader_direct_feasible_set_diagnostics(state, control, forecast, leader)
        diag.update(leader_diag)
        diag["distributed_grid_leader_target_projection_candidates"] = float(
            control.diagnostics.get("distributed_grid_leader_target_projection_candidates", 0.0)
        )
        diag["distributed_grid_leader_target_projection_improved_candidates"] = float(
            control.diagnostics.get("distributed_grid_leader_target_projection_improved_candidates", 0.0)
        )
        leader_violation = float(diag.get("distributed_grid_leader_total_constraint_violation", 0.0))
        feasible = violation <= 1.0e-9 and leader_violation <= 1.0e-9
        diag["distributed_grid_precheck_feasible"] = float(feasible)
        diag["distributed_grid_precheck_spillback_violation_veh"] = violation
        diag["distributed_grid_precheck_leader_constraint_violation"] = leader_violation
        return feasible, diag

    def _evaluate_grid_stage(
        self,
        state: TrafficState,
        candidates: list[GridControlCandidate],
        forecast: list[DemandStep],
        leader: Optional[LeaderAction],
        incumbent_obj: float = np.inf,
    ) -> list[tuple[GridControlCandidate, float, ControlAction, Dict[str, float]]]:
        if not candidates:
            return []
        prechecked = [
            (candidate, *self._grid_feasibility_precheck(state, candidate, forecast, leader))
            for candidate in candidates
        ]
        feasible_candidates = [item for item in prechecked if item[1]]
        selected = feasible_candidates if feasible_candidates else prechecked
        guards = {"previous", "no_control", "center"}
        guard_items = [item for item in selected if item[0].label in guards]
        rest_items = [item for item in selected if item[0].label not in guards]
        results: list[tuple[GridControlCandidate, float, ControlAction, Dict[str, float]]] = []
        stage_incumbent = incumbent_obj
        for candidate, _feasible, precheck_diag in guard_items:
            result = self._rollout_grid_objective(
                state,
                candidate,
                forecast,
                leader,
                incumbent_obj=np.inf,
                precheck_diag=precheck_diag,
            )
            results.append(result)
            stage_incumbent = min(stage_incumbent, result[1])
        rest_eval_items = [(candidate, precheck_diag) for candidate, _feasible, precheck_diag in rest_items]

        def evaluate_item(item: tuple[GridControlCandidate, Mapping[str, float]]):
            candidate, precheck_diag = item
            return self._rollout_grid_objective(
                state,
                candidate,
                forecast,
                leader,
                incumbent_obj=stage_incumbent,
                precheck_diag=precheck_diag,
            )

        process_payloads = build_chunk_payloads(
            rest_eval_items,
            static={
                "cfg": self.cfg,
                "ablation": self.ablation,
                "state": state,
                "forecast": forecast,
                "leader": leader,
                "incumbent_obj": stage_incumbent,
            },
            chunk_size=int(self.cfg.mpc.grid_parallel_chunk_size),
            max_workers=int(self.cfg.mpc.grid_parallel_max_workers),
        )
        parallel_run = evaluate_grid_items(
            self.cfg,
            rest_eval_items,
            evaluate_item,
            process_chunk_fn=_distributed_grid_process_chunk,
            process_payloads=process_payloads,
        )
        results.extend(parallel_run.results)
        parallel_diag = parallel_run.diagnostics("distributed_grid")
        for result in results:
            result[3]["distributed_grid_precheck_filtered_candidates"] = float(len(prechecked) - len(selected))
            result[3]["distributed_grid_precheck_evaluated_candidates"] = float(len(selected))
            result[3].update(parallel_diag)
        return results

    def _structured_grid_refinement(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        center: ControlAction,
        leader: Optional[LeaderAction],
    ) -> tuple[ControlAction, float, Dict[str, float]]:
        authority = self._grid_authority()
        refresh_sec = float(self.cfg.mpc.grid_global_refresh_sec)
        interval = max(self.cfg.simulation.control_interval, 1.0e-9)
        step_index = int(round(state.time_sec / interval))
        refresh_steps = max(1, int(round(refresh_sec / interval)))
        global_refresh = step_index == 0 or step_index % refresh_steps == 0
        coarse_scope = "global" if global_refresh else "local"
        coarse = structured_grid_candidates(
            self.cfg,
            previous,
            center,
            authority=authority,
            stage="coarse",
            scope=coarse_scope,
        )
        coarse_results = self._evaluate_grid_stage(state, coarse, forecast, leader, incumbent_obj=np.inf)
        best_obj = np.inf
        best_control = center.copy()
        best_diag: Dict[str, float] = {}
        for _candidate, obj, control, diag in coarse_results:
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        probes = sensitivity_probe_candidates(
            self.cfg,
            previous,
            best_control,
            authority=authority,
        )
        probe_results = self._evaluate_grid_stage(state, probes, forecast, leader, incumbent_obj=np.inf)
        probe_scores: list[tuple[GridControlCandidate, float]] = []
        for candidate, obj, control, diag in probe_results:
            probe_scores.append((candidate, obj))
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        directions = sensitivity_direction_candidates(
            self.cfg,
            previous,
            best_control,
            probe_scores,
            authority=authority,
            base_objective=best_obj,
        )
        direction_results = self._evaluate_grid_stage(
            state,
            directions,
            forecast,
            leader,
            incumbent_obj=best_obj,
        )
        for _candidate, obj, control, diag in direction_results:
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        fine = structured_grid_candidates(
            self.cfg,
            previous,
            best_control,
            authority=authority,
            stage="fine",
            scope="local",
        )
        fine_results = self._evaluate_grid_stage(state, fine, forecast, leader, incumbent_obj=best_obj)
        for _candidate, obj, control, diag in fine_results:
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        if not np.isfinite(best_obj):
            return center.copy(), np.inf, {}
        stage_results = [coarse_results, probe_results, direction_results, fine_results]

        def stage_diag_sum(key: str) -> float:
            return float(sum(
                stage[0][3].get(key, 0.0)
                for stage in stage_results
                if stage
            ))

        best_diag.update({
            "distributed_grid_search_active": 1.0,
            "distributed_grid_full_search_active": 1.0,
            "distributed_grid_leader_conditioned": 0.0,
            "distributed_grid_parallel_stages": 4.0,
            "distributed_grid_stage1_candidates": float(len(coarse)),
            "distributed_grid_stage2_candidates": float(len(fine)),
            "distributed_grid_sensitivity_probe_candidates": float(len(probes)),
            "distributed_grid_sensitivity_direction_candidates": float(len(directions)),
            "distributed_grid_total_candidates": float(
                len(coarse) + len(probes) + len(directions) + len(fine)
            ),
            "distributed_grid_early_terminated_candidates": float(sum(
                result[3].get("distributed_grid_early_terminated", 0.0)
                for result in coarse_results + probe_results + direction_results + fine_results
            )),
            "distributed_grid_precheck_filtered_candidates": stage_diag_sum(
                "distributed_grid_precheck_filtered_candidates"
            ),
            "distributed_grid_precheck_evaluated_candidates": stage_diag_sum(
                "distributed_grid_precheck_evaluated_candidates"
            ),
            "distributed_grid_global_refresh": float(global_refresh),
            "distributed_grid_scope_global": float(coarse_scope == "global"),
            "distributed_grid_scope_local": float(coarse_scope == "local"),
            "distributed_grid_refresh_interval_sec": float(refresh_sec),
            "distributed_grid_authority_wu": float(authority == "wu"),
            "distributed_grid_authority_proposed": float(authority == "proposed"),
        })
        return best_control, float(best_obj), best_diag

    def _allocation_green_phase_setpoints(
        self,
        allocation_plan: Optional[AllocationResult],
    ) -> Dict[str, float]:
        if allocation_plan is None:
            return {}
        specs = movement_specs(self.cfg)
        by_phase: Dict[str, list[float]] = {}
        for movement, green_sec in allocation_plan.movement_green_sec.items():
            phase = str(specs.get(movement, {}).get("phase", ""))
            if phase:
                by_phase.setdefault(phase, []).append(float(green_sec))
        return {
            phase: float(np.mean(values))
            for phase, values in by_phase.items()
            if values
        }

    def _allocation_control_map(
        self,
        allocation_plan: Optional[AllocationResult],
    ) -> Dict[str, float]:
        if allocation_plan is None:
            return {}
        allocation = dict(allocation_plan.movement_flows)
        for link in self.cfg.network.boundary_in_links:
            allocation[link] = sum(
                allocation.get(movement, 0.0)
                for movement, spec in self.cfg.network.urban_movements.items()
                if spec.get("origin") == link and spec.get("kind") == "boundary_in"
            )
        for link in self.cfg.network.boundary_out_links:
            allocation[link] = sum(
                allocation.get(movement, 0.0)
                for movement, spec in self.cfg.network.urban_movements.items()
                if spec.get("destination") == link and spec.get("kind") == "boundary_out"
            )
        return allocation

    def _bounded_leader_green(self, signal: str, phase_setpoints: Mapping[str, float], fallback_p1: float) -> float:
        net = self.cfg.network
        total = net.effective_green_total
        p1_target = phase_setpoints.get(f"{signal}_p1")
        p2_target = phase_setpoints.get(f"{signal}_p2")
        if p1_target is not None and p2_target is not None:
            p1 = 0.5 * (float(p1_target) + (total - float(p2_target)))
        elif p1_target is not None:
            p1 = float(p1_target)
        elif p2_target is not None:
            p1 = total - float(p2_target)
        else:
            p1 = float(fallback_p1)
        p1 = float(np.clip(p1, net.green_min, net.green_max))
        p2 = total - p1
        if p2 < net.green_min:
            p2 = net.green_min
            p1 = total - p2
        if p2 > net.green_max:
            p2 = net.green_max
            p1 = total - p2
        return float(p1)

    def _leader_allocation_band_green(
        self,
        signal: str,
        phase_setpoints: Mapping[str, float],
        candidate_p1: float,
    ) -> float:
        """Leader allocation 기준 band 안에서 공통 grid의 green 변화를 보존한다."""
        net = self.cfg.network
        total = float(net.effective_green_total)
        p1_target = phase_setpoints.get(f"{signal}_p1")
        p2_target = phase_setpoints.get(f"{signal}_p2")
        if p1_target is None and p2_target is None:
            return self._bounded_leader_green(signal, {}, candidate_p1)

        if p1_target is None:
            p1_target = total - float(p2_target)
        if p2_target is None:
            p2_target = total - float(p1_target)
        band = max(
            0.0,
            float(self.cfg.urban_follower.allocation_green_band_sec),
            float(self.cfg.urban_follower.eps_g),
        )
        low = max(float(net.green_min), float(p1_target) - band, total - (float(p2_target) + band))
        high = min(float(net.green_max), float(p1_target) + band, total - (float(p2_target) - band))
        if low > high:
            low, high = float(net.green_min), float(net.green_max)
        p1 = float(np.clip(candidate_p1, low, high))
        p2 = total - p1
        if p2 < net.green_min:
            p2 = float(net.green_min)
            p1 = total - p2
        if p2 > net.green_max:
            p2 = float(net.green_max)
            p1 = total - p2
        return float(p1)

    def _set_leader_green(self, control: ControlAction, signal: str, p1: float) -> None:
        p1 = self._bounded_leader_green(signal, {}, p1)
        control.green_times[f"{signal}_p1"] = p1
        control.green_times[f"{signal}_p2"] = float(self.cfg.network.effective_green_total - p1)

    def _leader_metering_projection(
        self,
        leader: LeaderAction,
        weights: Mapping[str, float],
    ) -> Dict[str, float]:
        upper = {ramp: float(self.cfg.network.ramp_capacity_veh_h[ramp]) for ramp in self.cfg.network.ramps}
        min_ratio = float(self.cfg.freeway_follower.ramp_metering_rate_min)
        lower = {ramp: min_ratio * upper[ramp] for ramp in self.cfg.network.ramps}
        return _project_to_bounded_target(float(leader.N_UF_star), lower, upper, weights)

    def _project_control_to_leader_constraints(
        self,
        control: ControlAction,
        leader: LeaderAction,
        allocation_plan: Optional[AllocationResult],
    ) -> ControlAction:
        """공통 structured-grid 후보를 Stackelberg leader 제약으로 사영한다.

        RM은 후보의 ramp별 비율을 최대한 보존하면서 `sum ~= N_UF_star`를 맞추고,
        allocation은 `N_P_star`가 만든 allocation module 결과를 사용한다. Green은
        allocation band 안에서만 clipping해서 grid/sensitivity 방향성을 남긴다.
        """
        net = self.cfg.network
        phase_setpoints = self._allocation_green_phase_setpoints(allocation_plan)
        allocation = self._allocation_control_map(allocation_plan)
        out = control.copy()
        out.N_P_star = float(leader.N_P_star)
        out.N_UF_star = float(leader.N_UF_star)
        out.inflow_outflow_allocation = dict(allocation)
        if allocation_plan is not None:
            mode = str(getattr(self.cfg.mpc, "stackelberg_allocation_mode", "direct"))
            out.diagnostics.update({
                key: float(value)
                for key, value in allocation_plan.diagnostics.items()
                if isinstance(value, (int, float, bool))
            })
            out.diagnostics["stackelberg_simplified_allocation_active"] = float(
                allocation_plan.diagnostics.get("allocation_simplified_module_active", 0.0)
            )
            out.diagnostics[f"stackelberg_allocation_mode_{mode}"] = 1.0
        else:
            out.diagnostics["stackelberg_allocation_mode_direct"] = 1.0
        weights = {
            ramp: max(1.0, float(out.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp])))
            for ramp in net.ramps
        }
        out.ramp_metering = self._leader_metering_projection(leader, weights)
        for signal in net.signals:
            candidate_p1 = float(out.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0))
            p1 = self._leader_allocation_band_green(signal, phase_setpoints, candidate_p1)
            out.green_times[f"{signal}_p1"] = p1
            out.green_times[f"{signal}_p2"] = float(net.effective_green_total - p1)
            out.offsets[signal] = float(out.offsets.get(signal, 0.0)) % max(float(net.cycle_length), 1.0e-9)
        for link in net.freeway_links:
            out.vsl[link] = float(out.vsl.get(link, max(self.cfg.freeway_follower.vsl_set)))
        return self._apply_green_vsl_only_authority(out)

    def _project_leader_conditioned_candidates(
        self,
        candidates: list[GridControlCandidate],
        leader: LeaderAction,
        allocation_plan: Optional[AllocationResult],
        label_prefix: str,
    ) -> list[GridControlCandidate]:
        """공통 structured/sensitivity 후보들을 leader-conditioned feasible set으로 변환한다."""
        out: list[GridControlCandidate] = []
        seen: set[tuple[float, ...]] = set()

        def key(control: ControlAction) -> tuple[float, ...]:
            return tuple(round(v, 4) for v in control.control_vector(self.cfg))

        for candidate in candidates:
            control = self._project_control_to_leader_constraints(
                candidate.control,
                leader,
                allocation_plan,
            )
            k = key(control)
            if k in seen:
                continue
            seen.add(k)
            out.append(GridControlCandidate(
                label=f"{label_prefix}_{candidate.label}",
                control=control,
                stage=candidate.stage,
                scope=candidate.scope,
                axis=candidate.axis,
                delta=candidate.delta,
            ))
        return out

    def _leader_green_candidate_values(self, current_p1: float) -> list[float]:
        net = self.cfg.network
        total = float(net.effective_green_total)
        raw = [
            float(net.green_min),
            float(net.green_max),
            0.5 * total,
            current_p1 - 24.0,
            current_p1 - 12.0,
            current_p1 - 6.0,
            current_p1,
            current_p1 + 6.0,
            current_p1 + 12.0,
            current_p1 + 24.0,
        ]
        out: list[float] = []
        for value in raw:
            p1 = float(np.clip(value, net.green_min, net.green_max))
            p2 = total - p1
            if p2 < net.green_min:
                p1 = total - float(net.green_min)
            if p2 > net.green_max:
                p1 = total - float(net.green_max)
            if not any(abs(p1 - existing) <= 1.0e-9 for existing in out):
                out.append(float(p1))
        return out

    def _leader_target_net_inflow_projected_control(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: list[DemandStep],
        leader: LeaderAction,
        allocation_plan: Optional[AllocationResult],
    ) -> tuple[ControlAction, Dict[str, float]]:
        """Adjust green splits around a seed so projected net inflow tracks N_P."""
        net = self.cfg.network
        best = self._project_control_to_leader_constraints(control, leader, allocation_plan)
        best_diag = self._leader_direct_feasible_set_diagnostics(state, best, forecast, leader)
        best_abs = float(best_diag.get("distributed_grid_leader_net_inflow_abs_residual_veh", np.inf))
        for values in product(
            (float(net.green_min), float(net.green_max)),
            repeat=len(net.signals),
        ):
            trial = best.copy()
            for signal, p1 in zip(net.signals, values):
                trial.green_times[f"{signal}_p1"] = float(p1)
                trial.green_times[f"{signal}_p2"] = float(net.effective_green_total - p1)
            trial = self._project_control_to_leader_constraints(trial, leader, allocation_plan)
            diag = self._leader_direct_feasible_set_diagnostics(state, trial, forecast, leader)
            residual_abs = float(diag.get(
                "distributed_grid_leader_net_inflow_abs_residual_veh",
                np.inf,
            ))
            if residual_abs < best_abs - 1.0e-9:
                best = trial
                best_diag = diag
                best_abs = residual_abs
        # Coordinate search is cheap compared with rollout and can move several
        # signals at once, unlike the one-axis structured grid.
        for _pass in range(2):
            improved = False
            for signal in net.signals:
                current_p1 = float(best.green_times.get(
                    f"{signal}_p1",
                    net.effective_green_total / 2.0,
                ))
                signal_best = best
                signal_diag = best_diag
                signal_abs = best_abs
                for p1 in self._leader_green_candidate_values(current_p1):
                    trial = best.copy()
                    trial.green_times[f"{signal}_p1"] = float(p1)
                    trial.green_times[f"{signal}_p2"] = float(net.effective_green_total - p1)
                    trial = self._project_control_to_leader_constraints(trial, leader, allocation_plan)
                    diag = self._leader_direct_feasible_set_diagnostics(state, trial, forecast, leader)
                    residual_abs = float(diag.get(
                        "distributed_grid_leader_net_inflow_abs_residual_veh",
                        np.inf,
                    ))
                    if residual_abs < signal_abs - 1.0e-9:
                        signal_best = trial
                        signal_diag = diag
                        signal_abs = residual_abs
                if signal_abs < best_abs - 1.0e-9:
                    best = signal_best
                    best_diag = signal_diag
                    best_abs = signal_abs
                    improved = True
            if not improved:
                break
        best_diag = dict(best_diag)
        best_diag["distributed_grid_leader_target_green_projection_active"] = 1.0
        return best, best_diag

    def _augment_leader_target_net_inflow_candidates(
        self,
        state: TrafficState,
        candidates: list[GridControlCandidate],
        forecast: list[DemandStep],
        leader: LeaderAction,
        allocation_plan: Optional[AllocationResult],
        label_prefix: str,
    ) -> list[GridControlCandidate]:
        out = list(candidates)
        seen = {
            tuple(round(v, 4) for v in candidate.control.control_vector(self.cfg))
            for candidate in out
        }
        added = 0
        improved = 0
        for candidate in candidates:
            base_diag = self._leader_direct_feasible_set_diagnostics(
                state,
                candidate.control,
                forecast,
                leader,
            )
            base_abs = float(base_diag.get(
                "distributed_grid_leader_net_inflow_abs_residual_veh",
                np.inf,
            ))
            projected, projected_diag = self._leader_target_net_inflow_projected_control(
                state,
                candidate.control,
                forecast,
                leader,
                allocation_plan,
            )
            projected_abs = float(projected_diag.get(
                "distributed_grid_leader_net_inflow_abs_residual_veh",
                np.inf,
            ))
            if projected_abs >= base_abs - 1.0e-9:
                continue
            key = tuple(round(v, 4) for v in projected.control_vector(self.cfg))
            if key in seen:
                continue
            seen.add(key)
            added += 1
            improved += float(projected_abs < base_abs - 1.0e-9)
            out.append(GridControlCandidate(
                label=f"{label_prefix}_target_net_inflow_{candidate.label}",
                control=projected,
                stage=candidate.stage,
                scope=candidate.scope,
                axis="target_net_inflow",
                delta=float(base_abs - projected_abs),
            ))
        for candidate in out:
            candidate.control.diagnostics["distributed_grid_leader_target_projection_candidates"] = float(added)
            candidate.control.diagnostics["distributed_grid_leader_target_projection_improved_candidates"] = float(
                improved
            )
        return out

    def _leader_conditioned_grid_candidates(
        self,
        previous: ControlAction,
        center: ControlAction,
        leader: LeaderAction,
        allocation_plan: Optional[AllocationResult],
        stage: str = "coarse",
        scope: str = "global",
        state: Optional[TrafficState] = None,
        forecast: Optional[list[DemandStep]] = None,
    ) -> list[GridControlCandidate]:
        raw = structured_grid_candidates(
            self.cfg,
            previous,
            center,
            authority="proposed",
            stage=stage,
            scope=scope,
        )
        projected = self._project_leader_conditioned_candidates(
            raw,
            leader,
            allocation_plan,
            f"leader_{stage}_{scope}",
        )
        if state is None or forecast is None:
            return projected
        return self._augment_leader_target_net_inflow_candidates(
            state,
            projected,
            forecast,
            leader,
            allocation_plan,
            f"leader_{stage}_{scope}",
        )

    def _leader_conditioned_grid_refinement(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        center: ControlAction,
        leader: LeaderAction,
        allocation_plan: Optional[AllocationResult],
        incumbent_obj: float = np.inf,
    ) -> tuple[ControlAction, float, Dict[str, float]]:
        refresh_sec = float(self.cfg.mpc.grid_global_refresh_sec)
        interval = max(self.cfg.simulation.control_interval, 1.0e-9)
        step_index = int(round(state.time_sec / interval))
        refresh_steps = max(1, int(round(refresh_sec / interval)))
        global_refresh = step_index == 0 or step_index % refresh_steps == 0
        coarse_scope = "global" if global_refresh else "local"
        projected_center = self._project_control_to_leader_constraints(center, leader, allocation_plan)
        projected_previous = self._project_control_to_leader_constraints(previous, leader, allocation_plan)

        coarse = self._leader_conditioned_grid_candidates(
            projected_previous,
            projected_center,
            leader,
            allocation_plan,
            stage="coarse",
            scope=coarse_scope,
            state=state,
            forecast=forecast,
        )
        coarse_results = self._evaluate_grid_stage(
            state,
            coarse,
            forecast,
            leader,
            incumbent_obj=incumbent_obj,
        )
        best_obj = np.inf
        best_control = projected_center.copy()
        best_diag: Dict[str, float] = {}
        for _candidate, obj, control, diag in coarse_results:
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        raw_probes = sensitivity_probe_candidates(
            self.cfg,
            projected_previous,
            best_control,
            authority="proposed",
        )
        probes = self._project_leader_conditioned_candidates(
            raw_probes,
            leader,
            allocation_plan,
            "leader_probe",
        )
        probes = self._augment_leader_target_net_inflow_candidates(
            state,
            probes,
            forecast,
            leader,
            allocation_plan,
            "leader_probe",
        )
        probe_results = self._evaluate_grid_stage(
            state,
            probes,
            forecast,
            leader,
            incumbent_obj=incumbent_obj,
        )
        probe_scores: list[tuple[GridControlCandidate, float]] = []
        for candidate, obj, control, diag in probe_results:
            probe_scores.append((candidate, obj))
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        raw_directions = sensitivity_direction_candidates(
            self.cfg,
            projected_previous,
            best_control,
            probe_scores,
            authority="proposed",
            base_objective=best_obj,
        )
        directions = self._project_leader_conditioned_candidates(
            raw_directions,
            leader,
            allocation_plan,
            "leader_direction",
        )
        directions = self._augment_leader_target_net_inflow_candidates(
            state,
            directions,
            forecast,
            leader,
            allocation_plan,
            "leader_direction",
        )
        direction_results = self._evaluate_grid_stage(
            state,
            directions,
            forecast,
            leader,
            incumbent_obj=min(best_obj, incumbent_obj),
        )
        for _candidate, obj, control, diag in direction_results:
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        fine = self._leader_conditioned_grid_candidates(
            projected_previous,
            best_control,
            leader,
            allocation_plan,
            stage="fine",
            scope="local",
            state=state,
            forecast=forecast,
        )
        fine_results = self._evaluate_grid_stage(
            state,
            fine,
            forecast,
            leader,
            incumbent_obj=min(best_obj, incumbent_obj),
        )
        for _candidate, obj, control, diag in fine_results:
            if self._response_is_better(obj, diag, best_obj, best_diag):
                best_obj = float(obj)
                best_control = control
                best_diag = diag

        if not np.isfinite(best_obj):
            return projected_center.copy(), np.inf, {}
        target = max(0.0, float(leader.N_UF_star))
        rm_sum = sum(best_control.ramp_metering.get(ramp, 0.0) for ramp in self.cfg.network.ramps)
        stage_results = [coarse_results, probe_results, direction_results, fine_results]

        def stage_diag_sum(key: str) -> float:
            return float(sum(
                stage[0][3].get(key, 0.0)
                for stage in stage_results
                if stage
            ))

        best_diag.update({
            "distributed_grid_search_active": 1.0,
            "distributed_grid_leader_conditioned": 1.0,
            "distributed_grid_full_search_active": 1.0,
            "distributed_grid_parallel_stages": 4.0,
            "distributed_grid_stage1_candidates": float(len(coarse)),
            "distributed_grid_stage2_candidates": float(len(fine)),
            "distributed_grid_sensitivity_probe_candidates": float(len(probes)),
            "distributed_grid_sensitivity_direction_candidates": float(len(directions)),
            "distributed_grid_total_candidates": float(
                len(coarse) + len(probes) + len(directions) + len(fine)
            ),
            "distributed_grid_early_terminated_candidates": float(sum(
                result[3].get("distributed_grid_early_terminated", 0.0)
                for result in coarse_results + probe_results + direction_results + fine_results
            )),
            "distributed_grid_precheck_filtered_candidates": stage_diag_sum(
                "distributed_grid_precheck_filtered_candidates"
            ),
            "distributed_grid_precheck_evaluated_candidates": stage_diag_sum(
                "distributed_grid_precheck_evaluated_candidates"
            ),
            "distributed_grid_leader_target_metering_veh_h": target,
            "distributed_grid_leader_selected_metering_sum_veh_h": float(rm_sum),
            "distributed_grid_leader_metering_sum_error_veh_h": float(rm_sum - target),
            "distributed_grid_leader_incumbent_active": float(np.isfinite(incumbent_obj)),
            "distributed_grid_leader_incumbent_objective": float(incumbent_obj if np.isfinite(incumbent_obj) else 0.0),
            "distributed_grid_global_refresh": float(global_refresh),
            "distributed_grid_scope_global": float(coarse_scope == "global"),
            "distributed_grid_scope_local": float(coarse_scope == "local"),
            "distributed_grid_refresh_interval_sec": float(refresh_sec),
            "distributed_grid_authority_proposed": 1.0,
        })
        return best_control, float(best_obj), best_diag

    def _simplified_allocation_seed_refinement(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        center: ControlAction,
        leader: LeaderAction,
        allocation_plan: AllocationResult,
        incumbent_obj: float = np.inf,
    ) -> tuple[ControlAction, float, Dict[str, float]]:
        """간소화 allocation ablation: broad grid 없이 allocation seed 하나만 평가한다."""
        projected = self._project_control_to_leader_constraints(center, leader, allocation_plan)
        candidate = GridControlCandidate(
            label="leader_simplified_allocation_seed",
            control=projected,
            stage="allocation_seed",
            scope="allocation",
        )
        _precheck_feasible, precheck_diag = self._grid_feasibility_precheck(
            state,
            candidate,
            forecast,
            leader,
        )
        _candidate, obj, control, diag = self._rollout_grid_objective(
            state,
            candidate,
            forecast,
            leader,
            incumbent_obj=incumbent_obj,
            precheck_diag=precheck_diag,
        )
        diag.update({
            "distributed_grid_search_active": 0.0,
            "distributed_grid_leader_conditioned": 1.0,
            "distributed_grid_full_search_active": 0.0,
            "distributed_grid_total_candidates": 1.0,
            "distributed_grid_stage1_candidates": 1.0,
            "distributed_grid_stage2_candidates": 0.0,
            "distributed_grid_sensitivity_probe_candidates": 0.0,
            "distributed_grid_sensitivity_direction_candidates": 0.0,
            "distributed_simplified_allocation_seed_active": 1.0,
            "distributed_grid_leader_allocation_module_active": 1.0,
            "distributed_grid_leader_allocation_module_disabled": 0.0,
            f"stackelberg_allocation_mode_{getattr(self.cfg.mpc, 'stackelberg_allocation_mode', 'direct')}": 1.0,
        })
        return control, float(obj), diag

    def solve(
        self,
        state: TrafficState,
        leader: Optional[LeaderAction],
        demand: DemandStep | Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
        leader_incumbent_obj: float = np.inf,
    ) -> NashResult:
        """leader=None이면 PROPOSED-FOLLOWERS-ONLY(spec 16.7, 2026-06-13 재정의) —
        allocation module 미사용, urban agent는 green 자유탐색 + offset, freeway agent는
        local objective로 metering/VSL을 결정한다. 숨은 전역 목표 없음."""
        forecast = [demand] if isinstance(demand, DemandStep) else list(demand)
        if not forecast:
            raise ValueError("DistributedCoordinator requires at least one demand step.")
        self._repair_diagnostics = {}
        first_demand = forecast[0]
        if previous_control is not None:
            reference_control = previous_control.copy()
        else:
            # leaderless 초기 기준은 물리적 no-control(allocation 비움) — fixed()의
            # 0.5cap allocation이 숨은 게이팅으로 남지 않게 한다.
            reference_control = (
                ControlAction.uncontrolled(self.cfg) if leader is None else ControlAction.fixed(self.cfg)
            )
        reference_control = self._apply_green_vsl_only_authority(reference_control)
        current = reference_control.copy()
        current.N_P_star = leader.N_P_star if leader is not None else 0.0
        current.N_UF_star = leader.N_UF_star if leader is not None else 0.0
        # 기본 Stackelberg path는 direct feasible set을 유지한다. 실험 플래그가 켜진
        # 경우에만 deterministic allocation plan을 follower candidate center로 쓴다.
        allocation_plan = self._stackelberg_allocation_plan(state, leader, forecast)
        if leader is not None:
            allocation_map = self._allocation_control_map(allocation_plan)
            reference_control.inflow_outflow_allocation = dict(allocation_map)
            current.inflow_outflow_allocation = dict(allocation_map)
            if allocation_plan is not None:
                reference_control.diagnostics.update({
                    key: float(value)
                    for key, value in allocation_plan.diagnostics.items()
                    if isinstance(value, (int, float, bool))
                })
                current.diagnostics.update(reference_control.diagnostics)
        coupling = self._extract_coupling(state, current, first_demand)
        best_control = current.copy()
        best_obj = np.inf
        best_diag: Dict[str, float] = {}
        last_solver_diag: Dict[str, float] = {}
        residual = np.inf
        converged = False
        iteration = 0
        guard_flags: Dict[str, float] = {}
        selected_guard_label = ""
        if leader is None:
            guard_candidates = self._full_controller_guard_candidates(current)
            guard_flags["distributed_full_controller_guard_active"] = 1.0
            guard_flags["distributed_guard_candidate_count"] = float(len(guard_candidates))
            guard_flags["distributed_guard_selected"] = 0.0
            for label, _guard in guard_candidates:
                guard_flags[f"distributed_{label}_guard_evaluated"] = 0.0
                guard_flags[f"distributed_guard_selected_{label}"] = 0.0
            for label, guard in guard_candidates:
                guard_obj, guard_diag = self._response_tts_objective(
                    state,
                    guard,
                    forecast,
                    residual=0.0,
                    proxy_objective=0.0,
                )
                guard_flags[f"distributed_{label}_guard_evaluated"] = 1.0
                guard_flags[f"distributed_{label}_guard_objective_tts"] = float(guard_obj)
                guard_flags[f"distributed_{label}_guard_spillback_violation_veh"] = float(
                    guard_diag.get("distributed_response_total_spillback_violation_veh", 0.0)
                )
                if self._response_is_better(guard_obj, guard_diag, best_obj, best_diag):
                    selected_flags = dict(guard_flags)
                    selected_flags["distributed_guard_selected"] = 1.0
                    for candidate_label, _candidate_guard in guard_candidates:
                        selected_flags[f"distributed_guard_selected_{candidate_label}"] = float(
                            label == candidate_label
                        )
                    guard_diag.update(selected_flags)
                    best_obj = float(guard_obj)
                    best_control = guard.copy()
                    best_diag = guard_diag
                    selected_guard_label = label
            if selected_guard_label:
                selected_flags = dict(guard_flags)
                selected_flags["distributed_guard_selected"] = 1.0
                for candidate_label, _candidate_guard in guard_candidates:
                    selected_flags[f"distributed_guard_selected_{candidate_label}"] = float(
                        selected_guard_label == candidate_label
                    )
                best_diag.update(selected_flags)

        if leader is None:
            grid_control, grid_obj, grid_diag = self._structured_grid_refinement(
                state,
                forecast,
                reference_control,
                current,
                leader,
            )
        elif allocation_plan is not None:
            grid_control, grid_obj, grid_diag = self._simplified_allocation_seed_refinement(
                state,
                forecast,
                current,
                leader,
                allocation_plan,
                incumbent_obj=leader_incumbent_obj,
            )
        else:
            grid_control, grid_obj, grid_diag = self._leader_conditioned_grid_refinement(
                state,
                forecast,
                reference_control,
                current,
                leader,
                allocation_plan,
                incumbent_obj=leader_incumbent_obj,
            )
        if grid_diag:
            grid_diag.update(guard_flags)
            if self._response_is_better(grid_obj, grid_diag, best_obj, best_diag):
                best_obj = float(grid_obj)
                best_control = grid_control.copy()
                best_diag = dict(grid_diag)
            current = grid_control.copy()
            coupling = self._extract_coupling(state, current, first_demand)

        for iteration in range(1, self.cfg.mpc.max_nash_iter + 1):
            # FIXED_* ablation: coupling player의 strategic 결정을 고정 정책으로 대체.
            # physical subsystem은 그대로 — strategic controller role만 제거(plan §11).
            freeway_solves = [
                self._fixed_freeway_solve(agent) if self._freeway_player_fixed(agent)
                else self._solve_freeway_agent(agent, state, leader, forecast, current, coupling)
                for agent in self.freeway_agents
            ]
            freeway_response = self._freeway_response(freeway_solves)
            urban_solves = [
                self._fixed_urban_solve(agent) if self._urban_player_fixed(agent)
                else self._solve_urban_agent(
                    agent,
                    state,
                    leader,
                    forecast,
                    freeway_response,
                    current,
                    allocation_plan,
                    coupling,
                )
                for agent in self.urban_agents
            ]
            candidate = self._merge_agent_controls(
                leader,
                current,
                freeway_solves,
                urban_solves,
            )
            candidate.offsets = self._clamp_offsets_to_reference(candidate.offsets, reference_control)
            candidate.vsl = self._clamp_vsl_to_reference(candidate.vsl, reference_control)
            candidate = self._apply_green_vsl_only_authority(candidate)
            nash_target_projection_diag: Dict[str, float] = {}
            if leader is not None:
                candidate, nash_target_projection_diag = self._leader_target_net_inflow_projected_control(
                    state,
                    candidate,
                    forecast,
                    leader,
                    allocation_plan,
                )
            new_coupling = self._extract_coupling(state, candidate, first_demand)
            residual = self._coupling_residual(coupling, new_coupling)
            proxy_obj = sum(s.objective for s in freeway_solves) + sum(s.objective for s in urban_solves)
            diagnostics = self._diagnostics(freeway_solves, urban_solves, residual, iteration)
            if leader is None:
                obj, response_diag = self._response_tts_objective(
                    state,
                    candidate,
                    forecast,
                    residual=residual,
                    proxy_objective=proxy_obj,
                )
            else:
                nash_candidate = GridControlCandidate(
                    label=f"nash_iteration_{iteration}",
                    control=candidate,
                    stage="nash",
                    scope="local",
                )
                _precheck_feasible, precheck_diag = self._grid_feasibility_precheck(
                    state,
                    nash_candidate,
                    forecast,
                    leader,
                )
                _candidate, obj, candidate, response_diag = self._rollout_grid_objective(
                    state,
                    nash_candidate,
                    forecast,
                    leader,
                    incumbent_obj=best_obj,
                    precheck_diag=precheck_diag,
                )
                response_diag.update({
                    "distributed_nash_candidate_rollout_evaluated": 1.0,
                    "distributed_nash_candidate_target_green_projection_active": 1.0,
                    "distributed_nash_candidate_proxy_objective": float(proxy_obj),
                    "distributed_nash_candidate_iteration": float(iteration),
                })
                response_diag.update(nash_target_projection_diag)
            diagnostics.update(response_diag)
            diagnostics.update(guard_flags)
            last_solver_diag = diagnostics
            if self._response_is_better(obj, diagnostics, best_obj, best_diag):
                best_obj = float(obj)
                best_control = candidate
                best_diag = diagnostics
            current = candidate
            coupling = new_coupling
            if residual < self.cfg.mpc.distributed_coupling_tol:
                converged = True
                break

        if last_solver_diag and "distributed_player_active" not in best_diag:
            merged_diag = dict(last_solver_diag)
            merged_diag.update(best_diag)
            best_diag = merged_diag
        best_control.diagnostics.update(best_diag)
        if leader is not None:
            if allocation_plan is None:
                best_control.inflow_outflow_allocation = {}
                best_control.diagnostics["distributed_grid_leader_allocation_module_disabled"] = 1.0
                best_control.diagnostics["distributed_grid_leader_allocation_module_active"] = 0.0
                best_control.diagnostics["distributed_stackelberg_direct_feasible_set_active"] = 1.0
                best_control.diagnostics["stackelberg_allocation_mode_direct"] = 1.0
            else:
                best_control.inflow_outflow_allocation.update(self._allocation_control_map(allocation_plan))
                best_control.diagnostics.update({
                    key: float(value)
                    for key, value in allocation_plan.diagnostics.items()
                    if isinstance(value, (int, float, bool))
                })
                mode = str(getattr(self.cfg.mpc, "stackelberg_allocation_mode", "direct"))
                best_control.diagnostics["distributed_grid_leader_allocation_module_disabled"] = 0.0
                best_control.diagnostics["distributed_grid_leader_allocation_module_active"] = 1.0
                best_control.diagnostics["distributed_stackelberg_direct_feasible_set_active"] = 0.0
                best_control.diagnostics[f"stackelberg_allocation_mode_{mode}"] = 1.0
        best_control.diagnostics["nash_converged"] = converged
        best_control.diagnostics["nash_iterations"] = iteration
        return NashResult(
            control=best_control,
            objective_value=float(best_obj if np.isfinite(best_obj) else 0.0),
            iterations=iteration,
            converged=converged,
            residual_objective=float(residual if np.isfinite(residual) else 0.0),
            residual_control=float(residual if np.isfinite(residual) else 0.0),
            diagnostics=best_diag,
        )

    def _append_metering_candidate(
        self,
        candidates: list[Dict[str, float]],
        seen: set[tuple[float, ...]],
        release: Mapping[str, float],
        upper: Mapping[str, float],
        ramps: Iterable[str],
        clip_to_upper: bool = True,
    ) -> bool:
        """Spec 18.6/18.7: ramp metering guard/proposal 후보를 feasible flow로 보정해 추가한다."""
        net = self.cfg.network
        values = {
            ramp: float(np.clip(
                release.get(ramp, 0.0),
                0.0,
                upper.get(ramp, 0.0) if clip_to_upper else net.ramp_capacity_veh_h[ramp],
            ))
            for ramp in ramps
        }
        for ramp in ramps:
            upper_bound = upper.get(ramp, 0.0) if clip_to_upper else net.ramp_capacity_veh_h[ramp]
            if upper_bound <= 1.0e-9:
                continue
            lower_bound = self.cfg.freeway_follower.ramp_metering_rate_min * net.ramp_capacity_veh_h[ramp]
            values[ramp] = float(max(values[ramp], min(lower_bound, upper_bound)))
        key = tuple(round(values[ramp], 6) for ramp in ramps)
        if key in seen:
            return False
        seen.add(key)
        candidates.append(values)
        return True

    def _metering_candidates(
        self,
        agent: AgentSpec,
        upper: Mapping[str, float],
        weights: Mapping[str, float],
        target: float,
        current: ControlAction,
        spillback_min_release: Optional[Mapping[str, float]] = None,
    ) -> list[Dict[str, float]]:
        """Spec 18.7: no-control, previous, target projection과 작은 주변 후보를 만든다."""
        if not agent.ramps:
            return [{}]
        ramps = tuple(agent.ramps)
        candidates: list[Dict[str, float]] = []
        seen: set[tuple[float, ...]] = set()
        self._append_metering_candidate(
            candidates,
            seen,
            {r: self.cfg.network.ramp_capacity_veh_h[r] for r in ramps},
            upper,
            ramps,
            clip_to_upper=False,
        )
        self._append_metering_candidate(
            candidates,
            seen,
            {r: current.ramp_metering.get(r, self.cfg.network.ramp_capacity_veh_h[r]) for r in ramps},
            upper,
            ramps,
            clip_to_upper=False,
        )
        projected = _project_to_target(target, upper, weights)
        self._append_metering_candidate(candidates, seen, projected, upper, ramps)
        if spillback_min_release and any(value > 1.0e-9 for value in spillback_min_release.values()):
            self._append_metering_candidate(
                candidates,
                seen,
                spillback_min_release,
                upper,
                ramps,
                clip_to_upper=False,
            )
            self._append_metering_candidate(
                candidates,
                seen,
                {
                    ramp: max(projected.get(ramp, 0.0), spillback_min_release.get(ramp, 0.0))
                    for ramp in ramps
                },
                upper,
                ramps,
                clip_to_upper=False,
            )
        # Keep explicit intermediate metering regimes in the final argmin set.
        # The leaderless target can collapse toward 0.5 * upper, while medium
        # demand needs rates around 0.7-0.9 * upper to avoid mainline breakdown.
        for fraction in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
            self._append_metering_candidate(
                candidates,
                seen,
                {ramp: fraction * upper.get(ramp, 0.0) for ramp in ramps},
                upper,
                ramps,
            )
        for factor in (0.85, 1.15):
            self._append_metering_candidate(
                candidates,
                seen,
                _project_to_target(target * factor, upper, weights),
                upper,
                ramps,
            )
        return candidates

    def _onramp_spillback_min_release_rates(
        self,
        state: TrafficState,
        agent: AgentSpec,
        ramp_arrivals_veh: Mapping[str, float],
        horizon_h: float,
    ) -> Dict[str, float]:
        rates: Dict[str, float] = {}
        for ramp in agent.ramps:
            zero_release = assess_onramp_spillback(
                state,
                self.cfg,
                ramp,
                ramp_arrivals_veh.get(ramp, 0.0),
                0.0,
            )
            rates[ramp] = min(
                self.cfg.network.ramp_capacity_veh_h[ramp],
                zero_release.violation_veh / max(horizon_h, 1.0e-9),
            )
        return rates

    def _agent_queue_tts_terms(
        self,
        agent: AgentSpec,
        state: TrafficState,
        ramp_metering: Mapping[str, float],
        coupling: Mapping[str, float],
        horizon_h: float,
    ) -> tuple[float, float]:
        """Ramp reservoir와 upstream urban queue의 후보별 TTS 근사[veh*h]."""
        net = self.cfg.network
        ramp_start = sum(max(0.0, state.ramp_queue.get(ramp, 0.0)) for ramp in agent.ramps)
        incoming = sum(max(0.0, float(coupling.get(f"u_on_{ramp}", 0.0))) * horizon_h for ramp in agent.ramps)
        release = sum(max(0.0, ramp_metering.get(ramp, 0.0)) * horizon_h for ramp in agent.ramps)
        capacity = net.ramp_queue_max_veh * max(len(agent.ramps), 1)
        ramp_terminal_unclipped = max(0.0, ramp_start + incoming - release)
        ramp_terminal = min(capacity, ramp_terminal_unclipped)
        blocked_to_urban = max(0.0, ramp_terminal_unclipped - capacity)
        ramp_tts = 0.5 * (ramp_start + ramp_terminal) * horizon_h
        # Existing urban on-ramp approach queues are already charged by the
        # urban model. The freeway agent only prices additional upstream
        # spillback caused by filling the ramp reservoir under this candidate.
        urban_tts = 0.5 * blocked_to_urban * horizon_h
        return float(ramp_tts), float(urban_tts)

    def _candidate_freeway_tts_terms(
        self,
        agent: AgentSpec,
        state: TrafficState,
        ramp_metering: Mapping[str, float],
        upper: Mapping[str, float],
        forecast: list[DemandStep],
        lane_profile: Mapping[str, list[float]],
    ) -> tuple[float, float, float, float, float]:
        """Approximate candidate-dependent freeway TTS from predicted merge rho."""
        net = self.cfg.network
        horizon_steps = forecast[: max(1, self.cfg.mpc.horizon_steps)]
        horizon_h = self.cfg.simulation.T_c_h * max(1, len(horizon_steps))
        all_rhos = list(state.freeway_density.get(agent.link, []))
        if not all_rhos:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        lanes_for_link = lane_profile.get(agent.link, [float(net.freeway_lanes) for _ in all_rhos])

        if not agent.ramps:
            vehicle_tts = 0.0
            density_excess_tts = 0.0
            peak_density = 0.0
            if 0 <= agent.segment_index < len(all_rhos):
                indices = [agent.segment_index]
            else:
                indices = list(range(len(all_rhos)))
            for idx in indices:
                rho = all_rhos[idx]
                lane = float(lanes_for_link[idx] if idx < len(lanes_for_link) else net.freeway_lanes)
                vehicle_tts += max(0.0, rho) * net.freeway_segment_length_km * max(lane, 1.0e-9) * horizon_h
                density_excess_tts += max(0.0, rho - net.rho_crit) * horizon_h
                peak_density = max(peak_density, float(rho))
            return float(vehicle_tts), float(density_excess_tts), float(peak_density), float(peak_density), 0.0

        idx = agent.segment_index if 0 <= agent.segment_index < len(all_rhos) else len(all_rhos) // 2
        rho = max(0.0, float(all_rhos[idx]))
        speed_values = state.freeway_speed.get(agent.link, [])
        speed = max(net.v_min, float(speed_values[idx] if idx < len(speed_values) else net.v_free))
        lane = max(1.0e-9, float(lanes_for_link[idx] if idx < len(lanes_for_link) else net.freeway_lanes))
        segment_veh_per_density = net.freeway_segment_length_km * lane
        release = sum(
            min(
                max(0.0, float(ramp_metering.get(ramp, 0.0))),
                max(0.0, float(upper.get(ramp, ramp_metering.get(ramp, 0.0)))),
            )
            for ramp in agent.ramps
        )
        flows = state.freeway_flow.get(agent.link, [])
        upstream_flow = (
            max(0.0, float(flows[idx - 1]))
            if idx > 0 and idx - 1 < len(flows)
            else max(0.0, float(horizon_steps[0].freeway_mainline.get(agent.link, 0.0)))
        )
        dt_h = self.cfg.simulation.T_c_h
        vehicle_tts = 0.0
        density_excess_tts = 0.0
        peak_density = rho
        for step in horizon_steps:
            q_upstream = upstream_flow if idx > 0 else max(0.0, float(step.freeway_mainline.get(agent.link, 0.0)))
            q_out = rho * speed * lane
            rho_next = max(
                0.0,
                min(
                    net.rho_max,
                    rho + (q_upstream + release - q_out) * dt_h / max(segment_veh_per_density, 1.0e-9),
                ),
            )
            vehicle_tts += 0.5 * (rho + rho_next) * segment_veh_per_density * dt_h
            density_excess_tts += 0.5 * (
                max(0.0, rho - net.rho_crit) + max(0.0, rho_next - net.rho_crit)
            ) * dt_h
            rho = rho_next
            peak_density = max(peak_density, rho)
        return (
            float(vehicle_tts),
            float(density_excess_tts),
            float(rho),
            float(peak_density),
            float(release),
        )

    def _vsl_pressure_proposal(
        self,
        rhos: list[float],
        previous_vsl: float,
        lane_loss: float,
        neighbor_pressure: float,
        offramp_storage_veh: float,
        offramp_capacity_veh: float,
    ) -> float:
        """Pressure rule은 최종 선택이 아니라 relaxed VSL 후보 중심으로만 사용한다."""
        vsl_set = sorted(float(v) for v in self.cfg.freeway_follower.vsl_set)
        density_ratio = (max(rhos) / max(self.cfg.network.rho_crit, 1.0e-9)) if rhos else 0.0
        storage_ratio = offramp_storage_veh / max(offramp_capacity_veh, 1.0e-9)
        pressure = max(0.0, density_ratio - 0.9) + 0.02 * neighbor_pressure + 0.5 * lane_loss + storage_ratio
        target = max(vsl_set) - 25.0 * min(2.0, pressure)
        return float(0.65 * target + 0.35 * previous_vsl)

    def _solve_freeway_agent(
        self,
        agent: AgentSpec,
        state: TrafficState,
        leader: Optional[LeaderAction],
        forecast: list[DemandStep],
        current: ControlAction,
        coupling: Mapping[str, float],
    ) -> AgentSolve:
        net = self.cfg.network
        dt_h = self.cfg.simulation.T_f_h
        demand = forecast[0]
        lane_profile, lane_diag = effective_lane_profile(state, self.cfg, demand)
        neighbor_pressure = self._freeway_neighbor_pressure(agent, state, coupling, lane_profile)
        neighbor_metering_factor = 1.0 - 0.15 * float(np.clip(
            neighbor_pressure / max(2.0 * net.rho_crit, 1.0e-9),
            0.0,
            1.0,
        ))
        link_capacity = sum(net.ramp_capacity_veh_h[ramp] for ramp in agent.ramps)
        total_capacity = max(sum(net.ramp_capacity_veh_h.values()), 1.0e-9)
        upper: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        receiving_limit: Dict[str, float] = {}
        min_receiving = 1.0
        for ramp in agent.ramps:
            merge_idx = agent.segment_index if agent.segment_index >= 0 else len(state.freeway_density[agent.link]) // 2
            rho_merge = state.freeway_density[agent.link][merge_idx]
            receiving = float(np.clip(
                (net.rho_max - rho_merge) / max(net.rho_max - net.rho_crit, 1.0e-9),
                0.0,
                1.0,
            ))
            min_receiving = min(min_receiving, receiving)
            # NO_U_TO_F/LOCAL_ONLY ablation: urban 예측 방출 정보 차단 — 측정된 현재
            # w_r만 사용(zero-order hold). 물리 차량 이동은 plant에서 그대로 일어난다.
            urban_release = 0.0 if self._block_u_to_f(agent) else max(0.0, coupling.get(f"u_on_{ramp}", 0.0))
            available = state.ramp_queue.get(ramp, 0.0) / max(dt_h, 1.0e-9) + urban_release
            receiving_limit[ramp] = min(
                net.ramp_capacity_veh_h[ramp],
                net.freeway_capacity_veh_h * receiving * neighbor_metering_factor,
            )
            upper[ramp] = min(
                net.ramp_capacity_veh_h[ramp],
                available,
                receiving_limit[ramp],
            )
            weights[ramp] = state.ramp_queue.get(ramp, 0.0) + urban_release * self.cfg.simulation.T_c_h + 1.0
        if leader is not None:
            target = max(0.0, leader.N_UF_star) * link_capacity / total_capacity
        else:
            # leaderless(spec 16.7): 전역 N_UF 목표 없이 agent가 local objective로 방출
            # 수준을 고른다 — 후보 분율을 1-구획 merge 밀도 예측으로 평가해 최소 비용 선택.
            target = self._leaderless_metering_target(agent, state, upper, demand)
        all_rhos = state.freeway_density.get(agent.link, [])
        rhos = [all_rhos[agent.segment_index]] if 0 <= agent.segment_index < len(all_rhos) else all_rhos
        lanes_for_link = lane_profile.get(agent.link, [net.freeway_lanes])
        lane_idx = agent.segment_index if 0 <= agent.segment_index < len(lanes_for_link) else len(lanes_for_link) - 1
        lane_loss = max(0.0, net.freeway_lanes - lanes_for_link[lane_idx])
        # off-ramp 램프 storage 재귀속(design 2026-06-17): 이 freeway link에서 갈라지는
        # off-ramp storage 점유[veh]를 계산해 freeway agent 자기 비용(objective)에 가산한다.
        offramp_storage_veh = 0.0
        offramp_capacity_veh = 0.0
        offramp_storage_pressure: Dict[str, float] = {}
        for off_ramp in net.off_ramps:
            if net.off_ramp_from_freeway.get(off_ramp) != agent.link:
                continue
            storage_link = net.off_ramp_storage_link.get(off_ramp, "")
            capacity = float(net.urban_link_storage_veh.get(storage_link, 0.0))
            if capacity <= 0.0:
                continue
            avail = float(state.urban_link_storage.get(storage_link, capacity))
            occupied = max(0.0, capacity - avail)
            offramp_storage_veh += occupied
            offramp_capacity_veh += capacity
            offramp_storage_pressure[off_ramp] = float(occupied / max(capacity, 1.0e-9))
        # forecast horizon에 걸친 off-ramp 예측 유입[veh] — VSL이 낮을수록 diverge
        # 도달량이 줄어 off-ramp storage 유입이 줄어드는 emergence를 후보 평가에 반영한다.
        offramp_forecast_by_ramp = self._forecast_offramp_arrivals_by_ramp(state, forecast, agent.link)
        offramp_forecast_veh = sum(offramp_forecast_by_ramp.values())
        prev_vsl = current.vsl.get(agent.link, max(self.cfg.freeway_follower.vsl_set))
        density_excess = sum(max(0.0, rho - net.rho_crit) for rho in rhos)
        # 잔차는 달성가능 목표(min(target, Σ물리상한)) 기준 — 수요 부족으로 덜 방출한 것을
        # "추적 실패"로 만들어 urban 쪽에 가짜 freeway 압력을 보내지 않게 한다.
        vsl_proposal = self._vsl_pressure_proposal(
            rhos,
            prev_vsl,
            lane_loss,
            neighbor_pressure,
            offramp_storage_veh,
            offramp_capacity_veh,
        )
        horizon_steps = forecast[: max(1, self.cfg.mpc.horizon_steps)]
        horizon_h = self.cfg.simulation.T_c_h * max(1, len(horizon_steps))
        ramp_arrivals_veh = ramp_arrivals_over_horizon(horizon_steps, self.cfg, tuple(agent.ramps))
        spillback_min_release = self._onramp_spillback_min_release_rates(
            state,
            agent,
            ramp_arrivals_veh,
            horizon_h,
        )
        metering_candidates = self._metering_candidates(
            agent,
            upper,
            weights,
            target,
            current,
            spillback_min_release,
        )
        if self._green_vsl_only_ttt_mode():
            metering_candidates = [self._no_metering_control(agent.ramps)] if agent.ramps else [{}]
        vsl_candidates = self._vsl_candidates(prev_vsl, vsl_proposal)
        target_feasible = min(target, sum(upper.values()))
        ramp_metering: Dict[str, float] = {}
        desired = float(prev_vsl)
        objective = float("inf")
        best_constraint_feasible = False
        best_spillback_violation = float("inf")
        metering_error = 0.0
        best_ramp_queue_tts = 0.0
        best_urban_queue_tts = 0.0
        best_onramp_spillback_violation = 0.0
        best_onramp_combined_terminal = 0.0
        best_onramp_combined_capacity = 0.0
        best_offramp_spillback_violation = 0.0
        best_offramp_combined_terminal = 0.0
        best_offramp_combined_capacity = 0.0
        best_projected_freeway_tts = 0.0
        best_projected_density_excess_tts = 0.0
        best_projected_terminal_density = max(rhos) if rhos else 0.0
        best_projected_peak_density = max(rhos) if rhos else 0.0
        best_projected_release_flow = 0.0
        joint_evals = 0
        feasible_evals = 0
        for ramp_candidate in metering_candidates:
            candidate_metering_error = abs(sum(ramp_candidate.values()) - target_feasible)
            (
                candidate_freeway_tts,
                candidate_density_excess_tts,
                candidate_terminal_density,
                candidate_peak_density,
                candidate_release_flow,
            ) = self._candidate_freeway_tts_terms(
                agent,
                state,
                ramp_candidate,
                upper,
                horizon_steps,
                lane_profile,
            )
            ramp_queue_tts, urban_queue_tts = self._agent_queue_tts_terms(
                agent,
                state,
                ramp_candidate,
                coupling,
                horizon_h,
            )
            onramp_assessments = [
                assess_onramp_spillback(
                    state,
                    self.cfg,
                    ramp,
                    ramp_arrivals_veh.get(ramp, 0.0),
                    min(
                        max(0.0, ramp_candidate.get(ramp, 0.0)),
                        max(0.0, upper.get(ramp, ramp_candidate.get(ramp, 0.0))),
                    )
                    * horizon_h,
                )
                for ramp in agent.ramps
            ]
            onramp_spillback_violation = sum(item.violation_veh for item in onramp_assessments)
            onramp_combined_terminal = sum(item.terminal_veh for item in onramp_assessments)
            onramp_combined_capacity = sum(item.capacity_veh for item in onramp_assessments)
            metering_smooth = sum(
                abs(ramp_candidate.get(r, 0.0) - current.ramp_metering.get(r, ramp_candidate.get(r, 0.0)))
                for r in agent.ramps
            )
            receiving_overrequest = sum(
                max(0.0, ramp_candidate.get(r, 0.0) - receiving_limit.get(r, 0.0))
                for r in agent.ramps
            )
            for vsl_candidate in vsl_candidates:
                vsl_fraction = self._offramp_release_fraction(vsl_candidate)
                offramp_assessments = [
                    assess_offramp_spillback(
                        state,
                        self.cfg,
                        off_ramp,
                        max(0.0, vehicles) * vsl_fraction,
                    )
                    for off_ramp, vehicles in offramp_forecast_by_ramp.items()
                ]
                offramp_spillback_violation = sum(item.violation_veh for item in offramp_assessments)
                offramp_combined_terminal = sum(item.terminal_veh for item in offramp_assessments)
                offramp_combined_capacity = sum(item.capacity_veh for item in offramp_assessments)
                spillback_violation = onramp_spillback_violation + offramp_spillback_violation
                constraint_feasible = spillback_violation <= 1.0e-9
                if constraint_feasible:
                    feasible_evals += 1
                cost = self._freeway_agent_objective(
                    rhos,
                    density_excess,
                    candidate_metering_error,
                    ramp_candidate,
                    vsl_candidate,
                    prev_vsl,
                    offramp_forecast_veh,
                    offramp_storage_veh,
                    offramp_capacity_veh,
                    ramp_queue_tts=ramp_queue_tts,
                    onramp_urban_queue_tts=urban_queue_tts,
                    horizon_h=horizon_h,
                    freeway_vehicle_tts=candidate_freeway_tts,
                    density_excess_tts=candidate_density_excess_tts,
                )
                cost += self.cfg.freeway_follower.metering_smoothness_weight * metering_smooth
                cost += self.cfg.freeway_follower.ramp_queue_penalty * receiving_overrequest * horizon_h
                cost += self.cfg.freeway_follower.ramp_queue_penalty * spillback_violation * horizon_h
                joint_evals += 1
                should_select = (
                    (constraint_feasible and not best_constraint_feasible)
                    or (
                        constraint_feasible
                        and best_constraint_feasible
                        and cost < objective - 1.0e-12
                    )
                    or (
                        not constraint_feasible
                        and not best_constraint_feasible
                        and (
                            spillback_violation < best_spillback_violation - 1.0e-9
                            or (
                                abs(spillback_violation - best_spillback_violation) <= 1.0e-9
                                and cost < objective - 1.0e-12
                            )
                        )
                    )
                )
                if should_select:
                    objective = float(cost)
                    best_constraint_feasible = bool(constraint_feasible)
                    best_spillback_violation = float(spillback_violation)
                    desired = float(vsl_candidate)
                    ramp_metering = dict(ramp_candidate)
                    metering_error = float(candidate_metering_error)
                    best_ramp_queue_tts = float(ramp_queue_tts)
                    best_urban_queue_tts = float(urban_queue_tts)
                    best_onramp_spillback_violation = float(onramp_spillback_violation)
                    best_onramp_combined_terminal = float(onramp_combined_terminal)
                    best_onramp_combined_capacity = float(onramp_combined_capacity)
                    best_offramp_spillback_violation = float(offramp_spillback_violation)
                    best_offramp_combined_terminal = float(offramp_combined_terminal)
                    best_offramp_combined_capacity = float(offramp_combined_capacity)
                    best_projected_freeway_tts = float(candidate_freeway_tts)
                    best_projected_density_excess_tts = float(candidate_density_excess_tts)
                    best_projected_terminal_density = float(candidate_terminal_density)
                    best_projected_peak_density = float(candidate_peak_density)
                    best_projected_release_flow = float(candidate_release_flow)
        vsl_fraction = self._offramp_release_fraction(desired)
        selected_offramp_arrival = {
            off_ramp: float(vehicles * vsl_fraction)
            for off_ramp, vehicles in offramp_forecast_by_ramp.items()
        }
        diagnostics = {
            f"agent_{agent.id}_density_excess": float(density_excess),
            f"agent_{agent.id}_metering_error": float(metering_error),
            f"agent_{agent.id}_min_receiving_factor": float(min_receiving),
            f"agent_{agent.id}_lane_loss": float(lane_loss),
            f"agent_{agent.id}_freeway_neighbor_pressure": float(neighbor_pressure),
            f"agent_{agent.id}_freeway_neighbor_metering_factor": float(neighbor_metering_factor),
            f"agent_{agent.id}_offramp_storage_veh": float(offramp_storage_veh),
            f"agent_{agent.id}_offramp_forecast_veh": float(offramp_forecast_veh),
            f"agent_{agent.id}_metering_candidates": float(len(metering_candidates)),
            f"agent_{agent.id}_vsl_candidates": float(len(vsl_candidates)),
            f"agent_{agent.id}_joint_candidate_evaluations": float(joint_evals),
            f"agent_{agent.id}_spillback_feasible_evaluations": float(feasible_evals),
            f"agent_{agent.id}_spillback_constraint_feasible": float(best_constraint_feasible),
            f"agent_{agent.id}_spillback_min_release_flow": float(sum(spillback_min_release.values())),
            f"agent_{agent.id}_candidate_density_projection_active": 1.0,
            f"agent_{agent.id}_projected_freeway_tts": float(best_projected_freeway_tts),
            f"agent_{agent.id}_projected_density_excess_tts": float(best_projected_density_excess_tts),
            f"agent_{agent.id}_projected_terminal_density": float(best_projected_terminal_density),
            f"agent_{agent.id}_projected_peak_density": float(best_projected_peak_density),
            f"agent_{agent.id}_projected_release_flow": float(best_projected_release_flow),
            f"agent_{agent.id}_onramp_combined_capacity_veh": float(best_onramp_combined_capacity),
            f"agent_{agent.id}_onramp_combined_terminal_veh": float(best_onramp_combined_terminal),
            f"agent_{agent.id}_onramp_spillback_violation_veh": float(best_onramp_spillback_violation),
            f"agent_{agent.id}_offramp_combined_capacity_veh": float(best_offramp_combined_capacity),
            f"agent_{agent.id}_offramp_combined_terminal_veh": float(best_offramp_combined_terminal),
            f"agent_{agent.id}_offramp_spillback_violation_veh": float(best_offramp_spillback_violation),
            f"agent_{agent.id}_default_metering_guard_evaluated": 1.0,
            f"agent_{agent.id}_green_vsl_only_no_metering": float(self._green_vsl_only_ttt_mode()),
            f"agent_{agent.id}_ramp_queue_tts": float(best_ramp_queue_tts),
            f"agent_{agent.id}_onramp_urban_queue_tts": float(best_urban_queue_tts),
            f"agent_{agent.id}_vsl_selected": float(desired),
        }
        for off_ramp, vehicles in selected_offramp_arrival.items():
            diagnostics[f"agent_{agent.id}_offramp_selected_arrival_{off_ramp}_veh"] = float(vehicles)
            diagnostics[f"agent_{agent.id}_offramp_selected_flow_{off_ramp}"] = float(
                vehicles / max(horizon_h, 1.0e-9)
            )
            diagnostics[f"agent_{agent.id}_offramp_storage_pressure_{off_ramp}"] = float(
                offramp_storage_pressure.get(off_ramp, 0.0)
            )
        diagnostics.update({f"agent_{agent.id}_{key}": value for key, value in lane_diag.items()})
        infeasibility = {
            "metering_tracking_residual": float(metering_error),
            "density_excess": float(density_excess),
            "min_ramp_receiving_factor": float(min_receiving),
            "ramp_projection_first_step_capacity": float(sum(upper.values())),
            "spillback_constraint_feasible": float(best_constraint_feasible),
            "onramp_spillback_violation_veh": float(best_onramp_spillback_violation),
            "offramp_spillback_violation_veh": float(best_offramp_spillback_violation),
            "total_spillback_violation_veh": float(
                best_onramp_spillback_violation + best_offramp_spillback_violation
            ),
        }
        for off_ramp, vehicles in selected_offramp_arrival.items():
            infeasibility[f"offramp_predicted_arrival_{off_ramp}_veh"] = float(vehicles)
            infeasibility[f"offramp_predicted_flow_{off_ramp}"] = float(vehicles / max(horizon_h, 1.0e-9))
            infeasibility[f"offramp_storage_pressure_{off_ramp}"] = float(
                offramp_storage_pressure.get(off_ramp, 0.0)
            )
        return AgentSolve(
            agent_id=agent.id,
            objective=float(objective),
            ramp_metering=ramp_metering,
            vsl={agent.link: desired},
            infeasibility=infeasibility,
            diagnostics=diagnostics,
        )

    def _leaderless_metering_target(
        self,
        agent: AgentSpec,
        state: TrafficState,
        upper: Mapping[str, float],
        demand: DemandStep,
    ) -> float:
        """leaderless freeway agent의 국소 metering 수준 선택.

        후보 = Σupper의 분율 {1.0, 0.85, 0.7, 0.5}. 1-구획 근사로 한 control interval 뒤
        merge 밀도를 예측해 비용 = density_penalty×pos(ρ_pred−ρ_crit) + 잡아둔 차량의
        대기비용(veh·h)으로 평가한다 — 전역 목표 없이 자기 목적만 사용(spec 16.7)."""
        net = self.cfg.network
        dt_h = self.cfg.simulation.T_c_h
        total_upper = sum(max(0.0, v) for v in upper.values())
        if total_upper <= 1.0e-9 or not agent.ramps:
            return total_upper
        merge_idx = agent.segment_index if agent.segment_index >= 0 else len(state.freeway_density[agent.link]) // 2
        rho_merge = state.freeway_density[agent.link][merge_idx]
        speed = max(state.freeway_speed[agent.link][merge_idx], net.v_min)
        seg_cap_veh = net.freeway_segment_length_km * net.freeway_lanes
        q_out = rho_merge * speed * net.freeway_lanes
        if merge_idx > 0:
            q_upstream = max(0.0, state.freeway_flow[agent.link][merge_idx - 1])
        else:
            q_upstream = max(0.0, demand.freeway_mainline.get(agent.link, 0.0))
        # ramp/on-ramp 큐 비용을 제대로 가격화한다(진단 문서 §"Relation To Wu"): 이미 큐가
        # 쌓인 ramp에 metering을 더 하면 큐 대기손실이 비선형으로 커진다. no-metering(=용량
        # 방출, fraction=1.0)을 보호 baseline 후보로 명시 — 국소 density만으로 과도하게
        # metering해 TTT가 악화되지 않게 한다.
        existing_ramp_queue = sum(max(0.0, state.ramp_queue.get(r, 0.0)) for r in agent.ramps)
        ramp_queue_max = max(net.ramp_queue_max_veh * max(len(agent.ramps), 1), 1.0e-9)
        queue_saturation = min(1.0, existing_ramp_queue / ramp_queue_max)
        best_target, best_cost = total_upper, float("inf")  # baseline = no-metering(용량 방출).
        for fraction in (1.0, 0.85, 0.7, 0.5):
            release = fraction * total_upper
            # Spec 3.1.2 conservation: merge 유입은 본선 상류 유량과 ramp release의 합이다.
            rho_pred = max(
                0.0,
                rho_merge + (q_upstream + release - q_out) * dt_h / max(seg_cap_veh, 1.0e-9),
            )
            held = (total_upper - release) * dt_h  # 잡아둔 차량수[veh] — 대기비용으로 환산.
            # 기존 ramp 큐가 포화에 가까울수록 추가로 잡아두는 비용을 가중(spillback 위험).
            held_cost = held * (1.0 + queue_saturation)
            cost = (
                self.cfg.freeway_follower.density_penalty * max(0.0, rho_pred - net.rho_crit)
                + held_cost
            )
            # 동률(자유류 ρ_pred<ρ_crit, 모든 cost 동일)이면 no-metering을 보호: strict 비교라
            # fraction=1.0이 먼저 best로 잡혀 유지된다.
            if cost < best_cost - 1.0e-12:
                best_cost, best_target = cost, release
        return float(best_target)

    def _vsl_candidates(self, previous_vsl: float, proposal: Optional[float] = None) -> list[float]:
        """이 control interval에 freeway agent가 고를 수 있는 VSL 후보 집합[km/h].

        full 모드: vsl_set 중 직전 VSL ±max_vsl_step 안에 드는 discrete 값(보통 3~5개,
        Cartesian 폭증 없이 per-link 1차원). relaxed-quantized 모드: 연속 target(=max,
        한 단계 낮춤, 두 단계 낮춤)을 공통 repair로 양자화해 소수 생성. 어느 모드든
        후보 수가 vsl_set 크기를 넘지 않는다(Nash 루프 비용 bound)."""
        fc = self.cfg.freeway_follower
        vsl_set = sorted(float(v) for v in fc.vsl_set)
        if not self.cfg.mpc.relaxed_quantized_controls:
            feasible = [
                v for v in vsl_set
                if previous_vsl - fc.max_vsl_step - 1.0e-9 <= v <= previous_vsl + fc.max_vsl_step + 1.0e-9
            ]
            return feasible or vsl_set
        max_vsl = max(vsl_set)
        step = max(1.0e-9, self.cfg.mpc.relaxed_vsl_quantum_km_h)
        out: list[float] = []
        center = float(previous_vsl if proposal is None else proposal)
        for raw in (max_vsl, previous_vsl, center, center - step, center + step):
            repaired = repair_vsl_value(float(raw), float(previous_vsl), self.cfg)
            if not any(abs(repaired.value - v) <= 1.0e-9 for v in out):
                accumulate_repair_diagnostics(self._repair_diagnostics, vsl=repaired)
                out.append(repaired.value)
        return out

    def _offramp_release_fraction(self, vsl: float) -> float:
        """VSL[km/h] → diverge segment 도달(=off-ramp 유입) 비율 근사 [0,1].

        VSL이 낮을수록 상류 유출(=diverge 도달)이 줄어 off-ramp 유입이 준다는 단조
        관계를 1차로 근사한다. 정밀 plant 예측이 아니라 후보 순위용 경량 surrogate."""
        max_vsl = max(float(v) for v in self.cfg.freeway_follower.vsl_set)
        return float(np.clip(vsl / max(max_vsl, 1.0e-9), 0.0, 1.0))

    def _freeway_agent_objective(
        self,
        rhos: list[float],
        density_excess: float,
        metering_error: float,
        ramp_metering: Mapping[str, float],
        vsl: float,
        previous_vsl: float,
        offramp_forecast_veh: float,
        offramp_storage_veh: float,
        offramp_capacity_veh: float = 0.0,
        ramp_queue_tts: float = 0.0,
        onramp_urban_queue_tts: float = 0.0,
        horizon_h: float = 1.0,
        freeway_vehicle_tts: Optional[float] = None,
        density_excess_tts: Optional[float] = None,
    ) -> float:
        """freeway agent 자기 비용(horizon emergence). 본선 차량·density penalty·
        off-ramp 큐(현재 점유 + VSL이 통과시키는 예측 유입의 spillback 가중분)·본선 hold·
        Δvsl smooth의 합. off-ramp가 포화에 가까우면 추가 유입의 spillback 비용이 비선형으로
        커져, VSL을 낮춰(예측 유입↓) 비용을 줄이는 게 emergent하게 유리해진다. off-ramp가
        비어 있으면 spillback 가중≈1이라 VSL을 낮출 유인이 없어 max VSL이 선택된다."""
        net = self.cfg.network
        fc = self.cfg.freeway_follower
        fraction = self._offramp_release_fraction(vsl)
        admitted = offramp_forecast_veh * fraction
        # off-ramp 포화도[0~1+]: 점유가 용량에 가까울수록 추가 유입의 spillback 비용이 커진다.
        occupancy_ratio = offramp_storage_veh / max(offramp_capacity_veh, 1.0e-9)
        spillback_weight = 1.0 + max(0.0, occupancy_ratio)
        offramp_cost = offramp_storage_veh + admitted * spillback_weight
        # VSL을 낮춰 통과시키지 못한 차량은 본선에 잡힘(hold) — 본선 대기 비용(가중 1)으로 가산.
        held_mainline = offramp_forecast_veh * (1.0 - fraction)
        # Δvsl smooth: 작은 off-ramp 압력에 과민하게 VSL을 흔들지 않도록 [km/h] 단위 그대로
        # 가격화한다. off-ramp 압력 이득이 smooth 비용을 넘어설 때만 VSL을 낮춘다(단조 emergence).
        vehicle_tts = (
            float(freeway_vehicle_tts)
            if freeway_vehicle_tts is not None
            else sum(max(0.0, rho) * net.freeway_segment_length_km * net.freeway_lanes for rho in rhos) * horizon_h
        )
        density_term = (
            float(density_excess_tts)
            if density_excess_tts is not None
            else density_excess * horizon_h
        )
        return float(
            vehicle_tts
            + ramp_queue_tts
            + onramp_urban_queue_tts
            + fc.density_penalty * density_term
            + (offramp_cost + held_mainline) * horizon_h
            + fc.vsl_smoothness_weight * abs(vsl - previous_vsl)
        )

    def _search_agent_vsl(
        self,
        agent: AgentSpec,
        rhos: list[float],
        lane_loss: float,
        previous_vsl: float,
        offramp_storage_veh: float,
        offramp_forecast_veh: float,
        offramp_capacity_veh: float,
        ramp_metering: Mapping[str, float],
    ) -> tuple[float, int]:
        """VSL 후보를 horizon objective로 평가해 최소 비용 후보를 고른다(emergence, option 2).

        트리거 없음 — off-ramp storage backup·예측 유입이 objective에 들어 있어 후보 평가
        과정에서 VSL이 자연히 낮아진다. lane-drop은 물리 제약이라 후보 평가와 별개로
        density_excess를 통해 반영된다."""
        net = self.cfg.network
        # lane-drop은 통과 용량을 줄이는 물리 제약 — density_excess에 가산해 후보 평가가
        # 차선 손실 segment에서 더 낮은 VSL을 선호하게 한다(트리거 아님, 비용 가중).
        density_excess = sum(max(0.0, rho - net.rho_crit) for rho in rhos) + lane_loss
        metering_error = 0.0  # VSL 선택은 metering_error와 독립 — 순위에 영향 없는 상수.
        candidates = self._vsl_candidates(previous_vsl)
        best_vsl, best_cost = previous_vsl, float("inf")
        for vsl in candidates:
            cost = self._freeway_agent_objective(
                rhos,
                density_excess,
                metering_error,
                ramp_metering,
                vsl,
                previous_vsl,
                offramp_forecast_veh,
                offramp_storage_veh,
                offramp_capacity_veh,
            )
            if cost < best_cost - 1.0e-12:
                best_cost, best_vsl = cost, float(vsl)
        return float(best_vsl), len(candidates)

    def _phase_arrival_coupling(
        self,
        agent: AgentSpec,
        coupling: Mapping[str, float],
    ) -> Dict[str, float]:
        """Wu식 arr_* flow[veh/h]를 UrbanFollower의 horizon arrival[veh]로 변환한다."""
        dt_h = self.cfg.simulation.T_c_h
        horizon = max(1, self.cfg.mpc.horizon_steps)
        out: Dict[str, float] = {}
        for phase_id in ("p1", "p2"):
            phase = f"{agent.signal}_{phase_id}"
            flow = max(0.0, float(coupling.get(f"arr_{phase}", 0.0)))
            if flow > 0.0:
                out[phase] = flow * dt_h * horizon
        return out

    def _coupling_active_flags(self) -> Dict[str, float]:
        """ablation 설정을 반영한 direction별 strategic coupling 활성 플래그."""
        u_to_f = 0.0 if self.ablation in {
            "NO_U_TO_F_INFO",
            "NO_CROSS_NETWORK_INFO",
            "LOCAL_ONLY_COUPLING_PLAYERS",
        } else 1.0
        f_to_u = 0.0 if self.ablation in {
            "NO_F_TO_U_INFO",
            "NO_CROSS_NETWORK_INFO",
            "LOCAL_ONLY_COUPLING_PLAYERS",
        } else 1.0
        return {
            "distributed_u_to_f_coupling_active": u_to_f,
            "distributed_f_to_u_coupling_active": f_to_u,
            "distributed_u_to_u_coupling_active": 1.0,
            "distributed_f_to_f_coupling_active": 1.0,
        }

    def _solve_urban_agent(
        self,
        agent: AgentSpec,
        state: TrafficState,
        leader: Optional[LeaderAction],
        forecast: list[DemandStep],
        freeway_response: FreewayFollowerResult,
        current: ControlAction,
        allocation_plan: Optional[AllocationResult],
        coupling: Mapping[str, float],
    ) -> AgentSolve:
        demand = forecast[0]
        # NO_F_TO_U/LOCAL_ONLY ablation: freeway 예측 압력 정보 차단 — urban은 측정된
        # 현재 off-ramp 도착(plant 경유)만 disturbance로 받는다.
        if self._block_f_to_u(agent):
            freeway_response = None
        phase_arrival_coupling = self._phase_arrival_coupling(agent, coupling)
        result = self.urban_follower.solve(
            state.copy(),
            leader,
            demand,
            freeway_response,
            current,
            allocation_plan,
            forecast=forecast,
            phase_arrival_coupling=phase_arrival_coupling,
        )
        specs = movement_specs(self.cfg)
        green = {
            key: value
            for key, value in result.green_times.items()
            if key.startswith(f"{agent.signal}_")
        }
        offsets = (
            {agent.signal: 0.0}
            if self._green_vsl_only_ttt_mode()
            else {agent.signal: result.offsets.get(agent.signal, current.offsets.get(agent.signal, 0.0))}
        )
        # follower allocation에 없는 movement(internal 등)는 0이 아니라 "비제어"다 —
        # 0으로 머지하면 내부 그리드 이동이 동결돼 출구 보급이 끊긴다(그리드 라우팅 후 치명적).
        # leaderless(P-FO)는 allocation 자체가 비어 있으므로 아래 합산도 자연히 건너뛴다.
        allocation = {
            movement: result.inflow_outflow_allocation[movement]
            for movement in agent.movements
            if movement in result.inflow_outflow_allocation
        }
        if self._green_vsl_only_ttt_mode():
            allocation = {}
        for movement in agent.movements:
            if movement not in allocation:
                continue
            spec = specs.get(movement, {})
            origin = str(spec.get("origin", ""))
            destination = str(spec.get("destination", ""))
            kind = str(spec.get("kind", ""))
            # _legacy_boundary_allocations와 동일하게 kind까지 맞춰 합산한다
            # (corner boundary_in→out movement가 out 링크 합에 중복 산입되지 않게).
            if origin in self.cfg.network.boundary_in_links and kind == "boundary_in":
                allocation[origin] = allocation.get(origin, 0.0) + allocation[movement]
            if destination in self.cfg.network.boundary_out_links and kind == "boundary_out":
                allocation[destination] = allocation.get(destination, 0.0) + allocation[movement]
        local_queue = sum(state.urban_movement_queue.get(movement, 0.0) for movement in agent.movements)
        local_objective = float(local_queue + result.objective_value / max(len(self.urban_agents), 1))
        diagnostics = {
            f"agent_{agent.id}_local_queue": float(local_queue),
            f"agent_{agent.id}_freeway_pressure_used": float(result.metrics.get("freeway_response_used", 0.0)),
            f"agent_{agent.id}_allocation_module_used": float(result.metrics.get("allocation_module_active", 0.0)),
        }
        merge_repair_diagnostics(diagnostics, result.metrics)
        diagnostics.update({
            key: float(value)
            for key, value in result.metrics.items()
            if key.startswith("allocation_")
        })
        diagnostics.update({
            key: float(value)
            for key, value in result.metrics.items()
            if key.startswith("urban_uncontrolled_node_")
        })
        return AgentSolve(
            agent_id=agent.id,
            objective=local_objective,
            green_times=green,
            offsets=offsets,
            allocation=allocation,
            infeasibility=dict(result.infeasibility),
            diagnostics=diagnostics,
        )

    def _freeway_response(self, solves: list[AgentSolve]) -> FreewayFollowerResult:
        ramp_metering: Dict[str, float] = {}
        vsl = self._aggregate_link_vsl(solves)
        objective = 0.0
        density_excess = 0.0
        metering_residual = 0.0
        step_capacity = 0.0
        min_receiving = 1.0
        coupling_payload: Dict[str, float] = {}
        for solve in solves:
            ramp_metering.update(solve.ramp_metering)
            objective += solve.objective
            density_excess += solve.infeasibility.get("density_excess", 0.0)
            metering_residual += solve.infeasibility.get("metering_tracking_residual", 0.0)
            step_capacity += solve.infeasibility.get("ramp_projection_first_step_capacity", 0.0)
            min_receiving = min(min_receiving, solve.infeasibility.get("min_ramp_receiving_factor", 1.0))
            for key, value in solve.infeasibility.items():
                if key.startswith(("offramp_predicted_arrival_", "offramp_predicted_flow_")):
                    coupling_payload[key] = max(coupling_payload.get(key, 0.0), float(value))
                elif key.startswith("offramp_storage_pressure_"):
                    coupling_payload[key] = max(coupling_payload.get(key, 0.0), float(value))
        infeasibility = {
            "density_excess": float(density_excess),
            "metering_tracking_residual": float(metering_residual),
            "ramp_projection_first_step_capacity": float(step_capacity),
            "min_ramp_receiving_factor": float(min_receiving),
            "freeway_follower_coupled_prediction": 0.0,
            "freeway_follower_lightweight_prediction": 1.0,
        }
        infeasibility.update(coupling_payload)
        return FreewayFollowerResult(
            ramp_metering=ramp_metering,
            vsl=vsl,
            objective_value=float(objective),
            infeasibility=infeasibility,
        )

    def _aggregate_link_vsl(self, solves: list[AgentSolve]) -> Dict[str, float]:
        """segment agent의 제안을 하나의 link-level VSL actuator로 합의한다.

        동일 link를 여러 agent가 소유한 것처럼 순서대로 덮어쓰지 않고, local
        congestion constraint 중 가장 제한적인 제안을 consensus projection으로 쓴다.
        """
        by_link: Dict[str, list[float]] = {link: [] for link in self.cfg.network.freeway_links}
        for solve in solves:
            for link, value in solve.vsl.items():
                by_link.setdefault(link, []).append(float(value))
        maximum = max(self.cfg.freeway_follower.vsl_set)
        return {
            link: float(min(values)) if values else float(maximum)
            for link, values in by_link.items()
        }

    def _merge_agent_controls(
        self,
        leader: Optional[LeaderAction],
        current: ControlAction,
        freeway_solves: list[AgentSolve],
        urban_solves: list[AgentSolve],
    ) -> ControlAction:
        alpha = float(np.clip(self.cfg.mpc.nash_relaxation_alpha, 0.0, 1.0))
        ramp_metering = dict(current.ramp_metering)
        vsl = dict(current.vsl)
        green_times = dict(current.green_times)
        offsets = dict(current.offsets)
        allocation = dict(current.inflow_outflow_allocation)
        infeasibility: Dict[str, float] = {}
        diagnostics: Dict[str, float] = {}
        for solve in freeway_solves:
            ramp_metering.update(solve.ramp_metering)
            infeasibility.update(solve.infeasibility)
            merge_repair_diagnostics(diagnostics, solve.diagnostics)
            diagnostics.update({
                k: v for k, v in solve.diagnostics.items()
                if k not in diagnostics or "quantization" not in k and "repair_count" not in k
            })
        vsl.update(self._aggregate_link_vsl(freeway_solves))
        for solve in urban_solves:
            green_times.update(solve.green_times)
            offsets.update(solve.offsets)
            allocation.update(solve.allocation)
            infeasibility.update(solve.infeasibility)
            merge_repair_diagnostics(diagnostics, solve.diagnostics)
            diagnostics.update({
                k: v for k, v in solve.diagnostics.items()
                if k not in diagnostics or "quantization" not in k and "repair_count" not in k
            })
        merge_repair_diagnostics(diagnostics, self._repair_diagnostics)
        if leader is None:
            # P-FO(spec 16.7 재정의): allocation 비제어 — plant 포화유율 fallback.
            allocation = {}
        else:
            allocation.update(self._legacy_boundary_allocations(allocation))
        ramp_out = _relax_map(current.ramp_metering, ramp_metering, alpha)
        offset_out = _relax_map(current.offsets, offsets, alpha)
        allocation_out = (
            {} if leader is None
            else _relax_map(current.inflow_outflow_allocation, allocation, alpha)
        )
        if self._green_vsl_only_ttt_mode():
            ramp_out = self._no_metering_control()
            offset_out = self._zero_offsets()
            allocation_out = {}
            diagnostics["wu_green_vsl_only_ttt_authority"] = 1.0
        return ControlAction(
            N_P_star=leader.N_P_star if leader is not None else 0.0,
            N_UF_star=leader.N_UF_star if leader is not None else 0.0,
            ramp_metering=ramp_out,
            vsl=vsl,
            green_times=_relax_map(current.green_times, green_times, alpha),
            offsets=offset_out,
            inflow_outflow_allocation=allocation_out,
            infeasibility=infeasibility,
            diagnostics=diagnostics,
        )

    def _clamp_offsets_to_reference(
        self,
        offsets: Mapping[str, float],
        reference: ControlAction,
    ) -> Dict[str, float]:
        """분산 내부 iteration이 실제 control-interval offset 제약을 누적 위반하지 않게 막는다."""
        cycle = self.cfg.network.cycle_length
        max_step = self.cfg.urban_follower.max_offset_step
        out: Dict[str, float] = {}
        for signal in self.cfg.network.signals:
            prev = reference.offsets.get(signal, 0.0)
            value = offsets.get(signal, prev)
            delta = (value - prev + 0.5 * cycle) % cycle - 0.5 * cycle
            delta = float(np.clip(delta, -max_step, max_step))
            out[signal] = float((prev + delta) % cycle)
        return out

    def _fixed_urban_solve(self, agent: AgentSpec) -> AgentSolve:
        """FIXED_URBAN_COUPLING_PLAYERS: green 50:50·offset 0·allocation 0.5cap 고정 정책."""
        net = self.cfg.network
        fixed = ControlAction.fixed(self.cfg)
        return AgentSolve(
            agent_id=agent.id,
            objective=0.0,
            green_times={
                f"{agent.signal}_p1": fixed.green_times[f"{agent.signal}_p1"],
                f"{agent.signal}_p2": fixed.green_times[f"{agent.signal}_p2"],
            },
            offsets={agent.signal: 0.0},
            allocation={m: fixed.inflow_outflow_allocation.get(m, 0.5 * net.movement_capacity_veh_h)
                        for m in agent.movements},
            diagnostics={f"agent_{agent.id}_fixed_policy": 1.0},
        )

    def _fixed_freeway_solve(self, agent: AgentSpec) -> AgentSolve:
        """FIXED_FREEWAY_COUPLING_PLAYERS: VSL=max·metering=용량(neutral) 고정 정책."""
        net = self.cfg.network
        return AgentSolve(
            agent_id=agent.id,
            objective=0.0,
            ramp_metering={r: net.ramp_capacity_veh_h[r] for r in agent.ramps},
            vsl={agent.link: max(self.cfg.freeway_follower.vsl_set)},
            infeasibility={
                "metering_tracking_residual": 0.0,
                "density_excess": 0.0,
                "min_ramp_receiving_factor": 1.0,
                "ramp_projection_first_step_capacity": sum(net.ramp_capacity_veh_h[r] for r in agent.ramps),
            },
            diagnostics={f"agent_{agent.id}_fixed_policy": 1.0},
        )

    def _clamp_vsl_to_reference(
        self,
        vsl: Mapping[str, float],
        reference: ControlAction,
    ) -> Dict[str, float]:
        """내부 iteration의 VSL 누적 드리프트가 interval 간 max_vsl_step 제약을
        위반하지 않게, 직전 적용 control 기준 ±step 범위의 discrete 값으로 스냅한다."""
        fc = self.cfg.freeway_follower
        vsl_set = sorted(float(v) for v in fc.vsl_set)
        out: Dict[str, float] = {}
        for link in self.cfg.network.freeway_links:
            prev = float(reference.vsl.get(link, max(vsl_set)))
            value = float(vsl.get(link, prev))
            feasible = [
                v for v in vsl_set
                if prev - fc.max_vsl_step - 1.0e-9 <= v <= prev + fc.max_vsl_step + 1.0e-9
            ] or vsl_set
            if self.cfg.mpc.relaxed_quantized_controls:
                repaired = repair_vsl_value(value, prev, self.cfg)
                accumulate_repair_diagnostics(self._repair_diagnostics, vsl=repaired)
                out[link] = repaired.value
            else:
                out[link] = float(min(feasible, key=lambda v: (abs(v - value), v)))
        return out

    def _legacy_boundary_allocations(self, allocation: Mapping[str, float]) -> Dict[str, float]:
        specs = movement_specs(self.cfg)
        out: Dict[str, float] = {}
        for link in self.cfg.network.boundary_in_links:
            out[link] = float(sum(
                allocation.get(movement, 0.0)
                for movement, spec in specs.items()
                if spec.get("origin") == link and spec.get("kind") == "boundary_in"
            ))
        for link in self.cfg.network.boundary_out_links:
            out[link] = float(sum(
                allocation.get(movement, 0.0)
                for movement, spec in specs.items()
                if spec.get("destination") == link and spec.get("kind") == "boundary_out"
            ))
        return out

    def _extract_coupling(
        self,
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, float]:
        ensure_urban_state(state, self.cfg)
        net = self.cfg.network
        # Spec 3.4/Wu coupling: ramp queue 공간 cap을 빼고 green 후보가 만든 접근부
        # reservoir inflow[veh/h]를 freeway agent에 전달한다.
        onramp = estimate_onramp_reservoir_inflow(
            state.copy(),
            control,
            demand,
            self.cfg,
            interval_h=self.cfg.simulation.T_c_h,
        )
        values: Dict[str, float] = {}
        for ramp, value in onramp.items():
            values[f"u_on_{ramp}"] = float(value)
            values[f"w_ramp_{ramp}"] = float(state.ramp_queue.get(ramp, 0.0))
        # urban→urban coupling: 상류 green release rate를 하류 phase arrival pressure로 보낸다.
        for signal in net.signals:
            for phase_id in ("p1", "p2"):
                phase = f"{signal}_{phase_id}"
                arrival_flow = 0.0
                for _up_signal, up_movement, beta in self._upstream_leaving_map.get(phase, []):
                    arrival_flow += beta * self._signal_leaving_rate(up_movement, control, state, demand)
                values[f"arr_{phase}"] = float(max(0.0, arrival_flow))
        for off_ramp in net.off_ramps:
            link = net.off_ramp_from_freeway[off_ramp]
            split = net.off_ramp_split_ratio.get(off_ramp, 0.0)
            flow = state.freeway_flow.get(link, [0.0])[-1] if state.freeway_flow.get(link) else 0.0
            values[f"q_off_{off_ramp}"] = float(max(0.0, flow * split))
        for link in net.freeway_links:
            rhos = state.freeway_density.get(link, [])
            speeds = state.freeway_speed.get(link, [])
            flows = state.freeway_flow.get(link, [])
            lanes = state.freeway_effective_lanes.get(link, [net.freeway_lanes for _ in rhos])
            values[f"rho_boundary_{link}"] = float(rhos[-1] if rhos else 0.0)
            values[f"speed_boundary_{link}"] = float(speeds[-1] if speeds else 0.0)
            for idx, rho in enumerate(rhos):
                values[f"rho_{link}_seg{idx}"] = float(rho)
                values[f"speed_{link}_seg{idx}"] = float(speeds[idx] if idx < len(speeds) else net.v_free)
                values[f"flow_{link}_seg{idx}"] = float(flows[idx] if idx < len(flows) else 0.0)
                lane_eff = float(lanes[idx] if idx < len(lanes) else net.freeway_lanes)
                values[f"lane_loss_{link}_seg{idx}"] = float(max(0.0, net.freeway_lanes - lane_eff))
        for agent in self.urban_agents:
            values[f"n_{agent.id}"] = float(sum(
                state.urban_movement_queue.get(movement, 0.0)
                for movement in agent.movements
            ))
        return values

    @staticmethod
    def _coupling_residual(old: Mapping[str, float], new: Mapping[str, float]) -> float:
        residual = 0.0
        for key in set(old) | set(new):
            a = float(old.get(key, 0.0))
            b = float(new.get(key, 0.0))
            residual = max(residual, abs(a - b) / max(1.0, abs(a), abs(b)))
        return float(residual)

    def _response_horizon_demand(self, forecast: list[DemandStep]) -> tuple[DemandStep, float, list[DemandStep]]:
        steps = forecast[: max(1, self.cfg.mpc.horizon_steps)] or forecast[:1]
        dt_h = self.cfg.simulation.T_c_h
        horizon_h = max(dt_h * max(len(steps), 1), 1.0e-9)

        def average(attr: str) -> Dict[str, float]:
            totals: Dict[str, float] = {}
            for step in steps:
                for key, value in getattr(step, attr).items():
                    totals[key] = totals.get(key, 0.0) + max(0.0, float(value)) * dt_h
            return {key: value / horizon_h for key, value in totals.items()}

        demand = DemandStep(
            freeway_mainline=average("freeway_mainline"),
            urban_boundary=average("urban_boundary"),
            ramp_arrival=average("ramp_arrival"),
            incident_capacity_factor=min(float(getattr(step, "incident_capacity_factor", 1.0)) for step in steps),
            freeway_lane_loss=merge_freeway_lane_loss(steps),
        )
        return demand, horizon_h, steps

    def _movement_forecast_arrivals_veh(
        self,
        steps: list[DemandStep],
    ) -> Dict[str, float]:
        net = self.cfg.network
        dt_h = self.cfg.simulation.T_c_h
        arrivals: Dict[str, float] = {}
        onramp_by_movement = {
            movement: ramp
            for ramp, movements in net.on_ramp_to_movement.items()
            for movement in movements
        }
        for step in steps:
            for movement, spec in self._specs.items():
                kind = str(spec.get("kind", ""))
                if kind == "boundary_in":
                    origin = str(spec.get("origin", ""))
                    beta = float(spec.get("beta", 1.0))
                    arrivals[movement] = arrivals.get(movement, 0.0) + (
                        max(0.0, step.urban_boundary.get(origin, 0.0)) * beta * dt_h
                    )
                elif kind == "on_ramp":
                    ramp = onramp_by_movement.get(movement, "")
                    if not ramp:
                        continue
                    movements = net.on_ramp_to_movement.get(ramp, [])
                    share = 1.0 / max(len(movements), 1)
                    arrivals[movement] = arrivals.get(movement, 0.0) + (
                        max(0.0, step.ramp_arrival.get(ramp, 0.0)) * share * dt_h
                    )
        return arrivals

    def _estimate_urban_service_veh(
        self,
        state: TrafficState,
        control: ControlAction,
        arrivals: Mapping[str, float],
        horizon_h: float,
    ) -> tuple[float, float]:
        service_total = 0.0
        boundary_out_sink = 0.0
        for movement, spec in self._specs.items():
            if str(spec.get("kind", "")) == "on_ramp":
                continue
            available = max(0.0, state.urban_movement_queue.get(movement, 0.0)) + max(
                0.0,
                arrivals.get(movement, 0.0),
            )
            cap_veh = horizon_h * _phase_green_fraction(control, self.cfg, spec) * _movement_capacity_flow(
                control,
                self.cfg,
                movement,
                spec,
            )
            served = min(available, max(0.0, cap_veh))
            service_total += served
            if str(spec.get("kind", "")) == "boundary_out":
                boundary_out_sink += served
        return float(service_total), float(boundary_out_sink)

    def _estimate_offramp_storage_departure_veh(
        self,
        state: TrafficState,
        control: ControlAction,
        horizon_h: float,
    ) -> float:
        return float(sum(self._estimate_offramp_storage_departures_by_ramp(state, control, horizon_h).values()))

    def _estimate_offramp_storage_departures_by_ramp(
        self,
        state: TrafficState,
        control: ControlAction,
        horizon_h: float,
    ) -> Dict[str, float]:
        net = self.cfg.network
        out: Dict[str, float] = {}
        for off_ramp in net.off_ramps:
            storage_link = net.off_ramp_storage_link.get(off_ramp, "")
            capacity = net.urban_link_storage_veh.get(storage_link, 0.0)
            occupancy = max(0.0, capacity - state.urban_link_storage.get(storage_link, capacity))
            if occupancy <= 0.0:
                out[off_ramp] = 0.0
                continue
            requested = 0.0
            for movement in net.off_ramp_to_movement.get(off_ramp, []):
                spec = self._specs.get(movement)
                if spec is None:
                    continue
                beta = max(0.0, float(spec.get("beta", 0.0)))
                cap_veh = horizon_h * _phase_green_fraction(control, self.cfg, spec) * _movement_capacity_flow(
                    control,
                    self.cfg,
                    movement,
                    spec,
                )
                requested += min(beta * occupancy, max(0.0, cap_veh))
            out[off_ramp] = float(min(occupancy, requested))
        return out

    def _estimate_ramp_release_veh(
        self,
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
        onramp_green_veh: Mapping[str, float],
        horizon_h: float,
    ) -> tuple[Dict[str, float], float]:
        net = self.cfg.network
        cap_factor = float(getattr(demand, "incident_capacity_factor", 1.0))
        release: Dict[str, float] = {}
        total = 0.0
        for ramp in net.ramps:
            link = net.ramp_to_freeway[ramp]
            densities = state.freeway_density.get(link, [])
            merge_idx = _configured_segment_index(
                getattr(net, "ramp_merge_segment_index", {}),
                ramp,
                len(densities) // 2,
                max(len(densities), 1),
            )
            rho_merge = densities[merge_idx] if densities else 0.0
            receiving = float(np.clip(
                (net.rho_max - rho_merge) / max(net.rho_max - net.rho_crit, 1.0e-9),
                0.0,
                1.0,
            ))
            available_veh = max(0.0, state.ramp_queue.get(ramp, 0.0)) + max(
                0.0,
                onramp_green_veh.get(ramp, 0.0),
            )
            requested_veh = max(0.0, control.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp])) * horizon_h
            capacity_veh = net.ramp_capacity_veh_h[ramp] * horizon_h
            receiving_veh = net.freeway_capacity_veh_h * cap_factor * receiving * horizon_h
            value = min(available_veh, requested_veh, capacity_veh, receiving_veh)
            release[ramp] = float(max(0.0, value))
            total += release[ramp]
        return release, float(total)

    def _estimate_freeway_density_excess_tts(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        ramp_release_veh: Mapping[str, float],
        horizon_h: float,
    ) -> tuple[float, float, float]:
        """Approximate candidate-dependent freeway density-excess vehicle-hours.

        The response objective used to charge density excess from the current
        state only. This rollout pushes ramp release into the configured merge
        segment so metering changes affect downstream density penalties.
        """
        net = self.cfg.network
        steps = forecast[: max(1, self.cfg.mpc.horizon_steps)]
        if not steps or horizon_h <= 0.0:
            return 0.0, state.freeway_segment_vehicles(net), 0.0
        lane_profile, _lane_diag = effective_lane_profile(state, self.cfg, steps[0])
        release_rate_by_segment: Dict[tuple[str, int], float] = {}
        for ramp, vehicles in ramp_release_veh.items():
            link = net.ramp_to_freeway.get(ramp, "")
            densities = state.freeway_density.get(link, [])
            if not link or not densities:
                continue
            merge_idx = _configured_segment_index(
                getattr(net, "ramp_merge_segment_index", {}),
                ramp,
                len(densities) // 2,
                max(len(densities), 1),
            )
            key = (link, merge_idx)
            release_rate_by_segment[key] = release_rate_by_segment.get(key, 0.0) + (
                max(0.0, float(vehicles)) / max(horizon_h, 1.0e-9)
            )

        dt_h = self.cfg.simulation.T_c_h
        density_excess_tts = 0.0
        terminal_freeway_vehicles = 0.0
        peak_density = 0.0
        for link in net.freeway_links:
            rhos = [max(0.0, float(rho)) for rho in state.freeway_density.get(link, [])]
            if not rhos:
                continue
            speeds = list(state.freeway_speed.get(link, []))
            lanes = lane_profile.get(link, [float(net.freeway_lanes) for _ in rhos])
            for step in steps:
                outflows: list[float] = []
                next_rhos: list[float] = []
                for idx, rho in enumerate(rhos):
                    lane = max(1.0e-9, float(lanes[idx] if idx < len(lanes) else net.freeway_lanes))
                    speed = max(
                        net.v_min,
                        float(speeds[idx] if idx < len(speeds) else net.v_free),
                    )
                    q_upstream = (
                        max(0.0, float(step.freeway_mainline.get(link, 0.0)))
                        if idx == 0
                        else outflows[idx - 1]
                    )
                    release_rate = release_rate_by_segment.get((link, idx), 0.0)
                    q_out = max(0.0, rho * speed * lane)
                    segment_veh_per_density = net.freeway_segment_length_km * lane
                    rho_next = float(np.clip(
                        rho + (q_upstream + release_rate - q_out) * dt_h
                        / max(segment_veh_per_density, 1.0e-9),
                        0.0,
                        net.rho_max,
                    ))
                    density_excess_tts += 0.5 * (
                        max(0.0, rho - net.rho_crit)
                        + max(0.0, rho_next - net.rho_crit)
                    ) * segment_veh_per_density * dt_h
                    outflows.append(q_out)
                    next_rhos.append(rho_next)
                    peak_density = max(peak_density, rho, rho_next)
                rhos = next_rhos
            for idx, rho in enumerate(rhos):
                lane = max(1.0e-9, float(lanes[idx] if idx < len(lanes) else net.freeway_lanes))
                terminal_freeway_vehicles += max(0.0, rho) * net.freeway_segment_length_km * lane
        return float(density_excess_tts), float(terminal_freeway_vehicles), float(peak_density)

    def _estimate_mainline_exit_veh(
        self,
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
        ramp_release_by_link_veh: Mapping[str, float],
        horizon_h: float,
    ) -> float:
        net = self.cfg.network
        cap_factor = float(getattr(demand, "incident_capacity_factor", 1.0))
        total = 0.0
        for link in net.freeway_links:
            current = sum(
                max(0.0, rho) * net.freeway_segment_length_km * net.freeway_lanes
                for rho in state.freeway_density.get(link, [])
            )
            arrivals = max(0.0, demand.freeway_mainline.get(link, 0.0)) * horizon_h
            arrivals += max(0.0, ramp_release_by_link_veh.get(link, 0.0))
            fallback_vsl = control.vsl.get(link, max(self.cfg.freeway_follower.vsl_set))
            vsl_factor = min(1.0, max(self.cfg.network.v_min, fallback_vsl) / max(net.v_free, 1.0e-9))
            exit_capacity = net.freeway_capacity_veh_h * cap_factor * vsl_factor * horizon_h
            total += min(max(0.0, current + arrivals), max(0.0, exit_capacity))
        return float(total)

    def _estimate_offramp_inflow_veh(
        self,
        state: TrafficState,
        control: ControlAction,
        steps: list[DemandStep],
    ) -> float:
        return float(sum(self._estimate_offramp_inflow_by_ramp(state, control, steps).values()))

    def _estimate_offramp_inflow_by_ramp(
        self,
        state: TrafficState,
        control: ControlAction,
        steps: list[DemandStep],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for link in self.cfg.network.freeway_links:
            forecast_by_ramp = self._forecast_offramp_arrivals_by_ramp(state, steps, link)
            fallback_vsl = control.vsl.get(link, max(self.cfg.freeway_follower.vsl_set))
            fraction = self._offramp_release_fraction(fallback_vsl)
            for off_ramp, vehicles in forecast_by_ramp.items():
                out[off_ramp] = out.get(off_ramp, 0.0) + max(0.0, vehicles) * fraction
        return out

    def _response_tts_objective(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: list[DemandStep],
        residual: float,
        proxy_objective: float,
    ) -> tuple[float, Dict[str, float]]:
        """Return a lightweight follower response objective in vehicle-hour units."""
        ensure_urban_state(state, self.cfg)
        net = self.cfg.network
        demand, horizon_h, steps = self._response_horizon_demand(forecast)
        movement_arrivals = self._movement_forecast_arrivals_veh(steps)
        urban_arrivals_veh = sum(max(0.0, value) for value in movement_arrivals.values())
        freeway_mainline_arrivals_veh = sum(
            max(0.0, step.freeway_mainline.get(link, 0.0)) * self.cfg.simulation.T_c_h
            for step in steps
            for link in net.freeway_links
        )

        urban_service_veh, boundary_out_sink_veh = self._estimate_urban_service_veh(
            state,
            control,
            movement_arrivals,
            horizon_h,
        )
        onramp_green_flows = estimate_onramp_green_release_flows(
            state.copy(),
            control,
            demand,
            self.cfg,
            interval_h=horizon_h,
        )
        onramp_green_veh = {
            ramp: max(0.0, flow) * horizon_h
            for ramp, flow in onramp_green_flows.items()
        }
        onramp_green_total = sum(onramp_green_veh.values())
        ramp_release_veh, ramp_release_total = self._estimate_ramp_release_veh(
            state,
            control,
            demand,
            onramp_green_veh,
            horizon_h,
        )
        ramp_release_by_link: Dict[str, float] = {}
        for ramp, vehicles in ramp_release_veh.items():
            link = net.ramp_to_freeway.get(ramp, "")
            ramp_release_by_link[link] = ramp_release_by_link.get(link, 0.0) + vehicles
        mainline_exit_veh = self._estimate_mainline_exit_veh(
            state,
            control,
            demand,
            ramp_release_by_link,
            horizon_h,
        )
        offramp_inflow_by_ramp = self._estimate_offramp_inflow_by_ramp(state, control, steps)
        offramp_inflow_veh = sum(offramp_inflow_by_ramp.values())
        offramp_departure_by_ramp = self._estimate_offramp_storage_departures_by_ramp(state, control, horizon_h)
        offramp_departure_veh = sum(offramp_departure_by_ramp.values())
        ramp_arrivals_veh = ramp_arrivals_over_horizon(steps, self.cfg, tuple(net.ramps))
        onramp_spillback_violation = sum(
            assess_onramp_spillback(
                state,
                self.cfg,
                ramp,
                ramp_arrivals_veh.get(ramp, 0.0),
                ramp_release_veh.get(ramp, 0.0),
            ).violation_veh
            for ramp in net.ramps
        )
        offramp_spillback_violation = sum(
            assess_offramp_spillback(
                state,
                self.cfg,
                off_ramp,
                offramp_inflow_by_ramp.get(off_ramp, 0.0),
                offramp_departure_by_ramp.get(off_ramp, 0.0),
            ).violation_veh
            for off_ramp in net.off_ramps
        )
        total_spillback_violation = onramp_spillback_violation + offramp_spillback_violation

        urban_start = state.total_urban_vehicles(net)
        uncontrolled_node_start = state.uncontrolled_node_vehicles(net)
        ramp_start = sum(max(0.0, value) for value in state.ramp_queue.values())
        freeway_segment_start = state.freeway_segment_vehicles(net)
        freeway_total_including_queues = state.total_freeway_vehicles(net)
        off_storage_start = state.off_ramp_storage_occupancy_veh(net)
        origin_start = sum(max(0.0, value) for value in state.mainline_origin_queue.values())
        current_total = urban_start + freeway_segment_start + ramp_start + off_storage_start + origin_start

        urban_terminal = max(
            0.0,
            urban_start + urban_arrivals_veh + offramp_departure_veh
            - boundary_out_sink_veh - onramp_green_total,
        )
        ramp_terminal = max(0.0, ramp_start + onramp_green_total - ramp_release_total)
        freeway_terminal = max(
            0.0,
            freeway_segment_start + freeway_mainline_arrivals_veh + ramp_release_total
            - mainline_exit_veh - offramp_inflow_veh,
        )
        off_storage_terminal = max(0.0, off_storage_start + offramp_inflow_veh - offramp_departure_veh)
        origin_terminal = max(0.0, origin_start)
        terminal_total = urban_terminal + ramp_terminal + freeway_terminal + off_storage_terminal + origin_terminal

        density_excess_veh = sum(
            net.freeway_segment_length_km
            * net.freeway_lanes
            * max(0.0, rho - net.rho_crit)
            for values in state.freeway_density.values()
            for rho in values
        )
        density_excess_tts, density_rollout_terminal, density_rollout_peak = (
            self._estimate_freeway_density_excess_tts(
                state,
                steps,
                ramp_release_veh,
                horizon_h,
            )
        )
        residual_penalty = max(0.0, residual if np.isfinite(residual) else 0.0) * current_total * horizon_h
        objective = 0.5 * (current_total + terminal_total) * horizon_h
        objective += self.cfg.freeway_follower.density_penalty * density_excess_tts
        objective += residual_penalty
        spillback_penalty = self.cfg.freeway_follower.ramp_queue_penalty * total_spillback_violation * horizon_h
        objective += spillback_penalty

        diagnostics = {
            "distributed_response_objective_tts": float(objective),
            "distributed_response_rollout_active": 0.0,
            "distributed_response_rollout_ttt": 0.0,
            "distributed_response_rollout_freeway_ttt": 0.0,
            "distributed_response_rollout_urban_ttt": 0.0,
            "distributed_response_proxy_objective": float(proxy_objective),
            "distributed_response_horizon_h": float(horizon_h),
            "distributed_response_current_vehicles": float(current_total),
            "distributed_response_uncontrolled_node_urban_vehicles": float(uncontrolled_node_start),
            "distributed_response_freeway_segment_vehicles": float(freeway_segment_start),
            "distributed_response_freeway_total_vehicles_including_queues": float(
                freeway_total_including_queues
            ),
            "distributed_response_ramp_queue_start_veh": float(ramp_start),
            "distributed_response_origin_queue_start_veh": float(origin_start),
            "distributed_response_terminal_proxy_vehicles": float(terminal_total),
            "distributed_response_terminal_rollout_vehicles": 0.0,
            "distributed_response_terminal_urban_vehicles": float(urban_terminal),
            "distributed_response_terminal_ramp_queue_veh": float(ramp_terminal),
            "distributed_response_terminal_freeway_vehicles": float(freeway_terminal),
            "distributed_response_terminal_offramp_storage_veh": float(off_storage_terminal),
            "distributed_response_terminal_origin_queue_veh": float(origin_terminal),
            "distributed_response_urban_service_veh": float(urban_service_veh + onramp_green_total),
            "distributed_response_boundary_out_sink_veh": float(boundary_out_sink_veh),
            "distributed_response_onramp_green_veh": float(onramp_green_total),
            "distributed_response_ramp_release_veh": float(ramp_release_total),
            "distributed_response_mainline_exit_veh": float(mainline_exit_veh),
            "distributed_response_offramp_inflow_veh": float(offramp_inflow_veh),
            "distributed_response_offramp_departure_veh": float(offramp_departure_veh),
            "distributed_response_arrivals_veh": float(urban_arrivals_veh + freeway_mainline_arrivals_veh),
            "distributed_response_residual_penalty": float(residual_penalty),
            "distributed_response_residual": float(residual if np.isfinite(residual) else 0.0),
            "distributed_response_density_excess_veh": float(density_excess_veh),
            "distributed_response_density_excess_current_tts": float(density_excess_veh * horizon_h),
            "distributed_response_density_excess_tts": float(density_excess_tts),
            "distributed_response_density_rollout_terminal_freeway_vehicles": float(density_rollout_terminal),
            "distributed_response_density_rollout_peak_density": float(density_rollout_peak),
            "distributed_response_spillback_penalty": float(spillback_penalty),
            "distributed_response_onramp_spillback_violation_veh": float(onramp_spillback_violation),
            "distributed_response_offramp_spillback_violation_veh": float(offramp_spillback_violation),
            "distributed_response_total_spillback_violation_veh": float(total_spillback_violation),
            "distributed_response_spillback_constraint_feasible": float(total_spillback_violation <= 1.0e-9),
            "distributed_response_onramp_green_shortfall_veh": 0.0,
            "distributed_response_ramp_release_shortfall_veh": 0.0,
            "distributed_response_ramp_queue_overflow_count": float(
                sum(1 for value in state.ramp_queue.values() if value > net.ramp_queue_max_veh)
            ),
            "distributed_response_movement_queue_projection_veh": float(
                sum(max(0.0, value) for value in state.urban_movement_queue.values())
            ),
            "distributed_response_ttt_compatible": 1.0,
        }
        return float(objective), diagnostics

    def _diagnostics(
        self,
        freeway_solves: list[AgentSolve],
        urban_solves: list[AgentSolve],
        residual: float,
        iteration: int,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {
            "distributed_player_active": 1.0,
            "nash_per_agent_active": 1.0,
            "distributed_urban_agent_count": float(len(self.urban_agents)),
            "distributed_freeway_agent_count": float(len(self.freeway_agents)),
            "distributed_coupling_residual": float(residual if np.isfinite(residual) else 0.0),
            "distributed_iterations": float(iteration),
            "nash_mutual_response_active": 1.0,
            "nash_urban_used_freeway_response": 1.0,
            "wu_green_vsl_only_ttt_authority": float(self._green_vsl_only_ttt_mode()),
        }
        coupling_flags = self._coupling_active_flags()
        out.update(coupling_flags)
        out["nash_freeway_used_coupled_prediction"] = coupling_flags["distributed_u_to_f_coupling_active"]
        out["nash_urban_used_freeway_response"] = coupling_flags["distributed_f_to_u_coupling_active"]
        out["distributed_neighbor_coupling_active"] = float(max(coupling_flags.values()))
        for agent in self.urban_agents + self.freeway_agents:
            out[f"distributed_agent_{agent.id}_active"] = 1.0
        for solve in freeway_solves + urban_solves:
            out[f"agent_{solve.agent_id}_objective"] = float(solve.objective)
            merge_repair_diagnostics(out, solve.diagnostics)
            out.update({
                k: v for k, v in solve.diagnostics.items()
                if k not in out or "quantization" not in k and "repair_count" not in k
            })
        merge_repair_diagnostics(out, self._repair_diagnostics)
        return out
