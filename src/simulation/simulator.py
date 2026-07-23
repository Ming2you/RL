from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.simulation.coupling import run_coupled_interval


@dataclass
class StepLog:
    step: int
    time_sec: float
    freeway_ttt: float
    urban_ttt: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class MixedTrafficSimulator:
    def __init__(self, cfg: ExperimentConfig, initial_state: TrafficState | None = None):
        self.cfg = cfg
        self.state = initial_state.copy() if initial_state else TrafficState.initial(cfg)
        self.freeway_ttt = 0.0
        self.urban_ttt = 0.0
        self.logs: List[StepLog] = []

    def copy(self) -> "MixedTrafficSimulator":
        return copy.deepcopy(self)

    def step(self, control: ControlAction, demand: DemandStep, step_idx: int) -> StepLog:
        coupled = run_coupled_interval(self.state, control, demand, self.cfg)
        self.state.time_sec += self.cfg.simulation.control_interval
        self.freeway_ttt += coupled.freeway_ttt
        self.urban_ttt += coupled.urban_ttt
        diag = {**coupled.diagnostics, **control.diagnostics}
        log = StepLog(
            step=step_idx,
            time_sec=self.state.time_sec,
            freeway_ttt=coupled.freeway_ttt,
            urban_ttt=coupled.urban_ttt,
            diagnostics=diag,
        )
        self.logs.append(log)
        return log

    @property
    def total_ttt(self) -> float:
        return float(self.freeway_ttt + self.urban_ttt)


def state_row(state: TrafficState, cfg: ExperimentConfig, step: int) -> Dict[str, Any]:
    state.ensure_freeway_lane_profile(cfg.network)
    row: Dict[str, Any] = {
        "step": step,
        "time_sec": state.time_sec,
        "freeway_vehicles": state.total_freeway_vehicles(cfg.network),
        "urban_vehicles": state.total_urban_vehicles(),
        "urban_protected_accumulation_veh": state.protected_accumulation_veh(cfg.network),
    }
    for link, values in state.freeway_density.items():
        row[f"rho_{link}_mean"] = sum(values) / len(values)
    for link, values in state.freeway_speed.items():
        row[f"speed_{link}_mean"] = sum(values) / len(values)
    for link, values in state.freeway_flow.items():
        row[f"flow_{link}_mean"] = sum(values) / len(values)
    for link, values in state.freeway_effective_lanes.items():
        row[f"lambda_eff_{link}_mean"] = sum(values) / len(values)
        row[f"lambda_eff_{link}_last"] = values[-1] if values else cfg.network.freeway_lanes
    for ramp, value in state.ramp_queue.items():
        row[f"ramp_queue_{ramp}"] = value
    for link, value in state.boundary_queue.items():
        row[f"boundary_queue_{link}"] = value
    for movement, value in state.urban_movement_queue.items():
        row[f"movement_queue_{movement}"] = value
    for link, value in state.urban_link_storage.items():
        row[f"storage_available_{link}"] = value
    return row


def control_row(control: ControlAction, cfg: ExperimentConfig, step: int, time_sec: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "step": step,
        "time_sec": time_sec,
        "N_P_star": control.N_P_star,
        "N_UF_star": control.N_UF_star,
    }
    for ramp in cfg.network.ramps:
        row[f"ramp_metering_{ramp}"] = control.ramp_metering.get(ramp, 0.0)
    vsl_max = max(cfg.freeway_follower.vsl_set)
    for link in cfg.network.freeway_links:
        # Option C: segment별 VSL 열 + 기존 link 열은 min-over-segment(병목 metering 가시화).
        seg_vsls = []
        for i in range(cfg.network.freeway_segments_per_link):
            value = segment_vsl(control, link, i, cfg)
            row[f"vsl_{link}_seg{i}"] = value
            seg_vsls.append(value)
        # 기존 분석/metrics가 읽는 link 열은 segment 최소값으로 유지(하위호환 + 작동 가시화).
        row[f"vsl_{link}"] = min(seg_vsls) if seg_vsls else control.vsl.get(link, vsl_max)
    for signal in cfg.network.signals:
        row[f"green_{signal}_p1"] = control.green_times.get(f"{signal}_p1", 0.0)
        row[f"green_{signal}_p2"] = control.green_times.get(f"{signal}_p2", 0.0)
        row[f"offset_{signal}"] = control.offsets.get(signal, 0.0)
    for link in cfg.network.movement_links:
        row[f"allocation_{link}"] = control.inflow_outflow_allocation.get(link, 0.0)
    for movement in cfg.network.urban_movements:
        row[f"allocation_{movement}"] = control.inflow_outflow_allocation.get(movement, 0.0)
    row.update({f"diag_{k}": v for k, v in control.diagnostics.items() if isinstance(v, (int, float, bool))})
    return row
