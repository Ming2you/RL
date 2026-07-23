from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from src.controllers.inflow_outflow_allocation import (
    BALANCE_INFLOW_KINDS,
    BALANCE_OUTFLOW_KINDS,
    INFLOW_KINDS,
    OUTFLOW_KINDS,
    AllocationResult,
    InflowOutflowAllocationModule,
)
from src.controllers.leader import LeaderAction
from src.models.demand import DemandStep
from src.models.state import ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    boundary_group_key,
    movement_specs,
    movement_storage_capacity,
    safe_balance_index,
)


class SimplifiedInflowOutflowAllocationModule(InflowOutflowAllocationModule):
    """Stackelberg ablation용 deterministic inflow-outflow allocation module.

    기존 PSO allocation path는 보존하고, 이 모듈은 `N_P_star`를 보호영역 net inflow
    target[veh]로 해석한 뒤 follower horizon의 flow-rate[veh/h]로만 변환한다. 목적은
    allocation 구조 자체가 문제였는지 빠르게 분리 진단하는 것이다.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)

    def solve(
        self,
        state: TrafficState,
        leader: LeaderAction,
        forecast: "list[DemandStep] | None" = None,
    ) -> AllocationResult:
        self.solve_count += 1
        specs = movement_specs(self.cfg)
        movements = [
            movement
            for movement, spec in specs.items()
            if spec.get("kind") in INFLOW_KINDS | OUTFLOW_KINDS
        ]
        if not movements:
            return AllocationResult({}, {}, 0.0, 0.0, 0.0, {
                "allocation_module_active": 1.0,
                "allocation_simplified_module_active": 1.0,
            })

        kinds = [str(specs[movement].get("kind", "")) for movement in movements]
        target = self._leader_net_inflow_target_veh_h(leader, forecast)
        lower, upper = self._bounds(movements, specs, leader)
        target = self._clip_target(target, lower, upper, kinds)
        seed = self._balanced_seed(state, movements, kinds, specs, lower, upper, forecast)
        projected = self._project_net_flow(seed, lower, upper, kinds, target)

        flows = {movement: float(projected[idx]) for idx, movement in enumerate(movements)}
        green = {
            movement: self._flow_to_green_sec(flows[movement])
            for movement in movements
        }
        in_values = self._densities_after_service(
            state,
            movements,
            kinds,
            projected,
            BALANCE_INFLOW_KINDS,
        )
        out_values = self._densities_after_service(
            state,
            movements,
            kinds,
            projected,
            BALANCE_OUTFLOW_KINDS,
        )
        net = self._net_flow(projected, kinds)
        residual = abs(net - target)
        horizon_h = self._target_horizon_h(forecast)
        diagnostics = {
            "allocation_module_active": 1.0,
            "allocation_simplified_module_active": 1.0,
            "allocation_simplified_calls": float(self.solve_count),
            "allocation_target_net_inflow_veh": float(leader.N_P_star),
            "allocation_target_net_inflow_veh_h": float(target),
            "allocation_projected_net_inflow_veh": float(net * horizon_h),
            "allocation_projected_net_inflow_veh_h": float(net),
            "allocation_net_inflow_residual_veh": float(residual * horizon_h),
            "allocation_net_inflow_residual_veh_h": float(residual),
            "allocation_B_in": safe_balance_index(in_values),
            "allocation_B_out": safe_balance_index(out_values),
            "allocation_inflow_dim": float(len(in_values)),
            "allocation_outflow_dim": float(len(out_values)),
        }
        return AllocationResult(flows, green, float(target), float(net), float(residual), diagnostics)

    def _balanced_seed(
        self,
        state: TrafficState,
        movements: list[str],
        kinds: list[str],
        specs: Mapping[str, Mapping[str, object]],
        lower: np.ndarray,
        upper: np.ndarray,
        forecast: "list[DemandStep] | None",
    ) -> np.ndarray:
        """현 queue/forecast로 균형 seed를 만든 뒤 net-inflow projection에 넘긴다."""
        horizon_h = self._target_horizon_h(forecast)
        arrivals = self._forecast_arrivals_veh(movements, specs, forecast)
        queues = np.asarray([
            max(0.0, state.urban_movement_queue.get(movement, 0.0))
            + max(0.0, arrivals.get(movement, 0.0))
            for movement in movements
        ], dtype=float)
        queue_service = queues / max(horizon_h, 1.0e-9)
        balance_service = self._balance_excess_service(
            movements,
            kinds,
            specs,
            queues,
            horizon_h,
        )
        # queue clearing과 balance equalization을 섞어서 PSO 없이 안정적인 초기점을 만든다.
        seed = 0.5 * queue_service + 0.5 * balance_service
        return np.clip(seed, lower, upper)

    def _forecast_arrivals_veh(
        self,
        movements: list[str],
        specs: Mapping[str, Mapping[str, object]],
        forecast: "list[DemandStep] | None",
    ) -> Dict[str, float]:
        steps = forecast[: max(1, self.cfg.mpc.horizon_steps)] if forecast else []
        arrivals: Dict[str, float] = {movement: 0.0 for movement in movements}
        dt_h = self.cfg.simulation.T_c_h
        for step in steps:
            for movement in movements:
                spec = specs[movement]
                kind = str(spec.get("kind", ""))
                beta = float(spec.get("beta", 1.0))
                if kind == "boundary_in":
                    origin = str(spec.get("origin", ""))
                    arrivals[movement] += max(0.0, step.urban_boundary.get(origin, 0.0)) * beta * dt_h
                elif kind == "on_ramp":
                    ramp = str(spec.get("ramp", ""))
                    arrivals[movement] += max(0.0, step.ramp_arrival.get(ramp, 0.0)) * beta * dt_h
        return arrivals

    def _balance_excess_service(
        self,
        movements: list[str],
        kinds: list[str],
        specs: Mapping[str, Mapping[str, object]],
        queues: np.ndarray,
        horizon_h: float,
    ) -> np.ndarray:
        service = np.zeros(len(movements), dtype=float)
        for accepted in (BALANCE_INFLOW_KINDS, BALANCE_OUTFLOW_KINDS):
            grouped: Dict[str, list[int]] = {}
            caps: Dict[str, float] = {}
            for idx, movement in enumerate(movements):
                if kinds[idx] not in accepted:
                    continue
                key = boundary_group_key(specs.get(movement, {}))
                grouped.setdefault(key, []).append(idx)
                caps[key] = caps.get(key, 0.0) + max(
                    movement_storage_capacity(self.cfg, movement, specs.get(movement, {})),
                    1.0e-9,
                )
            if not grouped:
                continue
            densities = {
                key: float(sum(queues[idx] for idx in indices) / max(caps[key], 1.0e-9))
                for key, indices in grouped.items()
            }
            target_density = float(np.mean(list(densities.values()))) if densities else 0.0
            for key, indices in grouped.items():
                excess = max(0.0, densities[key] - target_density) * caps[key]
                share = excess / max(len(indices), 1)
                for idx in indices:
                    service[idx] = max(service[idx], share / max(horizon_h, 1.0e-9))
        return service
