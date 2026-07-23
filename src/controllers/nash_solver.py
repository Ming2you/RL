from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from src.controllers.freeway_follower import FreewayFollower
from src.controllers.leader import LeaderAction
from src.controllers.urban_follower import UrbanFollower
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState


@dataclass
class NashResult:
    control: ControlAction
    objective_value: float
    iterations: int
    converged: bool
    residual_objective: float
    residual_control: float
    diagnostics: Dict[str, float]


def _relax_map(old: Dict[str, float], new: Dict[str, float], alpha: float) -> Dict[str, float]:
    keys = set(old) | set(new)
    return {k: float((1.0 - alpha) * old.get(k, new.get(k, 0.0)) + alpha * new.get(k, old.get(k, 0.0))) for k in keys}


class NashSolver:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.freeway = FreewayFollower(cfg)
        self.urban = UrbanFollower(cfg)

    def solve(
        self,
        state: TrafficState,
        leader: LeaderAction,
        demand: DemandStep | Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
    ) -> NashResult:
        forecast = [demand] if isinstance(demand, DemandStep) else list(demand)
        if not forecast:
            raise ValueError("NashSolver requires at least one demand step.")
        first_demand = forecast[0]
        alpha = float(np.clip(self.cfg.mpc.nash_relaxation_alpha, 0.0, 1.0))
        current = (previous_control.copy() if previous_control is not None else ControlAction.fixed(self.cfg))
        current.N_P_star = leader.N_P_star
        current.N_UF_star = leader.N_UF_star
        prev_obj = np.inf
        best_control = current
        best_obj = np.inf
        best_diagnostics: Dict[str, float] = {}
        converged = False
        residual_obj = np.inf
        residual_control = np.inf
        diagnostics: Dict[str, float] = {}

        for iteration in range(1, self.cfg.mpc.max_nash_iter + 1):
            fw = self.freeway.solve(state, leader, forecast, current)
            tmp = ControlAction(
                N_P_star=leader.N_P_star,
                N_UF_star=leader.N_UF_star,
                ramp_metering=_relax_map(current.ramp_metering, fw.ramp_metering, alpha),
                vsl=fw.vsl,
                green_times=dict(current.green_times),
                offsets=dict(current.offsets),
                inflow_outflow_allocation=dict(current.inflow_outflow_allocation),
            )
            urban_reference = previous_control.copy() if previous_control is not None else current.copy()
            urban = self.urban.solve(state, leader, first_demand, fw, urban_reference)
            candidate = ControlAction(
                N_P_star=leader.N_P_star,
                N_UF_star=leader.N_UF_star,
                ramp_metering=tmp.ramp_metering,
                vsl=tmp.vsl,
                green_times=_relax_map(current.green_times, urban.green_times, alpha),
                offsets=_relax_map(current.offsets, urban.offsets, alpha),
                inflow_outflow_allocation=_relax_map(
                    current.inflow_outflow_allocation,
                    urban.inflow_outflow_allocation,
                    alpha,
                ),
                infeasibility={**fw.infeasibility, **urban.infeasibility},
                diagnostics={
                    **urban.metrics,
                    **fw.diagnostics,
                    "freeway_follower_coupled_prediction": fw.infeasibility.get(
                        "freeway_follower_coupled_prediction",
                        0.0,
                    ),
                },
            )
            obj = fw.objective_value + urban.objective_value
            if obj < best_obj:
                best_obj = float(obj)
                best_control = candidate
            residual_obj = abs(prev_obj - obj) if np.isfinite(prev_obj) else np.inf
            old_vec = np.asarray(current.control_vector(self.cfg), dtype=float)
            new_vec = np.asarray(candidate.control_vector(self.cfg), dtype=float)
            residual_control = float(np.max(np.abs(new_vec - old_vec))) if old_vec.size else 0.0
            current = candidate
            prev_obj = obj
            diagnostics = {
                "freeway_objective": float(fw.objective_value),
                "urban_objective": float(urban.objective_value),
                "nash_residual_objective": float(residual_obj if np.isfinite(residual_obj) else obj),
                "nash_residual_control": float(residual_control),
                "nash_mutual_response_active": 1.0,
                "nash_urban_used_freeway_response": float(urban.metrics.get("freeway_response_used", 0.0)),
                "nash_freeway_used_coupled_prediction": float(fw.infeasibility.get(
                    "freeway_follower_coupled_prediction",
                    0.0,
                )),
                **urban.metrics,
                **fw.diagnostics,
            }
            if obj <= best_obj + 1.0e-12:
                best_diagnostics = dict(diagnostics)
            if (
                residual_obj < self.cfg.mpc.nash_obj_tol
                and residual_control < self.cfg.mpc.nash_control_tol
            ):
                converged = True
                break
        if not converged:
            current = best_control
            diagnostics = best_diagnostics or diagnostics
            prev_obj = best_obj
        current.diagnostics.update(diagnostics)
        current.diagnostics["nash_converged"] = converged
        current.diagnostics["nash_iterations"] = iteration
        return NashResult(
            control=current,
            objective_value=float(prev_obj if np.isfinite(prev_obj) else best_obj),
            iterations=iteration,
            converged=converged,
            residual_objective=float(residual_obj if np.isfinite(residual_obj) else best_obj),
            residual_control=float(residual_control),
            diagnostics=diagnostics,
        )
