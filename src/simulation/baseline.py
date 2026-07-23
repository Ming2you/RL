from __future__ import annotations

from src.controllers.freeway_follower import FreewayFollower
from src.controllers.leader import LeaderAction
from src.controllers.urban_follower import UrbanFollower
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def baseline_control(
    mode: str,
    cfg: ExperimentConfig,
    state: TrafficState,
    demand: DemandStep,
    previous: ControlAction | None = None,
) -> ControlAction:
    if mode in {"no_control", "fixed_signal_fixed_speed"}:
        return ControlAction.uncontrolled(cfg)
    control = ControlAction.fixed(cfg)
    if mode == "local_control_only":
        nuf = min(cfg.network.total_ramp_capacity, sum(demand.ramp_arrival.values()))
        leader = LeaderAction(N_P_star=0.0, N_UF_star=nuf)
        fw = FreewayFollower(cfg).solve(state, leader, demand, previous or control)
        urban = UrbanFollower(cfg).solve(state, leader, demand, fw, previous or control)
        return ControlAction(
            N_P_star=0.0,
            N_UF_star=nuf,
            ramp_metering=fw.ramp_metering,
            vsl=fw.vsl,
            green_times=urban.green_times,
            offsets=control.offsets,
            inflow_outflow_allocation=urban.inflow_outflow_allocation,
            infeasibility={**fw.infeasibility, **urban.infeasibility},
            diagnostics={"baseline_mode": 1.0},
        )
    raise ValueError(f"Unknown baseline mode: {mode}")
