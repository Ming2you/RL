from __future__ import annotations

from dataclasses import dataclass
from src.models.state import ExperimentConfig, TrafficState
from src.models.urban_queue_model import movement_specs, movement_storage_capacity


@dataclass(frozen=True)
class SpillbackAssessment:
    violation_veh: float
    capacity_veh: float
    occupancy_veh: float
    terminal_veh: float


def onramp_combined_capacity_veh(cfg: ExperimentConfig, ramp: str) -> float:
    """Capacity of the urban approach legs plus the ramp reservoir."""
    specs = movement_specs(cfg)
    approach_capacity = sum(
        movement_storage_capacity(cfg, movement, specs[movement])
        for movement in cfg.network.on_ramp_to_movement.get(ramp, [])
    )
    return float(cfg.network.ramp_queue_max_veh + approach_capacity)


def onramp_combined_occupancy_veh(
    state: TrafficState,
    cfg: ExperimentConfig,
    ramp: str,
) -> float:
    approach_queue = sum(
        max(0.0, state.urban_movement_queue.get(movement, 0.0))
        for movement in cfg.network.on_ramp_to_movement.get(ramp, [])
    )
    return float(max(0.0, state.ramp_queue.get(ramp, 0.0)) + approach_queue)


def assess_onramp_spillback(
    state: TrafficState,
    cfg: ExperimentConfig,
    ramp: str,
    ramp_arrival_veh: float,
    metering_release_veh: float,
) -> SpillbackAssessment:
    capacity = onramp_combined_capacity_veh(cfg, ramp)
    occupancy = onramp_combined_occupancy_veh(state, cfg, ramp)
    terminal = max(0.0, occupancy + max(0.0, ramp_arrival_veh) - max(0.0, metering_release_veh))
    return SpillbackAssessment(
        violation_veh=float(max(0.0, terminal - capacity)),
        capacity_veh=float(capacity),
        occupancy_veh=float(occupancy),
        terminal_veh=float(terminal),
    )


def offramp_combined_capacity_veh(cfg: ExperimentConfig, off_ramp: str) -> float:
    """Capacity of the off-ramp storage plus downstream intersection legs."""
    specs = movement_specs(cfg)
    storage_link = cfg.network.off_ramp_storage_link.get(off_ramp, "")
    storage_capacity = float(cfg.network.urban_link_storage_veh.get(storage_link, 0.0))
    leg_capacity = sum(
        movement_storage_capacity(cfg, movement, specs[movement])
        for movement in cfg.network.off_ramp_to_movement.get(off_ramp, [])
    )
    return float(storage_capacity + leg_capacity)


def offramp_combined_occupancy_veh(
    state: TrafficState,
    cfg: ExperimentConfig,
    off_ramp: str,
) -> float:
    storage_link = cfg.network.off_ramp_storage_link.get(off_ramp, "")
    storage_capacity = float(cfg.network.urban_link_storage_veh.get(storage_link, 0.0))
    storage_available = float(state.urban_link_storage.get(storage_link, storage_capacity))
    storage_occupancy = max(0.0, storage_capacity - storage_available)
    leg_queue = sum(
        max(0.0, state.urban_movement_queue.get(movement, 0.0))
        for movement in cfg.network.off_ramp_to_movement.get(off_ramp, [])
    )
    return float(storage_occupancy + leg_queue)


def assess_offramp_spillback(
    state: TrafficState,
    cfg: ExperimentConfig,
    off_ramp: str,
    offramp_inflow_veh: float,
    service_veh: float = 0.0,
) -> SpillbackAssessment:
    capacity = offramp_combined_capacity_veh(cfg, off_ramp)
    occupancy = offramp_combined_occupancy_veh(state, cfg, off_ramp)
    terminal = max(0.0, occupancy + max(0.0, offramp_inflow_veh) - max(0.0, service_veh))
    return SpillbackAssessment(
        violation_veh=float(max(0.0, terminal - capacity)),
        capacity_veh=float(capacity),
        occupancy_veh=float(occupancy),
        terminal_veh=float(terminal),
    )


def ramp_arrivals_over_horizon(
    forecast: list,
    cfg: ExperimentConfig,
    ramps: tuple[str, ...],
) -> dict[str, float]:
    dt_h = cfg.simulation.T_c_h
    return {
        ramp: float(sum(max(0.0, step.ramp_arrival.get(ramp, 0.0)) * dt_h for step in forecast))
        for ramp in ramps
    }
