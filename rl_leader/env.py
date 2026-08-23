"""Gym-like wrapper around the production RL Stackelberg controller path."""
from __future__ import annotations

import sys
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.coordination import (
    CoordinationActionSchema,
    CoordinationMask,
    CoordinationObservationSchema,
    StaticCoordinationProvider,
)
from src.controllers.rl_stackelberg import (
    OptimizerCoordinationProvider,
    RLStackelbergController,
    configure_pstack_b13_follower_contract,
)
from src.controllers.f1_wu_faithful_follower import F1StackelbergWuMeteredController
from src.models.demand import (
    DemandProfile,
    ScenarioConfig,
    apply_scenario_network_overrides,
    load_scenarios,
    merge_freeway_lane_loss,
)
from src.models.metanet import desired_speed_kmh, segment_flow_veh_h
from src.models.state import ControlAction, ExperimentConfig, segment_vsl
from src.simulation.simulator import MixedTrafficSimulator


WANG = {
    "v_free": 115.0,
    "rho_crit": 31.5,
    "metanet_tau_h": 0.0056111,
    "metanet_nu_km2_h": 22.5,
    "metanet_kappa_veh_km_lane": 10.0,
    "metanet_delta_merge": 0.9,
}

TARGETED_SCENARIO_WEIGHTS = (
    ("sweet_155_w60", 0.20),
    ("sweet_170_w60", 0.10),
    ("sweet_170_incident_w60", 0.35),
    ("sweet_170_skew15_w60", 0.10),
    ("sweet_190_w60", 0.25),
)
OPTIMIZER_ANCHOR_CONTRACT = "pstack_b13_coordination_no_supervisor_v1"


def make_cfg(scenario, fw_buffer: int = 8) -> tuple[ExperimentConfig, ScenarioConfig]:
    cfg = ExperimentConfig.from_file(str(ROOT / "src" / "config" / "default.yaml"), {})
    if isinstance(scenario, str):
        scenario = load_scenarios(str(ROOT / "src" / "config" / "scenarios.yaml"))[scenario]
    elif isinstance(scenario, dict):
        scenario = ScenarioConfig.from_mapping("random", scenario)
    cfg = apply_scenario_network_overrides(cfg, scenario)
    for key, value in WANG.items():
        setattr(cfg.network, key, value)
    cfg.network.freeway_buffer_segments = int(fw_buffer)
    cfg.network.terminal_zero_gradient = True
    return cfg, scenario


def make_random_scenario(rng, holdout_demand: float = 1.80):
    demand = float(rng.uniform(1.55, 2.40))
    stressor = rng.choice(["none", "skew", "incident"], p=[0.4, 0.3, 0.3])
    if stressor != "none":
        demand = min(demand, holdout_demand)
    demand *= float(rng.uniform(0.98, 1.02))
    scenario = {
        "urban_scale": demand,
        "freeway_scale": demand,
        "ramp_scale": demand,
        "incident_capacity_factor": 1.0,
        "pulse_base_scale": 0.5,
        "pulse_start_sec": 900.0,
        "pulse_rampup_sec": 360.0,
        "pulse_plateau_sec": 3600.0,
        "pulse_rampdown_sec": 360.0,
        "required": False,
    }
    if stressor == "skew":
        scenario["urban_west_east_ratio"] = float(rng.uniform(1.3, 2.0))
    elif stressor == "incident":
        segment = int(rng.integers(3, 8))
        start = float(rng.choice([1260.0, 1800.0, 2400.0]))
        duration = float(rng.choice([1200.0, 1800.0, 2400.0]))
        scenario["freeway_lane_closures"] = [{
            "link": str(rng.choice(["FW_E", "FW_W"])),
            "segment": segment,
            "lane_loss": 1.0,
            "start_sec": start,
            "end_sec": start + duration,
        }]
    return scenario


def make_targeted_scenario(rng):
    """Jitter the five diagnosis cells while retaining their stressor identity."""
    names, weights = zip(*TARGETED_SCENARIO_WEIGHTS)
    target = str(rng.choice(names, p=weights))
    source = load_scenarios(str(ROOT / "src" / "config" / "scenarios.yaml"))[target]
    scenario = {key: value for key, value in asdict(source).items() if value is not None}
    scenario.pop("name", None)
    scenario.pop("metadata", None)
    demand = float(source.urban_scale) * float(rng.uniform(0.98, 1.02))
    scenario.update({
        "urban_scale": demand,
        "freeway_scale": demand,
        "ramp_scale": demand,
        "target_scenario": target,
    })
    if target == "sweet_170_skew15_w60":
        scenario["urban_west_east_ratio"] = float(rng.uniform(1.4, 1.6))
    elif target == "sweet_170_incident_w60":
        start = float(rng.choice((1620.0, 1800.0, 1980.0)))
        duration = float(rng.choice((1440.0, 1800.0, 2160.0)))
        scenario["freeway_lane_closures"] = [{
            "link": str(rng.choice(("FW_E", "FW_E", "FW_E", "FW_W"))),
            "segment": int(rng.choice((5, 6, 6, 7))),
            "lane_loss": 1.0,
            "start_sec": start,
            "end_sec": start + duration,
        }]
    return scenario


class RLLeaderEnv:
    """One full coordination action -> shared follower/safety/plant pipeline."""

    def __init__(
        self,
        scenario_name: str = "sweet_170_incident_w60",
        T_total: float = 14400.0,
        warmup_nc_steps: int = 5,
        scenario_dict: dict | None = None,
        action_mode: str = "full",
        mask: str = "RL-FULL",
    ):
        self.scenario_name = scenario_name if scenario_dict is None else "random"
        self.cfg, self.scenario = make_cfg(scenario_dict if scenario_dict is not None else scenario_name)
        self.cfg.simulation.T_total = float(T_total)
        self.T_total = float(T_total)
        self.dt = float(self.cfg.simulation.control_interval)
        self.n_steps = int(self.T_total / self.dt)
        self.warmup = int(warmup_nc_steps)
        self.net = self.cfg.network
        self.action_mode = str(action_mode)
        if self.action_mode not in {"full", "legacy_budget"}:
            raise ValueError("action_mode must be 'full' or 'legacy_budget'")
        self.mask = CoordinationMask.named(mask)
        self.action_schema = CoordinationActionSchema(self.cfg)
        self.observation_schema = CoordinationObservationSchema(self.cfg)
        self.action_dim = self.action_schema.dimension if self.action_mode == "full" else 2
        self.obs_dim = self.observation_schema.dimension if self.action_mode == "full" else 13
        self.reset()

    def _new_controller(self) -> None:
        neutral = self.action_schema.decode(
            np.zeros(self.action_schema.dimension, dtype=np.float32),
            self._fixed_prev(),
            CoordinationMask.named("ZERO"),
        )
        self.provider = StaticCoordinationProvider(neutral)
        self.controller = RLStackelbergController(self.cfg, self.provider)
        self.follower = self.controller.nash_solver
        self.optimizer_controller = None
        self.optimizer_cfg = None
        self._optimizer_fargate_stress = False
        self._active_controller = self.controller

    def reset(self):
        self.sim = MixedTrafficSimulator(self.cfg)
        self.profile = DemandProfile(self.cfg, self.scenario)
        self.previous = self._fixed_prev()
        self.step_idx = 0
        self._new_controller()
        for _ in range(self.warmup):
            self._advance(self._fixed_prev())
        return self._observe()

    def _full_raw_action(self, action: Sequence[float]) -> np.ndarray:
        raw = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if self.action_mode == "full":
            if raw.size != self.action_schema.dimension:
                raise ValueError(f"expected action dimension {self.action_schema.dimension}, got {raw.size}")
            return raw.astype(np.float32)
        if raw.size != 2:
            raise ValueError(f"expected legacy budget action dimension 2, got {raw.size}")
        full = np.zeros(self.action_schema.dimension, dtype=np.float32)
        full[:2] = raw
        return full

    def step(self, action):
        self._active_controller = self.controller
        full_raw = self._full_raw_action(action)
        mask = self.mask if self.action_mode == "full" else CoordinationMask.named("RL-BUDGET")
        coordination = self.action_schema.decode(full_raw, self.previous, mask)
        self.provider.action = coordination
        forecast = self._forecast()
        inventory_before = self._inventory()
        result = self.controller.decide_with_info(self.sim.state.copy(), forecast, self.previous)
        control = result.control
        log = self.sim.step(control, forecast[0], self.step_idx)
        inventory_after = self._inventory()
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        done = self.step_idx >= self.n_steps
        info = self._step_info(
            coordination, control, log.diagnostics, forecast[0], inventory_before,
            inventory_after, step_ttt, log.urban_ttt, log.freeway_ttt,
        )
        return self._observe(), -step_ttt, done, info

    def step_optimizer_anchor(self):
        """Run native P-Stack directly for parity diagnostics, bypassing the RL adapter."""
        result, forecast, inventory_before, coordination, encoded = self._optimizer_decision()
        self._active_controller = self.optimizer_controller
        log = self.sim.step(result.control, forecast[0], self.step_idx)
        inventory_after = self._inventory()
        self.previous = result.control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        done = self.step_idx >= self.n_steps
        info = self._step_info(
            coordination, result.control, log.diagnostics, forecast[0], inventory_before,
            inventory_after, step_ttt, log.urban_ttt, log.freeway_ttt,
        )
        return self._observe(), -step_ttt, done, info, encoded

    def optimizer_anchor_action(self) -> np.ndarray:
        """Return the native P-Stack action for the current state without advancing the plant."""
        _, _, _, _, encoded = self._optimizer_decision()
        return encoded

    def _ensure_optimizer_controller(self):
        if self.optimizer_controller is None:
            self.optimizer_cfg = copy.deepcopy(self.cfg)
            configure_pstack_b13_follower_contract(self.optimizer_cfg)
            mpc = self.optimizer_cfg.mpc
            mpc.leader_rollout_box_walk = True
            mpc.leader_rollout_box_walk_vg = True
            mpc.leader_mfd_far_state_aware = True
            mpc.leader_mfd_far_real_speed = True
            optimizer = F1StackelbergWuMeteredController(self.optimizer_cfg)
            optimizer.nash_solver.f1_spillback_weight = 0.0
            optimizer.signal_price_enabled = True
            optimizer.offset_price_enabled = True
            optimizer.metering_price_enabled = True
            optimizer.vsl_price_enabled = True
            optimizer.nash_solver.joint_green_offset_enabled = True
            optimizer.metering_price_delta_veh_h = 300.0
            optimizer.metering_price_trust_frac = 0.2
            optimizer.green_offset_cross_price_enabled = False
            optimizer.vsl_meter_cross_price_enabled = False
            optimizer.nash_solver.segment_agents = True
            self.optimizer_controller = optimizer
        return self.optimizer_controller

    def _optimizer_decision(self):
        self._ensure_optimizer_controller()
        forecast = self._forecast()
        self._update_optimizer_far_gate(forecast)
        inventory_before = self._inventory()
        result = self.optimizer_controller.decide_with_info(
            self.sim.state.copy(), forecast, self.previous
        )
        coordination = OptimizerCoordinationProvider.from_controller(
            self.optimizer_controller, result.control, self.action_schema
        )
        encoded = self.action_schema.encode(coordination)
        return result, forecast, inventory_before, coordination, encoded

    def _update_optimizer_far_gate(self, forecast) -> None:
        """Match the b13 hybrid incident/capacity-drop FAR gate for anchor decisions."""
        cfg = self.optimizer_cfg
        rho_crit = float(cfg.network.rho_crit)
        drop_seen = False
        all_subcritical = True
        for link in cfg.network.freeway_links:
            densities = self.sim.state.freeway_density.get(link, [])
            speeds = self.sim.state.freeway_speed.get(link, [])
            lanes = self.sim.state.freeway_effective_lanes.get(link, [])
            for index, density in enumerate(densities):
                density = float(density)
                if density <= rho_crit:
                    continue
                all_subcritical = False
                lane_count = float(lanes[index]) if index < len(lanes) else float(cfg.network.freeway_lanes)
                speed = float(speeds[index]) if index < len(speeds) else 0.0
                flow = segment_flow_veh_h(density, speed, lane_count)
                capacity = segment_flow_veh_h(
                    rho_crit,
                    desired_speed_kmh(rho_crit, cfg.network.v_free, rho_crit),
                    lane_count,
                )
                if flow < 0.95 * capacity:
                    drop_seen = True
        if drop_seen:
            self._optimizer_fargate_stress = True
        elif all_subcritical:
            self._optimizer_fargate_stress = False
        incident_forecast = any(
            float(loss) > 0.0
            for segments in merge_freeway_lane_loss(list(forecast)).values()
            for loss in segments.values()
        )
        cfg.mpc.leader_mfd_far_enabled = bool(
            self._optimizer_fargate_stress or incident_forecast
        )

    def _step_info(
        self, coordination, control, diagnostics, demand, inventory_before,
        inventory_after, step_ttt, urban_ttt, freeway_ttt,
    ):
        mainline_arrivals = sum(float(v) for v in demand.freeway_mainline.values()) * self.cfg.simulation.T_c_h
        external_arrivals = (
            mainline_arrivals
            + float(diagnostics.get("urban_demand_arrivals_veh", 0.0))
            + float(diagnostics.get("onramp_arrivals_veh", 0.0))
        )
        completed = (
            float(diagnostics.get("boundary_out_sink_veh", 0.0))
            + float(diagnostics.get("mainline_exit_flow_total", 0.0)) * self.cfg.simulation.T_c_h
        )
        residual = inventory_after - inventory_before - external_arrivals + completed
        projection = float(diagnostics.get("movement_queue_projection_veh", 0.0))
        rejected = float(diagnostics.get("coupling_offramp_arrivals_rejected_veh", 0.0))
        overflow = float(diagnostics.get("ramp_queue_overflow_count", 0.0)) + float(
            diagnostics.get("queue_overflow_count", 0.0)
        )
        valid = projection <= 1.0e-9 and rejected <= 1.0e-9 and abs(residual) <= 1.0e-3 and overflow <= 0.0
        raw_n_p, raw_n_uf = coordination.raw_budget or (coordination.N_P_star, coordination.N_UF_star)
        info = {
            "step_ttt": float(step_ttt),
            "cum_ttt": float(self.sim.total_ttt),
            "urban_ttt": float(urban_ttt),
            "freeway_ttt": float(freeway_ttt),
            "raw_N_P": float(raw_n_p),
            "raw_N_UF": float(raw_n_uf),
            "bounded_N_P": float(diagnostics.get("leader_bounded_N_P_star", coordination.N_P_star)),
            "bounded_N_UF": float(diagnostics.get("leader_bounded_N_UF_star", coordination.N_UF_star)),
            "projected_N_P": float(control.N_P_star),
            "projected_N_UF": float(control.N_UF_star),
            "realized_metering_total": float(sum(control.ramp_metering.values())),
            "movement_queue_projection_veh": projection,
            "coupling_offramp_arrivals_rejected_veh": rejected,
            "queue_overflow_count": overflow,
            "external_arrivals_veh": float(external_arrivals),
            "completed_departures_veh": float(completed),
            "inventory_change_veh": float(inventory_after - inventory_before),
            "conservation_residual_veh": float(residual),
            "throughput_veh": float(completed),
            "validity_gate_pass": float(valid),
            "native_price_refresh_count": float(diagnostics.get("wu_b2_price_refresh_count", 0.0)),
        }
        info.update({
            key: float(value)
            for key, value in diagnostics.items()
            if isinstance(value, (int, float, bool)) and key not in info
        })
        for block in coordination.urban_blocks:
            values = (
                float(control.green_times.get(f"{block.owner}_p1", block.reference[0])),
                float(control.offsets.get(block.owner, block.reference[1])),
            )
            info[f"potential_cost_urban_{block.owner}"] = block.value(values, coordination.mask)
        for block in coordination.freeway_blocks:
            link = self.net.ramp_to_freeway[block.owner]
            segment = int(self.net.ramp_merge_segment_index.get(block.owner, 0))
            values = (
                float(control.ramp_metering.get(block.owner, block.reference[0])),
                float(segment_vsl(control, link, segment, self.cfg)),
            )
            info[f"potential_cost_freeway_{block.owner}"] = block.value(values, coordination.mask)
        for block in coordination.vsl_blocks:
            link, segment_text = block.owner.rsplit("__seg", 1)
            value = segment_vsl(control, link, int(segment_text), self.cfg)
            info[f"potential_cost_vsl_{block.owner}"] = block.value(value, coordination.mask)
        return info

    def response_vector(self) -> np.ndarray:
        values = []
        for signal in self.action_schema.signals:
            values.extend((
                float(self.previous.green_times.get(f"{signal}_p1", 0.0)),
                float(self.previous.offsets.get(signal, 0.0)),
            ))
        for ramp in self.action_schema.ramps:
            link = self.net.ramp_to_freeway[ramp]
            segment = int(self.net.ramp_merge_segment_index.get(ramp, 0))
            values.extend((
                float(self.previous.ramp_metering.get(ramp, 0.0)),
                float(segment_vsl(self.previous, link, segment, self.cfg)),
            ))
        for key in self.action_schema.nonmerge_vsl_keys:
            link, segment_text = key.rsplit("__seg", 1)
            values.append(float(segment_vsl(
                self.previous, link, int(segment_text), self.cfg,
            )))
        return np.asarray(values, dtype=np.float32)

    def budget_to_action(self, n_p: float, n_uf: float) -> np.ndarray:
        n_p_lo, n_p_hi = map(float, self.cfg.leader.N_P_star_range)
        n_uf_lo, n_uf_hi = map(float, self.cfg.leader.N_UF_star_range)
        return np.clip(np.asarray([
            2.0 * (float(n_p) - n_p_lo) / max(n_p_hi - n_p_lo, 1.0e-9) - 1.0,
            2.0 * (float(n_uf) - n_uf_lo) / max(n_uf_hi - n_uf_lo, 1.0e-9) - 1.0,
        ], dtype=np.float32), -1.0, 1.0)

    def step_with_control(self, control):
        forecast = self._forecast()
        log = self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        return self._observe(), step_ttt, self.step_idx >= self.n_steps

    def _forecast(self):
        return self.profile.horizon(
            self.step_idx * self.dt,
            self.cfg.mpc.horizon_steps + max(0, self.cfg.mpc.leader_value_depth),
        )

    def _fixed_prev(self):
        return ControlAction.fixed(self.cfg)

    def _advance(self, control):
        forecast = self._forecast()
        self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1

    def _inventory(self) -> float:
        state = self.sim.state
        buffer_vehicles = 0.0
        for density_map in (
            state.freeway_buffer_up_density,
            state.freeway_buffer_down_density,
        ):
            buffer_vehicles += sum(
                max(0.0, float(rho))
                * float(self.net.freeway_segment_length_km)
                * float(self.net.freeway_lanes)
                for values in density_map.values()
                for rho in values
            )
        return float(
            state.total_urban_vehicles(self.net)
            + state.total_freeway_vehicles(self.net)
            + state.off_ramp_storage_occupancy_veh(self.net)
            + buffer_vehicles
        )

    def _observe(self):
        if self.action_mode == "legacy_budget":
            return self._observe_legacy()
        return self.observation_schema.observe(
            self.sim.state, self._forecast(), self.previous, self._active_controller, self.n_steps,
        )

    def _observe_legacy(self):
        state = self.sim.state
        ramp_queue = float(sum(state.ramp_queue.values()))
        origin_queue = float(sum(max(0.0, value) for value in state.mainline_origin_queue.values()))
        density = np.asarray([
            value
            for link in self.net.freeway_links
            for value in state.freeway_density.get(link, [])
        ] or [0.0], dtype=float)
        speed = [
            value
            for link in self.net.freeway_links
            for value in state.freeway_speed.get(link, [])
        ]
        demand = self._forecast()[0]
        return np.asarray([
            state.total_urban_vehicles(self.net) / 1000.0,
            state.freeway_segment_vehicles(self.net) / 1000.0,
            ramp_queue / 100.0,
            origin_queue / 100.0,
            float(density.mean()) / self.net.rho_crit,
            float(density.max()) / self.net.rho_crit,
            float((density > self.net.rho_crit).sum()) / 10.0,
            float(np.mean(speed)) / self.net.v_free if speed else 1.0,
            sum(demand.freeway_mainline.values()) / 5000.0,
            sum(demand.ramp_arrival.values()) / 2000.0,
            float(self.previous.N_P_star) / max(float(self.cfg.leader.N_P_star_range[1]), 1.0),
            float(self.previous.N_UF_star) / max(float(self.cfg.leader.N_UF_star_range[1]), 1.0),
            float(self.step_idx) / max(self.n_steps, 1),
        ], dtype=np.float32)


def main() -> None:
    env = RLLeaderEnv(T_total=1260.0)
    observation = env.reset()
    print(f"obs_dim={env.obs_dim} action_dim={env.action_dim}")
    for _ in range(2):
        observation, reward, done, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
        print(
            f"step={env.step_idx} reward={reward:.3f} "
            f"valid={int(info['validity_gate_pass'])} native_refresh={info['native_price_refresh_count']:.0f}"
        )
        if done:
            break


if __name__ == "__main__":
    main()
