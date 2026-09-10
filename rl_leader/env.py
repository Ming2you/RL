"""Gym-like wrapper around the production RL Stackelberg controller path."""
from __future__ import annotations

import sys
import copy
import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.coordination import (
    CoordinationAction,
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
from src.controllers.pstack_factory import (
    CANONICAL_PSTACK_OPTIONS,
    make_pstack_allprice_joint_controller,
)
from src.controllers.stackelberg_mpc import mfd_far_cost_to_go
from src.controllers.wu_faithful_follower import WuFaithfulFollower
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
from src.simulation.coupling import run_coupled_interval

from rl_leader.experiment_contract import (
    ExperimentContract,
    canonicalize_experiment_config,
    experiment_contract_fingerprint,
    experiment_contract_payload,
)


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
OPTIMIZER_ANCHOR_CONTRACT = "pstack_allprice_joint_grid_native_v3"
OPTIMIZER_ANCHOR_SUPERVISOR_CONTRACT = "pstack_b13_hybrid_far_link_pfo_supervisor_v2"
OPTIMIZER_ANCHOR_TRANSITION_CONTRACT = (
    "pstack_anchor_gate_exact_common_follower_seed_v5"
)
OPTIMIZER_PREVIEW_CONTRACT = "deepcopy_side_effect_free_v1"
WARMUP_CONTROL_CONTRACT = "uncontrolled_native_v1"
PSTACK_ANCHOR_CONTEXT_CONTRACT = (
    "pstack_same_state_anchor_context_v3_hidden_follower_fingerprint"
)


def _normalize_runtime_fingerprint(value):
    if is_dataclass(value):
        return _normalize_runtime_fingerprint(asdict(value))
    if isinstance(value, np.ndarray):
        return _normalize_runtime_fingerprint(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _normalize_runtime_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "diagnostics"
        }
    if isinstance(value, (list, tuple, deque)):
        return [_normalize_runtime_fingerprint(item) for item in value]
    if isinstance(value, set):
        return sorted(
            (_normalize_runtime_fingerprint(item) for item in value), key=repr
        )
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _normalize_runtime_fingerprint(item)
            for key, item in sorted(value.__dict__.items())
            if key not in {
                "cfg", "_specs", "_phase_movements", "_local_models",
                "_local_freeway_models", "_segment_agent_models",
            }
        }
    return repr(value)


@dataclass(frozen=True)
class PStackAnchorContext:
    """A native P-Stack decision tied to one exact pre-action state."""

    contract_version: str
    experiment_contract_sha256: str
    state_fingerprint: str
    step_idx: int
    simulation_time_sec: float
    result: object
    forecast: tuple
    inventory_before: float
    coordination: CoordinationAction
    raw_action: np.ndarray
    anchor_fingerprint: str
    optimizer_controller: object
    optimizer_pfo_supervisor: object | None
    follower_seed: object


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
    return canonicalize_experiment_config(cfg), scenario


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


def make_targeted_scenario(rng, target_scenarios=None):
    """Jitter the five diagnosis cells while retaining their stressor identity."""
    weighted = TARGETED_SCENARIO_WEIGHTS
    if target_scenarios:
        requested = set(map(str, target_scenarios))
        weighted = tuple(
            (name, weight) for name, weight in weighted if name in requested
        )
        missing = requested - {name for name, _ in weighted}
        if missing:
            raise ValueError(f"unknown targeted scenarios: {sorted(missing)}")
    names, weights = zip(*weighted)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
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
        response_candidate_count: int = 1,
        response_value_depth: int = 0,
        strict_pfo_gate: bool = False,
        pfo_supervisor: bool = False,
        pstack_anchor: bool = False,
        action_parameterization: str = "absolute",
        experiment_contract: ExperimentContract | None = None,
    ):
        if experiment_contract is None:
            self.scenario_name = scenario_name if scenario_dict is None else "random"
            self.cfg, self.scenario = make_cfg(
                scenario_dict if scenario_dict is not None else scenario_name
            )
            self.cfg.simulation.T_total = float(T_total)
            self.T_total = float(T_total)
            self.warmup = int(warmup_nc_steps)
            self.pstack_options = CANONICAL_PSTACK_OPTIONS
        else:
            self.cfg, self.scenario, self.pstack_options = (
                experiment_contract.materialize()
            )
            payload = experiment_contract.payload
            self.scenario_name = str(payload["scenario_name"])
            self.T_total = float(payload["simulation"]["T_total_sec"])
            self.warmup = int(payload["warmup"]["steps"])
            pfo_supervisor = payload["supervisor"]["mode"] != "none"
        self.dt = float(self.cfg.simulation.control_interval)
        self.n_steps = int(self.T_total / self.dt)
        self.net = self.cfg.network
        self.action_mode = str(action_mode)
        if self.action_mode not in {"full", "legacy_budget"}:
            raise ValueError("action_mode must be 'full' or 'legacy_budget'")
        self.mask = CoordinationMask.named(mask)
        self.response_candidate_count = int(response_candidate_count)
        if not 1 <= self.response_candidate_count <= 10:
            raise ValueError("response_candidate_count must be between 1 and 10")
        self.response_value_depth = int(response_value_depth)
        if self.response_value_depth < 0:
            raise ValueError("response_value_depth must be nonnegative")
        self.strict_pfo_gate = bool(strict_pfo_gate)
        self.pfo_supervisor_enabled = bool(pfo_supervisor)
        self.pstack_anchor_enabled = bool(pstack_anchor)
        if self.pstack_anchor_enabled and self.pfo_supervisor_enabled:
            raise ValueError(
                "P-Stack anchor cannot be combined with the PFO supervisor; "
                "the anchor baseline must remain native P-Stack"
            )
        self.action_parameterization = str(action_parameterization)
        if self.action_parameterization not in {"absolute", "pstack_residual"}:
            raise ValueError("action_parameterization must be absolute or pstack_residual")
        self.action_schema = CoordinationActionSchema(self.cfg)
        self.observation_schema = CoordinationObservationSchema(self.cfg)
        self.action_dim = self.action_schema.dimension if self.action_mode == "full" else 2
        self.obs_dim = self.observation_schema.dimension if self.action_mode == "full" else 13
        self.reset()
        resolved_contract = ExperimentContract.from_env(self)
        if (
            experiment_contract is not None
            and resolved_contract.sha256 != experiment_contract.sha256
        ):
            raise ValueError(
                "materialized experiment contract drifted during environment construction: "
                f"expected {experiment_contract.sha256}, got {resolved_contract.sha256}"
            )
        self._experiment_contract = resolved_contract

    def _new_controller(self) -> None:
        neutral = self.action_schema.decode(
            np.zeros(self.action_schema.dimension, dtype=np.float32),
            self._fixed_prev(),
            CoordinationMask.named("ZERO"),
        )
        self.provider = StaticCoordinationProvider(neutral)
        self.controller = RLStackelbergController(
            self.cfg,
            self.provider,
            response_candidate_count=self.response_candidate_count,
            response_value_depth=self.response_value_depth,
            strict_pfo_gate=self.strict_pfo_gate,
            allow_internal_pfo_fallback=not self.pstack_anchor_enabled,
        )
        self.optimizer_controller = None
        self.optimizer_cfg = None
        self.pfo_supervisor = (
            self._make_link_pfo_supervisor(self.cfg)
            if self.pfo_supervisor_enabled else None
        )
        self.optimizer_pfo_supervisor = None
        self._pending_optimizer_trial = None
        self.last_optimizer_anchor_metadata: dict[str, float | str] = {}
        self.last_optimizer_anchor_response: np.ndarray | None = None
        self.last_optimizer_anchor_coordination = None
        self._rl_fargate_stress = False
        self._optimizer_fargate_stress = False
        self._active_controller = self.controller
        self.last_policy_raw_action: np.ndarray | None = None
        self.last_deployed_residual: np.ndarray | None = None
        self.last_anchor_raw_action: np.ndarray | None = None
        self.last_applied_raw_action: np.ndarray | None = None
        self.last_requested_coordination = None

    @staticmethod
    def _make_link_pfo_supervisor(cfg):
        pfo_cfg = copy.deepcopy(cfg)
        pfo_cfg.mpc.seg13_meter_box_veh_h = None
        if hasattr(pfo_cfg.mpc, "seg13_meter_box_up_veh_h"):
            pfo_cfg.mpc.seg13_meter_box_up_veh_h = None
        pfo_cfg.mpc.seg13_vsl_box_kmh = None
        pfo_cfg.mpc.baseline_move_box = True
        return WuFaithfulFollower(pfo_cfg)

    @property
    def follower(self):
        return self.controller.nash_solver

    @property
    def experiment_contract_payload(self) -> dict:
        contract = getattr(self, "_experiment_contract", None)
        return contract.payload if contract is not None else experiment_contract_payload(self)

    @property
    def experiment_contract_fingerprint(self) -> str:
        contract = getattr(self, "_experiment_contract", None)
        if contract is not None:
            return contract.sha256
        return experiment_contract_fingerprint(self.experiment_contract_payload)

    @property
    def experiment_contract(self) -> ExperimentContract:
        return self._experiment_contract

    def reset(self):
        self.sim = MixedTrafficSimulator(self.cfg)
        self.profile = DemandProfile(self.cfg, self.scenario)
        self.previous = self._fixed_prev()
        self.step_idx = 0
        self._new_controller()
        for _ in range(self.warmup):
            self._advance(ControlAction.uncontrolled(self.cfg))
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

    def _policy_raw_action(self, action: Sequence[float]) -> np.ndarray:
        raw = self._full_raw_action(action)
        if self.action_parameterization == "pstack_residual":
            certificate_count = len(self.action_schema.certificate_ramps)
            if certificate_count:
                raw[-certificate_count:] = 0.0
        return raw

    def step(self, action):
        return self._step(action)

    def step_anchored_candidate(
        self,
        action,
        anchor_context: PStackAnchorContext,
    ):
        """Apply one executable residual against a precomputed same-state anchor."""
        return self._step(
            action,
            anchor_context=anchor_context,
            apply_anchor_gate=False,
        )

    def _step(
        self,
        action,
        *,
        anchor_context: PStackAnchorContext | None = None,
        apply_anchor_gate: bool = True,
    ):
        self._active_controller = self.controller
        if anchor_context is not None:
            self._validate_anchor_context(anchor_context)
        policy_raw = self._policy_raw_action(action)
        forecast = self._forecast()
        anchor_result = None
        anchor_coordination = None
        anchor_raw = None
        if anchor_context is not None:
            forecast = list(copy.deepcopy(anchor_context.forecast))
            anchor_coordination = anchor_context.coordination
            anchor_raw = np.asarray(anchor_context.raw_action, dtype=np.float32).copy()
            self.last_optimizer_anchor_coordination = anchor_context.coordination
            self.last_optimizer_anchor_response = self.response_vector(
                anchor_context.result.control
            )
            if apply_anchor_gate:
                anchor_result = anchor_context.result
                self._pending_optimizer_trial = (
                    anchor_context.optimizer_controller,
                    anchor_context.optimizer_pfo_supervisor,
                )
        elif self.pstack_anchor_enabled or self.action_parameterization == "pstack_residual":
            (
                anchor_result,
                anchor_forecast,
                _,
                anchor_coordination,
                anchor_raw,
            ) = self._optimizer_decision(
                sync_follower_state=True,
                commit=False,
            )
            if self.action_parameterization == "pstack_residual":
                forecast = anchor_forecast
        self._update_rl_far_gate(forecast)
        prepared_rl_controller = self.controller
        prepared_rl_pfo = self.pfo_supervisor
        prepared_rl_controller.prepare_applied_transition(self.sim.state)
        trial_rl_controller = self._clone_controller_for_cfg(
            prepared_rl_controller, self.cfg
        )
        anchor_follower_seed = None
        if anchor_context is not None:
            anchor_follower_seed = anchor_context.follower_seed
        elif self._pending_optimizer_trial is not None:
            anchor_follower_seed = getattr(
                self._pending_optimizer_trial[0],
                "last_candidate_common_solver",
                None,
            )
        if anchor_follower_seed is not None:
            trial_rl_controller._anchor_follower_seed = self._clone_follower_for_cfg(
                anchor_follower_seed,
                self.cfg,
            )
        elif self.action_parameterization == "pstack_residual":
            self._discard_optimizer_trial()
            raise RuntimeError(
                "P-Stack residual evaluation is missing the native follower seed"
            )
        trial_rl_pfo = self._clone_follower_for_cfg(prepared_rl_pfo, self.cfg)
        if self.action_parameterization == "pstack_residual":
            if (
                (anchor_context is None and not self.pstack_anchor_enabled)
                or anchor_raw is None
                or anchor_coordination is None
            ):
                raise RuntimeError("P-Stack residual actions require the P-Stack anchor gate")
            full_raw = np.clip(anchor_raw + policy_raw, -1.0, 1.0).astype(np.float32)
        else:
            full_raw = policy_raw
        mask = self.mask if self.action_mode == "full" else CoordinationMask.named("RL-BUDGET")
        if self.action_parameterization == "pstack_residual":
            coordination = self.action_schema.decode_anchored_residual(
                policy_raw, anchor_coordination, mask
            )
        else:
            coordination = self.action_schema.decode(full_raw, self.previous, mask)
        self.controller = trial_rl_controller
        self.provider = self.controller.coordination_provider
        self.provider.action = coordination
        self.pfo_supervisor = trial_rl_pfo
        self.last_policy_raw_action = policy_raw.copy()
        self.last_anchor_raw_action = None if anchor_raw is None else anchor_raw.copy()
        self.last_requested_coordination = coordination
        inventory_before = self._inventory()
        try:
            result = self.controller.decide_with_info(
                self.sim.state.copy(), forecast, self.previous
            )
        except Exception:
            self.controller = prepared_rl_controller
            self.provider = self.controller.coordination_provider
            self.pfo_supervisor = prepared_rl_pfo
            self._discard_optimizer_trial()
            raise
        control = result.control
        if self.pfo_supervisor is not None:
            control, selected_nash, supervisor_metadata = self._pfo_supervisor_select(
                control,
                forecast,
                self.pfo_supervisor,
                far_enabled=bool(self.cfg.mpc.leader_mfd_far_enabled),
            )
            if selected_nash is not None:
                self._copy_follower_runtime_state(
                    self.controller.nash_solver,
                    self.pfo_supervisor,
                )
            control.diagnostics.update(supervisor_metadata)
        applied_coordination = self.controller.last_coordination_action or coordination
        selected_raw = full_raw
        deployed_residual = (
            policy_raw.copy()
            if self.action_parameterization == "pstack_residual"
            else None
        )
        if anchor_result is not None:
            control, anchor_metadata = self._pstack_anchor_select(
                control,
                anchor_result.control,
                forecast,
            )
            if anchor_metadata["leader_rl_pstack_anchor_pick_pstack"] > 0.5:
                self._commit_optimizer_branch(
                    prepared_rl_controller,
                    anchor_coordination,
                )
                applied_coordination = anchor_coordination
                selected_raw = anchor_raw
                if deployed_residual is not None:
                    deployed_residual = np.zeros_like(policy_raw)
            else:
                self._discard_optimizer_trial()
                self._active_controller = self.controller
            control = control.copy()
            control.diagnostics.update(anchor_metadata)
        else:
            self._discard_optimizer_trial()
            self._active_controller = self.controller
        self.last_applied_raw_action = np.asarray(
            selected_raw, dtype=np.float32
        ).copy()
        self.last_deployed_residual = (
            None
            if deployed_residual is None
            else np.asarray(deployed_residual, dtype=np.float32).copy()
        )
        log = self.sim.step(control, forecast[0], self.step_idx)
        inventory_after = self._inventory()
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        done = self.step_idx >= self.n_steps
        info = self._step_info(
            applied_coordination, control, log.diagnostics, forecast[0], inventory_before,
            inventory_after, step_ttt, log.urban_ttt, log.freeway_ttt,
        )
        return self._observe(), -step_ttt, done, info

    def prepare_pstack_anchor_context(self) -> PStackAnchorContext:
        """Compute one reusable native anchor without advancing the plant."""
        state_fingerprint = self._anchor_context_state_fingerprint()
        try:
            result, forecast, inventory_before, coordination, encoded = (
                self._optimizer_decision(
                    sync_follower_state=True,
                    commit=False,
                )
            )
            if self._pending_optimizer_trial is None:
                raise RuntimeError("native P-Stack anchor did not retain its trial state")
            optimizer_controller, optimizer_pfo_supervisor = self._pending_optimizer_trial
            follower_seed = getattr(
                optimizer_controller,
                "last_candidate_common_solver",
                None,
            )
            if follower_seed is None:
                raise RuntimeError("native P-Stack anchor did not retain its follower seed")
            envelope = self.action_schema.serialize_anchor(coordination)
            return PStackAnchorContext(
                contract_version=PSTACK_ANCHOR_CONTEXT_CONTRACT,
                experiment_contract_sha256=self.experiment_contract_fingerprint,
                state_fingerprint=state_fingerprint,
                step_idx=int(self.step_idx),
                simulation_time_sec=float(self.sim.state.time_sec),
                result=result,
                forecast=tuple(copy.deepcopy(forecast)),
                inventory_before=float(inventory_before),
                coordination=coordination,
                raw_action=np.asarray(encoded, dtype=np.float32).copy(),
                anchor_fingerprint=self.action_schema.anchor_fingerprint(
                    envelope,
                    coordination.selected_branch,
                ),
                optimizer_controller=optimizer_controller,
                optimizer_pfo_supervisor=optimizer_pfo_supervisor,
                follower_seed=follower_seed,
            )
        finally:
            self._discard_optimizer_trial()

    def step_prepared_optimizer_anchor(
        self,
        anchor_context: PStackAnchorContext,
        *,
        sync_follower_state: bool = True,
    ):
        """Commit the native branch stored in a same-state anchor context."""
        self._validate_anchor_context(anchor_context)
        forecast = self._forecast()
        self._update_rl_far_gate(forecast)
        self.controller.prepare_applied_transition(self.sim.state)
        self._pending_optimizer_trial = (
            anchor_context.optimizer_controller,
            anchor_context.optimizer_pfo_supervisor,
        )
        self._commit_optimizer_trial()
        self._active_controller = self.optimizer_controller
        if sync_follower_state:
            self._adopt_optimizer_follower_state(anchor_context.coordination)
        control = anchor_context.result.control
        self.last_policy_raw_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_deployed_residual = np.zeros(self.action_dim, dtype=np.float32)
        self.last_anchor_raw_action = np.asarray(
            anchor_context.raw_action, dtype=np.float32
        ).copy()
        self.last_applied_raw_action = self.last_anchor_raw_action.copy()
        self.last_requested_coordination = anchor_context.coordination
        self.last_optimizer_anchor_coordination = anchor_context.coordination
        inventory_before = self._inventory()
        log = self.sim.step(control, anchor_context.forecast[0], self.step_idx)
        inventory_after = self._inventory()
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        done = self.step_idx >= self.n_steps
        info = self._step_info(
            anchor_context.coordination,
            control,
            log.diagnostics,
            anchor_context.forecast[0],
            inventory_before,
            inventory_after,
            step_ttt,
            log.urban_ttt,
            log.freeway_ttt,
        )
        return (
            self._observe(),
            -step_ttt,
            done,
            info,
            self.last_anchor_raw_action.copy(),
        )

    def step_optimizer_anchor(self, *, sync_follower_state: bool = False):
        """Run native P-Stack directly, optionally continuing from RL follower memory."""
        rl_forecast = self._forecast()
        self._update_rl_far_gate(rl_forecast)
        self.controller.prepare_applied_transition(self.sim.state)
        result, forecast, inventory_before, coordination, encoded = self._optimizer_decision(
            sync_follower_state=sync_follower_state,
            commit=True,
        )
        self._active_controller = self.optimizer_controller
        if sync_follower_state:
            self._adopt_optimizer_follower_state(coordination)
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
        """Return a native P-Stack preview without mutating the live decision state."""
        preview = copy.deepcopy(self)
        _, _, _, _, encoded = preview._optimizer_decision(sync_follower_state=True)
        return encoded

    def _anchor_context_state_fingerprint(self) -> str:
        if self.action_mode == "legacy_budget":
            observation = self._observe_legacy()
        else:
            observation = self.observation_schema.observe(
                self.sim.state,
                self._actor_forecast(),
                self.previous,
                self.controller,
                self.n_steps,
            )
        digest = hashlib.sha256()
        digest.update(self.experiment_contract_fingerprint.encode("ascii"))
        digest.update(np.asarray([self.step_idx], dtype="<i8").tobytes())
        digest.update(np.asarray([self.sim.state.time_sec], dtype="<f8").tobytes())
        digest.update(
            np.ascontiguousarray(np.asarray(observation, dtype="<f8")).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                np.asarray(self.response_vector(self.previous), dtype="<f8")
            ).tobytes()
        )
        digest.update(
            self._follower_runtime_fingerprint(
                self.controller.nash_solver
            ).encode("ascii")
        )
        return digest.hexdigest()

    @staticmethod
    def _follower_runtime_fingerprint(follower) -> str:
        excluded = {
            "cfg", "_wu", "_specs", "_phase_movements", "_local_models",
            "_local_freeway_models", "_segment_agent_models",
            "last_candidate_trace",
        }
        payload = {
            key: value
            for key, value in follower.__dict__.items()
            if key not in excluded
        }
        wu = follower._wu
        wu_excluded = {
            "cfg", "_specs", "_phase_movements", "_local_models",
            "_local_freeway_models",
        }
        payload["wu_runtime"] = {
            key: value
            for key, value in wu.__dict__.items()
            if key not in wu_excluded
        }
        canonical = json.dumps(
            _normalize_runtime_fingerprint(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _validate_anchor_context(self, context: PStackAnchorContext) -> None:
        if not isinstance(context, PStackAnchorContext):
            raise TypeError("anchor_context must be a PStackAnchorContext")
        if context.contract_version != PSTACK_ANCHOR_CONTEXT_CONTRACT:
            raise ValueError(
                f"anchor context contract mismatch: {context.contract_version}"
            )
        if context.experiment_contract_sha256 != self.experiment_contract_fingerprint:
            raise ValueError("anchor context experiment contract mismatch")
        if int(context.step_idx) != int(self.step_idx):
            raise ValueError(
                f"anchor context step mismatch: {context.step_idx} != {self.step_idx}"
            )
        if abs(float(context.simulation_time_sec) - float(self.sim.state.time_sec)) > 1.0e-9:
            raise ValueError("anchor context simulation time mismatch")
        if context.state_fingerprint != self._anchor_context_state_fingerprint():
            raise ValueError("anchor context state fingerprint mismatch")
        encoded = np.asarray(context.raw_action, dtype=np.float32).reshape(-1)
        if encoded.shape != (self.action_schema.dimension,):
            raise ValueError("anchor context action dimension mismatch")
        expected_encoded = self.action_schema.encode(context.coordination)
        if not np.array_equal(encoded, expected_encoded):
            raise ValueError("anchor context encoded action mismatch")
        envelope = self.action_schema.serialize_anchor(context.coordination)
        expected_fingerprint = self.action_schema.anchor_fingerprint(
            envelope,
            context.coordination.selected_branch,
        )
        if context.anchor_fingerprint != expected_fingerprint:
            raise ValueError("anchor context coordination fingerprint mismatch")

    def _record_optimizer_anchor(self, result, encoded: np.ndarray) -> None:
        """Record the native branch label and response for collection diagnostics."""
        certificate_start = self.action_dim - len(self.action_schema.certificate_ramps)
        saturated = np.flatnonzero(np.abs(encoded[:certificate_start]) >= 1.0 - 1.0e-7)
        metadata = result.metadata
        internal_pfo_selected = float(
            metadata.get("leader_pfo_incumbent_selected", 0.0)
        )
        outer_pfo_selected = float(metadata.get("sup_pick_pfo", 0.0))
        pfo_selected = float(
            internal_pfo_selected > 0.5 or outer_pfo_selected > 0.5
        )
        if outer_pfo_selected > 0.5:
            selected_stage = "supervisor_pfo"
        elif internal_pfo_selected > 0.5:
            selected_stage = "fallback_pfo"
        elif float(metadata.get("leader_selected_stage_refined", 0.0)) > 0.5:
            selected_stage = "refined"
        elif float(metadata.get("leader_selected_stage_coarse", 0.0)) > 0.5:
            selected_stage = "coarse"
        else:
            selected_stage = "other"
        self.last_optimizer_anchor_metadata = {
            "teacher_pfo_selected": pfo_selected,
            "teacher_internal_pfo_selected": internal_pfo_selected,
            "teacher_outer_pfo_selected": outer_pfo_selected,
            "teacher_selected_stage": selected_stage,
            "teacher_encoded_saturated_count": float(saturated.size),
            "teacher_encoded_budget_saturated_count": float(
                np.count_nonzero(saturated < 2)
            ),
            "teacher_native_N_P_star": float(result.control.N_P_star),
            "teacher_native_N_UF_star": float(result.control.N_UF_star),
            "teacher_far_enabled": float(self.optimizer_cfg.mpc.leader_mfd_far_enabled),
        }
        self.last_optimizer_anchor_response = self.response_vector(result.control)

    @staticmethod
    def _optimizer_selected_branch(metadata: dict) -> str:
        if float(metadata.get("sup_pick_pfo", 0.0)) > 0.5:
            return "supervisor_pfo"
        if float(metadata.get("leader_pfo_incumbent_selected", 0.0)) > 0.5:
            return "fallback_pfo"
        if float(metadata.get("leader_selected_stage_refined", 0.0)) > 0.5:
            return "refined"
        if float(metadata.get("leader_selected_stage_coarse", 0.0)) > 0.5:
            return "coarse"
        return "other"

    def _sync_optimizer_follower_state(self) -> None:
        """Condition a teacher query on the deployable follower's actual memory state."""
        optimizer = self._ensure_optimizer_controller()
        self._copy_follower_runtime_state(
            optimizer.nash_solver, self.controller.nash_solver,
        )
        if self.pfo_supervisor_enabled and self.pfo_supervisor is not None:
            self.optimizer_pfo_supervisor = copy.deepcopy(self.pfo_supervisor)

    @staticmethod
    def _copy_follower_runtime_state(target, source) -> None:
        runtime_fields = (
            "_prev_coupling",
            "_lambda_P",
            "_lambda_UF",
            "_np_last_sum_nin",
            "_np_prev_accum",
            "_np_step_time",
            "_np_corrector_pending",
            "_np_last_real_q",
            "_np_bias_ratio",
            "_seg13_diag",
            "_seg_traj",
            "_segment_agent_models",
            "_phase_resolved_active_signals",
            "last_candidate_trace",
        )
        for name in runtime_fields:
            if hasattr(source, name):
                setattr(target, name, copy.deepcopy(getattr(source, name)))
        for name in (
            "_last_offramp_flow",
            "_has_last_offramp_flow",
            "_repair_diagnostics",
            "_omega_f",
        ):
            if hasattr(source._wu, name):
                setattr(target._wu, name, copy.deepcopy(getattr(source._wu, name)))

    @staticmethod
    def _clone_controller_for_cfg(controller, cfg):
        return copy.deepcopy(controller, {id(controller.cfg): cfg})

    @staticmethod
    def _clone_follower_for_cfg(follower, cfg):
        if follower is None:
            return None
        return copy.deepcopy(follower, {id(follower.cfg): cfg})

    def _ensure_optimizer_controller(self):
        if self.optimizer_controller is None:
            self.optimizer_cfg = copy.deepcopy(self.cfg)
            configure_pstack_b13_follower_contract(self.optimizer_cfg)
            mpc = self.optimizer_cfg.mpc
            mpc.leader_rollout_box_walk = True
            mpc.leader_rollout_box_walk_vg = True
            mpc.leader_mfd_far_state_aware = True
            mpc.leader_mfd_far_real_speed = True
            optimizer = make_pstack_allprice_joint_controller(
                self.optimizer_cfg,
                options=self.pstack_options,
            )
            self.optimizer_controller = optimizer
            if self.pfo_supervisor_enabled:
                self.optimizer_pfo_supervisor = self._make_link_pfo_supervisor(
                    self.optimizer_cfg
                )
        return self.optimizer_controller

    def _optimizer_decision(
        self,
        *,
        sync_follower_state: bool = False,
        commit: bool = True,
    ):
        self._ensure_optimizer_controller()
        self.optimizer_controller.retain_candidate_common_solver_snapshot = True
        if sync_follower_state:
            self._sync_optimizer_follower_state()
        forecast = self._optimizer_forecast()
        self._update_optimizer_far_gate(forecast)
        inventory_before = self._inventory()
        live_controller = self.optimizer_controller
        live_pfo = self.optimizer_pfo_supervisor
        live_controller.prepare_applied_transition(self.sim.state)
        trial_controller = self._clone_controller_for_cfg(
            live_controller, self.optimizer_cfg
        )
        trial_pfo = self._clone_follower_for_cfg(live_pfo, self.optimizer_cfg)
        self.optimizer_controller = trial_controller
        self.optimizer_pfo_supervisor = trial_pfo
        try:
            result = self.optimizer_controller.decide_with_info(
                self.sim.state.copy(), forecast, self.previous
            )
            if self.optimizer_pfo_supervisor is not None:
                selected_control, selected_nash, supervisor_metadata = (
                    self._pfo_supervisor_select(
                        result.control,
                        forecast,
                        self.optimizer_pfo_supervisor,
                        far_enabled=bool(self.optimizer_cfg.mpc.leader_mfd_far_enabled),
                        score_cfg=self.optimizer_cfg,
                    )
                )
                result.metadata.update(supervisor_metadata)
                selected_control.diagnostics.update(supervisor_metadata)
                result.control = selected_control
                if selected_nash is not None:
                    result.nash = selected_nash
                    result.leader_objective = float(supervisor_metadata["sup_v_pfo"])
                    self._copy_follower_runtime_state(
                        self.optimizer_controller.nash_solver,
                        self.optimizer_pfo_supervisor,
                    )
            coordination = OptimizerCoordinationProvider.from_controller(
                self.optimizer_controller,
                result.control,
                self.action_schema,
                selected_branch=self._optimizer_selected_branch(result.metadata),
            )
            self.last_optimizer_anchor_coordination = coordination
            encoded = self.action_schema.encode(coordination)
            self._record_optimizer_anchor(result, encoded)
        except Exception:
            self.optimizer_controller = live_controller
            self.optimizer_pfo_supervisor = live_pfo
            raise
        if commit:
            self._pending_optimizer_trial = None
        else:
            self._pending_optimizer_trial = (
                self.optimizer_controller,
                self.optimizer_pfo_supervisor,
            )
            self.optimizer_controller = live_controller
            self.optimizer_pfo_supervisor = live_pfo
        return result, forecast, inventory_before, coordination, encoded

    def _commit_optimizer_trial(self) -> None:
        if self._pending_optimizer_trial is None:
            raise RuntimeError("missing optimizer trial for selected P-Stack branch")
        self.optimizer_controller, self.optimizer_pfo_supervisor = (
            self._pending_optimizer_trial
        )
        self._pending_optimizer_trial = None

    def _discard_optimizer_trial(self) -> None:
        self._pending_optimizer_trial = None

    def _fixed_control_score(self, control, forecast, cfg=None) -> float:
        score_cfg = self.cfg if cfg is None else cfg
        state = self.sim.state.copy()
        total = 0.0
        horizon = max(1, int(score_cfg.mpc.horizon_steps))
        for demand in list(forecast)[:horizon]:
            result = run_coupled_interval(state, control, demand, score_cfg)
            total += float(result.urban_ttt + result.freeway_ttt)
            state.time_sec += score_cfg.simulation.control_interval
        return float(total + mfd_far_cost_to_go(score_cfg, state))

    def _fixed_control_metrics(self, control, forecast, cfg=None) -> dict[str, float]:
        score_cfg = self.cfg if cfg is None else cfg
        state = self.sim.state.copy()
        total_ttt = 0.0
        horizon = max(1, int(score_cfg.mpc.horizon_steps))
        for demand in list(forecast)[:horizon]:
            result = run_coupled_interval(state, control, demand, score_cfg)
            total_ttt += float(result.urban_ttt + result.freeway_ttt)
            state.time_sec += score_cfg.simulation.control_interval
        return {
            "ttt": float(total_ttt),
            "terminal_inventory": self._state_inventory(state, score_cfg.network),
        }

    def _pstack_anchor_select(
        self,
        rl_control,
        pstack_control,
        forecast,
        *,
        force_pstack: bool = False,
    ):
        rl_metrics = self._fixed_control_metrics(rl_control, forecast)
        pstack_metrics = self._fixed_control_metrics(pstack_control, forecast)
        rl_ttt = float(rl_metrics["ttt"])
        pstack_ttt = float(pstack_metrics["ttt"])
        rl_inventory = float(rl_metrics["terminal_inventory"])
        pstack_inventory = float(pstack_metrics["terminal_inventory"])
        gain = float(pstack_ttt - rl_ttt)
        required_gain = max(0.1, 1.0e-3 * max(abs(pstack_ttt), 1.0))
        inventory_blocked = rl_inventory > pstack_inventory + 1.0e-6
        unforced_pick_rl = gain > required_gain and not inventory_blocked
        pick_rl = bool(unforced_pick_rl and not force_pstack)
        metadata = {
            "leader_rl_pstack_anchor_enabled": 1.0,
            "leader_rl_pstack_anchor_pick_rl": float(pick_rl),
            "leader_rl_pstack_anchor_pick_pstack": float(not pick_rl),
            "leader_rl_pstack_anchor_identity_forced": float(force_pstack),
            "leader_rl_pstack_anchor_unforced_pick_rl": float(unforced_pick_rl),
            "leader_rl_pstack_anchor_rl_score": rl_ttt,
            "leader_rl_pstack_anchor_pstack_score": pstack_ttt,
            "leader_rl_pstack_anchor_rl_ttt": rl_ttt,
            "leader_rl_pstack_anchor_pstack_ttt": pstack_ttt,
            "leader_rl_pstack_anchor_rl_terminal_inventory": rl_inventory,
            "leader_rl_pstack_anchor_pstack_terminal_inventory": pstack_inventory,
            "leader_rl_pstack_anchor_inventory_blocked": float(inventory_blocked),
            "leader_rl_pstack_anchor_gain": gain,
            "leader_rl_pstack_anchor_required_gain": float(required_gain),
        }
        return (rl_control if pick_rl else pstack_control), metadata

    def _adopt_optimizer_follower_state(self, coordination) -> None:
        """Commit the follower memory belonging to the selected P-Stack branch."""
        self._copy_follower_runtime_state(
            self.controller.nash_solver,
            self.optimizer_controller.nash_solver,
        )
        if self.pfo_supervisor_enabled and self.optimizer_pfo_supervisor is not None:
            self.pfo_supervisor = copy.deepcopy(self.optimizer_pfo_supervisor)
        self.controller.last_coordination_action = coordination

    def _commit_optimizer_branch(self, prepared_rl_controller, coordination) -> None:
        """Install the selected optimizer trial and discard the RL proposal lane."""
        if prepared_rl_controller is None:
            raise RuntimeError("missing pre-RL controller snapshot for P-Stack commit")
        self._commit_optimizer_trial()
        self.controller = self._clone_controller_for_cfg(
            prepared_rl_controller, self.cfg
        )
        self.provider = self.controller.coordination_provider
        self._adopt_optimizer_follower_state(coordination)
        self._active_controller = self.optimizer_controller

    def _pfo_supervisor_select(
        self,
        candidate_control,
        forecast,
        supervisor,
        *,
        far_enabled: bool,
        score_cfg=None,
    ):
        metadata = {
            "sup_enabled": 1.0,
            "sup_active": float(not far_enabled),
            "sup_far_gate_blocked": float(far_enabled),
            "sup_pick_pfo": 0.0,
            "sup_v_candidate": 0.0,
            "sup_v_pstack": 0.0,
            "sup_v_pfo": 0.0,
        }
        if far_enabled:
            return candidate_control, None, metadata
        pfo_nash = supervisor.solve(
            self.sim.state.copy(),
            None,
            list(forecast),
            self.previous,
        )
        candidate_score = self._fixed_control_score(
            candidate_control, forecast, score_cfg,
        )
        pfo_score = self._fixed_control_score(pfo_nash.control, forecast, score_cfg)
        pick_pfo = pfo_score < candidate_score - 1.0e-9
        metadata.update({
            "sup_pick_pfo": float(pick_pfo),
            "sup_v_candidate": float(candidate_score),
            "sup_v_pstack": float(candidate_score),
            "sup_v_pfo": float(pfo_score),
        })
        return (
            (pfo_nash.control, pfo_nash, metadata)
            if pick_pfo else (candidate_control, None, metadata)
        )

    def _update_optimizer_far_gate(self, forecast) -> None:
        """Match the b13 hybrid incident/capacity-drop FAR gate for anchor decisions."""
        self._update_far_gate(
            self.optimizer_cfg,
            forecast,
            stress_attribute="_optimizer_fargate_stress",
        )

    def _update_rl_far_gate(self, forecast) -> None:
        """Apply the same b13 FAR gate to the deployable RL controller."""
        self._update_far_gate(
            self.cfg,
            forecast,
            stress_attribute="_rl_fargate_stress",
        )

    def _update_far_gate(self, cfg, forecast, *, stress_attribute: str) -> None:
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
            setattr(self, stress_attribute, True)
        elif all_subcritical:
            setattr(self, stress_attribute, False)
        incident_forecast = any(
            float(loss) > 0.0
            for segments in merge_freeway_lane_loss(list(forecast)).values()
            for loss in segments.values()
        )
        cfg.mpc.leader_mfd_far_enabled = bool(
            getattr(self, stress_attribute) or incident_forecast
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
        active_cfg = getattr(self._active_controller, "cfg", self.cfg)
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
            "leader_mfd_far_enabled": float(active_cfg.mpc.leader_mfd_far_enabled),
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

    def response_vector(self, control=None) -> np.ndarray:
        control = self.previous if control is None else control
        values = []
        for signal in self.action_schema.signals:
            values.extend((
                float(control.green_times.get(f"{signal}_p1", 0.0)),
                float(control.offsets.get(signal, 0.0)),
            ))
        for ramp in self.action_schema.ramps:
            link = self.net.ramp_to_freeway[ramp]
            segment = int(self.net.ramp_merge_segment_index.get(ramp, 0))
            values.extend((
                float(control.ramp_metering.get(ramp, 0.0)),
                float(segment_vsl(control, link, segment, self.cfg)),
            ))
        for key in self.action_schema.nonmerge_vsl_keys:
            link, segment_text = key.rsplit("__seg", 1)
            values.append(float(segment_vsl(
                control, link, int(segment_text), self.cfg,
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

    def _optimizer_forecast(self):
        self._ensure_optimizer_controller()
        return self.profile.horizon(
            self.step_idx * self.dt,
            self.optimizer_cfg.mpc.horizon_steps
            + max(0, self.optimizer_cfg.mpc.leader_value_depth),
        )

    def _actor_forecast(self):
        return self._forecast()[: max(1, int(self.cfg.mpc.horizon_steps))]

    def _fixed_prev(self):
        return ControlAction.fixed(self.cfg)

    def _advance(self, control):
        forecast = self._forecast()
        self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1

    def _inventory(self) -> float:
        return self._state_inventory(self.sim.state, self.net)

    @staticmethod
    def _state_inventory(state, net) -> float:
        buffer_vehicles = 0.0
        for density_map in (
            state.freeway_buffer_up_density,
            state.freeway_buffer_down_density,
        ):
            buffer_vehicles += sum(
                max(0.0, float(rho))
                * float(net.freeway_segment_length_km)
                * float(net.freeway_lanes)
                for values in density_map.values()
                for rho in values
            )
        return float(
            state.total_urban_vehicles(net)
            + state.total_freeway_vehicles(net)
            + state.off_ramp_storage_occupancy_veh(net)
            + buffer_vehicles
        )

    def _observe(self):
        if self.action_mode == "legacy_budget":
            return self._observe_legacy()
        return self.observation_schema.observe(
            self.sim.state,
            self._actor_forecast(),
            self.previous,
            self._active_controller,
            self.n_steps,
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
