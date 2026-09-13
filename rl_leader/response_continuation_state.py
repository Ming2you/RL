"""Conservative post-commit continuation identity with an exact contract scope.

The original 170-incident entry point remains unchanged. The separate five-cell
entry point accepts only five pinned scenarios sharing every non-scenario field.

This is NOT the strict snapshot fingerprint and is not physical-control
equivalence. Equal controls can retain different prices, predictor/corrector
memory, regret history, phase-active sets, FAR hysteresis, or previous budgets.
Those differences must survive. Unknown serializable runtime fields survive too;
unknown objects, config changes and unaudited execution modes are rejected.

Static exclusions (source locations describe the audited implementation):
* env.py:_step/step_prepared_optimizer_anchor replace the provider action and
  action-reporting fields. _active_controller is not an identity label; its
  observation-relevant memory must agree with the canonical RL follower.
* coordination.py:CoordinationPotentialAdapter.clear/apply (1085-1245) replaces
  RL-FULL residual input prices, references, trusts and runtime toggles BEFORE
  the next residual solve. env.py:_copy_follower_runtime_state (860-896) does
  not copy those inputs into native pricing. Native inputs are NEVER excluded.
* stackelberg_wu_metered.py:decide_with_info (474-487) replaces link-share context
  and per-decision dedupe cache; _evaluate_fallback_candidates (2163-2171)
  replaces incumbent scratch. env always supplies its previous control.
* rl_stackelberg.py:_single_coordination_search (149-164) replaces response
  scratch. Pending seeds/trials are rejected, not discarded as if committed.
* wu_faithful_follower.py:_solve_followers (3898-3900) resets trajectory and
  repair/segment diagnostics before use. last_candidate_trace is reporting.
  _phase_resolved_active_signals is NOT excluded: local_green_costs can read it
  before that reset on a native price refresh.
* Simulator cumulative TTT/logs and solve wall time are accounting, not inputs
  to the next interval. The supplied interval reward is separately retained.
* leader.objective_terms emits three accumulation/boundary reports that are
  copied to previous.diagnostics but never consumed by the next decision.
  Excluding those copies does not exclude plant inventory or objective inputs.

Small static topology descriptors are retained conservatively. Their repeated
config graphs are replaced by validated config digests, not serialized inside
each model. No candidate solver graphs/rollouts/process pools are serialized.
Additional diagnostics and native history may still over-distinguish responses.
In particular previous.diagnostics price/objective/stage reports, the RL
_signal_price_last_step and _price_rollout_count still participate. This first
version intentionally does not infer their irrelevance from their names.
Paired real replay parity probes are required before relying on new merges.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.experiment_contract import ExperimentContract
from src.controllers.coordination import (
    CoordinationActionSchema, CoordinationMask, CoordinationObservationSchema,
    CoordinationPotentialAdapter, StaticCoordinationProvider,
)
from src.controllers.f1_wu_faithful_follower import (
    F1StackelbergWuMeteredController, F1WuFaithfulFollower,
)
from src.controllers.leader import Leader
from src.controllers.local_freeway_plant import LocalFreewayModel
from src.controllers.local_signal_plant import LocalSignalModel
from src.controllers.pstack_factory import (
    PStackControllerOptions, configure_pstack_allprice_joint,
)
from src.controllers.rl_stackelberg import RLStackelbergController
from src.controllers.segment_local_plant import SegmentAgentModel
from src.controllers.wu_distributed import WuDistributedController
from src.models import state as state_types
from src.models.demand import DemandProfile, ScenarioConfig
from src.simulation.simulator import MixedTrafficSimulator


IDENTITY_VERSION = "post_commit_continuation_v1"
AUDITED_CONTRACT_SHA256 = (
    "95694fc1e5bb06da621e784a7e4d4bad56135b6d75ac360bb3a42b2d18501831"
)
FIVE_CELL_IDENTITY_VERSION = "post_commit_continuation_five_cell_v1"
# These full contracts include canonical follower configuration performed by reset.
# They can also be resolved without simulation from make_cfg plus
# configure_pstack_b13_follower_contract. Only scenario_name and scenario differ;
# all execution/accounting fields are pinned.
FIVE_CELL_CONTRACT_SHA256 = MappingProxyType({
    "sweet_155_w60": "d0ffa21d79b61bcca7d052bc152b7ea36e5729eedc8840bb673bc9a0eeecf74c",
    "sweet_170_w60": "4f74f8ed8e13d86a535350141ffcfc309b39faad3ffb263a0ab787398fab6c98",
    "sweet_170_incident_w60": AUDITED_CONTRACT_SHA256,
    "sweet_170_skew15_w60": "ff5944e72e4c899f2debafbbd13e0ab9dc7f67551f4fc83a6cbded6e916aff26",
    "sweet_190_w60": "23f6b73e86dbb57f4ef5231b518ff2724b85ba7cd137b1d76158247fba4f4fbd",
})
FIVE_CELL_SHARED_CONTRACT_SHA256 = (
    "89e46bd3cc7a35f5dce5633ab7c24f11bd6331020bdb27706ee177c7214bb98e"
)

# Exact names only: do not generalize these to price/diagnostic name prefixes.
OBSOLETE_RL_INPUTS = frozenset({
    "signal_marginal_price", "offset_marginal_price", "metering_marginal_price",
    "vsl_marginal_price", "green_offset_cross_price", "vsl_meter_cross_price",
    "signal_quadratic_price", "offset_quadratic_price", "metering_quadratic_price",
    "vsl_quadratic_price", "metering_release_certified", "offset_directive",
    "signal_marginal_price_ref", "offset_marginal_price_ref",
    "green_offset_cross_ref", "metering_marginal_price_ref",
    "vsl_marginal_price_ref", "vsl_meter_cross_ref",
    "signal_marginal_price_trust_sec", "offset_marginal_price_trust_sec",
    "metering_marginal_price_trust_frac", "vsl_marginal_price_trust_kmh",
    "joint_green_offset_enabled", "ramp_offset_enabled",
    "priced_vsl_segment_candidates_enabled",
})
_FOLLOWER_SCRATCH = frozenset({"_seg13_diag", "_seg_traj", "last_candidate_trace"})
_CONTROLLER_SCRATCH = frozenset({
    "previous_control", "last_decision", "_link_share_ctx", "_nuf_solve_cache",
    "_dedupe_hits", "_pfo_incumbent_center", "_pfo_incumbent_eval",
    "last_candidate_common_solver",
})
_RL_CONTROLLER_SCRATCH = frozenset({
    "last_coordination_action", "last_coordination_metadata",
    "last_response_candidate_trace", "_response_candidate_actions",
    "_response_candidate_solvers", "_response_pfo_solver", "_signal_price_meta",
})
_ENV_REPORTS = frozenset({
    "_active_controller", "last_optimizer_anchor_metadata",
    "last_optimizer_anchor_response", "last_optimizer_anchor_coordination",
    "last_policy_raw_action", "last_deployed_residual", "last_anchor_raw_action",
    "last_applied_raw_action", "last_requested_coordination",
})
_DIAGNOSTIC_REPORTS = frozenset({
    "leader_base_accumulation", "leader_state_accumulation_base",
    "leader_boundary_leg_excluded_veh",
    "wu_faithful_solve_time_sec", "leader_provider_rl", "leader_provider_optimizer",
    "leader_search_bypassed", "coordination_native_price_disabled",
    "coordination_urban_block_count", "coordination_freeway_block_count",
    "coordination_vsl_block_count", "coordination_linear_active",
    "coordination_quadratic_active", "coordination_cross_active",
    "coordination_ramp_offset_active", "coordination_priced_vsl_candidates_active",
})
_OBSERVED_MEMORY = (
    "_lambda_P", "_lambda_UF", "_prev_coupling", "_np_last_sum_nin",
    "_np_bias_ratio", "_np_prev_accum", "_np_last_real_q", "_np_corrector_pending",
)
_DATA_TYPES = frozenset({
    state_types.SimulationConfig, state_types.NetworkConfig,
    state_types.CapacityDropConfig, state_types.MPCConfig, state_types.LeaderConfig,
    state_types.FreewayFollowerConfig, state_types.UrbanFollowerConfig,
    state_types.EvaluationConfig, state_types.AutoTuningConfig,
    state_types.ExperimentConfig, state_types.TrafficState, state_types.ControlAction,
    ScenarioConfig, CoordinationMask, PStackControllerOptions,
})
_MODEL_TYPES = frozenset({LocalSignalModel, LocalFreewayModel, SegmentAgentModel})


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _type_name(value) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _require_type(value, expected, path):
    if type(value) is not expected:
        raise TypeError(f"{path}: unsupported object type {_type_name(value)}")


def _view(value, excluded=()):
    return {key: item for key, item in vars(value).items() if key not in excluded}


def _encode(value, path="state", active=None, model_config=None):
    """Typed, lossless value encoding; never repr(), pickle, or rounded floats."""
    kind = type(value)
    if value is None or kind in (str, bool):
        return [kind.__name__, value]
    if kind is int:
        return ["int", str(value)]
    if kind is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float")
        return ["float", value.hex()]
    if kind is bytes:
        return ["bytes", value.hex()]
    if kind is np.ndarray or isinstance(value, np.generic):
        array = np.asarray(value)
        if kind is not np.ndarray and kind is not array.dtype.type:
            raise TypeError(f"{path}: unsupported numpy scalar subclass")
        if array.dtype.kind not in "biuf" or array.dtype.fields or array.dtype.metadata:
            raise TypeError(f"{path}: unsupported numpy dtype {array.dtype}")
        if array.dtype.kind == "f" and (array.dtype.itemsize not in (2, 4, 8)
                                       or not np.isfinite(array).all()):
            raise ValueError(f"{path}: unsupported or non-finite numpy float")
        return ["ndarray" if kind is np.ndarray else "numpy_scalar", array.dtype.str,
                list(array.shape), array.tobytes(order="C").hex()]

    active = set() if active is None else active
    if id(value) in active:
        raise ValueError(f"{path}: cyclic state is unsupported")
    active.add(id(value))
    try:
        def encode(item, suffix):
            return _encode(item, path + suffix, active, model_config)

        if kind is dict:
            entries = [[encode(key, ".<key>"), encode(item, "." + str(key))]
                       for key, item in value.items()]
            entries.sort(key=lambda pair: _json(pair[0]))
            return ["dict", entries]
        if kind in (list, tuple, deque):
            entries = [encode(item, f"[{i}]") for i, item in enumerate(value)]
            return [kind.__name__, entries, value.maxlen] if kind is deque else [kind.__name__, entries]
        if kind in (set, frozenset):
            return [kind.__name__, sorted((encode(item, ".<set>") for item in value), key=_json)]
        if kind in _DATA_TYPES:
            return [_type_name(value), encode(vars(value), ".fields")]
        if kind in _MODEL_TYPES and model_config is not None:
            fields = _view(value, {"cfg"})
            config_digest = model_config(value.cfg) if hasattr(value, "cfg") else None
            return [_type_name(value), config_digest, encode(fields, ".descriptor")]
        raise TypeError(f"{path}: unsupported object type {_type_name(value)}")
    finally:
        active.remove(id(value))


def _expect(obj, expected, path):
    for name, value in expected.items():
        if not hasattr(obj, name) or _encode(getattr(obj, name)) != _encode(value):
            raise ValueError(f"{path}.{name}: unsupported mode/value")


def validate_five_cell_contract(env, *, expected_contract_sha256: str) -> dict:
    """Validate an explicitly opted-in, exact five-cell contract without solves.

    The static allowlist cannot be broadened by passing an arbitrary expected hash.
    Runtime FAR activation remains the sole config exception already audited by
    _Projection; runtime state, aliases and controller modes are still validated
    by the unchanged projection after each candidate commits.
    """
    _require_type(env, RLLeaderEnv, "env")
    scenario_name = env.scenario_name
    allowed = FIVE_CELL_CONTRACT_SHA256.get(scenario_name)
    if allowed is None:
        raise ValueError("five-cell continuation: unsupported scenario")
    if type(expected_contract_sha256) is not str or expected_contract_sha256 != allowed:
        raise ValueError("five-cell continuation: expected contract is not the scenario's pinned contract")
    contract = env._experiment_contract
    _require_type(contract, ExperimentContract, "env._experiment_contract")
    if set(vars(contract)) != {"canonical_json", "sha256"}:
        raise ValueError("env: unsupported experiment contract fields")
    validated = ExperimentContract.from_artifact(contract.payload, allowed)
    if contract.sha256 != validated.sha256:
        raise ValueError("env: unsupported experiment contract digest")
    payload = validated.payload
    if payload["scenario_name"] != scenario_name:
        raise ValueError("five-cell continuation: scenario name differs from pinned contract")
    shared = {key: value for key, value in payload.items() if key not in {"scenario", "scenario_name"}}
    if _sha(shared) != FIVE_CELL_SHARED_CONTRACT_SHA256:
        raise ValueError("five-cell continuation: non-scenario contract fields changed")
    base_cfg, scenario, _ = validated.materialize()
    _expect(env, {"scenario": scenario}, "env")
    if _Projection.config_shape(env.cfg) != _Projection.config_shape(base_cfg):
        raise ValueError("five-cell continuation: runtime config differs from pinned contract")
    return {
        "version": FIVE_CELL_IDENTITY_VERSION,
        "scenario": scenario_name,
        "expected_contract_sha256": allowed,
        "experiment_contract_sha256": validated.sha256,
        "shared_contract_sha256": FIVE_CELL_SHARED_CONTRACT_SHA256,
    }


class _Projection:
    def __init__(self, env, *, expected_contract_sha256=AUDITED_CONTRACT_SHA256):
        _require_type(env, RLLeaderEnv, "env")
        contract = env._experiment_contract
        _require_type(contract, ExperimentContract, "env._experiment_contract")
        if set(vars(contract)) != {"canonical_json", "sha256"}:
            raise ValueError("env: unsupported experiment contract fields")
        if expected_contract_sha256 not in FIVE_CELL_CONTRACT_SHA256.values():
            raise ValueError("env: unsupported expected experiment contract")
        validated = ExperimentContract.from_artifact(contract.payload, expected_contract_sha256)
        if contract.sha256 != validated.sha256:
            raise ValueError("env: unsupported experiment contract digest")
        self.base_cfg, scenario, options = validated.materialize()
        native_cfg, _, _ = validated.materialize()
        configure_pstack_allprice_joint(native_cfg, options=options)
        self.config_shapes = (self.config_shape(self.base_cfg), self.config_shape(native_cfg))
        self.config_cache = {}
        _expect(env, {
            "action_mode": "full", "mask": CoordinationMask(),
            "pstack_anchor_enabled": True, "action_parameterization": "pstack_residual",
            "response_candidate_count": 1, "response_value_depth": 0,
            "strict_pfo_gate": False, "pfo_supervisor_enabled": False,
            "pfo_supervisor": None, "optimizer_pfo_supervisor": None,
            "_pending_optimizer_trial": None, "pstack_options": options,
            "scenario": scenario, "scenario_name": validated.payload["scenario_name"],
            "T_total": float(validated.payload["simulation"]["T_total_sec"]),
            "dt": float(validated.payload["simulation"]["control_interval_sec"]),
            "warmup": validated.payload["warmup"]["steps"],
        }, "env")
        if type(env.step_idx) is not int or not env.warmup <= env.step_idx <= env.n_steps:
            raise ValueError("env.step_idx: invalid committed decision boundary")
        if env.n_steps != int(env.T_total / env.dt):
            raise ValueError("env.n_steps: inconsistent horizon")
        for name in ("_rl_fargate_stress", "_optimizer_fargate_stress"):
            _require_type(getattr(env, name), bool, "env." + name)
        self.config(env.cfg, role=0)
        self.config(env.optimizer_cfg, role=1)
        if env.cfg is env.optimizer_cfg:
            raise ValueError("env: RL/native runtime configs must be independent")
        _require_type(env.previous, state_types.ControlAction, "env.previous")
        _require_type(env.sim, MixedTrafficSimulator, "env.sim")
        _require_type(env.sim.state, state_types.TrafficState, "env.sim.state")
        _require_type(env.profile, DemandProfile, "env.profile")
        _require_type(env.action_schema, CoordinationActionSchema, "env.action_schema")
        _require_type(env.observation_schema, CoordinationObservationSchema, "env.observation_schema")
        # Conditional adapter assignments require the complete audited blocks.
        # Changed/partial schemas are not covered by the RL-FULL exclusions.
        for schema, factory, dimension in (
            (env.action_schema, CoordinationActionSchema, env.action_dim),
            (env.observation_schema, CoordinationObservationSchema, env.obs_dim),
        ):
            expected = factory(self.base_cfg)
            if _encode(_view(schema, {"cfg"})) != _encode(_view(expected, {"cfg"})):
                raise ValueError("env: unsupported action/observation schema")
            if dimension != expected.dimension:
                raise ValueError("env: inconsistent schema dimension")
        _require_type(env.provider, StaticCoordinationProvider, "env.provider")
        for name in ("sim", "profile", "action_schema", "observation_schema", "controller"):
            if getattr(env, name).cfg is not env.cfg:
                raise ValueError(f"env.{name}.cfg: unsupported config alias")
        if env.net is not env.cfg.network or env.profile.scenario is not env.scenario:
            raise ValueError("env: unsupported network/scenario alias")
        if env.optimizer_controller.cfg is not env.optimizer_cfg:
            raise ValueError("env.optimizer_controller.cfg: unsupported config alias")
        if env.controller.coordination_provider is not env.provider:
            raise ValueError("env: unsupported provider alias")
        if env._active_controller is not env.controller and env._active_controller is not env.optimizer_controller:
            raise ValueError("env: unsupported active controller")
        # Ignore the branch label, not an actual difference in policy inputs.
        active_follower = env._active_controller.nash_solver
        canonical = env.controller.nash_solver
        if _encode({k: getattr(active_follower, k) for k in _OBSERVED_MEMORY}) != _encode(
            {k: getattr(canonical, k) for k in _OBSERVED_MEMORY}
        ):
            raise ValueError("env: active observation memory differs from canonical RL memory")

    @staticmethod
    def config_shape(cfg):
        _require_type(cfg, state_types.ExperimentConfig, "config")
        _require_type(cfg.mpc, state_types.MPCConfig, "config.mpc")
        _require_type(cfg.mpc.leader_mfd_far_enabled, bool, "config.mpc.leader_mfd_far_enabled")
        fields = _view(cfg)
        fields["mpc"] = _view(cfg.mpc, {"leader_mfd_far_enabled"})
        return _encode(fields)

    def config(self, cfg, role=None):
        shape = self.config_shape(cfg)
        allowed = self.config_shapes if role is None else (self.config_shapes[role],)
        if shape not in allowed:
            raise ValueError("config: runtime config differs from audited contract/modes")
        if id(cfg) not in self.config_cache:
            self.config_cache[id(cfg)] = _sha(_encode(cfg))
        return self.config_cache[id(cfg)]

    def digest(self, value):
        return _sha(_encode(value, model_config=self.config))

    def follower(self, follower, cfg, *, residual):
        _require_type(follower, F1WuFaithfulFollower, "follower")
        _require_type(follower._wu, WuDistributedController, "follower._wu")
        if follower.cfg is not cfg or follower._wu.cfg is not cfg:
            raise ValueError("follower: unsupported config alias")
        _expect(follower, {"authority": "proposed", "segment_agents": True,
                           "metering_enabled": True, "offset_enabled": False,
                           "f1_spillback_weight": 0.0}, "follower")
        _expect(follower._wu, {"leader_enabled": False}, "follower._wu")
        excluded = {"cfg", "_wu"} | _FOLLOWER_SCRATCH
        if residual:
            excluded |= OBSOLETE_RL_INPUTS
        fields = _view(follower, excluded)
        fields["_wu"] = _view(follower._wu, {"cfg", "_repair_diagnostics"})
        return fields

    def controller(self, controller, *, residual):
        expected_type = RLStackelbergController if residual else F1StackelbergWuMeteredController
        _require_type(controller, expected_type, "controller")
        _require_type(controller.leader, Leader, "controller.leader")
        if controller.leader.cfg is not controller.cfg:
            raise ValueError("controller.leader: unsupported config alias")
        _expect(controller, {
            "price_spsa_enabled": False, "price_lite": False, "price_iter_max": 1,
            "price_refresh_interval": 1, "nuf_link_share_mode": "density",
            "price_far_enabled": False, "price_hinge_enabled": False,
            "barrier_price_enabled": False, "candidate_dedupe_enabled": False,
            "cross_cliff_gate_enabled": False, "offset_joint_enabled": False,
            "leader_offset_enabled": False, "green_offset_cross_price_enabled": False,
            "vsl_meter_cross_price_enabled": False, "_leader_process_pool": None,
            "_leader_process_pool_workers": 0, "_pfo_fallback_previous_control": None,
            "signal_price_enabled": not residual, "metering_price_enabled": not residual,
            "vsl_price_enabled": not residual, "offset_price_enabled": not residual,
        }, "controller")
        for name in ("_candidate_common_solver", "_anchor_follower_seed"):
            if getattr(controller, name, None) is not None:
                raise ValueError(f"controller.{name}: uncommitted solver scratch")
        excluded = {"cfg", "nash_solver", "leader"} | _CONTROLLER_SCRATCH
        if residual:
            _expect(controller, {"response_candidate_count": 1, "response_value_depth": 0,
                                 "strict_pfo_gate": False, "allow_internal_pfo_fallback": False},
                    "controller")
            _require_type(controller.potential_adapter, CoordinationPotentialAdapter, "adapter")
            excluded |= _RL_CONTROLLER_SCRATCH | {"coordination_provider", "potential_adapter"}
        fields = _view(controller, excluded)
        fields["leader"] = _view(controller.leader, {"cfg"})
        if residual:
            fields["potential_adapter"] = _view(controller.potential_adapter)
        return fields


def _continuation_identity(env, *, interval_reward: float, terminal: bool,
                           expected_contract_sha256: str, identity_version: str,
                           contract_metadata=None) -> dict:
    """Return exact component SHA256s; read-only, with no solves or forecasting.

    Call only AFTER a candidate has committed its plant interval and controller
    memory. Missing/uninitialized native controllers are intentionally unsupported.
    This identity makes no claim that every physical duplicate can be merged.
    """
    if type(interval_reward) is not float or not math.isfinite(interval_reward):
        raise ValueError("interval_reward must be a finite Python float")
    if type(terminal) is not bool:
        raise ValueError("terminal must be a Python bool")
    projection = _Projection(env, expected_contract_sha256=expected_contract_sha256)
    previous = _view(env.previous)
    _require_type(env.previous.diagnostics, dict, "previous.diagnostics")
    previous["diagnostics"] = {key: value for key, value in env.previous.diagnostics.items()
                               if key not in _DIAGNOSTIC_REPORTS}
    components = {
        "plant": env.sim.state,
        "previous_control": previous,
        "rl_controller": projection.controller(env.controller, residual=True),
        "native_controller": projection.controller(env.optimizer_controller, residual=False),
        "rl_follower": projection.follower(env.controller.nash_solver, env.cfg, residual=True),
        "native_follower": projection.follower(env.optimizer_controller.nash_solver,
                                                env.optimizer_cfg, residual=False),
        "runtime": {
            "env": _view(env, _ENV_REPORTS | {
                "cfg", "optimizer_cfg", "net", "_experiment_contract", "sim", "previous",
                "controller", "optimizer_controller", "profile", "action_schema",
                "observation_schema", "provider",
            }),
            "rl_config_sha256": projection.config(env.cfg, role=0),
            "native_config_sha256": projection.config(env.optimizer_cfg, role=1),
            "sim": _view(env.sim, {"cfg", "state", "freeway_ttt", "urban_ttt", "logs"}),
            "profile": _view(env.profile, {"cfg"}),
            "action_schema": _view(env.action_schema, {"cfg"}),
            "observation_schema": _view(env.observation_schema, {"cfg"}),
            "provider": _view(env.provider, {"action"}),
        },
        "transition": {"interval_reward": interval_reward, "terminal": terminal,
                       "reward_semantics": "interval_negative_ttt",
                       "done_semantics": "environment_terminal"},
    }
    result = {
        "version": identity_version,
        "experiment_contract_sha256": expected_contract_sha256,
        "component_sha256": {name: projection.digest(value) for name, value in components.items()},
    }
    if contract_metadata is not None:
        result["contract_scope"] = contract_metadata
    result["sha256"] = _sha(result)
    # Field digests explain a failed replay comparison without storing solver graphs.
    result["field_sha256"] = {}
    for component, value in components.items():
        fields = value if type(value) is dict else vars(value)
        result["field_sha256"][component] = {}
        for key, item in fields.items():
            name = str(key)
            if type(item) is dict:
                result["field_sha256"][component][name] = {
                    str(subkey): projection.digest(subvalue) for subkey, subvalue in item.items()
                }
            else:
                result["field_sha256"][component][name] = projection.digest(item)
    return result


def continuation_identity(env, *, interval_reward: float, terminal: bool) -> dict:
    """Original single-scenario identity; its default bytes and contract are unchanged."""
    return _continuation_identity(
        env, interval_reward=interval_reward, terminal=terminal,
        expected_contract_sha256=AUDITED_CONTRACT_SHA256, identity_version=IDENTITY_VERSION,
    )


def five_cell_continuation_identity(env, *, interval_reward: float, terminal: bool,
                                    expected_contract_sha256: str) -> dict:
    """Exact committed identity for the explicit five-cell opt-in contract."""
    metadata = validate_five_cell_contract(env, expected_contract_sha256=expected_contract_sha256)
    return _continuation_identity(
        env, interval_reward=interval_reward, terminal=terminal,
        expected_contract_sha256=expected_contract_sha256,
        identity_version=FIVE_CELL_IDENTITY_VERSION, contract_metadata=metadata,
    )
