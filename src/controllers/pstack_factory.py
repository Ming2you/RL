from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from src.controllers.f1_wu_faithful_follower import F1StackelbergWuMeteredController
from src.models.state import ExperimentConfig


@dataclass(frozen=True)
class PStackControllerOptions:
    """Explicit behavior contract for the ALLPRICE-JOINT controller factory."""

    leader_value_depth: int = 3
    skip_local_refinement: bool = True
    rollout_early_stop: bool = False
    grad_seed: bool = False
    bias_sample: bool = False
    bias_sample_pow: float = 0.4
    signal_price_enabled: bool = True
    metering_price_enabled: bool = True
    vsl_price_enabled: bool = True
    green_offset_cross_price_enabled: bool = False
    vsl_meter_cross_price_enabled: bool = False
    joint_green_offset_enabled: bool = True
    signal_price_trust_sec: float | None = None
    offset_price_enabled: bool = True
    offset_price_inner_iters: int = 4
    ramp_offset_enabled: bool = True
    metering_price_delta_veh_h: float = 300.0
    metering_price_trust_frac: float = 0.20
    segment_agents: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PStackControllerOptions":
        return cls(**dict(raw))

    @classmethod
    def from_legacy_environment(
        cls,
        environment: Mapping[str, str],
        cfg: ExperimentConfig,
    ) -> "PStackControllerOptions":
        """Translate the legacy environment surface at its compatibility edge."""
        depth = int(environment.get("LEADER_V_DEPTH", cfg.mpc.leader_value_depth or 3))
        has_external_meter_delta = "METER_PRICE_DELTA" in environment
        return cls(
            leader_value_depth=depth,
            skip_local_refinement=environment.get("OPT12") != "0",
            rollout_early_stop=environment.get("OPT12") != "0",
            grad_seed=environment.get("GRAD_SEED") == "1",
            bias_sample=environment.get("BIAS_SAMPLE") == "1",
            bias_sample_pow=float(environment.get("BIAS_POW", "0.4")),
            signal_price_enabled=environment.get("GREEN_PRICE") != "0",
            signal_price_trust_sec=(
                float(environment["GREEN_TRUST_SEC"])
                if environment.get("GREEN_TRUST_SEC") else None
            ),
            offset_price_enabled=environment.get("OFFSET_PRICE") != "0",
            offset_price_inner_iters=int(environment.get("OFFSET_INNER_ITER", "4")),
            ramp_offset_enabled=environment.get("RAMP_OFFSET") != "0",
            # Preserve the historical factory edge: an externally managed
            # delta leaves the controller's constructor defaults untouched.
            metering_price_delta_veh_h=(60.0 if has_external_meter_delta else 300.0),
            metering_price_trust_frac=(0.25 if has_external_meter_delta else 0.20),
            # The historical standalone runner applies this separately.
            segment_agents=False,
        )


CANONICAL_PSTACK_OPTIONS = PStackControllerOptions()


def configure_pstack_allprice_joint(
    cfg: ExperimentConfig,
    environment: Mapping[str, str] | None = None,
    *,
    options: PStackControllerOptions | None = None,
) -> PStackControllerOptions:
    """Apply the deterministic config portion of the ALLPRICE-JOINT factory."""
    if options is not None and environment is not None:
        raise ValueError("pass PStack options or a legacy environment, not both")
    if options is None:
        env = os.environ if environment is None else environment
        options = PStackControllerOptions.from_legacy_environment(env, cfg)
    cfg.mpc.relaxed_quantized_controls = True
    cfg.mpc.grid_parallel_backend = "serial"
    cfg.mpc.leader_search_mode = "grid"
    cfg.mpc.stackelberg_leader_parallel_backend = "serial"
    if int(cfg.mpc.leader_value_depth) == 0:
        cfg.mpc.leader_value_depth = int(options.leader_value_depth)
    cfg.mpc.leader_skip_local_refinement = bool(options.skip_local_refinement)
    cfg.mpc.leader_rollout_early_stop = bool(options.rollout_early_stop)
    return options


def make_pstack_allprice_joint_controller(
    cfg: ExperimentConfig,
    environment: Mapping[str, str] | None = None,
    *,
    options: PStackControllerOptions | None = None,
) -> F1StackelbergWuMeteredController:
    """Build the production ALLPRICE-JOINT P-Stack controller."""
    options = configure_pstack_allprice_joint(
        cfg,
        environment,
        options=options,
    )

    controller = F1StackelbergWuMeteredController(cfg)
    if options.grad_seed:
        from src.controllers.gradseed_mpc import enable_gradseed

        enable_gradseed(controller)
    if options.bias_sample:
        from src.controllers.biasedsample_mpc import enable_biased_sampling

        cfg.mpc.leader_bias_sample_pow = float(options.bias_sample_pow)
        enable_biased_sampling(controller)

    controller.nash_solver.f1_spillback_weight = 0.0
    controller.signal_price_enabled = bool(options.signal_price_enabled)
    controller.metering_price_enabled = bool(options.metering_price_enabled)
    controller.vsl_price_enabled = bool(options.vsl_price_enabled)
    controller.green_offset_cross_price_enabled = bool(
        options.green_offset_cross_price_enabled
    )
    controller.vsl_meter_cross_price_enabled = bool(
        options.vsl_meter_cross_price_enabled
    )
    controller.nash_solver.joint_green_offset_enabled = bool(
        options.joint_green_offset_enabled
    )
    if options.signal_price_trust_sec is not None:
        controller.signal_price_trust_sec = float(options.signal_price_trust_sec)
    controller.offset_price_enabled = bool(options.offset_price_enabled)
    controller.offset_price_inner_iters = int(options.offset_price_inner_iters)
    controller.nash_solver.ramp_offset_enabled = bool(options.ramp_offset_enabled)
    controller.metering_price_delta_veh_h = float(options.metering_price_delta_veh_h)
    controller.metering_price_trust_frac = float(options.metering_price_trust_frac)
    controller.nash_solver.segment_agents = bool(options.segment_agents)
    return controller
