from __future__ import annotations

from typing import Dict, List

from src.controllers.coordination import (
    CoordinationAction,
    CoordinationActionProvider,
    CoordinationPotentialAdapter,
)
from src.controllers.f1_wu_faithful_follower import F1StackelbergWuMeteredController
from src.controllers.leader import LeaderAction
from src.controllers.stackelberg_mpc import _LeaderCandidateEvaluation
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def configure_pstack_b13_follower_contract(cfg: ExperimentConfig) -> None:
    """Match the candidate and dual settings used by the b13 P-Stack follower."""
    cfg.mpc.seg13_vsl_box_kmh = 15.0
    cfg.mpc.seg13_meter_box_veh_h = 300.0
    cfg.mpc.np_primal_dual_iters = 4
    cfg.mpc.np_bias_correction = True
    cfg.mpc.baseline_move_box = True


class RLStackelbergController(F1StackelbergWuMeteredController):
    """Production P-Stack pipeline with only the coordination provider replaced."""

    def __init__(self, cfg: ExperimentConfig, provider: CoordinationActionProvider):
        configure_pstack_b13_follower_contract(cfg)
        super().__init__(cfg)
        self.coordination_provider = provider
        self.potential_adapter = CoordinationPotentialAdapter()
        self.last_coordination_action: CoordinationAction | None = None
        self.last_coordination_metadata: Dict[str, float] = {}
        self.nash_solver.f1_spillback_weight = 0.0
        self.nash_solver.segment_agents = True
        self._disable_native_price_ownership()

    def _disable_native_price_ownership(self) -> None:
        self.signal_price_enabled = False
        self.offset_price_enabled = False
        self.metering_price_enabled = False
        self.vsl_price_enabled = False
        self.green_offset_cross_price_enabled = False
        self.vsl_meter_cross_price_enabled = False
        self.offset_joint_enabled = False
        self.leader_offset_enabled = False

    def _maybe_refresh_signal_prices(self, state, forecast, previous, force=False) -> None:
        self._disable_native_price_ownership()
        self.potential_adapter.clear(self.nash_solver)
        self._signal_price_meta = {
            "wu_b2_price_refreshed": 0.0,
            "wu_b2_price_refresh_count": 0.0,
            "coordination_native_price_disabled": 1.0,
        }

    def _coordination_search(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
        fallback_incumbent_obj: float,
    ):
        requested = self.coordination_provider.provide(state, forecast, previous, self)
        bounds = self.leader._candidate_bounds(state, previous, forecast[0], forecast)
        bounded = requested.with_budget(
            min(max(float(requested.N_P_star), float(bounds.np_lower)), float(bounds.np_upper)),
            min(max(float(requested.N_UF_star), float(bounds.nuf_lower)), float(bounds.nuf_upper)),
        )
        self.last_coordination_action = bounded
        potential_meta = self.potential_adapter.apply(bounded, self.nash_solver)
        evaluation = self._evaluate_full_candidate(
            0,
            LeaderAction(bounded.N_P_star, bounded.N_UF_star),
            state,
            forecast,
            previous,
            stage="rl_coordination",
        )
        raw_n_p, raw_n_uf = requested.raw_budget or (requested.N_P_star, requested.N_UF_star)
        meta = {
            "leader_provider_rl": 1.0,
            "leader_provider_optimizer": 0.0,
            "leader_search_bypassed": 1.0,
            "leader_candidate_global_refresh": float(global_refresh),
            "leader_raw_N_P_star": float(raw_n_p),
            "leader_raw_N_UF_star": float(raw_n_uf),
            "leader_bounded_N_P_star": float(bounded.N_P_star),
            "leader_bounded_N_UF_star": float(bounded.N_UF_star),
            **potential_meta,
        }
        self.last_coordination_metadata = meta
        return [evaluation], meta, {}, {}

    def _grid_leader_search(self, state, forecast, previous, global_refresh, fallback_incumbent_obj):
        return self._coordination_search(state, forecast, previous, global_refresh, fallback_incumbent_obj)

    def _continuous_leader_search(self, state, forecast, previous, global_refresh, fallback_incumbent_obj):
        return self._coordination_search(state, forecast, previous, global_refresh, fallback_incumbent_obj)

    def decide_with_info(self, state, demand_forecast, previous_control=None, config=None):
        result = super().decide_with_info(state, demand_forecast, previous_control, config)
        result.metadata.update(self.last_coordination_metadata)
        result.control.diagnostics.update(self.last_coordination_metadata)
        return result


class OptimizerCoordinationProvider:
    """Expose a completed P-Stack decision and its native linear prices in the common schema."""

    @staticmethod
    def from_controller(controller, control: ControlAction, action_schema) -> CoordinationAction:
        follower = controller.nash_solver
        intent_n_p = float(control.diagnostics.get("leader_intent_N_P_star", control.N_P_star))
        intent_n_uf = float(control.diagnostics.get("leader_intent_N_UF_star", control.N_UF_star))
        raw = action_schema.encode(CoordinationAction(intent_n_p, intent_n_uf))
        previous = control
        decoded = action_schema.decode(raw, previous)
        urban = []
        for block in decoded.urban_blocks:
            tg, to = block.trust_radius
            urban.append(type(block)(
                owner=block.owner,
                reference=block.reference,
                trust_radius=block.trust_radius,
                linear=(
                    float((follower.signal_marginal_price or {}).get(block.owner, 0.0)) * tg,
                    float((follower.offset_marginal_price or {}).get(block.owner, 0.0)) * to,
                ),
                cholesky=(0.0, 0.0, 0.0),
                lever_keys=block.lever_keys,
            ))
        freeway = []
        for block in decoded.freeway_blocks:
            tm, tv = block.trust_radius
            vsl_key = block.lever_keys[1]
            freeway.append(type(block)(
                owner=block.owner,
                reference=block.reference,
                trust_radius=block.trust_radius,
                linear=(
                    float((follower.metering_marginal_price or {}).get(block.owner, 0.0)) * tm,
                    float((follower.vsl_marginal_price or {}).get(vsl_key, 0.0)) * tv,
                ),
                cholesky=(0.0, 0.0, 0.0),
                lever_keys=block.lever_keys,
            ))
        vsl_blocks = []
        for block in decoded.vsl_blocks:
            vsl_blocks.append(type(block)(
                owner=block.owner,
                reference=block.reference,
                trust_radius=block.trust_radius,
                linear=(
                    float((follower.vsl_marginal_price or {}).get(block.owner, 0.0))
                    * block.trust_radius
                ),
                cholesky=0.0,
                lever_key=block.lever_key,
            ))
        release_map = getattr(follower, "metering_release_certified", None)
        certificates = (
            tuple(bool(release_map.get(ramp, False)) for ramp in action_schema.ramps)
            if release_map is not None
            else None
        )
        return CoordinationAction(
            N_P_star=intent_n_p,
            N_UF_star=intent_n_uf,
            urban_blocks=tuple(urban),
            freeway_blocks=tuple(freeway),
            mask=decoded.mask,
            raw_budget=(float(control.N_P_star), float(control.N_UF_star)),
            vsl_blocks=tuple(vsl_blocks),
            metering_release_certified=certificates,
        )
