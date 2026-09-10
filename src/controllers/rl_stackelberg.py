from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Dict, List

from src.controllers.coordination import (
    CoordinationAction,
    CoordinationActionProvider,
    CoordinationPotentialAdapter,
    FollowerPotentialRuntime,
)
from src.controllers.f1_wu_faithful_follower import F1StackelbergWuMeteredController
from src.controllers.leader import LeaderAction
from src.controllers.stackelberg_mpc import _LeaderCandidateEvaluation
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def configure_pstack_b13_follower_contract(cfg: ExperimentConfig) -> None:
    """Match the candidate and dual settings used by the b13 P-Stack follower."""
    cfg.mpc.relaxed_quantized_controls = True
    cfg.mpc.seg13_vsl_box_kmh = 15.0
    cfg.mpc.seg13_meter_box_veh_h = 300.0
    cfg.mpc.np_primal_dual_iters = 4
    cfg.mpc.np_bias_correction = True
    cfg.mpc.baseline_move_box = True
    cfg.mpc.leader_rollout_box_walk = True
    cfg.mpc.leader_rollout_box_walk_vg = True
    cfg.mpc.leader_mfd_far_state_aware = True
    cfg.mpc.leader_mfd_far_real_speed = True


class RLStackelbergController(F1StackelbergWuMeteredController):
    """Production P-Stack pipeline with only the coordination provider replaced."""

    _RESPONSE_CANDIDATE_SPECS = (
        (1.0, 1.0),
        (0.5, 1.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (0.5, 2.0),
        (0.0, 2.0),
        (1.0, 4.0),
        (0.5, 4.0),
        (0.0, 4.0),
        (1.0, 0.5),
    )

    def __init__(
        self,
        cfg: ExperimentConfig,
        provider: CoordinationActionProvider,
        response_candidate_count: int = 1,
        response_value_depth: int = 0,
        strict_pfo_gate: bool = False,
        allow_internal_pfo_fallback: bool = True,
    ):
        if not 1 <= int(response_candidate_count) <= len(self._RESPONSE_CANDIDATE_SPECS):
            raise ValueError("response_candidate_count must be between 1 and 10")
        if int(response_value_depth) < 0:
            raise ValueError("response_value_depth must be nonnegative")
        configure_pstack_b13_follower_contract(cfg)
        if int(response_value_depth) > 0:
            cfg.mpc.leader_value_depth = int(response_value_depth)
            # The legacy guard reads the follower's H=3 response TTT. Extended
            # response ranking must compare the common long-horizon objective.
            cfg.mpc.stackelberg_fallback_guard_use_rollout_ttt = False
        super().__init__(cfg)
        self.coordination_provider = provider
        self.response_candidate_count = int(response_candidate_count)
        self.response_value_depth = int(response_value_depth)
        self.strict_pfo_gate = bool(strict_pfo_gate)
        self.allow_internal_pfo_fallback = bool(allow_internal_pfo_fallback)
        self.response_horizon_steps = int(cfg.mpc.horizon_steps) + self.response_value_depth
        self.potential_adapter = CoordinationPotentialAdapter()
        self.last_coordination_action: CoordinationAction | None = None
        self.last_coordination_metadata: Dict[str, float] = {}
        self.last_response_candidate_trace: list[dict[str, float]] = []
        self._response_candidate_actions: dict[int, CoordinationAction] = {}
        self._response_candidate_solvers: dict[int, object] = {}
        self._response_pfo_solver = None
        self.nash_solver.f1_spillback_weight = 0.0
        self.nash_solver.segment_agents = True
        self.nash_solver.ramp_offset_enabled = True
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
        bounded = (
            requested
            if requested.native_budget_exact
            else requested.with_budget(
                min(
                    max(float(requested.N_P_star), float(bounds.np_lower)),
                    float(bounds.np_upper),
                ),
                min(
                    max(float(requested.N_UF_star), float(bounds.nuf_lower)),
                    float(bounds.nuf_upper),
                ),
            )
        )
        if self.response_candidate_count == 1:
            return self._single_coordination_search(
                requested, bounded, state, forecast, previous, global_refresh,
            )

        return self._response_coordination_search(
            requested, bounded, state, forecast, previous, global_refresh, bounds,
        )

    def _single_coordination_search(
        self,
        requested: CoordinationAction,
        bounded: CoordinationAction,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
    ):
        self.last_coordination_action = bounded
        self.last_response_candidate_trace = []
        self._response_candidate_actions = {}
        self._response_candidate_solvers = {}
        self._response_pfo_solver = None
        live_solver = self.nash_solver
        common_solver = getattr(self, "_candidate_common_solver", None)
        candidate_solver = self._clone_follower_solver(
            live_solver if common_solver is None else common_solver
        )
        self._anchor_follower_seed = None
        self.nash_solver = candidate_solver
        try:
            potential_meta = self.potential_adapter.apply(
                bounded,
                candidate_solver,
            )
            solver_n_p = float(bounded.N_P_star)
            solver_n_uf = float(bounded.N_UF_star)
            if requested.native_budget_exact and requested.raw_budget is not None:
                solver_n_p, solver_n_uf = map(float, requested.raw_budget)
            evaluation = self._evaluate_full_candidate(
                0,
                LeaderAction(solver_n_p, solver_n_uf),
                state,
                forecast,
                previous,
                stage="rl_coordination",
            )
            evaluation.follower_solver_snapshot = self._clone_follower_solver(
                self.nash_solver
            )
        finally:
            self.nash_solver = live_solver
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
            "leader_solver_input_N_P_star": float(solver_n_p),
            "leader_solver_input_N_UF_star": float(solver_n_uf),
            **potential_meta,
        }
        self.last_coordination_metadata = meta
        return [evaluation], meta, {}, {}

    @staticmethod
    def _scaled_block(block, price_scale: float):
        root_scale = math.sqrt(max(float(price_scale), 0.0))
        return replace(
            block,
            linear=tuple(float(price_scale) * float(value) for value in block.linear),
            cholesky=tuple(root_scale * float(value) for value in block.cholesky),
            physical_linear=(
                tuple(
                    float(price_scale) * float(value)
                    for value in block.physical_linear
                )
                if block.physical_linear is not None else None
            ),
            physical_hessian=(
                tuple(
                    float(price_scale) * float(value)
                    for value in block.physical_hessian
                )
                if block.physical_hessian is not None else None
            ),
        )

    @staticmethod
    def _scaled_scalar_block(block, price_scale: float):
        root_scale = math.sqrt(max(float(price_scale), 0.0))
        return replace(
            block,
            linear=float(price_scale) * float(block.linear),
            cholesky=root_scale * float(block.cholesky),
            physical_linear=(
                float(price_scale) * float(block.physical_linear)
                if block.physical_linear is not None else None
            ),
            physical_curvature=(
                float(price_scale) * float(block.physical_curvature)
                if block.physical_curvature is not None else None
            ),
        )

    def _response_candidates(
        self,
        base: CoordinationAction,
        bounds,
    ) -> list[tuple[CoordinationAction, float, float]]:
        center = getattr(self, "_pfo_incumbent_center", None)
        anchor_np = float(center.N_P_star) if center is not None else float(base.N_P_star)
        anchor_nuf = float(center.N_UF_star) if center is not None else float(base.N_UF_star)
        candidates = []
        for budget_scale, price_scale in self._RESPONSE_CANDIDATE_SPECS:
            n_p = anchor_np + budget_scale * (float(base.N_P_star) - anchor_np)
            n_uf = anchor_nuf + budget_scale * (float(base.N_UF_star) - anchor_nuf)
            candidate = replace(
                base,
                N_P_star=min(max(n_p, float(bounds.np_lower)), float(bounds.np_upper)),
                N_UF_star=min(max(n_uf, float(bounds.nuf_lower)), float(bounds.nuf_upper)),
                raw_budget=(float(n_p), float(n_uf)),
                urban_blocks=tuple(
                    self._scaled_block(block, price_scale) for block in base.urban_blocks
                ),
                freeway_blocks=tuple(
                    self._scaled_block(block, price_scale) for block in base.freeway_blocks
                ),
                vsl_blocks=tuple(
                    self._scaled_scalar_block(block, price_scale) for block in base.vsl_blocks
                ),
            )
            if not any(candidate == existing[0] for existing in candidates):
                candidates.append((candidate, float(budget_scale), float(price_scale)))
            if len(candidates) >= self.response_candidate_count:
                break
        return candidates

    def _response_coordination_search(
        self,
        requested: CoordinationAction,
        bounded: CoordinationAction,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
        bounds,
    ):
        candidates = self._response_candidates(bounded, bounds)
        if not candidates:
            raise RuntimeError("RL response candidate generation produced no candidates")

        pfo_solver = self.nash_solver
        pfo_snapshot = copy.deepcopy(pfo_solver)
        evaluations = []
        candidate_actions: dict[int, CoordinationAction] = {}
        candidate_solvers: dict[int, object] = {}
        candidate_trace = []
        try:
            for index, (candidate, budget_scale, price_scale) in enumerate(candidates):
                candidate_solver = copy.deepcopy(pfo_snapshot)
                self.nash_solver = candidate_solver
                potential_meta = self.potential_adapter.apply(candidate, candidate_solver)
                evaluation = self._evaluate_full_candidate(
                    index,
                    LeaderAction(candidate.N_P_star, candidate.N_UF_star),
                    state,
                    forecast,
                    previous,
                    stage="rl_response",
                )
                evaluation.metadata.update({
                    "leader_rl_response_candidate_index": float(index),
                    "leader_rl_response_budget_scale": float(budget_scale),
                    "leader_rl_response_price_scale": float(price_scale),
                    **potential_meta,
                })
                rollout_ttt = self._evaluation_rollout_ttt(evaluation)
                response_score = self._response_score_value(evaluation)
                evaluations.append(evaluation)
                candidate_actions[index] = candidate
                candidate_solvers[index] = candidate_solver
                candidate_trace.append({
                    "index": float(index),
                    "budget_scale": float(budget_scale),
                    "price_scale": float(price_scale),
                    "requested_N_P_star": float(candidate.N_P_star),
                    "requested_N_UF_star": float(candidate.N_UF_star),
                    "realized_N_P_star": float(evaluation.action.N_P_star),
                    "realized_N_UF_star": float(evaluation.action.N_UF_star),
                    "objective": float(evaluation.objective),
                    "rollout_ttt": float(rollout_ttt) if rollout_ttt is not None else 0.0,
                    "long_rollout_ttt": float(evaluation.objective_terms.get(
                        "leader_follower_ttt_base", evaluation.objective,
                    )),
                    "objective_penalty": float(
                        evaluation.objective
                        - evaluation.objective_terms.get(
                            "leader_follower_ttt_base", evaluation.objective,
                        )
                    ),
                    "target_penalty": float(evaluation.objective_terms.get(
                        "leader_target_penalty", 0.0,
                    )),
                    "mfd_storage_penalty": float(evaluation.objective_terms.get(
                        "leader_mfd_storage_penalty", 0.0,
                    )),
                    "boundary_queue_penalty": float(evaluation.objective_terms.get(
                        "leader_boundary_in_queue_penalty", 0.0,
                    )),
                    "density_penalty": float(evaluation.objective_terms.get(
                        "leader_density_penalty", 0.0,
                    )),
                    "ramp_queue_penalty": float(evaluation.objective_terms.get(
                        "leader_ramp_queue_penalty", 0.0,
                    )),
                    "response_score": float(response_score),
                    "response_value_depth": float(self.response_value_depth),
                    "response_horizon_steps": float(self.response_horizon_steps),
                    "terminal_proxy_veh": float(self._evaluation_terminal_proxy(evaluation)),
                    "completed_proxy_veh": float(self._evaluation_completed_proxy(evaluation)),
                    "selected": 0.0,
                })
        finally:
            self.nash_solver = pfo_solver

        self._response_pfo_solver = pfo_solver
        self._response_candidate_actions = candidate_actions
        self._response_candidate_solvers = candidate_solvers
        self.last_response_candidate_trace = candidate_trace
        self.last_coordination_action = bounded
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
            "leader_rl_response_candidate_mode": 1.0,
            "leader_rl_response_candidate_requested_count": float(self.response_candidate_count),
            "leader_rl_response_candidate_evaluated_count": float(len(evaluations)),
            "leader_rl_response_value_depth": float(self.response_value_depth),
            "leader_rl_response_horizon_steps": float(self.response_horizon_steps),
            "leader_rl_response_score_metric_long_objective": float(
                self.response_value_depth > 0
            ),
        }
        self.last_coordination_metadata = meta
        return evaluations, meta, {}, {}

    def _response_score_value(self, evaluation) -> float:
        if self.response_value_depth > 0:
            return float(evaluation.objective)
        rollout_ttt = self._evaluation_rollout_ttt(evaluation)
        return float(rollout_ttt) if rollout_ttt is not None else float(evaluation.objective)

    def _fallback_guard_rejects(self, leader_best, fallback_best):
        reject, metadata = super()._fallback_guard_rejects(leader_best, fallback_best)
        if not self.strict_pfo_gate:
            metadata["leader_rl_strict_pfo_gate_active"] = 0.0
            return reject, metadata

        leader_score = self._response_score_value(leader_best)
        fallback_score = self._response_score_value(fallback_best)
        gain = float(fallback_score) - float(leader_score)
        required_gain = max(0.1, 1.0e-3 * max(abs(float(fallback_score)), 1.0))
        insufficient_gain = gain <= required_gain
        metadata.update({
            "leader_rl_strict_pfo_gate_active": 1.0,
            "leader_rl_strict_pfo_gate_score_gain": float(gain),
            "leader_rl_strict_pfo_gate_required_gain": float(required_gain),
            "leader_rl_strict_pfo_gate_insufficient_gain": float(insufficient_gain),
        })
        return bool(reject or insufficient_gain), metadata

    def _select_with_fallback_guard(self, leader_evaluations, fallback_evaluations):
        selectable_fallbacks = (
            fallback_evaluations if self.allow_internal_pfo_fallback else []
        )
        if self.response_candidate_count == 1 or not self._response_candidate_solvers:
            return self._select_parent_with_configured_fallback(
                leader_evaluations,
                selectable_fallbacks,
            )

        def response_score(evaluation):
            return (
                self._response_score_value(evaluation),
                float(evaluation.objective),
                int(evaluation.index),
            )

        response_best = min(leader_evaluations, key=response_score)
        selected_solver = self._response_candidate_solvers[int(response_best.index)]
        self.nash_solver = selected_solver
        self._rl_response_rank_by_ttt = self.response_value_depth == 0
        best, metadata = self._select_parent_with_configured_fallback(
            [response_best], selectable_fallbacks,
        )

        pfo_eval = next(
            (item for item in selectable_fallbacks if item.stage == "fallback_pfo"),
            None,
        )
        candidate_ttt = self._evaluation_rollout_ttt(response_best)
        pfo_ttt = self._evaluation_rollout_ttt(pfo_eval) if pfo_eval is not None else None
        candidate_score = self._response_score_value(response_best)
        pfo_score = self._response_score_value(pfo_eval) if pfo_eval is not None else None
        candidate_long_base = float(response_best.objective_terms.get(
            "leader_follower_ttt_base", response_best.objective,
        ))
        pfo_long_base = (
            float(pfo_eval.objective_terms.get("leader_follower_ttt_base", pfo_eval.objective))
            if pfo_eval is not None else 0.0
        )
        required_gain = (
            max(0.1, 1.0e-3 * max(abs(float(pfo_score)), 1.0))
            if pfo_score is not None else 0.0
        )
        response_score_gain = (
            float(pfo_score) - float(candidate_score)
            if pfo_score is not None else 0.0
        )
        response_ttt_gain = (
            float(pfo_ttt) - float(candidate_ttt)
            if candidate_ttt is not None and pfo_ttt is not None else 0.0
        )
        strict_accept = pfo_eval is None or (
            pfo_score is not None and response_score_gain > required_gain
        )
        if best.stage != "fallback_pfo" and not strict_accept and pfo_eval is not None:
            best = pfo_eval
            metadata.update({
                "leader_fallback_guard_selected": 1.0,
                "leader_fallback_guard_selected_pfo": 1.0,
                "leader_fallback_guard_rejected_leader": 1.0,
                "leader_pfo_incumbent_selected": 1.0,
            })

        selected_index = (
            int(response_best.index)
            if best.stage == "rl_response" and int(response_best.index) in self._response_candidate_solvers
            else -1
        )
        if selected_index >= 0:
            self.nash_solver = self._response_candidate_solvers[selected_index]
            self.last_coordination_action = self._response_candidate_actions[selected_index]
        else:
            self.nash_solver = (
                best.follower_solver_snapshot or self._response_pfo_solver
            )
        for row in self.last_response_candidate_trace:
            row["selected"] = float(int(row["index"]) == selected_index)
        metadata.update({
            "leader_rl_response_selected_index": float(selected_index),
            "leader_rl_response_best_index": float(response_best.index),
            "leader_rl_response_best_rollout_ttt": float(candidate_ttt or 0.0),
            "leader_rl_response_pfo_rollout_ttt": float(pfo_ttt or 0.0),
            "leader_rl_response_gain_ttt": float(response_ttt_gain),
            "leader_rl_response_best_score": float(candidate_score),
            "leader_rl_response_pfo_score": float(pfo_score or 0.0),
            "leader_rl_response_gain_score": float(response_score_gain),
            "leader_rl_response_required_gain_score": float(required_gain),
            "leader_rl_response_best_long_base": float(candidate_long_base),
            "leader_rl_response_pfo_long_base": float(pfo_long_base),
            "leader_rl_response_gain_long_base": float(pfo_long_base - candidate_long_base),
            "leader_rl_response_value_depth": float(self.response_value_depth),
            "leader_rl_response_horizon_steps": float(self.response_horizon_steps),
            "leader_rl_response_score_metric_long_objective": float(
                self.response_value_depth > 0
            ),
            "leader_rl_response_candidate_accepted": float(selected_index >= 0),
            "leader_rl_response_pfo_selected": float(selected_index < 0),
        })
        self._response_candidate_solvers = (
            {selected_index: self.nash_solver} if selected_index >= 0 else {}
        )
        return best, metadata

    def _select_parent_with_configured_fallback(
        self,
        leader_evaluations,
        fallback_evaluations,
    ):
        if self.allow_internal_pfo_fallback:
            return super()._select_with_fallback_guard(
                leader_evaluations,
                fallback_evaluations,
            )
        saved_incumbent = getattr(self, "_pfo_incumbent_eval", None)
        self._pfo_incumbent_eval = None
        try:
            best, metadata = super()._select_with_fallback_guard(
                leader_evaluations,
                [],
            )
        finally:
            self._pfo_incumbent_eval = saved_incumbent
        metadata.update({
            "leader_rl_internal_pfo_fallback_enabled": 0.0,
            "leader_pfo_incumbent_selected": 0.0,
        })
        return best, metadata

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
    def from_controller(
        controller,
        control: ControlAction,
        action_schema,
        *,
        selected_branch: str = "unspecified",
    ) -> CoordinationAction:
        follower = controller.nash_solver

        def price_map(name: str):
            return getattr(follower, name, None) or {}

        def trust_value(name: str, fallback: float) -> float:
            value = getattr(follower, name, None)
            return float(fallback if value is None else value)

        def normalized_cholesky(
            physical: tuple[float, float, float] | None,
            trust: tuple[float, float],
        ) -> tuple[float, float, float]:
            if physical is None:
                return (0.0, 0.0, 0.0)
            t0, t1 = trust
            a = max(float(physical[0]) * t0 * t0, 0.0)
            b = float(physical[1]) * t0 * t1
            c = max(float(physical[2]) * t1 * t1, 0.0)
            l11 = math.sqrt(a)
            l21 = b / l11 if l11 > 1.0e-15 else 0.0
            l22 = math.sqrt(max(c - l21 * l21, 0.0))
            return (float(l11), float(l21), float(l22))

        signal_linear = price_map("signal_marginal_price")
        offset_linear = price_map("offset_marginal_price")
        meter_linear = price_map("metering_marginal_price")
        vsl_linear = price_map("vsl_marginal_price")
        signal_ref = price_map("signal_marginal_price_ref")
        offset_ref = price_map("offset_marginal_price_ref")
        meter_ref = price_map("metering_marginal_price_ref")
        vsl_ref = price_map("vsl_marginal_price_ref")
        signal_quad_raw = getattr(follower, "signal_quadratic_price", None)
        offset_quad_raw = getattr(follower, "offset_quadratic_price", None)
        meter_quad_raw = getattr(follower, "metering_quadratic_price", None)
        vsl_quad_raw = getattr(follower, "vsl_quadratic_price", None)
        go_cross_raw = getattr(follower, "green_offset_cross_price", None)
        vm_cross_raw = getattr(follower, "vsl_meter_cross_price", None)
        signal_quad = signal_quad_raw or {}
        offset_quad = offset_quad_raw or {}
        meter_quad = meter_quad_raw or {}
        vsl_quad = vsl_quad_raw or {}
        go_cross = go_cross_raw or {}
        vm_cross = vm_cross_raw or {}
        intent_n_p = float(control.diagnostics.get("leader_intent_N_P_star", control.N_P_star))
        intent_n_uf = float(control.diagnostics.get("leader_intent_N_UF_star", control.N_UF_star))
        raw = action_schema.encode(CoordinationAction(intent_n_p, intent_n_uf))
        previous = control
        decoded = action_schema.decode(raw, previous)
        urban = []
        for block in decoded.urban_blocks:
            tg = trust_value("signal_marginal_price_trust_sec", block.trust_radius[0])
            to = trust_value("offset_marginal_price_trust_sec", block.trust_radius[1])
            physical_linear = (
                float(signal_linear.get(block.owner, 0.0)),
                float(offset_linear.get(block.owner, 0.0)),
            )
            physical_hessian = (
                (
                    float(signal_quad.get(block.owner, 0.0)),
                    float(go_cross.get(block.owner, 0.0)),
                    float(offset_quad.get(block.owner, 0.0)),
                )
                if any(value is not None for value in (
                    signal_quad_raw, go_cross_raw, offset_quad_raw
                ))
                else None
            )
            urban.append(type(block)(
                owner=block.owner,
                reference=(
                    float(signal_ref.get(block.owner, block.reference[0])),
                    float(offset_ref.get(block.owner, block.reference[1])),
                ),
                trust_radius=(tg, to),
                linear=(
                    physical_linear[0] * tg,
                    physical_linear[1] * to,
                ),
                cholesky=normalized_cholesky(physical_hessian, (tg, to)),
                lever_keys=block.lever_keys,
                physical_linear=physical_linear,
                physical_hessian=physical_hessian,
            ))
        freeway = []
        for block in decoded.freeway_blocks:
            trust_fraction = getattr(
                follower, "metering_marginal_price_trust_frac", None
            )
            tm = (
                float(trust_fraction) * float(controller.cfg.network.ramp_capacity_veh_h[block.owner])
                if trust_fraction is not None else float(block.trust_radius[0])
            )
            tv = trust_value("vsl_marginal_price_trust_kmh", block.trust_radius[1])
            vsl_key = block.lever_keys[1]
            physical_linear = (
                float(meter_linear.get(block.owner, 0.0)),
                float(vsl_linear.get(vsl_key, 0.0)),
            )
            physical_hessian = (
                (
                    float(meter_quad.get(block.owner, 0.0)),
                    float(vm_cross.get(block.owner, 0.0)),
                    float(vsl_quad.get(vsl_key, 0.0)),
                )
                if any(value is not None for value in (
                    meter_quad_raw, vm_cross_raw, vsl_quad_raw
                ))
                else None
            )
            freeway.append(type(block)(
                owner=block.owner,
                reference=(
                    float(meter_ref.get(block.owner, block.reference[0])),
                    float(vsl_ref.get(vsl_key, block.reference[1])),
                ),
                trust_radius=(tm, tv),
                linear=(
                    physical_linear[0] * tm,
                    physical_linear[1] * tv,
                ),
                cholesky=normalized_cholesky(physical_hessian, (tm, tv)),
                lever_keys=block.lever_keys,
                physical_linear=physical_linear,
                physical_hessian=physical_hessian,
            ))
        vsl_blocks = []
        for block in decoded.vsl_blocks:
            trust = trust_value(
                "vsl_marginal_price_trust_kmh", block.trust_radius
            )
            physical_linear = float(vsl_linear.get(block.owner, 0.0))
            physical_curvature = (
                float(vsl_quad.get(block.owner, 0.0))
                if vsl_quad_raw is not None else None
            )
            vsl_blocks.append(type(block)(
                owner=block.owner,
                reference=float(vsl_ref.get(block.owner, block.reference)),
                trust_radius=trust,
                linear=physical_linear * trust,
                cholesky=(
                    math.sqrt(max(physical_curvature * trust * trust, 0.0))
                    if physical_curvature is not None else 0.0
                ),
                lever_key=block.lever_key,
                physical_linear=physical_linear,
                physical_curvature=physical_curvature,
            ))
        release_map = getattr(follower, "metering_release_certified", None)
        certificates = (
            tuple(bool(release_map.get(ramp, False)) for ramp in action_schema.ramps)
            if release_map is not None
            else None
        )
        return CoordinationAction(
            N_P_star=float(control.N_P_star),
            N_UF_star=float(control.N_UF_star),
            urban_blocks=tuple(urban),
            freeway_blocks=tuple(freeway),
            mask=decoded.mask,
            raw_budget=(intent_n_p, intent_n_uf),
            vsl_blocks=tuple(vsl_blocks),
            metering_release_certified=certificates,
            selected_branch=str(selected_branch),
            follower_runtime=FollowerPotentialRuntime(
                ramp_offset_enabled=bool(
                    getattr(follower, "ramp_offset_enabled", True)
                ),
                priced_vsl_segment_candidates_enabled=bool(
                    getattr(follower, "priced_vsl_segment_candidates_enabled", True)
                ),
                joint_green_offset_enabled=bool(
                    getattr(follower, "joint_green_offset_enabled", True)
                ),
                metering_trust_fraction=(
                    float(trust_fraction) if trust_fraction is not None else None
                ),
                vsl_trust_kmh=(
                    float(getattr(follower, "vsl_marginal_price_trust_kmh"))
                    if getattr(follower, "vsl_marginal_price_trust_kmh", None)
                    is not None else None
                ),
            ),
            native_budget_exact=True,
        )
