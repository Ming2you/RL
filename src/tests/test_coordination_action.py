import unittest
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from rl_leader.collect_full_action import (
    action_block_layout,
    loose_anchor_action,
    optimizer_replay_anchor_step,
    priority_block_indices,
    structured_action,
    temporally_correlated_action,
)
from rl_leader.env import RLLeaderEnv, make_cfg, make_targeted_scenario
from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    CoordinationActionSchema,
    CoordinationMask,
    CoordinationPotentialAdapter,
    StaticCoordinationProvider,
)
from src.controllers.f1_wu_faithful_follower import F1WuFaithfulFollower
from src.controllers.leader import LeaderAction
from src.controllers.nash_solver import NashResult
from src.controllers.rl_stackelberg import OptimizerCoordinationProvider, RLStackelbergController
from src.controllers.stackelberg_mpc import _LeaderCandidateEvaluation
from src.models.state import ControlAction


class CoordinationActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = make_cfg("medium_demand")
        cls.schema = CoordinationActionSchema(cls.cfg)
        cls.previous = ControlAction.fixed(cls.cfg)

    def test_full_schema_has_fixed_all_channel_shape(self):
        all_vsl = len(self.cfg.network.freeway_links) * self.cfg.network.freeway_segments_per_link
        merge_vsl = len(self.cfg.network.ramps)
        expected = (
            2
            + 5 * len(self.cfg.network.signals)
            + 5 * len(self.cfg.network.ramps)
            + 2 * (all_vsl - merge_vsl)
            + len(self.cfg.network.ramps)
        )
        self.assertEqual(self.schema.dimension, expected)
        self.assertEqual(self.schema.dimension, len(self.schema.names))
        self.assertEqual(self.schema.metadata()["version"], ACTION_SCHEMA_VERSION)
        for name in ("RL-FULL", "RL-LINEAR", "RL-BUDGET", "RL-URBAN", "RL-FREEWAY", "RL-NO-CROSS"):
            action = self.schema.decode(
                np.zeros(self.schema.dimension), self.previous, CoordinationMask.named(name)
            )
            self.assertEqual(self.schema.encode(action).shape, (self.schema.dimension,))

    def test_full_schema_represents_every_vsl_segment_price(self):
        expected = {
            f"{link}__seg{index}"
            for link in self.cfg.network.freeway_links
            for index in range(self.cfg.network.freeway_segments_per_link)
        }

        self.assertEqual(set(self.schema.represented_vsl_keys), expected)
        action = self.schema.decode(
            np.full(self.schema.dimension, 0.5),
            self.previous,
            CoordinationMask.named("RL-FULL"),
        )
        follower = F1WuFaithfulFollower(self.cfg)
        CoordinationPotentialAdapter().apply(action, follower)
        self.assertEqual(set(follower.vsl_marginal_price), expected)
        self.assertEqual(set(follower.vsl_quadratic_price), expected)

    def test_optimizer_encoding_preserves_deployed_budget_intent_and_vsl_prices(self):
        control = self.previous.copy()
        control.N_P_star = 500.0
        control.N_UF_star = 5900.0
        control.diagnostics.update({
            "leader_intent_N_P_star": 450.0,
            "leader_intent_N_UF_star": 5700.0,
        })
        native_vsl = {key: 0.2 for key in self.schema.represented_vsl_keys}
        follower = SimpleNamespace(
            signal_marginal_price={},
            offset_marginal_price={},
            metering_marginal_price={},
            vsl_marginal_price=native_vsl,
            metering_release_certified={ramp: True for ramp in self.schema.ramps},
        )
        controller = SimpleNamespace(nash_solver=follower)

        coordination = OptimizerCoordinationProvider.from_controller(
            controller, control, self.schema,
        )
        encoded = self.schema.encode(coordination)
        replay = self.schema.decode(encoded, self.previous, CoordinationMask.named("RL-LINEAR"))
        replay_follower = F1WuFaithfulFollower(self.cfg)
        CoordinationPotentialAdapter().apply(replay, replay_follower)

        self.assertAlmostEqual(coordination.N_P_star, 500.0)
        self.assertAlmostEqual(coordination.N_UF_star, 5900.0)
        self.assertEqual(coordination.raw_budget, (450.0, 5700.0))
        self.assertTrue(coordination.native_budget_exact)
        self.assertLess(
            float(np.abs(encoded[2:-len(self.schema.certificate_ramps)]).max()), 1.0,
        )
        self.assertTrue(all(coordination.metering_release_certified))
        self.assertTrue(all(replay_follower.metering_release_certified.values()))
        for key, price in native_vsl.items():
            self.assertAlmostEqual(replay_follower.vsl_marginal_price[key], price)

    def test_anchored_residual_zero_preserves_native_coordination_exactly(self):
        raw = np.full(self.schema.dimension, 0.25, dtype=np.float32)
        anchor = self.schema.decode(
            raw, self.previous, CoordinationMask.named("RL-FULL")
        )
        anchor = replace(
            anchor,
            urban_blocks=(
                replace(
                    anchor.urban_blocks[0],
                    reference=(47.0, 75.0),
                    trust_radius=(3.0, 11.25),
                    linear=(17.0, -13.0),
                ),
                *anchor.urban_blocks[1:],
            ),
            selected_branch="refined",
        )

        replay = self.schema.decode_anchored_residual(
            np.zeros(self.schema.dimension, dtype=np.float32),
            anchor,
            CoordinationMask.named("RL-FULL"),
        )

        self.assertEqual(replay, anchor)
        self.assertEqual(replay.urban_blocks[0].linear, (17.0, -13.0))

    def test_anchored_residual_changes_only_native_coordinates(self):
        raw = np.full(self.schema.dimension, 0.1, dtype=np.float32)
        anchor = replace(
            self.schema.decode(raw, self.previous, CoordinationMask.named("RL-FULL")),
            selected_branch="fallback_pfo",
        )
        residual = np.zeros(self.schema.dimension, dtype=np.float32)
        residual[0] = 0.25
        residual[2] = -0.4

        replay = self.schema.decode_anchored_residual(
            residual, anchor, CoordinationMask.named("RL-FULL")
        )

        expected_np = np.clip(
            anchor.N_P_star
            + 0.25 * 0.5 * (
                self.cfg.leader.N_P_star_range[1]
                - self.cfg.leader.N_P_star_range[0]
            ),
            *self.cfg.leader.N_P_star_range,
        )
        self.assertAlmostEqual(replay.N_P_star, expected_np)
        self.assertAlmostEqual(
            replay.urban_blocks[0].linear[0],
            anchor.urban_blocks[0].linear[0] - 0.4 * self.schema.linear_scale,
            places=6,
        )
        self.assertEqual(
            replay.urban_blocks[0].reference,
            anchor.urban_blocks[0].reference,
        )
        self.assertEqual(
            replay.urban_blocks[0].trust_radius,
            anchor.urban_blocks[0].trust_radius,
        )
        self.assertEqual(
            replay.metering_release_certified,
            anchor.metering_release_certified,
        )
        self.assertEqual(replay.selected_branch, "fallback_pfo")

    def test_native_anchor_envelope_round_trip_and_fingerprint(self):
        raw = np.linspace(
            -0.8, 0.8, self.schema.dimension, dtype=np.float32
        )
        anchor = replace(
            self.schema.decode(raw, self.previous, CoordinationMask.named("RL-FULL")),
            urban_blocks=(
                replace(
                    self.schema.decode(
                        raw, self.previous, CoordinationMask.named("RL-FULL")
                    ).urban_blocks[0],
                    reference=(43.125, 67.5),
                    trust_radius=(2.75, 9.25),
                    linear=(17.5, -12.25),
                ),
                *self.schema.decode(
                    raw, self.previous, CoordinationMask.named("RL-FULL")
                ).urban_blocks[1:],
            ),
            raw_budget=(321.25, 5432.75),
            selected_branch="fallback_pfo",
        )

        envelope = self.schema.serialize_anchor(anchor)
        restored = self.schema.deserialize_anchor(
            envelope, selected_branch=anchor.selected_branch
        )
        fingerprint = self.schema.anchor_fingerprint(
            envelope, anchor.selected_branch
        )
        changed = envelope.copy()
        changed[0] += 1.0

        self.assertEqual(envelope.dtype, np.float64)
        self.assertEqual(
            envelope.shape,
            (self.schema.anchor_envelope_metadata()["dimension"],),
        )
        self.assertEqual(restored, anchor)
        self.assertEqual(len(fingerprint), 64)
        self.assertNotEqual(
            fingerprint,
            self.schema.anchor_fingerprint(changed, anchor.selected_branch),
        )

    def test_psd_parameterization_is_positive_semidefinite(self):
        rng = np.random.default_rng(7)
        action = self.schema.decode(rng.uniform(-1.0, 1.0, self.schema.dimension), self.previous)
        for block in (*action.urban_blocks, *action.freeway_blocks):
            self.assertGreaterEqual(float(np.linalg.eigvalsh(block.hessian()).min()), -1.0e-10)

    def test_zero_mask_is_exactly_unpriced(self):
        action = self.schema.decode(
            np.zeros(self.schema.dimension), self.previous, CoordinationMask.named("ZERO")
        )
        follower = F1WuFaithfulFollower(self.cfg)
        CoordinationPotentialAdapter().apply(action, follower)
        self.assertIsNone(follower.signal_marginal_price)
        self.assertIsNone(follower.offset_marginal_price)
        self.assertIsNone(follower.metering_marginal_price)
        self.assertIsNone(follower.vsl_marginal_price)
        self.assertIsNone(follower.signal_quadratic_price)
        self.assertIsNone(follower.vsl_quadratic_price)

    def test_linear_adapter_matches_existing_price_units(self):
        raw = np.zeros(self.schema.dimension, dtype=np.float32)
        raw[2] = 0.5
        first_freeway = 2 + 5 * len(self.cfg.network.signals)
        raw[first_freeway] = -0.4
        action = self.schema.decode(raw, self.previous, CoordinationMask.named("RL-LINEAR"))
        follower = F1WuFaithfulFollower(self.cfg)
        CoordinationPotentialAdapter().apply(action, follower)
        urban = action.urban_blocks[0]
        freeway = action.freeway_blocks[0]
        self.assertAlmostEqual(
            follower.signal_marginal_price[urban.owner],
            urban.linear[0] / urban.trust_radius[0],
        )
        self.assertAlmostEqual(
            follower.metering_marginal_price[freeway.owner],
            freeway.linear[0] / freeway.trust_radius[0],
        )
        self.assertTrue(all(value == 0.0 for value in follower.signal_quadratic_price.values()))
        self.assertTrue(all(value == 0.0 for value in follower.metering_quadratic_price.values()))

    def test_rl_adapter_activates_offset_and_priced_vsl_candidates(self):
        action = self.schema.decode(
            np.zeros(self.schema.dimension), self.previous, CoordinationMask.named("RL-BUDGET")
        )
        follower = F1WuFaithfulFollower(self.cfg)
        diagnostics = CoordinationPotentialAdapter().apply(action, follower)
        self.assertTrue(follower.ramp_offset_enabled)
        self.assertTrue(follower.priced_vsl_segment_candidates_enabled)
        self.assertEqual(diagnostics["coordination_ramp_offset_active"], 1.0)
        self.assertEqual(
            action.freeway_blocks[0].trust_radius[1],
            self.cfg.freeway_follower.max_vsl_step,
        )

    def test_rl_controller_owns_all_price_channels(self):
        action = self.schema.decode(
            np.zeros(self.schema.dimension), self.previous, CoordinationMask.named("RL-BUDGET")
        )
        controller = RLStackelbergController(self.cfg, StaticCoordinationProvider(action))
        controller._maybe_refresh_signal_prices(None, None, None)
        self.assertFalse(controller.signal_price_enabled)
        self.assertFalse(controller.offset_price_enabled)
        self.assertFalse(controller.metering_price_enabled)
        self.assertFalse(controller.vsl_price_enabled)
        self.assertIsNone(controller.nash_solver.signal_marginal_price)
        self.assertEqual(controller._signal_price_meta["wu_b2_price_refresh_count"], 0.0)

    def test_rl_response_candidates_are_pfo_centered_and_scale_full_price(self):
        raw = np.full(self.schema.dimension, 0.2, dtype=np.float32)
        action = self.schema.decode(raw, self.previous, CoordinationMask.named("RL-FULL"))
        controller = RLStackelbergController(
            self.cfg,
            StaticCoordinationProvider(action),
            response_candidate_count=10,
        )
        controller._pfo_incumbent_center = SimpleNamespace(
            N_P_star=400.0,
            N_UF_star=5200.0,
        )
        bounds = SimpleNamespace(
            np_lower=0.0,
            np_upper=2000.0,
            nuf_lower=0.0,
            nuf_upper=6000.0,
        )

        candidates = controller._response_candidates(action, bounds)

        self.assertEqual(len(candidates), 10)
        base, budget_scale, price_scale = candidates[0]
        self.assertEqual((budget_scale, price_scale), (1.0, 1.0))
        self.assertAlmostEqual(base.N_P_star, action.N_P_star)
        self.assertAlmostEqual(base.N_UF_star, action.N_UF_star)
        pfo_budget, budget_scale, price_scale = candidates[2]
        self.assertEqual((budget_scale, price_scale), (0.0, 1.0))
        self.assertAlmostEqual(pfo_budget.N_P_star, 400.0)
        self.assertAlmostEqual(pfo_budget.N_UF_star, 5200.0)
        self.assertEqual(pfo_budget.raw_budget, (400.0, 5200.0))
        strong_price, budget_scale, price_scale = candidates[6]
        self.assertEqual((budget_scale, price_scale), (1.0, 4.0))
        self.assertTrue(np.allclose(
            strong_price.urban_blocks[0].linear,
            4.0 * np.asarray(action.urban_blocks[0].linear),
        ))
        self.assertTrue(np.allclose(
            strong_price.urban_blocks[0].hessian(),
            4.0 * action.urban_blocks[0].hessian(),
        ))

    def test_rl_response_candidate_count_must_be_positive(self):
        action = self.schema.decode(
            np.zeros(self.schema.dimension), self.previous, CoordinationMask.named("RL-BUDGET")
        )
        with self.assertRaisesRegex(ValueError, "response_candidate_count"):
            RLStackelbergController(
                self.cfg,
                StaticCoordinationProvider(action),
                response_candidate_count=0,
            )
        with self.assertRaisesRegex(ValueError, "response_candidate_count"):
            RLStackelbergController(
                self.cfg,
                StaticCoordinationProvider(action),
                response_candidate_count=11,
            )

    def test_response_value_depth_extends_only_leader_ranking_horizon(self):
        short_env = RLLeaderEnv(
            scenario_name="sweet_170_incident_w60",
            response_candidate_count=10,
            response_value_depth=0,
        )
        long_env = RLLeaderEnv(
            scenario_name="sweet_170_incident_w60",
            response_candidate_count=10,
            response_value_depth=3,
        )

        short_observation = short_env.reset()
        long_observation = long_env.reset()

        self.assertEqual(len(short_env._forecast()), 3)
        self.assertEqual(len(long_env._forecast()), 6)
        self.assertEqual(len(long_env._actor_forecast()), 3)
        self.assertEqual(long_env.controller.response_horizon_steps, 6)
        self.assertTrue(np.array_equal(short_observation, long_observation))

    def test_response_value_depth_uses_long_objective_score(self):
        cfg, _ = make_cfg("medium_demand")
        schema = CoordinationActionSchema(cfg)
        action = schema.decode(
            np.zeros(schema.dimension), ControlAction.fixed(cfg), CoordinationMask.named("RL-BUDGET")
        )
        controller = RLStackelbergController(
            cfg,
            StaticCoordinationProvider(action),
            response_candidate_count=10,
            response_value_depth=3,
        )
        evaluation = SimpleNamespace(
            objective=12.0,
            nash=SimpleNamespace(
                diagnostics={"distributed_response_rollout_ttt": 3.0},
                control=SimpleNamespace(diagnostics={}),
            ),
        )

        self.assertEqual(controller._response_score_value(evaluation), 12.0)
        self.assertEqual(controller.cfg.mpc.leader_value_depth, 3)
        self.assertFalse(controller.cfg.mpc.stackelberg_fallback_guard_use_rollout_ttt)

    def test_response_value_depth_must_be_nonnegative(self):
        cfg, _ = make_cfg("medium_demand")
        schema = CoordinationActionSchema(cfg)
        action = schema.decode(
            np.zeros(schema.dimension), ControlAction.fixed(cfg), CoordinationMask.named("RL-BUDGET")
        )
        with self.assertRaisesRegex(ValueError, "response_value_depth"):
            RLStackelbergController(
                cfg,
                StaticCoordinationProvider(action),
                response_value_depth=-1,
            )

    def test_strict_pfo_gate_requires_positive_response_gain(self):
        cfg, _ = make_cfg("medium_demand")
        schema = CoordinationActionSchema(cfg)
        action = schema.decode(
            np.zeros(schema.dimension), ControlAction.fixed(cfg), CoordinationMask.named("RL-BUDGET")
        )
        controller = RLStackelbergController(
            cfg,
            StaticCoordinationProvider(action),
            strict_pfo_gate=True,
        )

        def evaluation(stage, rollout_ttt):
            control = ControlAction.fixed(cfg)
            control.diagnostics["distributed_response_rollout_ttt"] = float(rollout_ttt)
            nash = NashResult(
                control=control,
                objective_value=float(rollout_ttt),
                iterations=1,
                converged=True,
                residual_objective=0.0,
                residual_control=0.0,
                diagnostics=dict(control.diagnostics),
            )
            return _LeaderCandidateEvaluation(
                index=0,
                action=LeaderAction(control.N_P_star, control.N_UF_star),
                nash=nash,
                objective=float(rollout_ttt),
                objective_terms={"leader_follower_ttt_base": float(rollout_ttt)},
                metadata={},
                rollout_used=True,
                stage=stage,
            )

        reject, metadata = controller._fallback_guard_rejects(
            evaluation("rl", 99.95), evaluation("fallback_pfo", 100.0)
        )
        self.assertTrue(reject)
        self.assertEqual(metadata["leader_rl_strict_pfo_gate_active"], 1.0)
        self.assertEqual(metadata["leader_rl_strict_pfo_gate_insufficient_gain"], 1.0)

        reject, metadata = controller._fallback_guard_rejects(
            evaluation("rl", 99.0), evaluation("fallback_pfo", 100.0)
        )
        self.assertFalse(reject)
        self.assertEqual(metadata["leader_rl_strict_pfo_gate_insufficient_gain"], 0.0)

    def test_rl_follower_uses_pstack_b13_candidate_contract(self):
        env = RLLeaderEnv(scenario_name="sweet_170_incident_w60")
        optimizer = env._ensure_optimizer_controller()

        self.assertTrue(env.controller.nash_solver.segment_agents)
        self.assertEqual(
            env.controller.nash_solver.segment_agents,
            optimizer.nash_solver.segment_agents,
        )
        for name in (
            "seg13_meter_box_veh_h",
            "seg13_vsl_box_kmh",
            "baseline_move_box",
            "np_primal_dual_iters",
            "np_bias_correction",
        ):
            self.assertEqual(
                getattr(env.cfg.mpc, name),
                getattr(env.optimizer_cfg.mpc, name),
            )

    def test_rl_controller_uses_hybrid_far_gate(self):
        env = RLLeaderEnv(scenario_name="sweet_190_w60")
        forecast = env._forecast()

        env._update_rl_far_gate(forecast)
        self.assertFalse(env.cfg.mpc.leader_mfd_far_enabled)

        link = env.net.freeway_links[0]
        env.sim.state.freeway_density[link][0] = 1.1 * env.net.rho_crit
        env.sim.state.freeway_speed[link][0] = 0.0
        env._update_rl_far_gate(forecast)
        self.assertTrue(env.cfg.mpc.leader_mfd_far_enabled)

        for freeway_link in env.net.freeway_links:
            env.sim.state.freeway_density[freeway_link] = [
                0.5 * env.net.rho_crit
                for _ in env.sim.state.freeway_density[freeway_link]
            ]
        env._update_rl_far_gate(forecast)
        self.assertFalse(env.cfg.mpc.leader_mfd_far_enabled)

    def test_link_pfo_supervisor_selects_lower_common_score(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        candidate = env.previous.copy()
        candidate.N_UF_star = 5000.0
        pfo_control = env.previous.copy()
        pfo_control.N_UF_star = 4500.0
        pfo_nash = NashResult(
            control=pfo_control,
            objective_value=0.0,
            iterations=1,
            converged=True,
            residual_objective=0.0,
            residual_control=0.0,
            diagnostics={},
        )
        supervisor = SimpleNamespace(solve=lambda *args, **kwargs: pfo_nash)
        env._fixed_control_score = lambda control, forecast, cfg=None: (
            10.0 if control is pfo_control else 20.0
        )

        selected, selected_nash, metadata = env._pfo_supervisor_select(
            candidate,
            env._forecast(),
            supervisor,
            far_enabled=False,
        )

        self.assertIs(selected, pfo_control)
        self.assertIs(selected_nash, pfo_nash)
        self.assertEqual(metadata["sup_pick_pfo"], 1.0)
        self.assertEqual(metadata["sup_v_candidate"], 20.0)
        self.assertEqual(metadata["sup_v_pfo"], 10.0)

    def test_link_pfo_supervisor_is_disabled_while_far_gate_is_active(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        candidate = env.previous.copy()

        class UnexpectedSupervisor:
            def solve(self, *args, **kwargs):
                raise AssertionError("PFO supervisor should not run while FAR is active")

        selected, selected_nash, metadata = env._pfo_supervisor_select(
            candidate,
            env._forecast(),
            UnexpectedSupervisor(),
            far_enabled=True,
        )

        self.assertIs(selected, candidate)
        self.assertIsNone(selected_nash)
        self.assertEqual(metadata["sup_active"], 0.0)

    def test_pstack_anchor_requires_common_score_improvement(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        rl_control = env.previous.copy()
        pstack_control = env.previous.copy()
        metrics = {
            id(rl_control): {"ttt": 99.95, "terminal_inventory": 99.0},
            id(pstack_control): {"ttt": 100.0, "terminal_inventory": 100.0},
        }
        env._fixed_control_metrics = lambda control, forecast, cfg=None: metrics[id(control)]

        selected, metadata = env._pstack_anchor_select(
            rl_control, pstack_control, env._forecast()
        )

        self.assertIs(selected, pstack_control)
        self.assertEqual(metadata["leader_rl_pstack_anchor_pick_rl"], 0.0)
        metrics[id(rl_control)] = {"ttt": 99.0, "terminal_inventory": 99.0}
        selected, metadata = env._pstack_anchor_select(
            rl_control, pstack_control, env._forecast()
        )
        self.assertIs(selected, rl_control)
        self.assertEqual(metadata["leader_rl_pstack_anchor_pick_rl"], 1.0)

        metrics[id(rl_control)] = {"ttt": 90.0, "terminal_inventory": 100.1}
        selected, metadata = env._pstack_anchor_select(
            rl_control, pstack_control, env._forecast()
        )
        self.assertIs(selected, pstack_control)
        self.assertEqual(metadata["leader_rl_pstack_anchor_inventory_blocked"], 1.0)

    def test_pstack_anchor_rejects_pfo_supervisor_baseline_override(self):
        with self.assertRaisesRegex(ValueError, "P-Stack anchor"):
            RLLeaderEnv(
                scenario_name="medium_demand",
                pfo_supervisor=True,
                pstack_anchor=True,
            )

    def test_pstack_residual_policy_keeps_native_release_certificates(self):
        env = RLLeaderEnv(
            scenario_name="medium_demand",
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        policy = np.ones(env.action_dim, dtype=np.float32)

        prepared = env._policy_raw_action(policy)

        certificate_count = len(env.action_schema.certificate_ramps)
        np.testing.assert_array_equal(prepared[:-certificate_count], 1.0)
        np.testing.assert_array_equal(prepared[-certificate_count:], 0.0)

    def test_observation_accepts_segment_lane_loss_mapping(self):
        env = RLLeaderEnv(scenario_name="sweet_190_incident_w60")
        env.step_idx = int(2400.0 / env.dt)
        observation = env._observe()
        lane_loss_indices = [
            index
            for index, name in enumerate(env.observation_schema.names)
            if name.endswith(".lane_loss")
        ]
        self.assertTrue(lane_loss_indices)
        self.assertGreater(float(observation[lane_loss_indices].max()), 0.0)

    def test_observation_distinguishes_previous_price_references(self):
        env = RLLeaderEnv(scenario_name="sweet_170_incident_w60")
        state = env.sim.state.copy()
        forecast = env._forecast()
        previous_a = env.previous.copy()
        previous_b = env.previous.copy()
        previous_b.green_times["A_p1"] += 6.0
        previous_b.green_times["A_p2"] -= 6.0
        previous_b.offsets["A"] = 15.0
        previous_b.vsl["FW_E__seg0"] = 100.0

        observation_a = env.observation_schema.observe(
            state, forecast, previous_a, env.controller, env.n_steps,
        )
        observation_b = env.observation_schema.observe(
            state, forecast, previous_b, env.controller, env.n_steps,
        )

        self.assertFalse(np.array_equal(observation_a, observation_b))

    def test_observation_distinguishes_segment_location(self):
        env = RLLeaderEnv(scenario_name="sweet_170_incident_w60")
        state_a = env.sim.state.copy()
        state_b = env.sim.state.copy()
        link = env.net.freeway_links[0]
        count = len(state_a.freeway_density[link])
        density = np.linspace(0.5, 1.5, count) * float(env.net.rho_crit)
        speed = np.linspace(0.95, 0.45, count) * float(env.net.v_free)
        state_a.freeway_density[link] = density.tolist()
        state_a.freeway_speed[link] = speed.tolist()
        state_b.freeway_density[link] = density[::-1].tolist()
        state_b.freeway_speed[link] = speed[::-1].tolist()
        forecast_a = copy.deepcopy(env._forecast())
        forecast_b = copy.deepcopy(forecast_a)
        for demand in forecast_a:
            demand.freeway_lane_loss = {link: {5: 1.0}}
        for demand in forecast_b:
            demand.freeway_lane_loss = {link: {6: 1.0}}

        observation_a = env.observation_schema.observe(
            state_a, forecast_a, env.previous, env.controller, env.n_steps,
        )
        observation_b = env.observation_schema.observe(
            state_b, forecast_b, env.previous, env.controller, env.n_steps,
        )

        self.assertFalse(np.array_equal(observation_a, observation_b))

    def test_observation_distinguishes_follower_warm_start_state(self):
        env = RLLeaderEnv(scenario_name="sweet_170_incident_w60")
        follower = env.controller.nash_solver
        follower._prev_coupling = {"arr_A_p1": 100.0}
        follower._np_bias_ratio = 0.8
        observation_a = env._observe()
        follower._prev_coupling = {"arr_A_p1": 200.0}
        follower._np_bias_ratio = 1.2
        observation_b = env._observe()

        self.assertFalse(np.array_equal(observation_a, observation_b))

    def test_optimizer_anchor_collection_uses_rl_adapter_transition(self):
        class FakeEnv:
            def __init__(self):
                self.mask = None
                self.action = np.asarray([0.25, -0.5], dtype=np.float32)
                self.received = None

            def optimizer_anchor_action(self):
                return self.action

            def step(self, action):
                self.received = action
                return "next", -1.0, False, {"path": "adapter"}

        env = FakeEnv()
        next_obs, reward, done, info, action, mask = optimizer_replay_anchor_step(env)

        self.assertIs(env.received, action)
        self.assertEqual(next_obs, "next")
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)
        self.assertEqual(info["path"], "adapter")
        self.assertEqual(mask, CoordinationMask.named("RL-LINEAR"))
        self.assertEqual(env.mask, mask)

    def test_optimizer_teacher_query_syncs_deployable_follower_state(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        optimizer = env._ensure_optimizer_controller()
        env.controller.nash_solver._prev_coupling = {"arr_A_p1": 123.0}
        env.controller.nash_solver._lambda_P = 0.75
        optimizer.nash_solver._prev_coupling = {"arr_A_p1": 999.0}
        optimizer.nash_solver._lambda_P = 9.0

        env._sync_optimizer_follower_state()

        self.assertIsNot(optimizer.nash_solver, env.controller.nash_solver)
        self.assertEqual(optimizer.nash_solver._prev_coupling, {"arr_A_p1": 123.0})
        self.assertEqual(optimizer.nash_solver._lambda_P, 0.75)
        optimizer.nash_solver._prev_coupling["arr_A_p1"] = 456.0
        self.assertEqual(env.controller.nash_solver._prev_coupling["arr_A_p1"], 123.0)

    def test_optimizer_teacher_sync_preserves_production_follower_configuration(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        optimizer = env._ensure_optimizer_controller()
        optimizer.nash_solver.ramp_offset_enabled = True
        env.controller.nash_solver.ramp_offset_enabled = False
        env.controller.nash_solver._lambda_P = 0.75

        env._sync_optimizer_follower_state()

        self.assertTrue(optimizer.nash_solver.ramp_offset_enabled)
        self.assertEqual(optimizer.nash_solver._lambda_P, 0.75)

    def test_rl_controller_uses_production_b13_rollout_contract(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        optimizer = env._ensure_optimizer_controller()

        self.assertTrue(env.cfg.mpc.leader_rollout_box_walk)
        self.assertTrue(env.cfg.mpc.leader_rollout_box_walk_vg)
        self.assertTrue(env.cfg.mpc.leader_mfd_far_state_aware)
        self.assertTrue(env.cfg.mpc.leader_mfd_far_real_speed)
        self.assertTrue(env.controller.nash_solver.ramp_offset_enabled)
        self.assertEqual(optimizer.cfg.mpc.leader_search_mode, "grid")
        self.assertEqual(optimizer.cfg.mpc.grid_parallel_backend, "serial")
        self.assertEqual(
            optimizer.cfg.mpc.stackelberg_leader_parallel_backend, "serial"
        )
        self.assertEqual(optimizer.cfg.mpc.leader_value_depth, 3)
        self.assertTrue(optimizer.cfg.mpc.leader_skip_local_refinement)
        self.assertFalse(optimizer.cfg.mpc.leader_rollout_early_stop)
        self.assertEqual(optimizer.offset_price_inner_iters, 4)
        self.assertTrue(optimizer.nash_solver.ramp_offset_enabled)
        self.assertEqual(len(env._optimizer_forecast()), 6)

    def test_rl_warmup_matches_production_uncontrolled_control(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=5)
        manual = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        control = ControlAction.uncontrolled(manual.cfg)
        for _ in range(5):
            manual.step_with_control(control)

        self.assertAlmostEqual(
            env.sim.state.total_urban_vehicles(env.net),
            manual.sim.state.total_urban_vehicles(manual.net),
            places=12,
        )
        self.assertAlmostEqual(
            env.sim.state.total_freeway_vehicles(env.net),
            manual.sim.state.total_freeway_vehicles(manual.net),
            places=12,
        )
        self.assertEqual(env.sim.state.ramp_queue, manual.sim.state.ramp_queue)

    def test_structured_exploration_can_target_priority_block(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        action, _ = structured_action(
            env,
            np.random.default_rng(11),
            "single_block",
            priority_blocks=("B",),
            priority_probability=1.0,
        )
        layout = action_block_layout(env.action_schema)
        target = next(block for _, owner, block, _ in layout if owner == "B")
        self.assertGreater(float(np.abs(action[target]).sum()), 0.0)
        other = np.ones(action.size, dtype=bool)
        other[:2] = False
        other[target] = False
        other[-len(env.action_schema.certificate_ramps):] = False
        self.assertEqual(float(np.abs(action[other]).sum()), 0.0)
        self.assertTrue(np.all(action[-len(env.action_schema.certificate_ramps):] == -1.0))

    def test_family_qualified_priority_excludes_same_owner_certificate(self):
        layout = action_block_layout(self.schema)

        qualified = priority_block_indices(layout, ("freeway:R_D_W",))
        legacy = priority_block_indices(layout, ("R_D_W",))

        self.assertEqual([layout[index][0] for index in qualified], ["freeway"])
        self.assertEqual(
            {layout[index][0] for index in legacy}, {"freeway", "certificate"}
        )

    def test_structured_exploration_treats_release_certificate_as_binary(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        certificate_owner = env.action_schema.certificate_ramps[-1]
        certificate_slice = next(
            block for family, owner, block, _ in action_block_layout(env.action_schema)
            if family == "certificate" and owner == certificate_owner
        )
        observed = []
        for seed in range(20):
            action, _ = structured_action(
                env,
                np.random.default_rng(seed),
                "correlated_block",
                priority_blocks=(certificate_owner,),
                priority_probability=1.0,
            )
            observed.append(float(action[certificate_slice][0]))
        self.assertTrue(set(observed).issubset({-1.0, 0.0, 1.0}))
        self.assertTrue(any(value != 0.0 for value in observed))

    def test_targeted_scenarios_retain_five_cell_identity(self):
        scenarios = [make_targeted_scenario(np.random.default_rng(seed)) for seed in range(100)]
        allowed = {
            "sweet_155_w60", "sweet_170_w60", "sweet_170_incident_w60",
            "sweet_170_skew15_w60", "sweet_190_w60",
        }
        self.assertEqual({scenario["target_scenario"] for scenario in scenarios}, allowed)
        for scenario in scenarios:
            target = scenario["target_scenario"]
            nominal = float(target.split("_")[1]) / 100.0
            self.assertLessEqual(abs(scenario["urban_scale"] / nominal - 1.0), 0.02001)
            if "incident" in target:
                self.assertTrue(scenario["freeway_lane_closures"])
            else:
                self.assertFalse(scenario.get("freeway_lane_closures"))
            if "skew" in target:
                self.assertGreaterEqual(scenario["urban_west_east_ratio"], 1.4)
                self.assertLessEqual(scenario["urban_west_east_ratio"], 1.6)
            else:
                self.assertIsNone(scenario.get("urban_west_east_ratio"))

    def test_targeted_scenario_can_be_restricted_for_topup(self):
        scenario = make_targeted_scenario(
            np.random.default_rng(0), ("sweet_190_w60",),
        )

        self.assertEqual(scenario["target_scenario"], "sweet_190_w60")
        with self.assertRaisesRegex(ValueError, "unknown targeted scenarios"):
            make_targeted_scenario(np.random.default_rng(0), ("missing",))

    def test_loose_anchor_is_max_nuf_with_zero_prices(self):
        env = RLLeaderEnv(scenario_name="medium_demand")
        action = loose_anchor_action(env)
        self.assertAlmostEqual(float(action[1]), 1.0)
        certificate_count = len(env.action_schema.certificate_ramps)
        self.assertEqual(float(np.abs(action[2:-certificate_count]).sum()), 0.0)
        self.assertTrue(np.all(action[-certificate_count:] == -1.0))

    def test_temporal_perturbation_only_changes_selected_blocks_after_activation(self):
        base = np.zeros(self.schema.dimension, dtype=np.float32)
        noise = np.zeros_like(base)
        layout = action_block_layout(self.schema)
        target = layout[2][2]
        positive = tuple(index for _, _, _, indices in layout for index in indices)
        inactive, noise = temporally_correlated_action(
            base, noise, np.random.default_rng(4), (target,), positive,
            rho=0.95, budget_scale=0.05, block_scale=0.1, active=False,
        )
        self.assertTrue(np.array_equal(inactive, base))
        active, _ = temporally_correlated_action(
            base, noise, np.random.default_rng(5), (target,), positive,
            rho=0.95, budget_scale=0.05, block_scale=0.1, active=True,
        )
        changed = np.flatnonzero(np.abs(active) > 1.0e-9)
        allowed = set(range(2)) | set(range(target.start, target.stop))
        self.assertTrue(changed.size)
        self.assertTrue(set(changed).issubset(allowed))
        self.assertGreaterEqual(float(active[target.start + 2]), 0.0)
        self.assertGreaterEqual(float(active[target.start + 4]), 0.0)


if __name__ == "__main__":
    unittest.main()
