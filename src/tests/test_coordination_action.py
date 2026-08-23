import unittest
import copy
from types import SimpleNamespace

import numpy as np

from rl_leader.collect_full_action import (
    action_block_layout,
    loose_anchor_action,
    optimizer_replay_anchor_step,
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
from src.controllers.rl_stackelberg import OptimizerCoordinationProvider, RLStackelbergController
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

    def test_optimizer_encoding_preserves_intent_and_all_vsl_prices(self):
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

        self.assertAlmostEqual(coordination.N_P_star, 450.0)
        self.assertAlmostEqual(coordination.N_UF_star, 5700.0)
        self.assertLess(
            float(np.abs(encoded[2:-len(self.schema.certificate_ramps)]).max()), 1.0,
        )
        self.assertTrue(all(coordination.metering_release_certified))
        self.assertTrue(all(replay_follower.metering_release_certified.values()))
        for key, price in native_vsl.items():
            self.assertAlmostEqual(replay_follower.vsl_marginal_price[key], price)

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
