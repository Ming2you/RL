"""No-solve mutation tests, using an existing checkpoint only as a read-only fixture.

Run: python -B -m unittest src.tests.test_response_continuation_state -v
The checkpoint suite explicitly skips if the local experiment artifact is absent;
the standalone encoder tests do not require experiment results or a simulator.
"""
from collections import deque
import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.experiment_contract import ExperimentContract
from rl_leader.response_continuation_state import (
    AUDITED_CONTRACT_SHA256, IDENTITY_VERSION, OBSOLETE_RL_INPUTS,
    _encode, _sha, continuation_identity,
)
from src.controllers.f1_wu_faithful_follower import F1WuFaithfulFollower
from src.controllers.rl_stackelberg import RLStackelbergController
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.models.state import TrafficState
from src.simulation.simulator import MixedTrafficSimulator


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / (
    "results/response_dqn_170_incident/sequential_cql_reference_v1/"
    "with_references/evaluation/checkpoint.pkl"
)


def digest(value):
    return _sha(_encode(value))


class ExactEncodingTests(unittest.TestCase):
    def test_mapping_and_set_order_are_canonical(self):
        left = {1: "integer", "1": "string", "set": {"a", "b"}}
        right = {"set": {"b", "a"}, "1": "string", 1: "integer"}
        self.assertEqual(digest(left), digest(right))
        self.assertNotEqual(digest({1: 2}), digest({"1": 2}))

    def test_sequence_type_order_and_deque_limit_are_retained(self):
        samples = [[1, 2], [2, 1], (1, 2), deque([1, 2]), deque([1, 2], maxlen=3)]
        self.assertEqual(len({digest(value) for value in samples}), len(samples))

    def test_no_rounding_or_negative_zero_loss(self):
        self.assertNotEqual(digest(0.0), digest(-0.0))
        self.assertNotEqual(digest(1.0), digest(float(np.nextafter(1.0, 2.0))))
        self.assertNotEqual(digest(1), digest(1.0))

    def test_arrays_preserve_dtype_shape_exact_values_not_strides(self):
        array = np.arange(6, dtype=np.float64).reshape(2, 3)
        self.assertEqual(digest(array), digest(np.asfortranarray(array)))
        for different in (array.astype(np.float32), array.reshape(3, 2), array + 1):
            self.assertNotEqual(digest(array), digest(different))
        different = array.copy()
        different[0, 0] = -0.0
        self.assertNotEqual(digest(array), digest(different))
        self.assertNotEqual(digest(np.float64(1.0)), digest(1.0))

    def test_pickle_stability(self):
        value = {"array": np.array([1., -0.]), "queue": deque([(1, 2)], maxlen=4)}
        for protocol in (4, 5):
            self.assertEqual(digest(value), digest(pickle.loads(pickle.dumps(value, protocol))))

    def test_unsupported_objects_do_not_use_repr(self):
        class Unstable:
            def __repr__(self):
                raise AssertionError("repr must not be called")

        for value in (Unstable(), SimpleNamespace(x=1), np.array([object()], dtype=object)):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(TypeError):
                    digest(value)

    def test_nonfinite_and_cycles_reject(self):
        for value in (float("nan"), float("inf"), np.array([float("nan")])):
            with self.assertRaises(ValueError):
                digest(value)
        cycle = {}
        cycle["self"] = cycle
        with self.assertRaisesRegex(ValueError, "cyclic"):
            digest(cycle)

    def test_known_dataclass_extra_field_is_retained(self):
        # A tiny actual data-class mock, not a full environment construction.
        plant = TrafficState.__new__(TrafficState)
        plant.time_sec = 0.
        before = digest(plant)
        plant.future_causal_buffer = {1: [0., 2.]}
        self.assertNotEqual(before, digest(plant))
        plant.future_causal_buffer = object()
        with self.assertRaises(TypeError):
            digest(plant)

    def test_unknown_environment_and_bad_reward_types_reject(self):
        with self.assertRaises(TypeError):
            continuation_identity(SimpleNamespace(), interval_reward=-1., terminal=False)
        for reward in (1, np.float64(1), float("nan")):
            with self.assertRaisesRegex(ValueError, "interval_reward"):
                continuation_identity(None, interval_reward=reward, terminal=False)
        with self.assertRaisesRegex(ValueError, "terminal"):
            continuation_identity(None, interval_reward=-1., terminal=1)


class CheckpointContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.is_file():
            raise unittest.SkipTest(f"read-only local checkpoint unavailable: {CHECKPOINT}")
        with CHECKPOINT.open("rb") as handle:
            cls.fixture = pickle.load(handle)["env"]
        # Class patches are outside instance state and catch accidental simulation.
        for owner, method in (
            (RLLeaderEnv, "reset"), (RLLeaderEnv, "step"),
            (RLLeaderEnv, "_forecast"), (RLLeaderEnv, "_actor_forecast"),
            (MixedTrafficSimulator, "step"),
            (RLStackelbergController, "decide_with_info"),
            (StackelbergWuMeteredController, "decide_with_info"),
            (F1WuFaithfulFollower, "_solve_followers"),
        ):
            guard = patch.object(owner, method, side_effect=AssertionError("simulation forbidden"))
            guard.start()
            cls.addClassCleanup(guard.stop)

    def setUp(self):
        self.env = copy.deepcopy(self.fixture)
        # Mutating one follower below must not create an inconsistent active
        # observation by accident; the canonical RL role is a valid active role.
        self.env._active_controller = self.env.controller
        self.baseline = self.identity()

    def identity(self, env=None, reward=-1.0, terminal=True):
        return continuation_identity(self.env if env is None else env,
                                     interval_reward=reward, terminal=terminal)

    def assert_component_changed(self, component):
        result = self.identity()
        self.assertNotEqual(result["sha256"], self.baseline["sha256"])
        self.assertNotEqual(result["component_sha256"][component],
                            self.baseline["component_sha256"][component])

    def test_json_version_and_component_contract(self):
        self.assertEqual(json.loads(json.dumps(self.baseline)), self.baseline)
        self.assertEqual(self.baseline["version"], IDENTITY_VERSION)
        self.assertEqual(self.baseline["experiment_contract_sha256"], AUDITED_CONTRACT_SHA256)
        self.assertEqual(set(self.baseline["component_sha256"]), {
            "plant", "previous_control", "rl_controller", "native_controller",
            "rl_follower", "native_follower", "runtime", "transition",
        })
        for value in self.baseline["component_sha256"].values():
            self.assertEqual(len(bytes.fromhex(value)), 32)

    def test_helper_is_read_only_and_pickle_stable(self):
        before = pickle.dumps(self.env, protocol=5)
        self.assertEqual(self.identity(), self.baseline)
        self.assertEqual(pickle.dumps(self.env, protocol=5), before)
        for protocol in (4, 5):
            restored = pickle.loads(pickle.dumps(self.env, protocol=protocol))
            self.assertEqual(self.identity(restored), self.baseline)

    def test_all_follower_warm_state_is_retained(self):
        replacements = {
            "_prev_coupling": {"arr_A_p1": 1.234}, "_lambda_P": 3.25,
            "_lambda_UF": 0.25, "_np_last_sum_nin": 12., "_np_prev_accum": 13.,
            "_np_step_time": 14., "_np_corrector_pending": (0.75, 33.),
            "_np_last_real_q": 15., "_np_bias_ratio": 0.99,
            "_phase_resolved_active_signals": {"A", "D"},
        }
        for role in ("controller", "optimizer_controller"):
            component = "rl_follower" if role == "controller" else "native_follower"
            for field, value in replacements.items():
                with self.subTest(role=role, field=field):
                    follower = getattr(self.env, role).nash_solver
                    original = getattr(follower, field)
                    setattr(follower, field, value)
                    self.assert_component_changed(component)
                    setattr(follower, field, original)

    def test_inner_follower_warm_state_is_retained(self):
        for role in ("controller", "optimizer_controller"):
            component = "rl_follower" if role == "controller" else "native_follower"
            wu = getattr(self.env, role).nash_solver._wu
            for field in ("_last_offramp_flow", "_omega_f", "_omega_p"):
                with self.subTest(role=role, field=field):
                    original = getattr(wu, field)
                    setattr(wu, field, {"new": 13.})
                    self.assert_component_changed(component)
                    setattr(wu, field, original)
            wu._has_last_offramp_flow = not wu._has_last_offramp_flow
            self.assert_component_changed(component)
            wu._has_last_offramp_flow = not wu._has_last_offramp_flow

    def test_obsolete_residual_inputs_do_not_split_but_strict_snapshot_does(self):
        follower = self.env.controller.nash_solver
        strict_before = self.env._follower_runtime_fingerprint(follower)
        for field in sorted(OBSOLETE_RL_INPUTS):
            with self.subTest(field=field):
                original = getattr(follower, field)
                setattr(follower, field, {"obsolete": 876.25})
                self.assertEqual(self.identity(), self.baseline)
                setattr(follower, field, original)
        follower.signal_marginal_price = {"A": 876.25}
        self.assertNotEqual(self.env._follower_runtime_fingerprint(follower), strict_before)
        self.assertEqual(self.identity(), self.baseline)

    def test_same_inputs_on_native_are_all_retained(self):
        follower = self.env.optimizer_controller.nash_solver
        for field in sorted(OBSOLETE_RL_INPUTS):
            with self.subTest(field=field):
                original = getattr(follower, field)
                setattr(follower, field, {"native": 876.25})
                self.assert_component_changed("native_follower")
                setattr(follower, field, original)

    def test_reset_scratch_does_not_split(self):
        for role in ("controller", "optimizer_controller"):
            controller = getattr(self.env, role)
            for field in ("_link_share_ctx", "_nuf_solve_cache", "_dedupe_hits",
                          "_pfo_incumbent_center", "_pfo_incumbent_eval", "last_decision",
                          "previous_control", "last_candidate_common_solver"):
                setattr(controller, field, {"obsolete": 333.})
            follower = controller.nash_solver
            follower._seg13_diag = {"obsolete": 1.}
            follower._seg_traj = {"obsolete": [3.]}
            follower.last_candidate_trace = {"obsolete": 2.}
            follower._wu._repair_diagnostics = {"obsolete": 3.}
        for field in ("last_response_candidate_trace", "_response_candidate_actions",
                      "_response_candidate_solvers", "_response_pfo_solver", "_signal_price_meta"):
            setattr(self.env.controller, field, {"obsolete": 333.})
        self.assertEqual(self.identity(), self.baseline)

    def test_controller_pricing_and_regret_history_are_retained(self):
        replacements = {
            "_signal_price_refresh_count": 913, "_signal_price_last_step": 73,
            "_beta_hat": 0.77, "_beta_ewma_weight": 0.33,
            "_beta_ratio_clip": (0.1, 0.2), "_beta_pending": deque([(7., 8.)], maxlen=3),
            "_beta_prev_total_veh": 111., "_beta_last_realized": 8.2,
            "_beta_drift_streak": 99, "_regret_window": deque([(1., 9.)], maxlen=2),
            "_regret_pending_inc_pred": 57., "_regret_forced_remaining": 7,
            "_regret_force_this_step": True, "_regret_last_gap": 4.6,
            "_beta_regret_entry_time": 99., "_beta_regret_entry_meta": {"retained": 4.},
        }
        for role in ("controller", "optimizer_controller"):
            component = "rl_controller" if role == "controller" else "native_controller"
            controller = getattr(self.env, role)
            for field, value in replacements.items():
                with self.subTest(role=role, field=field):
                    original = getattr(controller, field)
                    # The fixture's RL last-refresh step can equal this probe.
                    replacement = value + 1 if type(value) is int and original == value else value
                    setattr(controller, field, replacement)
                    self.assert_component_changed(component)
                    setattr(controller, field, original)
        self.env.optimizer_controller._signal_price_meta["retained"] = 2.
        self.assert_component_changed("native_controller")

    def test_previous_budgets_controls_and_causal_diagnostic_are_retained(self):
        for field in vars(self.env.previous):
            if field == "diagnostics":
                continue
            with self.subTest(field=field):
                original = getattr(self.env.previous, field)
                replacement = original + 0.01 if type(original) is float else {"changed": 7.}
                setattr(self.env.previous, field, replacement)
                self.assert_component_changed("previous_control")
                setattr(self.env.previous, field, original)
        self.env.previous.diagnostics["wu_b3_release_ratio_FW_W"] = 0.12345
        self.assert_component_changed("previous_control")

    def test_full_plant_including_buffers_and_unknown_fields_is_retained(self):
        for field in vars(self.env.sim.state):
            with self.subTest(field=field):
                original = getattr(self.env.sim.state, field)
                replacement = original + 0.01 if type(original) is float else {"changed": [7.]}
                setattr(self.env.sim.state, field, replacement)
                self.assert_component_changed("plant")
                setattr(self.env.sim.state, field, original)
        self.env.sim.state.future_causal_buffer = {3: [1., 9.]}
        self.assert_component_changed("plant")

    def test_far_flags_and_runtime_config_are_retained(self):
        for field in ("_rl_fargate_stress", "_optimizer_fargate_stress"):
            setattr(self.env, field, not getattr(self.env, field))
            self.assert_component_changed("runtime")
            setattr(self.env, field, not getattr(self.env, field))
        for cfg in (self.env.cfg, self.env.optimizer_cfg):
            cfg.mpc.leader_mfd_far_enabled = not cfg.mpc.leader_mfd_far_enabled
            self.assert_component_changed("runtime")
            cfg.mpc.leader_mfd_far_enabled = not cfg.mpc.leader_mfd_far_enabled

    def test_step_reward_done_are_retained_and_horizon_is_guarded(self):
        self.env.step_idx -= 1
        self.assert_component_changed("runtime")
        self.env.step_idx += 1
        for reward, terminal in ((-1.0000000000000002, True), (-1., False)):
            result = self.identity(reward=reward, terminal=terminal)
            self.assertNotEqual(result["component_sha256"]["transition"],
                                self.baseline["component_sha256"]["transition"])
        self.env.n_steps += 1
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.identity()

    def test_branch_and_action_reporting_are_neutral(self):
        self.env._active_controller = self.env.optimizer_controller
        self.env.last_policy_raw_action = np.full(self.env.action_dim, 0.9)
        self.env.last_deployed_residual = np.full(self.env.action_dim, -0.9)
        self.env.last_anchor_raw_action = np.zeros(self.env.action_dim)
        self.env.last_applied_raw_action = np.ones(self.env.action_dim)
        self.env.last_requested_coordination = None
        self.env.last_optimizer_anchor_metadata = {"label": "different"}
        self.env.last_optimizer_anchor_response = np.array([99.])
        self.env.last_optimizer_anchor_coordination = None
        self.env.provider.action = None
        self.env.controller.last_coordination_action = None
        self.env.controller.last_coordination_metadata = {"label": "different"}
        self.assertEqual(self.identity(), self.baseline)

    def test_branch_label_cannot_hide_different_observation_memory(self):
        self.env._active_controller = self.env.optimizer_controller
        self.env.optimizer_controller.nash_solver._lambda_P += 1.
        with self.assertRaisesRegex(ValueError, "observation memory"):
            self.identity()

    def test_known_reporting_diagnostics_and_accounting_are_neutral(self):
        for field in ("leader_provider_rl", "leader_provider_optimizer", "leader_search_bypassed",
                      "coordination_linear_active", "wu_faithful_solve_time_sec"):
            self.env.previous.diagnostics[field] = 123.456
        self.env.sim.freeway_ttt += 1000.
        self.env.sim.urban_ttt += 1000.
        self.env.sim.logs = []
        self.assertEqual(self.identity(), self.baseline)

    def test_indirect_price_objective_reports_and_rl_counters_still_split(self):
        # Intentional v1 limitation, measured on the saved committed state.
        # Do not accidentally turn this into a broad diagnostics exclusion.
        for field in ("wu_b2_price_A", "wu_b3_meter_price_R_D_W",
                      "leader_total_objective", "leader_selected_stage_fallback"):
            with self.subTest(field=field):
                original = self.env.previous.diagnostics[field]
                self.env.previous.diagnostics[field] += 0.125
                self.assert_component_changed("previous_control")
                self.env.previous.diagnostics[field] = original
        for field in ("_signal_price_last_step", "_price_rollout_count"):
            original = getattr(self.env.controller, field)
            setattr(self.env.controller, field, original + 1)
            self.assert_component_changed("rl_controller")
            setattr(self.env.controller, field, original)

    def test_accumulation_report_copies_do_not_split_continuation(self):
        for field in ("leader_base_accumulation", "leader_state_accumulation_base",
                      "leader_boundary_leg_excluded_veh"):
            with self.subTest(field=field):
                self.env.previous.diagnostics[field] = 123456.789
                self.assertEqual(self.identity(), self.baseline)
        self.env.previous.diagnostics["wu_b3_release_ratio_FW_W"] = 0.12345
        self.assert_component_changed("previous_control")

    def test_unknown_runtime_and_diagnostic_fields_are_not_silently_dropped(self):
        targets = ((self.env, "runtime"), (self.env.sim, "runtime"),
                   (self.env.controller, "rl_controller"),
                   (self.env.controller.nash_solver, "rl_follower"),
                   (self.env.controller.nash_solver._wu, "rl_follower"))
        for obj, component in targets:
            obj.future_causal_field = {"x": 1.}
            self.assert_component_changed(component)
            del obj.future_causal_field
            obj.future_causal_field = object()
            with self.assertRaisesRegex(TypeError, "future_causal_field"):
                self.identity()
            del obj.future_causal_field
        self.env.previous.diagnostics["unknown_future_diagnostic"] = 1.
        self.assert_component_changed("previous_control")

    def test_static_descriptor_mutation_and_unknown_model_object_are_not_ignored(self):
        follower = self.env.controller.nash_solver
        model = next(iter(follower._local_models.values()))
        model.future_causal_field = 1.
        self.assert_component_changed("rl_follower")
        del model.future_causal_field
        model.cap_flow_of[model.movements[0]] += 1.
        self.assert_component_changed("rl_follower")
        follower._local_models["unknown"] = object()
        with self.assertRaises(TypeError):
            self.identity()

    def test_mapping_reinsertion_order_does_not_split(self):
        self.env.previous.diagnostics = dict(reversed(list(self.env.previous.diagnostics.items())))
        self.env.controller.nash_solver._prev_coupling = dict(reversed(list(
            self.env.controller.nash_solver._prev_coupling.items())))
        self.env.sim.state.urban_movement_queue = dict(reversed(list(
            self.env.sim.state.urban_movement_queue.items())))
        self.assertEqual(self.identity(), self.baseline)

    def test_unaudited_modes_reject(self):
        cases = (
            (self.env, "action_mode", "legacy_budget"),
            (self.env, "response_candidate_count", 2),
            (self.env, "strict_pfo_gate", True),
            (self.env, "pfo_supervisor_enabled", True),
            (self.env.controller, "allow_internal_pfo_fallback", True),
            (self.env.optimizer_controller, "price_spsa_enabled", True),
            (self.env.optimizer_controller, "price_iter_max", 2),
            (self.env.optimizer_controller, "price_refresh_interval", 2),
            (self.env.optimizer_controller, "nuf_link_share_mode", "search"),
            (self.env.optimizer_controller, "price_lite", True),
        )
        for obj, field, value in cases:
            with self.subTest(field=field):
                original = getattr(obj, field)
                setattr(obj, field, value)
                with self.assertRaises(ValueError):
                    self.identity()
                setattr(obj, field, original)

    def test_uncommitted_trials_or_seeds_reject(self):
        for obj, field in ((self.env, "_pending_optimizer_trial"),
                           (self.env.controller, "_anchor_follower_seed"),
                           (self.env.optimizer_controller, "_candidate_common_solver")):
            with self.subTest(field=field):
                original = getattr(obj, field, None)
                setattr(obj, field, "uncommitted")
                with self.assertRaises(ValueError):
                    self.identity()
                setattr(obj, field, original)

    def test_contract_and_live_config_tampering_reject(self):
        original = self.env._experiment_contract
        payload = original.payload
        payload["simulation"]["T_total_sec"] += 180.
        self.env._experiment_contract = ExperimentContract.from_payload(payload)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.identity()
        self.env._experiment_contract = original
        self.env.cfg.mpc.leader_gradseed_enabled = True
        with self.assertRaisesRegex(ValueError, "config"):
            self.identity()

    def test_config_alias_and_unknown_config_fields_reject(self):
        original = self.env.sim.cfg
        self.env.sim.cfg = copy.deepcopy(original)
        with self.assertRaisesRegex(ValueError, "alias"):
            self.identity()
        self.env.sim.cfg = original
        self.env.cfg.mpc.future_mode = True
        with self.assertRaisesRegex(ValueError, "config"):
            self.identity()

    def test_partial_or_changed_schema_rejects_conditional_exclusions(self):
        original = self.env.action_schema.signals
        self.env.action_schema.signals = ()
        with self.assertRaisesRegex(ValueError, "schema"):
            self.identity()
        self.env.action_schema.signals = original
        self.env.observation_schema.memory_names = ()
        with self.assertRaisesRegex(ValueError, "schema"):
            self.identity()


if __name__ == "__main__":
    unittest.main()
