"""Opt-in contract extension without simulator rollouts or model training."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.experiment_contract import ExperimentContract
from rl_leader.response_continuation_state import (
    AUDITED_CONTRACT_SHA256, FIVE_CELL_CONTRACT_SHA256, FIVE_CELL_IDENTITY_VERSION,
    FIVE_CELL_SHARED_CONTRACT_SHA256, continuation_identity,
    five_cell_continuation_identity, validate_five_cell_contract, _sha,
)
from rl_leader.response_dqn_collect import (
    _candidate_trial, _equivalence_mode, _validate_policy_equivalence,
    commit_action, evaluate_anchor_response, evaluate_executable_responses,
)
from rl_leader.response_dqn_mask import FIVE_CELL_CONTINUATION_EQUIVALENCE
from src.simulation.simulator import MixedTrafficSimulator
from src.controllers.rl_stackelberg import configure_pstack_b13_follower_contract
from src.tests.test_response_continuation_integration import TrialEnv, identity


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "results/response_dqn_170_incident/sequential_cql_reference_v1/with_references/evaluation/checkpoint.pkl"
FIXTURE_SHA256 = "87cd6f7a676be532293464296ea6f4c9319998039484597fe73988be27a14f2a"
ORIGINAL_IDENTITY_JSON_SHA256 = "b7d5de507ec0a39ae8a16e3a6b60e5aa4090af51a4dd7c8082c6ff0ba78ae2d2"


def make_env(name):
    # The real constructor calls reset, which advances five warmup intervals.
    # Apply only its canonical config transformation; do not create/step a plant.
    with patch.object(RLLeaderEnv, "reset",
                      new=lambda env: configure_pstack_b13_follower_contract(env.cfg)):
        return RLLeaderEnv(scenario_name=name, T_total=14400., action_mode="full",
                           mask="RL-FULL", pstack_anchor=True,
                           action_parameterization="pstack_residual")


class FiveCellContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for owner, method in ((RLLeaderEnv, "reset"), (RLLeaderEnv, "step"),
                              (MixedTrafficSimulator, "step")):
            guard = patch.object(owner, method, side_effect=AssertionError("simulation forbidden"))
            guard.start()
            cls.addClassCleanup(guard.stop)
        cls.environments = {name: make_env(name) for name in FIVE_CELL_CONTRACT_SHA256}

    def test_exact_five_contracts_share_every_non_scenario_field(self):
        self.assertEqual(set(FIVE_CELL_CONTRACT_SHA256), {
            "sweet_155_w60", "sweet_170_w60", "sweet_170_incident_w60",
            "sweet_170_skew15_w60", "sweet_190_w60",
        })
        shared = []
        for name, env in self.environments.items():
            with self.subTest(name=name):
                expected = FIVE_CELL_CONTRACT_SHA256[name]
                metadata = validate_five_cell_contract(env, expected_contract_sha256=expected)
                self.assertEqual(env.experiment_contract_fingerprint, expected)
                self.assertEqual(metadata["scenario"], name)
                self.assertEqual(metadata["experiment_contract_sha256"], expected)
                fields = {k: v for k, v in env._experiment_contract.payload.items()
                          if k not in {"scenario", "scenario_name"}}
                self.assertEqual(_sha(fields), FIVE_CELL_SHARED_CONTRACT_SHA256)
                shared.append(fields)
        self.assertTrue(all(value == shared[0] for value in shared))

    def test_allowlist_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            FIVE_CELL_CONTRACT_SHA256["sweet_200_w60"] = "0" * 64

    def test_unknown_scenario_and_wrong_or_missing_pin_reject(self):
        env = copy.deepcopy(self.environments["sweet_155_w60"])
        for pin in (None, "", AUDITED_CONTRACT_SHA256, "0" * 64):
            with self.subTest(pin=pin), self.assertRaisesRegex(ValueError, "expected contract"):
                validate_five_cell_contract(env, expected_contract_sha256=pin)
        env.scenario_name = "sweet_155_unreviewed"
        with self.assertRaisesRegex(ValueError, "unsupported scenario"):
            validate_five_cell_contract(env, expected_contract_sha256=FIVE_CELL_CONTRACT_SHA256["sweet_155_w60"])

    def test_config_and_scenario_mutation_reject_even_with_new_hash(self):
        expected = FIVE_CELL_CONTRACT_SHA256["sweet_155_w60"]
        env = copy.deepcopy(self.environments["sweet_155_w60"])
        env.cfg.mpc.leader_gradseed_enabled = not env.cfg.mpc.leader_gradseed_enabled
        with self.assertRaisesRegex(ValueError, "runtime config"):
            validate_five_cell_contract(env, expected_contract_sha256=expected)
        env = copy.deepcopy(self.environments["sweet_155_w60"])
        payload = env._experiment_contract.payload
        payload["scenario"]["urban_scale"] += .01
        env._experiment_contract = ExperimentContract.from_payload(payload)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_five_cell_contract(env, expected_contract_sha256=expected)
        with self.assertRaisesRegex(ValueError, "expected contract"):
            validate_five_cell_contract(env, expected_contract_sha256=env._experiment_contract.sha256)

    def test_opt_in_mode_requires_explicit_expected_contract(self):
        env = copy.deepcopy(self.environments["sweet_155_w60"])
        env.response_equivalence_mode = FIVE_CELL_IDENTITY_VERSION
        with self.assertRaisesRegex(ValueError, "expected contract"):
            _equivalence_mode(env)
        env.response_expected_contract_sha256 = FIVE_CELL_CONTRACT_SHA256[env.scenario_name]
        self.assertEqual(_equivalence_mode(env), FIVE_CELL_IDENTITY_VERSION)
        with self.assertRaisesRegex(ValueError, "explicit executable previews"):
            evaluate_anchor_response(env, np.zeros(238), None)


class FiveCellCommittedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FIXTURE.is_file():
            raise unittest.SkipTest("read-only committed checkpoint fixture unavailable")
        raw = FIXTURE.read_bytes()
        if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
            raise unittest.SkipTest("checkpoint differs from the pre-extension byte-parity fixture")
        cls.fixture = pickle.loads(raw)["env"]
        cls.fixture._active_controller = cls.fixture.controller
        for owner, method in ((RLLeaderEnv, "reset"), (RLLeaderEnv, "step"),
                              (RLLeaderEnv, "_forecast"), (MixedTrafficSimulator, "step")):
            guard = patch.object(owner, method, side_effect=AssertionError("simulation forbidden"))
            guard.start()
            cls.addClassCleanup(guard.stop)

    def test_default_identity_is_byte_identical_to_original_7a8f1a8(self):
        identity_value = continuation_identity(self.fixture, interval_reward=-1., terminal=True)
        serialized = json.dumps(identity_value, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(serialized).hexdigest(), ORIGINAL_IDENTITY_JSON_SHA256)
        self.assertNotIn("contract_scope", identity_value)

    def test_exact_projection_accepts_all_five_contexts_with_metadata(self):
        # Replace only context in a typed committed fixture; no trajectory claim.
        for name, expected in FIVE_CELL_CONTRACT_SHA256.items():
            with self.subTest(name=name):
                canonical = make_env(name)
                env = copy.deepcopy(self.fixture)
                env._experiment_contract = canonical._experiment_contract
                env.scenario_name, env.scenario = canonical.scenario_name, canonical.scenario
                env.profile.scenario = env.scenario
                env.response_equivalence_mode = FIVE_CELL_IDENTITY_VERSION
                env.response_expected_contract_sha256 = expected
                before = pickle.dumps(env, protocol=5)
                value = five_cell_continuation_identity(
                    env, interval_reward=-1., terminal=True, expected_contract_sha256=expected)
                self.assertEqual(value["version"], FIVE_CELL_IDENTITY_VERSION)
                self.assertEqual(value["contract_scope"]["scenario"], name)
                self.assertEqual(value["contract_scope"]["expected_contract_sha256"], expected)
                self.assertEqual(pickle.dumps(env, protocol=5), before)
                if name != "sweet_170_incident_w60":
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        continuation_identity(env, interval_reward=-1., terminal=True)

    def test_new_identity_keeps_strict_runtime_projection(self):
        env = copy.deepcopy(self.fixture)
        env.optimizer_controller.price_iter_max = 2
        with self.assertRaisesRegex(ValueError, "price_iter_max"):
            five_cell_continuation_identity(env, interval_reward=-1., terminal=True,
                                            expected_contract_sha256=AUDITED_CONTRACT_SHA256)
        env = copy.deepcopy(self.fixture)
        env.controller.nash_solver.future_causal_field = object()
        with self.assertRaisesRegex(TypeError, "future_causal_field"):
            five_cell_continuation_identity(env, interval_reward=-1., terminal=True,
                                            expected_contract_sha256=AUDITED_CONTRACT_SHA256)


class FiveCellCollectorTests(unittest.TestCase):
    def test_new_mode_uses_exact_preview_commit_validation(self):
        env = TrialEnv()
        env.response_equivalence_mode = FIVE_CELL_CONTINUATION_EQUIVALENCE
        env.response_expected_contract_sha256 = AUDITED_CONTRACT_SHA256
        catalog = SimpleNamespace(size=2, actions=[SimpleNamespace(action_id=i) for i in range(2)],
                                  residual=lambda i: np.array([i]))
        with patch("rl_leader.response_continuation_state.validate_five_cell_contract", return_value={}), \
             patch("rl_leader.response_dqn_collect._continuation_identity", side_effect=identity):
            evaluated = evaluate_executable_responses(env, np.array([0]), catalog)
            self.assertEqual(env.step, 0)
            self.assertEqual(evaluated.response_equivalence_mode, FIVE_CELL_CONTINUATION_EQUIVALENCE)
            commit_action(env, evaluated, catalog, 0)
            self.assertEqual(env.step, 1)
            with self.assertRaisesRegex(ValueError, "preview.*commit"):
                commit_action(env, evaluated, catalog, 0)

    def test_old_model_cannot_silently_use_new_identity(self):
        evaluated = SimpleNamespace(response_equivalence_mode=FIVE_CELL_CONTINUATION_EQUIVALENCE)
        for mode in ("legacy_follower_runtime_v1", "post_commit_continuation_v1"):
            with self.assertRaisesRegex(ValueError, "equivalence"):
                _validate_policy_equivalence([SimpleNamespace(response_equivalence_mode=mode)], evaluated)
        _validate_policy_equivalence([
            SimpleNamespace(response_equivalence_mode=FIVE_CELL_CONTINUATION_EQUIVALENCE)], evaluated)

    def test_model_serialization_and_inference_preserve_new_mode_without_training(self):
        from rl_leader.response_dqn import load_trained_response_dqn
        model_path = ROOT / "results/response_dqn_170_incident/sequential_multistep_v1/one_step/model/response_dqn_member_00.pt"
        replay_path = ROOT / "results/response_dqn_170_incident/sequential_multistep_v1/training_replay.npz"
        if not model_path.is_file() or not replay_path.is_file():
            self.skipTest("read-only model/replay fixture unavailable")
        from rl_leader.response_dqn_data import load_frozen_response_replay
        model = load_trained_response_dqn(model_path)
        replay = load_frozen_response_replay(replay_path)
        before = model.q_values(replay.observation[:1], replay.response_features[:1])
        # A temporary serialization fixture only; this does not promote the old model.
        model.response_equivalence_mode = FIVE_CELL_CONTINUATION_EQUIVALENCE
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serialization_fixture.pt"
            model.save(path)
            restored = load_trained_response_dqn(path)
            self.assertEqual(restored.response_equivalence_mode, FIVE_CELL_CONTINUATION_EQUIVALENCE)
            _validate_policy_equivalence([restored], SimpleNamespace(
                response_equivalence_mode=FIVE_CELL_CONTINUATION_EQUIVALENCE))
            np.testing.assert_array_equal(
                restored.q_values(replay.observation[:1], replay.response_features[:1]), before)


if __name__ == "__main__":
    unittest.main()
