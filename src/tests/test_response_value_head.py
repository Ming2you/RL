"""Raw-phase finite-horizon costs are opt-in and preserve sequential recovery."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from rl_leader import train_response_dqn
from rl_leader.response_dqn import (
    CandidateQNetwork, FiniteHorizonCostContract, ResponseDQNConfig,
    _normalized_tensors, load_trained_response_dqn, train_response_dqn_member,
)
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest


COST = "finite_horizon_cost_v1"


def phase_replay(horizon=80, start=5):
    steps = np.arange(start, horizon)
    n = steps.size
    observation = np.column_stack((np.ones(n), steps / horizon)).astype(np.float32)
    successor = np.column_stack((np.ones(n), (steps + 1) / horizon)).astype(np.float32)
    done = (steps + 1 == horizon).astype(np.float32)
    mask = np.ones((n, 2), bool)
    next_mask = mask.copy()
    next_mask[done == 1, 1] = False
    manifest = make_replay_manifest(
        transition_count=n, action_count=2, catalog_fingerprint="phase-toy",
        observation_schema={"dimension": 2, "names": ["constant", "time.phase"]},
        response_contract="toy", scenario="toy", source="unit-test",
    )
    manifest.update(reward_semantics="interval_negative_ttt", done_semantics="environment_terminal")
    return FrozenResponseReplay(
        observation=observation, next_observation=successor,
        action_id=np.arange(n, dtype=np.int64) % 2, reward=-np.ones(n, np.float32),
        done=done, option_steps=np.ones(n, np.int64), action_mask=mask, next_action_mask=next_mask,
        response_features=np.zeros((n, 2, 1), np.float32),
        next_response_features=np.zeros((n, 2, 1), np.float32),
        event_group=np.full(n, "episode"), episode=np.zeros(n, np.int64),
        control_step=np.arange(n, dtype=np.int64), manifest=manifest,
    ).validate()


def recovery_replay():
    # Root action 1 costs more now; only its later recovery makes it preferable.
    replay = phase_replay(horizon=6, start=0)
    observation = np.array([
        [1, 0, 0, 0], [1, 0, 0, 0],
        [0, .5, 1, 0], [0, .5, 1, 0],
        [0, .5, 0, 1], [0, .5, 0, 1],
    ], np.float32)
    successor = np.array([
        [0, .5, 1, 0], [0, .5, 0, 1],
        [0, 1, 0, 0], [0, 1, 0, 0],
        [0, 1, 0, 0], [0, 1, 0, 0],
    ], np.float32)
    done = np.array([0, 0, 1, 1, 1, 1], np.float32)
    next_mask = replay.next_action_mask.copy()
    next_mask[done == 1, 1] = False
    return replace(
        replay, observation=observation, next_observation=successor,
        reward=np.array([-1, -2, -5, -5, -9, -1], np.float32), done=done,
        next_action_mask=next_mask,
        manifest={**replay.manifest, "observation_schema": {
            "dimension": 4, "names": ["root", "time.phase", "bad", "recovery"],
        }},
    ).validate()


def train_tiny(replay, **overrides):
    options = dict(gamma=1, reward_scale=1, batch_size=replay.size, gradient_steps=3,
                   target_update_interval=2, hidden=(8,), ensemble_size=1,
                   mask_constant_features=True)
    options.update(overrides)
    return train_response_dqn_member(
        replay, np.eye(2, dtype=np.float32), catalog_fingerprint="phase-toy",
        config=ResponseDQNConfig(**options), seed=17, bootstrap=False,
    )


class ValueHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_response_dqn._configure_torch_threads(1)

    def test_default_free_q_training_and_old_checkpoint_are_bit_exact(self):
        replay = recovery_replay()
        default = train_tiny(replay)
        explicit = train_tiny(replay, value_parameterization="free_q")
        self.assertEqual(default.training_losses, explicit.training_losses)
        for network in ("online", "target"):
            for key, value in getattr(default, network).state_dict().items():
                torch.testing.assert_close(value, getattr(explicit, network).state_dict()[key], rtol=0, atol=0)
        checkpoint = default.checkpoint()
        self.assertNotIn("value_contract", checkpoint)
        checkpoint["config"].pop("value_parameterization")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(checkpoint, path)
            restored = load_trained_response_dqn(path)
        self.assertEqual(restored.config.value_parameterization, "free_q")
        self.assertIsNone(restored.value_contract)
        tensors = _normalized_tensors(replay, default.normalizer, torch.device("cpu"))
        with torch.no_grad():
            legacy = default.online(tensors["observation"], torch.eye(2), tensors["response"]).numpy()
        np.testing.assert_array_equal(legacy, default.q_values(replay.observation, replay.response_features))
        np.testing.assert_array_equal(legacy, restored.q_values(replay.observation, replay.response_features))
        np.testing.assert_array_equal(legacy, explicit.q_values(replay.observation, replay.response_features))

    def test_derives_full_80_interval_horizon_not_75_decisions(self):
        replay = phase_replay()
        contract = FiniteHorizonCostContract.from_replay(replay)
        self.assertEqual((contract.phase_index, contract.horizon_intervals), (1, 80))
        np.testing.assert_array_equal(contract.remaining_intervals(replay.observation), np.arange(75, 0, -1))
        np.testing.assert_array_equal(contract.remaining_intervals(replay.next_observation), np.arange(74, -1, -1))

    def test_last_decision_is_one_interval_and_only_end_has_zero_factor(self):
        replay = phase_replay(start=78)
        model = train_tiny(replay, value_parameterization=COST, gradient_steps=0)
        with torch.no_grad():
            for parameter in model.online.parameters():
                parameter.zero_()
        # Even a masked/shifted normalized phase cannot change the raw-phase head.
        model.normalizer = replace(model.normalizer, observation_mean=np.array([500, 10], np.float32),
                                   observation_active_mask=np.zeros(2, bool))
        raw = np.concatenate((replay.observation, replay.next_observation[-1:]))
        response = np.zeros((3, 2, 1), np.float32)
        q = model.q_values(raw, response)
        expected = -np.array([2, 1, 0], np.float32)[:, None] * F.softplus(torch.tensor(0.)).item()
        np.testing.assert_array_equal(q, np.repeat(expected, 2, axis=1))
        self.assertTrue((q[1] < 0).all())
        np.testing.assert_array_equal(model.q_values(raw[-1], response[-1]), np.zeros((1, 2), np.float32))

    def test_extreme_ood_features_and_logits_cannot_make_candidate_q_positive(self):
        replay = phase_replay(start=78)
        model = train_tiny(replay, value_parameterization=COST, gradient_steps=0, mask_constant_features=False)
        raw = np.repeat(replay.observation[-1:], 3, axis=0)
        raw[:, 0] = [-1.0e6, 0, 1.0e6]
        response = np.array([[[-1.0e6], [1.0e6]], [[1.0e6], [-1.0e6]], [[0], [1.0e6]]], np.float32)
        for bias in (-1.0e6, 0, 1.0e6):
            with self.subTest(bias=bias), torch.no_grad():
                model.online.network[-1].bias.fill_(bias)
                q = model.q_values(raw, response)
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue((q <= 0).all())
                end = raw.copy()
                end[:, 1] = 1
                np.testing.assert_array_equal(model.q_values(end, response), np.zeros_like(q))

    def test_training_uses_head_for_all_three_forwards_and_bounds_td_targets(self):
        replay = recovery_replay()
        original_forward = CandidateQNetwork.forward
        original_loss = F.smooth_l1_loss
        seen = []
        targets = []

        def forward(network, observation, candidate, response, **kwargs):
            remaining = kwargs["remaining_intervals"]
            self.assertIsNotNone(remaining)
            q = original_forward(network, observation, candidate, response, **kwargs)
            self.assertTrue((q <= 0).all())
            seen.append(remaining.detach().numpy().copy())
            return q

        def loss(selected, target, *args, **kwargs):
            targets.append(target.detach().numpy().copy())
            return original_loss(selected, target, *args, **kwargs)

        with patch.object(CandidateQNetwork, "forward", autospec=True, side_effect=forward), \
             patch("rl_leader.response_dqn.sample_training_indices", return_value=np.arange(replay.size)), \
             patch("rl_leader.response_dqn.F.smooth_l1_loss", side_effect=loss):
            model = train_tiny(replay, value_parameterization=COST, conservative_alpha=.1)
        for index in range(0, len(seen), 3):
            np.testing.assert_array_equal(seen[index], [2, 2, 1, 1, 1, 1])
            np.testing.assert_array_equal(seen[index + 1], [1, 1, 0, 0, 0, 0])
            np.testing.assert_array_equal(seen[index + 2], [1, 1, 0, 0, 0, 0])
        self.assertEqual(len(seen), 9)
        for target in targets:
            self.assertTrue((target <= replay.reward).all())
            np.testing.assert_array_equal(target[replay.done == 1], replay.reward[replay.done == 1])
        tensors = _normalized_tensors(replay, model.normalizer, torch.device("cpu"), model.value_contract)
        with torch.no_grad():
            current = model.online(tensors["observation"], torch.eye(2), tensors["response"],
                                   remaining_intervals=tensors["remaining_intervals"]).numpy()
            successor = model.online(tensors["next_observation"], torch.eye(2), tensors["next_response"],
                                     remaining_intervals=tensors["next_remaining_intervals"]).numpy()
        np.testing.assert_array_equal(current, model.q_values(replay.observation, replay.response_features))
        np.testing.assert_array_equal(successor, model.q_values(replay.next_observation, replay.next_response_features))

    def test_prediction_itself_is_not_bounded_by_immediate_reward(self):
        replay = recovery_replay()
        model = train_tiny(replay, value_parameterization=COST, gradient_steps=0)
        with torch.no_grad():
            for parameter in model.online.parameters():
                parameter.zero_()
        q = model.q_values(replay.observation, replay.response_features)
        self.assertGreater(q[4, 0], replay.reward[4])
        self.assertLess(q[4, 0], 0)

    def test_rejects_wrong_gamma_mode_reward_semantics_and_nonnegative_rewards(self):
        replay = recovery_replay()
        for options in ({"gamma": .99, "value_parameterization": COST}, {"value_parameterization": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                train_tiny(replay, gradient_steps=0, **options)
        for field, value in (("reward_semantics", "terminal_return"), ("done_semantics", "truncated")):
            bad = replace(replay, manifest={**replay.manifest, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                train_tiny(bad, value_parameterization=COST, gradient_steps=0)
        for value in (0., 1., float("nan"), float("inf"), -float("inf")):
            reward = replay.reward.copy()
            reward[0] = value
            with self.subTest(reward=value), self.assertRaises(ValueError):
                train_tiny(replace(replay, reward=reward), value_parameterization=COST, gradient_steps=0)

    def test_rejects_missing_phase_bad_grid_increments_and_done(self):
        replay = phase_replay()
        for names in ([], ["constant", "other"], ["time.phase", "time.phase"]):
            bad = replace(replay, manifest={**replay.manifest, "observation_schema": {"names": names}})
            with self.subTest(names=names), self.assertRaises(ValueError):
                FiniteHorizonCostContract.from_replay(bad)
        for field, row, value in (("observation", 0, -.1), ("next_observation", -1, 1.1),
                                  ("observation", 0, .0626), ("next_observation", 0, .0875),
                                  ("next_observation", 0, .0625), ("observation", 0, float("nan"))):
            values = getattr(replay, field).copy()
            values[row, 1] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                FiniteHorizonCostContract.from_replay(replace(replay, **{field: values}))
        for row in (0, -1):
            done = replay.done.copy()
            done[row] = 1 - done[row]
            with self.subTest(done_row=row), self.assertRaisesRegex(ValueError, "done"):
                FiniteHorizonCostContract.from_replay(replace(replay, done=done))
        with self.assertRaisesRegex(ValueError, "one-step"):
            FiniteHorizonCostContract.from_replay(replace(replay, option_steps=replay.option_steps * 2))
        bad = phase_replay(horizon=3, start=0)
        obs, nxt = bad.observation.copy(), bad.next_observation.copy()
        obs[:, 1], nxt[:, 1] = [0, .3, .6], [.3, .6, .9]
        with self.assertRaises(ValueError):
            FiniteHorizonCostContract.from_replay(replace(bad, observation=obs, next_observation=nxt))

    def test_inference_rejects_bad_phase_and_cost_network_requires_remaining(self):
        replay = phase_replay(start=78)
        model = train_tiny(replay, value_parameterization=COST, gradient_steps=0)
        for value in (-.1, 1.1, .1234, float("nan")):
            obs = replay.observation.copy()
            obs[0, 1] = value
            with self.subTest(phase=value), self.assertRaises(ValueError):
                model.q_values(obs, replay.response_features)
        with self.assertRaisesRegex(ValueError, "remaining"):
            model.online(torch.zeros(2, 2), torch.eye(2), torch.zeros(2, 2, 1))

    def test_checkpoint_roundtrip_and_contract_validation(self):
        replay = recovery_replay()
        model = train_tiny(replay, value_parameterization=COST)
        contract = model.checkpoint()["value_contract"]
        self.assertEqual(contract["horizon_intervals"], 2)
        self.assertTrue(contract["remaining_includes_current_interval"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.pt"
            model.save(path)
            restored = load_trained_response_dqn(path)
            self.assertEqual(restored.value_contract, model.value_contract)
            for obs, response in ((replay.observation, replay.response_features),
                                  (replay.next_observation, replay.next_response_features)):
                np.testing.assert_array_equal(model.q_values(obs, response), restored.q_values(obs, response))
            for change in (None, {"phase_index": 99}, {"horizon_intervals": 0},
                           {"remaining_includes_current_interval": False}, {"gamma": .99}):
                payload = model.checkpoint()
                payload["value_contract"] = None if change is None else {**contract, **change}
                torch.save(payload, path)
                with self.subTest(contract=change), self.assertRaises(ValueError):
                    load_trained_response_dqn(path)
            payload = model.checkpoint()
            payload["config"]["value_parameterization"] = "free_q"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_trained_response_dqn(path)

    def test_later_recovery_remains_learnable(self):
        replay = recovery_replay()
        model = train_tiny(replay, value_parameterization=COST, gradient_steps=500,
                           hidden=(16, 16), learning_rate=.01, target_update_interval=10)
        q = model.q_values(replay.observation, replay.response_features)
        self.assertTrue(np.isfinite(model.training_losses).all())
        self.assertGreater(q[0, 1], q[0, 0] + 1)
        self.assertGreater(q[4, 1], q[4, 0] + 5)
        self.assertAlmostEqual(float(q[0, 1]), -3, delta=.6)
        self.assertAlmostEqual(float(q[0, 0]), -6, delta=.6)

    def test_cli_value_choice_reaches_worker_and_manifest(self):
        replay = recovery_replay()
        replay = replace(replay, manifest={**replay.manifest, "catalog": {}})
        for mode in ("free_q", COST):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                argv = ["train_response_dqn", "--data", "unused.npz", "--output-dir", directory,
                        "--ensemble-size", "1", "--gamma", "1"]
                if mode != "free_q":
                    argv.extend(("--value-parameterization", mode))
                with patch("sys.argv", argv), \
                     patch.object(train_response_dqn, "load_frozen_response_replay", return_value=replay), \
                     patch.object(train_response_dqn.StructuredActionCatalog, "from_manifest") as catalog, \
                     patch.object(train_response_dqn, "_train_member_worker") as worker, \
                     contextlib.redirect_stdout(io.StringIO()):
                    catalog.return_value.fingerprint = "phase-toy"
                    worker.return_value = {"member": 0, "checkpoint": "mock.pt"}
                    train_response_dqn.main()
                self.assertEqual(worker.call_args.args[0]["config"]["value_parameterization"], mode)
                manifest = json.loads((Path(directory) / "ensemble_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["config"]["value_parameterization"], mode)


if __name__ == "__main__":
    unittest.main()
