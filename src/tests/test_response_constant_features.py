"""Constant masking and operational STOP preserve default training behavior."""
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

from rl_leader import train_response_dqn
from rl_leader.response_dqn import (
    FeatureNormalizer, ResponseDQNConfig, _normalized_tensors,
    load_trained_response_dqn, sample_training_indices, train_response_dqn_member,
)
from src.tests.test_sequential_response_learning import recovery_replay


def constant_replay():
    replay = recovery_replay()
    extra = np.tile(np.array([7, 0, 1, 0], np.float32), (replay.size, 1))
    next_extra = extra.copy()
    extra[0, 2] = np.nextafter(np.float32(1), np.float32(2))
    next_extra[1, 3] = 0.25
    response = np.zeros((replay.size, replay.action_count, 5), np.float32)
    response[:, :, 1:3] = [7, 1]
    response[:, 1, 3] = np.arange(1, replay.size + 1)
    next_response = response.copy()
    response[0, 0, 2] = np.nextafter(np.float32(1), np.float32(2))
    next_response[1, 0, 4] = 0.25
    return replace(
        replay,
        observation=np.concatenate((replay.observation, extra), axis=1),
        next_observation=np.concatenate((replay.next_observation, next_extra), axis=1),
        response_features=response, next_response_features=next_response,
    ).validate()


def train_tiny(replay, *, seed=17, stop_files=(), **options):
    config = ResponseDQNConfig(
        gamma=1, reward_scale=1, batch_size=4, gradient_steps=4,
        target_update_interval=2, hidden=(8,), ensemble_size=1,
        conservative_alpha=0.1, terminal_batch_fraction=0.25, **options,
    )
    return train_response_dqn_member(
        replay, np.eye(2, dtype=np.float32), catalog_fingerprint="recovery-toy",
        config=config, seed=seed, bootstrap=False, stop_files=stop_files,
    )


class ConstantFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_response_dqn._configure_torch_threads(1)

    def test_default_normalizer_remains_legacy(self):
        replay = constant_replay()
        normalizer = FeatureNormalizer.fit(replay, np.arange(replay.size))
        self.assertFalse(ResponseDQNConfig().mask_constant_features)
        self.assertIsNone(normalizer.observation_active_mask)
        self.assertIsNone(normalizer.response_active_mask)
        self.assertEqual(set(normalizer.as_dict()), {
            "observation_mean", "observation_scale", "response_mean", "response_scale",
        })
        tensors = _normalized_tensors(replay, normalizer, torch.device("cpu"))
        for key, values, mean, scale in (
            ("observation", replay.observation, normalizer.observation_mean, normalizer.observation_scale),
            ("next_observation", replay.next_observation, normalizer.observation_mean, normalizer.observation_scale),
            ("response", replay.response_features, normalizer.response_mean, normalizer.response_scale),
            ("next_response", replay.next_response_features, normalizer.response_mean, normalizer.response_scale),
        ):
            np.testing.assert_array_equal(tensors[key].numpy(), (values - mean) / scale)

    def test_exact_variance_uses_current_and_successor_valid_candidates(self):
        replay = constant_replay()
        normalizer = FeatureNormalizer.fit(replay, np.arange(replay.size), mask_constant_features=True)
        np.testing.assert_array_equal(normalizer.observation_active_mask, [1, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(normalizer.response_active_mask, [0, 0, 1, 1, 1])
        # Positive variance below the legacy scale cutoff must remain active.
        self.assertEqual(normalizer.observation_scale[5], 1)
        self.assertEqual(normalizer.response_scale[2], 1)
        only_anchor = np.tile([True, False], (replay.size, 1))
        replay = replace(replay, action_mask=only_anchor, next_action_mask=only_anchor,
                         action_id=np.zeros(replay.size, np.int64)).validate()
        normalizer = FeatureNormalizer.fit(replay, np.arange(replay.size), mask_constant_features=True)
        np.testing.assert_array_equal(normalizer.response_active_mask, [0, 0, 1, 0, 1])

    def test_refit_reactivates_new_variation_and_respects_fit_indices(self):
        replay = constant_replay()
        observation = replay.observation.copy()
        response = replay.response_features.copy()
        observation[-1, 4] = 1.0e-12
        response[-1, 0, 0] = 1.0e-12
        augmented = replace(replay, observation=observation, response_features=response)
        partial = FeatureNormalizer.fit(augmented, np.arange(replay.size - 1), mask_constant_features=True)
        refit = FeatureNormalizer.fit(augmented, np.arange(replay.size), mask_constant_features=True)
        self.assertFalse(partial.observation_active_mask[4])
        self.assertFalse(partial.response_active_mask[0])
        self.assertTrue(refit.observation_active_mask[4])
        self.assertTrue(refit.response_active_mask[0])
        self.assertFalse(refit.observation_active_mask[3])
        self.assertFalse(refit.response_active_mask[1])

    def test_matched_seed_training_and_in_support_predictions_are_identical(self):
        replay = constant_replay()
        for seed in (17, 18):
            with self.subTest(seed=seed):
                legacy = train_tiny(replay, seed=seed)
                masked = train_tiny(replay, seed=seed, mask_constant_features=True)
                self.assertEqual(legacy.training_losses, masked.training_losses)
                for network in ("online", "target"):
                    for key, value in getattr(legacy, network).state_dict().items():
                        torch.testing.assert_close(value, getattr(masked, network).state_dict()[key], rtol=0, atol=0)
                for obs, response in ((replay.observation, replay.response_features),
                                      (replay.next_observation, replay.next_response_features)):
                    np.testing.assert_array_equal(legacy.q_values(obs, response), masked.q_values(obs, response))

    def test_masked_perturbations_are_neutral_and_active_values_are_not_clipped(self):
        replay = constant_replay()
        model = train_tiny(replay, mask_constant_features=True)
        norm = model.normalizer
        rng = np.random.default_rng(19)
        changed = {}
        for key in ("observation", "next_observation", "response_features", "next_response_features"):
            values = getattr(replay, key).copy()
            active = norm.observation_active_mask if "observation" in key else norm.response_active_mask
            values[..., ~active] = rng.uniform(-1.0e6, 1.0e6, values[..., ~active].shape)
            changed[key] = values
        perturbed = replace(replay, **changed)
        expected = _normalized_tensors(replay, norm, torch.device("cpu"))
        actual = _normalized_tensors(perturbed, norm, torch.device("cpu"))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        np.testing.assert_array_equal(
            model.q_values(replay.observation, replay.response_features),
            model.q_values(perturbed.observation, perturbed.response_features),
        )
        np.testing.assert_array_equal(
            model.q_values(replay.observation[0], replay.response_features[0]),
            model.q_values(perturbed.observation[0], perturbed.response_features[0]),
        )
        for kind, values in (("observation", replay.observation.copy()),
                             ("response", replay.response_features.copy())):
            active = getattr(norm, kind + "_active_mask")
            values[..., active] += 10000
            actual = getattr(norm, "normalize_" + kind)(values)
            raw = (values - getattr(norm, kind + "_mean")) / getattr(norm, kind + "_scale")
            np.testing.assert_array_equal(actual[..., active], raw[..., active])
            self.assertGreater(abs(actual[..., active]).max(), 100)
        obs = replay.observation.copy()
        obs[:, 0] += 10000
        self.assertFalse(np.array_equal(model.q_values(obs, replay.response_features),
                                        model.q_values(replay.observation, replay.response_features)))

    def test_masked_checkpoint_roundtrip(self):
        replay = constant_replay()
        model = train_tiny(replay, mask_constant_features=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "masked.pt"
            model.save(path)
            restored = load_trained_response_dqn(path)
        self.assertTrue(restored.config.mask_constant_features)
        for name in ("observation_active_mask", "response_active_mask"):
            saved = getattr(restored.normalizer, name)
            self.assertEqual(saved.dtype, np.dtype(bool))
            np.testing.assert_array_equal(saved, getattr(model.normalizer, name))
            self.assertIn(name, model.checkpoint()["normalizer"])
        obs = replay.observation.copy()
        response = replay.response_features.copy()
        obs[:, ~model.normalizer.observation_active_mask] = 10000
        response[:, :, ~model.normalizer.response_active_mask] = -10000
        np.testing.assert_array_equal(model.q_values(replay.observation, replay.response_features),
                                      restored.q_values(obs, response))

    def test_absent_masks_load_as_legacy_even_for_out_of_support_inputs(self):
        replay = constant_replay()
        model = train_tiny(replay)
        checkpoint = model.checkpoint()
        checkpoint["config"].pop("mask_constant_features")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(checkpoint, path)
            restored = load_trained_response_dqn(path)
        self.assertFalse(restored.config.mask_constant_features)
        self.assertIsNone(restored.normalizer.observation_active_mask)
        self.assertIsNone(restored.normalizer.response_active_mask)
        obs = replay.observation.copy()
        response = replay.response_features.copy()
        obs[:, 3:5] = 10000
        response[:, :, :2] = -10000
        np.testing.assert_array_equal(model.q_values(obs, response), restored.q_values(obs, response))
        self.assertFalse(np.array_equal(restored.q_values(obs, response),
                                        restored.q_values(replay.observation, replay.response_features)))

    def test_cli_opt_in_reaches_worker_config_and_manifest(self):
        replay = constant_replay()
        replay = replace(replay, manifest={**replay.manifest, "catalog": {}})
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                argv = ["train_response_dqn", "--data", "unused.npz", "--output-dir", directory,
                        "--ensemble-size", "1", "--workers", "1", "--member-threads", "1"]
                if enabled:
                    argv.append("--mask-constant-features")
                stop_files = tuple(str(Path(directory) / name) for name in ("STOP", "STOP.parent")) if enabled else ()
                for path in stop_files:
                    argv.extend(("--stop-file", path))
                with patch("sys.argv", argv):
                    self.assertEqual(train_response_dqn._parse_args().mask_constant_features, enabled)
                    with patch.object(train_response_dqn, "load_frozen_response_replay", return_value=replay), \
                         patch.object(train_response_dqn.StructuredActionCatalog, "from_manifest") as catalog, \
                         patch.object(train_response_dqn, "_train_member_worker") as worker, \
                         contextlib.redirect_stdout(io.StringIO()):
                        catalog.return_value.fingerprint = "recovery-toy"
                        worker.return_value = {"member": 0, "checkpoint": "mock.pt"}
                        train_response_dqn.main()
                self.assertEqual(worker.call_args.args[0]["config"]["mask_constant_features"], enabled)
                self.assertEqual(worker.call_args.args[0]["stop_files"], stop_files)
                self.assertNotIn("stop_files", worker.call_args.args[0]["config"])
                manifest = json.loads((Path(directory) / "ensemble_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["config"]["mask_constant_features"], enabled)
                self.assertNotIn("stop_files", manifest["config"])


class TrainingStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_response_dqn._configure_torch_threads(1)

    def test_preexisting_stop_prevents_expensive_setup(self):
        replay = constant_replay()
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            stop.touch()
            with patch("rl_leader.response_dqn.set_seed") as seed, \
                 patch.object(FeatureNormalizer, "fit") as fit:
                with self.assertRaisesRegex(InterruptedError, "STOP"):
                    train_tiny(replay, stop_files=(Path(directory) / "missing", stop))
            seed.assert_not_called()
            fit.assert_not_called()

    def test_stop_during_updates_is_checked_within_100_steps_and_before_return(self):
        replay = constant_replay()
        original_exists = Path.exists
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            for steps in (3, 101):
                with self.subTest(steps=steps), \
                     patch("rl_leader.response_dqn.sample_training_indices", wraps=sample_training_indices) as updates:
                    def exists(path):
                        return updates.call_count > 0 if path == stop else original_exists(path)

                    with patch.object(Path, "exists", autospec=True, side_effect=exists):
                        with self.assertRaisesRegex(InterruptedError, "STOP"):
                            train_response_dqn_member(
                                replay, np.eye(2, dtype=np.float32), catalog_fingerprint="recovery-toy",
                                config=ResponseDQNConfig(gradient_steps=steps, batch_size=4, hidden=(8,)),
                                bootstrap=False, stop_files=(stop,),
                            )
                    self.assertGreater(updates.call_count, 0)
                    self.assertLessEqual(updates.call_count, min(100, steps))

    def test_missing_stop_preserves_matched_seed_training_and_hyperparameters(self):
        replay = constant_replay()
        default = train_tiny(replay)
        with tempfile.TemporaryDirectory() as directory:
            checked = train_tiny(replay, stop_files=(Path(directory) / "STOP",))
        self.assertEqual(default.training_losses, checked.training_losses)
        self.assertEqual(default.checkpoint()["config"], checked.checkpoint()["config"])
        self.assertNotIn("stop_files", checked.checkpoint()["config"])
        for network in ("online", "target"):
            for key, value in getattr(default, network).state_dict().items():
                torch.testing.assert_close(value, getattr(checked, network).state_dict()[key], rtol=0, atol=0)
        np.testing.assert_array_equal(default.q_values(replay.observation, replay.response_features),
                                      checked.q_values(replay.observation, replay.response_features))

    def test_worker_propagates_stop_and_preserves_completed_member(self):
        replay = constant_replay()
        replay = replace(replay, manifest={**replay.manifest, "catalog": {}})
        with tempfile.TemporaryDirectory() as directory:
            completed = Path(directory) / "response_dqn_member_00.pt"
            completed.write_bytes(b"completed-member")
            stop_files = (str(Path(directory) / "STOP"), str(Path(directory) / "STOP.parent"))
            payload = {
                "data": "unused.npz", "output_dir": directory, "member": 1,
                "member_threads": 1, "seed": 17, "device": "cpu", "bootstrap": False,
                "config": {}, "stop_files": stop_files,
            }
            with patch.object(train_response_dqn, "load_frozen_response_replay", return_value=replay), \
                 patch.object(train_response_dqn.StructuredActionCatalog, "from_manifest"), \
                 patch.object(train_response_dqn, "train_response_dqn_member", side_effect=InterruptedError("STOP")) as member:
                with self.assertRaises(InterruptedError):
                    train_response_dqn._train_member_worker(payload)
            self.assertEqual(member.call_args.kwargs["stop_files"], stop_files)
            self.assertFalse(hasattr(member.call_args.kwargs["config"], "stop_files"))
            self.assertEqual(completed.read_bytes(), b"completed-member")
            self.assertFalse((Path(directory) / "response_dqn_member_01.pt").exists())


if __name__ == "__main__":
    unittest.main()
