"""Terminal replay weighting must retain true sequential recovery targets."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_leader.response_dqn import (
    ResponseDQNConfig, load_trained_response_dqn, sample_training_indices,
    train_response_dqn_member,
)
from src.tests.test_sequential_response_learning import recovery_replay


class TerminalSamplingTest(unittest.TestCase):
    def test_zero_fraction_preserves_original_rng_path(self):
        indices = np.array([0, 1, 1, 2, 3])
        left, right = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(5):
            actual = sample_training_indices(left, indices, np.array([0, 0, 0, 1]), 4, 0)
            np.testing.assert_array_equal(actual, right.choice(indices, size=4, replace=False))

    def test_fraction_and_available_training_members_are_respected(self):
        indices = np.arange(20)
        done = np.zeros(21)
        done[19:] = 1
        batch = sample_training_indices(np.random.default_rng(4), indices, done, 16, .25)
        self.assertEqual(len(batch), 16)
        self.assertEqual(int(done[batch].sum()), 4)
        self.assertNotIn(20, batch)
        self.assertEqual(len(set(batch[done[batch] == 0])), 12)

    def test_mixed_sampling_requires_both_strata_and_two_slots(self):
        for done, size in ((np.zeros(3), 3), (np.ones(3), 3), (np.array([0, 0, 1]), 1)):
            with self.assertRaises(ValueError):
                sample_training_indices(np.random.default_rng(3), np.arange(3), done, size, .25)

    def test_fraction_validation(self):
        for fraction in (-.1, 1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ResponseDQNConfig(terminal_batch_fraction=fraction).validate()

    def test_legacy_checkpoint_defaults_to_uniform(self):
        torch.set_num_threads(1)
        replay = recovery_replay()
        model = train_response_dqn_member(
            replay, np.eye(2, dtype=np.float32), catalog_fingerprint='recovery-toy',
            config=ResponseDQNConfig(gradient_steps=1, hidden=(8,)), bootstrap=False,
        )
        payload = model.checkpoint()
        payload['config'].pop('terminal_batch_fraction')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save(payload, path)
            restored = load_trained_response_dqn(path)
        self.assertEqual(restored.config.terminal_batch_fraction, 0)
        np.testing.assert_array_equal(
            model.q_values(replay.observation, replay.response_features),
            restored.q_values(replay.observation, replay.response_features),
        )

    def test_terminal_weighting_does_not_remove_later_recovery(self):
        torch.set_num_threads(1)
        replay = recovery_replay()
        config = ResponseDQNConfig(
            gamma=1, reward_scale=1, learning_rate=.003, gradient_steps=1600,
            target_update_interval=20, hidden=(32, 32), batch_size=4,
            ensemble_size=1, conservative_alpha=.1, terminal_batch_fraction=.5,
        )
        model = train_response_dqn_member(
            replay, np.eye(2, dtype=np.float32), catalog_fingerprint='recovery-toy',
            config=config, seed=17, bootstrap=False,
        )
        q = model.q_values(replay.observation[0], replay.response_features[0])[0]
        self.assertGreater(q[1], q[0] + 5)
        self.assertAlmostEqual(float(q[1]), -2, delta=2)


if __name__ == '__main__':
    unittest.main()
