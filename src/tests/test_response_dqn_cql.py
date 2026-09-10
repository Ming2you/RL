"""CQL regularization must preserve masking and the opt-out DDQN path."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rl_leader.response_dqn import (
    ResponseDQNConfig, load_trained_response_dqn, masked_cql_penalty,
    train_response_dqn_member,
)
from src.tests.test_sequential_response_learning import recovery_replay


class ConservativeQTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_candidates_have_no_loss_or_gradient(self):
        q = torch.tensor([[1.0, 3.0, 10000.0]], requires_grad=True)
        loss = masked_cql_penalty(q, torch.tensor([0]), torch.tensor([[True, True, False]]))
        self.assertAlmostEqual(loss.item(), np.logaddexp(1.0, 3.0) - 1.0, places=6)
        loss.backward()
        self.assertLess(q.grad[0, 0].item(), 0)
        self.assertGreater(q.grad[0, 1].item(), 0)
        self.assertEqual(q.grad[0, 2].item(), 0)

    def test_candidate_permutation_does_not_change_penalty(self):
        q = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float64)
        mask = torch.tensor([[True, True, True]])
        first = masked_cql_penalty(q, torch.tensor([1]), mask)
        second = masked_cql_penalty(q[:, [1, 2, 0]], torch.tensor([0]), mask)
        self.assertAlmostEqual(first.item(), second.item(), places=12)

    def test_shift_invariance_and_single_action(self):
        q = torch.tensor([[-10.0, -3.0]], dtype=torch.float64)
        mask = torch.ones_like(q, dtype=torch.bool)
        before = masked_cql_penalty(q, torch.tensor([1]), mask)
        after = masked_cql_penalty(q + 100, torch.tensor([1]), mask)
        self.assertAlmostEqual(before.item(), after.item(), places=12)
        self.assertEqual(masked_cql_penalty(q, torch.tensor([0]), torch.tensor([[True, False]])).item(), 0)

    def test_rejects_empty_or_invalid_behavior_mask(self):
        for mask in (torch.tensor([[False, False]]), torch.tensor([[False, True]])):
            with self.assertRaises(ValueError):
                masked_cql_penalty(torch.zeros(1, 2), torch.tensor([0]), mask)

    def test_alpha_must_be_finite_and_nonnegative(self):
        for alpha in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ResponseDQNConfig(conservative_alpha=alpha).validate()

    def test_zero_alpha_skips_penalty_and_old_checkpoint_loads(self):
        replay = recovery_replay()
        config = ResponseDQNConfig(gradient_steps=2, hidden=(8,), conservative_alpha=0.0)
        with patch("rl_leader.response_dqn.masked_cql_penalty", side_effect=AssertionError("must skip")):
            model = train_response_dqn_member(
                replay, np.eye(2, dtype=np.float32), catalog_fingerprint="recovery-toy",
                config=config, seed=17, bootstrap=False,
            )
        checkpoint = model.checkpoint()
        checkpoint["config"].pop("conservative_alpha")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(checkpoint, path)
            restored = load_trained_response_dqn(path)
        self.assertEqual(restored.config.conservative_alpha, 0.0)
        np.testing.assert_array_equal(
            model.q_values(replay.observation, replay.response_features),
            restored.q_values(replay.observation, replay.response_features),
        )

    def test_regularized_learner_still_learns_later_recovery(self):
        replay = recovery_replay()
        config = ResponseDQNConfig(
            gamma=1, reward_scale=1, learning_rate=0.003, gradient_steps=1600,
            target_update_interval=20, hidden=(32, 32), batch_size=4,
            ensemble_size=1, conservative_alpha=0.1,
        )
        model = train_response_dqn_member(
            replay, np.eye(2, dtype=np.float32), catalog_fingerprint="recovery-toy",
            config=config, seed=17, bootstrap=False,
        )
        root = model.q_values(replay.observation[0], replay.response_features[0])[0]
        self.assertTrue(np.isfinite(model.training_losses).all())
        self.assertGreater(root[1], root[0] + 5)
        self.assertAlmostEqual(float(root[1]), -2, delta=2)


if __name__ == "__main__":
    unittest.main()
