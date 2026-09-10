"""Regression tests for recovery through Bellman updates, not fixed labels."""
from dataclasses import replace
import unittest

import numpy as np
import torch

from rl_leader.response_dqn import ResponseDQNConfig, train_response_dqn_member
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from rl_leader.train_response_dqn import validate_sequential_td_replay


def recovery_replay():
    # Action 1 is only good when another action 1 can be selected at its successor.
    observation = np.eye(3, dtype=np.float32)[[0, 0, 1, 1]]
    next_observation = np.eye(3, dtype=np.float32)[[2, 1, 2, 2]]
    manifest = make_replay_manifest(
        transition_count=4, action_count=2, catalog_fingerprint="recovery-toy",
        observation_schema={"dimension": 3}, response_contract="toy",
        scenario="toy", source="test",
    )
    manifest.update(reward_semantics="interval_negative_ttt", done_semantics="environment_terminal")
    return FrozenResponseReplay(
        observation=observation, next_observation=next_observation,
        action_id=np.array([0, 1, 0, 1]), reward=np.array([-10, -1, -100, -1], np.float32),
        done=np.array([1, 0, 1, 1], np.float32), option_steps=np.ones(4, np.int64),
        action_mask=np.ones((4, 2), bool), next_action_mask=np.ones((4, 2), bool),
        response_features=np.zeros((4, 2, 1), np.float32),
        next_response_features=np.zeros((4, 2, 1), np.float32),
        event_group=np.array(["root", "root", "recovery", "recovery"]),
        episode=np.zeros(4, np.int64), control_step=np.array([0, 0, 1, 1]),
        manifest=manifest,
    ).validate()


class SequentialLearningTest(unittest.TestCase):
    def test_later_recovery_action_improves_earlier_action_value(self):
        torch.set_num_threads(1)
        replay = recovery_replay()
        model = train_response_dqn_member(
            replay, np.eye(2, dtype=np.float32), catalog_fingerprint="recovery-toy",
            config=ResponseDQNConfig(
                gamma=1.0, learning_rate=0.003, reward_scale=1.0, batch_size=4,
                gradient_steps=1600, target_update_interval=20, hidden=(32, 32),
                ensemble_size=1,
            ), seed=17, bootstrap=False,
        )
        q_root = model.q_values(replay.observation[0], replay.response_features[0])[0]
        q_recovery = model.q_values(replay.observation[2], replay.response_features[2])[0]
        self.assertGreater(q_root[1], q_root[0] + 5)
        self.assertGreater(q_recovery[1], q_recovery[0] + 50)
        self.assertAlmostEqual(float(q_root[1]), -2.0, delta=1.0)

    def test_sequential_mode_rejects_fixed_labels_and_discount_mismatch(self):
        replay = recovery_replay()
        validate_sequential_td_replay(replay, gamma=1.0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            validate_sequential_td_replay(replay, gamma=0.99)
        fixed = replace(replay, done=np.ones(4, np.float32))
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            validate_sequential_td_replay(fixed, gamma=1.0)
        advantage = replace(replay, manifest={**replay.manifest,
            "reward_semantics": "terminal_recovery_advantage_vs_action0_return"})
        with self.assertRaisesRegex(ValueError, "interval_negative_ttt"):
            validate_sequential_td_replay(advantage, gamma=1.0)


if __name__ == "__main__":
    unittest.main()
