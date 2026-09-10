from __future__ import annotations

import unittest

import numpy as np

from rl_leader.convert_response_replay_to_mc import convert_replay_to_mc_targets
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest


def _replay() -> FrozenResponseReplay:
    n = 4
    action_count = 3
    response_dim = 2
    mask = np.ones((n, action_count), dtype=bool)
    manifest = make_replay_manifest(
        transition_count=n,
        action_count=action_count,
        catalog_fingerprint="catalog",
        observation_schema={"dimension": 2},
        response_contract="test",
        scenario="scenario",
        source="unit",
    )
    manifest["catalog"] = {"fingerprint": "catalog"}
    return FrozenResponseReplay(
        observation=np.zeros((n, 2), dtype=np.float32),
        action_id=np.asarray([0, 1, 2, 0], dtype=np.int64),
        reward=np.asarray([-1.0, -2.0, -3.0, -10.0], dtype=np.float32),
        next_observation=np.zeros((n, 2), dtype=np.float32),
        done=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        option_steps=np.ones(n, dtype=np.int64),
        action_mask=mask,
        next_action_mask=mask.copy(),
        response_features=np.zeros((n, action_count, response_dim), dtype=np.float32),
        next_response_features=np.zeros((n, action_count, response_dim), dtype=np.float32),
        event_group=np.asarray(["episode-a", "episode-a", "episode-a", "episode-b"]),
        episode=np.asarray([1, 1, 1, 2], dtype=np.int64),
        control_step=np.asarray([0, 1, 2, 1], dtype=np.int64),
        manifest=manifest,
    ).validate()


class ConvertResponseReplayToMCTest(unittest.TestCase):
    def test_converts_grouped_episode_rewards_to_terminal_mc_targets(self):
        converted = convert_replay_to_mc_targets(_replay(), gamma=1.0)

        np.testing.assert_allclose(converted.reward, [-6.0, -5.0, -3.0, -10.0])
        np.testing.assert_array_equal(converted.option_steps, [3, 2, 1, 1])
        np.testing.assert_array_equal(converted.done, [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.all(converted.next_action_mask[:, 0]))
        self.assertFalse(np.any(converted.next_action_mask[:, 1:]))

    def test_can_keep_only_selected_control_steps(self):
        converted = convert_replay_to_mc_targets(
            _replay(),
            gamma=1.0,
            control_steps=(1,),
        )

        np.testing.assert_array_equal(converted.control_step, [1, 1])
        np.testing.assert_array_equal(converted.action_id, [1, 0])
        np.testing.assert_allclose(converted.reward, [-5.0, -10.0])
        self.assertEqual(converted.manifest["selected_control_steps"], [1])


if __name__ == "__main__":
    unittest.main()
