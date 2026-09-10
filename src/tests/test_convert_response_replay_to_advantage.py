from __future__ import annotations

import unittest

import numpy as np

from rl_leader.convert_response_replay_to_advantage import (
    convert_replay_to_action0_advantage,
)
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest


def _replay() -> FrozenResponseReplay:
    n = 5
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
    return FrozenResponseReplay(
        observation=np.zeros((n, 2), dtype=np.float32),
        action_id=np.asarray([0, 1, 2, 1, 0], dtype=np.int64),
        reward=np.asarray([-10.0, -8.0, -12.5, -3.0, -4.0], dtype=np.float32),
        next_observation=np.zeros((n, 2), dtype=np.float32),
        done=np.ones(n, dtype=np.float32),
        option_steps=np.ones(n, dtype=np.int64),
        action_mask=mask,
        next_action_mask=mask.copy(),
        response_features=np.zeros((n, action_count, response_dim), dtype=np.float32),
        next_response_features=np.zeros((n, action_count, response_dim), dtype=np.float32),
        event_group=np.asarray(["paired", "paired", "paired", "unpaired", "anchor-only"]),
        episode=np.asarray([1, 1, 1, 2, 3], dtype=np.int64),
        control_step=np.asarray([10, 10, 10, 10, 11], dtype=np.int64),
        manifest=manifest,
    ).validate()


class ConvertResponseReplayToAdvantageTest(unittest.TestCase):
    def test_converts_paired_rows_to_action0_relative_advantages(self):
        converted = convert_replay_to_action0_advantage(
            _replay(),
            source="unit-test",
        )

        np.testing.assert_array_equal(converted.action_id, [0, 1, 2])
        np.testing.assert_allclose(converted.reward, [0.0, 2.0, -2.5])
        self.assertEqual(converted.manifest["reward_mode"], "action0_advantage")
        self.assertEqual(converted.manifest["skipped_rows_without_action0"], 1)
        self.assertEqual(converted.manifest["skipped_anchor_only_rows"], 1)

    def test_can_keep_anchor_only_groups(self):
        converted = convert_replay_to_action0_advantage(
            _replay(),
            source="unit-test",
            require_non_anchor=False,
        )

        np.testing.assert_array_equal(converted.action_id, [0, 1, 2, 0])
        np.testing.assert_allclose(converted.reward, [0.0, 2.0, -2.5, 0.0])


if __name__ == "__main__":
    unittest.main()
