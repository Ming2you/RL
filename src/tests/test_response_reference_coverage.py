"""Reference trajectories are raw sequential data and must reproduce their source."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from work.run_response_reference_coverage import persist_merged_replay, verify_reference


def tiny_episode():
    manifest = make_replay_manifest(
        transition_count=2, action_count=2, catalog_fingerprint="test",
        observation_schema={"dimension": 3}, response_contract="test", scenario="test", source="test",
    )
    manifest.update(reward_semantics="interval_negative_ttt", done_semantics="environment_terminal")
    return FrozenResponseReplay(
        observation=np.eye(3, dtype=np.float32)[:2], next_observation=np.eye(3, dtype=np.float32)[1:],
        action_id=np.array([0, 0]), reward=np.array([-1, -2], np.float32), done=np.array([0, 1], np.float32),
        option_steps=np.ones(2, np.int64), action_mask=np.ones((2, 2), bool), next_action_mask=np.ones((2, 2), bool),
        response_features=np.zeros((2, 2, 1), np.float32), next_response_features=np.zeros((2, 2, 1), np.float32),
        event_group=np.array(["test", "test"]), episode=np.zeros(2, np.int64), control_step=np.arange(2), manifest=manifest,
    ).validate()


class ReferenceCoverageTest(unittest.TestCase):
    def test_reference_total_interval_and_actions_are_verified(self):
        replay = tiny_episode()
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text('\n'.join(json.dumps({"control_step": i, "interval_ttt": i + 1}) for i in range(2)), encoding="utf-8")
            reference = {"expected_total_ttt": 3, "expected_trace": str(trace),
                         "expected_interval_field": "interval_ttt", "expected_interval_sign": -1,
                         "first_action": None}
            verify_reference({"total_ttt": 3}, replay, reference, 1e-4)
            with self.assertRaises(ValueError):
                verify_reference({"total_ttt": 4}, replay, reference, 1e-4)
            with self.assertRaises(AssertionError):
                verify_reference({"total_ttt": 3}, replace(replay, reward=np.array([-2, -1], np.float32)), reference, 1e-4)
            with self.assertRaises(AssertionError):
                verify_reference({"total_ttt": 3}, replace(replay, action_id=np.array([0, 1])), reference, 1e-4)

    def test_reference_keeps_later_transition_nonterminal_until_real_end(self):
        with self.assertRaises(ValueError):
            verify_reference({"total_ttt": 3}, replace(tiny_episode(), done=np.ones(2, np.float32)), {}, 1e-4)

    def test_existing_merged_replay_must_equal_sources_without_overwrite(self):
        replay = tiny_episode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            digest = persist_merged_replay(replay, path)
            self.assertEqual(persist_merged_replay(replay, path), digest)
            with self.assertRaises(AssertionError):
                persist_merged_replay(replace(replay, reward=np.array([-2, -3], np.float32)), path)
            self.assertEqual(persist_merged_replay(replay, path), digest)


if __name__ == "__main__":
    unittest.main()
