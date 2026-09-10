from __future__ import annotations

import unittest

from rl_leader.diagnose_balanced_remaining_horizon import (
    exact_balanced_h12_replay,
)


class BalancedRemainingHorizonTest(unittest.TestCase):
    def test_exact_h12_replay_checks_state_and_metrics(self):
        candidate_checkpoint = {
            "ttt": 10.0,
            "terminal_inventory": 20.0,
            "follower_memory_sha256": "candidate-memory",
            "physical_state_sha256": "candidate-physical",
            "validity_gate_pass": True,
        }
        native_checkpoint = {
            "ttt": 11.0,
            "terminal_inventory": 21.0,
            "follower_memory_sha256": "native-memory",
            "physical_state_sha256": "native-physical",
            "validity_gate_pass": True,
        }
        expected = {
            "candidate_ttt": 10.0,
            "native_ttt": 11.0,
            "candidate_terminal_inventory": 20.0,
            "native_terminal_inventory": 21.0,
            "candidate_follower_memory_sha256": "candidate-memory",
            "native_follower_memory_sha256": "native-memory",
            "candidate_physical_state_sha256": "candidate-physical",
            "native_physical_state_sha256": "native-physical",
            "validity_gate_pass": True,
        }
        replay = exact_balanced_h12_replay(
            {"checkpoints": {"12": candidate_checkpoint}},
            {"checkpoints": {"12": native_checkpoint}},
            expected,
        )
        self.assertTrue(replay["passed"])

        expected["candidate_ttt"] = 10.1
        replay = exact_balanced_h12_replay(
            {"checkpoints": {"12": candidate_checkpoint}},
            {"checkpoints": {"12": native_checkpoint}},
            expected,
        )
        self.assertFalse(replay["passed"])
        self.assertFalse(replay["candidate_ttt_exact"])


if __name__ == "__main__":
    unittest.main()
