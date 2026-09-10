from __future__ import annotations

import unittest

from rl_leader.complete_balanced_h12 import _checkpoint_matches_label


class CompleteBalancedH12Test(unittest.TestCase):
    def test_checkpoint_contract_binds_metrics_memory_and_physical_state(self):
        candidate = {
            "ttt": 1.0,
            "terminal_inventory": 2.0,
            "follower_memory_sha256": "candidate-memory",
            "physical_state_sha256": "candidate-physical",
        }
        native = {
            "ttt": 3.0,
            "terminal_inventory": 4.0,
            "follower_memory_sha256": "native-memory",
            "physical_state_sha256": "native-physical",
        }
        label = {
            "candidate_ttt": 1.0,
            "native_ttt": 3.0,
            "candidate_terminal_inventory": 2.0,
            "native_terminal_inventory": 4.0,
            "candidate_follower_memory_sha256": "candidate-memory",
            "native_follower_memory_sha256": "native-memory",
            "candidate_physical_state_sha256": "candidate-physical",
            "native_physical_state_sha256": "native-physical",
        }
        self.assertTrue(_checkpoint_matches_label(candidate, native, label))
        label["candidate_physical_state_sha256"] = "drift"
        self.assertFalse(_checkpoint_matches_label(candidate, native, label))


if __name__ == "__main__":
    unittest.main()
