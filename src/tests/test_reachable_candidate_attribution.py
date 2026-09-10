from __future__ import annotations

import unittest

from rl_leader.diagnose_reachable_candidate_attribution import (
    _require_price_anchor_branch,
    _require_requested_horizon_capacity,
    _validate_h1_probe_replay,
)


class ReachableCandidateAttributionContractTest(unittest.TestCase):
    def test_price_attribution_rejects_native_pfo_branches(self):
        _require_price_anchor_branch("coarse")
        _require_price_anchor_branch("refined")
        for branch in ("fallback_pfo", "supervisor_pfo", "other"):
            with self.subTest(branch=branch):
                with self.assertRaisesRegex(RuntimeError, "coarse/refined"):
                    _require_price_anchor_branch(branch)

    def test_requested_horizon_cannot_be_silently_truncated(self):
        _require_requested_horizon_capacity(12, (1, 3, 6, 12))
        with self.assertRaisesRegex(RuntimeError, "H12"):
            _require_requested_horizon_capacity(6, (1, 3, 6, 12))

    def test_h1_replay_requires_response_and_follower_memory(self):
        record = {
            "response": [1.0, 2.0],
            "post_follower_sha256": "a" * 64,
            "post_physical_sha256": "c" * 64,
        }
        rollout = {
            "first_step_response": [1.0, 2.0],
            "checkpoints": {
                "1": {
                    "follower_memory_sha256": "a" * 64,
                    "physical_state_sha256": "c" * 64,
                },
            },
        }

        evidence = _validate_h1_probe_replay(record, rollout, 1.0e-6)
        self.assertTrue(evidence["passed"])

        changed = {
            **rollout,
            "checkpoints": {
                "1": {"follower_memory_sha256": "b" * 64},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "follower memory"):
            _validate_h1_probe_replay(record, changed, 1.0e-6)

        missing = {"response": [1.0, 2.0]}
        with self.assertRaisesRegex(RuntimeError, "missing.*follower memory"):
            _validate_h1_probe_replay(missing, rollout, 1.0e-6)

        changed = {
            **rollout,
            "checkpoints": {
                "1": {
                    "follower_memory_sha256": "a" * 64,
                    "physical_state_sha256": "d" * 64,
                },
            },
        }
        with self.assertRaisesRegex(RuntimeError, "physical state"):
            _validate_h1_probe_replay(record, changed, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
