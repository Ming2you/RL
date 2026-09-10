from __future__ import annotations

import unittest

from rl_leader.diagnose_oracle_remaining_horizon import (
    select_minimal_execution_alias,
)


class OracleRemainingHorizonTest(unittest.TestCase):
    def test_minimal_alias_prefers_sparse_small_target_independent_action(self):
        base = {
            "execution_branch": "coordination",
            "response_memory_outcome_sha256": "outcome",
            "post_physical_sha256": "physical",
        }
        rows = [
            {
                **base,
                "candidate_id": "guided",
                "continuous_residual": [-0.25, 0.25],
                "uses_pcent_target": True,
            },
            {
                **base,
                "candidate_id": "axis-large",
                "continuous_residual": [-0.5, 0.0],
                "uses_pcent_target": False,
            },
            {
                **base,
                "candidate_id": "axis-small",
                "continuous_residual": [-0.25, 0.0],
                "uses_pcent_target": False,
            },
        ]
        representative = {
            **rows[0],
            "candidate_aliases": [row["candidate_id"] for row in rows],
        }
        selected = select_minimal_execution_alias(
            {"rows": rows}, representative
        )
        self.assertEqual(selected["candidate_id"], "axis-small")

    def test_aliases_must_share_one_realized_outcome(self):
        rows = [
            {
                "candidate_id": "first",
                "execution_branch": "coordination",
                "continuous_residual": [0.1],
                "uses_pcent_target": False,
                "response_memory_outcome_sha256": "first",
                "post_physical_sha256": "physical",
            },
            {
                "candidate_id": "second",
                "execution_branch": "coordination",
                "continuous_residual": [0.2],
                "uses_pcent_target": False,
                "response_memory_outcome_sha256": "second",
                "post_physical_sha256": "physical",
            },
        ]
        representative = {
            **rows[0],
            "candidate_aliases": ["first", "second"],
        }
        with self.assertRaisesRegex(ValueError, "one realized outcome"):
            select_minimal_execution_alias({"rows": rows}, representative)


if __name__ == "__main__":
    unittest.main()
