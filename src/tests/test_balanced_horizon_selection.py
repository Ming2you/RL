from __future__ import annotations

import unittest

from rl_leader.evaluate_balanced_horizons import (
    select_h12_representatives,
    select_h12_representatives_v2,
)


def _row(candidate_id, gain, inventory, distance):
    return {
        "candidate_id": candidate_id,
        "h3_label": {
            "validity_gate_pass": True,
            "ttt_gain": gain,
            "terminal_inventory_delta": inventory,
        },
        "native_response_distance": {"overall_rmse": distance},
    }


class BalancedHorizonSelectionTest(unittest.TestCase):
    def test_selector_unions_three_complementary_criteria(self):
        rows = [
            _row("ttt", 2.0, 1.0, 0.1),
            _row("inventory", 0.0, -3.0, 0.2),
            _row("diverse", -1.0, 2.0, 0.9),
        ]
        selected, evidence = select_h12_representatives(rows)
        self.assertEqual({row["candidate_id"] for row in selected}, {
            "ttt", "inventory", "diverse",
        })
        self.assertEqual(set(evidence["selected_by"]), {
            "ttt", "inventory", "diverse",
        })

    def test_selector_collapses_overlapping_criteria(self):
        rows = [_row("winner", 2.0, -3.0, 0.9), _row("other", 0.0, 0.0, 0.1)]
        selected, evidence = select_h12_representatives(rows)
        self.assertEqual([row["candidate_id"] for row in selected], ["winner"])
        self.assertEqual(len(evidence["selected_by"]["winner"]), 3)

    def test_v2_preserves_h3_best_response_per_owner(self):
        rows = [
            _row("structured:urban:A:axis:first", 1.0, 0.0, 0.2),
            _row("structured:urban:A:axis:second", 2.0, 0.0, 0.2),
            _row("structured:urban:C:axis:delayed", 0.1, 0.0, 0.2),
            _row("structured:urban:F:axis:identity", 5.0, 0.0, 0.0),
        ]
        selected, evidence = select_h12_representatives_v2(rows)
        self.assertEqual([row["candidate_id"] for row in selected], [
            "structured:urban:A:axis:second",
            "structured:urban:C:axis:delayed",
        ])
        self.assertEqual(evidence["owner_coverage"], ["urban:A", "urban:C"])


if __name__ == "__main__":
    unittest.main()
