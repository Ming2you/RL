from __future__ import annotations

import unittest

from rl_leader.build_tail_pairwise_dataset import (
    _assign_group_weights,
    _deduplicate_rows,
    _margin_ratio,
)


def _row(group, outcome, target, gain):
    return {
        "event_group_id": group,
        "realized_outcome_id": outcome,
        "target_valid": True,
        "target": target,
        "tail_status": "positive" if target else "negative",
        "gain_veh_h": gain,
        "required_gain_veh_h": 2.0,
        "margin_ratio": _margin_ratio(gain, 2.0),
        "candidate_aliases": [],
        "residual_aliases": [],
        "candidate_id": outcome,
        "candidate_residual": [0.25, 0.0],
        "source_provenance": {"source": outcome},
        "sample_weight": 0.0,
    }


class TailPairwiseDatasetTests(unittest.TestCase):
    def test_margin_ratio_is_relative_to_material_margin(self):
        self.assertEqual(_margin_ratio(3.0, 2.0), 0.5)
        self.assertEqual(_margin_ratio(1.0, 2.0), -0.5)

    def test_each_event_group_has_unit_total_weight(self):
        rows = [
            _row("a", "a1", 1, 3.0),
            _row("a", "a2", 0, 1.0),
            _row("b", "b1", 0, -1.0),
        ]
        _assign_group_weights(rows)
        self.assertEqual([row["sample_weight"] for row in rows], [0.5, 0.5, 1.0])

    def test_conflicting_realized_outcome_labels_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            _deduplicate_rows([
                _row("a", "same", 1, 3.0),
                _row("a", "same", 0, 1.0),
            ])


if __name__ == "__main__":
    unittest.main()
