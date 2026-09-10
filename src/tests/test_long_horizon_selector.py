from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.build_tail_pairwise_dataset import TAIL_PAIRWISE_FORMAT
from rl_leader.select_long_horizon_candidates import (
    LONG_HORIZON_SELECTOR_FORMAT,
    select_long_horizon_candidates,
)


def _row(group, candidate, target, gain, required=2.0, inventory_delta=0.0):
    return {
        "event_group_id": group,
        "scenario": f"scenario-{group}",
        "stratum": "plateau",
        "policy_step": 9,
        "candidate_id": candidate,
        "candidate_residual": [0.25, 0.0],
        "tail_status": "positive" if target else "negative",
        "target_valid": True,
        "target": int(target),
        "gain_veh_h": float(gain),
        "required_gain_veh_h": float(required),
        "margin_ratio": float((gain - required) / required),
        "terminal_inventory_delta_veh": float(inventory_delta),
    }


class LongHorizonSelectorTests(unittest.TestCase):
    def _select(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            output_path = Path(directory) / "selection.json"
            dataset_path.write_text(
                json.dumps({
                    "format_version": TAIL_PAIRWISE_FORMAT,
                    "rows": rows,
                }),
                encoding="utf-8",
            )
            result = select_long_horizon_candidates(
                dataset_path, output_path, **kwargs
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result, written)
            return result

    def test_selects_best_tail_positive_per_event_group(self):
        result = self._select([
            _row("a", "negative", 0, -1.0),
            _row("a", "weak-positive", 1, 3.0),
            _row("a", "strong-positive", 1, 5.0),
        ])

        self.assertEqual(result["format_version"], LONG_HORIZON_SELECTOR_FORMAT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["selected_event_groups"], 1)
        self.assertEqual(
            result["selections"][0]["selected_candidate_id"],
            "strong-positive",
        )
        self.assertEqual(result["selections"][0]["decision"], "select_candidate")

    def test_falls_back_to_pstack_when_no_tail_positive_passes(self):
        result = self._select([
            _row("a", "negative", 0, -1.0),
            _row("a", "margin-failure", 1, 2.0),
        ])

        self.assertEqual(result["summary"]["selected_event_groups"], 0)
        self.assertEqual(result["selections"][0]["decision"], "pstack_fallback")
        self.assertIsNone(result["selections"][0]["selected_candidate_id"])

    def test_optional_inventory_cap_blocks_positive_candidate(self):
        result = self._select(
            [_row("a", "inventory-heavy-positive", 1, 5.0, inventory_delta=1.5)],
            max_terminal_inventory_delta_veh=0.0,
        )

        self.assertEqual(result["selections"][0]["decision"], "pstack_fallback")
        self.assertEqual(
            result["selections"][0]["rejected_candidates"][0]["reason"],
            "terminal_inventory_delta_above_threshold",
        )


if __name__ == "__main__":
    unittest.main()
