from __future__ import annotations

import unittest

from rl_leader.run_balanced_owner_block_pilot import (
    _frozen_event_differences,
    _parse_csv,
    _parse_int_csv,
    _selected_strata,
)


class BalancedOwnerBlockPilotTest(unittest.TestCase):
    def test_parse_csv_drops_empty_items_and_whitespace(self):
        self.assertEqual(
            _parse_csv(" early, peak,,recovery "),
            ("early", "peak", "recovery"),
        )

    def test_parse_int_csv_drops_empty_items_and_whitespace(self):
        self.assertEqual(_parse_int_csv(" 1, 3,,5 "), (1, 3, 5))

    def test_frozen_event_comparison_normalizes_json_tuple_round_trip(self):
        actual = {"physical_snapshot": {"shape": (2, 3)}, "anchor": "same"}
        expected = {"physical_snapshot": {"shape": [2, 3]}, "anchor": "same"}
        self.assertEqual(_frozen_event_differences(actual, expected), [])
        expected["anchor"] = "drift"
        self.assertEqual(_frozen_event_differences(actual, expected), ["anchor"])

    def test_selected_strata_can_use_all_manifest_strata(self):
        manifest = {
            "scenarios": [
                {
                    "events": [
                        {"stratum": "ramp_up"},
                        {"stratum": "late_recovery"},
                    ]
                }
            ]
        }
        self.assertEqual(
            _selected_strata(manifest, ("all",)),
            {"ramp_up", "late_recovery"},
        )


if __name__ == "__main__":
    unittest.main()
