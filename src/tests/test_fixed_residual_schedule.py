import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.evaluate_fixed_residual_schedule import (
    load_oracle_choice_schedule,
    merge_schedules,
    parse_sparse_schedule_items,
)


class FixedResidualScheduleTests(unittest.TestCase):
    def test_parse_manual_sparse_schedule(self):
        names = ("urban.A.g_green", "urban.A.g_offset", "freeway.R_F_W.g_meter")
        schedule = parse_sparse_schedule_items(
            ["21=urban.A.g_green:0.5,freeway.R_F_W.g_meter:-0.25"],
            names,
        )
        self.assertEqual([21], list(schedule))
        self.assertEqual(
            (0.5, 0.0, -0.25),
            schedule[21].residual,
        )

    def test_parse_coordination_zero_schedule(self):
        names = ("urban.A.g_green", "urban.A.g_offset")
        schedule = parse_sparse_schedule_items(["13=coordination_zero"], names)
        self.assertEqual((0.0, 0.0), schedule[13].residual)
        self.assertEqual("coordination_zero", schedule[13].label)

    def test_load_oracle_choice_schedule_uses_event_choice(self):
        names = ("urban.A.g_green", "freeway.R_F_W.g_meter")
        artifact = {
            "events": [
                {
                    "step": 17,
                    "oracle_choice": {"selected": "toward:meter:0.5"},
                    "candidates_by_ttt": [
                        {
                            "representative_label": "toward:meter:0.5",
                            "aliases": ["alias:meter"],
                            "residual_nonzero": {"freeway.R_F_W.g_meter": 0.5},
                        }
                    ],
                },
                {
                    "step": 21,
                    "oracle_choice": {"selected": "coordination_zero"},
                    "candidates_by_ttt": [],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oracle.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            schedule = load_oracle_choice_schedule(path, names)
        self.assertEqual((0.0, 0.5), schedule[17].residual)
        self.assertEqual((0.0, 0.0), schedule[21].residual)

    def test_merge_rejects_duplicate_steps(self):
        names = ("urban.A.g_green",)
        left = parse_sparse_schedule_items(["1=urban.A.g_green:0.25"], names)
        right = parse_sparse_schedule_items(["1=coordination_zero"], names)
        with self.assertRaises(ValueError):
            merge_schedules(left, right)


if __name__ == "__main__":
    unittest.main()
