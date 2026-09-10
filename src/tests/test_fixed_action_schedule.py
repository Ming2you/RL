from __future__ import annotations

import unittest

from rl_leader.evaluate_fixed_action_schedule import (
    parse_schedule,
    resolve_schedule,
)
from rl_leader.response_dqn_catalog import DiscreteLeaderAction, StructuredActionCatalog


def _catalog() -> StructuredActionCatalog:
    return StructuredActionCatalog(
        ("freeway.R_F_W.g_meter",),
        (
            DiscreteLeaderAction(
                action_id=0,
                key="anchor",
                domain="anchor",
                owner="P-Stack",
                template="identity",
                magnitude=0.0,
                family="anchor",
                residual=(0.0,),
            ),
            DiscreteLeaderAction(
                action_id=1,
                key="freeway:R_F_W:linear_first_positive:m0.25",
                domain="freeway",
                owner="R_F_W",
                template="linear_first_positive",
                magnitude=0.25,
                family="linear",
                residual=(0.25,),
            ),
        ),
    )


class FixedActionScheduleTest(unittest.TestCase):
    def test_parse_schedule_orders_and_validates_entries(self):
        self.assertEqual(
            parse_schedule([
                "21=freeway:R_F_W:linear_first_positive:m0.5",
                "17=freeway:R_F_W:linear_first_positive:m0.25",
            ]),
            {
                17: "freeway:R_F_W:linear_first_positive:m0.25",
                21: "freeway:R_F_W:linear_first_positive:m0.5",
            },
        )
        with self.assertRaises(ValueError):
            parse_schedule(["bad-entry"])
        with self.assertRaises(ValueError):
            parse_schedule(["3=a", "3=b"])

    def test_resolve_schedule_maps_keys_to_action_ids(self):
        self.assertEqual(
            resolve_schedule(
                _catalog(),
                {17: "freeway:R_F_W:linear_first_positive:m0.25"},
            ),
            {17: 1},
        )
        with self.assertRaises(ValueError):
            resolve_schedule(_catalog(), {17: "missing"})
        with self.assertRaises(ValueError):
            resolve_schedule(_catalog(), {17: "anchor"})


if __name__ == "__main__":
    unittest.main()
