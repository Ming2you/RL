from __future__ import annotations

import unittest

import numpy as np

from rl_leader.response_dqn_catalog import DiscreteLeaderAction, StructuredActionCatalog
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from rl_leader.subset_response_replay_catalog import subset_replay_catalog


def _catalog() -> StructuredActionCatalog:
    names = ("freeway.R_F_E.g_meter", "freeway.R_F_E.g_vsl")
    return StructuredActionCatalog(
        names,
        (
            DiscreteLeaderAction(0, "anchor", "anchor", "P-Stack", "identity", 0.0, "anchor", (0.0, 0.0)),
            DiscreteLeaderAction(1, "linear", "freeway", "R_F_E", "linear_first_positive", 0.25, "linear", (0.25, 0.0)),
            DiscreteLeaderAction(2, "hybrid-a", "freeway", "R_F_E", "hybrid_first_pos_quad_first_inc", 0.25, "hybrid", (0.24, 0.05)),
            DiscreteLeaderAction(3, "hybrid-b", "freeway", "R_F_E", "hybrid_corner_pp_cross_pos", 0.25, "hybrid", (0.18, 0.18)),
        ),
    )


def _replay() -> FrozenResponseReplay:
    catalog = _catalog()
    n = 5
    action_count = catalog.size
    response_dim = 3
    base = np.arange(n * action_count * response_dim, dtype=np.float32).reshape(
        n, action_count, response_dim,
    )
    manifest = make_replay_manifest(
        transition_count=n,
        action_count=action_count,
        catalog_fingerprint=catalog.fingerprint,
        observation_schema={"dimension": 2},
        response_contract="test",
        scenario="scenario",
        source="unit",
    )
    manifest["catalog"] = catalog.as_manifest()
    return FrozenResponseReplay(
        observation=np.zeros((n, 2), dtype=np.float32),
        action_id=np.asarray([0, 1, 2, 3, 2], dtype=np.int64),
        reward=np.asarray([0.0, 10.0, 20.0, 30.0, 21.0], dtype=np.float32),
        next_observation=np.ones((n, 2), dtype=np.float32),
        done=np.ones(n, dtype=np.float32),
        option_steps=np.ones(n, dtype=np.int64),
        action_mask=np.ones((n, action_count), dtype=bool),
        next_action_mask=np.ones((n, action_count), dtype=bool),
        response_features=base,
        next_response_features=base + 1000.0,
        event_group=np.asarray([f"group-{index}" for index in range(n)]),
        episode=np.zeros(n, dtype=np.int64),
        control_step=np.arange(n, dtype=np.int64),
        manifest=manifest,
    ).validate()


class SubsetResponseReplayCatalogTest(unittest.TestCase):
    def test_subsets_catalog_and_remaps_action_ids(self):
        subset = subset_replay_catalog(
            _replay(),
            source="unit-test",
            families={"hybrid"},
        )

        np.testing.assert_array_equal(subset.action_id, [0, 1, 2, 1])
        np.testing.assert_allclose(subset.reward, [0.0, 20.0, 30.0, 21.0])
        self.assertEqual(subset.action_count, 3)
        self.assertEqual(subset.action_support_counts().tolist(), [1, 2, 1])
        self.assertEqual(
            subset.manifest["catalog_subset"]["old_action_ids"],
            [0, 2, 3],
        )
        restored = StructuredActionCatalog.from_manifest(subset.manifest["catalog"])
        self.assertEqual(restored.fingerprint, subset.manifest["catalog_fingerprint"])
        np.testing.assert_allclose(
            subset.response_features[1, 1],
            _replay().response_features[2, 2],
        )


if __name__ == "__main__":
    unittest.main()
