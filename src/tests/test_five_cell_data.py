"""No-simulator tests for strict, balanced common-policy replay construction."""
from __future__ import annotations

import copy
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl_leader.experiment_contract import ExperimentContract, EXPERIMENT_PROFILE_ID
from rl_leader.five_cell_data import (
    FIVE_CELL_EQUIVALENCE, FIVE_CELL_SCENARIOS, merge_five_cell_replays,
    validate_five_cell_replay,
)
from rl_leader.response_dqn_catalog import DiscreteLeaderAction, StructuredActionCatalog
from rl_leader.response_dqn_data import (
    FrozenResponseReplay, load_frozen_response_replay, make_replay_manifest,
    merge_frozen_response_replays,
)


def fixture(episodes=1):
    catalog = StructuredActionCatalog(["residual"], [
        DiscreteLeaderAction(0, "anchor", "anchor", "all", "identity", 0., "anchor", (0.,)),
        DiscreteLeaderAction(1, "linear", "freeway", "ramp", "linear", .5, "linear", (.5,)),
    ])
    contracts, replays = {}, []
    for cell, scenario in enumerate(FIVE_CELL_SCENARIOS):
        payload = {
            "experiment_contract_version": 1, "profile_id": EXPERIMENT_PROFILE_ID,
            "scenario_name": scenario, "scenario": {"name": scenario, "urban_scale": 1.55 + cell / 10},
            "resolved_config": {"network": {"physical_coefficient": 12}},
            "pstack_options": {"controller": "native"},
            "warmup": {"control": "uncontrolled", "steps": 5, "far_updates": False, "included_in_total_ttt": True},
            "supervisor": {"mode": "none"}, "dynamic_far": {"mode": "incident_or_capacity_drop"},
            "simulation": {"T_total_sec": 14400., "control_interval_sec": 180., "stochastic_seed": 0},
        }
        contracts[scenario] = payload
        n = episodes * 75
        steps = np.tile(np.arange(75), episodes)
        observation = np.column_stack((np.full(n, cell), (steps + 5) / 80)).astype(np.float32)
        successor = np.column_stack((np.full(n, cell), (steps + 6) / 80)).astype(np.float32)
        response = np.repeat(steps[:, None, None], 2, axis=1).astype(np.float32)
        next_response = response + 1
        mask = np.ones((n, 2), dtype=bool)
        done = (steps == 74).astype(np.float32)
        next_mask = mask.copy()
        next_mask[done == 1, 1] = False
        manifest = make_replay_manifest(
            transition_count=n, action_count=2, catalog_fingerprint=catalog.fingerprint,
            observation_schema={"dimension": 2, "names": ["cell_observation", "time.phase"]},
            response_contract="fixture_response", scenario=scenario, source=f"fixture-{cell}",
        )
        manifest.update(
            catalog=catalog.as_manifest(), reward_semantics="interval_negative_ttt",
            done_semantics="environment_terminal", response_equivalence_mode=FIVE_CELL_EQUIVALENCE,
            t_total_sec=14400., experiment_contract_sha256=ExperimentContract.from_payload(payload).sha256,
        )
        replays.append(FrozenResponseReplay(
            observation=observation, next_observation=successor,
            action_id=steps % 2, reward=-np.ones(n, dtype=np.float32), done=done,
            option_steps=np.ones(n, dtype=np.int64), action_mask=mask, next_action_mask=next_mask,
            response_features=response, next_response_features=next_response,
            event_group=np.full(n, "same-original-group"), episode=np.repeat(np.arange(episodes), 75),
            control_step=steps, manifest=manifest,
        ).validate())
    return replays, contracts


def select_rows(replay, indices):
    arrays = {field.name: getattr(replay, field.name)[indices].copy()
              for field in fields(replay) if field.name != "manifest"}
    return replace(replay, **arrays, manifest={**replay.manifest, "transition_count": len(indices)})


class FiveCellDataTest(unittest.TestCase):
    def test_balanced_complete_episodes_and_colliding_ids_roundtrip(self):
        replays, contracts = fixture(2)
        original = [replay.episode.copy() for replay in replays]
        aggregate = merge_five_cell_replays(replays, contracts, "test")
        summary = validate_five_cell_replay(aggregate)
        self.assertEqual(summary["total_rows"], 750)
        self.assertEqual(summary["total_episodes"], 10)
        self.assertEqual(summary["true_terminals"], 10)
        self.assertEqual(summary["verified_links"], 740)
        np.testing.assert_array_equal(np.unique(aggregate.episode), np.arange(10))
        for cell in summary["scenario_counts"].values():
            self.assertEqual(cell, {"episodes": 2, "rows": 150, "terminals": 2,
                                    "row_fraction": .2, "terminal_fraction": .2})
        self.assertNotIn("experiment_contract_sha256", aggregate.manifest)
        self.assertEqual(len(aggregate.unique_event_groups()), 10)
        for replay, before in zip(replays, original):
            np.testing.assert_array_equal(replay.episode, before)
            self.assertIn("experiment_contract_sha256", replay.manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            aggregate.save(path)
            loaded = load_frozen_response_replay(path)
            self.assertEqual(validate_five_cell_replay(loaded), summary)
            np.testing.assert_array_equal(loaded.observation, aggregate.observation)

    def test_existing_single_scenario_merge_remains_strict_and_is_called(self):
        replays, contracts = fixture()
        with self.assertRaisesRegex(ValueError, "different"):
            merge_frozen_response_replays(replays, source="must-fail")
        with patch("rl_leader.five_cell_data.merge_frozen_response_replays",
                   wraps=merge_frozen_response_replays) as merge:
            merge_five_cell_replays(replays, contracts, "test")
        self.assertEqual(merge.call_count, 5)

    def test_each_source_file_may_reuse_episode_ids(self):
        replays, contracts = fixture()
        aggregate = merge_five_cell_replays([*replays, *copy.deepcopy(replays)], contracts, "test")
        self.assertEqual(len(np.unique(aggregate.episode)), 10)
        self.assertEqual({entry["original_episode"] for entry in aggregate.manifest["episode_sources"].values()}, {0})

    def test_missing_extra_or_unbalanced_cells_rejected(self):
        replays, contracts = fixture()
        with self.assertRaisesRegex(ValueError, "all five"):
            merge_five_cell_replays(replays[:-1], contracts, "test")
        with self.assertRaisesRegex(ValueError, "exactly the five"):
            merge_five_cell_replays(replays, {key: value for key, value in list(contracts.items())[:-1]}, "test")
        with self.assertRaisesRegex(ValueError, "equal complete episode"):
            merge_five_cell_replays([*replays, replays[0]], contracts, "test")
        replays[0].manifest["scenario"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "unexpected source scenario"):
            merge_five_cell_replays(replays, contracts, "test")

    def test_shared_physics_contract_pollution_rejected_even_with_new_hash(self):
        for key in ("resolved_config", "pstack_options", "supervisor", "dynamic_far"):
            replays, contracts = fixture()
            contracts[FIVE_CELL_SCENARIOS[1]][key]["pollution"] = True
            replays[1].manifest["experiment_contract_sha256"] = ExperimentContract.from_payload(contracts[FIVE_CELL_SCENARIOS[1]]).sha256
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "shared physical"):
                merge_five_cell_replays(replays, contracts, "test")

    def test_wrong_hash_identity_and_warmup_rejected(self):
        replays, contracts = fixture()
        replays[0].manifest["experiment_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            merge_five_cell_replays(replays, contracts, "test")
        replays, contracts = fixture()
        contracts[FIVE_CELL_SCENARIOS[0]]["scenario"]["name"] = "other"
        with self.assertRaisesRegex(ValueError, "scenario identity"):
            merge_five_cell_replays(replays, contracts, "test")
        replays, contracts = fixture()
        contracts[FIVE_CELL_SCENARIOS[0]]["warmup"]["steps"] = 4
        with self.assertRaisesRegex(ValueError, "normal reset"):
            merge_five_cell_replays(replays, contracts, "test")

    def test_missing_duplicate_shifted_or_fractional_steps_rejected(self):
        for kind in ("missing", "duplicate", "shifted", "fractional"):
            replays, contracts = fixture()
            if kind == "missing":
                replays[0] = select_rows(replays[0], np.arange(74))
            elif kind == "duplicate":
                replays[0].control_step[-1] = 73
            elif kind == "shifted":
                replays[0].control_step[:] += 1
            else:
                replays[0] = replace(replays[0], control_step=replays[0].control_step.astype(float) + .5)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "control step|integer"):
                merge_five_cell_replays(replays, contracts, "test")

    def test_false_early_or_missing_terminal_rejected(self):
        for kind in ("early", "missing", "all"):
            replays, contracts = fixture()
            if kind == "early":
                replays[0].done[40] = 1
            elif kind == "missing":
                replays[0].done[-1] = 0
            else:
                replays[0].done[:] = 1
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "true terminal"):
                merge_five_cell_replays(replays, contracts, "test")

    def test_link_corruption_and_wrong_phase_rejected(self):
        for field in ("next_observation", "next_response_features", "next_action_mask"):
            replays, contracts = fixture()
            value = getattr(replays[0], field)
            if field == "next_action_mask":
                value[10, 1] = False
            else:
                value[(10,) + (0,) * (value.ndim - 1)] += .01
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "boundary mismatch"):
                merge_five_cell_replays(replays, contracts, "test")
        replays, contracts = fixture()
        replays[0].observation[:, 1] += .001
        replays[0].next_observation[:, 1] += .001
        with self.assertRaisesRegex(ValueError, "time.phase"):
            merge_five_cell_replays(replays, contracts, "test")

    def test_schema_response_and_catalog_mismatch_rejected(self):
        for field, value in (("response_contract", "other"),
                             ("observation_schema", {"dimension": 2, "names": ["other", "time.phase"]})):
            replays, contracts = fixture()
            replays[1].manifest[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "shared replay contract mismatch"):
                merge_five_cell_replays(replays, contracts, "test")
        replays, contracts = fixture()
        replays[1].manifest["catalog"]["actions"][1]["magnitude"] = .7
        with self.assertRaisesRegex(ValueError, "catalog manifest fingerprint mismatch"):
            merge_five_cell_replays(replays, contracts, "test")

    def test_legacy_equivalence_and_transformed_rewards_rejected(self):
        for field, value in (("response_equivalence_mode", "post_commit_continuation_v1"),
                             ("reward_semantics", "monte_carlo_return"),
                             ("done_semantics", "rollout_terminal")):
            replays, contracts = fixture()
            replays[0].manifest[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                merge_five_cell_replays(replays, contracts, "test")

    def test_aggregate_tampering_cannot_claim_single_contract_or_missing_provenance(self):
        replays, contracts = fixture()
        original = merge_five_cell_replays(replays, contracts, "test")
        for kind in ("single", "hash", "map", "provenance"):
            aggregate = copy.deepcopy(original)
            if kind == "single":
                aggregate.manifest["experiment_contract_sha256"] = replays[0].manifest["experiment_contract_sha256"]
            elif kind == "hash":
                aggregate.manifest["aggregate_contract_sha256"] = "0" * 64
            elif kind == "map":
                aggregate.manifest["episode_scenarios"].pop("0")
            else:
                aggregate.manifest["episode_sources"]["0"]["experiment_contract_sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_five_cell_replay(aggregate)


if __name__ == "__main__":
    unittest.main()
