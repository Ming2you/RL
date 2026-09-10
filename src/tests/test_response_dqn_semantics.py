from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from rl_leader.convert_response_replay_to_mc import convert_replay_to_mc_targets
from rl_leader.response_dqn_data import (
    FrozenResponseReplay,
    make_replay_manifest,
    merge_frozen_response_replays,
)


def _replay(**metadata) -> FrozenResponseReplay:
    manifest = make_replay_manifest(
        transition_count=2,
        action_count=2,
        catalog_fingerprint="catalog-test",
        observation_schema={"dimension": 2},
        response_contract="response-test",
        scenario="scenario-test",
        source="unit-test",
    )
    manifest.update(metadata)
    return FrozenResponseReplay(
        observation=np.asarray([[0, 1], [1, 2]], dtype=np.float32),
        action_id=np.asarray([0, 1], dtype=np.int64),
        reward=np.asarray([-1, -2], dtype=np.float32),
        next_observation=np.asarray([[1, 2], [2, 3]], dtype=np.float32),
        done=np.asarray([0, 1], dtype=np.float32),
        option_steps=np.ones(2, dtype=np.int64),
        action_mask=np.ones((2, 2), dtype=bool),
        next_action_mask=np.ones((2, 2), dtype=bool),
        response_features=np.zeros((2, 2, 1), dtype=np.float32),
        next_response_features=np.zeros((2, 2, 1), dtype=np.float32),
        event_group=np.asarray(["episode-a", "episode-a"]),
        episode=np.zeros(2, dtype=np.int64),
        control_step=np.arange(2, dtype=np.int64),
        manifest=manifest,
    ).validate()


def _recovery(**metadata) -> FrozenResponseReplay:
    fields = {
        "collector_format": "response_ddqn_recovery_replay_v1",
        "reward_semantics": "terminal_best_recovery_return_negative_ttt",
        "done_semantics": "recovery_labels_are_terminal_mc_targets",
        "reward_mode": "return",
        "recovery_config": {"max_rollout_steps": 20, "recovery_depth": 0},
    }
    fields.update(metadata)
    return replace(_replay(**fields), done=np.ones(2, dtype=np.float32))


class ResponseDQNSemanticsTest(unittest.TestCase):
    def assert_incompatible(self, first, second, field):
        for replays in ([first, second], [second, first]):
            with self.subTest(field=field, first=replays[0].manifest):
                with self.assertRaisesRegex(ValueError, field):
                    merge_frozen_response_replays(replays, source="merge-test")

    def test_legacy_raw_merge_preserves_arrays_and_support(self):
        first = _replay()
        second = replace(first, episode=first.episode + 1)
        merged = merge_frozen_response_replays([first, second], source="merge-test")
        self.assertEqual(merged.size, 4)
        np.testing.assert_array_equal(merged.reward, [-1, -2, -1, -2])
        np.testing.assert_array_equal(merged.done, [0, 1, 0, 1])
        np.testing.assert_array_equal(merged.episode, [0, 0, 1, 1])
        np.testing.assert_array_equal(merged.action_support_counts(), [2, 2])
        self.assertEqual(merged.manifest["merged_batch_count"], 2)
        self.assertEqual(merged.manifest["reward_semantics"], "interval_negative_ttt")
        self.assertEqual(merged.manifest["done_semantics"], "environment_terminal")
        self.assertNotIn("reward_semantics", first.manifest)

    def test_legacy_and_explicit_raw_semantics_are_compatible(self):
        legacy = _replay()
        explicit = _replay(
            reward_semantics="interval_negative_ttt",
            done_semantics="environment_terminal",
        )
        for replays in ([legacy, explicit], [explicit, legacy]):
            merged = merge_frozen_response_replays(replays, source="merge-test")
            self.assertEqual(merged.size, 4)
            self.assertEqual(merged.manifest["reward_semantics"], "interval_negative_ttt")

    def test_interval_and_terminal_mc_returns_cannot_mix(self):
        raw = _replay()
        mc = convert_replay_to_mc_targets(raw)
        self.assert_incompatible(raw, mc, "reward_semantics")

    def test_interval_and_terminal_advantages_cannot_mix(self):
        advantage = _recovery(
            reward_mode="advantage",
            reward_semantics="terminal_recovery_advantage_vs_action0_return",
        )
        self.assert_incompatible(_replay(), advantage, "reward_semantics")
        self.assert_incompatible(_recovery(), advantage, "reward_semantics")

    def test_different_done_semantics_cannot_mix(self):
        first = _recovery()
        second = _recovery(done_semantics="monte_carlo_targets_are_terminal")
        self.assert_incompatible(first, second, "done_semantics")

    def test_matching_mc_objectives_remain_compatible(self):
        mc = convert_replay_to_mc_targets(_replay(), gamma=0.95)
        merged = merge_frozen_response_replays([mc, mc], source="merge-test")
        self.assertEqual(merged.manifest["mc_gamma"], 0.95)
        np.testing.assert_allclose(merged.reward, [-2.9, -2, -2.9, -2])
        np.testing.assert_array_equal(merged.done, np.ones(4))

    def test_different_mc_gamma_cannot_mix(self):
        self.assert_incompatible(
            convert_replay_to_mc_targets(_replay(), gamma=1.0),
            convert_replay_to_mc_targets(_replay(), gamma=0.95),
            "mc_gamma",
        )

    def test_different_recovery_horizon_and_depth_cannot_mix(self):
        for field, value in (("max_rollout_steps", 12), ("recovery_depth", 1)):
            config = dict(_recovery().manifest["recovery_config"], **{field: value})
            self.assert_incompatible(_recovery(), _recovery(recovery_config=config), field)

    def test_recovery_worker_settings_do_not_change_objective(self):
        first = _recovery()
        config = dict(first.manifest["recovery_config"], branch_workers=8,
                      response_workers=1, response_backend="serial")
        merged = merge_frozen_response_replays(
            [first, _recovery(recovery_config=config)], source="merge-test",
        )
        self.assertEqual(merged.size, 4)

    def test_different_reward_mode_cannot_mix_even_with_same_semantic_strings(self):
        self.assert_incompatible(_recovery(), _recovery(reward_mode="advantage"), "reward_mode")

    def test_different_known_environment_metadata_cannot_mix(self):
        for field, first, second in (
            ("experiment_contract_sha256", "contract-a", "contract-b"),
            ("t_total_sec", 14400.0, 7200.0),
        ):
            self.assert_incompatible(_replay(**{field: first}), _replay(**{field: second}), field)

    def test_known_metadata_cannot_mix_with_missing_metadata(self):
        for field, value in (
            ("experiment_contract_sha256", "contract-a"),
            ("t_total_sec", 14400.0),
            ("mc_gamma", 1.0),
            ("reward_mode", "return"),
        ):
            known = _recovery(**{field: value})
            manifest = dict(known.manifest)
            manifest.pop(field)
            self.assert_incompatible(known, replace(known, manifest=manifest), field)
        for field in ("max_rollout_steps", "recovery_depth"):
            config = dict(_recovery().manifest["recovery_config"])
            config.pop(field)
            self.assert_incompatible(_recovery(), _recovery(recovery_config=config), field)

    def test_matching_known_environment_metadata_remains_compatible(self):
        replay = _replay(experiment_contract_sha256="contract-a", t_total_sec=14400.0)
        merged = merge_frozen_response_replays([replay, replay], source="merge-test")
        self.assertEqual(merged.manifest["experiment_contract_sha256"], "contract-a")
        self.assertEqual(merged.manifest["t_total_sec"], 14400.0)

    def test_matching_legacy_terminal_semantics_remain_compatible(self):
        replay = _recovery(
            reward_mode="advantage",
            reward_semantics="terminal_recovery_advantage_vs_action0_return",
        )
        merged = merge_frozen_response_replays([replay, replay], source="merge-test")
        self.assertEqual(merged.manifest["reward_semantics"], replay.manifest["reward_semantics"])
        self.assertEqual(merged.size, 4)

    def test_nested_known_contracts_are_checked(self):
        first = _recovery(recovery_state={"experiment_contract_sha256": "contract-a"})
        second = _recovery(recovery_states=[{"experiment_contract_sha256": "contract-b"}])
        self.assert_incompatible(first, second, "experiment_contract_sha256")
        self.assert_incompatible(first, _recovery(), "experiment_contract_sha256")

    def test_matching_nested_and_top_level_metadata_remain_compatible(self):
        metadata = {"experiment_contract_sha256": "contract-a", "t_total_sec": 14400.0}
        nested = _recovery(recovery_states=[dict(metadata), dict(metadata)])
        for replays in ([nested, _recovery(**metadata)], [_recovery(**metadata), nested]):
            merged = merge_frozen_response_replays(replays, source="merge-test")
            for field, value in metadata.items():
                self.assertEqual(merged.manifest[field], value)

    def test_conflicting_metadata_within_one_manifest_is_rejected(self):
        replay = _recovery(recovery_states=[
            {"experiment_contract_sha256": "contract-a"},
            {"experiment_contract_sha256": "contract-b"},
        ])
        with self.assertRaisesRegex(ValueError, "experiment_contract_sha256"):
            merge_frozen_response_replays([replay], source="merge-test")

    def test_label_clues_prevent_legacy_interval_defaults(self):
        clues = (
            {"mc_gamma": 1.0},
            {"recovery_config": {"max_rollout_steps": 20, "recovery_depth": 0}},
            {"collector_format": "response_ddqn_recovery_replay_v1"},
            {"response_evaluation_mode": "recovery_aware_counterfactual_tree"},
            {"advantage_format_version": "response_aware_action0_advantage_replay_v1"},
            {"reward_mode": "advantage"},
            {"source": "rl_leader.convert_response_replay_to_mc"},
            {"done_semantics": "monte_carlo_targets_are_terminal"},
            {"reward_semantics": "terminal_best_recovery_return_negative_ttt"},
        )
        for metadata in clues:
            with self.subTest(metadata=metadata):
                self.assert_incompatible(_replay(), _replay(**metadata), "missing.*semantics")


if __name__ == "__main__":
    unittest.main()
