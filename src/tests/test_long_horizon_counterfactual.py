import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rl_leader.generate_long_horizon_labels import (
    _process_file,
    accepted_local_indices,
    counterfactual_verdict,
    replay_reward_parity,
)
from rl_leader.env import RLLeaderEnv
from rl_leader.relabel_long_horizon import (
    build_label_arrays,
    quarantined_missing_mask,
)


class LongHorizonCounterfactualTest(unittest.TestCase):
    @staticmethod
    def _manifest_with_contract(episodes):
        env = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        contract = env.experiment_contract
        return {
            "experiment_contracts": {contract.sha256: contract.payload},
            "episode_summaries": [
                {
                    "episode": episode,
                    "scenario": {},
                    "experiment_contract_sha256": contract.sha256,
                }
                for episode in episodes
            ],
        }

    def test_optimizer_anchor_preview_does_not_mutate_live_state(self):
        class PreviewProbe(RLLeaderEnv):
            def __init__(self):
                self.hidden_decision_state = 0

            def _optimizer_decision(self, *, sync_follower_state=False):
                self.hidden_decision_state += 1
                encoded = np.asarray([self.hidden_decision_state], dtype=np.float32)
                return None, None, None, None, encoded

        env = PreviewProbe()

        np.testing.assert_array_equal(env.optimizer_anchor_action(), [1.0])
        self.assertEqual(env.hidden_decision_state, 0)

    def test_only_gate_accepted_local_rows_are_replayed(self):
        dataset = {
            "behavior_mode": np.asarray([
                "optimizer_anchor", "optimizer_local", "optimizer_local",
            ]),
            "pstack_anchor_pick_rl": np.asarray([1.0, 1.0, 0.0]),
        }

        np.testing.assert_array_equal(accepted_local_indices(dataset), [1])

    def test_relabel_arrays_validate_row_identity_and_keep_negative_labels(self):
        dataset = {
            "obs": np.zeros((2, 1), dtype=np.float32),
            "behavior_mode": np.asarray(["optimizer_local", "optimizer_local"]),
            "pstack_anchor_pick_rl": np.asarray([1.0, 1.0]),
            "episode": np.asarray([3, 3]),
            "step": np.asarray([7, 8]),
        }
        labels = [{
            "source_row_index": 1,
            "episode": 3,
            "step": 8,
            "long_horizon_positive": False,
            "ttt_gain": -2.0,
            "required_gain": 0.5,
            "terminal_inventory_delta": 4.0,
            "rollout_steps": 12,
        }]

        arrays = build_label_arrays(dataset, labels)

        np.testing.assert_array_equal(arrays["long_horizon_label_valid"], [0.0, 1.0])
        np.testing.assert_array_equal(arrays["long_horizon_positive"], [0.0, 0.0])
        self.assertAlmostEqual(float(arrays["long_horizon_ttt_gain"][1]), -2.0)

    def test_quarantine_covers_only_unlabeled_accepted_rows_in_failed_episode(self):
        dataset = {
            "behavior_mode": np.asarray([
                "optimizer_local", "optimizer_local", "optimizer_anchor",
            ]),
            "pstack_anchor_pick_rl": np.asarray([1.0, 1.0, 0.0]),
            "episode": np.asarray([3, 4, 3]),
        }
        label_arrays = {
            "long_horizon_label_valid": np.asarray([0.0, 1.0, 0.0]),
        }

        mask = quarantined_missing_mask(dataset, label_arrays, {3})

        np.testing.assert_array_equal(mask, [True, False, False])

    def test_generator_isolates_episode_replay_errors(self):
        manifest = self._manifest_with_contract((0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.npz"
            np.savez_compressed(
                path,
                behavior_mode=np.asarray(["optimizer_local", "optimizer_local"]),
                pstack_anchor_pick_rl=np.asarray([1.0, 1.0]),
                episode=np.asarray([0, 1]),
                manifest_json=np.asarray(json.dumps(manifest)),
            )
            with patch(
                "rl_leader.generate_long_horizon_labels.validate_pstack_residual_rows"
            ), patch(
                "rl_leader.generate_long_horizon_labels._process_episode",
                side_effect=[RuntimeError("bad replay"), [{"episode": 1}]],
            ):
                result = _process_file(str(path), 12, (1, 3, 6, 12), 1.0e-4, 0)

        self.assertEqual(result["labels"], [{"episode": 1}])
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["episode"], 0)

    def test_generator_can_retry_only_the_failed_episode(self):
        manifest = self._manifest_with_contract((0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.npz"
            np.savez_compressed(
                path,
                behavior_mode=np.asarray(["optimizer_local", "optimizer_local"]),
                pstack_anchor_pick_rl=np.asarray([1.0, 1.0]),
                episode=np.asarray([0, 1]),
                manifest_json=np.asarray(json.dumps(manifest)),
            )
            with patch(
                "rl_leader.generate_long_horizon_labels.validate_pstack_residual_rows"
            ), patch(
                "rl_leader.generate_long_horizon_labels._process_episode",
                return_value=[{"episode": 1}],
            ) as process_episode:
                result = _process_file(
                    str(path), 12, (1, 3, 6, 12), 1.0e-4, 0, (1,)
                )

        self.assertEqual(result["labels"], [{"episode": 1}])
        self.assertEqual(process_episode.call_count, 1)
        self.assertEqual(process_episode.call_args.args[2], 1)

    def test_generator_rejects_missing_experiment_contract(self):
        manifest = {
            "episode_summaries": [{"episode": 0, "scenario": {}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.npz"
            np.savez_compressed(
                path,
                behavior_mode=np.asarray(["optimizer_local"]),
                pstack_anchor_pick_rl=np.asarray([1.0]),
                episode=np.asarray([0]),
                manifest_json=np.asarray(json.dumps(manifest)),
            )
            with self.assertRaisesRegex(ValueError, "missing verified"):
                _process_file(str(path), 12, (1, 3, 6, 12), 1.0e-4, 0)

    def test_relabel_rejects_episode_contract_mismatch(self):
        dataset = {
            "obs": np.zeros((1, 1), dtype=np.float32),
            "behavior_mode": np.asarray(["optimizer_local"]),
            "pstack_anchor_pick_rl": np.asarray([1.0]),
            "episode": np.asarray([3]),
            "step": np.asarray([7]),
        }
        labels = [{
            "source_row_index": 0,
            "episode": 3,
            "step": 7,
            "experiment_contract_sha256": "wrong",
            "long_horizon_positive": False,
            "ttt_gain": -1.0,
            "required_gain": 0.1,
            "terminal_inventory_delta": 0.0,
            "rollout_steps": 12,
        }]

        with self.assertRaisesRegex(ValueError, "experiment contract mismatch"):
            build_label_arrays(dataset, labels, {3: "expected"})

    def test_wall_clock_failure_cost_is_removed_from_reward_parity(self):
        parity = replay_reward_parity(
            saved_reward=-5021.123046875,
            replay_reward=-21.12326369148,
            termination_reason="wall_clock_abort",
            failure_cost=5000.0,
            parity_tolerance=1.0e-4,
        )

        self.assertTrue(parity["passes"])
        self.assertAlmostEqual(parity["adjustment"], 5000.0)
        self.assertGreater(parity["raw_error"], 4999.0)

    def test_natural_reward_does_not_receive_failure_cost_adjustment(self):
        parity = replay_reward_parity(
            saved_reward=-5021.123046875,
            replay_reward=-21.12326369148,
            termination_reason="natural",
            failure_cost=5000.0,
            parity_tolerance=1.0e-4,
        )

        self.assertFalse(parity["passes"])
        self.assertEqual(parity["adjustment"], 0.0)

    def test_short_gain_with_long_loss_is_false_positive(self):
        verdict = counterfactual_verdict(
            candidate_ttt=110.0,
            pstack_ttt=100.0,
            candidate_terminal_inventory=40.0,
            pstack_terminal_inventory=35.0,
        )

        self.assertFalse(verdict["long_horizon_positive"])
        self.assertAlmostEqual(verdict["ttt_gain"], -10.0)
        self.assertTrue(verdict["inventory_blocked"])

    def test_material_ttt_gain_without_inventory_increase_is_positive(self):
        verdict = counterfactual_verdict(
            candidate_ttt=98.0,
            pstack_ttt=100.0,
            candidate_terminal_inventory=34.0,
            pstack_terminal_inventory=35.0,
        )

        self.assertTrue(verdict["long_horizon_positive"])
        self.assertAlmostEqual(verdict["required_gain"], 0.1)


if __name__ == "__main__":
    unittest.main()
