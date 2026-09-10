import json
import unittest

import numpy as np

from rl_leader.data_contract import (
    balanced_sampling_probabilities,
    PSTACK_DATASET_FORMAT,
    PSTACK_RESIDUAL_DATA_CONTRACT,
    continuous_actor_supervision_mask,
    iql_bootstrap_done,
    iql_training_rewards,
    pstack_residual_deployed_actions,
    pstack_residual_actor_supervision_mask,
    pstack_residual_trainable_dimension_mask,
    pstack_residual_targets,
    scenario_phase_cells,
    validate_pstack_residual_rows,
)
from rl_leader.validate_contract_v4_dataset import REQUIRED_TARGETS, validate as validate_data
from rl_leader.env import (
    OPTIMIZER_ANCHOR_CONTRACT,
    OPTIMIZER_ANCHOR_TRANSITION_CONTRACT,
    WARMUP_CONTROL_CONTRACT,
)
from rl_leader.eval_full_action import (
    _clip_action_to_support,
    observation_schema_adapter,
)
from rl_leader.experiment_contract import EXPERIMENT_PROFILE_ID
from rl_leader.validate_five_cell_policy import (
    REQUIRED_SCENARIOS,
    RECOVERY_GATED,
    validate as validate_policy,
)
from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    RL_RESPONSE_CONTRACT_VERSION,
)


def _anchored_fields(policy, deployed=None):
    policy = np.asarray(policy, dtype=np.float32)
    deployed = policy if deployed is None else np.asarray(deployed, dtype=np.float32)
    rows = policy.shape[0]
    return {
        "policy_residual": policy,
        "deployed_residual": deployed,
        "anchor_envelope": np.ones((rows, 2), dtype=np.float64),
        "anchor_selected_branch": np.asarray(["refined"] * rows),
        "anchor_fingerprint": np.asarray(["a" * 64] * rows),
    }


class RLValidationGateTest(unittest.TestCase):
    def test_exact_native_runtime_rejects_legacy_anchor_dataset_format(self):
        dataset = {
            **_anchored_fields([[0.0]], [[0.0]]),
            "pstack_anchor_pick_rl": np.asarray([0.0]),
        }
        manifest = {"format_version": "rl_coordination_dataset_v3_anchor_reference"}

        self.assertEqual(PSTACK_DATASET_FORMAT, "rl_coordination_dataset_v4_exact_native")
        with self.assertRaisesRegex(ValueError, "dataset format"):
            validate_pstack_residual_rows(dataset, manifest)

    def test_iql_bootstraps_external_collection_limits_but_not_true_terminal(self):
        dataset = {
            "done": np.asarray([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "termination_reason": np.asarray([
                "", "natural", "collection_time_limit", "wall_clock_abort",
                "safety_abort",
            ]),
        }

        np.testing.assert_array_equal(
            iql_bootstrap_done(dataset), [0.0, 1.0, 0.0, 0.0, 1.0]
        )

    def test_iql_removes_only_wall_clock_failure_cost_from_rewards(self):
        dataset = {
            "rew": np.asarray([-20.0, -5021.0, -5022.0], dtype=np.float32),
            "termination_reason": np.asarray([
                "natural", "wall_clock_abort", "solver_error",
            ]),
        }

        np.testing.assert_array_equal(
            iql_training_rewards(dataset, 5000.0), [-20.0, -21.0, -5022.0]
        )

    def test_full_evaluation_clips_policy_to_checkpoint_support(self):
        action, outside = _clip_action_to_support(
            np.asarray([-0.4, 0.2, 0.8]),
            np.asarray([-0.2, 0.0, 0.0]),
            np.asarray([0.1, 0.5, 0.0]),
        )

        np.testing.assert_allclose(action, [-0.2, 0.2, 0.0])
        np.testing.assert_array_equal(outside, [True, False, True])

    def test_evaluation_adapts_pure_observation_name_permutations(self):
        checkpoint_schema = {
            "version": "v1",
            "dimension": 3,
            "names": ["a", "b", "c"],
        }
        environment_schema = {
            "version": "v1",
            "dimension": 3,
            "names": ["c", "a", "b"],
        }

        indices, metadata = observation_schema_adapter(
            checkpoint_schema, environment_schema
        )

        np.testing.assert_array_equal(indices, [1, 2, 0])
        np.testing.assert_array_equal(np.asarray([30, 10, 20])[indices], [10, 20, 30])
        self.assertEqual(metadata["mode"], "name_permutation")
        self.assertEqual(metadata["moved_feature_count"], 3)

    def test_evaluation_rejects_observation_feature_drift(self):
        checkpoint_schema = {
            "version": "v1",
            "dimension": 2,
            "names": ["a", "b"],
        }
        environment_schema = {
            "version": "v1",
            "dimension": 2,
            "names": ["a", "c"],
        }

        with self.assertRaisesRegex(ValueError, "features do not match"):
            observation_schema_adapter(checkpoint_schema, environment_schema)

    def test_continuous_actor_excludes_pfo_and_nonreplayable_anchor_labels(self):
        dataset = {
            "behavior_mode": np.asarray([
                "optimizer_anchor", "optimizer_anchor", "optimizer_anchor", "full",
            ]),
            "teacher_pfo_selected": np.asarray([0.0, 1.0, 0.0, -1.0]),
            "response": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
            "teacher_response": np.asarray([[1.0], [2.0], [3.5], [np.nan]]),
        }

        mask = continuous_actor_supervision_mask(dataset)

        np.testing.assert_array_equal(mask, [True, False, False, True])

    def test_continuous_actor_fails_closed_for_legacy_anchor_labels(self):
        dataset = {
            "behavior_mode": np.asarray(["optimizer_anchor", "optimizer_local"]),
        }

        mask = continuous_actor_supervision_mask(dataset)

        np.testing.assert_array_equal(mask, [False, True])

    def test_pstack_residual_actor_uses_deployed_anchor_and_gate_accepted_local(self):
        dataset = {
            "behavior_mode": np.asarray([
                "optimizer_anchor", "optimizer_anchor",
                "optimizer_local", "optimizer_local", "full",
            ]),
            "teacher_pfo_selected": np.asarray([0.0, 1.0, 0.0, 1.0, -1.0]),
            "response": np.asarray([[1.0], [2.0], [30.0], [40.0], [50.0]]),
            "teacher_response": np.asarray([[1.0], [2.0], [3.0], [4.0], [np.nan]]),
            "act": np.asarray([[0.2], [0.4], [0.5], [0.7], [0.9]]),
            "anchor_action": np.asarray([[0.2], [0.4], [0.3], [0.4], [np.nan]]),
            "pstack_anchor_pick_rl": np.asarray([0.0, 0.0, 1.0, 1.0, -1.0]),
            **_anchored_fields(
                [[0.0], [0.0], [0.2], [0.3], [0.4]],
                [[0.0], [0.0], [0.2], [0.3], [0.0]],
            ),
        }

        mask = pstack_residual_actor_supervision_mask(dataset)
        targets = pstack_residual_targets(dataset)

        np.testing.assert_array_equal(mask, [True, True, True, True, False])
        np.testing.assert_allclose(targets[mask], [[0.0], [0.0], [0.2], [0.3]])

    def test_pstack_residual_actor_uses_only_long_horizon_positive_local_labels(self):
        dataset = {
            "behavior_mode": np.asarray([
                "optimizer_anchor", "optimizer_local", "optimizer_local",
                "optimizer_local",
            ]),
            "act": np.asarray([[0.2], [0.5], [0.7], [0.9]]),
            "anchor_action": np.asarray([[0.2], [0.3], [0.4], [0.5]]),
            "pstack_anchor_pick_rl": np.asarray([0.0, 1.0, 1.0, 1.0]),
            "long_horizon_label_valid": np.asarray([0.0, 1.0, 1.0, 0.0]),
            "long_horizon_positive": np.asarray([0.0, 1.0, 0.0, 0.0]),
            **_anchored_fields([[0.0], [0.2], [0.3], [0.4]]),
        }

        mask = pstack_residual_actor_supervision_mask(dataset)

        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_pstack_residual_critic_uses_only_actions_selected_by_anchor_gate(self):
        dataset = {
            "behavior_mode": np.asarray([
                "optimizer_anchor", "optimizer_local", "optimizer_local",
            ]),
            "act": np.asarray([[0.2], [0.7], [0.9]], dtype=np.float32),
            "anchor_action": np.asarray([[0.2], [0.4], [0.5]], dtype=np.float32),
            "pstack_anchor_pick_rl": np.asarray([0.0, 1.0, 0.0]),
            **_anchored_fields(
                [[0.0], [0.3], [0.4]],
                [[0.0], [0.3], [0.0]],
            ),
        }

        deployed = pstack_residual_deployed_actions(dataset)

        np.testing.assert_allclose(deployed, [[0.0], [0.3], [0.0]])

    def test_pstack_residual_freezes_discrete_certificate_dimensions(self):
        manifest = {
            "action_schema": {
                "names": ["budget.N_P", "urban.A.g_green", "certificate.R.release"],
            },
        }
        dataset = {
            "act": np.asarray([[0.3, 0.4, -0.8]], dtype=np.float32),
            "anchor_action": np.asarray([[0.1, 0.1, -1.0]], dtype=np.float32),
            "manifest_json": np.asarray(json.dumps(manifest)),
            "pstack_anchor_pick_rl": np.asarray([1.0]),
            **_anchored_fields(
                [[0.2, 0.3, 0.2]],
                [[0.2, 0.3, 0.2]],
            ),
        }

        trainable = pstack_residual_trainable_dimension_mask(dataset)
        targets = pstack_residual_targets(dataset)

        np.testing.assert_array_equal(trainable, [True, True, False])
        np.testing.assert_allclose(targets, [[0.2, 0.3, 0.0]])

    def test_pstack_residual_uses_persisted_delta_at_saturated_anchor(self):
        dataset = {
            "act": np.asarray([[1.0]], dtype=np.float32),
            "anchor_action": np.asarray([[1.0]], dtype=np.float32),
            "pstack_anchor_pick_rl": np.asarray([1.0]),
            **_anchored_fields([[0.2]], [[0.2]]),
        }

        np.testing.assert_allclose(pstack_residual_targets(dataset), [[0.2]])
        self.assertEqual(float(dataset["act"][0, 0] - dataset["anchor_action"][0, 0]), 0.0)

    def test_pstack_residual_rejects_legacy_flat_rows(self):
        dataset = {
            "act": np.asarray([[0.3]], dtype=np.float32),
            "anchor_action": np.asarray([[0.1]], dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "native-anchor residual fields"):
            pstack_residual_targets(dataset)

    def test_scenario_phase_balancing_gives_each_cell_equal_mass(self):
        dataset = {
            "episode": np.asarray([0, 0, 0, 1]),
            "simulation_time_sec": np.asarray([1000.0, 2000.0, 6000.0, 6000.0]),
        }
        manifest = {"episode_summaries": [
            {"episode": 0, "scenario": {"target_scenario": "170"}},
            {"episode": 1, "scenario": {"target_scenario": "190"}},
        ]}

        cells = scenario_phase_cells(dataset, manifest)
        probabilities = balanced_sampling_probabilities(cells)

        np.testing.assert_array_equal(cells, ["170|peak", "170|peak", "170|recovery", "190|recovery"])
        self.assertAlmostEqual(float(probabilities[:2].sum()), 1.0 / 3.0)
        self.assertAlmostEqual(float(probabilities[2]), 1.0 / 3.0)
        self.assertAlmostEqual(float(probabilities[3]), 1.0 / 3.0)

    def test_dataset_gate_accepts_complete_contract(self):
        report = {
            "experiment_contract_valid": True,
            "experiment_contract_sha256": ["0" * 64],
            "experiment_profiles": [EXPERIMENT_PROFILE_ID],
            "transitions": 10000,
            "validity_pass_fraction": 1.0,
            "response_contract": RL_RESPONSE_CONTRACT_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "optimizer_anchor_transition_contracts": [
                OPTIMIZER_ANCHOR_TRANSITION_CONTRACT
            ],
            "warmup_control_contracts": [WARMUP_CONTROL_CONTRACT],
            "optimizer_anchor_contracts": [OPTIMIZER_ANCHOR_CONTRACT],
            "pfo_supervisor_flags": [False],
            "pstack_anchor_flags": [True],
            "action_parameterization_supports": [PSTACK_RESIDUAL_DATA_CONTRACT],
            "optimizer_teacher_labels": {
                "labeled_transitions": 2,
                "leader_transitions": 2,
                "pfo_transitions": 0,
                "outer_pfo_transitions": 0,
                "replay_exact_fraction": 1.0,
                "leader_replay": {"exact_transitions": 1, "exact_fraction": 1.0},
                "pstack_selected_anchor_replay": {
                    "exact_transitions": 1,
                    "exact_fraction": 1.0,
                },
            },
            "continuous_actor_supervision": {"eligible_transitions": 1},
            "pstack_residual_actor_supervision": {
                "eligible_transitions": 1,
                "residual_abs_max": 0.2,
                "dead_dimension_indices": [],
                "nonzero_peak_transitions": 1,
                "nonzero_recovery_transitions": 1,
                "anchor_gate_rl_pick_transitions": 1,
                "frozen_dimension_indices": [2],
            },
            "target_scenarios": {scenario: 1 for scenario in REQUIRED_TARGETS},
            "phase_transitions": {"recovery": 1},
            "action_support": {"dead_dimension_indices": [2]},
            "blocks": [
                {"family": "certificate", "owner": f"ramp_{index}", "release_true_fraction": 0.5}
                for index in range(4)
            ],
        }

        self.assertEqual(validate_data(report, 10000), [])

    def test_dataset_gate_rejects_missing_experiment_contract(self):
        failures = validate_data({
            "experiment_contract_valid": False,
            "experiment_contract_sha256": [],
            "experiment_profiles": [],
            "action_support": {"dead_dimension_indices": []},
            "blocks": [],
        }, 1)

        self.assertTrue(any("experiment contracts" in failure for failure in failures))

    def test_dataset_gate_rejects_adapter_bypass(self):
        failures = validate_data({
            "optimizer_anchor_transition_contracts": ["native_bypass"],
            "action_support": {"dead_dimension_indices": []},
            "blocks": [],
        }, 1)

        self.assertTrue(any("anchor gate" in failure for failure in failures))

    def test_dataset_gate_rejects_nonreplayable_teacher_responses(self):
        failures = validate_data({
            "optimizer_teacher_labels": {
                "labeled_transitions": 10,
                "leader_transitions": 5,
                "pfo_transitions": 5,
                "replay_exact_fraction": 0.9,
                "leader_replay": {"exact_transitions": 0, "exact_fraction": 0.8},
            },
            "continuous_actor_supervision": {"eligible_transitions": 5},
            "pstack_residual_actor_supervision": {
                "eligible_transitions": 5,
                "residual_abs_max": 0.2,
                "dead_dimension_indices": [],
            },
            "action_support": {"dead_dimension_indices": []},
            "blocks": [],
        }, 1)

        self.assertTrue(any("zero residuals" in failure for failure in failures))

    def test_dataset_gate_rejects_recovery_only_residual_exploration(self):
        failures = validate_data({
            "pstack_residual_actor_supervision": {
                "eligible_transitions": 5,
                "residual_abs_max": 0.2,
                "dead_dimension_indices": [],
                "nonzero_peak_transitions": 0,
                "nonzero_recovery_transitions": 5,
            },
            "action_support": {"dead_dimension_indices": []},
            "blocks": [],
        }, 1)

        self.assertTrue(any("peak residual exploration" in failure for failure in failures))

    def test_policy_gate_accepts_frozen_thresholds(self):
        comparison = [
            {
                "scenario": scenario,
                "all_complete": "True",
                "all_valid": "True",
                "vs_pstack_percent": "-1.0",
                "support_out_mean": "0.10",
            }
            for scenario in REQUIRED_SCENARIOS
        ]
        phases = [
            {"scenario": scenario, "rl_recovery_vs_pstack_percent": "-2.0"}
            for scenario in RECOVERY_GATED
        ]

        self.assertEqual(validate_policy(comparison, phases, 0.10), [])

    def test_policy_gate_rejects_recovery_regression(self):
        comparison = [
            {
                "scenario": scenario,
                "all_complete": "True",
                "all_valid": "True",
                "vs_pstack_percent": "0.0",
                "support_out_mean": "0.0",
            }
            for scenario in REQUIRED_SCENARIOS
        ]
        phases = [
            {"scenario": scenario, "rl_recovery_vs_pstack_percent": "-2.1"}
            for scenario in RECOVERY_GATED
        ]

        failures = validate_policy(comparison, phases, 0.10)
        self.assertEqual(len(failures), len(RECOVERY_GATED))


if __name__ == "__main__":
    unittest.main()
