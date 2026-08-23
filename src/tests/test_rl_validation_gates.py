import unittest

from rl_leader.validate_contract_v4_dataset import REQUIRED_TARGETS, validate as validate_data
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


class RLValidationGateTest(unittest.TestCase):
    def test_dataset_gate_accepts_complete_contract(self):
        report = {
            "transitions": 10000,
            "validity_pass_fraction": 1.0,
            "response_contract": RL_RESPONSE_CONTRACT_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "optimizer_anchor_transition_contracts": ["rl_adapter_replay_v1"],
            "target_scenarios": {scenario: 1 for scenario in REQUIRED_TARGETS},
            "phase_transitions": {"recovery": 1},
            "action_support": {"dead_dimension_indices": []},
            "blocks": [
                {"family": "certificate", "owner": f"ramp_{index}", "release_true_fraction": 0.5}
                for index in range(4)
            ],
        }

        self.assertEqual(validate_data(report, 10000), [])

    def test_dataset_gate_rejects_adapter_bypass(self):
        failures = validate_data({
            "optimizer_anchor_transition_contracts": ["native_bypass"],
            "action_support": {"dead_dimension_indices": []},
            "blocks": [],
        }, 1)

        self.assertTrue(any("adapter replay" in failure for failure in failures))

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
