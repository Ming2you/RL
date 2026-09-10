from __future__ import annotations

import copy
import unittest

from rl_leader.oracle_label_contract import (
    DEPLOYMENT_RANKER_FEATURE_FIELDS,
    ORACLE_CANDIDATE_ROW_FORMAT,
    ORACLE_LABEL_DATASET_FORMAT,
    _validate_h3_selection_replay_evidence,
    validate_oracle_label_artifact,
)
from rl_leader.oracle_candidate_ablation import residual_sha256
from src.controllers.coordination import ACTION_SCHEMA_VERSION


SHA = "a" * 64


def _horizon_label(
    horizon: int,
    *,
    candidate_ttt: float,
    native_ttt: float,
    candidate_inventory: float,
    native_inventory: float,
    valid: bool = True,
):
    if not valid:
        return {"label_valid": False, "positive": False}
    gain = native_ttt - candidate_ttt
    required = max(0.1, 1.0e-3 * max(abs(native_ttt), 1.0))
    inventory_delta = candidate_inventory - native_inventory
    guard = inventory_delta <= 1.0e-6
    return {
        "label_valid": True,
        "candidate_steps": horizon,
        "native_steps": horizon,
        "candidate_ttt": candidate_ttt,
        "native_ttt": native_ttt,
        "ttt_gain": gain,
        "required_gain": required,
        "candidate_terminal_inventory": candidate_inventory,
        "native_terminal_inventory": native_inventory,
        "terminal_inventory_delta": inventory_delta,
        "inventory_guard_pass": guard,
        "validity_gate_pass": True,
        "positive": gain > required and guard,
        "candidate_follower_memory_sha256": SHA,
        "native_follower_memory_sha256": SHA,
    }


def _artifact(*, positive: bool = True):
    native_labels = {
        "1": _horizon_label(
            1, candidate_ttt=10.0, native_ttt=10.0,
            candidate_inventory=20.0, native_inventory=20.0,
        ),
        "12": _horizon_label(
            12, candidate_ttt=100.0, native_ttt=100.0,
            candidate_inventory=30.0, native_inventory=30.0,
        ),
    }
    coordination_labels = {
        "1": _horizon_label(
            1, candidate_ttt=9.9, native_ttt=10.0,
            candidate_inventory=19.0, native_inventory=20.0,
        ),
        "12": _horizon_label(
            12,
            candidate_ttt=90.0 if positive else 100.0,
            native_ttt=100.0,
            candidate_inventory=28.0,
            native_inventory=30.0,
        ),
    }
    return {
        "format_version": ORACLE_LABEL_DATASET_FORMAT,
        "row_format": ORACLE_CANDIDATE_ROW_FORMAT,
        "label_horizons": [1, 12],
        "decision_horizon": 12,
        "inventory_guard_tolerance_veh": 1.0e-6,
        "action_schema": {
            "version": ACTION_SCHEMA_VERSION,
            "dimension": 3,
            "names": ["a", "b", "c"],
        },
        "ranker_feature_fields": list(DEPLOYMENT_RANKER_FEATURE_FIELDS),
        "legacy_migration_allowed": False,
        "implementation_sha256": {"oracle.py": SHA},
        "candidate_pools": [{
            "candidate_pool_id": "sweet_170_w60:13",
            "native_anchor_branch": "fallback_pfo",
            "anchor_fingerprint": SHA,
            "pre_runtime_sha256": SHA,
            "forecast_sha256": SHA,
            "probe_order_verified": True,
            "source_isolation_verified": True,
            "identity_parity": {"passed": True},
            "zero_round_trip": {"passed": True},
            "candidate_manifest_sha256": SHA,
            "selected_h12_candidate_ids": ["coordination-zero"],
            "rows": [
                {
                    "candidate_id": "native",
                    "execution_branch": "native_anchor",
                    "candidate_family": "native_anchor",
                    "continuous_residual_valid": False,
                    "continuous_residual": None,
                    "validity_gate_pass": True,
                    "horizon_labels": native_labels,
                    "ranker_eligible": True,
                    "oracle_is_winner": not positive,
                },
                {
                    "candidate_id": "coordination-zero",
                    "execution_branch": "coordination",
                    "candidate_family": "coordination_zero",
                    "continuous_residual_valid": True,
                    "continuous_residual": [0.0, 0.0, 0.0],
                    "residual_sha256": residual_sha256([0.0, 0.0, 0.0]),
                    "post_follower_sha256": SHA,
                    "post_physical_sha256": SHA,
                    "candidate_response": [1.0, 2.0],
                    "requested_budget": {
                        "N_P_star": 1.0,
                        "N_UF_star": 2.0,
                        "raw_budget": [1.0, 2.0],
                    },
                    "budget_intent_checks": {
                        "N_P_star_exact": True,
                        "N_UF_star_exact": True,
                        "raw_budget_exact": True,
                        "certificate_exact": True,
                    },
                    "h1_replay_evidence": {
                        "probe_response": [1.0, 2.0],
                        "rollout_response": [1.0, 2.0],
                        "response_linf": 0.0,
                        "response_tolerance": 1.0e-6,
                        "response_exact": True,
                        "probe_follower_memory_sha256": SHA,
                        "rollout_follower_memory_sha256": SHA,
                        "follower_memory_exact": True,
                        "probe_physical_sha256": SHA,
                        "rollout_physical_sha256": SHA,
                        "physical_exact": True,
                        "probe_step_ttt": 9.9,
                        "rollout_step_ttt": 9.9,
                        "step_ttt_exact": True,
                        "probe_terminal_inventory": 19.0,
                        "rollout_terminal_inventory": 19.0,
                        "terminal_inventory_exact": True,
                        "probe_validity_gate_pass": True,
                        "rollout_validity_gate_pass": True,
                        "validity_gate_exact": True,
                        "passed": True,
                    },
                    "validity_gate_pass": True,
                    "horizon_labels": coordination_labels,
                    "ranker_eligible": True,
                    "oracle_is_winner": positive,
                },
            ],
        }],
    }


class OracleLabelContractTest(unittest.TestCase):
    def test_fallback_pfo_and_coordination_zero_are_distinct_valid_branches(self):
        self.assertEqual(
            validate_oracle_label_artifact(_artifact()),
            {"candidate_pools": 1, "rows": 2, "positive_rows": 1},
        )

    def test_native_anchor_cannot_be_encoded_as_zero_residual(self):
        artifact = _artifact()
        native = artifact["candidate_pools"][0]["rows"][0]
        native["continuous_residual_valid"] = True
        native["continuous_residual"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "native anchor must use a null"):
            validate_oracle_label_artifact(artifact)

    def test_coordination_zero_requires_exact_finite_zero_vector(self):
        artifact = _artifact()
        row = artifact["candidate_pools"][0]["rows"][1]
        row["continuous_residual"][1] = 0.1
        with self.assertRaisesRegex(ValueError, "exact zero residual"):
            validate_oracle_label_artifact(artifact)

    def test_missing_horizon_or_memory_hash_fails_closed(self):
        artifact = _artifact()
        row = artifact["candidate_pools"][0]["rows"][1]
        row["horizon_labels"].pop("1")
        with self.assertRaisesRegex(ValueError, "horizon labels are incomplete"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        label = artifact["candidate_pools"][0]["rows"][1]["horizon_labels"]["12"]
        label["candidate_follower_memory_sha256"] = "missing"
        with self.assertRaisesRegex(ValueError, "memory hash is invalid"):
            validate_oracle_label_artifact(artifact)

    def test_inventory_debt_cannot_be_labeled_positive(self):
        artifact = _artifact()
        label = artifact["candidate_pools"][0]["rows"][1]["horizon_labels"]["12"]
        label["candidate_terminal_inventory"] = 31.0
        label["terminal_inventory_delta"] = 1.0
        label["inventory_guard_pass"] = False
        label["positive"] = True
        with self.assertRaisesRegex(ValueError, "positive label is inconsistent"):
            validate_oracle_label_artifact(artifact)

    def test_no_positive_pool_must_choose_native_anchor(self):
        artifact = _artifact(positive=False)
        self.assertEqual(
            validate_oracle_label_artifact(artifact)["positive_rows"], 0
        )
        artifact["candidate_pools"][0]["rows"][0]["oracle_is_winner"] = False
        artifact["candidate_pools"][0]["rows"][1]["oracle_is_winner"] = True
        with self.assertRaisesRegex(ValueError, "fall back to native anchor"):
            validate_oracle_label_artifact(artifact)

    def test_required_decision_horizon_rejects_native_only_labels(self):
        artifact = _artifact(positive=False)
        row = artifact["candidate_pools"][0]["rows"][1]
        row["horizon_labels"]["12"] = {"label_valid": False, "positive": False}
        row["ranker_eligible"] = False
        row["h1_replay_evidence"] = None
        artifact["candidate_pools"][0]["selected_h12_candidate_ids"] = []
        with self.assertRaisesRegex(ValueError, "coordination decision-horizon"):
            validate_oracle_label_artifact(
                artifact, require_decision_horizon=True
            )

    def test_privileged_or_legacy_contracts_are_rejected(self):
        artifact = _artifact()
        artifact["ranker_feature_fields"].append("pcent_response")
        with self.assertRaisesRegex(ValueError, "ranker feature contract mismatch"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        artifact["legacy_migration_allowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot be migrated"):
            validate_oracle_label_artifact(artifact)

    def test_direct_pcent_cannot_be_an_executable_candidate(self):
        artifact = _artifact()
        artifact["candidate_pools"][0]["rows"][1]["candidate_family"] = "direct_pcent"
        with self.assertRaisesRegex(ValueError, "invalid family"):
            validate_oracle_label_artifact(artifact)

    def test_pool_provenance_is_required_fail_closed(self):
        for field in (
            "pre_runtime_sha256",
            "forecast_sha256",
            "source_isolation_verified",
            "identity_parity",
            "zero_round_trip",
            "candidate_manifest_sha256",
        ):
            with self.subTest(field=field):
                artifact = _artifact()
                artifact["candidate_pools"][0].pop(field)
                with self.assertRaisesRegex(ValueError, "provenance|isolation|parity"):
                    validate_oracle_label_artifact(artifact)

    def test_coordination_row_provenance_is_required_fail_closed(self):
        for field in (
            "residual_sha256",
            "post_follower_sha256",
            "post_physical_sha256",
            "requested_budget",
            "budget_intent_checks",
        ):
            with self.subTest(field=field):
                artifact = _artifact()
                artifact["candidate_pools"][0]["rows"][1].pop(field)
                with self.assertRaisesRegex(ValueError, "provenance|budget|hash"):
                    validate_oracle_label_artifact(artifact)

    def test_selected_h12_row_requires_exact_h1_replay_evidence(self):
        artifact = _artifact()
        artifact["candidate_pools"][0]["rows"][1].pop("h1_replay_evidence")
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        row = artifact["candidate_pools"][0]["rows"][1]
        row["post_follower_sha256"] = "b" * 64
        evidence = row["h1_replay_evidence"]
        evidence["probe_follower_memory_sha256"] = "b" * 64
        evidence["rollout_follower_memory_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        evidence = artifact["candidate_pools"][0]["rows"][1]["h1_replay_evidence"]
        evidence["probe_physical_sha256"] = "b" * 64
        evidence["rollout_physical_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        evidence = artifact["candidate_pools"][0]["rows"][1]["h1_replay_evidence"]
        evidence["probe_follower_memory_sha256"] = "b" * 64
        evidence["rollout_follower_memory_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        evidence = artifact["candidate_pools"][0]["rows"][1]["h1_replay_evidence"]
        evidence["probe_response"] = [9.0, 9.0]
        evidence["rollout_response"] = [9.0, 9.0]
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        evidence = artifact["candidate_pools"][0]["rows"][1]["h1_replay_evidence"]
        evidence["rollout_follower_memory_sha256"] = "b" * 64
        evidence["follower_memory_exact"] = False
        evidence["passed"] = False
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

        artifact = _artifact()
        evidence = artifact["candidate_pools"][0]["rows"][1]["h1_replay_evidence"]
        evidence["probe_step_ttt"] = 123.0
        evidence["rollout_step_ttt"] = 123.0
        with self.assertRaisesRegex(ValueError, "H1 replay"):
            validate_oracle_label_artifact(artifact)

    def test_h3_selection_replay_binds_metrics_and_hashes(self):
        label = _horizon_label(
            3,
            candidate_ttt=30.0,
            native_ttt=31.0,
            candidate_inventory=40.0,
            native_inventory=41.0,
        )
        label["candidate_physical_state_sha256"] = SHA
        label["native_physical_state_sha256"] = SHA
        evidence = {
            "selector_ttt": 30.0,
            "rollout_ttt": 30.0,
            "ttt_exact": True,
            "selector_terminal_inventory": 40.0,
            "rollout_terminal_inventory": 40.0,
            "terminal_inventory_exact": True,
            "selector_follower_memory_sha256": SHA,
            "rollout_follower_memory_sha256": SHA,
            "follower_memory_exact": True,
            "selector_physical_sha256": SHA,
            "rollout_physical_sha256": SHA,
            "physical_exact": True,
            "selector_validity_gate_pass": True,
            "rollout_validity_gate_pass": True,
            "validity_gate_exact": True,
            "passed": True,
        }
        _validate_h3_selection_replay_evidence(
            evidence, "candidate", expected_h3_label=label
        )
        evidence["selector_ttt"] = 29.0
        with self.assertRaisesRegex(ValueError, "unbound"):
            _validate_h3_selection_replay_evidence(
                evidence, "candidate", expected_h3_label=label
            )

if __name__ == "__main__":
    unittest.main()
