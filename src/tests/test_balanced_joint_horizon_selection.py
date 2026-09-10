from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl_leader.evaluate_balanced_joint_horizons import (
    _assert_matching_sources,
    _merge_residual_specs,
    _source_pool_and_event,
    _validate_recorded_dependencies,
    evaluate_joint_horizons,
    joint_candidate_specs,
)
from rl_leader.oracle_candidate_ablation import residual_sha256


def _row(candidate_id, residual, *, h12=True, distance=0.2):
    return {
        "candidate_id": candidate_id,
        "continuous_residual": residual,
        "native_response_distance": {"overall_rmse": distance},
        "h12_label": {"positive": False} if h12 else None,
    }


class JointCandidateTests(unittest.TestCase):
    def test_crosses_only_nonidentity_h12_representatives(self):
        urban = {"outcomes": [
            _row("structured:urban:A:axis:a", [0.25, 0.0, 0.0]),
            _row("structured:urban:B:axis:b", [0.5, 0.0, 0.0], h12=False),
            _row("structured:urban:C:axis:c", [-0.25, 0.0, 0.0], distance=0.0),
        ]}
        freeway = {"outcomes": [
            _row("structured:freeway:R_D_W:axis:a", [0.0, 0.25, 0.0]),
            _row("structured:freeway:R_F_W:axis:b", [0.0, 0.0, -0.5]),
        ]}

        specs = joint_candidate_specs(urban, freeway)

        self.assertEqual(len(specs), 2)
        self.assertEqual(
            {tuple(row["continuous_residual"]) for row in specs},
            {(0.25, 0.25, 0.0), (0.25, 0.0, -0.5)},
        )

    def test_component_blocks_must_be_disjoint(self):
        urban = {"outcomes": [
            _row("structured:urban:A:axis:a", [0.25, 0.0]),
        ]}
        freeway = {"outcomes": [
            _row("structured:freeway:R_D_W:axis:a", [-0.25, 0.0]),
        ]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            joint_candidate_specs(urban, freeway)

    def test_identical_residuals_preserve_all_component_pairs(self):
        specs = [
            {
                "candidate_id": "joint:urban+freeway:same",
                "component_candidate_ids": ["urban-a", "freeway-a"],
                "continuous_residual": [0.25, -0.25],
            },
            {
                "candidate_id": "joint:urban+freeway:same",
                "component_candidate_ids": ["urban-alias", "freeway-alias"],
                "continuous_residual": [0.25, -0.25],
            },
        ]
        merged = _merge_residual_specs(specs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["component_aliases"], [
            ["urban-a", "freeway-a"],
            ["urban-alias", "freeway-alias"],
        ])

    def test_source_provenance_must_match(self):
        common = {
            "scenario": "sweet_155_w60",
            "stratum": "plateau",
            "policy_step": 9,
            "anchor_fingerprint": "anchor",
            "source_manifest_sha256": "manifest",
        }
        mismatch = dict(common, policy_step=10)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            _assert_matching_sources(common, mismatch)

    def test_recorded_dependency_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            h1 = root / "h1.json"
            manifest = root / "manifest.json"
            h1.write_text("{}", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            source = {
                "source_h1_artifact": str(h1),
                "source_h1_sha256": hashlib.sha256(h1.read_bytes()).hexdigest(),
                "source_manifest": str(manifest),
                "source_manifest_sha256": "wrong",
            }
            with self.assertRaisesRegex(ValueError, "manifest SHA mismatch"):
                _validate_recorded_dependencies(source, "urban")

    def test_h1_pool_must_bind_to_source_and_frozen_event(self):
        pool = {
            "scenario": "sweet_155_w60",
            "policy_step": 9,
            "simulation_step": 109,
            "simulation_time_sec": 2520.0,
            "native_anchor_branch": "refined",
            "pre_runtime_sha256": "runtime",
            "forecast_sha256": "forecast",
            "anchor_fingerprint": "anchor",
            "observation": [1.0],
            "anchor_envelope": [2.0],
        }
        from rl_leader.diagnose_phase0_parity import _digest
        event = {
            **{key: pool[key] for key in (
                "policy_step", "simulation_step", "simulation_time_sec",
                "native_anchor_branch", "pre_runtime_sha256", "forecast_sha256",
                "anchor_fingerprint",
            )},
            "stratum": "plateau",
            "coordination_eligible": True,
            "observation_sha256": _digest(pool["observation"]),
            "anchor_envelope_sha256": _digest(pool["anchor_envelope"]),
        }
        source = {
            "scenario": pool["scenario"],
            "policy_step": 9,
            "stratum": "plateau",
            "anchor_fingerprint": "wrong",
        }
        h1 = {
            "implementation_sha256": {"core": "sha"},
            "candidate_pools": [pool],
        }
        manifest = {
            "scenarios": [{"scenario": pool["scenario"], "events": [event]}]
        }
        with patch(
            "rl_leader.evaluate_balanced_joint_horizons._implementation_fingerprints",
            return_value={"core": "sha"},
        ):
            with self.assertRaisesRegex(ValueError, "frozen event mismatch"):
                _source_pool_and_event(h1, manifest, source, "urban")


class JointEvaluationContractTests(unittest.TestCase):
    def test_positive_output_is_accepted_by_drain_out_validator(self):
        module = "rl_leader.evaluate_balanced_joint_horizons"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest_path = root / "manifest.json"
            output_path = root / "joint.json"
            pool_base = {
                "scenario": "sweet_155_w60",
                "policy_step": 9,
                "simulation_step": 109,
                "simulation_time_sec": 2520.0,
                "native_anchor_branch": "refined",
                "pre_runtime_sha256": "runtime",
                "forecast_sha256": "forecast",
                "anchor_fingerprint": "anchor",
                "observation": [1.0],
                "anchor_envelope": [2.0],
            }
            from rl_leader.diagnose_phase0_parity import _digest
            event = {
                **{key: pool_base[key] for key in (
                    "policy_step", "simulation_step", "simulation_time_sec",
                    "native_anchor_branch", "pre_runtime_sha256",
                    "forecast_sha256", "anchor_fingerprint",
                )},
                "stratum": "plateau",
                "coordination_eligible": True,
                "observation_sha256": _digest(pool_base["observation"]),
                "anchor_envelope_sha256": _digest(pool_base["anchor_envelope"]),
            }
            manifest = {
                "scenarios": [{
                    "scenario": pool_base["scenario"], "events": [event]
                }]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            h1_paths = {}
            for domain in ("urban", "freeway"):
                path = root / f"{domain}-h1.json"
                h1 = {
                    "implementation_sha256": {"core": "sha"},
                    "candidate_pools": [{**pool_base, "rows": []}],
                }
                path.write_text(json.dumps(h1), encoding="utf-8")
                h1_paths[domain] = path

            urban_residual = [0.25, 0.0, 0.0, 0.0]
            freeway_residual = [0.0, -0.25, 0.0, 0.0]
            source_paths = {}
            for domain, residual in (
                ("urban", urban_residual), ("freeway", freeway_residual)
            ):
                candidate_id = (
                    f"structured:{domain}:owner:axis:"
                    f"{residual_sha256(residual)[:12]}"
                )
                evaluator = Path(module.replace(".", "/")).parent / (
                    "evaluate_balanced_horizons.py"
                    if domain == "urban"
                    else "evaluate_balanced_freeway_horizons.py"
                )
                source = {
                    "format_version": "balanced_owner_block_h3_selective_h12_v1",
                    "candidate_domain": None if domain == "urban" else "freeway",
                    "source_h1_artifact": str(h1_paths[domain]),
                    "source_h1_sha256": hashlib.sha256(
                        h1_paths[domain].read_bytes()
                    ).hexdigest(),
                    "source_manifest": str(manifest_path),
                    "source_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "implementation_sha256": {"core": "sha"},
                    "sidecar_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
                    "scenario": pool_base["scenario"],
                    "stratum": "plateau",
                    "policy_step": 9,
                    "anchor_fingerprint": "anchor",
                    "selector_version": "v2",
                    "outcomes": [{
                        "candidate_id": candidate_id,
                        "continuous_residual": residual,
                        "native_response_distance": {"overall_rmse": 0.2},
                        "h1_replay_exact": True,
                        "h3_label": {"positive": False},
                        "h12_label": {"positive": False},
                        "h3_to_h12_replay": {"passed": True},
                    }],
                    "passed": True,
                }
                path = root / f"{domain}.json"
                path.write_text(json.dumps(source), encoding="utf-8")
                source_paths[domain] = path

            native = {
                "first_step_response": [0.0, 0.0],
                "checkpoints": {
                    str(horizon): {
                        "follower_memory_sha256": "native-follower",
                        "physical_state_sha256": "native-physical",
                        "ttt": 10.0,
                        "terminal_inventory": 2.0,
                        "validity_gate_pass": True,
                    }
                    for horizon in (1, 3, 12)
                },
            }
            candidate = {
                "first_step_response": [1.0, 0.0],
                "checkpoints": {
                    str(horizon): {
                        "follower_memory_sha256": "candidate-follower",
                        "physical_state_sha256": "candidate-physical",
                        "ttt": 9.0,
                        "terminal_inventory": 1.0,
                        "validity_gate_pass": True,
                    }
                    for horizon in (1, 3, 12)
                },
            }
            with (
                patch(f"{module}._implementation_fingerprints", return_value={"core": "sha"}),
                patch(f"{module}.validate_oracle_label_artifact"),
                patch(f"{module}.validate_frozen_manifest"),
                patch(
                    f"{module}._replay_event",
                    return_value=(object(), SimpleNamespace(anchor_fingerprint="anchor")),
                ),
                patch(f"{module}._rollout_pstack", return_value=native),
                patch(f"{module}._rollout_price_candidate", return_value=candidate),
                patch(f"{module}._require_rollout_coverage"),
                patch(f"{module}._response_scales_and_families", return_value=(None, None)),
                patch(f"{module}.response_distance", return_value={"overall_rmse": 0.2}),
                patch(
                    f"{module}._horizon_label",
                    side_effect=[
                        {"positive": False, "validity_gate_pass": True},
                        {"positive": True, "validity_gate_pass": True},
                    ],
                ),
                patch(
                    f"{module}._h3_selection_replay_evidence",
                    return_value={"passed": True},
                ),
            ):
                result = evaluate_joint_horizons(
                    source_paths["urban"], source_paths["freeway"], output_path
                )

            self.assertEqual(
                result["outcomes"][0]["representative_candidate_id"],
                result["outcomes"][0]["candidate_id"],
            )
            self.assertEqual(
                result["outcomes"][0]["response_memory_outcome_sha256"],
                "candidate-follower",
            )
            from rl_leader.diagnose_balanced_drain_out import _validate_source
            with (
                patch(
                    "rl_leader.diagnose_balanced_drain_out._implementation_fingerprints",
                    return_value={"core": "sha"},
                ),
                patch(
                    "rl_leader.evaluate_balanced_joint_horizons._implementation_fingerprints",
                    return_value={"core": "sha"},
                ),
                patch("rl_leader.diagnose_balanced_drain_out.validate_oracle_label_artifact"),
                patch("rl_leader.diagnose_balanced_drain_out.validate_frozen_manifest"),
            ):
                *_, positives = _validate_source(output_path)
            self.assertEqual(len(positives), 1)

            unknown = json.loads(output_path.read_text(encoding="utf-8"))
            unknown["contract"] = "unknown-producer"
            output_path.write_text(json.dumps(unknown), encoding="utf-8")
            with patch(
                "rl_leader.diagnose_balanced_drain_out._implementation_fingerprints",
                return_value={"core": "sha"},
            ):
                with self.assertRaisesRegex(ValueError, "unknown balanced source contract"):
                    _validate_source(output_path)


if __name__ == "__main__":
    unittest.main()
