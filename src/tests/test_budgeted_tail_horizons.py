from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.evaluate_budgeted_tail_horizons import (
    BUDGETED_SELECTOR_VERSION,
    BUDGETED_TAIL_DOMAIN,
    BUDGETED_TAIL_HORIZON_CONTRACT,
    budgeted_candidate_specs,
    evaluate_budgeted_tail_horizons,
)


def _row(
    candidate_id: str,
    gain: float,
    residual: list[float],
    *,
    outcome: str | None = None,
    physical: str | None = None,
):
    return {
        "candidate_id": candidate_id,
        "execution_branch": "coordination",
        "candidate_aliases": [candidate_id],
        "continuous_residual_valid": True,
        "continuous_residual": residual,
        "candidate_response": [1.0, 0.0],
        "response_memory_outcome_sha256": outcome or candidate_id,
        "post_follower_sha256": f"follower-{candidate_id}",
        "post_physical_sha256": physical or f"physical-{candidate_id}",
        "h1_step_ttt": 1.0,
        "h1_terminal_inventory": 1.0,
        "validity_gate_pass": True,
        "horizon_labels": {
            "1": {
                "label_valid": True,
                "validity_gate_pass": True,
                "positive": gain > 0.1,
                "ttt_gain": gain,
                "terminal_inventory_delta": 0.0,
            }
        },
    }


def _rollout(first_response=(1.0, 0.0)):
    checkpoints = {
        "1": {
            "ttt": 1.0,
            "terminal_inventory": 1.0,
            "follower_memory_sha256": "follower",
            "physical_state_sha256": "physical",
            "validity_gate_pass": True,
        },
        "3": {
            "ttt": 3.0,
            "terminal_inventory": 1.0,
            "follower_memory_sha256": "follower-h3",
            "physical_state_sha256": "physical-h3",
            "validity_gate_pass": True,
        },
        "12": {
            "ttt": 12.0,
            "terminal_inventory": 1.0,
            "follower_memory_sha256": "follower-h12",
            "physical_state_sha256": "physical-h12",
            "validity_gate_pass": True,
        },
    }
    return {
        "first_step_response": list(first_response),
        "checkpoints": checkpoints,
    }


class BudgetedCandidateSelectionTests(unittest.TestCase):
    def test_dedupes_h1_outcomes_and_prefers_stronger_representative(self):
        weak = _row(
            "structured:urban:A:axis:weak",
            0.1,
            [0.25, 0.0, 0.0],
            outcome="same",
            physical="same-physical",
        )
        strong = _row(
            "structured:urban:A:axis:strong",
            2.0,
            [0.5, 0.0, 0.0],
            outcome="same",
            physical="same-physical",
        )
        freeway = _row(
            "structured:freeway:R_D_W:axis:strong",
            1.0,
            [0.0, 0.5, 0.0],
        )

        specs = budgeted_candidate_specs(
            {"rows": [weak, strong, freeway]},
            max_h12_candidates_per_event=3,
            domain_pool_size=1,
        )

        component_ids = [
            tuple(spec["component_candidate_ids"]) for spec in specs
        ]
        self.assertNotIn((weak["candidate_id"],), component_ids)
        self.assertIn((strong["candidate_id"],), component_ids)

    def test_singles_are_prioritized_before_joint_candidates(self):
        urban = _row(
            "structured:urban:A:axis:best",
            3.0,
            [0.5, 0.0, 0.0],
        )
        freeway = _row(
            "structured:freeway:R_D_W:axis:best",
            2.0,
            [0.0, -0.5, 0.0],
        )

        specs = budgeted_candidate_specs(
            {"rows": [urban, freeway]},
            max_h12_candidates_per_event=1,
            domain_pool_size=1,
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["selection_source"], "single_urban_h1")
        self.assertEqual(
            specs[0]["component_candidate_ids"],
            [urban["candidate_id"]],
        )

    def test_domain_top_is_owner_diverse(self):
        urban_a_best = _row(
            "structured:urban:A:axis:best",
            3.0,
            [0.5, 0.0, 0.0, 0.0],
        )
        urban_a_second = _row(
            "structured:urban:A:axis:second",
            2.0,
            [0.25, 0.0, 0.0, 0.0],
        )
        urban_b = _row(
            "structured:urban:B:axis:covered",
            0.1,
            [0.0, 0.5, 0.0, 0.0],
        )
        freeway = _row(
            "structured:freeway:R_D_W:axis:covered",
            0.05,
            [0.0, 0.0, 0.5, 0.0],
        )

        specs = budgeted_candidate_specs(
            {"rows": [urban_a_best, urban_a_second, urban_b, freeway]},
            max_h12_candidates_per_event=3,
            domain_pool_size=2,
        )

        component_ids = [
            tuple(spec["component_candidate_ids"]) for spec in specs
        ]
        self.assertIn((urban_a_best["candidate_id"],), component_ids)
        self.assertNotIn((urban_a_second["candidate_id"],), component_ids)
        self.assertIn((urban_b["candidate_id"],), component_ids)


class BudgetedEvaluationTests(unittest.TestCase):
    def test_writes_drain_compatible_budgeted_source(self):
        module = "rl_leader.evaluate_budgeted_tail_horizons"
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            source_path = directory / "h1.json"
            manifest_path = directory / "manifest.json"
            output_path = directory / "budgeted.json"
            row = _row(
                "structured:urban:A:axis:best",
                2.0,
                [0.5, 0.0],
            )
            pool = {
                "scenario": "sweet_155_w60",
                "policy_step": 6,
                "simulation_step": 11,
                "simulation_time_sec": 1080.0,
                "native_anchor_branch": "coarse",
                "pre_runtime_sha256": "pre",
                "forecast_sha256": "forecast",
                "anchor_fingerprint": "anchor",
                "observation": [1.0, 2.0],
                "anchor_envelope": [3.0, 4.0],
                "candidate_mode": "owner_block_v2_all",
                "rows": [row],
            }
            event = {
                "stratum": "growth",
                "policy_step": pool["policy_step"],
                "coordination_eligible": True,
                "simulation_step": pool["simulation_step"],
                "simulation_time_sec": pool["simulation_time_sec"],
                "native_anchor_branch": pool["native_anchor_branch"],
                "pre_runtime_sha256": pool["pre_runtime_sha256"],
                "forecast_sha256": pool["forecast_sha256"],
                "anchor_fingerprint": pool["anchor_fingerprint"],
                "observation_sha256": _digest(pool["observation"]),
                "anchor_envelope_sha256": _digest(pool["anchor_envelope"]),
            }
            source = {
                "implementation_sha256": {"core": "sha"},
                "candidate_pools": [pool],
            }
            manifest = {
                "scenarios": [{
                    "scenario": pool["scenario"],
                    "events": [event],
                }]
            }
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with (
                patch(f"{module}.validate_oracle_label_artifact"),
                patch(f"{module}.validate_frozen_manifest"),
                patch(f"{module}._implementation_fingerprints", return_value={"core": "sha"}),
                patch(
                    f"{module}._replay_event",
                    return_value=(object(), SimpleNamespace(anchor_fingerprint="anchor")),
                ),
                patch(f"{module}._rollout_pstack", return_value=_rollout((0.0, 0.0))),
                patch(f"{module}._rollout_price_candidate", return_value=_rollout((1.0, 0.0))),
                patch(f"{module}._require_rollout_coverage"),
                patch(f"{module}._response_scales_and_families", return_value=(None, None)),
                patch(f"{module}.response_distance", return_value={"overall_rmse": 1.0}),
                patch(f"{module}._h1_exact", return_value=True),
                patch(
                    f"{module}._horizon_label",
                    return_value={
                        "validity_gate_pass": True,
                        "positive": True,
                        "ttt_gain": 1.0,
                    },
                ),
            ):
                result = evaluate_budgeted_tail_horizons(
                    source_path,
                    manifest_path,
                    output_path,
                    max_h12_candidates_per_event=1,
                    domain_pool_size=1,
                )

            self.assertEqual(result, json.loads(output_path.read_text(encoding="utf-8")))
            self.assertEqual(result["format_version"], "balanced_owner_block_h3_selective_h12_v1")
            self.assertEqual(result["contract"], BUDGETED_TAIL_HORIZON_CONTRACT)
            self.assertEqual(result["candidate_domain"], BUDGETED_TAIL_DOMAIN)
            self.assertEqual(result["selector_version"], BUDGETED_SELECTOR_VERSION)
            self.assertEqual(result["source_h1_sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
            self.assertEqual(len(result["outcomes"]), 1)
            self.assertTrue(result["outcomes"][0]["h12_label"]["positive"])


if __name__ == "__main__":
    unittest.main()
