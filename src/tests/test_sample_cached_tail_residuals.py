from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_leader.evaluate_fixed_residual_schedule import load_sampled_residual_schedule
from rl_leader.oracle_candidate_ablation import residual_sha256
from rl_leader.sample_cached_tail_residuals import (
    _chunked,
    build_sampled_residual_specs,
    evaluation_horizons,
    parse_magnitudes,
    select_h12_records,
)


def _action_names() -> tuple[str, ...]:
    names = ["budget.N_P", "budget.N_UF"]
    for signal in ("A", "B", "C", "D", "F"):
        names.extend(
            f"urban.{signal}.{term}"
            for term in ("g_green", "g_offset", "l11", "l21", "l22")
        )
    for ramp in ("R_D_W", "R_F_W", "R_D_E", "R_F_E"):
        names.extend(
            f"freeway.{ramp}.{term}"
            for term in ("g_meter", "g_vsl", "l11", "l21", "l22")
        )
    for link in ("FW_W", "FW_E"):
        for segment in (0, 1, 2, 4, 6, 7):
            names.extend(
                (
                    f"vsl.{link}__seg{segment}.g_vsl",
                    f"vsl.{link}__seg{segment}.sqrt_h",
                )
            )
    names.extend(
        f"certificate.{ramp}.release"
        for ramp in ("R_D_W", "R_F_W", "R_D_E", "R_F_E")
    )
    return tuple(names)


def _h1_record(
    candidate_id: str,
    gain: float,
    *,
    outcome: str | None = None,
    distance: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "response_memory_outcome_sha256": outcome or candidate_id,
        "post_physical_sha256": outcome or candidate_id,
        "native_response_distance": {"overall_rmse": distance},
        "residual_nonzero_count": 2,
        "horizon_labels": {
            "1": {
                "label_valid": True,
                "validity_gate_pass": True,
                "ttt_gain": gain,
                "terminal_inventory_delta": 0.0,
            }
        },
    }


class SampledResidualSpecTests(unittest.TestCase):
    def test_chunked_partitions_values_without_duplicates(self):
        chunks = _chunked(tuple(range(10)), 3)
        flattened = [value for chunk in chunks for value in chunk]
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(len(chunks), 3)

    def test_parse_magnitudes_rejects_duplicate_or_out_of_box_values(self):
        self.assertEqual(parse_magnitudes("0.25,0.5"), (0.25, 0.5))
        with self.assertRaises(ValueError):
            parse_magnitudes("0.25,0.25")
        with self.assertRaises(ValueError):
            parse_magnitudes("1.5")

    def test_evaluation_horizons_never_exceed_rollout_horizon(self):
        self.assertEqual(evaluation_horizons(1), (1,))
        self.assertEqual(evaluation_horizons(3), (1, 3))
        self.assertEqual(evaluation_horizons(12), (1, 3, 12))
        with self.assertRaises(ValueError):
            evaluation_horizons(0)

    def test_builds_bounded_mixed_current_contract_samples(self):
        names = _action_names()
        specs = build_sampled_residual_specs(
            names,
            seed=7,
            max_h1_candidates=48,
            magnitudes=(0.25, 0.5),
        )
        self.assertEqual(len(specs), 48)
        digests = {residual_sha256(spec.residual) for spec in specs}
        self.assertEqual(len(digests), len(specs))
        residuals = [spec.residual_array() for spec in specs]
        self.assertTrue(all(residual.shape == (len(names),) for residual in residuals))
        self.assertTrue(all(np.max(np.abs(residual)) <= 1.0 for residual in residuals))
        rfw_meter = names.index("freeway.R_F_W.g_meter")
        self.assertTrue(any(residual[rfw_meter] > 0.0 for residual in residuals))
        nonlinear_names = [
            name for name in names
            if name.endswith((".l11", ".l21", ".l22", ".sqrt_h"))
        ]
        self.assertTrue(any(
            any(abs(residual[names.index(name)]) > 0.0 for name in nonlinear_names)
            for residual in residuals
        ))
        self.assertTrue(
            any(np.count_nonzero(np.abs(residual) > 1.0e-9) >= 3 for residual in residuals)
        )


class H12SelectionTests(unittest.TestCase):
    def test_selects_diverse_representatives_and_dedupes_outcomes(self):
        weak_duplicate = _h1_record("weak", 1.0, outcome="same")
        strong_duplicate = _h1_record("strong", 3.0, outcome="same")
        far = _h1_record("far", 0.1, distance=10.0)

        chosen = select_h12_records(
            [weak_duplicate, strong_duplicate, far],
            max_h12_candidates=2,
            seed=1,
        )

        chosen_ids = {row["candidate_id"] for row in chosen}
        self.assertIn("strong", chosen_ids)
        self.assertNotIn("weak", chosen_ids)


class SampleArtifactScheduleTests(unittest.TestCase):
    def test_loads_best_h12_sample_as_schedule(self):
        names = _action_names()
        residual = [0.0] * len(names)
        residual[names.index("freeway.R_F_W.g_meter")] = 0.5
        artifact = {
            "format_version": "cached_tail_residual_sampler_v1",
            "control_step": 21,
            "horizon_steps": 12,
            "h12_records": [
                {
                    "candidate_id": "bad",
                    "continuous_residual": [0.0] * len(names),
                    "horizon_labels": {
                        "12": {
                            "ttt_gain": -1.0,
                            "positive": False,
                            "validity_gate_pass": True,
                        }
                    },
                },
                {
                    "candidate_id": "good",
                    "continuous_residual": residual,
                    "horizon_labels": {
                        "12": {
                            "ttt_gain": 5.0,
                            "positive": True,
                            "validity_gate_pass": True,
                        }
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "summary.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            schedule = load_sampled_residual_schedule(path, names)

        self.assertEqual(list(schedule), [21])
        self.assertEqual(schedule[21].label, "good")
        self.assertEqual(
            schedule[21].residual[names.index("freeway.R_F_W.g_meter")],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
