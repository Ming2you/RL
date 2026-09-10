import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.run_residual_neighborhood_search import (
    generate_neighborhood_candidates,
    load_residual_seed,
)


class ResidualNeighborhoodSearchTests(unittest.TestCase):
    def test_load_residual_seed_from_best_h12(self):
        artifact = {
            "format_version": "cached_tail_residual_sampler_v1",
            "control_step": 18,
            "horizon_steps": 1,
            "action_names": ["a", "b", "c"],
            "h12_records": [
                {
                    "candidate_id": "low",
                    "control_step": 18,
                    "continuous_residual": [0.0, 0.25, 0.0],
                    "horizon_labels": {"1": {"ttt_gain": -1.0, "positive": False}},
                },
                {
                    "candidate_id": "high",
                    "control_step": 18,
                    "continuous_residual": [0.0, 1.0, -0.5],
                    "horizon_labels": {"1": {"ttt_gain": 2.0, "positive": False}},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            seed = load_residual_seed(path)
        self.assertEqual(18, seed.source_step)
        self.assertEqual("high", seed.source_label)
        self.assertEqual((0.0, 1.0, -0.5), seed.residual)

    def test_generate_neighborhood_candidates_includes_scaled_subsets_and_repeats(self):
        seed = type(
            "Seed",
            (),
            {
                "source_path": Path("source.json"),
                "source_step": 18,
                "source_label": "base",
                "action_names": ("a", "b", "c"),
                "residual": (1.0, -0.5, 0.0),
            },
        )()
        candidates = generate_neighborhood_candidates(
            seed,
            placement_steps=(17, 18),
            scales=(0.5, 1.0),
            repeat_sizes=(2,),
            subset_pool_limit=2,
            max_subset_size=2,
            include_sign_flips=True,
            max_candidates=64,
        )
        labels = {candidate.label for candidate in candidates}
        self.assertIn("all_s1_at18", labels)
        self.assertIn("subset1_s0.5_at17_a", labels)
        self.assertIn("repeat2_s1_steps17-18", labels)
        signatures = [
            tuple((item.step, item.residual) for item in candidate.schedule)
            for candidate in candidates
        ]
        self.assertEqual(len(signatures), len(set(signatures)))


if __name__ == "__main__":
    unittest.main()
