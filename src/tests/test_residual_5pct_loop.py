from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.run_residual_5pct_loop import (
    build_sampler_jobs,
    candidate_artifact_sets,
    parse_int_csv,
    peak_steps_from_trace,
    recover_partial_sampler_summary,
    select_best_step_artifacts,
)


def _sampler_summary(step: int, seed: int, gain: float, positive: bool) -> dict:
    return {
        "control_step": step,
        "seed": seed,
        "horizon_steps": 3,
        "h12_records_path": f"samplers/step{step:02d}_seed{seed}/h12_records.jsonl",
        "best_h12": {
            "candidate_id": f"c{step}_{seed}",
            "horizon_labels": {
                "3": {
                    "ttt_gain": gain,
                    "positive": positive,
                    "validity_gate_pass": True,
                }
            },
        },
    }


class TestResidual5PctLoop(unittest.TestCase):
    def test_parse_int_csv_rejects_duplicates(self):
        self.assertEqual(parse_int_csv("3,1,2"), (3, 1, 2))
        with self.assertRaises(ValueError):
            parse_int_csv("1,2,1")

    def test_peak_steps_from_trace_selects_top_ttt_inside_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [
                {"control_step": 16, "interval_ttt": 99.0},
                {"control_step": 17, "interval_ttt": 2.0},
                {"control_step": 18, "interval_ttt": 9.0},
                {"control_step": 19, "interval_ttt": 7.0},
                {"control_step": 20, "interval_ttt": 1.0},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                peak_steps_from_trace(path, min_step=17, max_step=20, limit=2),
                (18, 19),
            )

    def test_select_best_step_artifacts_filters_and_keeps_best_per_step(self):
        summaries = [
            _sampler_summary(21, 1, 0.2, True),
            _sampler_summary(21, 2, 0.5, True),
            _sampler_summary(22, 1, 0.9, False),
            _sampler_summary(23, 1, 0.1, True),
        ]

        artifacts = select_best_step_artifacts(summaries, require_positive=True)

        self.assertEqual([item.step for item in artifacts], [21, 23])
        self.assertEqual(artifacts[0].seed, 2)
        self.assertAlmostEqual(artifacts[0].gain, 0.5)

    def test_candidate_artifact_sets_builds_gain_prefixes(self):
        artifacts = select_best_step_artifacts(
            [
                _sampler_summary(21, 1, 0.2, True),
                _sampler_summary(23, 1, 0.7, True),
                _sampler_summary(24, 1, 0.4, True),
            ],
            require_positive=True,
        )

        sets = candidate_artifact_sets(artifacts, max_schedules=3)

        self.assertEqual(sets[0][0], "top1_by_horizon_gain")
        self.assertEqual([item.step for item in sets[0][1]], [23])
        self.assertEqual([item.step for item in sets[1][1]], [23, 24])
        self.assertEqual([item.step for item in sets[2][1]], [21, 23, 24])

    def test_candidate_artifact_sets_adds_nonprefix_combos(self):
        artifacts = select_best_step_artifacts(
            [
                _sampler_summary(17, 1, 5.0, True),
                _sampler_summary(18, 1, 4.0, True),
                _sampler_summary(21, 1, 3.0, True),
            ],
            require_positive=True,
        )

        sets = candidate_artifact_sets(
            artifacts,
            max_schedules=8,
            max_combo_size=2,
            combo_pool_limit=3,
        )
        keys = {tuple(item.step for item in group) for _label, group in sets}

        self.assertIn((17, 21), keys)

    def test_build_sampler_jobs_orders_steps_then_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = build_sampler_jobs(
                caches={
                    21: root / "cache21.pkl",
                    22: root / "cache22.pkl",
                },
                output_dir=root / "out",
                steps=(21, 22),
                seeds=(3, 4),
                max_h1_candidates=80,
                max_h12_candidates=16,
                horizon_steps=12,
            )

            self.assertEqual(
                [(job.step, job.seed) for job in jobs],
                [(21, 3), (21, 4), (22, 3), (22, 4)],
            )
            self.assertEqual(jobs[0].replay_cache, root / "cache21.pkl")
            self.assertTrue(
                str(jobs[-1].output_dir).endswith("step22_seed4_h180_h12x16")
            )

    def test_recover_partial_sampler_summary_uses_best_h12_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h12 = root / "h12_records.jsonl"
            records = [
                {
                    "control_step": 21,
                    "candidate_id": "slow",
                    "continuous_residual": [0.0, 0.2],
                    "horizon_labels": {
                        "12": {
                            "ttt_gain": -1.0,
                            "positive": False,
                            "validity_gate_pass": True,
                        }
                    },
                },
                {
                    "control_step": 21,
                    "candidate_id": "best",
                    "continuous_residual": [0.0, 0.4],
                    "horizon_labels": {
                        "12": {
                            "ttt_gain": 3.0,
                            "positive": True,
                            "validity_gate_pass": True,
                        }
                    },
                },
            ]
            h12.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )

            summary = recover_partial_sampler_summary(
                output_dir=root,
                seed=7,
                max_h1_candidates=80,
                max_h12_candidates=16,
                horizon_steps=12,
                magnitudes=(0.25,),
                pstack_summary={"total_ttt": 100.0},
                workers=2,
                native_rollout_cache_dir=None,
                h1_chunks_per_worker=4,
                h12_chunks_per_worker=4,
            )

            self.assertIsNotNone(summary)
            self.assertTrue((root / "summary.json").is_file())
            self.assertEqual(summary["best_h12"]["candidate_id"], "best")
            self.assertEqual(summary["h12_record_count"], 2)
            self.assertEqual(summary["h12_positive_count"], 1)

    def test_recover_partial_sampler_summary_promotes_h1_for_horizon_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h1 = root / "h1_records.jsonl"
            records = [
                {
                    "control_step": 21,
                    "candidate_id": "a",
                    "continuous_residual": [0.1, 0.0],
                    "horizon_labels": {
                        "1": {
                            "ttt_gain": 0.1,
                            "positive": False,
                            "validity_gate_pass": True,
                            "label_valid": True,
                        }
                    },
                    "response_memory_outcome_sha256": "m1",
                    "post_physical_sha256": "p1",
                    "native_response_distance_score": 0.1,
                    "residual_nonzero_count": 1,
                },
                {
                    "control_step": 21,
                    "candidate_id": "b",
                    "continuous_residual": [0.0, 0.2],
                    "horizon_labels": {
                        "1": {
                            "ttt_gain": 0.2,
                            "positive": False,
                            "validity_gate_pass": True,
                            "label_valid": True,
                        }
                    },
                    "response_memory_outcome_sha256": "m2",
                    "post_physical_sha256": "p2",
                    "native_response_distance_score": 0.2,
                    "residual_nonzero_count": 1,
                },
            ]
            h1.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )

            summary = recover_partial_sampler_summary(
                output_dir=root,
                seed=7,
                max_h1_candidates=80,
                max_h12_candidates=1,
                horizon_steps=1,
                magnitudes=(0.25,),
                pstack_summary={"total_ttt": 100.0},
                workers=2,
                native_rollout_cache_dir=None,
                h1_chunks_per_worker=4,
                h12_chunks_per_worker=4,
            )

            self.assertIsNotNone(summary)
            self.assertTrue((root / "h12_records.jsonl").is_file())
            self.assertEqual(summary["h12_record_count"], 1)
            self.assertEqual(summary["best_h12"]["candidate_id"], "b")


if __name__ == "__main__":
    unittest.main()
