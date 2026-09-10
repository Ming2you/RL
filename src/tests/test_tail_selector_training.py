from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.build_tail_pairwise_dataset import TAIL_PAIRWISE_FORMAT
from rl_leader.train_tail_selector import (
    TAIL_SELECTOR_MODEL_FORMAT,
    train_tail_selector,
)


def _row(group, scenario, candidate, target, residual_value):
    gain = 8.0 if target else -2.0
    required = 1.0
    return {
        "event_group_id": group,
        "scenario": scenario,
        "stratum": "plateau",
        "policy_step": 9,
        "native_anchor_branch": "coarse",
        "candidate_execution_branch": "coordination",
        "observation": [float(residual_value), 0.0],
        "anchor_envelope": [0.0, 1.0],
        "candidate_residual": [float(residual_value), 0.0],
        "candidate_id": candidate,
        "target_valid": True,
        "target": int(target),
        "tail_status": "positive" if target else "negative",
        "gain_veh_h": gain,
        "required_gain_veh_h": required,
        "margin_ratio": gain - required,
        "sample_weight": 1.0,
    }


class TailSelectorTrainingTests(unittest.TestCase):
    def _dataset(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        dataset_path = Path(directory.name) / "dataset.json"
        output_path = Path(directory.name) / "model.json"
        dataset_path.write_text(
            json.dumps({
                "format_version": TAIL_PAIRWISE_FORMAT,
                "rows": rows,
                "folds": [
                    {
                        "held_out_scenario": "s0",
                        "train_event_group_ids": ["g1"],
                        "test_event_group_ids": ["g0"],
                    },
                    {
                        "held_out_scenario": "s1",
                        "train_event_group_ids": ["g0"],
                        "test_event_group_ids": ["g1"],
                    },
                ],
            }),
            encoding="utf-8",
        )
        return dataset_path, output_path

    def test_training_writes_fail_closed_non_deployable_smoke_model(self):
        dataset_path, output_path = self._dataset([
            _row("g0", "s0", "positive", 1, 1.0),
            _row("g0", "s0", "negative", 0, -1.0),
            _row("g1", "s1", "positive-2", 1, 1.0),
            _row("g1", "s1", "negative-2", 0, -1.0),
        ])

        result = train_tail_selector(
            dataset_path,
            output_path,
            ensemble_size=1,
            logistic_steps=300,
            logistic_lr=0.2,
            min_prob_lcb=0.5,
            min_event_groups=50,
            min_positive_rows=20,
        )

        self.assertEqual(result["format_version"], TAIL_SELECTOR_MODEL_FORMAT)
        self.assertTrue(output_path.exists())
        self.assertTrue(result["passed"])
        self.assertFalse(result["summary"]["deployable_gate_pass"])
        self.assertEqual(result["summary"]["rows"], 4)
        self.assertEqual(result["summary"]["positive_rows"], 2)
        self.assertGreaterEqual(result["summary"]["train_selected_event_groups"], 1)

    def test_training_requires_positive_and_negative_rows(self):
        dataset_path, output_path = self._dataset([
            _row("g0", "s0", "positive", 1, 1.0),
            _row("g1", "s1", "positive-2", 1, 1.0),
        ])

        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            train_tail_selector(dataset_path, output_path)

    def test_deployable_gate_rejects_all_fallback_cross_validation(self):
        dataset_path, output_path = self._dataset([
            _row("g0", "s0", "positive", 1, 1.0),
            _row("g0", "s0", "negative", 0, -1.0),
            _row("g1", "s1", "positive-2", 1, 1.0),
            _row("g1", "s1", "negative-2", 0, -1.0),
        ])

        result = train_tail_selector(
            dataset_path,
            output_path,
            ensemble_size=1,
            logistic_steps=100,
            min_prob_lcb=2.0,
            min_event_groups=2,
            min_positive_rows=2,
        )

        self.assertEqual(result["summary"]["fold_false_positive_event_groups"], 0)
        self.assertEqual(result["summary"]["fold_missed_positive_event_groups"], 2)
        self.assertFalse(result["summary"]["deployable_gate_pass"])


if __name__ == "__main__":
    unittest.main()
