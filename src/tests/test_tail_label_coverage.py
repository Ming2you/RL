from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.summarize_tail_label_coverage import (
    BALANCED_HORIZON_FORMAT,
    DRAIN_OUT_FORMAT,
    summarize_tail_label_coverage,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TailLabelCoverageTests(unittest.TestCase):
    def test_coverage_separates_complete_running_and_undrained_h12_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            _write(
                source,
                {
                    "format_version": BALANCED_HORIZON_FORMAT,
                    "passed": True,
                    "scenario": "sweet_170_w60",
                    "stratum": "plateau",
                    "policy_step": 9,
                    "candidate_domain": "urban+freeway",
                    "outcomes": [
                        {
                            "candidate_id": "done",
                            "continuous_residual": [1.0, 0.0],
                            "h12_label": {
                                "positive": True,
                                "ttt_gain": 3.0,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                        {
                            "candidate_id": "running",
                            "continuous_residual": [2.0, 0.0],
                            "h12_label": {
                                "positive": True,
                                "ttt_gain": 2.0,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                        {
                            "candidate_id": "todo",
                            "continuous_residual": [3.0, 0.0],
                            "h12_label": {
                                "positive": True,
                                "ttt_gain": 1.5,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                    ],
                },
            )
            _write(
                root / "done_drain.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "source_artifact": str(source),
                    "status": "complete",
                    "passed": True,
                    "parameters": {"requested_candidate_ids": None},
                    "outcomes": [
                        {
                            "candidate_id": "done",
                            "continuous_residual": [1.0, 0.0],
                            "verdict": {
                                "status": "positive",
                                "ttt_gain": 4.0,
                                "required_gain": 1.0,
                            },
                        }
                    ],
                },
            )
            _write(
                root / "running_drain.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "source_artifact": str(source),
                    "status": "running",
                    "passed": False,
                    "parameters": {"requested_candidate_ids": ["running"]},
                    "outcomes": [],
                },
            )
            result = summarize_tail_label_coverage(
                [root], root / "coverage.json"
            )

        self.assertEqual(result["summary"]["h12_positive_candidate_rows"], 3)
        self.assertEqual(result["summary"]["strict_completed_candidate_rows"], 1)
        self.assertEqual(result["summary"]["running_candidate_rows"], 1)
        self.assertEqual(result["summary"]["undrained_h12_positive_candidate_rows"], 1)
        self.assertEqual(
            result["undrained_h12_positive_candidates"][0]["candidate_id"], "todo"
        )

    def test_default_running_drain_covers_source_h12_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            _write(
                source,
                {
                    "format_version": BALANCED_HORIZON_FORMAT,
                    "passed": True,
                    "scenario": "sweet_155_w60",
                    "stratum": "plateau",
                    "policy_step": 9,
                    "candidate_domain": "urban+freeway",
                    "outcomes": [
                        {
                            "candidate_id": "positive",
                            "continuous_residual": [1.0, 0.0],
                            "h12_label": {
                                "positive": True,
                                "ttt_gain": 2.0,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                        {
                            "candidate_id": "negative",
                            "continuous_residual": [0.0, 1.0],
                            "h12_label": {
                                "positive": False,
                                "ttt_gain": -1.0,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                    ],
                },
            )
            _write(
                root / "default_drain.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "source_artifact": str(source),
                    "status": "running",
                    "passed": False,
                    "parameters": {"requested_candidate_ids": None},
                    "outcomes": [],
                },
            )
            result = summarize_tail_label_coverage(
                [root], root / "coverage.json"
            )

        by_id = {
            row["candidate_id"]: row["coverage_status"]
            for row in result["coverage_rows"]
        }
        self.assertEqual(by_id["positive"], "running")
        self.assertEqual(by_id["negative"], "not_drained")
        self.assertEqual(result["summary"]["running_candidate_rows"], 1)

    def test_partial_default_running_drain_keeps_remaining_positives_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            _write(
                source,
                {
                    "format_version": BALANCED_HORIZON_FORMAT,
                    "passed": True,
                    "scenario": "sweet_155_w60",
                    "stratum": "plateau",
                    "policy_step": 9,
                    "candidate_domain": "urban+freeway",
                    "outcomes": [
                        {
                            "candidate_id": "done",
                            "continuous_residual": [1.0, 0.0],
                            "h12_label": {
                                "positive": True,
                                "ttt_gain": 2.0,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                        {
                            "candidate_id": "still-running",
                            "continuous_residual": [2.0, 0.0],
                            "h12_label": {
                                "positive": True,
                                "ttt_gain": 1.5,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                    ],
                },
            )
            _write(
                root / "default_drain.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "source_artifact": str(source),
                    "status": "running",
                    "passed": False,
                    "parameters": {"requested_candidate_ids": None},
                    "outcomes": [
                        {
                            "candidate_id": "done",
                            "continuous_residual": [1.0, 0.0],
                            "verdict": {
                                "status": "negative",
                                "ttt_gain": 0.5,
                                "required_gain": 1.0,
                            },
                        }
                    ],
                },
            )
            result = summarize_tail_label_coverage(
                [root], root / "coverage.json"
            )

        by_id = {
            row["candidate_id"]: row["coverage_status"]
            for row in result["coverage_rows"]
        }
        self.assertEqual(by_id["done"], "running")
        self.assertEqual(by_id["still-running"], "running")
        self.assertEqual(result["summary"]["undrained_h12_positive_candidate_rows"], 0)

    def test_partial_explicit_running_drain_keeps_missing_requested_ids_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            _write(
                source,
                {
                    "format_version": BALANCED_HORIZON_FORMAT,
                    "passed": True,
                    "scenario": "sweet_170_incident_w60",
                    "stratum": "recovery_boundary",
                    "policy_step": 24,
                    "candidate_domain": "urban+freeway",
                    "outcomes": [
                        {
                            "candidate_id": "done",
                            "continuous_residual": [1.0, 0.0],
                            "h12_label": {
                                "positive": False,
                                "ttt_gain": -0.5,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                        {
                            "candidate_id": "requested",
                            "continuous_residual": [2.0, 0.0],
                            "h12_label": {
                                "positive": False,
                                "ttt_gain": 0.0,
                                "required_gain": 1.0,
                                "validity_gate_pass": True,
                            },
                        },
                    ],
                },
            )
            _write(
                root / "explicit_drain.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "source_artifact": str(source),
                    "status": "running",
                    "passed": False,
                    "parameters": {"requested_candidate_ids": ["done", "requested"]},
                    "outcomes": [
                        {
                            "candidate_id": "done",
                            "continuous_residual": [1.0, 0.0],
                            "verdict": {
                                "status": "negative",
                                "ttt_gain": 0.5,
                                "required_gain": 1.0,
                            },
                        }
                    ],
                },
            )
            result = summarize_tail_label_coverage(
                [root], root / "coverage.json"
            )

        by_id = {
            row["candidate_id"]: row["coverage_status"]
            for row in result["coverage_rows"]
        }
        self.assertEqual(by_id["done"], "running")
        self.assertEqual(by_id["requested"], "running")

    def test_duplicate_source_positive_is_not_a_new_drain_backlog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_done = root / "source_done.json"
            source_duplicate = root / "source_duplicate.json"
            source_payload = {
                "format_version": BALANCED_HORIZON_FORMAT,
                "passed": True,
                "scenario": "sweet_170_w60",
                "stratum": "plateau",
                "policy_step": 9,
                "candidate_domain": "urban",
                "outcomes": [
                    {
                        "candidate_id": "same",
                        "continuous_residual": [0.0, -0.5],
                        "h12_label": {
                            "positive": True,
                            "ttt_gain": 5.0,
                            "required_gain": 1.0,
                            "validity_gate_pass": True,
                        },
                    }
                ],
            }
            _write(source_done, source_payload)
            _write(source_duplicate, source_payload)
            _write(
                root / "done_drain.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "source_artifact": str(source_done),
                    "status": "complete",
                    "passed": True,
                    "parameters": {"requested_candidate_ids": None},
                    "outcomes": [
                        {
                            "candidate_id": "same",
                            "continuous_residual": [0.0, -0.5],
                            "verdict": {
                                "status": "positive",
                                "ttt_gain": 8.0,
                                "required_gain": 1.0,
                            },
                        }
                    ],
                },
            )
            result = summarize_tail_label_coverage(
                [root], root / "coverage.json"
            )

        statuses = sorted(row["coverage_status"] for row in result["coverage_rows"])
        self.assertEqual(statuses, ["complete", "duplicate_complete"])
        self.assertEqual(result["summary"]["undrained_h12_positive_candidate_rows"], 0)


if __name__ == "__main__":
    unittest.main()
