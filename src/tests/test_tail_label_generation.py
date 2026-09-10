from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_leader.run_tail_label_generation import (
    BUDGETED_SELECTOR_VERSION,
    _is_complete_h1,
    _is_complete_horizon,
    _summarize,
    eligible_manifest_events,
    populate_missing_h1_batches,
    run_tail_label_generation,
    select_drain_candidate_ids,
)


class TailLabelGenerationTest(unittest.TestCase):
    def test_eligible_manifest_events_filters_and_orders(self):
        manifest = {
            "scenarios": [
                {
                    "scenario": "sweet_170_w60",
                    "events": [
                        {
                            "stratum": "plateau",
                            "policy_step": 9,
                            "native_anchor_branch": "coarse",
                            "coordination_eligible": True,
                            "anchor_fingerprint": "b",
                        },
                        {
                            "stratum": "plateau",
                            "policy_step": 11,
                            "native_anchor_branch": "fallback_pfo",
                            "coordination_eligible": False,
                            "anchor_fingerprint": "c",
                        },
                    ],
                },
                {
                    "scenario": "sweet_155_w60",
                    "events": [
                        {
                            "stratum": "growth",
                            "policy_step": 6,
                            "native_anchor_branch": "coarse",
                            "coordination_eligible": True,
                            "anchor_fingerprint": "a",
                        }
                    ],
                },
            ]
        }
        rows = eligible_manifest_events(manifest)
        self.assertEqual(
            [(row["scenario"], row["policy_step"]) for row in rows],
            [("sweet_155_w60", 6), ("sweet_170_w60", 9)],
        )
        self.assertEqual(
            eligible_manifest_events(manifest, policy_steps=(9,))[0]["scenario"],
            "sweet_170_w60",
        )

    def test_select_drain_candidate_ids_keeps_positives_then_promising(self):
        source = {
            "outcomes": [
                {
                    "candidate_id": "bad",
                    "h12_label": {"validity_gate_pass": True, "positive": False, "ttt_gain": -5.0},
                },
                {
                    "candidate_id": "pos2",
                    "h12_label": {"validity_gate_pass": True, "positive": True, "ttt_gain": 1.0},
                },
                {
                    "candidate_id": "skip",
                    "h12_label": None,
                },
                {
                    "candidate_id": "neg_best",
                    "h12_label": {"validity_gate_pass": True, "positive": False, "ttt_gain": 0.5},
                },
                {
                    "candidate_id": "pos1",
                    "h12_label": {"validity_gate_pass": True, "positive": True, "ttt_gain": 3.0},
                },
            ]
        }
        self.assertEqual(
            select_drain_candidate_ids(source, limit=3),
            ["pos1", "pos2", "neg_best"],
        )

    def test_select_drain_candidate_ids_limit_zero_keeps_all_drain_worthy(self):
        source = {
            "outcomes": [
                {
                    "candidate_id": "neg",
                    "h12_label": {"validity_gate_pass": True, "positive": False, "ttt_gain": 0.0},
                },
                {
                    "candidate_id": "pos",
                    "h12_label": {"validity_gate_pass": True, "positive": True, "ttt_gain": 1.0},
                },
            ]
        }
        self.assertEqual(
            select_drain_candidate_ids(source, limit=0),
            ["pos"],
        )

    def test_select_drain_candidate_ids_can_skip_weak_h12_gains(self):
        source = {
            "outcomes": [
                {
                    "candidate_id": "weak_pos",
                    "h12_label": {"validity_gate_pass": True, "positive": True, "ttt_gain": 1.0},
                },
                {
                    "candidate_id": "strong_promising",
                    "h12_label": {"validity_gate_pass": True, "positive": False, "ttt_gain": 6.0},
                },
            ]
        }
        self.assertEqual(
            select_drain_candidate_ids(
                source,
                limit=4,
                min_h12_gain_for_drain=5.0,
            ),
            ["strong_promising"],
        )

    def test_select_drain_candidate_ids_returns_empty_without_positive_h12(self):
        source = {
            "outcomes": [
                {
                    "candidate_id": "neg",
                    "h12_label": {
                        "validity_gate_pass": True,
                        "positive": False,
                        "ttt_gain": -1.0,
                    },
                },
            ]
        }
        self.assertEqual(select_drain_candidate_ids(source, limit=3), [])

    def test_summarize_counts_no_drain_candidate_as_terminal_not_complete(self):
        summary = _summarize([
            {"status": "complete", "drain_statuses": ["positive"]},
            {"status": "h1_complete", "drain_statuses": []},
            {"status": "no_drain_candidate", "drain_statuses": []},
        ])
        self.assertEqual(summary["complete_events"], 1)
        self.assertEqual(summary["h1_complete_events"], 1)
        self.assertEqual(summary["no_drain_candidate_events"], 1)
        self.assertEqual(summary["terminal_events"], 3)
        self.assertEqual(summary["drain_outcomes"], 1)

    def test_complete_h1_requires_candidate_mode_when_requested(self):
        payload = {
            "format_version": "pcent_guided_oracle_labels_v1_branch_aware",
            "passed": True,
            "candidate_pools": [{
                "scenario": "sweet_155_w60",
                "policy_step": 6,
                "candidate_mode": "owner_block_v2_all",
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h1.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                _is_complete_h1(
                    path,
                    "sweet_155_w60",
                    6,
                    candidate_mode="owner_block_v2_all",
                )
            )
            self.assertFalse(
                _is_complete_h1(
                    path,
                    "sweet_155_w60",
                    6,
                    candidate_mode="owner_block_v2_urban",
                )
            )

    def test_complete_budgeted_horizon_checks_selection_policy(self):
        payload = {
            "format_version": "balanced_owner_block_h3_selective_h12_v1",
            "passed": True,
            "scenario": "sweet_155_w60",
            "policy_step": 6,
            "selector_version": BUDGETED_SELECTOR_VERSION,
            "selection_policy": {
                "max_h12_candidates_per_event": 2,
                "domain_pool_size": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h12.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                _is_complete_horizon(
                    path,
                    "sweet_155_w60",
                    6,
                    selector_version=BUDGETED_SELECTOR_VERSION,
                    parameters={
                        "max_h12_candidates_per_event": 2,
                        "domain_pool_size": 1,
                    },
                )
            )
            self.assertFalse(
                _is_complete_horizon(
                    path,
                    "sweet_155_w60",
                    6,
                    selector_version=BUDGETED_SELECTOR_VERSION,
                    parameters={
                        "max_h12_candidates_per_event": 1,
                        "domain_pool_size": 1,
                    },
                )
            )

    def test_populate_missing_h1_batches_splits_one_pool_per_event(self):
        events = [
            {
                "scenario": "sweet_155_w60",
                "stratum": "growth",
                "policy_step": 5,
                "native_anchor_branch": "coarse",
                "anchor_fingerprint": "a5",
            },
            {
                "scenario": "sweet_155_w60",
                "stratum": "growth",
                "policy_step": 7,
                "native_anchor_branch": "coarse",
                "anchor_fingerprint": "a7",
            },
        ]

        def fake_pilot(_manifest_path, output_dir, **kwargs):
            self.assertEqual(kwargs["policy_steps"], (5, 7))
            for step, payload_path in kwargs["replay_cache_payloads"].items():
                payload_path.parent.mkdir(parents=True, exist_ok=True)
                payload_path.write_bytes(f"cache-{step}".encode("ascii"))
            artifact = {
                "format_version": "pcent_guided_oracle_labels_v1_branch_aware",
                "passed": True,
                "candidate_pools": [
                    {
                        "scenario": "sweet_155_w60",
                        "policy_step": 5,
                        "candidate_mode": "owner_block_v2_all",
                        "rows": [],
                    },
                    {
                        "scenario": "sweet_155_w60",
                        "policy_step": 7,
                        "candidate_mode": "owner_block_v2_all",
                        "rows": [],
                    },
                ],
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sweet_155_w60.json").write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            return artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with (
                patch(
                    "rl_leader.run_tail_label_generation.run_balanced_pilot",
                    side_effect=fake_pilot,
                ) as pilot,
                patch(
                    "rl_leader.run_tail_label_generation.validate_oracle_label_artifact"
                ),
            ):
                populate_missing_h1_batches(manifest, root / "out", events)

            self.assertEqual(pilot.call_count, 1)
            for step in (5, 7):
                event_root = root / "out" / f"sweet_155_w60_step{step:02d}"
                h1 = event_root / "h1_all" / "sweet_155_w60.json"
                metadata = event_root / "replay_cache.json"
                payload = event_root / "replay_cache.pkl"
                written = json.loads(h1.read_text(encoding="utf-8"))
                self.assertEqual(len(written["candidate_pools"]), 1)
                self.assertEqual(written["candidate_pools"][0]["policy_step"], step)
                self.assertTrue(metadata.is_file())
            self.assertTrue(payload.is_file())

    def test_run_tail_label_generation_h1_only_writes_replay_index(self):
        event = {
            "scenario": "sweet_155_w60",
            "stratum": "growth",
            "policy_step": 5,
            "native_anchor_branch": "coarse",
            "coordination_eligible": True,
            "anchor_fingerprint": "a5",
        }

        def fake_validate_manifest(_manifest, **_kwargs):
            return None

        def fake_populate(_manifest_path, output_dir, events):
            self.assertEqual(len(events), 1)
            root = output_dir / "sweet_155_w60_step05"
            h1 = root / "h1_all" / "sweet_155_w60.json"
            h1.parent.mkdir(parents=True, exist_ok=True)
            h1.write_text(
                json.dumps({
                    "format_version": "pcent_guided_oracle_labels_v1_branch_aware",
                    "passed": True,
                    "candidate_pools": [{
                        "scenario": "sweet_155_w60",
                        "policy_step": 5,
                        "candidate_mode": "owner_block_v2_all",
                        "rows": [],
                    }],
                }),
                encoding="utf-8",
            )
            (root / "replay_cache.pkl").write_bytes(b"cache")
            (root / "replay_cache.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"scenarios": [{"scenario": "sweet_155_w60", "events": [event]}]}),
                encoding="utf-8",
            )
            with (
                patch(
                    "rl_leader.run_tail_label_generation.validate_frozen_manifest",
                    side_effect=fake_validate_manifest,
                ),
                patch(
                    "rl_leader.run_tail_label_generation.populate_missing_h1_batches",
                    side_effect=fake_populate,
                ),
            ):
                result = run_tail_label_generation(
                    manifest,
                    root / "out",
                    scenarios=("sweet_155_w60",),
                    policy_steps=(5,),
                    h1_only=True,
                )

        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["h1_complete_events"], 1)
        self.assertEqual(result["events"][0]["status"], "h1_complete")


if __name__ == "__main__":
    unittest.main()
