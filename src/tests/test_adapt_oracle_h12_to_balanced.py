from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl_leader.adapt_oracle_h12_to_balanced import adapt_oracle_h12_to_balanced
from rl_leader.diagnose_balanced_drain_out import _validate_source
from rl_leader.diagnose_phase0_parity import _digest


CORE = {"core.py": "current-core-sha"}
ROOT = Path(__file__).resolve().parents[2]
REAL_SOURCE = (
    ROOT
    / "results/rl_phase0_implementation_20260828"
    / "owner_block_v2_urban_axes_corners_step9_exhaustive_h12"
    / "sweet_170_w60.json"
)
REAL_MANIFEST = (
    ROOT
    / "results/rl_phase0_implementation_20260828"
    / "balanced_owner_block_frozen_states_v2/manifest.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(candidate_id: str, *, positive: bool, aliases: list[str]) -> dict:
    return {
        "candidate_id": candidate_id,
        "execution_branch": "coordination",
        "candidate_aliases": aliases,
        "continuous_residual": [0.25, 0.0],
        "response_memory_outcome_sha256": "response-outcome",
        "post_physical_sha256": "physical-outcome",
        "horizon_labels": {
            "1": {"positive": False, "evidence": "h1"},
            "3": {"positive": False, "evidence": "h3"},
            "12": {"positive": positive, "evidence": "h12"},
        },
        "h1_replay_evidence": {"passed": True},
        "h3_selection_replay_evidence": {"passed": True},
    }


def _fixture(*, positive: bool = True, second_domain: str = "urban") -> tuple[dict, dict]:
    first = "structured:urban:A:axis:first"
    second = f"structured:{second_domain}:B:axis:second"
    aliases = [first, second]
    pool = {
        "scenario": "sweet_170_w60",
        "policy_step": 9,
        "simulation_step": 14,
        "simulation_time_sec": 2520.0,
        "native_anchor_branch": "coarse",
        "pre_runtime_sha256": "pre-runtime",
        "forecast_sha256": "forecast",
        "anchor_fingerprint": "anchor",
        "observation": [1.0, 2.0],
        "anchor_envelope": [3.0, 4.0],
        "candidate_mode": "owner_block_v2_urban",
        "rows": [
            _row(first, positive=positive, aliases=aliases),
            _row(second, positive=False, aliases=aliases),
        ],
    }
    event = {
        "stratum": "plateau",
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
        "physical_snapshot_sha256": "physical-snapshot",
        "physical_snapshot": {"state": "frozen"},
        "rl_controller_sha256": "rl-controller",
        "optimizer_controller_sha256": "optimizer-controller",
    }
    source = {
        "passed": True,
        "implementation_sha256": CORE,
        "candidate_pools": [pool],
    }
    manifest = {"scenarios": [{"scenario": pool["scenario"], "events": [event]}]}
    return source, manifest


class AdaptOracleH12ToBalancedTests(unittest.TestCase):
    def _paths(self, source: dict, manifest: dict) -> tuple[Path, Path, Path]:
        directory = Path(tempfile.mkdtemp())
        source_path = directory / "source.json"
        manifest_path = directory / "manifest.json"
        output_path = directory / "balanced.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return source_path, manifest_path, output_path

    @contextmanager
    def _mocked_gates(self, *, replay_side_effect=None):
        with ExitStack() as stack:
            oracle = stack.enter_context(patch(
                "rl_leader.adapt_oracle_h12_to_balanced.validate_oracle_label_artifact"
            ))
            frozen = stack.enter_context(patch(
                "rl_leader.adapt_oracle_h12_to_balanced.validate_frozen_manifest"
            ))
            stack.enter_context(patch(
                "rl_leader.adapt_oracle_h12_to_balanced._implementation_fingerprints",
                return_value=CORE,
            ))
            replay = stack.enter_context(patch(
                "rl_leader.adapt_oracle_h12_to_balanced._replay_event",
                side_effect=replay_side_effect,
                return_value=(object(), SimpleNamespace(anchor_fingerprint="anchor")),
            ))
            yield oracle, frozen, replay

    def _assert_validators_called(self, oracle, frozen, source, manifest) -> None:
        oracle.assert_called_once_with(source, require_decision_horizon=True)
        frozen.assert_called_once_with(manifest, require_all_scenarios=True)

    def _assert_failure(self, source, manifest, message, *, replay_side_effect=None) -> None:
        source_path, manifest_path, output_path = self._paths(source, manifest)
        with self._mocked_gates(replay_side_effect=replay_side_effect) as gates:
            with self.assertRaisesRegex(ValueError, message):
                adapt_oracle_h12_to_balanced(source_path, manifest_path, output_path)
        self._assert_validators_called(*gates[:2], source, manifest)
        self.assertFalse(output_path.exists())

    def test_happy_schema_is_accepted_by_balanced_drain_out(self):
        source, manifest = _fixture()
        source_path, manifest_path, output_path = self._paths(source, manifest)
        with self._mocked_gates() as gates:
            result = adapt_oracle_h12_to_balanced(
                source_path, manifest_path, output_path
            )
        self._assert_validators_called(*gates[:2], source, manifest)
        event = manifest["scenarios"][0]["events"][0]
        gates[2].assert_called_once_with(manifest["scenarios"][0], event)

        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate_domain"], "urban")
        self.assertEqual(result["source_h1_artifact"], str(source_path))
        self.assertEqual(result["source_h1_sha256"], _sha(source_path))
        self.assertEqual(result["source_manifest_sha256"], _sha(manifest_path))
        self.assertEqual(result["frozen_event_replay"]["frozen_event_sha256"], _digest(event))
        self.assertEqual(
            result["adaptation_provenance"]["execution"],
            "no candidate rollout; frozen event replayed exactly",
        )
        self.assertEqual(result["outcomes"][0]["candidate_aliases"], [
            "structured:urban:A:axis:first", "structured:urban:B:axis:second",
        ])
        self.assertEqual(result["outcomes"][0]["h3_label"]["evidence"], "h3")

        with ExitStack() as stack:
            stack.enter_context(patch(
                "rl_leader.diagnose_balanced_drain_out.validate_oracle_label_artifact"
            ))
            stack.enter_context(patch(
                "rl_leader.diagnose_balanced_drain_out.validate_frozen_manifest"
            ))
            stack.enter_context(patch(
                "rl_leader.diagnose_balanced_drain_out._implementation_fingerprints",
                return_value=CORE,
            ))
            loaded = _validate_source(output_path)
        self.assertEqual(loaded[0]["source_h1_artifact"], str(source_path))
        self.assertEqual(loaded[4]["stratum"], "plateau")

    def test_nonpositive_source_fails_without_writing_output(self):
        source, manifest = _fixture(positive=False)
        self._assert_failure(source, manifest, "no positive H12")

    def test_implementation_drift_fails_closed(self):
        source, manifest = _fixture()
        source["implementation_sha256"] = {"core.py": "stale"}
        self._assert_failure(source, manifest, "implementation drift")

    def test_event_provenance_mismatch_fails_closed(self):
        source, manifest = _fixture()
        manifest["scenarios"][0]["events"][0]["forecast_sha256"] = "other-forecast"
        self._assert_failure(source, manifest, "frozen event mismatch")

    def test_alias_corruption_fails_closed(self):
        source, manifest = _fixture()
        source["candidate_pools"][0]["rows"][1]["candidate_aliases"] = [
            "structured:urban:B:axis:second"
        ]
        self._assert_failure(source, manifest, "inconsistent source aliases")

    def test_mixed_selected_domains_fail_closed(self):
        source, manifest = _fixture(second_domain="freeway")
        source["candidate_pools"][0]["candidate_mode"] = "owner_block_v2_all"
        source["candidate_pools"][0]["rows"][1]["horizon_labels"]["12"]["positive"] = True
        self._assert_failure(source, manifest, "mixed candidate domains")

    def test_physical_snapshot_corruption_fails_frozen_event_replay(self):
        source, manifest = _fixture()
        manifest["scenarios"][0]["events"][0]["physical_snapshot_sha256"] = "corrupt"

        def replay(_, event):
            if event["physical_snapshot_sha256"] != "physical-snapshot":
                raise ValueError("frozen event replay drift: ['physical_snapshot_sha256']")
            return object(), SimpleNamespace(anchor_fingerprint="anchor")

        self._assert_failure(source, manifest, "frozen event replay drift", replay_side_effect=replay)

    def test_controller_fingerprint_corruption_fails_frozen_event_replay(self):
        for field in ("rl_controller_sha256", "optimizer_controller_sha256"):
            with self.subTest(field=field):
                source, manifest = _fixture()
                manifest["scenarios"][0]["events"][0][field] = "corrupt"

                def replay(_, event, field=field):
                    if event[field] == "corrupt":
                        raise ValueError(f"frozen event replay drift: ['{field}']")
                    return object(), SimpleNamespace(anchor_fingerprint="anchor")

                self._assert_failure(
                    source, manifest, "frozen event replay drift", replay_side_effect=replay
                )

    @unittest.skipUnless(
        REAL_SOURCE.is_file() and REAL_MANIFEST.is_file(),
        "validated H12 source or frozen manifest is unavailable",
    )
    def test_real_artifact_replays_and_is_accepted_by_drain_out(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "adapted.json"
            started = time.monotonic()
            result = adapt_oracle_h12_to_balanced(
                REAL_SOURCE, REAL_MANIFEST, output_path
            )
            elapsed = time.monotonic() - started
            source, _, _, _, _, positives = _validate_source(output_path)

        self.assertTrue(result["passed"])
        self.assertTrue(source["frozen_event_replay"]["passed"])
        self.assertTrue(source["frozen_event_replay"]["anchor_fingerprint_exact"])
        self.assertGreater(len(positives), 0)
        self.assertGreater(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
