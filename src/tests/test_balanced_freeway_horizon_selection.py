from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.evaluate_balanced_freeway_horizons import (
    BALANCED_HORIZON_FORMAT,
    _prefilter_freeway_representatives,
    _source_pool_and_event,
    evaluate_freeway_horizons,
    freeway_outcome_representatives,
    select_h12_freeway_representatives,
    select_minimal_freeway_execution_alias,
)


def _row(
    candidate_id,
    outcome,
    physical,
    *,
    aliases=None,
    residual=(0.0, 0.0),
    h1_gain=0.0,
    response=(0.0, 0.0),
):
    return {
        "candidate_id": candidate_id,
        "execution_branch": "coordination",
        "candidate_aliases": list(aliases or [candidate_id]),
        "response_memory_outcome_sha256": outcome,
        "post_physical_sha256": physical,
        "continuous_residual": list(residual),
        "uses_pcent_target": False,
        "candidate_response": list(response),
        "horizon_labels": {"1": {"ttt_gain": h1_gain}},
        "post_follower_sha256": "follower",
        "h1_step_ttt": 10.0,
        "h1_terminal_inventory": 2.0,
        "validity_gate_pass": True,
    }


def _evaluated(candidate_id, gain, distance, aliases=None):
    return {
        "candidate_id": candidate_id,
        "candidate_aliases": list(aliases or [candidate_id]),
        "h3_label": {"validity_gate_pass": True, "ttt_gain": gain},
        "native_response_distance": {"overall_rmse": distance},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreewayFilteringTests(unittest.TestCase):
    def test_filtering_dedupe_and_cross_domain_alias_preservation(self):
        freeway_a = "structured:freeway:R_D_W:axis:a"
        freeway_b = "structured:freeway:R_D_W:axis:b"
        urban = "structured:urban:A:axis:a"
        aliases = [urban, freeway_a, freeway_b]
        rows = [
            _row(urban, "shared", "physical", aliases=aliases),
            _row(freeway_b, "shared", "physical", aliases=aliases, residual=(0.5, 0.0)),
            _row(freeway_a, "shared", "physical", aliases=aliases),
        ]
        representatives = freeway_outcome_representatives({"rows": rows})

        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives[0]["candidate_id"], freeway_a)
        self.assertEqual(representatives[0]["candidate_aliases"], sorted(aliases))
        execution = select_minimal_freeway_execution_alias(
            {"rows": rows}, representatives[0]
        )
        self.assertEqual(execution["candidate_id"], freeway_a)

    def test_inconsistent_source_alias_provenance_fails_closed(self):
        first = _row(
            "structured:freeway:R_D_W:axis:a",
            "shared",
            "physical",
            aliases=["structured:freeway:R_D_W:axis:a", "structured:urban:A:axis:a"],
        )
        second = _row(
            "structured:urban:A:axis:a",
            "shared",
            "physical",
            aliases=["structured:urban:A:axis:a"],
        )
        with self.assertRaisesRegex(ValueError, "inconsistent source aliases"):
            freeway_outcome_representatives({"rows": [first, second]})

    def test_source_aliases_must_include_grouped_candidates(self):
        aliases = ["structured:freeway:R_D_W:axis:a"]
        first = _row("structured:freeway:R_D_W:axis:a", "shared", "physical", aliases=aliases)
        second = _row("structured:freeway:R_D_W:axis:b", "shared", "physical", aliases=aliases)
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            freeway_outcome_representatives({"rows": [first, second]})

    def test_extraneous_nonexistent_freeway_alias_fails_closed(self):
        candidate_id = "structured:freeway:R_D_W:axis:a"
        row = _row(
            candidate_id,
            "shared",
            "physical",
            aliases=[candidate_id, "structured:freeway:R_F_W:axis:nonexistent"],
        )
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            freeway_outcome_representatives({"rows": [row]})


class FreewaySelectionTests(unittest.TestCase):
    def test_prefilter_keeps_best_h1_candidate_per_owner(self):
        rows = [
            _row(
                "structured:freeway:R_D_W:axis:low-gain",
                "rdw-low",
                "p1",
                h1_gain=1.0,
                response=(9.0, 0.0),
            ),
            _row(
                "structured:freeway:R_D_W:axis:high-gain",
                "rdw-high",
                "p2",
                h1_gain=4.0,
                response=(1.0, 0.0),
            ),
            _row(
                "structured:freeway:R_F_W:axis:identity",
                "rfw-identity",
                "p3",
                h1_gain=100.0,
                response=(0.0, 0.0),
            ),
            _row(
                "structured:freeway:R_F_W:axis:nonidentity",
                "rfw-nonidentity",
                "p4",
                h1_gain=2.0,
                response=(2.0, 0.0),
            ),
        ]
        representatives = freeway_outcome_representatives({"rows": rows})

        with patch(
            "rl_leader.evaluate_balanced_freeway_horizons.response_distance",
            side_effect=lambda response, *_: {"overall_rmse": float(response[0])},
        ):
            selected = _prefilter_freeway_representatives(
                {"rows": rows},
                representatives,
                anchor_response=[],
                response_scales=None,
                response_families=None,
                max_per_owner=1,
            )

        self.assertEqual(
            [item[1]["candidate_id"] for item in selected],
            [
                "structured:freeway:R_D_W:axis:high-gain",
                "structured:freeway:R_F_W:axis:nonidentity",
            ],
        )

    def test_identity_is_excluded_from_owner_selection(self):
        rows = [
            _evaluated("structured:freeway:R_D_W:axis:identity", 9.0, 0.0),
            _evaluated("structured:freeway:R_D_W:axis:nonidentity", 1.0, 0.2),
            _evaluated("structured:freeway:R_F_W:axis:best", 2.0, 0.2),
            _evaluated("structured:freeway:R_D_E:axis:best", 3.0, 0.2),
            _evaluated("structured:freeway:R_F_E:axis:best", 4.0, 0.2),
        ]
        selected, _ = select_h12_freeway_representatives(rows)
        self.assertNotIn(
            "structured:freeway:R_D_W:axis:identity",
            [row["candidate_id"] for row in selected],
        )

    def test_cross_owner_collision_reuses_one_h12_execution(self):
        collision = "structured:freeway:R_D_W:axis:shared"
        rows = [
            _evaluated(
                collision,
                8.0,
                0.2,
                aliases=[
                    collision,
                    "structured:freeway:R_F_W:axis:shared",
                    "structured:urban:A:axis:shared",
                ],
            ),
            _evaluated("structured:freeway:R_D_E:axis:best", 3.0, 0.2),
            _evaluated("structured:freeway:R_F_E:axis:best", 4.0, 0.2),
        ]
        selected, evidence = select_h12_freeway_representatives(rows)

        self.assertEqual(len(selected), 3)
        self.assertEqual(evidence["selected_by"][collision], [
            "owner_h3_best:R_D_W", "owner_h3_best:R_F_W",
        ])
        self.assertEqual(evidence["owner_coverage"], [
            "R_D_W", "R_F_W", "R_D_E", "R_F_E",
        ])

    def test_missing_required_owner_fails_closed(self):
        rows = [
            _evaluated("structured:freeway:R_D_W:axis:best", 1.0, 0.2),
            _evaluated("structured:freeway:R_F_W:axis:best", 2.0, 0.2),
            _evaluated("structured:freeway:R_D_E:axis:best", 3.0, 0.2),
        ]
        with self.assertRaisesRegex(ValueError, "does not cover exactly"):
            select_h12_freeway_representatives(rows)


class FreewayEvaluationTests(unittest.TestCase):
    def _fixture(self, directory: Path):
        source_path = directory / "h1.json"
        manifest_path = directory / "manifest.json"
        output_path = directory / "output.json"
        owners = ("R_D_W", "R_F_W", "R_D_E", "R_F_E")
        rows = [
            _row(f"structured:freeway:{owner}:axis:a", f"outcome-{owner}", f"physical-{owner}")
            for owner in owners
        ]
        pool = {
            "scenario": "sweet_155_w60",
            "policy_step": 1,
            "simulation_step": 101,
            "simulation_time_sec": 120.0,
            "native_anchor_branch": "refined",
            "pre_runtime_sha256": "pre-runtime",
            "forecast_sha256": "forecast",
            "anchor_fingerprint": "anchor",
            "observation": [1.0, 2.0],
            "anchor_envelope": [3.0, 4.0],
            "candidate_mode": "owner_block_v2_all",
            "rows": rows,
        }
        event = {
            "policy_step": 1,
            "stratum": "ramp_up",
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
        source = {"implementation_sha256": {"core": "sha"}, "candidate_pools": [pool]}
        manifest = {"scenarios": [{"scenario": pool["scenario"], "events": [event]}]}
        source_path.write_text(json.dumps(source), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return source_path, manifest_path, output_path, source, manifest

    def _rollout(self):
        checkpoints = {
            "1": {
                "follower_memory_sha256": "follower", "physical_state_sha256": "physical",
                "ttt": 10.0, "terminal_inventory": 2.0, "validity_gate_pass": True,
            },
            "3": {"ttt": 8.0, "terminal_inventory": 1.0, "validity_gate_pass": True},
            "12": {"ttt": 6.0, "terminal_inventory": 0.5, "validity_gate_pass": True},
        }
        return {"first_step_response": [0.0, 0.0], "checkpoints": checkpoints}

    def _patches(self, module, *, h1_exact=True, h3_passed=True):
        rollout = self._rollout()
        return (
            patch(f"{module}.validate_oracle_label_artifact"),
            patch(f"{module}.validate_frozen_manifest"),
            patch(f"{module}._implementation_fingerprints", return_value={"core": "sha"}),
            patch(f"{module}._replay_event", return_value=(object(), SimpleNamespace(anchor_fingerprint="anchor"))),
            patch(f"{module}._rollout_pstack", return_value=rollout),
            patch(f"{module}._response_scales_and_families", return_value=(None, None)),
            patch(f"{module}.response_distance", return_value={"overall_rmse": 0.2}),
            patch(f"{module}._rollout_price_candidate", return_value=rollout),
            patch(f"{module}._require_rollout_coverage"),
            patch(f"{module}._h1_exact", return_value=h1_exact),
            patch(f"{module}._horizon_label", return_value={"validity_gate_pass": True, "ttt_gain": 1.0}),
            patch(f"{module}._h3_selection_replay_evidence", return_value={"passed": h3_passed}),
        )

    @contextmanager
    def _patch_context(self, module, **kwargs):
        with ExitStack() as stack:
            for patcher in self._patches(module, **kwargs):
                stack.enter_context(patcher)
            yield

    def test_pool_event_provenance_mismatch_fails_closed(self):
        module = "rl_leader.evaluate_balanced_freeway_horizons"
        with tempfile.TemporaryDirectory() as raw:
            _, _, _, source, manifest = self._fixture(Path(raw))
            source["candidate_pools"][0]["forecast_sha256"] = "wrong-forecast"
            with patch(f"{module}._implementation_fingerprints", return_value={"core": "sha"}):
                with self.assertRaisesRegex(ValueError, "frozen event mismatch"):
                    _source_pool_and_event(source, manifest)

    def test_exact_replay_mismatch_fails_closed(self):
        module = "rl_leader.evaluate_balanced_freeway_horizons"
        with tempfile.TemporaryDirectory() as raw:
            source, manifest, output, _, _ = self._fixture(Path(raw))
            with self._patch_context(module, h1_exact=False):
                with self.assertRaisesRegex(ValueError, "failed H1 replay"):
                    evaluate_freeway_horizons(source, manifest, output)

    def test_h3_replay_mismatch_fails_closed(self):
        module = "rl_leader.evaluate_balanced_freeway_horizons"
        with tempfile.TemporaryDirectory() as raw:
            source, manifest, output, _, _ = self._fixture(Path(raw))
            with self._patch_context(module, h3_passed=False):
                with self.assertRaisesRegex(ValueError, "failed H3 replay"):
                    evaluate_freeway_horizons(source, manifest, output)

    def test_provenance_failure_fails_closed(self):
        module = "rl_leader.evaluate_balanced_freeway_horizons"
        with tempfile.TemporaryDirectory() as raw:
            source, manifest, output, _, _ = self._fixture(Path(raw))
            with self._patch_context(module):
                with patch(f"{module}._implementation_fingerprints", return_value={"changed": "sha"}):
                    with self.assertRaisesRegex(ValueError, "implementation drift"):
                        evaluate_freeway_horizons(source, manifest, output)

    def test_schema_is_drain_out_compatible(self):
        module = "rl_leader.evaluate_balanced_freeway_horizons"
        with tempfile.TemporaryDirectory() as raw:
            source, manifest, output, _, _ = self._fixture(Path(raw))
            with self._patch_context(module):
                result = evaluate_freeway_horizons(source, manifest, output)

            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result, written)
            written["outcomes"][0]["h12_label"] = {"positive": True}
            output.write_text(json.dumps(written), encoding="utf-8")
            self.assertEqual(written["format_version"], BALANCED_HORIZON_FORMAT)
            self.assertEqual(written["candidate_domain"], "freeway")
            self.assertEqual(written["source_h1_sha256"], _sha(source))
            self.assertEqual(written["source_manifest_sha256"], _sha(manifest))
            self.assertTrue(written["passed"])
            self.assertEqual(written["selector_version"], "v2")
            self.assertIn("sidecar_sha256", written)
            self.assertIn("core_implementation_sha256", written)
            from rl_leader.diagnose_balanced_drain_out import _validate_source

            with (
                patch(
                    "rl_leader.diagnose_balanced_drain_out._implementation_fingerprints",
                    return_value={"core": "sha"},
                ),
                patch("rl_leader.diagnose_balanced_drain_out.validate_oracle_label_artifact"),
                patch("rl_leader.diagnose_balanced_drain_out.validate_frozen_manifest"),
            ):
                _, _, _, frozen, event, positives = _validate_source(output)
            self.assertEqual(frozen["scenario"], written["scenario"])
            self.assertEqual(event["stratum"], written["stratum"])
            self.assertEqual(len(positives), 1)


if __name__ == "__main__":
    unittest.main()
