from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_leader.balanced_oracle_manifest import (
    BALANCED_SCENARIOS,
    DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS,
    FROZEN_MANIFEST_FORMAT_V4,
    _new_manifest,
    _new_dense_manifest,
    _new_dense_extra_manifest,
    _new_dense_extended_manifest,
    _partial_dense_events,
    dense_extra_events_for_scenario,
    dense_frozen_events_for_scenario,
    dense_stratum_for_policy_step,
    merge_dense_extended_manifests,
    frozen_events_for_scenario,
    validate_frozen_manifest,
    validate_oracle_against_frozen,
)
from rl_leader.diagnose_phase0_parity import _digest


def _event(stratum, step):
    simulation_step = 5 + step
    observation = [float(step)]
    anchor = [float(-step)]
    physical = {"step": simulation_step}
    return {
        "stratum": stratum,
        "stratum_predicate": {
            "simulation_time_sec": simulation_step * 180.0,
            "pulse_fraction": 0.5,
            "incident_active": False,
        },
        "policy_step": step,
        "simulation_step": simulation_step,
        "simulation_time_sec": simulation_step * 180.0,
        "native_anchor_branch": "coarse",
        "coordination_eligible": True,
        "anchor_fingerprint": f"anchor-{step}",
        "pre_runtime_sha256": f"runtime-{step}",
        "forecast_sha256": f"forecast-{step}",
        "forecast": [],
        "current_demand": {},
        "observation_sha256": _digest(observation),
        "observation": observation,
        "anchor_envelope_sha256": _digest(anchor),
        "anchor_envelope": anchor,
        "physical_snapshot_sha256": _digest(physical),
        "physical_snapshot": physical,
        "rl_controller_sha256": f"rl-{step}",
        "optimizer_controller_sha256": f"optimizer-{step}",
    }


def _scenario(name):
    contract = {"scenario": name}
    return {
        "scenario": name,
        "experiment_contract_version": "test",
        "experiment_contract_sha256": _digest(contract),
        "experiment_contract": contract,
        "action_schema": {},
        "observation_schema": {},
        "control_interval_sec": 180.0,
        "warmup_steps": 5,
        "events": [
            _event(stratum, step)
            for stratum, step in frozen_events_for_scenario(name)
        ],
    }


def _dense_scenario(name, *, eligible=True):
    contract = {"scenario": name}
    return {
        "scenario": name,
        "experiment_contract_version": "test",
        "experiment_contract_sha256": _digest(contract),
        "experiment_contract": contract,
        "action_schema": {},
        "observation_schema": {},
        "control_interval_sec": 180.0,
        "warmup_steps": 5,
        "events": [
            {
                **_event(stratum, step),
                "coordination_eligible": bool(eligible),
            }
            for stratum, step in dense_frozen_events_for_scenario(name)
        ],
    }


def _dense_extra_scenario(name, policy_steps, *, eligible=True):
    contract = {"scenario": name}
    return {
        "scenario": name,
        "experiment_contract_version": "test",
        "experiment_contract_sha256": _digest(contract),
        "experiment_contract": contract,
        "action_schema": {},
        "observation_schema": {},
        "control_interval_sec": 180.0,
        "warmup_steps": 5,
        "events": [
            {
                **_event(stratum, step),
                "coordination_eligible": bool(eligible),
            }
            for stratum, step in dense_extra_events_for_scenario(
                name, policy_steps
            )
        ],
    }


class BalancedOracleManifestTest(unittest.TestCase):
    def _manifest(self):
        manifest = _new_manifest()
        manifest["scenarios"] = [_scenario(name) for name in BALANCED_SCENARIOS]
        manifest["passed"] = True
        return manifest

    def test_complete_manifest_validates(self):
        manifest = self._manifest()
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ):
            validate_frozen_manifest(manifest, require_all_scenarios=True)

    def test_simulation_time_drift_is_rejected(self):
        manifest = self._manifest()
        manifest["scenarios"][0]["events"][1]["simulation_time_sec"] += 180.0
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ), self.assertRaisesRegex(ValueError, "simulation time drift"):
            validate_frozen_manifest(manifest, require_all_scenarios=True)

    def test_downstream_consumers_can_allow_implementation_drift(self):
        manifest = self._manifest()
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value={"changed.py": "sha"},
        ):
            with self.assertRaisesRegex(ValueError, "implementation drift"):
                validate_frozen_manifest(manifest, require_all_scenarios=True)
            validate_frozen_manifest(
                manifest,
                require_all_scenarios=True,
                allow_implementation_drift=True,
            )

    def test_downstream_consumers_can_allow_dependency_drift(self):
        manifest = _new_dense_manifest()
        manifest["scenarios"] = [
            _dense_scenario(name) for name in BALANCED_SCENARIOS
        ]
        manifest["passed"] = True
        manifest["dependency_sha256"] = {"changed.py": "sha"}
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ):
            with self.assertRaisesRegex(ValueError, "dependency drift"):
                validate_frozen_manifest(manifest, require_all_scenarios=True)
            validate_frozen_manifest(
                manifest,
                require_all_scenarios=True,
                allow_dependency_drift=True,
            )

    def test_oracle_round_trip_binds_anchor_runtime_and_forecast(self):
        frozen = _scenario(BALANCED_SCENARIOS[0])
        pools = []
        for event in frozen["events"]:
            pools.append({
                "policy_step": event["policy_step"],
                "simulation_step": event["simulation_step"],
                "simulation_time_sec": event["simulation_time_sec"],
                "native_anchor_branch": event["native_anchor_branch"],
                "anchor_fingerprint": event["anchor_fingerprint"],
                "pre_runtime_sha256": event["pre_runtime_sha256"],
                "forecast_sha256": event["forecast_sha256"],
            })
        result = validate_oracle_against_frozen(
            {"candidate_pools": pools}, frozen
        )
        self.assertTrue(result["passed"])
        peak_only = validate_oracle_against_frozen(
            {"candidate_pools": [pools[1]]},
            frozen,
            strata={frozen["events"][1]["stratum"]},
        )
        self.assertTrue(peak_only["passed"])
        drifted = copy.deepcopy(pools)
        drifted[1]["anchor_fingerprint"] = "drift"
        self.assertFalse(validate_oracle_against_frozen(
            {"candidate_pools": drifted}, frozen
        )["passed"])

    def test_oracle_round_trip_can_filter_policy_steps(self):
        frozen = _scenario(BALANCED_SCENARIOS[0])
        event = frozen["events"][1]
        pool = {
            "policy_step": event["policy_step"],
            "simulation_step": event["simulation_step"],
            "simulation_time_sec": event["simulation_time_sec"],
            "native_anchor_branch": event["native_anchor_branch"],
            "anchor_fingerprint": event["anchor_fingerprint"],
            "pre_runtime_sha256": event["pre_runtime_sha256"],
            "forecast_sha256": event["forecast_sha256"],
        }
        result = validate_oracle_against_frozen(
            {"candidate_pools": [pool]},
            frozen,
            policy_steps={event["policy_step"]},
        )
        self.assertTrue(result["passed"])

    def test_dense_plan_has_enough_candidate_event_slots(self):
        total = sum(
            len(dense_frozen_events_for_scenario(name))
            for name in BALANCED_SCENARIOS
        )
        self.assertGreaterEqual(total, DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS)
        incident = {
            step: stratum
            for stratum, step in dense_frozen_events_for_scenario(
                "sweet_170_incident_w60"
            )
        }
        self.assertEqual(incident[5], "incident_onset")
        self.assertEqual(incident[15], "incident_end")

    def test_dense_manifest_validates_separately_from_v2(self):
        manifest = _new_dense_manifest()
        manifest["scenarios"] = [
            _dense_scenario(name) for name in BALANCED_SCENARIOS
        ]
        manifest["passed"] = True
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ):
            validate_frozen_manifest(manifest, require_all_scenarios=True)

    def test_dense_manifest_requires_minimum_eligible_events(self):
        manifest = _new_dense_manifest()
        manifest["scenarios"] = [
            _dense_scenario(name, eligible=False) for name in BALANCED_SCENARIOS
        ]
        manifest["passed"] = True
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ), self.assertRaisesRegex(ValueError, "too few coordination-eligible"):
            validate_frozen_manifest(manifest, require_all_scenarios=True)

    def test_partial_dense_resume_requires_prefix_events(self):
        manifest = _new_dense_manifest()
        scenario = _dense_scenario(BALANCED_SCENARIOS[0])
        scenario["events"] = [scenario["events"][1]]
        manifest["scenarios"] = [scenario]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefix"):
                _partial_dense_events(path, BALANCED_SCENARIOS[0])

    def test_partial_dense_resume_accepts_prefix_events(self):
        manifest = _new_dense_manifest()
        scenario = _dense_scenario(BALANCED_SCENARIOS[0])
        scenario["events"] = scenario["events"][:2]
        manifest["scenarios"] = [scenario]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            events = _partial_dense_events(path, BALANCED_SCENARIOS[0])
        self.assertEqual([event["policy_step"] for event in events], [1, 3])

    def test_dense_extra_plan_maps_non_base_steps(self):
        self.assertEqual(
            dense_extra_events_for_scenario(BALANCED_SCENARIOS[0], (10, 12)),
            (("plateau", 10), ("plateau", 12)),
        )
        self.assertEqual(
            dense_stratum_for_policy_step("sweet_170_incident_w60", 15),
            "incident_end",
        )
        with self.assertRaisesRegex(ValueError, "duplicates dense base steps"):
            dense_extra_events_for_scenario(BALANCED_SCENARIOS[0], (10, 11))

    def test_dense_extra_manifest_validates(self):
        manifest = _new_dense_extra_manifest(BALANCED_SCENARIOS[0], (10, 12))
        manifest["scenarios"] = [
            _dense_extra_scenario(BALANCED_SCENARIOS[0], (10, 12))
        ]
        manifest["passed"] = True
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ):
            validate_frozen_manifest(manifest, require_all_scenarios=False)

    def test_dense_extended_manifest_validates_dynamic_plan(self):
        manifest = _new_dense_extended_manifest([
            _dense_scenario(name) for name in BALANCED_SCENARIOS
        ])
        manifest["scenarios"] = [
            _dense_scenario(name) for name in BALANCED_SCENARIOS
        ]
        manifest["passed"] = True
        with patch(
            "rl_leader.balanced_oracle_manifest._implementation_fingerprints",
            return_value=manifest["implementation_sha256"],
        ):
            validate_frozen_manifest(manifest, require_all_scenarios=True)

    def test_merge_dense_extended_manifests_adds_supplement_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_paths = []
            for name in BALANCED_SCENARIOS:
                manifest = _new_dense_manifest()
                manifest["scenarios"] = [_dense_scenario(name)]
                manifest["passed"] = True
                path = root / f"{name}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                base_paths.append(path)
            extra = _new_dense_extra_manifest(BALANCED_SCENARIOS[0], (10, 12))
            extra["scenarios"] = [
                _dense_extra_scenario(BALANCED_SCENARIOS[0], (10, 12))
            ]
            extra["passed"] = True
            extra_path = root / "extra.json"
            extra_path.write_text(json.dumps(extra), encoding="utf-8")
            out = root / "extended.json"
            result = merge_dense_extended_manifests(base_paths, [extra_path], out)
        self.assertEqual(result["format_version"], FROZEN_MANIFEST_FORMAT_V4)
        steps = [
            event["policy_step"] for event in result["scenarios"][0]["events"]
        ]
        self.assertIn(10, steps)
        self.assertIn(12, steps)
        self.assertEqual(steps, sorted(steps))


if __name__ == "__main__":
    unittest.main()
