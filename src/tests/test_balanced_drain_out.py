from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rl_leader.diagnose_balanced_drain_out import (
    ZeroDemandAfterCutoff,
    _cutoff_check,
    _inventory_parity,
    _rollout_drain_out,
    component_inventory,
    drain_out_verdict,
    run_balanced_drain_out,
    select_drain_candidates,
    update_clearance_stability,
)
from rl_leader.env import RLLeaderEnv
from src.models.demand import DemandStep


class _FakeProfile:
    def __init__(self):
        self.cfg = SimpleNamespace(
            simulation=SimpleNamespace(control_interval=60.0)
        )
        self.marker = "delegated"

    def at(self, time_sec):
        return DemandStep(
            freeway_mainline={"FW": 1000.0 + time_sec},
            urban_boundary={"in": 500.0, "out": 400.0},
            ramp_arrival={"R": 200.0},
            incident_capacity_factor=0.7,
            freeway_lane_loss={"FW": {2: 1.0}},
        )


class _FakeState:
    freeway_buffer_up_density = {"FW": [2.0, -1.0]}
    freeway_buffer_down_density = {"FW": [3.0]}

    def total_urban_vehicles(self, net):
        return 11.0

    def total_freeway_vehicles(self, net):
        return 13.0

    def off_ramp_storage_occupancy_veh(self, net):
        return 17.0


class _RolloutProfile:
    def __init__(self):
        self.cfg = SimpleNamespace(
            simulation=SimpleNamespace(control_interval=1200.0)
        )

    def at(self, time_sec):
        return DemandStep(
            freeway_mainline={"FW": 1.0},
            urban_boundary={"in": 1.0},
            ramp_arrival={"R": 1.0},
            incident_capacity_factor=0.8,
            freeway_lane_loss={"FW": {1: 1.0}},
        )


class _RolloutState:
    def __init__(self):
        self.inventory = 1.0
        self.time_sec = 0.0
        self.freeway_buffer_up_density = {}
        self.freeway_buffer_down_density = {}

    def total_urban_vehicles(self, net):
        return float(self.inventory)

    def total_freeway_vehicles(self, net):
        return 0.0

    def off_ramp_storage_occupancy_veh(self, net):
        return 0.0


class _FakeRolloutEnv:
    def __init__(self, tail_inventories=(0.05, 0.05)):
        self.T_total = 14400.0
        self.dt = 1200.0
        self.n_steps = 12
        self.step_idx = 0
        self.cfg = SimpleNamespace(
            simulation=SimpleNamespace(T_total=14400.0, control_interval=1200.0)
        )
        self.net = SimpleNamespace(
            freeway_segment_length_km=0.5,
            freeway_lanes=2.0,
        )
        self.profile = _RolloutProfile()
        self.sim = SimpleNamespace(state=_RolloutState())
        self.tail_inventories = list(tail_inventories)
        self.first_action_n_steps = []
        self.pstack_n_steps = []
        self.first_action_seen = False

    def _inventory(self):
        return float(self.sim.state.inventory)

    def _step(self, action):
        if action in {"candidate", "native"}:
            if self.first_action_seen:
                raise AssertionError("more than one first action")
            if self.n_steps != 12:
                raise AssertionError("n_steps was extended before the first action")
            self.first_action_seen = True
            self.first_action_n_steps.append(int(self.n_steps))
        else:
            if not self.first_action_seen:
                raise AssertionError("P-Stack ran before the first action")
            if self.n_steps != 15:
                raise AssertionError("P-Stack ran before n_steps extension")
            self.pstack_n_steps.append(int(self.n_steps))
        demand = self.profile.at(self.step_idx * self.dt)
        arrivals = float(
            sum(demand.freeway_mainline.values())
            + sum(demand.urban_boundary.values())
            + sum(demand.ramp_arrival.values())
        )
        if self.step_idx >= 12:
            tail_index = self.step_idx - 12
            self.sim.state.inventory = float(
                self.tail_inventories[min(tail_index, len(self.tail_inventories) - 1)]
            )
        else:
            self.sim.state.inventory = 1.0
        self.step_idx += 1
        self.sim.state.time_sec += self.dt
        info = {
            "validity_gate_pass": 1.0,
            "external_arrivals_veh": arrivals,
            "completed_departures_veh": 0.0,
            "conservation_residual_veh": 0.0,
            "movement_queue_projection_veh": 0.0,
            "coupling_offramp_arrivals_rejected_veh": 0.0,
            "queue_overflow_count": 0.0,
        }
        return -1.0, self.step_idx >= self.n_steps, info

    def step_anchored_candidate(self, residual, context):
        reward, done, info = self._step("candidate")
        return None, reward, done, info

    def step_prepared_optimizer_anchor(self, context, *, sync_follower_state=True):
        reward, done, info = self._step("native")
        return None, reward, done, info, np.zeros(1, dtype=np.float32)

    def step_optimizer_anchor(self, *, sync_follower_state=True):
        reward, done, info = self._step("pstack")
        return None, reward, done, info, np.zeros(1, dtype=np.float32)


def _metrics(**updates):
    base = {
        "drain_complete": True,
        "validity_all_steps": True,
        "cumulative_external_arrivals_veh": 100.0,
        "main_external_arrivals_veh": 100.0,
        "tail_external_arrivals_veh": 0.0,
        "max_abs_tail_step_external_arrivals_veh": 0.0,
        "main_external_arrivals_sequence_veh": [50.0, 50.0],
        "main_external_arrivals_sha256": "same-main-arrivals",
        "cutoff_check": {"passed": True},
        "inventory_parity": {
            "main_boundary": {"exact": True},
            "terminal": {"exact": True},
        },
        "max_abs_conservation_residual_veh": 0.0005,
        "cumulative_projection_veh": 0.0,
        "cumulative_rejected_flow_veh": 0.0,
        "cumulative_overflow_count": 0.0,
        "total_ttt": 100.0,
    }
    base.update(updates)
    return base


class ZeroDemandProfileTests(unittest.TestCase):
    def test_wrapper_delegates_before_cutoff_and_zeroes_external_inflow_after(self):
        wrapped = ZeroDemandAfterCutoff(_FakeProfile(), 120.0)
        before = wrapped.at(60.0)
        after = wrapped.at(120.0)

        self.assertEqual(before.freeway_mainline, {"FW": 1060.0})
        self.assertEqual(after.freeway_mainline, {"FW": 0.0})
        self.assertEqual(after.urban_boundary, {"in": 0.0, "out": 0.0})
        self.assertEqual(after.ramp_arrival, {"R": 0.0})
        self.assertEqual(after.incident_capacity_factor, 0.7)
        self.assertEqual(after.freeway_lane_loss, {"FW": {2: 1.0}})
        self.assertEqual(wrapped.marker, "delegated")

    def test_horizon_uses_wrapped_at_across_cutoff(self):
        wrapped = ZeroDemandAfterCutoff(_FakeProfile(), 120.0)
        horizon = wrapped.horizon(60.0, 3)
        self.assertEqual([step.freeway_mainline["FW"] for step in horizon], [1060.0, 0.0, 0.0])


class ComponentInventoryTests(unittest.TestCase):
    def test_components_sum_to_environment_inventory_contract(self):
        net = SimpleNamespace(freeway_segment_length_km=0.5, freeway_lanes=2.0)
        values = component_inventory(_FakeState(), net)
        expected_buffers = (2.0 + 3.0) * 0.5 * 2.0
        self.assertEqual(values["freeway_buffers"], expected_buffers)
        self.assertEqual(values["total"], 11.0 + 13.0 + 17.0 + expected_buffers)
        self.assertEqual(
            values["total"],
            sum(values[key] for key in ("urban", "freeway", "off_ramp_storage", "freeway_buffers")),
        )

    def test_real_environment_component_sum_has_exact_parity(self):
        env = RLLeaderEnv(T_total=14400.0)
        values = component_inventory(env.sim.state, env.net)
        self.assertEqual(values["total"], env._inventory())

    def test_component_parity_mismatch_fails_closed(self):
        env = SimpleNamespace(_inventory=lambda: 2.0)
        with self.assertRaisesRegex(RuntimeError, "component inventory parity failure"):
            _inventory_parity(env, {"total": 1.0}, "test")

    def test_component_parity_allows_float_rounding_error(self):
        env = SimpleNamespace(_inventory=lambda: 423.46652701664266)
        evidence = _inventory_parity(
            env, {"total": 423.4665270166427}, "rounding"
        )
        self.assertTrue(evidence["exact"])
        self.assertLess(evidence["absolute_error_veh"], 1.0e-9)


class ClearanceTests(unittest.TestCase):
    def test_clearance_requires_consecutive_stable_steps(self):
        count = 0
        observed = []
        for inventory in (0.05, 0.2, 0.1, 0.09):
            count, complete = update_clearance_stability(
                inventory,
                threshold=0.1,
                stable_count=count,
                required_stable_steps=2,
            )
            observed.append((count, complete))
        self.assertEqual(observed, [(1, False), (0, False), (1, False), (2, True)])


class FakeRolloutTests(unittest.TestCase):
    def _run(self, *, residual, tail_inventories=(0.05, 0.05), tail_cap=3):
        with (
            patch(
                "rl_leader.diagnose_balanced_drain_out._terminal_follower_hash",
                return_value="follower",
            ),
            patch(
                "rl_leader.diagnose_balanced_drain_out._terminal_physical_hash",
                return_value="physical",
            ),
        ):
            return _rollout_drain_out(
                _FakeRolloutEnv(tail_inventories),
                SimpleNamespace(),
                residual=residual,
                tail_cap_steps=tail_cap,
                inventory_threshold=0.1,
                stable_steps=2,
            )

    def test_boundary_demand_call_order_extension_and_clearance(self):
        candidate = self._run(residual=np.zeros(1, dtype=np.float32))
        native = self._run(residual=None)

        for rollout, first_action in (
            (candidate, "candidate"),
            (native, "native_pstack"),
        ):
            self.assertEqual(rollout["first_action"], first_action)
            self.assertEqual(rollout["n_steps_before_first_action"], 12)
            self.assertEqual(rollout["n_steps_after_first_action_before_extension"], 12)
            self.assertEqual(rollout["extended_n_steps"], 15)
            self.assertTrue(rollout["n_steps_extension_applied_after_first_action"])
            self.assertEqual(rollout["main_steps"], 12)
            self.assertEqual(rollout["tail_steps"], 2)
            self.assertEqual(rollout["pstack_steps_after_first_action"], 13)
            self.assertEqual(rollout["main_external_arrivals_sequence_veh"], [3.0] * 12)
            self.assertEqual(rollout["tail_external_arrivals_sequence_veh"], [0.0, 0.0])
            self.assertEqual(rollout["tail_external_arrivals_veh"], 0.0)
            self.assertEqual(rollout["max_abs_tail_step_external_arrivals_veh"], 0.0)
            self.assertIsInstance(rollout["main_external_arrivals_sha256"], str)
            self.assertIsInstance(rollout["tail_external_arrivals_sha256"], str)
            self.assertIsInstance(rollout["external_arrivals_sha256"], str)
            self.assertTrue(rollout["drain_complete"])
            self.assertEqual(rollout["stable_steps_observed"], 2)
            self.assertEqual(rollout["tail_clearance_time_sec"], 2400.0)
            self.assertTrue(rollout["cutoff_check"]["passed"])
            self.assertTrue(rollout["inventory_parity"]["main_boundary"]["exact"])
            self.assertTrue(rollout["inventory_parity"]["terminal"]["exact"])

        checks = drain_out_verdict(candidate, native)["checks"]
        self.assertTrue(checks["paired_main_external_arrival_sequence_exact"])
        self.assertTrue(checks["paired_main_external_arrival_hash_exact"])

    def test_v1_cutoff_contract_rejects_any_value_other_than_14400(self):
        env = _FakeRolloutEnv()
        env.T_total = 14399.0
        with self.assertRaisesRegex(RuntimeError, "cutoff contract mismatch"):
            _cutoff_check(env, env.n_steps)

    def test_tail_cap_produces_quarantine_without_stable_clearance(self):
        candidate = self._run(
            residual=np.zeros(1, dtype=np.float32),
            tail_inventories=(1.0, 1.0, 1.0),
            tail_cap=3,
        )
        native = self._run(
            residual=None,
            tail_inventories=(1.0, 1.0, 1.0),
            tail_cap=3,
        )
        self.assertEqual(candidate["tail_steps"], 3)
        self.assertFalse(candidate["drain_complete"])
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "quarantine")
        self.assertFalse(verdict["final_negative"])


class DrainCandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def _source():
        return {"outcomes": [
            {
                "candidate_id": "positive",
                "h12_label": {"validity_gate_pass": True, "positive": True},
            },
            {
                "candidate_id": "negative",
                "h12_label": {"validity_gate_pass": True, "positive": False},
            },
            {
                "candidate_id": "invalid",
                "h12_label": {"validity_gate_pass": False, "positive": True},
            },
        ]}

    def test_default_selects_only_h12_positive(self):
        selected = select_drain_candidates(self._source())
        self.assertEqual([row["candidate_id"] for row in selected], ["positive"])

    def test_explicit_mode_can_select_h12_nonpositive(self):
        selected = select_drain_candidates(self._source(), ("negative",))
        self.assertEqual([row["candidate_id"] for row in selected], ["negative"])

    def test_explicit_mode_rejects_invalid_or_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "absent or invalid"):
            select_drain_candidates(self._source(), ("invalid",))
        with self.assertRaisesRegex(ValueError, "nonempty unique"):
            select_drain_candidates(self._source(), ("negative", "negative"))


class VerdictTests(unittest.TestCase):
    def test_positive_requires_strict_total_ttt_margin(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(total_ttt=998.0)
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "positive")
        self.assertTrue(verdict["final_positive"])
        self.assertEqual(verdict["required_gain"], 1.0)

    def test_margin_failure_is_negative_only_after_valid_drain(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(total_ttt=999.5)
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "negative")
        self.assertTrue(verdict["final_negative"])
        self.assertFalse(verdict["invalid"])

    def test_incomplete_drain_is_quarantined_not_negative(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(total_ttt=900.0, drain_complete=False)
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "quarantine")
        self.assertTrue(verdict["quarantine"])
        self.assertFalse(verdict["final_positive"])
        self.assertFalse(verdict["final_negative"])

    def test_failed_quality_gate_is_invalid(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(total_ttt=900.0, cumulative_projection_veh=0.01)
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "invalid")
        self.assertFalse(verdict["final_positive"])
        self.assertFalse(verdict["final_negative"])

    def test_paired_arrival_mismatch_is_invalid(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(
            total_ttt=900.0,
            main_external_arrivals_sequence_veh=[50.0, 49.999999999],
            main_external_arrivals_sha256="different-main-arrivals",
        )
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "invalid")
        self.assertFalse(
            verdict["checks"]["paired_main_external_arrival_sequence_exact"]
        )
        self.assertFalse(verdict["checks"]["paired_main_external_arrival_hash_exact"])

    def test_nonzero_tail_arrival_is_invalid(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(
            total_ttt=900.0,
            tail_external_arrivals_veh=0.01,
            max_abs_tail_step_external_arrivals_veh=0.01,
        )
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "invalid")
        self.assertFalse(verdict["checks"]["candidate_tail_external_arrivals_zero"])
        self.assertFalse(
            verdict["checks"]["candidate_max_tail_step_external_arrivals_zero"]
        )

    def test_inventory_parity_gate_is_required(self):
        native = _metrics(total_ttt=1000.0)
        candidate = _metrics(total_ttt=900.0)
        candidate["inventory_parity"] = {
            "main_boundary": {"exact": True},
            "terminal": {"exact": False},
        }
        verdict = drain_out_verdict(candidate, native)
        self.assertEqual(verdict["status"], "invalid")
        self.assertFalse(
            verdict["checks"]["candidate_terminal_inventory_parity_exact"]
        )


class DrainOutOrchestrationTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        source = {
            "source_h1_artifact": "mock_h1.json",
            "source_h1_sha256": "h1-sha",
            "source_manifest": "mock_manifest.json",
            "source_manifest_sha256": "manifest-sha",
            "scenario": "sweet_190_w60",
            "stratum": "recovery_boundary",
            "policy_step": 42,
            "anchor_fingerprint": "anchor-sha",
        }
        frozen = {"scenario": source["scenario"]}
        event = {"policy_step": 42, "stratum": source["stratum"]}
        positive = {
            "candidate_id": "candidate-D",
            "representative_candidate_id": "representative-D",
            "candidate_aliases": ["candidate-D"],
            "continuous_residual": [0.25, -0.5],
            "h12_label": {"positive": True, "native_ttt": 12.0},
        }
        native = _metrics(total_ttt=100.0)
        candidate = _metrics(total_ttt=90.0)
        validated = (source, {}, {}, frozen, event, [positive])
        context = SimpleNamespace(anchor_fingerprint=source["anchor_fingerprint"])
        return source, event, positive, native, candidate, validated, context

    def test_exact_h12_replay_success_writes_complete_passed_outcome(self):
        (
            source,
            _,
            positive,
            native,
            candidate,
            validated,
            context,
        ) = self._fixture()
        module = "rl_leader.diagnose_balanced_drain_out"
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "output.json"
            source_path.write_text("{}", encoding="utf-8")
            with (
                patch(f"{module}._validate_source", return_value=validated),
                patch(f"{module}._implementation_fingerprints", return_value={}),
                patch(f"{module}._replay_event", return_value=(object(), context)),
                patch(
                    f"{module}._rollout_drain_out",
                    side_effect=[native, candidate],
                ) as rollout,
                patch(
                    f"{module}.exact_balanced_h12_replay",
                    return_value={"passed": True, "candidate_ttt_exact": True},
                ) as exact_replay,
            ):
                result = run_balanced_drain_out(source_path, output_path)

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertTrue(written["passed"])
            self.assertEqual(written["status"], "complete")
            self.assertEqual(written["scenario"], source["scenario"])
            self.assertEqual(len(written["outcomes"]), 1)
            outcome = written["outcomes"][0]
            self.assertEqual(outcome["candidate_id"], positive["candidate_id"])
            self.assertTrue(outcome["h12_replay"]["passed"])
            self.assertEqual(outcome["verdict"]["status"], "positive")
            self.assertEqual(rollout.call_count, 2)
            exact_replay.assert_called_once_with(
                candidate, native, positive["h12_label"]
            )

    def test_h12_replay_mismatch_raises_and_writes_failed_result(self):
        (
            _,
            _,
            positive,
            native,
            candidate,
            validated,
            context,
        ) = self._fixture()
        module = "rl_leader.diagnose_balanced_drain_out"
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "output.json"
            source_path.write_text("{}", encoding="utf-8")
            with (
                patch(f"{module}._validate_source", return_value=validated),
                patch(f"{module}._implementation_fingerprints", return_value={}),
                patch(f"{module}._replay_event", return_value=(object(), context)),
                patch(
                    f"{module}._rollout_drain_out",
                    side_effect=[native, candidate],
                ),
                patch(
                    f"{module}.exact_balanced_h12_replay",
                    return_value={"passed": False, "candidate_ttt_exact": False},
                ) as exact_replay,
            ):
                with self.assertRaisesRegex(RuntimeError, "failed exact H12 replay"):
                    run_balanced_drain_out(source_path, output_path)

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(written["passed"])
            self.assertEqual(written["status"], "failed")
            self.assertEqual(written["outcomes"], [])
            self.assertEqual(written["error"]["type"], "RuntimeError")
            self.assertIn("failed exact H12 replay", written["error"]["message"])
            exact_replay.assert_called_once_with(
                candidate, native, positive["h12_label"]
            )


if __name__ == "__main__":
    unittest.main()
