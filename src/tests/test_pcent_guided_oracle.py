from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from rl_leader.diagnose_pcent_guided_oracle import (
    OracleGateError,
    _dedupe_candidates,
    _event_integrity_passed,
    _long_zero_parity,
    _validate_branch_contract,
    choose_oracle_candidate,
    parse_event_steps,
    select_probe_candidates,
)
from rl_leader.diagnose_pcent_reachability import _directed_groups
from rl_leader.env import RLLeaderEnv
from src.models.state import ControlAction


class PcentGuidedOracleTest(unittest.TestCase):
    def test_event_parser_uses_canonical_defaults_and_rejects_mismatch(self):
        self.assertEqual(
            parse_event_steps("", ("sweet_170_w60",)),
            {"sweet_170_w60": (9, 13, 25)},
        )
        self.assertEqual(
            parse_event_steps("sweet_170_w60:5,2,5", ("sweet_170_w60",)),
            {"sweet_170_w60": (2, 5)},
        )
        with self.assertRaisesRegex(ValueError, "scenario mismatch"):
            parse_event_steps("sweet_190_w60:2", ("sweet_170_w60",))

    def test_candidate_dedupe_keeps_distinct_follower_memory(self):
        base = {
            "representative_label": "a",
            "aliases": ["a"],
            "response": [1.0, 2.0],
            "post_follower_sha256": "memory-a",
            "post_physical_sha256": "physical-a",
            "distance_to_pcent": {"overall_rmse": 0.2},
            "step_ttt": 10.0,
        }
        alias = {**base, "representative_label": "b", "aliases": ["b"]}
        distinct = {
            **base,
            "representative_label": "c",
            "aliases": ["c"],
            "post_follower_sha256": "memory-c",
        }
        distinct_physical = {
            **base,
            "representative_label": "d",
            "aliases": ["d"],
            "post_physical_sha256": "physical-d",
        }

        rows = _dedupe_candidates([base, alias, distinct, distinct_physical])

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["aliases"], ["a", "b"])

    def test_oracle_defaults_to_anchor_without_guarded_positive(self):
        rejected = [{
            "representative_label": "candidate",
            "residual_nonzero": {"urban.A.g_green": 0.5},
            "rollout": {"validity_gate_pass": True, "ttt": 9.0},
            "paired": {"long_horizon_positive": False, "ttt_gain": 1.0},
        }]

        self.assertEqual(choose_oracle_candidate(rejected)["selected"], "anchor")

        accepted = copy.deepcopy(rejected)
        accepted[0]["paired"]["long_horizon_positive"] = True
        self.assertEqual(
            choose_oracle_candidate(accepted)["selected"], "candidate"
        )

    def test_coordination_zero_requires_an_explicit_branch_switch(self):
        zero = {
            "representative_label": "coordination_zero",
            "residual_nonzero": {},
            "validity_gate_pass": True,
        }

        self.assertEqual(select_probe_candidates([zero], 1), [])
        self.assertEqual(
            select_probe_candidates(
                [zero], 1, include_coordination_zero=True
            ),
            [zero],
        )

        zero_parity = {
            "anchor_selected_branch": "fallback_pfo",
            "coordination_zero_is_native_identity": False,
            "identity_execution_branch": "native_anchor",
        }
        choice = {
            "selected": "coordination_zero",
            "execution_branch": "coordination",
            "residual_nonzero": {},
        }
        evaluated = [{
            "representative_label": "coordination_zero",
            "execution_branch": "coordination",
        }]
        self.assertTrue(
            _validate_branch_contract(
                zero_parity, evaluated, choice
            )["passed"]
        )

        zero_parity["coordination_zero_is_native_identity"] = True
        zero_parity["identity_execution_branch"] = "coordination_zero"
        with self.assertRaisesRegex(OracleGateError, "branch contract failed"):
            _validate_branch_contract(zero_parity, evaluated, choice)

    def test_long_zero_parity_checks_every_requested_horizon(self):
        rollout = {
            "steps": 6,
            "ttt": 12.0,
            "terminal_inventory": 30.0,
            "first_step_response": [1.0],
            "validity_gate_pass": True,
            "terminal_follower_memory_sha256": "terminal-memory",
            "terminal_physical_state_sha256": "terminal-physical",
            "checkpoints": {
                "1": {
                    "ttt": 2.0,
                    "terminal_inventory": 10.0,
                    "follower_memory_sha256": "memory-1",
                    "physical_state_sha256": "physical-1",
                },
                "3": {
                    "ttt": 6.0,
                    "terminal_inventory": 20.0,
                    "follower_memory_sha256": "memory-3",
                    "physical_state_sha256": "physical-3",
                },
                "6": {
                    "ttt": 12.0,
                    "terminal_inventory": 30.0,
                    "follower_memory_sha256": "terminal-memory",
                    "physical_state_sha256": "terminal-physical",
                },
            },
        }

        self.assertTrue(
            _long_zero_parity(rollout, copy.deepcopy(rollout), (1, 3, 6))["passed"]
        )
        changed = copy.deepcopy(rollout)
        changed["checkpoints"]["3"]["ttt"] += 0.1
        with self.assertRaisesRegex(RuntimeError, "h3_exact"):
            _long_zero_parity(rollout, changed, (1, 3, 6))

        changed = copy.deepcopy(rollout)
        changed["checkpoints"]["6"]["follower_memory_sha256"] = "different"
        with self.assertRaisesRegex(OracleGateError, "h6_exact"):
            _long_zero_parity(rollout, changed, (1, 3, 6))

        truncated = copy.deepcopy(rollout)
        truncated["steps"] = 3
        truncated["checkpoints"].pop("6")
        with self.assertRaisesRegex(OracleGateError, "every requested horizon"):
            _long_zero_parity(truncated, copy.deepcopy(truncated), (1, 3, 6))

    def test_event_integrity_fails_closed_when_order_check_was_skipped(self):
        event = {
            "zero_residual_parity": {"passed": True},
            "long_horizon_identity_parity": {"passed": True},
            "branch_contract": {"passed": True},
            "source_query_isolation": {"passed": True},
            "candidate_order_invariance": {"performed": True, "passed": True},
        }
        self.assertTrue(_event_integrity_passed(event))

        event["candidate_order_invariance"] = {
            "performed": False,
            "passed": None,
        }
        self.assertFalse(_event_integrity_passed(event))

    def test_pcent_direction_uses_schema_names_instead_of_fixed_offsets(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        anchor = ControlAction.fixed(env.cfg)
        target = anchor.copy()
        signal = env.action_schema.signals[0]
        target.green_times[f"{signal}_p1"] += 6.0
        reordered_names = tuple(reversed(env.action_schema.names))
        proxy = SimpleNamespace(
            action_dim=env.action_dim,
            action_schema=SimpleNamespace(
                names=reordered_names,
                signals=env.action_schema.signals,
                ramps=env.action_schema.ramps,
                nonmerge_vsl_keys=env.action_schema.nonmerge_vsl_keys,
            ),
            net=env.net,
            cfg=env.cfg,
        )

        groups = _directed_groups(proxy, anchor, target)
        expected = reordered_names.index(f"urban.{signal}.g_green")

        self.assertEqual(groups["green"][expected], -1.0)
        self.assertEqual(int((groups["green"] != 0.0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
