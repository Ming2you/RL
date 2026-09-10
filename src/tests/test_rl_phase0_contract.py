from __future__ import annotations

import copy
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rl_leader.env import (
    PSTACK_ANCHOR_CONTEXT_CONTRACT,
    PStackAnchorContext,
    RLLeaderEnv,
)
from rl_leader.diagnose_frozen_event_attribution import action_from_trace_row
from rl_leader.diagnose_pcent_reachability import (
    _shortest_offset_delta,
    response_distance,
)
from rl_leader.diagnose_reachable_candidate_attribution import (
    _paired_metrics,
    residual_from_record,
)
from rl_leader.experiment_contract import (
    ExperimentContract,
    experiment_contract_fingerprint,
    experiment_contract_payload,
)
from rl_leader.export_pcent_teacher_replay import control_from_payload
from rl_leader.run_pcent_matched import (
    _compact_result,
    _step_validity,
    _teacher_contract,
)
from src.controllers.pstack_factory import CANONICAL_PSTACK_OPTIONS
from src.controllers.leader import LeaderAction
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, TrafficState


class _CandidateIsolationProbe(StackelbergWuMeteredController):
    def __init__(self):
        self.nash_solver = SimpleNamespace(counter=0)

    def _evaluate_full_candidate(
        self,
        index,
        action,
        state,
        forecast,
        previous,
        stage="coarse",
        incumbent_obj=float("inf"),
        rollout_abort_obj=float("inf"),
    ):
        del action, state, forecast, previous, incumbent_obj, rollout_abort_obj
        before = int(self.nash_solver.counter)
        self.nash_solver.counter += 1
        return SimpleNamespace(
            index=index,
            objective=float(before),
            metadata={},
            stage=stage,
        )


class RLPhaseZeroContractTest(unittest.TestCase):
    @staticmethod
    def _fake_anchor_context(env: RLLeaderEnv) -> PStackAnchorContext:
        follower_seed = copy.deepcopy(env.controller.nash_solver)
        optimizer_controller = SimpleNamespace(
            last_candidate_common_solver=follower_seed,
        )
        coordination = env.action_schema.decode(
            np.zeros(env.action_schema.dimension, dtype=np.float32),
            env.previous,
        )
        raw_action = env.action_schema.encode(coordination)
        envelope = env.action_schema.serialize_anchor(coordination)
        return PStackAnchorContext(
            contract_version=PSTACK_ANCHOR_CONTEXT_CONTRACT,
            experiment_contract_sha256=env.experiment_contract_fingerprint,
            state_fingerprint=env._anchor_context_state_fingerprint(),
            step_idx=env.step_idx,
            simulation_time_sec=env.sim.state.time_sec,
            result=SimpleNamespace(control=ControlAction.fixed(env.cfg)),
            forecast=tuple(env._optimizer_forecast()),
            inventory_before=env._inventory(),
            coordination=coordination,
            raw_action=raw_action,
            anchor_fingerprint=env.action_schema.anchor_fingerprint(
                envelope,
                coordination.selected_branch,
            ),
            optimizer_controller=optimizer_controller,
            optimizer_pfo_supervisor=None,
            follower_seed=follower_seed,
        )

    def test_anchor_context_preparation_discards_pending_trial(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        context = self._fake_anchor_context(env)

        def fake_decision(**_kwargs):
            env._pending_optimizer_trial = (
                context.optimizer_controller,
                context.optimizer_pfo_supervisor,
            )
            return (
                context.result,
                list(context.forecast),
                context.inventory_before,
                context.coordination,
                context.raw_action,
            )

        with patch.object(env, "_optimizer_decision", side_effect=fake_decision):
            prepared = env.prepare_pstack_anchor_context()

        self.assertIsNone(env._pending_optimizer_trial)
        self.assertEqual(prepared.state_fingerprint, context.state_fingerprint)
        self.assertEqual(prepared.anchor_fingerprint, context.anchor_fingerprint)

    def test_anchored_candidate_reuses_context_without_optimizer_query(self):
        env = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        context = self._fake_anchor_context(env)
        residual = np.zeros(env.action_dim, dtype=np.float32)
        residual[2] = 0.1
        fake_result = SimpleNamespace(
            control=ControlAction.fixed(env.cfg),
            metadata={},
        )

        with (
            patch.object(env, "_optimizer_decision", side_effect=AssertionError),
            patch(
                "src.controllers.rl_stackelberg.RLStackelbergController.decide_with_info",
                return_value=fake_result,
            ),
        ):
            env.step_anchored_candidate(residual, context)

        np.testing.assert_array_equal(env.last_policy_raw_action, residual)
        self.assertIsNone(env._pending_optimizer_trial)
        self.assertEqual(
            env.last_requested_coordination,
            env.action_schema.decode_anchored_residual(
                residual,
                context.coordination,
                env.mask,
            ),
        )

    def test_regular_residual_path_uses_native_common_follower_seed(self):
        env = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        context = self._fake_anchor_context(env)
        seen_seed_fingerprints = []

        def fake_optimizer_decision(**_kwargs):
            env._pending_optimizer_trial = (
                context.optimizer_controller,
                context.optimizer_pfo_supervisor,
            )
            return (
                context.result,
                list(context.forecast),
                context.inventory_before,
                context.coordination,
                context.raw_action,
            )

        def fake_rl_decision(controller, _state, _forecast, _previous):
            seen_seed_fingerprints.append(
                env._follower_runtime_fingerprint(controller._anchor_follower_seed)
            )
            return SimpleNamespace(
                control=ControlAction.fixed(env.cfg),
                metadata={},
            )

        pick_rl_metadata = {
            "leader_rl_pstack_anchor_pick_rl": 1.0,
            "leader_rl_pstack_anchor_pick_pstack": 0.0,
        }
        with (
            patch.object(env, "_optimizer_decision", side_effect=fake_optimizer_decision),
            patch(
                "src.controllers.rl_stackelberg.RLStackelbergController.decide_with_info",
                autospec=True,
                side_effect=fake_rl_decision,
            ),
            patch.object(
                env,
                "_pstack_anchor_select",
                side_effect=lambda control, *_args, **_kwargs: (
                    control, pick_rl_metadata
                ),
            ),
        ):
            env.step(np.zeros(env.action_dim, dtype=np.float32))

        self.assertEqual(len(seen_seed_fingerprints), 1)
        self.assertEqual(
            seen_seed_fingerprints[0],
            env._follower_runtime_fingerprint(context.follower_seed),
        )

    def test_anchor_context_rejects_wrong_step(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        context = replace(self._fake_anchor_context(env), step_idx=env.step_idx + 1)

        with self.assertRaisesRegex(ValueError, "step mismatch"):
            env._validate_anchor_context(context)

    def test_anchor_context_rejects_hidden_follower_runtime_change(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        context = self._fake_anchor_context(env)
        env.controller.nash_solver._lambda_P += 1.0

        with self.assertRaisesRegex(ValueError, "state fingerprint"):
            env._validate_anchor_context(context)

    def test_pcent_teacher_contract_is_stable_and_separate(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)

        first, first_sha = _teacher_contract(env)
        second, second_sha = _teacher_contract(env)

        self.assertEqual(first, second)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first["version"], "pcent_teacher_contract_v1")
        self.assertNotEqual(first_sha, env.experiment_contract_fingerprint)

    def test_pcent_teacher_control_payload_is_reconstructed_without_aliasing(self):
        payload = {
            "N_P_star": 100.0,
            "N_UF_star": 200.0,
            "green_times": {"A_p1": 50.0},
            "offsets": {"A": 10.0},
            "vsl": {"FW_E": 100.0},
            "ramp_metering": {"R_D_E": 900.0},
            "inflow_outflow_allocation": {"in_A": 300.0},
        }

        control = control_from_payload(payload)
        payload["green_times"]["A_p1"] = 1.0

        self.assertEqual(control.N_P_star, 100.0)
        self.assertEqual(control.green_times["A_p1"], 50.0)
        self.assertEqual(control.inflow_outflow_allocation["in_A"], 300.0)

    def test_reachable_candidate_residual_uses_action_schema_order(self):
        residual = residual_from_record(
            {"residual_nonzero": {"third": -0.5, "first": 0.25}},
            ("first", "second", "third"),
        )

        np.testing.assert_allclose(residual, [0.25, 0.0, -0.5])
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            residual_from_record(
                {"residual_nonzero": {"missing": 1.0}},
                ("first",),
            )

    def test_reachable_candidate_pairing_reports_ttt_and_inventory(self):
        candidate = {
            "steps": 3,
            "ttt": 9.0,
            "terminal_inventory": 90.0,
            "checkpoints": {
                "1": {"ttt": 4.0, "terminal_inventory": 99.0},
                "3": {"ttt": 9.0, "terminal_inventory": 90.0},
            },
        }
        pstack = {
            "steps": 3,
            "ttt": 10.0,
            "terminal_inventory": 100.0,
            "checkpoints": {
                "1": {"ttt": 4.5, "terminal_inventory": 100.0},
                "3": {"ttt": 10.0, "terminal_inventory": 100.0},
            },
        }

        paired = _paired_metrics(candidate, pstack)

        self.assertAlmostEqual(paired["ttt_gain"], 1.0)
        self.assertAlmostEqual(paired["terminal_inventory_delta"], -10.0)
        self.assertAlmostEqual(paired["horizons"]["1"]["ttt_gain"], 0.5)

    def test_pcent_reachability_distance_uses_physical_scales(self):
        distance = response_distance(
            np.asarray([2.0, 4.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([2.0, 4.0]),
            {"green": [0], "offset": [1]},
        )

        self.assertAlmostEqual(distance["overall_rmse"], 1.0)
        self.assertAlmostEqual(distance["green_rmse"], 1.0)
        self.assertAlmostEqual(distance["offset_rmse"], 1.0)
        self.assertAlmostEqual(_shortest_offset_delta(350.0, 10.0, 360.0), -20.0)

    def test_pcent_summary_recovers_profile_from_embedded_contract(self):
        row = {
            "experiment_contract_sha256": "abc",
            "experiment_contract": {"profile_id": "profile"},
            "scenario": "scenario",
            "policy_steps": 75,
            "completed": True,
            "controlled_ttt": 1.0,
            "terminal_inventory": 2.0,
            "validity_gate_pass": True,
            "elapsed_sec": 3.0,
            "trace_path": "trace.jsonl",
        }

        compact = _compact_result(row)

        self.assertEqual(compact["experiment_profile_id"], "profile")

    def test_frozen_trace_action_is_reconstructed_in_schema_order(self):
        row = {"raw_action": {"second": 2.0, "first": 1.0}}

        action = action_from_trace_row(row, ("first", "second"))

        np.testing.assert_array_equal(action, np.asarray([1.0, 2.0], np.float32))
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            action_from_trace_row(row, ("first", "missing"))

    def test_pcent_runner_uses_environment_conservation_validity_contract(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        demand = env._forecast()[0]
        diagnostics = {
            "urban_demand_arrivals_veh": 12.0,
            "onramp_arrivals_veh": 8.0,
            "boundary_out_sink_veh": 5.0,
            "mainline_exit_flow_total": 100.0,
            "movement_queue_projection_veh": 0.0,
            "coupling_offramp_arrivals_rejected_veh": 0.0,
            "ramp_queue_overflow_count": 0.0,
            "queue_overflow_count": 0.0,
        }
        mainline = sum(float(value) for value in demand.freeway_mainline.values())
        external = mainline * env.cfg.simulation.T_c_h + 20.0
        completed = 5.0 + 100.0 * env.cfg.simulation.T_c_h
        inventory_before = 1000.0
        inventory_after = inventory_before + external - completed

        self.assertTrue(_step_validity(
            env,
            demand,
            diagnostics,
            inventory_before,
            inventory_after,
        ))
        self.assertFalse(_step_validity(
            env,
            demand,
            diagnostics,
            inventory_before,
            inventory_after + 0.01,
        ))

    def test_native_candidates_start_from_one_common_follower_snapshot(self):
        controller = _CandidateIsolationProbe()
        candidates = [LeaderAction(1.0, 10.0), LeaderAction(2.0, 20.0)]

        evaluations = controller._evaluate_candidate_set(
            candidates,
            [0, 1],
            state=None,
            forecast=[],
            previous=None,
        )

        self.assertEqual([item.objective for item in evaluations], [0.0, 0.0])
        self.assertEqual(controller.nash_solver.counter, 0)
        self.assertEqual(
            [item.follower_solver_snapshot.counter for item in evaluations],
            [1, 1],
        )

    def test_metering_selection_commits_winning_trial_runtime_without_recompute(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        follower = env._ensure_optimizer_controller().nash_solver
        follower.cfg.mpc.baseline_move_box = False
        follower.ramp_metering_fractions = (1.0, 0.5, 0.25)
        link = next(
            item
            for item in follower.cfg.network.freeway_links
            if any(
                follower.cfg.network.ramp_to_freeway.get(ramp) == item
                for ramp in follower.cfg.network.ramps
            )
        )
        ramps = [
            ramp
            for ramp in follower.cfg.network.ramps
            if follower.cfg.network.ramp_to_freeway.get(ramp) == link
        ]
        capacities = {
            ramp: float(follower.cfg.network.ramp_capacity_veh_h[ramp])
            for ramp in ramps
        }
        selected_meter = {ramp: 0.5 * capacities[ramp] for ramp in ramps}
        calls = []

        def fake_local(_link, _state, _coupling, _demand, previous):
            meter = tuple(float(previous.ramp_metering[ramp]) for ramp in ramps)
            calls.append(meter)
            follower._wu._last_offramp_flow = {
                f"selected_meter_{ramp}": float(previous.ramp_metering[ramp])
                for ramp in ramps
            }
            follower._wu._has_last_offramp_flow = True
            objective = sum(
                abs(float(previous.ramp_metering[ramp]) - selected_meter[ramp])
                for ramp in ramps
            )
            return ({link: 100.0}, objective, 1)

        state = TrafficState.initial(follower.cfg)
        demand = DemandProfile(
            follower.cfg,
            ScenarioConfig("probe"),
        ).at(0.0)
        previous = ControlAction.uncontrolled(follower.cfg)
        with patch.object(follower, "_solve_freeway_agent_local", side_effect=fake_local):
            _, metering, evals = follower._solve_freeway_agent_metered(
                link,
                state,
                {},
                demand,
                previous,
                leader=None,
                previous=previous,
            )

        selected_tuple = tuple(selected_meter[ramp] for ramp in ramps)
        self.assertNotEqual(calls[-1], selected_tuple)
        self.assertEqual(len(calls), 1 + 2 * len(ramps))
        self.assertEqual(evals, len(calls))
        for ramp in ramps:
            self.assertEqual(metering[ramp], selected_meter[ramp])
            self.assertEqual(
                follower._wu._last_offramp_flow[f"selected_meter_{ramp}"],
                selected_meter[ramp],
            )

    def test_optimizer_branch_restores_rejected_rl_controller_and_full_follower(self):
        env = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        optimizer = env._ensure_optimizer_controller()
        before_rl = copy.deepcopy(env.controller)
        before_rl._regret_force_this_step = False
        env.controller._regret_force_this_step = True
        optimizer.nash_solver._phase_resolved_active_signals = {"optimizer"}
        env._pending_optimizer_trial = (optimizer, None)
        coordination = env.action_schema.decode(
            np.zeros(env.action_dim, dtype=np.float32),
            env.previous,
        )

        env._commit_optimizer_branch(before_rl, coordination)

        self.assertFalse(env.controller._regret_force_this_step)
        self.assertEqual(
            env.controller.nash_solver._phase_resolved_active_signals,
            {"optimizer"},
        )
        self.assertIs(env.provider, env.controller.coordination_provider)
        self.assertIsNot(env.controller.nash_solver, optimizer.nash_solver)
        self.assertIs(env.controller.cfg, env.cfg)
        self.assertIs(env.controller.leader.cfg, env.cfg)
        self.assertIs(env.controller.nash_solver.cfg, env.cfg)
        self.assertIs(env.controller.nash_solver._wu.cfg, env.cfg)

    def test_anchor_identity_overrides_an_improving_rl_proposal(self):
        env = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        rl_control = ControlAction.uncontrolled(env.cfg)
        pstack_control = ControlAction.fixed(env.cfg)
        with patch.object(
            env,
            "_fixed_control_metrics",
            side_effect=[
                {"ttt": 10.0, "terminal_inventory": 100.0},
                {"ttt": 20.0, "terminal_inventory": 100.0},
            ],
        ):
            selected, metadata = env._pstack_anchor_select(
                rl_control,
                pstack_control,
                [],
                force_pstack=True,
            )

        self.assertIs(selected, pstack_control)
        self.assertEqual(metadata["leader_rl_pstack_anchor_pick_pstack"], 1.0)
        self.assertEqual(metadata["leader_rl_pstack_anchor_identity_forced"], 1.0)
        self.assertEqual(metadata["leader_rl_pstack_anchor_unforced_pick_rl"], 1.0)

    def test_canonical_factory_ignores_ambient_pstack_switches(self):
        with patch.dict(
            os.environ,
            {"OPT12": "0", "OFFSET_PRICE": "0", "RAMP_OFFSET": "0"},
            clear=False,
        ):
            env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
            optimizer = env._ensure_optimizer_controller()

        self.assertTrue(optimizer.cfg.mpc.leader_skip_local_refinement)
        self.assertFalse(optimizer.cfg.mpc.leader_rollout_early_stop)
        self.assertTrue(optimizer.offset_price_enabled)
        self.assertTrue(optimizer.nash_solver.ramp_offset_enabled)
        self.assertEqual(
            env.experiment_contract_payload["pstack_options"],
            CANONICAL_PSTACK_OPTIONS.to_dict(),
        )

    def test_contract_fingerprint_covers_resolved_config_and_runtime(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        payload = experiment_contract_payload(env)

        self.assertEqual(
            experiment_contract_fingerprint(payload),
            experiment_contract_fingerprint(copy.deepcopy(payload)),
        )
        changed = copy.deepcopy(payload)
        changed["warmup"]["steps"] += 1
        self.assertNotEqual(
            experiment_contract_fingerprint(payload),
            experiment_contract_fingerprint(changed),
        )
        self.assertEqual(env.experiment_contract_fingerprint, experiment_contract_fingerprint(payload))

    def test_execution_arm_does_not_change_semantic_contract(self):
        direct = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=False,
            action_parameterization="absolute",
        )
        anchor = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=0,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )

        self.assertEqual(
            direct.experiment_contract_fingerprint,
            anchor.experiment_contract_fingerprint,
        )

    def test_contract_round_trip_is_fresh_and_rejects_tampering(self):
        env = RLLeaderEnv(scenario_name="medium_demand", warmup_nc_steps=0)
        contract = env.experiment_contract
        first_cfg, first_scenario, first_options = contract.materialize()
        second_cfg, second_scenario, second_options = contract.materialize()

        self.assertIsNot(first_cfg, second_cfg)
        self.assertIsNot(first_scenario, second_scenario)
        self.assertEqual(first_options, second_options)
        self.assertTrue(first_cfg.network.terminal_zero_gradient)
        self.assertIn(
            "leader_skip_local_refinement",
            contract.payload["resolved_config"]["mpc"],
        )

        tampered = contract.payload
        tampered["warmup"]["steps"] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            ExperimentContract.from_artifact(tampered, contract.sha256)

    def test_environment_materializes_persisted_contract(self):
        source = RLLeaderEnv(
            scenario_name="medium_demand",
            warmup_nc_steps=2,
            pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        replay = RLLeaderEnv(
            pstack_anchor=True,
            action_parameterization="pstack_residual",
            experiment_contract=source.experiment_contract,
        )

        self.assertEqual(
            replay.experiment_contract_fingerprint,
            source.experiment_contract_fingerprint,
        )
        self.assertEqual(replay.warmup, 2)


if __name__ == "__main__":
    unittest.main()
