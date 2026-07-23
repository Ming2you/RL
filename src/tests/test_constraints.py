import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

# 이 런타임에는 scipy가 없을 수 있다 — SLSQP 경로 테스트는 존재 시에만 실행.
_HAS_SCIPY = importlib.util.find_spec("scipy") is not None

from src.controllers.centralized_mpc import CentralizedMPC
from src.controllers.distributed_coordinator import AgentSolve, DistributedCoordinator, build_agent_specs
from src.controllers.freeway_follower import FreewayFollower, FreewayFollowerResult
from src.controllers.leader import Leader, LeaderAction
from src.controllers.inflow_outflow_allocation import InflowOutflowAllocationModule
from src.controllers.nash_solver import NashResult
from src.controllers.relaxed_quantization import repair_green_pair, repair_vsl_value
from src.controllers.simplified_inflow_outflow_allocation import SimplifiedInflowOutflowAllocationModule
from src.controllers.spillback_constraints import (
    assess_offramp_spillback,
    assess_onramp_spillback,
    offramp_combined_capacity_veh,
    onramp_combined_capacity_veh,
)
from src.controllers.stackelberg_mpc import StackelbergMPCController, _LeaderCandidateEvaluation
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.structured_grid import sensitivity_probe_candidates, structured_grid_candidates
from src.controllers.urban_follower import UrbanFollower
from src.controllers.wu_distributed import WuDistributedController, _wu_fixed_control
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.evaluation.metrics import validate_controls
from src.models.demand import DemandProfile, DemandStep, ScenarioConfig
from src.models.metanet import effective_lane_profile
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    movement_storage_capacity,
    sync_onramp_queues_from_freeway,
    sync_onramp_queues_to_freeway,
    urban_substep,
)
from src.simulation.coupling import CoupledStepResult, run_coupled_interval
from src.simulation.simulator import MixedTrafficSimulator


def short_config():
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {"simulation": {"T_total": 360.0}, "mpc": {"leader_candidate_count": 5, "max_nash_iter": 3}},
    )


class ConstraintTests(unittest.TestCase):
    def test_relaxed_quantized_controls_are_off_by_default(self):
        cfg = short_config()
        self.assertFalse(cfg.mpc.relaxed_quantized_controls)
        self.assertEqual(cfg.mpc.relaxed_rounding_mode, "floor")
        self.assertIsInstance(UrbanFollower(cfg).allocation_module, InflowOutflowAllocationModule)

    @unittest.skipUnless(_HAS_SCIPY, "scipy 미설치 — SLSQP 경로는 graceful degrade(available=0)")
    def test_centralized_slsqp_solver_path_runs_and_logs(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {
                    "horizon_steps": 1,
                    "centralized_solver_mode": "slsqp",
                    "optimizer_maxiter": 1,
                    "optimizer_n_starts": 1,
                },
            },
        )
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        info = CentralizedMPC(cfg, mode="wu").decide_with_info(state, [demand])

        self.assertGreater(info.solver_evaluations, 0)
        self.assertEqual(info.control.diagnostics["centralized_solver_mode_slsqp"], 1.0)
        self.assertEqual(info.control.diagnostics["centralized_slsqp_available"], 1.0)
        self.assertIn("centralized_slsqp_variable_count", info.control.diagnostics)

    def test_centralized_solver_mode_validation(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file(
                "src/config/default.yaml",
                {"mpc": {"centralized_solver_mode": "not_a_solver"}},
            )

    def test_allocation_batch_objective_matches_scalar_objective(self):
        cfg = short_config()
        module = InflowOutflowAllocationModule(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        leader = LeaderAction(cfg.leader.N_P_crit_veh, 5000.0)
        specs = {
            movement: spec
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") in {"boundary_in", "off_ramp", "boundary_out", "on_ramp"}
        }
        movements = list(specs)
        kinds = [str(specs[movement].get("kind", "")) for movement in movements]
        lower, upper = module._bounds(movements, specs, leader)
        target = module._clip_target(
            module.solve(state, leader, demand).target_net_inflow_veh_h,
            lower,
            upper,
            kinds,
        )
        rows = lower + 0.37 * (upper - lower)
        particles = rows[None, :] + np.asarray([0.0, 0.1, -0.1])[:, None] * (upper - lower)[None, :]
        particles = particles.clip(lower, upper)

        context = module._objective_context(state, movements, kinds)
        batch = module._objective_many(context, kinds, particles, target)
        scalar = [
            module._objective(state, movements, kinds, row, target)
            for row in particles
        ]

        for got, expected in zip(batch, scalar):
            # batch/scalar 등가 주장 — 연산 순서 차이의 1e-12 부동소수 드리프트는 허용.
            self.assertAlmostEqual(float(got), float(expected), places=9)

    def test_simplified_allocation_uses_np_star_as_net_inflow_vehicles(self):
        cfg = short_config().with_updates({
            "mpc": {"stackelberg_allocation_mode": "simplified", "horizon_steps": 3},
        })
        module = SimplifiedInflowOutflowAllocationModule(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        leader = LeaderAction(300.0, cfg.network.total_ramp_capacity)

        plan = module.solve(state, leader, forecast)
        horizon_h = cfg.simulation.T_c_h * cfg.mpc.horizon_steps

        self.assertAlmostEqual(plan.target_net_inflow_veh_h * horizon_h, 300.0)
        self.assertLessEqual(plan.residual_veh_h * horizon_h, cfg.urban_follower.eps_U + 1.0e-9)
        self.assertEqual(plan.diagnostics["allocation_simplified_module_active"], 1.0)

    def test_stackelberg_simplified_allocation_projection_keeps_allocation_map(self):
        cfg = short_config().with_updates({
            "mpc": {"stackelberg_allocation_mode": "simplified", "horizon_steps": 3},
        })
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        leader = LeaderAction(300.0, cfg.network.total_ramp_capacity)

        plan = coordinator._stackelberg_allocation_plan(state, leader, forecast)
        self.assertIsNotNone(plan)
        control = coordinator._project_control_to_leader_constraints(
            ControlAction.fixed(cfg),
            leader,
            plan,
        )
        diagnostics = coordinator._leader_direct_feasible_set_diagnostics(
            state,
            control,
            forecast,
            leader,
        )

        self.assertTrue(control.inflow_outflow_allocation)
        self.assertEqual(control.diagnostics["stackelberg_allocation_mode_simplified"], 1.0)
        self.assertEqual(diagnostics["distributed_grid_leader_allocation_module_active"], 1.0)
        self.assertEqual(diagnostics["distributed_grid_leader_allocation_module_disabled"], 0.0)

    def test_stackelberg_pso_allocation_uses_original_module(self):
        cfg = short_config().with_updates({
            "mpc": {"stackelberg_allocation_mode": "pso", "horizon_steps": 2},
            "urban_follower": {"allocation_pso_particles": 4, "allocation_pso_iterations": 3},
        })
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        leader = LeaderAction(300.0, cfg.network.total_ramp_capacity)

        plan = coordinator._stackelberg_allocation_plan(state, leader, forecast)
        self.assertIsNotNone(plan)
        self.assertIsInstance(coordinator.allocation_module, InflowOutflowAllocationModule)
        self.assertEqual(plan.diagnostics["allocation_module_active"], 1.0)
        self.assertNotIn("allocation_simplified_module_active", plan.diagnostics)

        control = coordinator._project_control_to_leader_constraints(
            ControlAction.fixed(cfg),
            leader,
            plan,
        )
        self.assertTrue(control.inflow_outflow_allocation)
        self.assertEqual(control.diagnostics["stackelberg_allocation_mode_pso"], 1.0)

    def test_relaxed_green_repair_satisfies_cycle_and_bounds(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        repaired = repair_green_pair(cfg.network.green_max + 37.0, cfg)
        self.assertAlmostEqual(
            repaired.p1 + repaired.p2 + cfg.network.lost_time,
            cfg.network.cycle_length,
        )
        self.assertGreaterEqual(repaired.p1, cfg.network.green_min)
        self.assertLessEqual(repaired.p1, cfg.network.green_max)
        self.assertGreaterEqual(repaired.p2, cfg.network.green_min)
        self.assertLessEqual(repaired.p2, cfg.network.green_max)

    def test_relaxed_vsl_repair_is_discrete_and_step_limited(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        previous = 100.0
        repaired = repair_vsl_value(53.0, previous, cfg)
        self.assertIn(repaired.value, {float(v) for v in cfg.freeway_follower.vsl_set})
        self.assertLessEqual(abs(repaired.value - previous), cfg.freeway_follower.max_vsl_step + 1e-9)

    def test_relaxed_urban_follower_green_is_feasible(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        state = TrafficState.initial(cfg)
        for movement, spec in cfg.network.urban_movements.items():
            if spec.get("phase") == "A_p1":
                state.urban_movement_queue[movement] = 250.0
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        result = UrbanFollower(cfg).solve(state, None, demand)
        self.assertGreater(result.green_times["A_p1"], cfg.network.effective_green_total / 2.0)
        for signal in cfg.network.signals:
            p1 = result.green_times[f"{signal}_p1"]
            p2 = result.green_times[f"{signal}_p2"]
            self.assertAlmostEqual(p1 + p2, cfg.network.effective_green_total)
            self.assertGreaterEqual(p1, cfg.network.green_min)
            self.assertGreaterEqual(p2, cfg.network.green_min)

    def test_relaxed_wu_freeway_evaluates_fewer_vsl_candidates(self):
        full_cfg = short_config()
        relaxed_cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        state = TrafficState.initial(full_cfg)
        demand = DemandProfile(full_cfg, ScenarioConfig("test", freeway_scale=1.3)).at(0.0)
        previous = _wu_fixed_control(full_cfg)
        full_ctl = WuDistributedController(full_cfg)
        relaxed_ctl = WuDistributedController(relaxed_cfg)
        link = full_cfg.network.freeway_links[0]
        full_count = len(full_ctl._freeway_segment_candidates(
            link,
            full_cfg.network.freeway_segments_per_link,
            previous,
        ))
        coupling = full_ctl._coupling(state, previous, demand)
        _, _, relaxed_evals = relaxed_ctl._solve_freeway_agent(
            link,
            state,
            coupling,
            demand,
            _wu_fixed_control(relaxed_cfg),
            None,
        )
        self.assertLess(relaxed_evals, full_count)

    def test_freeway_ramp_candidates_include_default_and_previous_guard(self):
        cfg = short_config().with_updates({
            "freeway_follower": {"horizon_ramp_candidate_limit": 1},
        })
        state = TrafficState.initial(cfg)
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 100.0
        previous = ControlAction.uncontrolled(cfg)
        previous.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        follower = FreewayFollower(cfg)

        candidates, _ = follower._ramp_candidates(
            state,
            LeaderAction(0.0, cfg.network.total_ramp_capacity),
            [demand],
            previous,
        )

        totals = sorted(round(sum(candidate.values()), 6) for candidate in candidates)
        self.assertIn(0.0, totals)
        self.assertIn(round(cfg.network.total_ramp_capacity, 6), totals)

    def test_relaxed_urban_stage2_evaluates_default_guard_and_neighborhood(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        state = TrafficState.initial(cfg)
        previous = ControlAction.uncontrolled(cfg)
        previous.green_times["A_p1"] = cfg.network.green_min
        previous.green_times["A_p2"] = cfg.network.effective_green_total - cfg.network.green_min
        previous.offsets["A"] = cfg.urban_follower.max_offset_step
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)

        result = UrbanFollower(cfg).solve(state, None, demand, previous_control=previous)

        self.assertEqual(result.metrics["urban_stage2_default_guard_evaluated"], 1.0)
        self.assertGreater(result.metrics["urban_stage2_candidate_evaluations"], len(cfg.network.signals))
        for signal in cfg.network.signals:
            p1 = result.green_times[f"{signal}_p1"]
            p2 = result.green_times[f"{signal}_p2"]
            self.assertAlmostEqual(p1 + p2, cfg.network.effective_green_total)
            self.assertGreaterEqual(result.offsets[signal], 0.0)
            self.assertLess(result.offsets[signal], cfg.network.cycle_length)

    def test_distributed_freeway_agent_jointly_evaluates_metering_and_vsl_guards(self):
        cfg = short_config().with_updates({
            "mpc": {"relaxed_quantized_controls": True, "max_nash_iter": 1},
        })
        state = TrafficState.initial(cfg)
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 60.0
        demand = DemandProfile(cfg, ScenarioConfig("test", ramp_scale=1.2)).at(0.0)
        controller = DistributedCoordinator(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)
        previous = ControlAction.uncontrolled(cfg)
        coupling = {f"u_on_{ramp}": 900.0 for ramp in cfg.network.ramps}

        solve = controller._solve_freeway_agent(agent, state, None, [demand], previous, coupling)

        self.assertEqual(solve.diagnostics[f"agent_{agent.id}_default_metering_guard_evaluated"], 1.0)
        self.assertGreaterEqual(solve.diagnostics[f"agent_{agent.id}_metering_candidates"], 2.0)
        self.assertGreaterEqual(solve.diagnostics[f"agent_{agent.id}_vsl_candidates"], 2.0)
        expected = (
            solve.diagnostics[f"agent_{agent.id}_metering_candidates"]
            * solve.diagnostics[f"agent_{agent.id}_vsl_candidates"]
        )
        self.assertEqual(solve.diagnostics[f"agent_{agent.id}_joint_candidate_evaluations"], expected)

    def test_distributed_freeway_urban_queue_term_counts_only_spillback(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        controller = DistributedCoordinator(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)
        for ramp in agent.ramps:
            for movement in cfg.network.on_ramp_to_movement.get(ramp, []):
                state.urban_movement_queue[movement] = 100.0

        upper_release = {ramp: cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps}
        _, no_spillback_tts = controller._agent_queue_tts_terms(
            agent,
            state,
            upper_release,
            {f"u_on_{ramp}": 0.0 for ramp in agent.ramps},
            cfg.simulation.T_c_h,
        )

        self.assertEqual(no_spillback_tts, 0.0)

    def test_distributed_freeway_candidates_include_ratio_one_guards(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        controller = DistributedCoordinator(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)
        upper = {ramp: 0.5 * cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps}
        weights = {ramp: 1.0 for ramp in agent.ramps}
        current = ControlAction.uncontrolled(cfg)

        metering_candidates = controller._metering_candidates(agent, upper, weights, 0.5 * sum(upper.values()), current)
        vsl_candidates = controller._vsl_candidates(max(cfg.freeway_follower.vsl_set), min(cfg.freeway_follower.vsl_set))

        self.assertIn(round(sum(cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps), 6), {
            round(sum(candidate.values()), 6) for candidate in metering_candidates
        })
        self.assertIn(float(max(cfg.freeway_follower.vsl_set)), vsl_candidates)

    def test_distributed_freeway_candidates_include_intermediate_upper_fractions(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        controller = DistributedCoordinator(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)
        upper = {ramp: cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps}
        current = ControlAction.uncontrolled(cfg)

        metering_candidates = controller._metering_candidates(
            agent,
            upper,
            {ramp: 1.0 for ramp in agent.ramps},
            0.5 * sum(upper.values()),
            current,
        )
        totals = {round(sum(candidate.values()), 6) for candidate in metering_candidates}

        self.assertIn(round(0.8 * sum(upper.values()), 6), totals)
        self.assertIn(round(0.9 * sum(upper.values()), 6), totals)

    def test_distributed_freeway_projection_prices_metering_density_relief(self):
        # horizon 1-step: 이 proxy(고정속도 Euler, dt=T_c)는 2-step부터 과보정 진동으로
        # rho가 0/rho_max에 클리핑돼 서열이 무대 기하에 따라 뒤집힌다(망 변경으로 agent당
        # ramp 1개가 되며 표면화). relief 방향성(release↑→merge 밀도·TTS↑)은 1-step이 충분.
        cfg = short_config().with_updates({"mpc": {"horizon_steps": 1}})
        controller = DistributedCoordinator(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)
        state = TrafficState.initial(cfg)
        idx = agent.segment_index
        state.freeway_density[agent.link][idx] = cfg.network.rho_crit - 2.0
        state.freeway_speed[agent.link][idx] = 60.0
        state.freeway_flow[agent.link][idx - 1] = 2500.0
        lane_profile, _ = effective_lane_profile(state, cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        upper = {ramp: cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps}
        no_metering = {ramp: cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps}
        intermediate = {ramp: 0.8 * upper[ramp] for ramp in agent.ramps}

        no_meter_terms = controller._candidate_freeway_tts_terms(
            agent,
            state,
            no_metering,
            upper,
            demand,
            lane_profile,
        )
        intermediate_terms = controller._candidate_freeway_tts_terms(
            agent,
            state,
            intermediate,
            upper,
            demand,
            lane_profile,
        )

        self.assertGreater(no_meter_terms[0], intermediate_terms[0])
        self.assertGreater(no_meter_terms[3], intermediate_terms[3])

    def test_distributed_freeway_candidates_include_spillback_min_release_boundary(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        controller = DistributedCoordinator(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)
        state = TrafficState.initial(cfg)
        ramp = agent.ramps[0]
        for movement in cfg.network.on_ramp_to_movement[ramp]:
            state.urban_movement_queue[movement] = onramp_combined_capacity_veh(cfg, ramp)
        ramp_arrivals = {candidate_ramp: 0.0 for candidate_ramp in agent.ramps}
        ramp_arrivals[ramp] = 90.0
        min_release = controller._onramp_spillback_min_release_rates(
            state,
            agent,
            ramp_arrivals,
            cfg.simulation.T_c_h,
        )
        upper = {candidate_ramp: cfg.network.ramp_capacity_veh_h[candidate_ramp] for candidate_ramp in agent.ramps}
        candidates = controller._metering_candidates(
            agent,
            upper,
            {candidate_ramp: 1.0 for candidate_ramp in agent.ramps},
            target=0.0,
            current=ControlAction.uncontrolled(cfg),
            spillback_min_release=min_release,
        )

        keys = {
            tuple(round(candidate[candidate_ramp], 6) for candidate_ramp in agent.ramps)
            for candidate in candidates
        }
        expected = tuple(
            round(
                max(
                    min_release[candidate_ramp],
                    cfg.freeway_follower.ramp_metering_rate_min * cfg.network.ramp_capacity_veh_h[candidate_ramp],
                ),
                6,
            )
            for candidate_ramp in agent.ramps
        )
        self.assertIn(expected, keys)
        self.assertGreater(min_release[ramp], 0.0)

    def test_spillback_assessment_uses_combined_ramp_and_intersection_capacity(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)

        ramp = cfg.network.ramps[0]
        onramp_capacity = onramp_combined_capacity_veh(cfg, ramp)
        state.ramp_queue[ramp] = cfg.network.ramp_queue_max_veh
        for movement in cfg.network.on_ramp_to_movement[ramp]:
            state.urban_movement_queue[movement] = onramp_capacity
        no_release = assess_onramp_spillback(
            state,
            cfg,
            ramp,
            ramp_arrival_veh=120.0,
            metering_release_veh=0.0,
        )
        released = assess_onramp_spillback(
            state,
            cfg,
            ramp,
            ramp_arrival_veh=120.0,
            metering_release_veh=120.0,
        )

        self.assertGreater(no_release.violation_veh, 0.0)
        self.assertGreater(no_release.violation_veh, released.violation_veh)

        off_ramp = cfg.network.off_ramps[0]
        offramp_capacity = offramp_combined_capacity_veh(cfg, off_ramp)
        storage_link = cfg.network.off_ramp_storage_link[off_ramp]
        state.urban_link_storage[storage_link] = 0.0
        for movement in cfg.network.off_ramp_to_movement[off_ramp]:
            state.urban_movement_queue[movement] = offramp_capacity
        no_service = assess_offramp_spillback(
            state,
            cfg,
            off_ramp,
            offramp_inflow_veh=80.0,
            service_veh=0.0,
        )
        serviced = assess_offramp_spillback(
            state,
            cfg,
            off_ramp,
            offramp_inflow_veh=80.0,
            service_veh=80.0,
        )

        self.assertGreater(no_service.violation_veh, 0.0)
        self.assertGreater(no_service.violation_veh, serviced.violation_veh)

    def test_distributed_freeway_agent_reports_spillback_constraint_diagnostics(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True, "max_nash_iter": 1}})
        controller = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        agent = next(agent for agent in controller.freeway_agents if agent.ramps)

        for ramp in agent.ramps:
            state.ramp_queue[ramp] = cfg.network.ramp_queue_max_veh
            cap = onramp_combined_capacity_veh(cfg, ramp)
            for movement in cfg.network.on_ramp_to_movement[ramp]:
                state.urban_movement_queue[movement] = cap
        for off_ramp in cfg.network.off_ramps:
            if cfg.network.off_ramp_from_freeway.get(off_ramp) != agent.link:
                continue
            storage_link = cfg.network.off_ramp_storage_link[off_ramp]
            state.urban_link_storage[storage_link] = 0.0
            cap = offramp_combined_capacity_veh(cfg, off_ramp)
            for movement in cfg.network.off_ramp_to_movement[off_ramp]:
                state.urban_movement_queue[movement] = cap

        demand = DemandProfile(cfg, ScenarioConfig("test", freeway_scale=1.3, ramp_scale=1.5)).at(0.0)
        solve = controller._solve_freeway_agent(
            agent,
            state,
            None,
            [demand],
            ControlAction.uncontrolled(cfg),
            {f"u_on_{ramp}": 0.0 for ramp in cfg.network.ramps},
        )

        prefix = f"agent_{agent.id}"
        self.assertIn(f"{prefix}_spillback_feasible_evaluations", solve.diagnostics)
        self.assertIn(f"{prefix}_spillback_constraint_feasible", solve.diagnostics)
        self.assertIn(f"{prefix}_onramp_spillback_violation_veh", solve.diagnostics)
        self.assertIn(f"{prefix}_offramp_spillback_violation_veh", solve.diagnostics)
        self.assertIn("spillback_constraint_feasible", solve.infeasibility)
        self.assertLessEqual(
            solve.diagnostics[f"{prefix}_spillback_feasible_evaluations"],
            solve.diagnostics[f"{prefix}_joint_candidate_evaluations"],
        )

    def test_urban_freeway_pressure_ignores_metering_tracking_residual(self):
        cfg = short_config()
        response = FreewayFollowerResult(
            ramp_metering={},
            vsl={},
            objective_value=0.0,
            infeasibility={
                "metering_tracking_residual": cfg.network.freeway_capacity_veh_h,
                "ramp_projection_first_step_capacity": cfg.network.freeway_capacity_veh_h,
            },
        )

        pressure = UrbanFollower(cfg)._freeway_pressure(response)

        self.assertGreater(pressure["metering_pressure"], 0.0)
        self.assertEqual(pressure["total_pressure"], 0.0)

    def test_distributed_freeway_response_deduplicates_offramp_forecasts(self):
        cfg = short_config()
        controller = DistributedCoordinator(cfg)
        solves = [
            AgentSolve(
                agent_id=f"F_E{i}",
                objective=0.0,
                infeasibility={
                    "offramp_predicted_arrival_OR_D_E_veh": 12.5,
                    "offramp_predicted_flow_OR_D_E": 83.0,
                },
            )
            for i in range(4)
        ]

        response = controller._freeway_response(solves)

        self.assertEqual(response.infeasibility["offramp_predicted_arrival_OR_D_E_veh"], 12.5)
        self.assertEqual(response.infeasibility["offramp_predicted_flow_OR_D_E"], 83.0)

    def test_distributed_urban_arrival_coupling_is_queue_limited(self):
        cfg = short_config()
        controller = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        control = ControlAction.uncontrolled(cfg)
        zero_demand = DemandProfile(
            cfg,
            ScenarioConfig("zero", urban_scale=0.0, freeway_scale=0.0, ramp_scale=0.0),
        ).at(0.0)

        coupling = controller._extract_coupling(state, control, zero_demand)

        self.assertTrue(all(
            abs(value) <= 1.0e-9
            for key, value in coupling.items()
            if key.startswith("arr_")
        ))

    def test_relaxed_wu_urban_green_evaluates_neighborhood_candidates(self):
        cfg = short_config().with_updates({"mpc": {"relaxed_quantized_controls": True}})
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        controller = WuDistributedController(cfg)
        previous = _wu_fixed_control(cfg)
        coupling = controller._coupling(state, previous, forecast[0])

        p1, _, evals = controller._solve_urban_agent("A", state, coupling, previous, None)

        self.assertGreater(evals, 1)
        self.assertGreaterEqual(p1, cfg.network.green_min)
        self.assertLessEqual(p1, cfg.network.green_max)

    def test_wu_urban_arrival_coupling_is_queue_limited(self):
        cfg = short_config()
        controller = WuDistributedController(cfg)
        state = TrafficState.initial(cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        for link in state.freeway_flow:
            state.freeway_flow[link] = [0.0 for _ in state.freeway_flow[link]]
        control = _wu_fixed_control(cfg)
        zero_demand = DemandProfile(
            cfg,
            ScenarioConfig("zero", urban_scale=0.0, freeway_scale=0.0, ramp_scale=0.0),
        ).at(0.0)

        coupling = controller._coupling(state, control, zero_demand)

        self.assertTrue(all(
            abs(value) <= 1.0e-9
            for key, value in coupling.items()
            if key.startswith("arr_")
        ))

    def test_wu_upstream_map_uses_all_downstream_phase_movements(self):
        cfg = short_config()
        controller = WuDistributedController(cfg)
        origin = "A_to_D"
        phase = "D_p1"
        expected_beta = sum(
            float(spec.get("beta", 0.0))
            for spec in cfg.network.urban_movements.values()
            if spec.get("phase") == phase and spec.get("origin") == origin
        )
        actual_betas = [
            beta
            for _up_signal, up_movement, beta in controller._upstream_leaving_map[phase]
            if cfg.network.urban_movements[up_movement].get("destination") == origin
        ]

        self.assertGreater(expected_beta, 0.0)
        self.assertGreater(len(actual_betas), 0)
        for beta in actual_betas:
            self.assertAlmostEqual(beta, expected_beta)

    def test_vsl_values_are_discrete(self):
        cfg = short_config()
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        control = StackelbergMPCController(cfg).decide(TrafficState.initial(cfg), demand)
        self.assertTrue(all(v in cfg.freeway_follower.vsl_set for v in control.vsl.values()))

    def test_distributed_agent_partition_matches_topology(self):
        cfg = short_config()
        urban_agents, freeway_agents = build_agent_specs(cfg)
        n = cfg.network.freeway_segments_per_link
        m_d = cfg.network.ramp_merge_segment_index["R_D_W"]
        m_f = cfg.network.ramp_merge_segment_index["R_F_W"]
        o_d = cfg.network.off_ramp_segment_index["OR_D_W"]
        self.assertEqual(len(urban_agents), 5)
        self.assertEqual(len(freeway_agents), 2 * n)
        self.assertEqual({agent.id for agent in urban_agents}, {"U_A", "U_B", "U_C", "U_D", "U_F"})
        self.assertEqual(
            {agent.id for agent in freeway_agents},
            {f"F_{side}{i}" for side in ("W", "E") for i in range(n)},
        )
        d_agent = next(agent for agent in urban_agents if agent.id == "U_D")
        self.assertIn("D_N_to_onW", d_agent.movements)
        self.assertIn("D_offW_to_N", d_agent.movements)
        # segment-player 매핑(기하 무관, cfg 유도): merge seg가 해당 on-ramp 소유 —
        # ramp 소유 agent가 link당 2개로 분리(8-seg 디폴트: F_W4=R_D, F_W6=R_F).
        fw_d_agent = next(agent for agent in freeway_agents if agent.id == f"F_W{m_d}")
        self.assertEqual(tuple(fw_d_agent.ramps), ("R_D_W",))
        fw_f_agent = next(agent for agent in freeway_agents if agent.id == f"F_W{m_f}")
        self.assertEqual(tuple(fw_f_agent.ramps), ("R_F_W",))
        freeway_ids = {agent.id for agent in freeway_agents}
        for agent in urban_agents:
            self.assertTrue(set(agent.neighbors).issubset(freeway_ids))
        self.assertEqual(
            set(d_agent.neighbors),
            {f"F_W{o_d}", f"F_W{m_d}", f"F_E{o_d}", f"F_E{m_d}"},
        )

    def test_uncontrolled_E_vehicles_are_counted_in_ttt_coverage(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        net = cfg.network
        base_total = state.total_urban_vehicles(net)
        e_movement = next(
            movement
            for movement, spec in net.urban_movements.items()
            if spec.get("intersection") == "E"
        )
        state.urban_movement_queue[e_movement] = 7.0
        state.urban_link_storage["B_to_E"] = net.urban_link_storage_veh["B_to_E"] - 11.0
        state.urban_link_storage["E_to_F"] = net.urban_link_storage_veh["E_to_F"] - 5.0

        self.assertAlmostEqual(state.uncontrolled_node_movement_queue_veh(net), 7.0)
        self.assertAlmostEqual(state.uncontrolled_node_storage_occupancy_veh(net), 16.0)
        self.assertAlmostEqual(state.uncontrolled_node_vehicles(net), 23.0)
        self.assertAlmostEqual(state.total_urban_vehicles(net) - base_total, 23.0)

        plant_state = TrafficState.initial(cfg)
        plant_state.urban_link_storage["B_to_E"] = net.urban_link_storage_veh["B_to_E"] - 13.0
        plant_state.urban_link_storage["E_to_D"] = net.urban_link_storage_veh["E_to_D"] - 2.0
        demand = DemandStep(
            freeway_mainline={link: 0.0 for link in net.freeway_links},
            urban_boundary={link: 0.0 for link in net.boundary_in_links},
            ramp_arrival={ramp: 0.0 for ramp in net.ramps},
        )
        urban_ttt, diagnostics = urban_substep(
            plant_state,
            ControlAction.uncontrolled(cfg),
            demand,
            cfg,
            urban_step_index=0,
        )
        self.assertGreaterEqual(diagnostics["urban_uncontrolled_node_storage_occupancy_veh"], 15.0)
        self.assertGreaterEqual(diagnostics["urban_uncontrolled_node_vehicles_veh"], 15.0)
        self.assertAlmostEqual(
            diagnostics["urban_uncontrolled_node_ttt"],
            diagnostics["urban_uncontrolled_node_vehicles_veh"] * cfg.simulation.T_u_h,
        )
        self.assertGreaterEqual(urban_ttt, diagnostics["urban_uncontrolled_node_ttt"])

    def test_urban_follower_objective_covers_uncontrolled_E_vehicles(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        net = cfg.network
        e_movement = next(
            movement
            for movement, spec in net.urban_movements.items()
            if spec.get("intersection") == "E"
        )
        state.urban_movement_queue[e_movement] = 9.0
        state.urban_link_storage["D_to_E"] = net.urban_link_storage_veh["D_to_E"] - 4.0
        state.urban_link_storage["E_to_B"] = net.urban_link_storage_veh["E_to_B"] - 6.0
        demand = DemandStep(
            freeway_mainline={link: 0.0 for link in net.freeway_links},
            urban_boundary={link: 0.0 for link in net.boundary_in_links},
            ramp_arrival={ramp: 0.0 for ramp in net.ramps},
        )

        result = UrbanFollower(cfg).solve(
            state,
            None,
            demand,
            previous_control=ControlAction.uncontrolled(cfg),
        )
        expected_tts = 19.0 * cfg.simulation.T_c_h * cfg.mpc.horizon_steps

        self.assertEqual(result.metrics["urban_uncontrolled_node_objective_covered"], 1.0)
        self.assertAlmostEqual(result.metrics["urban_uncontrolled_node_vehicles_veh"], 19.0)
        self.assertAlmostEqual(result.metrics["urban_uncontrolled_node_objective_tts"], expected_tts)
        self.assertGreaterEqual(result.objective_value, expected_tts)

    def test_freeway_follower_vsl_candidates_are_segment_level(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "freeway_follower": {"horizon_vsl_candidate_limit_per_link": 4},
            },
        )
        state = TrafficState.initial(cfg)
        previous = ControlAction.fixed(cfg)

        candidates = FreewayFollower(cfg)._vsl_candidates(state, previous)

        for link in cfg.network.freeway_links:
            self.assertTrue(candidates[link])
            for candidate in candidates[link]:
                for i in range(cfg.network.freeway_segments_per_link):
                    self.assertIn(f"{link}__seg{i}", candidate)
            self.assertTrue(any(
                len({candidate[f"{link}__seg{i}"] for i in range(cfg.network.freeway_segments_per_link)}) > 1
                for candidate in candidates[link]
            ))

    def test_distributed_coordinator_returns_per_agent_diagnostics(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {
                    "follower_solver_mode": "distributed",
                    "max_nash_iter": 2,
                    "leader_candidate_count": 2,
                },
            },
        )
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        result = DistributedCoordinator(cfg).solve(
            state,
            LeaderAction(cfg.leader.N_P_crit_veh, 1200.0),
            demand,
            ControlAction.fixed(cfg),
        )
        self.assertEqual(result.control.diagnostics["distributed_player_active"], 1.0)
        self.assertEqual(result.control.diagnostics["distributed_urban_agent_count"], 5.0)
        self.assertEqual(
            result.control.diagnostics["distributed_freeway_agent_count"],
            2.0 * cfg.network.freeway_segments_per_link,
        )
        self.assertIn("agent_U_A_objective", result.control.diagnostics)
        self.assertIn("agent_F_W2_objective", result.control.diagnostics)
        self.assertIn("distributed_response_objective_tts", result.control.diagnostics)
        self.assertIn("urban_uncontrolled_node_vehicles_veh", result.control.diagnostics)
        self.assertIn("distributed_response_uncontrolled_node_urban_vehicles", result.control.diagnostics)
        self.assertEqual(result.control.diagnostics["distributed_grid_search_active"], 1.0)
        self.assertEqual(result.control.diagnostics["distributed_grid_parallel_stages"], 4.0)
        self.assertEqual(result.control.diagnostics["distributed_grid_leader_conditioned"], 1.0)
        self.assertEqual(result.control.diagnostics["distributed_grid_full_search_active"], 1.0)
        self.assertGreater(result.control.diagnostics["distributed_grid_sensitivity_probe_candidates"], 0.0)
        self.assertIn("distributed_grid_sensitivity_direction_candidates", result.control.diagnostics)
        self.assertIn("distributed_grid_rollout_objective", result.control.diagnostics)
        self.assertAlmostEqual(
            result.objective_value,
            result.control.diagnostics["distributed_grid_rollout_objective"],
        )
        self.assertEqual(set(result.control.vsl), set(cfg.network.freeway_links))
        # NOTE: boundary_out allocation 집계 단언은 PSO allocation 모드 전용이라 제거됨.
        # 기본 stackelberg_allocation_mode=direct에서는 inflow_outflow_allocation을 채우지
        # 않으므로(leader가 net-inflow/metering을 직접 적용), 이 테스트는 grid 진단만 검증한다.

    def test_leader_conditioned_grid_projects_metering_target(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {
                    "follower_solver_mode": "distributed",
                    "max_nash_iter": 1,
                },
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        previous = ControlAction.fixed(cfg)
        leader = LeaderAction(cfg.leader.N_P_crit_veh, 0.75 * cfg.network.total_ramp_capacity)
        candidates = coordinator._leader_conditioned_grid_candidates(
            previous,
            previous.copy(),
            leader,
            None,
            state=state,
            forecast=forecast,
        )

        self.assertGreater(len(candidates), 40)
        self.assertTrue(any(candidate.label.startswith("leader_coarse_global_green_") for candidate in candidates))
        self.assertTrue(any("target_net_inflow" in candidate.label for candidate in candidates))
        self.assertTrue(any(candidate.label.startswith("leader_coarse_global_vsl_") for candidate in candidates))
        self.assertTrue(any("offset" in candidate.label for candidate in candidates))
        base_candidates = coordinator._leader_conditioned_grid_candidates(
            previous,
            previous.copy(),
            leader,
            None,
        )
        base_best_residual = min(
            abs(coordinator._leader_direct_feasible_set_diagnostics(
                state,
                candidate.control,
                forecast,
                leader,
            )["distributed_grid_leader_net_inflow_residual_veh"])
            for candidate in base_candidates
        )
        augmented_best_residual = min(
            abs(coordinator._leader_direct_feasible_set_diagnostics(
                state,
                candidate.control,
                forecast,
                leader,
            )["distributed_grid_leader_net_inflow_residual_veh"])
            for candidate in candidates
        )
        self.assertLessEqual(augmented_best_residual, base_best_residual)
        for candidate in candidates:
            total_metering = sum(
                candidate.control.ramp_metering.get(ramp, 0.0)
                for ramp in cfg.network.ramps
            )
            self.assertAlmostEqual(total_metering, leader.N_UF_star, places=6)
            self.assertEqual(candidate.stage, "coarse")
            self.assertEqual(candidate.scope, "global")
            self.assertEqual(candidate.control.N_UF_star, leader.N_UF_star)
            self.assertFalse(candidate.control.inflow_outflow_allocation)

    def test_distributed_link_vsl_consensus_is_order_independent(self):
        cfg = short_config()
        coordinator = DistributedCoordinator(cfg)
        solves = [
            AgentSolve(agent_id="F_W0", objective=0.0, vsl={"FW_W": 100.0}),
            AgentSolve(agent_id="F_W1", objective=0.0, vsl={"FW_W": 80.0}),
            AgentSolve(agent_id="F_W2", objective=0.0, vsl={"FW_W": 60.0}),
        ]
        self.assertEqual(coordinator._aggregate_link_vsl(solves)["FW_W"], 60.0)
        self.assertEqual(coordinator._aggregate_link_vsl(list(reversed(solves)))["FW_W"], 60.0)

    def test_leaderless_metering_prediction_includes_upstream_mainline_flow(self):
        cfg = short_config()
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        agent = next(item for item in coordinator.freeway_agents if "R_D_W" in item.ramps)
        upper = {ramp: cfg.network.ramp_capacity_veh_h[ramp] for ramp in agent.ramps}

        upstream_idx = agent.segment_index - 1
        state.freeway_flow["FW_W"][upstream_idx] = 0.0
        low_inflow_target = coordinator._leaderless_metering_target(agent, state, upper, demand)
        state.freeway_flow["FW_W"][upstream_idx] = 8000.0
        high_inflow_target = coordinator._leaderless_metering_target(agent, state, upper, demand)

        self.assertLess(high_inflow_target, low_inflow_target)

    def test_distributed_response_objective_rewards_ramp_service(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 3, "max_nash_iter": 1},
                "freeway_follower": {"density_penalty": 1.0},
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [0.25 * cfg.network.rho_crit for _ in state.freeway_density[link]]
            state.freeway_speed[link] = [cfg.network.v_free for _ in state.freeway_speed[link]]
        state.refresh_freeway_flow(cfg.network)
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 300.0
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 3)

        low = ControlAction.fixed(cfg)
        low.N_UF_star = 0.0
        low.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        high = ControlAction.fixed(cfg)
        high.N_UF_star = cfg.network.total_ramp_capacity
        high.ramp_metering = dict(cfg.network.ramp_capacity_veh_h)

        low_obj, low_diag = coordinator._response_tts_objective(
            state,
            low,
            forecast,
            residual=0.0,
            proxy_objective=999.0,
        )
        high_obj, high_diag = coordinator._response_tts_objective(
            state,
            high,
            forecast,
            residual=0.0,
            proxy_objective=999.0,
        )

        self.assertEqual(low_diag["distributed_response_ttt_compatible"], 1.0)
        self.assertGreater(high_diag["distributed_response_ramp_release_veh"], low_diag["distributed_response_ramp_release_veh"])
        self.assertLess(high_obj, low_obj)

    def test_distributed_response_density_rollout_charges_ramp_release(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
                "freeway_follower": {"density_penalty": 1.0},
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [
                0.98 * cfg.network.rho_crit for _ in state.freeway_density[link]
            ]
            state.freeway_speed[link] = [cfg.network.v_min for _ in state.freeway_speed[link]]
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = cfg.network.ramp_queue_max_veh
        forecast = [
            DemandStep(
                freeway_mainline={link: 0.0 for link in cfg.network.freeway_links},
                urban_boundary={},
                ramp_arrival={ramp: 0.0 for ramp in cfg.network.ramps},
            )
        ]

        low = ControlAction.fixed(cfg)
        low.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        high = ControlAction.fixed(cfg)
        high.ramp_metering = dict(cfg.network.ramp_capacity_veh_h)

        _, low_diag = coordinator._response_tts_objective(
            state,
            low,
            forecast,
            residual=0.0,
            proxy_objective=0.0,
        )
        _, high_diag = coordinator._response_tts_objective(
            state,
            high,
            forecast,
            residual=0.0,
            proxy_objective=0.0,
        )

        self.assertGreater(
            high_diag["distributed_response_ramp_release_veh"],
            low_diag["distributed_response_ramp_release_veh"],
        )
        self.assertGreater(
            high_diag["distributed_response_density_excess_tts"],
            low_diag["distributed_response_density_excess_tts"],
        )
        self.assertGreater(
            high_diag["distributed_response_density_rollout_peak_density"],
            low_diag["distributed_response_density_rollout_peak_density"],
        )

    def test_distributed_response_objective_uses_lightweight_proxy_without_rollout(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 2, "max_nash_iter": 1},
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        control = ControlAction.fixed(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)

        with patch("src.simulation.coupling.run_coupled_interval") as coupled_step:
            objective, diagnostics = coordinator._response_tts_objective(
                state,
                control,
                forecast,
                residual=0.0,
                proxy_objective=123.0,
            )

        coupled_step.assert_not_called()
        self.assertGreater(objective, 0.0)
        self.assertAlmostEqual(diagnostics["distributed_response_objective_tts"], objective)
        self.assertEqual(diagnostics["distributed_response_rollout_active"], 0.0)
        self.assertEqual(diagnostics["distributed_response_rollout_ttt"], 0.0)
        self.assertGreater(diagnostics["distributed_response_terminal_proxy_vehicles"], 0.0)
        self.assertIn("distributed_response_total_spillback_violation_veh", diagnostics)

    def test_distributed_response_objective_does_not_double_count_freeway_queues(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 10.0
        for link in cfg.network.freeway_links:
            state.mainline_origin_queue[link] = 5.0
        control = ControlAction.fixed(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)

        _, diagnostics = coordinator._response_tts_objective(
            state,
            control,
            forecast,
            residual=0.0,
            proxy_objective=0.0,
        )

        ramp_queue = sum(state.ramp_queue.values())
        origin_queue = sum(state.mainline_origin_queue.values())
        segment_vehicles = state.freeway_segment_vehicles(cfg.network)
        self.assertAlmostEqual(
            state.total_freeway_vehicles(cfg.network),
            segment_vehicles + ramp_queue + origin_queue,
        )
        expected_current = (
            state.total_urban_vehicles(cfg.network)
            + segment_vehicles
            + ramp_queue
            + state.off_ramp_storage_occupancy_veh(cfg.network)
            + origin_queue
        )
        double_counted_current = (
            state.total_urban_vehicles(cfg.network)
            + state.total_freeway_vehicles(cfg.network)
            + ramp_queue
            + state.off_ramp_storage_occupancy_veh(cfg.network)
            + origin_queue
        )

        self.assertAlmostEqual(diagnostics["distributed_response_current_vehicles"], expected_current)
        self.assertNotAlmostEqual(
            diagnostics["distributed_response_current_vehicles"],
            double_counted_current,
        )
        self.assertAlmostEqual(
            diagnostics["distributed_response_freeway_segment_vehicles"],
            segment_vehicles,
        )
        self.assertAlmostEqual(diagnostics["distributed_response_ramp_queue_start_veh"], ramp_queue)
        self.assertAlmostEqual(diagnostics["distributed_response_origin_queue_start_veh"], origin_queue)

    def test_leaderless_distributed_evaluates_full_controller_guards(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
                "freeway_follower": {
                    "horizon_beam_width": 1,
                    "horizon_ramp_candidate_limit": 1,
                    "horizon_vsl_candidate_limit_per_link": 1,
                },
            },
        )
        coordinator = DistributedCoordinator(cfg)
        previous = ControlAction.uncontrolled(cfg)
        previous.green_times["A_p1"] = cfg.network.green_min
        previous.green_times["A_p2"] = cfg.network.effective_green_total - cfg.network.green_min
        previous.offsets["A"] = 12.0

        guards = dict(coordinator._full_controller_guard_candidates(previous))
        self.assertEqual(set(guards), {"previous", "no_control", "default"})
        self.assertEqual(guards["default"].inflow_outflow_allocation, {})

        result = coordinator.solve(
            TrafficState.initial(cfg),
            None,
            DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1),
            previous,
        )
        diagnostics = result.diagnostics

        self.assertEqual(diagnostics["distributed_full_controller_guard_active"], 1.0)
        self.assertEqual(diagnostics["distributed_previous_guard_evaluated"], 1.0)
        self.assertEqual(diagnostics["distributed_no_control_guard_evaluated"], 1.0)
        self.assertEqual(diagnostics["distributed_default_guard_evaluated"], 1.0)
        self.assertIn("distributed_previous_guard_objective_tts", diagnostics)
        self.assertIn("distributed_no_control_guard_objective_tts", diagnostics)
        self.assertIn("distributed_default_guard_objective_tts", diagnostics)
        selected_sum = (
            diagnostics["distributed_guard_selected_previous"]
            + diagnostics["distributed_guard_selected_no_control"]
            + diagnostics["distributed_guard_selected_default"]
        )
        self.assertIn(selected_sum, {0.0, 1.0})

    def test_leaderless_guard_selection_respects_spillback_constraint(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        ramp = cfg.network.ramps[0]
        for movement in cfg.network.on_ramp_to_movement[ramp]:
            state.urban_movement_queue[movement] = onramp_combined_capacity_veh(cfg, ramp)
        previous = ControlAction.uncontrolled(cfg)
        previous.ramp_metering = {candidate_ramp: 0.0 for candidate_ramp in cfg.network.ramps}

        result = coordinator.solve(
            state,
            None,
            DemandProfile(cfg, ScenarioConfig("test", ramp_scale=1.5)).horizon(0.0, 1),
            previous,
        )
        diagnostics = result.diagnostics

        self.assertGreater(diagnostics["distributed_previous_guard_spillback_violation_veh"], 0.0)
        self.assertEqual(diagnostics["distributed_guard_selected_previous"], 0.0)
        self.assertLess(
            diagnostics["distributed_response_total_spillback_violation_veh"],
            diagnostics["distributed_previous_guard_spillback_violation_veh"],
        )

    def test_distributed_freeway_agent_reports_neighbor_pressure(self):
        cfg = short_config()
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        previous = ControlAction.fixed(cfg)
        agent = next(item for item in coordinator.freeway_agents if item.id == "F_W1")
        state.freeway_density["FW_W"][2] = cfg.network.rho_crit + 25.0
        state.freeway_speed["FW_W"][2] = cfg.network.v_min
        state.refresh_freeway_flow(cfg.network)
        coupling = coordinator._extract_coupling(state, previous, demand)
        coupling["lane_loss_FW_W_seg2"] = 1.0

        solve = coordinator._solve_freeway_agent(
            agent,
            state,
            LeaderAction(0.0, 1200.0),
            [demand],
            previous,
            coupling,
        )

        self.assertGreater(solve.diagnostics["agent_F_W1_freeway_neighbor_pressure"], 0.0)
        self.assertLess(solve.diagnostics["agent_F_W1_freeway_neighbor_metering_factor"], 1.0)

    def test_distributed_ablation_diagnostics_report_blocked_coupling(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        previous = ControlAction.fixed(cfg)
        coordinator = DistributedCoordinator(cfg, ablation="NO_U_TO_F_INFO")

        result = coordinator.solve(
            state,
            LeaderAction(0.0, 1200.0),
            [demand],
            previous,
        )

        self.assertEqual(result.diagnostics["distributed_u_to_f_coupling_active"], 0.0)
        self.assertEqual(result.diagnostics["nash_freeway_used_coupled_prediction"], 0.0)
        self.assertEqual(result.diagnostics["distributed_f_to_u_coupling_active"], 1.0)

    def test_green_vsl_only_ttt_mode_preserves_wu_authority(self):
        cfg = short_config().with_updates({
            "mpc": {"relaxed_quantized_controls": True, "max_nash_iter": 2}
        })
        state = TrafficState.initial(cfg)
        for movement, spec in cfg.network.urban_movements.items():
            if spec.get("phase") == "A_p1":
                state.urban_movement_queue[movement] = 150.0
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("test", urban_scale=1.2, freeway_scale=1.2),
        ).horizon(0.0, cfg.mpc.horizon_steps)
        previous = ControlAction.fixed(cfg)
        for ramp in cfg.network.ramps:
            previous.ramp_metering[ramp] = 0.5 * cfg.network.ramp_capacity_veh_h[ramp]
        previous.offsets = {signal: 12.0 for signal in cfg.network.signals}
        previous.inflow_outflow_allocation = {
            movement: 123.0 for movement in cfg.network.urban_movements
        }

        result = DistributedCoordinator(
            cfg,
            ablation="WU_GREEN_VSL_ONLY_TTT",
        ).solve(state, None, forecast, previous)
        control = result.control

        self.assertEqual(control.diagnostics["wu_green_vsl_only_ttt_authority"], 1.0)
        self.assertEqual(control.inflow_outflow_allocation, {})
        for ramp in cfg.network.ramps:
            self.assertAlmostEqual(
                control.ramp_metering[ramp],
                cfg.network.ramp_capacity_veh_h[ramp],
            )
        for signal in cfg.network.signals:
            self.assertAlmostEqual(control.offsets[signal], 0.0)
            p1 = control.green_times[f"{signal}_p1"]
            p2 = control.green_times[f"{signal}_p2"]
            self.assertAlmostEqual(p1 + p2, cfg.network.effective_green_total)
            self.assertGreaterEqual(p1, cfg.network.green_min)
            self.assertGreaterEqual(p2, cfg.network.green_min)
        for link in cfg.network.freeway_links:
            self.assertIn(control.vsl[link], {float(v) for v in cfg.freeway_follower.vsl_set})

    def test_wu_cd_f_adapter_uses_green_vsl_only_ttt_coordinator(self):
        from src.experiments.six_controller_comparison import _ControllerAdapter

        cfg = short_config().with_updates({"mpc": {"max_nash_iter": 1}})
        adapter = _ControllerAdapter(cfg, "WU-CD-F")
        # d11cd16(2026-06-30): WU-CD-F default impl이 WuFaithfulFollower(authority="wu")로
        # 교체됨(six_controller_comparison.py 주석 참조) — 구 DistributedCoordinator 기대는 폐기.
        self.assertIsInstance(adapter._impl, WuFaithfulFollower)
        self.assertEqual(adapter._impl.authority, "wu")

        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        control, diag = adapter.decide(TrafficState.initial(cfg), forecast)

        # 구 DistributedCoordinator의 진단 키 대신 authority 게이트 플래그로 검증 —
        # 실질 권한(green+VSL only)은 아래 allocation/metering/offset 행동 검증이 담당.
        self.assertFalse(adapter._impl.metering_enabled)
        self.assertGreater(diag["solver_evaluations"], 0.0)
        self.assertEqual(control.inflow_outflow_allocation, {})
        for ramp in cfg.network.ramps:
            self.assertAlmostEqual(
                control.ramp_metering[ramp],
                cfg.network.ramp_capacity_veh_h[ramp],
            )
        for signal in cfg.network.signals:
            self.assertAlmostEqual(control.offsets[signal], 0.0)

    def test_wu_structured_grid_exposes_only_green_and_vsl_authority(self):
        cfg = short_config()
        previous = ControlAction.fixed(cfg)
        for ramp in cfg.network.ramps:
            previous.ramp_metering[ramp] = 0.5 * cfg.network.ramp_capacity_veh_h[ramp]
        previous.offsets = {signal: 17.0 for signal in cfg.network.signals}
        previous.inflow_outflow_allocation = {
            movement: 111.0 for movement in cfg.network.urban_movements
        }
        center = previous.copy()
        for link in cfg.network.freeway_links:
            center.vsl[link] = min(cfg.freeway_follower.vsl_set)

        grid = structured_grid_candidates(
            cfg,
            previous,
            center,
            authority="wu",
            stage="coarse",
            scope="global",
        )
        probes = sensitivity_probe_candidates(cfg, previous, center, authority="wu")
        candidates = grid + probes

        self.assertGreater(len(candidates), 0)
        self.assertTrue(any(c.label.startswith("green_") or c.axis.startswith("green") for c in candidates))
        self.assertTrue(any(c.label.startswith("vsl_") or c.axis.startswith("vsl") for c in candidates))
        self.assertFalse(any(c.label.startswith(("rm_", "offset_", "combo_rm")) for c in candidates))
        self.assertFalse(any(c.axis.startswith(("rm", "offset")) for c in candidates if c.axis))
        for candidate in candidates:
            control = candidate.control
            self.assertEqual(control.diagnostics["wu_green_vsl_only_ttt_authority"], 1.0)
            self.assertEqual(control.inflow_outflow_allocation, {})
            for ramp in cfg.network.ramps:
                self.assertAlmostEqual(
                    control.ramp_metering[ramp],
                    cfg.network.ramp_capacity_veh_h[ramp],
                )
            for signal in cfg.network.signals:
                self.assertAlmostEqual(control.offsets[signal], 0.0)

    def test_leader_candidate_budget_covers_extremes_and_previous_action(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        previous = ControlAction.fixed(cfg)
        # previous N_P_star는 후보에 보존되되, 도달 가능 net-inflow 범위로 클램프된다(2026-06-22).
        # N_P_star는 horizon당 net-inflow 목표라 빈/저혼잡 state에서 큰 음수(-250)는 도달 불가 →
        # 범위 하한으로 클램프되는 것이 올바른 동작이다.
        prev_np = -250.0
        previous.N_P_star = prev_np
        previous.N_UF_star = 3333.0
        candidates = Leader(cfg).candidates(state, previous, demand)

        pairs = {(round(c.N_P_star, 6), round(c.N_UF_star, 6)) for c in candidates}
        np_values = [c.N_P_star for c in candidates]
        nuf_values = [c.N_UF_star for c in candidates]
        self.assertIn((round(min(np_values), 6), round(min(nuf_values), 6)), pairs)
        self.assertIn((round(max(np_values), 6), round(max(nuf_values), 6)), pairs)
        clamped_prev = min(max(prev_np, min(np_values)), max(np_values))
        self.assertIn((round(clamped_prev, 6), 3333.0), pairs)

    def test_leader_nuf_candidates_and_projection_respect_ramp_bounds(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        previous = ControlAction.fixed(cfg)
        min_total = sum(
            cfg.freeway_follower.ramp_metering_rate_min * cfg.network.ramp_capacity_veh_h[ramp]
            for ramp in cfg.network.ramps
        )

        candidates = Leader(cfg).candidates(state, previous, demand)
        self.assertGreaterEqual(min(c.N_UF_star for c in candidates), min_total - 1.0e-6)

        coordinator = DistributedCoordinator(cfg)
        weights = {ramp: cfg.network.ramp_capacity_veh_h[ramp] for ramp in cfg.network.ramps}
        projected = coordinator._leader_metering_projection(
            LeaderAction(0.0, 0.0),
            weights,
        )
        self.assertAlmostEqual(sum(projected.values()), min_total)
        for ramp, release in projected.items():
            self.assertGreaterEqual(
                release,
                cfg.freeway_follower.ramp_metering_rate_min * cfg.network.ramp_capacity_veh_h[ramp] - 1.0e-6,
            )

    def test_stackelberg_normalizes_no_control_previous_nuf_reference(self):
        cfg = short_config()
        previous = ControlAction.fixed(cfg)
        controller = StackelbergMPCController(cfg)

        normalized = controller._normalize_previous_leader_reference(previous)

        self.assertEqual(previous.N_UF_star, 0.0)
        self.assertAlmostEqual(normalized.N_UF_star, cfg.network.total_ramp_capacity)

    def test_stackelberg_fallback_guard_rejects_terminal_worse_leader(self):
        cfg = short_config()
        controller = StackelbergMPCController(cfg)

        def evaluation(stage: str, objective: float, terminal: float, completed: float) -> _LeaderCandidateEvaluation:
            control = ControlAction.fixed(cfg)
            control.diagnostics.update({
                "distributed_response_terminal_proxy_vehicles": terminal,
                "distributed_response_mainline_exit_veh": 0.5 * completed,
                "distributed_response_boundary_out_sink_veh": 0.5 * completed,
            })
            nash = NashResult(
                control=control,
                objective_value=objective,
                iterations=1,
                converged=True,
                residual_objective=0.0,
                residual_control=0.0,
                diagnostics=dict(control.diagnostics),
            )
            return _LeaderCandidateEvaluation(
                index=0,
                action=LeaderAction(control.N_P_star, control.N_UF_star),
                nash=nash,
                objective=objective,
                objective_terms={
                    "leader_total_objective": objective,
                    "leader_objective_base": objective,
                    "leader_follower_ttt_base": objective,
                },
                metadata={},
                rollout_used=False,
                stage=stage,
            )

        leader = evaluation("refined", 80.0, terminal=5000.0, completed=20.0)
        fallback = evaluation("fallback_pfo", 100.0, terminal=1000.0, completed=40.0)

        best, meta = controller._select_with_fallback_guard([leader], [fallback])

        self.assertEqual(best.stage, "fallback_pfo")
        self.assertEqual(meta["leader_fallback_guard_selected"], 1.0)
        self.assertEqual(meta["leader_fallback_guard_rejected_leader"], 1.0)
        self.assertEqual(meta["leader_fallback_guard_terminal_severe"], 1.0)

    def test_stackelberg_wu_pfo_incumbent_can_be_selected_when_leader_is_worse(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {
                    "horizon_steps": 1,
                    "max_nash_iter": 1,
                    "stackelberg_enable_fallback": False,
                    "stackelberg_enable_pfo_incumbent": True,
                    "stackelberg_leader_parallel_backend": "serial",
                    "grid_parallel_backend": "serial",
                },
            },
        )
        controller = StackelbergWuMeteredController(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)

        pfo_candidates = controller._evaluate_fallback_candidates(
            state,
            forecast,
            previous,
            start_index=1000000,
        )
        self.assertEqual(len(pfo_candidates), 1)
        pfo = pfo_candidates[0]
        worse_control = ControlAction.fixed(cfg)
        worse_control.N_P_star = pfo.action.N_P_star
        worse_control.N_UF_star = pfo.action.N_UF_star
        worse_nash = NashResult(
            control=worse_control,
            objective_value=pfo.objective + 10.0,
            iterations=1,
            converged=True,
            residual_objective=0.0,
            residual_control=0.0,
            diagnostics={},
        )
        worse_leader = _LeaderCandidateEvaluation(
            index=0,
            action=LeaderAction(pfo.action.N_P_star, pfo.action.N_UF_star),
            nash=worse_nash,
            objective=pfo.objective + 10.0,
            objective_terms={
                "leader_total_objective": pfo.objective + 10.0,
                "leader_objective_base": pfo.objective + 10.0,
                "leader_follower_ttt_base": pfo.objective + 10.0,
            },
            metadata={},
            rollout_used=False,
            stage="refined",
        )

        best, meta = controller._select_with_fallback_guard([worse_leader], pfo_candidates)

        self.assertEqual(best.stage, "fallback_pfo")
        self.assertEqual(meta["leader_pfo_incumbent_active"], 1.0)
        self.assertEqual(meta["leader_pfo_incumbent_selected"], 1.0)
        for key in (
            "leader_pfo_incumbent_N_P_star",
            "leader_pfo_incumbent_N_UF_star",
            "leader_pfo_incumbent_objective",
        ):
            self.assertIsInstance(meta[key], float)

    def test_stackelberg_wu_pfo_incumbent_flag_can_disable_candidate(self):
        cfg = short_config().with_updates({
            "mpc": {
                "stackelberg_enable_fallback": False,
                "stackelberg_enable_pfo_incumbent": False,
            },
        })
        controller = StackelbergWuMeteredController(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)

        self.assertFalse(controller._pfo_incumbent_fallback_enabled())
        self.assertEqual(
            controller._evaluate_fallback_candidates(
                state,
                forecast,
                ControlAction.fixed(cfg),
                start_index=1000000,
            ),
            [],
        )

    def test_stackelberg_leader_evaluates_coarse_and_refined_grid(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {
                    "follower_solver_mode": "two_block",
                    "horizon_steps": 1,
                    "leader_search_mode": "grid",
                    "leader_candidate_count": 4,
                    "leader_refinement_candidate_count": 5,
                    "stackelberg_prefilter_top_k": 2,
                    "stackelberg_prefilter_local_top_k": 2,
                    "stackelberg_enable_fallback": False,
                    "max_nash_iter": 1,
                    "stackelberg_leader_parallel_backend": "serial",
                    "grid_parallel_backend": "serial",
                },
            },
        )
        result = StackelbergMPCController(cfg).decide_with_info(
            TrafficState.initial(cfg),
            DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps),
            ControlAction.fixed(cfg),
        )
        meta = result.metadata

        self.assertGreater(meta["leader_candidate_coarse_count"], 0.0)
        self.assertGreater(meta["leader_candidate_refined_count"], 0.0)
        self.assertEqual(meta["leader_candidate_refinement_active"], 1.0)
        self.assertEqual(meta["leader_candidate_global_refresh"], 1.0)
        self.assertEqual(meta["leader_candidate_prefilter_active"], 1.0)
        self.assertEqual(meta["leader_candidate_prefilter_scope_global"], 1.0)
        self.assertEqual(meta["leader_candidate_refined_prefilter_active"], 1.0)
        self.assertEqual(meta["leader_candidate_refined_prefilter_scope_local"], 1.0)
        self.assertAlmostEqual(
            meta["leader_candidate_full_evaluated_count"],
            meta["leader_candidate_coarse_evaluated_count"] + meta["leader_candidate_refined_evaluated_count"],
        )
        self.assertLess(meta["leader_candidate_full_evaluated_count"], meta["leader_candidate_count"])
        self.assertEqual(meta["leader_fallback_enabled"], 0.0)
        self.assertEqual(meta["leader_fallback_incumbent_seed_active"], 0.0)
        self.assertGreaterEqual(
            meta["leader_candidate_best_stage_coarse"]
            + meta["leader_candidate_best_stage_refined"]
            + meta["leader_candidate_best_stage_fallback"],
            1.0,
        )

    def test_stackelberg_leader_continuous_search_evaluates_targets(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {
                    "follower_solver_mode": "two_block",
                    "horizon_steps": 1,
                    "leader_search_mode": "continuous",
                    "leader_continuous_max_evals": 3,
                    "leader_continuous_seed_count": 2,
                    "leader_continuous_prefilter_samples": 5,
                    "leader_continuous_prefilter_top_k": 2,
                    "leader_continuous_local_iterations": 0,
                    "stackelberg_enable_fallback": False,
                    "stackelberg_leader_parallel_backend": "serial",
                    "max_nash_iter": 1,
                    "grid_parallel_backend": "serial",
                },
            },
        )
        result = StackelbergMPCController(cfg).decide_with_info(
            TrafficState.initial(cfg),
            DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps),
            ControlAction.fixed(cfg),
        )
        meta = result.metadata

        self.assertEqual(meta["leader_search_mode_continuous"], 1.0)
        self.assertEqual(meta["leader_search_mode_grid"], 0.0)
        self.assertGreater(meta["leader_candidate_full_evaluated_count"], 0.0)
        self.assertLessEqual(meta["leader_candidate_full_evaluated_count"], 3.0)
        self.assertEqual(meta["leader_candidate_parallel_backend_serial"], 1.0)
        self.assertEqual(meta["leader_continuous_prefilter_active"], 1.0)
        self.assertGreater(meta["leader_continuous_prefilter_proxy_evaluated_count"], 0.0)
        self.assertLessEqual(
            meta["leader_continuous_prefilter_selected_count"],
            meta["leader_continuous_prefilter_top_k"] + 1.0,
        )
        self.assertGreaterEqual(meta["leader_continuous_search_bound_np_upper"], meta["leader_continuous_search_bound_np_lower"])
        self.assertGreaterEqual(
            meta["leader_continuous_search_bound_nuf_upper"],
            meta["leader_continuous_search_bound_nuf_lower"],
        )

    def test_allocation_net_inflow_binding_uses_inflow_outflow_extremes(self):
        cfg = short_config()
        module = InflowOutflowAllocationModule(cfg)
        specs = {
            movement: spec
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") in {"boundary_in", "off_ramp", "boundary_out", "on_ramp"}
        }
        movements = list(specs)
        kinds = [str(specs[movement].get("kind", "")) for movement in movements]
        lower, upper = module._bounds(
            movements,
            specs,
            LeaderAction(cfg.leader.N_P_crit_veh, cfg.network.total_ramp_capacity),
        )
        inflow = np.asarray([kind in {"boundary_in", "off_ramp"} for kind in kinds], dtype=bool)
        outflow = np.asarray([kind in {"boundary_out", "on_ramp"} for kind in kinds], dtype=bool)
        expected_min = float(np.sum(lower[inflow]) - np.sum(upper[outflow]))
        expected_max = float(np.sum(upper[inflow]) - np.sum(lower[outflow]))

        self.assertAlmostEqual(module._clip_target(-1.0e9, lower, upper, kinds), expected_min)
        self.assertAlmostEqual(module._clip_target(1.0e9, lower, upper, kinds), expected_max)

    def test_internal_movement_not_throttled_by_allocation(self):
        # 불변식: 내부(internal) movement는 inflow_outflow_allocation으로 cap되지 않는다.
        # perimeter movement만 allocation cap이 적용된다(green×saturation은 내부 전용 제어).
        from src.models.urban_queue_model import (
            PERIMETER_MOVEMENT_KINDS,
            _movement_capacity_flow,
            movement_specs,
        )

        cfg = short_config()
        net = cfg.network
        specs = movement_specs(cfg)
        internal = next(m for m, s in specs.items() if s.get("kind") == "internal")
        perimeter = next(
            m for m, s in specs.items() if s.get("kind") in PERIMETER_MOVEMENT_KINDS
        )

        control = ControlAction.fixed(cfg)
        # fixed()는 내부 movement에 allocation을 사전충전하지 않는다.
        self.assertNotIn(internal, control.inflow_outflow_allocation)

        throttle = net.movement_capacity_veh_h * 0.5
        control.inflow_outflow_allocation[internal] = throttle
        control.inflow_outflow_allocation[perimeter] = throttle
        # 내부 movement는 allocation 700이 들어 있어도 물리용량(1400)을 반환한다.
        self.assertEqual(
            _movement_capacity_flow(control, cfg, internal, specs[internal]),
            net.movement_capacity_veh_h,
        )
        # perimeter movement는 allocation cap이 그대로 적용된다.
        self.assertEqual(
            _movement_capacity_flow(control, cfg, perimeter, specs[perimeter]),
            throttle,
        )

    def test_stackelberg_can_use_distributed_follower_solver(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {
                    "follower_solver_mode": "distributed",
                    "max_nash_iter": 2,
                    "leader_candidate_count": 2,
                },
            },
        )
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        control = StackelbergMPCController(cfg).decide(TrafficState.initial(cfg), demand)
        self.assertEqual(control.diagnostics["distributed_player_active"], 1.0)
        self.assertEqual(control.diagnostics["nash_per_agent_active"], 1.0)

    def test_green_times_sum_to_cycle_length(self):
        cfg = short_config()
        control = ControlAction.fixed(cfg)
        for signal in cfg.network.signals:
            total = control.green_times[f"{signal}_p1"] + control.green_times[f"{signal}_p2"] + cfg.network.lost_time
            self.assertAlmostEqual(total, cfg.network.cycle_length)

    def test_green_time_bounds(self):
        cfg = short_config()
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        control = StackelbergMPCController(cfg).decide(TrafficState.initial(cfg), demand)
        for value in control.green_times.values():
            self.assertGreaterEqual(value, cfg.network.green_min)
            self.assertLessEqual(value, cfg.network.green_max)

    def test_urban_follower_returns_movement_level_allocations(self):
        cfg = short_config()
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        control = StackelbergMPCController(cfg).decide(TrafficState.initial(cfg), demand)
        movement_keys = [
            movement for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") in {"boundary_in", "off_ramp", "on_ramp"}
        ]
        self.assertTrue(all(movement in control.inflow_outflow_allocation for movement in movement_keys))

    def test_offset_range(self):
        cfg = short_config()
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        control = StackelbergMPCController(cfg).decide(TrafficState.initial(cfg), demand)
        for value in control.offsets.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, cfg.network.cycle_length)

    def test_leader_np_candidates_use_feasible_net_inflow_range(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "mpc": {"leader_candidate_count": 6},
                "leader": {
                    "N_P_star_range": [-4000.0, 4000.0],
                    "N_P_crit_veh": 172.0,
                    "N_P_candidate_lower_factor": 0.9,
                    "N_P_candidate_upper_factor": 1.05,
                },
            },
        )
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        previous = ControlAction.fixed(cfg)
        previous.N_P_star = 333.0

        leader = Leader(cfg)
        bounds = leader._candidate_bounds(state, previous, forecast[0], forecast)
        actions = leader.candidates(state, previous, forecast[0], forecast)
        nps = [action.N_P_star for action in actions]
        self.assertLess(bounds.np_lower, 0.0)
        self.assertGreater(bounds.np_upper, 0.0)
        self.assertTrue(all(bounds.np_lower - 1.0e-9 <= value <= bounds.np_upper + 1.0e-9 for value in nps))
        self.assertTrue(any(value < 0.0 for value in nps))
        # N_P_star는 horizon당 net-inflow 목표라 후보 상한은 도달 가능 범위를 따른다
        # (종전 capacity 기반 고정 >1000 기대는 폐기, 2026-06-22). 후보가 도달 가능 상한부를 덮는지 확인.
        self.assertGreater(bounds.np_upper, 0.0)
        self.assertTrue(any(value > 0.5 * bounds.np_upper for value in nps))
        self.assertTrue(any(abs(value) <= 1.0e-9 for value in nps))
        self.assertFalse(all(0.9 * 172.0 <= value <= 1.05 * 172.0 for value in nps))

        # internal movement 큐는 보호영역 누적 N_P에 포함된다(그리드 라우팅 이후 정의).

    def test_default_leader_np_grid_covers_feasible_net_inflow_range(self):
        cfg = ExperimentConfig.from_file("src/config/default.yaml")
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, cfg.mpc.horizon_steps)
        bounds = Leader(cfg)._candidate_bounds(state, None, forecast[0], forecast)

        # 수정(2026-06-22): N_P_star는 horizon당 net-inflow 목표[veh]이며 도달 가능 범위
        # (movement별 큐+도착 servable)로 클램프된다. 종전 capacity envelope(±3500 근처)는
        # 도달 불가 영역을 포함해 N_P_star 탐색을 saturation 평원으로 퇴화시켰다.
        # 도달 가능 클램프가 활성(config 범위보다 안쪽)이고, 양/음 net-inflow를 모두 표현 가능한지 확인한다.
        self.assertGreater(bounds.np_lower, float(cfg.leader.N_P_star_range[0]))
        self.assertLess(bounds.np_upper, float(cfg.leader.N_P_star_range[1]))
        self.assertGreater(bounds.np_upper, 0.0)
        self.assertLessEqual(bounds.np_lower, 0.0)

    def test_leader_search_area_expands_under_high_stress(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"mpc": {"leader_candidate_count": 21}},
        )
        leader = Leader(cfg)
        low_state = TrafficState.initial(cfg)
        high_state = TrafficState.initial(cfg)
        for link, values in high_state.freeway_density.items():
            high_state.freeway_density[link] = [
                cfg.network.rho_crit * 1.05 for _ in values
            ]
        for ramp in cfg.network.ramps:
            high_state.ramp_queue[ramp] = 0.75 * cfg.network.ramp_queue_max_veh

        low_forecast = DemandProfile(cfg, ScenarioConfig("low")).horizon(0.0, cfg.mpc.horizon_steps)
        high_forecast = DemandProfile(
            cfg,
            ScenarioConfig("sweet_220_like", urban_scale=2.2, freeway_scale=2.2, ramp_scale=2.2),
        ).horizon(0.0, cfg.mpc.horizon_steps)

        low_bounds = leader._candidate_bounds(low_state, None, low_forecast[0], low_forecast)
        high_bounds = leader._candidate_bounds(high_state, None, high_forecast[0], high_forecast)
        high_actions = leader.candidates(high_state, ControlAction.fixed(cfg), high_forecast[0], high_forecast)

        self.assertGreater(high_bounds.stress_index, low_bounds.stress_index)
        self.assertGreater(
            high_bounds.np_upper - high_bounds.np_lower,
            low_bounds.np_upper - low_bounds.np_lower,
        )
        self.assertGreaterEqual(high_bounds.nuf_upper, 0.75 * cfg.network.total_ramp_capacity)
        self.assertTrue(any(
            action.N_P_star <= high_bounds.movement_capacity_np_lower + 1.0e-6
            or action.N_P_star >= min(high_bounds.movement_capacity_np_upper, high_bounds.np_upper) - 1.0e-6
            for action in high_actions
        ))
        self.assertTrue(any(
            abs(action.N_UF_star - high_bounds.nuf_queue_drain_target) <= 1.0e-6
            or abs(action.N_UF_star - high_bounds.nuf_upper) <= 1.0e-6
            for action in high_actions
        ))

    def test_leader_objective_matches_spec_accumulation_form(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "leader": {
                    "objective_mode": "state_accumulation",
                    "w_P": 2.0,
                    "w_F": 3.0,
                    "w_L": 0.5,
                    "N_P_crit_veh": 100.0,
                    "mfd_penalty_mode": "protected_exceed",
                    "non_convergence_penalty": 0.0,
                }
            },
        )
        state = TrafficState.initial(cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        internal_movement = next(
            movement
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "internal"
        )
        boundary_movement = next(
            movement
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "boundary_in"
        )
        state.urban_movement_queue[internal_movement] = 120.0
        state.urban_movement_queue[boundary_movement] = 77.0
        # 보호영역 누적(내부 link storage 점유)을 perimeter penalty 경로가 동작하도록 설정한다.
        grid_cap = cfg.network.urban_link_storage_veh["A_to_D"]
        state.urban_link_storage["A_to_D"] = grid_cap - 150.0
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [cfg.network.rho_crit + 2.0 for _ in state.freeway_density[link]]
            state.freeway_speed[link] = [cfg.network.v_free for _ in state.freeway_speed[link]]
        state.refresh_freeway_flow(cfg.network)

        action = ControlAction.fixed(cfg)
        action.N_P_star = 170.0
        action.N_UF_star = 300.0
        previous = ControlAction.fixed(cfg)
        previous.N_P_star = 160.0
        previous.N_UF_star = 250.0

        n_p = state.objective_urban_vehicles(cfg.network)
        n_p_protected = state.protected_accumulation_veh(cfg.network)
        n_f = state.total_freeway_vehicles(cfg.network)
        density_excess = sum(
            cfg.network.freeway_segment_length_km
            * cfg.network.freeway_lanes
            * max(0.0, rho - cfg.network.rho_crit)
            for values in state.freeway_density.values()
            for rho in values
        )
        # boundary_in queue is part of Total TTT coverage and enters the objective.
        boundary_in_queue = state.boundary_in_queue_vehicles(cfg.network)
        expected = (
            n_p
            + n_f
            + 2.0 * max(0.0, n_p_protected - 100.0)
            + cfg.leader.w_boundary_in * boundary_in_queue
            + 3.0 * density_excess
        )
        leader = Leader(cfg)
        terms = leader.objective_terms(
            [state], action, previous, follower_objective=9999.0, nash_converged=True
        )
        actual = leader.objective(
            [state], action, previous, follower_objective=9999.0, nash_converged=True
        )
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(terms["leader_total_objective"], expected)
        self.assertAlmostEqual(terms["leader_boundary_leg_excluded_veh"], 77.0)
        self.assertAlmostEqual(terms["leader_target_penalty"], 2.0 * max(0.0, n_p_protected - 100.0))
        self.assertAlmostEqual(terms["leader_boundary_in_queue_penalty"], cfg.leader.w_boundary_in * boundary_in_queue)
        self.assertAlmostEqual(terms["leader_smoothness_penalty"], 0.0)

    def test_default_leader_objective_uses_follower_ttt_base(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "leader": {
                    "w_P": 0.0,
                    "w_F": 0.0,
                    "w_L": 0.0,
                    "w_boundary_in": 5.0,
                    "non_convergence_penalty": 500.0,
                }
            },
        )
        state = TrafficState.initial(cfg)
        boundary_movement = next(
            movement
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "boundary_in"
        )
        state.urban_movement_queue[boundary_movement] = 40.0

        terms = Leader(cfg).objective_terms(
            [state],
            ControlAction.fixed(cfg),
            previous=None,
            follower_objective=1234.0,
            nash_converged=False,
            nash_residual_objective=0.2,
            nash_residual_control=0.1,
        )

        self.assertEqual(cfg.leader.objective_mode, "follower_ttt")
        self.assertAlmostEqual(terms["leader_follower_ttt_base"], 1234.0)
        self.assertAlmostEqual(terms["leader_objective_base"], 1234.0)
        self.assertGreater(terms["leader_boundary_in_queue_penalty"], 0.0)
        self.assertGreater(terms["leader_nonconvergence_penalty"], 0.0)
        expected_boundary = (
            5.0 * state.boundary_in_queue_vehicles(cfg.network) * cfg.simulation.T_c_h
        )
        self.assertAlmostEqual(terms["leader_boundary_in_queue_penalty"], expected_boundary)
        self.assertAlmostEqual(terms["leader_total_objective"], 1234.0 + expected_boundary)

    def test_leader_all_urban_halfcap_penalty_counts_boundary_and_storage(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "leader": {
                    "w_P": 99.0,
                    "w_F": 0.0,
                    "w_boundary_in": 0.0,
                    "mfd_penalty_mode": "all_urban_halfcap",
                    "mfd_storage_threshold_ratio": 0.5,
                    "mfd_storage_weight": 2.0,
                    "mfd_boundary_queue_capacity_veh": 220.0,
                }
            },
        )
        state = TrafficState.initial(cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0

        boundary_movement, boundary_spec = next(
            (movement, spec)
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "boundary_in"
        )
        boundary_out_movement, boundary_out_spec = next(
            (movement, spec)
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "boundary_out"
        )
        internal_movement, internal_spec = next(
            (movement, spec)
            for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "internal"
        )
        boundary_capacity = cfg.leader.mfd_boundary_queue_capacity_veh
        boundary_out_capacity = movement_storage_capacity(cfg, boundary_out_movement, boundary_out_spec)
        internal_capacity = movement_storage_capacity(cfg, internal_movement, internal_spec)
        state.urban_movement_queue[boundary_movement] = 0.5 * boundary_capacity + 10.0
        state.urban_movement_queue[boundary_out_movement] = 0.5 * boundary_out_capacity + 15.0
        state.urban_movement_queue[internal_movement] = 0.5 * internal_capacity + 20.0

        link = "A_to_D"
        link_capacity = cfg.network.urban_link_storage_veh[link]
        state.urban_link_storage[link] = link_capacity - (0.5 * link_capacity + 30.0)

        terms = Leader(cfg).objective_terms(
            [state],
            ControlAction.fixed(cfg),
            previous=None,
            follower_objective=100.0,
            nash_converged=True,
        )

        expected_excess = 10.0 + 15.0 + 20.0 + 30.0
        expected_penalty = 2.0 * expected_excess * cfg.simulation.T_c_h
        self.assertEqual(cfg.leader.mfd_penalty_mode, "all_urban_halfcap")
        self.assertAlmostEqual(terms["leader_target_penalty"], 0.0)
        self.assertAlmostEqual(terms["leader_mfd_movement_excess_veh"], 45.0)
        self.assertAlmostEqual(terms["leader_mfd_boundary_queue_capacity_veh"], 220.0)
        self.assertAlmostEqual(terms["leader_mfd_link_excess_veh"], 30.0)
        self.assertAlmostEqual(terms["leader_mfd_storage_excess_veh"], expected_excess)
        self.assertAlmostEqual(terms["leader_mfd_storage_penalty"], expected_penalty)
        self.assertAlmostEqual(terms["leader_total_objective"], 100.0 + expected_penalty)

    def test_default_leader_accumulation_penalties_use_control_interval_hours(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "leader": {
                    "w_P": 2.0,
                    "w_F": 3.0,
                    "w_L": 0.0,
                    "N_P_crit_veh": 100.0,
                    "mfd_penalty_mode": "protected_exceed",
                }
            },
        )
        state = TrafficState.initial(cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        grid_cap = cfg.network.urban_link_storage_veh["A_to_D"]
        state.urban_link_storage["A_to_D"] = grid_cap - 150.0
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [
                cfg.network.rho_crit + 2.0
                for _ in state.freeway_density[link]
            ]
            state.freeway_speed[link] = [cfg.network.v_free for _ in state.freeway_speed[link]]
        state.refresh_freeway_flow(cfg.network)

        n_p_excess = max(0.0, state.protected_accumulation_veh(cfg.network) - 100.0)
        density_excess = sum(
            cfg.network.freeway_segment_length_km
            * cfg.network.freeway_lanes
            * max(0.0, rho - cfg.network.rho_crit)
            for values in state.freeway_density.values()
            for rho in values
        )
        dt_h = cfg.simulation.T_c_h
        expected_target = 2.0 * n_p_excess * dt_h
        expected_density = 3.0 * density_excess * dt_h

        terms = Leader(cfg).objective_terms(
            [state],
            ControlAction.fixed(cfg),
            previous=None,
            follower_objective=1000.0,
            nash_converged=True,
        )

        self.assertEqual(cfg.leader.objective_mode, "follower_ttt")
        self.assertAlmostEqual(terms["leader_target_penalty"], expected_target)
        self.assertAlmostEqual(terms["leader_density_penalty"], expected_density)
        self.assertAlmostEqual(
            terms["leader_total_objective"],
            1000.0 + expected_target + expected_density,
        )

    def test_leader_non_convergence_penalty_uses_residuals(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "leader": {
                    "w_P": 0.0,
                    "w_F": 0.0,
                    "non_convergence_penalty": 500.0,
                    "non_convergence_objective_residual_scale": 0.5,
                    "non_convergence_control_residual_scale": 0.25,
                }
            },
        )
        state = TrafficState.initial(cfg)
        action = ControlAction.fixed(cfg)
        terms = Leader(cfg).objective_terms(
            [state],
            action,
            previous=None,
            follower_objective=77.0,
            nash_converged=False,
            nash_residual_objective=0.2,
            nash_residual_control=0.1,
        )
        expected_penalty = 500.0 * ((0.2 / 0.5) + (0.1 / 0.25))
        expected_boundary = (
            cfg.leader.w_boundary_in
            * state.boundary_in_queue_vehicles(cfg.network)
            * cfg.simulation.T_c_h
        )
        self.assertAlmostEqual(terms["leader_nonconvergence_penalty"], expected_penalty)
        self.assertAlmostEqual(terms["leader_total_objective"], 77.0 + expected_boundary)

    def test_leader_density_penalty_uses_effective_lanes_only_when_changed(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"leader": {"w_F": 1.0, "use_effective_lanes_for_density_penalty": True}},
        )
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [
                cfg.network.rho_crit + 2.0
                for _ in state.freeway_density[link]
            ]
            state.freeway_effective_lanes[link] = [
                float(cfg.network.freeway_lanes)
                for _ in state.freeway_density[link]
            ]
        first_link = cfg.network.freeway_links[0]
        state.freeway_effective_lanes[first_link][0] = cfg.network.freeway_lanes - 0.35

        nominal_count = sum(len(values) for values in state.freeway_density.values())
        expected_excess = (
            cfg.network.freeway_segment_length_km
            * (
                (cfg.network.freeway_lanes - 0.35)
                + (nominal_count - 1) * cfg.network.freeway_lanes
            )
            * 2.0
        )
        terms = Leader(cfg).objective_terms(
            [state],
            ControlAction.fixed(cfg),
            previous=None,
            follower_objective=0.0,
            nash_converged=True,
        )
        self.assertAlmostEqual(terms["leader_density_excess"], expected_excess)
        self.assertAlmostEqual(
            terms["leader_density_penalty"],
            expected_excess * cfg.simulation.T_c_h,
        )
        self.assertEqual(terms["leader_density_effective_lane_weight_count"], 1.0)

    def test_ramp_metering_bounds(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        result = FreewayFollower(cfg).solve(state, LeaderAction(0.0, 5000.0), demand)
        for ramp, value in result.ramp_metering.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, cfg.network.ramp_capacity_veh_h[ramp])

    def test_total_metering_tracking_or_infeasibility_flag(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        result = FreewayFollower(cfg).solve(state, LeaderAction(0.0, 10000.0), demand)
        error = abs(sum(result.ramp_metering.values()) - 10000.0)
        # N_UF는 ceiling 목표: 원목표(10000)를 못 채우면 추적잔차 대신
        # metering_target_infeasible로 명시 로깅된다(acceptance 기준 문서 참조).
        self.assertTrue(
            error <= cfg.freeway_follower.eps_F
            or result.infeasibility["metering_residual"] > 0.0
            or result.infeasibility["metering_target_infeasible"] > 0.0
        )

    def test_ramp_metering_respects_downstream_receiving_capacity(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        # ramp별 merge segment를 config 기준으로 전부 jam(R_D=seg2, R_F=seg3 — 망 변경 반영).
        for ramp, link in cfg.network.ramp_to_freeway.items():
            merge_idx = cfg.network.ramp_merge_segment_index[ramp]
            state.freeway_density[link][merge_idx] = cfg.network.rho_max
        demand = DemandProfile(cfg, ScenarioConfig("test", ramp_scale=3.0)).at(0.0)
        result = FreewayFollower(cfg).solve(state, LeaderAction(0.0, 3000.0), demand)
        self.assertTrue(all(value <= 1.0e-9 for value in result.ramp_metering.values()))
        # receiving 붕괴로 목표(3000)를 풀 수 없는 상황 — 명시적 infeasible 플래그가
        # 핵심 검증이다. (큐-증가 기반 잔차는 경량 예측 경로에선 큐를 안 키워 0일 수 있음.)
        self.assertGreater(result.infeasibility["metering_target_infeasible"], 0.0)

    def test_freeway_follower_scores_over_forecast_horizon(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 3},
                "freeway_follower": {"vsl_set": [100], "max_vsl_step": 0.0},
            },
        )
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 3)
        used_demand_ids = []

        def fake_lightweight_transition(*args):
            state_arg = args[-3]
            demand_step = args[-1]
            used_demand_ids.append(id(demand_step))
            return (
                state_arg.copy(),
                1.0,
                {"total_metering_error": 0.0, "mean_ramp_receiving_factor": 1.0},
            )

        with patch(
            "src.controllers.freeway_follower.FreewayFollower._lightweight_transition",
            side_effect=fake_lightweight_transition,
        ):
            result = FreewayFollower(cfg).solve(
                state,
                LeaderAction(0.0, 1200.0),
                demand,
                ControlAction.fixed(cfg),
            )

        self.assertIn(id(demand[2]), used_demand_ids)
        self.assertEqual(result.infeasibility["freeway_follower_horizon_steps"], 3.0)
        self.assertEqual(result.infeasibility["freeway_follower_sequence_optimized"], 1.0)

    def test_freeway_follower_expands_time_varying_vsl_sequence(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 2},
                "freeway_follower": {
                    "vsl_set": [60, 80, 100],
                    "max_vsl_step": 20.0,
                    "horizon_beam_width": 1,
                    "horizon_ramp_candidate_limit": 1,
                    "horizon_vsl_candidate_limit_per_link": 3,
                },
            },
        )
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        second_step_vsl_values = []

        def fake_lightweight_transition(*args):
            state_arg = args[-3]
            control = args[-2]
            demand_step = args[-1]
            if demand_step is demand[1]:
                second_step_vsl_values.extend(control.vsl.values())
            if demand_step is demand[0] and min(control.vsl.values()) < 100.0:
                return (
                    state_arg.copy(),
                    1.0,
                    {"total_metering_error": 0.0, "mean_ramp_receiving_factor": 1.0},
                )
            return (
                state_arg.copy(),
                10.0,
                {"total_metering_error": 0.0, "mean_ramp_receiving_factor": 1.0},
            )

        with patch(
            "src.controllers.freeway_follower.FreewayFollower._lightweight_transition",
            side_effect=fake_lightweight_transition,
        ):
            FreewayFollower(cfg).solve(
                state,
                LeaderAction(0.0, 1200.0),
                demand,
                ControlAction.fixed(cfg),
            )

        self.assertIn(60.0, second_step_vsl_values)

    def test_wu_faithful_vsl_sequence_reaches_lower_future_values(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "freeway_follower": {
                    "vsl_set": [50, 60, 70, 80, 90, 100],
                    "max_vsl_step": 20.0,
                    "vsl_sequence_search": True,
                    "vsl_sequence_horizon_steps": 4,
                    "vsl_sequence_candidate_limit": 128,
                },
            },
        )
        follower = WuFaithfulFollower(cfg)
        previous = ControlAction.fixed(cfg)
        link = cfg.network.freeway_links[0]
        n_seg = cfg.network.freeway_segments_per_link
        sequences = follower._freeway_vsl_sequence_candidates(
            link,
            n_seg,
            previous,
            [[max(cfg.freeway_follower.vsl_set)] * n_seg],
            horizon=4,
        )
        vsl_set = {float(value) for value in cfg.freeway_follower.vsl_set}
        bottleneck_idx = {
            int(cfg.network.off_ramp_segment_index.get(off_ramp, n_seg - 1))
            for off_ramp in cfg.network.off_ramps
            if cfg.network.off_ramp_from_freeway.get(off_ramp) == link
        } or {n_seg - 1}
        upstream_control_idx = {i for i in range(max(0, min(bottleneck_idx)))}
        vsl_max = float(max(cfg.freeway_follower.vsl_set))
        self.assertTrue(sequences)
        self.assertTrue(
            any(
                sequence[0][0] == 80.0
                and sequence[1][0] == 60.0
                and sequence[2][0] == 50.0
                for sequence in sequences
            ),
            "Wu-faithful VSL sequence search did not include 100->80->60->50 prevention path",
        )
        for sequence in sequences:
            for vec in sequence:
                self.assertTrue(all(value in vsl_set for value in vec))
                self.assertTrue(
                    all(
                        value == vsl_max
                        for index, value in enumerate(vec)
                        if index not in upstream_control_idx
                    )
                )
            for before, after in zip(sequence, sequence[1:]):
                self.assertTrue(
                    all(
                        abs(after[i] - before[i]) <= cfg.freeway_follower.max_vsl_step + 1.0e-9
                        for i in range(len(after))
                    )
                )

    def test_wu_faithful_freeway_release_uses_start_reservoir_before_current_arrivals(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {
                    "T_total": 10.0,
                    "T_f": 10.0,
                    "control_interval": 10.0,
                },
                "mpc": {"horizon_steps": 1},
                "freeway_follower": {
                    "freeway_prediction_horizon_steps": 1,
                    "vsl_sequence_search": False,
                },
            },
        )
        follower = WuFaithfulFollower(cfg)
        link = next(
            link
            for link, model in follower._local_freeway_models.items()
            if model.owned_ramps
        )
        model = follower._local_freeway_models[link]
        ramp = model.owned_ramps[0]
        state = TrafficState.initial(cfg)
        state.ramp_queue[ramp] = 0.0
        previous = ControlAction.fixed(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test", ramp_scale=0.0)).at(0.0)
        coupling = {f"u_on_{ramp}": 3600.0}
        release_seen_queue: list[dict[str, float]] = []

        def fake_local_ramp_release(
            link_arg,
            rhos,
            ramp_queue,
            candidate_control,
            demand_step,
        ):
            del rhos, candidate_control, demand_step
            if link_arg == link:
                release_seen_queue.append(dict(ramp_queue))
            return {owned: 0.0 for owned in follower._local_freeway_models[link_arg].owned_ramps}

        def fake_freeway_substep_local(
            local_model,
            rhos,
            speeds,
            prev_lanes,
            occ,
            origin_q,
            ramp_release,
            offramp_capacity,
            candidate_control,
            demand_step,
        ):
            del occ, ramp_release, offramp_capacity, candidate_control, demand_step
            return (
                list(rhos),
                list(speeds),
                list(prev_lanes),
                origin_q,
                {off_ramp: 0.0 for off_ramp in local_model.owned_offramps},
                [0.0 for _ in range(local_model.n_seg)],
            )

        # Spec 3.4.3: current urban→ramp arrival은 같은 T_f의 release 결정 이후
        # reservoir에 적재되어야 한다. 버그가 있으면 여기서 3600 veh/h * 10 s = 10 veh가
        # release 계산 전에 보인다.
        with patch.object(
            follower,
            "_freeway_vsl_sequence_candidates",
            return_value=[[[100.0 for _ in range(model.n_seg)]]],
        ):
            with patch.object(follower, "_local_ramp_release", side_effect=fake_local_ramp_release):
                with patch(
                    "src.controllers.wu_faithful_follower.freeway_substep_local",
                    side_effect=fake_freeway_substep_local,
                ):
                    follower._solve_freeway_agent_local(
                        link,
                        state,
                        coupling,
                        demand,
                        previous,
                    )

        self.assertTrue(release_seen_queue)
        self.assertAlmostEqual(release_seen_queue[0][ramp], 0.0, places=6)

    def test_wu_faithful_np_target_projects_to_signed_feasible_range(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
            },
        )
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("urban_med", urban_scale=1.15, freeway_scale=1.0, ramp_scale=1.15),
        ).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)

        for target in (9999.0, -9999.0):
            result = WuFaithfulFollower(cfg).solve(
                state.copy(),
                LeaderAction(target, 1000.0),
                forecast,
                previous,
            )
            diagnostics = result.diagnostics
            projected = float(diagnostics["wu_faithful_np_projected_target_veh"])
            feasible_min = float(diagnostics["wu_faithful_np_feasible_min_veh"])
            feasible_max = float(diagnostics["wu_faithful_np_feasible_max_veh"])
            sum_nin = float(diagnostics["wu_faithful_sum_nin"])

            self.assertLessEqual(feasible_min - 1.0e-9, projected)
            self.assertLessEqual(projected, feasible_max + 1.0e-9)
            self.assertLessEqual(abs(sum_nin - projected), 1.0)
            self.assertEqual(result.control.N_P_star, target)
            self.assertEqual(
                float(diagnostics["urban_net_inflow_target_veh"]),
                projected,
            )
            self.assertNotEqual(
                float(diagnostics["urban_net_inflow_original_target_veh"]),
                projected,
            )

    def test_wu_faithful_np_predictor_modes_project_in_vehicle_units(self):
        for mode, code in (
            ("storage_aware", 1.0),
            ("current_interval", 2.0),
            ("phase_substep", 3.0),
        ):
            cfg = ExperimentConfig.from_file(
                "src/config/default.yaml",
                {
                    "simulation": {"T_total": 180.0},
                    "mpc": {
                        "horizon_steps": 3,
                        "max_nash_iter": 1,
                        "wu_faithful_np_predictor_mode": mode,
                    },
                },
            )
            state = TrafficState.initial(cfg)
            forecast = DemandProfile(
                cfg,
                ScenarioConfig("urban_med", urban_scale=1.15, freeway_scale=1.0, ramp_scale=1.15),
            ).horizon(0.0, 3)
            result = WuFaithfulFollower(cfg).solve(
                state.copy(),
                LeaderAction(9999.0, 1000.0),
                forecast,
                ControlAction.fixed(cfg),
            )
            diagnostics = result.diagnostics
            projected = float(diagnostics["wu_faithful_np_projected_target_veh"])
            sum_nin = float(diagnostics["wu_faithful_sum_nin"])

            self.assertAlmostEqual(
                float(diagnostics["wu_faithful_np_predictor_mode_code"]),
                code,
            )
            self.assertAlmostEqual(
                float(diagnostics[f"wu_faithful_np_predictor_{mode}"]),
                1.0,
            )
            self.assertLessEqual(
                float(diagnostics["wu_faithful_np_feasible_min_veh"]) - 1.0e-9,
                projected,
            )
            self.assertLessEqual(
                projected,
                float(diagnostics["wu_faithful_np_feasible_max_veh"]) + 1.0e-9,
            )
            self.assertLessEqual(abs(sum_nin - projected), 2.0)
            self.assertAlmostEqual(
                float(diagnostics["urban_net_inflow_target_veh"]),
                projected,
            )

    def test_wu_faithful_storage_predictor_caps_blocked_receiving_space(self):
        overrides = {
            "simulation": {"T_total": 180.0},
            "mpc": {"horizon_steps": 3, "max_nash_iter": 1},
        }
        cfg_legacy = ExperimentConfig.from_file("src/config/default.yaml", overrides)
        cfg_storage = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                **overrides,
                "mpc": {
                    **overrides["mpc"],
                    "wu_faithful_np_predictor_mode": "storage_aware",
                },
            },
        )
        state_legacy = TrafficState.initial(cfg_legacy)
        state_storage = TrafficState.initial(cfg_storage)
        for link in state_storage.urban_link_storage:
            state_storage.urban_link_storage[link] = 0.0
        forecast = DemandProfile(
            cfg_legacy,
            ScenarioConfig("urban_med", urban_scale=1.15, freeway_scale=1.0, ramp_scale=1.15),
        ).horizon(0.0, 3)
        horizon_h = cfg_legacy.simulation.T_c_h * 3
        legacy = WuFaithfulFollower(cfg_legacy)
        storage = WuFaithfulFollower(cfg_storage)
        legacy_arrivals = legacy._movement_forecast_arrivals_veh(forecast)
        storage_arrivals = storage._movement_forecast_arrivals_veh(forecast)

        _, legacy_max, _ = legacy._np_feasible_sum_range(
            state_legacy, legacy_arrivals, horizon_h,
        )
        _, storage_max, _ = storage._np_feasible_sum_range(
            state_storage, storage_arrivals, horizon_h,
        )

        self.assertLess(storage_max, legacy_max)

    def test_freeway_follower_handles_capacity_drop_with_valid_vsl(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {"horizon_steps": 3},
                "network": {
                    "off_ramp_split_ratio": {
                        "OR_D_W": 0.25,
                        "OR_F_W": 0.25,
                        "OR_D_E": 0.25,
                        "OR_F_E": 0.25,
                    }
                },
                "freeway_offramp_capacity_drop": {
                    "enabled": True,
                    "lane_reduction": 0.75,
                    "gamma": 0.2,
                    "b": 2.0,
                },
                "freeway_follower": {
                    "vsl_smoothness_weight": 0.0,
                    "horizon_beam_width": 4,
                    "horizon_vsl_candidate_limit_per_link": 3,
                },
            },
        )
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [28.0, 28.0, 34.0, 45.0]
            state.freeway_speed[link] = [90.0, 90.0, 75.0, 45.0]
            state.freeway_effective_lanes[link] = [2.0, 2.0, 2.0, 2.0]
        for storage_link in cfg.network.off_ramp_storage_link.values():
            state.urban_link_storage[storage_link] = 0.0

        _, lane_diag = effective_lane_profile(state, cfg)
        self.assertEqual(lane_diag["capacity_drop_active"], 1.0)
        self.assertLess(
            min(
                lane_diag[f"lambda_eff_FW_W_seg{i}"]
                for i in range(len(state.freeway_density["FW_W"]))
            ),
            cfg.network.freeway_lanes,
        )

        demand = DemandProfile(
            cfg,
            ScenarioConfig(
                "forced_capacity_drop",
                urban_scale=0.0,
                freeway_scale=1.4,
                ramp_scale=0.8,
                incident_capacity_factor=1.0,
            ),
        ).horizon(0.0, cfg.mpc.horizon_steps)
        result = FreewayFollower(cfg).solve(
            state,
            LeaderAction(0.0, 0.0),
            demand,
            ControlAction.fixed(cfg),
        )

        self.assertTrue(result.infeasibility["freeway_follower_sequence_optimized"])
        self.assertTrue(all(
            float(value) in {float(v) for v in cfg.freeway_follower.vsl_set}
            for value in result.vsl.values()
        ))

    def test_boundary_queue_balance_safe_division(self):
        from src.models.urban_queue_model import safe_balance_index

        self.assertEqual(safe_balance_index([0.0, 0.0]), 0.0)

    def test_boundary_balance_gate_uses_movement_level_b_not_cv(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"evaluation": {"eps_balance": 0.03}, "urban_follower": {"eps_U": 100.0}},
        )
        baseline_state = TrafficState.initial(cfg)
        proposed_state = TrafficState.initial(cfg)
        for idx, link in enumerate(cfg.network.movement_links):
            proposed_state.boundary_queue[link] = 120.0 if idx == 0 else 10.0
        for movement, spec in cfg.network.urban_movements.items():
            if spec.get("kind") in {"boundary_in", "off_ramp", "boundary_out", "on_ramp"}:
                proposed_state.urban_movement_queue[movement] = 0.5 * movement_storage_capacity(cfg, movement, spec)
        validation = validate_controls(
            {"final_state": baseline_state, "run_rows": [], "control_rows": []},
            {"final_state": proposed_state, "run_rows": [], "control_rows": []},
            cfg,
        )
        self.assertGreater(validation["boundary_balance"]["CV_boundary"], 0.0)
        self.assertEqual(validation["boundary_balance"]["boundary_balance_degenerate"], 0.0)
        self.assertTrue(validation["boundary_balance"]["pass"])

    def test_degenerate_boundary_balance_does_not_trivially_pass(self):
        cfg = ExperimentConfig.from_file("src/config/default.yaml", {"evaluation": {"eps_balance": 0.03}})
        baseline_state = TrafficState.initial(cfg)
        proposed_state = TrafficState.initial(cfg)
        for movement in proposed_state.urban_movement_queue:
            proposed_state.urban_movement_queue[movement] = 0.0
        validation = validate_controls(
            {"final_state": baseline_state, "run_rows": [], "control_rows": []},
            {"final_state": proposed_state, "run_rows": [], "control_rows": []},
            cfg,
        )
        self.assertEqual(validation["boundary_balance"]["B_in"], 0.0)
        self.assertEqual(validation["boundary_balance"]["B_out"], 0.0)
        self.assertEqual(validation["boundary_balance"]["boundary_balance_degenerate"], 1.0)
        self.assertFalse(validation["boundary_balance"]["pass"])

    def test_no_negative_density_speed_queue(self):
        cfg = short_config()
        sim = MixedTrafficSimulator(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        sim.step(ControlAction.fixed(cfg), demand, 0)
        for values in sim.state.freeway_density.values():
            self.assertTrue(all(v >= 0.0 for v in values))
        for values in sim.state.freeway_speed.values():
            self.assertTrue(all(v >= 0.0 for v in values))
        self.assertTrue(all(v >= 0.0 for v in sim.state.ramp_queue.values()))
        self.assertTrue(all(v >= 0.0 for v in sim.state.boundary_queue.values()))
        self.assertTrue(all(v >= 0.0 for v in sim.state.urban_movement_queue.values()))
        self.assertTrue(all(v >= 0.0 for v in sim.state.urban_link_storage.values()))

    def test_simulator_uses_coupling_module_diagnostics(self):
        cfg = short_config()
        sim = MixedTrafficSimulator(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        log = sim.step(ControlAction.fixed(cfg), demand, 0)
        self.assertEqual(log.diagnostics["coupling_freeway_substeps"], float(cfg.simulation.K_cf))
        self.assertEqual(log.diagnostics["coupling_urban_substeps"], float(cfg.simulation.K_cu))
        self.assertEqual(log.diagnostics["coupling_nested_order_active"], 1.0)
        self.assertEqual(log.diagnostics["coupling_aggregate_urban_model"], 0.0)
        self.assertEqual(log.diagnostics["coupling_movement_urban_model"], 1.0)
        self.assertEqual(log.diagnostics["coupling_onramp_sync_active"], 1.0)
        self.assertEqual(log.diagnostics["coupling_onramp_two_reservoir_active"], 1.0)
        self.assertEqual(log.diagnostics["coupling_offramp_storage_active"], 1.0)

    def test_onramp_uses_two_reservoirs_instead_of_syncing_queues(self):
        cfg = short_config()
        state = TrafficState.initial(cfg)
        state.ramp_queue["R_D_W"] = 30.0
        state.urban_movement_queue["D_N_to_onW"] = 70.0
        sync_onramp_queues_from_freeway(state, cfg)
        self.assertAlmostEqual(state.ramp_queue["R_D_W"], 30.0)
        self.assertAlmostEqual(state.urban_movement_queue["D_N_to_onW"], 70.0)
        sync_onramp_queues_to_freeway(state, cfg)
        self.assertAlmostEqual(state.ramp_queue["R_D_W"], 30.0)
        self.assertAlmostEqual(state.urban_movement_queue["D_N_to_onW"], 70.0)

    def test_onramp_demand_enters_urban_movement_queue_when_metering_closed(self):
        cfg = short_config()
        sim = MixedTrafficSimulator(cfg)
        control = ControlAction.fixed(cfg)
        control.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        demand = DemandProfile(cfg, ScenarioConfig("test", ramp_scale=2.0)).at(0.0)
        log = sim.step(control, demand, 0)
        self.assertGreater(log.diagnostics["onramp_arrivals_veh"], 0.0)
        self.assertAlmostEqual(log.diagnostics["ramp_metering_releases_veh"], 0.0)
        self.assertGreater(log.diagnostics["onramp_green_releases_veh"], 0.0)
        self.assertGreater(
            log.diagnostics["onramp_approach_queue_veh"] + log.diagnostics["ramp_queue_veh"],
            0.0,
        )

    def test_onramp_green_controls_approach_release_to_ramp_queue(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0}},
        )
        demand = DemandProfile(cfg, ScenarioConfig("test", ramp_scale=0.0)).at(0.0)
        low = TrafficState.initial(cfg)
        high = TrafficState.initial(cfg)
        for state in (low, high):
            for ramp in cfg.network.ramps:
                state.ramp_queue[ramp] = 0.0
            for movements in cfg.network.on_ramp_to_movement.values():
                for movement in movements:
                    state.urban_movement_queue[movement] = 40.0

        low_control = ControlAction.fixed(cfg)
        high_control = ControlAction.fixed(cfg)
        # on_ramp 행 movement는 incoming approach 축으로 phase가 갈린다 — 양 phase 모두 조인다.
        for phase in ("D_p1", "D_p2"):
            low_control.green_times[phase] = cfg.network.green_min
            high_control.green_times[phase] = cfg.network.green_max
        for movement in cfg.network.on_ramp_to_movement["R_D_W"]:
            low_control.inflow_outflow_allocation[movement] = cfg.network.movement_capacity_veh_h
            high_control.inflow_outflow_allocation[movement] = cfg.network.movement_capacity_veh_h
        ramp_release = {ramp: 0.0 for ramp in cfg.network.ramps}

        # cycle 위상 plant에서는 substep별 green이 이진(window)이므로 한 cycle을
        # 누적해 비교한다(green이 길수록 cycle당 방출이 커야 한다).
        cycle_steps = int(cfg.network.cycle_length / cfg.simulation.T_u_sec)
        low_release = 0.0
        high_release = 0.0
        for step in range(cycle_steps):
            _, low_diag = urban_substep(low, low_control, demand, cfg, urban_step_index=step, ramp_release_veh_h=ramp_release)
            _, high_diag = urban_substep(high, high_control, demand, cfg, urban_step_index=step, ramp_release_veh_h=ramp_release)
            low_release += low_diag["onramp_green_releases_veh"]
            high_release += high_diag["onramp_green_releases_veh"]

        self.assertGreater(high.ramp_queue["R_D_W"], low.ramp_queue["R_D_W"])
        self.assertGreater(high_release, low_release)

    def test_coupling_passes_actual_ramp_release_to_freeway_step(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0}},
        )
        state = TrafficState.initial(cfg)
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 0.0
        # 게이트→ramp 직결 movement(β 자연 분산)도 w_r를 채우므로 ramp행 큐를 전부 비운다.
        for movement, spec in cfg.network.urban_movements.items():
            if spec.get("ramp"):
                state.urban_movement_queue[movement] = 0.0
        control = ControlAction.fixed(cfg)
        # urban 게이트 수요가 같은 interval 안에서 ramp로 넘어가지 않게 urban_scale=0.
        demand = DemandProfile(cfg, ScenarioConfig("test", urban_scale=0.0, ramp_scale=0.0)).at(0.0)
        requested_release = {ramp: 1000.0 for ramp in cfg.network.ramps}
        seen_release = []

        def fake_compute_release(*_args, **_kwargs):
            return requested_release, {
                "total_metering_flow": sum(requested_release.values()),
                "total_no_meter_flow": sum(requested_release.values()),
                "mean_ramp_receiving_factor": 1.0,
            }

        def fake_freeway_substep(*_args, **kwargs):
            actual = dict(kwargs["ramp_release_veh_h"])
            seen_release.append(actual)
            return 0.0, {
                "total_metering_flow": sum(actual.values()),
                "total_metering_error": 0.0,
                "mean_ramp_receiving_factor": 1.0,
                "offramp_flow_total": 0.0,
                "offramp_blocked_flow_total": 0.0,
            }

        with patch("src.simulation.coupling.compute_ramp_release_flows", side_effect=fake_compute_release):
            with patch("src.simulation.coupling.freeway_substep", side_effect=fake_freeway_substep):
                result = run_coupled_interval(state, control, demand, cfg)

        self.assertTrue(seen_release)
        self.assertTrue(all(value == 0.0 for value in seen_release[0].values()))
        self.assertGreater(result.diagnostics["ramp_metering_release_shortfall_veh"], 0.0)

    def test_offramp_storage_limits_freeway_boundary_flow(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "network": {
                    "urban_link_storage_veh": {
                        "OR_D_W_storage": 0.0,
                        "OR_F_W_storage": 0.0,
                        "OR_D_E_storage": 0.0,
                        "OR_F_E_storage": 0.0,
                    }
                },
            },
        )
        sim = MixedTrafficSimulator(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        log = sim.step(ControlAction.fixed(cfg), demand, 0)
        self.assertGreater(log.diagnostics["offramp_storage_binding"], 0.0)
        self.assertGreater(log.diagnostics["offramp_blocked_flow_total"], 0.0)

    def test_stackelberg_prediction_uses_coupling_module(self):
        cfg = short_config()
        controller = StackelbergMPCController(cfg)
        state = TrafficState.initial(cfg)
        control = ControlAction.fixed(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        calls = []

        def fake_coupled_step(*args):
            calls.append(args)
            return CoupledStepResult(freeway_ttt=1.25, urban_ttt=2.75)

        with patch("src.simulation.coupling.run_coupled_interval", side_effect=fake_coupled_step):
            states, total_ttt = controller._predict(state, control, demand)

        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(total_ttt, 4.0)
        self.assertAlmostEqual(states[0].time_sec, cfg.simulation.control_interval)

    def test_stackelberg_default_objective_uses_follower_response_with_future_penalty_states(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 2, "leader_candidate_count": 2, "max_nash_iter": 1},
                "leader": {"objective_mode": "follower_ttt", "w_P": 0.0, "w_F": 0.0, "w_L": 0.0},
                "freeway_follower": {
                    "horizon_beam_width": 1,
                    "horizon_ramp_candidate_limit": 1,
                    "horizon_vsl_candidate_limit_per_link": 1,
                },
            },
        )
        controller = StackelbergMPCController(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 2)
        nash = NashResult(
            control=ControlAction.fixed(cfg),
            objective_value=1234.0,
            iterations=1,
            converged=True,
            residual_objective=0.0,
            residual_control=0.0,
            diagnostics={},
        )

        states, follower_ttt, rollout_used = controller._leader_evaluation_base(state, nash, demand)

        self.assertTrue(rollout_used)
        self.assertEqual(len(states), 2)
        self.assertGreater(states[-1].time_sec, state.time_sec)
        self.assertAlmostEqual(follower_ttt, nash.objective_value)

    def test_distributed_follower_does_not_mutate_previous_control(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
                "freeway_follower": {
                    "horizon_beam_width": 1,
                    "horizon_ramp_candidate_limit": 1,
                    "horizon_vsl_candidate_limit_per_link": 1,
                },
            },
        )
        coordinator = DistributedCoordinator(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)
        previous.N_P_star = 111.0
        previous.N_UF_star = 222.0
        previous.green_times["A_p1"] = 12.0
        before_vector = previous.control_vector(cfg)

        coordinator.solve(state, LeaderAction(333.0, 4444.0), demand, previous)

        self.assertEqual(previous.N_P_star, 111.0)
        self.assertEqual(previous.N_UF_star, 222.0)
        self.assertEqual(previous.green_times["A_p1"], 12.0)
        self.assertEqual(previous.control_vector(cfg), before_vector)

    def test_two_block_nash_solver_does_not_mutate_previous_control(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
                "freeway_follower": {
                    "horizon_beam_width": 1,
                    "horizon_ramp_candidate_limit": 1,
                    "horizon_vsl_candidate_limit_per_link": 1,
                },
            },
        )
        from src.controllers.nash_solver import NashSolver

        solver = NashSolver(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)
        previous.N_P_star = 111.0
        previous.N_UF_star = 222.0
        before_vector = previous.control_vector(cfg)

        solver.solve(state, LeaderAction(333.0, 4444.0), demand, previous)

        self.assertEqual(previous.N_P_star, 111.0)
        self.assertEqual(previous.N_UF_star, 222.0)
        self.assertEqual(previous.control_vector(cfg), before_vector)

    def test_freeway_follower_prediction_preserves_urban_control_context(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1},
                "freeway_follower": {"vsl_set": [100], "max_vsl_step": 0.0},
            },
        )
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)
        previous.green_times["A_p1"] = cfg.network.green_min
        previous.green_times["A_p2"] = cfg.network.effective_green_total - cfg.network.green_min
        seen_green = []

        def fake_lightweight_transition(*args):
            state_arg = args[-3]
            control = args[-2]
            seen_green.append(dict(control.green_times))
            return (
                state_arg.copy(),
                1.0,
                {"total_metering_error": 0.0, "mean_ramp_receiving_factor": 1.0},
            )

        with patch(
            "src.controllers.freeway_follower.FreewayFollower._lightweight_transition",
            side_effect=fake_lightweight_transition,
        ):
            result = FreewayFollower(cfg).solve(state, LeaderAction(0.0, 1200.0), demand, previous)

        self.assertEqual(result.infeasibility["freeway_follower_coupled_prediction"], 0.0)
        self.assertEqual(result.infeasibility["freeway_follower_lightweight_prediction"], 1.0)
        self.assertTrue(seen_green)
        self.assertEqual(seen_green[0]["A_p1"], previous.green_times["A_p1"])

    def test_urban_follower_uses_freeway_response_pressure(self):
        cfg = short_config()
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        state = TrafficState.initial(cfg)
        follower = UrbanFollower(cfg)
        leader = LeaderAction(0.0, 1200.0)
        previous = ControlAction.fixed(cfg)
        no_response = follower.solve(state.copy(), leader, demand, None, previous)
        freeway_response = FreewayFollowerResult(
            ramp_metering={},
            vsl={},
            objective_value=0.0,
            infeasibility={
                "metering_tracking_residual": 1500.0,
                "ramp_projection_first_step_capacity": 1500.0,
                "ramp_queue_overflow": cfg.network.ramp_queue_max_veh,
                "density_excess": cfg.network.rho_crit,
                "min_ramp_receiving_factor": 0.2,
            },
        )
        with_response = follower.solve(state.copy(), leader, demand, freeway_response, previous)
        outbound = [
            movement for movement, spec in cfg.network.urban_movements.items()
            if spec.get("kind") == "off_ramp"
        ]
        no_out = sum(
            min(no_response.inflow_outflow_allocation.get(movement, 0.0), cfg.network.movement_capacity_veh_h)
            * no_response.green_times[cfg.network.urban_movements[movement]["phase"]]
            / cfg.network.cycle_length
            for movement in outbound
        )
        yes_out = sum(
            min(with_response.inflow_outflow_allocation.get(movement, 0.0), cfg.network.movement_capacity_veh_h)
            * with_response.green_times[cfg.network.urban_movements[movement]["phase"]]
            / cfg.network.cycle_length
            for movement in outbound
        )
        self.assertEqual(with_response.metrics["freeway_response_used"], 1.0)
        self.assertGreater(with_response.metrics["freeway_total_pressure"], 0.0)
        self.assertGreaterEqual(yes_out, no_out)


if __name__ == "__main__":
    unittest.main()
