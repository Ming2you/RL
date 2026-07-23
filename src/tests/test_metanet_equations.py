import math
import unittest

from src.models.demand import DemandStep
from src.models.metanet import (
    desired_speed_kmh,
    effective_lane_profile,
    effective_desired_speed_kmh,
    freeway_substep,
    freeway_step,
    metanet_speed_update_kmh,
    offramp_spillback_lambda_eff,
    segment_flow_veh_h,
)
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import off_ramp_capacity_by_freeway_link, schedule_offramp_arrivals


class MetanetEquationTests(unittest.TestCase):
    def test_segment_flow_matches_spec_equation(self):
        self.assertAlmostEqual(segment_flow_veh_h(20.0, 80.0, 2.0), 3200.0)

    def test_desired_speed_matches_spec_equation(self):
        rho = 30.0
        v_free = 100.0
        rho_crit = 33.5
        a = 1.867
        expected = v_free * math.exp(-(1.0 / a) * ((rho / rho_crit) ** a))
        self.assertAlmostEqual(desired_speed_kmh(rho, v_free, rho_crit, a), expected)

    def test_vsl_effect_uses_alpha_compliance(self):
        rho = 10.0
        no_vsl = desired_speed_kmh(rho, 100.0, 33.5, 1.867)
        self.assertGreater(no_vsl, 90.0)
        self.assertAlmostEqual(
            effective_desired_speed_kmh(rho, 100.0, 33.5, 90.0, alpha_vsl=0.1),
            min(no_vsl, 99.0),
        )
        self.assertAlmostEqual(
            effective_desired_speed_kmh(rho, 100.0, 33.5, 90.0, vsl_active=False),
            no_vsl,
        )

    def test_speed_update_matches_relaxation_convection_anticipation_sum(self):
        speed = 80.0
        upstream_speed = 90.0
        rho = 25.0
        downstream_rho = 30.0
        v_eff = 85.0
        dt_h = 10.0 / 3600.0
        length_km = 0.5
        tau_h = 0.005
        nu = 65.0
        kappa = 40.0
        v_min = 5.0
        expected = (
            speed
            + dt_h / tau_h * (v_eff - speed)
            + dt_h / length_km * speed * (upstream_speed - speed)
            - nu * dt_h / (tau_h * length_km) * (downstream_rho - rho) / (rho + kappa)
        )
        self.assertAlmostEqual(
            metanet_speed_update_kmh(
                speed,
                upstream_speed,
                rho,
                downstream_rho,
                v_eff,
                dt_h,
                length_km,
                tau_h,
                nu,
                kappa,
                v_min,
            ),
            max(v_min, expected),
        )

    def test_config_exposes_time_ratios_and_units(self):
        cfg = ExperimentConfig.from_file("src/config/default.yaml")
        self.assertAlmostEqual(cfg.simulation.T_f_h, cfg.simulation.T_f / 3600.0)
        self.assertAlmostEqual(cfg.simulation.T_u_h, cfg.simulation.T_u / 3600.0)
        self.assertAlmostEqual(cfg.simulation.T_c_h, cfg.simulation.control_interval / 3600.0)
        self.assertEqual(cfg.simulation.K_fu, 2)
        self.assertEqual(cfg.simulation.K_cf, 18)
        self.assertEqual(cfg.simulation.K_cu, 36)
        self.assertEqual(cfg.leader.objective_mode, "follower_ttt")
        self.assertEqual(cfg.leader.N_P_star_unit, "veh")
        # d11cd16(2026-06-30) N_P_crit recalibration 반영 — 구 기대값 509.4488은 재보정 전.
        self.assertAlmostEqual(cfg.leader.N_P_crit_veh, 1142.058)
        self.assertLessEqual(cfg.leader.N_P_candidate_lower_factor, cfg.leader.N_P_candidate_upper_factor)
        self.assertEqual(cfg.leader.N_UF_star_unit, "veh_per_hour")
        self.assertGreaterEqual(cfg.evaluation.eps_balance, 0.0)
        self.assertGreaterEqual(cfg.evaluation.boundary_degenerate_saturation_fraction, 0.0)
        self.assertLessEqual(cfg.evaluation.boundary_degenerate_saturation_fraction, 1.0)
        self.assertGreaterEqual(cfg.freeway_follower.horizon_beam_width, 1)
        self.assertGreaterEqual(cfg.freeway_follower.horizon_ramp_candidate_limit, 1)
        self.assertGreaterEqual(cfg.freeway_follower.horizon_vsl_candidate_limit_per_link, 1)
        self.assertIn("R_D_W", cfg.network.on_ramp_to_movement)
        self.assertIn("OR_D_W", cfg.network.off_ramps)
        self.assertIn("OR_D_W_storage", cfg.network.urban_link_storage_veh)
        # 다이아몬드 인터체인지(2026-07-13 승인): D(off 2/on 3), F(off 4/on 5) —
        # merge+off 동거(구 seg4) 해소, 각 인터체인지 내 exit가 entrance 상류.
        self.assertEqual(cfg.network.freeway_segments_per_link, 8)
        self.assertEqual(cfg.network.ramp_merge_segment_index["R_D_W"], 3)
        self.assertEqual(cfg.network.ramp_merge_segment_index["R_F_W"], 5)
        self.assertEqual(cfg.network.off_ramp_segment_index["OR_D_W"], 2)
        self.assertEqual(cfg.network.off_ramp_segment_index["OR_F_W"], 4)
        self.assertEqual(cfg.network.off_ramp_segment_index["OR_D_E"], 2)
        self.assertEqual(cfg.network.off_ramp_segment_index["OR_F_E"], 4)
        self.assertTrue(cfg.freeway_offramp_capacity_drop.enabled)
        self.assertGreaterEqual(cfg.freeway_offramp_capacity_drop.lane_reduction, 0.0)
        self.assertLess(cfg.freeway_offramp_capacity_drop.lane_reduction, cfg.network.freeway_lanes)

    def test_config_rejects_non_integer_time_ratios(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file("src/config/default.yaml", {"simulation": {"T_f": 7.0}})

    def test_config_rejects_unknown_nuf_unit(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file("src/config/default.yaml", {"leader": {"N_UF_star_unit": "vehicles"}})

    def test_config_rejects_unknown_np_unit(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file("src/config/default.yaml", {"leader": {"N_P_star_unit": "veh_per_hour"}})

    def test_config_rejects_unknown_leader_objective_mode(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file("src/config/default.yaml", {"leader": {"objective_mode": "bad_mode"}})

    def test_config_rejects_invalid_np_candidate_band(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file(
                "src/config/default.yaml",
                {"leader": {"N_P_candidate_lower_factor": 1.1, "N_P_candidate_upper_factor": 0.9}},
            )

    def test_config_rejects_invalid_capacity_drop_parameters(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file(
                "src/config/default.yaml",
                {"freeway_offramp_capacity_drop": {"lane_reduction": 2.0}},
            )
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file(
                "src/config/default.yaml",
                {"freeway_offramp_capacity_drop": {"gamma": 0.0}},
            )
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file(
                "src/config/default.yaml",
                {"freeway_offramp_capacity_drop": {"b": 0.0}},
            )

    def test_config_rejects_invalid_follower_solver_mode(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_file(
                "src/config/default.yaml",
                {"mpc": {"follower_solver_mode": "centralized"}},
            )

    def test_traffic_state_stores_freeway_flow(self):
        cfg = ExperimentConfig.from_file("src/config/default.yaml")
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            for rho, speed, lane, flow in zip(
                state.freeway_density[link],
                state.freeway_speed[link],
                state.freeway_effective_lanes[link],
                state.freeway_flow[link],
            ):
                self.assertAlmostEqual(
                    flow,
                    segment_flow_veh_h(rho, speed, lane),
                )

    def test_capacity_drop_lambda_eff_boundaries(self):
        self.assertAlmostEqual(
            offramp_spillback_lambda_eff(0.0, 120.0, 2.0, 0.35, 0.5, 2.0),
            2.0,
        )
        self.assertAlmostEqual(
            offramp_spillback_lambda_eff(120.0, 120.0, 2.0, 0.35, 0.5, 2.0),
            1.65,
        )

    def test_capacity_drop_preserves_vehicle_count_when_lambda_changes(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0},
                "network": {"freeway_segments_per_link": 1},
                "freeway_offramp_capacity_drop": {"enabled": True, "lane_reduction": 0.35},
            },
        )
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [cfg.network.rho_max]
            state.freeway_speed[link] = [0.0]
            state.freeway_effective_lanes[link] = [2.0]
        for storage_link in cfg.network.off_ramp_storage_link.values():
            state.urban_link_storage[storage_link] = 0.0
        before = state.total_freeway_vehicles(cfg.network)
        demand = DemandStep(
            freeway_mainline={link: 0.0 for link in cfg.network.freeway_links},
            urban_boundary={link: 0.0 for link in cfg.network.movement_links},
            ramp_arrival={ramp: 0.0 for ramp in cfg.network.ramps},
        )
        control = ControlAction.fixed(cfg)
        control.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}

        freeway_substep(state, control, demand, cfg, include_ramp_queue_ttt=False)

        after = state.total_freeway_vehicles(cfg.network)
        self.assertAlmostEqual(after, before)
        self.assertAlmostEqual(state.freeway_effective_lanes["FW_W"][0], 1.65)
        self.assertAlmostEqual(
            state.freeway_density["FW_W"][0] * 0.5 * 1.65,
            cfg.network.rho_max * 0.5 * 2.0,
        )

    def test_capacity_drop_uses_lane_corrected_density_for_speed_update(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0},
                "network": {"freeway_segments_per_link": 1},
                "freeway_offramp_capacity_drop": {"enabled": True, "lane_reduction": 0.35},
            },
        )
        base = TrafficState.initial(cfg)
        drop = TrafficState.initial(cfg)
        for state in (base, drop):
            for link in cfg.network.freeway_links:
                state.freeway_density[link] = [20.0]
                state.freeway_speed[link] = [80.0]
                state.freeway_effective_lanes[link] = [2.0]
        for storage_link in cfg.network.off_ramp_storage_link.values():
            drop.urban_link_storage[storage_link] = 0.0

        lane_profile, _ = effective_lane_profile(drop, cfg)
        rho_drop = 20.0 * 2.0 / lane_profile["FW_W"][0]
        self.assertGreater(rho_drop, 20.0)
        self.assertLess(
            desired_speed_kmh(rho_drop, cfg.network.v_free, cfg.network.rho_crit, cfg.network.metanet_a_m),
            desired_speed_kmh(20.0, cfg.network.v_free, cfg.network.rho_crit, cfg.network.metanet_a_m),
        )

        demand = DemandStep(
            freeway_mainline={link: segment_flow_veh_h(20.0, 80.0, 2.0) for link in cfg.network.freeway_links},
            urban_boundary={link: 0.0 for link in cfg.network.movement_links},
            ramp_arrival={ramp: 0.0 for ramp in cfg.network.ramps},
        )
        control = ControlAction.fixed(cfg)
        control.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        freeway_substep(base, control, demand, cfg, include_ramp_queue_ttt=False)
        freeway_substep(drop, control, demand, cfg, include_ramp_queue_ttt=False)

        self.assertLess(drop.freeway_speed["FW_W"][0], base.freeway_speed["FW_W"][0])

    def test_capacity_drop_uses_configured_offramp_segment(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0},
                "freeway_offramp_capacity_drop": {"enabled": True, "lane_reduction": 0.35},
            },
        )
        state = TrafficState.initial(cfg)
        for storage_link in cfg.network.off_ramp_storage_link.values():
            state.urban_link_storage[storage_link] = cfg.network.urban_link_storage_veh[storage_link]
        state.urban_link_storage["OR_D_W_storage"] = 0.0

        lane_profile, diag = effective_lane_profile(state, cfg)

        idx = cfg.network.off_ramp_segment_index["OR_D_W"]
        for i in range(cfg.network.freeway_segments_per_link):
            if i == idx:
                self.assertLess(lane_profile["FW_W"][i], cfg.network.freeway_lanes)
            else:
                self.assertAlmostEqual(lane_profile["FW_W"][i], cfg.network.freeway_lanes)
        self.assertAlmostEqual(diag["capacity_drop_lane_loss_FW_W_seg0"], 0.0)
        self.assertGreater(diag[f"capacity_drop_lane_loss_FW_W_seg{idx}"], 0.0)

    def test_offramp_flow_uses_segment_specific_storage_capacity(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0}},
        )
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [0.0 for _ in range(cfg.network.freeway_segments_per_link)]
            state.freeway_speed[link] = [80.0 for _ in range(cfg.network.freeway_segments_per_link)]
            state.freeway_effective_lanes[link] = [
                cfg.network.freeway_lanes for _ in range(cfg.network.freeway_segments_per_link)
            ]
        dens = [0.0] * cfg.network.freeway_segments_per_link
        dens[cfg.network.off_ramp_segment_index["OR_D_W"]] = 20.0
        dens[cfg.network.off_ramp_segment_index["OR_F_W"]] = 20.0
        state.freeway_density["FW_W"] = dens
        for storage_link in cfg.network.off_ramp_storage_link.values():
            state.urban_link_storage[storage_link] = cfg.network.urban_link_storage_veh[storage_link]
        state.urban_link_storage["OR_D_W_storage"] = 0.0

        demand = DemandStep(
            freeway_mainline={link: 0.0 for link in cfg.network.freeway_links},
            urban_boundary={link: 0.0 for link in cfg.network.movement_links},
            ramp_arrival={ramp: 0.0 for ramp in cfg.network.ramps},
        )
        control = ControlAction.fixed(cfg)
        control.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        capacity = off_ramp_capacity_by_freeway_link(state, cfg, interval_h=cfg.simulation.T_f_h)

        _, diag = freeway_substep(
            state,
            control,
            demand,
            cfg,
            offramp_capacity_veh_h=capacity,
            include_ramp_queue_ttt=False,
        )

        self.assertAlmostEqual(diag["offramp_flow_OR_D_W"], 0.0)
        self.assertGreater(diag["offramp_blocked_flow_OR_D_W"], 0.0)
        self.assertGreater(diag["offramp_flow_OR_F_W"], 0.0)
        self.assertAlmostEqual(
            diag["offramp_flow_FW_W"],
            diag["offramp_flow_OR_D_W"] + diag["offramp_flow_OR_F_W"],
        )
        _, rejected_d = schedule_offramp_arrivals(
            state,
            cfg,
            "OR_D_W",
            diag["offramp_flow_OR_D_W"] * cfg.simulation.T_f_h,
            1,
        )
        _, rejected_f = schedule_offramp_arrivals(
            state,
            cfg,
            "OR_F_W",
            diag["offramp_flow_OR_F_W"] * cfg.simulation.T_f_h,
            1,
        )
        self.assertAlmostEqual(rejected_d, 0.0)
        self.assertAlmostEqual(rejected_f, 0.0)

    def test_vsl_changes_speed_mechanism_under_spillback_density(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0},
                "network": {"freeway_segments_per_link": 1},
                "freeway_offramp_capacity_drop": {"enabled": True, "lane_reduction": 0.35},
            },
        )
        no_vsl = TrafficState.initial(cfg)
        with_vsl = TrafficState.initial(cfg)
        for state in (no_vsl, with_vsl):
            for link in cfg.network.freeway_links:
                state.freeway_density[link] = [20.0]
                state.freeway_speed[link] = [90.0]
                state.freeway_effective_lanes[link] = [2.0]
            for storage_link in cfg.network.off_ramp_storage_link.values():
                state.urban_link_storage[storage_link] = 0.0

        lane_profile, _ = effective_lane_profile(with_vsl, cfg)
        rho_for_flow = 20.0 * 2.0 / lane_profile["FW_W"][0]
        self.assertLess(
            effective_desired_speed_kmh(rho_for_flow, 100.0, 33.5, 60.0, alpha_vsl=0.0),
            effective_desired_speed_kmh(rho_for_flow, 100.0, 33.5, 100.0, alpha_vsl=0.0),
        )

        demand = DemandStep(
            freeway_mainline={link: segment_flow_veh_h(20.0, 90.0, 2.0) for link in cfg.network.freeway_links},
            urban_boundary={link: 0.0 for link in cfg.network.movement_links},
            ramp_arrival={ramp: 0.0 for ramp in cfg.network.ramps},
        )
        control_100 = ControlAction.fixed(cfg)
        control_60 = ControlAction.fixed(cfg)
        control_60.vsl = {link: 60.0 for link in cfg.network.freeway_links}
        for control in (control_100, control_60):
            control.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}

        freeway_substep(no_vsl, control_100, demand, cfg, include_ramp_queue_ttt=False)
        freeway_substep(with_vsl, control_60, demand, cfg, include_ramp_queue_ttt=False)

        self.assertLess(with_vsl.freeway_speed["FW_W"][0], no_vsl.freeway_speed["FW_W"][0])

    def test_freeway_step_density_uses_metanet_conservation(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0},
                "network": {"freeway_segments_per_link": 1},
            },
        )
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [20.0]
            state.freeway_speed[link] = [50.0]
        demand = DemandStep(
            freeway_mainline={link: 1000.0 for link in cfg.network.freeway_links},
            urban_boundary={link: 0.0 for link in cfg.network.movement_links},
            ramp_arrival={ramp: 0.0 for ramp in cfg.network.ramps},
        )
        control = ControlAction.fixed(cfg)
        control.ramp_metering = {ramp: 0.0 for ramp in cfg.network.ramps}
        freeway_step(state, control, demand, cfg)
        expected = 20.0 + cfg.simulation.T_f_h / (
            cfg.network.freeway_segment_length_km * cfg.network.freeway_lanes
        ) * (1000.0 - segment_flow_veh_h(20.0, 50.0, cfg.network.freeway_lanes))
        self.assertAlmostEqual(state.freeway_density["FW_W"][0], expected)
        self.assertAlmostEqual(
            state.freeway_flow["FW_W"][0],
            segment_flow_veh_h(
                state.freeway_density["FW_W"][0],
                state.freeway_speed["FW_W"][0],
                cfg.network.freeway_lanes,
            ),
        )

    def test_freeway_step_adds_full_ramp_flow_to_merge_segment(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"simulation": {"T_total": 10.0, "T_f": 10.0, "T_u": 5.0, "control_interval": 10.0}},
        )
        state = TrafficState.initial(cfg)
        for link in cfg.network.freeway_links:
            state.freeway_density[link] = [
                10.0 for _ in range(cfg.network.freeway_segments_per_link)
            ]
            state.freeway_speed[link] = [
                0.0 for _ in range(cfg.network.freeway_segments_per_link)
            ]
        demand = DemandStep(
            freeway_mainline={link: 0.0 for link in cfg.network.freeway_links},
            urban_boundary={link: 0.0 for link in cfg.network.movement_links},
            ramp_arrival={ramp: (900.0 if ramp == "R_D_W" else 0.0) for ramp in cfg.network.ramps},
        )
        control = ControlAction.fixed(cfg)
        control.ramp_metering = {ramp: (900.0 if ramp == "R_D_W" else 0.0) for ramp in cfg.network.ramps}
        control.N_UF_star = 900.0
        freeway_step(state, control, demand, cfg)
        expected_merge_density = 10.0 + cfg.simulation.T_f_h / (
            cfg.network.freeway_segment_length_km * cfg.network.freeway_lanes
        ) * 900.0
        merge_idx = cfg.network.ramp_merge_segment_index["R_D_W"]
        self.assertAlmostEqual(state.freeway_density["FW_W"][merge_idx], expected_merge_density)
        for i, rho in enumerate(state.freeway_density["FW_W"]):
            if i != merge_idx:
                self.assertAlmostEqual(rho, 10.0)


if __name__ == "__main__":
    unittest.main()
