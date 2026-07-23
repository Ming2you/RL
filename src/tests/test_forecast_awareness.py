# 분산 follower forecast-awareness(진단 문서 2026-06-17) 검증 단위테스트
import copy
import unittest

from src.controllers.distributed_coordinator import DistributedCoordinator
from src.controllers.freeway_follower import FreewayFollowerResult
from src.controllers.leader import LeaderAction
from src.controllers.stackelberg_mpc import StackelbergMPCController
from src.controllers.urban_follower import UrbanFollower
from src.models.demand import DemandProfile, DemandStep, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def _scale_future_demand(forecast: list[DemandStep], factor: float) -> list[DemandStep]:
    """forecast[0]은 그대로 두고 미래 스텝(forecast[1:])의 모든 수요만 factor배 한다.

    현재 상태·현재 스텝 수요는 고정한 채 '미래 도착'만 바꾸므로, 결과가 변하면
    follower가 forecast[0] 외 미래 스텝을 실제로 사용한다는 증거가 된다."""
    out = [forecast[0]]
    for step in forecast[1:]:
        out.append(DemandStep(
            freeway_mainline={k: v * factor for k, v in step.freeway_mainline.items()},
            urban_boundary={k: v * factor for k, v in step.urban_boundary.items()},
            ramp_arrival={k: v * factor for k, v in step.ramp_arrival.items()},
            incident_capacity_factor=step.incident_capacity_factor,
        ))
    return out


class ForecastAwarenessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig.from_file("src/config/default.yaml")
        self.net = self.cfg.network
        self.profile = DemandProfile(
            self.cfg, ScenarioConfig(name="peak", urban_scale=2.0, freeway_scale=1.5, ramp_scale=1.5)
        )
        self.forecast = self.profile.horizon(0.0, max(3, self.cfg.mpc.horizon_steps))
        self.leader = LeaderAction(N_P_star=float(self.cfg.leader.N_P_crit_veh), N_UF_star=1000.0)

    # ---------- (a) 미래 스텝을 실제로 사용 ----------
    def test_urban_green_uses_future_arrivals(self):
        """현재 큐 고정, 미래 phase 도착만 바꾸면 green이 변한다(forecast[0] 외 사용)."""
        state = TrafficState.initial(self.cfg)
        follower = UrbanFollower(self.cfg)
        low = follower.solve(
            state.copy(), self.leader, self.forecast[0],
            forecast=_scale_future_demand(self.forecast, 0.1),
        )
        high = follower.solve(
            state.copy(), self.leader, self.forecast[0],
            forecast=_scale_future_demand(self.forecast, 5.0),
        )
        # 미래 도착이 5배 다른데 green split이 같으면 forecast를 안 쓰는 것.
        self.assertNotEqual(low.green_times, high.green_times)

    def test_allocation_target_uses_future_offramp(self):
        """현재 N_P 고정, 미래 본선 수요(→off-ramp 외란)만 바꾸면 allocation target이 변한다."""
        state = TrafficState.initial(self.cfg)
        module = UrbanFollower(self.cfg).allocation_module
        low = module.solve(state.copy(), self.leader, _scale_future_demand(self.forecast, 0.1))
        high = module.solve(state.copy(), self.leader, _scale_future_demand(self.forecast, 5.0))
        self.assertNotEqual(
            low.target_net_inflow_veh_h, high.target_net_inflow_veh_h,
            "allocation target이 미래 off-ramp 외란 예측에 반응하지 않음",
        )

    def test_freeway_vsl_uses_future_offramp_inflow(self):
        """off-ramp storage를 backup시키고 미래 본선 수요만 바꾸면 VSL이 변한다."""
        state = TrafficState.initial(self.cfg)
        # 30% 점유의 중간 spillback 압력만 만든다. 98% 점유는 low/high가 모두 최저 후보로 포화된다.
        for storage_link in set(self.net.off_ramp_storage_link.values()):
            cap = float(self.net.urban_link_storage_veh[storage_link])
            state.urban_link_storage[storage_link] = cap * 0.70
        coord = DistributedCoordinator(self.cfg)
        prev = ControlAction.fixed(self.cfg)
        low = coord.solve(state.copy(), self.leader, _scale_future_demand(self.forecast, 0.1), prev)
        high = coord.solve(state.copy(), self.leader, _scale_future_demand(self.forecast, 5.0), prev)
        low_diag = {
            key: value for key, value in low.diagnostics.items()
            if key.endswith(("_offramp_forecast_veh", "_vsl_selected"))
        }
        high_diag = {
            key: value for key, value in high.diagnostics.items()
            if key.endswith(("_offramp_forecast_veh", "_vsl_selected"))
        }
        low_forecasts = [value for key, value in low_diag.items() if key.endswith("_offramp_forecast_veh")]
        high_forecasts = [value for key, value in high_diag.items() if key.endswith("_offramp_forecast_veh")]
        low_vsl_selected = [value for key, value in low_diag.items() if key.endswith("_vsl_selected")]
        high_vsl_selected = [value for key, value in high_diag.items() if key.endswith("_vsl_selected")]
        self.assertTrue(low_forecasts and high_forecasts)
        self.assertGreater(max(high_forecasts), max(low_forecasts))
        self.assertTrue(low_vsl_selected and high_vsl_selected)
        self.assertLess(min(high_vsl_selected), max(low_vsl_selected))
        self.assertNotEqual(
            low.control.vsl, high.control.vsl,
            f"freeway VSL이 미래 off-ramp 예측 유입에 반응하지 않음; low={low_diag}, high={high_diag}",
        )

    # ---------- (b) off-ramp backup 시 VSL이 objective 최소화로 낮아짐 ----------
    def test_freeway_vsl_lower_when_offramp_backed_up(self):
        """off-ramp가 backup하면(트리거 아님, 후보 평가로) VSL이 비backup 대비 낮거나 같다."""
        coord = DistributedCoordinator(self.cfg)
        prev = ControlAction.fixed(self.cfg)
        # 큰 미래 본선 수요 → off-ramp 예측 유입 큼.
        forecast = _scale_future_demand(self.forecast, 5.0)

        empty = TrafficState.initial(self.cfg)  # off-ramp 비어 있음.
        backed = empty.copy()
        for storage_link in set(self.net.off_ramp_storage_link.values()):
            cap = float(self.net.urban_link_storage_veh[storage_link])
            backed.urban_link_storage[storage_link] = cap * 0.02  # 거의 가득.

        vsl_empty = coord.solve(empty, self.leader, forecast, prev).control.vsl
        vsl_backed = coord.solve(backed, self.leader, forecast, prev).control.vsl
        for link in self.net.freeway_links:
            self.assertLessEqual(
                vsl_backed.get(link, 0.0), vsl_empty.get(link, 0.0) + 1.0e-6,
                f"{link}: off-ramp backup인데 VSL이 비backup보다 높다(emergence 실패)",
            )
        # 적어도 한 link에서는 backup 시 VSL이 엄격히 낮아져야 emergence가 발현한 것.
        self.assertTrue(
            any(
                vsl_backed.get(link, 0.0) < vsl_empty.get(link, 0.0) - 1.0e-6
                for link in self.net.freeway_links
            ),
            "off-ramp backup에서 어떤 link도 VSL을 낮추지 않음",
        )

    # ---------- (c) leader 후보가 forecast 요약 반영 ----------
    def test_onramp_coupling_preserves_green_difference_when_ramp_full(self):
        """urban-to-freeway u_on coupling은 ramp space cap 때문에 green 차이를 잃지 않는다."""
        coord = DistributedCoordinator(self.cfg)
        state = TrafficState.initial(self.cfg)
        demand = DemandProfile(self.cfg, ScenarioConfig("empty", ramp_scale=0.0)).at(0.0)
        ramp = "R_D_W"
        state.ramp_queue[ramp] = self.net.ramp_queue_max_veh
        for movement in self.net.on_ramp_to_movement[ramp]:
            state.urban_movement_queue[movement] = 0.0
        state.urban_movement_queue["D_N_to_onW"] = 80.0

        low = ControlAction.fixed(self.cfg)
        high = ControlAction.fixed(self.cfg)
        low.green_times["D_p1"] = self.net.green_min
        low.green_times["D_p2"] = self.net.effective_green_total - low.green_times["D_p1"]
        high.green_times["D_p1"] = self.net.green_max
        high.green_times["D_p2"] = self.net.effective_green_total - high.green_times["D_p1"]
        for movement in self.net.on_ramp_to_movement[ramp]:
            low.inflow_outflow_allocation[movement] = self.net.movement_capacity_veh_h
            high.inflow_outflow_allocation[movement] = self.net.movement_capacity_veh_h

        low_coupling = coord._extract_coupling(state.copy(), low, demand)
        high_coupling = coord._extract_coupling(state.copy(), high, demand)

        self.assertGreater(high_coupling[f"u_on_{ramp}"], low_coupling[f"u_on_{ramp}"])
        self.assertGreater(high_coupling[f"u_on_{ramp}"], 0.0)

    def test_upstream_green_release_enters_downstream_phase_coupling(self):
        """urban-to-urban arr_* coupling은 상류 green release를 하류 phase pressure로 보낸다."""
        coord = DistributedCoordinator(self.cfg)
        state = TrafficState.initial(self.cfg)
        demand = DemandProfile(self.cfg, ScenarioConfig("empty", urban_scale=0.0, ramp_scale=0.0)).at(0.0)
        d_p1_entries = coord._upstream_leaving_map["D_p1"]
        self.assertIn(("A", "A_E_to_S", 1.0), d_p1_entries)

        low = ControlAction.fixed(self.cfg)
        high = ControlAction.fixed(self.cfg)
        low.green_times["A_p2"] = self.net.green_min
        low.green_times["A_p1"] = self.net.effective_green_total - low.green_times["A_p2"]
        high.green_times["A_p2"] = self.net.green_max
        high.green_times["A_p1"] = self.net.effective_green_total - high.green_times["A_p2"]

        low_coupling = coord._extract_coupling(state.copy(), low, demand)
        high_coupling = coord._extract_coupling(state.copy(), high, demand)

        self.assertGreater(high_coupling["arr_D_p1"], low_coupling["arr_D_p1"])

    def test_urban_follower_uses_selected_offramp_arrival_response(self):
        """freeway-to-urban coupling은 선택된 off-ramp arrival를 phase forecast에 반영한다."""
        state = TrafficState.initial(self.cfg)
        follower = UrbanFollower(self.cfg)
        previous = ControlAction.fixed(self.cfg)
        freeway_response = FreewayFollowerResult(
            ramp_metering={},
            vsl={},
            objective_value=0.0,
            infeasibility={
                "offramp_predicted_arrival_OR_D_W_veh": 500.0,
                "offramp_storage_pressure_OR_D_W": 0.8,
            },
        )

        result = follower.solve(
            state.copy(),
            self.leader,
            self.forecast[0],
            freeway_response,
            previous,
            forecast=_scale_future_demand(self.forecast, 0.1),
        )

        self.assertEqual(result.metrics["freeway_selected_offramp_arrival_used"], 1.0)
        self.assertGreater(result.metrics["freeway_offramp_storage_pressure"], 0.0)
        self.assertGreater(result.green_times["D_p1"], self.net.effective_green_total / 2.0)

    def test_leader_candidates_reflect_forecast_summary(self):
        """후보 집합 차이가 아니라 forecast별 후보 평가/선택 민감도를 검증한다."""
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 60.0, "control_interval": 30.0},
                "mpc": {"horizon_steps": 2, "leader_candidate_count": 3, "max_nash_iter": 1},
                "freeway_follower": {
                    "horizon_beam_width": 1,
                    "horizon_ramp_candidate_limit": 1,
                    "horizon_vsl_candidate_limit_per_link": 1,
                },
            },
        )
        state = TrafficState.initial(cfg)
        prev = ControlAction.fixed(cfg)
        forecast = DemandProfile(
            cfg,
            ScenarioConfig(name="peak", urban_scale=2.0, freeway_scale=1.5, ramp_scale=1.5),
        ).horizon(0.0, max(3, cfg.mpc.horizon_steps))
        low_future = _scale_future_demand(forecast, 0.1)
        high_future = _scale_future_demand(forecast, 5.0)
        low_prev = ControlAction.fixed(cfg)
        high_prev = ControlAction.fixed(cfg)

        low = StackelbergMPCController(cfg).decide_with_info(state.copy(), low_future, low_prev)
        high = StackelbergMPCController(cfg).decide_with_info(state.copy(), high_future, high_prev)

        required_keys = [
            "leader_forecast_total_future_mean_veh_h",
            "leader_candidate_best_objective",
            "leader_candidate_second_objective",
            "leader_candidate_objective_spread",
            "leader_selected_N_P_star",
            "leader_selected_N_UF_star",
            "leader_search_stress_index",
            "leader_np_bound_lower",
            "leader_np_bound_upper",
        ]
        for key in required_keys:
            self.assertIn(key, low.metadata)
            self.assertIn(key, high.metadata)

        self.assertAlmostEqual(
            low.metadata["leader_forecast_total_first_veh_h"],
            high.metadata["leader_forecast_total_first_veh_h"],
        )
        self.assertGreater(
            high.metadata["leader_forecast_total_future_mean_veh_h"],
            low.metadata["leader_forecast_total_future_mean_veh_h"],
        )
        self.assertAlmostEqual(
            low.metadata["leader_candidate_best_objective"],
            low.leader_objective,
        )
        self.assertAlmostEqual(
            high.metadata["leader_candidate_best_objective"],
            high.leader_objective,
        )
        low_candidate_set = (
            round(low.metadata["N_P_min"], 6),
            round(low.metadata["N_P_max"], 6),
            round(low.metadata["N_UF_min"], 6),
            round(low.metadata["N_UF_max"], 6),
            round(low.metadata["leader_candidate_count"], 6),
        )
        high_candidate_set = (
            round(high.metadata["N_P_min"], 6),
            round(high.metadata["N_P_max"], 6),
            round(high.metadata["N_UF_min"], 6),
            round(high.metadata["N_UF_max"], 6),
            round(high.metadata["leader_candidate_count"], 6),
        )
        self.assertNotEqual(low_candidate_set, high_candidate_set)
        self.assertGreaterEqual(
            high.metadata["leader_search_stress_index"],
            low.metadata["leader_search_stress_index"],
        )
        self.assertGreaterEqual(
            high.metadata["leader_np_bound_upper"] - high.metadata["leader_np_bound_lower"],
            low.metadata["leader_np_bound_upper"] - low.metadata["leader_np_bound_lower"],
        )

        # 후보 N_UF 집합이 같아도 평가값, top-2 ranking, selected action 중 하나가 달라지면
        # leader가 forecast-sensitive 평가를 수행한다고 본다.
        low_signature = (
            round(low.metadata["leader_candidate_best_index"], 6),
            round(low.metadata["leader_candidate_second_index"], 6),
            round(low.metadata["leader_candidate_best_objective"], 6),
            round(low.metadata["leader_candidate_second_objective"], 6),
            round(low.metadata["leader_candidate_objective_spread"], 6),
            round(low.metadata["leader_selected_N_P_star"], 6),
            round(low.metadata["leader_selected_N_UF_star"], 6),
        )
        high_signature = (
            round(high.metadata["leader_candidate_best_index"], 6),
            round(high.metadata["leader_candidate_second_index"], 6),
            round(high.metadata["leader_candidate_best_objective"], 6),
            round(high.metadata["leader_candidate_second_objective"], 6),
            round(high.metadata["leader_candidate_objective_spread"], 6),
            round(high.metadata["leader_selected_N_P_star"], 6),
            round(high.metadata["leader_selected_N_UF_star"], 6),
        )
        self.assertNotEqual(
            low_signature,
            high_signature,
            "leader 후보 평가/랭킹/선택/action objective가 미래 forecast 변화에 반응하지 않음",
        )


if __name__ == "__main__":
    unittest.main()
