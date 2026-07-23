# 그리드 내부 라우팅(β분할·차량보존·2-phase) 검증 테스트 — grid_routing_proposal §8
import unittest

from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    approach_routing,
    ensure_urban_state,
    schedule_offramp_arrivals,
    urban_substep,
)
from src.simulation.simulator import MixedTrafficSimulator


def grid_config():
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {"simulation": {"T_total": 720.0}},
    )


def urban_total_vehicles(state: TrafficState, cfg: ExperimentConfig) -> float:
    occupancy = sum(
        max(0.0, cap - state.urban_link_storage.get(link, cap))
        for link, cap in cfg.network.urban_link_storage_veh.items()
    )
    return float(sum(state.urban_movement_queue.values()) + occupancy)


class TurningRatioTests(unittest.TestCase):
    def test_beta_sums_to_one_per_approach(self):
        cfg = grid_config()
        sums: dict[tuple[str, str], float] = {}
        for spec in cfg.network.urban_movements.values():
            key = (str(spec["intersection"]), str(spec["approach"]))
            sums[key] = sums.get(key, 0.0) + float(spec["beta"])
        self.assertTrue(sums)
        for key, total in sums.items():
            self.assertAlmostEqual(total, 1.0, places=9, msg=f"sum_d beta != 1 at {key}")

    def test_no_uturn_movements(self):
        cfg = grid_config()
        for movement, spec in cfg.network.urban_movements.items():
            approach = str(spec["approach"])
            exit_token = str(spec["exit"])
            # U-turn = 같은 leg로 회귀: 방위 동일 또는 램프 leg(S)의 off->on 회귀.
            self.assertNotEqual(approach, exit_token, msg=movement)
            if approach.startswith("off"):
                self.assertFalse(exit_token.startswith("on"), msg=f"off->on U-turn: {movement}")

    def test_straight_preference_beta(self):
        cfg = grid_config()
        specs = cfg.network.urban_movements
        # 4-leg 교차로: 직진(정반대 leg) = 0.5, 나머지 균등.
        self.assertAlmostEqual(specs["A_N_to_S"]["beta"], 0.5)
        self.assertAlmostEqual(specs["A_N_to_E"]["beta"], 0.25)
        self.assertAlmostEqual(specs["A_N_to_W"]["beta"], 0.25)
        # D의 직진(S) 몫은 on_ramp W/E 균등 분할.
        self.assertAlmostEqual(specs["D_N_to_onW"]["beta"], 0.25)
        self.assertAlmostEqual(specs["D_N_to_onE"]["beta"], 0.25)
        # E(3-leg)의 N approach는 직진 leg가 없어 균등.
        self.assertAlmostEqual(specs["E_N_to_E"]["beta"], 0.5)
        self.assertAlmostEqual(specs["E_N_to_W"]["beta"], 0.5)

    def test_two_phase_assignment_by_incoming_axis(self):
        cfg = grid_config()
        for movement, spec in cfg.network.urban_movements.items():
            node = str(spec["intersection"])
            approach = str(spec["approach"])
            phase = str(spec["phase"])
            if node not in cfg.network.signals:
                self.assertEqual(phase, "", msg=f"E는 비통제 통과 노드: {movement}")
                continue
            axis_dir = "S" if approach.startswith("off") else approach
            expected = f"{node}_p1" if axis_dir in {"N", "S"} else f"{node}_p2"
            self.assertEqual(phase, expected, msg=movement)
        # 램프 movement 재배정 확인: off_ramp(S 유입)→p1, on_ramp행은 incoming 축.
        self.assertEqual(cfg.network.urban_movements["D_offW_to_N"]["phase"], "D_p1")
        self.assertEqual(cfg.network.urban_movements["D_N_to_onW"]["phase"], "D_p1")
        self.assertEqual(cfg.network.urban_movements["D_E_to_onW"]["phase"], "D_p2")


class BetaSplitArrivalTests(unittest.TestCase):
    def test_arrival_splits_into_multiple_movements(self):
        cfg = grid_config()
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        # 내부 링크 A_to_D 끝(교차로 D, approach N)에 100대 도착 예약.
        state.urban_arrival_buffer["A_to_D"] = {0: 100.0}
        control = ControlAction.fixed(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test", urban_scale=0.0, ramp_scale=0.0)).at(0.0)
        urban_substep(state, control, demand, cfg, urban_step_index=0)
        # β[N,D,·] = onW 0.25 / onE 0.25 / E 0.25 / W 0.25 로 4개 movement에 분할.
        # (같은 substep에 green service로 일부 빠질 수 있어 합으로 검증하지 않고 분기만 확인.)
        served_or_queued = {
            movement: state.urban_movement_queue.get(movement, 0.0)
            for movement in ("D_N_to_onW", "D_N_to_onE", "D_N_to_E", "D_N_to_W")
        }
        nonzero = sum(1 for v in served_or_queued.values() if v > 0.0)
        self.assertGreaterEqual(nonzero, 3, msg=f"β분할 실패(1:1 체인 의심): {served_or_queued}")

    def test_offramp_vehicles_join_grid_not_terminate(self):
        cfg = grid_config()
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        accepted, rejected = schedule_offramp_arrivals(state, cfg, "OR_D_W", 60.0, 0)
        self.assertAlmostEqual(accepted, 60.0)
        self.assertAlmostEqual(rejected, 0.0)
        # OR storage 링크(=Wu 식17 off-ramp 큐)에 점유가 생긴다. 고정지연 arrival 스케줄은
        # 더 이상 쓰지 않고(Wu 식3 하류 게이트 드레인으로 대체), substep마다 하류 수용공간에
        # 게이트해 그리드로 방출된다.
        storage = cfg.network.off_ramp_storage_link["OR_D_W"]
        cap = cfg.network.urban_link_storage_veh[storage]
        self.assertAlmostEqual(cap - state.urban_link_storage[storage], 60.0)
        self.assertFalse(state.urban_arrival_buffer.get(storage))
        control = ControlAction.fixed(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test", urban_scale=0.0, ramp_scale=0.0)).at(0.0)
        # 하류가 비어 있으므로 여러 substep에 걸쳐 storage가 그리드로 정상 방출돼야 한다.
        for step in range(20):
            urban_substep(state, control, demand, cfg, urban_step_index=step)
        # "소멸"이 아니라 urban 시스템 안에 남아 있어야 한다(보존).
        total = urban_total_vehicles(state, cfg)
        self.assertGreater(total, 0.0)
        offw_movements = cfg.network.off_ramp_to_movement["OR_D_W"]
        self.assertEqual(
            sorted(offw_movements),
            sorted(["D_offW_to_N", "D_offW_to_E", "D_offW_to_W"]),
        )
        # storage 점유가 줄고(드레인) 차량은 하류 그리드 링크 점유로 옮겨가야 한다.
        storage_occ_after = cap - state.urban_link_storage[storage]
        self.assertLess(storage_occ_after, 60.0)
        occupancy_grid = sum(
            max(0.0, cfg.network.urban_link_storage_veh[l] - state.urban_link_storage.get(l, 0.0))
            for l in ("D_to_A", "D_to_E", "D_left_out")
        )
        self.assertGreater(occupancy_grid, 0.0)


class ConservationTests(unittest.TestCase):
    def test_substep_vehicle_conservation_identity(self):
        cfg = grid_config()
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        control = ControlAction.fixed(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        for step in range(40):
            before = urban_total_vehicles(state, cfg)
            ramp_before = sum(state.ramp_queue.values())
            _, diag = urban_substep(
                state,
                control,
                demand,
                cfg,
                urban_step_index=step,
                ramp_release_veh_h={r: 300.0 for r in cfg.network.ramps},
            )
            after = urban_total_vehicles(state, cfg)
            inflow = diag["urban_demand_arrivals_veh"] + diag["onramp_arrivals_veh"]
            outflow = diag["boundary_out_sink_veh"] + diag["onramp_green_releases_veh"]
            expected = before + inflow - outflow - diag["movement_queue_projection_veh"]
            self.assertAlmostEqual(after, expected, places=6, msg=f"step {step}")
            # on_ramp 전이량 = freeway ramp 저수지 유입량(차량 소멸 없음).
            ramp_after = sum(state.ramp_queue.values())
            self.assertAlmostEqual(
                ramp_after - ramp_before,
                diag["onramp_green_releases_veh"] - diag["ramp_metering_releases_veh"],
                places=6,
                msg=f"step {step}",
            )

    def test_closed_loop_interval_conservation_with_offramp(self):
        cfg = grid_config()
        sim = MixedTrafficSimulator(cfg)
        ensure_urban_state(sim.state, cfg)
        demand_profile = DemandProfile(cfg, ScenarioConfig("test"))
        control = ControlAction.fixed(cfg)
        for step_idx in range(3):
            before = urban_total_vehicles(sim.state, cfg)
            log = sim.step(control, demand_profile.at(sim.state.time_sec), step_idx)
            after = urban_total_vehicles(sim.state, cfg)
            d = log.diagnostics
            inflow = (
                d["urban_demand_arrivals_veh"]
                + d["onramp_arrivals_veh"]
                + d["coupling_offramp_arrivals_accepted_veh"]
            )
            outflow = d["boundary_out_sink_veh"] + d["onramp_green_releases_veh"]
            expected = before + inflow - outflow - d["movement_queue_projection_veh"]
            self.assertAlmostEqual(after, expected, places=5, msg=f"interval {step_idx}")

    def test_no_control_flow_reaches_exits_and_grid(self):
        cfg = grid_config()
        sim = MixedTrafficSimulator(cfg)
        demand_profile = DemandProfile(cfg, ScenarioConfig("test"))
        control = ControlAction.fixed(cfg)
        sink_total = 0.0
        transfer_total = 0.0
        for step_idx in range(4):
            log = sim.step(control, demand_profile.at(sim.state.time_sec), step_idx)
            sink_total += log.diagnostics["boundary_out_sink_veh"]
            transfer_total += log.diagnostics["onramp_green_releases_veh"]
        grid_links = [l for l in cfg.network.urban_link_storage_veh if "_to_" in l]
        occupied = [
            l for l in grid_links
            if cfg.network.urban_link_storage_veh[l] - sim.state.urban_link_storage.get(l, 0.0) > 1.0e-9
        ]
        # 게이트 진입 → β로 그리드 통과 → boundary_out 이탈 / on_ramp 전이가 모두 발생해야 한다.
        self.assertEqual(len(occupied), len(grid_links), msg=f"미점유 그리드 링크: {set(grid_links) - set(occupied)}")
        self.assertGreater(sink_total, 0.0)
        self.assertGreater(transfer_total, 0.0)

    def test_approach_routing_covers_all_internal_and_offramp_sources(self):
        cfg = grid_config()
        routing = approach_routing(cfg)
        for link in cfg.network.urban_link_storage_veh:
            if "_to_" in link:
                self.assertIn(link, routing, msg=f"내부 링크 routing 누락: {link}")
        for storage in cfg.network.off_ramp_storage_link.values():
            self.assertIn(storage, routing, msg=f"off-ramp storage routing 누락: {storage}")
        for source, targets in routing.items():
            self.assertAlmostEqual(
                sum(beta for _, beta in targets),
                1.0,
                places=9,
                msg=f"routing β합 != 1: {source}",
            )


if __name__ == "__main__":
    unittest.main()
