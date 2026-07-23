# Stage 2/3 사후분석 도구 검증 테스트 (plan §3, §12)
import unittest

from src.controllers.distributed_coordinator import ABLATION_MODES, DistributedCoordinator
from src.experiments.stage3_coupling_ablation import coupling_values
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def small_cfg():
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 360.0},
            "mpc": {"horizon_steps": 1, "max_nash_iter": 1, "leader_candidate_count": 2},
            "urban_follower": {"allocation_pso_particles": 6, "allocation_pso_iterations": 4},
        },
    )


class Stage3Tests(unittest.TestCase):
    def test_all_eight_ablation_modes_exist(self):
        self.assertEqual(len(ABLATION_MODES), 8)
        for mode in ABLATION_MODES:
            DistributedCoordinator(small_cfg(), ablation=mode)  # 생성 가능해야 함.
        with self.assertRaises(ValueError):
            DistributedCoordinator(small_cfg(), ablation="NOT_A_MODE")

    def test_coupling_value_formulas(self):
        # plan §12 수식의 산술 검증 — 손으로 계산 가능한 값.
        j = {
            "FULL_COUPLING": 100.0,
            "NO_U_TO_F_INFO": 110.0,
            "NO_F_TO_U_INFO": 105.0,
            "NO_CROSS_NETWORK_INFO": 118.0,
            "FIXED_URBAN_COUPLING_PLAYERS": 109.0,
            "FIXED_FREEWAY_COUPLING_PLAYERS": 112.0,
            "FIXED_ALL_COUPLING_PLAYERS": 120.0,
        }
        values = {row["metric"]: row["value"] for row in coupling_values(j)}
        self.assertAlmostEqual(values["Value_U_to_F_given_F_to_U"], 10.0)
        self.assertAlmostEqual(values["Value_F_to_U_given_U_to_F"], 5.0)
        self.assertAlmostEqual(values["BidirectionalSynergy"], 3.0)  # 118−110−105+100.
        self.assertAlmostEqual(values["Phi_U_to_F"], 11.5)  # 0.5×[(118−105)+(110−100)].
        self.assertAlmostEqual(values["Phi_F_to_U"], 6.5)
        self.assertAlmostEqual(values["UrbanCouplingPlayerValue"], 9.0)
        self.assertAlmostEqual(values["FreewayCouplingPlayerValue"], 12.0)
        # Shapley형 분해: Phi 합 = J_none − J_full.
        self.assertAlmostEqual(values["Phi_U_to_F"] + values["Phi_F_to_U"], 18.0)

    def test_ablation_keeps_physical_coupling(self):
        # 정보 차단 모드에서도 plant 결합(off-ramp 수용/이동)은 그대로 — 같은 control이면
        # plant 전이는 ablation과 무관해야 한다(ablation은 결정 경로에만 작용).
        from src.simulation.simulator import MixedTrafficSimulator

        cfg = small_cfg()
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        ttts = []
        for _ in ("FULL_COUPLING", "NO_CROSS_NETWORK_INFO"):
            sim = MixedTrafficSimulator(cfg)
            sim.step(ControlAction.fixed(cfg), demand, 0)
            ttts.append(sim.total_ttt)
        self.assertAlmostEqual(ttts[0], ttts[1])

    def test_fixed_players_pin_their_controls(self):
        cfg = small_cfg()
        cfg.mpc.follower_solver_mode = "distributed"
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).horizon(0.0, 1)
        from src.controllers.leader import LeaderAction

        coordinator = DistributedCoordinator(cfg, ablation="FIXED_FREEWAY_COUPLING_PLAYERS")
        result = coordinator.solve(state, LeaderAction(400.0, 1000.0), demand, ControlAction.fixed(cfg))
        net = cfg.network
        # coupling freeway player의 ramp는 용량 고정(중립 정책)이어야 한다.
        for ramp in net.ramps:
            self.assertAlmostEqual(
                result.control.ramp_metering.get(ramp, 0.0), net.ramp_capacity_veh_h[ramp],
                msg=f"{ramp} not pinned to capacity",
            )


class Stage2Tests(unittest.TestCase):
    def test_event_detection_and_replay_smoke(self):
        from src.analysis.stage2_mechanism import (
            detect_and_evaluate_events,
            run_traced_closed_loop,
        )

        cfg = small_cfg()
        scenario = ScenarioConfig("test")
        traces = run_traced_closed_loop(cfg, scenario)
        self.assertEqual(len(traces), 2)  # 360s / 180s.
        events = detect_and_evaluate_events(cfg, scenario, traces, max_events_per_control=1)
        controls = {e.control for e in events}
        self.assertEqual(controls, {"allocation_green", "offset", "vsl", "metering"})


if __name__ == "__main__":
    unittest.main()
