# off-ramp 램프 storage의 urban→freeway 재귀속(design 2026-06-17) 검증 단위테스트
import unittest

from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    _offramp_storage_occupancy,
    _storage_occupancy,
)
from src.controllers.leader import Leader
from src.simulation.coupling import run_coupled_interval


def _fill_offramp_storage(state: TrafficState, net, veh_per_ramp: float) -> float:
    """각 off-ramp 램프 storage에 veh_per_ramp만큼 점유를 채우고 총 점유[veh]를 반환한다.

    urban_link_storage는 available(가용공간)을 저장하므로 점유 = cap - available.
    available을 cap-veh_per_ramp로 낮춰 점유를 만든다.
    """
    total = 0.0
    for storage_link in set(net.off_ramp_storage_link.values()):
        cap = float(net.urban_link_storage_veh[storage_link])
        fill = min(veh_per_ramp, cap)
        state.urban_link_storage[storage_link] = cap - fill
        total += fill
    return total


class OffRampReattributionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig.from_file("src/config/default.yaml")
        self.net = self.cfg.network

    # ---------- G2: 재귀속 단위검사 ----------
    def test_storage_occupancy_excludes_offramp(self):
        """urban _storage_occupancy는 off-ramp 램프 storage를 제외, 전용 헬퍼가 그 양을 잡는다."""
        state = TrafficState.initial(self.cfg)
        urban_before = _storage_occupancy(state, self.cfg)
        offramp_before = _offramp_storage_occupancy(state, self.cfg)
        self.assertAlmostEqual(offramp_before, 0.0)

        filled = _fill_offramp_storage(state, self.net, 40.0)
        urban_after = _storage_occupancy(state, self.cfg)
        offramp_after = _offramp_storage_occupancy(state, self.cfg)

        # urban 점유는 off-ramp storage를 채워도 불변(제외됨).
        self.assertAlmostEqual(urban_after, urban_before)
        # 전용 헬퍼가 채운 양을 정확히 잡는다.
        self.assertAlmostEqual(offramp_after, filled)

    def test_np_and_urban_total_exclude_offramp_storage_keep_leg(self):
        """N_P·total_urban_vehicles에서 램프 storage는 빠지고 leg(off_ramp 점큐)는 유지된다."""
        state = TrafficState.initial(self.cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0

        # off_ramp movement(leg) 점큐 하나에 차량을 넣는다 — urban·N_P에 남아야 한다.
        leg_movement = next(
            m for m, sp in self.net.urban_movements.items() if sp.get("kind") == "off_ramp"
        )
        state.urban_movement_queue[leg_movement] = 33.0

        np_before = state.protected_accumulation_veh(self.net)
        urban_total_before = state.total_urban_vehicles(self.net)

        filled = _fill_offramp_storage(state, self.net, 25.0)

        np_after = state.protected_accumulation_veh(self.net)
        urban_total_after = state.total_urban_vehicles(self.net)

        # 램프 storage를 채워도 N_P·urban total 불변(freeway로 재귀속됨).
        self.assertAlmostEqual(np_after, np_before)
        self.assertAlmostEqual(urban_total_after, urban_total_before)
        # leg 점큐(33.0)는 여전히 N_P에 잡힌다(이중계상 아님 — leg는 urban에만).
        self.assertGreaterEqual(np_after, 33.0 - 1.0e-9)
        self.assertGreater(filled, 0.0)

    def test_offramp_storage_helper_matches_state_helper(self):
        """urban_queue_model 헬퍼와 state 헬퍼가 같은 off-ramp 점유를 보고한다(이중계상 0 확인)."""
        state = TrafficState.initial(self.cfg)
        _fill_offramp_storage(state, self.net, 50.0)
        self.assertAlmostEqual(
            _offramp_storage_occupancy(state, self.cfg),
            state.off_ramp_storage_occupancy_veh(self.net),
        )

    # ---------- G1: 보존 ----------
    def test_urban_substep_moves_offramp_ttt_out_of_urban(self):
        """urban_substep: 램프 storage 점유 TTT가 urban_ttt에서 빠지고 진단으로 노출된다.

        보존 항등: (램프 포함 urban_ttt) = urban_ttt + offramp_storage_ttt.
        즉 urban에서 빠진 양 = 진단으로 노출된 양과 정확히 일치(coupling이 freeway로 더함).
        """
        from src.models.demand import DemandStep
        from src.models.urban_queue_model import urban_substep

        state = TrafficState.initial(self.cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        filled = _fill_offramp_storage(state, self.net, 30.0)

        control = ControlAction.fixed(self.cfg)
        demand = DemandStep(freeway_mainline={}, urban_boundary={}, ramp_arrival={})

        urban_ttt, diag = urban_substep(state, control, demand, self.cfg, urban_step_index=0)
        moved = float(diag["offramp_storage_ttt"])
        T_u_h = self.cfg.simulation.T_u_h

        # 진단으로 노출된 램프 storage TTT = (남은 점유 × T_u_h). drain으로 점유가 일부
        # 빠질 수 있으므로 양수이고 채운 양 이하임을 확인.
        self.assertGreater(moved, 0.0)
        self.assertLessEqual(moved, filled * T_u_h + 1.0e-6)
        # 보존: 램프 포함 urban_ttt = urban_ttt(램프 제외) + 옮긴 양.
        urban_storage_occ = float(diag["urban_storage_occupancy"])
        offramp_occ = float(diag["offramp_storage_occupancy_veh"])
        urban_ttt_with_ramp = (
            sum(state.urban_movement_queue.values()) + urban_storage_occ + offramp_occ
        ) * T_u_h
        self.assertAlmostEqual(urban_ttt_with_ramp, urban_ttt + moved)

    def test_coupled_interval_total_ttt_identity(self):
        """coupled interval: total = freeway_ttt+urban_ttt, 옮긴 양이 freeway에 반영되고 진단됨."""
        from src.models.demand import DemandStep

        state = TrafficState.initial(self.cfg)
        _fill_offramp_storage(state, self.net, 30.0)
        control = ControlAction.fixed(self.cfg)
        demand = DemandStep(freeway_mainline={}, urban_boundary={}, ramp_arrival={})

        result = run_coupled_interval(state, control, demand, self.cfg)
        moved = float(result.diagnostics["coupling_offramp_storage_ttt_moved_to_freeway"])
        # 램프 storage가 차 있었으니 옮긴 양이 0보다 큼(emergence 경로 활성).
        self.assertGreater(moved, 0.0)
        # total은 분해와 무관하게 정의됨 — freeway에 옮긴 양이 포함됨.
        self.assertGreaterEqual(result.freeway_ttt, moved - 1.0e-6)

    # ---------- G2(b): freeway agent local TTT에 램프 storage 반영 ----------
    def test_wu_freeway_agent_objective_includes_offramp_storage(self):
        """WU _solve_freeway_agent local TTT가 off-ramp 램프 storage 점유를 비용에 더한다."""
        from src.controllers.wu_distributed import WuDistributedController, _wu_fixed_control
        from src.models.demand import DemandProfile, ScenarioConfig

        cfg = ExperimentConfig.from_file("src/config/default.yaml")
        net = cfg.network
        link = next(
            l
            for l in net.freeway_links
            if any(net.off_ramp_from_freeway.get(o) == l for o in net.off_ramps)
        )
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        previous = _wu_fixed_control(cfg)
        ctl = WuDistributedController(cfg)

        state_empty = TrafficState.initial(cfg)
        state_full = state_empty.copy()
        _fill_offramp_storage(state_full, net, 80.0)

        coupling_e = ctl._coupling(state_empty, previous, demand)
        coupling_f = ctl._coupling(state_full, previous, demand)
        _, obj_empty, _ = ctl._solve_freeway_agent(
            link, state_empty, coupling_e, demand, previous, None
        )
        _, obj_full, _ = ctl._solve_freeway_agent(
            link, state_full, coupling_f, demand, previous, None
        )
        # 램프 storage가 차 있으면 freeway agent 자기 비용이 더 크다(metering 유인의 원천).
        self.assertGreater(obj_full, obj_empty)

    # ---------- G3: leader boundary_in 큐 비용 ----------
    def test_leader_boundary_in_cost_enters_total_objective(self):
        """boundary_in 큐 비용 항은 leader_total_objective에 들어간다."""
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "leader": {
                    "w_P": 0.0,
                    "w_F": 0.0,
                    "w_boundary_in": 2.0,
                    "non_convergence_penalty": 0.0,
                }
            },
        )
        net = cfg.network
        boundary_movement = next(
            m for m, sp in net.urban_movements.items() if sp.get("kind") == "boundary_in"
        )

        state_low = TrafficState.initial(cfg)
        for m in state_low.urban_movement_queue:
            state_low.urban_movement_queue[m] = 0.0
        state_high = state_low.copy()
        state_high.urban_movement_queue[boundary_movement] = 100.0

        leader = Leader(cfg)
        action = ControlAction.fixed(cfg)
        terms_low = leader.objective_terms(
            [state_low], action, None, follower_objective=50.0, nash_converged=True
        )
        terms_high = leader.objective_terms(
            [state_high], action, None, follower_objective=50.0, nash_converged=True
        )

        # follower_ttt 모드에서는 T_c_h를 곱해 veh*h 비용으로 반영한다.
        self.assertAlmostEqual(terms_low["leader_boundary_in_queue_penalty"], 0.0)
        expected_high = 2.0 * 100.0 * cfg.simulation.T_c_h
        self.assertAlmostEqual(terms_high["leader_boundary_in_queue_penalty"], expected_high)
        self.assertAlmostEqual(terms_low["leader_total_objective"], 50.0)
        self.assertAlmostEqual(terms_high["leader_total_objective"], 50.0 + expected_high)


if __name__ == "__main__":
    unittest.main()
