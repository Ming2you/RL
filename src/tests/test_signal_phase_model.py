# cycle 위상(offset) plant 모델과 corridor offset 휴리스틱 검증 테스트
import unittest

from src.controllers.urban_follower import UrbanFollower
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    _phase_green_fraction,
    ensure_urban_state,
    urban_substep,
)


def cfg_default():
    return ExperimentConfig.from_file("src/config/default.yaml")


def empty_demand(cfg, boundary=None):
    return DemandStep(
        freeway_mainline={link: 0.0 for link in cfg.network.freeway_links},
        urban_boundary=boundary or {},
        ramp_arrival={r: 0.0 for r in cfg.network.ramps},
    )


class GreenWindowTests(unittest.TestCase):
    def test_phase_windows_follow_cycle_structure(self):
        cfg = cfg_default()
        control = ControlAction.fixed(cfg)  # g1=g2=56, lost 8, offset 0
        spec_p1 = {"phase": "A_p1"}
        spec_p2 = {"phase": "A_p2"}
        cycle_steps = int(cfg.network.cycle_length / cfg.simulation.T_u_sec)  # 24
        p1 = [_phase_green_fraction(control, cfg, spec_p1, urban_step_index=k) for k in range(cycle_steps)]
        p2 = [_phase_green_fraction(control, cfg, spec_p2, urban_step_index=k) for k in range(cycle_steps)]
        # p1 green [0,56): step 0~10 풀, step 11은 [55,60) 중 1초 → 0.2.
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in p1[:11]))
        self.assertAlmostEqual(p1[11], 0.2)
        self.assertTrue(all(v == 0.0 for v in p1[12:]))
        # p2 green [60,116): step 12~22 풀, step 23은 [115,120) 중 1초 → 0.2.
        self.assertTrue(all(v == 0.0 for v in p2[:12]))
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in p2[12:23]))
        self.assertAlmostEqual(p2[23], 0.2)
        # cycle 평균 서비스량 보존: Σ겹침×T_u = green_sec.
        t_u = cfg.simulation.T_u_sec
        self.assertAlmostEqual(sum(p1) * t_u, 56.0)
        self.assertAlmostEqual(sum(p2) * t_u, 56.0)
        # offset은 패턴을 그대로 평행이동시킨다(30s = 6 substep).
        control.offsets["A"] = 30.0
        p1_shift = [_phase_green_fraction(control, cfg, spec_p1, urban_step_index=k) for k in range(cycle_steps)]
        for k in range(cycle_steps):
            self.assertAlmostEqual(p1_shift[(k + 6) % cycle_steps], p1[k])

    def test_average_mode_unchanged_without_step(self):
        cfg = cfg_default()
        control = ControlAction.fixed(cfg)
        self.assertAlmostEqual(
            _phase_green_fraction(control, cfg, {"phase": "A_p1"}),
            56.0 / 120.0,
        )
        self.assertEqual(_phase_green_fraction(control, cfg, {"phase": ""}), 1.0)


class OffsetAlignmentEffectTests(unittest.TestCase):
    """offset이 plant 동역학을 실제로 바꾸는지 — 정렬 green wave vs 비정렬 비교."""

    def _run(self, offset_b: float, substeps: int) -> float:
        cfg = cfg_default()
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        for movement in state.urban_movement_queue:
            state.urban_movement_queue[movement] = 0.0
        control = ControlAction.fixed(cfg)
        control.offsets = {s: 0.0 for s in cfg.network.signals}
        control.offsets["B"] = offset_b
        # in_A_left(서쪽 게이트, A의 p2)에서만 유입 → A_to_B로 동진 플래툰 형성.
        demand = empty_demand(cfg, boundary={"in_A_left": 1200.0})
        total_ttt = 0.0
        for k in range(substeps):
            ttt, _ = urban_substep(state, control, demand, cfg, urban_step_index=k)
            total_ttt += ttt
        return total_ttt

    def test_aligned_offset_beats_antialigned(self):
        # A(p2 [60,116))에서 방출된 플래툰이 빈 링크 통과시간 ≈95s 뒤 B(approach W, p2)에
        # 도착한다. 정렬 offset_B = 95 → B의 p2 green이 도착 시점에 열림.
        # 비정렬 = 반 cycle 어긋난 35 → 도착분이 적색에 걸려 대기.
        substeps = 24 * 8  # 8 cycle
        aligned = self._run(95.0, substeps)
        anti = self._run(35.0, substeps)
        self.assertLess(aligned, anti, msg=f"aligned={aligned:.2f} anti={anti:.2f}")


class OffsetHeuristicTests(unittest.TestCase):
    def _converge(self, follower, state, control, iters=12):
        for _ in range(iters):
            offsets = follower._offsets(state, control, control.green_times)
            control.offsets = dict(offsets)
        return control.offsets

    def test_offsets_converge_to_urban_progression(self):
        cfg = cfg_default()
        follower = UrbanFollower(cfg)
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        control = ControlAction.fixed(cfg)
        control.offsets = {s: 0.0 for s in cfg.network.signals}
        offsets = self._converge(follower, state, control)
        # t_link = 220×6m / 50km/h = 95.04s. 균등 split이라 p2 보정항 0.
        t_link = 220.0 * 6.0 / 1000.0 / 50.0 * 3600.0
        cycle = cfg.network.cycle_length
        self.assertAlmostEqual((offsets["B"] - offsets["A"]) % cycle, t_link % cycle, places=6)
        self.assertAlmostEqual((offsets["C"] - offsets["B"]) % cycle, t_link % cycle, places=6)
        self.assertAlmostEqual((offsets["D"] - offsets["A"]) % cycle, t_link % cycle, places=6)
        self.assertAlmostEqual((offsets["F"] - offsets["D"]) % cycle, (2.0 * t_link) % cycle, places=6)

    def test_direction_flips_with_dominant_load(self):
        cfg = cfg_default()
        follower = UrbanFollower(cfg)
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        # 서행(westbound) 부하를 크게: B_to_A·C_to_B 점유.
        for link in ("B_to_A", "C_to_B"):
            state.urban_link_storage[link] = cfg.network.urban_link_storage_veh[link] - 150.0
        control = ControlAction.fixed(cfg)
        control.offsets = {s: 0.0 for s in cfg.network.signals}
        offsets = self._converge(follower, state, control)
        t_link = 220.0 * 6.0 / 1000.0 / 50.0 * 3600.0
        cycle = cfg.network.cycle_length
        # 서행 진행이면 B는 A보다 t_link 빨라야 한다(−t_link mod cycle).
        self.assertAlmostEqual((offsets["A"] - offsets["B"]) % cycle, t_link % cycle, places=6)

    def test_offset_step_constraint_respected(self):
        cfg = cfg_default()
        follower = UrbanFollower(cfg)
        state = TrafficState.initial(cfg)
        ensure_urban_state(state, cfg)
        control = ControlAction.fixed(cfg)
        control.offsets = {s: 0.0 for s in cfg.network.signals}
        offsets = follower._offsets(state, control, control.green_times)
        cycle = cfg.network.cycle_length
        for signal, value in offsets.items():
            delta = (value - control.offsets[signal] + 0.5 * cycle) % cycle - 0.5 * cycle
            self.assertLessEqual(abs(delta), cfg.urban_follower.max_offset_step + 1e-9)


if __name__ == "__main__":
    unittest.main()
