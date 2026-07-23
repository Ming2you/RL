# 13-player segment agent 국소 plant 검증 — link 전진과의 비트 일치·소유 매핑·결합 반응성
import unittest

from src.controllers.local_freeway_plant import (
    build_local_freeway_model,
    freeway_substep_local,
)
from src.controllers.segment_local_plant import (
    FrozenLinkTrajectory,
    SegmentLocalState,
    build_segment_agent_models,
    frozen_trajectory_from_state,
    segment_substep_local,
)
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState

CONFIG_PATH = "src/config/default.yaml"


def _cfg() -> ExperimentConfig:
    return ExperimentConfig.from_file(CONFIG_PATH, {})


def _demand(link: str = "FW_W") -> DemandStep:
    return DemandStep(
        freeway_mainline={link: 1500.0},
        urban_boundary={},
        ramp_arrival={},
    )


def _run_link_forward(cfg, link, control, demand, substeps):
    """link stepper를 substeps번 전진 — 매 substep의 (입력 스냅샷, 출력)을 기록.

    occupancy는 호출자 규약대로 offramp_flow 유입 − drain 유출로 갱신(고정 drain 300 veh/h).
    """
    model = build_local_freeway_model(cfg, link)
    net = cfg.network
    dt_h = cfg.simulation.T_f_h
    n = model.n_seg
    # 상류 자유류 → 하류 혼잡 경사 초기조건(세그먼트 수 무관 선형 보간).
    rhos = [30.0 + (90.0 - 30.0) * i / max(n - 1, 1) for i in range(n)]
    speeds = [80.0 + (25.0 - 80.0) * i / max(n - 1, 1) for i in range(n)]
    prev_lanes = [float(net.freeway_lanes)] * n
    origin_q = 12.0
    # capacity-drop 영역에 들어가도록 60% 점유에서 시작.
    occupancy = {o: 0.6 * model.offramp_storage_cap.get(o, 0.0) for o in model.owned_offramps}
    releases = {r: 800.0 for r in model.owned_ramps}
    caps = {o: 400.0 for o in model.owned_offramps}
    drain = 300.0

    inputs, outputs = [], []
    for _ in range(substeps):
        inputs.append(
            dict(
                rhos=list(rhos), speeds=list(speeds), prev_lanes=list(prev_lanes),
                origin_q=float(origin_q), occupancy=dict(occupancy),
                releases=dict(releases), caps=dict(caps),
            )
        )
        rhos, speeds, prev_lanes, origin_q, off_flow, veh = freeway_substep_local(
            model, rhos, speeds, prev_lanes, occupancy, origin_q, releases, caps,
            control, demand,
        )
        outputs.append(dict(rhos=list(rhos), speeds=list(speeds), origin_q=float(origin_q),
                            off_flow=dict(off_flow), veh=list(veh)))
        for o in model.owned_offramps:
            cap_v = model.offramp_storage_cap.get(o, 0.0)
            occ_new = occupancy[o] + dt_h * (off_flow.get(o, 0.0) - drain)
            occupancy[o] = min(max(occ_new, 0.0), cap_v)
    return model, inputs, outputs


class TestSegmentAgentMapping(unittest.TestCase):
    def test_segment_player_ownership(self):
        # 승인 매핑(기하 무관, cfg 유도): merge seg가 해당 ramp를 소유, seg0=origin queue,
        # off-ramp 유출 경계는 off_ramp_segment_index의 seg. 총 player = 5 urban + 2n seg
        # (8-seg 디폴트: 5 + 16 = 21).
        cfg = _cfg()
        net = cfg.network
        n = net.freeway_segments_per_link
        m_d = net.ramp_merge_segment_index["R_D_W"]
        m_f = net.ramp_merge_segment_index["R_F_W"]
        o_d = net.off_ramp_segment_index["OR_D_W"]
        o_f = net.off_ramp_segment_index["OR_F_W"]
        agents_w = build_segment_agent_models(cfg, "FW_W")
        agents_e = build_segment_agent_models(cfg, "FW_E")
        self.assertEqual(len(agents_w), n)
        self.assertEqual(
            len(cfg.network.signals) + len(agents_w) + len(agents_e), 5 + 2 * n,
        )
        self.assertTrue(agents_w[0].owns_origin_queue)
        self.assertEqual(agents_w[m_d].owned_ramps, ["R_D_W"])
        self.assertEqual(agents_w[m_f].owned_ramps, ["R_F_W"])
        for i, agent in enumerate(agents_w):
            if i not in (m_d, m_f):
                self.assertEqual(agent.owned_ramps, [])
        self.assertEqual(agents_w[o_d].boundary_offramps, ["OR_D_W"])
        self.assertEqual(agents_w[o_f].boundary_offramps, ["OR_F_W"])
        self.assertEqual(agents_e[m_d].owned_ramps, ["R_D_E"])
        self.assertEqual(agents_e[m_f].owned_ramps, ["R_F_E"])


class TestSegmentLocalExactness(unittest.TestCase):
    """참 이웃 궤적(y)을 동결 입력으로 주면 segment-local 전진이 link 전진과 비트 일치."""

    def test_matches_link_forward_bitwise(self):
        cfg = _cfg()
        link = "FW_W"
        control = ControlAction.uncontrolled(cfg)
        demand = _demand(link)
        substeps = 20
        model, inputs, outputs = _run_link_forward(cfg, link, control, demand, substeps)

        frozen = FrozenLinkTrajectory(
            rhos=[inp["rhos"] for inp in inputs],
            speeds=[inp["speeds"] for inp in inputs],
            prev_lanes=[inp["prev_lanes"] for inp in inputs],
            origin_queue=[inp["origin_q"] for inp in inputs],
            ramp_release=[inp["releases"] for inp in inputs],
            occupancy=[inp["occupancy"] for inp in inputs],
            offramp_capacity=[inp["caps"] for inp in inputs],
        )
        for agent in build_segment_agent_models(cfg, link):
            own = SegmentLocalState(
                rho=inputs[0]["rhos"][agent.seg],
                speed=inputs[0]["speeds"][agent.seg],
                prev_lane=inputs[0]["prev_lanes"][agent.seg],
                origin_queue=inputs[0]["origin_q"],
            )
            for t in range(substeps):
                own, off_flow, veh = segment_substep_local(
                    agent, frozen, t, own,
                    inputs[t]["releases"], control, demand,
                )
                self.assertAlmostEqual(
                    own.rho, outputs[t]["rhos"][agent.seg], places=9,
                    msg=f"seg{agent.seg} rho @t={t}",
                )
                self.assertAlmostEqual(
                    own.speed, outputs[t]["speeds"][agent.seg], places=9,
                    msg=f"seg{agent.seg} speed @t={t}",
                )
                self.assertAlmostEqual(veh, outputs[t]["veh"][agent.seg], places=9)
                if agent.owns_origin_queue:
                    self.assertAlmostEqual(own.origin_queue, outputs[t]["origin_q"], places=9)
                for o in agent.boundary_offramps:
                    self.assertAlmostEqual(
                        off_flow[o], outputs[t]["off_flow"][o], places=9,
                    )


class TestSegmentCouplingResponsiveness(unittest.TestCase):
    """결합변수가 살아있는지 — 동결 y·자기 lever 변화가 자기 seg 전진에 반영돼야 한다."""

    def _base(self):
        cfg = _cfg()
        link = "FW_W"
        control = ControlAction.uncontrolled(cfg)
        demand = _demand(link)
        agents = build_segment_agent_models(cfg, link)
        net = cfg.network
        n = net.freeway_segments_per_link
        m_f = net.ramp_merge_segment_index["R_F_W"]
        # R_F 합류 seg(와 그 하류)를 rho_max 근처로 — receiving이 병목이 되는 영역.
        rhos = [40.0] * n
        speeds = [60.0] * n
        rhos[0], speeds[0] = 30.0, 80.0
        rhos[1], speeds[1] = 45.0, 65.0
        rhos[m_f], speeds[m_f] = 90.0, 20.0
        if m_f + 1 < n:
            rhos[m_f + 1], speeds[m_f + 1] = 90.0, 20.0
        state_arrays = dict(
            rhos=rhos, speeds=speeds, lanes=[float(net.freeway_lanes)] * n,
        )
        return cfg, link, control, demand, agents, state_arrays, m_f

    def _frozen(self, cfg, arrays, releases):
        return FrozenLinkTrajectory(
            rhos=[list(arrays["rhos"])],
            speeds=[list(arrays["speeds"])],
            prev_lanes=[list(arrays["lanes"])],
            origin_queue=[0.0],
            ramp_release=[dict(releases)],
            occupancy=[{}],
            offramp_capacity=[{}],
        )

    def test_neighbor_ramp_release_in_y_moves_own_outflow(self):
        # 합류 상류 seg의 q_out은 하류 receiving에서 R_F owner의 방류를 차감 — 이웃 lever가
        # y를 통해 자기 전진을 바꿔야 결합이 실체다(동결이어도 iteration 간 전파의 기반).
        cfg, link, control, demand, agents, arrays, m_f = self._base()
        seg_up = agents[m_f - 1]
        own = SegmentLocalState(rho=arrays["rhos"][m_f - 1], speed=arrays["speeds"][m_f - 1],
                                prev_lane=arrays["lanes"][m_f - 1])
        results = []
        for rf_release in (0.0, 1200.0):
            frozen = self._frozen(cfg, arrays, {"R_D_W": 0.0, "R_F_W": rf_release})
            nxt, _, _ = segment_substep_local(
                seg_up, frozen, 0, own, {}, control, demand,
            )
            results.append(nxt.rho)
        # R_F 방류가 크면 합류 seg receiving이 줄어 상류 seg 유출이 막힘 → 밀도 상승.
        self.assertGreater(results[1], results[0] + 1.0e-6)

    def test_own_metering_moves_own_state_when_uncongested(self):
        # 비혼잡(receiving 여유) 영역: R_F owner의 자기 lever(방류)가 자기 밀도에 직접 반영.
        cfg, link, control, demand, agents, arrays, m_f = self._base()
        arrays["rhos"][m_f], arrays["speeds"][m_f] = 40.0, 60.0
        if m_f + 1 < len(arrays["rhos"]):
            arrays["rhos"][m_f + 1], arrays["speeds"][m_f + 1] = 40.0, 60.0
        owner = agents[m_f]
        own = SegmentLocalState(rho=arrays["rhos"][m_f], speed=arrays["speeds"][m_f],
                                prev_lane=arrays["lanes"][m_f])
        frozen = self._frozen(cfg, arrays, {"R_D_W": 0.0, "R_F_W": 0.0})
        rho_by_release = []
        for release in (0.0, 1200.0):
            nxt, _, _ = segment_substep_local(
                owner, frozen, 0, own, {"R_F_W": release}, control, demand,
            )
            rho_by_release.append(nxt.rho)
        self.assertGreater(rho_by_release[1], rho_by_release[0] + 1.0e-6)

    def test_own_metering_displaced_when_receiving_bound(self):
        # 혼잡(receiving 병목) 영역의 보존식 물리: 자기 방류가 본선 유입을 1:1로 밀어내
        # 자기 seg 총유입은 receiving에 포화 → **자기 밀도는 거의 불변**. metering의 이득은
        # 상류 seg 유출 개방으로 넘어간다(cross-agent externality) — segment-player에서 이
        # 배분을 교정하는 장치가 예산 equality + g_ext 가격(plan-13player-rebuild.md 리스크 2).
        cfg, link, control, demand, agents, arrays, m_f = self._base()
        owner = agents[m_f]
        own = SegmentLocalState(rho=arrays["rhos"][m_f], speed=arrays["speeds"][m_f],
                                prev_lane=arrays["lanes"][m_f])
        frozen = self._frozen(cfg, arrays, {"R_D_W": 0.0, "R_F_W": 0.0})
        rho_by_release = []
        for release in (0.0, 1200.0):
            nxt, _, _ = segment_substep_local(
                owner, frozen, 0, own, {"R_F_W": release}, control, demand,
            )
            rho_by_release.append(nxt.rho)
        self.assertAlmostEqual(rho_by_release[1], rho_by_release[0], delta=0.01)


class TestFrozenWarmStart(unittest.TestCase):
    def test_hold_constant_from_state(self):
        cfg = _cfg()
        state = TrafficState.initial(cfg)
        control = ControlAction.uncontrolled(cfg)
        frozen = frozen_trajectory_from_state(cfg, "FW_W", state, control, 6)
        self.assertEqual(len(frozen.rhos), 6)
        # hold-last: horizon 밖 t도 마지막 값.
        self.assertEqual(frozen.at(99)[0], frozen.at(5)[0])
        self.assertIn("R_D_W", frozen.ramp_release[0])
        self.assertIn("R_F_W", frozen.ramp_release[0])


if __name__ == "__main__":
    unittest.main()
