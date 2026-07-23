# Joint 모드 검증: full cross-grid ≥ 좌표하강(포함 관계), equality 계약 유지, directive 적용,
# 두 objective baseline(B2TR/F1) 비혼동(각 계열 상속 확인)
import unittest

from src.controllers.f1_wu_faithful_follower import F1WuFaithfulFollower
from src.controllers.joint_wu_controllers import (
    JointB2TRController,
    JointF1Controller,
    JointF1WuFaithfulFollower,
    JointWuFaithfulFollower,
)
from src.controllers.leader import LeaderAction
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def _build_cfg(nuf_mode="equality"):
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 360.0},
            "mpc": {
                "horizon_steps": 1,
                "relaxed_quantized_controls": True,
                "grid_parallel_backend": "serial",
                "wu_faithful_nuf_coordination_mode": nuf_mode,
                "leader_global_refresh_sec": 1.0e9,
            },
            "freeway_follower": {
                "freeway_prediction_horizon_steps": 1,
                "vsl_sequence_search": False,
            },
        },
    )


def _setup_freeway(cfg, follower):
    state = TrafficState.initial(cfg)
    demand = DemandProfile(
        cfg,
        ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
    ).horizon(0.0, 1)[0]
    snapshot = ControlAction.fixed(cfg)
    coupling = follower._wu._coupling(state, ControlAction.uncontrolled(cfg), demand)
    link = next(
        l for l in cfg.network.freeway_links
        if any(cfg.network.ramp_to_freeway.get(r) == l for r in cfg.network.ramps)
        and float(follower._wu._omega_f.get(l, 0.0)) > 0.0
    )
    owned = [r for r in cfg.network.ramps if cfg.network.ramp_to_freeway.get(r) == link]
    return state, demand, snapshot, coupling, link, owned


class TestJointFreewaySearch(unittest.TestCase):
    def test_full_grid_never_worse_than_descent_autonomous(self):
        # full cross-grid ⊇ 좌표하강 방문집합 → joint own-TTS ≤ 기존(양 objective 계열).
        for base_cls, joint_cls in (
            (WuFaithfulFollower, JointWuFaithfulFollower),
            (F1WuFaithfulFollower, JointF1WuFaithfulFollower),
        ):
            cfg = _build_cfg()
            base = base_cls(cfg)
            joint = joint_cls(cfg)
            state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg, base)
            _, m_base, _ = base._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, None,
            )
            _, cost_base, _ = base._solve_freeway_agent_local(
                link, state, coupling, demand, _with_meter(snapshot, m_base),
            )
            _, m_joint, _ = joint._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, None,
            )
            _, cost_joint, _ = joint._solve_freeway_agent_local(
                link, state, coupling, demand, _with_meter(snapshot, m_joint),
            )
            self.assertLessEqual(
                cost_joint, cost_base + 1e-6,
                msg=f"{joint_cls.__name__}: joint grid must not be worse than descent",
            )

    def test_equality_contract_preserved(self):
        # equality 모드: joint 후보들이 Σ=budget(클립 근사 이내)을 유지.
        cfg = _build_cfg("equality")
        joint = JointWuFaithfulFollower(cfg)
        state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg, joint)
        caps_sum = sum(float(cfg.network.ramp_capacity_veh_h[r]) for r in owned)
        omega = float(joint._wu._omega_f.get(link, 0.0))
        n_uf = 0.5 * caps_sum / omega
        budget = min(max(omega * n_uf, 0.0), caps_sum)
        _, meter, _ = joint._solve_freeway_agent_metered(
            link, state, coupling, demand, snapshot, LeaderAction(0.0, n_uf),
        )
        self.assertLessEqual(sum(meter.values()), budget + 1e-6)
        # 비례 스케일이므로 심하게 밑돌지도 않아야(클립이 없으면 정확히 budget).
        self.assertGreaterEqual(sum(meter.values()), 0.5 * budget)

    def test_objective_lineage_not_confused(self):
        # Joint-F1은 ρ hinge를 상속(포화 상태에서 B2TR 계열보다 비싼 채점),
        # Joint-B2TR은 hinge 없음 — 두 baseline objective의 비혼동 확인.
        cfg = _build_cfg()
        net = cfg.network
        state = TrafficState.initial(cfg)
        for link in net.freeway_links:
            n = len(state.freeway_density.get(link, []))
            state.freeway_density[link] = [float(net.rho_crit) + 20.0] * n
        demand = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)[0]
        previous = ControlAction.fixed(cfg)
        b2tr_f = JointWuFaithfulFollower(cfg)
        f1_f = JointF1WuFaithfulFollower(cfg)
        coupling = b2tr_f._wu._coupling(state, ControlAction.uncontrolled(cfg), demand)
        link = net.freeway_links[0]
        _, cost_b2tr, _ = b2tr_f._solve_freeway_agent_local(link, state, coupling, demand, previous)
        _, cost_f1, _ = f1_f._solve_freeway_agent_local(link, state, coupling, demand, previous)
        self.assertGreater(
            cost_f1, cost_b2tr + 1e-6,
            msg="Joint-F1 must inherit the rho hinge (dearer under saturation)",
        )

    def test_controllers_wire_expected_followers(self):
        cfg = _build_cfg()
        self.assertIsInstance(JointB2TRController(cfg).nash_solver, JointWuFaithfulFollower)
        jf1 = JointF1Controller(cfg)
        self.assertIsInstance(jf1.nash_solver, JointF1WuFaithfulFollower)
        self.assertEqual(jf1.nash_solver.f1_spillback_weight, 0.0)  # F1RHO 규약
        self.assertEqual(jf1.nash_solver.f1_rho_weight, 1.0)


def _with_meter(snapshot, meter):
    probe = ControlAction(
        ramp_metering=dict(snapshot.ramp_metering),
        vsl=dict(snapshot.vsl),
        green_times=dict(snapshot.green_times),
        offsets=dict(snapshot.offsets),
        inflow_outflow_allocation={},
    )
    probe.ramp_metering.update({r: float(v) for r, v in meter.items()})
    return probe


class TestJointOffsetDirective(unittest.TestCase):
    def test_directive_applied_and_guarded(self):
        cfg = _build_cfg()
        follower = WuFaithfulFollower(cfg)
        follower.offset_keep_margin = -1.0  # 가드 우회(역학 검증용 — 가드는 설계상 유지)
        cycle = float(cfg.network.cycle_length)
        non_ramp = [s for s in cfg.network.signals if not follower._local_models[s].has_ramps]
        directive = {s: cycle / 8.0 for s in non_ramp}
        follower.offset_directive = directive
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)
        nash = follower.solve(
            TrafficState.initial(cfg), None, forecast, ControlAction.fixed(cfg),
        )
        for s in non_ramp:
            self.assertAlmostEqual(
                nash.control.offsets.get(s, 0.0), cycle / 8.0, places=6,
                msg=f"directive offset must be applied for {s}",
            )


if __name__ == "__main__":
    unittest.main()
