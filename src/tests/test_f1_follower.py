# F1 사본 검증: 가중치 0 = 부모 비트동일(사본 무결성) + 페널티가 발화 조건에서만 비용 가산
import unittest

from src.controllers.f1_wu_faithful_follower import F1WuFaithfulFollower
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def _build_cfg():
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 360.0},
            "mpc": {
                "horizon_steps": 1,
                "relaxed_quantized_controls": True,
                "grid_parallel_backend": "serial",
            },
            "freeway_follower": {
                "freeway_prediction_horizon_steps": 1,
                "vsl_sequence_search": False,
            },
        },
    )


def _solve(follower, cfg, state=None):
    forecast = DemandProfile(
        cfg,
        ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
    ).horizon(0.0, 1)
    previous = ControlAction.fixed(cfg)
    s = state if state is not None else TrafficState.initial(cfg)
    return follower.solve(s.copy(), None, forecast, previous)


class TestF1CopyIntegrity(unittest.TestCase):
    def test_zero_weights_bit_identical_to_parent(self):
        # 사본 무결성: 가중치 0이면 F1 == 원본(모든 green·metering·VSL·objective 동일).
        cfg = _build_cfg()
        base = _solve(WuFaithfulFollower(cfg), cfg)

        f1 = F1WuFaithfulFollower(cfg)
        f1.f1_spillback_weight = 0.0
        f1.f1_rho_weight = 0.0
        mirror = _solve(f1, cfg)

        self.assertAlmostEqual(
            mirror.objective_value, base.objective_value, places=9,
            msg="zero-weight F1 must reproduce parent objective exactly",
        )
        for key, value in base.control.green_times.items():
            self.assertAlmostEqual(mirror.control.green_times[key], value, places=9)
        for ramp, value in base.control.ramp_metering.items():
            self.assertAlmostEqual(mirror.control.ramp_metering[ramp], value, places=9)

    def test_rho_hinge_raises_freeway_local_cost_when_saturated(self):
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
        link = net.freeway_links[0]

        base_f = WuFaithfulFollower(cfg)
        coupling = base_f._wu._coupling(state, ControlAction.uncontrolled(cfg), demand)
        _, cost_base, _ = base_f._solve_freeway_agent_local(link, state, coupling, demand, previous)

        f1 = F1WuFaithfulFollower(cfg)
        _, cost_f1, _ = f1._solve_freeway_agent_local(link, state, coupling, demand, previous)
        self.assertGreater(
            cost_f1, cost_base + 1e-6,
            msg="rho hinge must add cost on a saturated mainline",
        )

    def test_spill_hinge_inert_on_empty_network(self):
        # 빈 망에서는 spill hinge가 0 — F1(기본 가중치)과 부모의 green이 동일해야 한다.
        cfg = _build_cfg()
        base = _solve(WuFaithfulFollower(cfg), cfg)
        f1 = F1WuFaithfulFollower(cfg)
        f1.f1_rho_weight = 0.0  # freeway hinge도 빈 망에선 0이지만 격리 위해 차단
        mirror = _solve(f1, cfg)
        for key, value in base.control.green_times.items():
            self.assertAlmostEqual(
                mirror.control.green_times[key], value, places=9,
                msg="spill hinge must be inert on an empty network",
            )


if __name__ == "__main__":
    unittest.main()
