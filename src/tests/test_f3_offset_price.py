# F3 offset 가격: 휴면=offset 0 유지(비트동일), 가격이 그리드 1칸 내에서 offset을 움직임
import unittest

from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
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
                "leader_global_refresh_sec": 1.0e9,
            },
            "freeway_follower": {
                "freeway_prediction_horizon_steps": 1,
                "vsl_sequence_search": False,
            },
        },
    )


def _demand(cfg):
    return DemandProfile(
        cfg,
        ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
    ).horizon(0.0, 1)


class TestF3OffsetPrice(unittest.TestCase):
    def test_dormant_channel_keeps_offsets_zero(self):
        cfg = _build_cfg()
        follower = WuFaithfulFollower(cfg)
        nash = follower.solve(
            TrafficState.initial(cfg), None, _demand(cfg), ControlAction.fixed(cfg),
        )
        for signal, off in nash.control.offsets.items():
            self.assertAlmostEqual(off, 0.0, places=9,
                                   msg=f"dormant offset channel must keep {signal} at 0")

    def test_price_moves_offset_within_one_grid_step(self):
        cfg = _build_cfg()
        net = cfg.network
        cycle = float(net.cycle_length)
        grid = cycle / 8.0
        follower = WuFaithfulFollower(cfg)
        non_ramp = [s for s in net.signals if not follower._local_models[s].has_ramps]
        sig = non_ramp[0]
        # 큰 음수 가격 = "offset을 키워라". trust = 그리드 1칸.
        # corridor 가드(realized TTT 검증)는 빈 망에서 어떤 offset도 되돌리므로 테스트에선
        # 우회(-1.0 = 항상 유지) — 가드 자체는 F3에서도 최종 검증자로 살아있는 설계.
        follower.offset_keep_margin = -1.0
        follower.offset_marginal_price = {sig: -100.0}
        follower.offset_marginal_price_ref = {sig: 0.0}
        follower.offset_marginal_price_trust_sec = grid
        nash = follower.solve(
            TrafficState.initial(cfg), None, _demand(cfg), ControlAction.fixed(cfg),
        )
        off = float(nash.control.offsets.get(sig, 0.0))
        d = abs(off - 0.0) % cycle
        d = min(d, cycle - d)
        self.assertGreater(d, 1e-9, msg="huge price must move the offset off zero")
        self.assertLessEqual(d, grid + 1e-9,
                             msg="trust must keep offset within one grid step of ref")

    def test_controller_refresh_hands_offset_prices_for_non_ramp_signals(self):
        cfg = _build_cfg()
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = False  # offset 채널만 격리
        controller.offset_price_enabled = True
        state = TrafficState.initial(cfg)
        state.time_sec = float(cfg.simulation.control_interval)
        controller._maybe_refresh_signal_prices(
            state, _demand(cfg), ControlAction.fixed(cfg),
        )
        follower = controller.nash_solver
        self.assertIsNotNone(follower.offset_marginal_price)
        non_ramp = {
            s for s in cfg.network.signals if not follower._local_models[s].has_ramps
        }
        self.assertEqual(set(follower.offset_marginal_price), non_ramp)
        self.assertEqual(
            follower.offset_marginal_price_trust_sec,
            float(cfg.network.cycle_length) / 8.0,
        )


if __name__ == "__main__":
    unittest.main()
