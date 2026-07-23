# P1.5 auto 게이트: OFF/계측전용=기존 거동 비트동일 + 포화도 진단 기록 + band 활성화 검증
import unittest

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


def _solve(follower, cfg):
    forecast = DemandProfile(
        cfg,
        ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
    ).horizon(0.0, 1)
    previous = ControlAction.fixed(cfg)
    return follower.solve(TrafficState.initial(cfg), None, forecast, previous)


class TestP15AutoGate(unittest.TestCase):
    def test_instrument_only_band_is_bit_identical_and_logs_saturation(self):
        cfg = _build_cfg()
        base = _solve(WuFaithfulFollower(cfg), cfg)

        instrumented = WuFaithfulFollower(cfg)
        instrumented.ramp_aware_phase_auto = True
        instrumented.ramp_aware_phase_auto_band = (9.0e9, 9.0e9)  # 게이트 불발(계측 전용)
        probe = _solve(instrumented, cfg)

        for key, value in base.control.green_times.items():
            self.assertAlmostEqual(
                probe.control.green_times[key], value, places=12,
                msg=f"instrument-only auto gate must not move green {key}",
            )
        ramp_signals = [
            s for s in cfg.network.signals
            if instrumented._local_models[s].has_ramps
        ]
        self.assertTrue(ramp_signals)
        for s in ramp_signals:
            self.assertIn(f"wu_p15_sat_{s}", probe.control.diagnostics)
            self.assertGreaterEqual(probe.control.diagnostics[f"wu_p15_sat_{s}"], 0.0)
        self.assertEqual(probe.control.diagnostics["wu_p15_auto_active_count"], 0.0)

    def test_all_pass_band_activates_ramp_signals(self):
        cfg = _build_cfg()
        follower = WuFaithfulFollower(cfg)
        follower.ramp_aware_phase_auto = True
        follower.ramp_aware_phase_auto_band = (0.0, float("inf"))
        result = _solve(follower, cfg)
        ramp_count = sum(
            1 for s in cfg.network.signals if follower._local_models[s].has_ramps
        )
        self.assertEqual(
            result.control.diagnostics["wu_p15_auto_active_count"], float(ramp_count),
            msg="all-pass band must activate every ramp-aware signal",
        )


if __name__ == "__main__":
    unittest.main()
