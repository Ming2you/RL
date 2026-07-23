# G1: ramp 신호(D/F) offset 활성화 — 기본 off=비트동일, on=ramp 신호 offset 탐색 작동
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
    return follower.solve(TrafficState.initial(cfg), None, forecast, ControlAction.fixed(cfg))


class TestRampOffset(unittest.TestCase):
    def test_default_off_bit_identical(self):
        cfg = _build_cfg()
        base = _solve(WuFaithfulFollower(cfg), cfg)
        f = WuFaithfulFollower(cfg)
        # 기본 ramp_offset_enabled=False → offset 블록 자체가 안 돌아 offset 전부 0.
        g1 = _solve(f, cfg)
        for sig, off in base.control.offsets.items():
            self.assertAlmostEqual(g1.control.offsets.get(sig, 0.0), off, places=12)
        for sig in cfg.network.signals:
            self.assertAlmostEqual(g1.control.offsets.get(sig, 0.0), 0.0, places=12,
                                   msg=f"default must keep {sig} offset 0")

    def test_ramp_offset_searches_only_ramp_signals(self):
        # ramp_offset_enabled 단독 → ramp 신호(D/F)만 offset 탐색, 비-ramp는 0 유지.
        # (corridor 가드가 되돌릴 수 있으므로 가드 우회로 탐색 자체를 검증.)
        cfg = _build_cfg()
        f = WuFaithfulFollower(cfg)
        f.ramp_offset_enabled = True
        f.offset_keep_margin = -1.0  # 가드 우회(탐색 결과 유지)
        ramp_sigs = [s for s in cfg.network.signals if f._local_models[s].has_ramps]
        non_ramp = [s for s in cfg.network.signals if not f._local_models[s].has_ramps]
        self.assertTrue(ramp_sigs, "네트워크에 ramp 신호가 있어야 함")
        res = _solve(f, cfg)
        # 비-ramp는 ramp_only 모드에서 건너뛰므로 offset 0.
        for s in non_ramp:
            self.assertAlmostEqual(res.control.offsets.get(s, 0.0), 0.0, places=12,
                                   msg=f"ramp-only mode must keep non-ramp {s} at 0")
        # ramp 신호는 _solve_offset_local_ramp가 호출되어 후보 탐색이 일어났는지
        # (evals 진단) 확인 — offset 값 자체는 상태에 따라 0일 수 있으나 탐색은 발생.
        self.assertGreater(
            float(res.control.diagnostics.get("wu_faithful_offset_evals", 0.0)), 0.0,
            msg="ramp offset search must run (offset_evals > 0)",
        )

    def test_ramp_offset_method_returns_valid_offset(self):
        # _solve_offset_local_ramp 직접 호출: [0, cycle) 범위의 offset 반환.
        cfg = _build_cfg()
        f = WuFaithfulFollower(cfg)
        state = TrafficState.initial(cfg)
        demand = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)[0]
        snapshot = ControlAction.fixed(cfg)
        coupling = f._wu._coupling(state, ControlAction.uncontrolled(cfg), demand)
        s_eff = f._frozen_s_eff(state)
        ramp_sig = next(s for s in cfg.network.signals if f._local_models[s].has_ramps)
        arr = f._per_movement_arrivals(ramp_sig, state, snapshot, demand)
        off, evals = f._solve_offset_local_ramp(
            ramp_sig, cfg.network.effective_green_total / 2.0, state, coupling, arr,
            s_eff, snapshot, demand,
        )
        self.assertGreaterEqual(off, 0.0)
        self.assertLess(off, cfg.network.cycle_length)
        self.assertGreater(evals, 0)


if __name__ == "__main__":
    unittest.main()
