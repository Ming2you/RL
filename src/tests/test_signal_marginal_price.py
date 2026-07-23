# B2 per-signal externality 가격: follower 주입 하위호환(None=비트동일) + 컨트롤러 refresh 트리거 테스트
import unittest

from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def _build_cfg(extra_mpc=None):
    mpc = {
        "horizon_steps": 1,
        "relaxed_quantized_controls": True,
        "grid_parallel_backend": "serial",
        "leader_search_mode": "grid",
        "stackelberg_leader_parallel_backend": "serial",
    }
    if extra_mpc:
        mpc.update(extra_mpc)
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 360.0},
            "mpc": mpc,
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


class TestSignalMarginalPriceFollower(unittest.TestCase):
    """follower 가격항: None=완전 휴면(비트동일), 0가격=무영향, 부호가 argmin을 민다."""

    def test_zero_price_identical_to_none(self):
        # PRICE-TR(2026-07-09) 이후 "0가격 ≡ None"은 마찰 의미론(flag OFF)에서만 성립 —
        # 가격 모드 기본은 smoothness=0이라 0가격도 regularizer가 달라진다(trust만).
        # 이 테스트는 레거시 마찰 의미론의 채널 격리를 계속 고정한다.
        cfg = _build_cfg()
        forecast = _demand(cfg)
        previous = ControlAction.fixed(cfg)

        f_none = WuFaithfulFollower(cfg)
        base = f_none.solve(TrafficState.initial(cfg), None, forecast, previous)

        f_zero = WuFaithfulFollower(cfg)
        f_zero.price_smoothness_disabled = False
        f_zero.signal_marginal_price = {s: 0.0 for s in cfg.network.signals}
        f_zero.signal_marginal_price_ref = {
            s: float(previous.green_times.get(f"{s}_p1", 0.0))
            for s in cfg.network.signals
        }
        priced = f_zero.solve(TrafficState.initial(cfg), None, forecast, previous)

        for key, value in base.control.green_times.items():
            self.assertAlmostEqual(
                priced.control.green_times[key], value, places=12,
                msg=f"zero price must not move green {key}",
            )

    def test_price_sign_pushes_green_argmin(self):
        cfg = _build_cfg()
        forecast = _demand(cfg)
        previous = ControlAction.fixed(cfg)
        signal = cfg.network.signals[0]

        def solve_with_price(g_ext):
            follower = WuFaithfulFollower(cfg)
            if g_ext is not None:
                follower.signal_marginal_price = {signal: float(g_ext)}
                follower.signal_marginal_price_ref = {
                    signal: float(cfg.network.effective_green_total / 2.0)
                }
            nash = follower.solve(TrafficState.initial(cfg), None, forecast, previous)
            return float(nash.control.green_times[f"{signal}_p1"])

        p1_pos = solve_with_price(+10.0)   # 큰 양수 가격 → p1 증가가 비싸짐 → 작은 p1 선호
        p1_neg = solve_with_price(-10.0)   # 큰 음수 가격 → 큰 p1 선호
        self.assertLess(
            p1_pos, p1_neg,
            msg="positive externality price must push green p1 below the negative-price choice",
        )

    def test_trust_region_bounds_priced_argmin(self):
        # B2.1: trust 설정 시 거대 가격도 argmin을 ref의 유한차분 이웃 밖으로 못 끌어낸다.
        cfg = _build_cfg()
        forecast = _demand(cfg)
        previous = ControlAction.fixed(cfg)
        signal = cfg.network.signals[0]
        ref = float(cfg.network.effective_green_total / 2.0)

        def solve_with(trust):
            follower = WuFaithfulFollower(cfg)
            follower.signal_marginal_price = {signal: -100.0}  # "p1 올려라" 극단 가격
            follower.signal_marginal_price_ref = {signal: ref}
            follower.signal_marginal_price_trust_sec = trust
            nash = follower.solve(TrafficState.initial(cfg), None, forecast, previous)
            return float(nash.control.green_times[f"{signal}_p1"])

        p1_free = solve_with(None)
        p1_trusted = solve_with(6.0)
        self.assertGreater(
            p1_free, ref + 6.0 + 1e-9,
            msg="unbounded huge price must drag argmin far above ref (sanity)",
        )
        self.assertLessEqual(
            p1_trusted, ref + 6.0 + 1e-9,
            msg="trust region must keep priced argmin within the FD neighborhood",
        )

    def test_local_green_costs_ignores_active_price(self):
        # d_local은 비가격 own-TTS 기울기여야 한다(순환 방지) — 가격이 설정돼 있어도
        # local_green_costs 결과는 가격 없음과 동일해야 하고, 가격 상태는 복원돼야 한다.
        cfg = _build_cfg()
        forecast = _demand(cfg)
        previous = ControlAction.fixed(cfg)
        state = TrafficState.initial(cfg)
        signal = cfg.network.signals[0]
        requests = {signal: [40.0, 56.0, 70.0]}

        follower = WuFaithfulFollower(cfg)
        clean = follower.local_green_costs(requests, state, previous, forecast[0])

        follower.signal_marginal_price = {signal: 100.0}
        follower.signal_marginal_price_ref = {signal: 56.0}
        priced_state = follower.local_green_costs(requests, state, previous, forecast[0])

        self.assertEqual(follower.signal_marginal_price, {signal: 100.0},
                         msg="price must be restored after local_green_costs")
        for a, b in zip(clean[signal], priced_state[signal]):
            self.assertAlmostEqual(a, b, places=12,
                                   msg="local_green_costs must ignore the active price")


class TestSignalMarginalPriceController(unittest.TestCase):
    """P-Stack refresh 트리거: 최초/케이던스/event-trigger에서만 재계산, 사이엔 hold."""

    def _setup(self):
        # cadence가 안 걸리게 refresh 주기를 크게 — event-trigger 논리만 분리 검증.
        cfg = _build_cfg({"leader_global_refresh_sec": 1.0e9})
        controller = StackelbergWuMeteredController(cfg)
        # 기본 OFF(STOP 관례) — refresh 로직 검증을 위해 명시 opt-in.
        controller.signal_price_enabled = True
        state = TrafficState.initial(cfg)
        # step 0은 cadence가 무조건 활성이라 step 1 시점으로 옮긴다.
        state.time_sec = float(cfg.simulation.control_interval)
        forecast = _demand(cfg)
        previous = ControlAction.fixed(cfg)
        return cfg, controller, state, forecast, previous

    def test_refresh_then_hold_then_event_trigger(self):
        cfg, controller, state, forecast, previous = self._setup()
        follower = controller.nash_solver

        # (1) 가격 미존재 → 최초 refresh.
        controller._maybe_refresh_signal_prices(state, forecast, previous)
        self.assertIsNotNone(follower.signal_marginal_price)
        self.assertEqual(set(follower.signal_marginal_price), set(cfg.network.signals))
        self.assertEqual(controller._signal_price_meta["wu_b2_price_refreshed"], 1.0)
        count_1 = controller._signal_price_refresh_count

        # (2) 운영점 불변 → hold(재계산 없음).
        controller._maybe_refresh_signal_prices(state, forecast, previous)
        self.assertEqual(controller._signal_price_meta["wu_b2_price_refreshed"], 0.0)
        self.assertEqual(controller._signal_price_refresh_count, count_1)

        # (3) 운영점이 threshold 이상 이동 → event-trigger 재선형화.
        signal = cfg.network.signals[0]
        moved = previous.copy()
        base_p1 = float(moved.green_times.get(f"{signal}_p1", 56.0))
        moved.green_times[f"{signal}_p1"] = base_p1 + 10.0
        moved.green_times[f"{signal}_p2"] = (
            cfg.network.effective_green_total - (base_p1 + 10.0)
        )
        controller._maybe_refresh_signal_prices(state, forecast, moved)
        self.assertEqual(controller._signal_price_meta["wu_b2_price_refreshed"], 1.0)
        self.assertEqual(controller._signal_price_refresh_count, count_1 + 1)
        self.assertAlmostEqual(
            follower.signal_marginal_price_ref[signal],
            min(base_p1 + 10.0, controller._signal_price_p1_bounds()[1]),
            places=9,
        )

    def test_disabled_gate_clears_prices(self):
        cfg, controller, state, forecast, previous = self._setup()
        follower = controller.nash_solver
        controller._maybe_refresh_signal_prices(state, forecast, previous)
        self.assertIsNotNone(follower.signal_marginal_price)

        controller.signal_price_enabled = False
        controller._maybe_refresh_signal_prices(state, forecast, previous)
        self.assertIsNone(follower.signal_marginal_price)
        self.assertEqual(controller._signal_price_meta["wu_b2_price_enabled"], 0.0)


if __name__ == "__main__":
    unittest.main()
