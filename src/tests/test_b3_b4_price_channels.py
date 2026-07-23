# B3(metering/VSL 가격 포팅)·B4(barrier 가격): 휴면=비트동일, 부호 반응, refresh 하달, barrier 계산 검증
import unittest

from src.controllers.leader import LeaderAction
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


def _setup_freeway(cfg):
    follower = WuFaithfulFollower(cfg)
    # 이 모듈은 구 B3 계보(가격이 레벨 조절 + trust/cert)의 메커니즘을 고정하는 테스트 —
    # SPLIT-PRICE(2026-07-09) 기본값이 그 분기를 우회하므로 레벨 모드를 명시한다.
    follower.metering_price_split = False
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
    return follower, state, demand, snapshot, coupling, link, owned


class TestMeteringPriceFollower(unittest.TestCase):
    def test_zero_price_matches_none_in_pfo_branch(self):
        cfg = _build_cfg()
        follower, state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg)
        _, meter_none, _ = follower._solve_freeway_agent_metered(
            link, state, coupling, demand, snapshot, None,
        )
        follower.metering_marginal_price = {r: 0.0 for r in owned}
        follower.metering_marginal_price_ref = {
            r: float(snapshot.ramp_metering.get(r, 0.0)) for r in owned
        }
        _, meter_zero, _ = follower._solve_freeway_agent_metered(
            link, state, coupling, demand, snapshot, None,
        )
        for ramp, value in meter_none.items():
            self.assertAlmostEqual(
                meter_zero[ramp], value, places=9,
                msg=f"zero metering price must not move metering ({ramp})",
            )

    def test_price_sign_pushes_release(self):
        # 큰 음수 g_ext(방류 이득) → 방류 합이, 큰 양수(방류 비쌈) → 억제되어야 한다.
        cfg = _build_cfg()
        leader = LeaderAction(0.0, 3000.0)

        def solve_with(g_ext):
            follower, state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg)
            follower.metering_marginal_price = {r: float(g_ext) for r in owned}
            follower.metering_marginal_price_ref = {
                r: float(snapshot.ramp_metering.get(r, 0.0)) for r in owned
            }
            _, meter, _ = follower._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, leader,
            )
            return sum(meter.values())

        release_cheap = solve_with(-10.0)
        release_costly = solve_with(+10.0)
        self.assertGreater(
            release_cheap, release_costly,
            msg="negative metering price must yield more release than positive",
        )

    def test_metering_trust_region_bounds_priced_choice(self):
        # B3TR: trust(frac·cap) 설정 시 거대 가격도 metering을 ref 이웃 밖으로 못 끌어낸다.
        cfg = _build_cfg()
        leader = LeaderAction(0.0, 3000.0)

        def solve_with(trust_frac):
            follower, state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg)
            refs = {
                r: 0.5 * float(cfg.network.ramp_capacity_veh_h[r]) for r in owned
            }
            snapshot = snapshot.copy()
            snapshot.ramp_metering = dict(snapshot.ramp_metering)
            snapshot.ramp_metering.update(refs)
            follower.metering_marginal_price = {r: -100.0 for r in owned}  # "방류 최대로" 극단 가격
            follower.metering_marginal_price_ref = dict(refs)
            follower.metering_marginal_price_trust_frac = trust_frac
            _, meter, _ = follower._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, leader,
            )
            return meter, refs

        meter_free, refs = solve_with(None)
        self.assertTrue(
            any(meter_free[r] > refs[r] + 0.25 * float(cfg.network.ramp_capacity_veh_h[r]) + 1e-6
                for r in refs),
            msg="unbounded huge price must drag some ramp far above ref (sanity)",
        )
        meter_tr, refs = solve_with(0.25)
        for r, ref in refs.items():
            cap = float(cfg.network.ramp_capacity_veh_h[r])
            self.assertLessEqual(
                abs(meter_tr[r] - ref), 0.25 * cap + 1e-6,
                msg=f"trust must keep {r} within frac*cap of ref",
            )

    def test_metering_trust_preserves_mobility_from_cap(self):
        # 동결 회귀(2026-07-05 사고): ref=cap일 때 반경 0.25·cap < 첫 분율 간격 0.3·cap이면
        # 최근접 이웃 보장이 없을 경우 metering이 cap에 얼어붙는다. 큰 양수 가격("조여라")
        # 하에서 cap 아래 후보로 내려갈 수 있어야 한다.
        cfg = _build_cfg()
        leader = LeaderAction(0.0, 6000.0)
        follower, state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg)
        refs = {r: float(cfg.network.ramp_capacity_veh_h[r]) for r in owned}  # ref = cap
        follower.metering_marginal_price = {r: +100.0 for r in owned}  # "방류 줄여" 극단 가격
        follower.metering_marginal_price_ref = dict(refs)
        follower.metering_marginal_price_trust_frac = 0.25
        _, meter, _ = follower._solve_freeway_agent_metered(
            link, state, coupling, demand, snapshot, leader,
        )
        self.assertTrue(
            any(meter[r] < refs[r] - 1e-6 for r in owned),
            msg=f"nearest-neighbor guarantee must allow moving below cap: {meter}",
        )

    def test_release_certificate_blocks_upward_only(self):
        # B3CERT: 미인증이면 거대 음수 가격("방류 늘려")도 ref 위로 못 끌어냄, 아래로는 자유.
        # 인증되면 위로 이동 가능(B3TR 거동).
        cfg = _build_cfg()
        leader = LeaderAction(0.0, 6000.0)

        def solve_with(cert, g_ext):
            follower, state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg)
            refs = {r: 0.5 * float(cfg.network.ramp_capacity_veh_h[r]) for r in owned}
            snapshot = snapshot.copy()
            snapshot.ramp_metering = dict(snapshot.ramp_metering)
            snapshot.ramp_metering.update(refs)
            follower.metering_marginal_price = {r: float(g_ext) for r in owned}
            follower.metering_marginal_price_ref = dict(refs)
            follower.metering_marginal_price_trust_frac = 0.25
            follower.metering_release_certified = {r: cert for r in owned}
            _, meter, _ = follower._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, leader,
            )
            return meter, refs

        meter_blocked, refs = solve_with(False, -100.0)
        for r, ref in refs.items():
            self.assertLessEqual(
                meter_blocked[r], ref + 1e-6,
                msg=f"uncertified ramp {r} must not increase release above ref",
            )
        meter_ok, refs = solve_with(True, -100.0)
        self.assertTrue(
            any(meter_ok[r] > refs[r] + 1e-6 for r in refs),
            msg="certified ramps must be able to increase release",
        )
        meter_down, refs = solve_with(False, +100.0)
        self.assertTrue(
            any(meter_down[r] < refs[r] - 1e-6 for r in refs),
            msg="tightening must remain free even when uncertified",
        )

    def test_local_metering_costs_ignores_active_price(self):
        cfg = _build_cfg()
        follower, state, demand, snapshot, coupling, link, owned = _setup_freeway(cfg)
        ramp = owned[0]
        cap = float(cfg.network.ramp_capacity_veh_h[ramp])
        requests = {ramp: [0.5 * cap, cap]}
        control = ControlAction.fixed(cfg)

        clean = follower.local_metering_costs(requests, state, control, demand)
        follower.metering_marginal_price = {ramp: 100.0}
        priced = follower.local_metering_costs(requests, state, control, demand)

        self.assertEqual(follower.metering_marginal_price, {ramp: 100.0})
        for a, b in zip(clean[ramp], priced[ramp]):
            self.assertAlmostEqual(a, b, places=12)


class TestBarrierAndRefresh(unittest.TestCase):
    @staticmethod
    def _saturated_setup(cfg):
        """barrier가 horizon 동안 살아있는 과포화 셋업.

        기본 수요(scale 1.0)는 rho_crit+20에서 시작해도 한 interval(180s) 만에 임계
        아래로 배수된다(실측 9~23 veh/km) — barrier는 예측 상태에서 평가되므로 유지
        가능한 과수요(freeway/ramp ×4)와 높은 초기 밀도가 필요하다(실측 33~93 유지)."""
        state = TrafficState.initial(cfg)
        net = cfg.network
        for link in net.freeway_links:
            n_seg = len(state.freeway_density.get(link, []))
            state.freeway_density[link] = [float(net.rho_crit) + 40.0] * n_seg
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=2.0, freeway_scale=4.0, ramp_scale=4.0),
        ).horizon(0.0, 1)
        return state, forecast

    def test_barrier_zero_when_disabled_positive_when_saturated(self):
        cfg = _build_cfg()
        controller = StackelbergWuMeteredController(cfg)
        state, forecast = self._saturated_setup(cfg)
        control = ControlAction.fixed(cfg)

        controller.barrier_price_enabled = False
        _, barrier_off = controller._predict_ttt_and_barrier(state.copy(), control, forecast)
        self.assertEqual(barrier_off, 0.0)

        controller.barrier_price_enabled = True
        _, barrier_on = controller._predict_ttt_and_barrier(state.copy(), control, forecast)
        self.assertGreater(
            barrier_on, 0.0,
            msg="saturated freeway densities must produce positive barrier",
        )

    def test_refresh_hands_metering_prices_and_barrier_changes_them(self):
        cfg = _build_cfg()
        state, forecast = self._saturated_setup(cfg)
        state.time_sec = float(cfg.simulation.control_interval)
        net = cfg.network
        previous = ControlAction.fixed(cfg)
        # metering 운영점을 cap 미만으로(유한차분 양방향 확보).
        for ramp in net.ramps:
            previous.ramp_metering[ramp] = 0.5 * float(net.ramp_capacity_veh_h[ramp])

        controller = StackelbergWuMeteredController(cfg)
        controller.metering_price_enabled = True
        controller._maybe_refresh_signal_prices(state, forecast, previous)
        follower = controller.nash_solver
        self.assertIsNotNone(follower.metering_marginal_price)
        self.assertEqual(set(follower.metering_marginal_price), set(net.ramps))
        base_prices = dict(follower.metering_marginal_price)

        controller_b4 = StackelbergWuMeteredController(cfg)
        controller_b4.metering_price_enabled = True
        controller_b4.barrier_price_enabled = True
        controller_b4._maybe_refresh_signal_prices(state, forecast, previous)
        b4_prices = dict(controller_b4.nash_solver.metering_marginal_price)

        # barrier gradient(방류↑ → 초과밀도↑ → 양수)가 최소 한 ramp의 가격을 위로 민다.
        moved_up = any(
            b4_prices[r] > base_prices[r] + 1.0e-12 for r in net.ramps
        )
        self.assertTrue(
            moved_up,
            msg=f"barrier must push some metering price upward: base={base_prices}, b4={b4_prices}",
        )


if __name__ == "__main__":
    unittest.main()
