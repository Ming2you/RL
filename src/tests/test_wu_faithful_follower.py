# WuFaithfulFollower 국소 freeway 모델의 plant 정합 + Wu-metered leader prefilter action-분별 회귀 테스트
import sys
import unittest

from src.controllers.leader import LeaderAction
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


class TestLocalRampReleaseOrdering(unittest.TestCase):
    """Finding #3 회귀: 국소 모델 release가 유입 반영 전 reservoir 기준으로 결정돼야 한다."""

    def test_local_ramp_release_sees_pre_arrival_queue(self):
        # plant(run_coupled_interval)는 include_current_arrivals=False로 release를
        # 결정한 뒤 urban 유입을 reservoir에 적재한다. 버그 코드는 로컬 모델에서
        # 유입을 먼저 적재해 첫 substep의 ramp_queue가 approach*dt_h만큼 커진 채
        # release가 계산됐다. 여기서는 ramp_queue=0 + coupling>0으로 시작해 첫
        # _local_ramp_release 호출이 ramp_queue=0을 봐야 함을 검증한다(버그 코드는
        # 600 veh/h * T_f_h ≈ 1.67 veh > 0이 기록돼 실패한다 — 판별력 확보).
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {"horizon_steps": 1},
                "freeway_follower": {
                    "freeway_prediction_horizon_steps": 1,
                    "vsl_sequence_search": False,
                },
            },
        )
        follower = WuFaithfulFollower(cfg)
        link = cfg.network.freeway_links[0]
        owned = [r for r in cfg.network.ramps if cfg.network.ramp_to_freeway.get(r) == link]
        self.assertTrue(owned)

        state = TrafficState.initial(cfg)
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 0.0
        coupling = {f"u_on_{ramp}": 600.0 for ramp in owned}
        demand = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)[0]
        previous = ControlAction.fixed(cfg)

        recorded = []
        original = follower._local_ramp_release

        def spy(link_arg, rhos, ramp_queue, candidate_control, demand_arg):
            recorded.append({r: float(v) for r, v in dict(ramp_queue).items()})
            return original(link_arg, rhos, ramp_queue, candidate_control, demand_arg)

        follower._local_ramp_release = spy
        try:
            follower._solve_freeway_agent_local(link, state, coupling, demand, previous)
        finally:
            follower._local_ramp_release = original

        self.assertTrue(recorded, "no _local_ramp_release call recorded")
        first_call_queue = recorded[0]
        for ramp in owned:
            self.assertAlmostEqual(
                first_call_queue.get(ramp, 0.0),
                0.0,
                places=9,
                msg=(
                    f"first-substep release for {ramp} must be decided on the "
                    "pre-arrival reservoir (plant include_current_arrivals=False)"
                ),
            )


class TestWuMeteredProxyActionAware(unittest.TestCase):
    """Finding #5 회귀: prefilter proxy가 N_UF_star에 따라 다른 objective를 내야 한다."""

    def test_proxy_score_distinguishes_nuf_candidates(self):
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 180.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
            },
        )
        controller = StackelbergWuMeteredController(cfg)
        state = TrafficState.initial(cfg)
        # metering 차이가 rollout에 드러나도록 ramp queue를 수동 주입(혼잡 state).
        for ramp in cfg.network.ramps:
            state.ramp_queue[ramp] = 60.0
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.2, freeway_scale=1.0, ramp_scale=1.2),
        ).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)

        low = controller._proxy_score_candidate(
            0, LeaderAction(0.0, 200.0), state.copy(), forecast, previous,
        )
        high = controller._proxy_score_candidate(
            1, LeaderAction(0.0, 2800.0), state.copy(), forecast, previous,
        )

        for row in (low, high):
            for key in (
                "index", "N_P_star", "N_UF_star", "objective",
                "base", "follower_ttt", "spillback_violation",
            ):
                self.assertIn(key, row)
        self.assertGreater(
            abs(low["objective"] - high["objective"]),
            1.0e-9,
            "proxy objective must differ across N_UF_star candidates (action-blind regression)",
        )


class _LeaderStub:
    """N_P_star/N_UF_star float 속성만 가진 leader 대체(테스트 전용)."""

    def __init__(self, n_p_star: float = 50.0, n_uf_star: float = 0.0):
        self.N_P_star = float(n_p_star)
        self.N_UF_star = float(n_uf_star)


def _dual_test_config() -> ExperimentConfig:
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 180.0},
            "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
            "freeway_follower": {
                "freeway_prediction_horizon_steps": 1,
                "vsl_sequence_search": False,
            },
        },
    )


class TestLambdaDualIntegralUpdate(unittest.TestCase):
    """A1+A2 회귀: λ step 간 적분 갱신(비음수·cap·방향) + commit green == 합의 green."""

    def test_lambda_update_nonnegative_cap_direction(self):
        # A1 — λ_next = clip(λ + gain·(Σnin − target), 0, cap) 거동 검증.
        follower = WuFaithfulFollower(_dual_test_config())
        # 방향: Σnin > target(유입 과다) → λ 증가(억제 강화).
        self.assertAlmostEqual(
            follower._lambda_np_update(1.0, 200.0, 100.0),
            1.0 + follower.lambda_np_step_gain * 100.0,
            places=12,
        )
        self.assertGreater(follower._lambda_np_update(1.0, 200.0, 100.0), 1.0)
        # A1 핵심: target > Σnin이고 λ=0이면 0 유지(음수 λ 금지 — 유입 강제 보상 없음).
        self.assertEqual(follower._lambda_np_update(0.0, 50.0, 200.0), 0.0)
        # target > Σnin, λ>0이면 0을 향해 내려가되 음수로는 안 간다.
        self.assertEqual(follower._lambda_np_update(0.5, 0.0, 1.0e6), 0.0)
        # cap 초과 시 cap으로 clip.
        self.assertEqual(
            follower._lambda_np_update(follower.lambda_np_cap, 1.0e9, 0.0),
            follower.lambda_np_cap,
        )

    def test_commit_green_equals_last_consensus_sweep(self):
        # A2 — 이분법·commit sweep 폐지: 반환 green이 마지막 Jacobi 합의 sweep의 p1과
        # 일치해야 하고, _sum_nin_at_lambda 경유 urban solve가 아예 없어야 한다
        # (_np_feasible_range는 _agent_net_inflow_veh를 직접 쓰므로 여기 안 잡힌다).
        cfg = _dual_test_config()
        follower = WuFaithfulFollower(cfg)
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.2, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)

        calls = []  # (caller, signal, p1)
        original = follower._solve_urban_agent_local

        def spy(signal, *args, **kwargs):
            result = original(signal, *args, **kwargs)
            caller = sys._getframe(1).f_code.co_name
            calls.append((caller, signal, float(result[0])))
            return result

        follower._solve_urban_agent_local = spy
        try:
            result = follower.solve(state, _LeaderStub(), forecast, previous)
        finally:
            follower._solve_urban_agent_local = original

        self.assertTrue(calls, "no _solve_urban_agent_local call recorded")
        callers = {caller for caller, _, _ in calls}
        self.assertNotIn(
            "_sum_nin_at_lambda",
            callers,
            "bisection/commit sweep must be gone (no _sum_nin_at_lambda-driven urban solve)",
        )
        # 신호별 마지막 _solve_followers sweep의 p1 == commit된 green_times p1.
        last_p1 = {}
        for caller, signal, p1 in calls:
            if caller == "_solve_followers":
                last_p1[signal] = p1
        self.assertEqual(set(last_p1), set(cfg.network.signals))
        for signal in cfg.network.signals:
            self.assertAlmostEqual(
                float(result.control.green_times[f"{signal}_p1"]),
                last_p1[signal],
                places=9,
                msg=f"committed green for {signal} must equal the last consensus sweep",
            )

    def test_solve_does_not_mutate_persistent_lambda(self):
        # 오염 방지 — solve()는 self._lambda_P를 절대 바꾸지 않고 λ_next를 diagnostics로만 낸다.
        cfg = _dual_test_config()
        follower = WuFaithfulFollower(cfg)
        follower._lambda_P = 0.5
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.2, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)

        result = follower.solve(state, _LeaderStub(), forecast, previous)

        self.assertEqual(follower._lambda_P, 0.5)
        diag = result.control.diagnostics
        self.assertIn("wu_faithful_lambda_next", diag)
        self.assertGreaterEqual(float(diag["wu_faithful_lambda_next"]), 0.0)
        self.assertLessEqual(float(diag["wu_faithful_lambda_next"]), follower.lambda_np_cap)
        # solve에 사용된 λ는 warm-start 값 그대로여야 한다.
        self.assertAlmostEqual(float(diag["wu_faithful_lambda_P"]), 0.5, places=12)


class TestLeaderNpFeasibleRangeInterface(unittest.TestCase):
    """P0 회귀: follower가 leader_np_feasible_range를 노출해야 leader-side projection이
    무력화되지 않는다(stackelberg_mpc 1598행 getattr duck-typing)."""

    def test_leader_np_feasible_range_exists_and_returns_range(self):
        cfg = _dual_test_config()
        follower = WuFaithfulFollower(cfg)
        self.assertTrue(
            hasattr(follower, "leader_np_feasible_range"),
            "WuFaithfulFollower must expose leader_np_feasible_range "
            "(otherwise leader N_P projection silently deactivates)",
        )
        state = TrafficState.initial(cfg)
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.2, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)
        previous = ControlAction.fixed(cfg)
        # stackelberg_mpc._project_action_to_follower_feasible_np(1602행)와 동일한 호출 형태.
        sigma_min, sigma_max, diag = follower.leader_np_feasible_range(
            state.copy(), list(forecast), previous.copy()
        )
        self.assertLessEqual(float(sigma_min), float(sigma_max))
        self.assertIsInstance(diag, dict)
        # diag 값은 f"leader_{key}"로 승격돼 float() 변환되므로 float 호환이어야 한다.
        for key, value in diag.items():
            self.assertIsInstance(key, str)
            float(value)


class TestBlockedRampInflowCost(unittest.TestCase):
    """P1 회귀: freeway agent own-TTS가 reservoir 만석으로 막힌 urban 유입(가상 blocked 큐)을
    봐야 한다(count_blocked_ramp_inflow). OFF면 기존 거동과 완전 동일해야 한다."""

    def _freeway_link_inputs(self, cfg):
        link = None
        owned = []
        for cand in cfg.network.freeway_links:
            owned = [
                r for r in cfg.network.ramps
                if cfg.network.ramp_to_freeway.get(r) == cand
            ]
            if owned:
                link = cand
                break
        self.assertIsNotNone(link, "no freeway link with owned ramps")
        demand = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)[0]
        previous = ControlAction.fixed(cfg)
        return link, owned, demand, previous

    def test_saturated_reservoir_blocked_cost_visible(self):
        # reservoir 만석 + 큰 u_on coupling → 유입이 reservoir에 못 들어가 blocked 큐가
        # 쌓이고, flag ON 비용이 OFF보다 커야 한다(externality 가시화 판별력).
        cfg = _dual_test_config()
        follower = WuFaithfulFollower(cfg)
        link, owned, demand, previous = self._freeway_link_inputs(cfg)
        state = TrafficState.initial(cfg)
        for ramp in owned:
            state.ramp_queue[ramp] = float(cfg.network.ramp_queue_max_veh)
        coupling = {f"u_on_{ramp}": 5000.0 for ramp in owned}

        follower.count_blocked_ramp_inflow = False
        _, cost_off, _ = follower._solve_freeway_agent_local(
            link, state, coupling, demand, previous,
        )
        follower.count_blocked_ramp_inflow = True
        _, cost_on, _ = follower._solve_freeway_agent_local(
            link, state, coupling, demand, previous,
        )
        self.assertGreater(cost_on, cost_off)

    def test_unsaturated_reservoir_flag_has_no_effect(self):
        # 비포화(빈 reservoir + 작은 coupling)면 blocked 큐가 전혀 안 쌓여 ON/OFF 비용이
        # 완전 동일해야 한다(경부하 구조적 무영향 — sweet_128 회귀 안전장치).
        cfg = _dual_test_config()
        follower = WuFaithfulFollower(cfg)
        link, owned, demand, previous = self._freeway_link_inputs(cfg)
        state = TrafficState.initial(cfg)
        for ramp in owned:
            state.ramp_queue[ramp] = 0.0
        coupling = {f"u_on_{ramp}": 100.0 for ramp in owned}

        follower.count_blocked_ramp_inflow = False
        _, cost_off, _ = follower._solve_freeway_agent_local(
            link, state, coupling, demand, previous,
        )
        follower.count_blocked_ramp_inflow = True
        _, cost_on, _ = follower._solve_freeway_agent_local(
            link, state, coupling, demand, previous,
        )
        self.assertEqual(cost_on, cost_off)


# Step B2 앵커: production signal_marginal_price 주입이 probe(step_b) ext@w=1 argmin과 일치함을 증명.
# (원 테스트는 Codex eed5c51의 green_price API였으나, 같은 머신 A/B에서 signal_marginal_price
#  구현이 채택돼(merge 노트 §12) 그 API로 포팅. probe g_ext_i를 직접 주입 + ref=p1_0.)
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_B2_ROOT = Path(__file__).resolve().parents[2]
_B2_PROBE = _B2_ROOT / "outputs" / "_stepB_probe" / "step_b_results.json"
if not _B2_PROBE.exists():
    # 재생성본이 없는 머신은 커밋된 probe 결과 사본으로 대체(수치 동일).
    _B2_PROBE = _B2_ROOT / "2026-07-03" / "results" / "step_b_results.json"
_B2_LEGACY = (
    _B2_ROOT
    / "outputs"
    / "legacy_pstack_sweet190_7200_20260703"
    / "runs"
    / "sweet_190"
    / "LEGACY-STACKELBERG"
    / "control_timeseries.csv"
)


class _B2StubLeader:
    """leader-present 경로 확인용 최소 leader 스텁(가격 게이트 자체는 dict 설정 여부)."""

    N_P_star = 0.0


@unittest.skipUnless(
    _B2_PROBE.exists() and _B2_LEGACY.exists(),
    "step_b probe JSON 또는 legacy trace 부재 — 앵커 테스트 스킵",
)
class TestGreenPriceProbeAnchor(unittest.TestCase):
    """Step B2: follower.signal_marginal_price에 probe g_ext_i(ref=p1_0)를 주입하면 국소 green
    argmin이 probe의 ext@w=1 argmin으로 이동함을 검증(production 구현 = probe 메커니즘 동일)."""

    def test_step20_argmin_matches_probe_ext_w1(self):
        # probe replay 하네스 재사용(work/step_b·step_a와 동일 경로).
        from src.controllers.wu_faithful_follower import WuFaithfulFollower
        from src.models.demand import DemandProfile
        from src.models.state import ControlAction
        from work.step_a_oracle_probe import (
            build_cfg,
            load_legacy_controls,
            replay_to_step,
        )

        with open(_B2_PROBE) as _fh:
            probe = json.load(_fh)
        step_k = 20
        s20 = probe["steps"][str(step_k)]

        cfg, scenario = build_cfg("sweet_190")
        profile = DemandProfile(cfg, scenario)
        controls = load_legacy_controls(cfg, str(_B2_LEGACY))
        net = cfg.network
        follower = WuFaithfulFollower(cfg)

        state_k = replay_to_step(cfg, profile, controls, step_k)
        demand_k = profile.at(state_k.time_sec)

        control = ControlAction.uncontrolled(cfg)
        control.green_times = dict(controls[step_k].green_times)
        control.offsets = dict(controls[step_k].offsets)
        control.vsl = dict(controls[step_k].vsl)
        control.ramp_metering = dict(controls[step_k].ramp_metering)
        control.inflow_outflow_allocation = {}
        coupling = follower._wu._coupling(state_k, control, demand_k)
        s_eff_frozen = follower._frozen_s_eff(state_k)
        reservoir_drain = follower._frozen_reservoir_drain(state_k, control, demand_k)
        freeway_congestion = follower._frozen_freeway_congestion(state_k)
        snapshot = ControlAction(
            ramp_metering=dict(control.ramp_metering),
            vsl=dict(control.vsl),
            green_times=dict(control.green_times),
            offsets=dict(control.offsets),
            inflow_outflow_allocation={},
        )

        # probe가 계산한 per-signal g_ext_i(externality 가격)와 기준점 p1_0을 그대로 주입 —
        # production 가격항 w·g_ext·(p1−p1_ref)가 probe priced 곡선과 정확히 같은 식이 된다.
        # probe 기록은 smoothness 포함 곡선이므로 레거시 마찰 의미론(PRICE-TR OFF)으로 고정.
        follower.price_smoothness_disabled = False
        follower.signal_marginal_price = {
            sig: float(s20[sig]["g_ext_i"]) for sig in net.signals
        }
        follower.signal_marginal_price_ref = {
            sig: float(s20[sig]["p1_0"]) for sig in net.signals
        }

        for sig in net.signals:
            arr_movement = follower._per_movement_arrivals(sig, state_k, snapshot, demand_k)
            best_p1, _, _, _ = follower._solve_urban_agent_local(
                sig, state_k.copy(), coupling, arr_movement, s_eff_frozen,
                reservoir_drain, freeway_congestion, snapshot, None,
                0.0, None, 1.0, demand_k,
            )
            expect = float(s20[sig]["priced_argmin_ext"]["1.0"])
            self.assertAlmostEqual(
                best_p1, expect, places=6,
                msg=(
                    f"signal {sig}: production signal_marginal_price argmin {best_p1} != "
                    f"probe ext@w=1 argmin {expect} — B2 메커니즘 불일치"
                ),
            )

    def test_green_price_none_leaves_solve_unchanged(self):
        # signal_marginal_price=None(기본)이면 leader present여도 가격항 없음 → 기존과 동일.
        from src.controllers.wu_faithful_follower import WuFaithfulFollower
        from src.models.demand import DemandProfile
        from src.models.state import ControlAction
        from work.step_a_oracle_probe import (
            build_cfg,
            load_legacy_controls,
            replay_to_step,
        )

        step_k = 20
        cfg, scenario = build_cfg("sweet_190")
        profile = DemandProfile(cfg, scenario)
        controls = load_legacy_controls(cfg, str(_B2_LEGACY))
        net = cfg.network
        follower = WuFaithfulFollower(cfg)
        state_k = replay_to_step(cfg, profile, controls, step_k)
        demand_k = profile.at(state_k.time_sec)
        control = ControlAction.uncontrolled(cfg)
        control.green_times = dict(controls[step_k].green_times)
        control.offsets = dict(controls[step_k].offsets)
        control.vsl = dict(controls[step_k].vsl)
        control.ramp_metering = dict(controls[step_k].ramp_metering)
        control.inflow_outflow_allocation = {}
        coupling = follower._wu._coupling(state_k, control, demand_k)
        s_eff_frozen = follower._frozen_s_eff(state_k)
        reservoir_drain = follower._frozen_reservoir_drain(state_k, control, demand_k)
        freeway_congestion = follower._frozen_freeway_congestion(state_k)
        snapshot = ControlAction(
            ramp_metering=dict(control.ramp_metering),
            vsl=dict(control.vsl),
            green_times=dict(control.green_times),
            offsets=dict(control.offsets),
            inflow_outflow_allocation={},
        )
        leader = _B2StubLeader()
        # signal_marginal_price=None(기본)이면 leader present여도 국소 own-TTS argmin 그대로.
        for sig in net.signals:
            arr_movement = follower._per_movement_arrivals(sig, state_k, snapshot, demand_k)
            follower.signal_marginal_price = None
            p1_none, _, _, _ = follower._solve_urban_agent_local(
                sig, state_k.copy(), coupling, arr_movement, s_eff_frozen,
                reservoir_drain, freeway_congestion, snapshot, leader,
                0.0, None, 1.0, demand_k,
            )
            # PFO(leader=None)와도 동일해야 한다(price 게이트가 leader에도 걸리므로).
            p1_pfo, _, _, _ = follower._solve_urban_agent_local(
                sig, state_k.copy(), coupling, arr_movement, s_eff_frozen,
                reservoir_drain, freeway_congestion, snapshot, None,
                0.0, None, 1.0, demand_k,
            )
            self.assertAlmostEqual(p1_none, p1_pfo, places=9)


if __name__ == "__main__":
    unittest.main()
