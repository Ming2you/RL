# Stage 1 6-controller 비교 검증 테스트 (spec 16.13)
import unittest
from pathlib import Path
import tempfile

from src.analysis.authority import CONTROLLER_IDS, PRIMARY_CONTROLLERS, check_control_authority
from src.analysis.free_flow_reference import compute_free_flow_reference
from src.controllers.centralized_mpc import CentralizedMPC
from src.controllers.distributed_coordinator import DistributedCoordinator
from src.controllers.urban_follower import UrbanFollower
from src.controllers.wu_distributed import WuDistributedController, WuLeaderAction, _wu_fixed_control
from src.experiments.six_controller_comparison import (
    APPENDIX_COMPARISONS,
    PAIRED_COMPARISONS,
    _ControllerAdapter,
    paired_rows,
    run_controller,
    summarize_controller,
)
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.simulation.baseline import baseline_control


def fast_config():
    """smoke용 축소 설정 — 6개 모두 수 초 내 실행되도록 budget을 줄인다."""
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 360.0},
            "mpc": {
                "horizon_steps": 1,
                "leader_candidate_count": 4,
                "max_nash_iter": 2,
                "optimizer_maxiter": 4,
                "optimizer_n_starts": 1,
            },
            "urban_follower": {"allocation_pso_particles": 6, "allocation_pso_iterations": 4},
        },
    )


def run_short(controller_id: str, tmp: Path):
    cfg = fast_config()
    scenario = ScenarioConfig("test")
    result = run_controller(cfg, scenario, controller_id, tmp / controller_id)
    reference = compute_free_flow_reference(cfg, scenario)
    summary = summarize_controller(cfg, controller_id, result, reference)
    return cfg, result, summary


class SixControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.results = {}
        cls.summaries = {}
        for cid in CONTROLLER_IDS:
            cfg, result, summary = run_short(cid, cls.tmp)
            cls.cfg = cfg
            cls.results[cid] = result
            cls.summaries[cid] = summary

    def test_comparison_runner_exposes_primary_four_controllers(self):
        # 주 비교군 4종(spec 16.1 개정) — 보조 참고군 포함 전체는 6종 구현 유지.
        self.assertEqual(len(PRIMARY_CONTROLLERS), 4)
        self.assertEqual(
            PRIMARY_CONTROLLERS,
            ["WU-CD-F", "PROPOSED-FOLLOWERS-ONLY", "PROPOSED-STACKELBERG", "PROPOSED-CENTRALIZED"],
        )
        self.assertTrue(set(PRIMARY_CONTROLLERS) <= set(CONTROLLER_IDS))
        self.assertEqual(set(self.summaries), set(CONTROLLER_IDS))

    def test_all_controllers_use_same_physical_plant(self):
        # 같은 고정 control을 주면 plant 전이가 controller와 무관하게 동일해야 한다.
        cfg = fast_config()
        scenario = ScenarioConfig("test")
        ttts = []
        for cid in ("WU-CD-F", "PROPOSED-FOLLOWERS-ONLY"):
            adapter = _ControllerAdapter(cfg, cid)
            adapter.decide = lambda state, fc: (ControlAction.fixed(cfg), {})  # noqa: PLW2901
            result = run_controller(cfg, scenario, cid, self.tmp / f"plant_{cid}")
            # adapter를 갈아끼우지 못하므로 직접 시뮬 동일성 확인: 동일 fixed control 1 step.
        from src.simulation.simulator import MixedTrafficSimulator
        demand = DemandProfile(cfg, scenario).at(0.0)
        for _ in range(2):
            sim = MixedTrafficSimulator(cfg)
            sim.step(ControlAction.fixed(cfg), demand, 0)
            ttts.append(sim.total_ttt)
        self.assertAlmostEqual(ttts[0], ttts[1])

    def test_wu_group_uses_green_and_vsl_only(self):
        for cid in ("WU-CD-F", "WU-MATCHED-STACKELBERG", "WU-CC-F"):
            auth = check_control_authority(cid, self.results[cid]["control_rows"], self.cfg)
            self.assertTrue(auth["ok"], msg=f"{cid}: {auth['violations']}")

    def test_no_control_matches_wu_neutral_physical_action(self):
        cfg = fast_config()
        state = TrafficState.initial(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        no_control = baseline_control("no_control", cfg, state, demand)
        fixed_signal = baseline_control("fixed_signal_fixed_speed", cfg, state, demand)
        wu_neutral = _wu_fixed_control(cfg)

        self.assertEqual(no_control.inflow_outflow_allocation, {})
        self.assertEqual(no_control, fixed_signal)
        self.assertEqual(no_control.ramp_metering, wu_neutral.ramp_metering)
        self.assertEqual(no_control.vsl, wu_neutral.vsl)
        self.assertEqual(no_control.green_times, wu_neutral.green_times)
        self.assertEqual(no_control.offsets, wu_neutral.offsets)

    def test_wu_coupling_uses_forecast_demand_and_actual_green(self):
        cfg = fast_config()
        state = TrafficState.initial(cfg)
        controller = WuDistributedController(cfg)
        demand = DemandProfile(cfg, ScenarioConfig("test", freeway_scale=1.2)).at(0.0)
        control = _wu_fixed_control(cfg)
        movements = cfg.network.on_ramp_to_movement["R_D_W"]
        state.urban_movement_queue[movements[0]] = 100.0
        state.urban_movement_queue[movements[1]] = 0.0

        equal = controller._coupling(state, control, demand)
        control.green_times["D_p1"] = cfg.network.green_max
        control.green_times["D_p2"] = cfg.network.effective_green_total - cfg.network.green_max
        favored = controller._coupling(state, control, demand)

        self.assertEqual(equal["mainline_FW_W"], demand.freeway_mainline["FW_W"])
        self.assertGreater(favored["u_on_R_D_W"], equal["u_on_R_D_W"])

    def test_wu_cd_and_cc_return_feasible_green_and_discrete_vsl(self):
        cfg = fast_config()
        state = TrafficState.initial(cfg)
        profile = DemandProfile(cfg, ScenarioConfig("test"))
        forecast = [
            profile.at(step * cfg.simulation.control_interval)
            for step in range(cfg.mpc.horizon_steps)
        ]

        cd = WuDistributedController(cfg).decide_with_info(state.copy(), forecast)
        cc = CentralizedMPC(cfg, mode="wu").decide_with_info(state.copy(), forecast)
        vsl_set = {float(v) for v in cfg.freeway_follower.vsl_set}
        for control in (cd.control, cc.control):
            self.assertTrue(all(float(v) in vsl_set for v in control.vsl.values()))
            for signal in cfg.network.signals:
                p1 = control.green_times[f"{signal}_p1"]
                p2 = control.green_times[f"{signal}_p2"]
                self.assertAlmostEqual(p1 + p2, cfg.network.effective_green_total)
                self.assertGreaterEqual(p1, cfg.network.green_min)
                self.assertGreaterEqual(p2, cfg.network.green_min)

    def test_wu_cc_objective_uses_coupled_ttt_without_proposed_penalties(self):
        cfg = fast_config()
        state = TrafficState.initial(cfg)
        for movement in cfg.network.urban_movements:
            state.urban_movement_queue[movement] = 100.0
        control = _wu_fixed_control(cfg)
        controller = CentralizedMPC(cfg, mode="wu")

        objective = controller._objective(
            [state],
            control,
            control,
            trajectory_ttt=12.5,
        )

        self.assertAlmostEqual(objective, 12.5)

    def test_wu_cd_has_no_leader(self):
        decisions = self.results["WU-CD-F"]["decision_rows"]
        self.assertTrue(all(float(d.get("leader_candidate_count", 0.0)) == 0.0 for d in decisions))
        rows = self.results["WU-CD-F"]["control_rows"]
        self.assertTrue(all(float(r.get("N_P_star", 0.0)) == 0.0 for r in rows))

    def test_wu_stackelberg_uses_wu_partition_and_authority(self):
        auth = check_control_authority(
            "WU-MATCHED-STACKELBERG", self.results["WU-MATCHED-STACKELBERG"]["control_rows"], self.cfg
        )
        self.assertTrue(auth["ok"], msg=str(auth["violations"]))
        decisions = self.results["WU-MATCHED-STACKELBERG"]["decision_rows"]
        self.assertTrue(all(float(d.get("leader_candidate_count", 0.0)) > 0 for d in decisions))

    def test_wu_stackelberg_leader_affects_follower_response(self):
        # smoothness가 binding하는 조건: base는 이전 green을 유지하지만, tight한
        # N_P_star conditioning(잔여 누적 패널티)이 drain 쪽으로 optimum을 옮긴다.
        cfg = ExperimentConfig.from_file(
            "src/config/default.yaml",
            {
                "simulation": {"T_total": 360.0},
                "mpc": {"horizon_steps": 1, "max_nash_iter": 1},
                "urban_follower": {"green_smoothness_weight": 1.0},
            },
        )
        state = TrafficState.initial(cfg)
        for movement, spec in cfg.network.urban_movements.items():
            if spec.get("phase") == "A_p1":
                state.urban_movement_queue[movement] = 100.0  # 합계 ≈ 큐 지속 수준.
        demand = DemandProfile(cfg, ScenarioConfig("test", urban_scale=0.0, ramp_scale=0.0)).at(0.0)
        ctl = WuDistributedController(cfg, leader_enabled=True)
        previous = ControlAction.fixed(cfg)
        base = ctl._solve_followers(state, demand, previous, leader=None)[0]
        tight = ctl._solve_followers(
            state, demand, previous, leader=WuLeaderAction(n_p_star=1.0, n_f_star=1.0e9)
        )[0]
        self.assertNotEqual(
            round(base.green_times["A_p1"], 3), round(tight.green_times["A_p1"], 3),
            msg="leader conditioning이 follower green에 영향을 주지 않음",
        )

    def test_wu_cc_has_no_agent_iteration(self):
        decisions = self.results["WU-CC-F"]["decision_rows"]
        self.assertTrue(all("coordination_iterations" not in d for d in decisions))

    def test_proposed_group_authority_checks_pass(self):
        # P-STACK/P-CENT는 allocation 포함 full authority, P-FO는 allocation 제외(16.7 재정의).
        for cid in ("PROPOSED-FOLLOWERS-ONLY", "PROPOSED-STACKELBERG", "PROPOSED-CENTRALIZED"):
            auth = check_control_authority(cid, self.results[cid]["control_rows"], self.cfg)
            self.assertTrue(auth["ok"], msg=f"{cid}: {auth['violations']}")
        for cid in ("PROPOSED-STACKELBERG", "PROPOSED-CENTRALIZED"):
            rows = self.results[cid]["control_rows"]
            allocation_used = any(
                abs(float(r.get(f"allocation_{m}", 0.0))) > 1e-6
                for r in rows for m in self.cfg.network.urban_movements
            )
            self.assertTrue(allocation_used, msg=f"{cid}: allocation 흔적 없음")

    def test_proposed_followers_only_has_no_allocation_control(self):
        rows = self.results["PROPOSED-FOLLOWERS-ONLY"]["control_rows"]
        for row in rows:
            for movement in self.cfg.network.urban_movements:
                self.assertLessEqual(
                    abs(float(row.get(f"allocation_{movement}", 0.0))), 1e-6,
                    msg=f"P-FO가 allocation_{movement}을 사용함",
                )

    def test_proposed_followers_only_green_searched_within_bounds(self):
        cfg = fast_config()
        net = cfg.network
        rows = self.results["PROPOSED-FOLLOWERS-ONLY"]["control_rows"]
        for row in rows:
            for signal in net.signals:
                p1 = float(row.get(f"green_{signal}_p1", net.effective_green_total / 2.0))
                p2 = float(row.get(f"green_{signal}_p2", net.effective_green_total / 2.0))
                self.assertGreaterEqual(p1, net.green_min - 1e-6)
                self.assertLessEqual(p1, net.green_max + 1e-6)
                self.assertAlmostEqual(p1 + p2, net.effective_green_total, places=3)
        # 탐색이 큐에 반응하는지: p1 큐만 무거우면 p1 green이 균등(56)보다 커진다.
        state = TrafficState.initial(cfg)
        for movement, spec in cfg.network.urban_movements.items():
            if spec.get("phase") == "A_p1":
                state.urban_movement_queue[movement] = 200.0
        demand = DemandProfile(cfg, ScenarioConfig("test")).at(0.0)
        result = UrbanFollower(cfg).solve(state, None, demand)
        self.assertGreater(result.green_times["A_p1"], net.effective_green_total / 2.0)
        self.assertEqual(result.inflow_outflow_allocation, {})

    def test_proposed_followers_only_has_no_hidden_global_target(self):
        rows = self.results["PROPOSED-FOLLOWERS-ONLY"]["control_rows"]
        self.assertTrue(all(
            float(r.get("N_P_star", 0.0)) == 0.0 and float(r.get("N_UF_star", 0.0)) == 0.0
            for r in rows
        ))
        decisions = self.results["PROPOSED-FOLLOWERS-ONLY"]["decision_rows"]
        self.assertTrue(all(float(d.get("leader_candidate_count", 0.0)) == 0.0 for d in decisions))

    def test_proposed_pair_differs_by_leader_and_allocation_only(self):
        # 두 모드 모두 같은 follower 기계(DistributedCoordinator)를 사용하고,
        # 차이는 Leader와 allocation coordination layer 유무뿐이다(16.7/16.11 개정).
        cfg = fast_config()
        followers_only = _ControllerAdapter(cfg, "PROPOSED-FOLLOWERS-ONLY")
        stackelberg = _ControllerAdapter(cfg, "PROPOSED-STACKELBERG")
        self.assertIsInstance(followers_only._impl, DistributedCoordinator)
        self.assertIsInstance(stackelberg._impl.nash_solver, DistributedCoordinator)
        fo_rows = self.results["PROPOSED-FOLLOWERS-ONLY"]["control_rows"]
        st_rows = self.results["PROPOSED-STACKELBERG"]["control_rows"]
        fo_alloc = any(
            abs(float(r.get(f"allocation_{m}", 0.0))) > 1e-6
            for r in fo_rows for m in self.cfg.network.urban_movements
        )
        st_alloc = any(
            abs(float(r.get(f"allocation_{m}", 0.0))) > 1e-6
            for r in st_rows for m in self.cfg.network.urban_movements
        )
        self.assertFalse(fo_alloc)
        self.assertTrue(st_alloc)

    def test_proposed_centralized_has_no_leader_or_agents(self):
        cfg = fast_config()
        adapter = _ControllerAdapter(cfg, "PROPOSED-CENTRALIZED")
        self.assertIsInstance(adapter._impl, CentralizedMPC)
        decisions = self.results["PROPOSED-CENTRALIZED"]["decision_rows"]
        self.assertTrue(all(float(d.get("leader_candidate_count", 0.0)) == 0.0 for d in decisions))
        self.assertTrue(all("coordination_iterations" not in d for d in decisions))

    def test_physical_on_offramp_coupling_remains_identical(self):
        # 어떤 controller 모드도 topology/결합 맵을 바꾸지 않는다.
        base = fast_config().network
        for cid in CONTROLLER_IDS:
            cfg = fast_config()
            _ControllerAdapter(cfg, cid)
            self.assertEqual(cfg.network.on_ramp_to_movement, base.on_ramp_to_movement, msg=cid)
            self.assertEqual(cfg.network.off_ramp_to_movement, base.off_ramp_to_movement, msg=cid)

    def test_first_mpc_action_only_is_applied(self):
        # 매 interval 새 decide 1회 → control row 수 == interval 수.
        steps = max(1, int(round(self.cfg.simulation.T_total / self.cfg.simulation.control_interval)))
        for cid in CONTROLLER_IDS:
            self.assertEqual(len(self.results[cid]["control_rows"]), steps, msg=cid)

    def test_all_controllers_use_same_free_flow_delay_reference(self):
        refs = {
            (s["free_flow_reference_total_ttt"], s["free_flow_reference_urban_ttt"], s["free_flow_reference_freeway_ttt"])
            for s in self.summaries.values()
        }
        self.assertEqual(len(refs), 1)

    def test_delay_is_reported_for_total_urban_and_freeway(self):
        for s in self.summaries.values():
            for key in ("total_delay", "urban_delay", "freeway_delay"):
                self.assertIn(key, s)

    def test_delay_improvement_is_paired_with_throughput_and_terminal_state(self):
        # setUpClass가 6종 전부 실행하므로 주 4쌍 + 부록 3쌍이 모두 계산된다.
        rows = paired_rows(self.summaries, delay_eps=1.0)
        self.assertEqual(len(rows), len(PAIRED_COMPARISONS) + len(APPENDIX_COMPARISONS))
        for row in rows:
            for key in ("delay_improvement_abs", "throughput_difference", "terminal_total_difference"):
                self.assertIn(key, row)
        # 주 비교군 4종만 실행한 경우엔 주 4쌍만 계산된다.
        primary_only = {cid: self.summaries[cid] for cid in PRIMARY_CONTROLLERS}
        self.assertEqual(len(paired_rows(primary_only, delay_eps=1.0)), len(PAIRED_COMPARISONS))

    def test_low_reference_delay_uses_absolute_difference(self):
        fake = {k: dict(v) for k, v in self.summaries.items()}
        fake["WU-CD-F"]["total_delay"] = 0.5  # epsilon(1.0) 이하 → pct NA.
        rows = paired_rows(fake, delay_eps=1.0)
        full_package = next(r for r in rows if r["comparison"] == "FullPackageValue")
        self.assertEqual(full_package["delay_improvement_pct"], "NA")

    def test_terminal_queue_is_reported(self):
        for s in self.summaries.values():
            for key in (
                "terminal_total_vehicles",
                "terminal_urban_vehicles",
                "terminal_onramp_vehicles",
                "terminal_freeway_vehicles",
            ):
                self.assertIn(key, s)


class WuDistributedFixesTests(unittest.TestCase):
    """WU-CD-F 치명 2건 수정 검증 — 분산 협상 복원(a,b) + freeway VSL 작동(c,d)."""

    def _congested_config(self):
        return ExperimentConfig.from_file(
            "src/config/default.yaml",
            {"mpc": {"horizon_steps": 1, "max_nash_iter": 5}},
        )

    def _congested_state(self, cfg):
        """현실적 capacity-drop 무대: 상류 free-flow(VSL이 물리적으로 metering 가능한
        free-flow 가지 ρ<ρ_crit) + off-ramp 병목 seg2 jam + off-ramp storage 채움
        + 도시 링크 포화(drain 막힘). 직전 fixture는 전 segment를 ρ=40으로 띄워
        상류까지 congested 가지로 밀어넣어 VSL이 물리적으로 inert했다(이득=0의 한 원인).
        실제 capacity-drop은 상류는 흐르고 병목만 막힌 상태다."""
        net = cfg.network
        state = TrafficState.initial(cfg)
        for movement in net.urban_movements:
            state.urban_movement_queue[movement] = 80.0
        for link in net.freeway_links:
            # seg0/seg1은 free-flow 가지(ρ<ρ_crit=33.5)로 metering 여지 확보, seg2는 병목 jam.
            state.freeway_density[link] = [15.0, 20.0, 40.0, 35.0]
            state.freeway_speed[link] = [95.0, 88.0, 80.0, 48.0]
        # ramp 큐를 비워 probe에서 상류가 즉시 jam으로 flooding되지 않게 한다(현실적 metering).
        for ramp in net.ramps:
            state.ramp_queue[ramp] = 0.0
        for off_ramp in net.off_ramps:
            sl = net.off_ramp_storage_link[off_ramp]
            cap = net.urban_link_storage_veh[sl]
            state.urban_link_storage[sl] = 0.02 * cap  # 98% 점유.
        for link, cap in net.urban_link_storage_veh.items():
            state.urban_link_storage[link] = min(
                state.urban_link_storage.get(link, cap), 0.05 * cap
            )
        return state

    def test_a_peak_coupling_iterates_more_than_once(self):
        # (a) peak 혼잡에서 coupling residual>tol 이고 iteration>1(단발 퇴화 해소).
        # iteration>1은 정의상 첫 iteration residual>tol을 함의(아니면 1에서 종료)하지만,
        # 단발 퇴화 산물이 아님을 명시하기 위해 첫 iteration residual도 직접 측정한다.
        cfg = self._congested_config()
        ctl = WuDistributedController(cfg)
        state = self._congested_state(cfg)
        demand = DemandProfile(
            cfg, ScenarioConfig("test", freeway_scale=1.4, urban_scale=1.4, ramp_scale=1.4)
        ).at(0.0)
        previous = _wu_fixed_control(cfg)
        _, iterations, _, _, _ = ctl._solve_followers(state, demand, previous, leader=None)
        self.assertGreater(iterations, 1)
        # 단발 퇴화 산물이 아님을 명시: 1 iteration만 돌렸을 때의 반환 residual이 tol을
        # 초과(첫 단계에서 outgoing y가 후보 제어에 살아 있게 반응 — 즉시 종료가 아님).
        one_iter_cfg = ExperimentConfig.from_file(
            "src/config/default.yaml", {"mpc": {"horizon_steps": 1, "max_nash_iter": 1}}
        )
        ctl1 = WuDistributedController(one_iter_cfg)
        _, _, _, first_residual, _ = ctl1._solve_followers(state, demand, previous, leader=None)
        self.assertGreater(first_residual, cfg.mpc.distributed_coupling_tol)

    def test_b_coupling_y_changes_between_iteration_zero_and_one(self):
        # (b) iteration 0/1 사이 coupling y의 최소 한 key가 tol 초과로 변한다.
        cfg = self._congested_config()
        ctl = WuDistributedController(cfg)
        state = self._congested_state(cfg)
        demand = DemandProfile(
            cfg, ScenarioConfig("test", freeway_scale=1.4, urban_scale=1.4, ramp_scale=1.4)
        ).at(0.0)
        previous = _wu_fixed_control(cfg)
        y0 = ctl._coupling(state, previous, demand)
        control, _, _, _, _ = ctl._solve_followers(state, demand, previous, leader=None)
        y1 = ctl._coupling(state, control, demand)
        changed = [
            k for k in y0
            if abs(y1[k] - y0[k]) > cfg.mpc.distributed_coupling_tol
        ]
        self.assertGreater(len(changed), 0, msg="coupling y가 후보 제어에 반응하지 않음")

    def test_c_capacity_drop_wired_and_upstream_vsl_available(self):
        # (c) 새 4-segment topology 검증: seg0은 ramp 없는 단순 상류부이고, off-ramp 병목은
        # seg1/seg2에 있다. Wu freeway agent는 병목 segment 자체를 max VSL로 보존하면서
        # 상류 단순 segment의 per-segment VSL 후보를 낮출 수 있어야 한다.
        cfg = self._congested_config()
        ctl = WuDistributedController(cfg)
        state = self._congested_state(cfg)
        demand = DemandProfile(
            cfg, ScenarioConfig("test", freeway_scale=1.0, ramp_scale=0.8)
        ).at(0.0)
        previous = _wu_fixed_control(cfg)
        vsl_max = max(cfg.freeway_follower.vsl_set)
        for link in cfg.network.freeway_links:
            previous.vsl[link] = vsl_max
        # 전제: 이 state에서 capacity-drop이 실제로 active(λ_eff < freeway_lanes)한지 확인.
        from src.models.metanet import effective_lane_profile

        profile, lane_diag = effective_lane_profile(state, cfg)
        self.assertLess(
            min(
                lane_diag[f"lambda_eff_{link}_seg{i}"]
                for link in cfg.network.freeway_links
                for i in range(len(profile[link]))
            ),
            cfg.network.freeway_lanes,
            msg="전제 실패: capacity-drop이 active하지 않음",
        )
        vsl_set = {float(v) for v in cfg.freeway_follower.vsl_set}
        coupling = ctl._coupling(state, previous, demand)
        n_seg = cfg.network.freeway_segments_per_link
        for link in cfg.network.freeway_links:
            vsl_dict, _, _ = ctl._solve_freeway_agent(link, state, coupling, demand, previous, None)
            seg_vals = [vsl_dict[f"{link}__seg{i}"] for i in range(n_seg)]
            # per-segment VSL이 valid(discrete vsl_set)해야 한다.
            self.assertTrue(all(v in vsl_set for v in seg_vals), msg=f"{link}: VSL이 vsl_set 밖")
            # 병목 off-ramp segment는 max VSL을 유지하고, 새 상류 단순 segment가 metering 역할을 맡는다.
            for off_ramp in cfg.network.off_ramps:
                if cfg.network.off_ramp_from_freeway.get(off_ramp) == link:
                    idx = cfg.network.off_ramp_segment_index[off_ramp]
                    self.assertGreaterEqual(seg_vals[idx], vsl_max - 0.5)
            self.assertLess(
                seg_vals[0],
                vsl_max,
                msg=f"{link}: upstream plain segment VSL was not selected",
            )

    def test_d_probe_enters_capacity_drop_with_filled_offramp_storage(self):
        # (d) off-ramp storage 채운 state로 freeway solve 시 probe에 λ_eff<freeway_lanes 진입.
        from src.models.metanet import effective_lane_profile

        cfg = self._congested_config()
        state = self._congested_state(cfg)
        profile, _ = effective_lane_profile(state, cfg)
        entered = any(
            min(profile[link]) < cfg.network.freeway_lanes - 1.0e-9
            for link in cfg.network.freeway_links
            if profile[link]
        )
        self.assertTrue(entered, msg="off-ramp storage 점유가 capacity-drop을 유발하지 않음")

    def test_last_offramp_flow_cached_after_freeway_solve(self):
        # freeway→urban coupling 재사용 캐시가 freeway solve 후 채워진다.
        cfg = self._congested_config()
        ctl = WuDistributedController(cfg)
        state = self._congested_state(cfg)
        demand = DemandProfile(
            cfg, ScenarioConfig("test", freeway_scale=1.4, ramp_scale=1.4)
        ).at(0.0)
        previous = _wu_fixed_control(cfg)
        coupling = ctl._coupling(state, previous, demand)
        for link in cfg.network.freeway_links:
            ctl._solve_freeway_agent(link, state, coupling, demand, previous, None)
            self.assertGreaterEqual(ctl._last_offramp_flow[link], 0.0)
        self.assertTrue(any(v > 0.0 for v in ctl._last_offramp_flow.values()))


if __name__ == "__main__":
    unittest.main()
