# RL leader 환경 래퍼(Phase 0, 2026-07-22) — budget 액션 → follower 실행 → plant 전진
"""Gym형 환경: RL leader가 저차원 (N_P, N_UF) budget을 출력하면 principled follower가
detailed control을 생성(feasible 사영·안전 보장)하고 plant를 1 제어스텝 전진시킨다.
reward = -Δ(step TTT). Phase 0은 budget만(선형가격/볼록가격은 이후 phase).

관측(state)은 물리량 압축 벡터(일반화 유리). action은 [-1,1]^2 → feasible budget box로 매핑.
follower = StackelbergWuMeteredController.nash_solver (논문 P-Stack의 follower 그대로).

smoke: `python rl_leader/env.py` → reset + 랜덤정책 5스텝 (Phase 0 검증 게이트).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.models.state import ExperimentConfig
from src.models.demand import DemandProfile, apply_scenario_network_overrides, load_scenarios, ScenarioConfig
from src.simulation.simulator import MixedTrafficSimulator
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.leader import LeaderAction

# Wang 본선 물리(러너 env 훅과 동일)
WANG = dict(v_free=115.0, rho_crit=31.5, metanet_tau_h=0.0056111,
            metanet_nu_km2_h=22.5, metanet_kappa_veh_km_lane=10.0,
            metanet_delta_merge=0.9)


def make_cfg(scenario, fw_buffer: int = 8) -> tuple[ExperimentConfig, dict]:
    """scenario: yaml 이름(str) 또는 시나리오 dict(도메인 랜덤화용)."""
    cfg = ExperimentConfig.from_file(str(ROOT / "src" / "config" / "default.yaml"), {})
    if isinstance(scenario, str):
        scenario = load_scenarios(str(ROOT / "src" / "config" / "scenarios.yaml"))[scenario]
    elif isinstance(scenario, dict):
        scenario = ScenarioConfig.from_mapping("random", scenario)
    cfg = apply_scenario_network_overrides(cfg, scenario)
    for k, v in WANG.items():
        setattr(cfg.network, k, v)
    cfg.network.freeway_buffer_segments = int(fw_buffer)
    cfg.network.terminal_zero_gradient = True
    return cfg, scenario


def make_random_scenario(rng, holdout_demand: float = 1.80):
    """도메인 랜덤화 시나리오 dict. hold-out 프로토콜: stressor(사고/skew) 활성 시
    수요를 holdout_demand(=1.80) 이하로 제한 → 190+stressor(=1.90)는 학습에 없음(test 전용)."""
    demand = float(rng.uniform(1.55, 2.40))
    stressor = rng.choice(["none", "skew", "incident"], p=[0.4, 0.3, 0.3])
    if stressor != "none":
        demand = min(demand, holdout_demand)
    demand *= float(rng.uniform(0.98, 1.02))  # 소음
    scen = {
        "urban_scale": demand, "freeway_scale": demand, "ramp_scale": demand,
        "incident_capacity_factor": 1.0,
        "pulse_base_scale": 0.5, "pulse_start_sec": 900.0, "pulse_rampup_sec": 360.0,
        "pulse_plateau_sec": 3600.0, "pulse_rampdown_sec": 360.0, "required": False,
    }
    if stressor == "skew":
        scen["urban_west_east_ratio"] = float(rng.uniform(1.3, 2.0))
    elif stressor == "incident":
        seg = int(rng.integers(3, 8))                        # 하류 세그(off-ramp/merge 이후)
        start = float(rng.choice([1260.0, 1800.0, 2400.0]))  # plateau 중
        dur = float(rng.choice([1200.0, 1800.0, 2400.0]))
        scen["freeway_lane_closures"] = [{
            "link": str(rng.choice(["FW_E", "FW_W"])), "segment": seg,
            "lane_loss": 1.0, "start_sec": start, "end_sec": start + dur}]
    return scen


class RLLeaderEnv:
    """Phase 0: action=(N_P,N_UF) 정규화, follower 실행, reward=-ΔTTT."""

    def __init__(self, scenario_name: str = "sweet_170_incident_w60", T_total: float = 14400.0,
                 warmup_nc_steps: int = 5,
                 np_bounds: tuple[float, float] = (0.0, 2200.0),
                 nuf_bounds: tuple[float, float] = (0.0, 6000.0),
                 scenario_dict: dict | None = None):
        self.scenario_name = scenario_name if scenario_dict is None else "random"
        self.cfg, self.scenario = make_cfg(scenario_dict if scenario_dict is not None else scenario_name)
        self.T_total = float(T_total)
        self.dt = float(self.cfg.simulation.control_interval)
        self.n_steps = int(self.T_total / self.dt)
        self.warmup = int(warmup_nc_steps)
        self.np_lo, self.np_hi = np_bounds
        self.nuf_lo, self.nuf_hi = nuf_bounds
        self.net = self.cfg.network
        # follower(=P-Stack의 nash_solver). leader 탐색은 RL이 대체하므로 안 씀.
        self._stack = StackelbergWuMeteredController(self.cfg)
        self.follower = self._stack.nash_solver
        self.action_dim = 2                 # [N_P, N_UF] (Phase 3에서 price 추가)
        self.reset()
        self.obs_dim = int(len(self._observe()))

    # ---------- MDP ----------
    def reset(self):
        self.sim = MixedTrafficSimulator(self.cfg)
        self.profile = DemandProfile(self.cfg, self.scenario)
        self.previous = None
        self.step_idx = 0
        self._prev_ttt = 0.0
        # warmup: no-control 몇 스텝(분석창 진입 상태 동일화, 러너와 동일 관행)
        for _ in range(self.warmup):
            self._advance(control=None)
        return self._observe()

    def step(self, action):
        """action: array-like len=2, 각 성분 [-1,1]. → feasible budget box."""
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        n_p = self.np_lo + (a[0] + 1.0) * 0.5 * (self.np_hi - self.np_lo)
        n_uf = self.nuf_lo + (a[1] + 1.0) * 0.5 * (self.nuf_hi - self.nuf_lo)
        forecast = self._forecast()
        la = LeaderAction(float(n_p), float(n_uf))
        prev = self.previous if self.previous is not None else self._fixed_prev()
        nash = self.follower.solve(self.sim.state.copy(), la, forecast, prev)
        control = nash.control
        log = self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        reward = -step_ttt                  # dense; return = -(episodic TTT)
        done = self.step_idx >= self.n_steps
        info = {"step_ttt": step_ttt, "cum_ttt": float(self.sim.total_ttt),
                "N_P": float(n_p), "N_UF": float(n_uf)}
        return self._observe(), reward, done, info

    # ---------- BC 수집용 ----------
    def budget_to_action(self, n_p: float, n_uf: float) -> np.ndarray:
        a0 = 2.0 * (float(n_p) - self.np_lo) / max(self.np_hi - self.np_lo, 1e-9) - 1.0
        a1 = 2.0 * (float(n_uf) - self.nuf_lo) / max(self.nuf_hi - self.nuf_lo, 1e-9) - 1.0
        return np.clip(np.array([a0, a1], dtype=np.float32), -1.0, 1.0)

    def step_with_control(self, control):
        """teacher가 만든 control을 직접 적용(follower 재실행 없이). 반환: obs, step_ttt, done."""
        forecast = self._forecast()
        log = self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        return self._observe(), step_ttt, self.step_idx >= self.n_steps

    # ---------- 내부 ----------
    def _forecast(self):
        t = self.step_idx * self.dt
        return self.profile.horizon(t, self.cfg.mpc.horizon_steps + max(0, self.cfg.mpc.leader_value_depth))

    def _fixed_prev(self):
        from src.models.state import ControlAction
        return ControlAction.fixed(self.cfg)

    def _advance(self, control):
        """warmup용 no-control 전진."""
        forecast = self._forecast()
        if control is None:
            from src.models.state import ControlAction
            control = ControlAction.fixed(self.cfg)
        log = self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1

    def _observe(self):
        """물리량 압축 관측 벡터(일반화 유리)."""
        if not hasattr(self, "sim"):
            return np.zeros(24, dtype=np.float32)  # __init__ 형상용
        s = self.sim.state
        net = self.net
        rc = float(net.rho_crit)
        # subsystem 집계
        uveh = float(s.total_urban_vehicles(net))
        fveh = float(s.freeway_segment_vehicles(net))
        rampq = float(sum(s.ramp_queue.values()))
        originq = float(sum(max(0.0, q) for q in s.mainline_origin_queue.values()))
        # freeway 밀도(클리프 근접도) — 링크별 mean/max/#>rho_crit, ρ_crit 정규화
        dens = []
        for link in net.freeway_links:
            dens += list(s.freeway_density.get(link, []))
        dens = np.array(dens, dtype=float) if dens else np.array([0.0])
        rho_mean = float(dens.mean()) / rc
        rho_max = float(dens.max()) / rc
        over = float((dens > rc).sum())
        # 속도
        sp = []
        for link in net.freeway_links:
            sp += list(s.freeway_speed.get(link, []))
        v_mean = float(np.mean(sp)) if sp else float(net.v_free)
        # forecast 요약(다음 스텝 총유입)
        fc = self._forecast()[0]
        inflow = float(sum(getattr(fc, "freeway_mainline", {}).values())) if hasattr(fc, "freeway_mainline") else 0.0
        ramp_arr = float(sum(getattr(fc, "ramp_arrival", {}).values())) if hasattr(fc, "ramp_arrival") else 0.0
        # 이전 budget
        pnp = float(self.previous.N_P_star) if self.previous is not None else 0.0
        pnuf = float(self.previous.N_UF_star) if self.previous is not None else 0.0
        # phase(정규화 시간)
        phase = float(self.step_idx) / max(1, self.n_steps)
        obs = np.array([
            uveh / 1000.0, fveh / 1000.0, rampq / 100.0, originq / 100.0,
            rho_mean, rho_max, over / 10.0, v_mean / max(net.v_free, 1.0),
            inflow / 5000.0, ramp_arr / 2000.0,
            pnp / max(self.np_hi, 1.0), pnuf / max(self.nuf_hi, 1.0),
            phase,
        ], dtype=np.float32)
        return obs


if __name__ == "__main__":
    env = RLLeaderEnv(scenario_name="sweet_170_incident_w60")
    print("obs_dim =", env.obs_dim, " action_dim =", env.action_dim)
    obs = env.reset()
    print("reset obs shape:", obs.shape, " (warmup", env.warmup, "steps 후)")
    total_r = 0.0
    for i in range(5):
        a = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        obs, r, done, info = env.step(a)
        total_r += r
        print(f"  step {env.step_idx}: r={r:9.2f}  N_P={info['N_P']:6.1f} N_UF={info['N_UF']:7.1f}  cum_ttt={info['cum_ttt']:.1f}")
    print(f"5-step return = {total_r:.1f}  (Phase 0 smoke OK)")
