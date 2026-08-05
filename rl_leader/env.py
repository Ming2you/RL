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

# Phase 4(2026-07-31): per-ramp metering 가격 액션. 순서는 probe/§7.3과 동일하게 고정.
RAMPS = ["R_D_W", "R_F_W", "R_D_E", "R_F_E"]
# split 모드(총량 하드) 배분 가격. probe_hard_budget_split.py 실측: ±300이면 merge 내부가
# 이미 전폭(D_W 1376 → 1127/1500)이라 그 이상은 해상도 낭비다(§2.3의 ±500 포화와 정합).
PRICE_MAG = 300.0
PRICE_TRUST = 0.20       # probe 실측: 0.2 vs 0.6 동일(trust는 병목 아님)

# Phase 5(2026-08-01): 가격의 '총량' 채널. follower.metering_price_split=True(기본)면
# Σmeter ≡ ω·N_UF가 하드 고정되어 가격은 merge 내부 배분만 정한다(probe_price_level.py 실측:
# 가격 ±1000에도 총량 변동 0.0). split=False면 priced_metering 분기가 열려 budget이 soft
# anchor가 되고 가격이 방류 수준을 유도한다(실측 총량 4200~6000, 변동폭 1800 veh/h).
# 실측 응답곡선(probe_price_anchor.py, budget-ref, peak 상태, budget 5254):
#   g=-150 → 6000(cap) / g=0 → 5100 / g=+75 → 4200 / g=+250 → 4200(포화)
# 유효 구간이 대략 [-150,+75]이므로 mag를 150으로 둬야 수집 샘플이 반응 구간에 떨어진다
# (250이면 양수 쪽 다수가 포화라 데이터가 낭비된다). 포화점은 상태 의존이라 여유를 조금 둠.
PRICE_LEVEL_MAG = 150.0
# 링크(merge) 간 예산 분할 ω_F. RL env는 leader.solve를 우회해 이 값이 0.5/0.5로 영구
# 고정돼 있었다(wu_distributed.py:119 생성자 기본값). P-Stack은 이 축을 0.26~0.98까지 쓴다.
LINKS = ["FW_W", "FW_E"]
LINK_OF = {"R_D_W": "FW_W", "R_F_W": "FW_W", "R_D_E": "FW_E", "R_F_E": "FW_E"}
OMEGA_LO, OMEGA_HI = 0.20, 0.80   # price_level 모드용 고정 박스(레거시)

# Phase 7(2026-08-03): urban green 한계가격. §12.6·12.7 — P-CENT는 freeway에서 이기고
# urban에서 지불한다(green sd가 P-Stack의 3.5~10배). RL엔 per-signal urban 채널이 없어
# 그 거래를 못 했고, 유일한 urban 레버 λ_P는 89% 스텝에서 0이다.
# probe_green_price.py 실측(step12): |g|<=0.02 무반응, 0.1에서 한 신호가 6~12s 이동,
# |g|>=0.5면 전 신호가 레일(20 또는 92)로 붙는다. 사용 가능 대역 ≈ [0.05, 0.3].
SIGNALS = ["A", "B", "C", "D", "F"]
GREEN_PRICE_MAG = 0.20   # ±0.5는 액션 대부분이 포화 → 대역에 맞춰 축소


def omega_window(n_uf: float, cap_link: float) -> tuple[float, float]:
    """총량이 새지 않는 ω 구간. follower는 링크예산을 clip(ω·N_UF, 0, Σcap_link)로 자르므로
    ω·N_UF > cap이면 초과분이 버려져 Σmeter < N_UF가 된다(probe 실측: ω=0.2에서 −1203 veh/h).

    실측(N_UF=5254, cap=3000): 창 [0.429,0.571] 안에서 총량 정확히 보존, 밖은 누수.
    창 폭은 N_UF가 용량에 붙을수록 좁아진다(N_UF=6000이면 ω=0.5 한 점) — 물리적으로
    타당하다: 용량 근처에선 '어디로 보낼지' 고를 여지가 없다."""
    if n_uf <= 1e-9:
        return 0.5, 0.5
    lo = max(0.0, 1.0 - cap_link / n_uf)
    hi = min(1.0, cap_link / n_uf)
    if lo > hi:
        return 0.5, 0.5
    return lo, hi


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
                 np_bounds: tuple[float, float] | None = None,
                 nuf_bounds: tuple[float, float] = (0.0, 6000.0),
                 scenario_dict: dict | None = None,
                 price_action: bool = False,
                 price_mag: float | None = None,
                 price_trust_frac: float = PRICE_TRUST,
                 price_level: bool = False,
                 omega_action: bool = False,
                 green_action: bool = False,
                 green_mag: float = GREEN_PRICE_MAG,
                 np_wide: bool = False,
                 np_dual: bool = False):
        self.scenario_name = scenario_name if scenario_dict is None else "random"
        self.cfg, self.scenario = make_cfg(scenario_dict if scenario_dict is not None else scenario_name)
        self.T_total = float(T_total)
        self.dt = float(self.cfg.simulation.control_interval)
        self.n_steps = int(self.T_total / self.dt)
        self.warmup = int(warmup_nc_steps)
        # Phase 4: 가격 모드에선 N_P 하한을 음수로 개방(§6-3 — net-inflow라 음수 유의미,
        # movement 하한 실측 ≈ −950). 기본(budget-only) 경로는 기존 박스 비트 유지.
        self.price_action = bool(price_action)
        self.price_level = bool(price_level)      # True면 가격이 총량을 정한다(split=False)
        self.omega_action = bool(omega_action)    # True면 링크 배분 ω_F를 액션으로 낸다
        self.green_action = bool(green_action)    # True면 per-signal green 가격을 낸다
        self.green_mag = float(green_mag)
        # ★N_P dual 되쓰기(2026-08-05). follower.solve()는 λ_next를 diagnostics로만 내놓고
        # self._lambda_P를 갱신하지 않는다 — 되쓰기는 컨트롤러(stackelberg_wu_metered.py:2436-2446)
        # 에만 있는데 RL env는 그 레이어를 우회한다. 결과: _lambda_P가 초기값 0.0에 영구 고정되어
        # N_P 제약이 한 번도 binding하지 않는다(probe: 8상태 전부 λ_P=0, N_P −3500~+2200에
        # green 변동 1/8). 즉 액션 2차원 중 N_P가 사실상 죽어 있었다.
        # 기본 OFF — 켜면 env 동역학이 바뀌어 기존 데이터/체크포인트와 비교가 깨진다.
        self.np_dual = bool(np_dual)
        if price_mag is None:
            price_mag = PRICE_LEVEL_MAG if self.price_level else PRICE_MAG
        self.price_mag = float(price_mag)
        self.price_trust = float(price_trust_frac)
        if np_bounds is None:
            if np_wide:
                # ★2026-08-05: P-Stack leader는 N_P를 −3315까지 탐색하고(leader_intent_N_P_star)
                # 그 스텝에서 λ_P가 상한(10)에 포화한다 = 제약이 실제로 binding. corr(net_inflow,
                # N_P_star) = −0.564로 방향도 살아 있다. 기존 박스 [0,2200]은 이 레버의 절반을
                # 잘라낸 것이었다. realized N_P가 +255 아래로 안 내려가는 건 follower가 물리적
                # 순유출을 못 만들어서지 리더가 음수를 요청 안 해서가 아니다.
                np_bounds = (-3500.0, 2500.0)
            else:
                np_bounds = (-1000.0, 2200.0) if self.price_action else (0.0, 2200.0)
        self.np_lo, self.np_hi = np_bounds
        self.nuf_lo, self.nuf_hi = nuf_bounds
        self.net = self.cfg.network
        # 링크당 램프 용량 합 — ω 가용 창 계산에 쓴다(총량 보존 조건).
        _capd = getattr(self.net, "ramp_capacity_veh_h", {}) or {}
        self.cap_link = float(sum(float(_capd.get(rp, 0.0))
                                  for rp in RAMPS if LINK_OF[rp] == LINKS[0])) or 3000.0
        # follower(=P-Stack의 nash_solver). leader 탐색은 RL이 대체하므로 안 씀.
        self._stack = StackelbergWuMeteredController(self.cfg)
        self.follower = self._stack.nash_solver
        if self.price_level:
            # 가격의 총량 채널 개방. leader 탐색이 없는 RL 경로에선 원래 이 스위치를 껐던
            # 병리(incumbent↔후보 교대 커밋)가 발생하지 않는다(probe_price_stability.py).
            self.follower.metering_price_split = False
        # [N_P, N_UF] (+ per-ramp 가격 4) (+ ω 1) (+ per-signal green 가격 5)
        self.action_dim = (2 + (4 if self.price_action else 0)
                           + (1 if self.omega_action else 0)
                           + (len(SIGNALS) if self.green_action else 0))
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
        """action: array-like len=action_dim, 각 성분 [-1,1].
        [0]=N_P, [1]=N_UF → feasible budget box. price_action이면 [2:6]=per-ramp 가격(×price_mag)."""
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        n_p = self.np_lo + (a[0] + 1.0) * 0.5 * (self.np_hi - self.np_lo)
        n_uf = self.nuf_lo + (a[1] + 1.0) * 0.5 * (self.nuf_hi - self.nuf_lo)
        forecast = self._forecast()
        la = LeaderAction(float(n_p), float(n_uf))
        prev = self.previous if self.previous is not None else self._fixed_prev()
        prices, omega_w = None, None
        # ω를 가격보다 먼저 주입한다 — price_level의 ref가 ω·N_UF에서 나오므로(_price_ref)
        # 순서가 바뀌면 이번 스텝 가격이 '직전 스텝 ω' 기준으로 잡히는 불일치가 생긴다.
        if self.omega_action:
            # ω_F 주입: leader.solve를 우회하는 RL 경로에서 0.5 고정이던 축을 연다.
            # follower는 매 solve마다 이 값을 읽는다(wu_faithful_follower.py:2984, :3162).
            ai = 2 + (4 if self.price_action else 0)
            if self.price_level:
                lo, hi = OMEGA_LO, OMEGA_HI      # soft-budget 모드: cap 누수 개념이 다름
            else:
                lo, hi = omega_window(n_uf, self.cap_link)   # ★총량 보존 창으로 정규화
            omega_w = lo + (a[ai] + 1.0) * 0.5 * (hi - lo)
            self.follower._wu._omega_f = {LINKS[0]: float(omega_w),
                                          LINKS[1]: float(1.0 - omega_w)}
        if self.price_action:
            # probe(price_inverse_probe.py) 검증 패턴 그대로: price/ref/trust 주입 → solve → 원복.
            prices = {rp: float(self.price_mag * a[2 + i]) for i, rp in enumerate(RAMPS)}
            self.follower.metering_marginal_price = prices
            self.follower.metering_marginal_price_ref = self._price_ref(n_uf, prev)
            self.follower.metering_marginal_price_trust_frac = self.price_trust
        green_p = None
        if self.green_action:
            # per-signal green 가격. 소비: wu_faithful_follower.py:879-883
            #   cost += weight · g_ext · (p1 − ref),  ref = 직전 스텝 green(p1)
            # trust_sec은 None으로 둔다 — probe 실측상 None/6/18이 동일 결과였다
            # (green 후보가 6s 격자라 자연 이동폭이 이미 ±6s).
            gi = 2 + (4 if self.price_action else 0) + (1 if self.omega_action else 0)
            green_p = {s: float(self.green_mag * a[gi + k]) for k, s in enumerate(SIGNALS)}
            self.follower.signal_marginal_price = green_p
            self.follower.signal_marginal_price_ref = {
                s: float(prev.green_times.get(f"{s}_p1", 56.0)) for s in SIGNALS}
            self.follower.signal_marginal_price_trust_sec = None
        try:
            nash = self.follower.solve(self.sim.state.copy(), la, forecast, prev)
        finally:
            if self.price_action:
                self.follower.metering_marginal_price = None    # 오염 방지(probe 규약)
            if self.green_action:
                self.follower.signal_marginal_price = None
        control = nash.control
        if self.np_dual:
            self._commit_np_dual(control)
        log = self.sim.step(control, forecast[0], self.step_idx)
        self.previous = control.copy()
        self.step_idx += 1
        step_ttt = float(log.urban_ttt + log.freeway_ttt)
        reward = -step_ttt                  # dense; return = -(episodic TTT)
        done = self.step_idx >= self.n_steps
        info = {"step_ttt": step_ttt, "cum_ttt": float(self.sim.total_ttt),
                "N_P": float(n_p), "N_UF": float(n_uf),
                # subsystem 귀속(메커니즘 분석용) — 이득이 urban/freeway 어디서 오는지
                "urban_ttt": float(log.urban_ttt), "freeway_ttt": float(log.freeway_ttt)}
        if prices is not None:
            info["prices"] = prices
            info["ramp_release"] = {rp: float(control.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
        if omega_w is not None:
            info["omega_w"] = float(omega_w)
        if green_p is not None:
            info["green_prices"] = green_p
            info["green_times"] = {s: float(control.green_times.get(f"{s}_p1", float("nan")))
                                   for s in SIGNALS}
        return self._observe(), reward, done, info

    # ---------- BC 수집용 ----------
    def budget_to_action(self, n_p: float, n_uf: float) -> np.ndarray:
        a0 = 2.0 * (float(n_p) - self.np_lo) / max(self.np_hi - self.np_lo, 1e-9) - 1.0
        a1 = 2.0 * (float(n_uf) - self.nuf_lo) / max(self.nuf_hi - self.nuf_lo, 1e-9) - 1.0
        return np.clip(np.array([a0, a1], dtype=np.float32), -1.0, 1.0)

    def compose_action(self, n_p: float, n_uf: float, prices_veh=None,
                       omega_w: float | None = None, green_prices=None) -> np.ndarray:
        """(N_P, N_UF, per-ramp 가격[veh/h, RAMPS 순서], ω_W) → 정규화 액션. 수집기용."""
        a = self.budget_to_action(n_p, n_uf)
        if self.price_action:
            p = np.zeros(4, dtype=np.float32) if prices_veh is None else \
                np.clip(np.asarray(prices_veh, dtype=np.float32) / max(self.price_mag, 1e-9),
                        -1.0, 1.0)
            a = np.concatenate([a, p])
        if self.omega_action:
            # omega_w는 [0,1] 정규화 위치(0=창 하한, 1=창 상한). 창이 N_UF 의존이라
            # 절대 ω로 받으면 수집기가 창 밖을 뽑아 예산을 버리게 된다.
            u = 0.5 if omega_w is None else float(np.clip(omega_w, 0.0, 1.0))
            a = np.concatenate([a, np.array([2.0 * u - 1.0], dtype=np.float32)])
        if self.green_action:
            g = np.zeros(len(SIGNALS), dtype=np.float32) if green_prices is None else \
                np.clip(np.asarray(green_prices, dtype=np.float32) / max(self.green_mag, 1e-9),
                        -1.0, 1.0)
            a = np.concatenate([a, g])
        return a

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

    def _commit_np_dual(self, control):
        """컨트롤러의 λ_P 되쓰기를 복제(stackelberg_wu_metered.py:2436-2446).

        기본 설정은 np_candidate_lambda=True / np_primal_dual_iters=0 이므로 corrector 경로다:
        standing λ를 직접 쓰지 않고 (λ_k, 투영 target)을 pending으로 넘기면, 다음 solve 시작에서
        follower가 실현 유입 Q^real로 1회 교정해 self._lambda_P를 갱신한다
        (wu_faithful_follower.py:3788-3823). _np_last_real_q는 follower가 자체 계산하므로(3771)
        pending만 채워주면 루프가 닫힌다."""
        d = getattr(control, "diagnostics", None) or {}
        f = self.follower
        tgt = d.get("wu_faithful_np_projected_target")
        if tgt is not None:
            f._np_corrector_pending = (float(f._lambda_P), float(tgt))
        sn = d.get("wu_faithful_np_sum_nin")
        if sn is not None:
            f._np_last_sum_nin = float(sn)

    def _price_ref(self, n_uf, prev):
        """가격의 기준점(ref). cost = w·g·(meter − ref) 이므로 ref 선택이 곧 파라미터화다.

        price_level=False: 직전 운영점(§2.3 probe 규약). 총량이 하드 고정이라 안전.
        price_level=True : ★budget 함의 수준에 **고정**. 직전값을 쓰면 기준이 매 스텝 따라
          내려와 "지금보다 더 줄여라"가 반복되는 래칫이 생겨 미터링이 0까지 붕괴한다
          (probe_price_anchor.py 실측: [4200,3000,2100,1500,0,0,…], TTT +76%).
          고정 기준이면 가격은 '예산 대비 편차'를 정하는 레벨 제어가 되고 운영점이 안정화된다
          (동일 probe: 16스텝 전부 4200.0, std 0.0)."""
        if not self.price_level:
            return {rp: float(prev.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
        om = self.follower._wu._omega_f
        n_per_link = max(len(RAMPS) // max(len(LINKS), 1), 1)
        return {rp: float(om.get(LINK_OF[rp], 0.5)) * float(n_uf) / n_per_link
                for rp in RAMPS}

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
        """물리량 압축 관측 벡터(일반화 유리). price_action이면 공간 신호 10차원 추가
        (per-ramp 큐 4 + per-link rho_max 2 + per-ramp 도착예보 4) — 공간 타겟팅은
        공간을 봐야 가능하다(기존 13차원은 전부 집계량이라 서-동 비대칭이 안 보임)."""
        if not hasattr(self, "sim"):
            return np.zeros(23 if self.price_action else 13, dtype=np.float32)  # __init__ 형상용
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
        if not self.price_action:
            return obs
        # ---- Phase 4 공간 신호(앞 13차원은 budget-only와 동일 유지) ----
        per_ramp_q = [float(s.ramp_queue.get(rp, 0.0)) / 100.0 for rp in RAMPS]
        link_rho_max = []
        for link in net.freeway_links:                      # config 순서 고정(2링크)
            d = s.freeway_density.get(link, [])
            link_rho_max.append((float(max(d)) if len(d) else 0.0) / rc)
        ra = getattr(fc, "ramp_arrival", {}) or {}
        per_ramp_arr = [float(ra.get(rp, 0.0)) / 500.0 for rp in RAMPS]
        return np.concatenate([
            obs, np.array(per_ramp_q + link_rho_max + per_ramp_arr, dtype=np.float32)])


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
