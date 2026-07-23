# Wu(2022) §IV-D 충실 분산 follower — 진짜 per-signal 국소 rollout + Jacobi 합의 (새 코드)
"""SPEC_wu_faithful_follower.md 구현물.

이전 실패는 후보 채점을 전체망 plant(`urban_step`/`run_coupled_interval`)로 해서 진짜 local이
아니었고 목적이 global TTT였다. 이번엔:
1. agent i(=신호 1개)의 movement 큐만 `LocalSignalModel.rollout_local_tts`로 전진(이웃 동결).
2. 목적 = 자기 차량수 합(자기 TTS) + R_i·|Δg|.
3. Jacobi: S_max=5 반복, 결합변수 z̃ 동결·동시갱신, warm-start.

기존 파일 미변경 원칙: 결합변수 계산(`_coupling`), 토폴로지 맵(`_phase_movements`, `_specs`),
freeway agent VSL solve(`_solve_freeway_agent`)는 기존 `WuDistributedController` 인스턴스를
**조합(composition)**으로 재사용한다. urban agent solve만 진짜 국소 rollout으로 교체한다.

`solve(state, leader, demand, previous) -> NashResult`로 `DistributedCoordinator`와 동일 인터페이스.
leader는 None(PFO 모드)부터 구현한다.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from src.controllers.local_freeway_plant import (
    build_local_freeway_model,
    freeway_substep_local,
)
from src.controllers.segment_local_plant import (
    FrozenLinkTrajectory,
    SegmentLocalState,
    build_segment_agent_models,
    segment_substep_local,
)
from src.controllers.local_signal_plant import (
    build_local_model,
    rollout_local_tts,
    rollout_local_tts_phased,
    rollout_local_tts_ramp_aware,
)
from src.controllers.nash_solver import NashResult
from src.controllers.relaxed_quantization import (
    queue_pressure_green_target,
    repair_green_pair,
    repair_vsl_value,
)
from src.controllers.wu_distributed import WuDistributedController, _split_link_offramp_flow
from src.models.demand import DemandStep
from src.models.metanet import compute_ramp_release_flows
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.models.urban_queue_model import (
    _effective_available_space,
    _link_delay_steps,
    _phase_green_fraction,
    _urban_step_index,
)

# 보호영역 net-inflow 분류(inflow_outflow_allocation.py와 동일). 리더 N_P 정의와 일치시킨다.
_INFLOW_KINDS = {"boundary_in", "off_ramp"}
_OUTFLOW_KINDS = {"boundary_out", "on_ramp"}

# N_P predictor 모드 코드(진단용 one-hot과 함께 기록; 2026-06-30 리포트 원인분리 스위치).
_NP_PREDICTOR_MODE_CODES = {
    "legacy": 0.0,
    "storage_aware": 1.0,
    "current_interval": 2.0,
    "phase_substep": 3.0,
}


class WuFaithfulFollower:
    """Wu §IV-D 충실 분산 follower(PFO 모드 우선)."""

    def __init__(self, cfg: ExperimentConfig, authority: str = "proposed"):
        self.cfg = cfg
        # ---------- AUTHORITY 설정(어떤 액추에이터가 활성인지) ----------
        # "proposed"(기본): urban GREEN + OFFSET + ramp METERING + freeway VSL(전체 제안 follower).
        #   기존 거동 보존 — +56.63%(+offset 효과). default이라 기존 런 미영향.
        # "wu": urban GREEN + freeway VSL만. ramp metering은 capacity 고정(metering OFF),
        #   OFFSET 없음(0 고정). Wu 충실 green+VSL 분산 baseline — metering 부재라 약함(≈−1%대).
        authority = str(authority).lower()
        if authority not in ("proposed", "wu"):
            raise ValueError(f"authority must be 'proposed' or 'wu', got {authority!r}")
        self.authority = authority
        # 액추에이터 게이트 플래그(authority가 결정).
        self.metering_enabled = authority == "proposed"
        # offset은 per-signal로 끈다(2026-06-29 검증): offset(green wave)은 *joint·coordinated* 양이라
        # per-signal selfish 최적화는 corridor를 오히려 de-coordinate해 무가치/해롭다(대칭·양방향·one-way
        # 모두 음수). global TTT로 *joint* 채점하면 양수지만 이 망에선 ~+0.2%로 미미(짧은 corridor·metering
        # 지배). 따라서 offset은 follower 레버가 아니라 향후 leader-coordinated 레버로 둔다.
        # phase-resolved 서비스 + platoon 도착 인프라(_solve_offset_local 등)는 그 leader 레버용으로 보존.
        self.offset_enabled = False
        # 결합·freeway·토폴로지 재사용용 내부 인스턴스(기존 파일 미변경, 조합만).
        self._wu = WuDistributedController(cfg, leader_enabled=False)
        self._specs = self._wu._specs
        self._phase_movements = self._wu._phase_movements
        # 신호별 국소 모델(정적 데이터) 구성 — 매 step 재사용.
        self._local_models = {
            signal: build_local_model(cfg, signal, self._specs, self._phase_movements)
            for signal in cfg.network.signals
        }
        # freeway link별 국소 모델(정적 데이터) — per-link METANET rollout용. 매 step 재사용.
        self._local_freeway_models = {
            link: build_local_freeway_model(cfg, link)
            for link in cfg.network.freeway_links
        }
        # de facto ramp metering 패널티 계수: 0으로 비활성화한다(SPEC 갱신). 이전엔 urban
        # agent의 on_ramp green을 freeway 혼잡으로 가중 처벌해 metering을 '근사'했으나, 이는
        # 튜닝된 hack(이중 metering)이었다. 이제 freeway agent가 진짜 ramp_metering 액추에이터를
        # 자기 own-TTS로 직접 탐색하므로(아래 `_solve_freeway_agent_metered`), urban agent는
        # 순수 demand-responsive로 남기고 metering은 freeway가 단독 수행한다.
        self.ramp_metering_weight: float = 0.0
        # freeway agent가 탐색할 ramp metering 후보 분율(×capacity). +41.8% 검증 최적(≈0.5)을
        # 중심으로 0.25~1.0을 덮는다. cap=100%(=metering off)부터 강한 metering 25%까지.
        self.ramp_metering_fractions: tuple[float, ...] = (1.0, 0.7, 0.5, 0.35, 0.25)
        # freeway agent own-TTS에 urban 상류 blocked 큐(가상)를 포함할지 (P1, 2026-07-03).
        # 진단 근거: sweet_190(중부하)에서 PFO가 D-ramp metering을 ~546 veh/h로 조이면
        # reservoir(≤ramp_queue_max)가 차고 plant가 urban green release를 ramp_space로
        # 스케일다운해 차량이 urban movement 큐에 갇힌다. freeway agent own-TTS는
        # (본선+자기 reservoir+off-ramp storage)만 세서 이 상류 비용이 안 보였다 —
        # 보이는 비용(cap 180)과 안 보이는 비용(무한 urban 큐)을 맞바꾸는 externality.
        # True면 reservoir가 수용 못 한 coupling 유입을 ramp별 가상 blocked 큐로 이월
        # 적재하고 그 차량수를 own-TTS에 더한다(새 가중치 없음 — 차량 수 세기).
        # False면 기존 거동과 완전 동일(A/B probe용).
        self.count_blocked_ramp_inflow: bool = True
        # D/F(ramp-aware) 신호의 green 채점을 phase-resolved(platoon 도착 + offset-aware
        # green window)로 할지 (P1.5, 2026-07-03). 기존 cycle-평균 gf=green/cycle 상수는
        # green의 시간구조·offset이 채점에 전혀 안 들어가 legacy green 주입도 기각하는
        # 원인 일부로 의심됐다. 비-ramp(A/B/C)는 이미 같은 경로(use_phased)를 쓴다.
        # False면 기존 cycle-평균 경로 그대로(A/B probe용).
        # ── 기본 False(게이트 실패, 2026-07-03 closed-loop A/B): sweet_155 −1.47% 개선이나
        # sweet_190 +1.09%/sweet_128 +0.94% 악화 — sweet_190이 "1% 초과 악화 STOP" 기준에
        # 걸려 기본 비활성(opt-in)으로 남긴다. 채점 인프라·단위테스트는 유효(하위호환 완전).
        self.ramp_aware_phase_resolved: bool = False
        # P1.5 조건부(auto) 활성화(2026-07-03 재검토): phase-resolved 채점은 sweet_155(중부하)
        # −1.47% 개선이나 sweet_190/128에서 ~+1% 악화 — 상시 ON은 기각됐고, "중부하에서만"이
        # 재시도 방향이다. auto=True면 step당 1회 ramp 신호별 phase 포화도
        #   x = max_p (q0_p/horizon_h + arr_p) / (Σ cap_flow·g_p/total)
        # 를 계산해, **모든 ramp 신호의 x가 band [lo, hi) 안일 때만**(AND-게이트) phase-
        # resolved 채점을 켠다. band (1.0, 2.2)는 3600s PFO 계측(outputs/_p15_sat_*)에서:
        #   sweet_128(+0.94% 악화): D 0.68–0.96 미포화 → D가 band 밖 → OFF.
        #   sweet_155(−1.47% 개선): D 0.83–1.69 / F 1.0–2.1 → 대부분 ON.
        #   sweet_190(+1.09% 악화): step 8 이후 D 2.6–4.3 과포화 → OFF.
        # 직관: green window 시간구조가 채점에 유효한 건 막 포화된 green-binding 영역뿐 —
        # 미포화는 큐가 사소해 노이즈, 깊은 과포화는 큐가 cycle 내내 남아 cycle-평균이 이미
        # 정확하다. AND인 이유: sweet_128의 F(1.0–1.37)가 sweet_155 대역과 겹쳐 신호별
        # 게이트로는 분리 불가(회귀 재발) — 망 단위 "중부하 regime" 판정이 필요하다.
        # False(기본)면 완전 휴면(기존 거동 비트 동일). x는 wu_p15_sat_{signal}로 진단 기록.
        self.ramp_aware_phase_auto: bool = False
        self.ramp_aware_phase_auto_band: tuple[float, float] = (1.0, 2.2)
        self._phase_resolved_active_signals: set = set()
        # ---------- B2: leader-계산 per-signal externality 가격(2026-07-03 Step B1 검증) ----------
        # P-Stack 컨트롤러가 refresh마다 설정하는 신호별 가격 g_ext_i[veh·h/green-sec]와
        # 기준점 p1_ref. green 후보 비용에 + weight·g_ext_i·(p1 − p1_ref_i)를 더한다.
        # g_ext = d(전역TTT)/dp1 − d(own-TTS)/dp1 (Pigouvian — own-TTS 몫을 빼야 부호·크기가
        # 산다, Step B1 ext 9/15 vs full 1/15). 이 가격은 전역 rollout이 필요해 분산 follower가
        # 스스로 계산할 수 없다 — 순수 PFO 러너는 절대 설정하지 않으며, None이면 기존 거동과
        # 비트 동일(가격 채널 완전 휴면).
        self.signal_marginal_price: Optional[Dict[str, float]] = None
        self.signal_marginal_price_ref: Dict[str, float] = {}
        self.signal_marginal_price_weight: float = 1.0
        # B2.1 trust region(2026-07-05 진단): 가격이 유효한 범위를 유한차분 이웃으로 제한.
        # 폭주(sweet_155 C 56→92)의 기전은 g_ext = g_i − d_local(두 큰 수의 차)이 측정
        # 이웃(±δ) 밖으로 선형 외삽되며 국소 곡률 변화에 월권하는 것 — step36 직독에서
        # 전역 g_i는 +0.012("줄여라")로 옳았는데 가격 −0.27이 후보 92를 끌어감. None이면
        # 무제한(기존 거동). 값이 있으면 |p1 − p1_ref| ≤ trust 후보만 가격 대상으로 허용
        # (이웃 밖 후보는 탐색에서 제외 — 정렬된 개선은 refresh마다 한 이웃씩 plant 검증을
        # 통과하며 누적, 비정렬 표류는 전역 신호가 반전되는 즉시 정지).
        self.signal_marginal_price_trust_sec: Optional[float] = None
        # ---------- B3(Codex f18e920 포팅): metering/VSL 가격 채널 ----------
        # green과 동일 규약으로 통일(2026-07-04 병합): leader가 refresh마다 **동일 동결
        # 운영점**에서 g_ext(=g_i − d_local)를 완성해 하달하고, follower는 선형 가격항만
        # 더한다 — Codex 원안(solve 안에서 d_local 재계산: 운영점 혼합 + solve당 rollout
        # 2회/ramp 낭비)과 다른 지점. metering 가격 활성 시 leader 분기의 N_UF hard
        # budget은 soft |Σ−budget| 페널티로 완화된다(가격이 방류 수준을 결정, budget은
        # anchor — Codex soft-budget 설계 유지). 1차 TTT 가격 단독은 음성 판정(절벽
        # lever, 2026-07-04 §3)이라 기본 None; B4 barrier 가격과 함께 opt-in.
        self.metering_marginal_price: Optional[Dict[str, float]] = None
        self.metering_marginal_price_ref: Dict[str, float] = {}
        self.metering_marginal_price_weight: float = 1.0
        self.metering_budget_penalty_weight: float = float(cfg.simulation.T_c_h)
        # B3TR trust region(2026-07-05 §7과 동일 원리): 가격 유효 범위를 측정 이웃으로 제한.
        # metering은 후보 격자(0.1~0.3·cap)가 커서 반경을 **cap 분율**로 정의한다 —
        # trust_r = frac·cap_r, leader의 유한차분 δ_r도 같은 폭으로 측정(허용 이동폭만큼
        # 측정 원칙). None이면 무제한(-B3 재현용, 과소방류 나선).
        self.metering_marginal_price_trust_frac: Optional[float] = None
        # B3CERT 비대칭 안전 증명서(2026-07-05): capacity drop은 비가역이라 방류 증가
        # 방향은 탐색으로 검증 불가(B3TR v2 breakdown) — leader가 +δ rollout의 예측 본선
        # 밀도로 사전 인증한 ramp만 방류 증가 후보 허용. 조임(방류↓)은 가역이라 자유.
        # None이면 비활성(기존 거동).
        self.metering_release_certified: Optional[Dict[str, bool]] = None
        # ---------- SPLIT-PRICE(2026-07-09, 사용자 스펙 정정): 총량=수량, 배분=own_TTS+가격 ----------
        # True(기본): metering 가격이 있어도 **총량은 equality budget으로 강제**하고, 가격은
        # budget 내 ramp 간 배분 채점(own_TTS + Σ g_ext·(x−ref))에만 쓴다. Σ가 고정이라
        # 가격의 공통 성분은 후보 간 상수로 소거 — ramp 간 차이만 배분을 기울인다(가격이
        # 총량과 싸울 여지 원천 차단). 절벽 축(총량)=수량, 매끈 축(배분)=가격의 축단위
        # Weitzman 정합. False = 구 B3 계보(가격이 레벨 조절 + soft anchor) 재현용 —
        # d0 붕괴(22698)·레벨 배회의 원인이었던 모드.
        self.metering_price_split: bool = True
        # VSL 가격(세그먼트 키 "link__segN"). E2(2026-07-09): vsl_override 프리미티브로
        # local_vsl_costs(고정 VSL 벡터 국소 채점)가 생겨 g_ext = g_i − d_local로 정렬됨
        # (기존 raw g_i의 own 성분 이중계상 결함 해소). 기본 OFF.
        self.vsl_marginal_price: Optional[Dict[str, float]] = None
        self.vsl_marginal_price_ref: Dict[str, float] = {}
        self.vsl_marginal_price_weight: float = 1.0
        # ---------- F3(2026-07-06): offset 가격 채널 ----------
        # offset은 selfish 최적화가 corridor를 해치는 순수 조정 레버(2026-06-29 판정으로
        # 자율 offset OFF, "leader-coordinated 레버로 보존"). F3 = 그 보존된 계획의 실행:
        # leader가 전역 rollout으로 g_ext_off를 계산·하달하면 offset 탐색이 활성화되고,
        # 후보 채점 = 자기 phased 비용 + w·g_ext·Δ(원형 최단 변위), trust(그리드 1칸,
        # cycle/8)로 월권 방지. 기존 offset corridor 가드(realized TTT 검증)는 그대로
        # 최종 검증자로 작동. None이면 완전 휴면(offset 0 유지, 기존 거동 비트 동일).
        self.offset_marginal_price: Optional[Dict[str, float]] = None
        self.offset_marginal_price_ref: Dict[str, float] = {}
        self.offset_marginal_price_weight: float = 1.0
        self.offset_marginal_price_trust_sec: Optional[float] = None
        # ---------- JOINT(2026-07-09): bilinear cross-term 가격 ----------
        # per-lever 선형가격이 못 담는 lever쌍 교차곡률 ∂²(TTT+V)/∂a∂b를 leader가
        # h_ext = h_global − h_local(4-corner 스텐실, own_TTS 몫 차감)로 하달. follower는
        # 해당 쌍을 2D 공동탐색하며 priced += w·h_ext·(a−a_ref)·(b−b_ref).
        # green×offset(도시 신호쌍, non-ramp): 선형가격만으론 무력한 진짜 gap —
        #   green은 offset 동결 1D, offset은 green 동결 1D(coordinate descent)라 cross가
        #   퇴화. joint_green_offset_enabled면 non-ramp 신호를 2D 공동탐색으로 전환.
        # vsl×metering(freeway 합류쌍): primal joint은 _solve_freeway_agent_local이 이미
        #   포착(metering마다 VSL best-response) → 탐색 구조 불변, cross 가격만 추가.
        # None/False = 완전 휴면(비트동일).
        self.green_offset_cross_price: Optional[Dict[str, float]] = None  # signal → h_ext
        self.green_offset_cross_ref: Dict[str, tuple] = {}               # signal → (p1_ref, off_ref)
        self.green_offset_cross_weight: float = 1.0
        self.joint_green_offset_enabled: bool = False
        self.vsl_meter_cross_price: Optional[Dict[str, float]] = None    # ramp → h_ext
        self.vsl_meter_cross_ref: Dict[str, tuple] = {}                  # ramp → (meter_ref, vsl_ref)
        self.vsl_meter_cross_weight: float = 1.0
        # ---------- PRICE-TR(2026-07-09): 가격 모드 = trust region만, 마찰 0 (기본 OFF 복귀) ----------
        # ON이면 가격 활성 레버의 smoothness(proximal 마찰)를 0으로 — SLP/Frank-Wolfe 표준형.
        # 실측 판정(같은 날): 마찰 deadband는 저SNR 가격 신호의 **암묵적 신호검정**이었다.
        # 제거하자 노이즈성 gradient까지 매 스텝 trust 보폭으로 행동 → G1DF +139, APJOINT
        # 13493 참사. 평시 레짐(green g_ext 0.03~0.055 < 0.1)에선 마찰 유지가 우세 —
        # 원리적 대체는 dead zone(|g_ext|<θ 무시)이며 강신호 레짐(포화×skew·사고)용 옵션.
        # VSL trust(±10km/h, vsl_price_trust_kmh)는 보폭 제한이라 마찰과 무관하게 유지.
        self.price_smoothness_disabled: bool = False
        self.vsl_marginal_price_trust_kmh: Optional[float] = None
        # ---------- 13-player(2026-07-10 승인, plan-13player-rebuild.md) ----------
        # segment_agents=True면 freeway를 link agent 2개 대신 segment agent 8개로 분해:
        # F_L0=seg0+origin, F_L1=seg1, F_L2=seg2+R_D, F_L3=seg3+R_F. off-ramp storage는
        # urban 소유(여기선 동결 y로만 읽음). 예산 Σmeter=ω_F·N_UF*는 owner 2명의
        # best-response 후 simplex 사영(승인안 (ii)) — env SEG13=1로 활성.
        self.segment_agents: bool = False
        self._segment_agent_models: Dict[str, list] = {}
        self._seg13_diag: Dict[str, float] = {}
        # SPLIT-PRICE v1/v2 재현(13p): v2(기본)=incumbent(leader=None)는 meter 가격-레벨
        # 배제(가격은 예산 내 배분만 — 7p 플래그십 규약). v1=True면 incumbent도
        # g_meter/h로 레벨 조절(7p에서 레짐 플래핑 병리를 낸 구성) — env SEG13_V1=1.
        self.seg13_meter_price_standing: bool = False
        # 궤적 교환(Wu §IV-D의 ỹ): Jacobi iteration마다 각 segment agent가 자기 seg의
        # 예측 입력궤적(ρ,v,λ,release)을 내놓고, 다음 iteration의 동결 y가 α=0.5 블렌딩으로
        # 갱신된다. False(SEG13_TRAJ=0)면 v0의 hold-constant(현 상태 동결) — A/B용.
        self.seg13_traj_exchange: bool = True
        self._seg_traj: Dict[str, dict] = {}
        # radius-1 국소 rollout(PFO 강화 변형, env SEG13_NBR): 자기 ±1 seg를 함께 전진시키고
        # 그 차량수를 w_nbr 가중으로 비용에 포함 — own-TTS의 방류 무차별(보존식 변위)을
        # leader 없이 깨는 정직한 최대 분산 기준선. 플래그십(leader)은 가격이 이 역할을
        # 하므로 기본 0.0(OFF) — 켜면 가격 기여 귀속이 흐려진다.
        self.seg13_neighbor_weight: float = 0.0
        # NP-CAND-λ̂(2026-07-12): 직전 step 컨트롤러가 commit한 solve의 Σnin.
        # np_candidate_lambda=True일 때 후보별 λ̂ 선반영의 오차항으로 쓴다. None=첫 step.
        self._np_last_sum_nin: Optional[float] = None
        # (51) corrector 상태(원고 정식화 정렬): 직전 step 보호구역 accumulation(실현
        # 유입 측정용), step 가드 시각, pending=(λ_k, 커밋 후보의 투영 target),
        # 최근 실현 순유입(horizon 환산 veh). predictor(48)·corrector(51) 모두 실현
        # 유입을 오차 신호로 쓴다 — 예측 Σnin은 첫 스텝 fallback.
        self._np_prev_accum: Optional[float] = None
        self._np_step_time: Optional[float] = None
        self._np_corrector_pending: Optional[tuple] = None
        self._np_last_real_q: Optional[float] = None
        # r̂ 편향 보정(2026-07-13): 계획(예측 Σnin) 대비 실현(ΔN_P×H) 유입의 EWMA 비율.
        # 국소 모형의 낙관 편향으로 계획-공간 target(Ñ)과 실현-공간 오차 신호가 어긋나
        # λ̂가 구조적으로 휴면하는 문제의 다리 — 비교를 r̂·Ñ(실현 공간 환산)으로 수행.
        # np_bias_correction=False(기본)면 r̂=1 고정(비트동일).
        self._np_bias_ratio: Optional[float] = None
        # 예산 부등식 ablation(2026-07-11, env SEG13_INEQ=1): 등식 Σ=budget 대신 Σ≤budget.
        # 170_skew 진단 — 등식은 상향으로도 binding이라 leader N_UF=6000이 follower의
        # 자발적 하향 metering을 덮어써 절벽 근처 dip→과잉 교정(whipsaw)을 유발.
        # True면 자율 합이 budget 미만일 때 그대로 존중(하향 자유), 초과 시에만 사영.
        self.seg13_budget_inequality: bool = True  # 2026-07-14 동결: 부등식 예산이 기본(whipsaw 제거, 155 -909). SEG13_INEQ=0으로 등식 복원 A/B
        # 회랑 예산(2026-07-14, 사용자 지시): 링크별 α·budget ≤ Σmeter ≤ budget.
        # 순수 부등식(α=0)은 하한이 없어 大δ 가격이 전 램프를 동시 조일 때 총방류가
        # 나선형으로 떨어지는 과소방류 나선(구 B3 기각 사유)을 못 막고, 등식(α=1)은
        # 상향 binding이 whipsaw를 만든다 — α는 그 스펙트럼의 혼합도구(Roberts-Spence:
        # 가격(배분)과 수량(회랑 경계)의 결합). env RELEASE_FLOOR로 조정.
        self.seg13_release_floor_frac: float = 0.65
        # 부하 적응형 회랑 하한(2026-07-15): 본선 자유류면 α→0(경부하 자율), 임계 근방
        # α→α_max(나선 방어). RELEASE_FLOOR_FIXED=1로 고정 0.65 복원(A/B).
        self.seg13_release_floor_adaptive: bool = True
        self.seg13_floor_cong_lo: float = 0.7   # 본선ρ/ρ_crit 이하면 하한 0
        self.seg13_floor_cong_hi: float = 1.0   # 임계 도달 시 하한 α_max
        # ---------- J1(2026-07-06): joint offset 패턴 directive ----------
        # F3 판정: offset은 joint 결합 변수라 per-signal 가격(편미분)이 구조적으로 0.
        # J1 = leader가 corridor 패턴(여러 신호 offset 조합)을 통째로 rollout 평가해
        # 최선 조합을 directive로 하달 — legacy의 joint 평가를 패턴 후보로 저렴화.
        # 설정 시 per-signal offset 탐색 대신 directive를 그대로 적용하고, 기존
        # corridor 가드(realized TTT 검증)가 최종 검증자로 남는다. None=완전 휴면.
        self.offset_directive: Optional[Dict[str, float]] = None
        # LEADER-OFFSET(2026-07-07): directive가 leader의 전역 joint 결정(권위적)이면 follower의
        # corridor 검증 가드(아래 offset_keep_margin)를 우회한다. 근거: (i) 가드는 per-signal
        # selfish 국소 탐색이 corridor 전체를 해칠 위험을 막으려는 것인데, leader가 이미 global
        # rollout으로 joint 최적화했으므로 selfish 위험이 없다(가드 명분 소멸). (ii) 사용자 요지 —
        # 단일 스텝 이득이 작아도(가드 0.5% 마진 미달) 누적되어 혼잡을 예방 — 를 가드가 정면으로
        # 죽인다(작은 이득 = 되돌림). leader가 offset을 "소유"한다는 건 follower가 재검증(veto)하지
        # 않는다는 뜻. 안전망은 leader의 adopt 판정(global gain>margin)과 실험의 STOP(>1% 악화) 기준.
        self.offset_directive_authoritative: bool = False
        # ---------- G1(2026-07-06): ramp 신호(D/F) offset 활성화 ----------
        # 진단(scratchpad corridor_graph_check): plant offset 민감도가 F=10.7 ≫ 비-ramp
        # (A 0.30/B 0.12/C 1.56) — 가장 큰 offset 레버 F/D가 ramp 신호라는 이유로
        # `_solve_offset_local`에서 offset=0 고정(단순화)돼 있었다. legacy는 D/F offset을
        # std 44/35로 씀. G1은 ramp 신호도 phase-resolved ramp-aware rollout으로 offset을
        # 탐색한다(P1.5의 use_phased_ramp 채점 경로 재사용). 기본 False → 기존 비트 동일.
        self.ramp_offset_enabled: bool = False
        # 신호별 offset 후보 분율(×cycle_length). [0, cycle) 안의 작은 후보집합으로 국소 탐색.
        # 0.0은 현 baseline(offset off). 0~7/8 cycle을 8등분(끝점 cycle 제외).
        self.offset_fractions: tuple[float, ...] = (
            0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
        )
        # offset corridor 가드 마진(상대): realized horizon TTT가 offset-0 대비 이만큼 이상
        # 개선될 때만 offset을 commit한다. horizon proxy는 closed-loop의 불완전 대리라, 미세
        # 개선은 offset 전환 transient로 실제 손해가 되므로 noise-level flip을 차단한다.
        self.offset_keep_margin: float = 0.005
        # 직전 step 수렴 결합값(warm-start).
        self._prev_coupling: Optional[Dict[str, float]] = None
        # ---------- 듀얼 분해(dual decomposition) N_P 추적용 상태 ----------
        # λ_P: 보호영역 net-inflow 결합제약 Σ_i nin_i = N_P_star 의 Lagrange 가격(≥0).
        # 매 control step warm-start(직전 수렴 λ에서 출발). 단위 = h(=cost veh·h / nin veh).
        self._lambda_P: float = 0.0
        # 듀얼 모드 사용 여부(leader present일 때만 ON). 고정가중 패널티(w_P) 대체.
        # True면 _solve_urban_agent_local의 N_P 항이 fixed-weight가 아니라 λ·nin_i가 된다.
        self.use_dual_np: bool = True
        # N_P 가격/패널티 항 전체 게이트(2026-07-07, green 이중가격 격리용). False면 green
        # 비용에서 N_P 항(dual λ_P·nin 또는 legacy w_P 패널티)을 완전히 뺀다 — g_ext(B2
        # green 가격)만 남겨 "N_P가 g_ext와 이중계상돼 잉여/충돌인가"를 판정. 기본 True.
        self.np_price_enabled: bool = True
        # subgradient 스텝 정규화 상수(차원무관 O(1); 시나리오 비의존). 실제 스텝 α는
        # 매 step 측정한 듀얼 gain G=|dΣnin/dλ|로 α = dual_step_c·cost_norm/max(G,G_floor)
        # 로 자기보정한다(스케일 인지·과적합 방지). 아래 _solve_followers 참고.
        self.dual_step_c: float = 1.0
        # λ step 간 적분 갱신 게인[h²/veh]: 오차 100 veh/h가 1 step에 λ를 ~1 움직이는 스케일.
        # 진단(2026-07-02): λ 응답 임계 ≈1, Σnin 바닥 도달 ≈10 — cap은 바닥 기준.
        self.lambda_np_step_gain: float = 0.01
        self.lambda_np_cap: float = 10.0
        # ---------- N_UF dual λ_UF(2026-07-07, 사용자 제안 — λ_P와 대칭) ----------
        # N_UF 조정을 hard 등식이 아니라 **결합제약 Σmeter = N_UF*의 dual 가격**으로 푼다
        # (wu_faithful_nuf_coordination_mode="dual"). freeway agent 비용에 λ_UF·Σ(owned
        # metering)을 더하고(등식 dual의 per-link 분해; link_budget 상수는 argmin 무관),
        # λ_UF는 control step 간 적분 갱신: λ_next = clip(λ + gain·(Σmeter − N_UF*), −cap,
        # +cap). **signed**(N_UF는 두 방향 target — Σmeter<target이면 λ<0로 방류 보상,
        # >target이면 λ>0로 억제). λ_P(비음수·유입 억제)와 달리 양방향. metering이 절벽
        # 레버라 방류 보상이 과방류→breakdown을 부를 수 있음(가설 검증 대상). warm-start.
        self._lambda_UF: float = 0.0
        self.lambda_uf_step_gain: float = 1.0e-5
        self.lambda_uf_cap: float = 1.0
        # 어댑터가 n_agents를 셀 때 쓰는 속성(six_controller 어댑터 호환).
        self.urban_agents = list(cfg.network.signals)
        self.freeway_agents = list(cfg.network.freeway_links)

    # ---------- per-movement 도착 (결합변수 movement 분해) ----------

    def _per_movement_arrivals(
        self,
        signal: str,
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, float]:
        """신호 i의 각 movement m에 대한 고정 도착유량 arr_m[veh/h]을 **소스에서 직접** 계산.

        `_coupling`의 `arr_{signal}_{pid}` phase 집계를 β로 재분배(re-smear)하지 않는다.
        `_coupling`이 합산한 것과 동일한 항을 movement 단위로 그대로 구한다. _coupling은
        movement마다 (a) kind-specific 외생 항과 (b) origin 링크가 상류 신호에서 받는 유입의
        β 몫을 **둘 다 더한다**(예: on_ramp movement D_N_to_onW는 origin이 internal 링크
        A_to_D라 ramp_arrival·β + A_to_D 유입·β를 모두 받는다). 따라서 두 항을 합산한다:
          (a) kind-specific:
              boundary_in → β_m × demand.urban_boundary[origin_m]
              on_ramp     → β_m × demand.ramp_arrival[ramp_m]
              off_ramp    → β_m × off_inflow(off_ramp_m)  (freeway 후보 VSL 동결값 재사용)
          (b) upstream:
              origin 링크 L이 상류 신호 leaving을 받으면 + β_m × inflow(L).
              inflow(L) = Σ_{producer dest==L} leaving_rate(producer)  (_coupling과 동일).
        보존: Σ_{m∈phase} arr_m == arr_{signal}_{phase}(_coupling)."""
        net = self.cfg.network
        wu = self._wu
        arr_m: Dict[str, float] = {}
        # 상류 신호가 먹이는 origin 링크별 총 유입유량[veh/h]을 한 번만 계산해 캐시.
        # _coupling과 동일하게 producer movement의 leaving rate를 합산한다.
        upstream_inflow_by_link: Dict[str, float] = {}
        for phase_id in ("p1", "p2"):
            key = f"{signal}_{phase_id}"
            for up_signal, up_movement, _up_beta in wu._upstream_leaving_map.get(key, []):
                origin_link = str(wu._specs[up_movement].get("destination", ""))
                if not origin_link:
                    continue
                if origin_link not in upstream_inflow_by_link:
                    upstream_inflow_by_link[origin_link] = 0.0
                upstream_inflow_by_link[origin_link] += wu._signal_leaving_rate(
                    up_signal, up_movement, control, state, demand,
                )

        for phase_id in ("p1", "p2"):
            for movement in self._phase_movements[signal][phase_id]:
                spec = self._specs[movement]
                kind = str(spec.get("kind", ""))
                beta = float(spec.get("beta", 0.0))
                origin = str(spec.get("origin", ""))
                arrival = 0.0
                # (a) kind-specific 외생 항.
                if kind == "boundary_in":
                    arrival += beta * max(0.0, demand.urban_boundary.get(origin, 0.0))
                elif kind == "on_ramp":
                    ramp = str(spec.get("ramp", ""))
                    arrival += beta * max(0.0, demand.ramp_arrival.get(ramp, 0.0))
                elif kind == "off_ramp":
                    off_ramp = str(spec.get("off_ramp", ""))
                    link = net.off_ramp_from_freeway.get(off_ramp, "")
                    if wu._has_last_offramp_flow:
                        off_inflow = float(wu._last_offramp_flow.get(off_ramp, 0.0))
                    else:
                        base = (
                            state.freeway_flow.get(link, [0.0])[-1]
                            if state.freeway_flow.get(link) else 0.0
                        )
                        off_inflow = _split_link_offramp_flow(self.cfg, link, off_ramp, base)
                    arrival += beta * max(0.0, off_inflow)
                # (b) origin 링크가 상류 신호 leaving을 받으면 그 β 몫도 더한다(_coupling과 동일).
                arrival += beta * max(0.0, upstream_inflow_by_link.get(origin, 0.0))
                arr_m[movement] = arrival
        return arr_m

    def _frozen_offramp_inflow(self, off_ramp: str, state: TrafficState) -> float:
        """off_ramp별 frozen freeway→off-ramp 유출[veh/h].

        `_coupling`의 freeway→urban 결합과 동일 소스: freeway agent 후보 VSL의 off-ramp
        유출 캐시(`_last_offramp_flow`), 없으면 현재 본선 유량 폴백(`_split_link_offramp_flow`).
        `_per_movement_arrivals`가 β로 분배하기 전 per-off_ramp 원값이다."""
        net = self.cfg.network
        wu = self._wu
        link = net.off_ramp_from_freeway.get(off_ramp, "")
        if wu._has_last_offramp_flow:
            return max(0.0, float(wu._last_offramp_flow.get(off_ramp, 0.0)))
        base = (
            state.freeway_flow.get(link, [0.0])[-1]
            if state.freeway_flow.get(link) else 0.0
        )
        return max(0.0, _split_link_offramp_flow(self.cfg, link, off_ramp, base))

    def _frozen_freeway_congestion(self, state: TrafficState) -> Dict[str, float]:
        """ramp별 frozen freeway 혼잡 가중 w_fw ∈ [0,1] — de facto ramp metering(SPEC line 28).

        merge 지점 ρ로 `compute_ramp_release_flows`의 receiving_factor를 복제해
        w_fw = 1 − receiving_factor. freeway가 막히면(ρ_merge↑) w_fw→1, free-flow면 →0.
        on-ramp reservoir 적재(→freeway 유입)에 이 가중을 곱해 비용에 넣으면, 막힌 freeway로
        차를 더 보내는 p1(on_ramp 위주) green이 비용으로 잡혀 p2-heavy로 기운다. freeway가
        자유흐름이면 w_fw≈0이라 순수 국소 거동을 회복(무해)."""
        from src.models.metanet import _ramp_merge_index, _clip
        net = self.cfg.network
        w: Dict[str, float] = {}
        for ramp in net.ramps:
            link = net.ramp_to_freeway.get(ramp, "")
            densities = state.freeway_density.get(link, [])
            if not densities:
                w[ramp] = 0.0
                continue
            merge_idx = _ramp_merge_index(self.cfg, ramp, len(densities))
            rho_merge = densities[merge_idx]
            receiving_factor = _clip(
                (net.rho_max - rho_merge) / max(net.rho_max - net.rho_crit, 1.0e-9),
                0.0, 1.0,
            )
            w[ramp] = float(max(0.0, 1.0 - receiving_factor))
        return w

    def _frozen_reservoir_drain(
        self, state: TrafficState, control: ControlAction, demand: DemandStep,
    ) -> Dict[str, float]:
        """ramp별 frozen reservoir→freeway 방출률[veh/h](freeway가 reservoir를 비우는 속도).

        실제 plant는 매 T_f 경계에서 `compute_ramp_release_flows`(ρ_merge 기반 수용)로
        reservoir(w_r)를 freeway로 비운다. 국소 rollout이 reservoir 유출을 0으로 동결하면
        w_r이 ramp_queue_max에 고정돼 on_ramp green이 무력해진다(잘못된 flat 비용). 따라서
        freeway 본선 ρ로 결정되는 이 방출률을 동결 결합값으로 받아 substep마다 reservoir를
        비운다(green→reservoir 적재 vs freeway→reservoir 배출의 상충이 보이게)."""
        release, _ = compute_ramp_release_flows(state, control, demand, self.cfg)
        return {ramp: max(0.0, float(v)) for ramp, v in release.items()}

    def _offramp_occupancy(self, off_ramp: str, state: TrafficState) -> float:
        """off-ramp storage 초기 점유[veh] = cap − available(plant `_drain_offramp_storage` 정의)."""
        net = self.cfg.network
        storage = net.off_ramp_storage_link.get(off_ramp, "")
        if not storage:
            return 0.0
        cap = float(net.urban_link_storage_veh.get(storage, 0.0))
        return max(0.0, cap - float(state.urban_link_storage.get(storage, cap)))

    # ---------- 듀얼 분해용 net-inflow 정의(리더와 동일 소스) ----------

    def _movement_forecast_arrivals_veh(self, forecast: List[DemandStep]) -> Dict[str, float]:
        """movement별 horizon 도착량[veh] — 리더 `_movement_forecast_arrivals_veh`(distributed_
        coordinator.py 3147~3177)의 충실 복제. net-inflow의 available_m = queue_m + 이 도착량을
        리더와 동일하게 맞추기 위함(off_ramp 도착은 여기 없고, _agent_net_inflow_veh에서 frozen
        freeway→off-ramp 유출로 따로 더한다 — 리더 진단도 off_ramp는 movement_arrivals에 없다)."""
        net = self.cfg.network
        dt_h = self.cfg.simulation.T_c_h
        steps = forecast[: max(1, self.cfg.mpc.horizon_steps)] or forecast[:1]
        arrivals: Dict[str, float] = {}
        onramp_by_movement = {
            movement: ramp
            for ramp, movements in net.on_ramp_to_movement.items()
            for movement in movements
        }
        for step in steps:
            for movement, spec in self._specs.items():
                kind = str(spec.get("kind", ""))
                if kind == "boundary_in":
                    origin = str(spec.get("origin", ""))
                    beta = float(spec.get("beta", 1.0))
                    arrivals[movement] = arrivals.get(movement, 0.0) + (
                        max(0.0, step.urban_boundary.get(origin, 0.0)) * beta * dt_h
                    )
                elif kind == "on_ramp":
                    ramp = onramp_by_movement.get(movement, "")
                    if not ramp:
                        continue
                    movements = net.on_ramp_to_movement.get(ramp, [])
                    share = 1.0 / max(len(movements), 1)
                    arrivals[movement] = arrivals.get(movement, 0.0) + (
                        max(0.0, step.ramp_arrival.get(ramp, 0.0)) * share * dt_h
                    )
        return arrivals

    def _agent_net_inflow_veh(
        self,
        signal: str,
        green_p1: float,
        state: TrafficState,
        forecast_arrivals: Mapping[str, float],
        horizon_h: float,
    ) -> float:
        """신호 i의 보호영역 net-inflow nin_i(green_p1)[veh, horizon 적분] — 리더와 동일 정의.

        리더 `_leader_direct_feasible_set_diagnostics`(distributed_coordinator.py 757~799)의
        served 공식을 신호 i movement에만 적용한다:
          available_m = queue_m + forecast_arrivals_m  (off_ramp는 frozen freeway 유출·β·horizon)
          served_m    = min(available_m, horizon_h · green_fraction(phase) · cap_flow_m)
          (on_ramp는 ramp reservoir 여유로 추가 스케일 — 리더와 동일)
          nin_i = Σ_{kind∈INFLOW} served_m − Σ_{kind∈OUTFLOW} served_m
        INFLOW={boundary_in, off_ramp}, OUTFLOW={boundary_out, on_ramp}(inflow_outflow_allocation.py).
        green_fraction은 cycle 평균(green_sec/cycle) — 리더 `_phase_green_fraction`(urban_step_index
        =None)과 동일. 이 정의가 리더 N_P_star(=total net inflow target)와 직접 비교 가능하게 한다."""
        net = self.cfg.network
        model = self._local_models[signal]
        total = net.effective_green_total
        cycle = max(net.cycle_length, 1.0e-9)
        green = {"p1": float(green_p1), "p2": float(total - green_p1)}

        served: Dict[str, float] = {}
        raw_onramp_by_ramp: Dict[str, float] = {}
        onramp_by_movement = {
            m: r for r, mvs in net.on_ramp_to_movement.items() for m in mvs
        }
        for movement in model.movements:
            spec = self._specs[movement]
            kind = model.kind_of[movement]
            available = max(0.0, float(state.urban_movement_queue.get(movement, 0.0)))
            if kind == "off_ramp":
                off_ramp = str(spec.get("off_ramp", ""))
                inflow = self._frozen_offramp_inflow(off_ramp, state)
                available += inflow * horizon_h * float(spec.get("beta", 0.0))
            else:
                available += max(0.0, float(forecast_arrivals.get(movement, 0.0)))
            green_fraction = green[model.phase_of[movement]] / cycle
            cap_veh = horizon_h * green_fraction * model.cap_flow_of[movement]
            s = min(available, max(0.0, cap_veh))
            served[movement] = s
            if kind == "on_ramp":
                ramp = onramp_by_movement.get(movement, "")
                if ramp:
                    raw_onramp_by_ramp[ramp] = raw_onramp_by_ramp.get(ramp, 0.0) + s
        # on_ramp 서비스는 ramp reservoir 여유로 추가 스케일(리더 776~783 복제).
        for ramp, raw_total in raw_onramp_by_ramp.items():
            if raw_total <= 1.0e-9:
                continue
            ramp_space = max(
                0.0,
                float(net.ramp_queue_max_veh) - max(0.0, float(state.ramp_queue.get(ramp, 0.0))),
            )
            scale = min(1.0, ramp_space / raw_total)
            for movement in net.on_ramp_to_movement.get(ramp, []):
                if movement in served:
                    served[movement] *= scale
        inflow_veh = sum(
            served[m] for m in model.movements if model.kind_of[m] in _INFLOW_KINDS
        )
        outflow_veh = sum(
            served[m] for m in model.movements if model.kind_of[m] in _OUTFLOW_KINDS
        )
        return float(inflow_veh - outflow_veh)

    # ---------- per-signal green 후보 구성 (solver·feasible-range 공용) ----------

    def _urban_green_candidates(
        self,
        signal: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        snapshot: ControlAction,
    ) -> List[float]:
        # _solve_urban_agent_local의 green-p1 후보 구성을 그대로 떼어낸 공용 헬퍼 — solver와
        # _np_feasible_range가 **동일한** 후보집합을 보도록 한다(PFO/WU green 거동 불변).
        net = self.cfg.network
        sim = self.cfg.simulation
        model = self._local_models[signal]
        total = net.effective_green_total
        horizon = max(1, self.cfg.mpc.horizon_steps)
        substeps = horizon * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        q0 = {m: max(0.0, state.urban_movement_queue.get(m, 0.0)) for m in model.movements}
        arr_phase = {pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")}

        prev_p1 = float(snapshot.green_times.get(f"{signal}_p1", total / 2.0))
        # pressure 중심 + 주변 후보(완화 양자화). 기존 _solve_urban_agent와 같은 후보 구성 철학.
        p1_pressure = q0_sum(q0, model, "p1") + arr_phase["p1"] * dt_h * substeps
        p2_pressure = q0_sum(q0, model, "p2") + arr_phase["p2"] * dt_h * substeps
        pressure_center = queue_pressure_green_target(p1_pressure, p2_pressure, self.cfg)
        raw_candidates = [total / 2.0, prev_p1, pressure_center]
        if self.cfg.mpc.relaxed_quantized_controls:
            raw_candidates.extend([
                pressure_center - 1.0, pressure_center + 1.0,
                pressure_center - 2.0, pressure_center + 2.0,
                pressure_center - 5.0, pressure_center + 5.0,
            ])
        # 진짜 국소 rollout은 신호 1개만 돌아 싸므로 전 green 범위를 굵게 훑어 실제 국소
        # 최적을 찾는다(pressure-center 밴드는 옛 집계모델용이라 좁아 56을 못 벗어났다 —
        # 의도적 deviation, SPEC §2의 "argmin J_i,local" 충실). 후보 폭발 없음(13점/신호).
        raw_candidates.extend(float(v) for v in np.linspace(net.green_min, net.green_max, 13))

        candidates: List[float] = []
        for raw in raw_candidates:
            if self.cfg.mpc.relaxed_quantized_controls:
                p1_value = repair_green_pair(float(raw), self.cfg).p1
            else:
                p1_value = float(np.clip(raw, net.green_min, net.green_max))
                p2_value = total - p1_value
                if p2_value < net.green_min:
                    p1_value = total - net.green_min
                if p2_value > net.green_max:
                    p1_value = total - net.green_max
            if not any(abs(p1_value - existing) <= 1.0e-9 for existing in candidates):
                candidates.append(float(p1_value))
        return candidates

    # ---------- per-signal 국소 agent solve (핵심 신규) ----------

    def _solve_urban_agent_local(
        self,
        signal: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        arr_movement: Mapping[str, float],
        s_eff_frozen: Mapping[str, float],
        reservoir_drain: Mapping[str, float],
        freeway_congestion: Mapping[str, float],
        previous: ControlAction,
        leader: Optional[object] = None,
        lambda_p: float = 0.0,
        forecast_arrivals: Optional[Mapping[str, float]] = None,
        horizon_h: float = 1.0,
        demand: Optional[DemandStep] = None,
        candidates_override: Optional[Sequence[float]] = None,
        committed_prev: Optional[ControlAction] = None,
    ) -> tuple[float, float, int, float]:
        """green p1 후보 탐색 — 반환 (p1*, 자기 TTS objective, evaluations, nin_i*).

        후보 채점은 `rollout_local_tts`로 **신호 i movement만** 전진(전체망 plant 호출 없음).
        arr_movement: `_per_movement_arrivals`가 소스에서 직접 구한 movement별 도착유량.
        결합변수는 frozen이므로 phase 합이 frozen arr_{signal}_{pid}와 일치하도록 재정규화한다
        (phase 내 재귀속만, phase 총량 보존).

        리더 present + use_dual_np면 N_P 추적을 **듀얼 분해**로 한다: 후보 비용에
        `+ λ_P·nin_i(green)`(가격×자기 net-inflow)을 더한다. λ_P가 크면 agent가 자기 net
        inflow를 줄이는 green(boundary inflow hold-back)을 선호한다. 고정가중 w_P 패널티는
        쓰지 않는다(use_dual_np=True). nin_i*는 선택 후보의 net-inflow로, 호출처(_solve_followers)가
        Σ_i nin_i를 모아 subgradient λ 갱신에 쓴다. leader=None(PFO)이면 λ=0·항 없음 → +56.63% 보존."""
        net = self.cfg.network
        sim = self.cfg.simulation
        model = self._local_models[signal]
        total = net.effective_green_total
        horizon = max(1, self.cfg.mpc.horizon_steps)
        substeps = horizon * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        smooth_w = self.cfg.urban_follower.green_smoothness_weight

        # 자기 movement 초기 큐.
        q0 = {m: max(0.0, state.urban_movement_queue.get(m, 0.0)) for m in model.movements}
        # phase 단위 고정 도착(결합변수, frozen).
        arr_phase = {pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")}
        # ramp-aware 신호(D/F): off-ramp 유입을 phase 큐에서 분리해 storage로 보낸다. frozen
        # arr_phase는 `_coupling`에서 off-ramp inflow·β를 포함하므로, queue 도착에는 그 몫을
        # 빼고(phase별 off-ramp 기여), off-ramp inflow는 storage 유입으로 따로 넘긴다.
        offramp_inflow: Dict[str, float] = {}
        offramp_contrib_phase = {"p1": 0.0, "p2": 0.0}
        if model.has_ramps:
            for off_ramp, movements in model.offramp_movements.items():
                inflow = self._frozen_offramp_inflow(off_ramp, state)
                offramp_inflow[off_ramp] = inflow
                # off_ramp movement는 모두 같은 phase(Σβ=1.0); 그 phase 큐 기여 = inflow.
                for m in movements:
                    offramp_contrib_phase[model.phase_of[m]] += model.beta_of[m] * inflow
        # movement별 도착을 frozen phase 총량(off-ramp 몫 제외)에 맞춰 재정규화.
        arr_mv: Dict[str, float] = {}
        for pid in ("p1", "p2"):
            # off_ramp movement는 큐 도착 대상이 아님(storage로 유입).
            phase_movements = [
                m for m in model.movements
                if model.phase_of[m] == pid and model.kind_of[m] != "off_ramp"
            ]
            raw_sum = sum(max(0.0, float(arr_movement.get(m, 0.0))) for m in phase_movements)
            target = max(0.0, arr_phase[pid] - offramp_contrib_phase[pid])
            if raw_sum > 1.0e-12:
                scale = target / raw_sum
                for m in phase_movements:
                    arr_mv[m] = max(0.0, float(arr_movement.get(m, 0.0))) * scale
            else:
                for m in phase_movements:
                    arr_mv[m] = 0.0
        # off-ramp storage 초기 점유·on-ramp reservoir 초기 큐 스냅샷(자기 권역).
        offramp_occ0 = {
            off_ramp: self._offramp_occupancy(off_ramp, state)
            for off_ramp in model.offramp_movements
        }
        ramp_queue0 = {
            ramp: max(0.0, float(state.ramp_queue.get(ramp, 0.0)))
            for ramp in model.onramp_movements
        }
        # 이 신호 movement들의 receiving 링크 S_eff(동결 스냅샷).
        s_eff0 = {
            model.receiving_of[m]: float(s_eff_frozen.get(model.receiving_of[m], 0.0))
            for m in model.movements
            if model.receiving_of[m]
        }

        # ---------- PHASE-RESOLVED + PLATOON 셋업(non-ramp 신호; Task-B) ----------
        # non-ramp 신호는 cycle-평균 균일 도착 대신 phase-resolved 서비스 + platoon 도착으로
        # 채점한다. platoon profile은 상류 신호(snapshot 동결)에만 의존하므로 후보 green 루프
        # **밖에서 1회** 계산한다(green/offset에 불변). gf만 후보별로 갱신한다.
        # ramp-aware(D/F)는 storage 동역학이 복잡해 기존 cycle-평균 경로 유지(Task-B 범위 밖).
        use_phased = (not model.has_ramps) and demand is not None
        # P1.5: D/F(ramp-aware)도 phase-resolved + platoon 도착으로 채점. 상시 플래그
        # (ramp_aware_phase_resolved) 또는 auto 게이트가 이 신호를 활성화했을 때만.
        use_phased_ramp = (
            model.has_ramps
            and demand is not None
            and (
                self.ramp_aware_phase_resolved
                or signal in self._phase_resolved_active_signals
            )
        )
        start_idx = _urban_step_index(state, self.cfg)
        # green search 동안 offset은 snapshot 값(직전 best-response)으로 동결한다.
        offset_for_green = float(previous.offsets.get(signal, 0.0))
        arr_by_substep: Dict[str, List[float]] = {}
        if use_phased or use_phased_ramp:
            arr_by_substep = self._platoon_arrival_profiles(
                signal, state, previous, demand, arr_mv, substeps, start_idx,
            )
        if use_phased_ramp:
            # off_ramp movement는 큐 도착이 아니라 storage 유입(단계 (c))이므로 프로파일에서
            # 제외한다(rollout의 도착 주입은 queue_movements만 읽지만 명시적으로 필터).
            arr_by_substep = {
                m: prof for m, prof in arr_by_substep.items()
                if model.kind_of.get(m) != "off_ramp"
            }

        prev_p1 = float(previous.green_times.get(f"{signal}_p1", total / 2.0))
        # candidates_override: B2 가격 계산 등 외부에서 특정 후보만 채점할 때(단일 후보면
        # best_obj가 곧 그 후보의 cost). None이면 기존 후보 구성 그대로.
        if candidates_override is not None:
            candidates = [float(v) for v in candidates_override]
        else:
            candidates = self._urban_green_candidates(signal, state, coupling, previous)
            # B2.1 trust region: 가격 활성 + trust 설정 시, 가격의 선형 근사가 유효한
            # 유한차분 이웃(|p1 − ref| ≤ trust) 밖 후보를 제외한다. 이웃 안 후보가 하나도
            # 없으면 안전하게 전체 후보로 fallback(가격 월권보다 자율 탐색이 낫다).
            if (
                self.signal_marginal_price is not None
                and self.signal_marginal_price_trust_sec is not None
                and signal in self.signal_marginal_price
            ):
                trust = float(self.signal_marginal_price_trust_sec)
                ref = float(self.signal_marginal_price_ref.get(signal, prev_p1))
                trusted = [
                    p1 for p1 in candidates if abs(p1 - ref) <= trust + 1.0e-9
                ]
                if trusted:
                    candidates = trusted
            # BASELINE-BOX green(2026-07-17): PFO는 가격 부재 → 위 trust 미발동 →
            # green 무제한(실측 per-step 최대 57s). walk-MVG와 동일 한계 ±6s.
            # ★앵커는 committed_prev(직전 step commit)다 — 이 함수의 previous 인자는
            # 메인 루프가 snapshot(Jacobi 반복값)을 넘겨서(L3791) sweep마다 재앵커되면
            # 스텝당 12s 누수(실측). VSL 구멍과 동일 패턴.
            if (bool(getattr(self.cfg.mpc, "baseline_move_box", False))
                    and committed_prev is not None):
                _bg = float(committed_prev.green_times.get(f"{signal}_p1", prev_p1))
                _bk = [p1 for p1 in candidates if abs(p1 - _bg) <= 6.0 + 1.0e-9]
                candidates = _bk or [min(candidates, key=lambda p1: abs(p1 - _bg))]

        # Leader N_P 추적. 두 모드:
        #  (A) use_dual_np=True(기본, 듀얼 분해): 후보 비용에 + λ_P·nin_i(green)을 더한다.
        #      λ_P는 결합제약 Σ_i nin_i = N_P_star의 가격으로, _solve_followers가 subgradient
        #      ascent로 갱신한다. 가격이 높을수록 자기 net inflow가 낮은 green을 선호 → boundary
        #      inflow hold-back. nin_i(green)은 리더와 동일 served 정의(_agent_net_inflow_veh).
        #  (B) use_dual_np=False(레거시 고정가중): 옛 w_P setpoint 패널티(추적 안 됨). 비교용 보존.
        # leader=None(PFO)이면 두 모드 다 항 0 → 기존 +56.63% 거동 그대로 보존.
        dual_mode = leader is not None and self.use_dual_np
        legacy_mode = leader is not None and not self.use_dual_np
        n_p_star = float(getattr(leader, "N_P_star", 0.0)) if leader is not None else 0.0
        w_p = float(self.cfg.leader.w_P)
        omega_p = float(self._wu._omega_p.get(signal, 0.0))
        np_setpoint = omega_p * n_p_star
        cost_norm = max(1.0e-9, float(substeps) * dt_h)
        fa = forecast_arrivals if forecast_arrivals is not None else {}

        best_p1, best_obj, best_nin = prev_p1, float("inf"), 0.0
        evals = 0
        for p1 in candidates:
            p2 = total - p1
            if p2 < net.green_min - 1.0e-9 or p2 > net.green_max + 1.0e-9:
                continue
            if model.has_ramps:
                if use_phased_ramp:
                    # phase-resolved 서비스(offset-aware) + platoon 도착(P1.5). offset은
                    # 비-ramp 경로와 동일하게 snapshot 값으로 동결, gf만 후보별 갱신.
                    gf_by_substep = self._offset_green_fractions(
                        signal, p1, offset_for_green, substeps, start_idx,
                    )
                    cost = rollout_local_tts_ramp_aware(
                        model, q0, arr_mv, s_eff0,
                        offramp_inflow, offramp_occ0, ramp_queue0, reservoir_drain,
                        freeway_congestion, self.ramp_metering_weight,
                        p1, p2, substeps, dt_h,
                        arr_by_substep=arr_by_substep,
                        gf_by_substep=gf_by_substep,
                    )
                else:
                    cost = rollout_local_tts_ramp_aware(
                        model, q0, arr_mv, s_eff0,
                        offramp_inflow, offramp_occ0, ramp_queue0, reservoir_drain,
                        freeway_congestion, self.ramp_metering_weight,
                        p1, p2, substeps, dt_h,
                    )
            elif use_phased:
                # phase-resolved 서비스(offset-aware) + platoon 도착. offset은 green search 동안
                # snapshot 값으로 동결(offset best-response는 _solve_offset_local에서 별도).
                gf_by_substep = self._offset_green_fractions(
                    signal, p1, offset_for_green, substeps, start_idx,
                )
                cost = rollout_local_tts_phased(
                    model, q0, arr_by_substep, gf_by_substep, s_eff0, substeps, dt_h,
                )
            else:
                cost = rollout_local_tts(
                    model, q0, arr_mv, s_eff0, p1, p2, substeps, dt_h,
                )
            # PRICE-TR: 가격 활성 신호는 smoothness 마찰 0(trust region이 보폭 제약).
            if not (
                self.price_smoothness_disabled
                and self.signal_marginal_price is not None
                and signal in self.signal_marginal_price
            ):
                cost += smooth_w * abs(p1 - prev_p1)
            # B2 가격항: leader가 하달한 per-signal externality 가격(설정 시에만).
            # own-TTS는 그대로 두고 선형 가격만 더한다 — Step B1 검증 형태
            # priced = local + w·g_ext_i·(p1 − p1_ref_i). None이면 완전 휴면(비트 동일).
            if self.signal_marginal_price is not None:
                g_ext = self.signal_marginal_price.get(signal)
                if g_ext is not None:
                    ref = float(self.signal_marginal_price_ref.get(signal, prev_p1))
                    cost += self.signal_marginal_price_weight * float(g_ext) * (float(p1) - ref)
            # nin_i(green)은 리더 setpoint와 비교 가능한 net-inflow(듀얼·진단 공통).
            nin = self._agent_net_inflow_veh(signal, p1, state, fa, horizon_h)
            if self.np_price_enabled and dual_mode:
                cost += lambda_p * nin
            elif self.np_price_enabled and legacy_mode and w_p > 0.0:
                mean_accum = cost / cost_norm
                cost += w_p * max(0.0, mean_accum - np_setpoint) * cost_norm
            evals += 1
            if cost < best_obj:
                best_obj, best_p1, best_nin = cost, float(p1), float(nin)
        return best_p1, best_obj, evals, best_nin

    # ---------- B2: leader가 부르는 국소 green 비용 조회(가격항 제외) ----------

    def local_green_costs(
        self,
        requests: Mapping[str, Sequence[float]],
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, List[float]]:
        """신호별 green-p1 후보들의 국소 own-TTS 비용을 반환한다(B2 d_local 유한차분용).

        `_solve_followers` 프롤로그(동결 결합/스냅샷)를 1회 구성하고, 각 요청 후보를
        `_solve_urban_agent_local`(단일 후보, leader=None)로 채점한다 — Step A/B probe의
        score_candidate와 동일 경로. 가격항은 일시 비활성화한다: d_local은 **비가격**
        own-TTS의 기울기여야 한다(가격이 들어가면 g_ext = g_i − d_local이 자기 가격을
        다시 빼는 순환). warm-start coupling(`_prev_coupling`)은 읽지도 쓰지도 않아
        follower 영속 상태를 오염시키지 않는다.
        """
        ctrl = ControlAction.uncontrolled(self.cfg)
        ctrl.green_times = dict(control.green_times)
        ctrl.offsets = dict(control.offsets)
        ctrl.vsl = dict(control.vsl)
        ctrl.ramp_metering = dict(control.ramp_metering)
        ctrl.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, ctrl, demand)
        s_eff_frozen = self._frozen_s_eff(state)
        reservoir_drain = self._frozen_reservoir_drain(state, ctrl, demand)
        freeway_congestion = self._frozen_freeway_congestion(state)
        snapshot = ControlAction(
            ramp_metering=dict(ctrl.ramp_metering),
            vsl=dict(ctrl.vsl),
            green_times=dict(ctrl.green_times),
            offsets=dict(ctrl.offsets),
            inflow_outflow_allocation={},
        )
        saved_price = self.signal_marginal_price
        self.signal_marginal_price = None
        out: Dict[str, List[float]] = {}
        try:
            for signal, p1_list in requests.items():
                arr_movement = self._per_movement_arrivals(signal, state, snapshot, demand)
                costs: List[float] = []
                for p1 in p1_list:
                    _, obj, _, _ = self._solve_urban_agent_local(
                        signal, state.copy(), coupling, arr_movement, s_eff_frozen,
                        reservoir_drain, freeway_congestion, snapshot,
                        None, 0.0, None, 1.0, demand,
                        candidates_override=[float(p1)],
                    )
                    costs.append(float(obj))
                out[signal] = costs
        finally:
            self.signal_marginal_price = saved_price
        return out

    # ---------- B3: leader가 부르는 국소 metering 비용 조회(가격항 제외) ----------

    def local_metering_costs(
        self,
        requests: Mapping[str, Sequence[float]],
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, List[float]]:
        """ramp별 metering 후보들의 freeway-agent 국소 own-TTS 비용(B3 d_local 유한차분용).

        local_green_costs와 동일 규약: 프롤로그(동결 결합) 1회 구성, 가격은 일시 비활성
        (g_ext가 자기 가격을 다시 빼는 순환 방지), 영속 상태 미변경. 각 후보는 그 ramp
        소유 link의 `_solve_freeway_agent_local`(VSL best-response 포함 own-TTS)로
        채점한다 — follower가 metering 후보를 채점하는 `_solve_with`와 동일 경로.
        """
        ctrl = ControlAction.uncontrolled(self.cfg)
        ctrl.green_times = dict(control.green_times)
        ctrl.offsets = dict(control.offsets)
        ctrl.vsl = dict(control.vsl)
        ctrl.ramp_metering = dict(control.ramp_metering)
        ctrl.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, ctrl, demand)
        saved_meter_price = self.metering_marginal_price
        saved_vsl_price = self.vsl_marginal_price
        self.metering_marginal_price = None
        self.vsl_marginal_price = None
        out: Dict[str, List[float]] = {}
        try:
            for ramp, values in requests.items():
                link = self.cfg.network.ramp_to_freeway.get(ramp)
                if link is None:
                    out[ramp] = [0.0 for _ in values]
                    continue
                costs: List[float] = []
                for x in values:
                    probe_prev = ControlAction(
                        ramp_metering=dict(ctrl.ramp_metering),
                        vsl=dict(ctrl.vsl),
                        green_times=dict(ctrl.green_times),
                        offsets=dict(ctrl.offsets),
                        inflow_outflow_allocation={},
                    )
                    probe_prev.ramp_metering[ramp] = float(x)
                    _, cost, _ = self._solve_freeway_agent_local(
                        link, state, coupling, demand, probe_prev,
                    )
                    costs.append(float(cost))
                out[ramp] = costs
        finally:
            self.metering_marginal_price = saved_meter_price
            self.vsl_marginal_price = saved_vsl_price
        return out

    # ---------- F3: leader가 부르는 국소 offset 비용 조회(가격항 제외) ----------

    def local_offset_costs(
        self,
        requests: Mapping[str, Sequence[float]],
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, List[float]]:
        """신호별 offset 후보들의 국소 phased 비용(F3 d_local 유한차분용).

        local_green_costs와 동일 규약(프롤로그 1회, 가격 일시비활성, 영속상태 미변경).
        채점은 `_solve_offset_local`과 동일 경로 — snapshot 동결 platoon profile(offset
        불변, 신호당 1회) + 후보 offset의 offset-aware gf로 phased rollout. ramp 신호
        (D/F)는 offset 탐색 제외 대상이므로 0 반환."""
        net = self.cfg.network
        sim = self.cfg.simulation
        cycle = max(net.cycle_length, 1.0e-9)
        substeps = max(1, self.cfg.mpc.horizon_steps) * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        start_idx = _urban_step_index(state, self.cfg)
        ctrl = ControlAction.uncontrolled(self.cfg)
        ctrl.green_times = dict(control.green_times)
        ctrl.offsets = dict(control.offsets)
        ctrl.vsl = dict(control.vsl)
        ctrl.ramp_metering = dict(control.ramp_metering)
        ctrl.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, ctrl, demand)
        s_eff_frozen = self._frozen_s_eff(state)
        snapshot = ControlAction(
            ramp_metering=dict(ctrl.ramp_metering),
            vsl=dict(ctrl.vsl),
            green_times=dict(ctrl.green_times),
            offsets=dict(ctrl.offsets),
            inflow_outflow_allocation={},
        )
        saved_price = self.offset_marginal_price
        self.offset_marginal_price = None
        out: Dict[str, List[float]] = {}
        try:
            for signal, offsets in requests.items():
                model = self._local_models[signal]
                if model.has_ramps:
                    out[signal] = [0.0 for _ in offsets]
                    continue
                green_p1 = float(ctrl.green_times.get(
                    f"{signal}_p1", net.effective_green_total / 2.0
                ))
                arr_movement = self._per_movement_arrivals(signal, state, snapshot, demand)
                arr_phase = {
                    pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0))
                    for pid in ("p1", "p2")
                }
                arr_mv: Dict[str, float] = {}
                for pid in ("p1", "p2"):
                    phase_movements = [
                        m for m in model.movements
                        if model.phase_of[m] == pid and model.kind_of[m] != "off_ramp"
                    ]
                    raw_sum = sum(
                        max(0.0, float(arr_movement.get(m, 0.0))) for m in phase_movements
                    )
                    target = max(0.0, arr_phase[pid])
                    if raw_sum > 1.0e-12:
                        scale = target / raw_sum
                        for m in phase_movements:
                            arr_mv[m] = max(0.0, float(arr_movement.get(m, 0.0))) * scale
                    else:
                        for m in phase_movements:
                            arr_mv[m] = 0.0
                q0 = {
                    m: max(0.0, float(state.urban_movement_queue.get(m, 0.0)))
                    for m in model.movements
                }
                s_eff0 = {
                    model.receiving_of[m]: float(
                        s_eff_frozen.get(model.receiving_of[m], 0.0)
                    )
                    for m in model.movements
                    if model.receiving_of[m]
                }
                arr_by_substep = self._platoon_arrival_profiles(
                    signal, state, snapshot, demand, arr_mv, substeps, start_idx,
                )
                costs: List[float] = []
                for off in offsets:
                    offset = float(off) % cycle
                    gf_by_substep = self._offset_green_fractions(
                        signal, green_p1, offset, substeps, start_idx,
                    )
                    costs.append(float(rollout_local_tts_phased(
                        model, q0, arr_by_substep, gf_by_substep, s_eff0, substeps, dt_h,
                    )))
                out[signal] = costs
        finally:
            self.offset_marginal_price = saved_price
        return out

    # ---------- JOINT probe: (green_p1, offset) 쌍 own_TTS(가격 OFF, 신호당 셋업 1회) ----------

    def local_green_offset_costs(
        self,
        requests: Mapping[str, Sequence[tuple]],
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, List[float]]:
        """신호별 (p1, offset) 쌍들의 국소 phased own_TTS — cross-term h_local 4-corner용.

        local_offset_costs와 동일 규약. 셋업(arr_mv·q0·s_eff0·platoon)은 ego green·offset에
        불변이라 신호당 1회 계산 후 쌍마다 gf만 갱신해 채점한다. ramp 신호는 0 반환."""
        net = self.cfg.network
        sim = self.cfg.simulation
        cycle = max(net.cycle_length, 1.0e-9)
        substeps = max(1, self.cfg.mpc.horizon_steps) * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        start_idx = _urban_step_index(state, self.cfg)
        ctrl = ControlAction.uncontrolled(self.cfg)
        ctrl.green_times = dict(control.green_times)
        ctrl.offsets = dict(control.offsets)
        ctrl.vsl = dict(control.vsl)
        ctrl.ramp_metering = dict(control.ramp_metering)
        ctrl.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, ctrl, demand)
        s_eff_frozen = self._frozen_s_eff(state)
        snapshot = ControlAction(
            ramp_metering=dict(ctrl.ramp_metering),
            vsl=dict(ctrl.vsl),
            green_times=dict(ctrl.green_times),
            offsets=dict(ctrl.offsets),
            inflow_outflow_allocation={},
        )
        out: Dict[str, List[float]] = {}
        for signal, pairs in requests.items():
            model = self._local_models[signal]
            if model.has_ramps:
                out[signal] = [0.0 for _ in pairs]
                continue
            arr_movement = self._per_movement_arrivals(signal, state, snapshot, demand)
            arr_phase = {
                pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")
            }
            arr_mv: Dict[str, float] = {}
            for pid in ("p1", "p2"):
                phase_movements = [
                    m for m in model.movements
                    if model.phase_of[m] == pid and model.kind_of[m] != "off_ramp"
                ]
                raw_sum = sum(max(0.0, float(arr_movement.get(m, 0.0))) for m in phase_movements)
                target = max(0.0, arr_phase[pid])
                if raw_sum > 1.0e-12:
                    scale = target / raw_sum
                    for m in phase_movements:
                        arr_mv[m] = max(0.0, float(arr_movement.get(m, 0.0))) * scale
                else:
                    for m in phase_movements:
                        arr_mv[m] = 0.0
            q0 = {
                m: max(0.0, float(state.urban_movement_queue.get(m, 0.0)))
                for m in model.movements
            }
            s_eff0 = {
                model.receiving_of[m]: float(s_eff_frozen.get(model.receiving_of[m], 0.0))
                for m in model.movements
                if model.receiving_of[m]
            }
            arr_by_substep = self._platoon_arrival_profiles(
                signal, state, snapshot, demand, arr_mv, substeps, start_idx,
            )
            costs: List[float] = []
            for pair in pairs:
                p1 = float(pair[0])
                offset = float(pair[1]) % cycle
                gf_by_substep = self._offset_green_fractions(
                    signal, p1, offset, substeps, start_idx,
                )
                costs.append(float(rollout_local_tts_phased(
                    model, q0, arr_by_substep, gf_by_substep, s_eff0, substeps, dt_h,
                )))
            out[signal] = costs
        return out

    def local_vsl_meter_costs(
        self,
        requests: Mapping[str, Sequence[tuple]],
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, List[float]]:
        """ramp별 (meter, vsl) 쌍들의 freeway-agent 국소 own-TTS — vsl×metering h_local 4-corner용.

        각 쌍은 metering을 probe_prev에 고정하고 vsl_override로 VSL을 단일값에 고정해
        `_solve_freeway_agent_local`(단일 시퀀스)로 채점한다. 가격은 일시 비활성(자기 가격
        재차감 순환 방지). 영속 상태 미변경."""
        ctrl = ControlAction.uncontrolled(self.cfg)
        ctrl.green_times = dict(control.green_times)
        ctrl.offsets = dict(control.offsets)
        ctrl.vsl = dict(control.vsl)
        ctrl.ramp_metering = dict(control.ramp_metering)
        ctrl.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, ctrl, demand)
        n_seg = self.cfg.network.freeway_segments_per_link
        saved_meter = self.metering_marginal_price
        saved_vsl = self.vsl_marginal_price
        saved_cross = self.vsl_meter_cross_price
        self.metering_marginal_price = None
        self.vsl_marginal_price = None
        self.vsl_meter_cross_price = None
        out: Dict[str, List[float]] = {}
        try:
            for ramp, pairs in requests.items():
                link = self.cfg.network.ramp_to_freeway.get(ramp)
                if link is None:
                    out[ramp] = [0.0 for _ in pairs]
                    continue
                costs: List[float] = []
                for pair in pairs:
                    meter, vsl = float(pair[0]), float(pair[1])
                    probe_prev = ControlAction(
                        ramp_metering=dict(ctrl.ramp_metering),
                        vsl=dict(ctrl.vsl),
                        green_times=dict(ctrl.green_times),
                        offsets=dict(ctrl.offsets),
                        inflow_outflow_allocation={},
                    )
                    probe_prev.ramp_metering[ramp] = meter
                    _, cost, _ = self._solve_freeway_agent_local(
                        link, state, coupling, demand, probe_prev,
                        vsl_override=[vsl] * int(n_seg),
                    )
                    costs.append(float(cost))
                out[ramp] = costs
        finally:
            self.metering_marginal_price = saved_meter
            self.vsl_marginal_price = saved_vsl
            self.vsl_meter_cross_price = saved_cross
        return out

    # ---------- E2: leader가 부르는 국소 VSL 벡터 비용 조회(가격항 제외) ----------

    def local_vsl_costs(
        self,
        requests: Mapping[str, Sequence[Sequence[float]]],
        state: TrafficState,
        control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, List[float]]:
        """link별 VSL 벡터 후보들의 freeway-agent 국소 own-TTS — B3 VSL 가격 g_ext화(E2).

        vsl_override로 벡터를 고정 채점(단일 시퀀스) — 그간 "국소 고정 VSL 벡터 채점
        프리미티브 부재"로 VSL 가격이 raw g_i(d_local 미차감)였던 결함의 해소 재료.
        규약은 local_metering_costs와 동일: 프롤로그 1회, 가격 일시 비활성, 영속 상태 미변경."""
        ctrl = ControlAction.uncontrolled(self.cfg)
        ctrl.green_times = dict(control.green_times)
        ctrl.offsets = dict(control.offsets)
        ctrl.vsl = dict(control.vsl)
        ctrl.ramp_metering = dict(control.ramp_metering)
        ctrl.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, ctrl, demand)
        saved_meter = self.metering_marginal_price
        saved_vsl = self.vsl_marginal_price
        saved_cross = self.vsl_meter_cross_price
        self.metering_marginal_price = None
        self.vsl_marginal_price = None
        self.vsl_meter_cross_price = None
        out: Dict[str, List[float]] = {}
        try:
            for link, vectors in requests.items():
                costs: List[float] = []
                for vec in vectors:
                    _, cost, _ = self._solve_freeway_agent_local(
                        link, state, coupling, demand, ctrl,
                        vsl_override=[float(v) for v in vec],
                    )
                    costs.append(float(cost))
                out[link] = costs
        finally:
            self.metering_marginal_price = saved_meter
            self.vsl_marginal_price = saved_vsl
            self.vsl_meter_cross_price = saved_cross
        return out

    # ---------- P1.5 auto 게이트: phase 포화도 ----------

    def _ramp_phase_saturation(
        self,
        signal: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        control: ControlAction,
        horizon_h: float,
    ) -> float:
        """P1.5 auto 게이트용 phase 포화도 x(신호별, frozen coupling 기준).

        x_p = (q0_p/horizon_h + arr_p) / (Σ_{m∈p, 큐 movement} cap_flow_m · g_p/total),
        반환 max_p x_p. 서비스 항의 g_p는 현재 control(직전 commit) green — 게이트는
        후보 탐색 전에 step당 1회만 평가한다(후보 의존 아님). off_ramp movement는
        큐 서비스 대상이 아니므로 제외(도착도 storage로 가서 arr_phase에서 빠짐과 대칭)."""
        net = self.cfg.network
        model = self._local_models[signal]
        total = float(net.effective_green_total)
        q0 = {m: max(0.0, state.urban_movement_queue.get(m, 0.0)) for m in model.movements}
        x_max = 0.0
        for pid in ("p1", "p2"):
            g = float(control.green_times.get(f"{signal}_{pid}", total / 2.0))
            cap = sum(
                float(model.cap_flow_of.get(m, 0.0))
                for m in model.movements
                if model.phase_of[m] == pid and model.kind_of.get(m) != "off_ramp"
            ) * max(g, 0.0) / max(total, 1.0e-9)
            demand_rate = (
                q0_sum(q0, model, pid) / max(horizon_h, 1.0e-9)
                + float(coupling.get(f"arr_{signal}_{pid}", 0.0))
            )
            if cap > 1.0e-9:
                x_max = max(x_max, demand_rate / cap)
            elif demand_rate > 1.0e-9:
                x_max = float("inf")
        return float(x_max)

    # ---------- PLATOON 도착 재구성 + offset-aware service (Task-B 핵심) ----------

    def _platoon_arrival_profiles(
        self,
        signal: str,
        state: TrafficState,
        snapshot: ControlAction,
        demand: DemandStep,
        arr_movement: Mapping[str, float],
        substeps: int,
        start_idx: int,
    ) -> Dict[str, List[float]]:
        """신호 i의 movement별 시간분해 도착 profile arr[m][sub][veh/h] — PLATOON 재구성.

        `_per_movement_arrivals`(상수)를 TIME으로 재분배한다(총량 보존, 분포만 변경):
          (a) exogenous 항(boundary_in 수요·on_ramp·off_ramp): 상류 신호 platoon이 없으므로
              균일(uniform) — 매 substep 동일.
          (b) upstream 항(β_m × Σ_{producer p→origin(m)} leaving_rate(p)): 상류 신호 p가 자기
              green+offset window에서 방출하는 시간분해 discharge를 재구성한다. producer profile
              d_p(sub) = leaving_rate(p) × gf_p(start+sub)/mean(gf_p)  (예산 보존: mean=leaving_rate).
              gf_p = `_phase_green_fraction(snapshot p control, urban_step_index=start+sub)`(offset-aware).
              origin 링크 L별로 producer를 합쳐 D_L(sub), travel τ_L=`_link_delay_steps(L)` 만큼 지연한
              뒤 β_m로 split. τ 지연으로 horizon 끝에서 빠져나간 mass는 profile을 원래 총량으로
              **재정규화**해 정확히 보존한다(예산 불변, 분포만 platoon).
        반환 arr[m] = 길이 substeps의 리스트(균일 exogenous + platoon upstream).
        offset이 동역학에 들어가는 두 통로 중 도착 쪽(서비스 쪽은 gf_by_substep)이다."""
        wu = self._wu
        model = self._local_models[signal]

        # --- (1) origin 링크별 상류 producer의 시간분해 discharge profile D_L(sub) ---
        # 각 producer p: 상수 leaving_rate(p)에 offset-aware green window shape를 곱하고
        # mean으로 나눠 정규화(총량=leaving_rate(p) 보존).
        link_profile: Dict[str, List[float]] = {}
        for phase_id in ("p1", "p2"):
            key = f"{signal}_{phase_id}"
            for up_signal, up_movement, _agg_beta in wu._upstream_leaving_map.get(key, []):
                origin_link = str(wu._specs[up_movement].get("destination", ""))
                if not origin_link:
                    continue
                rate = wu._signal_leaving_rate(up_signal, up_movement, snapshot, state, demand)
                if rate <= 1.0e-12:
                    continue
                up_spec = wu._specs[up_movement]
                gf = [
                    _phase_green_fraction(
                        snapshot, self.cfg, up_spec, urban_step_index=start_idx + sub,
                    )
                    for sub in range(substeps)
                ]
                gf_mean = sum(gf) / max(len(gf), 1)
                prof = link_profile.setdefault(origin_link, [0.0] * substeps)
                if gf_mean > 1.0e-9:
                    for sub in range(substeps):
                        prof[sub] += rate * gf[sub] / gf_mean
                else:
                    # green window가 horizon에 전혀 안 걸리면 균일 fallback(총량 보존).
                    for sub in range(substeps):
                        prof[sub] += rate

        # --- (2) travel τ_L 지연 적용(빈 링크일수록 통과시간↑) ---
        delayed_link_profile: Dict[str, List[float]] = {}
        for link, prof in link_profile.items():
            tau = _link_delay_steps(state, self.cfg, link)
            delayed = [0.0] * substeps
            for sub in range(substeps):
                src = sub - tau
                if 0 <= src < substeps:
                    delayed[sub] = prof[src]
            delayed_link_profile[link] = delayed

        # --- (3) movement별 profile = 균일 exogenous + β-split platoon(upstream), 예산 재정규화 ---
        arr_prof: Dict[str, List[float]] = {}
        for phase_id in ("p1", "p2"):
            for movement in self._phase_movements[signal][phase_id]:
                spec = self._specs[movement]
                kind = str(spec.get("kind", ""))
                beta = float(spec.get("beta", 0.0))
                origin = str(spec.get("origin", ""))
                total_rate = max(0.0, float(arr_movement.get(movement, 0.0)))
                # upstream 기여(예산 target) = β_m × Σ_p leaving_rate(p) on origin. PRE-DELAY
                # link_profile 평균(=Σ leaving_rate)으로 잡아 τ 지연으로 horizon 밖에 나간 mass도
                # platoon으로 보존한다(flat exogenous로 새지 않게). exogenous = total − upstream.
                pre = link_profile.get(origin)
                up_total_rate = (
                    beta * sum(pre) / max(substeps, 1) if pre is not None else 0.0
                )
                exo_rate = max(0.0, total_rate - up_total_rate)
                prof = [exo_rate] * substeps
                dl = delayed_link_profile.get(origin)
                if dl is not None and beta > 0.0:
                    # 지연 후 profile을 PRE-DELAY 총량(up_total_rate)으로 재정규화 → τ 손실 보존.
                    cur_mean = beta * sum(dl) / max(substeps, 1)
                    scale = (up_total_rate / cur_mean) if cur_mean > 1.0e-12 else 0.0
                    for sub in range(substeps):
                        prof[sub] += beta * dl[sub] * scale
                arr_prof[movement] = prof
        return arr_prof

    def _offset_green_fractions(
        self,
        signal: str,
        green_p1: float,
        offset: float,
        substeps: int,
        start_idx: int,
    ) -> Dict[str, List[float]]:
        """movement별 offset-aware green fraction gf[m][sub] — service가 substep window 겹침."""
        net = self.cfg.network
        model = self._local_models[signal]
        total = net.effective_green_total
        green_p2 = total - green_p1
        probe = ControlAction.uncontrolled(self.cfg)
        probe.green_times = {f"{signal}_p1": float(green_p1), f"{signal}_p2": float(green_p2)}
        probe.offsets = {signal: float(offset)}
        probe.inflow_outflow_allocation = {}
        gf: Dict[str, List[float]] = {}
        # phase별 fraction은 movement 무관(같은 phase면 동일) → phase 단위로 1회 계산해 공유.
        phase_gf: Dict[str, List[float]] = {}
        for m in model.movements:
            pid = model.phase_of[m]
            if pid not in phase_gf:
                spec = model.specs[m]
                phase_gf[pid] = [
                    _phase_green_fraction(probe, self.cfg, spec, urban_step_index=start_idx + sub)
                    for sub in range(substeps)
                ]
            gf[m] = phase_gf[pid]
        return gf

    # ---------- JOINT(2026-07-09): (green_p1, offset) 2D 공동탐색 ----------

    def _solve_urban_agent_joint(
        self,
        signal: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        arr_movement: Mapping[str, float],
        s_eff_frozen: Mapping[str, float],
        snapshot: ControlAction,
        leader: Optional[object],
        lambda_p: float,
        forecast_arrivals: Optional[Mapping[str, float]],
        horizon_h: float,
        demand: DemandStep,
        previous: Optional[ControlAction] = None,
    ):
        """non-ramp 신호의 green_p1과 offset을 2D 격자로 함께 채점해 joint 최적점 반환.

        반환 (best_p1, best_off, best_obj, evals, best_nin) 또는 ramp 신호면 None.

        coordinate descent(green은 offset 동결·offset은 green 동결)와 달리 두 lever를 동시에
        움직여 bilinear cross-term 가격 h·(p1−p1ref)·Δoff가 실제로 작동한다. own_TTS(p1,offset)는
        offset 탐색과 같은 phased rollout으로 직접 계산하며, 셋업(arr_by_substep 등)은 ego
        green·offset에 불변이라 1회 계산 후 이중루프한다."""
        net = self.cfg.network
        sim = self.cfg.simulation
        model = self._local_models[signal]
        if model.has_ramps:
            return None  # ramp 신호(D/F)는 storage 동역학 복잡 → 기존 coordinate descent 유지.
        cycle = max(net.cycle_length, 1.0e-9)
        substeps = max(1, self.cfg.mpc.horizon_steps) * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        total = float(net.effective_green_total)
        start_idx = _urban_step_index(state, self.cfg)
        smooth_w = self.cfg.urban_follower.green_smoothness_weight
        fa = forecast_arrivals if forecast_arrivals is not None else {}
        dual_mode = leader is not None and self.use_dual_np

        # phase 단위 고정 도착 → movement 재정규화(_solve_offset_local과 동일).
        arr_phase = {pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")}
        arr_mv: Dict[str, float] = {}
        for pid in ("p1", "p2"):
            phase_movements = [
                m for m in model.movements
                if model.phase_of[m] == pid and model.kind_of[m] != "off_ramp"
            ]
            raw_sum = sum(max(0.0, float(arr_movement.get(m, 0.0))) for m in phase_movements)
            target = max(0.0, arr_phase[pid])
            if raw_sum > 1.0e-12:
                scale = target / raw_sum
                for m in phase_movements:
                    arr_mv[m] = max(0.0, float(arr_movement.get(m, 0.0))) * scale
            else:
                for m in phase_movements:
                    arr_mv[m] = 0.0
        q0 = {m: max(0.0, float(state.urban_movement_queue.get(m, 0.0))) for m in model.movements}
        s_eff0 = {
            model.receiving_of[m]: float(s_eff_frozen.get(model.receiving_of[m], 0.0))
            for m in model.movements
            if model.receiving_of[m]
        }
        # platoon profile(상류 snapshot 의존, ego green·offset 불변) — 1회 계산.
        arr_by_substep = self._platoon_arrival_profiles(
            signal, state, snapshot, demand, arr_mv, substeps, start_idx,
        )

        prev_p1 = float(snapshot.green_times.get(f"{signal}_p1", total / 2.0))

        def _circ_delta(off: float, ref: float) -> float:
            return ((off - ref + cycle / 2.0) % cycle) - cycle / 2.0

        # ---- 가격/참조점(설정 시에만) ----
        g_green = None
        green_ref = prev_p1
        green_trust = None
        if self.signal_marginal_price is not None and signal in self.signal_marginal_price:
            g_green = float(self.signal_marginal_price.get(signal, 0.0))
            green_ref = float(self.signal_marginal_price_ref.get(signal, prev_p1))
            green_trust = self.signal_marginal_price_trust_sec
        g_off = None
        off_ref = 0.0
        off_trust = None
        if self.offset_marginal_price is not None and signal in self.offset_marginal_price:
            g_off = float(self.offset_marginal_price.get(signal, 0.0))
            off_ref = float(self.offset_marginal_price_ref.get(signal, 0.0)) % cycle
            off_trust = self.offset_marginal_price_trust_sec
        h_cross = None
        if (
            self.green_offset_cross_price is not None
            and signal in self.green_offset_cross_price
        ):
            h_cross = float(self.green_offset_cross_price.get(signal, 0.0))
            gc_ref, oc_ref = self.green_offset_cross_ref.get(signal, (green_ref, off_ref))
            gc_ref = float(gc_ref)
            oc_ref = float(oc_ref) % cycle

        # ---- green 후보(trust 필터) × offset 후보(trust 필터) 2D ----
        green_cands = self._urban_green_candidates(signal, state, coupling, snapshot)
        if green_trust is not None:
            trusted = [p for p in green_cands if abs(p - green_ref) <= float(green_trust) + 1.0e-9]
            if trusted:
                green_cands = trusted
        # BASELINE-BOX green(joint 경로): local 경로와 동일 — PFO 무제한 green에 ±6s.
        if bool(getattr(self.cfg.mpc, "baseline_move_box", False)) and previous is not None:
            _bgj = float(previous.green_times.get(f"{signal}_p1", prev_p1))
            _bkj = [p for p in green_cands if abs(p - _bgj) <= 6.0 + 1.0e-9]
            green_cands = _bkj or [min(green_cands, key=lambda p: abs(p - _bgj))]
        offset_cands = []
        for frac in self.offset_fractions:
            offset = (frac * cycle) % cycle
            if off_trust is not None and abs(_circ_delta(offset, off_ref)) > float(off_trust) + 1.0e-9:
                continue
            offset_cands.append(offset)
        if not offset_cands:
            offset_cands = [float(snapshot.offsets.get(signal, 0.0)) % cycle]

        best_p1, best_off, best_obj, best_nin = prev_p1, 0.0, float("inf"), 0.0
        evals = 0
        for p1 in green_cands:
            p2 = total - p1
            if p2 < net.green_min - 1.0e-9 or p2 > net.green_max + 1.0e-9:
                continue
            nin = self._agent_net_inflow_veh(signal, p1, state, fa, horizon_h)
            # PRICE-TR: 가격 활성이면 smoothness 마찰 0(trust가 보폭 제약).
            green_lin = (
                0.0 if (self.price_smoothness_disabled and g_green is not None)
                else smooth_w * abs(p1 - prev_p1)
            )
            if g_green is not None:
                green_lin += self.signal_marginal_price_weight * g_green * (p1 - green_ref)
            if self.np_price_enabled and dual_mode:
                green_lin += lambda_p * nin
            for offset in offset_cands:
                gf_by_substep = self._offset_green_fractions(
                    signal, p1, offset, substeps, start_idx,
                )
                cost = rollout_local_tts_phased(
                    model, q0, arr_by_substep, gf_by_substep, s_eff0, substeps, dt_h,
                )
                cost += green_lin
                if g_off is not None:
                    cost += self.offset_marginal_price_weight * g_off * _circ_delta(offset, off_ref)
                if h_cross is not None:
                    cost += (
                        self.green_offset_cross_weight * h_cross
                        * (p1 - gc_ref) * _circ_delta(offset, oc_ref)
                    )
                evals += 1
                if cost < best_obj - 1.0e-12:
                    best_obj, best_p1, best_off, best_nin = cost, float(p1), float(offset), float(nin)
        return best_p1, best_off, best_obj, evals, best_nin

    # ---------- per-signal 국소 OFFSET 탐색 (PLATOON-AWARE, "proposed" authority 전용) ----------

    def _solve_offset_local(
        self,
        signal: str,
        green_p1: float,
        state: TrafficState,
        coupling: Mapping[str, float],
        arr_movement: Mapping[str, float],
        s_eff_frozen: Mapping[str, float],
        snapshot: ControlAction,
        demand: DemandStep,
    ) -> tuple[float, int]:
        """신호 i의 offset 후보집합을 훑어 자기 국소 objective 최소 offset을 반환 — (offset*, evals).

        Task-B: offset의 corridor 이득은 PLATOON 정렬에서 나온다. 따라서 offset 채점은
        `rollout_local_tts_phased`(phase-resolved 서비스 + platoon 도착)로 한다. 상류 신호 green+
        offset(snapshot 동결)에서 재구성한 platoon profile은 offset에 불변이므로 **1회** 계산하고,
        후보 offset마다 offset-aware gf만 갱신해 채점한다. offset이 platoon 도착 window와 자기 green
        window를 정렬할수록 stop-delay(자기 TTS)가 낮아져 비용 최소 offset이 선택된다.

        ramp-aware(D/F) 신호는 기본 offset 0(storage 동역학 복잡). G1(ramp_offset_enabled)이면
        ramp-aware phased rollout로 offset 탐색(아래 _solve_offset_local_ramp)."""
        net = self.cfg.network
        sim = self.cfg.simulation
        model = self._local_models[signal]
        if model.has_ramps:
            if not self.ramp_offset_enabled:
                return 0.0, 0
            return self._solve_offset_local_ramp(
                signal, green_p1, state, coupling, arr_movement,
                s_eff_frozen, snapshot, demand,
            )
        cycle = max(net.cycle_length, 1.0e-9)
        substeps = max(1, self.cfg.mpc.horizon_steps) * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        start_idx = _urban_step_index(state, self.cfg)

        # phase 단위 고정 도착(frozen 결합) → movement별 재정규화(_solve_urban_agent_local과 동일).
        arr_phase = {pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")}
        arr_mv: Dict[str, float] = {}
        for pid in ("p1", "p2"):
            phase_movements = [
                m for m in model.movements
                if model.phase_of[m] == pid and model.kind_of[m] != "off_ramp"
            ]
            raw_sum = sum(max(0.0, float(arr_movement.get(m, 0.0))) for m in phase_movements)
            target = max(0.0, arr_phase[pid])
            if raw_sum > 1.0e-12:
                scale = target / raw_sum
                for m in phase_movements:
                    arr_mv[m] = max(0.0, float(arr_movement.get(m, 0.0))) * scale
            else:
                for m in phase_movements:
                    arr_mv[m] = 0.0

        q0 = {m: max(0.0, float(state.urban_movement_queue.get(m, 0.0))) for m in model.movements}
        s_eff0 = {
            model.receiving_of[m]: float(s_eff_frozen.get(model.receiving_of[m], 0.0))
            for m in model.movements
            if model.receiving_of[m]
        }
        # platoon profile(상류 snapshot 의존, offset 불변) — 1회 계산.
        arr_by_substep = self._platoon_arrival_profiles(
            signal, state, snapshot, demand, arr_mv, substeps, start_idx,
        )

        # F3 offset 가격 준비(설정 시에만). Δ는 원형 최단 변위((-cycle/2, cycle/2]).
        price_active = (
            self.offset_marginal_price is not None
            and signal in self.offset_marginal_price
        )
        g_off = (
            float(self.offset_marginal_price.get(signal, 0.0)) if price_active else 0.0
        )
        ref_off = (
            float(self.offset_marginal_price_ref.get(signal, 0.0)) % cycle
            if price_active else 0.0
        )
        trust = self.offset_marginal_price_trust_sec if price_active else None

        def _circ_delta(off: float) -> float:
            return ((off - ref_off + cycle / 2.0) % cycle) - cycle / 2.0

        best_off, best_obj = 0.0, float("inf")
        evals = 0
        for frac in self.offset_fractions:
            offset = (frac * cycle) % cycle
            if trust is not None and abs(_circ_delta(offset)) > float(trust) + 1.0e-9:
                continue  # 가격이 측정된 이웃 밖(월권 방지) — trust는 그리드 1칸으로 설정됨.
            gf_by_substep = self._offset_green_fractions(
                signal, green_p1, offset, substeps, start_idx,
            )
            obj = rollout_local_tts_phased(
                model, q0, arr_by_substep, gf_by_substep, s_eff0, substeps, dt_h,
            )
            if price_active:
                obj += self.offset_marginal_price_weight * g_off * _circ_delta(offset)
            evals += 1
            if obj < best_obj - 1.0e-9:
                best_obj, best_off = obj, float(offset)
        return best_off, evals

    def _solve_offset_local_ramp(
        self,
        signal: str,
        green_p1: float,
        state: TrafficState,
        coupling: Mapping[str, float],
        arr_movement: Mapping[str, float],
        s_eff_frozen: Mapping[str, float],
        snapshot: ControlAction,
        demand: DemandStep,
    ) -> tuple[float, int]:
        """ramp 신호(D/F) offset 국소 탐색 — ramp-aware phased rollout 채점(G1).

        `_solve_urban_agent_local`의 ramp 셋업(offramp_inflow·occ0·ramp_queue0·reservoir_drain·
        freeway_congestion·arr 재정규화)을 재사용하고, `_solve_offset_local`의 offset 후보 루프
        구조를 미러링한다. platoon arr_by_substep은 offset 불변(상류 snapshot 의존)이라 1회 계산,
        후보 offset마다 offset-aware gf_by_substep만 갱신해 rollout_local_tts_ramp_aware로 채점한다.
        offset 가격(F3) 설정 시 가격항·trust도 non-ramp와 동일하게 적용."""
        net = self.cfg.network
        sim = self.cfg.simulation
        model = self._local_models[signal]
        cycle = max(net.cycle_length, 1.0e-9)
        substeps = max(1, self.cfg.mpc.horizon_steps) * max(1, sim.K_cu)
        dt_h = sim.T_u_h
        start_idx = _urban_step_index(state, self.cfg)
        total = net.effective_green_total

        # frozen ramp 입력(snapshot을 control로).
        reservoir_drain = self._frozen_reservoir_drain(state, snapshot, demand)
        freeway_congestion = self._frozen_freeway_congestion(state)

        arr_phase = {pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")}
        offramp_inflow: Dict[str, float] = {}
        offramp_contrib_phase = {"p1": 0.0, "p2": 0.0}
        for off_ramp, movements in model.offramp_movements.items():
            inflow = self._frozen_offramp_inflow(off_ramp, state)
            offramp_inflow[off_ramp] = inflow
            for m in movements:
                offramp_contrib_phase[model.phase_of[m]] += model.beta_of[m] * inflow
        arr_mv: Dict[str, float] = {}
        for pid in ("p1", "p2"):
            phase_movements = [
                m for m in model.movements
                if model.phase_of[m] == pid and model.kind_of[m] != "off_ramp"
            ]
            raw_sum = sum(max(0.0, float(arr_movement.get(m, 0.0))) for m in phase_movements)
            target = max(0.0, arr_phase[pid] - offramp_contrib_phase[pid])
            if raw_sum > 1.0e-12:
                scale = target / raw_sum
                for m in phase_movements:
                    arr_mv[m] = max(0.0, float(arr_movement.get(m, 0.0))) * scale
            else:
                for m in phase_movements:
                    arr_mv[m] = 0.0
        offramp_occ0 = {
            off_ramp: self._offramp_occupancy(off_ramp, state)
            for off_ramp in model.offramp_movements
        }
        ramp_queue0 = {
            ramp: max(0.0, float(state.ramp_queue.get(ramp, 0.0)))
            for ramp in model.onramp_movements
        }
        s_eff0 = {
            model.receiving_of[m]: float(s_eff_frozen.get(model.receiving_of[m], 0.0))
            for m in model.movements
            if model.receiving_of[m]
        }
        q0 = {m: max(0.0, float(state.urban_movement_queue.get(m, 0.0))) for m in model.movements}
        # platoon arr(offset 불변) — off_ramp movement 제외(storage 유입 별도).
        arr_by_substep = self._platoon_arrival_profiles(
            signal, state, snapshot, demand, arr_mv, substeps, start_idx,
        )
        arr_by_substep = {
            m: prof for m, prof in arr_by_substep.items()
            if model.kind_of.get(m) != "off_ramp"
        }
        p2 = total - green_p1

        price_active = (
            self.offset_marginal_price is not None
            and signal in self.offset_marginal_price
        )
        g_off = float(self.offset_marginal_price.get(signal, 0.0)) if price_active else 0.0
        ref_off = (
            float(self.offset_marginal_price_ref.get(signal, 0.0)) % cycle
            if price_active else 0.0
        )
        trust = self.offset_marginal_price_trust_sec if price_active else None

        def _circ_delta(off: float) -> float:
            return ((off - ref_off + cycle / 2.0) % cycle) - cycle / 2.0

        best_off, best_obj = 0.0, float("inf")
        evals = 0
        for frac in self.offset_fractions:
            offset = (frac * cycle) % cycle
            if trust is not None and abs(_circ_delta(offset)) > float(trust) + 1.0e-9:
                continue
            gf_by_substep = self._offset_green_fractions(
                signal, green_p1, offset, substeps, start_idx,
            )
            obj = rollout_local_tts_ramp_aware(
                model, q0, arr_mv, s_eff0,
                offramp_inflow, offramp_occ0, ramp_queue0, reservoir_drain,
                freeway_congestion, self.ramp_metering_weight,
                green_p1, p2, substeps, dt_h,
                arr_by_substep=arr_by_substep,
                gf_by_substep=gf_by_substep,
            )
            if price_active:
                obj += self.offset_marginal_price_weight * g_off * _circ_delta(offset)
            evals += 1
            if obj < best_obj - 1.0e-9:
                best_obj, best_off = obj, float(offset)
        return best_off, evals

    # ---------- freeway agent: 진짜 per-link 국소 METANET rollout (핵심 신규) ----------

    def _local_ramp_release(
        self,
        link: str,
        rhos: List[float],
        ramp_queue: Mapping[str, float],
        candidate_control: ControlAction,
        demand: DemandStep,
    ) -> Dict[str, float]:
        """이 link 소유 ramp에 대해서만 `compute_ramp_release_flows`를 복제(per-link 국소).

        plant의 `compute_ramp_release_flows`는 모든 ramp을 돌지만 각 ramp release는 자기 link의
        merge 밀도(rho_merge)·자기 ramp_queue·candidate metering만 읽으므로 per-link로 분리 가능.
        `_solve_freeway_agent` probe와 동일하게 include_current_arrivals=False(arrival=0)."""
        net = self.cfg.network
        model = self._local_freeway_models[link]
        dt_h = self.cfg.simulation.T_f_h
        cap_factor = getattr(demand, "incident_capacity_factor", 1.0)
        q_cap = net.freeway_capacity_veh_h * cap_factor
        release: Dict[str, float] = {}
        for ramp in model.owned_ramps:
            merge_idx = model.ramp_merge_idx[ramp]
            rho_merge = rhos[merge_idx] if merge_idx < len(rhos) else net.rho_crit
            receiving_factor = _clip_local(
                (net.rho_max - rho_merge) / max(net.rho_max - net.rho_crit, 1.0e-9), 0.0, 1.0,
            )
            cap = net.ramp_capacity_veh_h[ramp]
            requested = _clip_local(candidate_control.ramp_metering.get(ramp, cap), 0.0, cap)
            available = max(0.0, ramp_queue.get(ramp, 0.0) / max(dt_h, 1.0e-9))
            no_meter = min(available, cap, q_cap * receiving_factor)
            release[ramp] = min(no_meter, requested)
        return release

    def _local_offramp_capacity(self, link: str, storage_avail: Mapping[str, float]) -> Dict[str, float]:
        """이 link 소유 off-ramp의 cap[veh/h] — `off_ramp_capacity_by_freeway_link`의 per-link 복제.

        cap = storage 가용공간 / T_f_h. probe storage 가용공간(국소 추적값)을 받아 계산한다."""
        net = self.cfg.network
        model = self._local_freeway_models[link]
        dt_h = self.cfg.simulation.T_f_h
        cap: Dict[str, float] = {}
        link_total = 0.0
        for off_ramp in model.owned_offramps:
            available = max(0.0, float(storage_avail.get(off_ramp, 0.0)))
            flow_cap = available / max(dt_h, 1.0e-9)
            cap[off_ramp] = flow_cap
            link_total += flow_cap
        cap[link] = link_total
        return cap

    def _local_offramp_drain(
        self,
        off_ramp: str,
        occupied: float,
        recv_occ: Mapping[str, float],
        control: ControlAction,
        dt_h: float,
    ) -> tuple[float, Dict[str, float]]:
        """off-ramp storage 1개의 drain[veh] — `_update_probe_offramp_storage`의 per-off_ramp 복제.

        drain = 하류 신호 off_ramp movement green 처리율, 단 receiving 도시 링크 가용공간 제약.
        반환 (drain_total_flow[veh/h], receiving 링크별 intake[veh/h])."""
        net = self.cfg.network
        wu = self._wu
        drain = 0.0
        recv_intake: Dict[str, float] = {}
        for signal, movement in wu._offramp_drain_flow.get(off_ramp, []):
            rate = wu._signal_leaving_rate(signal, movement, control)
            recv_link = str(wu._specs[movement].get("receiving_link", ""))
            recv_cap = float(net.urban_link_storage_veh.get(recv_link, 0.0))
            if recv_cap > 0.0:
                recv_avail = max(0.0, recv_cap - float(recv_occ.get(recv_link, 0.0)))
                rate = min(rate, recv_avail / max(dt_h, 1.0e-9))
            drain += rate
            if recv_link:
                recv_intake[recv_link] = recv_intake.get(recv_link, 0.0) + rate
        return drain, recv_intake

    def _freeway_vsl_sequence_candidates(
        self,
        link: str,
        n_seg: int,
        previous: ControlAction,
        base_candidates: list[list[float]],
        horizon: int,
    ) -> list[list[list[float]]]:
        """Build bounded VSL sequences for the Wu-faithful freeway probe.

        The previous probe fixed one VSL vector over the whole horizon. With
        `max_vsl_step=20`, it could evaluate the first 100->80 move but could
        not see a later 80->60->50 preventive sequence. This helper keeps the
        plant commit to the first vector while letting the local rollout score
        bounded future VSL trajectories.
        """
        ff = self.cfg.freeway_follower
        horizon = max(1, int(horizon))
        sequences: list[list[list[float]]] = []
        seen: set[tuple[tuple[float, ...], ...]] = set()

        def add_sequence(sequence: list[list[float]]) -> None:
            normalized = [
                [float(v) for v in vec]
                for vec in (sequence + [sequence[-1]] * max(0, horizon - len(sequence)))
            ][:horizon]
            key = tuple(tuple(round(v, 6) for v in vec) for vec in normalized)
            if key not in seen:
                seen.add(key)
                sequences.append(normalized)

        if not ff.vsl_sequence_search:
            for vec in base_candidates:
                add_sequence([[float(v) for v in vec]])
            return sequences

        vsl_set = sorted(float(v) for v in ff.vsl_set)
        if not vsl_set:
            return sequences
        vsl_max = max(vsl_set)
        max_step = max(0.0, float(ff.max_vsl_step))
        sequence_steps = max(1, min(horizon, int(ff.vsl_sequence_horizon_steps)))
        net = self.cfg.network
        bottleneck_idx = {
            int(net.off_ramp_segment_index.get(off_ramp, n_seg - 1))
            for off_ramp in net.off_ramps
            if net.off_ramp_from_freeway.get(off_ramp) == link
        } or {n_seg - 1}
        upstream_control_idx = {i for i in range(max(0, min(bottleneck_idx)))}

        def sanitize_base_vector(vec: list[float]) -> list[float]:
            sanitized: list[float] = []
            for index in range(n_seg):
                value = float(vec[index]) if index < len(vec) else segment_vsl(previous, link, index, self.cfg)
                if index not in upstream_control_idx:
                    prev = segment_vsl(previous, link, index, self.cfg)
                    value = repair_vsl_value(vsl_max, prev, self.cfg).value
                sanitized.append(float(value))
            return sanitized

        for vec in base_candidates:
            add_sequence([sanitize_base_vector([float(v) for v in vec])])
        limit = max(len(sequences), int(ff.vsl_sequence_candidate_limit))

        def segment_sequences(index: int) -> list[list[float]]:
            prev = segment_vsl(previous, link, index, self.cfg)
            if index not in upstream_control_idx:
                repaired = repair_vsl_value(vsl_max, prev, self.cfg).value
                return [[float(repaired)] * sequence_steps]

            first_values = [
                value
                for value in vsl_set
                if value <= prev + 1.0e-9 and prev - value <= max_step + 1.0e-9
            ]
            if not first_values:
                first_values = [repair_vsl_value(prev, prev, self.cfg).value]

            out: list[list[float]] = []

            def extend(prefix: list[float]) -> None:
                if len(prefix) >= sequence_steps:
                    out.append([float(v) for v in prefix])
                    return
                current = prefix[-1]
                next_values = [
                    value
                    for value in vsl_set
                    if value <= current + 1.0e-9
                    and current - value <= max_step + 1.0e-9
                ]
                for value in sorted(set(next_values), reverse=True):
                    extend(prefix + [float(value)])

            for value in sorted(set(first_values), reverse=True):
                extend([float(value)])
            return out

        per_segment = [segment_sequences(i) for i in range(n_seg)]
        segment_combinations: list[list[list[float]]] = [[]]
        for options in per_segment:
            segment_combinations = [
                partial + [option]
                for partial in segment_combinations
                for option in options
            ]

        generated: list[list[list[float]]] = []
        for combo in segment_combinations:
            generated.append([
                [float(combo[seg][step]) for seg in range(n_seg)]
                for step in range(sequence_steps)
            ])

        def sequence_score(sequence: list[list[float]]) -> tuple[float, float, float]:
            flat = [v for vec in sequence for v in vec]
            first = sum(sequence[0]) / max(len(sequence[0]), 1)
            terminal = sum(sequence[-1]) / max(len(sequence[-1]), 1)
            mean = sum(flat) / max(len(flat), 1)
            return (mean, terminal, first)

        for sequence in sorted(generated, key=sequence_score):
            if len(sequences) >= limit:
                break
            add_sequence(sequence)
        return sequences

    def _solve_freeway_agent_local(
        self,
        link: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        demand: DemandStep,
        previous: ControlAction,
        vsl_override: Optional[Sequence[float]] = None,
    ) -> tuple[Dict[str, float], float, int]:
        """`_solve_freeway_agent`의 per-link 국소판 — 후보 채점이 **이 link 본선만** 전진한다.

        SPEC: parent `_solve_freeway_agent`는 후보마다 `freeway_substep`(전체 freeway link 루프)을
        돌려 비국소다. 여기서는 `freeway_substep_local`(이 link만)로 같은 own-TTS를 채점한다.
        본선 이웃 경계는 plant 규약(upstream=v_free, downstream=self)으로 이미 동결돼 있어 별도
        조작 없이 plant와 동일 거동을 낸다. on-ramp reservoir·off-ramp storage는 이 link 권역만
        국소 추적한다. 반환 (segment 키 vsl_dict, own-TTS, evaluations) — parent와 동일 시그니처.
        반환 dict의 off-ramp 유출은 `_last_offramp_flow`에 캐시(coupling freeway→urban 재사용)."""
        net = self.cfg.network
        sim = self.cfg.simulation
        ff = self.cfg.freeway_follower
        model = self._local_freeway_models[link]
        horizon = max(1, ff.freeway_prediction_horizon_steps or self.cfg.mpc.horizon_steps)
        dt_h = sim.T_f_h
        vsl_max = max(ff.vsl_set)
        smooth_w = ff.vsl_smoothness_weight
        n_seg = model.n_seg
        prev_vec = [segment_vsl(previous, link, i, self.cfg) for i in range(n_seg)]

        candidates = (
            self._wu._relaxed_freeway_segment_candidates(link, n_seg, state, coupling, previous, demand)
            if self.cfg.mpc.relaxed_quantized_controls
            else self._wu._freeway_segment_candidates(link, n_seg, previous)
        )
        vsl_sequences = self._freeway_vsl_sequence_candidates(
            link, n_seg, previous, candidates, horizon,
        )
        # JOINT h_local probe: vsl를 단일 값으로 고정 채점(고정 (meter,vsl) own-TTS).
        if vsl_override is not None:
            fixed_vec = [float(v) for v in vsl_override][:n_seg]
            if len(fixed_vec) < n_seg:
                fixed_vec += [float(fixed_vec[-1] if fixed_vec else vsl_max)] * (n_seg - len(fixed_vec))
            vsl_sequences = [[list(fixed_vec) for _ in range(horizon)]]
        # PRICE-TR: VSL 가격 활성 시 trust region(±vsl_marginal_price_trust_kmh) 밖 후보 제외
        # (선형화 유효 반경 — 가격이 측정된 이웃만). 전무 시 전체 fallback(이동성 보장).
        elif self.vsl_marginal_price and self.vsl_marginal_price_trust_kmh is not None:
            trust_v = float(self.vsl_marginal_price_trust_kmh)
            kept = []
            for seq in vsl_sequences:
                fv = seq[0] if seq else []
                ok = True
                for i, v in enumerate(fv):
                    ref = self.vsl_marginal_price_ref.get(f"{link}__seg{i}")
                    if ref is not None and abs(float(v) - float(ref)) > trust_v + 1.0e-9:
                        ok = False
                        break
                if ok:
                    kept.append(seq)
            if kept:
                vsl_sequences = kept

        # 후보 무관 초기 스냅샷(이 link 권역만).
        rhos0 = list(state.freeway_density.get(link, []))
        speeds0 = list(state.freeway_speed.get(link, []))
        lanes0 = list(state.freeway_effective_lanes.get(link, [])) or [
            float(net.freeway_lanes) for _ in range(n_seg)
        ]
        if len(lanes0) != n_seg:
            lanes0 = [float(net.freeway_lanes) for _ in range(n_seg)]
        origin_q0 = max(0.0, float(state.mainline_origin_queue.get(link, 0.0)))
        # Phase B(완충 동결 결합): 완충 plant면 상·하류 완충 경계를 결정시점 값으로 동결해 전달.
        _buf_bc = None
        if int(getattr(net, "freeway_buffer_segments", 0)) > 0:
            _bu_r_bc = state.freeway_buffer_up_density.get(link) or []
            _bu_v_bc = state.freeway_buffer_up_speed.get(link) or []
            _bd_r_bc = state.freeway_buffer_down_density.get(link) or []
            if _bu_r_bc and _bd_r_bc:
                from src.models.metanet import segment_flow_veh_h as _sfvh_bc
                _bu_send_bc = _sfvh_bc(_bu_r_bc[-1], _bu_v_bc[-1], float(net.freeway_lanes))
                _phi_bc = float(getattr(net, "capacity_drop_discharge_phi", 1.0) or 1.0)
                if _phi_bc < 1.0 and _bu_r_bc[-1] > float(net.rho_crit):
                    # plant capacity drop과 동일 cap(동결 BC 정합).
                    _bu_send_bc = min(_bu_send_bc, _phi_bc * float(net.freeway_capacity_veh_h))
                _buf_bc = (
                    _bu_send_bc,
                    float(_bu_v_bc[-1]),
                    float(_bd_r_bc[0]),
                )
        ramp_q0 = {r: max(0.0, float(state.ramp_queue.get(r, 0.0))) for r in model.owned_ramps}
        # off-ramp storage 초기 점유[veh] + receiving 도시 링크 초기 점유[veh].
        occ0: Dict[str, float] = {}
        for off_ramp in model.owned_offramps:
            cap = model.offramp_storage_cap.get(off_ramp, 0.0)
            storage = net.off_ramp_storage_link.get(off_ramp, "")
            avail = float(state.urban_link_storage.get(storage, cap))
            occ0[off_ramp] = max(0.0, cap - avail)
        # receiving 도시 링크 초기 점유(국소 추적; drain이 채우는 링크).
        recv_links: set[str] = set()
        for off_ramp in model.owned_offramps:
            for _signal, movement in self._wu._offramp_drain_flow.get(off_ramp, []):
                rl = str(self._wu._specs[movement].get("receiving_link", ""))
                if rl:
                    recv_links.add(rl)
        recv_occ0: Dict[str, float] = {}
        for rl in recv_links:
            cap = float(net.urban_link_storage_veh.get(rl, 0.0))
            avail = float(state.urban_link_storage.get(rl, cap))
            recv_occ0[rl] = max(0.0, cap - avail)
        urban_exit = float(net.boundary_out_capacity_veh_h)

        best_vec, best_obj = list(prev_vec), float("inf")
        best_offramp_flow: Dict[str, float] = {o: 0.0 for o in model.owned_offramps}
        evals = 0
        for sequence in vsl_sequences:
            candidate_control = ControlAction(
                ramp_metering=dict(previous.ramp_metering),
                vsl=dict(previous.vsl),
                green_times=dict(previous.green_times),
                offsets=dict(previous.offsets),
                inflow_outflow_allocation={},
            )
            first_vec = sequence[0] if sequence else list(prev_vec)
            for i, v in enumerate(first_vec):
                candidate_control.vsl[f"{link}__seg{i}"] = float(v)
            # 국소 상태 복사(후보별 독립).
            rhos = list(rhos0)
            speeds = list(speeds0)
            prev_lanes = list(lanes0)
            origin_q = origin_q0
            ramp_q = dict(ramp_q0)
            occ = dict(occ0)
            recv_occ = dict(recv_occ0)
            # 가상 상류 blocked 큐[veh]: reservoir 만석으로 수용 못 한 coupling 유입의 이월분
            # (P1 — urban 상류 externality를 own-TTS에 보이게 한다). 후보별 독립.
            blocked_q = {r: 0.0 for r in model.owned_ramps}
            cost = 0.0
            first_offramp_flow = dict(best_offramp_flow)
            first_substep = True
            for horizon_idx in range(horizon):
                current_vec = sequence[min(horizon_idx, len(sequence) - 1)]
                # Sequence MPC: score the bounded VSL trajectory inside the
                # horizon, but commit only first_vec to the plant controller.
                for i, v in enumerate(current_vec):
                    candidate_control.vsl[f"{link}__seg{i}"] = float(v)
                for _ in range(sim.K_cf):
                    # Spec 3.4.3: ramp metering release는 T_f 시작 시점의 reservoir만 본다.
                    # 같은 T_f 안에서 urban green으로 새로 들어온 차량은 이번 release가 아니라
                    # 다음 release 결정부터 사용할 수 있으므로, release를 먼저 계산하고 나중에 적재한다.
                    ramp_release = self._local_ramp_release(link, rhos, ramp_q, candidate_control, demand)
                    for ramp, rel in ramp_release.items():
                        ramp_q[ramp] = max(0.0, ramp_q.get(ramp, 0.0) - max(0.0, rel) * dt_h)
                    # urban->freeway coupling[veh/h]은 release 이후 ramp reservoir에 적재된다.
                    for ramp in model.owned_ramps:
                        approach = max(0.0, float(coupling.get(f"u_on_{ramp}", 0.0)))
                        if self.count_blocked_ramp_inflow:
                            # 가상 blocked 큐: reservoir 가용공간(space)에 blocked 이월분을
                            # 먼저(FIFO) 넣고, 남은 공간에 신규 유입을 넣는다. 못 들어간
                            # 차량은 blocked_q에 남아 own-TTS에서 세어진다(무한 큐 가시화).
                            q = max(0.0, ramp_q.get(ramp, 0.0))
                            space = max(0.0, net.ramp_queue_max_veh - q)
                            arrival = approach * dt_h
                            adm1 = min(blocked_q[ramp], space)
                            adm2 = min(arrival, space - adm1)
                            ramp_q[ramp] = min(net.ramp_queue_max_veh, q + adm1 + adm2)
                            blocked_q[ramp] = blocked_q[ramp] - adm1 + (arrival - adm2)
                        else:
                            ramp_q[ramp] = min(
                                net.ramp_queue_max_veh,
                                max(0.0, ramp_q.get(ramp, 0.0)) + approach * dt_h,
                            )
                    # off-ramp cap(국소 storage 가용공간 기반).
                    storage_avail = {
                        o: max(0.0, model.offramp_storage_cap.get(o, 0.0) - occ.get(o, 0.0))
                        for o in model.owned_offramps
                    }
                    offramp_capacity = self._local_offramp_capacity(link, storage_avail)
                    # per-link METANET 한 substep 전진(이 link 본선만).
                    rhos, speeds, prev_lanes, origin_q, offramp_flow, veh_count = freeway_substep_local(
                        model, rhos, speeds, prev_lanes, occ, origin_q,
                        ramp_release, offramp_capacity, candidate_control, demand,
                        buffer_bc=_buf_bc,
                    )
                    # off-ramp storage 점유 갱신(_update_probe_offramp_storage 국소 복제).
                    for off_ramp in model.owned_offramps:
                        cap = model.offramp_storage_cap.get(off_ramp, 0.0)
                        if cap <= 0.0:
                            continue
                        inflow = max(0.0, float(offramp_flow.get(off_ramp, 0.0)))
                        drain, recv_intake = self._local_offramp_drain(
                            off_ramp, occ.get(off_ramp, 0.0), recv_occ, candidate_control, dt_h,
                        )
                        occupied = min(cap, max(0.0, occ.get(off_ramp, 0.0) + (inflow - drain) * dt_h))
                        occ[off_ramp] = occupied
                        # 드레인 차량이 receiving 도시 링크를 점유(보존), 도시 유한출구로만 해소.
                        for recv_link, intake in recv_intake.items():
                            recv_cap = float(net.urban_link_storage_veh.get(recv_link, 0.0))
                            if recv_cap <= 0.0:
                                continue
                            relief = urban_exit if urban_exit > 0.0 else float("inf")
                            ro = min(recv_cap, max(0.0, recv_occ.get(recv_link, 0.0) + (intake - relief) * dt_h))
                            recv_occ[recv_link] = ro
                    if first_substep:
                        first_offramp_flow = {o: max(0.0, float(offramp_flow.get(o, 0.0))) for o in best_offramp_flow}
                        first_substep = False
                    # Wu own-TTS: 이 link segment 차량 + 이 link ramp queue + off-ramp storage 점유
                    # + (P1) reservoir가 수용 못 한 가상 blocked 큐(urban 상류 externality).
                    link_vehicles = sum(veh_count)
                    link_ramp_queue = sum(max(0.0, ramp_q.get(r, 0.0)) for r in model.owned_ramps)
                    link_offramp_storage = sum(occ.get(o, 0.0) for o in model.owned_offramps)
                    link_blocked_queue = sum(blocked_q.values())
                    cost += (
                        link_vehicles + link_ramp_queue + link_offramp_storage + link_blocked_queue
                    ) * dt_h
            # (ii-b 2026-07-19, 기본 OFF) follower terminal cost: rollout 끝에 남은 ramp+blocked
            # 큐의 삼각 배수 tail(Q²/2R)을 own-TTS에 가산 — 창 밖 배수 비용이 무가격이라
            # metering이 과대평가되는 근시 병리 교정. R = ramp_cap×receiving(ρ_merge_end),
            # far(MFD tail)의 ramp 항과 동일 형태·상수 0(상태 유도).
            if getattr(self.cfg.mpc, "follower_terminal_cost_enabled", False):
                _rho_crit_tc = float(net.rho_crit)
                _rho_max_tc = float(net.rho_max)
                for _ramp_tc in model.owned_ramps:
                    _q_end = max(0.0, ramp_q.get(_ramp_tc, 0.0)) + max(0.0, blocked_q.get(_ramp_tc, 0.0))
                    if _q_end <= 0.0:
                        continue
                    _m_idx = model.ramp_merge_idx[_ramp_tc]
                    _rho_m = float(rhos[_m_idx]) if _m_idx < len(rhos) else 0.0
                    _recv = min(1.0, max(0.0, (_rho_max_tc - _rho_m) / max(_rho_max_tc - _rho_crit_tc, 1.0e-9)))
                    _r_end = max(1.0, float(net.ramp_capacity_veh_h.get(_ramp_tc, 0.0)) * _recv)
                    cost += _q_end * _q_end / (2.0 * _r_end)
            # VdB4 보호큐 벌점 — 링크-agent metering 배선(2026-07-19 3차, 기본 OFF):
            # 방류-결합형(초과 지속 중 미방류 몫 과금). rollout 전체의 평균 release로 근사.
            _pq_mv_f = str(getattr(self.cfg.mpc, "protected_queue_movement", "") or "")
            _pq_w_f = float(getattr(self.cfg.mpc, "protected_queue_weight", 0.0))
            if _pq_mv_f and _pq_w_f > 0.0:
                _pq_spec_f = self.cfg.network.urban_movements.get(_pq_mv_f, {})
                _pq_ramp_f = str(_pq_spec_f.get("ramp", "") or "")
                if _pq_ramp_f in model.owned_ramps:
                    _pq_max_f = float(getattr(self.cfg.mpc, "protected_queue_max_veh", 50.0))
                    _pq_now_f = max(0.0, float(state.urban_movement_queue.get(_pq_mv_f, 0.0)))
                    _pq_exc_f = max(0.0, _pq_now_f - _pq_max_f)
                    if _pq_exc_f > 0.0:
                        _cap_f = max(float(net.ramp_capacity_veh_h.get(_pq_ramp_f, 1500.0)), 1.0e-9)
                        _rel_f = max(0.0, float(ramp_release.get(_pq_ramp_f, 0.0)))
                        _idle_f = 1.0 - min(1.0, _rel_f / _cap_f)
                        cost += _pq_w_f * _pq_exc_f * _idle_f * (sim.T_c_h)
            smooth = sum(abs(first_vec[i] - prev_vec[i]) for i in range(min(n_seg, len(first_vec))))
            for prev_step, next_step in zip(sequence, sequence[1:]):
                smooth += sum(
                    abs(next_step[i] - prev_step[i])
                    for i in range(min(len(prev_step), len(next_step), n_seg))
                )
            # PRICE-TR: VSL 가격 활성이면 smoothness 마찰 0(trust ±10km/h가 보폭 제약).
            if not (self.price_smoothness_disabled and self.vsl_marginal_price):
                cost += smooth_w * smooth
            # B3 VSL 가격항(설정 시에만): + w·g·(vsl_seg − ref_seg). 기본 None=완전 휴면.
            if self.vsl_marginal_price:
                for i, value in enumerate(first_vec):
                    key = f"{link}__seg{i}"
                    g_vsl = self.vsl_marginal_price.get(key)
                    if g_vsl is not None and i < len(prev_vec):
                        ref = float(self.vsl_marginal_price_ref.get(key, float(prev_vec[i])))
                        cost += self.vsl_marginal_price_weight * float(g_vsl) * (
                            float(value) - ref
                        )
            # JOINT vsl×metering cross항(설정 시에만): 이 link 소유 ramp의 metering(previous에
            # 고정)과 후보 link-binding VSL(min seg)의 교차. primal joint은 이미 metering
            # best-response로 포착되나, cross 가격이 externality 곡률을 반영한다.
            if self.vsl_meter_cross_price:
                vsl_bind = float(min(first_vec)) if first_vec else vsl_max
                for ramp in model.owned_ramps:
                    h_c = self.vsl_meter_cross_price.get(ramp)
                    if h_c is None:
                        continue
                    m_ref, v_ref = self.vsl_meter_cross_ref.get(ramp, (0.0, vsl_bind))
                    m_now = float(previous.ramp_metering.get(ramp, float(m_ref)))
                    cost += self.vsl_meter_cross_weight * float(h_c) * (
                        (m_now - float(m_ref)) * (vsl_bind - float(v_ref))
                    )
            evals += 1
            if cost < best_obj:
                best_obj, best_vec = cost, list(first_vec)
                best_offramp_flow = dict(first_offramp_flow)
        # 선택 후보 VSL의 off-ramp 유출 캐시(coupling freeway→urban 재사용).
        if best_obj < float("inf"):
            for off_ramp, flow in best_offramp_flow.items():
                self._wu._last_offramp_flow[off_ramp] = float(flow)
            self._wu._last_offramp_flow[link] = float(sum(best_offramp_flow.values()))
            self._wu._has_last_offramp_flow = True
        else:
            for off_ramp in best_offramp_flow:
                self._wu._last_offramp_flow[off_ramp] = 0.0
            self._wu._last_offramp_flow[link] = 0.0
        vsl_dict: Dict[str, float] = {f"{link}__seg{i}": float(v) for i, v in enumerate(best_vec)}
        vsl_dict[link] = float(min(best_vec)) if best_vec else vsl_max
        return vsl_dict, best_obj, evals

    def _meter_spillback_floor(self, ramp: str, state, coupling, cap: float) -> float:
        """spillback-방지 metering 하한(내재화 2026-07-19, 기본 OFF): 램프 큐가 임계
        (frac×ramp_queue_max)를 넘으면 다음 T_c 안에 임계 아래로 복귀시키는 최소 방류.
        ALINEA queue-override의 내부화 — 후보 격자·budget 사영이 이 하한을 공유해
        계획-집행 정합을 보장한다(외부 감독층 패치는 asym_200서 -4.4→-8.5%p 부정합 실측)."""
        if not getattr(self.cfg.mpc, "meter_queue_constraint_enabled", False):
            return 0.0
        net = self.cfg.network
        frac = float(getattr(self.cfg.mpc, "meter_queue_constraint_frac", 0.8))
        q_thr = frac * float(net.ramp_queue_max_veh)
        q_now = max(0.0, float(state.ramp_queue.get(ramp, 0.0)))
        arrival = max(0.0, float(coupling.get(f"u_on_{ramp}", 0.0)))
        tc_h = max(self.cfg.simulation.T_c_h, 1.0e-9)
        need = arrival + max(0.0, q_now - q_thr) / tc_h
        return min(float(cap), need)

    # ---------- 13-player: freeway segment agent 분해 (2026-07-10 승인 매핑) ----------

    def _solve_freeway_segment_agents(
        self,
        link: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        demand: DemandStep,
        snapshot: ControlAction,
        leader: Optional[object],
        previous: Optional[ControlAction] = None,
    ) -> tuple[Dict[str, float], Dict[str, float], int]:
        """link의 segment agent 4개(F_L0~F_L3)가 각자 own VSL(+소유 ramp meter)을 best-response.

        13-player 매핑(plan-13player-rebuild.md): F_L0=seg0+origin queue, F_L1=seg1,
        F_L2=seg2+R_D, F_L3=seg3+R_F. off-ramp storage는 urban 소유 — 여기서는 동결
        y(현 상태 hold)로만 읽고(λ_eff·유출 cap) own-TTS에 세지 않는다. 이웃 seg 본선과
        이웃 ramp 방류도 동결 y(snapshot metering). own-TTS = 자기 seg 차량 + 자기 ramp
        queue(+blocked) + (seg0) origin queue.

        가격(joint): 전 seg에 g_vsl·(v−ref); leader present 시 소유 ramp agent만
        g_meter·(m−ref) + h·(m−m_ref)(v−v_ref) 2D 탐색(ramp 없는 agent는 1D).
        예산 Σmeter = ω_F·N_UF*(equality)는 owner 2명의 best-response 후 simplex 사영
        (승인안 (ii)) — Jacobi iteration마다 재사영되어 cross-player 합의로 수렴.
        leader=None(PFO/incumbent probe)이면 예산·meter 가격 없이 자율 best-response
        (v2 규약: 자율 분기는 가격-레벨 배제).

        v0 단순화(문서화): VSL 후보는 horizon 내 상수(sequence 탐색 없음), 이웃 y는
        iteration 시작 상태 hold-constant(예측 궤적 교환은 후속 단계)."""
        net = self.cfg.network
        sim = self.cfg.simulation
        ff = self.cfg.freeway_follower
        cfg = self.cfg
        if link not in self._segment_agent_models:
            self._segment_agent_models[link] = build_segment_agent_models(cfg, link)
        agents = self._segment_agent_models[link]
        link_model = self._local_freeway_models[link]
        n_seg = link_model.n_seg
        horizon = max(1, ff.freeway_prediction_horizon_steps or cfg.mpc.horizon_steps)
        substeps = horizon * sim.K_cf
        dt_h = sim.T_f_h
        vsl_max = max(ff.vsl_set)
        smooth_w = ff.vsl_smoothness_weight

        # ---- 동결 y(hold-constant): 본선 상태·이웃 방류·storage 점유(urban 소유)·유출 cap ----
        rhos0 = list(state.freeway_density.get(link, [0.0] * n_seg))
        speeds0 = list(state.freeway_speed.get(link, [net.v_free] * n_seg))
        lanes0 = list(state.freeway_effective_lanes.get(link, [])) or [
            float(net.freeway_lanes) for _ in range(n_seg)
        ]
        if len(lanes0) != n_seg:
            lanes0 = [float(net.freeway_lanes) for _ in range(n_seg)]
        origin_q0 = max(0.0, float(state.mainline_origin_queue.get(link, 0.0)))
        occ0: Dict[str, float] = {}
        storage_avail0: Dict[str, float] = {}
        for off_ramp in link_model.owned_offramps:
            cap_o = link_model.offramp_storage_cap.get(off_ramp, 0.0)
            storage = net.off_ramp_storage_link.get(off_ramp, "")
            avail = float(state.urban_link_storage.get(storage, cap_o))
            occ0[off_ramp] = max(0.0, cap_o - avail)
            storage_avail0[off_ramp] = max(0.0, avail)
        offramp_cap0 = self._local_offramp_capacity(link, storage_avail0)
        release0 = {
            r: max(0.0, float(snapshot.ramp_metering.get(r, net.ramp_capacity_veh_h[r])))
            for r in link_model.owned_ramps
        }
        # 궤적 교환(ỹ): 직전 Jacobi iteration의 예측 입력궤적이 있으면 그걸 동결 y로 쓴다.
        # 없으면(첫 iteration) 현 상태 hold-constant. occupancy(urban 소유)·유출 cap은
        # 항상 state 동결 — 교환 대상은 본선 (ρ,v,λ)와 owner 방류 스케줄만.
        traj_prev = self._seg_traj.get(link) if self.seg13_traj_exchange else None
        if traj_prev is not None and traj_prev.get("rhos"):
            t_len = len(traj_prev["rhos"])
            frozen = FrozenLinkTrajectory(
                rhos=[list(r) for r in traj_prev["rhos"]],
                speeds=[list(r) for r in traj_prev["speeds"]],
                prev_lanes=[list(r) for r in traj_prev["lanes"]],
                origin_queue=[origin_q0 for _ in range(t_len)],
                ramp_release=[dict(r) for r in traj_prev["release"]],
                occupancy=[dict(occ0) for _ in range(t_len)],
                offramp_capacity=[dict(offramp_cap0) for _ in range(t_len)],
            )
            self._seg13_diag[f"wu_seg13_traj_used_{link}"] = 1.0
        else:
            frozen = FrozenLinkTrajectory(
                rhos=[rhos0], speeds=[speeds0], prev_lanes=[lanes0],
                origin_queue=[origin_q0], ramp_release=[release0],
                occupancy=[occ0], offramp_capacity=[offramp_cap0],
            )
        ramp_q0 = {
            r: max(0.0, float(state.ramp_queue.get(r, 0.0))) for r in link_model.owned_ramps
        }

        evals = 0
        vsl_out: Dict[str, float] = {}
        preferred_meter: Dict[str, float] = {}
        best_offramp_flow: Dict[str, float] = {}
        agent_best_traj: Dict[int, tuple] = {}
        leader_present = (
            leader is not None and float(getattr(leader, "N_UF_star", 0.0)) > 0.0
        )
        nuf_mode = str(getattr(
            self.cfg.mpc, "wu_faithful_nuf_coordination_mode", "equality",
        ))
        # 8단계 ablation(dual): λ_UF·m — DUAL-STANDING 규약(windup 수정): 영속 가격이라
        # incumbent(leader=None) solve에도 적용해 적분 루프의 액추에이터를 유지한다.
        lambda_uf = float(self._lambda_UF)
        dual_price_active = nuf_mode == "dual" and abs(lambda_uf) > 1.0e-12
        # METER-BOX 판별 카운터(박스 끝 선택 vs 내부 정착).
        _meter_box_total = 0
        _meter_box_edge = 0

        for agent in agents:
            seg = agent.seg
            key = f"{link}__seg{seg}"
            prev_v = float(segment_vsl(snapshot, link, seg, cfg))
            # VSL-BOX(2026-07-17, 사용자 지시): 기존 필터는 앵커가 Jacobi 반복 내부
            # snapshot이라 sweep마다 ±max_vsl_step 재앵커 → 스텝당 20×sweep수
            # (실측: ③ 10셀에서 max|Δ| 50 = 명목 20의 2.5배, 위반 112/7020).
            # 박스 ON이면 앵커를 직전 step commit(previous)으로 고정 — METER-BOX와 동일
            # 규약. vsl_set 간격 10이라 ±10이면 {prev-10, prev, prev+10}, 흡수 불가.
            _vbox = getattr(self.cfg.mpc, "seg13_vsl_box_kmh", None)
            if _vbox is not None and previous is not None:
                _v_anchor = float(segment_vsl(previous, link, seg, cfg))
                v_cands = [
                    float(v) for v in ff.vsl_set
                    if abs(float(v) - _v_anchor) <= float(_vbox) + 1.0e-9
                ] or [_v_anchor]
            else:
                v_cands = [
                    float(v) for v in ff.vsl_set
                    if abs(float(v) - prev_v) <= ff.max_vsl_step + 1.0e-9
                ] or [prev_v]
            own_ramp = agent.owned_ramps[0] if agent.owned_ramps else None
            if own_ramp is not None and self.metering_enabled:
                cap_r = float(net.ramp_capacity_veh_h[own_ramp])
                # METER-BOX(2026-07-17, 사용자 설계): 고정 격자 {cap·f} 대신 직전 step
                # commit(m_prev) 중심 ±R 박스 안에 등간격 5점을 찍는다.
                #   근거 1(실측): 선형 가격 × 이산 격자 → 내부 rung 선택 0/160, 끝점 60~62%,
                #     '부호→끝점' 적중 80~85%. 끝점이 진짜 최적이면 머물러야 하는데 왕복한다
                #     = 진짜 최적은 내부이고 선형 외삽(±300 측정을 1125까지 연장)이 지나친다.
                #   근거 2: R=300이면 박스=가격 FD 측정폭(d_r=0.20×1500) — 가격이 측정된
                #     구간 안에서만 쓰여 외삽이 소멸("허용 이동폭만큼 측정" 규약 충족).
                #   흡수 없음: 격자 필터(워크플로 설계)와 달리 점을 새로 찍으므로 m_prev
                #     자신이 항상 후보 → cap 동결 불가. 5점 유지로 평가수 동일(계산 중립).
                # 기준점은 previous(직전 step commit)다 — snapshot.ramp_metering은 Jacobi
                # iteration 1에서 uncontrolled=cap(state.py:1090)이라 기준이 될 수 없다.
                _box_r = getattr(self.cfg.mpc, "seg13_meter_box_veh_h", None)
                if _box_r is not None and previous is not None:
                    _R = float(_box_r)
                    # 비대칭 박스(2026-07-17, 사용자 설계 2차): 내림은 R(큐 생성 방향이라
                    # 보수적), 올림은 R_up(회복/방류 방향). 파국 2셀(170_w/200_w)이 전부
                    # '낮은 곳에 갇혀 못 올라옴'이어서 올림만 넓힌다. 미설정=R(대칭, 기존).
                    _bu = getattr(self.cfg.mpc, "seg13_meter_box_up_veh_h", None)
                    _R_up = float(_bu) if _bu is not None else _R
                    m_prev_r = min(max(float(
                        previous.ramp_metering.get(own_ramp, cap_r)), 0.0), cap_r)
                    m_cands = sorted(
                        {round(min(max(m_prev_r + off, 0.0), cap_r), 6)
                         for off in (-_R, -_R / 2.0, 0.0, _R_up / 2.0, _R_up)},
                        reverse=True,
                    )
                else:
                    # 내림차순(전량 방류 우선) — 7p 분율 순서(1.0, 0.7, …)와 동일 규약.
                    # own-TTS는 보존식 때문에 방류에 근사-무차별인 레짐이 흔해서 tie-break가
                    # 결정적: 오름차순이면 최소 방류로 쏠려 전면 질식(PFO 13p 실측 병리).
                    m_cands = sorted(
                        {round(cap_r * f, 6) for f in self.ramp_metering_fractions},
                        reverse=True,
                    )
            else:
                m_cands = [None]
            # 내재화(2026-07-19): spillback-방지 하한을 후보 격자에 적용 — 하한 아래
            # 후보는 하한으로 끌어올려(중복 제거) rollout 채점이 실제 집행값을 본다.
            if own_ramp is not None and m_cands and m_cands[0] is not None:
                _m_floor = self._meter_spillback_floor(own_ramp, state, coupling, cap_r)
                if _m_floor > 0.0:
                    m_cands = sorted(
                        {round(max(float(m), _m_floor), 6) for m in m_cands},
                        reverse=True,
                    )
            best_cost, best_v, best_m = float("inf"), prev_v, None
            best_flow_local: Dict[str, float] = {}
            best_traj: Optional[tuple] = None  # (rho[t], v[t], lane[t], rel[t]) — 입력시점 기록
            for v_cand in v_cands:
                for m_cand in m_cands:
                    cand = ControlAction(
                        ramp_metering=dict(snapshot.ramp_metering),
                        vsl=dict(snapshot.vsl),
                        green_times=dict(snapshot.green_times),
                        offsets=dict(snapshot.offsets),
                        inflow_outflow_allocation={},
                    )
                    cand.vsl[key] = float(v_cand)
                    if own_ramp is not None and m_cand is not None:
                        cand.ramp_metering[own_ramp] = float(m_cand)
                    own = SegmentLocalState(
                        rho=float(rhos0[seg]), speed=float(speeds0[seg]),
                        prev_lane=float(lanes0[seg]),
                        origin_queue=origin_q0 if agent.owns_origin_queue else 0.0,
                    )
                    ramp_q = float(ramp_q0.get(own_ramp, 0.0)) if own_ramp else 0.0
                    blocked = 0.0
                    first_flow: Dict[str, float] = {}
                    # radius-1 이웃 상태(활성 시): ±1 seg를 함께 전진(2차 이웃은 동결 y).
                    nbr_states: Dict[int, SegmentLocalState] = {}
                    if self.seg13_neighbor_weight > 0.0:
                        for j in (seg - 1, seg + 1):
                            if 0 <= j < n_seg:
                                nbr_states[j] = SegmentLocalState(
                                    rho=float(rhos0[j]), speed=float(speeds0[j]),
                                    prev_lane=float(lanes0[j]),
                                    origin_queue=origin_q0 if j == 0 else 0.0,
                                )
                    cost = 0.0
                    tr_rho: List[float] = []
                    tr_v: List[float] = []
                    tr_lane: List[float] = []
                    tr_rel: List[float] = []
                    for t in range(substeps):
                        # 궤적 교환용 입력시점 기록(substep t 시작 상태) — frozen.at(t) 의미와 정렬.
                        tr_rho.append(float(own.rho))
                        tr_v.append(float(own.speed))
                        tr_lane.append(float(own.prev_lane))
                        own_release: Dict[str, float] = {}
                        if own_ramp is not None:
                            # release는 T_f 시작 reservoir만 본다(spec 3.4.3) — 계산 후 적재.
                            rhos_asm = list(rhos0)
                            rhos_asm[seg] = float(own.rho)
                            rel = self._local_ramp_release(
                                link, rhos_asm, {own_ramp: ramp_q}, cand, demand,
                            )
                            r_own = max(0.0, float(rel.get(own_ramp, 0.0)))
                            ramp_q = max(0.0, ramp_q - r_own * dt_h)
                            approach = max(
                                0.0, float(coupling.get(f"u_on_{own_ramp}", 0.0))
                            )
                            if self.count_blocked_ramp_inflow:
                                space = max(0.0, net.ramp_queue_max_veh - ramp_q)
                                arrival = approach * dt_h
                                adm1 = min(blocked, space)
                                adm2 = min(arrival, space - adm1)
                                ramp_q = min(net.ramp_queue_max_veh, ramp_q + adm1 + adm2)
                                blocked = blocked - adm1 + (arrival - adm2)
                            else:
                                ramp_q = min(
                                    net.ramp_queue_max_veh, ramp_q + approach * dt_h,
                                )
                            own_release[own_ramp] = r_own
                        tr_rel.append(float(own_release.get(own_ramp, 0.0)) if own_ramp else 0.0)
                        # radius-1: 이웃을 time-t 상태 기준으로 동시(자코비) 전진 —
                        # 이웃 방류는 동결 스케줄, 이웃 차량수는 w_nbr 가중 비용.
                        new_nbr: Dict[int, SegmentLocalState] = {}
                        if nbr_states:
                            frz_rel = frozen.ramp_release[
                                min(t, len(frozen.ramp_release) - 1)
                            ]
                            cur_all: Dict[int, SegmentLocalState] = dict(nbr_states)
                            cur_all[seg] = own
                            for j, st_j in nbr_states.items():
                                agent_j = agents[j]
                                rel_j = {
                                    r: max(0.0, float(frz_rel.get(r, 0.0)))
                                    for r in agent_j.owned_ramps
                                }
                                ov = {k: v for k, v in cur_all.items() if k != j}
                                nst, _, veh_j = segment_substep_local(
                                    agent_j, frozen, t, st_j, rel_j, cand, demand,
                                    extra_overrides=ov,
                                )
                                new_nbr[j] = nst
                                cost += self.seg13_neighbor_weight * float(veh_j) * dt_h
                        own, off_flow, veh = segment_substep_local(
                            agent, frozen, t, own, own_release, cand, demand,
                            extra_overrides=nbr_states or None,
                        )
                        if nbr_states:
                            nbr_states = new_nbr
                        if t == 0:
                            first_flow = dict(off_flow)
                        cost += (
                            float(veh) + ramp_q + blocked
                            + (own.origin_queue if agent.owns_origin_queue else 0.0)
                        ) * dt_h
                        # VdB4 보호큐 벌점 — seg13 metering 배선(2026-07-19 3차, 기본 OFF).
                        # 2차(blocked 투영)는 후보 불변항 지배로 기울기 0(비트동일 실측) —
                        # 방류-결합형으로 교체: 보호 큐가 한계 초과인 동안 "방류 안 한 몫"
                        # (1 − release/cap)에 초과분×w를 부과 → release↑가 직접 벌점을 줄인다.
                        if own_ramp is not None:
                            _pq_w_s = float(getattr(self.cfg.mpc, "protected_queue_weight", 0.0))
                            if _pq_w_s > 0.0:
                                _pq_mv_s = str(getattr(self.cfg.mpc, "protected_queue_movement", "") or "")
                                if _pq_mv_s:
                                    _pq_ramp_s = str(self.cfg.network.urban_movements.get(_pq_mv_s, {}).get("ramp", "") or "")
                                    if _pq_ramp_s == own_ramp:
                                        _pq_max_s = float(getattr(self.cfg.mpc, "protected_queue_max_veh", 50.0))
                                        _pq_now_s = max(0.0, float(state.urban_movement_queue.get(_pq_mv_s, 0.0)))
                                        _pq_exc_s = max(0.0, _pq_now_s - _pq_max_s)
                                        if _pq_exc_s > 0.0:
                                            _idle_s = 1.0 - min(1.0, r_own / max(cap_r, 1.0e-9))
                                            cost += _pq_w_s * _pq_exc_s * _idle_s * dt_h
                    # 마찰(암묵적 신호검정) — 7p와 동일 규약(PRICE-TR 시 0).
                    if not (self.price_smoothness_disabled and self.vsl_marginal_price):
                        cost += smooth_w * abs(float(v_cand) - prev_v)
                    # 가격항: g_vsl(전 seg), 소유 ramp의 g_meter + h cross(leader present).
                    if self.vsl_marginal_price:
                        g_vsl = self.vsl_marginal_price.get(key)
                        if g_vsl is not None:
                            ref_v = float(self.vsl_marginal_price_ref.get(key, prev_v))
                            cost += self.vsl_marginal_price_weight * float(g_vsl) * (
                                float(v_cand) - ref_v
                            )
                    if own_ramp is not None and m_cand is not None and (
                        leader_present or self.seg13_meter_price_standing
                    ):
                        if self.metering_marginal_price:
                            g_m = self.metering_marginal_price.get(own_ramp)
                            if g_m is not None:
                                m_ref = float(self.metering_marginal_price_ref.get(
                                    own_ramp, float(m_cand),
                                ))
                                cost += (
                                    self.metering_marginal_price_weight
                                    * float(g_m) * (float(m_cand) - m_ref)
                                )
                        if self.vsl_meter_cross_price:
                            h_c = self.vsl_meter_cross_price.get(own_ramp)
                            if h_c is not None:
                                m_ref2, v_ref2 = self.vsl_meter_cross_ref.get(
                                    own_ramp, (0.0, vsl_max),
                                )
                                cost += self.vsl_meter_cross_weight * float(h_c) * (
                                    (float(m_cand) - float(m_ref2))
                                    * (float(v_cand) - float(v_ref2))
                                )
                    if own_ramp is not None and m_cand is not None and dual_price_active:
                        cost += lambda_uf * float(m_cand)
                    evals += 1
                    if cost < best_cost:
                        best_cost, best_v, best_m = cost, float(v_cand), m_cand
                        best_flow_local = dict(first_flow)
                        best_traj = (tr_rho, tr_v, tr_lane, tr_rel)
            if best_traj is not None:
                agent_best_traj[seg] = best_traj
            vsl_out[key] = float(best_v)
            if own_ramp is not None and best_m is not None:
                preferred_meter[own_ramp] = float(best_m)
                # METER-BOX 판별 진단: 박스 끝(±R)을 골랐나, 내부에 정착했나.
                # 끝점 비율이 계속 높으면 '진짜 최적이 박스 밖' — 사용자 가설 반증 쪽.
                if _box_r is not None and previous is not None:
                    _bu0 = getattr(self.cfg.mpc, "seg13_meter_box_up_veh_h", None)
                    _rup0 = float(_bu0) if _bu0 is not None else float(_box_r)
                    _mp0 = min(max(float(
                        previous.ramp_metering.get(own_ramp, cap_r)), 0.0), cap_r)
                    _blo = max(0.0, _mp0 - float(_box_r))
                    _bhi = min(cap_r, _mp0 + _rup0)
                    _meter_box_total += 1
                    if abs(float(best_m) - _blo) < 0.5 or abs(float(best_m) - _bhi) < 0.5:
                        _meter_box_edge += 1
            for o, fl in best_flow_local.items():
                best_offramp_flow[o] = float(fl)

        # ---- 궤적 교환 저장: 각 agent의 best 예측 입력궤적을 다음 iteration의 y로 ----
        # α=0.5 under-relaxation(루프의 coupling 블렌딩과 동일 규약)으로 진동을 누른다.
        if self.seg13_traj_exchange and len(agent_best_traj) == n_seg:
            alpha = 0.5
            new_rhos = [
                [float(agent_best_traj[j][0][t]) for j in range(n_seg)]
                for t in range(substeps)
            ]
            new_speeds = [
                [float(agent_best_traj[j][1][t]) for j in range(n_seg)]
                for t in range(substeps)
            ]
            new_lanes = [
                [float(agent_best_traj[j][2][t]) for j in range(n_seg)]
                for t in range(substeps)
            ]
            new_release: List[Dict[str, float]] = []
            for t in range(substeps):
                rel_t = dict(release0)
                for a in agents:
                    if a.owned_ramps and a.seg in agent_best_traj:
                        rel_t[a.owned_ramps[0]] = float(agent_best_traj[a.seg][3][t])
                new_release.append(rel_t)
            prev_stored = self._seg_traj.get(link)
            if prev_stored is not None and len(prev_stored.get("rhos", [])) == substeps:
                for t in range(substeps):
                    for j in range(n_seg):
                        new_rhos[t][j] = (1.0 - alpha) * prev_stored["rhos"][t][j] + alpha * new_rhos[t][j]
                        new_speeds[t][j] = (1.0 - alpha) * prev_stored["speeds"][t][j] + alpha * new_speeds[t][j]
                        new_lanes[t][j] = (1.0 - alpha) * prev_stored["lanes"][t][j] + alpha * new_lanes[t][j]
                    for r in new_release[t]:
                        old = float(prev_stored["release"][t].get(r, new_release[t][r]))
                        new_release[t][r] = (1.0 - alpha) * old + alpha * new_release[t][r]
            self._seg_traj[link] = {
                "rhos": new_rhos, "speeds": new_speeds,
                "lanes": new_lanes, "release": new_release,
            }

        # link 대표 VSL(plant fallback·진단) = min seg — 7p 규약 유지.
        seg_vals = [vsl_out.get(f"{link}__seg{i}", vsl_max) for i in range(n_seg)]
        vsl_out[link] = float(min(seg_vals)) if seg_vals else vsl_max
        # f→u 결합 캐시(선택 후보의 첫 substep off-ramp 유출) — 7p와 동일 규약.
        for off_ramp in link_model.owned_offramps:
            self._wu._last_offramp_flow[off_ramp] = float(
                best_offramp_flow.get(off_ramp, 0.0)
            )
        self._wu._last_offramp_flow[link] = float(
            sum(best_offramp_flow.get(o, 0.0) for o in link_model.owned_offramps)
        )
        self._wu._has_last_offramp_flow = True

        # ---- 예산 simplex 사영(승인안 (ii)): Σmeter = ω_F·N_UF* (equality) ----
        # dual 모드면 사영 없음 — 총량은 λ_UF(step 간 적분)가 추적, 레벨은 자율.
        # BUDGET_OFF(2026-07-17 수정): 이 SEG13 경로엔 게이트가 **없었다**. 비-SEG13 경로
        # (L2886)만 막혀 있어서 SEG13=1인 플래그십에선 BUDGET_OFF가 무효였고, 2026-07-16
        # wave2 20런이 통째로 ③ 재실행이 됐다(30/30 bit-identical). 여기서도 막는다.
        meter_out: Dict[str, float] = dict(preferred_meter)
        _budget_off_seg13 = bool(getattr(self.cfg.mpc, "leader_budget_off", False))
        if leader_present and not _budget_off_seg13 and nuf_mode == "equality" and preferred_meter:
            caps = {r: float(net.ramp_capacity_veh_h[r]) for r in preferred_meter}
            cap_sum = sum(caps.values())
            omega_f = float(self._wu._omega_f.get(link, 0.0))
            n_uf_star = float(getattr(leader, "N_UF_star", 0.0))
            budget = min(max(omega_f * n_uf_star, 0.0), cap_sum)
            total_pref = sum(preferred_meter.values())

            # METER-BOX Site B: 사영도 같은 박스 안에서. 안 묶으면 여기가 새는 구멍이다 —
            # 실측 격자밖 값 38~40%(160개 중)가 전부 이 사영의 산물이고, per-ramp 1199도
            # 여기서 나왔다(격자 최대 span=1125). None이면 lo=0/hi=caps → 기존과 비트동일.
            _box_r_b = getattr(self.cfg.mpc, "seg13_meter_box_veh_h", None)
            if _box_r_b is not None and previous is not None:
                _Rb = float(_box_r_b)
                _bu_b = getattr(self.cfg.mpc, "seg13_meter_box_up_veh_h", None)
                _Rb_up = float(_bu_b) if _bu_b is not None else _Rb
                _rl_lo, _rl_hi = {}, {}
                for r in preferred_meter:
                    _mp = min(max(float(previous.ramp_metering.get(r, caps[r])), 0.0), caps[r])
                    _rl_lo[r] = max(0.0, _mp - _Rb)
                    _rl_hi[r] = min(caps[r], _mp + _Rb_up)
            else:
                _rl_lo = {r: 0.0 for r in preferred_meter}
                _rl_hi = dict(caps)
            # 내재화(2026-07-19): spillback-방지 하한을 사영 회랑 lo에 합류 — 기존
            # "박스=하드, 예산이 양보" 규약에 제약이 올라타 leader budget이 자동 양보.
            # 하한이 박스 상단을 넘으면 제약 우선(hi도 상향).
            for r in preferred_meter:
                _fl_r = self._meter_spillback_floor(r, state, coupling, caps[r])
                if _fl_r > _rl_lo[r]:
                    _rl_lo[r] = _fl_r
                if _rl_hi[r] < _rl_lo[r]:
                    _rl_hi[r] = _rl_lo[r]

            def _scale_to(target: float) -> Dict[str, float]:
                # 목표 합 target으로 비례 사영(용량 클립 + 잔여 재분배) — 기존 등식
                # 로직을 회랑 상·하한이 공용하도록 함수화(로직 비트 동일).
                # METER-BOX: 박스가 target을 못 담으면 **예산이 양보한다**(박스=하드).
                target = min(max(target, sum(_rl_lo.values())), sum(_rl_hi.values()))
                if total_pref <= 1.0e-9:
                    return {
                        r: min(max(target * caps[r] / max(cap_sum, 1.0e-9),
                                   _rl_lo[r]), _rl_hi[r])
                        for r in preferred_meter
                    }
                scale = target / total_pref
                out = {
                    r: min(_rl_hi[r], max(_rl_lo[r], m * scale))
                    for r, m in preferred_meter.items()
                }
                deficit = target - sum(out.values())
                if deficit > 1.0e-9:
                    for r in sorted(
                        out, key=lambda x: _rl_hi[x] - out[x], reverse=True,
                    ):
                        room = _rl_hi[r] - out[r]
                        add = min(room, deficit)
                        out[r] += add
                        deficit -= add
                        if deficit <= 1.0e-9:
                            break
                # METER-BOX 신규: rl_lo>0이면 하한 클립이 target을 **초과**시킬 수 있다
                # (기존 lo=0에선 불가능했던 방향). 대칭 surplus 루프로 하한 여유가 큰
                # 램프부터 깎는다. 박스 OFF(lo=0)면 발화 불가 → 비트동일.
                surplus = sum(out.values()) - target
                if surplus > 1.0e-9:
                    for r in sorted(
                        out, key=lambda x: out[x] - _rl_lo[x], reverse=True,
                    ):
                        room = out[r] - _rl_lo[r]
                        cut = min(room, surplus)
                        out[r] -= cut
                        surplus -= cut
                        if surplus <= 1.0e-9:
                            break
                return out

            if self.seg13_budget_inequality and total_pref <= budget + 1.0e-9:
                # 회랑 예산(2026-07-14): α·budget ≤ Σmeter ≤ budget. 회랑 안이면 자율
                # 존중(하향 자유), 하한 아래(과소방류 나선 조짐)면 α·budget으로 비례
                # 상향. α=0이면 구 부등식(하한 없음), α=1이면 구 등식과 동치.
                # 부하 적응형 α(2026-07-15): 나선은 혼잡(breakdown 근방)에서만 발생하므로
                # 하한도 그때만 필요. 본선이 자유류면 α→0(follower 자율 존중, 경부하 과잉
                # 조임 해소), 임계 근방이면 α→α_max(나선 방어). 본선 최대밀도/ρ_crit로 게이팅.
                alpha_max = float(self.seg13_release_floor_frac)
                if self.seg13_release_floor_adaptive:
                    rho_crit = float(net.rho_crit)
                    seg_rho = state.freeway_density.get(link, [])
                    cong = (max(seg_rho) / rho_crit) if (seg_rho and rho_crit > 0) else 0.0
                    c_lo = float(self.seg13_floor_cong_lo)
                    c_hi = float(self.seg13_floor_cong_hi)
                    span = max(c_hi - c_lo, 1.0e-9)
                    frac = min(max((cong - c_lo) / span, 0.0), 1.0)
                    alpha_eff = alpha_max * frac
                    self._seg13_diag[f"wu_seg13_floor_cong_{link}"] = float(cong)
                    self._seg13_diag[f"wu_seg13_floor_alpha_{link}"] = float(alpha_eff)
                else:
                    alpha_eff = alpha_max
                floor_b = alpha_eff * budget
                if total_pref >= floor_b - 1.0e-9:
                    meter_out = dict(preferred_meter)
                else:
                    meter_out = _scale_to(floor_b)
            else:
                meter_out = _scale_to(budget)
            self._seg13_diag[f"wu_seg13_budget_{link}"] = float(budget)
            self._seg13_diag[f"wu_seg13_presplit_{link}"] = float(total_pref)
            self._seg13_diag[f"wu_seg13_postsplit_{link}"] = float(
                sum(meter_out.values())
            )
            # 나선 감시(2026-07-14): 실현 Σmeter/budget 비율 — 지속 하락 + 램프 큐
            # 상승 동반이면 과소방류 나선 서명. 컨트롤러의 release-트리거 재선형화 재료.
            self._seg13_diag[f"wu_b3_release_ratio_{link}"] = float(
                sum(meter_out.values()) / max(budget, 1.0e-9)
            )
        self._seg13_diag[f"wu_seg13_evals_{link}"] = float(evals)
        # METER-BOX 진단 — 플래그 ON일 때만 기록(OFF 런의 CSV 헤더 불변 유지).
        # box_r 존재 = 플래그가 SEG13 경로에 실제로 도달했다는 영수증(BUDGET_OFF 재발 방지).
        if getattr(self.cfg.mpc, "seg13_meter_box_veh_h", None) is not None:
            self._seg13_diag[f"wu_seg13_meter_box_r_{link}"] = float(
                self.cfg.mpc.seg13_meter_box_veh_h
            )
            self._seg13_diag[f"wu_seg13_meter_box_edge_{link}"] = float(_meter_box_edge)
            self._seg13_diag[f"wu_seg13_meter_box_total_{link}"] = float(_meter_box_total)
            _bu_d = getattr(self.cfg.mpc, "seg13_meter_box_up_veh_h", None)
            if _bu_d is not None:
                # 비대칭 영수증 — 이 컬럼 존재 = up 플래그가 SEG13 경로에 도달.
                self._seg13_diag[f"wu_seg13_meter_box_rup_{link}"] = float(_bu_d)
        _vbox_d = getattr(self.cfg.mpc, "seg13_vsl_box_kmh", None)
        if _vbox_d is not None:
            # VSL-BOX 영수증 — 존재 = previous 앵커 필터가 SEG13 경로에 도달.
            self._seg13_diag[f"wu_seg13_vsl_box_r_{link}"] = float(_vbox_d)
        # 내재화 최종 정합(2026-07-19): 어떤 경로(budget off/inequality 자율 존중 포함)든
        # 반환 metering은 spillback 하한 이상 — 컨트롤러 반환값 = plant 집행값 보장.
        if getattr(self.cfg.mpc, "meter_queue_constraint_enabled", False) and meter_out:
            for _r_fl in list(meter_out.keys()):
                _cap_fl = float(net.ramp_capacity_veh_h.get(_r_fl, meter_out[_r_fl]))
                _fl_v = self._meter_spillback_floor(_r_fl, state, coupling, _cap_fl)
                if meter_out[_r_fl] < _fl_v:
                    self._seg13_diag[f"wu_seg13_qcon_lift_{_r_fl}"] = _fl_v - meter_out[_r_fl]
                    meter_out[_r_fl] = _fl_v
        return vsl_out, meter_out, evals

    # ---------- freeway agent: 진짜 ramp metering 탐색 (핵심 신규) ----------

    def _solve_freeway_agent_metered(
        self,
        link: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        demand: DemandStep,
        snapshot: ControlAction,
        leader: Optional[object] = None,
        previous: Optional[ControlAction] = None,
    ) -> tuple[Dict[str, float], Dict[str, float], int]:
        """freeway agent의 VSL + ramp_metering 결합 탐색 — 반환 (vsl_dict, metering_dict, evals).

        SPEC: freeway agent가 자기 partition의 on-ramp를 소유하므로(`ramp_to_freeway[ramp]==link`),
        metering은 own-TTS 최소화 안에서 결정된다. leader 유무로 두 갈래로 나뉜다.

        **leader present (N_UF_star>0):** N_UF_star는 총 urban→freeway 교환유량 target이다.
        이 link 몫 budget B = ω_F[link]·N_UF_star(feasible [0,Σcap]로 clamp)로 소유 ramp metering
        합을 **hard로 고정**한다. follower는 B를 소유 ramp들에 어떻게 배분(allocation simplex)할지만
        탐색해 per-link own-TTS를 최소화한다(2 ramp이면 1-D, 7점). soft N_UF penalty는 제거됐다 —
        budget이 패널티가 아니라 제약으로 정확히 충족된다.

        **leader=None (PFO):** 기존 autonomous per-ramp metering 좌표하강(미변경). 이 link 소유
        ramp 각각에 대해 5개 분율(`ramp_metering_fractions`)을 훑고 나머지는 현재 best에 고정.
        호출 횟수 ≈ Σ_ramp(분율 수). de-facto metering이 own-TTS 최소화에서 창발한다.

        어느 분기든 각 후보 metering 하의 own-TTS는 `_solve_freeway_agent_local`(per-link 국소
        METANET rollout, VSL best-response 포함)로 채점한다.
        """
        net = self.cfg.network
        owned_ramps = [r for r in net.ramps if net.ramp_to_freeway.get(r) == link]
        caps = {r: float(net.ramp_capacity_veh_h[r]) for r in owned_ramps}
        # BASELINE-BOX(2026-07-17, 사용자 지시): PFO 경로 metering 이동 한계 — walk-MVG의
        # METER-BOX와 동일 규약(prev commit 앵커 ±300). 실측: 무제한 PFO는 per-step 최대
        # 1125(격자 전폭) 점프 → 공정비교(§2.4 authority group 내 bounds 통일) 위반.
        _bb_on = (
            bool(getattr(self.cfg.mpc, "baseline_move_box", False))
            and previous is not None
        )
        _bb: Dict[str, tuple] = {}
        _bb_cands: Dict[str, list] = {}
        if _bb_on:
            for _r in owned_ramps:
                _p = min(max(float(previous.ramp_metering.get(_r, caps[_r])), 0.0), caps[_r])
                _bb[_r] = (max(0.0, _p - 300.0), min(caps[_r], _p + 300.0))
                # 격자 흡수 방지(실측: ±300 필터만 걸면 1500에서 격자 간극 450에 막혀
                # 동결) — walk-MVG METER-BOX와 동일하게 격자 대신 박스 점 5개로 교체.
                _bb_cands[_r] = sorted(
                    {round(min(max(_p + _off, 0.0), caps[_r]), 6)
                     for _off in (-300.0, -150.0, 0.0, 150.0, 300.0)},
                    reverse=True,
                )

        def _bb_ok(_r: str, _v: float) -> bool:
            if not _bb_on:
                return True
            _lo, _hi = _bb[_r]
            return _lo - 1.0e-9 <= float(_v) <= _hi + 1.0e-9
        evals_total = 0

        def _solve_with(meter: Mapping[str, float]) -> tuple[Dict[str, float], float, int]:
            probe_prev = ControlAction(
                ramp_metering=dict(snapshot.ramp_metering),
                vsl=dict(snapshot.vsl),
                green_times=dict(snapshot.green_times),
                offsets=dict(snapshot.offsets),
                inflow_outflow_allocation={},
            )
            probe_prev.ramp_metering.update({r: float(v) for r, v in meter.items()})
            vsl_dict, cost, e = self._solve_freeway_agent_local(
                link, state, coupling, demand, probe_prev,
            )
            return vsl_dict, cost, e

        # ---- B3 metering 가격항 준비(설정 시에만; 기본 None=완전 휴면) ----
        # leader가 동결 운영점에서 완성해 하달한 g_ext를 선형으로 더한다. green과 동일
        # 규약 — solve 안에서 d_local을 재계산하지 않는다(운영점 혼합·중복 rollout 방지).
        n_uf_star = float(getattr(leader, "N_UF_star", 0.0)) if leader is not None else 0.0
        omega_f_price = float(self._wu._omega_f.get(link, 0.0))
        cap_sum_price = sum(caps[r] for r in owned_ramps) if owned_ramps else 0.0
        budget_price = (
            float(np.clip(omega_f_price * n_uf_star, 0.0, cap_sum_price))
            if cap_sum_price > 0.0 else 0.0
        )
        # DUAL×PRICE(2026-07-09, 사용자 지시): nuf dual 모드면 가격 분기에서도 soft anchor
        # 대신 λ_UF·Σmeter를 쓴다 — leader의 수량 target(N_UF*)을 추적하는 dual이 marginal
        # 가격(g_ext)과 공존. anchor(|Σ−budget|, w=T_c_h≈0.05)는 사실상 무력해 far-informed
        # 수량 결정이 집행 안 되던 구멍을 λ_UF(적분 피드백)가 메운다.
        nuf_mode_priced = str(getattr(
            self.cfg.mpc, "wu_faithful_nuf_coordination_mode", "equality"
        ))
        # DUAL-STANDING(2026-07-09 windup 수정): λ_UF는 follower의 **영속 가격** — green/vsl
        # 가격처럼 leader 부재 solve(PFO incumbent probe)에도 적용돼야 한다. 기존(leader
        # 있을 때만)은 incumbent가 λ-면제라 적분 루프에 액추에이터가 없었다: incumbent가
        # 이기는 동안 실현 Σmeter가 λ에 무반응 → 오차 양수 지속 → λ가 cap(1.0)까지 windup
        # → leader 후보만 오염(metering→0) → incumbent 선택 고착(실측 G1DF 11909/APJOINT
        # 13116, 전 스텝 incumbent). 순수 PFO 컨트롤러는 λ를 설정하지 않아(=0) 무영향.
        dual_standing = (
            nuf_mode_priced == "dual" and abs(float(self._lambda_UF)) > 1.0e-12
        )
        # SPLIT-PRICE: split 모드면 가격이 있어도 레벨-조절 분기(priced branch)를 타지
        # 않는다 — 총량은 아래 equality 분기가 강제하고, 가격은 그 배분 채점에만 합산.
        price_split = (
            self.metering_price_split
            and self.metering_marginal_price is not None
            and leader is not None
            and n_uf_star > 0.0
            and bool(owned_ramps)
        )
        priced_metering = (
            self.metering_marginal_price is not None
            and bool(owned_ramps)
            and not price_split
            and (
                (leader is not None and n_uf_star > 0.0)
                or dual_standing
            )
        )

        def _price_metering_cost(meter: Mapping[str, float]) -> float:
            if self.metering_marginal_price is None or not owned_ramps:
                return 0.0
            # SPLIT-PRICE 정합 v2(2026-07-09): split 모드에선 가격이 '총량'을 정하는
            # 경로를 컨트롤러 어디에도 두지 않는다 — leader=None(incumbent/PFO probe)
            # 자율 분기의 레벨 탐색에서도 가격항 제외. 실측 병리: incumbent(가격-레벨)와
            # leader 후보(budget)가 스텝마다 교대 커밋되며 레짐 플래핑(Σmeter TV 1.74×)
            # + 과소방류(평균 −800) 유발. dual-standing(λ_UF)은 windup 수정이라 보존.
            if self.metering_price_split and leader is None and not dual_standing:
                return 0.0
            total_price = 0.0
            for ramp in owned_ramps:
                g_ext = self.metering_marginal_price.get(ramp)
                if g_ext is None:
                    continue
                ref = float(self.metering_marginal_price_ref.get(
                    ramp, float(snapshot.ramp_metering.get(ramp, caps[ramp]))
                ))
                total_price += (
                    self.metering_marginal_price_weight
                    * float(g_ext)
                    * (float(meter.get(ramp, ref)) - ref)
                )
            if priced_metering:
                total_meter = sum(float(meter.get(r, 0.0)) for r in owned_ramps)
                if nuf_mode_priced == "dual":
                    # dual: λ_UF·Σmeter — G1DF dual 분기와 동일 신호(수량 target 추적).
                    total_price += float(self._lambda_UF) * total_meter
                else:
                    # 가격 모드에선 leader budget을 hard로 강제하지 않고 soft anchor로만
                    # 남긴다(|Σ−budget| 페널티, w=T_c_h — Codex f18e920 설계 유지).
                    total_price += self.metering_budget_penalty_weight * abs(
                        total_meter - budget_price
                    )
            return float(total_price)

        # ---- B3 가격 모드 leader 분기: soft budget + 자율 후보 sweep ----
        # metering 가격이 하달돼 있으면 hard budget/cap 분기 대신 자율 좌표하강(PFO와
        # 동일 후보 + 경계값 {0, budget}판)을 돌리고, 비용에 가격항+soft budget을 더한다.
        # 가격이 방류 수준을 유도하고 budget은 anchor — 절벽 과방류는 B4 barrier 가격이
        # 담당한다(1차 TTT 가격 단독은 음성 판정, 2026-07-04 §3).
        if priced_metering:
            best_meter = {
                r: float(np.clip(snapshot.ramp_metering.get(r, caps[r]), 0.0, caps[r]))
                for r in owned_ramps
            }
            if _bb_on:
                # snapshot(반복 내부값)이 박스 밖일 수 있어 초기점도 박스로 clip.
                best_meter = {r: min(max(best_meter[r], _bb[r][0]), _bb[r][1]) for r in owned_ramps}
            best_vsl, best_cost, e0 = _solve_with(best_meter)
            best_cost += _price_metering_cost(best_meter)
            evals_total += e0
            for ramp in owned_ramps:
                local_best = best_meter[ramp]
                values = {0.0, local_best, min(caps[ramp], budget_price)}
                values.update(
                    float(frac) * caps[ramp] for frac in self.ramp_metering_fractions
                )
                if _bb_on:
                    # BASELINE-BOX: 박스 밖 후보 제거 + 박스 점 추가(격자 흡수 방지).
                    values = ({v for v in values if _bb_ok(ramp, v)}
                              | set(_bb_cands[ramp]) | {local_best})
                # B3CERT(비대칭 안전 증명서): 미인증 ramp는 방류 증가(> ref) 후보 제외.
                # 조임 방향은 가역이라 항상 허용 — trust 이동성 보장보다 우선한다.
                cert_ok = True
                if (
                    self.metering_release_certified is not None
                    and ramp in (self.metering_marginal_price or {})
                ):
                    cert_ok = bool(self.metering_release_certified.get(ramp, False))
                    if not cert_ok:
                        ref_c = float(self.metering_marginal_price_ref.get(ramp, local_best))
                        kept = {
                            v for v in values
                            if float(np.clip(v, 0.0, caps[ramp])) <= ref_c + 1.0e-9
                        }
                        if kept:
                            values = kept
                # B3TR trust region: 가격이 측정된 이웃(|v − ref| ≤ frac·cap) 밖 후보 제외.
                # **이동성 보장**: 반경이 격자 간격보다 좁아도 ref의 최근접 아래/위 후보는
                # 포함한다(위쪽은 인증 시에만) — 반경 0.25·cap < 첫 분율 간격 0.3·cap로
                # metering이 cap에 동결돼 붕괴한 사고(2026-07-05 §11 v1)의 재발 방지.
                if (
                    self.metering_marginal_price_trust_frac is not None
                    and ramp in (self.metering_marginal_price or {})
                ):
                    ref_r = float(self.metering_marginal_price_ref.get(ramp, local_best))
                    trust_r = float(self.metering_marginal_price_trust_frac) * caps[ramp]
                    clipped_vals = sorted(
                        {float(np.clip(v, 0.0, caps[ramp])) for v in values}
                    )
                    trusted = {
                        v for v in clipped_vals if abs(v - ref_r) <= trust_r + 1.0e-9
                    }
                    below = [v for v in clipped_vals if v < ref_r - 1.0e-9]
                    above = [v for v in clipped_vals if v > ref_r + 1.0e-9]
                    if below:
                        trusted.add(below[-1])
                    if above and cert_ok:
                        trusted.add(above[0])
                    if trusted:
                        values = trusted
                for cand_val in sorted(
                    float(np.clip(v, 0.0, caps[ramp])) for v in values
                ):
                    if abs(cand_val - best_meter[ramp]) <= 1.0e-9:
                        continue
                    trial = dict(best_meter)
                    trial[ramp] = cand_val
                    vsl_dict, cost, e = _solve_with(trial)
                    cost += _price_metering_cost(trial)
                    evals_total += e
                    if cost < best_cost:
                        best_cost, best_vsl, local_best = cost, vsl_dict, cand_val
                best_meter[ramp] = local_best
            return best_vsl, best_meter, evals_total

        # ---- leader 분기: N_UF를 hard BUDGET으로 처리(simplex allocation) ----
        # leader가 N_UF_star(총 urban→freeway 교환유량 target, veh/h)를 주면, 이 link의 몫
        # B = ω_F[link]·N_UF_star로 소유 ramp metering 합을 **고정**한다(soft penalty 제거).
        # follower는 B를 소유 ramp들에 어떻게 배분(allocation)할지만 탐색해 per-link own-TTS를
        # 최소화한다. 2 ramp이면 1-D 탐색: meter_R1 ∈ [max(0,B−cap2), min(cap1,B)], R2 = B−R1.
        # 이 분기는 (a) leader의 N_UF를 정확히 실현하고, (b) full grid보다 훨씬 작은 탐색이며,
        # (c) 이전 soft penalty hack을 없앤다.
        # DUAL-STANDING: dual 모드는 leader 부재(incumbent probe)에도 λ≠0이면 진입 —
        # incumbent가 λ에 반응해야 적분 루프가 닫힌다(windup 수정, 위 priced 주석 참조).
        # BUDGET-OFF(2026-07-16 A/B, 사용자 제안): "예산 없이 가격만" arm.
        # 이 분기가 leader의 N_UF를 **hard 제약**으로 강제한다(follower는 배분만 탐색).
        # budget_off=True면 이 분기를 건너뛰고 **PFO autonomous 좌표하강**(위 분기)을 탄다 —
        # de-facto metering이 own-TTS 최소화에서 창발하고, leader는 **가격만** 넘긴다.
        # 남는 채널: green/metering/vsl/offset marginal price + λ_UF(가격이므로 유지).
        # 빠지는 것: N_UF hard budget(ω_F·N_UF_star). 기본 False=비트동일.
        # 목적: +4.78%(③ vs PFO)가 **예산 몫인지 가격 몫인지** 분해 — 사다리에 없던 rung.
        _budget_off = bool(getattr(self.cfg.mpc, "leader_budget_off", False))
        if owned_ramps and not _budget_off and (
            (leader is not None and n_uf_star > 0.0) or dual_standing
        ):
            omega_f = float(self._wu._omega_f.get(link, 0.0))
            cap_sum = sum(caps[r] for r in owned_ramps)
            # link budget을 가용 영역 [0, Σcap]으로 clamp(나머지는 follower가 분배).
            budget = float(np.clip(omega_f * n_uf_star, 0.0, cap_sum))

            # ---- N_UF cap 모드(2026-07-03): budget을 등식이 아니라 상한으로 처리 ----
            # 진단: PFO(P1)의 자율 metering이 똑똑해졌는데 등식 budget이 이를 덮어써
            # standalone 악화(sweet_190 7200s 14711.8)·bal_med N_UF 폭주를 만들었다.
            # cap 모드는 자율 좌표하강(PFO 분기와 동일 후보)을 그대로 돌리되 모든 후보를
            # link 합 ≤ budget으로 비례 투영한다: 자율 최적이 budget 미만이면 leader가
            # 건드리지 않고(자율 존중), 초과할 때만 boundary(합=budget)로 눌린다.
            # N_P cap(wu_faithful_np_coordination_mode)과 대칭. 기본 equality(기존 거동).
            nuf_mode = nuf_mode_priced
            if nuf_mode == "dual":
                # N_UF dual: 자율 좌표하강(등식 강제 없음) + 비용에 λ_UF·Σowned_meter 추가.
                # λ_UF>0면 방류 억제(Σmeter가 target 초과), <0면 방류 보상(target 미달).
                lam_uf = float(self._lambda_UF)

                def _price_uf(meter: Mapping[str, float]) -> float:
                    return lam_uf * sum(float(meter.get(r, 0.0)) for r in owned_ramps)

                best_meter = {
                    r: float(np.clip(snapshot.ramp_metering.get(r, caps[r]), 0.0, caps[r]))
                    for r in owned_ramps
                }
                if _bb_on:
                    best_meter = {r: min(max(best_meter[r], _bb[r][0]), _bb[r][1]) for r in owned_ramps}
                best_vsl, best_cost, e0 = _solve_with(best_meter)
                best_cost += _price_uf(best_meter)
                evals_total += e0
                for ramp in owned_ramps:
                    local_best = best_meter[ramp]
                    _cand_vals = (_bb_cands[ramp] if _bb_on else
                                  [f * caps[ramp] for f in self.ramp_metering_fractions])
                    for cand_val in _cand_vals:
                        if abs(cand_val - best_meter[ramp]) <= 1.0e-9:
                            continue
                        trial = dict(best_meter)
                        trial[ramp] = cand_val
                        vsl_dict, cost, e = _solve_with(trial)
                        cost += _price_uf(trial)
                        evals_total += e
                        if cost < best_cost:
                            best_cost, best_vsl, local_best = cost, vsl_dict, cand_val
                    best_meter[ramp] = local_best
                return best_vsl, best_meter, evals_total

            if nuf_mode == "cap":
                def _project_to_cap(meter: Mapping[str, float]) -> Dict[str, float]:
                    clipped = {
                        r: float(np.clip(float(meter.get(r, caps[r])), 0.0, caps[r]))
                        for r in owned_ramps
                    }
                    total_m = sum(clipped.values())
                    if total_m > budget + 1.0e-9 and total_m > 0.0:
                        scale = budget / total_m
                        clipped = {r: v * scale for r, v in clipped.items()}
                    return clipped

                best_meter = _project_to_cap({
                    r: float(snapshot.ramp_metering.get(r, caps[r]))
                    for r in owned_ramps
                })
                best_vsl, best_cost, e0 = _solve_with(best_meter)
                evals_total += e0
                for ramp in owned_ramps:
                    _cand_vals2 = (_bb_cands[ramp] if _bb_on else
                                   [f * caps[ramp] for f in self.ramp_metering_fractions])
                    for _cv in _cand_vals2:
                        trial = dict(best_meter)
                        trial[ramp] = _cv
                        trial = _project_to_cap(trial)
                        if all(
                            abs(trial[r] - best_meter[r]) <= 1.0e-9
                            for r in owned_ramps
                        ):
                            continue
                        if _bb_on and any(not _bb_ok(r, trial[r]) for r in owned_ramps):
                            continue  # BASELINE-BOX: 사영 후에도 박스 준수 필요.
                        vsl_dict, cost, e = _solve_with(trial)
                        evals_total += e
                        if cost < best_cost:
                            best_cost, best_vsl, best_meter = cost, vsl_dict, dict(trial)
                return best_vsl, best_meter, evals_total

            best_meter: Dict[str, float] = {}
            best_vsl: Dict[str, float] = {}
            best_cost = float("inf")

            if len(owned_ramps) == 1:
                # ramp 1개면 배분 자유도가 없다: meter = budget(clamped). 가격은 상수라 무영향.
                r0 = owned_ramps[0]
                meter = {r0: float(np.clip(budget, 0.0, caps[r0]))}
                best_vsl, best_cost, e = _solve_with(meter)
                best_meter = meter
                evals_total += e
            else:
                # 2 ramp simplex: meter_R1 ∈ [lo, hi], meter_R2 = budget − meter_R1.
                r1, r2 = owned_ramps[0], owned_ramps[1]
                cap1, cap2 = caps[r1], caps[r2]
                lo = max(0.0, budget - cap2)
                hi = min(cap1, budget)
                # 분배 격자 7점(끝점 포함). 끝점 동일하면 1점.
                if hi - lo <= 1.0e-9:
                    splits = [lo]
                else:
                    splits = [float(v) for v in np.linspace(lo, hi, 7)]
                for m1 in splits:
                    m1c = float(np.clip(m1, 0.0, cap1))
                    m2c = float(np.clip(budget - m1c, 0.0, cap2))
                    meter = {r1: m1c, r2: m2c}
                    vsl_dict, cost, e = _solve_with(meter)
                    # SPLIT-PRICE: own_TTS + Σ g_ext·(x−ref)로 배분 랭킹. Σ=budget 고정이라
                    # 가격 공통성분은 후보 간 상수(자동 소거) — ramp 간 차이만 작동.
                    # priced_metering=False이므로 anchor/λ 항은 없음(순수 선형 가격만).
                    if price_split:
                        cost += _price_metering_cost(meter)
                    evals_total += e
                    if cost < best_cost:
                        best_cost, best_vsl, best_meter = cost, vsl_dict, dict(meter)

            return best_vsl, best_meter, evals_total

        # ---- leader=None(PFO) 분기: 기존 autonomous per-ramp metering 좌표하강 ----
        # 현재 best metering(절대 veh/h). 초기값 = capacity(=metering off, snapshot 기본).
        # B3 가격이 설정돼 있으면(P-Stack 내부 incumbent solve) 가격항이 더해진다 —
        # 순수 PFO 러너는 가격 dict가 None이라 비용 0(기존 거동 비트 동일).
        best_meter = {
            r: float(snapshot.ramp_metering.get(r, caps[r])) for r in owned_ramps
        }
        if _bb_on:
            best_meter = {r: min(max(best_meter[r], _bb[r][0]), _bb[r][1]) for r in owned_ramps}
        # 초기 best 비용·VSL(현재 metering에서).
        best_vsl, best_cost, e0 = _solve_with(best_meter)
        best_cost += _price_metering_cost(best_meter)
        evals_total += e0

        # ramp별 좌표하강: 각 ramp의 5개 분율을 훑어 own-TTS 최저 분율로 갱신.
        for ramp in owned_ramps:
            local_best_meter = best_meter[ramp]
            _cand_vals3 = (_bb_cands[ramp] if _bb_on else
                           [f * caps[ramp] for f in self.ramp_metering_fractions])
            for cand_val in _cand_vals3:
                if abs(cand_val - best_meter[ramp]) <= 1.0e-9:
                    continue  # 이미 평가된 현재값.

                # B3CERT: 미인증 ramp의 방류 증가 후보는 스킵(비대칭, trust보다 우선).
                if (
                    self.metering_marginal_price is not None
                    and self.metering_release_certified is not None
                    and ramp in self.metering_marginal_price
                    and not self.metering_release_certified.get(ramp, False)
                ):
                    ref_c = float(self.metering_marginal_price_ref.get(ramp, cand_val))
                    if cand_val > ref_c + 1.0e-9:
                        continue
                # B3TR trust: 가격 활성 시(P-Stack 내부 incumbent 포함) 측정 이웃 밖
                # 후보는 건너뜀 — 단, ref의 최근접 아래/위 분율은 이동성 보장을 위해
                # 항상 허용(반경<격자 간격일 때 동결 방지, priced 분기와 동일 규칙).
                if (
                    self.metering_marginal_price is not None
                    and self.metering_marginal_price_trust_frac is not None
                    and ramp in self.metering_marginal_price
                ):
                    ref_r = float(self.metering_marginal_price_ref.get(ramp, cand_val))
                    trust_r = float(self.metering_marginal_price_trust_frac) * caps[ramp]
                    if abs(cand_val - ref_r) > trust_r + 1.0e-9:
                        lattice = sorted(f * caps[ramp] for f in self.ramp_metering_fractions)
                        below = [v for v in lattice if v < ref_r - 1.0e-9]
                        above = [v for v in lattice if v > ref_r + 1.0e-9]
                        nearest = set()
                        if below:
                            nearest.add(round(below[-1], 9))
                        if above:
                            nearest.add(round(above[0], 9))
                        if round(cand_val, 9) not in nearest:
                            continue
                trial = dict(best_meter)
                trial[ramp] = cand_val
                vsl_dict, cost, e = _solve_with(trial)
                cost += _price_metering_cost(trial)
                evals_total += e
                if cost < best_cost:
                    best_cost, best_vsl, local_best_meter = cost, vsl_dict, cand_val
            best_meter[ramp] = local_best_meter

        return best_vsl, best_meter, evals_total

    # ---------- Jacobi 합의 루프 (Wu §IV-D) ----------

    def _frozen_s_eff(self, state: TrafficState) -> Dict[str, float]:
        """모든 urban 링크의 S_eff 스냅샷(이웃 downstream 동결값)."""
        s_eff: Dict[str, float] = {}
        for link in self.cfg.network.urban_link_storage_veh:
            s_eff[link] = float(_effective_available_space(state, self.cfg, link))
        return s_eff

    def _sum_nin_at_lambda(
        self,
        lambda_p: float,
        state: TrafficState,
        coupling: Mapping[str, float],
        s_eff_frozen: Mapping[str, float],
        reservoir_drain: Mapping[str, float],
        freeway_congestion: Mapping[str, float],
        snapshot: ControlAction,
        leader: Optional[object],
        forecast_arrivals: Mapping[str, float],
        horizon_h: float,
        demand: DemandStep,
    ) -> tuple[float, int]:
        """주어진 가격 λ에서 모든 urban agent best-response의 Σ_i nin_i[veh]과 평가수.

        gain 측정·진단 공용. snapshot(동결 결합·이웃)을 고정해 λ만 바꾼 효과를 본다."""
        total_nin = 0.0
        e_total = 0
        for signal in self.cfg.network.signals:
            arr_movement = self._per_movement_arrivals(signal, state, snapshot, demand)
            _, _, e, nin_i = self._solve_urban_agent_local(
                signal, state, coupling, arr_movement, s_eff_frozen,
                reservoir_drain, freeway_congestion, snapshot, leader,
                lambda_p, forecast_arrivals, horizon_h, demand,
            )
            total_nin += nin_i
            e_total += e
        return total_nin, e_total

    def _lambda_np_update(self, lambda_p: float, sum_nin: float, target: float) -> float:
        """λ의 control step 간 적분 갱신(A1+A2, 2026-07-02 진단 기반).

        λ_next = clip(λ + lambda_np_step_gain·(Σnin − target), 0, lambda_np_cap).
        방향: Σnin > target(유입 과다) → λ 증가(억제 강화). max(0,·)이 A1(음수 λ 금지)을
        강제한다 — target > Σnin이면 λ가 0으로 내려가 자연 유입만 허용하고, 유입을 강제로
        늘리는 보상은 없다. cap은 진단상 Σnin 바닥 도달 λ≈10 기준."""
        return min(
            max(0.0, lambda_p + self.lambda_np_step_gain * (sum_nin - target)),
            self.lambda_np_cap,
        )

    def _np_feasible_range(
        self,
        state: TrafficState,
        coupling: Mapping[str, float],
        snapshot: ControlAction,
        forecast_arrivals: Mapping[str, float],
        horizon_h: float,
    ) -> tuple[float, float]:
        # 리더 N_P target을 follower가 만들 수 있는 Σnin 범위로 투영 — 단위 [veh/horizon].
        # solver가 실제로 보는 동일 green 후보집합(_urban_green_candidates)에서 신호별 nin의
        # min/max를 합산해 (sigma_min, sigma_max)를 구한다. _agent_net_inflow_veh와 같은 정의·단위.
        sigma_min = 0.0
        sigma_max = 0.0
        for signal in self.cfg.network.signals:
            candidates = self._urban_green_candidates(signal, state, coupling, snapshot)
            nin_vals = [
                self._predicted_agent_net_inflow_veh(
                    signal, p1, state, forecast_arrivals, horizon_h
                )
                for p1 in candidates
            ]
            if not nin_vals:
                continue
            sigma_min += min(nin_vals)
            sigma_max += max(nin_vals)
        return float(sigma_min), float(sigma_max)

    def _predicted_agent_net_inflow_veh(
        self,
        signal: str,
        green_p1: float,
        state: TrafficState,
        forecast_arrivals: Mapping[str, float],
        horizon_h: float,
    ) -> float:
        """predictor 모드(cfg.mpc.wu_faithful_np_predictor_mode)를 반영한 nin_i[veh] 예측.

        "legacy"(기본)는 기존 `_agent_net_inflow_veh` 그대로 위임한다 — 기본 경로 거동 불변.
        나머지 모드는 2026-06-30 리포트의 원인분리용 진단 스위치를 복원한 것이다.
          storage_aware   — served 유입을 receiving 링크 가용 저장공간(state.urban_link_storage,
                            available-space 저장)으로 추가 상한(막힌 수용공간이 예측을 깎는다).
          current_interval — horizon 전체 forecast 도착을 즉시 처리 가능량으로 보지 않고
                            현 control interval 몫(1/steps)만 available에 넣는다.
          phase_substep   — cycle 평균 green 비율 대신 `_phase_green_fraction`의 substep
                            green window 겹침으로 용량을 적분한다.
        """
        mode = str(self.cfg.mpc.wu_faithful_np_predictor_mode)
        if mode == "legacy":
            return self._agent_net_inflow_veh(
                signal, green_p1, state, forecast_arrivals, horizon_h
            )
        net = self.cfg.network
        sim = self.cfg.simulation
        model = self._local_models[signal]
        total = net.effective_green_total
        cycle = max(net.cycle_length, 1.0e-9)
        green = {"p1": float(green_p1), "p2": float(total - green_p1)}
        steps = max(1, int(round(horizon_h / max(sim.T_c_h, 1.0e-9))))
        arr_scale = (1.0 / steps) if mode == "current_interval" else 1.0
        # phase_substep: 후보 green으로 만든 임시 control의 substep green window로 용량 적분.
        phased_cap: Dict[str, float] = {}
        if mode == "phase_substep":
            probe = ControlAction(
                ramp_metering={},
                vsl={},
                green_times={f"{signal}_p1": green["p1"], f"{signal}_p2": green["p2"]},
                offsets={},
                inflow_outflow_allocation={},
            )
            start_idx = _urban_step_index(state, self.cfg)
            substeps = max(1, int(round(horizon_h / max(sim.T_u_h, 1.0e-9))))
            for movement in model.movements:
                spec = self._specs[movement]
                cap = 0.0
                for k in range(substeps):
                    gf = _phase_green_fraction(probe, self.cfg, spec, start_idx + k)
                    cap += sim.T_u_h * gf * model.cap_flow_of[movement]
                phased_cap[movement] = cap
        served: Dict[str, float] = {}
        raw_onramp_by_ramp: Dict[str, float] = {}
        onramp_by_movement = {
            m: r for r, mvs in net.on_ramp_to_movement.items() for m in mvs
        }
        for movement in model.movements:
            spec = self._specs[movement]
            kind = model.kind_of[movement]
            available = max(0.0, float(state.urban_movement_queue.get(movement, 0.0)))
            if kind == "off_ramp":
                off_ramp = str(spec.get("off_ramp", ""))
                inflow = self._frozen_offramp_inflow(off_ramp, state)
                available += inflow * horizon_h * float(spec.get("beta", 0.0)) * arr_scale
            else:
                available += max(0.0, float(forecast_arrivals.get(movement, 0.0))) * arr_scale
            if mode == "phase_substep":
                cap_veh = phased_cap.get(movement, 0.0)
            else:
                green_fraction = green[model.phase_of[movement]] / cycle
                cap_veh = horizon_h * green_fraction * model.cap_flow_of[movement]
            s = min(available, max(0.0, cap_veh))
            served[movement] = s
            if kind == "on_ramp":
                ramp = onramp_by_movement.get(movement, "")
                if ramp:
                    raw_onramp_by_ramp[ramp] = raw_onramp_by_ramp.get(ramp, 0.0) + s
        # on_ramp 서비스는 ramp reservoir 여유로 추가 스케일(_agent_net_inflow_veh와 동일).
        for ramp, raw_total in raw_onramp_by_ramp.items():
            if raw_total <= 1.0e-9:
                continue
            ramp_space = max(
                0.0,
                float(net.ramp_queue_max_veh) - max(0.0, float(state.ramp_queue.get(ramp, 0.0))),
            )
            scale = min(1.0, ramp_space / raw_total)
            for movement in net.on_ramp_to_movement.get(ramp, []):
                if movement in served:
                    served[movement] *= scale
        if mode == "storage_aware":
            # receiving 링크별 served 합을 가용 저장공간으로 상한(비율 축소).
            recv_totals: Dict[str, float] = {}
            for movement in model.movements:
                recv = model.receiving_of[movement]
                if recv and recv in net.urban_link_storage_veh:
                    recv_totals[recv] = recv_totals.get(recv, 0.0) + served[movement]
            for recv, total_in in recv_totals.items():
                if total_in <= 1.0e-9:
                    continue
                space = max(
                    0.0,
                    float(state.urban_link_storage.get(recv, net.urban_link_storage_veh[recv])),
                )
                scale = min(1.0, space / total_in)
                if scale < 1.0:
                    for movement in model.movements:
                        if model.receiving_of[movement] == recv:
                            served[movement] *= scale
        inflow_veh = sum(
            served[m] for m in model.movements if model.kind_of[m] in _INFLOW_KINDS
        )
        outflow_veh = sum(
            served[m] for m in model.movements if model.kind_of[m] in _OUTFLOW_KINDS
        )
        return float(inflow_veh - outflow_veh)

    def _np_feasible_sum_range(
        self,
        state: TrafficState,
        forecast_arrivals: Mapping[str, float],
        horizon_h: float,
    ) -> tuple[float, float, Dict[str, float]]:
        """coupling 없이 state·도착 예보만으로 follower Σnin 실현가능범위[veh]를 예측한다.

        후보 green 집합은 solver와 동일한 `_urban_green_candidates`(중립 coupling=0,
        snapshot=fixed)로 만들고, per-signal nin은 predictor 모드
        (`_predicted_agent_net_inflow_veh`)로 평가해 min/max를 합산한다.
        반환 (sigma_min, sigma_max, diag) — diag 값은 전부 float."""
        snapshot = ControlAction.fixed(self.cfg)
        coupling: Dict[str, float] = {}
        sigma_min = 0.0
        sigma_max = 0.0
        for signal in self.cfg.network.signals:
            candidates = self._urban_green_candidates(signal, state, coupling, snapshot)
            nin_vals = [
                self._predicted_agent_net_inflow_veh(
                    signal, p1, state, forecast_arrivals, horizon_h
                )
                for p1 in candidates
            ]
            if not nin_vals:
                continue
            sigma_min += min(nin_vals)
            sigma_max += max(nin_vals)
        mode = str(self.cfg.mpc.wu_faithful_np_predictor_mode)
        diag = {
            "np_feasible_sum_sigma_min_veh": float(sigma_min),
            "np_feasible_sum_sigma_max_veh": float(sigma_max),
            "np_feasible_sum_horizon_h": float(horizon_h),
            "np_feasible_sum_predictor_mode_code": float(
                _NP_PREDICTOR_MODE_CODES.get(mode, 0.0)
            ),
        }
        return float(sigma_min), float(sigma_max), diag

    def leader_np_feasible_range(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
    ) -> tuple[float, float, Dict[str, float]]:
        """Stackelberg leader 후보 N_P 투영용 Σnin feasible range[veh over horizon].

        `stackelberg_mpc._project_action_to_follower_feasible_np`(1598행)가 duck-typing으로
        호출하는 인터페이스. `_solve_followers` 앞부분과 동일하게 coupling(warm-start 반영)·
        snapshot·forecast_arrivals·horizon_h를 구성한 뒤 `_np_feasible_range`를 감싼다.
        반환 (sigma_min, sigma_max, diag) — diag 키는 호출측에서 f"leader_{key}"로 승격돼
        float(value)로 변환되므로 float 호환 값만 담는다."""
        fc = list(forecast) if forecast else []
        if not fc:
            raise ValueError("leader_np_feasible_range requires at least one demand step.")
        demand = fc[0]
        control = ControlAction.uncontrolled(self.cfg)
        control.green_times = dict(previous.green_times)
        control.vsl = dict(previous.vsl)
        control.inflow_outflow_allocation = {}
        coupling = self._wu._coupling(state, control, demand)
        if self._prev_coupling is not None:
            for k in coupling:
                if k in self._prev_coupling:
                    coupling[k] = float(self._prev_coupling[k])
        snapshot = ControlAction(
            ramp_metering=dict(control.ramp_metering),
            vsl=dict(control.vsl),
            green_times=dict(control.green_times),
            offsets=dict(control.offsets),
            inflow_outflow_allocation={},
        )
        steps_h = fc[: max(1, self.cfg.mpc.horizon_steps)] or fc[:1]
        horizon_h = max(self.cfg.simulation.T_c_h * max(len(steps_h), 1), 1.0e-9)
        forecast_arrivals = self._movement_forecast_arrivals_veh(fc)
        sigma_min, sigma_max = self._np_feasible_range(
            state, coupling, snapshot, forecast_arrivals, horizon_h,
        )
        mode = str(self.cfg.mpc.wu_faithful_np_predictor_mode)
        diag = {
            "np_feasible_sigma_min_veh": float(sigma_min),
            "np_feasible_sigma_max_veh": float(sigma_max),
            "np_feasible_horizon_h": float(horizon_h),
            "np_feasible_predictor_mode_code": float(
                _NP_PREDICTOR_MODE_CODES.get(mode, 0.0)
            ),
        }
        return float(sigma_min), float(sigma_max), diag

    def _measure_dual_gain(
        self,
        state: TrafficState,
        coupling: Mapping[str, float],
        s_eff_frozen: Mapping[str, float],
        reservoir_drain: Mapping[str, float],
        freeway_congestion: Mapping[str, float],
        control: ControlAction,
        leader: Optional[object],
        forecast_arrivals: Mapping[str, float],
        horizon_h: float,
        demand: DemandStep,
    ) -> tuple[float, int]:
        """듀얼 gain G = |dΣnin/dλ|[veh per (veh·h/veh)] 측정 — subgradient 스텝 자기보정용.

        nin은 λ에 대해 **단조 비증가**(가격↑→inflow 억제)이고 green 격자상 조각선형이라 미분이
        조각상수다. λ를 0에서 λ_hi까지 스캔해 Σnin이 떨어지는 전체 기울기의 평균 크기를 G로 쓴다:
          G ≈ (Σnin(0) − Σnin(λ_hi)) / λ_hi.
        λ_hi는 스케일 인지 상한: 가격이 비용 스케일(자기 TTS 비용 스프레드)을 넘어 nin 항이 green
        선택을 지배하기 시작하는 값 근처. cost_norm·movement_cap(veh·h)와 nin 스케일(veh)의 비로
        잡는다 → λ_hi ≈ (자기 비용 스케일)/(nin 스케일). G가 0(전 구간 둔감)이면 0 반환→스텝 보류.
        평가수는 격자점수×agent수."""
        net = self.cfg.network
        substeps = max(1, self.cfg.mpc.horizon_steps) * max(1, self.cfg.simulation.K_cu)
        cost_norm = max(1.0e-9, float(substeps) * self.cfg.simulation.T_u_h)
        # 자기 TTS 비용 스케일 ≈ 가능 누적(veh)×cost_norm. movement_cap·horizon으로 상한 잡는다.
        cost_scale = cost_norm * float(net.movement_capacity_veh_h) * max(horizon_h, 1.0e-9)
        nin_scale = max(horizon_h * float(net.movement_capacity_veh_h), 1.0)
        # λ가 이 정도면 λ·nin이 비용 스프레드를 덮어 green 선택을 지배한다(스케일 인지).
        lambda_hi = max(cost_scale / (nin_scale * nin_scale), 1.0e-6)
        snapshot = ControlAction(
            ramp_metering=dict(control.ramp_metering),
            vsl=dict(control.vsl),
            green_times=dict(control.green_times),
            offsets=dict(control.offsets),
            inflow_outflow_allocation={},
        )
        nin_lo, e0 = self._sum_nin_at_lambda(
            0.0, state, coupling, s_eff_frozen, reservoir_drain, freeway_congestion,
            snapshot, leader, forecast_arrivals, horizon_h, demand,
        )
        nin_hi, e1 = self._sum_nin_at_lambda(
            lambda_hi, state, coupling, s_eff_frozen, reservoir_drain, freeway_congestion,
            snapshot, leader, forecast_arrivals, horizon_h, demand,
        )
        gain = max(0.0, (nin_lo - nin_hi)) / max(lambda_hi, 1.0e-12)
        return float(gain), int(e0 + e1)

    def _solve_followers(
        self,
        state: TrafficState,
        demand: DemandStep,
        previous: ControlAction,
        leader: Optional[object] = None,
        forecast: Optional[List[DemandStep]] = None,
    ) -> tuple[ControlAction, int, bool, float, int]:
        net = self.cfg.network
        # METER-BOX 가드: SEG13 경로 전용 플래그가 비-SEG13 구성에 꽂히면 침묵 무효
        # (BUDGET_OFF 2026-07-16, 20런 무효)가 재발한다 — 런을 살리지 말고 즉사시킨다.
        if getattr(self.cfg.mpc, "seg13_meter_box_veh_h", None) is not None and not (
            self.segment_agents and self.metering_enabled
        ):
            raise RuntimeError(
                "METER_BOX는 SEG13(segment_agents+metering) 전용인데 해당 경로가 꺼져 "
                "있다 — 비-SEG13은 metering_marginal_price_trust_frac이 이미 묶는다."
            )
        if getattr(self.cfg.mpc, "seg13_vsl_box_kmh", None) is not None and not self.segment_agents:
            raise RuntimeError(
                "VSL_BOX는 SEG13 segment agent 경로 전용인데 segment_agents가 꺼져 있다."
            )
        self._wu._repair_diagnostics = {}
        self._seg13_diag = {}
        self._seg_traj = {}  # 궤적 교환은 solve 단위 — 후보 간 오염 방지.
        control = ControlAction.uncontrolled(self.cfg)
        control.green_times = dict(previous.green_times)
        control.vsl = dict(previous.vsl)
        control.inflow_outflow_allocation = {}
        # warm-start 결합변수(직전 step 수렴값) 우선, 없으면 현재 control 기준 계산.
        coupling = self._wu._coupling(state, control, demand)
        if self._prev_coupling is not None:
            for k in coupling:
                if k in self._prev_coupling:
                    coupling[k] = float(self._prev_coupling[k])
        # 이웃 downstream S_eff 동결 스냅샷(한 step 내 고정).
        s_eff_frozen = self._frozen_s_eff(state)
        # freeway가 reservoir(w_r)를 비우는 frozen 방출률(ρ_merge 기반) — 한 step 내 고정.
        reservoir_drain = self._frozen_reservoir_drain(state, control, demand)
        # ramp별 frozen freeway 혼잡 가중(de facto ramp metering) — 한 step 내 고정.
        freeway_congestion = self._frozen_freeway_congestion(state)

        # ---------- 듀얼 분해 N_P 추적 셋업(leader present + use_dual_np) ----------
        # forecast horizon 도착량·horizon_h를 리더와 동일하게 구해 nin_i가 N_P_star와 비교 가능하게.
        fc = forecast if forecast else [demand]
        steps_h = fc[: max(1, self.cfg.mpc.horizon_steps)] or fc[:1]
        horizon_h = max(self.cfg.simulation.T_c_h * max(len(steps_h), 1), 1.0e-9)
        forecast_arrivals = self._movement_forecast_arrivals_veh(fc)
        # ---- P1.5 auto 게이트(신호별, step당 1회): 포화도 band 안의 ramp 신호만 활성 ----
        # 항상 reset(이전 step 잔존 방지). auto=False면 빈 집합 그대로 → 기존 거동 비트 동일.
        self._phase_resolved_active_signals = set()
        if self.ramp_aware_phase_auto:
            lo_band, hi_band = self.ramp_aware_phase_auto_band
            ramp_signals = [
                s for s in net.signals if self._local_models[s].has_ramps
            ]
            all_in_band = bool(ramp_signals)
            for signal in ramp_signals:
                x = self._ramp_phase_saturation(signal, state, coupling, control, horizon_h)
                control.diagnostics[f"wu_p15_sat_{signal}"] = float(x)
                if not (lo_band <= x < hi_band):
                    all_in_band = False
            # AND-게이트: 모든 ramp 신호가 band 안일 때만 전체 활성(망 단위 중부하 판정).
            if all_in_band:
                self._phase_resolved_active_signals = set(ramp_signals)
            control.diagnostics["wu_p15_auto_active_count"] = float(
                len(self._phase_resolved_active_signals)
            )
        dual_active = leader is not None and self.use_dual_np and self.np_price_enabled
        n_p_star = float(getattr(leader, "N_P_star", 0.0)) if leader is not None else 0.0
        np_cand_flag = bool(getattr(self.cfg.mpc, "np_candidate_lambda", False))
        # 방법 A(2026-07-13): candidate 내부 primal-dual 반복 K. K>0이면 λ̂ 1회 선반영
        # 대신 후보별 (λ, green) 안장점 반복으로 대체(np_cand_flag가 마스터 스위치).
        np_pd_iters = int(getattr(self.cfg.mpc, "np_primal_dual_iters", 0))
        np_pd_active = dual_active and np_cand_flag and np_pd_iters > 0
        # ---- (51) corrector(원고 정식화): λ_{k+1} = Π[λ_k + γ_c(Q^real − Ñ_{c*})] ----
        # 커밋 시점이 아니라 다음 step 시작 시(실현 유입 관측 후) standing λ를 1회 교정.
        # Q^real은 보호구역 accumulation 차분(구간)을 horizon 배수로 환산해 측정한다
        # (Σnin·target이 horizon 집계 veh 단위이므로). 후보 solve 간에는 state 시각
        # 가드로 스텝당 1회만 실행된다.
        # 방법 A(K>0)에서는 corrector를 통째로 생략한다 — λ는 스텝마다 warm start에서
        # 후보별로 재유도되고, 실현-공간 교정(Q^real) 채널 자체를 쓰지 않는다.
        if np_pd_active:
            self._np_corrector_pending = None
        if dual_active and np_cand_flag and not np_pd_active:
            now_t = float(getattr(state, "time_sec", 0.0))
            if self._np_step_time != now_t:
                self._np_step_time = now_t
                n_p_now_veh = float(state.protected_accumulation_veh(net))
                if self._np_prev_accum is not None:
                    self._np_last_real_q = (
                        n_p_now_veh - float(self._np_prev_accum)
                    ) * float(max(1, self.cfg.mpc.horizon_steps))
                self._np_prev_accum = n_p_now_veh
                # r̂ 갱신: 직전 commit 균형의 예측 Σnin(_np_last_sum_nin, commit 시 기록)과
                # 방금 측정한 실현 유입의 쌍으로 EWMA(β=0.3). 음수 실현·0 나눗셈 가드로
                # [0.05, 2.0] 클립. 플래그 OFF면 사용처에서 r̂=1로 무시된다.
                if (
                    self._np_last_real_q is not None
                    and self._np_last_sum_nin is not None
                ):
                    denom = max(abs(float(self._np_last_sum_nin)), 1.0e-6)
                    ratio = min(max(float(self._np_last_real_q) / denom, 0.05), 2.0)
                    if self._np_bias_ratio is None:
                        self._np_bias_ratio = ratio
                    else:
                        self._np_bias_ratio = 0.7 * self._np_bias_ratio + 0.3 * ratio
                if (
                    self._np_corrector_pending is not None
                    and self._np_last_real_q is not None
                ):
                    lam_base, tgt_committed = self._np_corrector_pending
                    r_hat_c = (
                        float(self._np_bias_ratio)
                        if bool(getattr(self.cfg.mpc, "np_bias_correction", False))
                        and self._np_bias_ratio is not None
                        else 1.0
                    )
                    deadband_c = float(getattr(self.cfg.mpc, "np_dual_deadband_frac", 0.0))
                    crit_c = float(getattr(self.cfg.leader, "N_P_crit_veh", 0.0) or 0.0)
                    low_stock_c = (
                        deadband_c > 0.0 and crit_c > 0.0
                        and n_p_now_veh < deadband_c * crit_c
                    )
                    # deadband v2(2026-07-13): 위반(실현 유입 > 환산 target)은 stock과 무관하게
                    # 적분한다. 펄스 loading edge는 stock이 flow를 지연 추종해 절대 게이트가
                    # 진짜 위반을 삼킨다(dhigh2 step7 +204 폐기 실측). 경부하 windup 수선은
                    # 위반 없는 저stock 국면에만 적용되므로 보존. 플래그 OFF면 비트동일.
                    viol_ovr = bool(
                        getattr(self.cfg.mpc, "np_deadband_violation_override", False)
                    )
                    violated_c = (
                        float(self._np_last_real_q) > r_hat_c * float(tgt_committed)
                    )
                    if low_stock_c and not (viol_ovr and violated_c):
                        self._lambda_P = 0.5 * float(lam_base)
                    else:
                        self._lambda_P = self._lambda_np_update(
                            float(lam_base),
                            float(self._np_last_real_q),
                            r_hat_c * float(tgt_committed),
                        )
                self._np_corrector_pending = None
        lambda_p = float(self._lambda_P) if dual_active else 0.0  # warm-start(직전 step 수렴값).
        # ---- NP-CAND-λ̂(2026-07-12, 리뷰 4안): 후보별 λ 1회 선반영 ----
        # 표준(플래그 OFF)에선 λ가 스텝 내 동결이라 follower 반응이 N_P 후보에 불변
        # (리뷰 지적: R_S(N_P)가 아니라 R_S(λ)). ON이면 직전 committed Σnin과 이 후보의
        # 투영 target으로 λ를 1회 적분 선반영한 λ̂ 아래서 Jacobi를 완전 재수렴시킨다.
        # 폐지된 step 내 이분법(A1+A2 주석 참조)과 달리 반복 중 λ 갱신이 없어
        # off-equilibrium commit이 없고, clip(0,cap)이 음수 보조금을 차단한다.
        np_cand_lambda_applied = 0.0
        # predictor(48)의 오차 신호 Q^prev: 실현 유입(우선) → 예측 Σnin(첫 스텝 fallback).
        q_prev = (
            self._np_last_real_q
            if self._np_last_real_q is not None
            else self._np_last_sum_nin
        )
        # 방법 A(K>0)면 1회 선반영을 건너뛴다 — λ는 아래 K-loop에서 계획 공간 반복으로 유도.
        if dual_active and np_cand_flag and not np_pd_active and q_prev is not None:
            pre_snapshot = ControlAction(
                ramp_metering=dict(control.ramp_metering),
                vsl=dict(control.vsl),
                green_times=dict(control.green_times),
                offsets=dict(control.offsets),
                inflow_outflow_allocation={},
            )
            pre_min, pre_max = self._np_feasible_range(
                state, coupling, pre_snapshot, forecast_arrivals, horizon_h,
            )
            interior_pre = float(getattr(self.cfg.mpc, "np_target_interior_frac", 0.0))
            pre_lo = pre_min + max(0.0, interior_pre) * max(0.0, pre_max - pre_min)
            pre_target = min(max(n_p_star, pre_lo), pre_max)
            deadband_pre = float(getattr(self.cfg.mpc, "np_dual_deadband_frac", 0.0))
            crit_pre = float(getattr(self.cfg.leader, "N_P_crit_veh", 0.0) or 0.0)
            r_hat_p = (
                float(self._np_bias_ratio)
                if bool(getattr(self.cfg.mpc, "np_bias_correction", False))
                and self._np_bias_ratio is not None
                else 1.0
            )
            low_stock_pre = deadband_pre > 0.0 and crit_pre > 0.0 and float(
                state.protected_accumulation_veh(net)
            ) < deadband_pre * crit_pre
            # deadband v2: 위반(q_prev > 환산 pre_target)은 stock 게이트를 우회(위 corrector 참조).
            viol_ovr_pre = bool(
                getattr(self.cfg.mpc, "np_deadband_violation_override", False)
            )
            violated_pre = float(q_prev) > r_hat_p * pre_target
            if low_stock_pre and not (viol_ovr_pre and violated_pre):
                lambda_p = 0.5 * lambda_p
            else:
                lambda_p = self._lambda_np_update(
                    lambda_p, float(q_prev), r_hat_p * pre_target,
                )
            np_cand_lambda_applied = 1.0
        sum_nin = 0.0
        evals = 0
        # 듀얼 분해 N_P 추적은 **control step 간 λ 적분 갱신**으로 한다(A1+A2, 2026-07-02 진단).
        # 이 step의 λ는 warm-start 값(직전 step에 컨트롤러가 commit한 λ) 하나뿐이고, green은
        # Jacobi 합의값 그대로 커밋한다. 폐지된 step 내 λ* 이분법+commit sweep의 병리(실측):
        #   (i) 음수 λ 분기가 dual solve의 34~41%에서 발동 — follower가 보호구역 유입을 늘리도록
        #       보상받아 target을 평균 67~89 veh/h 초과(overshoot).
        #   (iii) dual solve의 83~95%에서 λ* 재해 green이 합의 green을 중앙값 30초씩 뒤집은 채
        #        coupling 재수렴 없이 커밋(off-equilibrium commit).
        # λ_next = clip(λ + gain·(Σnin − projected_target), 0, cap)은 수렴 후 계산해 diagnostics로만
        # 내보내고, 선택된 후보의 λ만 컨트롤러가 commit한다(아래 post-loop 참고).

        s_max = max(1, min(self.cfg.mpc.max_nash_iter, 5))
        alpha = 0.5
        residual = float("inf")
        converged = False
        iteration = 0

        def _jacobi_consensus(lam_fixed: float) -> None:
            # 방법 A 지원 추출(2026-07-13): 기존 합의 루프 본체를 λ 인자만 받는 클로저로
            # 분리 — K=0(OFF) 경로는 이 클로저를 lambda_p로 1회 호출하므로 연산·순서가
            # 기존과 동일(비트동일). K>0이면 λ^(κ)마다 재호출해 현 결합값에서 재수렴한다.
            nonlocal coupling, sum_nin, evals, residual, converged, iteration
            residual = float("inf")
            converged = False
            for iteration in range(1, s_max + 1):
                # Jacobi: iteration 시작 control 스냅샷 고정 → 모든 agent 동일 z̃/previous 입력.
                snapshot = ControlAction(
                    ramp_metering=dict(control.ramp_metering),
                    vsl=dict(control.vsl),
                    green_times=dict(control.green_times),
                    offsets=dict(control.offsets),
                    inflow_outflow_allocation={},
                )
                new_green: Dict[str, float] = {}
                new_vsl: Dict[str, float] = {}
                sum_nin = 0.0  # 이 sweep의 Σ_i nin_i(현 λ에서 각 agent가 commit한 net inflow).
                for signal in net.signals:
                    arr_movement = self._per_movement_arrivals(signal, state, snapshot, demand)
                    p1, _, e, nin_i = self._solve_urban_agent_local(
                        signal, state, coupling, arr_movement, s_eff_frozen,
                        reservoir_drain, freeway_congestion, snapshot, leader,
                        lam_fixed, forecast_arrivals, horizon_h, demand,
                        committed_prev=previous,
                    )
                    new_green[f"{signal}_p1"] = p1
                    new_green[f"{signal}_p2"] = net.effective_green_total - p1
                    sum_nin += nin_i
                    evals += e
                # 합의 루프 동안 λ는 호출 인자 값으로 동결한다(스냅샷/결합 settle 우선). λ 갱신은
                # step 간 적분(수렴 후 λ_next 계산, 아래 post-loop) 또는 방법 A K-loop(κ 간).
                # PFO(leader=None)는 dual_active=False라 무영향.
                # freeway agent(Jacobi 내부): VSL solve만 cheap하게 — VSL은 여기서 inert이고
                # metering 좌표하강은 비싸므로 합의 루프 밖에서 1회만 돈다(아래 post-loop).
                # 13-player(segment_agents): segment agent 8개가 (VSL, meter)를 루프 안에서
                # best-response + 예산 사영 — meter 합의가 iteration을 필요로 하므로 in-loop.
                new_meter: Dict[str, float] = {}
                for link in net.freeway_links:
                    if self.segment_agents and self.metering_enabled:
                        vsl_dict, meter_dict, e = self._solve_freeway_segment_agents(
                            link, state, coupling, demand, snapshot, leader, previous,
                        )
                        new_meter.update(meter_dict)
                    else:
                        vsl_dict, _, e = self._solve_freeway_agent_local(
                            link, state, coupling, demand, snapshot,
                        )
                    new_vsl.update(vsl_dict)
                    evals += e
                control.green_times.update(new_green)
                control.vsl.update(new_vsl)
                if new_meter:
                    control.ramp_metering.update(new_meter)
                # outgoing 결합변수 갱신 후 under-relaxation 합성.
                predicted = self._wu._coupling(state, control, demand)
                relaxed = {
                    k: (1.0 - alpha) * coupling.get(k, 0.0) + alpha * predicted[k]
                    for k in predicted
                }
                residual = max(
                    (
                        abs(relaxed[k] - coupling.get(k, 0.0)) / max(1.0, abs(coupling.get(k, 0.0)))
                        for k in relaxed
                    ),
                    default=0.0,
                )
                coupling = relaxed
                if residual < self.cfg.mpc.distributed_coupling_tol:
                    converged = True
                    break

        # 방법 A 진단: 실사용 κ 횟수·최종 계획-공간 잔차(Σν − Ñ). OFF(K=0)면 0/0.
        np_pd_iters_used = 0.0
        np_pd_residual = 0.0
        # PD 종료사유 진단(2026-07-17): -1=루프 미실행, 0=K소진(미수렴), 1=잔차수렴, 2=λ고정점.
        # exit=2 & 최종 λ=cap이면 '수렴'이 아니라 '경계에 박힘' — 이 둘이 run_log에서 구분 불가였다.
        np_pd_exit = -1.0
        np_pd_lam_entry = 0.0
        np_pd_lam_path = ""
        if np_pd_active:
            # ---- 방법 A(2026-07-13): candidate 내부 primal-dual 반복 ----
            # λ^(κ+1) = Π[λ^(κ) + γ_P(Σν^(κ) − Ñ^(c))]를 K회(조기수렴 허용) 반복하며
            # 각 κ마다 Jacobi를 재수렴시켜 (λ*, green*) 안장점을 함께 얻는다.
            # target은 현행 선반영과 동일하게 계획 공간에서 투영하되 r̂ 환산은 제거한다
            # — 반복이 계획 공간 안에서 닫히므로 실현 보정(Q^real)이 필요 없다.
            # deadband/위반 override도 K-loop 안에서는 미적용(제약을 직접 강제).
            pd_snapshot = ControlAction(
                ramp_metering=dict(control.ramp_metering),
                vsl=dict(control.vsl),
                green_times=dict(control.green_times),
                offsets=dict(control.offsets),
                inflow_outflow_allocation={},
            )
            pd_min, pd_max = self._np_feasible_range(
                state, coupling, pd_snapshot, forecast_arrivals, horizon_h,
            )
            interior_pd = float(getattr(self.cfg.mpc, "np_target_interior_frac", 0.0))
            pd_lo = pd_min + max(0.0, interior_pd) * max(0.0, pd_max - pd_min)
            pd_target = min(max(n_p_star, pd_lo), pd_max)
            # γ_P 배율: 표준 gain(0.01)은 K≤5 안에 수렴 불가 — 25배≈0.25로 반복 수렴.
            gain_pd = self.lambda_np_step_gain * float(
                getattr(self.cfg.mpc, "np_pd_gain_mult", 25.0)
            )
            lam_curr = lambda_p  # warm start = standing _lambda_P(직전 step 커밋 λ).
            np_pd_lam_entry = float(lam_curr)
            _lam_path = [float(lam_curr)]
            np_pd_exit = 0.0  # 0=K소진(미수렴). break에서 1/2로 덮어씀.
            lam_last_sweep: Optional[float] = None
            for _kappa in range(1, np_pd_iters + 1):
                _jacobi_consensus(lam_curr)
                lam_last_sweep = lam_curr
                np_pd_iters_used = float(_kappa)
                np_pd_residual = float(sum_nin - pd_target)
                # 조기수렴 ①: 잔차가 target의 2% 이내 — λ 갱신 없이 종료(안장점 도달).
                if abs(np_pd_residual) <= 0.02 * max(pd_target, 1.0):
                    np_pd_exit = 1.0
                    break
                lam_new = min(
                    max(0.0, lam_curr + gain_pd * np_pd_residual), self.lambda_np_cap
                )
                # 조기수렴 ②: λ 고정점(cap/0 clip 포함) — green이 이미 그 λ의 균형.
                if abs(lam_new - lam_curr) <= 1.0e-6:
                    lam_curr = lam_new
                    np_pd_exit = 2.0
                    break
                lam_curr = lam_new
                _lam_path.append(float(lam_curr))
            np_pd_lam_path = "|".join("%.4f" % v for v in _lam_path)
            # 최종 λ로 1회 재수렴 — commit되는 green이 최종 λ의 균형이 되게
            # (off-equilibrium commit 금지). 마지막 sweep이 이미 그 λ였으면 생략.
            if lam_last_sweep is None or abs(lam_curr - lam_last_sweep) > 1.0e-6:
                _jacobi_consensus(lam_curr)
                np_pd_residual = float(sum_nin - pd_target)
            lambda_p = lam_curr
            np_cand_lambda_applied = 1.0
        else:
            _jacobi_consensus(lambda_p)
        # ---- 듀얼 분해 λ step 간 적분 갱신(A1+A2 — 이분법·commit sweep 폐지, 2026-07-02) ----
        # green은 Jacobi가 수렴시킨 값 그대로 커밋된다(A2: off-equilibrium commit 소멸). λ_next는
        # 마지막 합의 sweep의 Σnin과 투영 target으로 적분 갱신하되, 여기서 self._lambda_P에 쓰지
        # 않고 diagnostics로만 내보낸다(P-Stack이 step당 solve를 후보마다 여러 번 부르므로, 선택된
        # 후보의 λ만 컨트롤러가 commit해야 한다 — stackelberg_wu_metered._select_with_fallback_guard).
        # leader=None/use_dual_np=False면 dual_active=False라 이 블록 전체를 건너뛴다(PFO 무영향).
        projected_target = 0.0
        sigma_min = 0.0
        sigma_max = 0.0
        lambda_next = lambda_p
        if dual_active:
            commit_snapshot = ControlAction(
                ramp_metering=dict(control.ramp_metering),
                vsl=dict(control.vsl),
                green_times=dict(control.green_times),
                offsets=dict(control.offsets),
                inflow_outflow_allocation={},
            )
            # 리더 target을 follower가 실제로 만들 수 있는 Σnin 범위로 투영한 뒤 그 PROJECTED
            # target을 추적한다(plant control은 clip하지 않는다 — target만 투영).
            sigma_min, sigma_max = self._np_feasible_range(
                state, coupling, commit_snapshot, forecast_arrivals, horizon_h,
            )
            # windup 수선 ①(내부 투영): 모서리(feas_min)는 own-TTS와 타협하는 follower
            # 균형이 도달하지 못하는 점이라 오차가 구조적 양수 → λ 단방향 적분(8-seg
            # 155에서 cap 폭주, NP_OFF probe로 인과 확정). 내부점으로 클립하면 균형이
            # target 양쪽에 놓일 수 있어 λ가 자가 복원한다. frac=0이면 구거동(비트동일).
            interior = float(getattr(self.cfg.mpc, "np_target_interior_frac", 0.0))
            proj_lo = sigma_min + max(0.0, interior) * max(0.0, sigma_max - sigma_min)
            projected_target = min(max(n_p_star, proj_lo), sigma_max)
            # windup 수선 ②(경부하 deadband): 보호구역 accumulation이 임계 대비 충분히
            # 낮으면 보호가 무의미 — 적분 대신 감쇠로 λ를 회수(잔여 왜곡 제거).
            deadband = float(getattr(self.cfg.mpc, "np_dual_deadband_frac", 0.0))
            n_p_crit_veh = float(getattr(self.cfg.leader, "N_P_crit_veh", 0.0) or 0.0)
            n_p_now = float(state.protected_accumulation_veh(net))
            low_stock_post = (
                deadband > 0.0 and n_p_crit_veh > 0.0
                and n_p_now < deadband * n_p_crit_veh
            )
            # deadband v2: 위반(Σnin > projected_target)은 stock 게이트를 우회(corrector 참조).
            viol_ovr_post = bool(
                getattr(self.cfg.mpc, "np_deadband_violation_override", False)
            )
            if low_stock_post and not (viol_ovr_post and sum_nin > projected_target):
                lambda_next = 0.5 * lambda_p
            else:
                # 방향: Σnin > target(유입 과다) → λ 증가(억제 강화). max(0,·)이 A1(음수 금지)을 강제.
                lambda_next = self._lambda_np_update(lambda_p, sum_nin, projected_target)
        # ---- per-signal OFFSET 국소 탐색(PLATOON-AWARE, "proposed" authority 전용, step당 1회) ----
        # 수렴된 green/결합값을 고정한 뒤, 각 신호가 자기 corridor objective를 최소화하는 offset을
        # 탐색해 control.offsets에 commit한다. offset 채점은 phase-resolved 서비스 + 상류 platoon
        # 도착(`_solve_offset_local`)으로 한다 — offset이 platoon 도착 window와 자기 green을 정렬할수록
        # 자기 TTS가 낮아진다. snapshot=control(모든 신호 green+offset 동결)로 상류 platoon을 재구성한다.
        # "wu" authority(offset_enabled=False)면 offset 0 유지.
        # Jacobi 순서 의존성 주의: A의 offset 탐색은 D/B의 offset이 아직 commit되기 전(0) snapshot을
        # 본다(corridor 정렬은 상류 green window 위상이 주효과라 1-pass로 충분; 반복은 비용만↑).
        offset_evals = 0
        offsets_off_zero = 0
        # F3: offset 가격이 하달돼 있으면 offset 탐색을 활성화(자율 offset은 여전히 OFF —
        # 가격+trust가 방향과 보폭을 주는 leader-coordinated 모드).
        # J1: joint 패턴 directive가 있으면 per-signal 탐색 대신 그대로 적용(leader가
        # 조합을 통째로 평가했으므로 국소 재탐색은 조합을 되깨뜨릴 뿐) — 가드는 유지.
        offset_active = (
            self.offset_enabled
            or self.ramp_offset_enabled
            or (self.offset_marginal_price is not None)
            or (self.offset_directive is not None)
            or self.joint_green_offset_enabled
        )
        if offset_active:
            if self.offset_directive is not None:
                for signal in net.signals:
                    off = float(self.offset_directive.get(signal, 0.0))
                    control.offsets[signal] = off
                    if abs(off) > 1.0e-6:
                        offsets_off_zero += 1
            else:
                offset_snapshot = dict(control.green_times)
                # ramp_offset_enabled 단독(offset_enabled=False, 가격 없음)이면 ramp 신호만
                # 탐색(D/F offset 격리 실험). offset_enabled True거나 F3 가격 설정 시엔 전부.
                ramp_only = (
                    (not self.offset_enabled)
                    and (self.offset_marginal_price is None)
                    and self.ramp_offset_enabled
                )
                for signal in net.signals:
                    is_ramp = self._local_models[signal].has_ramps
                    # JOINT: non-ramp 신호는 (green_p1, offset) 2D 공동탐색으로 둘 다 재커밋
                    # (coordinate descent가 못 보는 cross 반영). ramp 신호는 아래 기존 경로.
                    if self.joint_green_offset_enabled and not is_ramp:
                        arr_movement = self._per_movement_arrivals(signal, state, control, demand)
                        joint = self._solve_urban_agent_joint(
                            signal, state, coupling, arr_movement, s_eff_frozen,
                            control, leader, lambda_p, forecast_arrivals, horizon_h, demand,
                            previous,
                        )
                        if joint is not None:
                            jp1, joff, _, je, _ = joint
                            control.green_times[f"{signal}_p1"] = float(jp1)
                            control.green_times[f"{signal}_p2"] = float(
                                net.effective_green_total - jp1
                            )
                            control.offsets[signal] = float(joff)
                            offset_evals += je
                            if abs(joff) > 1.0e-6:
                                offsets_off_zero += 1
                            continue
                    if ramp_only and not is_ramp:
                        continue
                    green_p1 = float(offset_snapshot.get(
                        f"{signal}_p1", net.effective_green_total / 2.0
                    ))
                    arr_movement = self._per_movement_arrivals(signal, state, control, demand)
                    off, e = self._solve_offset_local(
                        signal, green_p1, state, coupling, arr_movement, s_eff_frozen,
                        control, demand,
                    )
                    control.offsets[signal] = float(off)
                    offset_evals += e
                    if abs(off) > 1.0e-6:
                        offsets_off_zero += 1
            evals += offset_evals
        # ---- freeway agent ramp_metering 좌표하강(step당 1회, 수렴된 결합값 기준) ----
        # metering 탐색은 비싸므로(VSL probe sweep ×5분율) Jacobi 루프 밖에서 1회만 돈다.
        # 입력 snapshot = 합의 종료 control(최신 urban green·VSL). coupling['u_on_ramp']은
        # 수렴값이라 reservoir 적재가 안정적이고, metering이 own-TTS에서 창발한다.
        # "wu" authority(metering_enabled=False)면 metering 탐색을 건너뛰고 capacity 고정(=metering OFF).
        meter_snapshot = ControlAction(
            ramp_metering=dict(control.ramp_metering),
            vsl=dict(control.vsl),
            green_times=dict(control.green_times),
            offsets=dict(control.offsets),
            inflow_outflow_allocation={},
        )
        for link in net.freeway_links:
            if self.segment_agents and self.metering_enabled:
                # 13-player: VSL·metering은 Jacobi 합의 안에서 segment agent들이 이미
                # commit(예산 사영 포함) — post-loop 좌표하강 생략.
                continue
            if self.metering_enabled:
                vsl_dict, meter_dict, e = self._solve_freeway_agent_metered(
                    link, state, coupling, demand, meter_snapshot, leader, previous,
                )
                control.ramp_metering.update(meter_dict)
            else:
                # "wu" authority: metering OFF — ramp metering을 capacity로 고정하고 VSL만 푼다.
                for ramp in net.ramps:
                    if net.ramp_to_freeway.get(ramp) == link:
                        control.ramp_metering[ramp] = float(net.ramp_capacity_veh_h[ramp])
                vsl_dict, _, e = self._solve_freeway_agent_local(
                    link, state, coupling, demand, meter_snapshot,
                )
            control.vsl.update(vsl_dict)
            evals += e
        # ---- N_UF dual λ_UF 적분 갱신(dual 모드 + leader present) ----
        # 커밋된 총 metering(Σmeter)과 N_UF* 오차로 λ_UF를 signed 적분 갱신. solve() 내
        # 영속 상태(self._lambda_UF)는 건드리지 않고 diagnostic으로만 노출 — 선택된 후보의
        # λ_UF만 컨트롤러가 commit(λ_P와 동일 규약, 후보별 solve 오염 방지).
        nuf_dual_active = (
            leader is not None
            and str(getattr(self.cfg.mpc, "wu_faithful_nuf_coordination_mode", "equality")) == "dual"
        )
        lambda_uf_next = float(self._lambda_UF)
        if nuf_dual_active:
            n_uf_star = float(getattr(leader, "N_UF_star", 0.0))
            sum_meter = sum(float(v) for v in control.ramp_metering.values())
            lambda_uf_next = min(
                max(
                    -self.lambda_uf_cap,
                    float(self._lambda_UF) + self.lambda_uf_step_gain * (sum_meter - n_uf_star),
                ),
                self.lambda_uf_cap,
            )
            control.diagnostics["wu_faithful_lambda_uf_next"] = float(lambda_uf_next)
            control.diagnostics["wu_faithful_lambda_uf"] = float(self._lambda_UF)
            control.diagnostics["wu_faithful_sum_meter"] = float(sum_meter)
            control.diagnostics["wu_faithful_nuf_target"] = float(n_uf_star)
        # ---- OFFSET corridor 검증 가드(closed-loop, method rule 충실) ----
        # per-signal 국소 offset은 자기 TTS를 줄이지만 selfish라 corridor 전체(realized full-plant
        # 결합)에선 손해일 수 있다(downstream de-align). offset의 가치는 본질적으로 multi-signal
        # corridor 효과이므로, offset을 commit하기 전에 realized horizon TTT(offset-on vs offset-0)를
        # 같은 `run_coupled_interval`로 비교해 **개선될 때만** 유지한다(아니면 0으로 되돌림). 이는
        # offset을 actuator로 두되 그 효과를 closed-loop으로 검증하는 것으로, decisive check와 정합.
        offsets_kept = offsets_off_zero
        # leader 권위 directive면 가드 우회(leader가 이미 global joint 평가 — follower veto 금지).
        directive_authoritative = (
            self.offset_directive is not None and self.offset_directive_authoritative
        )
        if (
            offset_active
            and offsets_off_zero > 0
            and forecast is not None
            and not directive_authoritative
        ):
            ttt_on, _, _ = self._rollout_horizon_ttt(state, control, forecast)
            zero_control = control.copy()
            zero_control.offsets = {s: 0.0 for s in control.offsets}
            ttt_off, _, _ = self._rollout_horizon_ttt(state, zero_control, forecast)
            # 가드 마진: horizon-TTT는 closed-loop(매 step replan)의 불완전 proxy라, 미세 개선은
            # 실제로는 offset 전환 transient(platoon 구조 교란)로 손해가 된다(검증: 무마진 가드는
            # sweet_128서 56.33% < 56.63%). 따라서 **유의한** 상대 개선(≥ offset_keep_margin)일
            # 때만 offset을 유지한다. 기본 0.5%면 noise-level 개선은 0으로 되돌려 회귀를 막는다.
            if ttt_off <= ttt_on * (1.0 + self.offset_keep_margin) + 1.0e-9:
                control.offsets = dict(zero_control.offsets)
                offsets_kept = 0
            control.diagnostics["wu_faithful_offset_ttt_on"] = float(ttt_on)
            control.diagnostics["wu_faithful_offset_ttt_off"] = float(ttt_off)
        # 다음 step warm-start용 수렴 결합값 저장.
        self._prev_coupling = dict(coupling)
        # 듀얼 N_P 추적: λ_next는 diagnostics로만 노출 — solve() 내 영속 상태(self._lambda_P)
        # 변경 금지(후보별 solve가 가격을 오염시키지 않게). 선택된 후보의 λ_next만 컨트롤러가
        # commit한다. sum_nin은 이제 마지막 합의 sweep 기준.
        if dual_active:
            control.diagnostics["wu_faithful_lambda_next"] = float(lambda_next)
        control.diagnostics["wu_faithful_sum_nin"] = float(sum_nin)
        control.diagnostics["wu_faithful_lambda_P"] = float(lambda_p)
        control.diagnostics["wu_faithful_np_target"] = float(n_p_star)
        # N_P 투영 진단: 리더 의도(original)·실현가능범위로 투영한 target·잔차. dual path에서만
        # 유의값, leader=None이면 모두 0(default). plant/run-log가 읽는 rate는 PROJECTED target.
        control.diagnostics["wu_faithful_np_original_target"] = float(n_p_star)
        # ---- ε-best-response gap probe(2026-07-12, 진단 전용 — 행동 불변) ----
        # 고정점(최종 결합변수·최종 control)에서 각 urban follower를 단독 재최적화해
        # gap_i = J_i(committed) − J_i(BR) ≥ 0을 측정. freeway segment agent는 재-BR의
        # 행동 변화량(L1)만 측정(예산 사영·궤적 캐시 부작용은 save/restore로 차단).
        # 공유 등식 제약 하 metering의 단독 이탈은 실행가능집합 밖이므로 gap 정의에서 제외.
        if bool(getattr(self.cfg.mpc, "eps_gap_probe", False)):
            import copy as _copy
            probe_snapshot = ControlAction(
                ramp_metering=dict(control.ramp_metering),
                vsl=dict(control.vsl),
                green_times=dict(control.green_times),
                offsets=dict(control.offsets),
                inflow_outflow_allocation={},
            )
            gaps = []
            rel_gaps = []
            for signal in net.signals:
                arr_probe = self._per_movement_arrivals(signal, state, probe_snapshot, demand)
                _, j_br, _, _ = self._solve_urban_agent_local(
                    signal, state, coupling, arr_probe, s_eff_frozen,
                    reservoir_drain, freeway_congestion, probe_snapshot, leader,
                    lambda_p, forecast_arrivals, horizon_h, demand,
                )
                p1_com = float(control.green_times.get(f"{signal}_p1", 0.0))
                _, j_com, _, _ = self._solve_urban_agent_local(
                    signal, state, coupling, arr_probe, s_eff_frozen,
                    reservoir_drain, freeway_congestion, probe_snapshot, leader,
                    lambda_p, forecast_arrivals, horizon_h, demand,
                    candidates_override=[p1_com],
                )
                gap = max(0.0, float(j_com) - float(j_br))
                gaps.append(gap)
                rel_gaps.append(gap / max(abs(float(j_br)), 1.0e-9))
            fw_vsl_l1 = 0.0
            fw_meter_l1 = 0.0
            if self.segment_agents and self.metering_enabled:
                saved_traj = _copy.deepcopy(self._seg_traj)
                saved_off = dict(self._wu._last_offramp_flow)
                saved_has = bool(getattr(self._wu, "_has_last_offramp_flow", False))
                saved_diag = dict(self._seg13_diag)
                for link in net.freeway_links:
                    vsl_p, meter_p, _ = self._solve_freeway_segment_agents(
                        link, state, coupling, demand, probe_snapshot, leader, previous,
                    )
                    for k, v in vsl_p.items():
                        fw_vsl_l1 += abs(float(v) - float(control.vsl.get(k, v)))
                    for k, v in meter_p.items():
                        fw_meter_l1 += abs(float(v) - float(control.ramp_metering.get(k, v)))
                self._seg_traj = saved_traj
                self._wu._last_offramp_flow = saved_off
                self._wu._has_last_offramp_flow = saved_has
                self._seg13_diag = saved_diag
            control.diagnostics["wu_eps_gap_probe"] = 1.0
            control.diagnostics["wu_eps_gap_urban_max"] = float(max(gaps) if gaps else 0.0)
            control.diagnostics["wu_eps_gap_urban_mean"] = float(
                sum(gaps) / len(gaps) if gaps else 0.0
            )
            control.diagnostics["wu_eps_gap_urban_rel_max"] = float(
                max(rel_gaps) if rel_gaps else 0.0
            )
            control.diagnostics["wu_eps_fw_vsl_l1"] = float(fw_vsl_l1)
            control.diagnostics["wu_eps_fw_meter_l1"] = float(fw_meter_l1)
        control.diagnostics["wu_faithful_np_projected_target"] = float(projected_target)
        control.diagnostics["wu_faithful_np_sum_nin"] = float(sum_nin)
        control.diagnostics["wu_faithful_np_cand_lambda_applied"] = float(np_cand_lambda_applied)
        control.diagnostics["wu_faithful_np_cand_lambda"] = float(lambda_p)
        # 방법 A 진단: K-loop 실사용 횟수·최종 계획-공간 잔차(OFF면 0/0).
        control.diagnostics["wu_faithful_np_pd_iters"] = float(np_pd_iters_used)
        control.diagnostics["wu_faithful_np_pd_residual"] = float(np_pd_residual)
        # 종료사유·λ 경로(2026-07-17): exit=2 & λ=cap → '수렴'이 아니라 경계 고착.
        control.diagnostics["wu_faithful_np_pd_exit"] = float(np_pd_exit)
        control.diagnostics["wu_faithful_np_pd_lam_entry"] = float(np_pd_lam_entry)
        control.diagnostics["wu_faithful_np_pd_lam_path"] = np_pd_lam_path
        control.diagnostics["wu_faithful_np_bias_ratio"] = float(
            self._np_bias_ratio if self._np_bias_ratio is not None else 1.0
        )
        control.diagnostics["wu_faithful_np_feasible_min"] = float(sigma_min)
        control.diagnostics["wu_faithful_np_feasible_max"] = float(sigma_max)
        control.diagnostics["wu_faithful_np_projection_residual"] = float(n_p_star - projected_target)
        control.diagnostics["wu_faithful_np_target_error"] = float(sum_nin - projected_target)
        control.diagnostics["urban_net_inflow_target_veh_h"] = float(
            projected_target / max(horizon_h, 1.0e-9)
        )
        # ---- veh-단위 리더 인터페이스 진단(f9794b0 leader closure가 읽는 키; 테스트 스펙) ----
        # A1+A2에서 λ는 solve 내 동결이므로 이 step에 follower가 실현하는 Σnin은 합의값
        # 하나다. 따라서 veh-단위 "realizable" 밴드는 그 점으로 붕괴한다(min=max=sum_nin).
        # wide 후보범위(wu_faithful_np_feasible_min/max)는 λ 적분 target 투영용으로 그대로
        # 두고, leader closure/plant 로그가 읽는 projected_veh는 이 step 실현가능값으로
        # 보고한다(정직한 closure — 이분법 시대의 in-step tracking을 재도입하지 않는다).
        predictor_mode = str(self.cfg.mpc.wu_faithful_np_predictor_mode)
        control.diagnostics["wu_faithful_np_predictor_mode_code"] = float(
            _NP_PREDICTOR_MODE_CODES.get(predictor_mode, 0.0)
        )
        for mode_name in _NP_PREDICTOR_MODE_CODES:
            control.diagnostics[f"wu_faithful_np_predictor_{mode_name}"] = float(
                mode_name == predictor_mode
            )
        if dual_active:
            realizable_veh = float(sum_nin)
            projected_veh = float(min(max(n_p_star, realizable_veh), realizable_veh))
            control.diagnostics["wu_faithful_np_original_target_veh"] = float(n_p_star)
            control.diagnostics["wu_faithful_np_projected_target_veh"] = projected_veh
            control.diagnostics["wu_faithful_np_realized_sum_nin_veh"] = realizable_veh
            control.diagnostics["wu_faithful_np_feasible_min_veh"] = realizable_veh
            control.diagnostics["wu_faithful_np_feasible_max_veh"] = realizable_veh
            control.diagnostics["urban_net_inflow_original_target_veh"] = float(n_p_star)
            control.diagnostics["urban_net_inflow_target_veh"] = projected_veh
        # offset 진단: 탐색에서 0이 아닌 offset을 고른 신호 수 + 가드 후 실제 유지된 수.
        control.diagnostics["wu_faithful_offsets_off_zero"] = float(offsets_kept)
        control.diagnostics["wu_faithful_offsets_searched_off_zero"] = float(offsets_off_zero)
        control.diagnostics["wu_faithful_offset_evals"] = float(offset_evals)
        control.diagnostics.update(self._wu._repair_diagnostics)
        if self.segment_agents:
            control.diagnostics["wu_seg13_active"] = 1.0
            control.diagnostics.update(self._seg13_diag)
        return control, iteration, converged, float(residual), evals

    # ---------- 외부 인터페이스 (DistributedCoordinator.solve와 동일) ----------

    def _rollout_horizon_ttt(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: List[DemandStep],
    ) -> tuple[float, float, float]:
        """committed control을 horizon 동안 full coupled plant로 rollout해 realized TTT를 구한다.

        반환 (total_ttt, freeway_ttt, urban_ttt)[veh·h]. leader가 후보들을 follower_ttt 기준으로
        구분하려면(StackelbergMPCController `_leader_evaluation_base`가 `nash.objective_value`를
        follower TTT로 읽음) follower가 실제 realized horizon TTT를 내야 한다. leader가 읽는
        `_predict`와 동일한 `run_coupled_interval`을 써서 일관성을 맞춘다."""
        from src.simulation.coupling import run_coupled_interval

        s = state.copy()
        freeway_ttt = 0.0
        urban_ttt = 0.0
        for demand in forecast[: max(1, self.cfg.mpc.horizon_steps)]:
            result = run_coupled_interval(s, control, demand, self.cfg)
            s.time_sec += self.cfg.simulation.control_interval
            freeway_ttt += float(result.freeway_ttt)
            urban_ttt += float(result.urban_ttt)
        return float(freeway_ttt + urban_ttt), float(freeway_ttt), float(urban_ttt)

    def solve(
        self,
        state: TrafficState,
        leader: Optional[object],
        demand: DemandStep | Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
        leader_incumbent_obj: float = np.inf,
    ) -> NashResult:
        forecast = [demand] if isinstance(demand, DemandStep) else list(demand)
        if not forecast:
            raise ValueError("WuFaithfulFollower requires at least one demand step.")
        first_demand = forecast[0]
        start = time.perf_counter()
        previous = (
            previous_control.copy()
            if previous_control is not None
            else ControlAction.uncontrolled(self.cfg)
        )
        control, iteration, converged, residual, evals = self._solve_followers(
            state, first_demand, previous, leader, forecast,
        )
        # leader가 commit하는 setpoint를 control에 기록(leader=None이면 0).
        control.N_P_star = float(getattr(leader, "N_P_star", 0.0)) if leader is not None else 0.0
        control.N_UF_star = float(getattr(leader, "N_UF_star", 0.0)) if leader is not None else 0.0
        control.inflow_outflow_allocation = {}
        # objective_value = realized horizon TTT(leader follower_ttt base). leader=None(PFO closed
        # -loop 러너)는 objective를 안 읽지만, leader 평가에서는 후보 구분의 핵심이라 항상 채운다.
        # fallback guard가 읽는 distributed_response_rollout_ttt 진단도 함께 채운다.
        total_ttt, freeway_ttt, urban_ttt = self._rollout_horizon_ttt(state, control, forecast)
        objective_value = total_ttt
        control.diagnostics["wu_faithful_follower_active"] = 1.0
        control.diagnostics["wu_faithful_local_evals"] = float(evals)
        control.diagnostics["wu_faithful_solve_time_sec"] = float(time.perf_counter() - start)
        control.diagnostics["distributed_response_rollout_ttt"] = float(total_ttt)
        control.diagnostics["distributed_response_rollout_freeway_ttt"] = float(freeway_ttt)
        control.diagnostics["distributed_response_rollout_urban_ttt"] = float(urban_ttt)
        return NashResult(
            control=control,
            objective_value=float(objective_value),
            iterations=iteration,
            converged=converged,
            residual_objective=0.0,
            residual_control=float(residual),
            diagnostics=dict(control.diagnostics),
        )


def _clip_local(value: float, lower: float, upper: float) -> float:
    """metanet._clip와 동일(국소 ramp release 복제용; 미변경 파일 함수의 재선언)."""
    return min(max(value, lower), upper)


def q0_sum(q0: Mapping[str, float], model, phase_id: str) -> float:
    """phase별 초기 큐 합(pressure 중심 계산용)."""
    return float(sum(
        max(0.0, q0.get(m, 0.0))
        for m in model.movements
        if model.phase_of[m] == phase_id
    ))
