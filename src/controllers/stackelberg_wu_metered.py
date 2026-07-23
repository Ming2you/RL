# Wu충실 metering-PFO follower를 nash_solver로 주입하는 StackelbergMPCController 서브클래스(새 코드)
"""기존 `StackelbergMPCController`를 미변경으로 두고, follower만 `WuFaithfulFollower`로
교체하는 thin 서브클래스/팩토리.

근거(미변경 원칙): `StackelbergMPCController._make_follower_solver`는 cfg.mpc.follower_solver_mode가
"distributed"면 DistributedCoordinator, 아니면 NashSolver를 반환한다. 이 파일을 건드리지 않고
follower를 바꾸려면 `_make_follower_solver`만 오버라이드하면 된다. `decide_with_info` 등 나머지
경로는 `isinstance(self.nash_solver, DistributedCoordinator)` 분기에서 우리 follower가 모두 else
경로를 타므로(라인 ~301/991/1173/1338) 그대로 동작한다:
  - `_evaluate_full_candidate`: else 분기에서 `solve(state, action, forecast, previous)` 호출 —
    `WuFaithfulFollower.solve` 시그니처와 일치, `nash.control`/`nash.objective_value` 사용.
  - `_realized_net_inflow_veh` / `_evaluate_fallback_candidates`: DistributedCoordinator가 아니면
    각각 None / [] 반환 → output closure는 intent 유지, fallback 후보(no_control/pfo) 비활성.
    즉 우리 경우 fallback guard는 leader 후보만 보고 선택한다(leader vs PFO 비교는 러너가 한다).
  - `_proxy_score_candidate`: 베이스 else 분기는 action-blind(모든 후보 동점)라 이 파일에서
    action-aware 버전으로 오버라이드한다(용량비례 metering 근사 + plant rollout 채점).

따라서 서브클래스는 follower 주입만 한다. cfg는 호출처에서 follower_solver_mode를 임의값(예:
"wu_metered")으로 두거나 기본값 그대로 둬도 무방하다(우리는 mode와 무관하게 항상 주입).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from src.controllers.stackelberg_mpc import (
    DecisionResult,
    StackelbergMPCController,
    _LeaderCandidateEvaluation,
)
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.controllers.leader import Leader, LeaderAction


class StackelbergWuMeteredController(StackelbergMPCController):
    """follower=WuFaithfulFollower(metering-PFO)로 고정한 Stackelberg 컨트롤러."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        # ---------- B2: per-signal externality 가격(marginal price) 하달 ----------
        # Step B1(2026-07-03)이 검증한 가격 채널의 구현판. leader가 refresh마다 유한차분
        # 전역 rollout으로 g_ext_i = d(전역TTT)/dp1 − d(own-TTS)/dp1을 계산해 follower의
        # signal_marginal_price에 설정하고(사이엔 hold), follower는 green 후보 비용에
        # + w·g_ext_i·(p1 − p1_ref_i)를 더한다. w=1이 1차 정확값(B1 sweet spot, w=2는
        # overshoot). 이 가격은 전역 rollout이 필요해 leader 전용 — 순수 PFO 러너에는
        # 존재하지 않는다(P-Stack에서만 활성).
        # ── 기본 True + trust(2026-07-05 §8 승격): trust region과 함께 3개 regime 전부
        # 개선(7200s: sweet_128 −1.98% / 155 −0.63% / 190 −2.52%) — STOP 관례 첫 통과.
        # trust 없는 무제한 가격은 sweet_155 +10.3% 폭주(선형 월권, §7)라 -B2 변형으로만.
        self.signal_price_enabled: bool = True
        self.signal_price_delta_sec: float = 6.0  # 유한차분 스텝(B1 probe와 동일)
        self.signal_price_weight: float = 1.0
        # event-trigger 재선형화: 운영점(commit green)이 기준점에서 이만큼 이동하면
        # 재계산한다(B1 step35류 non-monotone은 재선형화 iteration으로 흡수). 그 외엔
        # 기존 leader_global_refresh cadence에 편승.
        self.signal_price_refresh_threshold_sec: float = 3.0
        # B2.1 trust region(2026-07-05): 가격 유효 범위를 유한차분 이웃으로 제한(폭주 기전
        # = 선형 가격의 이웃 밖 월권, 진단은 notes 2026-07-05 §6·§7). 기본 = δ(6.0s,
        # 가격을 측정한 바로 그 이웃 — §8 승격). None=무제한(구 -B2 재현용, 155 폭주).
        self.signal_price_trust_sec: Optional[float] = 6.0
        # ---------- SUBSET-PRICE(2026-07-09): 가격 대상 신호 선택 ----------
        # None=전 신호(기존). 집합 지정 시 해당 신호만 가격 계산·하달 — 나머지는 own-TTS
        # 자율(+마찰). 근거: 챔피언 런 실측 |g_ext| 계층이 극단적(F 4.07 ≫ D 0.32 ≫
        # A/B/C 0.16-0.20, 20배) — externality가 ramp interface에 집중. 강결합 신호만
        # 가격하면 refresh당 전역 rollout이 신호 수에 비례해 절감(대규모 망 스케일 논거).
        self.signal_price_signals: Optional[set] = None
        # ---------- B3(Codex f18e920 포팅) + B4: metering/VSL 가격 채널 ----------
        # green과 동일 흐름으로 통일: refresh(최초/cadence/event-trigger) 시 **동일 동결
        # 운영점**에서 g_i(전역 rollout)와 d_local(follower 국소 채점)을 모두 계산해
        # g_ext를 완성·하달, 사이엔 hold. 1차 TTT 가격 단독의 metering/VSL 가격은 음성
        # (절벽 lever가 과방류 → freeway breakdown, 2026-07-04 §3) → 기본 OFF.
        self.metering_price_enabled: bool = False
        self.metering_price_delta_veh_h: float = 60.0  # Codex f18e920과 동일 δ
        self.metering_price_refresh_threshold_veh_h: float = 30.0
        # B3TR(2026-07-05): metering 가격의 trust region — cap 분율 반경. green과 달리
        # 후보 격자(0.1~0.3·cap)가 δ(60veh/h)보다 커서, 반경과 **측정폭을 격자에 맞춘다**
        # (허용 이동폭만큼 측정: δ_r = trust·cap_r). None=무제한(-B3 재현, 과소방류 나선).
        # 기본 0.25 — metering_price_enabled를 켜면 trust가 함께 걸린다(러너 -B3TR).
        # 플래그십(ALLPRICE-JOINT)은 make_controller에서 δ=300·trust_frac=0.20 짝으로
        # override(2026-07-15 δ 스캔 승자 — 아래 make_controller 주석 참조).
        self.metering_price_trust_frac: Optional[float] = 0.25
        # B3CERT(2026-07-05, 사용자 승인 설계): **비대칭 안전 증명서**. trust의 전제(오류
        # 가역성)가 capacity drop에선 깨진다(B3TR v2: 한 칸씩 과방류로 걸어가 breakdown,
        # notes §11). 위험은 한 방향뿐이므로 — 조임(방류↓)은 가역이라 자유, 풂(방류↑)은
        # 가격 계산에 쓰는 +δ rollout의 예측 본선 밀도가 ρ_crit·(1−margin) 아래일 때만
        # leader가 인증(추가 rollout 0회). "성능은 가격, 안전-임계 방향은 증명서" —
        # CBF/safety-filter 구조의 가격 실현. 논문 서사의 통일성 회복(모든 레버가 가격
        # 프레임 안, 절벽 방향만 인증 동반).
        self.metering_release_cert_enabled: bool = True
        self.metering_release_cert_margin: float = 0.1
        self.vsl_price_enabled: bool = False
        self.vsl_price_delta_kmh: float = 10.0
        self.vsl_price_refresh_threshold_kmh: float = 5.0
        # PRICE-TR(2026-07-09): VSL trust region(±kmh) — 가격이 측정된 이웃 밖 후보 제외.
        # 기본 None(2026-07-09 분리실험 판정): 마찰이 살아있는 기본 구성에선 trust가
        # 중복 규제 + 유익한 큰 VSL 이동을 지연시켜 실측 +432 손해(E2 셀 12475→12043).
        # PRICE-TR(마찰 0)로 전환할 때만 boundary로 설정할 것(예: FD delta와 동일 ±10).
        self.vsl_price_trust_kmh: Optional[float] = None
        # ---------- F3(2026-07-06): offset 가격 채널 ----------
        # offset은 selfish로는 해로운 순수 조정 레버(2026-06-29 판정, leader-coordinated
        # 레버로 보존) — F3가 그 계획의 실행: leader가 전역 rollout FD로 g_ext_off를
        # 계산해 하달하면 follower의 offset 탐색이 가격+trust(그리드 1칸=cycle/8) 모드로
        # 활성화된다. legacy 잔존 격차(오프셋 coordinated 운영점)의 유력 재료. 기본 OFF.
        self.offset_price_enabled: bool = False
        self.offset_price_delta_sec: Optional[float] = None  # None=cycle/8(그리드 1칸)
        self.offset_price_refresh_threshold_sec: float = 7.0  # 그리드 반칸
        # 스텝 내 재선형화(SQP식 trust-region 걷기, 2026-07-15): 선형 offset 가격은
        # trust(±δ=cycle/8) 한 칸에 갇혀 뾰족한 platoon 골짜기를 못 가로지른다. K>0이면
        # 앵커가 trust 경계에 닿을 때마다 그 새 운영점에서 가격을 재측정(재선형화)하며
        # K회까지 15s씩 걸어 direct-search 운영점으로 수렴을 시도한다. 0=OFF(비트동일).
        self.offset_price_inner_iters: int = 0
        # ---------- SPSA(2026-07-10): 가격층 O(n) — per-lever FD를 동시섭동으로 대체 ----------
        # per-lever 유한차분은 lever수(O(n))×전역 rollout(O(n)) = refresh당 O(n²). SPSA(Spall
        # 1992)는 전 lever를 동시에 ±δ 섭동한 rollout 쌍 k개로 전 lever gradient를 한꺼번에
        # 추정(교차항은 독립 부호로 기대 0) → rollout 2k회 고정 = O(n). adjoint의 실용 대체.
        self.price_spsa_enabled: bool = False
        self.price_spsa_pairs: int = 4
        # ---------- JOINT(2026-07-09): bilinear cross-term 가격 ----------
        # per-lever 선형가격이 못 담는 lever쌍 교차곡률 h_ext = h_global − h_local(4-corner
        # 스텐실)를 하달. green×offset(non-ramp 신호): follower가 joint 2D 공동탐색해야 cross가
        # 작동(coordinate descent면 퇴화) → follower.joint_green_offset_enabled 동반. vsl×metering:
        # follower가 primal joint을 이미 포착 → cross 가격만 추가. 기본 OFF = 비트동일.
        self.green_offset_cross_price_enabled: bool = False
        self.green_offset_cross_weight: float = 1.0
        # green×offset cross의 offset FD 폭(2026-07-14 Task C probe): None=cycle/8(기존
        # 격자 1칸). 2차 혼합편미분은 소δ에서 잘 죽으므로 영역판별 시 cycle/4로 확대.
        self.green_offset_cross_offset_delta_sec: Optional[float] = None
        self.vsl_meter_cross_price_enabled: bool = False
        self.vsl_meter_cross_weight: float = 1.0
        # CROSS-GATE(2026-07-15): cross 2종을 capacity-drop 문턱 근방에서만 활성화.
        # 근거(8셀 실측): 절벽 무 → cross OFF 이득 −164~−420 / 절벽 유 → cross ON 이득 +27~+380.
        # 분리 축은 부하가 아님 — 170_w vs 170_incident_w는 NC부하 13,028로 같은데 부호가 반대.
        # 신호는 기존 wu_b3_cliff_both_*(metering FD가 이미 계산) 재사용 — 신규 문턱 없음.
        # 기본 False = 게이트 없이 현행 동작(비트동일).
        self.cross_cliff_gate_enabled: bool = False
        # ---------- A/B-패키지(2026-07-10): 계산비용 절감 ----------
        # A1 후보 N_UF 중복제거: step 내 follower 반응은 후보의 N_UF에만 의존(λ_P는
        # warm-start 고정, N_P는 dual로만 작용) → 같은 N_UF 후보는 solve+rollout 재사용,
        # N_P 의존 diagnostics(λ_next·target 계열)만 패치. warm-start 순서 효과로 미세
        # 드리프트 가능 — sweet_190 검증런으로 확인. 기본 OFF(env LEADER_DEDUPE=1).
        self.candidate_dedupe_enabled: bool = False
        self._nuf_solve_cache: Dict[float, tuple] = {}
        self._dedupe_hits: int = 0
        # B 가격-lite: 공용 baseline + one-sided FD + cross 스텐실 재활용 + 얕은 가격
        # rollout(H+1 — 가격은 배분 신호라 국소·단기; 후보 채점은 기존 깊이+far 유지).
        # refresh당 전역 rollout ~62(양측·4-corner·H+D) → ~30(H+1) = rollout·초 기준 −68%.
        # 기본 OFF(env PRICE_LITE=1) — 검증런 후 채택 결정.
        self.price_lite: bool = False
        self._price_rollout_count: int = 0
        # ---------- E1(2026-07-09): 가격 FD에 far(MFD tail) 합산 ----------
        # 비대칭 해소(사용자 지시): far는 leader 후보 채점에만 있고 가격 rollout엔 없었다 —
        # "far는 수량 신호(N_UF_star)만 똑똑하게 하고 가격(gradient)엔 안 들어간다". 활성 시
        # 모든 가격·cross rollout이 TTT + far(terminal state)로 채점된다. far는 leader 전용
        # 목적항이라 d_local 차감 없음(barrier와 동일 규약 — follower own-TTS에 대응물 부재).
        # 2026-07-09 기본 OFF 복귀: far-in-price는 gradient 크기와 노이즈를 함께 증폭 —
        # 마찰 문턱과 결합해 G1DF +139/APJOINT 13493 실측 악화(같은 날 ablation).
        # far는 leader 후보 채점(V=near+far)에서만 기본 작동. 켜면 E1(far도 가격에).
        # far 자체는 leader_mfd_far_enabled로도 게이트(내부 0 반환).
        self.price_far_enabled: bool = False
        # PRICE-HINGE(2026-07-23, 사용자 아이디어): rho_crit hinge를 가격 FD에도 합산.
        # 진단(190-skew): metering marginal price가 ~0인 이유 = 이득이 capacity-drop 문턱
        # 비선형이라 선형 FD가 보존식-평탄 영역서 0을 읽음. hinge(Σ max(0,ρ−ρ_crit)·L·λ)를
        # 가격 목적에 넣으면 ∂(문턱초과)/∂meter가 잡혀 rho_crit 근처 램프 가격이 살아난다.
        # 기본 OFF(price_hinge_enabled=False)면 비트동일. weight는 PRICE_HINGE_W 노브.
        self.price_hinge_enabled: bool = False
        self.price_hinge_weight: float = 1.0
        # ---------- LINK-SHARE(2026-07-09, 사용자 지시): network→link 배분 자유도 개방 ----------
        # ω_F 고정 균등분할(1/n_links)이 link 간 budget 배분을 pressure 무관하게 강제하던
        # 것을 개방. dual의 "link 간 재배분" 자유를 가격이 아닌 수량 채널로 흡수.
        #  - "density"(기본): 본선 headroom(ρ_crit까지 남은 수용량 veh, λ_eff 반영) 비례로
        #    매 스텝 상태 피드백 분할 — 비용 0, 연속 반응, HERO류 occupancy-배분 계열.
        #    λ_eff 반영이라 사고(유효차로 하락) 시 그 링크 몫이 자동 축소.
        #  - "search": 선택된 (N_P*,N_UF*) 위에서 s(첫 링크 몫) 좌표하강(grid, +2 eval/step,
        #    deep rollout+far로 채점 — 예측형이나 성긴 격자·추가비용). A/B ablation용.
        #  - "off": 기존 고정 균등(비트동일).
        self.nuf_link_share_mode: str = "density"
        self.nuf_link_share_grid: tuple = (0.35, 0.65)
        self._link_share_ctx = None
        # ---------- J1(2026-07-06): joint offset 패턴 ----------
        # F3 판정(offset 단독 편미분 = 0)의 처방: leader가 비-ramp 신호들의 offset
        # **조합**(3^k 패턴, 격자 {0, ±cycle/8})을 통째로 rollout 평가해 최선 조합을
        # directive로 하달 — 결합의 가치는 조합 후보에서만 보인다(legacy joint 평가의
        # 저렴판). 채택 마진은 corridor 가드와 동일 철학(미세 개선은 transient 손해).
        self.offset_joint_enabled: bool = False
        self.offset_joint_step_sec: Optional[float] = None  # None=cycle/8
        self.offset_joint_margin: float = 0.005
        # (green, offset) 결합: 상위 K 패턴에 대해 비-ramp p1 ±δ 공동 이동 변형을 함께
        # 평가 — 패턴 단독으론 안 보여도 green과 결합하면 이기는 조합을 채택 기준에
        # 반영한다(채택 후 green 자체는 가격+trust가 새 offset 운영점에서 재선형화되며
        # 따라감 — 계층적 joint). 변형 이득은 진단으로 기록.
        self.offset_joint_green_delta_sec: float = 6.0
        self.offset_joint_green_top_k: int = 3
        # ---------- LEADER-OFFSET(2026-07-07): offset 소유권을 follower→leader로 이전 ----------
        # 배경: offset은 joint 변수 — per-signal 가격≈0(F3 null), follower 국소 best-response는
        # de-coordinate(g1all 12638 > b2tr 12523), 양방향 mesh는 offset 하나로 두 방향 green-wave를
        # 못 맞춘다(MAXBAND=전역 bandwidth). 즉 follower(신호 하나)로는 원리적으로 못 정한다.
        # → offset 결정을 leader(전역 joint 평가자)로 옮긴다. green split은 국소라 follower 유지.
        # J1(offset_joint)과의 차이: (i) 전 신호(D/F ramp 포함), (ii) grid {0,±c/8,±c/4},
        # (iii) corridor 진행방향 lag 패턴 seed + 좌표하강(조합폭발 회피), (iv) follower offset
        # 탐색 완전 OFF(directive만 동결). 기본 OFF = 비트동일.
        self.leader_offset_enabled: bool = False
        self.leader_offset_method: str = "mpc"  # {"mpc", "maxband_lp"}
        # grid 분율(|frac|·cycle): {0, ±c/8, ±c/4}. legacy offset_std 28~45 커버.
        self.leader_offset_grid_fracs: tuple = (0.125, 0.25)
        # 채택 마진: 0.0 = per-step 개선이면 무조건 적용(단일 스텝 이득이 작아도 누적으로
        # 혼잡을 예방한다는 가설 — J1의 0.5% 마진이 0/40 채택을 낸 원인 제거).
        self.leader_offset_margin: float = 0.0
        # Stackelberg green×offset 결합: 최선 패턴에 green ±δ 공동이동 변형 평가(채점 반영).
        self.leader_offset_green_delta_sec: float = 6.0
        # 좌표하강 라운드(green-wave 결합은 라운드로 창발 — corridor_joint_offset_probe 검증).
        self.leader_offset_cd_rounds: int = 2
        # B4(사용자 제안, 2026-07-05 개정): 가격 채널들에 barrier 항의 marginal price를 합산.
        # ── 개정 이유(2026-07-04 §9 + 2026-07-05 probe): 제곱 barrier는 단위가 veh²·h라
        # 정체불명 가중치가 필요했고 얕은 초과에서 gradient가 소멸(g_TTT의 1/750)했다.
        # **선형 hinge·차량수 환산**으로 바꾸면 단위가 veh·h(TTT와 동일)가 되어 w=1이
        # "임계 초과 차량 1대·1시간 = TTT 1 veh·h"라는 물리적 의미를 갖고, 혼잡 상태
        # probe에서 gradient가 g_TTT와 같은 자릿수로 산다(fw: metering 축 +0.001~0.010
        # vs g_TTT −0.002~−0.005; spillback: green 축 A +0.055 vs g_TTT −0.030).
        # barrier 2종: (a) freeway rho_crit 초과차량(절벽=capacity drop 예고, 주 레버
        # metering), (b) urban 링크 spillback 여유부족(주 레버 green — B2의 sweet_155
        # 폭발이 겨냥하는 지점: 링크가 조여질 때만 자동 발화하는 내생적 regime 보정).
        # barrier는 leader 전용 목적항이라 d_local 차감이 없다(follower own-TTS에 부재).
        # 활성 시 green·metering 두 가격 채널 모두에 합산된다.
        self.barrier_price_enabled: bool = False
        self.barrier_weight: float = 1.0  # 선형·veh·h 단위라 1이 1차 정확값(fudge 아님)
        # spillback barrier 발화 문턱: 링크 여유공간이 저장고의 이 분율 아래로 떨어지면.
        self.barrier_spillback_frac: float = 0.2
        self._signal_price_refresh_count: int = 0
        self._signal_price_meta: Dict[str, float] = {}
        # ---- B: leader↔follower price↔response ADMM/dual-ascent iteration ----
        # single-shot price(=1)는 절벽·상보성서 joint를 못 재현(실측: 실제혼잡 ALLPRICE<PFO).
        # >1이면 price를 response에 되먹여 수렴까지 재선형화(under-relaxation으로 안정화).
        self.price_iter_max: int = 1
        self.price_iter_relax: float = 0.5
        # ★2단 채택 철회(2026-07-18): 26스텝 A/B는 +0.1%였으나 60스텝 전체 실행선 파국
        # (가격 스킵이 혼잡 plateau서 눈덩이). 기본 1=매 스텝 refresh(비트동일). N>1은 env로만.
        self.price_refresh_interval: int = 1
        self._signal_price_last_step: Optional[int] = None
        self.price_iter_tol: float = 1.0e-2  # 상대 control 변화 수렴 문턱
        # ---------- 층2(2026-07-14): β̂ 낙관편향 추정기 + trailing-regret 스위치 ----------
        # leader 내부 rollout은 체계적으로 낙관(제약 누락·capacity-drop 절벽 평활·horizon
        # 절단·동결 결합)이고 ~50 후보 argmax가 이를 증폭(optimizer's curse) — 예측 vs 예측
        # 비교인 fallback guard는 실현 +49% 파국에서도 발화하지 못했다.
        # 측정 기준(설계 선택, 2026-07-14): 이 코드베이스에서 1-step 예측(_predict)은 plant
        # (run_coupled_interval)와 같은 함수·같은 demand·같은 state라 첫-interval 예측이
        # 구성상 실현과 일치한다 — 첫-interval 비율은 낙관을 원리적으로 못 잰다. 따라서
        # 스펙의 허용 대안대로 **horizon TTT + lag buffer**를 쓴다: 예측 = 커밋 계획의
        # 채점 rollout TTT(distributed_response_rollout_ttt, held-plan H-step — guard가
        # 비교하는 바로 그 값), 실현 = 이후 H개 interval의 사다리꼴 TTT 합(lag H).
        # 닫힌루프 재결정·whipsaw·미계상 큐로 실현이 held-plan 예측에서 벗어나는 만큼이
        # β̂>1로 잡힌다(cfg.mpc.leader_bias_estimator, 기본 OFF=비트동일).
        self._beta_hat: Optional[float] = None
        self._beta_ewma_weight: float = 0.2       # EWMA 갱신 가중(스펙 고정)
        self._beta_ratio_clip: tuple = (0.5, 3.0)  # 단발 이상비율 클립(스펙 고정)
        # lag buffer: 커밋 시 {pred(H-step TTT), remaining(H), acc(실현 누적)} 등록,
        # 매 스텝 실현 interval을 전 항목에 가산 — remaining 소진 시 비율 확정·EWMA 갱신.
        self._beta_pending: deque = deque()
        self._beta_prev_total_veh: Optional[float] = None  # 직전 스텝 결정 시점 총 차량수 N_prev
        self._beta_last_realized: Optional[float] = None   # 이번 스텝에서 측정한 직전 interval 실현 TTT
        # β̂ 표류 재캘리브레이션 트리거: |β̂−1|>0.2 연속 스텝 카운터(5스텝 연속 시 제안).
        self._beta_drift_streak: int = 0
        # trailing-regret(cfg.mpc.regret_guard_steps=k>0): (실현 interval, incumbent 예측
        # interval) 쌍의 최근 k-창. incumbent 예측 interval = incumbent의 채점 horizon TTT/H
        # (이미 계산된 값 재사용 — 추가 rollout 0회). 창 전체에서 실현 합 > 예측 합×1.10이면
        # 다음 k스텝 강제 incumbent 커밋(hysteresis: 강제 중에도 창 계속 갱신, k 소진 후 복귀).
        self._regret_window: deque = deque()
        self._regret_pending_inc_pred: Optional[float] = None  # 직전 스텝 incumbent 예측 interval TTT
        self._regret_forced_remaining: int = 0
        self._regret_force_this_step: bool = False
        self._regret_last_gap: float = 0.0

    def _make_follower_solver(self, cfg: ExperimentConfig):
        return WuFaithfulFollower(cfg)

    # ---------- 층2(2026-07-14): β̂ 추정 + trailing-regret — 스텝 시작/커밋 훅 ----------

    def _beta_regret_on_entry(self, state: TrafficState) -> Dict[str, float]:
        """스텝 시작(decide 진입) 훅: 직전 interval의 실현 TTT를 측정해 β̂·regret 창을 갱신.

        실현 interval TTT ≈ ((N_prev + N_now)/2)·T_c_h — N = 총 urban+freeway 차량수
        (사다리꼴 적분 근사, 예측 rollout TTT와 동일 단위 veh·h). β̂은 lag buffer의
        H-스텝 창이 완성될 때마다 실현합/예측(H-step held-plan TTT) 비율로 EWMA 갱신.
        두 플래그 모두 OFF면 즉시 반환(계산 0, 비트동일)."""
        cfg_m = self.cfg.mpc
        est_on = bool(getattr(cfg_m, "leader_bias_estimator", False))
        regret_k = int(getattr(cfg_m, "regret_guard_steps", 0))
        self._regret_force_this_step = False
        if not est_on and regret_k <= 0:
            return {}
        net = self.cfg.network
        t_c_h = float(self.cfg.simulation.T_c_h)
        n_now = float(state.total_urban_vehicles(net)) + float(
            state.total_freeway_vehicles(net)
        )
        meta: Dict[str, float] = {}
        realized: Optional[float] = None
        if self._beta_prev_total_veh is not None:
            realized = 0.5 * (self._beta_prev_total_veh + n_now) * t_c_h
            # β̂ lag buffer: 미결 예측 항목마다 실현 interval을 누적, H개 채워지면 확정.
            if est_on:
                while self._beta_pending and self._beta_pending[0]["remaining"] <= 0:
                    self._beta_pending.popleft()
                for item in self._beta_pending:
                    item["acc"] += realized
                    item["remaining"] -= 1
                while self._beta_pending and self._beta_pending[0]["remaining"] <= 0:
                    done = self._beta_pending.popleft()
                    pred_h = float(done["pred"])
                    if pred_h > 1.0e-9:
                        ratio = float(done["acc"]) / pred_h
                        lo, hi = self._beta_ratio_clip
                        ratio = min(max(float(ratio), float(lo)), float(hi))
                        w = float(self._beta_ewma_weight)
                        self._beta_hat = (
                            ratio
                            if self._beta_hat is None
                            else (1.0 - w) * float(self._beta_hat) + w * ratio
                        )
            # regret 창: 같은 interval의 (실현, incumbent 예측 interval) 쌍으로 갱신.
            if regret_k > 0 and self._regret_pending_inc_pred is not None:
                self._regret_window.append(
                    (float(realized), float(self._regret_pending_inc_pred))
                )
                while len(self._regret_window) > regret_k:
                    self._regret_window.popleft()
        # 재캘리브레이션 트리거 ②: |β̂−1|>0.2가 5스텝 연속이면 표류 판정(진단·경고만).
        if est_on and self._beta_hat is not None:
            if abs(float(self._beta_hat) - 1.0) > 0.2:
                self._beta_drift_streak += 1
            else:
                self._beta_drift_streak = 0
            if self._beta_drift_streak >= 5:
                self._recalib_needed = True
        self._regret_pending_inc_pred = None
        self._beta_last_realized = realized
        # 이번 스텝 커밋 시점의 N — 다음 스텝 realized 계산 재료.
        self._beta_prev_total_veh = n_now
        if regret_k > 0:
            gap = 0.0
            if len(self._regret_window) >= regret_k:
                sum_real = sum(r for r, _ in self._regret_window)
                sum_inc = sum(p for _, p in self._regret_window)
                gap = sum_real - 1.10 * sum_inc
                # 트리거: 창 전체가 10% margin 넘게 나쁘고, 현재 강제 구간이 아니면 k스텝 강제.
                if self._regret_forced_remaining <= 0 and gap > 0.0:
                    self._regret_forced_remaining = regret_k
            self._regret_last_gap = float(gap)
            if self._regret_forced_remaining > 0:
                self._regret_force_this_step = True
                self._regret_forced_remaining -= 1
        if realized is not None:
            meta["leader_realized_interval_ttt"] = float(realized)
        if self._beta_hat is not None:
            meta["leader_beta_hat"] = float(self._beta_hat)
        if regret_k > 0:
            meta["leader_regret_active"] = float(self._regret_force_this_step)
            meta["leader_regret_gap"] = float(self._regret_last_gap)
            meta["leader_regret_window_len"] = float(len(self._regret_window))
        return meta

    def _committed_horizon_pred_ttt(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: List[DemandStep],
        n_steps: int,
    ) -> float:
        """커밋된 control의 held-plan H-step 예측 TTT — 채점 진단 재사용, 결측 시 재계산.

        1순위: follower solve가 채점 시 기록한 distributed_response_rollout_ttt
        (wu_faithful_follower._rollout_horizon_ttt, horizon_steps개 interval — fallback
        guard가 비교하는 바로 그 값이라 β̂ 보정과 차원이 정확히 맞는다, 추가 rollout 0회).
        결측(이례)이면 같은 정의로 `_predict`(depth_override=H) 재계산."""
        diag = getattr(control, "diagnostics", {}) or {}
        raw = diag.get("distributed_response_rollout_ttt")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if value == value and value not in (float("inf"), -float("inf")):
            return value
        _, pred = self._predict(state, control, forecast, depth_override=n_steps)
        return float(pred)

    def _beta_regret_on_commit(
        self,
        state: TrafficState,
        result: DecisionResult,
        forecast: List[DemandStep],
    ) -> Dict[str, float]:
        """커밋 직후 훅: 커밋 계획(+regret용 incumbent)의 예측 horizon TTT를 등록.

        예측은 채점 rollout 진단의 재사용이라 추가 rollout 0회. β̂용은 lag buffer에
        {pred, remaining=H}로 등록(entry 훅이 실현 H개를 채워 확정), regret용 incumbent는
        interval 평균(pred/H)으로 환산해 다음 실현 interval과 쌍을 만든다.
        두 플래그 모두 OFF면 즉시 반환(계산 0, 비트동일)."""
        cfg_m = self.cfg.mpc
        est_on = bool(getattr(cfg_m, "leader_bias_estimator", False))
        regret_k = int(getattr(cfg_m, "regret_guard_steps", 0))
        if not est_on and regret_k <= 0:
            return {}
        meta: Dict[str, float] = {}
        if not forecast:
            return meta
        n_steps = max(1, int(cfg_m.horizon_steps))
        pred_h = self._committed_horizon_pred_ttt(
            state, result.control, forecast, n_steps
        )
        meta["leader_pred_horizon_ttt"] = float(pred_h)
        meta["leader_pred_interval_ttt"] = float(pred_h) / float(n_steps)
        if est_on:
            self._beta_pending.append(
                {"pred": float(pred_h), "remaining": int(n_steps), "acc": 0.0}
            )
            # 안전 상한(정상 흐름에선 스텝당 1개 추가·1개 확정으로 길이 ≤ H+1).
            while len(self._beta_pending) > 4 * n_steps + 4:
                self._beta_pending.popleft()
        if regret_k > 0:
            pfo_eval = getattr(self, "_pfo_incumbent_eval", None)
            if pfo_eval is not None:
                inc_h = self._committed_horizon_pred_ttt(
                    state, pfo_eval.nash.control, forecast, n_steps
                )
                self._regret_pending_inc_pred = float(inc_h) / float(n_steps)
                meta["leader_regret_incumbent_pred_interval_ttt"] = float(
                    self._regret_pending_inc_pred
                )
        return meta

    # ---------- B2: per-signal externality 가격 계산/refresh ----------

    def decide_with_info(
        self,
        state: TrafficState,
        demand_forecast,
        previous_control: Optional[ControlAction] = None,
        config: Optional[ExperimentConfig] = None,
    ) -> DecisionResult:
        # config 교체는 base와 동일 로직을 먼저 수행한다 — 가격이 새 cfg의 follower에
        # 걸리도록(base가 나중에 nash_solver를 갈아치우면 하달한 가격이 유실된다).
        if config is not None and config is not self.cfg:
            self.close()
            self.cfg = config
            self.leader = Leader(config)
            self.nash_solver = self._make_follower_solver(config)
            self._pfo_fallback_previous_control = None
        forecast = list(demand_forecast)
        previous = (
            previous_control.copy()
            if previous_control is not None
            else self.previous_control.copy()
            if self.previous_control is not None
            else ControlAction.fixed(self.cfg)
        )
        # 층2(2026-07-14): β̂/regret 스텝 시작 갱신 — 직전 interval 실현 TTT 측정,
        # β̂ EWMA·regret 창·강제 incumbent 여부 확정(플래그 OFF면 무동작=비트동일).
        l2_meta = self._beta_regret_on_entry(state)
        # LINK-SHARE(2026-07-09): mode에 따라 step 시작 시 ω_F 설정.
        #  density → headroom 비례(상태 피드백), search/off → 균등(후보 평가 일관성).
        follower_ls = self.nash_solver
        if isinstance(follower_ls, WuFaithfulFollower):
            if self.nuf_link_share_mode == "density":
                follower_ls._wu._omega_f = self._link_share_omega(state)
            else:
                links_ls = list(self.cfg.network.freeway_links)
                n_l = max(len(links_ls), 1)
                follower_ls._wu._omega_f = {l: 1.0 / n_l for l in links_ls}
        self._link_share_ctx = (state, forecast, previous)
        # A1: N_UF 중복제거 캐시는 step 단위(state·가격·λ가 step 내 고정일 때만 유효).
        self._nuf_solve_cache = {}
        self._dedupe_hits = 0
        max_iter = max(1, int(self.price_iter_max))
        if max_iter <= 1:
            # single-shot(기존): price 1회 → follower 응답 1회 → commit.
            # 2단: refresh 간격 게이트. 스킵 스텝은 follower에 남은 직전 가격을 재사용한다.
            interval = max(1, int(self.price_refresh_interval))
            step_idx = int(round(
                state.time_sec / max(self.cfg.simulation.control_interval, 1.0e-9)))
            do_refresh = (
                interval <= 1
                or self._signal_price_last_step is None
                or step_idx % interval == 0
            )
            if do_refresh:
                self._maybe_refresh_signal_prices(state, forecast, previous)
                self._signal_price_last_step = step_idx
                if self._signal_price_meta:
                    self._signal_price_meta["wu_b2_price_skipped"] = 0.0
            else:
                self._signal_price_meta = dict(self._signal_price_meta)
                self._signal_price_meta["wu_b2_price_skipped"] = 1.0
                self._signal_price_meta["wu_b2_price_refreshed"] = 0.0
            result = super().decide_with_info(state, forecast, previous_control, config)
        else:
            # B: leader↔follower price↔response 반복(dual ascent). 매 iteration마다 현재
            # response 운영점에서 price 재선형화 → leader 재최적 → under-relaxation으로 되먹임.
            current = previous
            result = None
            k = 0
            for k in range(max_iter):
                self._maybe_refresh_signal_prices(state, forecast, current, force=True)
                # config는 최상단서 이미 적용 → super엔 None(재init 방지). previous=current(선형화점).
                result = super().decide_with_info(state, forecast, current, None)
                new = result.control
                if k > 0 and self._price_iter_converged(current, new):
                    break
                current = self._price_iter_relax(current, new, self.price_iter_relax)
            self._signal_price_meta = dict(self._signal_price_meta)
            self._signal_price_meta["wu_price_iter_count"] = float(k + 1)
        if self._signal_price_meta:
            result.metadata.update(self._signal_price_meta)
            result.control.diagnostics.update(self._signal_price_meta)
        # LINK-SHARE 진단: 이 step의 ω_F(밀도 기반이면 headroom 비례값)를 기록.
        if isinstance(self.nash_solver, WuFaithfulFollower):
            for link, w in self.nash_solver._wu._omega_f.items():
                result.control.diagnostics[f"leader_nuf_omega_{link}"] = float(w)
        # 층2: 커밋 계획(+incumbent)의 예측 첫-interval TTT 기록 + 진단 export
        # (플래그 OFF면 두 훅 모두 {} — 진단 키 추가 없음, 비트동일).
        l2_meta.update(self._beta_regret_on_commit(state, result, forecast))
        if l2_meta:
            result.metadata.update(l2_meta)
            result.control.diagnostics.update(l2_meta)
        return result

    def _link_share_omega(self, state: TrafficState) -> Dict[str, float]:
        """본선 headroom 비례 link budget 분할 — ω_l ∝ Σ_seg max(0, ρ_crit−ρ)·L·λ_eff.

        "임계까지 남은 수용량(veh)"이 큰 링크가 방류 예산을 더 받는다(절벽까지의 여유
        비례 — HERO류 occupancy-기반 배분 계열). λ_eff 반영이라 사고(유효차로 하락) 시
        해당 링크 몫이 자동 축소. 양쪽 다 여유 0(전부 임계 초과)이면 균등 fallback."""
        net = self.cfg.network
        rho_crit = float(net.rho_crit)
        seg_len = float(net.freeway_segment_length_km)
        headroom: Dict[str, float] = {}
        for link in net.freeway_links:
            rhos = state.freeway_density.get(link, [])
            lanes = state.freeway_effective_lanes.get(link, [])
            h = 0.0
            for i, rho in enumerate(rhos):
                lam = float(lanes[i]) if i < len(lanes) else float(net.freeway_lanes)
                h += max(0.0, rho_crit - float(rho)) * seg_len * lam
            headroom[link] = h
        total = sum(headroom.values())
        n_l = max(len(net.freeway_links), 1)
        if total <= 1.0e-9:
            return {link: 1.0 / n_l for link in net.freeway_links}
        return {link: h / total for link, h in headroom.items()}

    def _price_iter_relax(
        self, old: ControlAction, new: ControlAction, alpha: float
    ) -> ControlAction:
        """다음 선형화점 = (1−α)·old + α·new (under-relaxation, 절벽 진동 억제)."""
        c = new.copy()
        b = 1.0 - alpha
        c.N_UF_star = b * float(old.N_UF_star) + alpha * float(new.N_UF_star)
        c.N_P_star = b * float(old.N_P_star) + alpha * float(new.N_P_star)
        for attr in ("green_times", "vsl", "offsets", "ramp_metering"):
            od = getattr(old, attr, {}) or {}
            nd = dict(getattr(new, attr, {}) or {})
            for key, val in list(nd.items()):
                if key in od:
                    nd[key] = b * float(od[key]) + alpha * float(val)
            setattr(c, attr, nd)
        return c

    def _price_iter_converged(self, old: ControlAction, new: ControlAction) -> bool:
        """핵심 레버(N_UF·VSL·green)의 상대 변화가 tol 미만이면 수렴."""
        tol = float(self.price_iter_tol)
        if abs(float(old.N_UF_star) - float(new.N_UF_star)) / max(abs(float(old.N_UF_star)), 1.0) > tol:
            return False
        for attr, scale in (("vsl", 100.0), ("green_times", 120.0), ("ramp_metering", 1500.0)):
            od = getattr(old, attr, {}) or {}
            nd = getattr(new, attr, {}) or {}
            for key, val in nd.items():
                if key in od and abs(float(od[key]) - float(val)) / scale > tol:
                    return False
        return True

    def _signal_price_p1_bounds(self) -> tuple[float, float]:
        # p1 pair-feasible 범위: p1∈[green_min, green_max] ∧ p2=total−p1∈[green_min, green_max].
        net = self.cfg.network
        total = float(net.effective_green_total)
        lo = max(float(net.green_min), total - float(net.green_max))
        hi = min(float(net.green_max), total - float(net.green_min))
        return lo, hi

    def _global_rollout_ttt_with_green(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        signal: str,
        p1: float,
        depth_override: Optional[int] = None,
    ) -> float:
        """ego 신호만 green을 p1로 바꾸고 나머지는 previous 유지, horizon 전역 rollout TTT.

        B1 probe의 truth_horizon_ttt와 같은 구조지만, 미래 legacy trace 대신 현재
        committed control을 horizon 동안 hold한다(closed-loop에서 미래 제어는 미지)."""
        total = float(self.cfg.network.effective_green_total)
        control = previous.copy()
        control.green_times[f"{signal}_p1"] = float(p1)
        control.green_times[f"{signal}_p2"] = float(total - p1)
        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        return self._price_ttt(states, ttt)

    def _predict_ttt_and_barrier(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: List[DemandStep],
        depth_override: Optional[int] = None,
    ) -> tuple[float, float]:
        """전역 rollout의 (TTT, barrier). barrier는 B4 활성 시에만 계산(아니면 0).

        선형 hinge·차량수 환산(2026-07-05 개정 — 제곱은 단위 veh²·h + 얕은 초과에서
        gradient 소멸로 기각):
          barrier = w·Σ_states [ Σ_seg max(0, ρ−ρ_crit)·L_seg·lanes     (freeway 초과차량)
                               + Σ_link max(0, frac·cap − S_eff(link)) ] · T_c_h  (spillback 부족분)
        단위가 veh·h로 TTT와 동일 — w=1이 1차 정확값. TTT를 뽑는 같은 rollout의 상태에서
        계산하므로 추가 rollout 0회(green·metering 유한차분이 두 gradient를 동시에 얻음)."""
        from src.models.urban_queue_model import _effective_available_space

        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        return self._price_ttt(states, ttt), self._barrier_from_states(states)

    def _barrier_from_states(self, states: List[TrafficState]) -> float:
        """예측 상태 목록에서 barrier 합산(B4 활성 시에만, 아니면 0)."""
        from src.models.urban_queue_model import _effective_available_space

        if not self.barrier_price_enabled:
            return 0.0
        net = self.cfg.network
        t_c_h = float(self.cfg.simulation.T_c_h)
        seg_veh = float(net.freeway_segment_length_km) * float(net.freeway_lanes)
        rho_crit = float(net.rho_crit)
        spill_frac = float(self.barrier_spillback_frac)
        barrier = 0.0
        for s in states:
            for link in net.freeway_links:
                for rho in s.freeway_density.get(link, []):
                    excess = max(0.0, float(rho) - rho_crit) * seg_veh
                    barrier += self.barrier_weight * excess * t_c_h
            for u_link, cap in net.urban_link_storage_veh.items():
                space = float(_effective_available_space(s, self.cfg, u_link))
                deficit = max(0.0, spill_frac * float(cap) - space)
                barrier += self.barrier_weight * deficit * t_c_h
        return float(barrier)

    def _price_ttt(self, states: List[TrafficState], ttt: float, forecast=None) -> float:
        """가격 FD용 rollout 채점 — E1 활성 시 TTT + far(terminal state의 MFD tail).

        leader 후보 채점(_leader_evaluation_base)과 같은 V=near+far 형태로 가격을 정렬한다.
        far는 leader 전용 목적항이라 d_local 차감 없음(B4 barrier와 동일 규약). 기본 OFF."""
        if self.price_far_enabled and states:
            ttt += self._mfd_far_cost_to_go(states[-1])
        # PRICE-HINGE(2026-07-23): rho_crit 문턱 hinge를 가격 목적에 합산 → metering 한계가격이
        # capacity-drop 비선형(∂문턱초과/∂meter)을 잡는다. forecast는 폐쇄세그 면제용(skew=무관).
        if self.price_hinge_enabled and states:
            from src.controllers.stackelberg_mpc import leader_hinge_cost
            ttt += self.price_hinge_weight * leader_hinge_cost(
                self.cfg, states, forecast, force=True)
        # VdB4 보호큐 벌점 — 리더 가격 편입(2026-07-19 4차, 기본 OFF): 전역 rollout 상태의
        # 보호 movement 큐 초과분을 가격 목적에 가산 → 모든 리더 한계가격(green/metering/VSL)이
        # 제약-인지. 소거 실측: follower 벌점은 리더 B2 가격이 상쇄(가격OFF 시 green 30→89s,
        # 큐 459→150) — 제약은 가격 계산 지점(여기)에 있어야 계층이 한 방향을 가리킨다.
        _pq_mv = str(getattr(self.cfg.mpc, "protected_queue_movement", "") or "")
        if _pq_mv and states:
            _pq_w = float(getattr(self.cfg.mpc, "protected_queue_weight", 0.0))
            if _pq_w > 0.0:
                _pq_max = float(getattr(self.cfg.mpc, "protected_queue_max_veh", 50.0))
                _tc_h = float(self.cfg.simulation.T_c_h)
                for _s in states:
                    _q = max(0.0, float(_s.urban_movement_queue.get(_pq_mv, 0.0)))
                    ttt += _pq_w * max(0.0, _q - _pq_max) * _tc_h
        return float(ttt)

    def _global_rollout_metrics_with_green(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        signal: str,
        p1: float,
    ) -> tuple[float, float]:
        """ego 신호만 green을 p1로 바꾸고 나머지는 previous hold, (TTT, barrier)."""
        total = float(self.cfg.network.effective_green_total)
        control = previous.copy()
        control.green_times[f"{signal}_p1"] = float(p1)
        control.green_times[f"{signal}_p2"] = float(total - p1)
        return self._predict_ttt_and_barrier(state, control, forecast)

    def _global_rollout_metrics_with_metering(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        ramp: str,
        value: float,
        depth_override: Optional[int] = None,
    ) -> tuple[float, float, float]:
        """해당 ramp만 metering을 value로 바꾸고 previous hold — (TTT, barrier, max_rho).

        max_rho = 이 ramp가 합류하는 본선 링크의 예측 밀도 최대(전 세그먼트·전 horizon).
        B3CERT 안전 증명서의 재료: 가격 계산에 이미 쓰는 rollout에서 공짜로 얻는다."""
        control = previous.copy()
        control.ramp_metering = dict(previous.ramp_metering)
        control.ramp_metering[ramp] = float(value)
        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        barrier = self._barrier_from_states(states)
        link = self.cfg.network.ramp_to_freeway.get(ramp)
        max_rho = 0.0
        if link is not None:
            for s in states:
                for rho in s.freeway_density.get(link, []):
                    max_rho = max(max_rho, float(rho))
        return self._price_ttt(states, ttt, forecast), float(barrier), float(max_rho)

    def _global_rollout_ttt_with_vsl(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        link: str,
        seg_key: str,
        value: float,
        vsl_upper: float,
        depth_override: Optional[int] = None,
    ) -> float:
        """해당 segment만 VSL을 value로 바꾸고(link fallback 키 동기화) horizon TTT."""
        control = previous.copy()
        control.vsl = dict(previous.vsl)
        control.vsl[seg_key] = float(value)
        control.vsl[link] = min(float(control.vsl.get(link, vsl_upper)), float(value))
        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        return self._price_ttt(states, ttt)

    def _global_rollout_ttt_with_offset(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        signal: str,
        offset: float,
        depth_override: Optional[int] = None,
    ) -> float:
        """ego 신호만 offset을 바꾸고 나머지는 previous hold, horizon 전역 rollout TTT."""
        control = previous.copy()
        control.offsets = dict(previous.offsets)
        control.offsets[signal] = float(offset)
        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        return self._price_ttt(states, ttt)

    def _offset_price_relinearize_walk(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        signal: str,
        anchor0: float,
        delta_o: float,
        inner_k: int,
        grid_offsets: List[float],
        lc_map: Dict[float, float],
        o_weight: float,
        cycle: float,
    ) -> tuple[float, int, float]:
        """offset 가격의 스텝 내 재선형화 걷기 — (최종 앵커, 재선형화 횟수, g_ext) 반환.

        SQP식 trust-region walk. 앵커 a에서 externality 가격 g_ext = d(전역TTT)/d(off) −
        d(국소 phased)/d(off)를 측정하고, follower가 trust(±δ) 내 grid 후보 중 국소비용+
        선형가격을 최소화하는 offset*을 고른다(_solve_offset_local과 동일한 채점). offset*이
        trust 경계(|Δ|≈δ)에 닿으면 운영점을 그리로 옮겨 가격을 재측정(재선형화)하고, 내부
        최적(경계 미도달)이거나 K회 소진 시 종료한다. 국소비용(lc_map)은 offset 불변 platoon
        기반이라 신호당 1회 채점해 전달받고, 비싼 전역 rollout만 앵커 이웃을 캐시로 늘린다
        (총 전역 rollout ≤ 재선형화 횟수+2). 원형 offset 규약 보존(±δ 항상 유효)."""
        span = 2.0 * delta_o
        ttt_cache: Dict[float, float] = {}

        def _key(off: float) -> float:
            return round(float(off) % cycle, 6)

        def _gttt(off: float) -> float:
            k = _key(off)
            if k not in ttt_cache:
                ttt_cache[k] = self._global_rollout_ttt_with_offset(
                    state, previous, forecast, signal, k,
                )
            return ttt_cache[k]

        def _circ(o: float, ref: float) -> float:
            # 원형 최단 변위 ∈ (−cycle/2, cycle/2].
            return ((o - ref + cycle / 2.0) % cycle) - cycle / 2.0

        def _g_ext(a: float) -> float:
            a_hi = (a + delta_o) % cycle
            a_lo = (a - delta_o) % cycle
            g_i = (_gttt(a_hi) - _gttt(a_lo)) / span
            d_local = (lc_map.get(_key(a_hi), 0.0) - lc_map.get(_key(a_lo), 0.0)) / span
            return g_i - d_local

        def _resp(a: float, g: float) -> float:
            # trust(±δ) 내 grid 후보 중 국소비용+선형가격 최소(= follower best-response).
            best_o, best_obj = a, float("inf")
            for o in grid_offsets:
                d = _circ(o, a)
                if abs(d) > delta_o + 1.0e-9:
                    continue
                obj = lc_map.get(_key(o), 0.0) + o_weight * g * d
                if obj < best_obj - 1.0e-9:
                    best_obj, best_o = obj, float(o)
            return best_o

        anchor = float(anchor0) % cycle
        iters = 0
        g_ext = _g_ext(anchor)
        while True:
            o_star = _resp(anchor, g_ext)
            if abs(_circ(o_star, anchor)) < delta_o - 1.0e-9:
                break  # 내부 최적(trust 경계 미도달) — 수렴.
            if iters >= inner_k:
                break  # 재선형화 예산 소진 — 마지막 앵커에서 종료.
            anchor = float(o_star) % cycle  # trust 경계로 운영점 이동 → 재선형화.
            iters += 1
            g_ext = _g_ext(anchor)
        return anchor, iters, g_ext

    def _global_rollout_ttt_with_green_offset(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        signal: str,
        p1: float,
        offset: float,
        depth_override: Optional[int] = None,
    ) -> float:
        """ego 신호의 green(p1)·offset을 동시에 바꾸고 나머지 hold — horizon 전역 rollout TTT.

        JOINT green×offset cross-term 4-corner 스텐실용(h_global). green price와 동일하게
        `_predict`(leader_value_depth로 3+d 깊이) TTT + E1 활성 시 far를 쓴다."""
        total = float(self.cfg.network.effective_green_total)
        control = previous.copy()
        control.green_times[f"{signal}_p1"] = float(p1)
        control.green_times[f"{signal}_p2"] = float(total - p1)
        control.offsets = dict(previous.offsets)
        control.offsets[signal] = float(offset)
        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        return self._price_ttt(states, ttt)

    def _global_rollout_ttt_with_vsl_meter(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        ramp: str,
        link: str,
        meter: float,
        vsl: float,
        vsl_upper: float,
        depth_override: Optional[int] = None,
    ) -> float:
        """ego ramp의 metering·이 link 전 segment VSL을 동시에 바꾸고 나머지 hold — 전역 TTT.

        JOINT vsl×metering cross-term 4-corner 스텐실용(h_global). link-binding VSL 규약과
        일치하도록 전 segment 키 + link fallback 키를 vsl로 설정한다."""
        net = self.cfg.network
        control = previous.copy()
        control.ramp_metering = dict(previous.ramp_metering)
        control.ramp_metering[ramp] = float(meter)
        control.vsl = dict(previous.vsl)
        for index in range(int(net.freeway_segments_per_link)):
            control.vsl[f"{link}__seg{index}"] = float(vsl)
        control.vsl[link] = float(vsl)
        self._price_rollout_count += 1
        states, ttt = self._predict(state, control, forecast, depth_override=depth_override)
        return self._price_ttt(states, ttt)

    def _spsa_global_price_gradients(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
        levers: List[tuple],
        meter_cert_probe: Optional[Dict[str, float]] = None,
    ) -> tuple[Dict[tuple, float], float]:
        """SPSA(Spall 1992) — 전 lever 동시 ±δ rollout 쌍 k개로 전 lever의 전역 TTT gradient.

        levers: (kind, key, v_minus, v_plus, span). per-lever FD의 O(lever)회 rollout을 2k회
        고정으로 대체(가격층 O(n)). 교차 lever 항은 독립 Rademacher 부호로 기대 0(노이즈는
        k=price_spsa_pairs로 제어). refresh count 시드로 결정론적. meter_cert_probe가 있으면
        전 ramp 동시 +δ rollout의 본선 최대 예측밀도(개별 +δ보다 보수적)를 cert용으로 반환."""
        import numpy as np

        net = self.cfg.network
        total_green = float(net.effective_green_total)
        cycle = float(net.cycle_length)
        k = max(1, int(self.price_spsa_pairs))

        def build(sign_map: Dict[tuple, float]) -> ControlAction:
            c = previous.copy()
            c.green_times = dict(previous.green_times)
            c.ramp_metering = dict(previous.ramp_metering)
            c.vsl = dict(previous.vsl)
            c.offsets = dict(previous.offsets)
            touched_links: Dict[str, float] = {}
            for kind, key, v_minus, v_plus, _span in levers:
                v = v_plus if sign_map[(kind, key)] > 0 else v_minus
                if kind == "green":
                    c.green_times[f"{key}_p1"] = float(v)
                    c.green_times[f"{key}_p2"] = float(total_green - v)
                elif kind == "meter":
                    c.ramp_metering[key] = float(v)
                elif kind == "vsl":
                    c.vsl[key] = float(v)
                    link = key.split("__seg")[0]
                    touched_links[link] = min(touched_links.get(link, float("inf")), float(v))
                elif kind == "offset":
                    c.offsets[key] = float(v) % cycle
            for link, vmin in touched_links.items():
                c.vsl[link] = min(float(c.vsl.get(link, vmin)), vmin)
            return c

        g_acc: Dict[tuple, float] = {(kd, ky): 0.0 for kd, ky, _, _, _ in levers}
        for s_idx in range(k):
            rng = np.random.default_rng(100003 * int(self._signal_price_refresh_count) + s_idx)
            signs = {
                (kd, ky): (1.0 if rng.random() >= 0.5 else -1.0)
                for kd, ky, _, _, _ in levers
            }
            states_hi, t_hi_raw = self._predict(state, build(signs), forecast)
            states_lo, t_lo_raw = self._predict(
                state, build({q: -s for q, s in signs.items()}), forecast
            )
            t_hi = self._price_ttt(states_hi, t_hi_raw)
            t_lo = self._price_ttt(states_lo, t_lo_raw)
            for kd, ky, _vm, _vp, span in levers:
                if span > 1.0e-9:
                    g_acc[(kd, ky)] += (t_hi - t_lo) * signs[(kd, ky)] / span
        g = {q: v / float(k) for q, v in g_acc.items()}
        rho_joint = 0.0
        if meter_cert_probe:
            c = previous.copy()
            c.ramp_metering = dict(previous.ramp_metering)
            for ramp, m_hi in meter_cert_probe.items():
                c.ramp_metering[ramp] = float(m_hi)
            probe_states, _ = self._predict(state, c, forecast)
            for s in probe_states:
                for _link, dens in s.freeway_density.items():
                    if dens:
                        rho_joint = max(rho_joint, float(max(dens)))
        return g, rho_joint

    # 상단 arterial A-B-C / 하단 D-(E)-F / 수직 A-D·C-F(E는 비신호 통과) — 진행방향 lag seed용.
    _OFFSET_CORRIDORS = (("A", "B", "C"), ("D", "F"), ("A", "D"), ("C", "F"))

    def _solve_leader_offset(
        self,
        state: TrafficState,
        previous: ControlAction,
        forecast: List[DemandStep],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """LEADER-OFFSET(2A, MPC joint rollout): 전 신호 offset을 leader가 전역 rollout으로
        공동 결정한다. 후보 = corridor 진행방향 lag seed + 좌표하강(조합폭발 회피), grid
        {0,±c/8,±c/4}. 각 후보를 _predict 전역 rollout으로 채점(green×offset 결합은 최선
        패턴의 green ±δ 변형으로 반영 — Stackelberg follower green 반응 근사). 최소 TTT 패턴을
        offset_directive로 반환(가격 아닌 고정값). green split은 follower 유지."""
        net = self.cfg.network
        cycle = float(net.cycle_length)
        total = float(net.effective_green_total)
        signals = list(net.signals)
        lo, hi = self._signal_price_p1_bounds()

        def clamp(v: float) -> float:
            return max(lo, min(hi, float(v)))

        # grid = {0, ±frac·cycle}(중복 제거, 순서 보존).
        grid: List[float] = []
        for g in [0.0] + [
            sign * f * cycle
            for f in self.leader_offset_grid_fracs
            for sign in (1.0, -1.0)
        ]:
            gv = round(float(g) % cycle, 6)
            if gv not in grid:
                grid.append(gv)

        memo: Dict[tuple, float] = {}

        def key(pat: Dict[str, float]) -> tuple:
            return tuple(round(float(pat.get(s, 0.0)) % cycle, 3) for s in signals)

        def score(pat: Dict[str, float], green_shift: float = 0.0) -> float:
            k = (key(pat), round(float(green_shift), 3))
            if k in memo:
                return memo[k]
            control = previous.copy()
            control.offsets = dict(previous.offsets)
            for s in signals:
                control.offsets[s] = float(pat.get(s, 0.0)) % cycle
            if abs(green_shift) > 1.0e-9:
                control.green_times = dict(previous.green_times)
                for s in signals:
                    p1 = float(previous.green_times.get(f"{s}_p1", total / 2.0))
                    p1n = clamp(p1 + green_shift)
                    control.green_times[f"{s}_p1"] = p1n
                    control.green_times[f"{s}_p2"] = total - p1n
            _, ttt = self._predict(state, control, forecast)
            memo[k] = float(ttt)
            return float(ttt)

        zero_pat = {s: 0.0 for s in signals}
        zero_ttt = score(zero_pat)

        # ---- seed: corridor 진행방향 lag 패턴(양방향) + 직전 committed(hysteresis) ----
        seeds: List[Dict[str, float]] = [dict(zero_pat)]
        seeds.append({s: float(previous.offsets.get(s, 0.0)) % cycle for s in signals})
        sig_set = set(signals)
        for corr in self._OFFSET_CORRIDORS:
            seq0 = [s for s in corr if s in sig_set]
            if len(seq0) < 2:
                continue
            for seq in (seq0, list(reversed(seq0))):
                for f in self.leader_offset_grid_fracs:
                    tau = (f * cycle) % cycle
                    pat = dict(zero_pat)
                    for i, s in enumerate(seq):
                        pat[s] = (i * tau) % cycle
                    seeds.append(pat)
        best_pat = min(seeds, key=score)
        best_ttt = score(best_pat)

        # ---- 좌표하강 refinement(전 신호, green-wave 결합은 라운드로 창발) ----
        cur = dict(best_pat)
        for _ in range(max(0, int(self.leader_offset_cd_rounds))):
            improved = False
            for s in signals:
                base_v = float(cur.get(s, 0.0))
                best_v, best_s = base_v, score(cur)
                for g in grid:
                    trial = dict(cur)
                    trial[s] = g
                    t = score(trial)
                    if t < best_s - 1.0e-9:
                        best_s, best_v = t, g
                if abs(best_v - base_v) > 1.0e-9:
                    cur[s] = best_v
                    improved = True
            if not improved:
                break
        cd_ttt = score(cur)
        if cd_ttt < best_ttt - 1.0e-9:
            best_pat, best_ttt = dict(cur), cd_ttt

        # ---- Stackelberg green×offset 결합: 최선 패턴의 green ±δ 변형을 채점 반영 ----
        g_delta = float(self.leader_offset_green_delta_sec)
        best_shift = 0.0
        if g_delta > 1.0e-9:
            for sh in (g_delta, -g_delta):
                t = score(best_pat, sh)
                if t < best_ttt - 1.0e-9:
                    best_ttt, best_shift = t, sh

        gain = zero_ttt - best_ttt
        adopt = gain > float(self.leader_offset_margin) * max(zero_ttt, 1.0e-9)
        directive = dict(best_pat) if adopt else dict(zero_pat)
        diag = {
            "gain": float(gain),
            "adopt": float(bool(adopt)),
            "green_shift": float(best_shift),
            "n_eval": float(len(memo)),
        }
        return directive, diag

    def _maybe_refresh_signal_prices(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        force: bool = False,
    ) -> None:
        follower = self.nash_solver
        if not isinstance(follower, WuFaithfulFollower):
            return
        # 채널별 게이트: 꺼진 채널의 잔존 가격은 항상 제거(A/B 격리).
        if not self.signal_price_enabled:
            follower.signal_marginal_price = None
        if not self.metering_price_enabled:
            follower.metering_marginal_price = None
            follower.metering_release_certified = None
        if not self.vsl_price_enabled:
            follower.vsl_marginal_price = None
            follower.vsl_marginal_price_trust_kmh = None
        if not self.offset_price_enabled:
            follower.offset_marginal_price = None
        if not self.offset_joint_enabled and not self.leader_offset_enabled:
            follower.offset_directive = None
        if not self.green_offset_cross_price_enabled:
            follower.green_offset_cross_price = None
        if not self.vsl_meter_cross_price_enabled:
            follower.vsl_meter_cross_price = None
        if not (
            self.signal_price_enabled
            or self.metering_price_enabled
            or self.vsl_price_enabled
            or self.offset_price_enabled
            or self.offset_joint_enabled
            or self.leader_offset_enabled
            or self.green_offset_cross_price_enabled
            or self.vsl_meter_cross_price_enabled
        ):
            self._signal_price_meta = {
                "wu_b2_price_enabled": 0.0,
                "wu_b2_price_refreshed": 0.0,
            }
            return
        net = self.cfg.network
        total = float(net.effective_green_total)
        lo, hi = self._signal_price_p1_bounds()

        def clamp(v: float) -> float:
            return max(lo, min(hi, float(v)))

        # ---- 운영점 스냅샷: 모든 채널이 같은 previous(동결 운영점)를 본다 ----
        # SUBSET-PRICE: signal_price_signals 지정 시 그 신호만 가격 대상(나머지 자율).
        _price_sigs = self.signal_price_signals
        op_green: Dict[str, float] = (
            {
                signal: clamp(previous.green_times.get(f"{signal}_p1", total / 2.0))
                for signal in net.signals
                if _price_sigs is None or signal in _price_sigs
            }
            if self.signal_price_enabled else {}
        )
        ramp_caps = {r: float(net.ramp_capacity_veh_h[r]) for r in net.ramps}
        op_meter: Dict[str, float] = (
            {
                r: min(max(float(previous.ramp_metering.get(r, ramp_caps[r])), 0.0), ramp_caps[r])
                for r in net.ramps
            }
            if self.metering_price_enabled else {}
        )
        ff = self.cfg.freeway_follower
        vsl_values = [float(v) for v in ff.vsl_set]
        vsl_lower = min(vsl_values) if vsl_values else 0.0
        vsl_upper = max(vsl_values) if vsl_values else 0.0
        op_vsl: Dict[str, float] = {}
        if self.vsl_price_enabled and vsl_values:
            for link in net.freeway_links:
                for index in range(int(net.freeway_segments_per_link)):
                    key = f"{link}__seg{index}"
                    op_vsl[key] = float(segment_vsl(previous, link, index, self.cfg))
        cycle = float(net.cycle_length)

        def _circ(a: float, b: float) -> float:
            d = abs(a - b) % cycle
            return min(d, cycle - d)

        op_offset: Dict[str, float] = (
            {
                s: float(previous.offsets.get(s, 0.0)) % cycle
                for s in net.signals
                if not follower._local_models[s].has_ramps
            }
            if self.offset_price_enabled else {}
        )

        # ---- refresh 판정: 강제(ADMM iteration) ∨ 가격 부재 ∨ cadence ∨ event-trigger(운영점 이동) ----
        refresh = force or (
            (self.signal_price_enabled and follower.signal_marginal_price is None)
            or (self.metering_price_enabled and follower.metering_marginal_price is None)
            or (self.vsl_price_enabled and follower.vsl_marginal_price is None)
            or (self.offset_price_enabled and follower.offset_marginal_price is None)
            or (self.offset_joint_enabled and follower.offset_directive is None)
            or (self.leader_offset_enabled and follower.offset_directive is None)
            or (self.green_offset_cross_price_enabled and follower.green_offset_cross_price is None)
            or (self.vsl_meter_cross_price_enabled and follower.vsl_meter_cross_price is None)
        )
        if not refresh and self._leader_global_refresh_active(state):
            refresh = True
        if not refresh:
            # event-trigger: 운영점이 선형화 기준점에서 threshold 이상 이동한 레버가 있으면
            # 재선형화(dual ascent/SQP식 iteration — B1 step35 non-monotone 처방).
            for signal, p1_now in op_green.items():
                ref = float(follower.signal_marginal_price_ref.get(signal, p1_now))
                if abs(p1_now - ref) >= float(self.signal_price_refresh_threshold_sec):
                    refresh = True
                    break
        if not refresh:
            for ramp, x_now in op_meter.items():
                ref = float(follower.metering_marginal_price_ref.get(ramp, x_now))
                if abs(x_now - ref) >= float(self.metering_price_refresh_threshold_veh_h):
                    refresh = True
                    break
        if not refresh:
            for key, v_now in op_vsl.items():
                ref = float(follower.vsl_marginal_price_ref.get(key, v_now))
                if abs(v_now - ref) >= float(self.vsl_price_refresh_threshold_kmh):
                    refresh = True
                    break
        if not refresh:
            for signal, off_now in op_offset.items():
                ref = float(follower.offset_marginal_price_ref.get(signal, off_now))
                if _circ(off_now, ref) >= float(self.offset_price_refresh_threshold_sec):
                    refresh = True
                    break
        release_trigger = False
        if not refresh and self.metering_price_enabled:
            # release-트리거 재선형화(2026-07-14): 커밋된 직전 control의 실현
            # Σmeter/budget(wu_b3_release_ratio_*)이 회랑 하한 α 근방(<α+0.05)까지
            # 떨어지면(과소방류 나선 조짐) 가격 강제 재선형화 — 지연 불안정 완화.
            # min(·,0.999)로 등식(α=1) 재현 시 상시 발화 방지(ratio≡1.0).
            _floor = float(getattr(follower, "seg13_release_floor_frac", 0.0) or 0.0)
            _thr = min(_floor + 0.05, 0.999)
            for _k, _v in getattr(previous, "diagnostics", {}).items():
                if _k.startswith("wu_b3_release_ratio_") and float(_v) < _thr:
                    refresh = True
                    release_trigger = True
                    break
        if not refresh:
            self._signal_price_meta = dict(self._signal_price_meta)
            self._signal_price_meta["wu_b2_price_refreshed"] = 0.0
            return

        meta: Dict[str, float] = {"wu_b2_price_enabled": float(self.signal_price_enabled)}
        meta["wu_b3_release_refresh"] = float(release_trigger)
        self._price_rollout_count = 0

        # ---- B-패키지: price-lite 경로(공용 baseline + one-sided + 스텐실 재활용 + 얕은 rollout) ----
        if self.price_lite:
            self._compute_prices_lite(
                state, forecast, previous, follower, net, meta,
                op_green, clamp, total, ramp_caps, op_meter,
                vsl_values, vsl_lower, vsl_upper, op_vsl, cycle,
            )
            self._signal_price_refresh_count += 1
            meta["wu_b2_price_refreshed"] = 1.0
            meta["wu_b2_price_refresh_count"] = float(self._signal_price_refresh_count)
            meta["wu_price_rollout_count"] = float(self._price_rollout_count)
            meta["wu_price_lite"] = 1.0
            self._signal_price_meta = meta
            return

        # ---- green 채널(B2) ----
        # ---- SPSA(가격층 O(n)): 4채널 per-lever FD의 전역 rollout을 동시섭동 쌍 k개로 대체 ----
        # d_local(국소 채점)·trust·cross-term은 기존 그대로 — 전역 g_i의 추정 방식만 교체.
        spsa_g: Optional[Dict[tuple, float]] = None
        spsa_rho_joint = float("inf")
        if bool(getattr(self, "price_spsa_enabled", False)):
            spsa_levers: List[tuple] = []
            spsa_meter_probe: Dict[str, float] = {}
            if self.signal_price_enabled:
                d_g = float(self.signal_price_delta_sec)
                for signal, p1_0 in op_green.items():
                    lo_v, hi_v = clamp(p1_0 - d_g), clamp(p1_0 + d_g)
                    spsa_levers.append(("green", signal, lo_v, hi_v, hi_v - lo_v))
            if self.metering_price_enabled:
                d_m0 = float(self.metering_price_delta_veh_h)
                for ramp, x0 in op_meter.items():
                    cap = ramp_caps[ramp]
                    d_r = d_m0 if self.metering_price_trust_frac is None else max(
                        d_m0, float(self.metering_price_trust_frac) * cap
                    )
                    lo_v, hi_v = max(0.0, x0 - d_r), min(cap, x0 + d_r)
                    spsa_levers.append(("meter", ramp, lo_v, hi_v, hi_v - lo_v))
                    spsa_meter_probe[ramp] = hi_v
            if self.vsl_price_enabled:
                d_v = float(self.vsl_price_delta_kmh)
                for key, x0 in op_vsl.items():
                    lo_v, hi_v = max(vsl_lower, x0 - d_v), min(vsl_upper, x0 + d_v)
                    spsa_levers.append(("vsl", key, lo_v, hi_v, hi_v - lo_v))
            if self.offset_price_enabled:
                d_o = (
                    float(self.offset_price_delta_sec)
                    if self.offset_price_delta_sec is not None else cycle / 8.0
                )
                for signal, off0 in op_offset.items():
                    spsa_levers.append(
                        ("offset", signal, (off0 - d_o) % cycle, (off0 + d_o) % cycle, 2.0 * d_o)
                    )
            if spsa_levers:
                spsa_g, spsa_rho_joint = self._spsa_global_price_gradients(
                    state, previous, forecast, spsa_levers,
                    spsa_meter_probe if spsa_meter_probe else None,
                )
                meta["wu_spsa_enabled"] = 1.0
                meta["wu_spsa_pairs"] = float(self.price_spsa_pairs)
                meta["wu_spsa_levers"] = float(len(spsa_levers))

        if self.signal_price_enabled:
            delta = float(self.signal_price_delta_sec)
            pts: Dict[str, tuple[float, float, float]] = {}
            requests: Dict[str, List[float]] = {}
            for signal, p1_0 in op_green.items():
                p_hi = clamp(p1_0 + delta)
                p_lo = clamp(p1_0 - delta)
                pts[signal] = (p1_0, p_lo, p_hi)
                requests[signal] = [p_lo, p_hi]
            local_costs = follower.local_green_costs(requests, state, previous, forecast[0])
            prices: Dict[str, float] = {}
            refs: Dict[str, float] = {}
            for signal, (p1_0, p_lo, p_hi) in pts.items():
                two_delta = p_hi - p_lo
                if two_delta <= 1.0e-9:
                    g_ext = 0.0
                else:
                    if spsa_g is not None:
                        g_i = float(spsa_g.get(("green", signal), 0.0))
                        bar_hi = bar_lo = 0.0
                    else:
                        ttt_hi, bar_hi = self._global_rollout_metrics_with_green(
                            state, previous, forecast, signal, p_hi,
                        )
                        ttt_lo, bar_lo = self._global_rollout_metrics_with_green(
                            state, previous, forecast, signal, p_lo,
                        )
                        g_i = (ttt_hi - ttt_lo) / two_delta
                    cost_lo, cost_hi = local_costs[signal]
                    g_ext = g_i - (cost_hi - cost_lo) / two_delta
                    if self.barrier_price_enabled:
                        # B4: spillback/freeway barrier gradient 합산(green 축 —
                        # 주 신호는 spillback: 하류 링크가 조여질 때 green 증가를 막는다).
                        g_ext += (bar_hi - bar_lo) / two_delta
                prices[signal] = float(g_ext)
                refs[signal] = float(p1_0)
                meta[f"wu_b2_price_{signal}"] = float(g_ext)
                meta[f"wu_b2_price_ref_{signal}"] = float(p1_0)
            follower.signal_marginal_price = prices
            follower.signal_marginal_price_ref = refs
            follower.signal_marginal_price_weight = float(self.signal_price_weight)
            follower.signal_marginal_price_trust_sec = (
                float(self.signal_price_trust_sec)
                if self.signal_price_trust_sec is not None else None
            )
            meta["wu_b2_price_delta_sec"] = delta
            meta["wu_b2_price_trust_sec"] = (
                float(self.signal_price_trust_sec)
                if self.signal_price_trust_sec is not None else 0.0
            )

        # ---- metering 채널(B3, B4 barrier 합산 가능) ----
        # g_ext = d(전역TTT)/dx − d(own-TTS)/dx (+ d(barrier)/dx). 세 미분 모두 같은
        # 동결 운영점에서 — Codex 원안의 "g_i는 commit점·d_local은 snapshot점" 혼합을 제거.
        if self.metering_price_enabled:
            meta["wu_b3_meter_price_enabled"] = 1.0
            meta["wu_b4_barrier_enabled"] = float(self.barrier_price_enabled)
            delta_m = float(self.metering_price_delta_veh_h)
            m_pts: Dict[str, tuple[float, float, float]] = {}
            m_requests: Dict[str, List[float]] = {}
            for ramp, x0 in op_meter.items():
                cap = ramp_caps[ramp]
                # B3TR: trust 설정 시 측정폭을 trust 반경(격자 폭)에 맞춘다 —
                # 가격이 유효해야 하는 바로 그 구간을 측정하는 secant.
                d_r = delta_m
                if self.metering_price_trust_frac is not None:
                    d_r = max(delta_m, float(self.metering_price_trust_frac) * cap)
                m_hi = min(cap, x0 + d_r)
                m_lo = max(0.0, x0 - d_r)
                m_pts[ramp] = (x0, m_lo, m_hi)
                m_requests[ramp] = [m_lo, m_hi]
            local_m = follower.local_metering_costs(m_requests, state, previous, forecast[0])
            m_prices: Dict[str, float] = {}
            m_refs: Dict[str, float] = {}
            m_certs: Dict[str, bool] = {}
            rho_cert_limit = float(net.rho_crit) * (
                1.0 - float(self.metering_release_cert_margin)
            )
            # ---- 3점 곡률 진단(2026-07-14, δ 스캔): TTT(−Δ)/TTT(기준)/TTT(+Δ) ----
            # 기준점 rollout은 동결 운영점(previous 그대로 — ramp를 자기 op값으로 덮어써도
            # 동일 control)이라 전 램프 공용 → 1회만 추가 계산(refresh당 +1 rollout).
            fd3_base_ttt: Optional[float] = None
            if m_pts and spsa_g is None:
                _r0 = next(iter(m_pts))
                _x00 = m_pts[_r0][0]
                _ttt_b, _, _ = self._global_rollout_metrics_with_metering(
                    state, previous, forecast, _r0, _x00,
                )
                fd3_base_ttt = float(_ttt_b)
            fd3_cliff_up: Dict[str, bool] = {}
            for ramp, (x0, m_lo, m_hi) in m_pts.items():
                span = m_hi - m_lo
                if span <= 1.0e-9:
                    g_ext = 0.0
                    m_certs[ramp] = False  # 측정 불능이면 방류 증가 미인증(보수적)
                else:
                    if spsa_g is not None:
                        g_i = float(spsa_g.get(("meter", ramp), 0.0))
                        bar_hi = bar_lo = 0.0
                        # cert: 전 ramp 동시 +δ rollout의 최대 밀도(개별보다 보수적).
                        rho_hi = float(spsa_rho_joint)
                    else:
                        ttt_hi, bar_hi, rho_hi = self._global_rollout_metrics_with_metering(
                            state, previous, forecast, ramp, m_hi,
                        )
                        ttt_lo, bar_lo, _ = self._global_rollout_metrics_with_metering(
                            state, previous, forecast, ramp, m_lo,
                        )
                        g_i = (ttt_hi - ttt_lo) / span
                        if fd3_base_ttt is not None:
                            # 곡률 |hi+lo−2·base|: 0이면 선형. 문턱(breakdown) 구조면
                            # 한쪽 팔만 급등해 곡률이 1차 전폭을 압도한다.
                            curv = abs(ttt_hi + ttt_lo - 2.0 * fd3_base_ttt)
                            meta[f"wu_b3_meter_fd3_{ramp}_lo"] = float(ttt_lo)
                            meta[f"wu_b3_meter_fd3_{ramp}_base"] = float(fd3_base_ttt)
                            meta[f"wu_b3_meter_fd3_{ramp}_hi"] = float(ttt_hi)
                            meta[f"wu_b3_meter_fd3_{ramp}_curv"] = float(curv)
                            # 선형화 불가 깃발: 2차 잔차 > 1차 전폭 |hi−lo| — secant 대표성 상실.
                            meta[f"wu_b3_meter_fd3_{ramp}_nonlin"] = float(
                                curv > abs(ttt_hi - ttt_lo) + 1.0e-9
                            )
                            # 위쪽 절벽: TTT(+Δ) ≫ base ≈ TTT(−Δ) — 방류 증가 방향만 급락.
                            fd3_cliff_up[ramp] = bool(
                                (ttt_hi - fd3_base_ttt)
                                > 2.0 * abs(fd3_base_ttt - ttt_lo) + 1.0
                            )
                            meta[f"wu_b3_meter_fd3_{ramp}_cliffup"] = float(
                                fd3_cliff_up[ramp]
                            )
                    cost_lo, cost_hi = local_m[ramp]
                    g_ext = g_i - (cost_hi - cost_lo) / span
                    if self.barrier_price_enabled:
                        # barrier는 leader 전용 목적항 — follower own-TTS에 없으므로
                        # d_local 차감 없이 그대로 합산(B4).
                        g_ext += (bar_hi - bar_lo) / span
                    # B3CERT: +δ rollout의 본선 최대 예측 밀도가 안전선 아래일 때만
                    # 방류 증가 방향을 인증(절벽 비가역 — 사후 검증 불가, 사전 인증만).
                    m_certs[ramp] = bool(rho_hi < rho_cert_limit)
                m_prices[ramp] = float(g_ext)
                m_refs[ramp] = float(x0)
                meta[f"wu_b3_meter_price_{ramp}"] = float(g_ext)
                meta[f"wu_b3_meter_price_ref_{ramp}"] = float(x0)
                meta[f"wu_b3cert_{ramp}"] = float(m_certs[ramp])
            # 링크별 '양 램프 동시 위쪽 절벽' 깃발(2026-07-14) — 大δ 과대 신호:
            # 두 가격이 다 양수(전 램프 조임) → 과소방류 나선 위험의 사전 서명.
            if fd3_base_ttt is not None:
                for _link in net.freeway_links:
                    _ramps_l = [
                        r for r in m_pts if net.ramp_to_freeway.get(r) == _link
                    ]
                    if _ramps_l:
                        meta[f"wu_b3_cliff_both_{_link}"] = float(
                            all(fd3_cliff_up.get(r, False) for r in _ramps_l)
                        )
            follower.metering_marginal_price = m_prices
            follower.metering_marginal_price_ref = m_refs
            follower.metering_release_certified = (
                m_certs if self.metering_release_cert_enabled else None
            )
            follower.metering_marginal_price_trust_frac = (
                float(self.metering_price_trust_frac)
                if self.metering_price_trust_frac is not None else None
            )

        # ---- offset 채널(F3): g_ext = d(전역TTT)/d(off) − d(국소 phased)/d(off) ----
        # δ·trust = 후보 그리드 1칸(cycle/8) — 허용 이동폭만큼 측정. offset은 원형이라
        # 클램프 불필요(±δ가 항상 유효), FD span = 2δ 고정.
        if self.offset_price_enabled and op_offset:
            meta["wu_f3_offset_price_enabled"] = 1.0
            delta_o = (
                float(self.offset_price_delta_sec)
                if self.offset_price_delta_sec is not None else cycle / 8.0
            )
            # 재선형화 걷기는 δ가 grid 한 칸(cycle/8)일 때만 유효(앵커 이웃 a±δ가 grid 후보와
            # 정렬돼 국소비용을 신호당 1회 채점으로 공유). δ가 어긋나면 단일 선형화로 폴백.
            inner_k = max(0, int(getattr(self, "offset_price_inner_iters", 0)))
            grid_aligned = abs(delta_o - cycle / 8.0) <= 1.0e-6
            do_walk = inner_k > 0 and spsa_g is None and grid_aligned
            o_prices: Dict[str, float] = {}
            o_refs: Dict[str, float] = {}
            if do_walk:
                # 국소 phased 비용은 offset 불변 platoon profile 기반 → 8개 grid 후보를
                # 신호당 한 번에 채점하고(재선형화마다 재계산 불필요) 앵커가 밟는 이웃만
                # 전역 rollout을 캐시로 늘린다.
                o_weight = float(getattr(follower, "offset_marginal_price_weight", 1.0))
                grid_offsets = [
                    (frac * cycle) % cycle for frac in follower.offset_fractions
                ]
                grid_keys = [round(o, 6) for o in grid_offsets]
                local_all = follower.local_offset_costs(
                    {signal: grid_offsets for signal in op_offset},
                    state, previous, forecast[0],
                )
                for signal, off0 in op_offset.items():
                    lc_map = {
                        grid_keys[i]: float(local_all[signal][i])
                        for i in range(len(grid_offsets))
                    }
                    anchor, iters, g_ext = self._offset_price_relinearize_walk(
                        state, previous, forecast, signal, float(off0) % cycle,
                        delta_o, inner_k, grid_offsets, lc_map, o_weight, cycle,
                    )
                    o_prices[signal] = float(g_ext)
                    o_refs[signal] = float(anchor)
                    meta[f"wu_f3_offset_price_{signal}"] = float(g_ext)
                    meta[f"wu_f3_offset_ref_{signal}"] = float(anchor)
                    meta[f"wu_f3_offset_inner_iters_{signal}"] = float(iters)
                    meta[f"wu_f3_offset_walk_from_{signal}"] = float(off0) % cycle
            else:
                # 단일 선형화(K=0 ∨ SPSA ∨ δ 비정렬) — 기존 비트동일 경로.
                o_requests = {
                    signal: [(off - delta_o) % cycle, (off + delta_o) % cycle]
                    for signal, off in op_offset.items()
                }
                local_o = follower.local_offset_costs(
                    o_requests, state, previous, forecast[0]
                )
                for signal, off0 in op_offset.items():
                    o_lo, o_hi = o_requests[signal]
                    span = 2.0 * delta_o
                    if spsa_g is not None:
                        g_i = float(spsa_g.get(("offset", signal), 0.0))
                    else:
                        ttt_hi = self._global_rollout_ttt_with_offset(
                            state, previous, forecast, signal, o_hi,
                        )
                        ttt_lo = self._global_rollout_ttt_with_offset(
                            state, previous, forecast, signal, o_lo,
                        )
                        g_i = (ttt_hi - ttt_lo) / span
                    cost_lo, cost_hi = local_o[signal]
                    g_ext = g_i - (cost_hi - cost_lo) / span
                    o_prices[signal] = float(g_ext)
                    o_refs[signal] = float(off0)
                    meta[f"wu_f3_offset_price_{signal}"] = float(g_ext)
                    meta[f"wu_f3_offset_ref_{signal}"] = float(off0)
            follower.offset_marginal_price = o_prices
            follower.offset_marginal_price_ref = o_refs
            follower.offset_marginal_price_trust_sec = delta_o

        # ---- J1 joint offset 패턴(결합 후보 통째 평가 — F3 편미분=0의 처방) ----
        if self.offset_joint_enabled:
            step_o = (
                float(self.offset_joint_step_sec)
                if self.offset_joint_step_sec is not None else cycle / 8.0
            )
            joint_signals = [
                s for s in net.signals if not follower._local_models[s].has_ramps
            ]
            grid = (0.0, step_o % cycle, (cycle - step_o) % cycle)
            patterns: List[Dict[str, float]] = [{}]
            for s in joint_signals:
                patterns = [
                    dict(pat, **{s: g}) for pat in patterns for g in grid
                ]
            # 직전 committed 패턴도 후보에 포함(히스테리시스 — 채택된 조합의 유지 평가).
            patterns.append({
                s: float(previous.offsets.get(s, 0.0)) % cycle for s in joint_signals
            })
            def _pattern_ttt(pat: Dict[str, float], green_shift: float = 0.0) -> float:
                control_p = previous.copy()
                control_p.offsets = dict(previous.offsets)
                for s, off in pat.items():
                    control_p.offsets[s] = float(off)
                if abs(green_shift) > 1.0e-9:
                    control_p.green_times = dict(previous.green_times)
                    for s in joint_signals:
                        p1 = float(previous.green_times.get(f"{s}_p1", total / 2.0))
                        p1n = clamp(p1 + green_shift)
                        control_p.green_times[f"{s}_p1"] = p1n
                        control_p.green_times[f"{s}_p2"] = total - p1n
                _, ttt_p = self._predict(state, control_p, forecast)
                return float(ttt_p)

            scored = []
            zero_ttt = float("inf")
            for pat in patterns:
                ttt_p = _pattern_ttt(pat)
                if all(abs(v) < 1.0e-9 for v in pat.values()):
                    zero_ttt = min(zero_ttt, ttt_p)
                scored.append((ttt_p, pat))
            scored.sort(key=lambda t: t[0])
            # 2단계 (green, offset) 결합: 상위 K 패턴에 green 공동 이동(±δ) 변형을 평가 —
            # 패턴 점수 = min(기본, 변형들). green 결합으로만 이기는 조합도 채택 가능.
            g_delta = float(self.offset_joint_green_delta_sec)
            top_k = max(1, int(self.offset_joint_green_top_k))
            best_ttt, best_pat, best_shift = float("inf"), {}, 0.0
            for ttt_base, pat in scored[:top_k]:
                variants = [(ttt_base, 0.0)]
                if g_delta > 1.0e-9:
                    variants.append((_pattern_ttt(pat, +g_delta), +g_delta))
                    variants.append((_pattern_ttt(pat, -g_delta), -g_delta))
                v_ttt, v_shift = min(variants, key=lambda t: t[0])
                if v_ttt < best_ttt:
                    best_ttt, best_pat, best_shift = v_ttt, dict(pat), v_shift
            gain = zero_ttt - best_ttt
            adopt = gain > float(self.offset_joint_margin) * max(zero_ttt, 1.0e-9)
            directive = (
                best_pat if adopt else {s: 0.0 for s in joint_signals}
            )
            follower.offset_directive = directive
            # joint 가치 신호(관찰 목적): 패턴 예측 이득·채택 조합·green 결합 기여 기록 —
            # per-signal 편미분이 0이던 자리에서 조합 방향의 유한 차이가 갖는 값.
            # (green 이동 자체는 directive하지 않는다 — 채택된 offset 운영점에서 다음
            # refresh의 green 가격이 재선형화되며 trust 보폭으로 따라간다.)
            meta["wu_j_off_gain"] = float(gain)
            meta["wu_j_off_adopted"] = float(adopt)
            meta["wu_j_green_shift_at_best"] = float(best_shift)
            for s, off in directive.items():
                meta[f"wu_j_off_{s}"] = float(off)

        # ---- LEADER-OFFSET 채널(2026-07-07): offset 소유권을 leader로 이전 ----
        # J1과 별개 채널(전 신호·finer grid·corridor lag seed·follower 탐색 완전 OFF).
        # 최선 offset 패턴을 offset_directive로 하달 → follower는 그 값으로 동결(탐색 없음).
        if self.leader_offset_enabled:
            if self.leader_offset_method == "mpc":
                lead_directive, lead_diag = self._solve_leader_offset(
                    state, previous, forecast,
                )
            else:
                raise NotImplementedError(
                    f"leader_offset_method={self.leader_offset_method!r} 미구현 "
                    "(현재 'mpc'만 지원 — maxband_lp는 2B, 별도 스펙 필요)."
                )
            follower.offset_directive = lead_directive
            meta["wu_lead_off_enabled"] = 1.0
            meta["wu_lead_off_gain"] = float(lead_diag["gain"])
            meta["wu_lead_off_adopted"] = float(lead_diag["adopt"])
            meta["wu_lead_off_green_shift"] = float(lead_diag["green_shift"])
            meta["wu_lead_off_n_eval"] = float(lead_diag["n_eval"])
            for s, off in lead_directive.items():
                meta[f"wu_lead_off_{s}"] = float(off)

        # ---- VSL 채널(B3, E2로 g_ext화: g_ext = g_i − d_local, 기본 OFF) ----
        # 기존 raw g_i는 follower가 own-TTS로 이미 보는 성분을 이중계상(알려진 결함,
        # "활성화 전 g_ext화가 선행 과제"). vsl_override 프리미티브(local_vsl_costs)로
        # d_local(고정 VSL 벡터 own-TTS의 유한차분)을 차감해 다른 채널과 규약 정렬.
        if self.vsl_price_enabled and op_vsl:
            meta["wu_b3_vsl_price_enabled"] = 1.0
            delta_v = float(self.vsl_price_delta_kmh)
            n_seg_v = int(net.freeway_segments_per_link)
            # per-link 고정벡터 요청: seg i의 ±δ 두 벡터씩(순서 [hi_i, lo_i] × seg).
            v_corners: Dict[str, tuple] = {}
            v_requests: Dict[str, List[List[float]]] = {}
            for link in net.freeway_links:
                base = [
                    float(op_vsl.get(f"{link}__seg{i}", vsl_upper)) for i in range(n_seg_v)
                ]
                reqs: List[List[float]] = []
                for i in range(n_seg_v):
                    key = f"{link}__seg{i}"
                    if key not in op_vsl:
                        continue
                    x0 = base[i]
                    v_hi = min(vsl_upper, x0 + delta_v)
                    v_lo = max(vsl_lower, x0 - delta_v)
                    v_corners[key] = (x0, v_lo, v_hi, link, len(reqs))
                    hi_vec = list(base)
                    hi_vec[i] = v_hi
                    lo_vec = list(base)
                    lo_vec[i] = v_lo
                    reqs.extend([hi_vec, lo_vec])
                if reqs:
                    v_requests[link] = reqs
            local_v = follower.local_vsl_costs(v_requests, state, previous, forecast[0])
            v_prices: Dict[str, float] = {}
            v_refs: Dict[str, float] = {}
            for key, (x0, v_lo, v_hi, link, req_idx) in v_corners.items():
                span = v_hi - v_lo
                if span <= 1.0e-9:
                    g_ext = 0.0
                else:
                    if spsa_g is not None:
                        g_i = float(spsa_g.get(("vsl", key), 0.0))
                    else:
                        ttt_hi = self._global_rollout_ttt_with_vsl(
                            state, previous, forecast, link, key, v_hi, vsl_upper,
                        )
                        ttt_lo = self._global_rollout_ttt_with_vsl(
                            state, previous, forecast, link, key, v_lo, vsl_upper,
                        )
                        g_i = (ttt_hi - ttt_lo) / span
                    lc = local_v.get(link, [])
                    if req_idx + 1 < len(lc):
                        d_local = (lc[req_idx] - lc[req_idx + 1]) / span
                    else:
                        d_local = 0.0
                    g_ext = g_i - d_local
                v_prices[key] = float(g_ext)
                v_refs[key] = float(x0)
                # 가시성(2026-07-14): per-seg VSL 가격을 meta로 export(기존엔 enabled
                # 플래그만 나가 5채널 감사에서 vsl이 '안 보임'이었다 — 진단 전용).
                meta[f"wu_b3_vsl_price_{key}"] = float(g_ext)
                meta[f"wu_b3_vsl_price_ref_{key}"] = float(x0)
            follower.vsl_marginal_price = v_prices
            follower.vsl_marginal_price_ref = v_refs
            # PRICE-TR: VSL trust region 하달(±kmh — 가격이 측정된 이웃 밖 후보 제외).
            follower.vsl_marginal_price_trust_kmh = (
                float(self.vsl_price_trust_kmh)
                if self.vsl_price_trust_kmh is not None else None
            )

        # ---- CROSS-GATE(2026-07-15): cross는 capacity-drop 문턱 근방에서만 이득 ----
        # 8셀 실측: 절벽 무 → cross OFF가 −164~−420 이득 / 절벽 유 → cross ON이 +27~+380 이득.
        # 분리 축은 부하가 아니라 절벽(170_w vs 170_incident_w: 부하 13,028 동일, 부호 반대).
        # 게이트 신호 = 위에서 이미 계산된 wu_b3_cliff_both_*(신규 문턱·파라미터 없음).
        # 기본 OFF(미지정) = 게이트 없이 현행 동작 → 비트동일.
        cross_gate_on = True
        if self.cross_cliff_gate_enabled:
            _cliff_flags = [
                v for k, v in meta.items() if k.startswith("wu_b3_cliff_both_")
            ]
            # 어느 방향이든 양 램프 동시 위쪽 절벽이면 문턱 근방으로 보고 cross 허용.
            cross_gate_on = any(float(v) > 0.5 for v in _cliff_flags)
            meta["wu_joint_cross_gate_on"] = float(cross_gate_on)
            if not cross_gate_on:
                # ★None이 아니라 {}: None으로 지우면 L1129 refresh 판정의
                # `enabled and price is None`이 매 스텝 참이 되어 **전 채널(green·metering·VSL)
                # 가격 갱신 주기**까지 바뀐다. 1차 게이트 런이 이걸로 오염됨 — 0스텝 발화인
                # 155_incident가 cross OFF와 값이 갈렸다(1,471.6 vs 1,462.7).
                # {}면 refresh 판정은 그대로이고, follower는 `signal in {}`(L1531) /
                # truthy 검사(L2230·L2513)에서 자연히 cross를 건너뛴다.
                follower.green_offset_cross_price = {}
                follower.vsl_meter_cross_price = {}

        # ---- JOINT green×offset cross(2026-07-09): h_ext = h_global − h_local, 4-corner ----
        # non-ramp 신호만(ramp는 storage 동역학 복잡 → follower joint 제외). δp=green delta,
        # δo=cycle/8(offset 격자 1칸). h는 mixed 2차 편미분(부호 유지 — cross 곡률).
        if self.green_offset_cross_price_enabled and cross_gate_on:
            meta["wu_joint_go_cross_enabled"] = 1.0
            dp = float(self.signal_price_delta_sec)
            do = (
                float(self.green_offset_cross_offset_delta_sec)
                if self.green_offset_cross_offset_delta_sec is not None
                else cycle / 8.0
            )
            meta["wu_joint_go_cross_dp"] = float(dp)
            meta["wu_joint_go_cross_do"] = float(do)
            nonramp = [s for s in net.signals if not follower._local_models[s].has_ramps]
            # 신호별 4-corner (p1, offset) 쌍 구성.
            gp_pairs: Dict[str, List[tuple]] = {}
            gp_ref: Dict[str, tuple] = {}
            gp_corners: Dict[str, tuple] = {}
            for s in nonramp:
                p0 = clamp(previous.green_times.get(f"{s}_p1", total / 2.0))
                o0 = float(previous.offsets.get(s, 0.0)) % cycle
                p_hi = clamp(p0 + dp)
                p_lo = clamp(p0 - dp)
                o_hi = (o0 + do) % cycle
                o_lo = (o0 - do) % cycle
                gp_ref[s] = (float(p0), float(o0))
                gp_corners[s] = (p_lo, p_hi, o_lo, o_hi)
                # (p_hi,o_hi),(p_hi,o_lo),(p_lo,o_hi),(p_lo,o_lo)
                gp_pairs[s] = [(p_hi, o_hi), (p_hi, o_lo), (p_lo, o_hi), (p_lo, o_lo)]
            local_go = follower.local_green_offset_costs(gp_pairs, state, previous, forecast[0])
            cross_prices: Dict[str, float] = {}
            cross_refs: Dict[str, tuple] = {}
            for s in nonramp:
                p_lo, p_hi, o_lo, o_hi = gp_corners[s]
                dpp = p_hi - p_lo
                # offset 원형 변위(o_hi−o_lo → 2·do 근사).
                doo = ((o_hi - o_lo + cycle / 2.0) % cycle) - cycle / 2.0
                denom = dpp * doo
                if abs(denom) <= 1.0e-9:
                    cross_prices[s] = 0.0
                    cross_refs[s] = gp_ref[s]
                    continue
                t_pp = self._global_rollout_ttt_with_green_offset(state, previous, forecast, s, p_hi, o_hi)
                t_pm = self._global_rollout_ttt_with_green_offset(state, previous, forecast, s, p_hi, o_lo)
                t_mp = self._global_rollout_ttt_with_green_offset(state, previous, forecast, s, p_lo, o_hi)
                t_mm = self._global_rollout_ttt_with_green_offset(state, previous, forecast, s, p_lo, o_lo)
                h_global = (t_pp - t_pm - t_mp + t_mm) / denom
                lc = local_go.get(s, [0.0, 0.0, 0.0, 0.0])
                h_local = (lc[0] - lc[1] - lc[2] + lc[3]) / denom
                h_ext = h_global - h_local
                cross_prices[s] = float(h_ext)
                cross_refs[s] = gp_ref[s]
                meta[f"wu_joint_go_cross_{s}"] = float(h_ext)
            follower.green_offset_cross_price = cross_prices
            follower.green_offset_cross_ref = cross_refs
            follower.green_offset_cross_weight = float(self.green_offset_cross_weight)

        # ---- JOINT vsl×metering cross(2026-07-09): h_ext = h_global − h_local, ramp별 4-corner ----
        # δm=metering delta, δv=vsl delta. link-binding VSL(v0=이 link 최소 seg vsl)와 ramp
        # metering(m0)의 mixed 2차. h_local은 vsl_override 고정 own-TTS 4-corner로 계산.
        if self.vsl_meter_cross_price_enabled and cross_gate_on and vsl_values:
            meta["wu_joint_vm_cross_enabled"] = 1.0
            dm = float(self.metering_price_delta_veh_h)
            dv = float(self.vsl_price_delta_kmh)
            vm_pairs: Dict[str, List[tuple]] = {}
            vm_ref: Dict[str, tuple] = {}
            vm_corners: Dict[str, tuple] = {}
            for r in net.ramps:
                link = net.ramp_to_freeway.get(r)
                if link is None:
                    continue
                m0 = min(max(float(previous.ramp_metering.get(r, ramp_caps[r])), 0.0), ramp_caps[r])
                v0 = float(segment_vsl(previous, link, 0, self.cfg))
                m_hi = min(ramp_caps[r], m0 + dm)
                m_lo = max(0.0, m0 - dm)
                v_hi = min(vsl_upper, v0 + dv)
                v_lo = max(vsl_lower, v0 - dv)
                vm_ref[r] = (float(m0), float(v0))
                vm_corners[r] = (m_lo, m_hi, v_lo, v_hi, link)
                vm_pairs[r] = [(m_hi, v_hi), (m_hi, v_lo), (m_lo, v_hi), (m_lo, v_lo)]
            local_vm = follower.local_vsl_meter_costs(vm_pairs, state, previous, forecast[0])
            vm_prices: Dict[str, float] = {}
            vm_refs: Dict[str, tuple] = {}
            for r, (m_lo, m_hi, v_lo, v_hi, link) in vm_corners.items():
                denom = (m_hi - m_lo) * (v_hi - v_lo)
                if abs(denom) <= 1.0e-9:
                    vm_prices[r] = 0.0
                    vm_refs[r] = vm_ref[r]
                    continue
                t_pp = self._global_rollout_ttt_with_vsl_meter(state, previous, forecast, r, link, m_hi, v_hi, vsl_upper)
                t_pm = self._global_rollout_ttt_with_vsl_meter(state, previous, forecast, r, link, m_hi, v_lo, vsl_upper)
                t_mp = self._global_rollout_ttt_with_vsl_meter(state, previous, forecast, r, link, m_lo, v_hi, vsl_upper)
                t_mm = self._global_rollout_ttt_with_vsl_meter(state, previous, forecast, r, link, m_lo, v_lo, vsl_upper)
                h_global = (t_pp - t_pm - t_mp + t_mm) / denom
                lc = local_vm.get(r, [0.0, 0.0, 0.0, 0.0])
                h_local = (lc[0] - lc[1] - lc[2] + lc[3]) / denom
                vm_prices[r] = float(h_global - h_local)
                vm_refs[r] = vm_ref[r]
                meta[f"wu_joint_vm_cross_{r}"] = float(h_global - h_local)
            follower.vsl_meter_cross_price = vm_prices
            follower.vsl_meter_cross_ref = vm_refs
            follower.vsl_meter_cross_weight = float(self.vsl_meter_cross_weight)

        self._signal_price_refresh_count += 1
        meta["wu_b2_price_refreshed"] = 1.0
        meta["wu_b2_price_refresh_count"] = float(self._signal_price_refresh_count)
        meta["wu_price_rollout_count"] = float(self._price_rollout_count)
        self._signal_price_meta = meta

    def _compute_prices_lite(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        follower: "WuFaithfulFollower",
        net,
        meta: Dict[str, float],
        op_green: Dict[str, float],
        clamp,
        total: float,
        ramp_caps: Dict[str, float],
        op_meter: Dict[str, float],
        vsl_values: List[float],
        vsl_lower: float,
        vsl_upper: float,
        op_vsl: Dict[str, float],
        cycle: float,
    ) -> None:
        """B-패키지 가격 계산 — rollout 예산을 상수 절감(검증: sweet_190 A/B런).

        (B2) one-sided FD: 공용 baseline J0 1회 + lever당 1회 (양측 2회 대비 절반).
        (B1) cross 스텐실 재활용: h ≈ [J(a⁺,b⁺) − J(a⁺) − J(b⁺) + J0]/(Δa·Δb) —
             J(a⁺)·J(b⁺)는 per-lever FD가 이미 계산한 점 → 쌍당 신규 1~2회.
        (B3) 얕은 가격 rollout(H+1): 가격은 배분 신호라 국소·단기(level 고원 실측이
             방증). 후보 채점은 기존 깊이(H+D)+far 유지.
        flagship(5신호·4ramp·2링크·4seg) 기준: 62회@H+D → ~30회@H+1 ≈ rollout·초 −68%.
        한계: B4 barrier·B3CERT 미지원(각각 기본 OFF·split 모드에서 무관 — level 모드
        재현 시에는 price_lite를 끌 것)."""
        depth_p = int(self.cfg.mpc.horizon_steps) + 1
        # ---- 공용 baseline J0 ----
        self._price_rollout_count += 1
        base_states, base_raw = self._predict(
            state, previous, forecast, depth_override=depth_p,
        )
        j0 = self._price_ttt(base_states, base_raw)

        # ---- green (one-sided) ----
        green_pt: Dict[str, tuple] = {}
        if self.signal_price_enabled and op_green:
            dsec = float(self.signal_price_delta_sec)
            pts: Dict[str, tuple] = {}
            requests: Dict[str, List[float]] = {}
            for signal, p0 in op_green.items():
                p_pl = clamp(p0 + dsec)
                if abs(p_pl - p0) <= 1.0e-9:
                    p_pl = clamp(p0 - dsec)
                pts[signal] = (p0, p_pl)
                requests[signal] = [p0, p_pl]
            local_costs = follower.local_green_costs(requests, state, previous, forecast[0])
            prices: Dict[str, float] = {}
            refs: Dict[str, float] = {}
            for signal, (p0, p_pl) in pts.items():
                dp = p_pl - p0
                if abs(dp) <= 1.0e-9:
                    g_ext = 0.0
                    green_pt[signal] = (p0, p0, j0)
                else:
                    j_p = self._global_rollout_ttt_with_green(
                        state, previous, forecast, signal, p_pl, depth_override=depth_p,
                    )
                    c0, c_pl = local_costs[signal]
                    g_ext = (j_p - j0) / dp - (c_pl - c0) / dp
                    green_pt[signal] = (p0, p_pl, j_p)
                prices[signal] = float(g_ext)
                refs[signal] = float(p0)
                meta[f"wu_b2_price_{signal}"] = float(g_ext)
                meta[f"wu_b2_price_ref_{signal}"] = float(p0)
            follower.signal_marginal_price = prices
            follower.signal_marginal_price_ref = refs
            follower.signal_marginal_price_weight = float(self.signal_price_weight)
            follower.signal_marginal_price_trust_sec = (
                float(self.signal_price_trust_sec)
                if self.signal_price_trust_sec is not None else None
            )
            meta["wu_b2_price_delta_sec"] = dsec

        # ---- metering (one-sided; cert 미지원 → None) ----
        meter_pt: Dict[str, tuple] = {}
        if self.metering_price_enabled and op_meter:
            dm = float(self.metering_price_delta_veh_h)
            m_requests: Dict[str, List[float]] = {}
            m_pts: Dict[str, tuple] = {}
            for ramp, m0 in op_meter.items():
                cap = ramp_caps[ramp]
                m_pl = min(cap, m0 + dm)
                if abs(m_pl - m0) <= 1.0e-9:
                    m_pl = max(0.0, m0 - dm)
                m_pts[ramp] = (m0, m_pl)
                m_requests[ramp] = [m0, m_pl]
            local_m = follower.local_metering_costs(m_requests, state, previous, forecast[0])
            m_prices: Dict[str, float] = {}
            m_refs: Dict[str, float] = {}
            for ramp, (m0, m_pl) in m_pts.items():
                dmm = m_pl - m0
                if abs(dmm) <= 1.0e-9:
                    g_ext = 0.0
                    meter_pt[ramp] = (m0, m0, j0)
                else:
                    j_m, _bar, _rho = self._global_rollout_metrics_with_metering(
                        state, previous, forecast, ramp, m_pl, depth_override=depth_p,
                    )
                    lc = local_m[ramp]
                    g_ext = (j_m - j0) / dmm - (lc[1] - lc[0]) / dmm
                    meter_pt[ramp] = (m0, m_pl, j_m)
                m_prices[ramp] = float(g_ext)
                m_refs[ramp] = float(m0)
            follower.metering_marginal_price = m_prices
            follower.metering_marginal_price_ref = m_refs
            follower.metering_marginal_price_trust_frac = (
                float(self.metering_price_trust_frac)
                if self.metering_price_trust_frac is not None else None
            )
            follower.metering_release_certified = None
            meta["wu_b3_meter_price_enabled"] = 1.0

        # ---- VSL (one-sided, E2 차감) ----
        vsl_link_pt: Dict[str, tuple] = {}
        if self.vsl_price_enabled and vsl_values and op_vsl:
            dv = float(self.vsl_price_delta_kmh)
            n_seg_v = int(net.freeway_segments_per_link)
            v_corners: Dict[str, tuple] = {}
            v_requests: Dict[str, List[List[float]]] = {}
            for link in net.freeway_links:
                base = [
                    float(op_vsl.get(f"{link}__seg{i}", vsl_upper)) for i in range(n_seg_v)
                ]
                reqs: List[List[float]] = [list(base)]
                for i in range(n_seg_v):
                    key = f"{link}__seg{i}"
                    if key not in op_vsl:
                        continue
                    x0 = base[i]
                    v_pl = max(vsl_lower, x0 - dv)
                    if abs(v_pl - x0) <= 1.0e-9:
                        v_pl = min(vsl_upper, x0 + dv)
                    vec = list(base)
                    vec[i] = v_pl
                    v_corners[key] = (x0, v_pl, link, len(reqs))
                    reqs.append(vec)
                v_requests[link] = reqs
            local_v = follower.local_vsl_costs(v_requests, state, previous, forecast[0])
            v_prices: Dict[str, float] = {}
            v_refs: Dict[str, float] = {}
            for key, (x0, v_pl, link, ri) in v_corners.items():
                dvv = v_pl - x0
                if abs(dvv) <= 1.0e-9:
                    g_ext = 0.0
                else:
                    j_v = self._global_rollout_ttt_with_vsl(
                        state, previous, forecast, link, key, v_pl, vsl_upper,
                        depth_override=depth_p,
                    )
                    lc = local_v.get(link, [])
                    d_local = (lc[ri] - lc[0]) / dvv if ri < len(lc) else 0.0
                    g_ext = (j_v - j0) / dvv - d_local
                v_prices[key] = float(g_ext)
                v_refs[key] = float(x0)
            follower.vsl_marginal_price = v_prices
            follower.vsl_marginal_price_ref = v_refs
            follower.vsl_marginal_price_trust_kmh = (
                float(self.vsl_price_trust_kmh)
                if self.vsl_price_trust_kmh is not None else None
            )
            meta["wu_b3_vsl_price_enabled"] = 1.0

        # ---- cross green×offset (B1 재활용: 쌍당 신규 2회 — J(o⁺), J(p⁺,o⁺)) ----
        if self.green_offset_cross_price_enabled and green_pt:
            do = cycle / 8.0
            nonramp = [s for s in net.signals if not follower._local_models[s].has_ramps]
            gp_pairs: Dict[str, List[tuple]] = {}
            corners: Dict[str, tuple] = {}
            for s in nonramp:
                p0, p_pl, j_p = green_pt.get(
                    s, (clamp(previous.green_times.get(f"{s}_p1", total / 2.0)), None, None)
                )
                o0 = float(previous.offsets.get(s, 0.0)) % cycle
                o_pl = (o0 + do) % cycle
                corners[s] = (p0, p_pl, o0, o_pl, j_p)
                gp_pairs[s] = [(p0, o0), (p_pl if p_pl is not None else p0, o0),
                               (p0, o_pl), (p_pl if p_pl is not None else p0, o_pl)]
            local_go = follower.local_green_offset_costs(gp_pairs, state, previous, forecast[0])
            cross_prices: Dict[str, float] = {}
            cross_refs: Dict[str, tuple] = {}
            for s, (p0, p_pl, o0, o_pl, j_p) in corners.items():
                dp = (p_pl - p0) if p_pl is not None else 0.0
                doo = ((o_pl - o0 + cycle / 2.0) % cycle) - cycle / 2.0
                denom = dp * doo
                if abs(denom) <= 1.0e-9 or j_p is None:
                    cross_prices[s] = 0.0
                    cross_refs[s] = (float(p0), float(o0))
                    continue
                j_o = self._global_rollout_ttt_with_offset(
                    state, previous, forecast, s, o_pl, depth_override=depth_p,
                )
                j_po = self._global_rollout_ttt_with_green_offset(
                    state, previous, forecast, s, p_pl, o_pl, depth_override=depth_p,
                )
                h_global = (j_po - j_p - j_o + j0) / denom
                lc = local_go.get(s, [0.0, 0.0, 0.0, 0.0])
                h_local = (lc[3] - lc[1] - lc[2] + lc[0]) / denom
                cross_prices[s] = float(h_global - h_local)
                cross_refs[s] = (float(p0), float(o0))
                meta[f"wu_joint_go_cross_{s}"] = cross_prices[s]
            follower.green_offset_cross_price = cross_prices
            follower.green_offset_cross_ref = cross_refs
            follower.green_offset_cross_weight = float(self.green_offset_cross_weight)
            meta["wu_joint_go_cross_enabled"] = 1.0

        # ---- cross vsl×metering (B1 재활용: 링크당 J(v_s) 1회 + ramp당 J(m⁺,v_s) 1회) ----
        if self.vsl_meter_cross_price_enabled and vsl_values and meter_pt:
            dv = float(self.vsl_price_delta_kmh)
            vm_prices: Dict[str, float] = {}
            vm_refs: Dict[str, tuple] = {}
            j_vlink: Dict[str, tuple] = {}
            for r in net.ramps:
                link = net.ramp_to_freeway.get(r)
                if link is None or r not in meter_pt:
                    continue
                m0, m_pl, j_m = meter_pt[r]
                v0 = float(segment_vsl(previous, link, 0, self.cfg))
                v_s = max(vsl_lower, v0 - dv)
                if abs(v_s - v0) <= 1.0e-9:
                    v_s = min(vsl_upper, v0 + dv)
                dmm = m_pl - m0
                dvv = v_s - v0
                denom = dmm * dvv
                if abs(denom) <= 1.0e-9:
                    vm_prices[r] = 0.0
                    vm_refs[r] = (float(m0), float(v0))
                    continue
                if link not in j_vlink or abs(j_vlink[link][0] - v_s) > 1.0e-9:
                    jv = self._global_rollout_ttt_with_vsl_meter(
                        state, previous, forecast, r, link, m0, v_s, vsl_upper,
                        depth_override=depth_p,
                    )
                    j_vlink[link] = (v_s, jv)
                j_v = j_vlink[link][1]
                j_mv = self._global_rollout_ttt_with_vsl_meter(
                    state, previous, forecast, r, link, m_pl, v_s, vsl_upper,
                    depth_override=depth_p,
                )
                lc = follower.local_vsl_meter_costs(
                    {r: [(m0, v0), (m_pl, v0), (m0, v_s), (m_pl, v_s)]},
                    state, previous, forecast[0],
                )[r]
                h_global = (j_mv - j_m - j_v + j0) / denom
                h_local = (lc[3] - lc[1] - lc[2] + lc[0]) / denom
                vm_prices[r] = float(h_global - h_local)
                vm_refs[r] = (float(m0), float(v0))
                meta[f"wu_joint_vm_cross_{r}"] = vm_prices[r]
            follower.vsl_meter_cross_price = vm_prices
            follower.vsl_meter_cross_ref = vm_refs
            follower.vsl_meter_cross_weight = float(self.vsl_meter_cross_weight)
            meta["wu_joint_vm_cross_enabled"] = 1.0

    def _pfo_incumbent_fallback_enabled(self) -> bool:
        return bool(getattr(self.cfg.mpc, "stackelberg_enable_pfo_incumbent", True))

    @staticmethod
    def _finite(value: float) -> bool:
        return value == value and value not in (float("inf"), -float("inf"))

    def _pfo_equivalent_action(
        self,
        control: ControlAction,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
    ) -> tuple[LeaderAction, Dict[str, float]]:
        # PFO response를 leader local search의 target 좌표계로 변환한다.
        diagnostics = dict(control.diagnostics)
        raw_np = float(diagnostics.get("wu_faithful_sum_nin", control.N_P_star))
        if not self._finite(raw_np):
            raw_np = float(control.N_P_star)
        raw_nuf = float(sum(float(v) for v in control.ramp_metering.values()))
        if raw_nuf <= 1.0e-9 and self._finite(float(control.N_UF_star)):
            raw_nuf = float(control.N_UF_star)
        bounds = self.leader._candidate_bounds(state, previous, forecast[0], forecast)
        clipped_np = float(min(max(raw_np, bounds.np_lower), bounds.np_upper))
        clipped_nuf = float(min(max(raw_nuf, bounds.nuf_lower), bounds.nuf_upper))
        return LeaderAction(clipped_np, clipped_nuf), {
            "leader_pfo_incumbent_N_P_star": clipped_np,
            "leader_pfo_incumbent_N_UF_star": clipped_nuf,
            "leader_pfo_incumbent_raw_N_P_star": raw_np,
            "leader_pfo_incumbent_raw_N_UF_star": raw_nuf,
            "leader_pfo_incumbent_N_P_clipped": float(abs(clipped_np - raw_np) > 1.0e-9),
            "leader_pfo_incumbent_N_UF_clipped": float(abs(clipped_nuf - raw_nuf) > 1.0e-9),
        }

    def _evaluate_fallback_candidates(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        start_index: int,
    ) -> List[_LeaderCandidateEvaluation]:
        self._pfo_incumbent_center: Optional[LeaderAction] = None
        self._pfo_incumbent_eval: Optional[_LeaderCandidateEvaluation] = None
        if not self._pfo_incumbent_fallback_enabled():
            return []
        pfo_previous = previous.copy()
        pfo_previous.N_P_star = 0.0
        pfo_previous.N_UF_star = 0.0
        pfo_previous.inflow_outflow_allocation = {}
        pfo_nash = self.nash_solver.solve(state.copy(), None, forecast, pfo_previous)
        action, action_meta = self._pfo_equivalent_action(pfo_nash.control, state, forecast, previous)
        pfo_nash.control.N_P_star = float(action.N_P_star)
        pfo_nash.control.N_UF_star = float(action.N_UF_star)
        pfo_nash.control.diagnostics.update(action_meta)
        predicted_states, follower_ttt, rollout_used = self._leader_evaluation_base(
            state,
            pfo_nash,
            forecast,
            previous=pfo_previous,
        )
        objective_terms = self.leader.objective_terms(
            predicted_states,
            pfo_nash.control,
            previous,
            follower_ttt,
            pfo_nash.converged,
            pfo_nash.residual_objective,
            pfo_nash.residual_control,
        )
        metadata = {
            "leader_response_proxy_state_count": float(len(predicted_states)),
            "leader_pfo_incumbent_active": 1.0,
            "leader_pfo_incumbent_candidate": 1.0,
            "leader_pfo_incumbent_objective": float(objective_terms["leader_total_objective"]),
            **action_meta,
        }
        pfo_eval = _LeaderCandidateEvaluation(
            index=start_index,
            action=action,
            nash=pfo_nash,
            objective=float(objective_terms["leader_total_objective"]),
            objective_terms=objective_terms,
            metadata=metadata,
            rollout_used=rollout_used,
            stage="fallback_pfo",
        )
        self._pfo_incumbent_center = action
        self._pfo_incumbent_eval = pfo_eval
        self._append_progress_event(
            event="candidate_evaluated",
            stage="fallback_pfo",
            completed=1,
            total=1,
            evaluation=pfo_eval,
            best_objective=pfo_eval.objective,
        )
        return [pfo_eval]

    def _pfo_centered_previous(self, previous: ControlAction) -> ControlAction:
        center = getattr(self, "_pfo_incumbent_center", None)
        if center is None:
            return previous
        seeded = previous.copy()
        seeded.N_P_star = float(center.N_P_star)
        seeded.N_UF_star = float(center.N_UF_star)
        return seeded

    def _grid_leader_search(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
        fallback_incumbent_obj: float,
    ):
        return super()._grid_leader_search(
            state,
            forecast,
            self._pfo_centered_previous(previous),
            global_refresh,
            fallback_incumbent_obj,
        )

    @staticmethod
    def _leader_action_key(action: LeaderAction) -> tuple[float, float]:
        return (round(float(action.N_P_star), 6), round(float(action.N_UF_star), 6))

    def _pfo_global_scout_evaluations(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        local_evaluations: List[_LeaderCandidateEvaluation],
        fallback_incumbent_obj: float,
    ) -> tuple[List[_LeaderCandidateEvaluation], Dict[str, float]]:
        bounds = self.leader._candidate_bounds(state, previous, forecast[0], forecast)
        np_lower, np_upper = float(bounds.np_lower), float(bounds.np_upper)
        nuf_lower, nuf_upper = float(bounds.nuf_lower), float(bounds.nuf_upper)

        def clipped(np_value: float, nuf_value: float) -> LeaderAction:
            return LeaderAction(
                float(min(max(float(np_value), np_lower), np_upper)),
                float(min(max(float(nuf_value), nuf_lower), nuf_upper)),
            )

        seed_actions = self._unique_leader_actions(
            [LeaderAction(previous.N_P_star, previous.N_UF_star)]
            + self._continuous_seed_actions(
                previous,
                bounds,
                np_lower,
                np_upper,
                nuf_lower,
                nuf_upper,
                clipped,
            )
        )
        scout_top_k = max(1, min(2, int(self.cfg.mpc.leader_continuous_prefilter_top_k)))
        scout_actions, scout_meta = self._continuous_prefilter_actions(
            seed_actions,
            state,
            forecast,
            previous,
            np_lower,
            np_upper,
            nuf_lower,
            nuf_upper,
            prefilter_samples=int(self.cfg.mpc.leader_continuous_prefilter_samples),
            prefilter_top_k=scout_top_k,
        )
        seen = {self._leader_action_key(item.action) for item in local_evaluations}
        scout_actions = [
            action for action in scout_actions
            if self._leader_action_key(action) not in seen
        ]
        full_budget = max(
            0,
            min(2, int(self.cfg.mpc.leader_continuous_max_evals) - len(local_evaluations)),
        )
        if full_budget <= 0 or not scout_actions:
            return [], {
                "leader_pfo_anchor_scout_full_budget": float(full_budget),
                "leader_pfo_anchor_scout_candidate_count": float(len(scout_actions)),
                "leader_pfo_anchor_scout_full_evaluated_count": 0.0,
                "leader_pfo_anchor_scout_top_k": float(scout_top_k),
            }
        incumbent = min(
            [float(fallback_incumbent_obj)]
            + [float(item.objective) for item in local_evaluations],
        )
        scout_evals, _ = self._evaluate_continuous_action_set(
            scout_actions[:full_budget],
            state,
            forecast,
            previous,
            stage="continuous_global_scout",
            index_start=len(local_evaluations),
            incumbent_obj=incumbent,
        )
        prefixed = {
            f"leader_pfo_anchor_scout_{key}": float(value)
            for key, value in scout_meta.items()
            if isinstance(value, (int, float, bool))
        }
        prefixed.update({
            "leader_pfo_anchor_scout_full_budget": float(full_budget),
            "leader_pfo_anchor_scout_candidate_count": float(len(scout_actions)),
            "leader_pfo_anchor_scout_full_evaluated_count": float(len(scout_evals)),
            "leader_pfo_anchor_scout_top_k": float(scout_top_k),
        })
        return scout_evals, prefixed

    def _continuous_leader_search(
        self,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
        fallback_incumbent_obj: float,
    ):
        centered_previous = self._pfo_centered_previous(previous)
        if global_refresh and getattr(self, "_pfo_incumbent_center", None) is not None:
            local_evals, base_meta, proxy_meta, refined_meta = super()._continuous_leader_search(
                state,
                forecast,
                centered_previous,
                False,
                fallback_incumbent_obj,
            )
            scout_evals, scout_meta = self._pfo_global_scout_evaluations(
                state,
                forecast,
                centered_previous,
                local_evals,
                fallback_incumbent_obj,
            )
            full_evals = local_evals + scout_evals
            base_meta.update(scout_meta)
            base_meta.update({
                "leader_pfo_anchor_global_hybrid_active": 1.0,
                "leader_pfo_anchor_local_full_evaluated_count": float(len(local_evals)),
                "leader_pfo_anchor_total_full_evaluated_count": float(len(full_evals)),
                "leader_candidate_global_refresh": 1.0,
                "leader_candidate_coarse_global": 1.0,
                "leader_candidate_coarse_local": 0.0,
            })
            return full_evals, base_meta, proxy_meta, refined_meta
        return super()._continuous_leader_search(
            state,
            forecast,
            centered_previous,
            global_refresh,
            fallback_incumbent_obj,
        )

    def _select_with_fallback_guard(
        self,
        leader_evaluations: List[_LeaderCandidateEvaluation],
        fallback_evaluations: List[_LeaderCandidateEvaluation],
    ):
        best, metadata = super()._select_with_fallback_guard(leader_evaluations, fallback_evaluations)
        pfo_eval = getattr(self, "_pfo_incumbent_eval", None)
        pfo_tie_break_selected = False
        if pfo_eval is not None and best.stage != "fallback_pfo":
            eps = 1.0e-9
            if float(best.objective) >= float(pfo_eval.objective) - eps:
                best = pfo_eval
                pfo_tie_break_selected = True
                metadata["leader_fallback_guard_selected"] = 1.0
                metadata["leader_fallback_guard_selected_pfo"] = 1.0
                metadata["leader_fallback_guard_rejected_leader"] = 1.0
        # 층2(2026-07-14) trailing-regret 강제 커밋: 최근 k-창에서 실현 TTT가 incumbent
        # 예측×1.10을 넘었으면 leader 후보 대신 incumbent를 k스텝 강제 커밋한다
        # (예측 vs 예측 guard가 못 잡는 실현 기반 안전장치; regret_guard_steps=0=OFF).
        metadata["leader_regret_forced_commit"] = 0.0
        if (
            bool(getattr(self, "_regret_force_this_step", False))
            and pfo_eval is not None
            and best.stage != "fallback_pfo"
        ):
            best = pfo_eval
            metadata["leader_regret_forced_commit"] = 1.0
            metadata["leader_fallback_guard_selected"] = 1.0
            metadata["leader_fallback_guard_selected_pfo"] = 1.0
        elif bool(getattr(self, "_regret_force_this_step", False)) and best.stage == "fallback_pfo":
            # 이미 guard/tie-break가 incumbent를 선택한 경우도 강제 상태를 기록.
            metadata["leader_regret_forced_commit"] = 1.0
        metadata.update({
            "leader_pfo_incumbent_active": float(pfo_eval is not None),
            "leader_pfo_incumbent_selected": float(best.stage == "fallback_pfo"),
            "leader_pfo_incumbent_tie_break_selected": float(pfo_tie_break_selected),
            "leader_pfo_incumbent_N_P_star": float(pfo_eval.action.N_P_star) if pfo_eval else 0.0,
            "leader_pfo_incumbent_N_UF_star": float(pfo_eval.action.N_UF_star) if pfo_eval else 0.0,
            "leader_pfo_incumbent_objective": float(pfo_eval.objective) if pfo_eval else 0.0,
            "leader_pfo_incumbent_local_center_used": float(
                pfo_eval is not None and getattr(self, "_pfo_incumbent_center", None) is not None
            ),
        })
        # ---- LINK-SHARE 스윕(mode="search" 전용): 선택된 leader 후보 위에서 s 좌표하강 ----
        # incumbent(PFO) 선택이면 skip(자율 metering이라 ω 무관). 균등(0.5)은 best가 이미
        # 그 값으로 평가된 결과이므로 grid 2점만 추가 평가. 개선 시 best 교체 + ω 고정.
        if (
            self.nuf_link_share_mode == "search"
            and self._link_share_ctx is not None
            and best.stage != "fallback_pfo"
            and float(best.action.N_UF_star) > 0.0
            and isinstance(self.nash_solver, WuFaithfulFollower)
            and len(self.cfg.network.freeway_links) == 2
        ):
            ls_state, ls_forecast, ls_previous = self._link_share_ctx
            links = list(self.cfg.network.freeway_links)
            best_s = 1.0 / len(links)
            for s in self.nuf_link_share_grid:
                self.nash_solver._wu._omega_f = {
                    links[0]: float(s), links[1]: 1.0 - float(s),
                }
                trial = self._evaluate_full_candidate(
                    9000 + int(round(float(s) * 100)), best.action,
                    ls_state, ls_forecast, ls_previous, stage="link_share",
                )
                if float(trial.objective) < float(best.objective):
                    best = trial
                    best_s = float(s)
            # 커밋되는 best와 일치하는 ω로 고정(다음 step 시작 시 균등 리셋).
            self.nash_solver._wu._omega_f = {
                links[0]: float(best_s), links[1]: 1.0 - float(best_s),
            }
            metadata["leader_nuf_link_share"] = float(best_s)
            metadata["leader_nuf_link_share_adopted"] = float(abs(best_s - 0.5) > 1e-9)
        # λ step 간 적분 갱신(A1+A2): **선택된 후보**의 λ_next만 follower 영속 가격에 commit한다
        # (후보별 solve는 diagnostics로만 λ_next를 내놓고 self._lambda_P를 건드리지 않는다).
        # PFO incumbent 선택 시(stage=="fallback_pfo", leader=None이라 lambda_next 없음) λ는 갱신
        # 하지 않고 유지한다 — 커밋된 제어가 PFO면 λ는 plant에 영향이 없으므로 새 정보가 없다.
        lam_next = best.nash.control.diagnostics.get("wu_faithful_lambda_next")
        if lam_next is not None:
            if bool(getattr(self.cfg.mpc, "np_candidate_lambda", False)):
                if int(getattr(self.cfg.mpc, "np_primal_dual_iters", 0)) > 0:
                    # 방법 A: 선택 후보의 K-loop 최종 λ(안장점)를 다음 step warm start로
                    # 직접 commit. corrector 체인은 사용하지 않는다(pending 해제) —
                    # λ는 매 step 후보별 계획-공간 반복으로 재유도된다.
                    lam_pd = best.nash.control.diagnostics.get("wu_faithful_np_cand_lambda")
                    if lam_pd is not None:
                        self.nash_solver._lambda_P = float(lam_pd)
                    self.nash_solver._np_corrector_pending = None
                else:
                    # (51) corrector로 이관: 커밋 시점에는 standing λ를 갱신하지 않고
                    # (λ_k, 커밋 후보의 투영 target)을 pending으로 넘겨, 다음 step 시작 시
                    # 실현 유입 Q^real로 1회 교정한다(원고 정식화 정렬 — λ̂ 재적분 아님).
                    tgt_c = best.nash.control.diagnostics.get("wu_faithful_np_projected_target")
                    if tgt_c is not None:
                        self.nash_solver._np_corrector_pending = (
                            float(self.nash_solver._lambda_P), float(tgt_c),
                        )
            else:
                self.nash_solver._lambda_P = float(lam_next)
            # 첫 스텝 predictor fallback용 committed Σnin 저장(실현 유입 관측 전).
            sum_nin_d = best.nash.control.diagnostics.get("wu_faithful_np_sum_nin")
            if sum_nin_d is not None:
                self.nash_solver._np_last_sum_nin = float(sum_nin_d)
        metadata["leader_lambda_np_committed"] = float(lam_next is not None)
        # N_UF dual λ_UF: 선택 후보 diagnostics에서 commit(λ_P와 동일 규약)하되,
        # BOOTSTRAP(2026-07-09): incumbent/fallback 선택으로 diagnostics에 λ_next가 없어도
        # **무조건 적분**한다. 실측(g1df dual d3): λ=0 → dual 항 0 → leader 후보 ≈ PFO
        # incumbent → tie-break가 incumbent 선택(31/40, tie 19) → λ commit 9/40 → λ가
        # 영영 0 부근에 잠기는 자기잠금. 수량 오차(실현 Σmeter − leader-stage 최선 후보의
        # N_UF*)는 커밋 주체와 무관하게 관측되는 새 정보이므로 스텝마다 λ를 적분한다.
        lam_uf_next = best.nash.control.diagnostics.get("wu_faithful_lambda_uf_next")
        if lam_uf_next is not None:
            self.nash_solver._lambda_UF = float(lam_uf_next)
        else:
            follower = self.nash_solver
            nuf_mode = str(getattr(
                self.cfg.mpc, "wu_faithful_nuf_coordination_mode", "equality"
            ))
            if nuf_mode == "dual" and isinstance(follower, WuFaithfulFollower):
                leader_best = None
                for ev in leader_evaluations:
                    if leader_best is None or float(ev.objective) < float(leader_best.objective):
                        leader_best = ev
                if leader_best is not None:
                    target = float(leader_best.action.N_UF_star)
                    sum_meter = sum(
                        float(v) for v in best.nash.control.ramp_metering.values()
                    )
                    lam = float(follower._lambda_UF) + float(
                        follower.lambda_uf_step_gain
                    ) * (sum_meter - target)
                    follower._lambda_UF = float(min(
                        max(lam, -float(follower.lambda_uf_cap)),
                        float(follower.lambda_uf_cap),
                    ))
                    metadata["leader_lambda_uf_bootstrap_updated"] = 1.0
                    metadata["leader_lambda_uf_value"] = float(follower._lambda_UF)
        metadata["leader_lambda_uf_committed"] = float(lam_uf_next is not None)
        return best, metadata

    def _proxy_score_candidate(
        self,
        index: int,
        action: LeaderAction,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
    ) -> Dict[str, float]:
        """action-aware prefilter proxy(베이스의 action-blind else 분기 대체).

        베이스 `_proxy_score_candidate`의 else 분기는 현재 state만 읽어 모든 후보의 점수가
        동일했다(prefilter가 index 순서로 무작위 통과). 여기서는 후보-의존으로 만든다:
        (1) follower의 hard-budget 분기(N_UF_star>0)를 simplex 탐색 없이 용량비례 배분으로
        근사해 candidate metering을 구성하고, (2) leader full 평가와 동일한 plant rollout
        (`self._predict`)으로 예측 상태를 만들어 `objective_terms`로 채점한다(leader는
        centralized이므로 정보 철학 위반 아님).

        한계: green의 λ(N_P) 응답은 근사하지 않는다 — N_P 차원은 predicted states의
        protected accumulation 항을 통해서만 간접 반영되며, 주된 후보-분별력은
        N_UF(metering) 차원에서 나온다."""
        # base full 평가(_evaluate_full_candidate)는 follower-feasible로 투영된 좌표를
        # 채점하므로, prefilter proxy도 동일 좌표계에서 채점해야 랭킹이 어긋나지 않는다
        # (base _proxy_score_candidate 1537-1539행과 동일한 pre-projection).
        action, _projection_meta = self._project_action_to_follower_feasible_np(
            action, state, forecast, previous
        )
        control = previous.copy()
        control.N_P_star = float(action.N_P_star)
        control.N_UF_star = float(action.N_UF_star)
        net = self.cfg.network
        if float(action.N_UF_star) > 0.0:
            # follower hard-budget 분기 근사: link budget = ω_F[link]·N_UF_star를
            # 소유 ramp들에 용량비례로 배분(각 ramp은 capacity로 clamp).
            follower = self.nash_solver
            for link in net.freeway_links:
                model = follower._local_freeway_models[link]
                owned = list(model.owned_ramps)
                if not owned:
                    continue
                caps = {r: float(net.ramp_capacity_veh_h[r]) for r in owned}
                cap_sum = sum(caps.values())
                if cap_sum <= 0.0:
                    continue
                omega = float(follower._wu._omega_f.get(link, 0.0))
                budget = min(max(omega * float(action.N_UF_star), 0.0), cap_sum)
                for ramp in owned:
                    share = budget * (caps[ramp] / cap_sum)
                    control.ramp_metering[ramp] = float(min(max(share, 0.0), caps[ramp]))
        # N_UF_star<=0이면 follower autonomous 분기에 대응 — previous.ramp_metering 유지.
        # OPT3: proxy를 near(horizon)+far(MFD tail)로 — full depth(3+d)의 절반 비용으로
        # far-aware 랭킹(현행 proxy는 far 무시라 방류 후보 misrank). 기본 OFF=기존 그대로.
        if bool(getattr(self.cfg.mpc, "leader_proxy_near_far", False)):
            states, rollout_ttt = self._predict(
                state, control, forecast, depth_override=self.cfg.mpc.horizon_steps
            )
            rollout_ttt += self._mfd_far_cost_to_go(states[-1])
        else:
            states, rollout_ttt = self._predict(state, control, forecast)
        terms = self.leader.objective_terms(
            states,
            control,
            previous,
            float(rollout_ttt),
            True,
            0.0,
            0.0,
        )
        return {
            "index": float(index),
            "N_P_star": float(action.N_P_star),
            "N_UF_star": float(action.N_UF_star),
            "objective": float(terms["leader_total_objective"]),
            "base": float(terms["leader_objective_base"]),
            "follower_ttt": float(terms["leader_follower_ttt_base"]),
            "spillback_violation": 0.0,
        }

    # ---------- A1(2026-07-10): 후보 N_UF 중복제거 ----------

    def _evaluate_full_candidate(
        self,
        index: int,
        action: LeaderAction,
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        stage: str = "coarse",
        incumbent_obj: float = float("inf"),
        rollout_abort_obj: float = float("inf"),
    ) -> _LeaderCandidateEvaluation:
        """같은 N_UF 후보의 follower solve + rollout 재사용(A1).

        step 내에서 follower 반응은 후보의 N_UF에만 의존한다 — λ_P는 warm-start로
        고정이고 N_P는 그 λ를 통해서만 작용(dual 규약), metering budget은 ω·N_UF.
        따라서 (N_P, N_UF) 격자에서 N_UF가 같은 후보들은 nash.control과 rollout이
        동일하다. 대표 1회만 full 평가하고 나머지는 캐시 재사용, N_P 의존
        diagnostics(λ_next·target 계열)만 패치한다. objective_terms는 클론 control로
        정확 재계산(states 재사용, 저렴). base 메서드와 동기 유지 필요(꼬리 복제)."""
        if not (
            self.candidate_dedupe_enabled
            and isinstance(self.nash_solver, WuFaithfulFollower)
        ):
            return super()._evaluate_full_candidate(
                index, action, state, forecast, previous,
                stage=stage, incumbent_obj=incumbent_obj,
                rollout_abort_obj=rollout_abort_obj,
            )
        raw_action = LeaderAction(float(action.N_P_star), float(action.N_UF_star))
        action_p, projection_meta = self._project_action_to_follower_feasible_np(
            action, state, forecast, previous
        )
        key = round(float(action_p.N_UF_star), 6)
        if bool(getattr(self.cfg.mpc, "np_candidate_lambda", False)):
            # NP-CAND-λ̂: λ̂가 N_P 의존이라 follower 반응이 후보별로 달라짐 → 키에 N_P 포함.
            key = (key, round(float(action_p.N_P_star), 6))
        cached = self._nuf_solve_cache.get(key)
        dedupe_hit = cached is not None
        if dedupe_hit:
            self._dedupe_hits += 1
            nash_rep, predicted_states, follower_ttt, rollout_used = cached
            nash = self._clone_nash_for_candidate(nash_rep, action_p)
        else:
            nash = self.nash_solver.solve(state.copy(), action_p, forecast, previous)
            predicted_states, follower_ttt, rollout_used = self._leader_evaluation_base(
                state, nash, forecast, incumbent_obj=rollout_abort_obj, previous=previous,
            )
            # abort(inf)된 평가는 캐시 금지 — 같은 N_UF가 뒤에서 더 관대한 문턱으로
            # 재평가될 수 있다(선택 동일성 보존).
            if follower_ttt != float("inf"):
                self._nuf_solve_cache[key] = (
                    nash, predicted_states, follower_ttt, rollout_used,
                )
        evaluated_action, closure_metadata = self._close_nash_response_leader_action(
            action_p, nash, forecast, intent_action=raw_action,
        )
        objective_terms = self.leader.objective_terms(
            predicted_states, nash.control, previous, follower_ttt,
            nash.converged, nash.residual_objective, nash.residual_control,
        )
        metadata = {
            "leader_response_proxy_state_count": float(len(predicted_states)),
            "leader_candidate_raw_N_P_star": float(raw_action.N_P_star),
            "leader_candidate_raw_N_UF_star": float(raw_action.N_UF_star),
            **projection_meta,
            **closure_metadata,
            "leader_candidate_incumbent_active": float(incumbent_obj < float("inf")),
            "leader_candidate_incumbent_objective": float(
                incumbent_obj if incumbent_obj < float("inf") else 0.0
            ),
            "leader_candidate_follower_early_terminated_candidates": float(
                nash.diagnostics.get(
                    "distributed_grid_early_terminated_candidates",
                    nash.control.diagnostics.get(
                        "distributed_grid_early_terminated_candidates", 0.0
                    ),
                )
            ),
            "leader_candidate_dedupe_hit": float(dedupe_hit),
        }
        return _LeaderCandidateEvaluation(
            index=index,
            action=evaluated_action,
            nash=nash,
            objective=float(objective_terms["leader_total_objective"]),
            objective_terms=objective_terms,
            metadata=metadata,
            rollout_used=rollout_used,
            stage=stage,
        )

    def _clone_nash_for_candidate(self, nash_rep, action_p: LeaderAction):
        """A1 클론: 대표 nash의 control을 복사하고 N_P 의존 diagnostics만 패치.

        행동에 유효한 패치는 wu_faithful_lambda_next(λ_P commit 재료) 하나 —
        λ_next = clip(λ + gain·(Σnin − projected(N_P))). 나머지 target 계열은
        로그 정확성용. sigma/sum_nin/λ_P는 후보 무관이라 대표 값 재사용."""
        import dataclasses as _dc
        control = nash_rep.control.copy()
        control.N_P_star = float(action_p.N_P_star)
        control.N_UF_star = float(action_p.N_UF_star)
        d = control.diagnostics
        follower = self.nash_solver
        n_p = float(action_p.N_P_star)
        sum_nin = float(d.get("wu_faithful_sum_nin", 0.0))
        sig_min = float(d.get("wu_faithful_np_feasible_min", 0.0))
        sig_max = float(d.get("wu_faithful_np_feasible_max", sig_min))
        lambda_p = float(d.get("wu_faithful_lambda_P", 0.0))
        projected = min(max(n_p, sig_min), sig_max)
        if "wu_faithful_lambda_next" in d:
            d["wu_faithful_lambda_next"] = float(
                follower._lambda_np_update(lambda_p, sum_nin, projected)
            )
        # rate 환산용 horizon_h: 대표 값에서 역산, 불가 시 H·T_c_h.
        rep_proj = float(d.get("wu_faithful_np_projected_target", 0.0))
        rep_rate = float(d.get("urban_net_inflow_target_veh_h", 0.0))
        if abs(rep_rate) > 1.0e-9 and abs(rep_proj) > 1.0e-9:
            horizon_h = rep_proj / rep_rate
        else:
            horizon_h = float(self.cfg.mpc.horizon_steps) * float(
                self.cfg.simulation.T_c_h
            )
        d["wu_faithful_np_target"] = n_p
        d["wu_faithful_np_original_target"] = n_p
        d["wu_faithful_np_projected_target"] = projected
        d["wu_faithful_np_projection_residual"] = n_p - projected
        d["wu_faithful_np_target_error"] = sum_nin - projected
        d["urban_net_inflow_target_veh_h"] = projected / max(horizon_h, 1.0e-9)
        if "wu_faithful_np_original_target_veh" in d:
            d["wu_faithful_np_original_target_veh"] = n_p
            d["urban_net_inflow_original_target_veh"] = n_p
        return _dc.replace(nash_rep, control=control)

    def _evaluate_candidate_set(
        self,
        candidates: List[LeaderAction],
        selected_indices: List[int],
        state: TrafficState,
        forecast: List[DemandStep],
        previous: ControlAction,
        stage: str = "coarse",
        index_offset: int = 0,
        incumbent_obj: float = float("inf"),
    ) -> List[_LeaderCandidateEvaluation]:
        """후보 평가를 serial in-process로 강제한다(미변경 베이스의 worker 우회).

        베이스 `_evaluate_candidate_set`는 module-level `_stackelberg_candidate_worker`로
        후보를 평가하는데, 그 worker는 항상 베이스 `StackelbergMPCController`를 생성해
        nash_solver가 NashSolver가 된다(우리 follower 주입이 무시됨). 그래서 worker를 쓰지
        않고 self의 `_evaluate_full_candidate`(=self.nash_solver=WuFaithfulFollower)를 직접
        serial로 호출한다. metering 좌표하강이 무거우므로 process 풀 없이도 충분하다."""
        if not selected_indices:
            raise ValueError("Stackelberg leader prefilter removed every candidate.")
        results: List[_LeaderCandidateEvaluation] = []
        stage_incumbent = float(incumbent_obj)
        # OPT2의 abort 기준은 **같은 스케일**(full 후보 objective = (3+d) rollout+far)만 —
        # fallback incumbent(_response_tts_objective, 3스텝·far 없음)는 스케일이 작아 섞으면
        # 과잉 pruning으로 정상 후보를 기각(leader가 fallback으로 후퇴, 실측 +1000 손실).
        rollout_inc = float("inf")
        for idx in selected_indices:
            result = self._evaluate_full_candidate(
                idx + index_offset,
                candidates[idx],
                state,
                forecast,
                previous,
                stage=stage,
                incumbent_obj=stage_incumbent,
                rollout_abort_obj=rollout_inc,
            )
            results.append(result)
            stage_incumbent = min(stage_incumbent, float(result.objective))
            if float(result.objective) != float("inf"):
                rollout_inc = min(rollout_inc, float(result.objective))
        diag: Dict[str, float] = {
            "leader_candidate_parallel_backend_serial": 1.0,
            "leader_candidate_parallel_backend_thread": 0.0,
            "leader_candidate_parallel_backend_process": 0.0,
            "leader_candidate_parallel_workers": 1.0,
            "leader_candidate_wu_metered_serial_override": 1.0,
        }
        for result in results:
            result.metadata.update(diag)
        return results
