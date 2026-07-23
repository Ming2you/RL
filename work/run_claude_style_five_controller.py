from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.free_flow_reference import compute_free_flow_reference
from src.controllers.classical_hierarchical import ClassicalHierarchicalController
from src.controllers.distributed_coordinator import DistributedCoordinator
from src.controllers.f1_wu_faithful_follower import (
    F1StackelbergWuMeteredController,
    F1WuFaithfulFollower,
)
from src.controllers.joint_wu_controllers import (
    JointB2TRController,
    JointF1Controller,
)
from src.controllers.centralized_mpc import CentralizedMPC
from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, apply_scenario_network_overrides, load_scenarios
from src.models.state import ControlAction, ExperimentConfig
from src.simulation.baseline import baseline_control
from src.simulation.simulator import MixedTrafficSimulator, control_row, state_row


CONTROLLERS = [
    "NO-CONTROL",
    "WU-CD-F",
    "WU-FAITHFUL-FOLLOWER",
    "P-STACK-WU-FAITHFUL",
    "CLASSICAL-HIERARCHICAL",
]

# 3점 사다리(2026-07-03 §4d) 변형 ID — 기본 5종의 거동은 불변:
#   WU-FAITHFUL-FOLLOWER-NOP1 : rung1 = PFO 순수 own-TTS(P1 blocked-inflow 가격 OFF).
#   WU-FAITHFUL-FOLLOWER      : rung2 = PFO + tax(P1 ON, leader 없음) — 기존 기본값.
#   P-STACK-WU-FAITHFUL       : rung3 = leader + marginal price(B2 ON) — B2가 기본.
#   P-STACK-WU-FAITHFUL-NOB2  : rung3의 가격 채널 귀속용 A/B(B2 OFF, 구 P-Stack).
LADDER_VARIANTS = [
    "WU-FAITHFUL-FOLLOWER-NOP1",
    "P-STACK-WU-FAITHFUL-NOB2",
]

# P1.5 재검토(조건부 활성화) 변형:
#   WU-FAITHFUL-FOLLOWER-P15SAT  : 게이트는 닫아두고 포화도 x만 진단 기록(계측 전용).
#   WU-FAITHFUL-FOLLOWER-P15AUTO : 포화도 band 안의 ramp 신호만 phase-resolved 활성.
P15_VARIANTS = [
    "WU-FAITHFUL-FOLLOWER-P15SAT",
    "WU-FAITHFUL-FOLLOWER-P15AUTO",
]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cfg(scenario_name: str, t_total: float) -> tuple[ExperimentConfig, Any]:
    overrides = {
        "mpc": {
            "relaxed_quantized_controls": True,
            "grid_parallel_backend": "serial",
            "leader_search_mode": "grid",
            "stackelberg_leader_parallel_backend": "serial",
        },
        "simulation": {"T_total": float(t_total)},
    }
    import os as _os
    if _os.environ.get("NET4SEG") == "1":
        # legacy 4-seg 재현 전용(2026-07-12부터 default.yaml 자체가 8-seg).
        # 구 NET8SEG=1은 무의미(디폴트가 이미 8-seg).
        overrides["network"] = {
            "freeway_segments_per_link": 4,
            "ramp_merge_segment_index": {"R_D_W": 2, "R_F_W": 3, "R_D_E": 2, "R_F_E": 3},
            "off_ramp_segment_index": {"OR_D_W": 1, "OR_F_W": 2, "OR_D_E": 1, "OR_F_E": 2},
        }
    cfg = ExperimentConfig.from_file(str(ROOT / "src" / "config" / "default.yaml"), overrides)
    scenarios = load_scenarios(str(ROOT / "src" / "config" / "scenarios.yaml"))
    scenario = scenarios[scenario_name]
    cfg = apply_scenario_network_overrides(cfg, scenario)
    return cfg, scenario


def make_controller(controller_id: str, cfg: ExperimentConfig):
    if controller_id == "NO-CONTROL":
        return None
    if controller_id == "WU-CD-F":
        return DistributedCoordinator(cfg, ablation="WU_GREEN_VSL_ONLY_TTT")
    if controller_id == "WU-FAITHFUL-FOLLOWER":
        return WuFaithfulFollower(cfg)
    if controller_id == "WU-FAITHFUL-FOLLOWER-NOP1":
        follower = WuFaithfulFollower(cfg)
        follower.count_blocked_ramp_inflow = False
        return follower
    if controller_id == "WU-FAITHFUL-FOLLOWER-P15SAT":
        follower = WuFaithfulFollower(cfg)
        follower.ramp_aware_phase_auto = True
        follower.ramp_aware_phase_auto_band = (9.0e9, 9.0e9)  # 계측 전용(게이트 불발)
        return follower
    if controller_id == "WU-FAITHFUL-FOLLOWER-P15AUTO":
        follower = WuFaithfulFollower(cfg)
        follower.ramp_aware_phase_auto = True
        return follower
    if controller_id == "P-STACK-WU-FAITHFUL":
        return StackelbergWuMeteredController(cfg)
    # 2026-07-05 §8부터 P-STACK-WU-FAITHFUL 기본 = green 가격 + trust(=B2TR).
    # -B2 = 무제한 가격(trust 없음, sweet_155 폭주 재현/역사적 비교용).
    if controller_id == "P-STACK-WU-FAITHFUL-B2":
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = True
        controller.signal_price_trust_sec = None
        return controller
    # B3: green + metering 가격, trust 없음(과소방류 나선 재현/역사적 비교용).
    if controller_id == "P-STACK-WU-FAITHFUL-B3":
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = True
        controller.signal_price_trust_sec = None
        controller.metering_price_enabled = True
        controller.metering_price_trust_frac = None
        return controller
    # B3TR: 기본(green+trust) + metering 가격+trust, 증명서 없음(§11 v2 재현 — breakdown).
    if controller_id == "P-STACK-WU-FAITHFUL-B3TR":
        controller = StackelbergWuMeteredController(cfg)
        controller.metering_price_enabled = True
        controller.metering_release_cert_enabled = False
        return controller
    # B3CERT: metering 가격 + trust + **비대칭 안전 증명서**(방류↑는 leader의 +δ rollout
    # 예측밀도 인증 시에만, 방류↓는 자유) — "성능은 가격, 절벽 방향은 증명서".
    if controller_id == "P-STACK-WU-FAITHFUL-B3CERT":
        controller = StackelbergWuMeteredController(cfg)
        controller.metering_price_enabled = True
        return controller
    # F1(2026-07-06, 사본 실험): 안전 페널티를 follower objective로 이관 —
    # urban 0.5cap spill hinge + freeway rho_crit hinge. 가격 구성은 기본(B2TR) 그대로.
    if controller_id == "P-STACK-WU-FAITHFUL-F1":
        return F1StackelbergWuMeteredController(cfg)
    # F1RHO: freeway rho_crit hinge만(urban spill hinge OFF) — F1 혼합 판정(155 +1.26%/
    # 190 -2.91%)의 분해: 155 후퇴가 spill hinge 몫인지 검증.
    if controller_id == "P-STACK-WU-FAITHFUL-F1RHO":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        return controller
    # F1RHO05: rho hinge w=0.5 — hinge는 가격이 아니라 마진 항이라 w가 정당한 노브
    # (위험 회피 강도). 155 비용(+1.26%) 완화 vs 190 이득 보존의 트레이드오프 측정.
    if controller_id == "P-STACK-WU-FAITHFUL-F1RHO05":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.f1_rho_weight = 0.5
        return controller
    # F2: F1RHO(ρ hinge가 절벽 방어) + metering 가격+trust+cert — 가격 분기의 후보 채점
    # (_solve_with → F1의 _solve_freeway_agent_local)에 hinge가 들어가므로, B3 실패
    # (과방류 후보가 싸게 보임)가 교정된 상태에서 가격이 매끈한 몫만 나르는지 재도전.
    if controller_id == "P-STACK-WU-FAITHFUL-F2":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.metering_price_enabled = True
        return controller
    # F3: F1RHO + offset 가격(leader FD → follower offset 탐색 가격+trust 활성화) —
    # legacy 잔존 격차의 coordinated offset 재료 공략.
    if controller_id == "P-STACK-WU-FAITHFUL-F3":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.offset_price_enabled = True
        return controller
    # F23: F2 + F3 결합(metering·offset 가격 동시).
    if controller_id == "P-STACK-WU-FAITHFUL-F23":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.metering_price_enabled = True
        controller.offset_price_enabled = True
        return controller
    # ---- Joint 모드(2026-07-06 스펙): 두 objective baseline을 각각 보존한 joint 탐색 ----
    # JB2TR = B2TR objective(hinge 없음) + RM full cross-grid + (green,offset) 패턴 joint.
    if controller_id == "P-STACK-WU-FAITHFUL-JB2TR":
        controller = JointB2TRController(cfg)
        controller.offset_joint_enabled = True
        return controller
    # JF1 = F1RHO objective(ρ hinge) + RM full cross-grid + (green,offset) 패턴 joint.
    if controller_id == "P-STACK-WU-FAITHFUL-JF1":
        controller = JointF1Controller(cfg)
        controller.offset_joint_enabled = True
        return controller
    # freeway joint만(urban joint 없이 — 성분 분해용).
    if controller_id == "P-STACK-WU-FAITHFUL-JB2TR-FW":
        return JointB2TRController(cfg)
    if controller_id == "P-STACK-WU-FAITHFUL-JF1-FW":
        return JointF1Controller(cfg)
    # ---- G1(2026-07-06): ramp 신호(D/F) offset 활성화 (F1RHO base) ----
    # G1DF = D/F offset만 활성(A/B/C offset은 0 유지) — plant 민감도 F=10.7 레버 격리.
    if controller_id == "P-STACK-WU-FAITHFUL-G1DF":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.ramp_offset_enabled = True
        return controller
    # G1ALL = 전 신호 offset 활성(A/B/C 국소 + D/F ramp-aware) — legacy 전체 offset 비교.
    if controller_id == "P-STACK-WU-FAITHFUL-G1ALL":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.offset_enabled = True
        controller.nash_solver.ramp_offset_enabled = True
        return controller
    # ---- ALLPRICE(2026-07-07): 모든 lever가 ∂(TTT+V)/∂lever — 통일 marginal price ----
    # 사용자 설계 수렴: green·offset(urban) + VSL·metering(freeway) 전부 marginal price로.
    # V(terminal cost)는 HORIZON env로 rollout 깊이를 늘려 근사(horizon_steps↑ → leader eval +
    # 모든 price rollout + follower가 uniform하게 (9+d)분 → price=∂(TTT+V)/∂lever). base=F1RHO
    # (ρ hinge, VSL-FD plant서 rho_crit(VSL) 반영) + green price(B2TR 기본) + metering + VSL price.
    # VSL_FD=1 + HORIZON=k와 함께 실행. 깊이 sweep으로 legacy 회수율 frontier.
    if controller_id == "P-STACK-WU-FAITHFUL-ALLPRICE":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.signal_price_enabled = True   # green (B2TR 기본이지만 명시)
        controller.metering_price_enabled = True # RM
        controller.vsl_price_enabled = True      # VSL
        return controller
    # ---- ALLPRICE-JOINT(2026-07-09): per-lever 선형가격 + bilinear cross-term 가격 ----
    # ★ DEFAULT P-STACK(2026-07-09 사용자 결정): 논문 정합성 기준 기본 구성 —
    #   전 lever 통일 marginal price(g_ext = g_i − d_local, E2로 VSL도 정합) + 2쌍 cross
    #   + N_UF equality anchor + link-share density(headroom 비례, 컨트롤러 기본).
    #   성능 챔피언은 G1DF(green price only, 11283 d3)이며 ablation 상한으로 병기.
    # ALLPRICE(green·metering·vsl per-lever price) 위에 lever쌍 교차곡률을 얹는다:
    #   green×offset — leader h_ext 하달 + follower가 non-ramp 신호를 2D 공동탐색
    #                  (coordinate descent면 cross 퇴화 → joint_green_offset_enabled 필수).
    #   vsl×metering — leader h_ext 하달, follower 탐색구조 불변(primal joint 이미 포착)에
    #                  cross 가격만 freeway 채점에 추가.
    # LEADER_V_DEPTH=k와 함께 실행(far는 leader 채점 기본 ON, 가격 far=E1은 env 옵션).
    if controller_id == "P-STACK-WU-FAITHFUL-ALLPRICE-JOINT":
        # DEFAULT = SPLIT-PRICE v2 + DEPTH d3(18분) — 2026-07-09 최종 확정.
        # 플래그십 = hybrid v2(총량 equality + 배분 own_TTS+가격 + incumbent 가격-레벨
        # 배제): d3 = 11893, d0 무붕괴(13079 — level의 d0 22698 붕괴와 대조), 사용자
        # 스펙 정합. 수량 전송 구조라 depth 곡선이 G1DF형(깊어야 후보 랭킹 안정) →
        # 기본 d3. 구 level 모드(가격-레벨, 고원 11.7~12.0k)는 METER_PRICE_MODE=level.
        # env LEADER_V_DEPTH 지정 시 그 값 우선(명시적 0 포함).
        import os as _os_ap
        if "LEADER_V_DEPTH" not in _os_ap.environ and int(
            getattr(cfg.mpc, "leader_value_depth", 0)
        ) == 0:
            cfg.mpc.leader_value_depth = 3
        # OPT12 기본 ON(2026-07-10 확정): local 스텝 refined 재정련 생략 + rollout exact
        # 조기절단 — sweet_190 ablation TTT 11459 vs 11458(+1, 무손실)에 compute −28%
        # (81.7→59.2s/step). OPT12=0으로 해제. (OPT3·SPSA는 APJOINT서 유해 판정 — 제외.)
        if _os_ap.environ.get("OPT12") != "0":
            cfg.mpc.leader_skip_local_refinement = True
            cfg.mpc.leader_rollout_early_stop = True
        controller = F1StackelbergWuMeteredController(cfg)
        if _os_ap.environ.get("GRAD_SEED") == "1":
            # proxy-gradient 유도 시드(2026-07-21, 사용자 설계): 균등 Halton 대신 하강 방향
            # 시드 추가 — 적은 평가로 최적 예산 접근. 미설정 시 완전 비활성(원본 동일).
            from src.controllers.gradseed_mpc import enable_gradseed
            enable_gradseed(controller)
        if _os_ap.environ.get("BIAS_SAMPLE") == "1":
            # box 상단 편향 샘플링(2026-07-21, 사용자 설계): 선택점이 box 상단 0.85~0.94에
            # 몰려 있어 Halton을 상단 warp. BIAS_POW=p(<1). CAND와 병용해 후보 더 축소.
            from src.controllers.biasedsample_mpc import enable_biased_sampling
            cfg.mpc.leader_bias_sample_pow = float(_os_ap.environ.get("BIAS_POW", "0.4"))
            enable_biased_sampling(controller)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.signal_price_enabled = True
        controller.metering_price_enabled = True
        controller.vsl_price_enabled = True
        # ★동결 변경(2026-07-16): cross 2종 기본 OFF. 10셀 실측 —
        #   ②상수terminal+crossON(구 동결): 평균 +3.70% / 최악 −5.61% / 7승3패
        #   ③state-aware terminal+crossOFF(신): 평균 +4.78% / 최악 −4.34% / 8승2패
        # cross는 **눈먼 terminal의 대역**이었다(170_incident: 상수terminal에선 crossON이
        # +5.62%로 구제, state-aware에선 crossON이 −17.27% 파국 — 이중 보호로 과잉 조임).
        # terminal이 절벽을 보게 되자 cross는 중복이자 잡음. CROSS_ON=1로 구거동 복원.
        controller.green_offset_cross_price_enabled = False
        controller.vsl_meter_cross_price_enabled = False
        controller.nash_solver.joint_green_offset_enabled = True
        # offset price 편입(2026-07-18, 사용자 요청): "ALLPRICE"에 offset marginal price도 포함.
        # SQP식 inner-walk 4회로 trust 한 칸 갇힘 해소. 실측 이 망에선 offset 한계가치 무시 수준
        # (‖가격‖~0.01, wTTT −0.02%, High demand만 offset 3/5 이동) — §3 "죽은 채널"로 보고.
        # green 채널 진단 훅(2026-07-19, 보호큐/Wu격차 공통 병소 규명용):
        # GREEN_TRUST_SEC=k → 가격 trust 반경 확대(기본 ±6s — 보행 속도 제한 가설 A/B),
        # GREEN_PRICE=0 → 리더 B2 green 가격 해제(가격-벌점 상쇄 가설 A/B).
        if _os_ap.environ.get("GREEN_TRUST_SEC"):
            controller.signal_price_trust_sec = float(_os_ap.environ["GREEN_TRUST_SEC"])
        if _os_ap.environ.get("GREEN_PRICE") == "0":
            controller.signal_price_enabled = False
        # OFFSET_PRICE=0 env로 해제(구거동 재현). 미지정=ON.
        if _os_ap.environ.get("OFFSET_PRICE") != "0":
            controller.offset_price_enabled = True
            controller.offset_price_inner_iters = int(_os_ap.environ.get("OFFSET_INNER_ITER", "4"))
        # D/F(램프 신호) offset 편입(2026-07-18, 사용자 요청): A/B/C는 위 offset price(비램프),
        # D/F는 ramp-aware offset 탐색(G1DF 계보 nash_solver.ramp_offset_enabled). RAMP_OFFSET=0로 해제.
        if _os_ap.environ.get("RAMP_OFFSET") != "0":
            controller.nash_solver.ramp_offset_enabled = True
        # metering δ 스캔 승자(2026-07-15): δ=300 + trust_frac=0.20(=반경 300veh/h) 짝.
        # 170_skew_w wTTT 3244.7(baseline)→3089(회랑 floor 0.65 단독)→3028(δ=300 추가),
        # 190_w 5419.4→5357(floor)→5045(δ). METER_PRICE_DELTA env가 미지정일 때만 적용
        # (env가 δ·trust를 직접 주면 그 값 우선). **주의: δ·trust는 짝으로만 유효** —
        # trust=0.20 단독(δ=60)은 3186으로 오히려 열화, δ=300은 반드시 함께 줘야 한다.
        # **선결조건: 회랑 예산 floor(seg13_release_floor_frac=0.65)** — floor 없이 이
        # 반경이면 과소방류 나선(d300_floor0 wTTT 3768). 수량 floor와 짝으로만 안전.
        import os as _os_mpd
        if "METER_PRICE_DELTA" not in _os_mpd.environ:
            controller.metering_price_delta_veh_h = 300.0
            controller.metering_price_trust_frac = 0.20
        return controller
    # ---- G1DF-NORHO(2026-07-07): g1df에서 rho_crit 안전장치 2종 제거 — 진단 ----
    # 사용자 진단: freeway follower의 F1 ρ_crit hinge(own-TTS penalty)와 leader의 density_headroom
    # 캡(N_UF 예산을 merge 밀도가 rho_crit 닿는 flow로 상한)이 freeway 유입을 과하게 조여 격차의
    # 원인인가(legacy N_UF 5700 vs 우리 ~5084). 둘 다 제거하고 g1df 재실행 → 유입 증가가 격차를
    # 닫나 vs capacity-drop 붕괴하나. base=g1df(F1RHO+D/F offset).
    if controller_id == "P-STACK-WU-FAITHFUL-G1DF-NORHO":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.f1_rho_weight = 0.0           # F1 ρ_crit hinge 제거
        controller.nash_solver.ramp_offset_enabled = True
        controller.leader.rho_headroom_cap_enabled = False   # leader density_headroom 캡 제거
        return controller
    # ---- LEADER PROXY 격리(2026-07-07): rollout horizon vs candidate pruning, 2×2 ----
    # 사용자 4-run: {G1DF, NORHO} × {H15, FULLEVAL}. leader가 true global TTT가 아니라 proxy로
    # 후보를 채점하는 두 지점을 분리 검증:
    #   H15      = horizon_steps 3→5 (180s×5=900s=15분). leader가 더 긴 진짜 rollout으로 채점 →
    #              방류의 장기 배수 이득을 horizon 안으로. (penalty는 그대로 — 15분은 완만해
    #              cost-to-go 이중계상 경미; 필요시 별도 실험.)
    #   FULLEVAL = continuous prefilter의 proxy 사전선별 제거 — 샘플 후보 전부 full 평가
    #              (top_k=samples, max_evals 충분). proxy가 좋은 고방류 후보를 pruning하는지 검증.
    def _base_g1df():
        c = F1StackelbergWuMeteredController(cfg)
        c.nash_solver.f1_spillback_weight = 0.0
        c.nash_solver.ramp_offset_enabled = True
        return c
    def _base_norho():
        c = _base_g1df()
        c.nash_solver.f1_rho_weight = 0.0
        c.leader.rho_headroom_cap_enabled = False
        return c
    def _apply_h15():
        cfg.mpc.horizon_steps = 5  # 180s × 5 = 15분
    def _apply_fulleval():
        cfg.mpc.leader_continuous_prefilter_top_k = int(cfg.mpc.leader_continuous_prefilter_samples)
        cfg.mpc.leader_continuous_local_prefilter_top_k = int(cfg.mpc.leader_continuous_local_prefilter_samples)
        cfg.mpc.leader_continuous_max_evals = max(64, int(cfg.mpc.leader_continuous_prefilter_samples))
    if controller_id == "P-STACK-WU-FAITHFUL-G1DF-H15":
        _apply_h15(); return _base_g1df()
    if controller_id == "P-STACK-WU-FAITHFUL-G1DF-FULLEVAL":
        _apply_fulleval(); return _base_g1df()
    if controller_id == "P-STACK-WU-FAITHFUL-NORHO-H15":
        _apply_h15(); return _base_norho()
    if controller_id == "P-STACK-WU-FAITHFUL-NORHO-FULLEVAL":
        _apply_fulleval(); return _base_norho()
    # ---- RAMPQ(2026-07-07): leader에 선형 ramp-queue terminal cost 추가 (Codex "ramp=hidden space") ----
    # realized TTT의 위치 불변성(방류=차 이동, net-0)이 만든 under-release를, ramp 큐 선형 penalty로
    # 깬다. 보정(rampq_calib): 피크는 freeway 포화로 flat(자연 collapse 방지), buildup서 방류↑ →
    # 큐 폭발 예방. weight=8(보정상 buildup N_UF를 legacy 5700 방향으로). freeway follower는 단일
    # segment라 leader-단 항만으로 충분(사용자 지적). base: {GLEADOFF, NORHO}.
    import os as _os
    _RAMPQ_W = float(_os.environ.get("RAMPQ_W", "8.0"))  # 스윕용 env override
    if controller_id == "P-STACK-WU-FAITHFUL-NORHO-RQ":
        cfg.leader.w_ramp_queue = _RAMPQ_W
        return _base_norho()
    if controller_id == "P-STACK-WU-FAITHFUL-GLEADOFF-RQ":
        cfg.leader.w_ramp_queue = _RAMPQ_W
        c = F1StackelbergWuMeteredController(cfg)
        c.nash_solver.f1_spillback_weight = 0.0
        c.leader_offset_enabled = True
        c.leader_offset_method = "mpc"
        c.nash_solver.offset_directive_authoritative = True
        return c
    # ---- LEADER-OFFSET(2026-07-07): offset 소유권 follower→leader (green은 follower 유지) ----
    # G-LEAD-OFF-MPC = F1RHO(ρ hinge)+B2TR green price base + leader가 전 신호 offset을 MPC
    # joint rollout(corridor lag seed + 좌표하강)으로 결정. follower offset 탐색은 완전 OFF
    # (offset_enabled/ramp_offset_enabled/offset_marginal_price 모두 기본 OFF → directive만 동결).
    # 진단 처방: offset은 joint 변수라 follower 국소 best-response론 못 정한다(F3 per-signal
    # 가격≈0, g1all 12638 de-coordinate) → 전역 joint 평가자(leader)가 결정.
    if controller_id == "P-STACK-WU-FAITHFUL-GLEADOFF-MPC":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.leader_offset_enabled = True
        controller.leader_offset_method = "mpc"
        # leader가 offset 소유 → follower corridor 가드 우회(veto 금지, 작은 이득 누적 허용).
        controller.nash_solver.offset_directive_authoritative = True
        return controller
    # ---- 실험 ①(2026-07-07): N_UF dual λ_UF (등식→dual 가격, λ_P와 대칭) ----
    # G1DF base(신기록) + N_UF를 hard 등식 대신 signed dual 가격으로. 통일성 검증:
    # metering 절벽 레버가 수량 dual로도 되나(과방류 breakdown 위험 가설).
    if controller_id == "P-STACK-WU-FAITHFUL-NUFDUAL":
        cfg.mpc.wu_faithful_nuf_coordination_mode = "dual"
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.ramp_offset_enabled = True
        return controller
    # ---- 실험 ③(2026-07-07): green 이중가격 격리 — N_P가 g_ext와 이중계상/충돌인가 ----
    # G1DF(신기록) + N_P 항 전체 OFF(np_price_enabled=False). g_ext만 남김. 비슷하면
    # N_P 잉여(g_ext가 다 함), 나빠지면 N_P가 horizon 밖 보호를 실제 수행(시간대 분업).
    if controller_id == "P-STACK-WU-FAITHFUL-G1DF-NOLAMP":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.ramp_offset_enabled = True
        controller.nash_solver.np_price_enabled = False
        return controller
    # ---- 실험 ②(2026-07-07): blocked_q 격리 — metering price가 blocked_q와 간섭했나 ----
    # F2(metering price + hinge) 구성에서 count_blocked_ramp_inflow=False.
    # F2(19074, blocked_q ON)와 비교: 개선되면 간섭, 그대로면 절벽(§12) 재확인.
    if controller_id == "P-STACK-WU-FAITHFUL-F2NOBLK":
        controller = F1StackelbergWuMeteredController(cfg)
        controller.nash_solver.f1_spillback_weight = 0.0
        controller.nash_solver.count_blocked_ramp_inflow = False
        controller.metering_price_enabled = True
        return controller
    if controller_id == "WU-FAITHFUL-FOLLOWER-F1":
        return F1WuFaithfulFollower(cfg)
    # B2TR: green 가격 + trust region(가격 유효 범위 = 유한차분 이웃 ±6s) —
    # B2.1 폭주(선형 가격의 이웃 밖 월권, 2026-07-05 §6·§7 진단)의 원인 직결 처방.
    if controller_id == "P-STACK-WU-FAITHFUL-B2TR":
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = True
        controller.signal_price_trust_sec = controller.signal_price_delta_sec
        return controller
    # B2BAR: green 가격 + barrier(선형 fw+spillback) 보정, metering 가격 없음 —
    # sweet_155 B2 폭발(+10.3%)의 barrier 처방 검증(spillback이 조여질 때만 자동 발화).
    if controller_id == "P-STACK-WU-FAITHFUL-B2BAR":
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = True
        controller.barrier_price_enabled = True
        return controller
    # B2BAR50: 사용자 제안의 urban critical capacity(0.5·cap 초과분) 기준 —
    # 발화가 이르고(반차부터) 양방향(2026-07-05 probe: green축 C −0.21, A/B/F +0.03~0.04).
    if controller_id == "P-STACK-WU-FAITHFUL-B2BAR50":
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = True
        controller.barrier_price_enabled = True
        controller.barrier_spillback_frac = 0.5
        return controller
    # B4: B3 + rho_crit barrier 가격(절벽 예고 항), trust 없음(역사적 재현용 — 무력 판정).
    if controller_id == "P-STACK-WU-FAITHFUL-B4":
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = True
        controller.signal_price_trust_sec = None
        controller.metering_price_enabled = True
        controller.metering_price_trust_frac = None
        controller.barrier_price_enabled = True
        return controller
    if controller_id == "P-STACK-WU-FAITHFUL-NOB2":
        # 기본이 OFF가 되면서 P-STACK-WU-FAITHFUL과 동일 — 과거 런 재현용 별칭으로 유지.
        controller = StackelbergWuMeteredController(cfg)
        controller.signal_price_enabled = False
        return controller
    # N_UF 등식→cap A/B(2026-07-03): budget을 상한으로만 쓰는 cap 모드 변형.
    if controller_id == "P-STACK-WU-FAITHFUL-NUFCAP":
        cfg.mpc.wu_faithful_nuf_coordination_mode = "cap"
        return StackelbergWuMeteredController(cfg)
    # standalone(=PFO incumbent/fallback 없이 leader 후보만) — 등식 budget 악화 사례의 A/B.
    if controller_id == "P-STACK-WU-FAITHFUL-STANDALONE":
        cfg.mpc.stackelberg_enable_fallback = False
        cfg.mpc.stackelberg_enable_pfo_incumbent = False
        return StackelbergWuMeteredController(cfg)
    if controller_id == "P-STACK-WU-FAITHFUL-STANDALONE-NUFCAP":
        cfg.mpc.stackelberg_enable_fallback = False
        cfg.mpc.stackelberg_enable_pfo_incumbent = False
        cfg.mpc.wu_faithful_nuf_coordination_mode = "cap"
        return StackelbergWuMeteredController(cfg)
    if controller_id == "CLASSICAL-HIERARCHICAL":
        return ClassicalHierarchicalController(cfg)
    if controller_id == "P-CENT":
        # 중앙 천장 측정기(2026-07-10): 전권 joint 최적화 + far tail 채점(신규 이식).
        return CentralizedMPC(cfg, mode="proposed")
    if controller_id == "P-CENT-POLISH":
        # 진짜 중앙 상한(2026-07-10): 플래그십 해를 격자 중심(previous)으로 한 중앙 polish.
        # _structured_grid_search가 center/previous를 후보에 포함하므로 polish 채점(H rollout
        # +far) 기준으로 플래그십 이하로 내려갈 수 없음 — 개선분 = lever별 headroom 측정.
        class _PolishController:
            def __init__(self, cfg_):
                self.flag = make_controller("P-STACK-WU-FAITHFUL-ALLPRICE-JOINT", cfg_)
                self.cent = CentralizedMPC(cfg_, mode="proposed")

            def decide(self, state, forecast, previous):
                base = self.flag.decide(state, forecast, previous)
                info = self.cent.decide_with_info(state.copy(), forecast, previous_control=base)
                info.control.diagnostics["polish_improved"] = float(
                    info.control.diagnostics.get("centralized_grid_early_terminated", 0.0) == 0.0
                )
                return info.control

        return _PolishController(cfg)
    raise ValueError(f"Unknown controller: {controller_id}")


def decide(controller_id: str, controller, sim: MixedTrafficSimulator, forecast, previous, cfg, step: int):
    if controller_id == "NO-CONTROL":
        return baseline_control("no_control", cfg, sim.state, forecast[0])
    if controller_id == "P-CENT":
        return controller.decide_with_info(sim.state.copy(), forecast, previous).control
    if controller_id == "P-CENT-POLISH":
        return controller.decide(sim.state.copy(), forecast, previous)
    if controller_id in {
        "WU-CD-F",
        "WU-FAITHFUL-FOLLOWER",
        "WU-FAITHFUL-FOLLOWER-NOP1",
        "WU-FAITHFUL-FOLLOWER-P15SAT",
        "WU-FAITHFUL-FOLLOWER-P15AUTO",
        "WU-FAITHFUL-FOLLOWER-F1",
    }:
        return controller.solve(sim.state.copy(), None, forecast, previous).control
    if controller_id.startswith("P-STACK-WU-FAITHFUL"):
        return controller.decide(sim.state.copy(), forecast, previous, cfg)
    if controller_id == "CLASSICAL-HIERARCHICAL":
        return controller.decide(sim.state.copy(), forecast, previous)
    raise ValueError(f"Unknown controller: {controller_id}")


def summarize(controller_id: str, cfg: ExperimentConfig, sim: MixedTrafficSimulator, decision_rows, run_rows, reference) -> Dict[str, Any]:
    net = cfg.network
    horizon_h = cfg.simulation.T_total / 3600.0
    t_c_h = cfg.simulation.T_c_h
    completed_urban = sum(float(r.get("boundary_out_sink_veh", 0.0)) for r in run_rows)
    completed_freeway = sum(float(r.get("mainline_exit_flow_total", 0.0)) * t_c_h for r in run_rows)
    completed = completed_urban + completed_freeway
    final = sim.state
    onramp_terminal = sum(final.ramp_queue.values()) + sum(
        max(0.0, final.urban_movement_queue.get(m, 0.0))
        for movements in net.on_ramp_to_movement.values()
        for m in movements
    )
    compute_times = [float(row.get("computation_time_sec", 0.0)) for row in decision_rows]
    total_delay = sim.total_ttt - reference.total_ttt
    return {
        "controller_id": controller_id,
        "total_ttt": round(sim.total_ttt, 3),
        "urban_ttt": round(sim.urban_ttt, 3),
        "freeway_ttt": round(sim.freeway_ttt, 3),
        "free_flow_reference_total_ttt": round(reference.total_ttt, 3),
        "total_delay": round(total_delay, 3),
        "urban_delay": round(sim.urban_ttt - reference.urban_ttt, 3),
        "freeway_delay": round(sim.freeway_ttt - reference.freeway_ttt, 3),
        "completed_vehicles": round(completed, 1),
        "completed_urban_vehicles": round(completed_urban, 1),
        "completed_freeway_vehicles": round(completed_freeway, 1),
        "network_throughput_veh_h": round(completed / max(horizon_h, 1e-9), 1),
        "terminal_total_vehicles": round(final.total_urban_vehicles(net) + final.total_freeway_vehicles(net), 1),
        "terminal_urban_vehicles": round(final.total_urban_vehicles(net), 1),
        "terminal_onramp_vehicles": round(onramp_terminal, 1),
        "terminal_freeway_vehicles": round(final.total_freeway_vehicles(net) - sum(final.ramp_queue.values()), 1),
        "computation_time_sec": round(sum(compute_times), 3),
        "mean_step_compute_sec": round(float(np.mean(compute_times)) if compute_times else 0.0, 3),
        "max_step_compute_sec": round(float(np.max(compute_times)) if compute_times else 0.0, 3),
    }


def run_one(controller_id: str, scenario_name: str, t_total: float, output_root: Path, reference) -> Dict[str, Any]:
    import os as _os
    cfg, scenario = build_cfg(scenario_name, t_total)
    if _os.environ.get("VSL_FD") == "1":
        cfg.network.vsl_fd_two_branch = True  # two_branch VSL-FD 활성(진단)
    if _os.environ.get("RHO_CRIT_TB"):
        cfg.network.rho_crit_two_branch = float(_os.environ["RHO_CRIT_TB"])  # 삼각형 capacity 재calibration(예 19.6→cap 1950)
    if _os.environ.get("HORIZON"):
        cfg.mpc.horizon_steps = int(_os.environ["HORIZON"])  # V 깊이 sweep: leader eval+price+follower 동시 연장
    if _os.environ.get("LEADER_V_DEPTH"):
        cfg.mpc.leader_value_depth = int(_os.environ["LEADER_V_DEPTH"])  # leader full-rollout V 깊이(follower myopic 유지)
    if _os.environ.get("GLOBAL_REFRESH_SEC"):
        # 리더 전역 탐색 주기(2026-07-16 프로브). 기본 1800s = 10스텝마다만 candidates() 전역
        # 탐색이고 나머지 스텝은 refined_candidates()로 직전 해 주변만 본다(stackelberg_mpc L488).
        # 사고(구조 변화)는 incumbent를 무효화하는데 트리거가 시계뿐이라, 4500s 사고 발생 후
        # 다음 전역 탐색(step 30)까지 15분간 무효한 해 주변만 뒤진다 — 그 구멍의 상한 측정용.
        # =180이면 매 스텝 전역(상한). 미지정=기본 1800(비트동일).
        cfg.mpc.leader_global_refresh_sec = float(_os.environ["GLOBAL_REFRESH_SEC"])
    if _os.environ.get("CENT_REFRESH_SEC"):
        # P-CENT tightness: 전역 재탐색 주기[s]. 180=매 스텝 전역 그리드(기본 1800=10스텝마다).
        cfg.mpc.centralized_grid_refresh_sec = float(_os.environ["CENT_REFRESH_SEC"])
    if _os.environ.get("CENT_DENSE") == "1":
        cfg.mpc.centralized_grid_dense = True  # P-CENT tightness: 레버별 격자 레벨 조밀화
    if _os.environ.get("CENT_SLSQP") == "1":
        # P-CENT solver를 dense grid → SciPy SLSQP(연속 gradient 최적화)로 교체(2026-07-21,
        # Wang 물리 재검). capacity-drop=비볼록·비평활이라 gradient가 헤매는지 실측 —
        # "왜 grid인가/왜 분산이 강건한가"의 대조. 목적함수·far·refresh는 grid와 동일.
        cfg.mpc.centralized_solver_mode = "slsqp"
    if _os.environ.get("CENT_SLSQP_MAXEVAL"):
        # SLSQP 하드 평가예산(스텝당 rollout 상한). grid와 동일 예산(=maxiter*n_starts) 주고
        # 소진 시 best-so-far 반환 → 무제한 수렴(비평활서 폭주) 대신 완주+공정비교(2026-07-21).
        cfg.mpc.centralized_slsqp_max_eval = int(_os.environ["CENT_SLSQP_MAXEVAL"])
    if _os.environ.get("CAPDROP_PHI"):
        # queue-discharge capacity drop 민감도(2026-07-20): 혼잡 세그먼트 배출 ≤ φ·용량.
        # 문헌 실측 5~18% 감소 — 민감도 arm은 φ=0.85(사용자 채택). plant+복제본 동시 적용.
        cfg.network.capacity_drop_discharge_phi = float(_os.environ["CAPDROP_PHI"])
    if _os.environ.get("TAU_H"):
        # 정통 METANET 창발 capacity drop 프로브(2026-07-20): 완화시간 τ[h] 상향 → jam 머리
        # 가속 지연으로 배출 결손이 스위치 없이 창발(Papageorgiou 계열). 미설정=기본(비트동일).
        cfg.network.metanet_tau_h = float(_os.environ["TAU_H"])
    if _os.environ.get("MERGE_DELTA"):
        # METANET 표준 merge 항(2026-07-20): 합류 유입의 본선 속도 교란 δ(문헌 0.0122).
        cfg.network.metanet_delta_merge = float(_os.environ["MERGE_DELTA"])
    if _os.environ.get("FAR_REAL_V") == "1":
        # 실제-v far(2026-07-20, 사용자 설계): far 속도원 = 말단 state 실제 속도(캡드롭 반영).
        cfg.mpc.leader_mfd_far_real_speed = True
    if _os.environ.get("NU_BASE"):
        # Wang et al. 세트 재현용(2026-07-20): base ν(비혼잡 포함 단일 anticipation 계수).
        cfg.network.metanet_nu_km2_h = float(_os.environ["NU_BASE"])
    if _os.environ.get("KAPPA"):
        cfg.network.metanet_kappa_veh_km_lane = float(_os.environ["KAPPA"])
    if _os.environ.get("VFREE"):
        cfg.network.v_free = float(_os.environ["VFREE"])
    if _os.environ.get("RHO_CRIT"):
        cfg.network.rho_crit = float(_os.environ["RHO_CRIT"])
    if _os.environ.get("MFD_FAR") == "1":
        cfg.mpc.leader_mfd_far_enabled = True  # far(MFD tail) cost-to-go 가산 → near 깊이 절감 검증
    if _os.environ.get("MFD_FAR") == "0":
        # terminal cost 없는 리더 A/B(2026-07-19 밤샘 sweep): far cost-to-go 완전 해제.
        cfg.mpc.leader_mfd_far_enabled = False
    if _os.environ.get("FOLLOWER_TC") == "1":
        # (ii-b) follower own-TTS terminal tail(램프+blocked 삼각 배수) ON — 링크-agent 경로
        # (_solve_freeway_agent_local, PFO 해당). seg13 경로는 미구현(P-Stack TC축=MFD_FAR).
        cfg.mpc.follower_terminal_cost_enabled = True
    if _os.environ.get("FAR_STATE_AWARE") is not None:
        # ★동결 기본 ON(2026-07-16). far가 v_free·상수 g_fw=300으로 박혀 차선폐색·과포화를
        # 못 보던 결함 수정 — t_trav=Σℓ/V(ρ_i), g_fw=min_i cap(lanes_i)(METANET 기본도 유도,
        # 새 상수 0). 170_incident −10.45%→+1.21%, 최악 −10.45%→−4.34%.
        # FAR_STATE_AWARE=0으로 구 상수식 복원(재현용).
        cfg.mpc.leader_mfd_far_state_aware = _os.environ["FAR_STATE_AWARE"] == "1"
    # 6월말 hysteresis 작업 물리 원복 A/B 훅(2026-07-19): RHO_MAX=180(문헌값, 2e9d084가 95로
    # 반토막)·CAPDROP=0(ν regime-split OFF, f329696이 ON+250). 기본 미설정 시 현행 유지.
    if _os.environ.get("FW_BUFFER"):
        # 완충 세그먼트(2026-07-19): 양쪽 각 N셀 무제어 METANET 체인 — 혼잡의 자연 형성·해소를
        # 경계 가정 대신 실제 셀로. plant 전용(컨트롤러 무접촉), 최종 경계는 표준 자유출구.
        cfg.network.freeway_buffer_segments = int(_os.environ["FW_BUFFER"])
    if _os.environ.get("TERM_ZG") == "1":
        # 구경계 복원(2026-07-19): sink 밀도 = 직전 세그먼트(zero-gradient) + 배출 cap 해제.
        # ρ180 조합 검증용 — plant·예측모델 동시 적용(terminal_zero_gradient).
        cfg.network.terminal_zero_gradient = True
    if _os.environ.get("RHO_MAX"):
        cfg.network.rho_max = float(_os.environ["RHO_MAX"])
    if _os.environ.get("CAPDROP") is not None:
        cfg.network.capacity_drop_anticipation = _os.environ["CAPDROP"] == "1"
    if _os.environ.get("NU_CONG"):
        # ν_cong 재캘리브레이션(2026-07-19): ρ180 물리에서 실증 capacity drop 5~15%에 맞춤.
        cfg.network.metanet_nu_cong_km2_h = float(_os.environ["NU_CONG"])
    if _os.environ.get("QPROT_MV"):
        # VdB 시나리오4 재현: 보호 큐(최대길이 제약) 설정. QPROT_W>0이면 예측형 벌점
        # (local_signal_plant ramp-aware rollout에 소프트 제약), 0이면 계측만.
        cfg.mpc.protected_queue_movement = _os.environ["QPROT_MV"]
        cfg.mpc.protected_queue_max_veh = float(_os.environ.get("QPROT_MAX", "50"))
        cfg.mpc.protected_queue_weight = float(_os.environ.get("QPROT_W", "0"))
    if _os.environ.get("METER_QCON") == "1":
        # spillback-방지 제약 내재화(2026-07-19): follower 후보·budget 사영·최종 조립이
        # 공유하는 metering 하한(계획-집행 정합). 외부 감독층 METER_QOVR와 구분.
        cfg.mpc.meter_queue_constraint_enabled = True
        if _os.environ.get("METER_QCON_FRAC"):
            cfg.mpc.meter_queue_constraint_frac = float(_os.environ["METER_QCON_FRAC"])
    # far(MFD tail) 배수율 재캘리브레이션 훅(2026-07-19): 경계 수정 후 신물리 NC 실측으로
    # urban 혼잡 배수율이 구 캘리브레이션(g_cong=500)의 절반 수준(~280)임이 확인됨 —
    # 잔존 urban 비용 과소평가가 P-CENT/계층의 과잉 metering(근시 병리) 원인.
    for _far_env, _far_attr in (
        ("FAR_G_FREE", "leader_mfd_far_g_free"),
        ("FAR_G_CONG", "leader_mfd_far_g_cong"),
        ("FAR_NCRIT", "leader_mfd_far_ncrit"),
        ("FAR_G_FW", "leader_mfd_far_g_fw"),
    ):
        if _os.environ.get(_far_env):
            setattr(cfg.mpc, _far_attr, float(_os.environ[_far_env]))
    if _os.environ.get("BUDGET_OFF") == "1":
        # "가격만" arm: N_UF hard budget 제거, follower는 PFO autonomous metering.
        # 리더는 여전히 가격(green/metering/vsl/offset)을 계산·전달한다.
        cfg.mpc.leader_budget_off = True
    if _os.environ.get("BUDGET_FAIR") == "1":
        # 후보 budget을 축별로 균등 표본(np-major 절단으로 N_P 축이 굶는 문제). 기본 OFF.
        cfg.mpc.leader_budget_fair = True
    if _os.environ.get("NUF_RADIUS_STRICT") == "1":
        # 리더 국소 반경을 실제로 구속(앵커도 반경으로 clip). 기본 OFF=비트동일.
        cfg.mpc.leader_local_radius_strict = True
    if _os.environ.get("METER_BOX"):
        # METER-BOX(2026-07-17): SEG13 metering 후보를 m_prev±R 박스 5점으로 + 예산
        # 사영도 같은 박스. 값=R[veh/h]. 권장 300(=가격 FD 측정폭, 외삽 소멸).
        cfg.mpc.seg13_meter_box_veh_h = float(_os.environ["METER_BOX"])
    if _os.environ.get("METER_BOX_UP"):
        # 비대칭: 올림폭만 따로(내림=METER_BOX 유지). 파국=하방 고착이어서 회복만 가속.
        cfg.mpc.seg13_meter_box_up_veh_h = float(_os.environ["METER_BOX_UP"])
    if _os.environ.get("VSL_BOX"):
        # VSL 앵커를 previous로 + 반폭[km/h]. 기존 snapshot 앵커는 스텝당 최대 50 관측.
        cfg.mpc.seg13_vsl_box_kmh = float(_os.environ["VSL_BOX"])
    if _os.environ.get("BOX_WALK") == "1":
        # 리더 rollout에 박스 다중스텝 도달 모델링(METER_BOX 필요). 200_w 회복 시야 복원.
        cfg.mpc.leader_rollout_box_walk = True
    if _os.environ.get("BOX_WALK_VG") == "1":
        # VSL·green도 rollout에 끝 지속(edge persistence) walk. BOX_WALK와 병용 권장.
        cfg.mpc.leader_rollout_box_walk_vg = True
    if _os.environ.get("BASELINE_BOX") == "1":
        # PFO(비-SEG13) 이동 한계를 walk-MVG와 동일치로(meter 300/green 6s) — 공정비교용.
        cfg.mpc.baseline_move_box = True
    if _os.environ.get("NUF_RADIUS"):
        # 반경 상수 override. trust 산수 복원치 = 4램프 × 0.20 × 1500 = 1200.
        cfg.mpc.leader_local_nuf_radius_veh_h = float(_os.environ["NUF_RADIUS"])
    if _os.environ.get("NUF_UPPER"):
        # 반사실(2026-07-23): N_UF 상한 강제 → 리더를 tight metering으로 눌러 loose와 비교.
        cfg.mpc.N_UF_star_range = [0.0, float(_os.environ["NUF_UPPER"])]
    if _os.environ.get("VSL_SMOOTH_W"):
        # VSL 마찰 override(2026-07-23): cooldown VSL 미회복 원인 — 마찰(0.1×15=1.5) > 이득 → 0으로 제거 테스트.
        cfg.freeway_follower.vsl_smoothness_weight = float(_os.environ["VSL_SMOOTH_W"])
    if _os.environ.get("NO_FRICTION") == "1":
        # 모든 마찰(smoothness) 끄기(2026-07-23 사용자) — metering/vsl/offset/green 전부 0.
        cfg.freeway_follower.metering_smoothness_weight = 0.0
        cfg.freeway_follower.vsl_smoothness_weight = 0.0
        cfg.urban_follower.offset_smoothness_weight = 0.0
        cfg.urban_follower.green_smoothness_weight = 0.0
    if _os.environ.get("MFD_FAR_W"):
        cfg.mpc.leader_mfd_far_weight = float(_os.environ["MFD_FAR_W"])
    if _os.environ.get("FAR_D0") == "1":
        cfg.mpc.leader_mfd_far_at_d0 = True  # depth=0에서도 rollout+far 채점(얕은 leader 검정)
    if _os.environ.get("EARLY_STOP") == "1":
        cfg.mpc.leader_rollout_early_stop = True  # A2: 러닝베스트 초과 rollout 중도 폐기(선택 불변)
    if _os.environ.get("OPT12") == "1":
        cfg.mpc.leader_skip_local_refinement = True  # OPT1: local 스텝 refined 재정련 생략
        cfg.mpc.leader_rollout_early_stop = True     # OPT2: incumbent 초과 rollout exact 조기절단
    if _os.environ.get("OPT2") == "1":
        cfg.mpc.leader_rollout_early_stop = True     # OPT2 — 판정 기각(절감 미미 + inf metadata 부작용)
    if _os.environ.get("OPT3") == "1":
        cfg.mpc.leader_proxy_near_far = True         # OPT3 — 판정 기각(far 랭킹 불가, 단독 +1036)
    if _os.environ.get("OPT1") == "1":
        cfg.mpc.leader_skip_local_refinement = True  # OPT1 단독(local refined 생략) — 깨끗한 단독 측정
    profile = DemandProfile(cfg, scenario)
    sim = MixedTrafficSimulator(cfg)
    controller = make_controller(controller_id, cfg)
    if _os.environ.get("PRICE_ITER") and hasattr(controller, "price_iter_max"):
        controller.price_iter_max = int(_os.environ["PRICE_ITER"])  # B: leader↔follower dual-ascent 반복 횟수
    if _os.environ.get("PRICE_REFRESH_INTERVAL") and hasattr(controller, "price_refresh_interval"):
        # 2단(2026-07-18): 가격 refresh 스텝 간격. 1=매스텝(비트동일), 2=격스텝(FD rollout 절반).
        controller.price_refresh_interval = int(_os.environ["PRICE_REFRESH_INTERVAL"])
    if _os.environ.get("PRICE_SPSA") == "1" and hasattr(controller, "price_spsa_enabled"):
        controller.price_spsa_enabled = True  # SPSA: per-lever FD → 동시섭동(가격층 O(n))
    if _os.environ.get("SPSA_K") and hasattr(controller, "price_spsa_pairs"):
        controller.price_spsa_pairs = int(_os.environ["SPSA_K"])
    if _os.environ.get("MFD_FAR_PRICE") == "1" and hasattr(controller, "price_far_enabled"):
        controller.price_far_enabled = True  # E1: 가격 FD에도 far 합산(MFD_FAR=1 필요)
    if _os.environ.get("PRICE_HINGE") == "1" and hasattr(controller, "price_hinge_enabled"):
        controller.price_hinge_enabled = True  # rho_crit hinge를 가격 FD에(2026-07-23 사용자)
        if _os.environ.get("PRICE_HINGE_W"):
            controller.price_hinge_weight = float(_os.environ["PRICE_HINGE_W"])
    if _os.environ.get("ALLPRICE_OFF") == "1":
        # 전체 가격 채널 OFF(2026-07-23 사용자): budget(N_UF)만 남기고 green/metering/vsl/offset
        # 한계가격 전부 끔 → "가격이 TTT에 얼마나 기여하나" 격리. metering weight도 0.
        for _pa in ("signal_price_enabled", "metering_price_enabled", "vsl_price_enabled", "offset_price_enabled"):
            if hasattr(controller, _pa):
                setattr(controller, _pa, False)
        if hasattr(getattr(controller, "nash_solver", None), "metering_marginal_price_weight"):
            controller.nash_solver.metering_marginal_price_weight = 0.0
    if _os.environ.get("LINK_SHARE") and hasattr(controller, "nuf_link_share_mode"):
        controller.nuf_link_share_mode = str(_os.environ["LINK_SHARE"])  # {density, search, off}
    if _os.environ.get("VSL_TRUST") and hasattr(controller, "vsl_price_trust_kmh"):
        _vt = _os.environ["VSL_TRUST"]  # "none"=trust 해제, 숫자=반경(kmh) — E2 분리 실험용
        controller.vsl_price_trust_kmh = None if _vt.lower() == "none" else float(_vt)
    if _os.environ.get("PRICE_SIGNALS") and hasattr(controller, "signal_price_signals"):
        # SUBSET-PRICE: 강결합 신호만 가격(예: "D,F") — 나머지는 own-TTS 자율.
        controller.signal_price_signals = {
            s.strip() for s in _os.environ["PRICE_SIGNALS"].split(",") if s.strip()
        }
    if _os.environ.get("METER_PRICE_MODE") == "level" and hasattr(controller, "nash_solver"):
        # 구 B3 계보(가격이 레벨 조절 + soft anchor) 재현용 — 기본은 split(총량 equality).
        if hasattr(controller.nash_solver, "metering_price_split"):
            controller.nash_solver.metering_price_split = False
    if _os.environ.get("LEADER_DEDUPE") == "1" and hasattr(controller, "candidate_dedupe_enabled"):
        controller.candidate_dedupe_enabled = True  # A1: 같은 N_UF 후보 solve+rollout 재사용
    if _os.environ.get("PRICE_LITE") == "1" and hasattr(controller, "price_lite"):
        controller.price_lite = True  # B: baseline+one-sided+스텐실 재활용+얕은 가격 rollout
    if _os.environ.get("METER_PRICE_DELTA") and hasattr(controller, "metering_price_delta_veh_h"):
        # δ 스캔(2026-07-14): metering 가격 FD probe 폭 override. 기존 d_r =
        # max(δ, trust_frac·cap)에선 trust(0.25·1500=375)가 작은 δ를 가리므로, B3TR
        # 원칙(측정한 구간에서만 가격 유효 — 측정폭=trust 반경)을 유지한 채 δ가
        # 지배하도록 trust_frac도 δ/cap으로 연동(cap 균일 1500). 상향 cert 게이트 불변.
        _mpd = float(_os.environ["METER_PRICE_DELTA"])
        controller.metering_price_delta_veh_h = _mpd
        if getattr(controller, "metering_price_trust_frac", None) is not None:
            _cap_min = min(cfg.network.ramp_capacity_veh_h.values())
            controller.metering_price_trust_frac = _mpd / max(_cap_min, 1.0e-9)
    if _os.environ.get("OFFSET_PRICE") == "1" and hasattr(controller, "offset_price_enabled"):
        # 단일 offset 가격 진단 프로브(2026-07-15): 순수 offset 그래디언트 스파이크 관측용
        # (green×offset cross와 분리). 기본 OFF — env 지정 시에만.
        controller.offset_price_enabled = True
    if _os.environ.get("OFFSET_INNER_ITER") and hasattr(controller, "offset_price_inner_iters"):
        # 스텝 내 재선형화(SQP식 trust-region 걷기, 2026-07-15): offset 가격이 trust 한 칸에
        # 갇히는 문제를, 경계에 닿을 때마다 새 운영점에서 가격을 재측정하며 K회 걷어 해소.
        # OFFSET_PRICE=1과 함께 써야 발화(offset_price_enabled 게이트). 기본 0=OFF(비트동일).
        controller.offset_price_inner_iters = int(_os.environ["OFFSET_INNER_ITER"])
    if _os.environ.get("CROSS_ON") == "1":
        # 구 동결(②) 재현용 — cross 2종 복원. 2026-07-16부터 기본 OFF.
        for _attr in ("green_offset_cross_price_enabled", "vsl_meter_cross_price_enabled"):
            if hasattr(controller, _attr):
                setattr(controller, _attr, True)
    if _os.environ.get("CROSS_GATE") == "1" and hasattr(controller, "cross_cliff_gate_enabled"):
        # 절벽 게이팅(2026-07-15): cross 2종을 wu_b3_cliff_both_* 발화 스텝에만 활성화.
        # cross ON 고정=5승3패 / OFF 고정=7승1패 → 레짐별로 갈아끼우면 8승0패(오라클) 기대.
        # 기본 미지정=현행(비트동일). CROSS_OFF와 동시 지정 금지(OFF가 우선).
        controller.cross_cliff_gate_enabled = True
    if _os.environ.get("CROSS_OFF") == "1":
        # cross 가격 2종 동시 제거(2026-07-15). 8셀 실측: 비절벽 셀은 OFF가 −164~−420 이득,
        # 절벽 셀은 ON이 +27~+380 이득(분리 축은 부하 아닌 capacity-drop 문턱).
        # 종합 cross OFF=7승1패 / ON=5승3패. 기본 미지정=현행 유지(비트동일).
        for _attr in ("green_offset_cross_price_enabled", "vsl_meter_cross_price_enabled"):
            if hasattr(controller, _attr):
                setattr(controller, _attr, False)
    # ---- 채널 분리 A/B(2026-07-16): 두 cross를 따로 껐다 켠다 ----
    # 가설: 비절벽의 해악은 g×o(어디서나 활성: 전형무게 0.386/0.428),
    #       절벽의 이득은 v×m(절벽서만 각성: 0.008 → 0.091, 11배).
    # 맞으면 게이트·문턱 없이 "g×o OFF + v×m ON" 단일 고정으로 8승0패 가능.
    # 지금까지 CROSS_OFF가 둘을 동시에 껐으므로 이 칸은 미측정.
    if _os.environ.get("GO_CROSS_OFF") == "1" and hasattr(controller, "green_offset_cross_price_enabled"):
        controller.green_offset_cross_price_enabled = False   # v×m만 남김
    if _os.environ.get("VM_CROSS_OFF") == "1" and hasattr(controller, "vsl_meter_cross_price_enabled"):
        controller.vsl_meter_cross_price_enabled = False      # g×o만 남김
    if _os.environ.get("VSL_PRICE_DELTA") and hasattr(controller, "vsl_price_delta_kmh"):
        # Task C cross probe(2026-07-14): vsl 가격 FD 폭 override — vsl×meter cross의
        # δv를 δm과 비례 확대해 2차 혼합곡률 재측정(소δ에서 죽는지 vs 진짜 평탄).
        controller.vsl_price_delta_kmh = float(_os.environ["VSL_PRICE_DELTA"])
    if _os.environ.get("GREEN_PRICE_DELTA") and hasattr(controller, "signal_price_delta_sec"):
        # green 가격 FD 폭 override — green×offset cross의 δp 확대(6→24s 등) 겸용.
        controller.signal_price_delta_sec = float(_os.environ["GREEN_PRICE_DELTA"])
    if _os.environ.get("GO_CROSS_OFFSET_DELTA") and hasattr(controller, "green_offset_cross_offset_delta_sec"):
        controller.green_offset_cross_offset_delta_sec = float(_os.environ["GO_CROSS_OFFSET_DELTA"])
    if _os.environ.get("RELEASE_FLOOR") is not None:
        # 회랑 예산 하한 α override: 0=구 부등식(하한 없음), 1=구 등식, 기본 0.65.
        _rf_target = getattr(controller, "nash_solver", controller)
        if hasattr(_rf_target, "seg13_release_floor_frac"):
            _rf_target.seg13_release_floor_frac = float(_os.environ["RELEASE_FLOOR"])
    if _os.environ.get("SEG13") == "1":
        # 13-player(2026-07-10 승인): freeway를 segment agent 8개로 분해 + 예산 simplex 사영.
        # P-STACK 계열은 .nash_solver, PFO(WU-FAITHFUL-FOLLOWER)는 follower 자체가 controller.
        _seg_target = getattr(controller, "nash_solver", controller)
        if hasattr(_seg_target, "segment_agents"):
            _seg_target.segment_agents = True
        if _os.environ.get("SEG13_TRAJ") == "0" and hasattr(_seg_target, "seg13_traj_exchange"):
            _seg_target.seg13_traj_exchange = False  # v0(hold-constant) 재현 A/B용
        if _os.environ.get("SEG13_NBR") and hasattr(_seg_target, "seg13_neighbor_weight"):
            # radius-1 국소 rollout + 이웃 차량수 비용(PFO 강화 기준선; 플래그십은 OFF 유지).
            _seg_target.seg13_neighbor_weight = float(_os.environ["SEG13_NBR"])
        if _os.environ.get("SEG13_INEQ") == "0" and hasattr(_seg_target, "seg13_budget_inequality"):
            _seg_target.seg13_budget_inequality = False  # 등식 복원 A/B(2026-07-14 동결 후)
        if _os.environ.get("SEG13_INEQ") == "1" and hasattr(_seg_target, "seg13_budget_inequality"):
            # 예산 부등식 ablation: Σmeter ≤ budget(하향 자율 존중) — 170_skew whipsaw 인과 검증.
            _seg_target.seg13_budget_inequality = True
    if _os.environ.get("CAND"):
        # 헤드룸 탐침: leader 후보 격자 밀도(OPT12 절감분 재투자 실험).
        cfg.mpc.leader_candidate_count = int(_os.environ["CAND"])
    if _os.environ.get("FH"):
        # 헤드룸 탐침: freeway follower 예측 horizon(steps) 연장.
        cfg.freeway_follower.freeway_prediction_horizon_steps = int(_os.environ["FH"])
    if _os.environ.get("G_FW"):
        # far freeway drain 계수 재보정 probe(8-seg: 본선 2차항이 자유류까지 과대 계상 가설).
        cfg.mpc.leader_mfd_far_g_fw = float(_os.environ["G_FW"])
    if _os.environ.get("FAR_FF") == "1":
        # far 자유류 오프셋(자기정규화 수선) — 검증 전 기본 OFF.
        cfg.mpc.leader_mfd_far_freeflow_offset = True
    if _os.environ.get("METER_PRICE_W") and hasattr(controller, "nash_solver"):
        # metering marginal price 가중(기본 1.0). 실측: price 항이 own-TTS를 지배해
        # 내부 rung 선택 0/160(부호→끝점 80~85%) — w<1이면 own-TTS 곡률이 살아나
        # 내부 최적 가능. w→0 = 분해 손실 복귀(PFO metering)이므로 스윕으로 균형점 탐색.
        controller.nash_solver.metering_marginal_price_weight = float(
            _os.environ["METER_PRICE_W"])
    if _os.environ.get("NP_OFF") == "1" and hasattr(controller, "nash_solver"):
        # D-green 진단 probe: N_P dual(λ_P) 차단 — 8-seg 경부하 λ 폭주 인과 확인용.
        controller.nash_solver.np_price_enabled = False
    if _os.environ.get("LEADER_HINGE") == "1":
        # leader 채점 hinge 복원(F1 2종을 candidate 랭킹 전용으로 이관) — A/B용.
        cfg.mpc.leader_hinge_enabled = True
    if _os.environ.get("LEADER_HINGE") == "0":
        # 기본 ON 전환(2026-07-12) 이후의 해제 ablation용.
        cfg.mpc.leader_hinge_enabled = False
    if _os.environ.get("NP_BIAS") == "1":
        # r̂ 편향 보정: λ̂ target을 실현 공간으로 환산 — A/B용(기본 OFF).
        cfg.mpc.np_bias_correction = True
    if _os.environ.get("NP_CAP"):
        # 방법 A cap 포화 A/B: λ 상한 override(기본 10.0 — dhigh2 잔차 +476~534 진단).
        getattr(controller, "nash_solver", controller).lambda_np_cap = float(_os.environ["NP_CAP"])
    if _os.environ.get("RELEASE_FLOOR_FIXED") == "1":
        _seg_t = getattr(controller, "nash_solver", controller)
        if hasattr(_seg_t, "seg13_release_floor_adaptive"):
            _seg_t.seg13_release_floor_adaptive = False  # 고정 α=0.65 복원(A/B)
    if _os.environ.get("NP_DEADBAND_V2") == "1":
        # deadband v2: 위반 신호의 저stock 게이트 우회 — 잠금해제 A/B용(기본 OFF).
        cfg.mpc.np_deadband_violation_override = True
    if _os.environ.get("BETA_EST") == "1":
        # 층2 β̂ 낙관편향 추정기(추정 전용, 행동 불변) — leader_beta_hat 등 진단 export.
        cfg.mpc.leader_bias_estimator = True
    if _os.environ.get("GUARD_BETA") == "1":
        # 층2 β̂ 보정 guard: β̂·leader_pred vs incumbent_pred 비교(BETA_EST=1 필요).
        cfg.mpc.fallback_guard_beta = True
    if _os.environ.get("REGRET_K"):
        # 층2 trailing-regret: 최근 k스텝 실현 > incumbent 예측×1.10 → k스텝 강제 incumbent.
        cfg.mpc.regret_guard_steps = int(_os.environ["REGRET_K"])
    if _os.environ.get("NP_PD_ITER"):
        # 방법 A: candidate 내부 primal-dual 반복 K(0=OFF, 현행 λ̂ 1회 선반영).
        cfg.mpc.np_primal_dual_iters = int(_os.environ["NP_PD_ITER"])
    if _os.environ.get("NP_PD_GAIN"):
        # PD 스텝게인 배수. 기본 25는 gain 0.25 → 실측 잔차 324~580에 곱하면 81~145로
        # cap(10)을 12배 초과 → λ가 {0, cap} bang-bang(400스텝 중간값 0개, 2026-07-16 실측).
        cfg.mpc.np_pd_gain_mult = float(_os.environ["NP_PD_GAIN"])
    if _os.environ.get("FB_OFF") == "1":
        # fallback guard 해제 — N_P-active leader 기각 억압 제거(잠금해제 A/B용).
        cfg.mpc.stackelberg_enable_fallback = False
    if _os.environ.get("EPS_GAP") == "1":
        # ε-best-response gap probe(리뷰 2.2/2.8): 고정점 단독 재최적화 진단, 행동 불변.
        cfg.mpc.eps_gap_probe = True
    if _os.environ.get("NP_CAND_LAMBDA") == "1":
        # 리뷰 4안: 후보별 λ̂ 1회 선반영 — N_P가 당스텝 follower 반응에 작용(A/B용).
        cfg.mpc.np_candidate_lambda = True
    if _os.environ.get("NP_CAND_LAMBDA") == "0":
        # 구거동 재현(스텝 내 λ 동결) — 기본 ON 전환(2026-07-12) 이후의 ablation용.
        cfg.mpc.np_candidate_lambda = False
    if _os.environ.get("NP_FIX") == "0":
        # λ windup 수선 해제(구거동 재현 A/B): 내부 투영·deadband 모두 끔.
        cfg.mpc.np_target_interior_frac = 0.0
        cfg.mpc.np_dual_deadband_frac = 0.0
    if _os.environ.get("NUF_DUAL") == "1":
        # 8단계 ablation: N_UF 총량을 equality 사영 대신 λ_UF dual(step 간 적분)로 추적.
        cfg.mpc.wu_faithful_nuf_coordination_mode = "dual"
    if _os.environ.get("SEG13_V1") == "1":
        # SPLIT-PRICE v1 재현: incumbent도 meter 가격-레벨 조절(7p 플래핑 병리 A/B용).
        _seg_t = getattr(controller, "nash_solver", controller)
        if hasattr(_seg_t, "seg13_meter_price_standing"):
            _seg_t.seg13_meter_price_standing = True
    # ---- SUP_PFO(2026-07-20, 사용자 설계): 기권 후보 내장 감독자 ----
    # 매 스텝 P-Stack 해와 링크-PFO 에뮬레이션 해를 둘 다 계산, 리더 채점 V(=h3 결합
    # rollout TTT + far)로 공통 채점해 나은 쪽을 집행 — 리더 후보군에 "무개입(=PFO 위임)"
    # 옵션이 상시 포함되는 것과 등가라 채점 정확도 한도 내에서 P-Stack ≥ PFO 구조 보장.
    # PFO 인스턴스는 새로 생성(SEG13 env는 본 컨트롤러 인스턴스에만 적용됐으므로 링크 모드,
    # BASELINE_BOX=1이면 cfg 경유로 동일 이동한계). 미설정=기존 거동.
    _sup_pfo = None
    _sup_pfo_cfg = None
    if _os.environ.get("SUP_PFO") == "1" and controller_id.startswith("P-STACK"):
        import copy as _copy
        # PFO 인스턴스용 cfg 사본: seg13 전용 박스 필드를 기본값으로 되돌림(링크 경로 가드 회피)
        # + baseline_move_box 보장(walk-MVG와 동일 이동한계). 본 cfg는 무접촉.
        _sup_pfo_cfg = _copy.deepcopy(cfg)
        # seg13 전용 박스 가드는 `is not None` 검사 — 반드시 None으로 되돌린다.
        _sup_pfo_cfg.mpc.seg13_meter_box_veh_h = None
        if hasattr(_sup_pfo_cfg.mpc, "seg13_meter_box_up_veh_h"):
            _sup_pfo_cfg.mpc.seg13_meter_box_up_veh_h = None
        _sup_pfo_cfg.mpc.seg13_vsl_box_kmh = None
        _sup_pfo_cfg.mpc.baseline_move_box = True
        _sup_pfo = make_controller("WU-FAITHFUL-FOLLOWER", _sup_pfo_cfg)
        print("[SUP_PFO] 기권 후보(링크-PFO 에뮬레이션) 활성", flush=True)

    def _sup_score(ctrl: ControlAction, forecast) -> float:
        """감독자 공통 채점: 고정 control로 h3 결합 rollout TTT + far(리더 V와 동형)."""
        from src.simulation.coupling import run_coupled_interval
        from src.controllers.stackelberg_mpc import mfd_far_cost_to_go
        s = sim.state.copy()
        total = 0.0
        h = max(1, int(cfg.mpc.horizon_steps))
        for k in range(min(h, len(forecast))):
            res = run_coupled_interval(s, ctrl, forecast[k], cfg)
            total += float(res.urban_ttt + res.freeway_ttt)
            s.time_sec += cfg.simulation.control_interval
        total += float(mfd_far_cost_to_go(cfg, s))
        return total

    previous: Optional[ControlAction] = None
    steps = max(1, int(round(cfg.simulation.T_total / cfg.simulation.control_interval)))
    run_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    out_dir = output_root / controller_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tier1 체크포인트/재개(2026-07-20, 사용자 설계) ----
    # CHECKPOINT=1(기본): 매 스텝 plant 전체 상태+누적치를 checkpoints.pkl에 append-pickle.
    # INIT_STATE=<checkpoints.pkl 경로>(+INIT_STEP=k, 기본 -1=마지막): 그 스냅샷에서 재개.
    # 한계(문서화): 컨트롤러 내부 상태(λ·r̂·incumbent)는 미저장 — NC는 정확 재개,
    # 컨트롤러 런은 plant 정확·컨트롤러 cold start 근사(T연장·프로브·진단용).
    import pickle as _pk
    start_step = 0
    if _os.environ.get("INIT_STATE"):
        _want = int(_os.environ.get("INIT_STEP", "-1"))
        _snap = None
        with open(_os.environ["INIT_STATE"], "rb") as _f:
            while True:
                try:
                    _s = _pk.load(_f)
                except EOFError:
                    break
                if _want < 0 or _s["step"] <= _want:
                    _snap = _s
                if _want >= 0 and _s["step"] >= _want:
                    break
        if _snap is None:
            raise SystemExit(f"INIT_STATE에서 스냅샷을 못 찾음: {_os.environ['INIT_STATE']}")
        sim.state = _snap["state"]
        for _k, _v in _snap.get("sim_num", {}).items():
            setattr(sim, _k, _v)
        for _k, _v in _snap.get("sim_dicts", {}).items():
            setattr(sim, _k, dict(_v))
        previous = _snap.get("previous_control")
        start_step = int(_snap["step"]) + 1
        print(f"[resume] {_os.environ['INIT_STATE']} step {_snap['step']} → {start_step}부터 재개", flush=True)

    # ---- FAR_GATE(2026-07-20 밤, 사용자 설계): far를 구조적 교란에만 자동 연동 ----
    # b6 해부 판정: far는 평시 순손해(대리모형 오차 > 시력), incident에선 +7p(회복기 관리).
    # 게이트 = 폐쇄가 forecast에 존재 OR (폐쇄 이력 링크가 아직 임계 위 = 회복 미완).
    # 시계 상수 없음 — 이벤트(폐쇄 일정)+상태(배수 완료)만으로 개폐. FAR_GATE=1일 때만.
    # FAR_GATE=1: 폐쇄 일정 트리거. FAR_GATE=2(사용자 설계): capacity-drop 상태 트리거 —
    # "임계 초과 + 실배출 < 0.95·용량" 세그 존재 시 ON(예방 실패 → 회복 관리 국면),
    # 전 링크 최대 ρ < ρ_crit로 회복하면 OFF(히스테리시스, 시계 상수 없음).
    # FAR_GATE=3(하이브리드, 2026-07-21): 폐쇄 예보 OR capdrop 실측 — incident 선제 포석
    # (b4의 s5~10 사전 배치) + 평시 b9 동일. 폐쇄 없는 셀에선 m2와 완전 동치.
    _fargate_mode = _os.environ.get("FAR_GATE", "")
    _fargate_on = _fargate_mode in ("1", "2", "3")
    # capdrop 검출 문턱(2026-07-21): 실배출 < thr·용량이면 drop. 기본 0.95(실측 지속 ~10%의
    # 절반). FAR_GATE_THR로 하향(예 0.90) — 일시적 딥 무시, 진짜 지속붕괴에만 게이트 개방.
    _fargate_thr = float(_os.environ.get("FAR_GATE_THR", "0.95"))
    _fargate_links: set = set()
    _fargate_stress = False

    _ckpt_on = _os.environ.get("CHECKPOINT", "1") == "1"
    _pg_core = ["step", "time_sec", "step_total_ttt", "step_urban_ttt", "step_freeway_ttt",
                "cumulative_total_ttt", "cumulative_urban_ttt", "cumulative_freeway_ttt",
                "computation_time_sec"]

    print(f"=== {controller_id} ===", flush=True)
    for step in range(start_step, steps):
        t = step * cfg.simulation.control_interval
        # 리더 후보 곡면 로깅(2026-07-22, price 곡률 진단용). LEADER_CAND_LOG=<path> 지정 시
        # 스텝별 (N_P,N_UF,objective) 후보 평가를 그대로 남긴다(추가 계산 없음, env-gated).
        if _os.environ.get("LEADER_CAND_LOG"):
            _os.environ["NUMSIM_STACKELBERG_PROGRESS_FILE"] = _os.environ["LEADER_CAND_LOG"]
            _os.environ["NUMSIM_STACKELBERG_PROGRESS_STEP"] = str(step)
        forecast = profile.horizon(t, cfg.mpc.horizon_steps + max(0, cfg.mpc.leader_value_depth))
        t0 = time.perf_counter()
        # WARMUP_NC_STEPS: 웜업 구간은 전 arm 공통 no-control(분석창 진입 상태 동일화 +
        # 컨트롤러 계산 절약, 표준 관행). 미지정(0)이면 기존 거동.
        if _fargate_on:
            if _fargate_mode == "1":
                from src.models.demand import merge_freeway_lane_loss
                _fg_merged = merge_freeway_lane_loss(list(forecast))
                _fg_active = {
                    l for l, segs in _fg_merged.items()
                    if any(float(v) > 0.0 for v in segs.values())
                }
                _fargate_links |= _fg_active
                for _l in list(_fargate_links):
                    if _l in _fg_active:
                        continue
                    _dens = sim.state.freeway_density.get(_l, [])
                    if _dens and max(_dens) < float(cfg.network.rho_crit):
                        _fargate_links.discard(_l)  # 회복 완료 → 게이트 해제
                _fg_new = bool(_fargate_links)
            else:
                # mode 2: capacity-drop 상태 검출(사용자 설계).
                from src.models.metanet import segment_flow_veh_h, desired_speed_kmh
                _rc = float(cfg.network.rho_crit)
                _vf = float(cfg.network.v_free)
                _all_sub = True
                _drop_seen = False
                for _l in cfg.network.freeway_links:
                    _dens = sim.state.freeway_density.get(_l, [])
                    _spds = sim.state.freeway_speed.get(_l, [])
                    _lns = sim.state.freeway_effective_lanes.get(_l, [])
                    for _i in range(len(_dens)):
                        _rho = float(_dens[_i])
                        if _rho <= _rc:
                            continue
                        _all_sub = False
                        _lam = float(_lns[_i]) if _i < len(_lns) else float(cfg.network.freeway_lanes)
                        _v = float(_spds[_i]) if _i < len(_spds) else 0.0
                        _flow = segment_flow_veh_h(_rho, _v, _lam)
                        _cap = segment_flow_veh_h(_rc, desired_speed_kmh(_rc, _vf, _rc), _lam)
                        if _flow < _fargate_thr * _cap:
                            _drop_seen = True
                if _drop_seen:
                    _fargate_stress = True   # drop 발생 → ON 래치
                elif _all_sub:
                    _fargate_stress = False  # 전 세그 임계 아래 = 회복 완료 → OFF
                _fg_new = _fargate_stress
                if _fargate_mode == "3":
                    # 하이브리드: 예보 지평 내 폐쇄가 있으면 선제 ON(capdrop 실측 전이라도).
                    from src.models.demand import merge_freeway_lane_loss
                    _fg_m3 = merge_freeway_lane_loss(list(forecast))
                    if any(float(v) > 0.0 for segs in _fg_m3.values() for v in segs.values()):
                        _fg_new = True
            if _fg_new != bool(cfg.mpc.leader_mfd_far_enabled):
                print(f"[FAR_GATE m{_fargate_mode}] step {step}: far {'ON' if _fg_new else 'OFF'}", flush=True)
            cfg.mpc.leader_mfd_far_enabled = _fg_new
        if step < int(_os.environ.get("WARMUP_NC_STEPS", "0")):
            control = baseline_control("no_control", cfg, sim.state, forecast[0])
        else:
            control = decide(controller_id, controller, sim, forecast, previous, cfg, step)
            # PRICE-CF(2026-07-23, 사용자 설계): 같은 state·같은 budget에서 **전체 가격 OFF**
            # (green/metering/vsl/offset) follower 선택을 함께 계산·기록(적용 안 함). price가
            # lever를 얼마나 바꿨나의 clean 측정 — 발산 두 궤적(confound) 대신 동일 state 반사실.
            if _os.environ.get("PRICE_CF") == "1" and controller_id.startswith("P-STACK") and control.ramp_metering:
                _ns = getattr(controller, "nash_solver", None)
                if _ns is not None and hasattr(_ns, "metering_marginal_price_weight"):
                    from src.controllers.leader import LeaderAction as _LA
                    _nuf = float(control.diagnostics.get("leader_realized_N_UF_star", getattr(control, "N_UF_star", 0.0)))
                    _npv = float(control.diagnostics.get("leader_realized_N_P_star", getattr(control, "N_P_star", 0.0)))
                    _pas = ("signal_price_enabled", "metering_price_enabled", "vsl_price_enabled", "offset_price_enabled")
                    _saved = {_a: getattr(controller, _a) for _a in _pas if hasattr(controller, _a)}
                    _w = _ns.metering_marginal_price_weight
                    try:
                        for _a in _saved:
                            setattr(controller, _a, False)
                        _ns.metering_marginal_price_weight = 0.0
                        _cf = _ns.solve(sim.state.copy(), _LA(_npv, _nuf), list(forecast), previous)
                        _cfc = getattr(_cf, "control", _cf)
                        for _r in control.ramp_metering:
                            control.diagnostics[f"cfon_meter_{_r}"] = float(control.ramp_metering[_r])
                            control.diagnostics[f"cfoff_meter_{_r}"] = float(_cfc.ramp_metering.get(_r, 0.0))
                        for _sg in list(control.green_times.keys()):
                            control.diagnostics[f"cfon_green_{_sg}"] = float(control.green_times[_sg])
                            control.diagnostics[f"cfoff_green_{_sg}"] = float(_cfc.green_times.get(_sg, 0.0))
                    except Exception as _e:
                        control.diagnostics["cf_error"] = 1.0
                    finally:
                        for _a, _v in _saved.items():
                            setattr(controller, _a, _v)
                        _ns.metering_marginal_price_weight = _w
            # GREEN_SWEEP(2026-07-23, 사용자 가설검증): 혼잡 스텝서 green을 넓게(±범위) 쓸어
            # TTT_global(green) 곡선 기록. 6s 국소 probe가 큰-스케일 이득을 놓치나 판정(곡선이
            # 선형이면 6s 충분, 비선형이면 넓은 probe가 이득). 가격 FD가 쓰는 바로 그 함수 호출.
            if _os.environ.get("GREEN_SWEEP") == "1" and controller_id.startswith("P-STACK") \
               and hasattr(controller, "_global_rollout_metrics_with_green"):
                _sw_steps = [int(x) for x in _os.environ.get("GREEN_SWEEP_STEPS", "14,17,20").split(",")]
                if step in _sw_steps:
                    import csv as _csv2
                    _tot_g = float(cfg.network.effective_green_total)
                    _sigs_g = _os.environ.get("GREEN_SWEEP_SIGS", "A,D").split(",")
                    _rng = float(_os.environ.get("GREEN_SWEEP_RANGE", "30"))
                    _stp = float(_os.environ.get("GREEN_SWEEP_STEP", "2"))
                    _pth_g = f"outputs/_diag/green_sweep_{_os.environ.get('GREEN_SWEEP_TAG','x')}.csv"
                    _new_g = not _os.path.exists(_pth_g)
                    with open(_pth_g, "a", newline="") as _fg:
                        _wr = _csv2.writer(_fg)
                        if _new_g:
                            _wr.writerow(["step", "time_sec", "signal", "p1", "ttt", "barrier", "ref_p1"])
                        for _sig in _sigs_g:
                            _ref_g = float(control.green_times.get(f"{_sig}_p1", _tot_g / 2.0))
                            _lo_g = max(6.0, _ref_g - _rng)
                            _hi_g = min(_tot_g - 6.0, _ref_g + _rng)
                            _p1 = _lo_g
                            while _p1 <= _hi_g + 1e-9:
                                try:
                                    _ttt_g, _bar_g = controller._global_rollout_metrics_with_green(
                                        sim.state.copy(), previous, list(forecast), _sig, _p1)
                                    _wr.writerow([step, t, _sig, round(_p1, 1), _ttt_g, _bar_g, round(_ref_g, 1)])
                                except Exception as _eg:
                                    _wr.writerow([step, t, _sig, round(_p1, 1), "ERR", str(_eg)[:50], round(_ref_g, 1)])
                                _p1 += _stp
            # SUP_GATE=fargate(2026-07-21, 사용자 설계): far 게이트가 ON인 스텝엔 감독자 OFF.
            # 원리 — 동일한 물리 조건(조율 지배 국면: capdrop/사고)이 far를 부르고 감독자를
            # 쫓아낸다. far ON(회복 다단계 조율)일 때 근시 PFO 스위칭이 포석을 깨므로(b11
            # incident/220/240 병리), 그 국면에선 리더 계획을 그대로 집행. 별도 신호 없이
            # 기존 far 게이트 상태(leader_mfd_far_enabled) 재사용 — 단일 게이트, 이중 스위치.
            _sup_off = (_os.environ.get("SUP_GATE") == "fargate"
                        and bool(cfg.mpc.leader_mfd_far_enabled))
            if _sup_pfo is not None and not _sup_off:
                # 기권 후보: 링크-PFO 해를 같은 상태에서 계산해 공통 V로 채점, 승자 집행.
                _pfo_ctrl = decide("WU-FAITHFUL-FOLLOWER", _sup_pfo, sim, forecast, previous, _sup_pfo_cfg, step)
                # SUP_FAST(2026-07-21): P-Stack 해의 V는 리더가 방금 계산한 best_objective를
                # 재사용(rollout 1회 절감, 무손실). 리더 objective와 감독 채점이 동형(near+far).
                if _os.environ.get("SUP_FAST") == "1" and "leader_candidate_best_objective" in control.diagnostics:
                    _v_ps = float(control.diagnostics["leader_candidate_best_objective"])
                else:
                    _v_ps = _sup_score(control, forecast)
                _v_pfo = _sup_score(_pfo_ctrl, forecast)
                if _v_pfo < _v_ps - 1.0e-9:
                    _pfo_ctrl.diagnostics["sup_pick_pfo"] = 1.0
                    _pfo_ctrl.diagnostics["sup_v_pstack"] = _v_ps
                    _pfo_ctrl.diagnostics["sup_v_pfo"] = _v_pfo
                    control = _pfo_ctrl
                else:
                    control.diagnostics["sup_pick_pfo"] = 0.0
                    control.diagnostics["sup_v_pstack"] = _v_ps
                    control.diagnostics["sup_v_pfo"] = _v_pfo
        # METER_QOVR(2026-07-19, 사용자 제안): spillback 금지 queue-override — ALINEA queue
        # override/VdB max-queue 제약과 동형. 램프 큐가 임계(기본 0.8×180)를 넘으면 다음
        # 인터벌에 임계 아래로 돌아오도록 방류 하한을 강제(간선 역류=blocked 큐 형성 차단).
        # 감독층 clamp라 어떤 컨트롤러에도 동일 적용(A/B용, 기본 미설정=무변경).
        if _os.environ.get("METER_QOVR") == "1" and control.ramp_metering:
            _q_frac = float(_os.environ.get("METER_QOVR_FRAC", "0.8"))
            _q_thr = _q_frac * float(cfg.network.ramp_queue_max_veh)
            _tc_h = cfg.simulation.T_c_h
            for _ramp in list(control.ramp_metering.keys()):
                _q_now = max(0.0, float(sim.state.ramp_queue.get(_ramp, 0.0)))
                _arr = max(0.0, float(forecast[0].ramp_arrival.get(_ramp, 0.0)))
                _m_need = _arr + max(0.0, _q_now - _q_thr) / max(_tc_h, 1.0e-9)
                _cap = float(cfg.network.ramp_capacity_veh_h.get(_ramp, 1500.0))
                _m_new = max(float(control.ramp_metering[_ramp]), min(_cap, _m_need))
                if _m_new > float(control.ramp_metering[_ramp]) + 1.0e-9:
                    control.diagnostics[f"meter_qovr_{_ramp}"] = _m_new - float(control.ramp_metering[_ramp])
                control.ramp_metering[_ramp] = _m_new
        # QPROT_REACT(2026-07-19): VdB SCOOT/UTOPIA 아날로그 — 보호 큐가 한계 90%를 넘으면
        # 해당 phase green을 반응적으로 증가(예측 없음, 집행만). 예측형(QPROT_W 벌점)과 대조용.
        if _os.environ.get("QPROT_REACT") == "1" and _os.environ.get("QPROT_MV"):
            _mv_p = _os.environ["QPROT_MV"]
            _qmax_p = float(_os.environ.get("QPROT_MAX", "50"))
            _qnow_p = max(0.0, float(sim.state.urban_movement_queue.get(_mv_p, 0.0)))
            if _qnow_p > 0.9 * _qmax_p:
                _spec_p = cfg.network.urban_movements.get(_mv_p, {})
                _ph_p = str(_spec_p.get("phase", ""))
                _sig_p = str(_spec_p.get("signal", ""))
                if _ph_p and _sig_p:
                    _other_p = f"{_sig_p}_p2" if _ph_p.endswith("p1") else f"{_sig_p}_p1"
                    _tot_p = float(cfg.network.effective_green_total)
                    _cur_p = float(control.green_times.get(_ph_p, _tot_p / 2.0))
                    _boost_p = min(10.0, max(0.0, _tot_p - 6.0 - _cur_p))
                    if _boost_p > 0.0:
                        control.green_times[_ph_p] = _cur_p + _boost_p
                        control.green_times[_other_p] = _tot_p - control.green_times[_ph_p]
                        control.diagnostics["qprot_react_boost"] = _boost_p
        # 반사실(2026-07-23): cooldown 구간 freeway VSL 강제 override — VSL이 회복 안 하는 게 TTT 까먹나 검증.
        if _os.environ.get("FORCE_VSL_COOLDOWN") and step >= int(_os.environ.get("VSL_COOLDOWN_STEP", "29")):
            _vv = float(_os.environ["FORCE_VSL_COOLDOWN"])
            for _k in list(getattr(control, "vsl", {}).keys()):
                if str(_k).startswith("FW"):
                    control.vsl[_k] = _vv
        compute_time = time.perf_counter() - t0
        log = sim.step(control, forecast[0], step)
        previous = control.copy()
        decision = {
            "step": step,
            "computation_time_sec": compute_time,
            "N_P_star": control.N_P_star,
            "N_UF_star": control.N_UF_star,
            "solver_evaluations": float(control.diagnostics.get("wu_faithful_local_evals", 0.0)),
            "wu_faithful_active": float(control.diagnostics.get("wu_faithful_follower_active", 0.0)),
            "leader_candidate_count": float(control.diagnostics.get("leader_candidate_count", 0.0)),
            "leader_full_evaluated_count": float(control.diagnostics.get("leader_candidate_full_evaluated_count", 0.0)),
            "leader_serial_override": float(control.diagnostics.get("leader_candidate_wu_metered_serial_override", 0.0)),
        }
        # 가격(wu_b2_/b3_/b4_/f3_)·P1.5(wu_p15_)·joint(wu_j_) 진단은 control_row에 안 실리므로 수집.
        decision.update({
            k: float(v) for k, v in control.diagnostics.items()
            if k.startswith(("wu_b2_", "wu_b3_", "wu_b4_", "wu_p15_", "wu_f3_", "wu_j_", "wu_joint_", "wu_lead_off_", "wu_eps_", "wu_faithful_np_", "wu_seg13_", "sup_"))
            and isinstance(v, (int, float, bool))
        })
        decision_rows.append(decision)
        control_rows.append(control_row(control, cfg, step, sim.state.time_sec))
        state_rows.append(state_row(sim.state, cfg, step))
        row = {
            "step": step,
            "time_sec": sim.state.time_sec,
            "step_total_ttt": log.freeway_ttt + log.urban_ttt,
            "step_urban_ttt": log.urban_ttt,
            "step_freeway_ttt": log.freeway_ttt,
            "cumulative_total_ttt": sim.total_ttt,
            "cumulative_urban_ttt": sim.urban_ttt,
            "cumulative_freeway_ttt": sim.freeway_ttt,
            "computation_time_sec": compute_time,
        }
        row.update({k: v for k, v in log.diagnostics.items() if isinstance(v, (int, float, bool))})
        run_rows.append(row)
        # ---- Tier1: 체크포인트 append + progress.csv 즉시 기록(중단 시 데이터 소실 방지) ----
        if _ckpt_on:
            _sim_num = {k: v for k, v in vars(sim).items() if isinstance(v, (int, float)) and not k.startswith("_")}
            _sim_dicts = {
                k: dict(v) for k, v in vars(sim).items()
                if isinstance(v, dict) and v and not k.startswith("_")
                and all(isinstance(x, (int, float)) for x in v.values())
            }
            with open(out_dir / "checkpoints.pkl", "wb" if step == start_step else "ab") as _f:
                _pk.dump({
                    "step": step, "time_sec": sim.state.time_sec, "state": sim.state.copy(),
                    "previous_control": control.copy(), "sim_num": _sim_num, "sim_dicts": _sim_dicts,
                }, _f)
        _pg_path = out_dir / "progress.csv"
        _pg_new = (step == start_step and start_step == 0) or not _pg_path.exists()
        with open(_pg_path, "w" if (step == start_step and start_step == 0) else "a", encoding="utf-8") as _f:
            if _pg_new:
                _f.write(",".join(_pg_core) + "\n")
            _f.write(",".join(str(row.get(c, "")) for c in _pg_core) + "\n")
        print(
            f"{controller_id} step {step + 1}/{steps} "
            f"cum_ttt={sim.total_ttt:.3f} step_ttt={row['step_total_ttt']:.3f} "
            f"solve={compute_time * 1000.0:.0f}ms",
            flush=True,
        )

    if hasattr(controller, "close"):
        controller.close()

    write_csv(out_dir / "run_log.csv", run_rows)
    write_csv(out_dir / "control_timeseries.csv", control_rows)
    write_csv(out_dir / "state_timeseries.csv", state_rows)
    write_csv(out_dir / "decision_diagnostics.csv", decision_rows)
    summary = summarize(controller_id, cfg, sim, decision_rows, run_rows, reference)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="sweet_190")
    parser.add_argument("--T-total", type=float, default=3600.0)
    parser.add_argument("--output", default="outputs/claude_style_sweet190_3600_20260629")
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg, scenario = build_cfg(args.scenario, args.T_total)
    reference = compute_free_flow_reference(cfg, scenario)
    selected = [item.strip() for item in args.controllers.split(",") if item.strip()]
    summaries: List[Dict[str, Any]] = []
    for controller_id in selected:
        summaries.append(run_one(controller_id, args.scenario, args.T_total, output_root, reference))

    no_control = next((row for row in summaries if row["controller_id"] == "NO-CONTROL"), None)
    for row in summaries:
        if no_control and row["controller_id"] != "NO-CONTROL":
            row["ttt_improvement_vs_no_control_pct"] = round(
                100.0 * (no_control["total_ttt"] - row["total_ttt"]) / max(no_control["total_ttt"], 1.0e-9),
                3,
            )
            row["delay_improvement_vs_no_control_pct"] = round(
                100.0 * (no_control["total_delay"] - row["total_delay"]) / max(no_control["total_delay"], 1.0e-9),
                3,
            )
        else:
            row["ttt_improvement_vs_no_control_pct"] = 0.0
            row["delay_improvement_vs_no_control_pct"] = 0.0

    write_csv(output_root / "summary.csv", summaries)
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n========== SUMMARY ==========", flush=True)
    for row in summaries:
        print(
            f"{row['controller_id']}: total_ttt={row['total_ttt']:.3f} "
            f"delay={row['total_delay']:.3f} "
            f"impr={row['ttt_improvement_vs_no_control_pct']:+.2f}% "
            f"completed={row['completed_vehicles']:.1f} "
            f"terminal={row['terminal_total_vehicles']:.1f} "
            f"mean_solve={row['mean_step_compute_sec']:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
