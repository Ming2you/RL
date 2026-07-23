from __future__ import annotations

import json
import os
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from src.controllers.distributed_coordinator import DistributedCoordinator
from src.controllers.leader import Leader, LeaderAction, leader_metadata
from src.controllers.nash_solver import NashResult, NashSolver
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState


# ---------- 층3(2026-07-14): 수작업 캘리브레이션 성분의 기하 지문(경고 전용) ----------
# 손튜닝 성분들은 특정 plant 기하(ramp merge 인덱스)에서 캘리브레이션된 암묵 전제를
# 갖는다. 2026-07-13에 merge 인덱스가 {4,6}→{3,5}로 바뀌며 전제가 조용히 만료됐다 —
# 이 지문은 그 만료를 __init__에서 감지해 stderr 경고 + 진단 카운트로 드러낸다.
# 수치 경로에는 절대 개입하지 않는다(경고·진단만).
_CALIB_MERGE_OLD_46 = {"R_D_W": 4, "R_F_W": 6, "R_D_E": 4, "R_F_E": 6}
CALIBRATION_FINGERPRINTS: Dict[str, Dict[str, Dict[str, int]]] = {
    # leader hinge 가중(leader_hinge_weight/spill_frac) — 2026-07-12 A/B는 구 merge {4,6}.
    "leader_hinge": {"ramp_merge": dict(_CALIB_MERGE_OLD_46)},
    # N_P dual deadband/내부투영(np_dual_deadband_frac/np_target_interior_frac) — 구 plant 튜닝.
    "np_deadband": {"ramp_merge": dict(_CALIB_MERGE_OLD_46)},
    # far(MFD tail) g 계수(leader_mfd_far_g_free/g_cong/g_fw/ncrit) — 구 plant probe calib.
    "leader_mfd_far": {"ramp_merge": dict(_CALIB_MERGE_OLD_46)},
}
# 경고는 프로세스당 성분별 1회만(후보 worker 재생성 시 반복 스팸 방지).
_calib_fingerprint_warned: set = set()


def leader_hinge_cost(cfg: "ExperimentConfig", states: list, forecast=None, force: bool = False) -> float:
    """Leader 채점 전용 hinge(2026-07-12 복원, 사용자 지시 — follower/가격 채널 불변).

    구 F1 follower hinge 2종을 leader의 candidate 채점(rollout 궤적)으로 이관:
      freeway: Σ_t Σ_seg max(0, ρ − ρ_crit_eff)·L_seg·λ_eff  [임계 초과 차량수]
      urban:   Σ_t Σ_link max(0, 점유 − frac·cap)            [spillback 여유부족, frac=0.5]
    선형·차량수×T_c_h = veh·h(TTT 동단위, w=1이 1차 정확값). far와 같은 격리 수준 —
    후보 랭킹에만 작용하고 가격 probe·follower 목적에는 들어가지 않는다(이중계상 방지).
    leader_hinge_enabled=False(기본)면 0 — 비트동일.
    force=True면 플래그 무관 계산(price-hinge용 — 가격 FD에서 직접 호출)."""
    if not force and not getattr(cfg.mpc, "leader_hinge_enabled", False):
        return 0.0
    from src.models.metanet import effective_rho_crit
    net = cfg.network
    tc_h = float(cfg.simulation.T_c_h)
    w = float(getattr(cfg.mpc, "leader_hinge_weight", 1.0))
    frac = float(getattr(cfg.mpc, "leader_hinge_spill_frac", 0.5))
    seg_len = float(net.freeway_segment_length_km)
    rho_crit = float(effective_rho_crit(net, None))
    # hinge v2(2026-07-13, 155_incident +943 귀속 수선): 예정된 차로폐쇄(forecast의
    # lane closure)가 걸린 segment는 과금 면제 — 폐쇄 유발 임계 초과는 어떤 후보도
    # 회피할 수 없는 외생 혼잡이라, 과금하면 랭킹만 왜곡한다(통제 가능한 초과만 벌점).
    exempt: set = set()
    if forecast:
        from src.models.demand import merge_freeway_lane_loss
        merged = merge_freeway_lane_loss(list(forecast))
        for link, seg_losses in merged.items():
            for seg_idx, loss in seg_losses.items():
                if float(loss) > 0.0:
                    exempt.add((str(link), int(seg_idx)))
    total = 0.0
    for s in states:
        excess = 0.0
        for link in net.freeway_links:
            dens = s.freeway_density.get(link, [])
            lanes = s.freeway_effective_lanes.get(link, [])
            for i, rho in enumerate(dens):
                if (str(link), i) in exempt:
                    continue
                lam = float(lanes[i]) if i < len(lanes) else 1.0
                excess += max(0.0, float(rho) - rho_crit) * seg_len * lam
        for link, cap in net.urban_link_storage_veh.items():
            avail = float(s.urban_link_storage.get(link, cap))
            occ = max(0.0, float(cap) - avail)
            excess += max(0.0, occ - frac * float(cap))
        total += excess * tc_h
    return w * total


def mfd_far_cost_to_go(cfg: "ExperimentConfig", state: "TrafficState") -> float:
    """far(MFD tail): near rollout 밖 잔여 accumulation의 배수 cost-to-go(§3.2, uniform urban+freeway).

    **urban reservoir**: N_P를 outflow(N)로 배수 — N_crit 넘으면 outflow↓ → V 급증(gridlock 회피).
      삼각형 drain TTS ≈ N²/(2G)·T_c_h. G=probe① calib(Geroliminis/Haddad MFD).
    **freeway reservoir**: 본선 2차(exit 병목 큐잉) + ramp 큐 merge-병목 대기(congestion-aware).
      ramp: q²/(2·merge_rate)·T_c_h(대기) + q·T_ramp_traverse(합류 후 통과),
      merge_rate=ramp_cap·receiving(ρ_merge) → 혼잡할수록 배수 느림(hidden-space).
    leader_mfd_far_enabled 미설정=0(비트동일). cfg·state만 읽는 순수 함수 —
    P-CENT(중앙 천장 측정기) 등 타 컨트롤러 채점에 재사용(2026-07-10 승격)."""
    if not getattr(cfg.mpc, "leader_mfd_far_enabled", False):
        return 0.0
    from src.models.metanet import (
        _ramp_merge_index, _clip, desired_speed_kmh, segment_flow_veh_h,
    )
    net = cfg.network
    tc_h = float(cfg.simulation.T_c_h)
    w = float(getattr(cfg.mpc, "leader_mfd_far_weight", 1.0))
    # far 항의 t_trav·배수율을 state(ρ·유효차선)에서 유도할지 — 기본 False=기존 상수식(비트동일).
    state_aware = bool(getattr(cfg.mpc, "leader_mfd_far_state_aware", False))
    # 실제-v far(2026-07-20): 속도원을 FD V(ρ) 대신 말단 state 실제 속도로(상세 state.py 주석).
    real_v = bool(getattr(cfg.mpc, "leader_mfd_far_real_speed", False))
    # ---- urban reservoir ----
    # boundary_in 큐도 accumulation에 포함: gating(유입 조임)은 차를 boundary 큐로 옮길 뿐
    # (여전히 TTT 발생)이라, in-network(N_P)만 세면 gating이 far상 공짜로 보여 leader가 과-gate.
    n_u = float(state.protected_accumulation_veh(net)) + float(state.boundary_in_queue_vehicles(net))
    n_crit = float(getattr(cfg.mpc, "leader_mfd_far_ncrit", 1700.0))
    g_free = float(getattr(cfg.mpc, "leader_mfd_far_g_free", 640.0))
    g_cong = float(getattr(cfg.mpc, "leader_mfd_far_g_cong", 500.0))
    g_u = g_free if n_u < n_crit else g_cong
    far = (n_u * n_u) * tc_h / (2.0 * max(g_u, 1.0))
    # ---- freeway reservoir: 본선 2차(exit 병목 큐잉, urban과 대칭) ----
    seg_len = float(net.freeway_segment_length_km)
    v_free = float(net.v_free)
    ramp_total = sum(max(0.0, float(state.ramp_queue.get(r, 0.0))) for r in net.ramps)
    n_main = max(0.0, float(state.total_freeway_vehicles(net)) - ramp_total)
    g_fw = float(getattr(cfg.mpc, "leader_mfd_far_g_fw", 300.0))
    if getattr(cfg.mpc, "leader_mfd_far_freeflow_offset", False):
        # 자유류 오프셋(2026-07-11, 8-seg 경부하 과잉억제 수선): 링크 주행시간 내 자연
        # 배출분은 큐가 아님 — N_eff = max(0, N_l − G·t_trav)만 2차 벌점, 흐르는 재고는
        # 선형(평균 잔여 주행 t_trav/2). t_trav ∝ 링크 길이라 식이 자기정규화(튜닝 상수 0).
        # 링크별 제곱합(각 링크는 자기 출구로 배수 — 합산 제곱의 교차항 과대 제거).
        n_seg_l = int(net.freeway_segments_per_link)
        t_trav = (n_seg_l * seg_len) / max(v_free, 1.0)  # [h]
        drainable = g_fw * (t_trav / max(tc_h, 1.0e-9))  # [veh]
        for link in net.freeway_links:
            dens = state.freeway_density.get(link, [])
            lanes = state.freeway_effective_lanes.get(link, [])

            def _lane_at(i: int) -> float:
                return max(float(lanes[i]) if i < len(lanes) else float(net.freeway_lanes), 1.0e-9)

            n_l = sum(
                max(0.0, float(dens[i])) * seg_len * _lane_at(i)
                for i in range(len(dens))
            )
            t_trav_l, g_fw_l = t_trav, g_fw
            if state_aware and dens:
                # STATE-AWARE far(2026-07-16): 기존 식은 t_trav를 v_free로, 배수율 g_fw를
                # 상수 300으로 박아둬 **차선 폐색·과포화를 전혀 못 본다**(사고 시 리더가
                # freeway 유입 비용을 구조적으로 과소평가). ρ와 유효차선이 state에 있고
                # METANET 기본도가 있으므로 둘 다 유도한다 — 새 튜닝 상수 없음.
                #   t_trav = Σ_i ℓ / V(ρ_i)        (자유류 대신 실제 밀도 속도)
                #   g_fw   = min_i cap(lanes_i)    (병목 세그먼트 용량; 폐색이면 자동 하락)
                spd_l = state.freeway_speed.get(link, []) if real_v else []
                t_trav_l = 0.0
                caps = []
                for i in range(len(dens)):
                    if real_v and i < len(spd_l):
                        # 실제-v: rollout 동역학이 계산한 말단 속도(캡드롭 반영) 그대로.
                        v_i = max(float(spd_l[i]), 1.0)
                    else:
                        v_i = desired_speed_kmh(max(0.0, float(dens[i])), v_free, net.rho_crit)
                    t_trav_l += seg_len / max(v_i, 1.0)
                    cap_i = segment_flow_veh_h(
                        net.rho_crit,
                        desired_speed_kmh(net.rho_crit, v_free, net.rho_crit),
                        _lane_at(i),
                    )
                    if real_v and i < len(spd_l) and float(dens[i]) > float(net.rho_crit):
                        # 혼잡 병목의 배수율은 용량이 아니라 실측 유량(ρ·v·λ) — 자유류 세그는
                        # 용량 유지(경부하서 실유량으로 나누면 배수시간 과대라는 역왜곡 방지).
                        cap_i = min(cap_i, max(float(dens[i]) * max(float(spd_l[i]), 1.0) * _lane_at(i), 1.0))
                    caps.append(cap_i)
                g_fw_l = min(caps) if caps else g_fw
            drainable_l = g_fw_l * (t_trav_l / max(tc_h, 1.0e-9))
            n_eff = max(0.0, n_l - drainable_l)
            far += (n_eff * n_eff) * tc_h / (2.0 * max(g_fw_l, 1.0)) + n_l * t_trav_l / 2.0
    else:
        far += (n_main * n_main) * tc_h / (2.0 * max(g_fw, 1.0))
    # ---- freeway reservoir: ramp 큐 merge-병목 대기 + 통과 ----
    for ramp in net.ramps:
        q = max(0.0, float(state.ramp_queue.get(ramp, 0.0)))
        if q <= 0.0:
            continue
        link = net.ramp_to_freeway.get(ramp)
        dens = state.freeway_density.get(link, [])
        if not dens:
            continue
        midx = _ramp_merge_index(cfg, ramp, len(dens))
        rho_merge = float(dens[midx])
        recv = _clip((net.rho_max - rho_merge) / max(net.rho_max - net.rho_crit, 1.0e-9), 0.0, 1.0)
        merge_interval = float(net.ramp_capacity_veh_h[ramp]) * recv * tc_h  # veh/interval
        if state_aware:
            # 합류 후 하류 통과도 자유류가 아니라 실제 밀도 속도로(위와 동일 취지).
            spd_r = state.freeway_speed.get(link, []) if real_v else []
            t_ramp_traverse = sum(
                seg_len / max(
                    (float(spd_r[i]) if real_v and i < len(spd_r)
                     else desired_speed_kmh(max(0.0, float(dens[i])), v_free, net.rho_crit)),
                    1.0,
                )
                for i in range(midx, len(dens))
            )
        else:
            t_ramp_traverse = (len(dens) - midx) * seg_len / max(v_free, 1.0)
        far += (q * q) * tc_h / (2.0 * max(merge_interval, 1.0e-6)) + q * t_ramp_traverse
    return w * far


@dataclass
class DecisionResult:
    control: ControlAction
    leader_objective: float
    nash: NashResult
    metadata: Dict[str, float]


@dataclass
class _LeaderCandidateEvaluation:
    index: int
    action: LeaderAction
    nash: NashResult
    objective: float
    objective_terms: Dict[str, float]
    metadata: Dict[str, float]
    rollout_used: bool
    stage: str = "coarse"


def _stackelberg_candidate_worker(payload: dict) -> _LeaderCandidateEvaluation:
    controller = StackelbergMPCController(payload["cfg"])
    return controller._evaluate_full_candidate(
        payload["index"],
        payload["action"],
        payload["state"],
        payload["forecast"],
        payload["previous"],
        payload.get("stage", "coarse"),
        payload.get("incumbent_obj", float("inf")),
    )


class StackelbergMPCController:
    """Spec-first Stackelberg MPC controller.

    This implementation is intentionally self-contained under `src/` and does
    not import any root-level historical controller modules. Leader actions are
    enumerated, follower responses are solved by deterministic projection and
    queue-balancing heuristics, and the default leader base uses the follower
    response objective rather than an additional full-system rollout.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.leader = Leader(cfg)
        self.nash_solver = self._make_follower_solver(cfg)
        self.previous_control: Optional[ControlAction] = None
        self.last_decision: Optional[DecisionResult] = None
        self._pfo_fallback_previous_control: Optional[ControlAction] = None
        self._leader_process_pool: Optional[ProcessPoolExecutor] = None
        self._leader_process_pool_workers: int = 0
        # 층3: 캘리브레이션 지문 대조(경고 전용, 수치 경로 불변).
        self._calib_fingerprint_mismatch_count: int = self._check_calibration_fingerprints()
        # 재캘리브레이션 트리거(2026-07-14): 지문 불일치 또는 β̂ 표류 감지 시 진단
        # leader_recalib_needed=1 + stderr 제안(런당 1회). 수치 경로 불변.
        self._recalib_needed: bool = self._calib_fingerprint_mismatch_count > 0
        self._recalib_stderr_emitted: bool = False

    def _check_calibration_fingerprints(self) -> int:
        """층3: 손튜닝 성분별 캘리브레이션 기하 지문 vs 현재 cfg 기하 대조.

        불일치 성분 수를 반환하고 성분별 한국어 경고를 stderr에 1회 출력한다.
        경고·카운트만 — 어떤 수치 경로도 바꾸지 않는다."""
        current = {
            str(k): int(v)
            for k, v in dict(
                getattr(self.cfg.network, "ramp_merge_segment_index", {}) or {}
            ).items()
        }
        mismatches = 0
        for component, fingerprint in CALIBRATION_FINGERPRINTS.items():
            expected = {
                str(k): int(v) for k, v in fingerprint.get("ramp_merge", {}).items()
            }
            if not expected or not current:
                continue
            if expected == current:
                continue
            mismatches += 1
            if component not in _calib_fingerprint_warned:
                _calib_fingerprint_warned.add(component)
                exp_set = sorted(set(expected.values()))
                cur_set = sorted(set(current.values()))
                print(
                    f"[경고] {component} 성분은 merge {{{','.join(str(v) for v in exp_set)}}} "
                    f"기하에서 캘리브레이션됨 - 현재 {{{','.join(str(v) for v in cur_set)}}}, 재검증 필요",
                    file=sys.stderr,
                    flush=True,
                )
        return mismatches

    def _maybe_emit_recalib_suggestion(self) -> None:
        """재캘리브레이션 제안 stderr — 런(컨트롤러 인스턴스)당 최초 1회만."""
        if self._recalib_needed and not self._recalib_stderr_emitted:
            self._recalib_stderr_emitted = True
            print(
                "[제안] 캘리브레이션 표류 감지 - component_canary.py 실행 권장",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        if self._leader_process_pool is not None:
            self._leader_process_pool.shutdown(wait=False, cancel_futures=True)
            self._leader_process_pool = None
            self._leader_process_pool_workers = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _make_follower_solver(self, cfg: ExperimentConfig):
        if cfg.mpc.follower_solver_mode == "distributed":
            return DistributedCoordinator(cfg)
        return NashSolver(cfg)

    @staticmethod
    def _finite_float(value: object) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if out != out or out in (float("inf"), -float("inf")):
            return None
        return out

    @staticmethod
    def _merged_nash_diagnostics(nash: NashResult) -> Dict[str, float]:
        diag = dict(getattr(nash, "diagnostics", {}) or {})
        diag.update(getattr(nash.control, "diagnostics", {}) or {})
        return diag

    def _follower_realized_net_inflow_veh(
        self,
        nash: NashResult,
    ) -> tuple[Optional[float], Optional[float], str]:
        """Spec 4.1 follower projection: 후보 solve 뒤 실현 가능한 N_P[veh]를 읽는다."""
        diag = self._merged_nash_diagnostics(nash)
        projected_sources = [
            ("leader_projected_N_P_star", "leader_projected"),
            ("wu_faithful_np_projected_target_veh", "wu_projected"),
            ("wu_faithful_np_projected_target", "wu_projected"),
            ("distributed_grid_leader_projected_net_inflow_veh", "distributed_projected"),
            ("allocation_projected_net_inflow_veh", "allocation_projected"),
        ]
        realized_sources = [
            ("leader_realized_N_P_star", "leader_realized"),
            ("wu_faithful_sum_nin", "wu_sum_nin"),
            ("distributed_grid_leader_projected_net_inflow_veh", "distributed_projected"),
            ("allocation_projected_net_inflow_veh", "allocation_projected"),
        ]
        projected: Optional[float] = None
        for key, _source in projected_sources:
            value = self._finite_float(diag.get(key))
            if value is not None:
                projected = value
                break
        for key, source in realized_sources:
            value = self._finite_float(diag.get(key))
            if value is not None:
                return value, projected if projected is not None else value, source
        if projected is not None:
            return projected, projected, "projected_only"
        return None, None, "none"

    def _close_nash_response_leader_action(
        self,
        action: LeaderAction,
        nash: NashResult,
        forecast: Optional[list[DemandStep]] = None,
        intent_action: Optional[LeaderAction] = None,
    ) -> tuple[LeaderAction, Dict[str, float]]:
        """Follower solve 후 leader 좌표를 raw intent가 아니라 실현/사영 N_P로 닫는다."""
        raw_intent = intent_action if intent_action is not None else action
        intent_np = float(raw_intent.N_P_star)
        intent_nuf = float(raw_intent.N_UF_star)
        solver_np = float(action.N_P_star)
        solver_nuf = float(action.N_UF_star)
        realized_np, projected_np, source = self._follower_realized_net_inflow_veh(nash)
        closed_np = solver_np if realized_np is None else float(realized_np)
        closed_projected_np = solver_np if projected_np is None else float(projected_np)
        realized_nuf = sum(float(v) for v in nash.control.ramp_metering.values())
        if self._finite_float(realized_nuf) is None:
            realized_nuf = solver_nuf
        source_names = [
            "leader_realized",
            "wu_sum_nin",
            "wu_projected",
            "distributed_projected",
            "allocation_projected",
            "projected_only",
            "none",
        ]
        changed_np = abs(closed_np - intent_np) > 1.0e-9
        horizon_steps = len(forecast or [])
        if horizon_steps <= 0:
            horizon_steps = int(self.cfg.mpc.horizon_steps)
        horizon_steps = max(1, min(horizon_steps, int(self.cfg.mpc.horizon_steps)))
        horizon_h = max(float(self.cfg.simulation.T_c_h) * horizon_steps, 1.0e-9)
        metadata: Dict[str, float] = {
            "leader_intent_N_P_star": intent_np,
            "leader_intent_N_UF_star": intent_nuf,
            "leader_solver_N_P_star": solver_np,
            "leader_solver_N_UF_star": solver_nuf,
            "leader_realized_N_P_star": closed_np,
            "leader_projected_N_P_star": closed_projected_np,
            "leader_realized_N_UF_star": float(realized_nuf),
            "leader_response_closure_applied": float(realized_np is not None),
            "leader_response_closure_changed_N_P": float(changed_np),
            "leader_response_N_P_star_realization_residual": float(intent_np - closed_np),
            "leader_response_N_P_star_projection_residual": float(intent_np - closed_projected_np),
            "leader_response_N_UF_star_realization_residual": float(intent_nuf - realized_nuf),
            "leader_candidate_intent_N_P_star": intent_np,
            "leader_candidate_intent_N_UF_star": intent_nuf,
            "leader_candidate_solver_N_P_star": solver_np,
            "leader_candidate_solver_N_UF_star": solver_nuf,
            "leader_candidate_realized_N_P_star": closed_np,
            "leader_candidate_projected_N_P_star": closed_projected_np,
            "leader_candidate_realized_N_UF_star": float(realized_nuf),
            "urban_net_inflow_original_target_veh": intent_np,
            "urban_net_inflow_target_veh": closed_projected_np,
            "urban_net_inflow_target_veh_h": float(closed_projected_np / horizon_h),
        }
        for name in source_names:
            metadata[f"leader_response_net_inflow_source_{name}"] = float(source == name)
        # Wu follower의 horizon rollout objective는 control.N_P_star를 읽으므로 closure 후 다시 평가한다.
        metadata["leader_response_closure_use_rollout_objective"] = float(source.startswith("wu_"))
        nash.control.N_P_star = closed_np
        nash.control.N_UF_star = float(realized_nuf)
        nash.control.diagnostics.update(metadata)
        nash.diagnostics.update(metadata)
        return LeaderAction(closed_np, float(realized_nuf)), metadata

    def _append_progress_event(
        self,
        *,
        event: str,
        stage: str,
        completed: int,
        total: int,
        evaluation: Optional[_LeaderCandidateEvaluation] = None,
        best_objective: Optional[float] = None,
    ) -> None:
        path_text = os.environ.get("NUMSIM_STACKELBERG_PROGRESS_FILE", "")
        if not path_text:
            return
        row: Dict[str, object] = {
            "event": event,
            "step": os.environ.get("NUMSIM_STACKELBERG_PROGRESS_STEP", ""),
            "stage": stage,
            "completed": int(completed),
            "total": int(total),
        }
        if best_objective is not None:
            row["best_objective_so_far"] = float(best_objective)
        if evaluation is not None:
            row.update({
                "candidate_index": int(evaluation.index),
                "N_P_star": float(evaluation.action.N_P_star),
                "N_UF_star": float(evaluation.action.N_UF_star),
                "objective": float(evaluation.objective),
                "follower_ttt": float(evaluation.objective_terms.get("leader_follower_ttt_base", 0.0)),
                "mfd_storage_penalty": float(evaluation.objective_terms.get("leader_mfd_storage_penalty", 0.0)),
                "mfd_storage_excess_veh": float(
                    evaluation.objective_terms.get("leader_mfd_storage_excess_veh", 0.0)
                ),
            })
        try:
            path = Path(path_text)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            csv_path = path.with_suffix(".csv")
            fields = [
                "step",
                "event",
                "stage",
                "completed",
                "total",
                "candidate_index",
                "N_P_star",
                "N_UF_star",
                "objective",
                "best_objective_so_far",
                "follower_ttt",
                "mfd_storage_penalty",
                "mfd_storage_excess_veh",
            ]
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            return

    def decide(
        self,
        state: TrafficState,
        demand_forecast: Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
        config: Optional[ExperimentConfig] = None,
    ) -> ControlAction:
        result = self.decide_with_info(state, demand_forecast, previous_control, config)
        return result.control

    def decide_with_info(
        self,
        state: TrafficState,
        demand_forecast: Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
        config: Optional[ExperimentConfig] = None,
    ) -> DecisionResult:
        if config is not None and config is not self.cfg:
            self.close()
            self.cfg = config
            self.leader = Leader(config)
            self.nash_solver = self._make_follower_solver(config)
            self._pfo_fallback_previous_control = None
        forecast = list(demand_forecast)
        if not forecast:
            raise ValueError("StackelbergMPCController requires a non-empty demand forecast.")
        previous = (
            previous_control.copy()
            if previous_control is not None
            else self.previous_control.copy()
            if self.previous_control is not None
            else ControlAction.fixed(self.cfg)
        )
        previous = self._normalize_previous_leader_reference(previous)
        global_refresh = self._leader_global_refresh_active(state)
        search_mode = str(self.cfg.mpc.leader_search_mode)
        fallback_start_index = 1000000
        fallback_enabled = bool(self.cfg.mpc.stackelberg_enable_fallback)
        pfo_incumbent_enabled = bool(self._pfo_incumbent_fallback_enabled())
        fallback_evaluations = (
            self._evaluate_fallback_candidates(
                state,
                forecast,
                previous,
                start_index=fallback_start_index,
            )
            if fallback_enabled or pfo_incumbent_enabled
            else []
        )
        fallback_incumbent_obj = min(
            (item.objective for item in fallback_evaluations),
            default=float("inf"),
        )
        if search_mode == "continuous":
            (
                full_evaluations,
                base_metadata,
                proxy_metadata,
                refined_proxy_metadata,
            ) = self._continuous_leader_search(
                state,
                forecast,
                previous,
                global_refresh,
                fallback_incumbent_obj,
            )
        else:
            (
                full_evaluations,
                base_metadata,
                proxy_metadata,
                refined_proxy_metadata,
            ) = self._grid_leader_search(
                state,
                forecast,
                previous,
                global_refresh,
                fallback_incumbent_obj,
            )
        base_metadata.update({
            "leader_search_mode_grid": float(search_mode == "grid"),
            "leader_search_mode_continuous": float(search_mode == "continuous"),
            "leader_fallback_enabled": float(fallback_enabled),
            "leader_pfo_incumbent_enabled": float(pfo_incumbent_enabled),
            "leader_fallback_incumbent_seed_active": float(fallback_incumbent_obj < float("inf")),
            "leader_fallback_incumbent_objective": float(
                fallback_incumbent_obj if fallback_incumbent_obj < float("inf") else 0.0
            ),
        })
        all_evaluations = full_evaluations + fallback_evaluations
        evaluations: list[dict[str, object]] = [
            {
                "index": float(item.index),
                "N_P_star": float(item.action.N_P_star),
                "N_UF_star": float(item.action.N_UF_star),
                "intent_N_P_star": float(
                    item.metadata.get("leader_candidate_intent_N_P_star", item.action.N_P_star)
                ),
                "intent_N_UF_star": float(
                    item.metadata.get("leader_candidate_intent_N_UF_star", item.action.N_UF_star)
                ),
                "realized_N_P_star": float(
                    item.metadata.get("leader_candidate_realized_N_P_star", item.action.N_P_star)
                ),
                "projected_N_P_star": float(
                    item.metadata.get("leader_candidate_projected_N_P_star", item.action.N_P_star)
                ),
                "realized_N_UF_star": float(
                    item.metadata.get("leader_candidate_realized_N_UF_star", item.action.N_UF_star)
                ),
                "objective": float(item.objective),
                "base": float(item.objective_terms["leader_objective_base"]),
                "follower_ttt": float(item.objective_terms["leader_follower_ttt_base"]),
                "stage": item.stage,
            }
            for item in all_evaluations
        ]
        # price 곡률 진단(2026-07-22): LEADER_CAND_LOG 지정 시 스텝별 전 후보 (N_UF,objective)를
        # 남겨 목적함수 곡면 곡률을 사후 측정한다(추가 계산 없음, env-gated).
        _cand_log = os.environ.get("LEADER_CAND_LOG")
        if _cand_log:
            _step = os.environ.get("NUMSIM_STACKELBERG_PROGRESS_STEP", "")
            _p = Path(_cand_log).with_suffix(".allcand.csv")
            try:
                _p.parent.mkdir(parents=True, exist_ok=True)
                _new = not _p.exists()
                with _p.open("a", newline="", encoding="utf-8") as _fh:
                    _w = csv.writer(_fh)
                    if _new:
                        _w.writerow(["step", "index", "N_P_star", "N_UF_star",
                                     "objective", "base", "follower_ttt", "stage"])
                    for _it in all_evaluations:
                        _w.writerow([
                            _step, _it.index, _it.action.N_P_star, _it.action.N_UF_star,
                            _it.objective,
                            _it.objective_terms.get("leader_objective_base", ""),
                            _it.objective_terms.get("leader_follower_ttt_base", ""),
                            _it.stage,
                        ])
            except OSError:
                pass
        best_eval, fallback_metadata = self._select_with_fallback_guard(
            full_evaluations,
            fallback_evaluations,
        )
        # price 곡률 진단 능동 스윕(2026-07-22): LEADER_CURV_SWEEP 지정 시, 선택된 N_P 고정하고
        # N_UF를 넓게 11점 스윕하며 J를 찍는다(클리프 가로지르는 진짜 곡률용). search 후 별도
        # 평가라 결정·궤적 불변. env-gated.
        _sweep = os.environ.get("LEADER_CURV_SWEEP")
        if _sweep:
            _step_s = os.environ.get("NUMSIM_STACKELBERG_PROGRESS_STEP", "")
            _npv = float(best_eval.action.N_P_star)
            _nuf0 = float(best_eval.action.N_UF_star)
            _span = max(abs(_nuf0) * 0.5, 300.0)
            _sp = Path(_sweep).with_suffix(".sweep.csv")
            try:
                _sp.parent.mkdir(parents=True, exist_ok=True)
                _newp = not _sp.exists()
                with _sp.open("a", newline="", encoding="utf-8") as _sfh:
                    _sw = csv.writer(_sfh)
                    if _newp:
                        _sw.writerow(["step", "req_N_UF", "N_P", "objective", "realized_N_UF"])
                    for _k in range(11):
                        _nuf = _nuf0 - _span + 2.0 * _span * _k / 10.0
                        _ev = self._evaluate_full_candidate(
                            900000 + _k, LeaderAction(_npv, _nuf), state, forecast, previous,
                        )
                        _sw.writerow([_step_s, _nuf, _npv, _ev.objective, _ev.action.N_UF_star])
            except OSError:
                pass
        metadata = dict(base_metadata)
        metadata.update(proxy_metadata)
        metadata.update(refined_proxy_metadata)
        metadata.update(fallback_metadata)
        metadata.update({
            "N_P_crit": float(self.cfg.leader.N_P_crit_veh),
            "nash_iterations": float(best_eval.nash.iterations),
            "nash_converged": 1.0 if best_eval.nash.converged else 0.0,
            "nash_residual_control": best_eval.nash.residual_control,
            "nash_residual_objective": best_eval.nash.residual_objective,
            "leader_rollout_prediction_used": 1.0 if best_eval.rollout_used else 0.0,
            "leader_follower_response_objective_used": float(
                self.cfg.leader.objective_mode == "follower_ttt"
            ),
            "leader_rollout_objective_base_used": float(
                self.cfg.leader.objective_mode == "state_accumulation"
            ),
            "leader_response_proxy_state_count": float(best_eval.metadata.get("leader_response_proxy_state_count", 0.0)),
            "leader_candidate_full_evaluated_count": float(len(full_evaluations)),
            "leader_fallback_evaluated_count": float(len(fallback_evaluations)),
            # 층3: 캘리브레이션 지문 불일치 성분 수 + 재캘리브레이션 트리거(경고·진단 전용).
            "calib_fingerprint_mismatch_count": float(
                getattr(self, "_calib_fingerprint_mismatch_count", 0)
            ),
            "leader_recalib_needed": float(bool(getattr(self, "_recalib_needed", False))),
        })
        self._maybe_emit_recalib_suggestion()
        metadata.update(best_eval.metadata)
        metadata.update(best_eval.objective_terms)
        best = DecisionResult(best_eval.nash.control, best_eval.objective, best_eval.nash, metadata)
        assert best is not None
        best.metadata.update(self._candidate_evaluation_metadata(evaluations, best.control, best_eval))
        best.control.diagnostics.update(best.metadata)
        best.control.diagnostics["leader_objective"] = best.leader_objective
        self._apply_output_closure(best, state, forecast)
        self.previous_control = best.control.copy()
        self.last_decision = best
        return best

    def _realized_net_inflow_veh(
        self,
        control: ControlAction,
        state: TrafficState,
        forecast: list[DemandStep],
    ) -> Optional[float]:
        """committed control이 실제로 실현하는 보호영역 net-inflow[veh]를 계산한다.

        production decide 경로의 control.diagnostics에는 이 값이 전파되지 않으므로(direct
        feasible-set 경로에서만 생성), injection 진단과 동일하게 coordinator의 feasible-set
        진단을 committed control에 직접 적용해 service(inflow−outflow)를 구한다.
        distributed follower가 아니거나 키 부재 시 None(=닫기 보류)."""
        # WuFaithful follower는 leader target을 자체 feasible Σnin 범위로 투영한다.
        # output closure는 raw intent가 아니라 실제 follower response[veh/horizon]를
        # 다음 step seed와 로그의 committed target으로 사용해야 한다.
        diag = getattr(control, "diagnostics", {}) or {}
        for key in (
            "leader_response_realized_N_P_star",
            "leader_realized_N_P_star",
            "wu_faithful_realized_N_P_star",
            "wu_faithful_np_realized_sum_nin_veh",
            "wu_faithful_sum_nin",
            "wu_faithful_np_projected_target_veh",
            "wu_faithful_np_projected_target",
        ):
            if key not in diag:
                continue
            value = float(diag.get(key, 0.0))
            if value == value and value not in (float("inf"), -float("inf")):
                return value
        if not isinstance(self.nash_solver, DistributedCoordinator):
            return None
        probe_leader = LeaderAction(float(control.N_P_star), float(control.N_UF_star))
        diag = self.nash_solver._leader_direct_feasible_set_diagnostics(
            state.copy(), control.copy(), list(forecast), probe_leader
        )
        if "distributed_grid_leader_projected_net_inflow_veh" not in diag:
            return None
        return float(diag["distributed_grid_leader_projected_net_inflow_veh"])

    def _apply_output_closure(
        self,
        best: DecisionResult,
        state: TrafficState,
        forecast: list[DemandStep],
    ) -> None:
        """층1 출력 폐쇄: leader가 commit하는 N_P*/N_UF*를 follower가 실제 실현한 값으로
        덮어쓴다(self-consistent Stackelberg, 2026-06-25).

        leader 후보의 raw intent는 도달 불가일 수 있고, 보고값이 실현값과 다르면 정직하지
        않다. follower가 실현한 net-inflow/metering을 committed action으로 닫는다. raw intent는
        `leader_intent_*` 진단키로, 기존 `leader_selected_*`(=intent)도 그대로 보존해 추적성을 유지한다.
        """
        control = best.control
        diag = getattr(control, "diagnostics", None) or {}
        intent_np = self._finite_float(diag.get("leader_intent_N_P_star"))
        intent_nuf = self._finite_float(diag.get("leader_intent_N_UF_star"))
        if intent_np is None:
            intent_np = float(control.N_P_star)
        if intent_nuf is None:
            intent_nuf = float(control.N_UF_star)
        realized_np = self._finite_float(diag.get("leader_response_realized_N_P_star"))
        if realized_np is None:
            realized_np = self._finite_float(diag.get("leader_realized_N_P_star"))
        projected_np = self._finite_float(diag.get("leader_response_projected_N_P_star"))
        if projected_np is None:
            projected_np = self._finite_float(diag.get("leader_projected_N_P_star"))
        # realized net-inflow[veh]: committed control의 service로 직접 계산(부재 시 intent 유지).
        realized_net = self._realized_net_inflow_veh(control, state, forecast) if realized_np is None else realized_np
        realized_np = intent_np if realized_net is None else realized_net
        projected_np = realized_np if projected_np is None else projected_np
        # realized metering[veh/h]: follower가 적용한 ramp metering 합(이미 공급/수용 상한에 사영됨).
        realized_nuf = sum(float(v) for v in control.ramp_metering.values())
        control.diagnostics["leader_intent_N_P_star"] = intent_np
        control.diagnostics["leader_intent_N_UF_star"] = intent_nuf
        control.diagnostics["leader_realized_N_P_star"] = realized_np
        control.diagnostics["leader_projected_N_P_star"] = projected_np
        control.diagnostics["leader_realized_N_UF_star"] = realized_nuf
        control.diagnostics["leader_output_closure_applied"] = 1.0 if realized_net is not None else 0.0
        horizon_steps = max(1, min(len(forecast), int(self.cfg.mpc.horizon_steps)))
        horizon_h = max(float(self.cfg.simulation.T_c_h) * horizon_steps, 1.0e-9)
        control.diagnostics["urban_net_inflow_original_target_veh"] = intent_np
        control.diagnostics["urban_net_inflow_target_veh"] = projected_np
        control.diagnostics["urban_net_inflow_target_veh_h"] = float(projected_np / horizon_h)
        control.N_P_star = realized_np
        control.N_UF_star = realized_nuf

    def _normalize_previous_leader_reference(self, previous: ControlAction) -> ControlAction:
        out = previous.copy()
        # 층1 출력폐쇄가 적용된 previous는 commit값이 realized다. 다음 결정의 seeding은 raw
        # intent를 써서 탐색 동역학을 폐쇄 이전과 동일하게(=TTT 불변) 유지한다. 보고/플롯은
        # realized(commit), seed는 intent로 분리한다.
        diag = getattr(out, "diagnostics", None) or {}
        if float(diag.get("leader_output_closure_applied", 0.0)) >= 0.5:
            out.N_P_star = float(diag.get("leader_realized_N_P_star", out.N_P_star))
            out.N_UF_star = float(diag.get("leader_realized_N_UF_star", out.N_UF_star))
        if out.N_UF_star > 1.0e-9:
            return out
        net = self.cfg.network
        release = sum(float(out.ramp_metering.get(ramp, 0.0)) for ramp in net.ramps)
        if release >= 0.95 * float(net.total_ramp_capacity):
            out.N_UF_star = float(net.total_ramp_capacity)
        return out

    def _leader_global_refresh_active(self, state: TrafficState) -> bool:
        interval = max(float(self.cfg.simulation.control_interval), 1.0e-9)
        step_index = int(round(float(state.time_sec) / interval))
        refresh_steps = max(1, int(round(float(self.cfg.mpc.leader_global_refresh_sec) / interval)))
        return step_index == 0 or step_index % refresh_steps == 0

    def _grid_leader_search(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
        fallback_incumbent_obj: float,
    ) -> tuple[list[_LeaderCandidateEvaluation], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if global_refresh:
            coarse_candidates = self.leader.candidates(state, previous, forecast[0], forecast=forecast)
            coarse_stage = "coarse_global"
        else:
            previous_leader = LeaderAction(previous.N_P_star, previous.N_UF_star)
            coarse_candidates = self.leader.refined_candidates(
                state,
                previous_leader,
                previous,
                forecast[0],
                forecast=forecast,
                count=self.cfg.mpc.leader_candidate_count,
            )
            coarse_stage = "coarse_local"
        selected_indices, proxy_metadata = self._prefilter_leader_candidates(
            coarse_candidates,
            state,
            forecast,
            previous,
            global_scope=global_refresh,
        )
        coarse_evaluations = self._evaluate_candidate_set(
            coarse_candidates,
            selected_indices,
            state,
            forecast,
            previous,
            stage=coarse_stage,
            incumbent_obj=fallback_incumbent_obj,
        )
        coarse_best = min(coarse_evaluations, key=lambda item: item.objective)
        # OPT1: local(비-global) 스텝은 coarse-local 자체가 previous 주변 trust-region이라
        # refined 재정련이 중복(nash 5s×~5회) — 생략해 스텝 비용 절반. global 스텝은 유지.
        skip_refine = (
            not global_refresh
            and bool(getattr(self.cfg.mpc, "leader_skip_local_refinement", False))
        )
        refined_candidates = [] if skip_refine else self._unique_leader_actions(
            self.leader.refined_candidates(
                state,
                coarse_best.action,
                previous,
                forecast[0],
                forecast=forecast,
            )
        )
        refined_indices: list[int] = []
        refined_proxy_metadata: Dict[str, float] = {}
        if refined_candidates:
            refined_indices, raw_refined_proxy_metadata = self._prefilter_leader_candidates(
                refined_candidates,
                state,
                forecast,
                previous,
                global_scope=False,
            )
            refined_proxy_metadata = self._stage_prefilter_metadata(
                raw_refined_proxy_metadata,
                "refined",
            )
        refined_evaluations = self._evaluate_candidate_set(
            refined_candidates,
            refined_indices,
            state,
            forecast,
            previous,
            stage="refined",
            index_offset=len(coarse_candidates),
            incumbent_obj=min(coarse_best.objective, fallback_incumbent_obj),
        ) if refined_candidates else []
        full_evaluations = coarse_evaluations + refined_evaluations
        base_metadata = leader_metadata(coarse_candidates + refined_candidates)
        base_metadata.update(self.leader.candidate_bound_metadata(state, previous, forecast[0], forecast=forecast))
        base_metadata.update(self._forecast_demand_metadata(forecast))
        base_metadata.update(self._leader_search_common_metadata(
            full_evaluations,
            global_refresh,
            coarse_stage,
            len(coarse_candidates),
            len(coarse_evaluations),
            len(refined_candidates),
            len(refined_evaluations),
            bool(refined_candidates),
        ))
        return full_evaluations, base_metadata, proxy_metadata, refined_proxy_metadata

    def _leader_search_common_metadata(
        self,
        full_evaluations: list[_LeaderCandidateEvaluation],
        global_refresh: bool,
        coarse_stage: str,
        coarse_count: int,
        coarse_evaluated_count: int,
        refined_count: int,
        refined_evaluated_count: int,
        refinement_active: bool,
    ) -> Dict[str, float]:
        def candidate_meta_any(key: str) -> float:
            return float(max((item.metadata.get(key, 0.0) for item in full_evaluations), default=0.0))

        return {
            "leader_candidate_coarse_count": float(coarse_count),
            "leader_candidate_coarse_evaluated_count": float(coarse_evaluated_count),
            "leader_candidate_refined_count": float(refined_count),
            "leader_candidate_refined_evaluated_count": float(refined_evaluated_count),
            "leader_candidate_refinement_active": float(refinement_active),
            "leader_candidate_global_refresh": float(global_refresh),
            "leader_candidate_coarse_global": float(coarse_stage in {"coarse_global", "continuous_global"}),
            "leader_candidate_coarse_local": float(coarse_stage in {"coarse_local", "continuous_local"}),
            "leader_candidate_parallel_backend_serial": candidate_meta_any(
                "leader_candidate_parallel_backend_serial"
            ),
            "leader_candidate_parallel_backend_thread": candidate_meta_any(
                "leader_candidate_parallel_backend_thread"
            ),
            "leader_candidate_parallel_backend_process": candidate_meta_any(
                "leader_candidate_parallel_backend_process"
            ),
            "leader_candidate_process_pool_reuse_enabled": candidate_meta_any(
                "leader_candidate_process_pool_reuse_enabled"
            ),
            "leader_candidate_process_pool_reused_existing": candidate_meta_any(
                "leader_candidate_process_pool_reused_existing"
            ),
            "leader_candidate_incumbent_any_active": float(any(
                item.metadata.get("leader_candidate_incumbent_active", 0.0) > 0.5
                for item in full_evaluations
            )),
            "leader_candidate_follower_early_terminated_candidates_total": float(sum(
                item.metadata.get("leader_candidate_follower_early_terminated_candidates", 0.0)
                for item in full_evaluations
            )),
        }

    def _continuous_leader_search(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        global_refresh: bool,
        fallback_incumbent_obj: float,
    ) -> tuple[list[_LeaderCandidateEvaluation], Dict[str, float], Dict[str, float], Dict[str, float]]:
        bounds = self.leader._candidate_bounds(state, previous, forecast[0], forecast)
        np_lower, np_upper = float(bounds.np_lower), float(bounds.np_upper)
        nuf_lower, nuf_upper = float(bounds.nuf_lower), float(bounds.nuf_upper)
        if not global_refresh:
            np_lower = max(np_lower, float(previous.N_P_star) - float(self.cfg.mpc.leader_local_np_radius_veh))
            np_upper = min(np_upper, float(previous.N_P_star) + float(self.cfg.mpc.leader_local_np_radius_veh))
            nuf_lower = max(nuf_lower, float(previous.N_UF_star) - float(self.cfg.mpc.leader_local_nuf_radius_veh_h))
            nuf_upper = min(nuf_upper, float(previous.N_UF_star) + float(self.cfg.mpc.leader_local_nuf_radius_veh_h))
        if np_lower > np_upper:
            np_lower = np_upper = float(previous.N_P_star)
        if nuf_lower > nuf_upper:
            nuf_lower = nuf_upper = float(previous.N_UF_star)

        def clipped(np_value: float, nuf_value: float) -> LeaderAction:
            return LeaderAction(
                float(min(max(float(np_value), np_lower), np_upper)),
                float(min(max(float(nuf_value), nuf_lower), nuf_upper)),
            )

        # full 탐색은 global refresh 스텝에서만, local 스텝은 축소 예산으로 previous 인근만 본다.
        if global_refresh:
            eff_max_evals = int(self.cfg.mpc.leader_continuous_max_evals)
            eff_seed_count = int(self.cfg.mpc.leader_continuous_seed_count)
            eff_prefilter_samples = int(self.cfg.mpc.leader_continuous_prefilter_samples)
            eff_prefilter_top_k = int(self.cfg.mpc.leader_continuous_prefilter_top_k)
        else:
            eff_max_evals = int(self.cfg.mpc.leader_continuous_local_max_evals)
            eff_seed_count = int(self.cfg.mpc.leader_continuous_local_seed_count)
            eff_prefilter_samples = int(self.cfg.mpc.leader_continuous_local_prefilter_samples)
            eff_prefilter_top_k = int(self.cfg.mpc.leader_continuous_local_prefilter_top_k)
        max_evals = max(1, eff_max_evals)
        raw_seed_actions = self._continuous_seed_actions(
            previous,
            bounds,
            np_lower,
            np_upper,
            nuf_lower,
            nuf_upper,
            clipped,
        )
        # Continuous search도 grid prefilter와 같은 구조를 쓴다. 먼저 넓은
        # deterministic low-discrepancy sample을 cheap proxy로 정렬하고, full
        # follower evaluation은 top-K seed에만 쓴다.
        seed_actions, prefilter_metadata = self._continuous_prefilter_actions(
            raw_seed_actions,
            state,
            forecast,
            previous,
            np_lower,
            np_upper,
            nuf_lower,
            nuf_upper,
            prefilter_samples=eff_prefilter_samples,
            prefilter_top_k=eff_prefilter_top_k,
        )
        seed_actions = seed_actions[: max(1, eff_seed_count)]
        coarse_stage = "continuous_global" if global_refresh else "continuous_local"
        evaluated, incumbent_obj = self._evaluate_continuous_action_set(
            seed_actions[:max_evals],
            state,
            forecast,
            previous,
            stage=coarse_stage,
            index_start=0,
            incumbent_obj=fallback_incumbent_obj,
        )
        if not evaluated:
            raise ValueError("Continuous leader search produced no evaluated candidates.")
        seen = {
            (round(float(item.action.N_P_star), 6), round(float(item.action.N_UF_star), 6))
            for item in evaluated
        }
        best = min(evaluated, key=lambda item: item.objective)
        np_span = max(np_upper - np_lower, 1.0e-9)
        nuf_span = max(nuf_upper - nuf_lower, 1.0e-9)
        np_step = max(
            float(self.cfg.mpc.leader_continuous_min_np_step_veh),
            np_span * float(self.cfg.mpc.leader_continuous_initial_step_fraction),
        )
        nuf_step = max(
            float(self.cfg.mpc.leader_continuous_min_nuf_step_veh_h),
            nuf_span * float(self.cfg.mpc.leader_continuous_initial_step_fraction),
        )
        local_evaluations: list[_LeaderCandidateEvaluation] = []
        iterations_completed = 0
        next_index = len(evaluated)
        shrink = float(self.cfg.mpc.leader_continuous_shrink_factor)
        min_np_step = float(self.cfg.mpc.leader_continuous_min_np_step_veh)
        min_nuf_step = float(self.cfg.mpc.leader_continuous_min_nuf_step_veh_h)
        base_directions = [
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        ]
        for iteration in range(max(0, int(self.cfg.mpc.leader_continuous_local_iterations))):
            remaining = max_evals - len(evaluated) - len(local_evaluations)
            if remaining <= 0:
                break
            candidates: list[LeaderAction] = []
            directions = self._continuous_ranked_directions(
                best.action,
                state,
                forecast,
                previous,
                np_step,
                nuf_step,
                clipped,
                base_directions,
            )
            for dx, dy in directions:
                action = clipped(
                    best.action.N_P_star + dx * np_step,
                    best.action.N_UF_star + dy * nuf_step,
                )
                key = (round(float(action.N_P_star), 6), round(float(action.N_UF_star), 6))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(action)
                if len(candidates) >= remaining:
                    break
            if not candidates:
                np_step *= shrink
                nuf_step *= shrink
                if np_step < min_np_step and nuf_step < min_nuf_step:
                    break
                continue
            stage = f"continuous_refined_{iteration}"
            new_evaluations, incumbent_obj = self._evaluate_continuous_action_set(
                candidates,
                state,
                forecast,
                previous,
                stage=stage,
                index_start=next_index,
                incumbent_obj=incumbent_obj,
            )
            next_index += len(new_evaluations)
            local_evaluations.extend(new_evaluations)
            if new_evaluations:
                new_best = min(new_evaluations + [best], key=lambda item: item.objective)
                if new_best.objective < best.objective - 1.0e-12:
                    best = new_best
                else:
                    np_step *= shrink
                    nuf_step *= shrink
            iterations_completed += 1
            if np_step < min_np_step and nuf_step < min_nuf_step:
                break

        full_evaluations = evaluated + local_evaluations
        actions = [item.action for item in full_evaluations]
        base_metadata = leader_metadata(actions)
        base_metadata.update(self.leader.candidate_bound_metadata(state, previous, forecast[0], forecast=forecast))
        base_metadata.update(self._forecast_demand_metadata(forecast))
        base_metadata.update(self._leader_search_common_metadata(
            full_evaluations,
            global_refresh,
            coarse_stage,
            len(seed_actions),
            len(evaluated),
            len(local_evaluations),
            len(local_evaluations),
            bool(local_evaluations),
        ))
        base_metadata.update({
            "leader_continuous_search_bound_np_lower": float(np_lower),
            "leader_continuous_search_bound_np_upper": float(np_upper),
            "leader_continuous_search_bound_nuf_lower": float(nuf_lower),
            "leader_continuous_search_bound_nuf_upper": float(nuf_upper),
            "leader_continuous_max_evals": float(max_evals),
            "leader_continuous_seed_count": float(len(seed_actions)),
            "leader_continuous_iterations_completed": float(iterations_completed),
            "leader_continuous_final_np_step_veh": float(np_step),
            "leader_continuous_final_nuf_step_veh_h": float(nuf_step),
        })
        base_metadata.update(prefilter_metadata)
        return full_evaluations, base_metadata, {}, {}

    def _continuous_seed_actions(
        self,
        previous: ControlAction,
        bounds,
        np_lower: float,
        np_upper: float,
        nuf_lower: float,
        nuf_upper: float,
        clipped,
    ) -> list[LeaderAction]:
        np_mid = 0.5 * (np_lower + np_upper)
        nuf_mid = 0.5 * (nuf_lower + nuf_upper)
        heuristic_nuf = float(min(max(float(bounds.heuristic_nuf), nuf_lower), nuf_upper))
        actions = [
            clipped(previous.N_P_star, previous.N_UF_star),
            clipped(np_mid, nuf_mid),
            clipped(0.0, heuristic_nuf),
            clipped(np_lower, nuf_lower),
            clipped(np_lower, nuf_upper),
            clipped(np_upper, nuf_lower),
            clipped(np_upper, nuf_upper),
            clipped(np_lower, nuf_mid),
            clipped(np_upper, nuf_mid),
            clipped(np_mid, nuf_lower),
            clipped(np_mid, nuf_upper),
        ]
        np_anchors = sorted(self.leader._np_anchor_values(bounds, previous))
        nuf_anchors = sorted(self.leader._nuf_anchor_values(bounds, previous))
        for np_value in np_anchors:
            actions.append(clipped(np_value, heuristic_nuf))
        for nuf_value in nuf_anchors:
            actions.append(clipped(0.0, nuf_value))
        for np_value in (np_anchors[0], np_anchors[-1]) if np_anchors else ():
            for nuf_value in (nuf_anchors[0], nuf_anchors[-1]) if nuf_anchors else ():
                actions.append(clipped(np_value, nuf_value))
        return self._unique_leader_actions(actions)

    def _continuous_prefilter_actions(
        self,
        seed_actions: list[LeaderAction],
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        np_lower: float,
        np_upper: float,
        nuf_lower: float,
        nuf_upper: float,
        prefilter_samples: Optional[int] = None,
        prefilter_top_k: Optional[int] = None,
    ) -> tuple[list[LeaderAction], Dict[str, float]]:
        if prefilter_samples is None:
            prefilter_samples = int(self.cfg.mpc.leader_continuous_prefilter_samples)
        if prefilter_top_k is None:
            prefilter_top_k = int(self.cfg.mpc.leader_continuous_prefilter_top_k)
        samples = self._unique_leader_actions(
            seed_actions
            + self._continuous_low_discrepancy_samples(
                int(prefilter_samples),
                np_lower,
                np_upper,
                nuf_lower,
                nuf_upper,
            )
        )
        rows: list[Dict[str, float]] = []
        filtered = 0
        spillback_filtered = 0
        for idx, action in enumerate(samples):
            if not self._continuous_candidate_bounds_ok(action, np_lower, np_upper, nuf_lower, nuf_upper):
                filtered += 1
                continue
            row = self._proxy_score_candidate(idx, action, state, forecast, previous)
            if bool(self.cfg.mpc.leader_continuous_hard_precheck):
                tolerance = float(self.cfg.mpc.leader_continuous_precheck_spillback_tolerance_veh)
                if row["spillback_violation"] > tolerance:
                    filtered += 1
                    spillback_filtered += 1
                    continue
            rows.append(row)
        if not rows:
            # Hard pre-check가 지나치게 강하면 기존 seed를 보존한다. feasibility
            # 진단은 남기되 controller가 후보 0개로 죽지 않게 한다.
            rows = [
                self._proxy_score_candidate(idx, action, state, forecast, previous)
                for idx, action in enumerate(self._unique_leader_actions(seed_actions))
            ]
        rows = sorted(rows, key=lambda row: (row["objective"], row["spillback_violation"], row["index"]))
        top_k = max(1, int(prefilter_top_k))
        selected_actions = [
            LeaderAction(float(row["N_P_star"]), float(row["N_UF_star"]))
            for row in rows[:top_k]
        ]
        # 기본/직전 seed가 cheap proxy에서 살짝 밀려도 한 개는 남겨 guard 역할을 하게 한다.
        if seed_actions:
            seed0, _ = self._project_action_to_follower_feasible_np(
                seed_actions[0], state, forecast, previous
            )
            selected_actions.insert(0, seed0)
        selected_actions = self._unique_leader_actions(selected_actions)
        objectives = [float(row["objective"]) for row in rows] or [0.0]
        best = rows[0] if rows else {"index": 0.0, "objective": 0.0}
        second = rows[1] if len(rows) > 1 else best
        return selected_actions, {
            "leader_continuous_prefilter_active": 1.0,
            "leader_continuous_prefilter_samples": float(len(samples)),
            "leader_continuous_prefilter_proxy_evaluated_count": float(len(rows)),
            "leader_continuous_prefilter_selected_count": float(len(selected_actions)),
            "leader_continuous_prefilter_top_k": float(top_k),
            "leader_continuous_precheck_filtered_count": float(filtered),
            "leader_continuous_precheck_spillback_filtered_count": float(spillback_filtered),
            "leader_continuous_proxy_best_index": float(best["index"]),
            "leader_continuous_proxy_second_index": float(second["index"]),
            "leader_continuous_proxy_best_objective": float(best["objective"]),
            "leader_continuous_proxy_second_objective": float(second["objective"]),
            "leader_continuous_proxy_objective_gap": float(second["objective"] - best["objective"]),
            "leader_continuous_proxy_objective_spread": float(max(objectives) - min(objectives)),
        }

    def _continuous_low_discrepancy_samples(
        self,
        count: int,
        np_lower: float,
        np_upper: float,
        nuf_lower: float,
        nuf_upper: float,
    ) -> list[LeaderAction]:
        count = max(0, int(count))
        np_span = max(np_upper - np_lower, 0.0)
        nuf_span = max(nuf_upper - nuf_lower, 0.0)

        def vdc(index: int, base: int) -> float:
            value = 0.0
            denom = 1.0
            n = max(0, int(index))
            while n:
                n, rem = divmod(n, base)
                denom *= base
                value += rem / denom
            return value

        actions: list[LeaderAction] = []
        for i in range(1, count + 1):
            np_frac = vdc(i, 2)
            nuf_frac = vdc(i, 3)
            actions.append(LeaderAction(
                float(np_lower + np_frac * np_span),
                float(nuf_lower + nuf_frac * nuf_span),
            ))
        return actions

    def _continuous_candidate_bounds_ok(
        self,
        action: LeaderAction,
        np_lower: float,
        np_upper: float,
        nuf_lower: float,
        nuf_upper: float,
    ) -> bool:
        values = [action.N_P_star, action.N_UF_star, np_lower, np_upper, nuf_lower, nuf_upper]
        if any(value != value for value in values):
            return False
        eps = 1.0e-6
        return (
            np_lower - eps <= action.N_P_star <= np_upper + eps
            and nuf_lower - eps <= action.N_UF_star <= nuf_upper + eps
        )

    def _continuous_ranked_directions(
        self,
        center: LeaderAction,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        np_step: float,
        nuf_step: float,
        clipped,
        base_directions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not bool(self.cfg.mpc.leader_continuous_use_sensitivity_directions):
            return base_directions
        scored: list[tuple[float, float, float]] = []
        for dx, dy in base_directions:
            action = clipped(
                center.N_P_star + dx * np_step,
                center.N_UF_star + dy * nuf_step,
            )
            row = self._proxy_score_candidate(0, action, state, forecast, previous)
            scored.append((float(row["objective"]), dx, dy))
        scored.sort(key=lambda item: item[0])
        ranked = [(dx, dy) for _objective, dx, dy in scored]
        if len(ranked) >= 2:
            best_dx, best_dy = ranked[0]
            second_dx, second_dy = ranked[1]
            sensitivity = (
                float(max(-1.0, min(1.0, best_dx + 0.5 * second_dx))),
                float(max(-1.0, min(1.0, best_dy + 0.5 * second_dy))),
            )
            if abs(sensitivity[0]) > 1.0e-9 or abs(sensitivity[1]) > 1.0e-9:
                ranked.insert(0, sensitivity)
        return self._unique_directions(ranked)

    def _unique_directions(self, directions: list[tuple[float, float]]) -> list[tuple[float, float]]:
        seen: set[tuple[float, float]] = set()
        out: list[tuple[float, float]] = []
        for dx, dy in directions:
            key = (round(float(dx), 6), round(float(dy), 6))
            if key in seen:
                continue
            seen.add(key)
            out.append((float(dx), float(dy)))
        return out

    def _evaluate_continuous_action_set(
        self,
        actions: list[LeaderAction],
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        stage: str,
        index_start: int,
        incumbent_obj: float,
    ) -> tuple[list[_LeaderCandidateEvaluation], float]:
        actions = self._unique_leader_actions(actions)
        if not actions:
            return [], float(incumbent_obj)
        if not bool(self.cfg.mpc.leader_continuous_parallel_multistart):
            eval_cfg = self.cfg.with_updates({"mpc": {"stackelberg_leader_parallel_backend": "serial"}})
            controller = StackelbergMPCController(eval_cfg)
            try:
                results = controller._evaluate_candidate_set(
                    actions,
                    list(range(len(actions))),
                    state,
                    forecast,
                    previous,
                    stage=stage,
                    index_offset=index_start,
                    incumbent_obj=incumbent_obj,
                )
            finally:
                controller.close()
        else:
            # Grid path의 parallel evaluator를 재사용한다. 첫 후보는 serial로
            # incumbent를 만든 뒤 나머지 multi-start 후보를 thread/process로 평가한다.
            results = self._evaluate_candidate_set(
                actions,
                list(range(len(actions))),
                state,
                forecast,
                previous,
                stage=stage,
                index_offset=index_start,
                incumbent_obj=incumbent_obj,
            )
        stage_incumbent = min((float(result.objective) for result in results), default=float(incumbent_obj))
        return results, min(float(incumbent_obj), stage_incumbent)

    def _fallback_full_refresh_active(self, state: TrafficState) -> bool:
        interval = max(float(self.cfg.simulation.control_interval), 1.0e-9)
        step_index = int(round(float(state.time_sec) / interval))
        refresh_steps = max(1, int(round(float(self.cfg.mpc.stackelberg_fallback_full_refresh_sec) / interval)))
        return step_index == 0 or step_index % refresh_steps == 0

    def _pfo_incumbent_fallback_enabled(self) -> bool:
        return False

    def _unique_leader_actions(self, actions: list[LeaderAction]) -> list[LeaderAction]:
        seen: set[tuple[float, float]] = set()
        out: list[LeaderAction] = []
        for action in actions:
            key = (round(float(action.N_P_star), 6), round(float(action.N_UF_star), 6))
            if key in seen:
                continue
            seen.add(key)
            out.append(action)
        return out

    def _stage_prefilter_metadata(self, metadata: Dict[str, float], stage: str) -> Dict[str, float]:
        prefix = f"leader_candidate_{stage}_"
        out: Dict[str, float] = {}
        for key, value in metadata.items():
            if key.startswith("leader_candidate_"):
                out[prefix + key[len("leader_candidate_"):]] = float(value)
            else:
                out[f"{prefix}{key}"] = float(value)
        return out

    def _candidate_eval_config(self) -> ExperimentConfig:
        if (
            self.cfg.mpc.stackelberg_leader_parallel_backend == "process"
            and self.cfg.mpc.grid_parallel_backend == "process"
        ):
            return self.cfg.with_updates({
                "mpc": {
                    "grid_parallel_backend": self.cfg.mpc.stackelberg_inner_backend_when_outer_process,
                    "stackelberg_leader_parallel_backend": "serial",
                }
            })
        return self.cfg

    def _leader_process_executor(self, workers: int) -> tuple[ProcessPoolExecutor, bool]:
        if self._leader_process_pool is not None and self._leader_process_pool_workers == workers:
            return self._leader_process_pool, True
        self.close()
        self._leader_process_pool = ProcessPoolExecutor(max_workers=workers)
        self._leader_process_pool_workers = workers
        return self._leader_process_pool, False

    def _evaluate_full_candidate(
        self,
        index: int,
        action: LeaderAction,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        stage: str = "coarse",
        incumbent_obj: float = float("inf"),
        rollout_abort_obj: float = float("inf"),
    ) -> _LeaderCandidateEvaluation:
        raw_action = LeaderAction(float(action.N_P_star), float(action.N_UF_star))
        action, projection_meta = self._project_action_to_follower_feasible_np(
            action, state, forecast, previous
        )
        if isinstance(self.nash_solver, DistributedCoordinator):
            nash = self.nash_solver.solve(
                state.copy(),
                action,
                forecast,
                previous,
                leader_incumbent_obj=incumbent_obj,
            )
        else:
            nash = self.nash_solver.solve(state.copy(), action, forecast, previous)
        evaluated_action, closure_metadata = self._close_nash_response_leader_action(
            action,
            nash,
            forecast,
            intent_action=raw_action,
        )
        predicted_states, follower_ttt, rollout_used = self._leader_evaluation_base(
            state,
            nash,
            forecast,
            incumbent_obj=rollout_abort_obj,
            previous=previous,
        )
        objective_terms = self.leader.objective_terms(
            predicted_states,
            nash.control,
            previous,
            follower_ttt,
            nash.converged,
            nash.residual_objective,
            nash.residual_control,
        )
        metadata = {
            "leader_response_proxy_state_count": float(len(predicted_states)),
            "leader_candidate_raw_N_P_star": float(raw_action.N_P_star),
            "leader_candidate_raw_N_UF_star": float(raw_action.N_UF_star),
            **projection_meta,
            **closure_metadata,
            "leader_candidate_incumbent_active": float(incumbent_obj < float("inf")),
            "leader_candidate_incumbent_objective": float(incumbent_obj if incumbent_obj < float("inf") else 0.0),
            "leader_candidate_follower_early_terminated_candidates": float(
                nash.diagnostics.get(
                    "distributed_grid_early_terminated_candidates",
                    nash.control.diagnostics.get("distributed_grid_early_terminated_candidates", 0.0),
                )
            ),
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

    def _evaluation_terminal_proxy(self, evaluation: _LeaderCandidateEvaluation) -> float:
        diag = dict(evaluation.nash.diagnostics)
        diag.update(evaluation.nash.control.diagnostics)
        return float(diag.get("distributed_response_terminal_proxy_vehicles", 0.0))

    def _evaluation_completed_proxy(self, evaluation: _LeaderCandidateEvaluation) -> float:
        diag = dict(evaluation.nash.diagnostics)
        diag.update(evaluation.nash.control.diagnostics)
        return float(diag.get("distributed_response_mainline_exit_veh", 0.0)) + float(
            diag.get("distributed_response_boundary_out_sink_veh", 0.0)
        )

    def _evaluation_rollout_ttt(self, evaluation: _LeaderCandidateEvaluation) -> Optional[float]:
        """후보의 realized rollout TTT[veh·h]. 결측이면 None(guard가 기존 obj 로직으로 fallback)."""
        diag = dict(evaluation.nash.diagnostics)
        diag.update(evaluation.nash.control.diagnostics)
        if "distributed_response_rollout_ttt" not in diag:
            return None
        return float(diag["distributed_response_rollout_ttt"])

    def _fallback_guard_rejects(
        self,
        leader_best: _LeaderCandidateEvaluation,
        fallback_best: _LeaderCandidateEvaluation,
    ) -> tuple[bool, Dict[str, float]]:
        leader_terminal = self._evaluation_terminal_proxy(leader_best)
        fallback_terminal = self._evaluation_terminal_proxy(fallback_best)
        leader_completed = self._evaluation_completed_proxy(leader_best)
        fallback_completed = self._evaluation_completed_proxy(fallback_best)
        leader_obj = float(leader_best.objective)
        fallback_obj = float(fallback_best.objective)
        terminal_margin = max(20.0, 0.05 * max(fallback_terminal, 1.0))
        completed_margin = max(1.0, 0.05 * max(fallback_completed, 1.0))
        terminal_worse = leader_terminal > fallback_terminal + terminal_margin
        terminal_severe = leader_terminal > fallback_terminal + max(100.0, 0.25 * max(fallback_terminal, 1.0))
        completed_worse = leader_completed + completed_margin < fallback_completed
        completed_severe = leader_completed + max(2.0, 0.10 * max(fallback_completed, 1.0)) < fallback_completed
        objective_worse = leader_obj >= fallback_obj - 1.0e-12
        objective_gain = max(0.0, fallback_obj - leader_obj)
        required_gain = 0.05 * max(abs(fallback_obj), 1.0)
        insufficient_gain = objective_gain < required_gain
        # guard 비교 척도: realized rollout-TTT(기본) vs penalized objective(레거시).
        # penalized obj는 mfd/density/boundary 벌점이 실제 TTT와 어긋나 TTT 좋은 leader를
        # 잘못 기각했다(sweet_128 +13.5% 손실, 2026-06-25). rollout_ttt 결측 시 obj 로직으로 안전 fallback.
        use_ttt = bool(getattr(self.cfg.mpc, "stackelberg_fallback_guard_use_rollout_ttt", False))
        leader_ttt = self._evaluation_rollout_ttt(leader_best)
        fallback_ttt = self._evaluation_rollout_ttt(fallback_best)
        # 층2(2026-07-14): β̂ 보정 — leader측 예측 TTT만 낙관편향 배율 β̂로 보정한다.
        # leader측은 ~50 후보 argmax 선택이라 낙관이 증폭(optimizer's curse)되지만
        # incumbent(PFO)는 단일 해라 선택편향이 없어 무보정. β̂=None(추정 전)이면 기존 거동.
        beta_hat: Optional[float] = None
        if bool(getattr(self.cfg.mpc, "fallback_guard_beta", False)):
            beta_value = getattr(self, "_beta_hat", None)
            if beta_value is not None:
                beta_hat = float(beta_value)
        leader_ttt_guard = (
            leader_ttt
            if (leader_ttt is None or beta_hat is None)
            else float(leader_ttt) * beta_hat
        )
        ttt_available = use_ttt and leader_ttt is not None and fallback_ttt is not None
        if ttt_available:
            # leader가 PFO보다 예측 rollout-TTT 기준 (margin 넘게) 나쁘면 기각, 동률·개선이면 채택.
            # per-step 예측은 128(복리 이득)과 115(근소 손해)를 거의 구분 못 하므로 동률을 채택해
            # 128의 누적 이득(+13.7%)을 전부 살린다. 대가로 저혼잡(115)에서 PFO 대비 ~0.9% 회귀가
            # 있으나, leader 가치를 주장하지 않는 저부하라 수용한다(2026-06-25 사용자 결정).
            ttt_margin = max(1.0, 1.0e-3 * max(fallback_ttt, 1.0))
            ttt_worse = leader_ttt_guard > fallback_ttt + ttt_margin
            # 1차 기각을 TTT로. 잔류/완료 severe는 throughput 안전장치로 유지.
            reject = bool(ttt_worse or terminal_severe or completed_severe)
        else:
            ttt_worse = False
            reject = bool(
                objective_worse
                or terminal_severe
                or completed_severe
                or ((terminal_worse or completed_worse) and insufficient_gain)
            )
        return reject, {
            "leader_fallback_guard_rejected_leader": float(reject),
            "leader_fallback_guard_metric_ttt": float(ttt_available),
            "leader_fallback_guard_leader_rollout_ttt": float(leader_ttt) if leader_ttt is not None else 0.0,
            "leader_fallback_guard_fallback_rollout_ttt": float(fallback_ttt) if fallback_ttt is not None else 0.0,
            # 층2: β̂ 보정 적용 여부·값·보정된 leader측 비교값(미적용 시 원값 그대로).
            "leader_fallback_guard_beta_applied": float(beta_hat is not None),
            "leader_fallback_guard_beta_hat": float(beta_hat) if beta_hat is not None else 0.0,
            "leader_fallback_guard_leader_rollout_ttt_beta": (
                float(leader_ttt_guard) if leader_ttt_guard is not None else 0.0
            ),
            "leader_fallback_guard_ttt_worse": float(ttt_worse),
            "leader_fallback_guard_terminal_worse": float(terminal_worse),
            "leader_fallback_guard_terminal_severe": float(terminal_severe),
            "leader_fallback_guard_completed_worse": float(completed_worse),
            "leader_fallback_guard_completed_severe": float(completed_severe),
            "leader_fallback_guard_objective_worse": float(objective_worse),
            "leader_fallback_guard_insufficient_gain": float(insufficient_gain),
            "leader_fallback_guard_leader_objective": leader_obj,
            "leader_fallback_guard_fallback_objective": fallback_obj,
            "leader_fallback_guard_objective_gain": float(objective_gain),
            "leader_fallback_guard_required_gain": float(required_gain),
            "leader_fallback_guard_leader_terminal_proxy_veh": leader_terminal,
            "leader_fallback_guard_fallback_terminal_proxy_veh": fallback_terminal,
            "leader_fallback_guard_leader_completed_proxy_veh": leader_completed,
            "leader_fallback_guard_fallback_completed_proxy_veh": fallback_completed,
        }

    def _select_with_fallback_guard(
        self,
        leader_evaluations: list[_LeaderCandidateEvaluation],
        fallback_evaluations: list[_LeaderCandidateEvaluation],
    ) -> tuple[_LeaderCandidateEvaluation, Dict[str, float]]:
        if not leader_evaluations:
            raise ValueError("Stackelberg leader produced no evaluated candidates.")
        leader_best = min(leader_evaluations, key=lambda item: item.objective)
        metadata: Dict[str, float] = {
            "leader_fallback_guard_active": 1.0,
            "leader_fallback_guard_selected": 0.0,
            "leader_fallback_guard_selected_pfo": 0.0,
            "leader_fallback_guard_selected_no_control": 0.0,
            "leader_fallback_candidate_count": float(len(fallback_evaluations)),
        }
        if not fallback_evaluations:
            metadata["leader_fallback_guard_rejected_leader"] = 0.0
            return leader_best, metadata
        fallback_best = min(fallback_evaluations, key=lambda item: item.objective)
        reject, guard_meta = self._fallback_guard_rejects(leader_best, fallback_best)
        metadata.update(guard_meta)
        metadata.update({
            "leader_fallback_best_stage_pfo": float(fallback_best.stage == "fallback_pfo"),
            "leader_fallback_best_stage_no_control": float(fallback_best.stage == "fallback_no_control"),
            "leader_fallback_best_objective": float(fallback_best.objective),
            "leader_fallback_best_terminal_proxy_veh": self._evaluation_terminal_proxy(fallback_best),
            "leader_fallback_best_completed_proxy_veh": self._evaluation_completed_proxy(fallback_best),
        })
        for fallback in fallback_evaluations:
            prefix = f"leader_{fallback.stage}"
            metadata[f"{prefix}_objective"] = float(fallback.objective)
            metadata[f"{prefix}_terminal_proxy_veh"] = self._evaluation_terminal_proxy(fallback)
            metadata[f"{prefix}_completed_proxy_veh"] = self._evaluation_completed_proxy(fallback)
        if reject:
            metadata["leader_fallback_guard_selected"] = 1.0
            metadata["leader_fallback_guard_selected_pfo"] = float(fallback_best.stage == "fallback_pfo")
            metadata["leader_fallback_guard_selected_no_control"] = float(
                fallback_best.stage == "fallback_no_control"
            )
            return fallback_best, metadata
        return leader_best, metadata

    def _evaluate_fallback_candidates(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        start_index: int,
    ) -> list[_LeaderCandidateEvaluation]:
        if not isinstance(self.nash_solver, DistributedCoordinator):
            return []
        evaluations: list[_LeaderCandidateEvaluation] = []

        no_control = ControlAction.uncontrolled(self.cfg)
        no_obj, no_diag = self.nash_solver._response_tts_objective(
            state,
            no_control,
            forecast,
            residual=0.0,
            proxy_objective=0.0,
        )
        no_control.diagnostics.update(no_diag)
        no_nash = NashResult(
            control=no_control,
            objective_value=float(no_obj),
            iterations=0,
            converged=True,
            residual_objective=0.0,
            residual_control=0.0,
            diagnostics=dict(no_diag),
        )
        no_eval = self._make_fallback_evaluation(
            start_index,
            "fallback_no_control",
            no_nash,
            previous,
            state,
            forecast,
        )
        evaluations.append(no_eval)
        self._append_progress_event(
            event="candidate_evaluated",
            stage="fallback_no_control",
            completed=1,
            total=2,
            evaluation=no_eval,
            best_objective=no_eval.objective,
        )

        fallback_refresh = self._fallback_full_refresh_active(state)
        cached_previous_used = (
            bool(self.cfg.mpc.stackelberg_fallback_use_cached_pfo)
            and not fallback_refresh
            and self._pfo_fallback_previous_control is not None
        )
        pfo_previous = (
            self._pfo_fallback_previous_control.copy()
            if cached_previous_used and self._pfo_fallback_previous_control is not None
            else previous.copy()
        )
        pfo_previous.N_P_star = 0.0
        pfo_previous.N_UF_star = 0.0
        pfo_previous.inflow_outflow_allocation = {}
        pfo_nash = self.nash_solver.solve(
            state.copy(),
            None,
            forecast,
            pfo_previous,
        )
        self._pfo_fallback_previous_control = pfo_nash.control.copy()
        pfo_eval = self._make_fallback_evaluation(
            start_index + 1,
            "fallback_pfo",
            pfo_nash,
            previous,
            state,
            forecast,
            extra_metadata={
                "leader_fallback_pfo_global_refresh": float(fallback_refresh),
                "leader_fallback_pfo_local_search": float(not fallback_refresh),
                "leader_fallback_pfo_cached_previous_used": float(cached_previous_used),
                "leader_fallback_pfo_refresh_interval_sec": float(
                    self.cfg.mpc.stackelberg_fallback_full_refresh_sec
                ),
            },
        )
        evaluations.append(pfo_eval)
        self._append_progress_event(
            event="candidate_evaluated",
            stage="fallback_pfo",
            completed=2,
            total=2,
            evaluation=pfo_eval,
            best_objective=min(item.objective for item in evaluations),
        )
        return evaluations

    def _make_fallback_evaluation(
        self,
        index: int,
        stage: str,
        nash: NashResult,
        previous: ControlAction,
        state: TrafficState,
        forecast: list[DemandStep],
        extra_metadata: Optional[Dict[str, float]] = None,
    ) -> _LeaderCandidateEvaluation:
        action = LeaderAction(float(nash.control.N_P_star), float(nash.control.N_UF_star))
        horizon = max(1, len(forecast[: self.cfg.mpc.horizon_steps]))
        objective_terms = self.leader.objective_terms(
            [state.copy() for _ in range(horizon)],
            nash.control,
            previous,
            float(nash.objective_value),
            nash.converged,
            nash.residual_objective,
            nash.residual_control,
        )
        return _LeaderCandidateEvaluation(
            index=index,
            action=action,
            nash=nash,
            objective=float(objective_terms.get("leader_total_objective", nash.objective_value)),
            objective_terms={
                "leader_base_accumulation": float(objective_terms.get("leader_base_accumulation", 0.0)),
                "leader_objective_base": float(objective_terms.get("leader_objective_base", nash.objective_value)),
                "leader_state_accumulation_base": float(objective_terms.get("leader_state_accumulation_base", 0.0)),
                "leader_follower_ttt_base": float(objective_terms.get("leader_follower_ttt_base", nash.objective_value)),
                "leader_boundary_leg_excluded_veh": float(objective_terms.get("leader_boundary_leg_excluded_veh", 0.0)),
                "leader_target_penalty": float(objective_terms.get("leader_target_penalty", 0.0)),
                "leader_mfd_storage_excess_veh": float(
                    objective_terms.get("leader_mfd_storage_excess_veh", 0.0)
                ),
                "leader_mfd_movement_excess_veh": float(
                    objective_terms.get("leader_mfd_movement_excess_veh", 0.0)
                ),
                "leader_mfd_link_excess_veh": float(objective_terms.get("leader_mfd_link_excess_veh", 0.0)),
                "leader_mfd_storage_penalty": float(objective_terms.get("leader_mfd_storage_penalty", 0.0)),
                "leader_boundary_in_queue_penalty": float(objective_terms.get("leader_boundary_in_queue_penalty", 0.0)),
                "leader_density_excess": float(objective_terms.get("leader_density_excess", 0.0)),
                "leader_density_penalty": float(objective_terms.get("leader_density_penalty", 0.0)),
                "leader_density_effective_lane_weight_count": float(
                    objective_terms.get("leader_density_effective_lane_weight_count", 0.0)
                ),
                "leader_smoothness_penalty": float(objective_terms.get("leader_smoothness_penalty", 0.0)),
                "leader_nonconvergence_penalty": float(objective_terms.get("leader_nonconvergence_penalty", 0.0)),
                "leader_nonconvergence_obj_residual_component": float(
                    objective_terms.get("leader_nonconvergence_obj_residual_component", 0.0)
                ),
                "leader_nonconvergence_control_residual_component": float(
                    objective_terms.get("leader_nonconvergence_control_residual_component", 0.0)
                ),
                "leader_total_objective": float(objective_terms.get("leader_total_objective", nash.objective_value)),
            },
            metadata={
                "leader_response_proxy_state_count": float(horizon),
                "leader_fallback_candidate": 1.0,
                **(extra_metadata or {}),
            },
            rollout_used=False,
            stage=stage,
        )

    def _proxy_score_candidate(
        self,
        index: int,
        action: LeaderAction,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
    ) -> Dict[str, float]:
        action, projection_meta = self._project_action_to_follower_feasible_np(
            action, state, forecast, previous
        )
        control = previous.copy()
        control.N_P_star = float(action.N_P_star)
        control.N_UF_star = float(action.N_UF_star)
        if isinstance(self.nash_solver, DistributedCoordinator):
            weights = {
                ramp: float(self.cfg.network.ramp_capacity_veh_h[ramp])
                for ramp in self.cfg.network.ramps
            }
            control.ramp_metering = self.nash_solver._leader_metering_projection(action, weights)
            follower_obj, proxy_diag = self.nash_solver._response_tts_objective(
                state,
                control,
                forecast,
                residual=0.0,
                proxy_objective=0.0,
            )
        else:
            horizon_h = self.cfg.simulation.T_c_h * max(1, min(len(forecast), self.cfg.mpc.horizon_steps))
            follower_obj = (
                state.total_urban_vehicles(self.cfg.network)
                + state.total_freeway_vehicles(self.cfg.network)
            ) * horizon_h
            proxy_diag = {}
        horizon = max(1, len(forecast[: self.cfg.mpc.horizon_steps]))
        terms = self.leader.objective_terms(
            [state.copy() for _ in range(horizon)],
            control,
            previous,
            float(follower_obj),
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
            "spillback_violation": float(proxy_diag.get("distributed_response_total_spillback_violation_veh", 0.0)),
            **projection_meta,
        }

    def _project_action_to_follower_feasible_np(
        self,
        action: LeaderAction,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
    ) -> tuple[LeaderAction, Dict[str, float]]:
        """Follower가 구현 가능한 Σnin 범위로 leader N_P 후보를 투영한다.

        PFO anchor/fallback 없이도 Stackelberg leader가 raw infeasible target 공간을
        탐색하지 않도록, follower solver가 노출하는 현재 feasible range를 사용한다.
        N_UF는 기존 freeway follower projection 경로가 처리하므로 여기서는 N_P만
        조정한다. 단위는 controller horizon 전체 차량수[veh].
        """
        feasible_fn = getattr(self.nash_solver, "leader_np_feasible_range", None)
        if feasible_fn is None:
            return action, {"leader_np_follower_feasible_projection_active": 0.0}
        try:
            sigma_min, sigma_max, diag = feasible_fn(state.copy(), list(forecast), previous.copy())
        except Exception:
            return action, {
                "leader_np_follower_feasible_projection_active": 0.0,
                "leader_np_follower_feasible_projection_failed": 1.0,
            }
        lower = min(float(sigma_min), float(sigma_max))
        upper = max(float(sigma_min), float(sigma_max))
        raw_np = float(action.N_P_star)
        np_coordination_mode = str(
            getattr(self.cfg.mpc, "wu_faithful_np_coordination_mode", "equality")
        ).lower()
        if np_coordination_mode == "cap":
            projected_np = float(min(raw_np, upper))
        else:
            projected_np = float(min(max(raw_np, lower), upper))
        meta = {
            "leader_np_follower_feasible_projection_active": 1.0,
            "leader_np_follower_feasible_projection_failed": 0.0,
            "leader_np_follower_feasible_projection_cap_mode": float(np_coordination_mode == "cap"),
            "leader_np_follower_feasible_min": lower,
            "leader_np_follower_feasible_max": upper,
            "leader_np_raw_N_P_star": raw_np,
            "leader_np_projected_N_P_star": projected_np,
            "leader_np_projection_residual": float(raw_np - projected_np),
            "leader_np_projection_applied": float(abs(raw_np - projected_np) > 1.0e-9),
        }
        for key, value in diag.items():
            meta[f"leader_{key}"] = float(value)
        return LeaderAction(projected_np, float(action.N_UF_star)), meta

    def _prefilter_leader_candidates(
        self,
        candidates: list[LeaderAction],
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        global_scope: bool = False,
    ) -> tuple[list[int], Dict[str, float]]:
        rows = [
            self._proxy_score_candidate(idx, action, state, forecast, previous)
            for idx, action in enumerate(candidates)
        ]
        ranked = sorted(rows, key=lambda row: (row["objective"], row["spillback_violation"], row["index"]))
        objectives = [row["objective"] for row in rows] or [0.0]
        top_k = int(
            self.cfg.mpc.stackelberg_prefilter_top_k
            if global_scope
            else self.cfg.mpc.stackelberg_prefilter_local_top_k
        )
        if top_k <= 0 or top_k >= len(candidates):
            selected = list(range(len(candidates)))
            active = False
        else:
            selected = []
            seen = set()

            def add_index(idx: int) -> None:
                if idx in seen:
                    return
                seen.add(idx)
                selected.append(idx)

            for row in ranked[:top_k]:
                add_index(int(row["index"]))
            add_index(0)
            nearest_previous = min(
                range(len(candidates)),
                key=lambda idx: (
                    abs(candidates[idx].N_P_star - previous.N_P_star)
                    + abs(candidates[idx].N_UF_star - previous.N_UF_star)
                ),
            )
            add_index(int(nearest_previous))
            active = True
        best = ranked[0] if ranked else {"index": 0.0, "objective": 0.0}
        second = ranked[1] if len(ranked) > 1 else best
        return selected, {
            "leader_candidate_prefilter_active": float(active),
            "leader_candidate_prefilter_top_k": float(top_k),
            "leader_candidate_prefilter_scope_global": float(global_scope),
            "leader_candidate_prefilter_scope_local": float(not global_scope),
            "leader_candidate_proxy_evaluated_count": float(len(rows)),
            "leader_candidate_prefilter_selected_count": float(len(selected)),
            "leader_candidate_proxy_best_index": float(best["index"]),
            "leader_candidate_proxy_second_index": float(second["index"]),
            "leader_candidate_proxy_best_objective": float(best["objective"]),
            "leader_candidate_proxy_second_objective": float(second["objective"]),
            "leader_candidate_proxy_objective_gap": float(second["objective"] - best["objective"]),
            "leader_candidate_proxy_objective_spread": float(max(objectives) - min(objectives)),
        }

    def _evaluate_candidate_set(
        self,
        candidates: list[LeaderAction],
        selected_indices: list[int],
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        stage: str = "coarse",
        index_offset: int = 0,
        incumbent_obj: float = float("inf"),
    ) -> list[_LeaderCandidateEvaluation]:
        if not selected_indices:
            raise ValueError("Stackelberg leader prefilter removed every candidate.")
        backend = self.cfg.mpc.stackelberg_leader_parallel_backend
        workers = max(1, min(int(self.cfg.mpc.stackelberg_leader_parallel_max_workers), len(selected_indices)))
        eval_cfg = self._candidate_eval_config()
        inner_backend = eval_cfg.mpc.grid_parallel_backend
        payloads = [
            {
                "cfg": eval_cfg,
                "index": idx + index_offset,
                "action": candidates[idx],
                "state": state,
                "forecast": forecast,
                "previous": previous,
                "stage": stage,
                "incumbent_obj": incumbent_obj,
            }
            for idx in selected_indices
        ]
        results: list[_LeaderCandidateEvaluation] = []
        stage_incumbent = float(incumbent_obj)
        seed_used = False
        process_pool_reused_existing = False
        total_candidates = len(selected_indices)
        self._append_progress_event(
            event="stage_started",
            stage=stage,
            completed=0,
            total=total_candidates,
            best_objective=stage_incumbent if stage_incumbent < float("inf") else None,
        )
        if payloads:
            seed_payload = dict(payloads[0])
            seed_payload["incumbent_obj"] = stage_incumbent
            seed_result = _stackelberg_candidate_worker(seed_payload)
            results.append(seed_result)
            stage_incumbent = min(stage_incumbent, float(seed_result.objective))
            seed_used = True
            self._append_progress_event(
                event="candidate_evaluated",
                stage=stage,
                completed=len(results),
                total=total_candidates,
                evaluation=seed_result,
                best_objective=stage_incumbent,
            )
            payloads = payloads[1:]
            for payload in payloads:
                payload["incumbent_obj"] = stage_incumbent
        if backend == "serial" or workers <= 1:
            for payload in payloads:
                payload["incumbent_obj"] = stage_incumbent
                result = _stackelberg_candidate_worker(payload)
                results.append(result)
                stage_incumbent = min(stage_incumbent, float(result.objective))
                self._append_progress_event(
                    event="candidate_evaluated",
                    stage=stage,
                    completed=len(results),
                    total=total_candidates,
                    evaluation=result,
                    best_objective=stage_incumbent,
                )
            backend_used = "serial"
            chunks = 1
        elif backend == "thread":
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_stackelberg_candidate_worker, payload) for payload in payloads]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    stage_incumbent = min(stage_incumbent, float(result.objective))
                    self._append_progress_event(
                        event="candidate_evaluated",
                        stage=stage,
                        completed=len(results),
                        total=total_candidates,
                        evaluation=result,
                        best_objective=stage_incumbent,
                    )
            backend_used = "thread"
            chunks = max(1, len(payloads))
        else:
            if self.cfg.mpc.stackelberg_reuse_process_pool:
                executor, process_pool_reused_existing = self._leader_process_executor(workers)
                futures = [executor.submit(_stackelberg_candidate_worker, payload) for payload in payloads]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    stage_incumbent = min(stage_incumbent, float(result.objective))
                    self._append_progress_event(
                        event="candidate_evaluated",
                        stage=stage,
                        completed=len(results),
                        total=total_candidates,
                        evaluation=result,
                        best_objective=stage_incumbent,
                    )
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_stackelberg_candidate_worker, payload) for payload in payloads]
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        stage_incumbent = min(stage_incumbent, float(result.objective))
                        self._append_progress_event(
                            event="candidate_evaluated",
                            stage=stage,
                            completed=len(results),
                            total=total_candidates,
                            evaluation=result,
                            best_objective=stage_incumbent,
                        )
            backend_used = "process"
            chunks = max(1, len(payloads))
        diag = {
            "leader_candidate_parallel_backend_serial": float(backend_used == "serial"),
            "leader_candidate_parallel_backend_thread": float(backend_used == "thread"),
            "leader_candidate_parallel_backend_process": float(backend_used == "process"),
            "leader_candidate_parallel_workers": float(workers if backend_used != "serial" else 1),
            "leader_candidate_parallel_chunks": float(chunks),
            "leader_candidate_process_pool_reuse_enabled": float(
                bool(self.cfg.mpc.stackelberg_reuse_process_pool)
            ),
            "leader_candidate_process_pool_reused_existing": float(process_pool_reused_existing),
            "leader_candidate_inner_grid_backend_serial": float(inner_backend == "serial"),
            "leader_candidate_inner_grid_backend_thread": float(inner_backend == "thread"),
            "leader_candidate_inner_grid_backend_process": float(inner_backend == "process"),
            "leader_candidate_nested_process_avoided": float(
                backend_used == "process" and self.cfg.mpc.grid_parallel_backend == "process" and inner_backend != "process"
            ),
            "leader_candidate_incumbent_seed_used": float(seed_used),
            "leader_candidate_incumbent_initial_objective": float(
                incumbent_obj if incumbent_obj < float("inf") else 0.0
            ),
            "leader_candidate_incumbent_final_objective": float(
                min((result.objective for result in results), default=stage_incumbent)
            ),
            "leader_candidate_incumbent_early_termination_active": float(
                any(result.metadata.get("leader_candidate_incumbent_active", 0.0) > 0.5 for result in results)
            ),
            "leader_candidate_follower_early_terminated_candidates": float(sum(
                result.metadata.get("leader_candidate_follower_early_terminated_candidates", 0.0)
                for result in results
            )),
        }
        for result in results:
            result.metadata.update(diag)
        return results

    def _forecast_demand_metadata(self, forecast: list[DemandStep]) -> Dict[str, float]:
        """forecast 요약 진단을 compact하게 남긴다.

        후보 집합은 같아도 평가 입력 수요가 달라졌는지 test/진단에서 확인할 수 있도록,
        horizon 내부의 첫 스텝/미래 평균/peak 유량[veh/h]만 기록한다.
        """
        horizon = forecast[: max(1, self.cfg.mpc.horizon_steps)]

        def total(step: DemandStep, attr: str) -> float:
            values = getattr(step, attr)
            return float(sum(float(v) for v in values.values()))

        def mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        by_group = {
            "mainline": [total(step, "freeway_mainline") for step in horizon],
            "boundary": [total(step, "urban_boundary") for step in horizon],
            "ramp": [total(step, "ramp_arrival") for step in horizon],
        }
        metadata: Dict[str, float] = {"leader_forecast_steps": float(len(horizon))}
        total_by_step = [
            by_group["mainline"][idx] + by_group["boundary"][idx] + by_group["ramp"][idx]
            for idx in range(len(horizon))
        ]
        future_total = total_by_step[1:]
        metadata.update({
            "leader_forecast_total_first_veh_h": float(total_by_step[0]),
            "leader_forecast_total_future_mean_veh_h": mean(future_total),
            "leader_forecast_total_peak_veh_h": float(max(total_by_step, default=0.0)),
        })
        for name, values in by_group.items():
            future = values[1:]
            metadata.update({
                f"leader_forecast_{name}_first_veh_h": float(values[0]),
                f"leader_forecast_{name}_future_mean_veh_h": mean(future),
                f"leader_forecast_{name}_peak_veh_h": float(max(values, default=0.0)),
            })
        return metadata

    def _candidate_evaluation_metadata(
        self,
        evaluations: list[dict[str, object]],
        selected: ControlAction,
        selected_eval: Optional[_LeaderCandidateEvaluation] = None,
    ) -> Dict[str, float]:
        """candidate 평가/랭킹 요약만 diagnostics에 보존한다.

        전체 후보 로그는 커질 수 있으므로 best/second, objective spread, 선택 action만 남긴다.
        """
        if not evaluations:
            return {}
        ranked = sorted(evaluations, key=lambda row: row["objective"])
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        objectives = [row["objective"] for row in evaluations]
        selected_stage = selected_eval.stage if selected_eval is not None else str(best.get("stage", ""))
        selected_objective = float(selected_eval.objective) if selected_eval is not None else float(best["objective"])
        selected_diag = getattr(selected, "diagnostics", None) or {}
        selected_intent_np = self._finite_float(selected_diag.get("leader_intent_N_P_star"))
        selected_intent_nuf = self._finite_float(selected_diag.get("leader_intent_N_UF_star"))
        selected_realized_np = self._finite_float(selected_diag.get("leader_realized_N_P_star"))
        selected_projected_np = self._finite_float(selected_diag.get("leader_projected_N_P_star"))
        selected_realized_nuf = self._finite_float(selected_diag.get("leader_realized_N_UF_star"))
        if selected_intent_np is None:
            selected_intent_np = float(best.get("intent_N_P_star", selected.N_P_star))
        if selected_intent_nuf is None:
            selected_intent_nuf = float(best.get("intent_N_UF_star", selected.N_UF_star))
        if selected_realized_np is None:
            selected_realized_np = float(best.get("realized_N_P_star", selected.N_P_star))
        if selected_projected_np is None:
            selected_projected_np = float(best.get("projected_N_P_star", selected_realized_np))
        if selected_realized_nuf is None:
            selected_realized_nuf = float(best.get("realized_N_UF_star", selected.N_UF_star))
        return {
            "leader_selected_N_P_star": float(selected.N_P_star),
            "leader_selected_N_UF_star": float(selected.N_UF_star),
            "leader_selected_intent_N_P_star": float(selected_intent_np),
            "leader_selected_intent_N_UF_star": float(selected_intent_nuf),
            "leader_selected_realized_N_P_star": float(selected_realized_np),
            "leader_selected_projected_N_P_star": float(selected_projected_np),
            "leader_selected_realized_N_UF_star": float(selected_realized_nuf),
            "leader_selected_objective": selected_objective,
            "leader_selected_stage_coarse": float(str(selected_stage).startswith("coarse")),
            "leader_selected_stage_refined": float(selected_stage == "refined"),
            "leader_selected_stage_fallback": float(str(selected_stage).startswith("fallback")),
            "leader_selected_stage_fallback_pfo": float(selected_stage == "fallback_pfo"),
            "leader_selected_stage_fallback_no_control": float(selected_stage == "fallback_no_control"),
            "leader_candidate_best_index": float(best["index"]),
            "leader_candidate_second_index": float(second["index"]),
            "leader_candidate_best_stage_coarse": float(str(best.get("stage", "")).startswith("coarse")),
            "leader_candidate_best_stage_refined": float(best.get("stage") == "refined"),
            "leader_candidate_best_stage_fallback": float(str(best.get("stage", "")).startswith("fallback")),
            "leader_candidate_second_stage_coarse": float(str(second.get("stage", "")).startswith("coarse")),
            "leader_candidate_second_stage_refined": float(second.get("stage") == "refined"),
            "leader_candidate_second_stage_fallback": float(str(second.get("stage", "")).startswith("fallback")),
            "leader_candidate_best_N_P_star": float(best["N_P_star"]),
            "leader_candidate_best_N_UF_star": float(best["N_UF_star"]),
            "leader_candidate_second_N_P_star": float(second["N_P_star"]),
            "leader_candidate_second_N_UF_star": float(second["N_UF_star"]),
            "leader_candidate_best_intent_N_P_star": float(best.get("intent_N_P_star", best["N_P_star"])),
            "leader_candidate_best_intent_N_UF_star": float(best.get("intent_N_UF_star", best["N_UF_star"])),
            "leader_candidate_best_realized_N_P_star": float(best.get("realized_N_P_star", best["N_P_star"])),
            "leader_candidate_best_projected_N_P_star": float(best.get("projected_N_P_star", best["N_P_star"])),
            "leader_candidate_best_realized_N_UF_star": float(best.get("realized_N_UF_star", best["N_UF_star"])),
            "leader_candidate_second_intent_N_P_star": float(second.get("intent_N_P_star", second["N_P_star"])),
            "leader_candidate_second_intent_N_UF_star": float(second.get("intent_N_UF_star", second["N_UF_star"])),
            "leader_candidate_second_realized_N_P_star": float(second.get("realized_N_P_star", second["N_P_star"])),
            "leader_candidate_second_projected_N_P_star": float(second.get("projected_N_P_star", second["N_P_star"])),
            "leader_candidate_second_realized_N_UF_star": float(second.get("realized_N_UF_star", second["N_UF_star"])),
            "leader_candidate_best_objective": float(best["objective"]),
            "leader_candidate_second_objective": float(second["objective"]),
            "leader_candidate_objective_gap": float(second["objective"] - best["objective"]),
            "leader_candidate_objective_spread": float(max(objectives) - min(objectives)),
        }

    def _leader_evaluation_base(
        self,
        state: TrafficState,
        nash: NashResult,
        forecast: list[DemandStep],
        incumbent_obj: float = float("inf"),
        previous: Optional[ControlAction] = None,
    ) -> tuple[list[TrafficState], float, bool]:
        """Leader 기본 평가는 follower response objective를 그대로 사용한다.

        사용자가 제시한 Stackelberg 구조에서는 follower equilibrium/response가 산출한
        objective가 leader의 follower-TTT 항이다. 따라서 기본 `follower_ttt` 모드에서는
        후보 control을 다시 full coupled plant로 rollout하지 않는다. Legacy
        `state_accumulation` 모드는 state trajectory 자체가 base이므로 기존 rollout을 유지한다.
        """
        # In follower_ttt mode, keep the follower response as the base objective
        # but evaluate leader penalties on future MPC rollout states.
        abort_above = None
        if (
            bool(getattr(self.cfg.mpc, "leader_rollout_early_stop", False))
            and incumbent_obj != float("inf")
        ):
            abort_above = float(incumbent_obj)
        states, rollout_ttt = self._predict(
            state, nash.control, forecast, abort_above=abort_above, previous=previous)
        if self.cfg.leader.objective_mode == "state_accumulation":
            return states, rollout_ttt, True
        # leader_value_depth>0면 leader의 full (3+d) rollout TTT를 base로(=TTT_3+V). follower는
        # myopic-3로 control을 정하고, leader가 그 control을 full rollout으로 랭킹·pricing(∂(TTT+V)/∂lever).
        # FAR-D0(2026-07-09): leader_mfd_far_at_d0=True면 depth=0에서도 같은 형태
        # (H-step rollout TTT + far)로 채점 — 역사적 d0는 rollout이 아니라 follower 응답
        # proxy로 랭킹했으므로, "얕은 leader + far가 충분한가"를 채점 형태를 고정한 채
        # 검정하려면 이 게이트가 필요하다. 기본 False = 비트동일(legacy/proxy 경로 보존).
        if int(getattr(self.cfg.mpc, "leader_value_depth", 0)) > 0 or bool(
            getattr(self.cfg.mpc, "leader_mfd_far_at_d0", False)
        ):
            # far(MFD tail): near rollout 끝(states[-1]) 잔여 accumulation의 배수 cost-to-go를
            # 해석적으로 가산(§3.2). far가 temporal tail을 싸게 담으면 near 깊이 d를 줄일 수 있다.
            return (
                states,
                rollout_ttt
                + self._mfd_far_cost_to_go(states[-1])
                + leader_hinge_cost(self.cfg, states, forecast),
                True,
            )
        if float(nash.control.diagnostics.get("leader_response_closure_use_rollout_objective", 0.0)) >= 0.5:
            return states, rollout_ttt, True
        return states, float(nash.objective_value), True

        # target/density penalty는 현재 response-state proxy 위에서 평가한다. follower objective
        # 자체가 후보별 response 비용을 담고, full system rollout 비용은 의도적으로 배제한다.

    def _mfd_far_cost_to_go(self, state: TrafficState) -> float:
        """far(MFD tail) — 모듈 함수 `mfd_far_cost_to_go`로 위임(P-CENT 천장 이식용 승격)."""
        return mfd_far_cost_to_go(self.cfg, state)

    def _predict(
        self,
        state: TrafficState,
        control: ControlAction,
        forecast: list[DemandStep],
        abort_above: Optional[float] = None,
        depth_override: Optional[int] = None,
        previous: Optional[ControlAction] = None,
    ) -> tuple[list[TrafficState], float]:
        from src.simulation.coupling import run_coupled_interval

        s = state.copy()
        states: list[TrafficState] = []
        total_ttt = 0.0
        # leader full rollout: horizon + leader_value_depth 만큼(V 포함). _predict은 leader 전용이라
        # follower myopia에 영향 없음. depth=0이면 기존과 동일.
        depth = self.cfg.mpc.horizon_steps + max(0, int(getattr(self.cfg.mpc, "leader_value_depth", 0)))
        if depth_override is not None:
            depth = max(1, int(depth_override))
        # BOX-WALK(2026-07-17, 사용자 설계 3차): 기존 rollout은 solve된 control을 horizon에
        # 고정 — METER-BOX 하에선 m_prev가 낮을 때 박스 밖 모든 후보(N_UF* 3300이든
        # 6000이든)의 1스텝 실현값이 동일해 V가 구분 불가("박스 끝 너머가 안 보임").
        # 실측(200_w): ③ 리더는 회복기에 6000 전량방류, box 리더는 intent 3190~3374 고정
        # → 방류 3000 고착 → ramp 큐 2.4배 → −29.78%. walk ON이면 rollout의 2번째
        # interval부터 metering을 후보 intent(N_UF*) 방향으로 스텝당 램프별 ±R씩 전진시켜
        # 다중스텝 도달을 채점에 반영한다. 후보별로 목표가 달라 V가 다시 구분된다.
        # 가드: incumbent/PFO probe(leader=None)는 N_UF_star=0(follower L4244)이라 walk하면
        # 0으로 끌려가 오염 → N_UF_star>0일 때만. 기본 False=비트동일.
        _walk_r = getattr(self.cfg.mpc, "seg13_meter_box_veh_h", None)
        _do_walk = (
            bool(getattr(self.cfg.mpc, "leader_rollout_box_walk", False))
            and _walk_r is not None
            and bool(control.ramp_metering)
            and float(getattr(control, "N_UF_star", 0.0)) > 0.0
        )
        if _do_walk:
            _r_dn = float(_walk_r)
            _bu_w = getattr(self.cfg.mpc, "seg13_meter_box_up_veh_h", None)
            _r_up = float(_bu_w) if _bu_w is not None else _r_dn
            _net = self.cfg.network
            _caps = {r: float(_net.ramp_capacity_veh_h[r]) for r in control.ramp_metering}
            _target_total = float(control.N_UF_star)
        # BOX-WALK-VG(2026-07-17, 사용자 지적 "vsl도 green도 점진 탐색해야"): VSL·green도
        # rollout에 다중스텝 이동을 모델링. metering과 달리 후보가 목표를 안 주므로
        # **끝 지속(edge persistence)**: 이번 solve가 이동 한계 끝(VSL ±R_v, green ±trust)
        # 까지 밀었으면 "더 가고 싶었다"로 보고 rollout에서 같은 방향·같은 속도로 전진
        # (전역 한계에서 정지). 내부 정착(끝 미달)이면 수렴으로 보고 고정 유지.
        _vg_moves: list = []
        _gtot = 0.0
        if bool(getattr(self.cfg.mpc, "leader_rollout_box_walk_vg", False)) and previous is not None:
            _rv = getattr(self.cfg.mpc, "seg13_vsl_box_kmh", None)
            if _rv is not None:
                _rv = float(_rv)
                _vmin = float(min(self.cfg.freeway_follower.vsl_set))
                _vmax = float(max(self.cfg.freeway_follower.vsl_set))
                for _key, _v in control.vsl.items():
                    if "__seg" not in _key:
                        continue
                    _pv = previous.vsl.get(_key)
                    if _pv is None:
                        continue
                    _d0 = float(_v) - float(_pv)
                    if _d0 >= _rv - 1.0e-6:
                        _vg_moves.append(("vsl", _key, _rv, _vmin, _vmax))
                    elif _d0 <= -(_rv - 1.0e-6):
                        _vg_moves.append(("vsl", _key, -_rv, _vmin, _vmax))
            _gt = getattr(getattr(self, "nash_solver", None),
                          "signal_marginal_price_trust_sec", None)
            if _gt:
                _gt = float(_gt)
                _gtot = float(self.cfg.network.effective_green_total)
                _gmin = float(getattr(self.cfg.network, "green_min", 20.0))
                for _key, _g in control.green_times.items():
                    if not _key.endswith("_p1"):
                        continue
                    _pg = previous.green_times.get(_key)
                    if _pg is None:
                        continue
                    _d0 = float(_g) - float(_pg)
                    if _d0 >= _gt - 1.0e-6:
                        _vg_moves.append(("green", _key, _gt, _gmin, _gtot - _gmin))
                    elif _d0 <= -(_gt - 1.0e-6):
                        _vg_moves.append(("green", _key, -_gt, _gmin, _gtot - _gmin))
        if _do_walk or _vg_moves:
            import copy as _copy
            # 얕은 사본 + 변형할 dict만 교체 — 원본(commit 후보 control)은 불변.
            # 주의: ramp만 교체하고 vsl/green을 원본 dict째 만지면 커밋 후보가 오염된다.
            _ctrl_w = _copy.copy(control)
            if _do_walk:
                _ctrl_w.ramp_metering = dict(control.ramp_metering)
            if _vg_moves:
                _ctrl_w.vsl = dict(control.vsl)
                _ctrl_w.green_times = dict(control.green_times)
            control = _ctrl_w
        for _k, demand in enumerate(forecast[:depth]):
            if _vg_moves and _k >= 1:
                for _kind, _key, _rate, _lo, _hi in _vg_moves:
                    if _kind == "vsl":
                        control.vsl[_key] = min(max(
                            float(control.vsl[_key]) + _rate, _lo), _hi)
                    else:
                        _np1 = min(max(
                            float(control.green_times[_key]) + _rate, _lo), _hi)
                        control.green_times[_key] = _np1
                        _k2 = _key[:-3] + "_p2"
                        if _k2 in control.green_times:
                            # p1+p2 합 보존(사이클 예산 112 불변).
                            control.green_times[_k2] = _gtot - _np1
                # link 대표 VSL = min(seg) 재계산(plant fallback 정합).
                for _lnk in self.cfg.network.freeway_links:
                    _sv = [float(v) for k, v in control.vsl.items()
                           if k.startswith(_lnk + "__seg")]
                    if _sv:
                        control.vsl[_lnk] = min(_sv)
            if _do_walk and _k >= 1:
                _m = control.ramp_metering
                _gap = _target_total - sum(_m.values())
                if _gap > 1.0e-9:
                    # 여유 큰 램프부터 스텝당 ≤ R_up씩 올림(_scale_to deficit 루프와 동형).
                    for _r in sorted(_m, key=lambda x: _caps[x] - _m[x], reverse=True):
                        _step = min(_r_up, _caps[_r] - _m[_r], _gap)
                        if _step <= 0.0:
                            continue
                        _m[_r] += _step
                        _gap -= _step
                        if _gap <= 1.0e-9:
                            break
                elif _gap < -1.0e-9:
                    _need = -_gap
                    for _r in sorted(_m, key=lambda x: _m[x], reverse=True):
                        _step = min(_r_dn, _m[_r], _need)
                        if _step <= 0.0:
                            continue
                        _m[_r] -= _step
                        _need -= _step
                        if _need <= 1.0e-9:
                            break
            result = run_coupled_interval(s, control, demand, self.cfg)
            s.time_sec += self.cfg.simulation.control_interval
            total_ttt += result.freeway_ttt + result.urban_ttt
            states.append(s.copy())
            # OPT2: TTT는 비음 누적 + 모든 penalty/far ≥0 → 부분합이 incumbent를 넘으면 이
            # 후보의 최종 objective도 반드시 초과 = exact pruning(argmin 불변). inf로 즉시 기각.
            if abort_above is not None and total_ttt > abort_above:
                return states, float("inf")
        return states, float(total_ttt)
