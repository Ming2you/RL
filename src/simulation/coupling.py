from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.models.demand import DemandStep
from src.models.metanet import compute_ramp_release_flows, freeway_substep
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    aggregate_urban_diagnostics,
    off_ramp_capacity_by_freeway_link,
    schedule_offramp_arrivals,
    sync_onramp_queues_from_freeway,
    sync_onramp_queues_to_freeway,
    urban_substep,
)


def _offramp_flow_from_diagnostics(
    fw_diag: Dict[str, float],
    cfg: ExperimentConfig,
    off_ramp: str,
) -> float:
    direct_key = f"offramp_flow_{off_ramp}"
    if direct_key in fw_diag:
        return max(0.0, float(fw_diag.get(direct_key, 0.0)))
    link = cfg.network.off_ramp_from_freeway[off_ramp]
    link_ratio_total = sum(
        ratio
        for candidate, ratio in cfg.network.off_ramp_split_ratio.items()
        if cfg.network.off_ramp_from_freeway.get(candidate) == link
    )
    share = cfg.network.off_ramp_split_ratio.get(off_ramp, 0.0) / max(link_ratio_total, 1.0e-9)
    return max(0.0, float(fw_diag.get(f"offramp_flow_{link}", 0.0))) * share


@dataclass
class CoupledStepResult:
    freeway_ttt: float
    urban_ttt: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _aggregate_freeway_diagnostics(
    rows: list[Dict[str, float]],
    interval_h: float | None = None,
    rows_per_cycle: int | None = None,
    queue_cap_veh: float | None = None,
) -> Dict[str, float]:
    """`freeway_substep` diagnostics를 control interval 기준으로 묶는다."""
    if not rows:
        return {}
    avg_keys = {
        "total_metering_flow",
        "total_metering_error",
        "total_no_meter_flow",
        "nuf_target_flow",
        "metering_over_release_flow",
        "metering_shortfall_flow",
        "total_ramp_queue_start_veh",
        "total_ramp_queue_end_veh",
        "mean_ramp_receiving_factor",
        "mean_segment_flow",
        "offramp_flow_total",
        "offramp_blocked_flow_total",
        "mainline_exit_flow_total",
    }
    out: Dict[str, float] = {}
    keys = set().union(*(row.keys() for row in rows))
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        if (
            key in avg_keys
            or key.startswith("offramp_flow_")
            or key.startswith("offramp_blocked_flow_")
            or key.startswith("lambda_eff_")
            or key.startswith("capacity_drop_lane_loss_")
            or key.startswith("offramp_occupancy_ratio_")
        ):
            out[key] = float(sum(values) / max(len(values), 1))
        elif key in {"metering_target_infeasible", "offramp_storage_binding", "capacity_drop_active"}:
            out[key] = float(max(values))
        else:
            out[key] = float(sum(values))
    # metering 추적 잔차는 interval 수준에서 재채점한다 — cycle 위상 plant에서 T_f(10s)
    # 창의 w_r 증감은 green 펄스로 진동하고, max(0,·)로 정류된 per-T_f 잔차 평균은
    # 펄스 버퍼링을 '잡아둠'으로 오인한다. interval(=신호 cycle 이상) 시야에서
    # 순증가만 잡으면 진짜 restrictive metering만 남는다.
    if interval_h is not None and "nuf_target_flow" in rows[0]:
        target = float(rows[0].get("nuf_target_flow", 0.0))
        mean_actual = float(out.get("total_metering_flow", 0.0))
        start_veh = float(rows[0].get("total_ramp_queue_start_veh", 0.0))
        end_veh = float(rows[-1].get("total_ramp_queue_end_veh", 0.0))
        # 성장은 cycle 정렬 시점끼리 비교한다 — interval(1.5 cycle) 경계는 cycle 위상의
        # 반대편을 표본해 펄스 진동이 성장으로 앨리어싱된다. 같은 위상(정확히 1 cycle
        # 간격)의 w_r 차이만 진짜 누적이다.
        if rows_per_cycle and len(rows) > rows_per_cycle:
            ref_veh = float(rows[-1 - rows_per_cycle].get("total_ramp_queue_end_veh", start_veh))
            # T_f_h = interval_h/len(rows) → 1 cycle = rows_per_cycle × T_f_h.
            cycle_h = interval_h * rows_per_cycle / max(len(rows), 1)
            growth_flow = max(0.0, end_veh - ref_veh) / max(cycle_h, 1.0e-9)
        else:
            growth_flow = max(0.0, end_veh - start_veh) / max(interval_h, 1.0e-9)
        # 큐가 상한에 고착되면 성장이 0이어도 명백한 holding이다 — 전액 채점.
        if queue_cap_veh is not None and end_veh >= 0.9 * queue_cap_veh:
            growth_flow = float("inf")
        over_release = max(0.0, mean_actual - target)
        shortfall = min(max(0.0, target - mean_actual), growth_flow)
        out["total_metering_error"] = over_release + shortfall
        out["metering_over_release_flow"] = float(over_release)
        out["metering_shortfall_flow"] = float(shortfall)
        out["total_ramp_queue_start_veh"] = start_veh
        out["total_ramp_queue_end_veh"] = end_veh
    return out


def _actual_ramp_release_flows(
    rows: list[Dict[str, float]],
    cfg: ExperimentConfig,
) -> Dict[str, float]:
    """이번 T_f 안에서 w_r에서 실제로 빠진 metering 차량만 freeway 유입으로 사용한다."""
    interval_h = cfg.simulation.T_f_h
    return {
        ramp: float(sum(
            row.get(f"ramp_metering_release_actual_{ramp}_veh", 0.0)
            for row in rows
        ) / max(interval_h, 1.0e-9))
        for ramp in cfg.network.ramps
    }


def _with_actual_ramp_diagnostics(
    requested_diag: Dict[str, float],
    actual_release: Dict[str, float],
) -> Dict[str, float]:
    diag = dict(requested_diag)
    actual_total = float(sum(actual_release.values()))
    diag["total_metering_flow"] = actual_total
    diag["total_no_meter_flow"] = float(max(
        requested_diag.get("total_no_meter_flow", 0.0),
        actual_total,
    ))
    return diag


def run_coupled_interval(
    state: TrafficState,
    control: ControlAction,
    demand: DemandStep,
    cfg: ExperimentConfig,
) -> CoupledStepResult:
    """Spec 3.4.3의 `T_c -> T_f -> T_u` nested order로 한 control interval을 전진한다."""
    sim = cfg.simulation
    sync_onramp_queues_from_freeway(state, cfg)

    freeway_ttt = 0.0
    urban_ttt = 0.0
    offramp_storage_ttt_moved = 0.0  # urban→freeway로 옮긴 램프 storage TTT 누적(보존 진단).
    freeway_rows: list[Dict[str, float]] = []
    urban_rows: list[Dict[str, float]] = []
    accepted_offramp = 0.0
    rejected_offramp = 0.0
    start_urban_step = int(round(state.time_sec / max(sim.T_u_sec, 1.0e-9)))

    for freeway_substep_index in range(sim.K_cf):
        # 2저수지 구조에서는 현재 ramp 저수지 w_r만 이번 T_f의 metering release 후보가 된다.
        sync_onramp_queues_to_freeway(state, cfg)
        ramp_release, ramp_diag = compute_ramp_release_flows(
            state,
            control,
            demand,
            cfg,
            include_current_arrivals=False,
        )

        # freeway 한 스텝 사이에 들어있는 urban substep들을 먼저 진행해 off-ramp storage 여유를 갱신한다.
        urban_rows_in_freeway_step: list[Dict[str, float]] = []
        for urban_offset in range(sim.K_fu):
            step_idx = start_urban_step + freeway_substep_index * sim.K_fu + urban_offset
            ur_ttt, ur_diag = urban_substep(
                state,
                control,
                demand,
                cfg,
                urban_step_index=step_idx,
                ramp_release_veh_h=ramp_release,
            )
            urban_ttt += ur_ttt
            # off-ramp 램프 storage 재귀속(design 2026-06-17): urban_substep이 urban_ttt에서
            # 제외한 램프 storage 점유 TTT를 freeway_ttt로 옮긴다(보존: 총합 불변, T_u_h 단위 일관).
            moved = float(ur_diag.get("offramp_storage_ttt", 0.0))
            freeway_ttt += moved
            offramp_storage_ttt_moved += moved
            urban_rows.append(ur_diag)
            urban_rows_in_freeway_step.append(ur_diag)
        sync_onramp_queues_to_freeway(state, cfg)

        actual_ramp_release = _actual_ramp_release_flows(urban_rows_in_freeway_step, cfg)
        actual_ramp_diag = _with_actual_ramp_diagnostics(ramp_diag, actual_ramp_release)

        offramp_capacity = off_ramp_capacity_by_freeway_link(
            state,
            cfg,
            interval_h=sim.T_f_h,
        )
        fw_ttt, fw_diag = freeway_substep(
            state,
            control,
            demand,
            cfg,
            offramp_capacity_veh_h=offramp_capacity,
            ramp_release_veh_h=actual_ramp_release,
            ramp_release_diagnostics=actual_ramp_diag,
            update_ramp_queues=False,
            include_ramp_queue_ttt=True,
        )
        freeway_ttt += fw_ttt
        freeway_rows.append(fw_diag)

        # freeway에서 빠져나온 off-ramp 차량은 다음 urban substep 이후 storage/buffer로 들어간다.
        next_urban_step = start_urban_step + (freeway_substep_index + 1) * sim.K_fu
        for off_ramp in cfg.network.off_ramps:
            flow = _offramp_flow_from_diagnostics(fw_diag, cfg, off_ramp)
            vehicles = flow * sim.T_f_h
            accepted, rejected = schedule_offramp_arrivals(state, cfg, off_ramp, vehicles, next_urban_step)
            accepted_offramp += accepted
            rejected_offramp += rejected

    diagnostics: Dict[str, Any] = {
        "coupling_nested_order_active": 1.0,
        "coupling_freeway_substeps": float(cfg.simulation.K_cf),
        "coupling_urban_substeps": float(cfg.simulation.K_cu),
        "coupling_onramp_sync_active": 1.0,
        "coupling_onramp_two_reservoir_active": 1.0,
        "coupling_offramp_storage_active": 1.0,
        "coupling_aggregate_urban_model": 0.0,
        "coupling_movement_urban_model": 1.0,
        "coupling_offramp_arrivals_accepted_veh": float(accepted_offramp),
        "coupling_offramp_arrivals_rejected_veh": float(rejected_offramp),
        "coupling_offramp_storage_ttt_moved_to_freeway": float(offramp_storage_ttt_moved),
    }
    fw_diag = _aggregate_freeway_diagnostics(
        freeway_rows,
        interval_h=sim.T_c_h,
        rows_per_cycle=max(1, int(round(cfg.network.cycle_length / sim.T_f_sec))),
        queue_cap_veh=cfg.network.ramp_queue_max_veh * max(len(cfg.network.ramps), 1),
    )
    ur_diag = aggregate_urban_diagnostics(urban_rows, cfg, control, interval_h=sim.T_c_h)
    diagnostics.update(fw_diag)
    diagnostics.update(ur_diag)
    return CoupledStepResult(
        freeway_ttt=float(freeway_ttt),
        urban_ttt=float(urban_ttt),
        diagnostics=diagnostics,
    )
