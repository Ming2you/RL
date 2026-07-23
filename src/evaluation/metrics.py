from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np

from src.models.state import EvaluationResult, ExperimentConfig
from src.models.urban_queue_model import boundary_indices, movement_balance_summary


def improvement_rate(
    baseline: float,
    proposed: float,
    direction: str = "lower_is_better",
    eps: float = 1.0e-9,
) -> float:
    if direction == "higher_is_better":
        return float(100.0 * (proposed - baseline) / max(abs(baseline), eps))
    return float(100.0 * (baseline - proposed) / max(abs(baseline), eps))


def boundary_cv(values: Iterable[float], eps: float = 1.0e-9) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    return float(np.std(arr) / max(mean, eps)) if mean > eps else 0.0


def _balance_from_rows(rows: list, cfg: ExperimentConfig) -> Dict[str, float] | None:
    """boundary balance를 최종 스냅샷이 아니라 run 전체의 제어가능 구간에서 시간 집계한다.

    피크가 지나 큐가 비워진 최종 상태로 판정하면 "잘 비운 성공"이 degenerate로
    무효 처리된다. 따라서 interval별 B를 모아 degenerate가 아닌(부하가 걸린)
    구간 평균으로 B_in/B_out을 내고, 제어가능 구간 비율이 기준 미만일 때만
    run 전체를 degenerate로 본다."""
    balance_rows = [r for r in rows if "B_in" in r and "boundary_balance_degenerate" in r]
    if not balance_rows:
        return None
    controllable = [r for r in balance_rows if float(r.get("boundary_balance_degenerate", 0.0)) < 0.5]
    fraction = len(controllable) / len(balance_rows)
    degenerate = fraction < cfg.evaluation.boundary_controllable_min_fraction
    source = controllable if controllable else balance_rows
    mean = lambda key: float(np.mean([float(r.get(key, 0.0)) for r in source]))

    def load_weighted(key: str, load_key: str) -> float:
        """B는 스케일 불변이라 노이즈 수준 큐의 B가 실제 대기 큐의 B와 같은 무게로
        평균되면 안 된다 — 공평성은 실제 대기량에 비례해 중요하므로 부하 가중 평균."""
        values = [float(r.get(key, 0.0)) for r in source]
        weights = [max(0.0, float(r.get(load_key, 0.0))) for r in source]
        total = sum(weights)
        if total <= 0.0:
            return float(np.mean(values))
        return float(sum(v * w for v, w in zip(values, weights)) / total)

    return {
        "B_in": load_weighted("B_in", "boundary_in_load_veh"),
        "B_out": load_weighted("B_out", "boundary_out_load_veh"),
        "boundary_empty_ratio": mean("boundary_empty_ratio"),
        "boundary_saturation_ratio": mean("boundary_saturation_ratio"),
        "boundary_in_empty_ratio": mean("boundary_in_empty_ratio"),
        "boundary_out_empty_ratio": mean("boundary_out_empty_ratio"),
        "boundary_in_saturation_ratio": mean("boundary_in_saturation_ratio"),
        "boundary_out_saturation_ratio": mean("boundary_out_saturation_ratio"),
        "boundary_balance_degenerate": 1.0 if degenerate else 0.0,
        "boundary_balance_controllable": 0.0 if degenerate else 1.0,
        "boundary_controllable_fraction": float(fraction),
    }


def _net_inflow_tracking_from_rows(rows: list, cfg: ExperimentConfig) -> float | None:
    """net inflow 추적오차를 feedback horizon 창의 실현 dN_P/dt 기준으로 계산한다.

    기존 row 지표(realized = gross 서비스 유량)는 처리량 자체를 벌점화하는 stale
    정의다. feedback 법칙 (N_P_star−N_P)/horizon 은 누적 변화율 목표이므로, 같은
    horizon 창에서 (N_P(t)−N_P(t−H))/H 와 창 평균 목표를 비교한다."""
    # cycle 위상 plant에서 endpoint N_P는 신호 cycle(120s)과 interval(180s)의
    # 비정합으로 앨리어싱된다 — interval 평균 N_P가 있으면 그것을 쓴다.
    accs = [r.get("urban_accumulation_mean_veh", r.get("urban_accumulation_veh")) for r in rows]
    targets = [r.get("net_inflow_target") for r in rows]
    if len(rows) < 2 or any(v is None for v in accs) or any(v is None for v in targets):
        return None
    t_c_h = max(cfg.simulation.T_c_h, 1.0e-9)
    window = max(1, int(round(cfg.leader.N_P_feedback_horizon_h / t_c_h)))
    flow_limit = max(0.0, float(cfg.leader.N_P_feedback_flow_limit_veh_h))
    errors = []
    clipped_errors = []
    for i in range(window, len(rows)):
        realized = (float(accs[i]) - float(accs[i - window])) / (window * t_c_h)
        target = float(np.mean([float(t) for t in targets[i - window + 1: i + 1]]))
        # 목표가 feedback flow limit에 클립된 창은 정의상 달성 불가능한 요청
        # (deep over/undershoot 구간, infeasibility는 누적오차 진단에 따로 남음)이라
        # 추적 품질 채점에서 제외한다. 단 전 구간이 클립이면 그 값으로 폴백.
        if flow_limit > 0.0 and abs(target) >= flow_limit - 1.0e-6:
            clipped_errors.append(abs(realized - target))
            continue
        errors.append(abs(realized - target))
    if not errors:
        errors = clipped_errors
    if not errors:
        return None
    return float(np.mean(errors))


def summarize_run(result: Mapping[str, Any], cfg: ExperimentConfig) -> Dict[str, float]:
    rows = result.get("run_rows", [])
    final_state = result.get("final_state")
    boundary_values = final_state.boundary_queue.values() if final_state is not None else []
    balance = _balance_from_rows(list(rows), cfg)
    if balance is None:
        # run rows가 없으면(단위테스트 등) 기존처럼 최종 상태 스냅샷으로 계산한다.
        balance = (
            movement_balance_summary(
                final_state,
                cfg,
                saturation_fraction=cfg.evaluation.boundary_degenerate_saturation_fraction,
                degenerate_ratio=cfg.evaluation.boundary_degenerate_ratio,
                eps=cfg.evaluation.eps,
            )
            if final_state is not None
            else {
                "B_in": 0.0,
                "B_out": 0.0,
                "boundary_empty_ratio": 0.0,
                "boundary_saturation_ratio": 0.0,
                "boundary_in_empty_ratio": 0.0,
                "boundary_out_empty_ratio": 0.0,
                "boundary_in_saturation_ratio": 0.0,
                "boundary_out_saturation_ratio": 0.0,
                "boundary_balance_degenerate": 0.0,
                "boundary_balance_controllable": 1.0,
            }
        )
        balance.setdefault("boundary_controllable_fraction", 1.0 - float(balance.get("boundary_balance_degenerate", 0.0)))
    gross_net_inflow_errors = [
        r.get("urban_net_inflow_tracking_error_veh_h", r.get("net_inflow_tracking_error", 0.0))
        for r in rows
    ]
    tracking_error = _net_inflow_tracking_from_rows(list(rows), cfg)
    if tracking_error is None:
        tracking_error = float(np.mean(gross_net_inflow_errors)) if rows else 0.0
    accumulation_errors = [
        r.get("urban_accumulation_abs_error_veh", abs(r.get("urban_accumulation_error_veh", 0.0)))
        for r in rows
    ]
    return {
        "freeway_ttt": float(result.get("freeway_ttt", 0.0)),
        "urban_ttt": float(result.get("urban_ttt", 0.0)),
        "total_ttt": float(result.get("total_ttt", 0.0)),
        "mean_metering_error": float(np.mean([r.get("total_metering_error", 0.0) for r in rows])) if rows else 0.0,
        "max_metering_violation": float(np.max([r.get("total_metering_error", 0.0) for r in rows])) if rows else 0.0,
        "ramp_queue_overflow_duration": float(np.sum([r.get("ramp_queue_overflow_count", 0.0) > 0 for r in rows])) if rows else 0.0,
        "density_exceedance_duration": float(np.sum([r.get("density_exceedance_count", 0.0) > 0 for r in rows])) if rows else 0.0,
        "urban_net_inflow_tracking_error_veh_h": float(tracking_error),
        "net_inflow_tracking_error": float(tracking_error),
        # 구(gross 서비스 유량) 정의는 참고용 descriptive로만 남긴다.
        "urban_gross_service_net_inflow_error_veh_h": (
            float(np.mean(gross_net_inflow_errors)) if rows else 0.0
        ),
        "urban_accumulation_abs_error_veh": float(np.mean(accumulation_errors)) if rows else 0.0,
        "CV_boundary": boundary_cv(boundary_values, cfg.evaluation.eps),
        **balance,
        **(
            boundary_indices(final_state.boundary_queue.values(), cfg.network.boundary_queue_max_veh)
            if final_state is not None
            else {}
        ),
    }


def evaluate(
    baseline: Mapping[str, Any],
    proposed: Mapping[str, Any],
    cfg: ExperimentConfig,
) -> EvaluationResult:
    baseline_summary = summarize_run(baseline, cfg)
    proposed_summary = summarize_run(proposed, cfg)
    metric = cfg.evaluation.main_metric
    imp = improvement_rate(
        baseline_summary.get(metric, baseline_summary["total_ttt"]),
        proposed_summary.get(metric, proposed_summary["total_ttt"]),
        cfg.evaluation.main_metric_direction,
        cfg.evaluation.eps,
    )
    validation = validate_controls(baseline, proposed, cfg)
    passed = bool(imp >= cfg.evaluation.min_improvement_pct and all(v.get("pass", False) for v in validation.values()))
    metrics = {
        **{f"baseline_{k}": v for k, v in baseline_summary.items()},
        **{f"proposed_{k}": v for k, v in proposed_summary.items()},
        "improvement_pct": imp,
    }
    return EvaluationResult(metrics=metrics, improvement_pct=imp, passed=passed, control_validation=validation)


def validate_controls(
    baseline: Mapping[str, Any],
    proposed: Mapping[str, Any],
    cfg: ExperimentConfig,
) -> Dict[str, Dict[str, Any]]:
    rows = proposed.get("run_rows", [])
    controls = proposed.get("control_rows", [])
    baseline_summary = summarize_run(baseline, cfg)
    proposed_summary = summarize_run(proposed, cfg)
    vsl_values = [
        value for row in controls for key, value in row.items()
        if key.startswith("vsl_")
    ]
    vsl_active_steps = sum(
        any(float(row.get(f"vsl_{link}", max(cfg.freeway_follower.vsl_set))) < max(cfg.freeway_follower.vsl_set) - 0.5
            for link in cfg.network.freeway_links)
        for row in controls
    )
    metering_active_steps = sum(
        any(float(row.get(f"ramp_metering_{ramp}", cfg.network.ramp_capacity_veh_h[ramp]))
            < cfg.network.ramp_capacity_veh_h[ramp] - 1.0
            for ramp in cfg.network.ramps)
        for row in controls
    )
    vsl_set = set(float(v) for v in cfg.freeway_follower.vsl_set)
    vsl_ok = all(float(v) in vsl_set for v in vsl_values)
    vsl_change_violations = 0
    for prev, cur in zip(controls, controls[1:]):
        for link in cfg.network.freeway_links:
            keys = [f"vsl_{link}"] + [
                f"vsl_{link}_seg{i}"
                for i in range(cfg.network.freeway_segments_per_link)
            ]
            for key in keys:
                if key not in cur and key not in prev:
                    continue
                if abs(cur.get(key, 0.0) - prev.get(key, 0.0)) > cfg.freeway_follower.max_vsl_step + 1e-9:
                    vsl_change_violations += 1
    cycle_errors = []
    green_min_v = 0
    green_max_v = 0
    offset_v = 0
    offset_smooth_v = 0
    for row in controls:
        for signal in cfg.network.signals:
            p1 = row.get(f"green_{signal}_p1", 0.0)
            p2 = row.get(f"green_{signal}_p2", 0.0)
            cycle_errors.append(abs(p1 + p2 + cfg.network.lost_time - cfg.network.cycle_length))
            green_min_v += int(p1 < cfg.network.green_min - 1e-9 or p2 < cfg.network.green_min - 1e-9)
            green_max_v += int(p1 > cfg.network.green_max + 1e-9 or p2 > cfg.network.green_max + 1e-9)
            offset = row.get(f"offset_{signal}", 0.0)
            offset_v += int(offset < 0.0 or offset >= cfg.network.cycle_length)
    for prev, cur in zip(controls, controls[1:]):
        for signal in cfg.network.signals:
            delta = abs(cur.get(f"offset_{signal}", 0.0) - prev.get(f"offset_{signal}", 0.0))
            delta = min(delta, cfg.network.cycle_length - delta)
            if delta > cfg.urban_follower.max_offset_step + 1e-9:
                offset_smooth_v += 1
    mean_metering = proposed_summary["mean_metering_error"]
    boundary_balance_improvement = improvement_rate(
        baseline_summary["CV_boundary"],
        proposed_summary["CV_boundary"],
        "lower_is_better",
        cfg.evaluation.eps,
    )
    movement_balance_ok = (
        proposed_summary["B_in"] <= cfg.evaluation.eps_balance + 1.0e-9
        and proposed_summary["B_out"] <= cfg.evaluation.eps_balance + 1.0e-9
    )
    boundary_degenerate = proposed_summary.get("boundary_balance_degenerate", 0.0) > 0.5
    return {
        "ramp_metering": {
            "metering_active_steps": metering_active_steps,
            "mean_total_metering_error": mean_metering,
            "max_metering_violation": proposed_summary["max_metering_violation"],
            "ramp_queue_overflow_duration": proposed_summary["ramp_queue_overflow_duration"],
            "pass": mean_metering <= cfg.freeway_follower.eps_F
            and proposed_summary["ramp_queue_overflow_duration"] <= max(1.0, 0.05 * len(rows)),
        },
        "vsl": {
            "vsl_active_steps": vsl_active_steps,
            "vsl_feasible_rate": float(np.mean([float(v) in vsl_set for v in vsl_values])) if vsl_values else 1.0,
            "vsl_change_violation_count": vsl_change_violations,
            "density_exceedance_duration": proposed_summary["density_exceedance_duration"],
            "pass": vsl_ok and vsl_change_violations == 0
            and proposed_summary["density_exceedance_duration"] <= baseline_summary["density_exceedance_duration"] + 1.0,
        },
        "green_time": {
            "cycle_sum_error": max(cycle_errors) if cycle_errors else 0.0,
            "green_min_violation_count": green_min_v,
            "green_max_violation_count": green_max_v,
            "queue_overflow_count": proposed_summary.get("OverflowRatio_boundary", 0.0),
            "pass": (max(cycle_errors) if cycle_errors else 0.0) <= 1.0e-6
            and green_min_v == 0 and green_max_v == 0,
        },
        "offset": {
            "offset_range_violation_count": offset_v,
            "offset_smoothness_violation_count": offset_smooth_v,
            "corridor_delay_change": proposed_summary["urban_ttt"] - baseline_summary["urban_ttt"],
            "pass": offset_v == 0 and offset_smooth_v == 0,
        },
        "boundary_balance": {
            "B_in": proposed_summary["B_in"],
            "B_out": proposed_summary["B_out"],
            "eps_balance": cfg.evaluation.eps_balance,
            "CV_boundary": proposed_summary["CV_boundary"],
            "MaxMin_boundary": proposed_summary["MaxMin_boundary"],
            "OverflowRatio_boundary": proposed_summary["OverflowRatio_boundary"],
            "boundary_queue_balance_improvement": boundary_balance_improvement,
            "boundary_balance_degenerate": proposed_summary.get("boundary_balance_degenerate", 0.0),
            "boundary_balance_controllable": proposed_summary.get("boundary_balance_controllable", 1.0),
            "boundary_empty_ratio": proposed_summary.get("boundary_empty_ratio", 0.0),
            "boundary_saturation_ratio": proposed_summary.get("boundary_saturation_ratio", 0.0),
            "boundary_in_empty_ratio": proposed_summary.get("boundary_in_empty_ratio", 0.0),
            "boundary_out_empty_ratio": proposed_summary.get("boundary_out_empty_ratio", 0.0),
            "boundary_in_saturation_ratio": proposed_summary.get("boundary_in_saturation_ratio", 0.0),
            "boundary_out_saturation_ratio": proposed_summary.get("boundary_out_saturation_ratio", 0.0),
            "urban_net_inflow_tracking_error_veh_h": proposed_summary["urban_net_inflow_tracking_error_veh_h"],
            "urban_accumulation_abs_error_veh": proposed_summary["urban_accumulation_abs_error_veh"],
            "pass": movement_balance_ok
            and not boundary_degenerate
            and proposed_summary["OverflowRatio_boundary"] <= baseline_summary["OverflowRatio_boundary"] + 1e-9
            and proposed_summary["urban_net_inflow_tracking_error_veh_h"] <= cfg.urban_follower.eps_U + 1e-9,
        },
    }
