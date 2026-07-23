from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.models.state import ControlAction, ExperimentConfig


@dataclass(frozen=True)
class GridControlCandidate:
    label: str
    control: ControlAction
    stage: str
    scope: str = "global"
    axis: str = ""
    delta: float = 0.0


def _offset_delta(cycle: float, anchor: float, target: float) -> float:
    return float((target - anchor + 0.5 * cycle) % cycle - 0.5 * cycle)


def _bounded_green_values(cfg: ExperimentConfig, values: list[float]) -> list[float]:
    net = cfg.network
    out: list[float] = []
    for value in values:
        p1 = float(np.clip(value, net.green_min, net.green_max))
        p2 = net.effective_green_total - p1
        if p2 < net.green_min:
            p1 = net.effective_green_total - net.green_min
        if p2 > net.green_max:
            p1 = net.effective_green_total - net.green_max
        if not any(abs(p1 - existing) <= 1.0e-9 for existing in out):
            out.append(float(p1))
    return out


def _set_signal_green(control: ControlAction, cfg: ExperimentConfig, signal: str, p1_value: float) -> None:
    p1 = _bounded_green_values(cfg, [p1_value])[0]
    control.green_times[f"{signal}_p1"] = float(p1)
    control.green_times[f"{signal}_p2"] = float(cfg.network.effective_green_total - p1)


def _set_all_green(control: ControlAction, cfg: ExperimentConfig, p1_value: float) -> None:
    for signal in cfg.network.signals:
        _set_signal_green(control, cfg, signal, p1_value)


def _set_uniform_metering(control: ControlAction, cfg: ExperimentConfig, rate: float) -> None:
    net = cfg.network
    for ramp in net.ramps:
        lower = cfg.freeway_follower.ramp_metering_rate_min * net.ramp_capacity_veh_h[ramp]
        upper = cfg.freeway_follower.ramp_metering_rate_max * net.ramp_capacity_veh_h[ramp]
        control.ramp_metering[ramp] = float(np.clip(rate, lower, upper))


def _set_one_metering(control: ControlAction, cfg: ExperimentConfig, ramp: str, rate: float) -> None:
    net = cfg.network
    lower = cfg.freeway_follower.ramp_metering_rate_min * net.ramp_capacity_veh_h[ramp]
    upper = cfg.freeway_follower.ramp_metering_rate_max * net.ramp_capacity_veh_h[ramp]
    control.ramp_metering[ramp] = float(np.clip(rate, lower, upper))


def _set_uniform_vsl(control: ControlAction, cfg: ExperimentConfig, value: float) -> None:
    vsl_min = min(float(v) for v in cfg.freeway_follower.vsl_set)
    vsl_max = max(float(v) for v in cfg.freeway_follower.vsl_set)
    vsl = float(np.clip(value, vsl_min, vsl_max))
    for link in cfg.network.freeway_links:
        control.vsl[link] = vsl


def _set_one_vsl(control: ControlAction, cfg: ExperimentConfig, link: str, value: float) -> None:
    vsl_min = min(float(v) for v in cfg.freeway_follower.vsl_set)
    vsl_max = max(float(v) for v in cfg.freeway_follower.vsl_set)
    control.vsl[link] = float(np.clip(value, vsl_min, vsl_max))


def _apply_offset_pattern(
    control: ControlAction,
    cfg: ExperimentConfig,
    previous: ControlAction,
    multipliers: dict[str, float],
    delta_sec: float,
) -> None:
    net = cfg.network
    max_step = float(cfg.urban_follower.max_offset_step)
    cycle = max(float(net.cycle_length), 1.0e-9)
    for signal in net.signals:
        prev = float(previous.offsets.get(signal, 0.0))
        target = float(multipliers.get(signal, 0.0)) * float(delta_sec)
        step = float(np.clip(_offset_delta(cycle, prev, target), -max_step, max_step))
        control.offsets[signal] = float((prev + step) % cycle)


def _shift_one_offset(
    control: ControlAction,
    cfg: ExperimentConfig,
    previous: ControlAction,
    signal: str,
    delta_sec: float,
) -> None:
    net = cfg.network
    max_step = float(cfg.urban_follower.max_offset_step)
    prev = float(previous.offsets.get(signal, 0.0))
    step = float(np.clip(delta_sec, -max_step, max_step))
    control.offsets[signal] = float((prev + step) % max(float(net.cycle_length), 1.0e-9))


def _apply_authority(control: ControlAction, cfg: ExperimentConfig, authority: str) -> ControlAction:
    if authority == "wu":
        net = cfg.network
        control.N_P_star = 0.0
        control.N_UF_star = 0.0
        control.ramp_metering = {ramp: net.ramp_capacity_veh_h[ramp] for ramp in net.ramps}
        control.offsets = {signal: 0.0 for signal in net.signals}
        control.inflow_outflow_allocation = {}
        control.diagnostics["wu_green_vsl_only_ttt_authority"] = 1.0
    return control


def _control_key(control: ControlAction, cfg: ExperimentConfig) -> Tuple[float, ...]:
    net = cfg.network
    return tuple(
        round(v, 4)
        for v in (
            [control.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp]) for ramp in net.ramps]
            + [control.vsl.get(link, max(cfg.freeway_follower.vsl_set)) for link in net.freeway_links]
            + [control.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0) for signal in net.signals]
            + [control.offsets.get(signal, 0.0) % max(net.cycle_length, 1.0e-9) for signal in net.signals]
        )
    )


def _add(
    out: List[GridControlCandidate],
    seen: set[Tuple[float, ...]],
    cfg: ExperimentConfig,
    authority: str,
    label: str,
    stage: str,
    scope: str,
    control: ControlAction,
    axis: str = "",
    delta: float = 0.0,
) -> None:
    control = _apply_authority(control, cfg, authority)
    key = _control_key(control, cfg)
    if key in seen:
        return
    seen.add(key)
    out.append(GridControlCandidate(
        label=label,
        control=control,
        stage=stage,
        scope=scope,
        axis=axis,
        delta=float(delta),
    ))


def _cap_mean(cfg: ExperimentConfig) -> float:
    return float(np.mean([cfg.network.ramp_capacity_veh_h[ramp] for ramp in cfg.network.ramps]))


def _dense(cfg: ExperimentConfig) -> bool:
    return bool(getattr(cfg.mpc, "centralized_grid_dense", False))


def _global_metering_rates(cfg: ExperimentConfig) -> list[float]:
    cap = _cap_mean(cfg)
    if _dense(cfg):
        # tightness: 0.40~1.00을 0.05 간격(13레벨, 낮은 metering까지 확장)
        fracs = [round(0.40 + 0.05 * i, 2) for i in range(13)]
    else:
        fracs = [1.0, 0.9, 0.8, 0.7333333333, 0.667, 0.6]
    return [cap * fraction for fraction in fracs]


def _local_metering_rates(cfg: ExperimentConfig, center: ControlAction) -> list[float]:
    net = cfg.network
    center_rate = float(np.mean([
        center.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp]) for ramp in net.ramps
    ]))
    return [center_rate + delta for delta in (-100.0, -50.0, 0.0, 50.0, 100.0)]


def _global_green_values(cfg: ExperimentConfig) -> list[float]:
    if _dense(cfg):
        # tightness: 20~92를 4 간격(19레벨)
        return _bounded_green_values(cfg, [20.0 + 4.0 * i for i in range(19)])
    return _bounded_green_values(cfg, [20.0, 32.0, 44.0, 56.0, 68.0, 80.0, 92.0])


def _local_green_values(cfg: ExperimentConfig, center: ControlAction) -> list[float]:
    net = cfg.network
    seed = float(np.mean([
        center.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0)
        for signal in net.signals
    ]))
    values = [seed - 6.0, seed, seed + 6.0]
    return _bounded_green_values(cfg, values)


def _local_signal_green_values(cfg: ExperimentConfig, center: ControlAction, signal: str) -> list[float]:
    net = cfg.network
    seed = center.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0)
    return _bounded_green_values(cfg, [seed - 6.0, seed, seed + 6.0])


def _offset_patterns() -> dict[str, dict[str, float]]:
    return {
        "lower_east": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 0.0, "F": 2.0},
        "lower_west": {"A": 3.0, "B": 2.0, "C": 1.0, "D": 2.0, "F": 0.0},
        "top_east": {"A": 0.0, "B": 1.0, "C": 2.0, "D": 1.0, "F": 2.0},
        "top_west": {"A": 2.0, "B": 1.0, "C": 0.0, "D": 2.0, "F": 1.0},
    }


def _apply_axis_delta(
    control: ControlAction,
    cfg: ExperimentConfig,
    previous: ControlAction,
    axis: str,
    delta: float,
) -> None:
    net = cfg.network
    if axis == "rm_all":
        current = float(np.mean([
            control.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp])
            for ramp in net.ramps
        ]))
        _set_uniform_metering(control, cfg, current + delta)
        return
    if axis.startswith("rm:"):
        ramp = axis.split(":", 1)[1]
        current = control.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp])
        _set_one_metering(control, cfg, ramp, current + delta)
        return
    if axis == "green_all":
        current = float(np.mean([
            control.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0)
            for signal in net.signals
        ]))
        _set_all_green(control, cfg, current + delta)
        return
    if axis.startswith("green:"):
        signal = axis.split(":", 1)[1]
        current = control.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0)
        _set_signal_green(control, cfg, signal, current + delta)
        return
    if axis == "vsl_all":
        current = float(np.mean([
            control.vsl.get(link, max(cfg.freeway_follower.vsl_set))
            for link in net.freeway_links
        ]))
        _set_uniform_vsl(control, cfg, current + delta)
        return
    if axis.startswith("vsl:"):
        link = axis.split(":", 1)[1]
        current = control.vsl.get(link, max(cfg.freeway_follower.vsl_set))
        _set_one_vsl(control, cfg, link, current + delta)
        return
    if axis.startswith("offset:"):
        signal = axis.split(":", 1)[1]
        _shift_one_offset(control, cfg, previous, signal, delta)
        return
    raise ValueError(f"Unknown sensitivity axis: {axis}")


def sensitivity_probe_candidates(
    cfg: ExperimentConfig,
    previous: ControlAction,
    center: ControlAction,
    authority: str = "proposed",
) -> list[GridControlCandidate]:
    """Finite-difference probes around the previous selected solution."""
    out: list[GridControlCandidate] = []
    seen: set[Tuple[float, ...]] = set()
    axes: list[tuple[str, float]] = [("green_all", 6.0), ("vsl_all", 10.0)]
    axes.extend((f"green:{signal}", 6.0) for signal in cfg.network.signals)
    axes.extend((f"vsl:{link}", 10.0) for link in cfg.network.freeway_links)
    if authority != "wu":
        axes.append(("rm_all", 100.0))
        axes.extend((f"rm:{ramp}", 100.0) for ramp in cfg.network.ramps)
        axes.extend((f"offset:{signal}", 5.0) for signal in cfg.network.signals)

    for axis, step in axes:
        for sign in (-1.0, 1.0):
            delta = sign * step
            cand = center.copy()
            _apply_axis_delta(cand, cfg, previous, axis, delta)
            _add(
                out,
                seen,
                cfg,
                authority,
                f"sens_probe_{axis}_{delta:+.0f}",
                "sensitivity_probe",
                "local",
                cand,
                axis=axis,
                delta=delta,
            )
    return out


def sensitivity_direction_candidates(
    cfg: ExperimentConfig,
    previous: ControlAction,
    center: ControlAction,
    probe_scores: list[tuple[GridControlCandidate, float]],
    authority: str = "proposed",
    max_axes: int = 4,
    base_objective: float | None = None,
) -> list[GridControlCandidate]:
    """Create finite-difference descent candidates from evaluated probes.

    The probe stage evaluates one-axis positive/negative perturbations.  This
    stage turns those finite differences into local descent directions, then
    tests one-step, two-step, and small multi-axis combinations around the
    incumbent grid solution.
    """
    out: list[GridControlCandidate] = []
    seen: set[Tuple[float, ...]] = set()
    base_key = _control_key(center, cfg)
    by_axis: dict[str, list[tuple[GridControlCandidate, float]]] = {}
    for candidate, objective in probe_scores:
        if candidate.axis:
            by_axis.setdefault(candidate.axis, []).append((candidate, float(objective)))

    directions: list[tuple[float, str, float, float]] = []
    for axis, entries in by_axis.items():
        usable = [
            (candidate.delta, objective)
            for candidate, objective in entries
            if abs(candidate.delta) > 1.0e-12 and _control_key(candidate.control, cfg) != base_key
        ]
        if not usable:
            continue
        by_delta = {float(delta): float(objective) for delta, objective in usable}
        positives = [(delta, objective) for delta, objective in usable if delta > 0.0]
        negatives = [(delta, objective) for delta, objective in usable if delta < 0.0]
        best_delta, best_obj = min(usable, key=lambda item: item[1])
        slope = 0.0
        direction_delta = best_delta
        if positives and negatives:
            pos_delta, pos_obj = min(positives, key=lambda item: abs(item[0]))
            neg_delta, neg_obj = min(negatives, key=lambda item: abs(item[0]))
            denom = max(pos_delta - neg_delta, 1.0e-9)
            slope = (pos_obj - neg_obj) / denom
            step = min(abs(pos_delta), abs(neg_delta))
            direction_delta = -np.sign(slope) * step if abs(slope) > 1.0e-12 else best_delta
            best_obj = by_delta.get(float(direction_delta), best_obj)
        if base_objective is not None and np.isfinite(base_objective):
            benefit = float(base_objective) - float(best_obj)
            if benefit <= 1.0e-9:
                continue
        else:
            benefit = -float(best_obj)
        directions.append((-benefit, axis, float(direction_delta), float(slope)))
    directions.sort(key=lambda item: item[0])

    # leader 축(target_net_inflow 등)은 coordinator projection으로 probe 단계에서만 탐색되고,
    # 이 generic follower-control 섭동(_apply_axis_delta)으로는 재투영 불가하다(ValueError).
    # direction/combo 단계가 그 축을 perturbation하지 않도록 적용 가능한 축만 남긴다.
    def _axis_applicable(axis: str, delta: float) -> bool:
        try:
            _apply_axis_delta(center.copy(), cfg, previous, axis, delta)
            return True
        except ValueError:
            return False

    selected = [item for item in directions if _axis_applicable(item[1], item[2])][:max_axes]

    for rank, (_score, axis, delta, _slope) in enumerate(selected, start=1):
        cand = center.copy()
        _apply_axis_delta(cand, cfg, previous, axis, delta)
        _add(
            out,
            seen,
            cfg,
            authority,
            f"sens_dir_axis{rank}_{axis}_{delta:+.0f}",
            "sensitivity_direction",
            "local",
            cand,
            axis=axis,
            delta=delta,
        )
        cand2 = center.copy()
        _apply_axis_delta(cand2, cfg, previous, axis, 2.0 * delta)
        _add(
            out,
            seen,
            cfg,
            authority,
            f"sens_dir_axis{rank}_{axis}_{2.0 * delta:+.0f}",
            "sensitivity_direction",
            "local",
            cand2,
            axis=axis,
            delta=2.0 * delta,
        )

    for count in range(2, len(selected) + 1):
        cand = center.copy()
        labels: list[str] = []
        for _score, axis, delta, _slope in selected[:count]:
            _apply_axis_delta(cand, cfg, previous, axis, delta)
            labels.append(f"{axis}{delta:+.0f}")
        _add(
            out,
            seen,
            cfg,
            authority,
            f"sens_dir_combo{count}_{'_'.join(labels)}",
            "sensitivity_direction",
            "local",
            cand,
        )

    return out


def structured_grid_candidates(
    cfg: ExperimentConfig,
    previous: ControlAction,
    center: ControlAction,
    authority: str = "proposed",
    stage: str = "coarse",
    scope: str = "global",
) -> list[GridControlCandidate]:
    """Build broad periodic grid or previous-solution trust-region candidates.

    `scope="global"` is used at the start and at periodic refresh points.  It
    explores the physical feasible ranges with topology-aware packages.
    `scope="local"` is used between refreshes and explores a trust region around
    the previously selected solution.
    """
    if authority not in {"proposed", "wu"}:
        raise ValueError(f"Unknown structured grid authority: {authority}")
    if scope not in {"global", "local"}:
        raise ValueError(f"Unknown structured grid scope: {scope}")

    out: list[GridControlCandidate] = []
    seen: set[Tuple[float, ...]] = set()
    net = cfg.network

    def base_copy() -> ControlAction:
        return center.copy()

    _add(out, seen, cfg, authority, "previous", stage, scope, previous.copy())
    _add(out, seen, cfg, authority, "no_control", stage, scope, ControlAction.uncontrolled(cfg))
    _add(out, seen, cfg, authority, "center", stage, scope, center.copy())

    green_values = _global_green_values(cfg) if scope == "global" else _local_green_values(cfg, center)
    if stage == "fine" and scope == "global":
        green_values = _local_green_values(cfg, center)

    for p1 in green_values:
        cand = base_copy()
        _set_all_green(cand, cfg, p1)
        _add(out, seen, cfg, authority, f"green_uniform_{p1:.0f}", stage, scope, cand)

    for signal in net.signals:
        signal_values = green_values if scope == "global" else _local_signal_green_values(cfg, center, signal)
        for p1 in signal_values:
            cand = base_copy()
            _set_signal_green(cand, cfg, signal, p1)
            _add(out, seen, cfg, authority, f"green_{signal}_{p1:.0f}", stage, scope, cand)

    if "D" in net.signals and "F" in net.signals:
        d_values = green_values
        f_values = green_values
        if scope == "local" or stage == "fine":
            d0 = center.green_times.get("D_p1", net.effective_green_total / 2.0)
            f0 = center.green_times.get("F_p1", net.effective_green_total / 2.0)
            d_values = _bounded_green_values(cfg, [d0 - 6.0, d0, d0 + 6.0])
            f_values = _bounded_green_values(cfg, [f0 - 6.0, f0, f0 + 6.0])
        for d_p1 in d_values:
            for f_p1 in f_values:
                cand = base_copy()
                _set_signal_green(cand, cfg, "D", d_p1)
                _set_signal_green(cand, cfg, "F", f_p1)
                _add(out, seen, cfg, authority, f"green_df_D{d_p1:.0f}_F{f_p1:.0f}", stage, scope, cand)

    if authority != "wu":
        rm_rates = _global_metering_rates(cfg) if scope == "global" else _local_metering_rates(cfg, center)
        if stage == "fine" and scope == "global":
            rm_rates = _local_metering_rates(cfg, center)
        for rate in rm_rates:
            cand = base_copy()
            _set_uniform_metering(cand, cfg, rate)
            _add(out, seen, cfg, authority, f"rm_uniform_{rate:.0f}", stage, scope, cand)
        single_ramp_rates = rm_rates if scope == "global" else rm_rates[1:4]
        for ramp in net.ramps:
            for rate in single_ramp_rates:
                cand = base_copy()
                _set_one_metering(cand, cfg, ramp, rate)
                _add(out, seen, cfg, authority, f"rm_{ramp}_{rate:.0f}", stage, scope, cand)

    if scope == "global":
        # tightness dense: 40~100을 5 간격(13레벨), 낮은 VSL까지 확장
        vsl_values = [40.0 + 5.0 * i for i in range(13)] if _dense(cfg) else [100.0, 90.0, 80.0, 70.0, 60.0]
    else:
        vsl_values = []
    if scope == "local":
        for link in net.freeway_links:
            current = center.vsl.get(link, max(cfg.freeway_follower.vsl_set))
            vsl_values.extend([current - 10.0, current, current + 10.0])
        vsl_values = sorted({float(np.clip(v, min(cfg.freeway_follower.vsl_set), max(cfg.freeway_follower.vsl_set))) for v in vsl_values})
    for vsl in vsl_values:
        cand = base_copy()
        _set_uniform_vsl(cand, cfg, vsl)
        _add(out, seen, cfg, authority, f"vsl_uniform_{vsl:.0f}", stage, scope, cand)
    for link in net.freeway_links:
        for vsl in vsl_values:
            cand = base_copy()
            _set_one_vsl(cand, cfg, link, vsl)
            _add(out, seen, cfg, authority, f"vsl_{link}_{vsl:.0f}", stage, scope, cand)

    if authority != "wu":
        offset_deltas = [5.0, 10.0, 15.0] if scope == "global" else [5.0, 10.0]
        for pattern_name, pattern in _offset_patterns().items():
            for delta in offset_deltas:
                cand = base_copy()
                _apply_offset_pattern(cand, cfg, previous, pattern, delta)
                _add(out, seen, cfg, authority, f"offset_{pattern_name}_{delta:.0f}", stage, scope, cand)
        for signal in net.signals:
            for delta in offset_deltas:
                for sign in (-1.0, 1.0):
                    cand = base_copy()
                    _shift_one_offset(cand, cfg, previous, signal, sign * delta)
                    _add(out, seen, cfg, authority, f"offset_{signal}_{sign * delta:.0f}", stage, scope, cand)

    if authority != "wu":
        # Structured interactions: not a full Cartesian grid, but broad enough
        # that RM/green synergy is not hidden behind one-variable probes.
        rm_rates = _global_metering_rates(cfg) if scope == "global" else _local_metering_rates(cfg, center)
        combo_green = _global_green_values(cfg) if scope == "global" else _local_green_values(cfg, center)
        for rate in rm_rates:
            for p1 in combo_green:
                cand = base_copy()
                _set_uniform_metering(cand, cfg, rate)
                _set_all_green(cand, cfg, p1)
                _add(out, seen, cfg, authority, f"combo_rm{rate:.0f}_uniform_green{p1:.0f}", stage, scope, cand)
        if "D" in net.signals and "F" in net.signals:
            if scope == "global":
                d_values = combo_green
                f_values = combo_green
                rm_for_df = [rm_rates[0], rm_rates[3]]
            else:
                d0 = center.green_times.get("D_p1", net.effective_green_total / 2.0)
                f0 = center.green_times.get("F_p1", net.effective_green_total / 2.0)
                d_values = _bounded_green_values(cfg, [d0 - 6.0, d0, d0 + 6.0])
                f_values = _bounded_green_values(cfg, [f0 - 6.0, f0, f0 + 6.0])
                center_rate = float(np.mean([
                    center.ramp_metering.get(ramp, net.ramp_capacity_veh_h[ramp]) for ramp in net.ramps
                ]))
                rm_for_df = [center_rate - 50.0, center_rate, center_rate + 50.0]
            for rate in rm_for_df:
                for d_p1 in d_values:
                    for f_p1 in f_values:
                        cand = base_copy()
                        _set_uniform_metering(cand, cfg, rate)
                        _set_signal_green(cand, cfg, "D", d_p1)
                        _set_signal_green(cand, cfg, "F", f_p1)
                        _add(
                            out,
                            seen,
                            cfg,
                            authority,
                            f"combo_rm{rate:.0f}_D{d_p1:.0f}_F{f_p1:.0f}",
                            stage,
                            scope,
                            cand,
                        )

    return out
