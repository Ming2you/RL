from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import numpy as np

from src.models.state import ExperimentConfig


@dataclass(frozen=True)
class GreenRepairResult:
    p1: float
    p2: float
    quantization_residual_sec: float
    repair_count: int


@dataclass(frozen=True)
class VSLRepairResult:
    value: float
    quantization_residual_km_h: float
    repair_count: int


GREEN_RESIDUAL_KEY = "green_quantization_residual_sec"
VSL_RESIDUAL_KEY = "vsl_quantization_residual_km_h"
GREEN_REPAIR_COUNT_KEY = "green_repair_count"
VSL_REPAIR_COUNT_KEY = "vsl_repair_count"


def accumulate_repair_diagnostics(
    diagnostics: MutableMapping[str, float],
    *,
    green: GreenRepairResult | None = None,
    vsl: VSLRepairResult | None = None,
) -> None:
    """Spec 17 diagnostics: residual은 최대값, repair count는 합계로 누적한다."""
    if green is not None:
        diagnostics[GREEN_RESIDUAL_KEY] = max(
            float(diagnostics.get(GREEN_RESIDUAL_KEY, 0.0)),
            float(green.quantization_residual_sec),
        )
        diagnostics[GREEN_REPAIR_COUNT_KEY] = (
            float(diagnostics.get(GREEN_REPAIR_COUNT_KEY, 0.0)) + float(green.repair_count)
        )
    if vsl is not None:
        diagnostics[VSL_RESIDUAL_KEY] = max(
            float(diagnostics.get(VSL_RESIDUAL_KEY, 0.0)),
            float(vsl.quantization_residual_km_h),
        )
        diagnostics[VSL_REPAIR_COUNT_KEY] = (
            float(diagnostics.get(VSL_REPAIR_COUNT_KEY, 0.0)) + float(vsl.repair_count)
        )


def merge_repair_diagnostics(
    diagnostics: MutableMapping[str, float],
    source: Mapping[str, float],
) -> None:
    """여러 agent diagnostics를 합칠 때 residual/count 의미를 보존한다."""
    for key in (GREEN_RESIDUAL_KEY, VSL_RESIDUAL_KEY):
        if key in source:
            diagnostics[key] = max(float(diagnostics.get(key, 0.0)), float(source.get(key, 0.0)))
    for key in (GREEN_REPAIR_COUNT_KEY, VSL_REPAIR_COUNT_KEY):
        if key in source:
            diagnostics[key] = float(diagnostics.get(key, 0.0)) + float(source.get(key, 0.0))


def _quantize(value: float, quantum: float, mode: str) -> float:
    quantum = max(float(quantum), 1.0e-9)
    ratio = float(value) / quantum
    if mode == "nearest":
        return float(round(ratio) * quantum)
    return float(math.floor(ratio + 1.0e-12) * quantum)


def _project_to_vsl_set(value: float, vsl_set: Iterable[float], mode: str) -> float:
    values = sorted(float(v) for v in vsl_set)
    if not values:
        raise ValueError("vsl_set must not be empty.")
    if mode == "floor":
        lower = [v for v in values if v <= value + 1.0e-9]
        return float(lower[-1] if lower else values[0])
    return float(min(values, key=lambda v: (abs(v - value), v)))


def repair_vsl_value(
    target: float,
    previous: float,
    cfg: ExperimentConfig,
) -> VSLRepairResult:
    """Spec 17.4 VSL: continuous target을 step 제약과 이산 vsl_set으로 투영한다."""
    fc = cfg.freeway_follower
    mpc = cfg.mpc
    vsl_set = sorted(float(v) for v in fc.vsl_set)
    minimum = min(vsl_set)
    maximum = max(vsl_set)
    repair_count = 0

    clipped = float(np.clip(target, minimum, maximum))
    repair_count += int(abs(clipped - target) > 1.0e-9)

    step_low = max(minimum, float(previous) - fc.max_vsl_step)
    step_high = min(maximum, float(previous) + fc.max_vsl_step)
    stepped = float(np.clip(clipped, step_low, step_high))
    repair_count += int(abs(stepped - clipped) > 1.0e-9)

    quantized = _quantize(stepped, mpc.relaxed_vsl_quantum_km_h, mpc.relaxed_rounding_mode)
    repair_count += int(abs(quantized - stepped) > 1.0e-9)
    projected = _project_to_vsl_set(quantized, vsl_set, mpc.relaxed_rounding_mode)
    feasible = [v for v in vsl_set if step_low - 1.0e-9 <= v <= step_high + 1.0e-9] or vsl_set
    repaired = float(min(feasible, key=lambda v: (abs(v - projected), v)))
    repair_count += int(abs(repaired - quantized) > 1.0e-9 or abs(repaired - projected) > 1.0e-9)
    return VSLRepairResult(
        value=repaired,
        quantization_residual_km_h=abs(float(target) - repaired),
        repair_count=repair_count,
    )


def repair_green_pair(
    p1_target: float,
    cfg: ExperimentConfig,
) -> GreenRepairResult:
    """Spec 17.4 green: p1 target을 양 phase bounds와 cycle equality에 맞게 복구한다."""
    net = cfg.network
    mpc = cfg.mpc
    total = float(net.effective_green_total)
    gmin = float(net.green_min)
    gmax = float(net.green_max)
    low = max(gmin, total - gmax)
    high = min(gmax, total - gmin)
    repair_count = 0

    if low > high:
        midpoint = total / 2.0
        p1 = float(np.clip(midpoint, gmin, gmax))
        p2 = float(np.clip(total - p1, gmin, gmax))
        return GreenRepairResult(
            p1=p1,
            p2=p2,
            quantization_residual_sec=abs(float(p1_target) - p1),
            repair_count=1,
        )

    clipped = float(np.clip(p1_target, low, high))
    repair_count += int(abs(clipped - p1_target) > 1.0e-9)
    quantized = _quantize(clipped, mpc.relaxed_green_quantum_sec, mpc.relaxed_rounding_mode)
    repair_count += int(abs(quantized - clipped) > 1.0e-9)
    p1 = float(np.clip(quantized, low, high))
    repair_count += int(abs(p1 - quantized) > 1.0e-9)

    # cycle equality를 먼저 정확히 맞춘 뒤, p2 bound 위반 시 p1을 반대 방향으로 재보정한다.
    p2 = total - p1
    if p2 < gmin:
        p2 = gmin
        p1 = total - p2
        repair_count += 1
    elif p2 > gmax:
        p2 = gmax
        p1 = total - p2
        repair_count += 1
    p1 = float(np.clip(p1, gmin, gmax))
    p2 = float(total - p1)
    if p2 < gmin - 1.0e-9 or p2 > gmax + 1.0e-9:
        p2 = float(np.clip(p2, gmin, gmax))
        p1 = float(np.clip(total - p2, gmin, gmax))
        repair_count += 1
    return GreenRepairResult(
        p1=float(p1),
        p2=float(total - p1),
        quantization_residual_sec=abs(float(p1_target) - float(p1)),
        repair_count=repair_count,
    )


def queue_pressure_green_target(p1_pressure: float, p2_pressure: float, cfg: ExperimentConfig) -> float:
    """두 phase pressure를 cycle 내 연속 split으로 변환하는 완화 solver 공통 휴리스틱."""
    total = float(cfg.network.effective_green_total)
    pressure_sum = max(0.0, float(p1_pressure)) + max(0.0, float(p2_pressure))
    ratio = max(0.0, float(p1_pressure)) / max(pressure_sum, 1.0e-9) if pressure_sum > 0.0 else 0.5
    return float(total * ratio)
