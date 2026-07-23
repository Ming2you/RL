from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

from .state import ExperimentConfig, load_jsonish


@dataclass
class DemandStep:
    freeway_mainline: Dict[str, float]
    urban_boundary: Dict[str, float]
    ramp_arrival: Dict[str, float]
    incident_capacity_factor: float = 1.0
    freeway_lane_loss: Dict[str, Dict[int, float]] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    name: str
    urban_scale: float = 1.0
    freeway_scale: float = 1.0
    ramp_scale: float = 1.0
    incident_capacity_factor: float = 1.0
    required: bool = False
    # off-ramp split ratio를 이 시나리오에서만 덮어쓴다(None이면 NetworkConfig 0.06 유지).
    # capacity-drop 유발 무대를 만들기 위한 시나리오 한정 주입 — plant 보존식은 불변.
    off_ramp_split_ratio_override: Optional[Dict[str, float]] = None
    # boundary_out 유한 출구용량[veh/h]을 이 시나리오에서만 덮어쓴다(None이면 1600 유지).
    # heavy-transfer 시나리오에서만 cap을 낮춰 off-ramp 홍수가 urban을 포화시키게 한다(A″-4).
    boundary_out_capacity_override: Optional[float] = None
    # 시나리오 한정 boundary_in 게이트별 가중치(공간 skew). 적용 후 총 urban 유입은 baseline과
    # 같도록 renormalize해 skew 효과를 demand 크기와 분리한다. None이면 기존 gradient 유지.
    urban_boundary_weight_override: Optional[Dict[str, float]] = None
    urban_west_east_ratio: Optional[float] = None
    # 시변 skew 반전(2026-07-15): 지정 시 urban_west_east_ratio가 초기값, _late가 후기값이고
    # _reversal_sec 중심 ±_reversal_win/2 구간에서 선형 crossfade. offset/green×offset 가격
    # 활성화 무대(고정 skew면 offset 한 번 정해지고 불변 → 가격 0).
    urban_we_ratio_late: Optional[float] = None
    urban_we_reversal_sec: Optional[float] = None
    urban_we_reversal_win: float = 600.0
    freeway_lane_closures: list[Dict[str, float | str]] = field(default_factory=list)
    surge_start_sec: Optional[float] = None
    surge_peak_sec: Optional[float] = None
    surge_end_sec: Optional[float] = None
    surge_peak_scale: float = 1.0
    surge_recovery_scale: float = 1.0
    # 사다리꼴 펄스 프로파일(2026-07-12 승인, 청산형 설계): base 지대 → 램프업 →
    # 플래토(class scale 도달) → 램프다운 → base 복귀. pulse_start_sec가 None이면
    # 완전 휴면(기존 시나리오 비트 동일). 펄스 구성 시 내장 sine 피크파는 꺼서
    # 설계 곡선이 그대로 실현되게 한다(effective scale = base + (class−base)·pulse(t)).
    pulse_base_scale: Optional[float] = None
    pulse_start_sec: Optional[float] = None
    pulse_rampup_sec: float = 300.0
    pulse_plateau_sec: float = 900.0
    pulse_rampdown_sec: float = 300.0
    metadata: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, object]) -> "ScenarioConfig":
        known = {
            "urban_scale": float(raw.get("urban_scale", 1.0)),
            "freeway_scale": float(raw.get("freeway_scale", 1.0)),
            "ramp_scale": float(raw.get("ramp_scale", 1.0)),
            "incident_capacity_factor": float(raw.get("incident_capacity_factor", 1.0)),
            "required": bool(raw.get("required", False)),
        }
        raw_override = raw.get("off_ramp_split_ratio_override")
        if isinstance(raw_override, Mapping):
            known["off_ramp_split_ratio_override"] = {
                str(k): float(v) for k, v in raw_override.items()
            }
        cap_override = raw.get("boundary_out_capacity_override")
        if cap_override is not None:
            known["boundary_out_capacity_override"] = float(cap_override)
        weight_override = raw.get("urban_boundary_weight_override")
        if isinstance(weight_override, Mapping):
            known["urban_boundary_weight_override"] = {
                str(k): float(v) for k, v in weight_override.items()
            }
        west_east_ratio = raw.get("urban_west_east_ratio")
        if west_east_ratio is not None:
            known["urban_west_east_ratio"] = float(west_east_ratio)
        for _k in ("urban_we_ratio_late", "urban_we_reversal_sec", "urban_we_reversal_win"):
            if raw.get(_k) is not None:
                known[_k] = float(raw[_k])
        lane_closures = raw.get("freeway_lane_closures")
        if isinstance(lane_closures, list):
            known["freeway_lane_closures"] = [
                {
                    str(k): (str(v) if str(k) == "link" else float(v))
                    for k, v in closure.items()
                }
                for closure in lane_closures
                if isinstance(closure, Mapping)
            ]
        for key in ("surge_start_sec", "surge_peak_sec", "surge_end_sec"):
            value = raw.get(key)
            if value is not None:
                known[key] = float(value)
        known["surge_peak_scale"] = float(raw.get("surge_peak_scale", 1.0))
        known["surge_recovery_scale"] = float(raw.get("surge_recovery_scale", 1.0))
        for key in ("pulse_base_scale", "pulse_start_sec"):
            value = raw.get(key)
            if value is not None:
                known[key] = float(value)
        for key in ("pulse_rampup_sec", "pulse_plateau_sec", "pulse_rampdown_sec"):
            value = raw.get(key)
            if value is not None:
                known[key] = float(value)
        return cls(name=name, **known)


def load_scenarios(path: str | Path) -> Dict[str, ScenarioConfig]:
    raw = load_jsonish(path)
    scenarios = raw.get("scenarios", raw)
    return {
        name: ScenarioConfig.from_mapping(name, value)
        for name, value in scenarios.items()
    }


def merge_freeway_lane_loss(steps: list[DemandStep]) -> Dict[str, Dict[int, float]]:
    """Forecast 구간에서 발생하는 segment별 최대 incident lane loss를 합친다."""
    merged: Dict[str, Dict[int, float]] = {}
    for step in steps:
        for link, segment_losses in step.freeway_lane_loss.items():
            target = merged.setdefault(str(link), {})
            for segment, loss in segment_losses.items():
                segment_idx = int(segment)
                target[segment_idx] = max(
                    target.get(segment_idx, 0.0),
                    max(0.0, float(loss)),
                )
    return merged


def apply_scenario_network_overrides(
    cfg: ExperimentConfig, scenario: ScenarioConfig
) -> ExperimentConfig:
    """시나리오의 network 단위 override(off-ramp split)를 cfg에 반영한 새 cfg를 반환한다.

    off_ramp_split_ratio는 plant·모든 controller가 단일하게 cfg.network에서 읽으므로,
    여기서 한 번만 덮어쓰면 모든 사용처에 일관 적용된다. override가 None이면 cfg를 그대로
    반환(기존 0.06 동작 유지). 차량보존식·β합류 로직은 건드리지 않고 split·cap 값만 주입한다.
    boundary_out cap도 동일 패턴으로 시나리오 한정 주입한다(A″-4)."""
    network_updates: Dict[str, object] = {}
    override = scenario.off_ramp_split_ratio_override
    if override:
        merged = dict(cfg.network.off_ramp_split_ratio)
        merged.update({str(k): float(v) for k, v in override.items()})
        network_updates["off_ramp_split_ratio"] = merged
    if scenario.boundary_out_capacity_override is not None:
        network_updates["boundary_out_capacity_veh_h"] = float(
            scenario.boundary_out_capacity_override
        )
    if not network_updates:
        return cfg
    return cfg.with_updates({"network": network_updates})


class DemandProfile:
    """Deterministic demand generator with a mild peak wave."""

    def __init__(self, cfg: ExperimentConfig, scenario: ScenarioConfig):
        self.cfg = cfg
        self.scenario = scenario

    def _surge_scale(self, time_sec: float) -> float:
        # Medium surge는 기준 수요의 총 구성비를 유지하면서 모든 유입을 같은 배율로 증감한다.
        start = self.scenario.surge_start_sec
        peak_time = self.scenario.surge_peak_sec
        end = self.scenario.surge_end_sec
        if start is None or peak_time is None or end is None:
            return 1.0
        if not start < peak_time < end:
            raise ValueError("scenario surge timing must satisfy start < peak < end")
        peak_scale = max(0.0, self.scenario.surge_peak_scale)
        recovery_scale = max(0.0, self.scenario.surge_recovery_scale)
        if time_sec <= start:
            return 1.0
        if time_sec <= peak_time:
            fraction = (time_sec - start) / max(peak_time - start, 1.0e-9)
            return 1.0 + fraction * (peak_scale - 1.0)
        if time_sec <= end:
            fraction = (time_sec - peak_time) / max(end - peak_time, 1.0e-9)
            return peak_scale + fraction * (recovery_scale - peak_scale)
        return recovery_scale

    def _pulse_fraction(self, time_sec: float) -> Optional[float]:
        # 사다리꼴 펄스 위치 함수 pulse(t)∈[0,1]. base 지대 0, 램프 선형, 플래토 1.
        # 미구성(pulse_start_sec/base_scale None)이면 None → 기존 경로 비트 동일.
        start = self.scenario.pulse_start_sec
        base = self.scenario.pulse_base_scale
        if start is None or base is None:
            return None
        up = max(0.0, self.scenario.pulse_rampup_sec)
        plateau = max(0.0, self.scenario.pulse_plateau_sec)
        down = max(0.0, self.scenario.pulse_rampdown_sec)
        if time_sec <= start:
            return 0.0
        if time_sec <= start + up:
            return (time_sec - start) / max(up, 1.0e-9)
        if time_sec <= start + up + plateau:
            return 1.0
        if time_sec <= start + up + plateau + down:
            return 1.0 - (time_sec - start - up - plateau) / max(down, 1.0e-9)
        return 0.0

    def _effective_we_ratio(self, time_sec: float) -> Optional[float]:
        """시변 skew: _reversal_sec 중심 crossfade로 초기→후기 비율 보간. 미지정이면 고정값."""
        r0 = self.scenario.urban_west_east_ratio
        r1 = self.scenario.urban_we_ratio_late
        rev = self.scenario.urban_we_reversal_sec
        if r1 is None or rev is None or r0 is None:
            return r0
        win = max(float(self.scenario.urban_we_reversal_win), 1.0e-9)
        lo, hi = rev - win / 2.0, rev + win / 2.0
        if time_sec <= lo:
            return r0
        if time_sec >= hi:
            return r1
        f = (time_sec - lo) / win
        return r0 + (r1 - r0) * f

    def _active_lane_loss(self, time_sec: float) -> Dict[str, Dict[int, float]]:
        # 차로 폐쇄는 DemandStep에 넣어 plant와 MPC forecast가 같은 incident 상태를 보게 한다.
        lane_loss: Dict[str, Dict[int, float]] = {}
        for closure in self.scenario.freeway_lane_closures:
            link = str(closure.get("link", ""))
            segment = int(float(closure.get("segment", -1.0)))
            loss = max(0.0, float(closure.get("lane_loss", 0.0)))
            start = float(closure.get("start_sec", 0.0))
            end = float(closure.get("end_sec", float("inf")))
            if not link or segment < 0 or loss <= 0.0 or not start <= time_sec < end:
                continue
            lane_loss.setdefault(link, {})[segment] = max(
                lane_loss.get(link, {}).get(segment, 0.0),
                loss,
            )
        return lane_loss

    def at(self, time_sec: float) -> DemandStep:
        sim = self.cfg.simulation
        net = self.cfg.network
        x = time_sec / max(sim.T_total, 1.0)
        peak = 1.0 + 0.22 * math.sin(math.pi * min(max(x, 0.0), 1.0))

        surge = self._surge_scale(time_sec)
        pulse = self._pulse_fraction(time_sec)
        if pulse is None:
            urban_scale = self.scenario.urban_scale
            freeway_scale = self.scenario.freeway_scale
            ramp_scale = self.scenario.ramp_scale
        else:
            # 펄스 모드: class scale은 플래토 피크, base 지대는 pulse_base_scale.
            # 내장 sine 피크파는 꺼서(peak=1) 설계 사다리꼴이 그대로 실현되게 한다.
            peak = 1.0
            base = max(0.0, float(self.scenario.pulse_base_scale))
            urban_scale = base + (self.scenario.urban_scale - base) * pulse
            freeway_scale = base + (self.scenario.freeway_scale - base) * pulse
            ramp_scale = base + (self.scenario.ramp_scale - base) * pulse
        freeway_base = 1650.0 * freeway_scale * peak * surge
        ramp_base = 560.0 * ramp_scale * peak * surge
        urban_base = 500.0 * urban_scale * peak * surge

        freeway = {
            link: freeway_base * (1.0 + 0.05 * idx)
            for idx, link in enumerate(net.freeway_links)
        }
        urban = {}
        in_base = {
            link: urban_base * (1.0 + 0.10 * idx)
            for idx, link in enumerate(net.boundary_in_links)
        }
        weights = self.scenario.urban_boundary_weight_override
        if weights:
            # 게이트별 가중치 적용 후 총 유입 보존(renormalize) — skew를 demand 크기와 분리.
            weighted = {link: in_base[link] * float(weights.get(link, 1.0)) for link in in_base}
            base_total = sum(in_base.values())
            w_total = sum(weighted.values())
            renorm = (base_total / w_total) if w_total > 1.0e-9 else 1.0
            in_base = {link: weighted[link] * renorm for link in weighted}
        west_east_ratio = self._effective_we_ratio(time_sec)
        if west_east_ratio is not None:
            # 서/동 측면 진입의 합만 재배분하고 북측 진입과 전체 urban demand 합은 보존한다.
            if west_east_ratio <= 0.0:
                raise ValueError("urban_west_east_ratio must be positive")
            west_links = [link for link in ("in_A_left", "in_D_left") if link in in_base]
            east_links = [link for link in ("in_C_right", "in_F_right") if link in in_base]
            west_total = sum(in_base[link] for link in west_links)
            east_total = sum(in_base[link] for link in east_links)
            side_total = west_total + east_total
            target_west = side_total * west_east_ratio / (1.0 + west_east_ratio)
            target_east = side_total / (1.0 + west_east_ratio)
            west_factor = target_west / max(west_total, 1.0e-9)
            east_factor = target_east / max(east_total, 1.0e-9)
            for link in west_links:
                in_base[link] *= west_factor
            for link in east_links:
                in_base[link] *= east_factor
        urban.update(in_base)
        for idx, link in enumerate(net.boundary_out_links):
            urban[link] = urban_base * (0.82 + 0.08 * idx)
        ramp = {
            ramp_name: ramp_base * (1.0 + 0.05 * idx)
            for idx, ramp_name in enumerate(net.ramps)
        }
        return DemandStep(
            freeway_mainline=freeway,
            urban_boundary=urban,
            ramp_arrival=ramp,
            incident_capacity_factor=self.scenario.incident_capacity_factor,
            freeway_lane_loss=self._active_lane_loss(time_sec),
        )

    def horizon(self, start_time_sec: float, steps: int) -> list[DemandStep]:
        dt = self.cfg.simulation.control_interval
        return [self.at(start_time_sec + i * dt) for i in range(max(1, steps))]
