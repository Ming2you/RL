from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.models.urban_queue_model import (
    boundary_indices,
    ensure_urban_state,
    movement_balance_summary,
    movement_specs,
)


@dataclass(frozen=True)
class _PressureSummary:
    protected_accumulation_veh: float
    protected_reference_veh: float
    protected_signed_pressure: float
    protected_excess_pressure: float
    freeway_density_pressure: float
    freeway_density_max_ratio: float
    ramp_queue_pressure: float
    off_ramp_storage_pressure: float


class ClassicalHierarchicalController:
    """Stackelberg/Nash 해를 쓰지 않는 고전적 계층형 휴리스틱 기준 controller."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def _as_forecast(self, forecast: DemandStep | Iterable[DemandStep]) -> list[DemandStep]:
        if isinstance(forecast, DemandStep):
            return [forecast]
        steps = list(forecast)
        if not steps:
            raise ValueError("ClassicalHierarchicalController requires at least one demand step.")
        return steps[: max(1, int(self.cfg.mpc.horizon_steps))]

    def _link_density_pressure(self, state: TrafficState) -> tuple[Dict[str, float], float, float]:
        # Spec 4.3의 freeway 압력을 최적화 없이 현재 밀도 초과율로 근사한다.
        net = self.cfg.network
        link_pressure: Dict[str, float] = {}
        all_pressures: list[float] = []
        max_ratio = 0.0
        denom = max(net.rho_max - net.rho_crit, 1.0e-9)
        for link in net.freeway_links:
            values = state.freeway_density.get(link, [])
            if not values:
                link_pressure[link] = 0.0
                continue
            pressures = [max(0.0, (float(rho) - net.rho_crit) / denom) for rho in values]
            ratios = [max(0.0, float(rho) / max(net.rho_crit, 1.0e-9)) for rho in values]
            link_pressure[link] = float(np.clip(np.mean(pressures), 0.0, 1.0))
            all_pressures.extend(pressures)
            max_ratio = max(max_ratio, max(ratios))
        global_pressure = float(np.clip(np.mean(all_pressures), 0.0, 1.0)) if all_pressures else 0.0
        return link_pressure, global_pressure, float(max_ratio)

    def _off_ramp_storage_pressure(self, state: TrafficState) -> tuple[Dict[str, float], float]:
        # off-ramp 저장공간 압력은 가용공간(cap-S)을 cap으로 정규화한 현재 점유율이다.
        net = self.cfg.network
        by_off_ramp: Dict[str, float] = {}
        values: list[float] = []
        for off_ramp in net.off_ramps:
            storage_link = net.off_ramp_storage_link.get(off_ramp, "")
            cap = max(0.0, float(net.urban_link_storage_veh.get(storage_link, 0.0)))
            if cap <= 1.0e-9:
                by_off_ramp[off_ramp] = 0.0
                continue
            occupied = max(0.0, cap - state.urban_link_storage.get(storage_link, cap))
            pressure = float(np.clip(occupied / cap, 0.0, 1.0))
            by_off_ramp[off_ramp] = pressure
            values.append(pressure)
        return by_off_ramp, float(np.mean(values)) if values else 0.0

    def _pressure_summary(
        self,
        state: TrafficState,
        link_density_pressure: Mapping[str, float],
        off_ramp_pressure: Mapping[str, float],
    ) -> _PressureSummary:
        # 보호영역 압력은 N_P_crit 대비 현재 protected accumulation의 signed error로 둔다.
        net = self.cfg.network
        protected = float(state.protected_accumulation_veh(net))
        reference = max(float(self.cfg.leader.N_P_crit_veh), 1.0e-9)
        signed = float(np.clip((protected - reference) / reference, -1.0, 2.0))
        ramp_ratios = [
            max(0.0, state.ramp_queue.get(ramp, 0.0)) / max(net.ramp_queue_max_veh, 1.0e-9)
            for ramp in net.ramps
        ]
        return _PressureSummary(
            protected_accumulation_veh=protected,
            protected_reference_veh=reference,
            protected_signed_pressure=signed,
            protected_excess_pressure=max(0.0, signed),
            freeway_density_pressure=float(np.mean(list(link_density_pressure.values())))
            if link_density_pressure else 0.0,
            freeway_density_max_ratio=max(
                [
                    max(
                        [float(rho) / max(net.rho_crit, 1.0e-9) for rho in state.freeway_density.get(link, [])]
                        or [0.0]
                    )
                    for link in net.freeway_links
                ]
                or [0.0]
            ),
            ramp_queue_pressure=float(np.clip(np.mean(ramp_ratios), 0.0, 1.0)) if ramp_ratios else 0.0,
            off_ramp_storage_pressure=float(np.mean(list(off_ramp_pressure.values())))
            if off_ramp_pressure else 0.0,
        )

    def _movement_forecast_arrivals(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
    ) -> Dict[str, float]:
        # 단위 변환: forecast 유량[veh/h]을 제어 간격 T_c_h의 차량 수[veh]로 바꿔 phase 압력에 더한다.
        net = self.cfg.network
        specs = movement_specs(self.cfg)
        dt_h = self.cfg.simulation.T_c_h
        onramp_by_movement = {
            movement: ramp
            for ramp, movements in net.on_ramp_to_movement.items()
            for movement in movements
        }
        arrivals: Dict[str, float] = {}
        for movement, spec in specs.items():
            kind = str(spec.get("kind", ""))
            beta = max(0.0, float(spec.get("beta", 1.0)))
            if kind == "boundary_in":
                origin = str(spec.get("origin", ""))
                arrivals[movement] = sum(
                    max(0.0, step.urban_boundary.get(origin, 0.0)) * beta * dt_h
                    for step in forecast
                )
            elif kind == "on_ramp":
                ramp = str(spec.get("ramp", onramp_by_movement.get(movement, "")))
                share = 1.0 / max(len(net.on_ramp_to_movement.get(ramp, [])), 1)
                arrivals[movement] = sum(
                    max(0.0, step.ramp_arrival.get(ramp, 0.0)) * share * dt_h
                    for step in forecast
                )
            elif kind == "off_ramp":
                off_ramp = str(spec.get("off_ramp", spec.get("origin", "")))
                link = net.off_ramp_from_freeway.get(off_ramp, "")
                current_flow = max(0.0, (state.freeway_flow.get(link, []) or [0.0])[-1])
                split = max(0.0, float(net.off_ramp_split_ratio.get(off_ramp, 0.0)))
                arrivals[movement] = current_flow * split * beta * dt_h * len(forecast)
        return arrivals

    def _movement_weight(
        self,
        state: TrafficState,
        movement: str,
        spec: Mapping[str, object],
        pressure: _PressureSummary,
        link_density_pressure: Mapping[str, float],
        off_ramp_pressure: Mapping[str, float],
    ) -> float:
        # 보호영역이 과포화되면 진입 움직임은 낮추고 유출/off-ramp 움직임은 높이는 고전적 perimeter 논리다.
        net = self.cfg.network
        kind = str(spec.get("kind", ""))
        excess = min(1.0, pressure.protected_excess_pressure)
        deficit = min(1.0, max(0.0, -pressure.protected_signed_pressure))
        if kind == "boundary_in":
            return float(np.clip(1.0 - 0.70 * excess + 0.25 * deficit, 0.20, 1.40))
        if kind == "boundary_out":
            return float(np.clip(1.0 + 0.65 * excess, 0.50, 1.80))
        if kind == "off_ramp":
            off_ramp = str(spec.get("off_ramp", spec.get("origin", "")))
            link = net.off_ramp_from_freeway.get(off_ramp, "")
            return float(np.clip(
                1.0
                + 0.75 * link_density_pressure.get(link, pressure.freeway_density_pressure)
                + 0.65 * off_ramp_pressure.get(off_ramp, 0.0)
                + 0.30 * excess,
                0.60,
                2.20,
            ))
        if kind == "on_ramp":
            ramp = str(spec.get("ramp", ""))
            if not ramp:
                for candidate, movements in net.on_ramp_to_movement.items():
                    if movement in movements:
                        ramp = candidate
                        break
            link = net.ramp_to_freeway.get(ramp, "")
            queue_ratio = float(np.clip(
                self._onramp_queue_veh(state=state, ramp=ramp, movement=movement)
                / max(net.ramp_queue_max_veh, 1.0e-9),
                0.0,
                1.5,
            ))
            density = link_density_pressure.get(link, pressure.freeway_density_pressure)
            return float(np.clip(1.0 + 0.45 * queue_ratio - 0.45 * density + 0.15 * deficit, 0.35, 1.70))
        return float(np.clip(1.0 + 0.20 * excess, 0.70, 1.40))

    def _onramp_queue_veh(
        self,
        state: Optional[TrafficState],
        ramp: str,
        movement: str | None = None,
    ) -> float:
        net = self.cfg.network
        if state is None:
            return 0.0
        total = max(0.0, state.ramp_queue.get(ramp, 0.0))
        movements = [movement] if movement else net.on_ramp_to_movement.get(ramp, [])
        for item in movements:
            total += max(0.0, state.urban_movement_queue.get(item, 0.0))
        return float(total)

    def _repair_green_pair(self, p1_raw: float) -> tuple[float, float, float]:
        # cycle repair: p1+p2+lost_time=cycle_length를 보존하면서 green min/max를 만족시킨다.
        net = self.cfg.network
        total = max(0.0, float(net.effective_green_total))
        p1 = float(np.clip(p1_raw, net.green_min, net.green_max))
        p2 = total - p1
        if p2 < net.green_min:
            p2 = float(net.green_min)
            p1 = total - p2
        if p2 > net.green_max:
            p2 = float(net.green_max)
            p1 = total - p2
        p1 = float(np.clip(p1, 0.0, total))
        p2 = float(np.clip(total - p1, 0.0, total))
        residual = abs(p1 + p2 + net.lost_time - net.cycle_length)
        return p1, p2, float(residual)

    def _green_times(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        previous: ControlAction,
        pressure: _PressureSummary,
        link_density_pressure: Mapping[str, float],
        off_ramp_pressure: Mapping[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        # local pressure split: 각 신호의 두 phase queue+forecast 압력 비율로 green을 배분한다.
        net = self.cfg.network
        specs = movement_specs(self.cfg)
        arrivals = self._movement_forecast_arrivals(state, forecast)
        green: Dict[str, float] = {}
        diag: Dict[str, float] = {}
        repair_count = 0.0
        max_residual = 0.0

        for signal in net.signals:
            phase_scores = {"p1": 0.0, "p2": 0.0}
            for movement, spec in specs.items():
                phase = str(spec.get("phase", ""))
                if not phase.startswith(f"{signal}_"):
                    continue
                phase_id = phase.rsplit("_", 1)[-1]
                if phase_id not in phase_scores:
                    continue
                queue = max(0.0, state.urban_movement_queue.get(movement, 0.0))
                arrival = max(0.0, arrivals.get(movement, 0.0))
                weight = self._movement_weight(
                    state,
                    movement,
                    spec,
                    pressure,
                    link_density_pressure,
                    off_ramp_pressure,
                )
                phase_scores[phase_id] += (queue + arrival) * weight

            score_sum = phase_scores["p1"] + phase_scores["p2"]
            if score_sum <= 1.0e-9:
                prev_p1 = previous.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0)
                ratio = prev_p1 / max(net.effective_green_total, 1.0e-9)
            else:
                ratio = phase_scores["p1"] / score_sum
            p1_raw = net.effective_green_total * float(np.clip(ratio, 0.0, 1.0))
            p1, p2, residual = self._repair_green_pair(p1_raw)
            repair_count += float(abs(p1 - p1_raw) > 1.0e-9 or residual > 1.0e-9)
            max_residual = max(max_residual, residual)
            green[f"{signal}_p1"] = p1
            green[f"{signal}_p2"] = p2
            diag[f"classical_green_score_{signal}_p1"] = float(phase_scores["p1"])
            diag[f"classical_green_score_{signal}_p2"] = float(phase_scores["p2"])
            diag[f"classical_green_{signal}_p1_sec"] = p1
            diag[f"classical_green_{signal}_p2_sec"] = p2

        diag["classical_green_repair_count"] = repair_count
        diag["classical_green_cycle_residual_sec"] = max_residual
        return green, diag

    @staticmethod
    def _allocate_to_target(target: float, upper: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # ramp별 upper bound를 넘지 않도록 weighted water filling으로 총 방출 목표를 분배한다.
        release = np.zeros_like(upper, dtype=float)
        remaining = float(np.clip(target, 0.0, float(np.sum(upper))))
        active = upper > 1.0e-9
        while remaining > 1.0e-9 and bool(np.any(active)):
            w = np.where(active, np.maximum(weights, 1.0e-9), 0.0)
            w_sum = float(np.sum(w))
            if w_sum <= 1.0e-9:
                break
            proposed = remaining * w / w_sum
            spare = np.maximum(0.0, upper - release)
            capped = active & (proposed >= spare - 1.0e-9)
            if not bool(np.any(capped)):
                release += proposed
                remaining = 0.0
                break
            add = np.where(capped, spare, 0.0)
            release += add
            remaining -= float(np.sum(add))
            active = active & ~capped
        return np.minimum(np.maximum(release, 0.0), upper)

    def _ramp_metering(
        self,
        state: TrafficState,
        forecast: list[DemandStep],
        pressure: _PressureSummary,
        link_density_pressure: Mapping[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        # 단위 변환: ramp/on-ramp 대기 차량[veh]을 T_c_h로 나누어 방출 가능 유량[veh/h]에 반영한다.
        net = self.cfg.network
        fc = self.cfg.freeway_follower
        dt_h = max(self.cfg.simulation.T_c_h, 1.0e-9)
        incident_factor = min(1.0, min(getattr(step, "incident_capacity_factor", 1.0) for step in forecast))
        upper: list[float] = []
        weights: list[float] = []
        queue_ratios: list[float] = []
        arrival_rates: list[float] = []
        diag: Dict[str, float] = {}

        for ramp in net.ramps:
            cap = max(0.0, net.ramp_capacity_veh_h[ramp] * incident_factor)
            arrival_rate = sum(max(0.0, step.ramp_arrival.get(ramp, 0.0)) for step in forecast) / max(
                len(forecast),
                1,
            )
            onramp_queue = self._onramp_queue_veh(state, ramp)
            available_rate = arrival_rate + onramp_queue / dt_h
            link = net.ramp_to_freeway.get(ramp, "")
            density = link_density_pressure.get(link, pressure.freeway_density_pressure)
            queue_ratio = float(np.clip(onramp_queue / max(net.ramp_queue_max_veh, 1.0e-9), 0.0, 1.5))
            ramp_upper = min(cap, max(0.0, available_rate))
            upper.append(ramp_upper)
            arrival_norm = arrival_rate / max(cap, 1.0e-9)
            weights.append(max(0.05, (0.25 + 2.00 * queue_ratio + 0.40 * arrival_norm) * max(0.15, 1.0 - 0.70 * density)))
            queue_ratios.append(queue_ratio)
            arrival_rates.append(arrival_rate)
            diag[f"classical_metering_upper_{ramp}_veh_h"] = float(ramp_upper)
            diag[f"classical_ramp_queue_pressure_{ramp}"] = queue_ratio
            diag[f"classical_ramp_density_pressure_{ramp}"] = float(density)

        upper_arr = np.asarray(upper, dtype=float)
        weight_arr = np.asarray(weights, dtype=float)
        min_frac = float(np.clip(fc.ramp_metering_rate_min, 0.0, 1.0))
        max_frac = float(np.clip(fc.ramp_metering_rate_max, min_frac, 1.0))
        release_fraction = float(np.clip(
            1.0 - 0.55 * pressure.freeway_density_pressure + 0.35 * pressure.ramp_queue_pressure,
            min_frac,
            max_frac,
        ))
        target = float(np.sum(upper_arr) * release_fraction)
        release = self._allocate_to_target(target, upper_arr, weight_arr)

        for idx, ramp in enumerate(net.ramps):
            cap = max(0.0, net.ramp_capacity_veh_h[ramp] * incident_factor)
            queue_ratio = queue_ratios[idx]
            if queue_ratio > 0.75:
                guard_frac = min(max_frac, min_frac + 0.75 * (queue_ratio - 0.75) / 0.75)
                release[idx] = max(release[idx], min(upper_arr[idx], cap * guard_frac))

        metering = {
            ramp: float(np.clip(release[idx], 0.0, net.ramp_capacity_veh_h[ramp]))
            for idx, ramp in enumerate(net.ramps)
        }
        diag["classical_metering_target_veh_h"] = target
        diag["classical_metering_release_fraction"] = release_fraction
        diag["classical_metering_total_veh_h"] = float(sum(metering.values()))
        diag["classical_metering_available_total_veh_h"] = float(np.sum(upper_arr))
        for ramp, value in metering.items():
            diag[f"classical_metering_{ramp}_veh_h"] = float(value)
        return metering, diag

    def _supervisor_targets(
        self,
        pressure: _PressureSummary,
        forecast: list[DemandStep],
        metering: Mapping[str, float],
    ) -> tuple[float, float, Dict[str, float]]:
        """고전적 위계 controller의 feedback target을 기록한다.

        후보 탐색 없이 MFD perimeter feedback과 ramp pressure rule의 결과를 leader target과
        같은 단위로 남긴다. `N_P_star`는 horizon 순 net-inflow[veh], `N_UF_star`는 ramp
        release rate[veh/h]이다.
        """
        lc = self.cfg.leader
        sim = self.cfg.simulation
        horizon_h = max(sim.T_c_h * max(len(forecast), 1), 1.0e-9)
        feedback_h = max(float(lc.N_P_feedback_horizon_h), horizon_h, 1.0e-9)
        flow_limit = max(0.0, float(lc.N_P_feedback_flow_limit_veh_h))
        err_veh = pressure.protected_accumulation_veh - pressure.protected_reference_veh
        avg_boundary_demand = sum(
            sum(max(0.0, value) for value in step.urban_boundary.values())
            for step in forecast
        ) / max(len(forecast), 1)
        if err_veh > 0.0:
            q_p_cmd = -min(flow_limit, err_veh / feedback_h)
        else:
            q_p_cmd = min(flow_limit, avg_boundary_demand, 0.35 * (-err_veh) / feedback_h)

        np_low, np_high = sorted(float(v) for v in lc.N_P_star_range[:2])
        nuf_low, nuf_high = sorted(float(v) for v in lc.N_UF_star_range[:2])
        n_p_star = float(np.clip(q_p_cmd * horizon_h, np_low, np_high))
        n_uf_star = float(np.clip(sum(max(0.0, float(v)) for v in metering.values()), nuf_low, nuf_high))
        diag = {
            "classical_supervisor_nP_veh": pressure.protected_accumulation_veh,
            "classical_supervisor_nP_crit_veh": pressure.protected_reference_veh,
            "classical_supervisor_nP_error_veh": err_veh,
            "classical_supervisor_qP_cmd_veh_h": float(q_p_cmd),
            "classical_supervisor_N_P_star_veh": n_p_star,
            "classical_supervisor_N_UF_star_veh_h": n_uf_star,
        }
        return n_p_star, n_uf_star, diag

    def _target_vsl(self, density_ratio: float) -> float:
        # density threshold rule: rho/rho_crit 구간을 discrete VSL set의 가까운 값으로 사상한다.
        vsl_set = sorted(float(v) for v in self.cfg.freeway_follower.vsl_set)
        if density_ratio >= 1.50:
            raw = min(vsl_set)
        elif density_ratio >= 1.30:
            raw = 60.0
        elif density_ratio >= 1.15:
            raw = 70.0
        elif density_ratio >= 1.00:
            raw = 80.0
        elif density_ratio >= 0.90:
            raw = 90.0
        else:
            raw = max(vsl_set)
        return min(vsl_set, key=lambda value: (abs(value - raw), value))

    def _bounded_vsl(self, previous: float, target: float) -> float:
        # VSL step repair: 이전 표시속도에서 max_vsl_step 안에 있는 이산 후보만 허용한다.
        fc = self.cfg.freeway_follower
        vsl_set = sorted(float(v) for v in fc.vsl_set)
        feasible = [value for value in vsl_set if abs(value - previous) <= fc.max_vsl_step + 1.0e-9]
        if not feasible:
            nearest = min(vsl_set, key=lambda value: abs(value - previous))
            feasible = [nearest]
        return float(min(feasible, key=lambda value: (abs(value - target), value)))

    def _vsl(self, state: TrafficState, previous: ControlAction) -> tuple[Dict[str, float], Dict[str, float]]:
        # segment별 threshold VSL을 만들고 link 값은 plant/CSV 호환을 위해 segment 최솟값으로 둔다.
        net = self.cfg.network
        vsl: Dict[str, float] = {}
        diag: Dict[str, float] = {}
        lowered = 0.0
        for link in net.freeway_links:
            segment_values: list[float] = []
            densities = state.freeway_density.get(link, [])
            for idx, rho in enumerate(densities):
                ratio = max(0.0, float(rho) / max(net.rho_crit, 1.0e-9))
                target = self._target_vsl(ratio)
                prev = segment_vsl(previous, link, idx, self.cfg)
                value = self._bounded_vsl(prev, target)
                vsl[f"{link}__seg{idx}"] = value
                segment_values.append(value)
                lowered += float(value < max(self.cfg.freeway_follower.vsl_set) - 1.0e-9)
                diag[f"classical_vsl_target_{link}_seg{idx}_km_h"] = target
                diag[f"classical_vsl_{link}_seg{idx}_km_h"] = value
            vsl[link] = float(min(segment_values)) if segment_values else max(self.cfg.freeway_follower.vsl_set)
            diag[f"classical_vsl_{link}_km_h"] = vsl[link]
        diag["classical_vsl_lowered_segments"] = lowered
        return vsl, diag

    def decide(
        self,
        state: TrafficState,
        forecast: DemandStep | Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
    ) -> ControlAction:
        # 계층 구조: 압력 산정 -> green/metering/VSL 휴리스틱 -> 고정 offset, allocation 비사용.
        working = state.copy()
        ensure_urban_state(working, self.cfg)
        steps = self._as_forecast(forecast)
        previous = previous_control or ControlAction.uncontrolled(self.cfg)
        link_density_pressure, density_pressure, density_max_ratio = self._link_density_pressure(working)
        off_ramp_pressure, off_ramp_mean = self._off_ramp_storage_pressure(working)
        pressure = self._pressure_summary(working, link_density_pressure, off_ramp_pressure)
        pressure = _PressureSummary(
            protected_accumulation_veh=pressure.protected_accumulation_veh,
            protected_reference_veh=pressure.protected_reference_veh,
            protected_signed_pressure=pressure.protected_signed_pressure,
            protected_excess_pressure=pressure.protected_excess_pressure,
            freeway_density_pressure=density_pressure,
            freeway_density_max_ratio=density_max_ratio,
            ramp_queue_pressure=pressure.ramp_queue_pressure,
            off_ramp_storage_pressure=off_ramp_mean,
        )

        green, green_diag = self._green_times(
            working,
            steps,
            previous,
            pressure,
            link_density_pressure,
            off_ramp_pressure,
        )
        metering, metering_diag = self._ramp_metering(working, steps, pressure, link_density_pressure)
        n_p_star, n_uf_star, supervisor_diag = self._supervisor_targets(pressure, steps, metering)
        vsl, vsl_diag = self._vsl(working, previous)
        cycle = max(self.cfg.network.cycle_length, 1.0e-9)
        offsets = {
            signal: float(previous.offsets.get(signal, 0.0) % cycle)
            for signal in self.cfg.network.signals
        }

        balance = movement_balance_summary(
            working,
            self.cfg,
            saturation_fraction=self.cfg.evaluation.boundary_degenerate_saturation_fraction,
            degenerate_ratio=self.cfg.evaluation.boundary_degenerate_ratio,
            eps=self.cfg.evaluation.eps,
        )
        # 진단값은 controller 식별, 압력, actuator 결정을 모두 남겨 비교 runner에서 추적할 수 있게 한다.
        diagnostics: Dict[str, float] = {
            "controller_classical_hierarchical": 1.0,
            "classical_stackelberg_used": 0.0,
            "classical_follower_nash_used": 0.0,
            "classical_direct_supervisor_used": 1.0,
            "classical_inflow_outflow_allocation_used": 0.0,
            "classical_offset_optimization_used": 0.0,
            "classical_offset_fixed_previous": 1.0,
            "classical_protected_accumulation_veh": pressure.protected_accumulation_veh,
            "classical_protected_reference_veh": pressure.protected_reference_veh,
            "classical_protected_signed_pressure": pressure.protected_signed_pressure,
            "classical_protected_excess_pressure": pressure.protected_excess_pressure,
            "classical_freeway_density_pressure": pressure.freeway_density_pressure,
            "classical_freeway_density_max_ratio": pressure.freeway_density_max_ratio,
            "classical_ramp_queue_pressure": pressure.ramp_queue_pressure,
            "classical_off_ramp_storage_pressure": pressure.off_ramp_storage_pressure,
            "B_in": float(balance.get("B_in", 0.0)),
            "B_out": float(balance.get("B_out", 0.0)),
        }
        diagnostics.update(boundary_indices(working.boundary_queue.values(), self.cfg.network.boundary_queue_max_veh))
        diagnostics.update(supervisor_diag)
        diagnostics.update(green_diag)
        diagnostics.update(metering_diag)
        diagnostics.update(vsl_diag)

        return ControlAction(
            N_P_star=n_p_star,
            N_UF_star=n_uf_star,
            ramp_metering=metering,
            vsl=vsl,
            green_times=green,
            offsets=offsets,
            inflow_outflow_allocation={},
            infeasibility={
                "classical_green_cycle_residual_sec": diagnostics["classical_green_cycle_residual_sec"],
                "classical_metering_capacity_residual": 0.0,
            },
            diagnostics=diagnostics,
        )
