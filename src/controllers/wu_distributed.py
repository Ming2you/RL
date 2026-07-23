# Wu(2022) authority(green+VSL) 분산 컨트롤러 — WU-CD-F와 WU-MATCHED-STACKELBERG (spec 16.4~16.5)
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from src.models.demand import DemandStep
from src.controllers.relaxed_quantization import (
    accumulate_repair_diagnostics,
    queue_pressure_green_target,
    repair_green_pair,
    repair_vsl_value,
)
from src.models.metanet import compute_ramp_release_flows, effective_lane_profile, freeway_substep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.models.urban_queue_model import (
    _movement_capacity_flow,
    _phase_green_fraction,
    estimate_onramp_green_release_flows,
    estimate_onramp_reservoir_inflow,
    movement_specs,
    off_ramp_capacity_by_freeway_link,
)


@dataclass(frozen=True)
class WuLeaderAction:
    """WU-MATCHED-STACKELBERG leader action (spec 16.5) — 둘 다 [veh] 단위 누적 목표."""

    n_p_star: float
    n_f_star: float


@dataclass
class WuDecisionInfo:
    control: ControlAction
    iterations: int
    converged: bool
    coupling_residual: float
    solver_evaluations: int
    computation_time_sec: float
    leader_candidates: int = 0
    leader_selected: Optional[WuLeaderAction] = None
    leader_objective: float = 0.0


def _wu_fixed_control(cfg: ExperimentConfig) -> ControlAction:
    """Wu authority의 고정 요소 — offset 고정 0, metering=용량(no-metering 물리 유출),
    allocation 미사용(빈 dict → plant가 movement 포화유율 1400으로 fallback)."""
    return ControlAction.uncontrolled(cfg)


def _split_link_offramp_flow(
    cfg: ExperimentConfig,
    link: str,
    off_ramp: str,
    link_flow: float,
) -> float:
    net = cfg.network
    link_split = sum(
        ratio
        for candidate, ratio in net.off_ramp_split_ratio.items()
        if net.off_ramp_from_freeway.get(candidate) == link
    )
    this_split = float(net.off_ramp_split_ratio.get(off_ramp, 0.0))
    return max(0.0, float(link_flow)) * this_split / max(link_split, 1.0e-9)


def _offramp_flow_from_diagnostics(
    fw_diag: Mapping[str, float],
    cfg: ExperimentConfig,
    off_ramp: str,
) -> float:
    direct_key = f"offramp_flow_{off_ramp}"
    if direct_key in fw_diag:
        return max(0.0, float(fw_diag.get(direct_key, 0.0)))
    link = cfg.network.off_ramp_from_freeway.get(off_ramp, "")
    return _split_link_offramp_flow(cfg, link, off_ramp, float(fw_diag.get(f"offramp_flow_{link}", 0.0)))


class WuDistributedController:
    """Wu §IV-D 6단계 합의 루프의 경량 재구성.

    - urban agent(신호 단위): green p1 후보 탐색, 이웃 결합변수(도착유량) 고정.
    - freeway agent(링크 단위): VSL 후보 탐색, on-ramp 유입(urban 결합) 고정.
    - local 예측은 경량 큐/밀도 모델(원문 MILP/SQP 대신 결정적 후보 탐색 —
      wu2022_distributed_reference §8의 허용 근사, fidelity matrix에 기록).
    - leader_enabled=True면 (N_P_star, N_F_star)[veh] conditioning 항을 local objective에
      추가하고, leader가 후보별 coupled 예측으로 system objective를 평가한다(spec 16.5).
    """

    def __init__(self, cfg: ExperimentConfig, leader_enabled: bool = False):
        self.cfg = cfg
        self.leader_enabled = leader_enabled
        self.previous_control: Optional[ControlAction] = None
        self._repair_diagnostics: Dict[str, float] = {}
        self._specs = movement_specs(cfg)
        net = cfg.network
        # 신호별 phase 소속 movement와 포화유율 합(서비스율 = green비율 × Σ포화).
        self._phase_movements: Dict[str, Dict[str, list[str]]] = {}
        for signal in net.signals:
            self._phase_movements[signal] = {
                "p1": [m for m, s in self._specs.items() if s.get("phase") == f"{signal}_p1"],
                "p2": [m for m, s in self._specs.items() if s.get("phase") == f"{signal}_p2"],
            }
        # Leader conditioning 고정 가중 ω (spec 16.5: 평가 전에 고정, Σ=1).
        protected_kinds = {"internal", "boundary_out", "off_ramp"}
        cap_by_signal = {}
        for signal in net.signals:
            cap_by_signal[signal] = sum(
                1.0 for m, s in self._specs.items()
                if s.get("signal") == signal and str(s.get("kind")) in protected_kinds
            )
        total_cap = max(sum(cap_by_signal.values()), 1.0e-9)
        self._omega_p = {s: cap_by_signal[s] / total_cap for s in net.signals}
        self._omega_f = {link: 1.0 / max(len(net.freeway_links), 1) for link in net.freeway_links}
        self._build_coupling_maps()

    def _build_coupling_maps(self) -> None:
        """Jacobi coupling용 토폴로지 캐시를 movement spec에서 자동 유도한다(hand-list 금지).

        - `_upstream_leaving_map`: 하류 신호의 (phase, internal movement)마다 그 movement
          origin 링크로 흘려보내는 상류 (signal, 상류 movement, β) 리스트. 상류 후보
          green의 leaving rate를 하류 도착유량으로 전파하는 데 쓴다.
        - `_offramp_drain_flow`: off_ramp별 하류 신호 drain 근사(off_ramp movement의
          green×movement capacity 합). probe storage 유출에 쓴다.
        - `_last_offramp_flow`: freeway agent가 고른 후보 VSL의 off-ramp 유출[veh/h]
          캐시(link→flow). coupling freeway→urban이 재시뮬 없이 재사용한다.
        """
        net = self.cfg.network
        specs = self._specs
        # origin link(=internal incoming link)별로 그 링크를 destination으로 보내는
        # 상류 movement를 모은다.
        producers_by_link: Dict[str, list[tuple[str, str]]] = {}
        for up_mv, up_spec in specs.items():
            dest = str(up_spec.get("destination", ""))
            up_signal = str(up_spec.get("signal", ""))
            if dest and up_signal in set(net.signals):
                producers_by_link.setdefault(dest, []).append((up_signal, up_mv))
        self._upstream_leaving_map: Dict[str, list[tuple[str, str, float]]] = {}
        for signal in net.signals:
            for phase_id in ("p1", "p2"):
                key = f"{signal}_{phase_id}"
                entries: list[tuple[str, str, float]] = []
                beta_by_origin: Dict[str, float] = {}
                for movement in self._phase_movements[signal][phase_id]:
                    spec = specs[movement]
                    origin = str(spec.get("origin", ""))
                    if not origin:
                        continue
                    beta_by_origin[origin] = beta_by_origin.get(origin, 0.0) + float(spec.get("beta", 0.0))
                for origin, beta in beta_by_origin.items():
                    for up_signal, up_mv in producers_by_link.get(origin, []):
                        entries.append((up_signal, up_mv, beta))
                self._upstream_leaving_map[key] = entries
        # off_ramp drain: off_ramp movement(도시로 빠진 차량)의 green 용량 합.
        self._offramp_drain_flow: Dict[str, list[tuple[str, str]]] = {}
        for movement, spec in specs.items():
            if str(spec.get("kind", "")) == "off_ramp":
                off_ramp = str(spec.get("off_ramp", ""))
                if off_ramp:
                    self._offramp_drain_flow.setdefault(off_ramp, []).append(
                        (str(spec.get("signal", "")), movement)
                    )
        self._last_offramp_flow: Dict[str, float] = {
            **{link: 0.0 for link in net.freeway_links},
            **{off_ramp: 0.0 for off_ramp in net.off_ramps},
        }
        self._has_last_offramp_flow = False

    def _signal_leaving_rate(
        self,
        up_signal: str,
        up_movement: str,
        control: ControlAction,
        state: Optional[TrafficState] = None,
        demand: Optional[DemandStep] = None,
    ) -> float:
        """상류 movement의 후보 green leaving rate[veh/h] = green_fraction×movement capacity.

        β는 호출처에서 곱한다(여기서는 movement 자체의 방출 용량만 반환)."""
        spec = self._specs[up_movement]
        green_fraction = _phase_green_fraction(control, self.cfg, spec)
        cap_flow = _movement_capacity_flow(control, self.cfg, up_movement, spec)
        capacity_flow = float(green_fraction * cap_flow)
        if state is None or demand is None:
            return capacity_flow
        dt_h = max(self.cfg.simulation.T_c_h, 1.0e-9)
        available_flow = max(0.0, state.urban_movement_queue.get(up_movement, 0.0)) / dt_h
        kind = str(spec.get("kind", ""))
        beta = float(spec.get("beta", 1.0))
        if kind == "boundary_in":
            origin = str(spec.get("origin", ""))
            available_flow += beta * max(0.0, demand.urban_boundary.get(origin, 0.0))
        elif kind == "on_ramp":
            ramp = str(spec.get("ramp", ""))
            available_flow += beta * max(0.0, demand.ramp_arrival.get(ramp, 0.0))
        return float(min(capacity_flow, available_flow))

    # ---------- 결합변수 y (Wu §IV-B) ----------

    def _coupling(self, state: TrafficState, control: ControlAction, demand: DemandStep) -> Dict[str, float]:
        """agent 간 교환하는 결합변수 — local solve 동안 고정된다.

        urban→urban: 상류 신호의 후보 green leaving rate(주) + 링크 점유 방출(보조<1).
        urban→freeway: ramp별 reservoir inflow(w_r 캡 없는 green 후보 방출).
        freeway→urban: freeway agent가 고른 후보 VSL의 off-ramp 유출(_last_offramp_flow)."""
        net = self.cfg.network
        y: Dict[str, float] = {}
        t_link_h = (
            float(net.grid_link_storage_veh) * net.urban_avg_vehicle_length_m / 1000.0
            / max(net.urban_avg_speed_km_h, 1.0e-9)
        )
        occupancy_weight = 0.5  # 점유 방출 보조항 가중(<1) — 상류 후보 green이 주채널.
        # 신호·phase별 도착유량 추정: 게이트 수요(β분할) + off-ramp 후보 유입 + 상류
        # 신호의 후보 green leaving rate(+점유 방출 보조) — local solve 동안 고정.
        for signal in net.signals:
            arr = {"p1": 0.0, "p2": 0.0}
            for phase_id in ("p1", "p2"):
                key = f"{signal}_{phase_id}"
                for movement in self._phase_movements[signal][phase_id]:
                    spec = self._specs[movement]
                    kind = str(spec.get("kind", ""))
                    beta = float(spec.get("beta", 0.0))
                    origin = str(spec.get("origin", ""))
                    if kind == "boundary_in":
                        arr[phase_id] += beta * max(0.0, demand.urban_boundary.get(origin, 0.0))
                    elif kind == "on_ramp":
                        ramp = str(spec.get("ramp", ""))
                        arr[phase_id] += beta * max(0.0, demand.ramp_arrival.get(ramp, 0.0))
                    elif kind == "off_ramp":
                        # freeway→urban 결합: freeway agent 후보 VSL의 off-ramp 유출 재사용.
                        off_ramp = str(spec.get("off_ramp", ""))
                        link = net.off_ramp_from_freeway.get(off_ramp, "")
                        if self._has_last_offramp_flow:
                            off_inflow = float(self._last_offramp_flow.get(off_ramp, 0.0))
                        else:
                            # 초기(아직 freeway solve 전): 현재 본선 유량 폴백.
                            base = state.freeway_flow.get(link, [0.0])[-1] if state.freeway_flow.get(link) else 0.0
                            off_inflow = _split_link_offramp_flow(self.cfg, link, off_ramp, base)
                        arr[phase_id] += beta * max(0.0, off_inflow)
                    elif kind == "_legacy_occupancy_disabled":
                        # urban→urban: 상류 신호의 후보 green leaving rate(주채널) +
                        # 링크 점유 방출(보조, 가중<1).
                        cap = net.urban_link_storage_veh.get(origin, 0.0)
                        occupied = max(0.0, cap - state.urban_link_storage.get(origin, cap))
                        arr[phase_id] += occupancy_weight * beta * occupied / max(t_link_h, 1.0e-9) * 0.5
                # 상류 후보 green leaving rate를 β로 분배해 더한다(후보 반응형 주채널).
                for up_signal, up_movement, up_beta in self._upstream_leaving_map.get(key, []):
                    arr[phase_id] += up_beta * self._signal_leaving_rate(
                        up_signal,
                        up_movement,
                        control,
                        state,
                        demand,
                    )
            y[f"arr_{signal}_p1"] = float(arr["p1"])
            y[f"arr_{signal}_p2"] = float(arr["p2"])
        # urban→freeway: ramp별 접근(x_on) no-metering 방출 추정 = min(대기+수요, green×포화).
        # Spec 3.3.1: movement별 queue/arrival와 실제 green 용량을 먼저 제한한 뒤 ramp별로 합친다.
        # queue와 phase 용량을 각각 합산한 뒤 min을 취하면 phase split 효과가 소거된다.
        # WU-CD-F Jacobi: w_r 포화에 막히지 않는 reservoir inflow를 결합변수로 사용해
        # green 후보 차이를 freeway agent에 전파한다(green↑→u_on↑ 단조성 보존).
        onramp_release = estimate_onramp_reservoir_inflow(
            state,
            control,
            demand,
            self.cfg,
            interval_h=self.cfg.simulation.T_c_h,
        )
        for ramp in net.ramps:
            y[f"u_on_{ramp}"] = float(onramp_release.get(ramp, 0.0))
        # freeway boundary 상태(진단·잔차용).
        for link in net.freeway_links:
            rhos = state.freeway_density.get(link, [])
            y[f"rho_{link}"] = float(np.mean(rhos)) if rhos else 0.0
            y[f"mainline_{link}"] = float(max(0.0, demand.freeway_mainline.get(link, 0.0)))
        return y

    # ---------- urban agent local solve ----------

    def _solve_urban_agent(
        self,
        signal: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        previous: ControlAction,
        leader: Optional[WuLeaderAction],
    ) -> tuple[float, float, int]:
        """green p1 후보 탐색 — 반환 (p1*, local objective, evaluations).

        local 모델: phase별 집계 큐 q_p가 도착유량(결합변수 고정) − 서비스율로 진화.
        서비스율 = (green/cycle) × Σ포화유율(plant의 cycle 평균과 동일 회계).
        J = T_c Σ_k (q_p1+q_p2) + R(Δp1)² (+ leader conditioning)."""
        net = self.cfg.network
        sim = self.cfg.simulation
        horizon = max(1, self.cfg.mpc.horizon_steps)
        dt_h = sim.T_c_h
        total = net.effective_green_total
        sat = {
            pid: max(
                sum(
                    min(net.movement_capacity_veh_h, net.movement_capacity_veh_h)
                    for _ in self._phase_movements[signal][pid]
                ),
                1.0e-9,
            )
            for pid in ("p1", "p2")
        }
        q0 = {
            pid: sum(
                max(0.0, state.urban_movement_queue.get(m, 0.0))
                for m in self._phase_movements[signal][pid]
            )
            for pid in ("p1", "p2")
        }
        arr = {pid: float(coupling.get(f"arr_{signal}_{pid}", 0.0)) for pid in ("p1", "p2")}
        prev_p1 = float(previous.green_times.get(f"{signal}_p1", total / 2.0))
        smooth_w = self.cfg.urban_follower.green_smoothness_weight
        p1_pressure = q0["p1"] + arr["p1"] * dt_h * horizon
        p2_pressure = q0["p2"] + arr["p2"] * dt_h * horizon
        pressure_center = queue_pressure_green_target(p1_pressure, p2_pressure, self.cfg)
        raw_candidates = [total / 2.0, prev_p1, pressure_center]
        if self.cfg.mpc.relaxed_quantized_controls:
            # Spec 17.5/18.9: pressure split은 후보 중심일 뿐이며, 주변 후보를 같은 TTS 비용으로 평가한다.
            raw_candidates.extend([
                pressure_center - 1.0,
                pressure_center + 1.0,
                pressure_center - 2.0,
                pressure_center + 2.0,
                pressure_center - 5.0,
                pressure_center + 5.0,
            ])
        else:
            raw_candidates.extend(float(v) for v in np.linspace(net.green_min, net.green_max, 7))
        candidates: list[float] = []
        for raw in raw_candidates:
            if self.cfg.mpc.relaxed_quantized_controls:
                repaired = repair_green_pair(float(raw), self.cfg)
                accumulate_repair_diagnostics(self._repair_diagnostics, green=repaired)
                p1_value = repaired.p1
            else:
                p1_value = float(np.clip(raw, net.green_min, net.green_max))
                p2_value = total - p1_value
                if p2_value < net.green_min:
                    p2_value = net.green_min
                    p1_value = total - p2_value
                if p2_value > net.green_max:
                    p2_value = net.green_max
                    p1_value = total - p2_value
            if not any(abs(p1_value - existing) <= 1.0e-9 for existing in candidates):
                candidates.append(float(p1_value))

        best_p1, best_obj = prev_p1, float("inf")
        evals = 0
        for p1 in candidates:
            p2 = total - p1
            if p2 < net.green_min - 1.0e-9 or p2 > net.green_max + 1.0e-9:
                continue
            q = dict(q0)
            cost = 0.0
            for _ in range(horizon):
                for pid, g in (("p1", p1), ("p2", p2)):
                    service = (g / max(net.cycle_length, 1e-9)) * sat[pid] * dt_h
                    q[pid] = max(0.0, q[pid] + arr[pid] * dt_h - service)
                cost += (q["p1"] + q["p2"]) * dt_h
            cost += smooth_w * abs(p1 - prev_p1)
            if leader is not None:
                # spec 16.5 conditioning: 예측 국소 누적이 ω×N_P_star를 넘으면 패널티.
                n_pred = q["p1"] + q["p2"]
                cost += self.cfg.leader.w_P * max(0.0, n_pred - self._omega_p[signal] * leader.n_p_star)
            evals += 1
            if cost < best_obj:
                best_obj, best_p1 = cost, float(p1)
        return best_p1, best_obj, evals

    # ---------- freeway agent local solve ----------

    def _freeway_segment_candidates(
        self,
        link: str,
        n_seg: int,
        previous: ControlAction,
    ) -> list[list[float]]:
        """Option C segment 벡터 VSL 후보 집합 — 폭발 방지 가지치기.

        - 병목 segment(off-ramp seg, 보통 마지막): {max, 직전값} 2값으로 제한.
          (병목 자신은 max 유지해야 q_out 회복이 상쇄되지 않는다.)
        - 상류 segment(seg0/seg1): max_vsl_step(±20) 내 vsl_set 후보.
        총 수십 개 수준(상류 2개 × 후보^2 × 병목 2값).
        """
        net = self.cfg.network
        ff = self.cfg.freeway_follower
        vsl_max = max(ff.vsl_set)
        vsl_set = sorted(float(v) for v in ff.vsl_set)
        step = ff.max_vsl_step
        # off-ramp(병목) segment 인덱스 집합 — 이 링크에서 빠지는 off-ramp의 segment.
        bottleneck_idx = {
            int(net.off_ramp_segment_index.get(o, n_seg - 1))
            for o in net.off_ramps
            if net.off_ramp_from_freeway.get(o) == link
        }
        if not bottleneck_idx:
            bottleneck_idx = {n_seg - 1}

        per_seg_options: list[list[float]] = []
        for i in range(n_seg):
            prev_i = segment_vsl(previous, link, i, self.cfg)
            if i in bottleneck_idx:
                # 병목: max 유지 우선 + 직전값(스무딩). 중복 제거.
                opts = {vsl_max, prev_i}
            else:
                # 상류: max_vsl_step 내 vsl_set 후보로 metering 가지치기.
                opts = {v for v in vsl_set if abs(v - prev_i) <= step + 1e-9}
                opts.add(vsl_max)
                if not opts:
                    opts = set(vsl_set)
            per_seg_options.append(sorted(opts))

        # 데카르트 곱 — 후보 벡터.
        vectors: list[list[float]] = [[]]
        for opts in per_seg_options:
            vectors = [vec + [v] for vec in vectors for v in opts]
        return vectors

    def _relaxed_freeway_segment_candidates(
        self,
        link: str,
        n_seg: int,
        state: TrafficState,
        coupling: Mapping[str, float],
        previous: ControlAction,
        demand: DemandStep,
    ) -> list[list[float]]:
        """Spec 17.5 WU relaxed VSL: Cartesian 열거 대신 pressure target 벡터만 만든다."""
        net = self.cfg.network
        ff = self.cfg.freeway_follower
        vsl_max = float(max(ff.vsl_set))
        vsl_min = float(min(ff.vsl_set))
        densities = list(state.freeway_density.get(link, []))
        profile, _ = effective_lane_profile(state, self.cfg, demand)
        lane_profile = profile.get(link, [net.freeway_lanes for _ in range(n_seg)])
        # 2026-06-30: VSL-mod default 'pin off'. 옛 코드는 off-ramp 병목 segment를 vsl_max로 고정했으나
        # (단일 실험상 TTT 효과 0), default로 핀을 끄고 전 segment가 pressure 휴리스틱을 쓰게 한다.
        # 주의: 이론상 병목 자체를 늦추면 discharge가 줄 수 있어 약손해 가능 — 되돌리려면 아래를
        # `{off_ramp_segment_index...} or {n_seg-1}`로 복원.
        bottleneck_idx: set = set()
        downstream_pressure = max(0.0, float(coupling.get(f"p_down_{link}", 0.0)))
        target_vec: list[float] = []
        neutral_vec: list[float] = []
        for i in range(n_seg):
            prev_i = segment_vsl(previous, link, i, self.cfg)
            rho = float(densities[i]) if i < len(densities) else net.rho_crit
            density_pressure = max(0.0, rho / max(net.rho_crit, 1.0e-9) - 0.9)
            lane_loss = max(0.0, net.freeway_lanes - float(lane_profile[i] if i < len(lane_profile) else net.freeway_lanes))
            if i in bottleneck_idx:
                # Wu mechanism 보존: 병목/off-ramp segment는 discharge 회복을 위해 기본적으로 max VSL을 유지한다.
                continuous = vsl_max
            else:
                upstream_weight = max(0.0, 1.0 - (i / max(n_seg - 1, 1)))
                pressure = density_pressure + downstream_pressure * upstream_weight + 0.5 * lane_loss
                continuous = vsl_max - 25.0 * min(2.0, pressure)
                continuous = 0.65 * continuous + 0.35 * prev_i
                continuous = max(vsl_min, min(vsl_max, continuous))
            target_repaired = repair_vsl_value(continuous, prev_i, self.cfg)
            neutral_repaired = repair_vsl_value(vsl_max, prev_i, self.cfg)
            accumulate_repair_diagnostics(self._repair_diagnostics, vsl=target_repaired)
            accumulate_repair_diagnostics(self._repair_diagnostics, vsl=neutral_repaired)
            target_vec.append(target_repaired.value)
            neutral_vec.append(neutral_repaired.value)

        vectors: list[list[float]] = []
        raw_vectors = [target_vec]
        for delta in (-self.cfg.mpc.relaxed_vsl_quantum_km_h, self.cfg.mpc.relaxed_vsl_quantum_km_h):
            neighbor: list[float] = []
            for value, prev_i in zip(target_vec, [segment_vsl(previous, link, i, self.cfg) for i in range(n_seg)]):
                repaired = repair_vsl_value(value + delta, prev_i, self.cfg)
                accumulate_repair_diagnostics(self._repair_diagnostics, vsl=repaired)
                neighbor.append(repaired.value)
            raw_vectors.append(neighbor)
        if self.cfg.mpc.relaxed_wu_vsl_include_neutral:
            raw_vectors.append(neutral_vec)
        for vec in raw_vectors:
            if not any(all(abs(a - b) <= 1.0e-9 for a, b in zip(vec, existing)) for existing in vectors):
                vectors.append([float(v) for v in vec])
        return vectors

    def _solve_freeway_agent(
        self,
        link: str,
        state: TrafficState,
        coupling: Mapping[str, float],
        demand: DemandStep,
        previous: ControlAction,
        leader: Optional[WuLeaderAction],
    ) -> tuple[Dict[str, float], float, int]:
        """segment 벡터 VSL 후보 탐색 — 반환 (vsl_dict, local objective, evaluations).

        Option C: 후보는 segment별 VSL 조합(_freeway_segment_candidates). 각 후보를
        storage-aware probe로 Np(freeway_prediction_horizon_steps)×K_cf substep 적분해,
        상류 metering→seg1→seg2 유입↓→off-ramp storage 유입↓→λ_eff 회복(capacity-drop
        회피)이 horizon 누적 link 차량합(순수 TTS, 메커니즘 A)에 반영되게 한다. 인위적
        결합 페널티(메커니즘 B)는 없다 — VSL 활성화는 순수 TTS 예측에서만 나온다.
        반환 dict 키는 segment 키 `{link}__seg{i}`(plant가 segment_vsl로 읽음)."""
        net = self.cfg.network
        sim = self.cfg.simulation
        # Wu Np=10 정신: freeway agent probe는 λ-recovery가 여러 step에 걸쳐 누적되도록
        # freeway_prediction_horizon_steps를 쓴다(0 이하면 mpc.horizon_steps fallback).
        ff = self.cfg.freeway_follower
        horizon = max(1, ff.freeway_prediction_horizon_steps or self.cfg.mpc.horizon_steps)
        dt_h = sim.T_f_h
        vsl_max = max(ff.vsl_set)
        smooth_w = ff.vsl_smoothness_weight
        n_seg = len(state.freeway_density.get(link, [])) or net.freeway_segments_per_link
        prev_vec = [segment_vsl(previous, link, i, self.cfg) for i in range(n_seg)]

        candidates = (
            self._relaxed_freeway_segment_candidates(link, n_seg, state, coupling, previous, demand)
            if self.cfg.mpc.relaxed_quantized_controls
            else self._freeway_segment_candidates(link, n_seg, previous)
        )
        best_vec, best_obj = list(prev_vec), float("inf")
        best_offramp_flow: Dict[str, float] = {
            off_ramp: 0.0
            for off_ramp in net.off_ramps
            if net.off_ramp_from_freeway.get(off_ramp) == link
        }
        evals = 0
        for vec in candidates:
            probe = state.copy()
            candidate_control = ControlAction(
                ramp_metering=dict(previous.ramp_metering),
                vsl=dict(previous.vsl),
                green_times=dict(previous.green_times),
                offsets=dict(previous.offsets),
                inflow_outflow_allocation={},
            )
            # 후보 segment 벡터를 segment 키로 주입(plant가 segment_vsl로 읽음).
            for i, v in enumerate(vec):
                candidate_control.vsl[f"{link}__seg{i}"] = float(v)
            cost = 0.0
            first_offramp_flow = dict(best_offramp_flow)
            first_substep = True
            for _ in range(horizon):
                for _ in range(sim.K_cf):
                    # urban->freeway coupling [veh/h]을 ramp reservoir에 넣은 뒤,
                    # no-metering receiving constraint로 실제 freeway 진입량을 계산한다.
                    for ramp in net.ramps:
                        approach_flow = max(0.0, float(coupling.get(f"u_on_{ramp}", 0.0)))
                        probe.ramp_queue[ramp] = min(
                            net.ramp_queue_max_veh,
                            max(0.0, probe.ramp_queue.get(ramp, 0.0)) + approach_flow * dt_h,
                        )
                    ramp_release, ramp_diag = compute_ramp_release_flows(
                        probe,
                        candidate_control,
                        demand,
                        self.cfg,
                        include_current_arrivals=False,
                    )
                    for ramp, release in ramp_release.items():
                        probe.ramp_queue[ramp] = max(
                            0.0,
                            probe.ramp_queue.get(ramp, 0.0) - max(0.0, release) * dt_h,
                        )
                    offramp_capacity = off_ramp_capacity_by_freeway_link(
                        probe.copy(),
                        self.cfg,
                        interval_h=dt_h,
                    )
                    _, fw_diag = freeway_substep(
                        probe,
                        candidate_control,
                        demand,
                        self.cfg,
                        offramp_capacity_veh_h=offramp_capacity,
                        ramp_release_veh_h=ramp_release,
                        ramp_release_diagnostics=ramp_diag,
                        update_ramp_queues=False,
                        include_ramp_queue_ttt=False,
                    )
                    # storage-aware probe: off-ramp 유출을 storage로 유입, 하류 신호 drain으로
                    # 유출시켜 점유를 갱신한다. 다음 substep의 effective_lane_profile이
                    # capacity-drop을 반영해 VSL↓→다운스트림 유입↓→λ_eff 회복 이득을 내생화.
                    self._update_probe_offramp_storage(probe, fw_diag, candidate_control, dt_h)
                    if first_substep:
                        # coupling freeway→urban이 재사용할 후보 VSL의 off-ramp 유출[veh/h].
                        first_offramp_flow = {
                            off_ramp: _offramp_flow_from_diagnostics(fw_diag, self.cfg, off_ramp)
                            for off_ramp in best_offramp_flow
                        }
                        first_substep = False

                    # Wu local TTS: 해당 freeway agent의 segment 차량과 연결 ramp queue.
                    link_vehicles = sum(probe.freeway_vehicle_count_by_link(net).get(link, []))
                    link_ramp_queue = sum(
                        max(0.0, probe.ramp_queue.get(ramp, 0.0))
                        for ramp in net.ramps
                        if net.ramp_to_freeway.get(ramp) == link
                    )
                    # off-ramp 램프 storage 재귀속(design 2026-06-17): 이 freeway link에서
                    # 갈라지는 off-ramp의 storage 점유를 자기 TTS에 가산한다. probe storage는
                    # _update_probe_offramp_storage가 후보 VSL별로 예측하므로, VSL↓→off-ramp
                    # 유입↓→storage 점유↓가 자기 비용에 반영돼 metering이 emergent하게 발현.
                    link_offramp_storage = 0.0
                    for off_ramp in net.off_ramps:
                        if net.off_ramp_from_freeway.get(off_ramp) != link:
                            continue
                        storage_link = net.off_ramp_storage_link.get(off_ramp, "")
                        capacity = float(net.urban_link_storage_veh.get(storage_link, 0.0))
                        if capacity <= 0.0:
                            continue
                        avail = float(probe.urban_link_storage.get(storage_link, capacity))
                        link_offramp_storage += max(0.0, capacity - avail)
                    # Wu 순수 TTS: Σ L·λ_eff·ρ(=segment 차량수) + ramp 큐 + off-ramp 램프 storage.
                    # 비-Wu density_penalty 항은 제거 — storage-aware probe가 capacity-drop을
                    # TTS에 내생화하므로 별도 패널티 없이 VSL이 혼잡 시 작동한다.
                    cost += (link_vehicles + link_ramp_queue + link_offramp_storage) * dt_h
            # smoothness: segment별 직전값 대비 변화량 합.
            cost += smooth_w * sum(abs(v - prev_vec[i]) for i, v in enumerate(vec))
            if leader is not None:
                n_pred = sum(probe.freeway_vehicle_count_by_link(net).get(link, []))
                cost += self.cfg.leader.w_F * max(0.0, n_pred - self._omega_f[link] * leader.n_f_star)
            evals += 1
            if cost < best_obj:
                best_obj, best_vec = cost, list(vec)
                best_offramp_flow = dict(first_offramp_flow)
        # 선택된 후보 VSL의 off-ramp 유출을 캐시 — coupling freeway→urban이 재시뮬
        # 없이 후보 반응형 off-ramp inflow로 재사용한다.
        if best_obj < float("inf"):
            for off_ramp, flow in best_offramp_flow.items():
                self._last_offramp_flow[off_ramp] = float(flow)
            self._last_offramp_flow[link] = float(sum(best_offramp_flow.values()))
            self._has_last_offramp_flow = True
        else:
            for off_ramp in best_offramp_flow:
                self._last_offramp_flow[off_ramp] = 0.0
            self._last_offramp_flow[link] = 0.0
        # segment 키 dict 반환(+ link 키는 min-over-segment로 하위호환 fallback 보존).
        vsl_dict: Dict[str, float] = {f"{link}__seg{i}": float(v) for i, v in enumerate(best_vec)}
        vsl_dict[link] = float(min(best_vec)) if best_vec else vsl_max
        return vsl_dict, best_obj, evals

    def _update_probe_offramp_storage(
        self,
        probe: TrafficState,
        fw_diag: Mapping[str, float],
        control: ControlAction,
        dt_h: float,
    ) -> None:
        """probe의 off-ramp storage 점유를 한 substep 갱신한다.

        유입 = freeway_substep이 보고한 off-ramp 유출(link 합산을 split ratio로 분배),
        유출 = 하류 신호의 off_ramp movement drain(green×movement capacity 합).
        storage 가용량(available)을 감소/증가시켜 effective_lane_profile이 다음 substep에
        capacity-drop을 반영하게 한다(원논문 storage-aware 다운스트림 인식)."""
        net = self.cfg.network
        for off_ramp in net.off_ramps:
            link = net.off_ramp_from_freeway.get(off_ramp, "")
            storage_link = net.off_ramp_storage_link.get(off_ramp, "")
            if not link or not storage_link:
                continue
            capacity = float(net.urban_link_storage_veh.get(storage_link, 0.0))
            inflow = _offramp_flow_from_diagnostics(fw_diag, self.cfg, off_ramp)
            # drain = 하류 신호 off_ramp movement green 처리량, 단 receiving 도시 링크
            # 가용 공간으로 제약(도시 포화 시 spillback → drain 막힘 → storage 정체 →
            # capacity-drop 지속). 이게 혼잡 interval에서 VSL이 작동하는 물리 경로다.
            drain = 0.0
            recv_intake: Dict[str, float] = {}
            for signal, movement in self._offramp_drain_flow.get(off_ramp, []):
                rate = self._signal_leaving_rate(signal, movement, control)
                recv_link = str(self._specs[movement].get("receiving_link", ""))
                recv_cap = float(net.urban_link_storage_veh.get(recv_link, 0.0))
                if recv_cap > 0.0:
                    recv_avail = float(probe.urban_link_storage.get(recv_link, recv_cap))
                    rate = min(rate, max(0.0, recv_avail) / max(dt_h, 1.0e-9))
                drain += rate
                if recv_link:
                    recv_intake[recv_link] = recv_intake.get(recv_link, 0.0) + rate
            available = float(probe.urban_link_storage.get(storage_link, capacity))
            occupied = max(0.0, capacity - available)
            occupied = min(capacity, max(0.0, occupied + (inflow - drain) * dt_h))
            probe.urban_link_storage[storage_link] = capacity - occupied
            # A″-3: 드레인된 차량이 receiving 도시 링크 공간을 점유하게 한다(보존).
            # 자유-sink 시절엔 drain이 receiving 공간을 갉아먹지 않아 도시가 절대 포화되지
            # 않았고, 그래서 off-ramp storage가 VSL과 무관하게 항상 빠져 λ_eff가 늘 회복돼
            # VSL이 이득을 못 봤다(메커니즘 A inert). 점유분은 도시 자체 유한출구(A″-1의
            # boundary_out_capacity)로만 풀려 도시 혼잡 주원인=off-ramp 홍수일 때 도시가
            # 포화 유지되고, VSL↓로 유입을 끊으면 풀린다(Wu λ-recovery 채널).
            urban_exit = float(net.boundary_out_capacity_veh_h)
            for recv_link, intake in recv_intake.items():
                recv_cap = float(net.urban_link_storage_veh.get(recv_link, 0.0))
                if recv_cap <= 0.0:
                    continue
                recv_avail = float(probe.urban_link_storage.get(recv_link, recv_cap))
                recv_occ = max(0.0, recv_cap - recv_avail)
                relief = urban_exit if urban_exit > 0.0 else float("inf")
                recv_occ = min(recv_cap, max(0.0, recv_occ + (intake - relief) * dt_h))
                probe.urban_link_storage[recv_link] = recv_cap - recv_occ

    # ---------- 합의 루프 (Wu §IV-D) ----------

    def _solve_followers(
        self,
        state: TrafficState,
        demand: DemandStep,
        previous: ControlAction,
        leader: Optional[WuLeaderAction],
    ) -> tuple[ControlAction, int, bool, float, int]:
        net = self.cfg.network
        self._repair_diagnostics = {}
        control = _wu_fixed_control(self.cfg)
        control.green_times = dict(previous.green_times)
        control.vsl = dict(previous.vsl)
        coupling = self._coupling(state, control, demand)
        # Wu 원논문: Jacobi 병렬 + S_max=5 절단. WU-CD-F 경로는 5로 클램프해
        # green→arr 과결합 발산을 막는다(위험요소: under-relaxation+점유 보조항과 함께).
        s_max = max(1, min(self.cfg.mpc.max_nash_iter, 5))
        alpha = 0.5  # under-relaxation: y_new = (1-α)y_old + α·y_pred.
        evals = 0
        residual = float("inf")
        converged = False
        iteration = 0
        for iteration in range(1, s_max + 1):
            # Jacobi: iteration 시작 control을 스냅샷으로 고정 → 그 iteration 내 모든 agent가
            # 동일 고정 y와 동일 previous(스냅샷)를 입력으로 푼다(green을 통한 간접 결합 차단).
            snapshot = ControlAction(
                ramp_metering=dict(control.ramp_metering),
                vsl=dict(control.vsl),
                green_times=dict(control.green_times),
                offsets=dict(control.offsets),
                inflow_outflow_allocation={},
            )
            new_green: Dict[str, float] = {}
            new_vsl: Dict[str, float] = {}
            for signal in net.signals:
                p1, _, e = self._solve_urban_agent(signal, state, coupling, snapshot, leader)
                new_green[f"{signal}_p1"] = p1
                new_green[f"{signal}_p2"] = net.effective_green_total - p1
                evals += e
            for link in net.freeway_links:
                vsl_dict, _, e = self._solve_freeway_agent(
                    link,
                    state,
                    coupling,
                    demand,
                    snapshot,
                    leader,
                )
                new_vsl.update(vsl_dict)
                evals += e
            control.green_times.update(new_green)
            control.vsl.update(new_vsl)
            # outgoing y를 후보 제어로 갱신한 뒤 under-relaxation으로 합성.
            predicted = self._coupling(state, control, demand)
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
        control.diagnostics.update(self._repair_diagnostics)
        return control, iteration, converged, float(residual), evals

    # ---------- leader 평가 (WU-MATCHED-STACKELBERG) ----------

    def _leader_candidates(self, state: TrafficState) -> List[WuLeaderAction]:
        net = self.cfg.network
        n_p_now = state.protected_accumulation_veh(net)
        n_f_now = state.total_freeway_vehicles(net)
        # 후보 밴드: 현재 누적과 임계 부근 — 평가 전 고정(결과 보고 후 재조정 금지).
        p_values = [0.8 * max(n_p_now, 50.0), max(n_p_now, 50.0), self.cfg.leader.N_P_crit_veh]
        f_values = [0.8 * max(n_f_now, 50.0), max(n_f_now, 50.0), 1.2 * max(n_f_now, 50.0)]
        return [WuLeaderAction(float(p), float(f)) for p in p_values for f in f_values]

    def _system_objective(self, states: List[TrafficState]) -> float:
        """spec 16.8형 system objective(공통 비교 비용) — n_P+n_F+초과 패널티."""
        net = self.cfg.network
        lc = self.cfg.leader
        total = 0.0
        for s in states:
            n_p = s.total_urban_vehicles(net)
            n_f = s.total_freeway_vehicles(net)
            # off-ramp 램프 storage 재귀속(design 2026-06-17): urban에서 빠진 램프 storage를
            # freeway 쪽 비용으로 계상(total objective 보존 + freeway 귀속, plant 공통).
            n_f += s.off_ramp_storage_occupancy_veh(net)
            total += n_p + n_f
            total += lc.w_P * max(0.0, s.protected_accumulation_veh(net) - lc.N_P_crit_veh)
            total += lc.w_F * sum(
                net.freeway_segment_length_km * net.freeway_lanes * max(0.0, rho - net.rho_crit)
                for values in s.freeway_density.values()
                for rho in values
            )
        return float(total)

    def _add_uncontrolled_node_diagnostics(self, control: ControlAction, state: TrafficState) -> None:
        """WU 계열도 E 비통제 노드 TTT coverage를 같은 키로 노출한다."""
        net = self.cfg.network
        movement = state.uncontrolled_node_movement_queue_veh(net)
        storage = state.uncontrolled_node_storage_occupancy_veh(net)
        vehicles = movement + storage
        horizon_h = self.cfg.simulation.T_c_h * max(1, self.cfg.mpc.horizon_steps)
        control.diagnostics["urban_uncontrolled_node_movement_queue_veh"] = float(movement)
        control.diagnostics["urban_uncontrolled_node_storage_occupancy_veh"] = float(storage)
        control.diagnostics["urban_uncontrolled_node_vehicles_veh"] = float(vehicles)
        control.diagnostics["urban_uncontrolled_node_objective_tts"] = float(vehicles * horizon_h)
        control.diagnostics["urban_uncontrolled_node_objective_covered"] = 1.0

    def decide_with_info(
        self,
        state: TrafficState,
        demand_forecast: Iterable[DemandStep],
        previous_control: Optional[ControlAction] = None,
    ) -> WuDecisionInfo:
        from src.simulation.coupling import run_coupled_interval

        start = time.perf_counter()
        forecast = list(demand_forecast)
        previous = previous_control or self.previous_control or _wu_fixed_control(self.cfg)
        total_evals = 0

        if not self.leader_enabled:
            control, iters, converged, residual, evals = self._solve_followers(
                state, forecast[0], previous, leader=None,
            )
            total_evals += evals
            info = WuDecisionInfo(
                control=control,
                iterations=iters,
                converged=converged,
                coupling_residual=residual,
                solver_evaluations=total_evals,
                computation_time_sec=time.perf_counter() - start,
            )
            self._add_uncontrolled_node_diagnostics(control, state)
            self.previous_control = control
            return info

        # WU-MATCHED-STACKELBERG: 후보별 follower 응답 → coupled 예측 → system objective.
        best: tuple[float, ControlAction, WuLeaderAction, int, bool, float] | None = None
        candidates = self._leader_candidates(state)
        for action in candidates:
            control, iters, converged, residual, evals = self._solve_followers(
                state, forecast[0], previous, leader=action,
            )
            total_evals += evals
            sim_state = state.copy()
            states: List[TrafficState] = []
            for demand in forecast[: self.cfg.mpc.horizon_steps]:
                run_coupled_interval(sim_state, control, demand, self.cfg)
                sim_state.time_sec += self.cfg.simulation.control_interval
                states.append(sim_state.copy())
            obj = self._system_objective(states)
            total_evals += len(states)
            if best is None or obj < best[0]:
                best = (obj, control, action, iters, converged, residual)
        assert best is not None
        obj, control, action, iters, converged, residual = best
        info = WuDecisionInfo(
            control=control,
            iterations=iters,
            converged=converged,
            coupling_residual=residual,
            solver_evaluations=total_evals,
            computation_time_sec=time.perf_counter() - start,
            leader_candidates=len(candidates),
            leader_selected=action,
            leader_objective=obj,
        )
        self._add_uncontrolled_node_diagnostics(control, state)
        self.previous_control = control
        return info
