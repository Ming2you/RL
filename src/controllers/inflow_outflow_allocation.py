from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import numpy as np

from src.controllers.leader import LeaderAction
from src.models.demand import DemandStep
from src.models.state import ExperimentConfig, TrafficState
from src.models.urban_queue_model import (
    boundary_group_key,
    movement_storage_capacity,
    movement_specs,
    safe_balance_index,
)


INFLOW_KINDS = {"boundary_in", "off_ramp"}
OUTFLOW_KINDS = {"boundary_out", "on_ramp"}
# 균형화(B) 대상은 외부 게이트(in)와 통제 가능한 on_ramp(out)만 — boundary_out은
# 자유 유출 sink, off_ramp는 우선 방출 transfer 큐(상시 ≈0)라 균등화가 정의되지
# 않는다(net-flow 회계에는 INFLOW/OUTFLOW_KINDS 전체를 그대로 쓴다).
BALANCE_INFLOW_KINDS = {"boundary_in"}
BALANCE_OUTFLOW_KINDS = {"on_ramp"}


@dataclass
class AllocationResult:
    movement_flows: Dict[str, float]
    movement_green_sec: Dict[str, float]
    target_net_inflow_veh_h: float
    projected_net_inflow_veh_h: float
    residual_veh_h: float
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _ObjectiveContext:
    """PSO particle batch objective를 같은 수식으로 빠르게 평가하기 위한 고정 항."""

    queue: np.ndarray
    signs: np.ndarray
    inflow_groups: list[np.ndarray]
    inflow_caps: np.ndarray
    outflow_groups: list[np.ndarray]
    outflow_caps: np.ndarray


class InflowOutflowAllocationModule:
    """논문 §3.2 density-balancing inflow/outflow allocation module.

    이 모듈은 leader의 `N_P_star`를 받아 perimeter 전체 movement의 기준 service flow와
    green setpoint를 한 번 산출한다. Urban agent는 이 결과를 `±eps_g` 범위에서만
    fine-tune한다.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.solve_count = 0

    def _target_horizon_h(self, forecast: "list[DemandStep] | None") -> float:
        steps = forecast[: max(1, self.cfg.mpc.horizon_steps)] if forecast else []
        count = len(steps) if steps else max(1, int(self.cfg.mpc.horizon_steps))
        return float(max(self.cfg.simulation.T_c_h * count, 1.0e-9))

    def _leader_net_inflow_target_veh_h(
        self,
        leader: LeaderAction,
        forecast: "list[DemandStep] | None",
    ) -> float:
        return float(leader.N_P_star) / self._target_horizon_h(forecast)

    def solve(
        self,
        state: TrafficState,
        leader: LeaderAction,
        forecast: "list[DemandStep] | None" = None,
    ) -> AllocationResult:
        # 2026-06-13 재정의(spec 16.7): leaderless 모드 제거 — 이 모듈은 Leader target을
        # 입력으로 갖는 coordination 기구이므로 leader 없이는 호출되지 않는다.
        # forecast가 주어지면 target이 예측 off-ramp 외란 유입을 반영한다(진단 문서 §4).
        self.solve_count += 1
        specs = movement_specs(self.cfg)
        movements = [
            movement
            for movement, spec in specs.items()
            if spec.get("kind") in INFLOW_KINDS | OUTFLOW_KINDS
        ]
        if not movements:
            return AllocationResult({}, {}, 0.0, 0.0, 0.0, {"allocation_module_active": 1.0})

        target = self._leader_net_inflow_target_veh_h(leader, forecast)
        lower, upper = self._bounds(movements, specs, leader)
        kinds = [str(specs[m].get("kind", "")) for m in movements]
        target = self._clip_target(target, lower, upper, kinds)
        # 같은 state/leader에서는 호출 순서와 무관하게 같은 allocation plan을 재현한다.
        time_index = int(round(state.time_sec / max(self.cfg.simulation.control_interval, 1.0e-9)))
        seed = int(self.cfg.simulation.random_seed + time_index)
        best = self._run_pso(state, movements, kinds, lower, upper, target, seed)
        best = self._project_net_flow(best, lower, upper, kinds, target)
        flows = {movement: float(best[idx]) for idx, movement in enumerate(movements)}
        green = {
            movement: self._flow_to_green_sec(flows[movement])
            for movement in movements
        }
        in_values = self._densities_after_service(state, movements, kinds, best, BALANCE_INFLOW_KINDS)
        out_values = self._densities_after_service(state, movements, kinds, best, BALANCE_OUTFLOW_KINDS)
        net = self._net_flow(best, kinds)
        residual = abs(net - target)
        diagnostics = {
            "allocation_module_active": 1.0,
            "allocation_pso_calls": float(self.solve_count),
            "allocation_pso_particles": float(self.cfg.urban_follower.allocation_pso_particles),
            "allocation_pso_iterations": float(self.cfg.urban_follower.allocation_pso_iterations),
            "allocation_target_net_inflow_veh_h": float(target),
            "allocation_projected_net_inflow_veh_h": float(net),
            "allocation_net_inflow_residual_veh_h": float(residual),
            "allocation_B_in": safe_balance_index(in_values),
            "allocation_B_out": safe_balance_index(out_values),
            "allocation_inflow_dim": float(len(in_values)),
            "allocation_outflow_dim": float(len(out_values)),
        }
        return AllocationResult(flows, green, float(target), float(net), float(residual), diagnostics)

    def _bounds(
        self,
        movements: list[str],
        specs: Mapping[str, Mapping[str, object]],
        leader: LeaderAction,
    ) -> tuple[np.ndarray, np.ndarray]:
        net = self.cfg.network
        flow_max = float(net.green_max) / max(net.cycle_length, 1.0e-9) * float(net.movement_capacity_veh_h)
        # on_ramp 상한 = leader N_UF_star의 ramp 용량비 배분. urban→ramp 전이와 freeway
        # metering이 같은 N_UF_star를 공유해(중복 제어 방지) 피크에 본선이 받을 수 없는
        # 유량을 w_r로 밀어넣지 않는다. (전이를 풀면 적체가 x_on→w_r/본선으로 자리만
        # 옮겨 총 TTT는 같고, 피크 본선 붕괴 위험만 커지는 것을 3600s A/B로 확인.)
        ramp_counts: Dict[str, int] = {}
        for movement in movements:
            ramp = str(specs[movement].get("ramp", ""))
            if ramp and str(specs[movement].get("kind", "")) == "on_ramp":
                ramp_counts[ramp] = ramp_counts.get(ramp, 0) + 1
        total_ramp_cap = max(float(net.total_ramp_capacity), 1.0e-9)
        nuf_total = max(0.0, float(leader.N_UF_star))
        lower = []
        upper = []
        for movement in movements:
            kind = str(specs[movement].get("kind", ""))
            if kind == "boundary_out":
                # 자유 유출 sink는 균형화 대상이 아니다 — 항상 최대 서비스로 고정
                # (B_out 균형화가 출구를 조이면 내부 누적이 폭주한다).
                lower.append(flow_max)
                upper.append(flow_max)
            elif kind == "on_ramp":
                ramp = str(specs[movement].get("ramp", ""))
                share = float(net.ramp_capacity_veh_h.get(ramp, 0.0)) / total_ramp_cap
                per_movement = nuf_total * share / max(ramp_counts.get(ramp, 1), 1)
                lower.append(0.0)
                upper.append(float(np.clip(per_movement, 0.0, flow_max)))
            else:
                # 하한 0: 2-phase 공유 green에서 phase green_min은 movement별 최소
                # 유량이 아니다. 그리드 라우팅 후 유입 movement 35개의 g_min 하한 합
                # (≈8.2k veh/h)이 수요를 넘어 perimeter 게이팅을 무력화했던 것을 복원.
                lower.append(0.0)
                upper.append(flow_max)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def _clip_target(
        self,
        target: float,
        lower: np.ndarray,
        upper: np.ndarray,
        kinds: list[str],
    ) -> float:
        inflow = np.asarray([kind in INFLOW_KINDS for kind in kinds], dtype=bool)
        outflow = np.asarray([kind in OUTFLOW_KINDS for kind in kinds], dtype=bool)
        min_net = float(np.sum(lower[inflow]) - np.sum(upper[outflow]))
        max_net = float(np.sum(upper[inflow]) - np.sum(lower[outflow]))
        return float(np.clip(target, min_net, max_net))

    def _run_pso(
        self,
        state: TrafficState,
        movements: list[str],
        kinds: list[str],
        lower: np.ndarray,
        upper: np.ndarray,
        target: float,
        seed: int,
    ) -> np.ndarray:
        particles = max(4, int(self.cfg.urban_follower.allocation_pso_particles))
        iterations = max(1, int(self.cfg.urban_follower.allocation_pso_iterations))
        rng = np.random.default_rng(seed)
        span = np.maximum(upper - lower, 1.0e-9)
        pos = lower + rng.random((particles, len(movements))) * span
        vel = rng.normal(0.0, 0.10, size=pos.shape) * span
        context = self._objective_context(state, movements, kinds)
        personal = pos.copy()
        personal_score = self._objective_many(context, kinds, pos, target)
        best_idx = int(np.argmin(personal_score))
        global_best = personal[best_idx].copy()
        global_score = float(personal_score[best_idx])

        for _ in range(iterations):
            r1 = rng.random(pos.shape)
            r2 = rng.random(pos.shape)
            vel = 0.55 * vel + 1.45 * r1 * (personal - pos) + 1.45 * r2 * (global_best - pos)
            pos = np.clip(pos + vel, lower, upper)
            scores = self._objective_many(context, kinds, pos, target)
            improved = scores < personal_score
            personal[improved] = pos[improved]
            personal_score[improved] = scores[improved]
            best_idx = int(np.argmin(personal_score))
            if float(personal_score[best_idx]) < global_score:
                global_score = float(personal_score[best_idx])
                global_best = personal[best_idx].copy()
        return global_best

    def _objective(
        self,
        state: TrafficState,
        movements: list[str],
        kinds: list[str],
        flows: np.ndarray,
        target: float,
    ) -> float:
        inflow_density = self._densities_after_service(state, movements, kinds, flows, BALANCE_INFLOW_KINDS)
        outflow_density = self._densities_after_service(state, movements, kinds, flows, BALANCE_OUTFLOW_KINDS)
        balance = safe_balance_index(inflow_density) ** 2 + safe_balance_index(outflow_density) ** 2
        residual = self._net_flow(flows, kinds) - target
        return float(balance + 10.0 * (residual / max(self.cfg.network.movement_capacity_veh_h, 1.0)) ** 2)

    def _objective_context(
        self,
        state: TrafficState,
        movements: list[str],
        kinds: list[str],
    ) -> _ObjectiveContext:
        """Spec 4.4의 balance objective에서 particle과 무관한 항을 한 번만 구성한다."""
        specs = movement_specs(self.cfg)
        queue = np.asarray([
            max(0.0, state.urban_movement_queue.get(movement, 0.0))
            for movement in movements
        ], dtype=float)
        signs = np.asarray([
            1.0 if kind in INFLOW_KINDS else -1.0 if kind in OUTFLOW_KINDS else 0.0
            for kind in kinds
        ], dtype=float)

        def groups_for(accepted_kinds: set[str]) -> tuple[list[np.ndarray], np.ndarray]:
            grouped_indices: Dict[str, list[int]] = {}
            grouped_caps: Dict[str, float] = {}
            for idx, movement in enumerate(movements):
                if kinds[idx] not in accepted_kinds:
                    continue
                key = boundary_group_key(specs.get(movement, {}))
                grouped_indices.setdefault(key, []).append(idx)
                grouped_caps[key] = grouped_caps.get(key, 0.0) + max(
                    self._movement_storage_capacity(movement),
                    1.0e-9,
                )
            keys = sorted(grouped_indices)
            return (
                [np.asarray(grouped_indices[key], dtype=int) for key in keys],
                np.asarray([grouped_caps[key] for key in keys], dtype=float),
            )

        inflow_groups, inflow_caps = groups_for(BALANCE_INFLOW_KINDS)
        outflow_groups, outflow_caps = groups_for(BALANCE_OUTFLOW_KINDS)
        return _ObjectiveContext(
            queue=queue,
            signs=signs,
            inflow_groups=inflow_groups,
            inflow_caps=inflow_caps,
            outflow_groups=outflow_groups,
            outflow_caps=outflow_caps,
        )

    def _objective_many(
        self,
        context: _ObjectiveContext,
        kinds: list[str],
        flows: np.ndarray,
        target: float,
    ) -> np.ndarray:
        """기존 scalar objective와 같은 값을 particle batch 단위로 계산한다."""
        del kinds  # signs는 context에 고정되어 있어 난수/탐색 순서를 바꾸지 않는다.
        rows = np.atleast_2d(flows.astype(float, copy=False))
        dt_h = self.cfg.simulation.T_c_h
        remaining = np.maximum(0.0, context.queue[None, :] - np.maximum(rows, 0.0) * dt_h)
        b_in = self._balance_index_many(remaining, context.inflow_groups, context.inflow_caps)
        b_out = self._balance_index_many(remaining, context.outflow_groups, context.outflow_caps)
        residual = rows @ context.signs - target
        scale = max(self.cfg.network.movement_capacity_veh_h, 1.0)
        return b_in * b_in + b_out * b_out + 10.0 * (residual / scale) ** 2

    @staticmethod
    def _balance_index_many(
        remaining: np.ndarray,
        groups: list[np.ndarray],
        caps: np.ndarray,
        eps: float = 1.0e-9,
    ) -> np.ndarray:
        """safe_balance_index를 여러 particle에 대해 같은 정의로 평가한다."""
        if not groups:
            return np.zeros(remaining.shape[0], dtype=float)
        values = np.column_stack([
            np.sum(remaining[:, indices], axis=1) / caps[idx]
            for idx, indices in enumerate(groups)
        ])
        l1 = np.sum(np.abs(values), axis=1)
        l2_sq = np.sum(values * values, axis=1)
        raw = l2_sq / np.maximum(l1 * l1, eps) - 1.0 / values.shape[1]
        return np.where(l1 <= eps, 0.0, np.maximum(0.0, raw))

    def _densities_after_service(
        self,
        state: TrafficState,
        movements: list[str],
        kinds: list[str],
        flows: np.ndarray,
        accepted_kinds: set[str],
    ) -> list[float]:
        """서비스 후 잔여 큐를 경계 요소(게이트/off-ramp/ramp) 단위로 집계한 밀도.

        balance 지표(movement_balance_summary)와 같은 집계라야 PSO가 게이트가
        측정하는 양을 최적화한다(β분할 movement 조각 단위 균형화는 ill-posed)."""
        specs = movement_specs(self.cfg)
        dt_h = self.cfg.simulation.T_c_h
        queues: Dict[str, float] = {}
        caps: Dict[str, float] = {}
        for idx, movement in enumerate(movements):
            if kinds[idx] not in accepted_kinds:
                continue
            key = boundary_group_key(specs.get(movement, {}))
            queue = max(0.0, state.urban_movement_queue.get(movement, 0.0))
            remaining = max(0.0, queue - max(0.0, flows[idx]) * dt_h)
            queues[key] = queues.get(key, 0.0) + remaining
            caps[key] = caps.get(key, 0.0) + max(self._movement_storage_capacity(movement), 1.0e-9)
        return [queues[key] / caps[key] for key in sorted(queues)]

    def _movement_storage_capacity(self, movement: str) -> float:
        return movement_storage_capacity(self.cfg, movement)

    @staticmethod
    def _net_flow(flows: np.ndarray, kinds: list[str]) -> float:
        net = 0.0
        for idx, kind in enumerate(kinds):
            if kind in INFLOW_KINDS:
                net += float(flows[idx])
            elif kind in OUTFLOW_KINDS:
                net -= float(flows[idx])
        return float(net)

    def _project_net_flow(
        self,
        flows: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        kinds: list[str],
        target: float,
    ) -> np.ndarray:
        out = np.clip(flows.astype(float), lower, upper)
        for _ in range(2 * len(out) + 1):
            residual = target - self._net_flow(out, kinds)
            if abs(residual) <= max(self.cfg.urban_follower.eps_U, 1.0e-9):
                break
            if residual > 0.0:
                candidates = [i for i, kind in enumerate(kinds) if kind in INFLOW_KINDS and out[i] < upper[i] - 1.0e-9]
                candidates += [i for i, kind in enumerate(kinds) if kind in OUTFLOW_KINDS and out[i] > lower[i] + 1.0e-9]
                sign = 1.0
            else:
                candidates = [i for i, kind in enumerate(kinds) if kind in INFLOW_KINDS and out[i] > lower[i] + 1.0e-9]
                candidates += [i for i, kind in enumerate(kinds) if kind in OUTFLOW_KINDS and out[i] < upper[i] - 1.0e-9]
                sign = -1.0
            if not candidates:
                break
            share = abs(residual) / len(candidates)
            for idx in candidates:
                if kinds[idx] in INFLOW_KINDS:
                    delta = sign * min(share, upper[idx] - out[idx] if sign > 0.0 else out[idx] - lower[idx])
                    out[idx] += delta
                else:
                    delta = -sign * min(share, out[idx] - lower[idx] if sign > 0.0 else upper[idx] - out[idx])
                    out[idx] += delta
        return np.clip(out, lower, upper)

    def _flow_to_green_sec(self, flow: float) -> float:
        net = self.cfg.network
        raw = max(0.0, flow) / max(net.movement_capacity_veh_h, 1.0e-9) * net.cycle_length
        return float(np.clip(raw, net.green_min, net.green_max))
