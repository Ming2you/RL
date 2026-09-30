"""정확한 coupled rollout을 사용하는 두 지역 sensitivity/SQP DMPC 실험.

상태를 제거한 constant move-block 문제이며 물리 목적은 순수 TTT이다.
원 S-DMPC의 로컬 nonlinear NLP 대신 1차 일관적인 proximal SQP 모델을 쓴다.
공유 resource QP와 nonlinear 재검증은 가격만으로 등식 만족을 가정하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import lsq_linear, minimize

from src.controllers.leader import Leader
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.simulation.coupling import run_coupled_interval
from src.simulation.player_cost_accounting import freeway_buffer_state_values, freeway_buffer_vehicle_counts


@dataclass(frozen=True)
class Budget:
    np_change_veh: float
    nuf_veh_h: float

    def vector(self) -> np.ndarray:
        return np.array([self.np_change_veh, self.nuf_veh_h], dtype=float)


@dataclass
class SDMPCOptions:
    horizon_steps: int = 3
    max_iterations: int = 8
    local_max_iterations: int = 1
    tolerance_np_veh: float = 1.0
    tolerance_nuf_veh_h: float = 20.0
    trust_radius: float = 0.15
    min_trust_radius: float = 0.002
    proximal: float = 10.0
    stationarity_tolerance: float = 0.001
    fd_green_sec: float = 1.0
    fd_offset_sec: float = 1.0
    fd_metering_veh_h: float = 30.0
    fd_allocation_veh_h: float = 30.0
    budget_scale_np_veh: float = 100.0
    budget_scale_nuf_veh_h: float = 1000.0
    restoration_iterations: int = 3
    line_search_steps: int = 4
    max_candidates: int = 3
    optimize_allocation: bool = True
    externality_enabled: bool = True
    discrete_poll: bool = True
    discrete_poll_every: int = 2
    objective_improvement_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        for name in ("horizon_steps", "max_iterations", "local_max_iterations", "line_search_steps", "max_candidates", "discrete_poll_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("tolerance_np_veh", "tolerance_nuf_veh_h", "trust_radius", "min_trust_radius", "proximal", "stationarity_tolerance", "fd_green_sec", "fd_offset_sec", "fd_metering_veh_h", "fd_allocation_veh_h", "budget_scale_np_veh", "budget_scale_nuf_veh_h"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.restoration_iterations < 0:
            raise ValueError("restoration_iterations must be nonnegative")


@dataclass
class RolloutEvaluation:
    total_ttt: float
    regional_costs: dict[str, float]
    achieved_np_change_veh: float
    achieved_nuf_veh_h: float
    first_state: TrafficState
    states: list[TrafficState]
    diagnostics: dict[str, Any]
    control_valid: bool
    physical_valid: bool
    evaluation_seconds: float

    @property
    def cost_vector(self) -> np.ndarray:
        return np.array([self.regional_costs["F"], self.regional_costs["U"]])

    @property
    def budget_vector(self) -> np.ndarray:
        return np.array([self.achieved_np_change_veh, self.achieved_nuf_veh_h])


@dataclass
class SDMPCResult:
    control: ControlAction
    objective: float
    residual_np_veh: float
    residual_nuf_veh_h: float
    status: str
    converged: bool
    feasible: bool
    iterations: int
    evaluation: RolloutEvaluation
    iteration_rows: list[dict[str, Any]] = field(default_factory=list)
    sensitivity_rows: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = ""


class NoExecutableControlError(RuntimeError):
    """모든 후보가 원 budget/물리/actuator 실행 검사를 실패했을 때 발생."""


@dataclass
class ResourceStep:
    step: np.ndarray
    multiplier: np.ndarray
    linear_residual: float
    success: bool
    message: str


def joint_resource_step(
    local_response: np.ndarray,
    jacobian: np.ndarray,
    residual: np.ndarray,
    lower_step: np.ndarray,
    upper_step: np.ndarray,
    proximal: float = 1.0,
) -> ResourceStep:
    """모든 지역의 움직임을 합치는 bounded equality resource QP.

    min 0.5*tau*||d-d_local||^2, A*d=-c. 승수 부호는 +nu*(A*d+c).
    상수/종속 행은 SVD로 제거하되 원 A*d+c를 마지막에 직접 검사한다.
    선형화 불가능은 전역 물리 infeasibility의 증명이 아니다.
    """
    p = np.asarray(local_response, dtype=float)
    A = np.asarray(jacobian, dtype=float)
    c = np.asarray(residual, dtype=float)
    lo, hi = np.asarray(lower_step, dtype=float), np.asarray(upper_step, dtype=float)
    if np.any(lo > hi):
        raise ValueError("invalid resource bounds")
    U, singular, _ = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(singular > max(1.0e-10, (singular[0] if len(singular) else 0.0) * 1.0e-9)))
    Ar, cr = U[:, :rank].T @ A, U[:, :rank].T @ c
    constraints = [] if rank == 0 else [{
        "type": "eq", "fun": lambda d: Ar @ d + cr, "jac": lambda d: Ar,
    }]
    out = minimize(
        lambda d: 0.5 * proximal * float(np.dot(d - p, d - p)),
        np.clip(p, lo, hi), jac=lambda d: proximal * (d - p),
        bounds=list(zip(lo, hi)), constraints=constraints, method="SLSQP",
        options={"maxiter": 80, "ftol": 1.0e-11},
    )
    d = np.clip(out.x, lo, hi)
    violation = float(np.max(np.abs(A @ d + c), initial=0.0))
    if violation > 1.0e-6:
        # QP equality가 실패하면 bounded 선형 least-squares 복원 방향만 반환한다.
        # 실패 표시를 유지하므로 이 step이 feasible certificate가 되지 않는다.
        free = (hi - lo) > 1.0e-12
        d = np.clip(np.zeros_like(p), lo, hi)
        if np.any(free):
            fit = lsq_linear(A[:, free], -c - A[:, ~free] @ d[~free], bounds=(lo[free], hi[free]), tol=1.0e-10)
            d[free] = fit.x
        violation = float(np.max(np.abs(A @ d + c), initial=0.0))
    # 활성 actuator bound의 법선은 equality 승수로 잘못 흡수하지 않는다.
    free = (d > lo + 1.0e-7) & (d < hi - 1.0e-7)
    multiplier = np.zeros(len(c))
    if np.any(free) and rank:
        multiplier = np.linalg.lstsq(A[:, free].T, -proximal * (d[free] - p[free]), rcond=1.0e-9)[0]
    return ResourceStep(d, multiplier, violation, bool(violation <= 1.0e-6 and out.success), str(out.message))


def finite_difference_model(
    evaluate: Callable[[np.ndarray], RolloutEvaluation],
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normalized_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """동일한 +/- full rollout에서 두 지역 total derivative와 자원 Jacobian 획득.

    bound에서는 실제 stencil 폭으로 one-sided/비대칭 차분한다. dynamics를 이미
    제거했으므로 이 기울기에 별도 dynamics multiplier를 더하면 중복이다.
    """
    gradients = np.zeros((2, len(value)))
    jacobian = np.zeros((2, len(value)))
    for j in range(len(value)):
        hi, lo = value.copy(), value.copy()
        hi[j] = min(upper[j], value[j] + normalized_steps[j])
        lo[j] = max(lower[j], value[j] - normalized_steps[j])
        span = hi[j] - lo[j]
        if span <= 1.0e-12:
            continue
        ehi, elo = evaluate(hi), evaluate(lo)
        gradients[:, j] = (ehi.cost_vector - elo.cost_vector) / span
        jacobian[:, j] = (ehi.budget_vector - elo.budget_vector) / span
    return gradients, jacobian


def physical_inventory(state: TrafficState, cfg: ExperimentConfig) -> float:
    """off-ramp 재귀속·실제 freeway 완충 포함; 도시 예약/legacy 중복 view는 제외."""
    return (state.total_freeway_vehicles(cfg.network) + state.total_urban_vehicles(cfg.network)
            + state.off_ramp_storage_occupancy_veh(cfg.network)
            + sum(freeway_buffer_vehicle_counts(state, cfg).values()))


def validate_control(control: ControlAction, previous: ControlAction, cfg: ExperimentConfig) -> dict[str, Any]:
    """구간 실행 actuator 제한을 검사한다. offset 변화는 원 위의 최단거리이다."""
    net, eps = cfg.network, 1.0e-7
    errors: list[str] = []
    for r in net.ramps:
        x = control.ramp_metering.get(r, 0.0)
        if not math.isfinite(x) or not -eps <= x <= net.ramp_capacity_veh_h[r] + eps:
            errors.append(f"metering:{r}")
    for s in net.signals:
        g1, g2 = control.green_times.get(f"{s}_p1", 0.0), control.green_times.get(f"{s}_p2", 0.0)
        if not all(math.isfinite(g) and net.green_min - eps <= g <= net.green_max + eps for g in (g1, g2)) or abs(g1 + g2 - net.effective_green_total) > eps:
            errors.append(f"green:{s}")
        offset = control.offsets.get(s, 0.0)
        delta = (offset - previous.offsets.get(s, 0.0) + net.cycle_length / 2) % net.cycle_length - net.cycle_length / 2
        if not math.isfinite(offset) or not 0 <= offset < net.cycle_length or abs(delta) > cfg.urban_follower.max_offset_step + eps:
            errors.append(f"offset:{s}")
    for link in net.freeway_links:
        v = control.vsl.get(link, max(cfg.freeway_follower.vsl_set))
        old = previous.vsl.get(link, max(cfg.freeway_follower.vsl_set))
        if v not in cfg.freeway_follower.vsl_set or abs(v - old) > cfg.freeway_follower.max_vsl_step + eps:
            errors.append(f"vsl:{link}")
    for key, value in control.inflow_outflow_allocation.items():
        if not math.isfinite(value) or not -eps <= value <= net.movement_capacity_veh_h + eps:
            errors.append(f"allocation:{key}")
    return {"valid": not errors, "violations": errors}


class _ControlCoordinates:
    """물리 스케일을 명시한 continuous 축; VSL은 별도 이산 poll로만 변경."""

    def __init__(self, cfg: ExperimentConfig, options: SDMPCOptions, previous: ControlAction):
        self.cfg, self.options, self.previous = cfg, options, previous.copy()
        net = cfg.network
        # tuple(kind,key,scale,low,high,physical FD step,owner)
        axes: list[tuple[str, str, float, float, float, float, int]] = []
        for r in net.ramps:
            cap = net.ramp_capacity_veh_h[r]
            axes.append(("meter", r, cap, 0.0, cap, options.fd_metering_veh_h, 0))
        for s in net.signals:
            lower = max(net.green_min, net.effective_green_total - net.green_max)
            upper = min(net.green_max, net.effective_green_total - net.green_min)
            axes.append(("green", s, net.cycle_length, lower, upper, options.fd_green_sec, 1))
            step = cfg.urban_follower.max_offset_step
            axes.append(("offset", s, net.cycle_length, -step, step, options.fd_offset_sec, 1))
        if options.optimize_allocation:
            for gate in net.boundary_in_links:
                axes.append(("gate", gate, net.movement_capacity_veh_h, 0.0, net.movement_capacity_veh_h, options.fd_allocation_veh_h, 1))
            for r in net.ramps:
                axes.append(("ramp_allocation", r, net.movement_capacity_veh_h, 0.0, net.movement_capacity_veh_h, options.fd_allocation_veh_h, 1))
        self.axes = axes
        self.scales = np.array([a[2] for a in axes])
        self.lower = np.array([a[3] / a[2] for a in axes])
        self.upper = np.array([a[4] / a[2] for a in axes])
        self.fd = np.array([a[5] / a[2] for a in axes])
        self.owners = np.array([a[6] for a in axes])

    def encode(self, control: ControlAction) -> np.ndarray:
        net = self.cfg.network
        values = []
        for kind, key, scale, _, _, _, _ in self.axes:
            if kind == "meter":
                value = control.ramp_metering.get(key, scale)
            elif kind == "green":
                value = control.green_times.get(f"{key}_p1", net.effective_green_total / 2)
            elif kind == "offset":
                value = (control.offsets.get(key, 0.0) - self.previous.offsets.get(key, 0.0) + net.cycle_length / 2) % net.cycle_length - net.cycle_length / 2
            elif kind == "gate":
                value = control.inflow_outflow_allocation.get(key, scale)
            else:
                values_m = [control.inflow_outflow_allocation.get(m, scale) for m in net.on_ramp_to_movement.get(key, [])]
                value = float(np.mean(values_m)) if values_m else scale
            values.append(value / scale)
        return np.clip(values, self.lower, self.upper)

    def decode(self, values: np.ndarray, vsl: dict[str, float]) -> ControlAction:
        net = self.cfg.network
        c = ControlAction.uncontrolled(self.cfg)
        c.vsl = dict(vsl)
        # 모든 perimeter의 capacity를 명시적으로 기록한다. green fraction은 plant에서
        # 한번만 곱하며 여기서는 saturation cap[veh/h]을 전달한다.
        c.inflow_outflow_allocation = {m: net.movement_capacity_veh_h for m, spec in net.urban_movements.items() if spec.get("kind") != "internal"}
        for raw, axis in zip(values, self.axes):
            kind, key, scale, _, _, _, _ = axis
            value = float(raw * scale)
            if kind == "meter":
                c.ramp_metering[key] = value
            elif kind == "green":
                c.green_times[f"{key}_p1"] = value
                c.green_times[f"{key}_p2"] = net.effective_green_total - value
            elif kind == "offset":
                offset = (self.previous.offsets.get(key, 0.0) + value) % net.cycle_length
                # 0 근처 상쇄오차가 작은 음수이면 float modulo가 정확히 cycle로
                # 반올림될 수 있다. 같은 위상인 0으로 정규화해 [0,cycle)를 보존한다.
                c.offsets[key] = 0.0 if offset >= net.cycle_length else offset
            elif kind == "gate":
                c.inflow_outflow_allocation[key] = value
                for m, spec in net.urban_movements.items():
                    if spec.get("kind") == "boundary_in" and str(spec.get("origin", "")) == key:
                        c.inflow_outflow_allocation[m] = value
            else:
                for m in net.on_ramp_to_movement.get(key, []):
                    c.inflow_outflow_allocation[m] = value
        return c

    def initial_vsl(self, control: ControlAction) -> dict[str, float]:
        result = {}
        for link in self.cfg.network.freeway_links:
            old = self.previous.vsl.get(link, max(self.cfg.freeway_follower.vsl_set))
            allowed = [v for v in self.cfg.freeway_follower.vsl_set if abs(v - old) <= self.cfg.freeway_follower.max_vsl_step + 1.0e-9]
            desired = control.vsl.get(link, old)
            result[link] = float(min(allowed, key=lambda v: (abs(v - desired), -v)))
        return result


class SensitivityDMPC:
    def __init__(self, cfg: ExperimentConfig, options: SDMPCOptions | None = None):
        self.cfg, self.options = cfg, options or SDMPCOptions()
        self.candidate_rows: list[dict[str, Any]] = []
        self.last_results: list[SDMPCResult] = []
        self.last_result: SDMPCResult | None = None
        self.previous_control: ControlAction | None = None
        # 외부 runner가 PFO 등으로 만든 초기값을 명시적으로 전달할 수 있다.
        # 이 값의 rollout은 witness 생성이며 하위 수렴 판정을 대신하지 않는다.
        self.warm_start_control: ControlAction | None = None

    def evaluate_control(self, state: TrafficState, forecast: Sequence[DemandStep], control: ControlAction, previous: ControlAction | None = None) -> RolloutEvaluation:
        if not forecast:
            raise ValueError("nonempty demand forecast required")
        tic, cfg = time.perf_counter(), self.cfg
        s = state.copy()
        inventory_start = physical_inventory(s, cfg)
        protected_start = s.protected_accumulation_veh(cfg.network)
        costs = {"F": 0.0, "U": 0.0}
        states, diag_rows = [], []
        external_exits = 0.0
        for demand in list(forecast)[:self.options.horizon_steps]:
            # Spec 3.4의 plant 순서/시간격자는 수정하지 않는다. 자체 terminal reward 없음.
            result = run_coupled_interval(s, control, demand, cfg)
            s.time_sec += cfg.simulation.T_c_sec
            costs["F"] += result.freeway_ttt
            costs["U"] += result.urban_ttt
            external_exits += result.diagnostics.get("mainline_exit_flow_total", 0.0) * cfg.simulation.T_c_h + result.diagnostics.get("boundary_out_sink_veh", 0.0)
            states.append(s.copy())
            diag_rows.append(result.diagnostics)
        validation = validate_control(control, previous or control, cfg)
        nonnegative = all(
            min([0.0] + list(st.ramp_queue.values()) + list(st.urban_movement_queue.values()) + list(st.mainline_origin_queue.values()) + [x for xs in st.freeway_density.values() for x in xs] + freeway_buffer_state_values(st)[0]) >= -1.0e-7
            for st in states
        )
        finite_states = all(np.all(np.isfinite(
            list(st.ramp_queue.values()) + list(st.urban_movement_queue.values()) + list(st.mainline_origin_queue.values())
            + list(st.urban_link_storage.values()) + [x for xs in st.freeway_density.values() for x in xs]
            + [x for xs in st.freeway_speed.values() for x in xs]
            + freeway_buffer_state_values(st)[0] + freeway_buffer_state_values(st)[1]
        )) for st in states)
        storage_valid = all(all(-1.0e-7 <= st.urban_link_storage.get(k, cap) <= cap + 1.0e-7 for k, cap in cfg.network.urban_link_storage_veh.items()) for st in states)
        numerical_loss = sum(float(d.get("movement_queue_projection_veh", 0.0)) + float(d.get("coupling_offramp_arrivals_rejected_veh", 0.0)) for d in diag_rows)
        diagnostics = {
            "first_interval": diag_rows[0], "intervals": diag_rows,
            "inventory_start_veh": inventory_start, "inventory_end_veh": physical_inventory(s, cfg),
            "external_exit_veh": external_exits, "numerical_vehicle_loss_veh": numerical_loss,
            "control_validation": validation, "horizon_steps": len(states),
            "commanded_nuf_veh_h": sum(control.ramp_metering.values()),
        }
        return RolloutEvaluation(
            costs["F"] + costs["U"], costs,
            states[0].protected_accumulation_veh(cfg.network) - protected_start,
            float(diag_rows[0].get("total_metering_flow", 0.0)), states[0], states,
            diagnostics, bool(validation["valid"]), bool(finite_states and np.all(np.isfinite(list(costs.values()))) and nonnegative and storage_valid and numerical_loss <= 1.0e-6), time.perf_counter() - tic,
        )

    def _scales(self) -> np.ndarray:
        return np.array([self.options.budget_scale_np_veh, self.options.budget_scale_nuf_veh_h])

    def _feasible(self, evaluation: RolloutEvaluation, budget: Budget) -> bool:
        residual = evaluation.budget_vector - budget.vector()
        return bool(evaluation.control_valid and evaluation.physical_valid and math.isfinite(evaluation.total_ttt) and np.all(np.isfinite(residual))
                    and abs(residual[0]) <= self.options.tolerance_np_veh and abs(residual[1]) <= self.options.tolerance_nuf_veh_h)

    def _residual_norm(self, evaluation: RolloutEvaluation, budget: Budget) -> float:
        return float(np.max(np.abs((evaluation.budget_vector - budget.vector()) / self._scales())))

    def solve_fixed_budget(self, state: TrafficState, forecast: Sequence[DemandStep], previous: ControlAction | None, budget: Budget, initial_control: ControlAction | None = None) -> SDMPCResult:
        opts, cfg = self.options, self.cfg
        if not np.all(np.isfinite(budget.vector())):
            raise ValueError("budget must be finite")
        previous = (previous or ControlAction.uncontrolled(cfg)).copy()
        coords = _ControlCoordinates(cfg, opts, previous)
        initial = initial_control or previous
        y, vsl = coords.encode(initial), coords.initial_vsl(initial)
        cache: dict[tuple[float, ...], RolloutEvaluation] = {}
        evaluations = 0

        def evaluate(z: np.ndarray, speed: dict[str, float] | None = None) -> RolloutEvaluation:
            nonlocal evaluations
            speed = vsl if speed is None else speed
            key = tuple(np.round(z, 12)) + tuple(speed[k] for k in cfg.network.freeway_links)
            if key not in cache:
                cache[key] = self.evaluate_control(state, forecast, coords.decode(z, speed), previous)
                evaluations += 1
            return cache[key]

        current = evaluate(y)
        rows: list[dict[str, Any]] = []
        sensitivities: list[dict[str, Any]] = []
        # 단순하고 엄밀한 물리 bound만 infeasible_proven으로 분류한다.
        impossible = (budget.nuf_veh_h < 0.0
                      or budget.nuf_veh_h > cfg.network.total_ramp_capacity + 1.0e-9
                      or budget.np_change_veh < -state.protected_accumulation_veh(cfg.network) - 1.0e-9)
        best = (y.copy(), dict(vsl), current) if self._feasible(current, budget) else None
        multipliers, trust = np.zeros(2), opts.trust_radius
        converged, reason = False, "iteration_limit"
        discrete_complete = not opts.discrete_poll
        last_stationarity = float("inf")

        def restore(z: np.ndarray, speed: dict[str, float], A_start: np.ndarray) -> tuple[np.ndarray, RolloutEvaluation, int]:
            """선형 budget correction 이후 original nonlinear rollout 잔차를 줄인다.

            intermediate infeasible 시도는 허용하지만 실행 후보에는 넣지 않는다.
            작은 local Broyden 업데이트로 FD를 반복하지 않으며 실패를 숨기지 않는다.
            """
            ev = evaluate(z, speed)
            A = A_start.copy()
            for k in range(opts.restoration_iterations):
                if self._feasible(ev, budget):
                    return z, ev, k
                c = (ev.budget_vector - budget.vector()) / self._scales()
                correction = joint_resource_step(np.zeros_like(z), A, c, np.maximum(coords.lower - z, -trust), np.minimum(coords.upper - z, trust))
                if np.max(np.abs(correction.step), initial=0.0) < 1.0e-10:
                    break
                accepted = False
                for alpha in (1.0, 0.5, 0.25):
                    trial = np.clip(z + alpha * correction.step, coords.lower, coords.upper)
                    e_trial = evaluate(trial, speed)
                    if self._residual_norm(e_trial, budget) < self._residual_norm(ev, budget) - 1.0e-10:
                        d = trial - z
                        change = (e_trial.budget_vector - ev.budget_vector) / self._scales()
                        A += np.outer(change - A @ d, d) / max(float(d @ d), 1.0e-20)
                        z, ev, accepted = trial, e_trial, True
                        break
                if not accepted:
                    break
            return z, ev, opts.restoration_iterations

        def log_row(iteration: int, accepted: bool, stationarity: float, step_norm: float, qp: ResourceStep | None, restoration: int, polled: bool) -> None:
            residual = current.budget_vector - budget.vector()
            rows.append({
                "iteration": iteration, "objective_ttt": current.total_ttt,
                "freeway_ttt": current.regional_costs["F"], "urban_ttt": current.regional_costs["U"],
                "requested_np_change_veh": budget.np_change_veh, "requested_nuf_veh_h": budget.nuf_veh_h,
                "achieved_np_change_veh": current.achieved_np_change_veh, "achieved_nuf_veh_h": current.achieved_nuf_veh_h,
                "residual_np_veh": float(residual[0]), "residual_nuf_veh_h": float(residual[1]),
                "residual_scaled_inf": self._residual_norm(current, budget),
                "feasible": not impossible and self._feasible(current, budget), "accepted": accepted,
                "stationarity_norm": stationarity, "step_norm": step_norm, "trust_radius": trust,
                "lambda_np_h": float(multipliers[0] / self._scales()[0]),
                "lambda_nuf_h2": float(multipliers[1] / self._scales()[1]),
                "resource_qp_success": qp.success if qp else False,
                "linear_residual_scaled_inf": qp.linear_residual if qp else 0.0,
                "restoration_iterations": restoration, "discrete_poll_complete": polled,
                "rollout_evaluations": evaluations, "local_model": "first_order_consistent_proximal_SQP",
                "local_iterations": 1,
            })

        log_row(0, True, last_stationarity, 0.0, None, 0, False)
        if impossible:
            reason = "physical_bound_violation"
        else:
            for iteration in range(1, opts.max_iterations + 1):
                gradients, A_physical = finite_difference_model(lambda z: evaluate(z), y, coords.lower, coords.upper, coords.fd)
                A = A_physical / self._scales()[:, None]
                own = gradients[coords.owners, np.arange(len(y))]
                externality = gradients.sum(axis=0) - own
                g = own + (externality if opts.externality_enabled else 0.0)
                # 두 로컬 모델은 같은 incumbent에서 별도로 응답한다(Jacobi emulation).
                # own의 SQP linearization + 외부효과 + 공통 승수 + proximal. 공유 등식은
                # 로컬 단일 블록을 얼리는 대신 아래 공동 resource QP에서만 부과한다.
                budget_price = A.T @ multipliers
                lambda_used = multipliers.copy()
                local_response = np.zeros_like(y)
                for owner in (0, 1):
                    mask = coords.owners == owner
                    local_response[mask] = -(g[mask] + budget_price[mask]) / opts.proximal
                c = (current.budget_vector - budget.vector()) / self._scales()
                qp = joint_resource_step(local_response, A, c, np.maximum(coords.lower - y, -trust), np.minimum(coords.upper - y, trust), opts.proximal)
                # correction multiplier는 기존 lambda에 더해져 원 QP KKT 부호와 일치한다.
                if qp.success:
                    multipliers = multipliers + qp.multiplier
                for j, (kind, key, scale, _, _, _, owner) in enumerate(coords.axes):
                    sensitivities.append({
                        "iteration": iteration, "region": "F" if owner == 0 else "U", "control": f"{kind}:{key}",
                        "physical_scale": scale, "own_gradient": float(own[j] / scale),
                        "externality_gradient": float(externality[j] / scale),
                        "budget_gradient": float(budget_price[j] / scale),
                        "budget_gradient_next": float((A.T @ multipliers)[j] / scale),
                        "total_ttt_gradient": float(gradients[:, j].sum() / scale),
                        "d_np_d_control": float(A_physical[0, j] / scale),
                        "d_nuf_d_control": float(A_physical[1, j] / scale),
                        "lambda_np_h": float(lambda_used[0] / self._scales()[0]),
                        "lambda_nuf_h2": float(lambda_used[1] / self._scales()[1]),
                        "lambda_np_next_h": float(multipliers[0] / self._scales()[0]),
                        "lambda_nuf_next_h2": float(multipliers[1] / self._scales()[1]),
                        "resource_price_update_valid": qp.success,
                        "externality_enabled": opts.externality_enabled,
                    })
                # stationarity는 축소된 trust 반경으로 판정하지 않는다. 원 box의 tangent
                # projected gradient를 별도 QP로 계산하여 작은 trust/정체를 수렴이라 하지 않는다.
                stationarity_qp = joint_resource_step(-g / opts.proximal, A, np.zeros(2), coords.lower - y, coords.upper - y, opts.proximal)
                # gradient mapping=tau*d: proximal만 크게 설정해 d를 축소하는 것이
                # 가짜 stationarity가 되지 않도록 물리 TTT gradient 스케일을 복원한다.
                last_stationarity = opts.proximal * float(np.max(np.abs(stationarity_qp.step))) if stationarity_qp.success else float("inf")
                old_y, old_objective = y.copy(), current.total_ttt
                accepted, restoration_count = False, 0
                for ls in range(opts.line_search_steps):
                    trial = np.clip(y + (0.5 ** ls) * qp.step, coords.lower, coords.upper)
                    trial, e_trial, restorations = restore(trial, vsl, A)
                    restoration_count += restorations
                    improvement = e_trial.total_ttt < current.total_ttt - opts.objective_improvement_tolerance
                    if self._feasible(e_trial, budget) and (best is None or improvement):
                        y, current, accepted = trial, e_trial, True
                        break
                    if best is None and e_trial.control_valid and e_trial.physical_valid and self._residual_norm(e_trial, budget) < self._residual_norm(current, budget) - 1.0e-9:
                        y, current, accepted = trial, e_trial, True
                        break
                if self._feasible(current, budget) and (best is None or current.total_ttt < best[2].total_ttt):
                    best = y.copy(), dict(vsl), current
                # VSL는 feasible 이산 set 전체의 인접값을 poll. 변경마다 continuous
                # budget를 복원하고 nonlinear 공유 잔차를 재검사한다.
                do_poll = opts.discrete_poll and (iteration % opts.discrete_poll_every == 0 or last_stationarity <= opts.stationarity_tolerance or iteration == opts.max_iterations)
                discrete_complete, discrete_improved = not opts.discrete_poll, False
                if do_poll and self._feasible(current, budget):
                    discrete_complete = True
                    base_y, base_vsl, base_eval = y.copy(), dict(vsl), current
                    for link in cfg.network.freeway_links:
                        old_v = previous.vsl.get(link, max(cfg.freeway_follower.vsl_set))
                        allowed = sorted(v for v in cfg.freeway_follower.vsl_set if abs(v - old_v) <= cfg.freeway_follower.max_vsl_step + 1.0e-9)
                        idx = allowed.index(base_vsl[link])
                        neighbors = allowed[max(0, idx - 1):idx] + allowed[idx + 1:idx + 2]
                        for value in neighbors:
                            speeds = dict(base_vsl)
                            speeds[link] = value
                            trial, e_trial, restorations = restore(base_y.copy(), speeds, A)
                            restoration_count += restorations
                            if not self._feasible(e_trial, budget):
                                # 복원 실패는 해당 discrete fiber에 feasible 개선이 없다는 증명 아님.
                                discrete_complete = False
                            elif e_trial.total_ttt < current.total_ttt - opts.objective_improvement_tolerance:
                                y, vsl, current = trial, speeds, e_trial
                                accepted, discrete_improved = True, True
                    if discrete_improved:
                        discrete_complete = False
                        best = y.copy(), dict(vsl), current
                log_row(iteration, accepted, last_stationarity, float(np.max(np.abs(y - old_y))), qp, restoration_count, discrete_complete)
                # 기울기는 old_y에서 얻었으므로 새 해로 이동한 반복에는 수렴 판정을 미룬다.
                if self._feasible(current, budget) and not discrete_improved and np.max(np.abs(y - old_y)) <= opts.stationarity_tolerance and last_stationarity <= opts.stationarity_tolerance and discrete_complete:
                    converged, reason = True, "feasible_projected_stationarity_and_discrete_poll"
                    break
                if not accepted:
                    trust *= 0.5
                    if trust < opts.min_trust_radius:
                        reason = "trust_region_stall_not_convergence"
                        break
                elif old_objective - current.total_ttt > opts.objective_improvement_tolerance:
                    trust = min(opts.trust_radius, trust * 1.25)

        if best is not None:
            y, vsl, current = best
        control = coords.decode(y, vsl)
        control.N_P_star, control.N_UF_star = budget.np_change_veh, budget.nuf_veh_h
        feasible = not impossible and self._feasible(current, budget)
        status = "infeasible_proven" if impossible else ("converged_feasible" if feasible and converged else "feasible_but_not_converged" if feasible else "unresolved_algorithm_failure")
        residual = current.budget_vector - budget.vector()
        control.diagnostics.update({
            "sdmpc_active": True, "sdmpc_feasible": feasible, "sdmpc_converged": feasible and converged,
            "budget_contract_failed": not feasible, "fallback": not feasible,
            "sdmpc_residual_np_veh": float(residual[0]), "sdmpc_residual_nuf_veh_h": float(residual[1]),
            "sdmpc_achieved_np_change_veh": current.achieved_np_change_veh,
            "sdmpc_achieved_nuf_veh_h": current.achieved_nuf_veh_h,
            "sdmpc_rollout_evaluations": evaluations, "sdmpc_stationarity_norm": last_stationarity,
            "sdmpc_lambda_np_h": float(multipliers[0] / self._scales()[0]),
            "sdmpc_lambda_nuf_h2": float(multipliers[1] / self._scales()[1]),
            "urban_net_inflow_target_veh_h": budget.np_change_veh / cfg.simulation.T_c_h,
            "sdmpc_first_interval_budget": True, "sdmpc_constant_move_block": True,
        })
        return SDMPCResult(control, current.total_ttt, float(residual[0]), float(residual[1]), status, feasible and converged, feasible, len(rows) - 1, current, rows, sensitivities, reason)

    def decide(self, state: TrafficState, forecast: Sequence[DemandStep], previous: ControlAction | None = None, cfg: ExperimentConfig | None = None) -> ControlAction:
        previous = (previous or self.previous_control or ControlAction.uncontrolled(self.cfg)).copy()
        coords = _ControlCoordinates(self.cfg, self.options, previous)
        hint = self.warm_start_control or previous
        warm = coords.decode(coords.encode(hint), coords.initial_vsl(hint))
        self.warm_start_control = None
        witness = self.evaluate_control(state, forecast, warm, previous)
        witness_budget = Budget(witness.achieved_np_change_veh, witness.achieved_nuf_veh_h)
        candidates: list[tuple[Budget, str]] = [(witness_budget, "warm_start_rollout_witness")]
        # 기존 상위 candidate 생성만 재사용한다. 그 뒤 raw pair를 project/완화하지 않는다.
        raw = Leader(self.cfg).candidates(state, previous, forecast=list(forecast))
        raw = sorted(raw, key=lambda b: (abs(b.N_P_star - witness_budget.np_change_veh) / self._scales()[0] + abs(b.N_UF_star - witness_budget.nuf_veh_h) / self._scales()[1], b.N_P_star, b.N_UF_star))
        for action in raw:
            if len(candidates) >= self.options.max_candidates:
                break
            budget = Budget(action.N_P_star, action.N_UF_star)
            if budget not in [b for b, _ in candidates]:
                candidates.append((budget, "existing_leader_raw_pair"))
            if len(candidates) >= self.options.max_candidates:
                break
        self.last_results, self.candidate_rows = [], []
        for index, (budget, origin) in enumerate(candidates):
            start = time.perf_counter()
            result = self.solve_fixed_budget(state, forecast, previous, budget, warm)
            self.last_results.append(result)
            self.candidate_rows.append({
                "candidate": index, "origin": origin,
                "requested_np_change_veh": budget.np_change_veh, "requested_nuf_veh_h": budget.nuf_veh_h,
                "achieved_np_change_veh": result.evaluation.achieved_np_change_veh,
                "achieved_nuf_veh_h": result.evaluation.achieved_nuf_veh_h,
                "residual_np_veh": result.residual_np_veh, "residual_nuf_veh_h": result.residual_nuf_veh_h,
                "objective_ttt": result.objective, "status": result.status, "feasible": result.feasible,
                "converged": result.converged, "iterations": result.iterations,
                "termination_reason": result.termination_reason, "computation_time_sec": time.perf_counter() - start,
            })
        # 최종 실행 경계에서 원 residual과 physical/actuator gate를 다시 확인한다.
        # 하위 탐색의 실패 제어안은 진단용 결과일 뿐 operational fallback이 아니다.
        eligible = [r for r in self.last_results if r.feasible and self._feasible(
            r.evaluation, Budget(r.control.N_P_star, r.control.N_UF_star))]
        if not eligible:
            self.last_result = None
            for row in self.candidate_rows:
                row["selected"] = False
            # 이전 실행 제어와 현재 state를 변경하지 않은 채 원 후보/반복 근거를 남긴다.
            raise NoExecutableControlError(
                "No candidate satisfies the original budget, physical and actuator execution constraints."
            )
        chosen = min(eligible, key=lambda r: r.objective)
        self.last_result = chosen
        self.previous_control = chosen.control.copy()
        chosen_index = self.last_results.index(chosen)
        for row in self.candidate_rows:
            row["selected"] = row["candidate"] == chosen_index
        return chosen.control.copy()
