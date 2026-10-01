"""정본 7개 응답 player의 nonlinear-own S-DMPC 실험.

Spec 03/04/10/12/15 및 player_solver_design.md를 사용한다. 사용자 지정 순수
TTT/정본 권한/NP horizon cap·UF command equality가 과거 proxy 규칙에 우선한다.
지역 own TTT를 매 trial 실제 결합 plant로 계산하며 master는 자원 복원만 한다.
단일 프로세스의 동시-incumbent 응답 모사, H구간 constant move block이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import lsq_linear, minimize

from src.controllers.leader import Leader
from src.controllers.relaxed_quantization import repair_vsl_value
from src.controllers.sensitivity_dmpc import NoExecutableControlError, ResourceStep, physical_inventory
from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl
from src.simulation.coupling import run_coupled_interval
from src.simulation.player_cost_accounting import PlayerCostLedger, PlayerCostOwnership, freeway_buffer_state_values


@dataclass(frozen=True)
class PlayerBudget:
    np_cap_veh: float
    nuf_veh_h: float

    def vector(self) -> np.ndarray:
        return np.array([self.np_cap_veh, self.nuf_veh_h], dtype=float)


@dataclass
class PlayerSDMPCOptions:
    horizon_steps: int = 3
    max_iterations: int = 4
    local_solver: str = "slsqp"
    local_max_iterations: int = 4
    local_max_evaluations: int = 32
    local_tolerance: float = 0.001
    local_mesh_min: float = 0.002
    tolerance_np_veh: float = 0.1
    tolerance_nuf_veh_h: float = 2.0
    trust_radius: float = 0.15
    min_trust_radius: float = 0.002
    proximal: float = 10.0
    stationarity_tolerance: float = 0.001
    fd_green_sec: float = 1.0
    fd_offset_sec: float = 1.0
    fd_metering_veh_h: float = 30.0
    budget_scale_np_veh: float = 100.0
    budget_scale_nuf_veh_h: float = 1000.0
    dual_step: float = 0.5
    restoration_iterations: int = 3
    line_search_steps: int = 4
    discrete_poll: bool = True
    discrete_poll_every: int = 2
    max_candidates: int = 3
    externality_enabled: bool = True
    objective_improvement_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.local_solver not in {"slsqp", "pattern"}:
            raise ValueError("local_solver must be slsqp or pattern")
        for name in ("horizon_steps", "max_iterations", "local_max_iterations", "local_max_evaluations", "line_search_steps", "discrete_poll_every", "max_candidates"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("local_tolerance", "local_mesh_min", "tolerance_np_veh", "tolerance_nuf_veh_h", "trust_radius", "min_trust_radius", "proximal", "stationarity_tolerance", "fd_green_sec", "fd_offset_sec", "fd_metering_veh_h", "budget_scale_np_veh", "budget_scale_nuf_veh_h", "dual_step"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.restoration_iterations < 0:
            raise ValueError("restoration_iterations must be nonnegative")


@dataclass
class PlayerRolloutEvaluation:
    total_ttt: float
    player_costs: dict[str, float]
    regional_costs: dict[str, float]
    rawpassivecosts: dict[str, float]
    achieved_np_veh: float
    commanded_nuf_veh_h: float
    actual_nuf_veh_h: float
    first_state: TrafficState
    states: list[TrafficState]
    diagnostics: dict[str, Any]
    control_valid: bool
    physical_valid: bool
    evaluation_seconds: float

    @property
    def cost_vector(self) -> np.ndarray:
        return np.array(list(self.player_costs.values()))

    @property
    def budget_vector(self) -> np.ndarray:
        return np.array([self.achieved_np_veh, self.commanded_nuf_veh_h])


@dataclass
class PlayerSDMPCResult:
    control: ControlAction
    objective: float
    evaluation: PlayerRolloutEvaluation
    residual_np_veh: float
    residual_nuf_veh_h: float
    feasible: bool
    converged: bool
    status: str
    iterations: int
    iteration_rows: list[dict[str, Any]] = field(default_factory=list)
    sensitivity_rows: list[dict[str, Any]] = field(default_factory=list)
    local_rows: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = ""


def mixed_violation(residual: np.ndarray) -> float:
    """NP cap은 양의 초과만, UF command equality는 양방향 오차를 센다."""
    return float(max(0.0, residual[0], abs(residual[1])))


def update_budget_prices(prices: np.ndarray, raw_local_residual: np.ndarray, step: float) -> np.ndarray:
    """정규화 surrogate의 projected dual ascent; resource QP dual과 다르다."""
    updated = np.asarray(prices) + step * np.asarray(raw_local_residual)
    updated[0] = max(0.0, updated[0])
    return updated


def mixed_resource_step(local_response: np.ndarray, jacobian: np.ndarray, residual: np.ndarray,
                        lower_step: np.ndarray, upper_step: np.ndarray, metric: float = 1.0) -> ResourceStep:
    """min .5*metric*||d-local_response||², NP 선형 cap와 UF 등식. TTT 항 없음."""
    p, A, c = np.asarray(local_response), np.asarray(jacobian), np.asarray(residual)
    lo, hi = np.asarray(lower_step), np.asarray(upper_step)
    if np.any(lo > hi):
        raise ValueError("invalid resource bounds")
    constraints = []
    if np.linalg.norm(A[0]) > 1e-12:
        constraints.append({"type": "ineq", "fun": lambda d: -c[0] - A[0] @ d, "jac": lambda d: -A[0]})
    if np.linalg.norm(A[1]) > 1e-12:
        constraints.append({"type": "eq", "fun": lambda d: c[1] + A[1] @ d, "jac": lambda d: A[1]})
    out = minimize(lambda d: .5 * metric * float((d-p) @ (d-p)), np.clip(p, lo, hi),
                   jac=lambda d: metric*(d-p), bounds=list(zip(lo, hi)), constraints=constraints,
                   method="SLSQP", options={"maxiter": 60, "ftol": 1e-11})
    d = np.clip(out.x, lo, hi)
    violation = mixed_violation(A @ d + c)
    succeeded = bool(out.success and violation <= 1e-6)
    if not succeeded and violation > 1e-6:
        # QP 실패 때의 방향은 순수 선형 feasibility 복원이며 성공 증명으로 쓰지 않는다.
        def phase_one(x):
            r = A @ x + c
            r[0] = max(0.0, r[0])
            return .5*float(r @ r), A.T @ r
        repair = minimize(lambda x: phase_one(x)[0], d, jac=lambda x: phase_one(x)[1],
                          bounds=list(zip(lo, hi)), method="L-BFGS-B",
                          options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-10})
        if mixed_violation(A @ repair.x + c) < violation:
            d = np.clip(repair.x, lo, hi)
            violation = mixed_violation(A @ d + c)
    # 이 dual은 projection 목적의 KKT 추정치다. 원 TTT 가격으로 누적하지 않는다.
    dual = np.zeros(2)
    free = (d > lo + 1e-7) & (d < hi - 1e-7)
    active = [1] if (A @ d + c)[0] < -1e-7 else [0, 1]
    if succeeded and np.any(free):
        matrix = A[active][:, free].T
        if np.linalg.norm(matrix) > 1e-12:
            low = np.array([0. if k == 0 else -np.inf for k in active])
            fit = lsq_linear(matrix, -metric*(d-p)[free], bounds=(low, np.full(len(active), np.inf)))
            dual[active] = fit.x
    return ResourceStep(d, dual, violation, succeeded, str(out.message))


def player_finite_difference_model(evaluate: Callable, value: np.ndarray, lower: np.ndarray,
                                   upper: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """전체 결합 rollout의 동일 stencil에서 모든 player의 total derivative를 얻는다."""
    gradients = np.zeros((len(evaluate(value).cost_vector), len(value)))
    A = np.zeros((2, len(value)))
    for j in range(len(value)):
        hi, lo = value.copy(), value.copy()
        hi[j], lo[j] = min(upper[j], value[j]+steps[j]), max(lower[j], value[j]-steps[j])
        span = hi[j]-lo[j]
        if span > 1e-12:
            high, low = evaluate(hi), evaluate(lo)
            gradients[:, j] = (high.cost_vector-low.cost_vector)/span
            A[:, j] = (high.budget_vector-low.budget_vector)/span
    return gradients, A


@dataclass
class LocalNonlinearResult:
    step: np.ndarray
    success: bool
    status: str
    evaluations: int
    iterations: int
    own_initial: float
    own_final: float
    model_initial: float
    model_final: float
    stationarity: float
    mesh_size: float


def solve_nonlinear_local(own_objective: Callable[[np.ndarray], float], linear_term: np.ndarray,
                          lower: np.ndarray, upper: np.ndarray, fd_steps: np.ndarray, proximal: float,
                          *, solver: str = "slsqp", max_iterations: int = 4, max_evaluations: int = 32,
                          tolerance: float = .001, mesh_min: float = .002) -> LocalNonlinearResult:
    """Own은 매 trial nonlinear 함수 그대로다. e+budget price만 1차항으로 고정한다."""
    lo, hi, linear = np.asarray(lower), np.asarray(upper), np.asarray(linear_term)
    zero = np.clip(np.zeros(len(lo)), lo, hi)
    cache: dict[bytes, float] = {}
    best = zero.copy()
    best_value = math.inf
    calls, iterations = 0, 0
    class EvaluationLimit(Exception):
        pass
    def own(x):
        nonlocal calls
        key = np.asarray(x, dtype=float).tobytes()
        if key not in cache:
            if calls >= max_evaluations:
                raise EvaluationLimit
            cache[key] = float(own_objective(np.asarray(x)))
            calls += 1
        return cache[key]
    def objective(x):
        nonlocal best, best_value
        value = own(x) + float(linear @ x) + .5*proximal*float(x @ x)
        if math.isfinite(value) and value < best_value:
            best, best_value = x.copy(), value
        return value
    def jac(x):
        gradient = linear + proximal*x
        gradient = gradient.copy()
        for j in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[j], xm[j] = min(hi[j], x[j]+fd_steps[j]), max(lo[j], x[j]-fd_steps[j])
            if xp[j]-xm[j] > 1e-12:
                gradient[j] += (own(xp)-own(xm))/(xp[j]-xm[j])
        return gradient
    initial_own, initial_value = own(zero), objective(zero)
    success, status, stationarity, mesh = False, "iteration_limit", math.inf, math.inf
    try:
        if solver == "slsqp":
            out = minimize(objective, zero, jac=jac, bounds=list(zip(lo, hi)), method="SLSQP",
                           options={"maxiter": max_iterations, "ftol": 1e-9})
            iterations = int(out.nit)
            objective(np.clip(out.x, lo, hi))
            gradient = jac(best)
            stationarity = float(np.max(np.abs(best-np.clip(best-gradient, lo, hi)), initial=0.))
            success = bool(out.success and stationarity <= tolerance)
            status = "local_projected_stationarity" if success else f"slsqp:{out.message}"
        elif solver == "pattern":
            mesh = max(float(np.max(hi-lo))/2, mesh_min)
            for iterations in range(1, max_iterations+1):
                anchor, start_value = best.copy(), best_value
                directions = list(np.eye(len(lo))) + list(-np.eye(len(lo)))
                if len(lo) == 2:
                    directions += [np.array(v) for v in ((1,1),(1,-1),(-1,1),(-1,-1))]
                for direction in directions:
                    objective(np.clip(anchor+mesh*direction, lo, hi))
                if best_value >= start_value-1e-9:
                    if mesh <= mesh_min:
                        success, status = True, "local_finite_mesh_no_improvement"
                        break
                    mesh = max(mesh/2, mesh_min)
            if not success:
                status = "local_mesh_search_incomplete"
        else:
            raise ValueError("unknown local solver")
    except EvaluationLimit:
        success, status = False, "local_evaluation_limit"
    return LocalNonlinearResult(best, success, status, calls, iterations, initial_own, own(best),
                                initial_value, best_value, stationarity, mesh)


def active_vsl_indices(cfg: ExperimentConfig, link: str) -> range:
    """정본 sequence sanitize처럼 첫 off-ramp segment 이전만 VSL 탐색 권한을 둔다."""
    net = cfg.network
    bottlenecks = [int(net.off_ramp_segment_index.get(r, net.freeway_segments_per_link-1))
                   for r in net.off_ramps if net.off_ramp_from_freeway.get(r) == link]
    return range(max(0, min(bottlenecks or [net.freeway_segments_per_link-1])))


class PlayerControlCoordinates:
    """정본 active 12개 연속축과 freeway별 2개 active VSL; 나머지는 고정 권한."""
    def __init__(self, cfg: ExperimentConfig, options: PlayerSDMPCOptions, previous: ControlAction):
        self.cfg, self.options, self.previous = cfg, options, previous.copy()
        net = cfg.network
        self.axes = []
        for ramp in net.ramps:
            cap = net.ramp_capacity_veh_h[ramp]
            self.axes.append(("meter", ramp, cap, 0., cap, options.fd_metering_veh_h, f"FW:{net.ramp_to_freeway[ramp]}"))
        for signal in net.signals:
            low = max(net.green_min, net.effective_green_total-net.green_max)
            high = min(net.green_max, net.effective_green_total-net.green_min)
            self.axes.append(("green", signal, net.cycle_length, low, high, options.fd_green_sec, f"URBAN:{signal}"))
            if signal in {"A", "B", "C"}:
                step = cfg.urban_follower.max_offset_step
                self.axes.append(("offset", signal, net.cycle_length, -step, step, options.fd_offset_sec, f"URBAN:{signal}"))
        self.scales = np.array([a[2] for a in self.axes])
        self.lower = np.array([a[3]/a[2] for a in self.axes])
        self.upper = np.array([a[4]/a[2] for a in self.axes])
        self.fd = np.array([a[5]/a[2] for a in self.axes])
        self.owners = tuple(a[6] for a in self.axes)
        self.active_vsl = tuple((link, i) for link in net.freeway_links for i in active_vsl_indices(cfg,link))

    def encode(self, control: ControlAction) -> np.ndarray:
        net, values = self.cfg.network, []
        for kind, key, scale, *_ in self.axes:
            if kind == "meter":
                value = control.ramp_metering.get(key, scale)
            elif kind == "green":
                value = control.green_times.get(f"{key}_p1", net.effective_green_total/2)
            else:
                value = (control.offsets.get(key, 0.)-self.previous.offsets.get(key, 0.)+net.cycle_length/2)%net.cycle_length-net.cycle_length/2
            values.append(value/scale)
        return np.clip(values, self.lower, self.upper)

    def allowed_vsl(self, link: str, index: int) -> list[float]:
        old = segment_vsl(self.previous, link, index, self.cfg)
        return sorted(float(v) for v in self.cfg.freeway_follower.vsl_set
                      if abs(v-old) <= self.cfg.freeway_follower.max_vsl_step+1e-9)

    def initial_vsl(self, control: ControlAction) -> dict[str, float]:
        net, neutral = self.cfg.network, float(max(self.cfg.freeway_follower.vsl_set))
        values = {}
        for link in net.freeway_links:
            for i in range(net.freeway_segments_per_link):
                desired = segment_vsl(control, link, i, self.cfg)
                values[f"{link}__seg{i}"] = (min(self.allowed_vsl(link, i), key=lambda v: (abs(v-desired), -v))
                    if (link,i) in self.active_vsl else repair_vsl_value(neutral,segment_vsl(self.previous,link,i,self.cfg),self.cfg).value)
            values[link] = min(values[f"{link}__seg{i}"] for i in range(net.freeway_segments_per_link))
        return values

    def decode(self, y: np.ndarray, vsl: dict[str, float]) -> ControlAction:
        net = self.cfg.network
        control = ControlAction.uncontrolled(self.cfg)
        control.inflow_outflow_allocation = {}
        control.offsets = {s: 0. for s in net.signals}
        control.vsl = dict(vsl)
        for link in net.freeway_links:
            control.vsl[link] = min(control.vsl[f"{link}__seg{i}"] for i in range(net.freeway_segments_per_link))
        for value, (kind, key, scale, *_rest) in zip(y, self.axes):
            value = float(value*scale)
            if kind == "meter":
                control.ramp_metering[key] = value
            elif kind == "green":
                control.green_times[f"{key}_p1"] = value
                control.green_times[f"{key}_p2"] = net.effective_green_total-value
            else:
                offset = (self.previous.offsets.get(key, 0.)+value)%net.cycle_length
                control.offsets[key] = 0. if offset >= net.cycle_length else offset
        return control


def validate_player_control(control: ControlAction, previous: ControlAction, cfg: ExperimentConfig) -> dict[str, Any]:
    """실제 segment VSL와 D/F 고정 offset 및 allocation 비권한까지 검사한다."""
    from src.controllers.sensitivity_dmpc import validate_control
    result = validate_control(control, previous, cfg)
    errors = list(result["violations"])
    net, neutral = cfg.network, max(cfg.freeway_follower.vsl_set)
    if set(control.ramp_metering) != set(net.ramps):
        errors.append("metering_owner_keys")
    allowed_vsl_keys = set(net.freeway_links) | {f"{link}__seg{i}" for link in net.freeway_links for i in range(net.freeway_segments_per_link)}
    if set(control.vsl) != allowed_vsl_keys:
        errors.append("vsl_owner_keys")
    if control.inflow_outflow_allocation:
        errors.append("allocation_authority")
    for signal in set(net.signals)-{"A", "B", "C"}:
        if abs(control.offsets.get(signal, 0.)) > 1e-9:
            errors.append(f"fixed_offset:{signal}")
    for link in net.freeway_links:
        values = []
        for i in range(net.freeway_segments_per_link):
            key = f"{link}__seg{i}"
            value = control.vsl.get(key, float("nan"))
            values.append(value)
            old = segment_vsl(previous,link,i,cfg)
            inactive_target = repair_vsl_value(neutral,old,cfg).value
            if value not in cfg.freeway_follower.vsl_set or abs(value-old) > cfg.freeway_follower.max_vsl_step+1e-9 or (i not in active_vsl_indices(cfg,link) and value != inactive_target):
                errors.append(f"segment_vsl:{key}")
        if control.vsl.get(link) != min(values):
            errors.append(f"vsl_summary:{link}")
    return {"valid": not errors, "violations": errors}


class PlayerSensitivityDMPC:
    def __init__(self, cfg: ExperimentConfig, options: PlayerSDMPCOptions | None = None):
        self.cfg, self.options = cfg, options or PlayerSDMPCOptions()
        self.ownership = PlayerCostOwnership.from_config(cfg)
        self.player_ids = self.ownership.active_player_ids
        self.candidate_rows: list[dict[str, Any]] = []
        self.last_results: list[PlayerSDMPCResult] = []
        self.last_result: PlayerSDMPCResult | None = None
        self.previous_control: ControlAction | None = None
        self.warm_start_control: ControlAction | None = None
        self._decision_cache: dict | None = None

    def evaluate_control(self, state: TrafficState, forecast: Sequence[DemandStep], control: ControlAction,
                         previous: ControlAction | None = None) -> PlayerRolloutEvaluation:
        if len(forecast) < self.options.horizon_steps:
            raise ValueError("forecast must cover the complete NP budget horizon")
        start, cfg = time.perf_counter(), self.cfg
        s, ledger = state.copy(), PlayerCostLedger(cfg, self.ownership)
        states, rows, conservation = [], [], []
        total, np_service, exits_total = 0., 0., 0.
        physical = True
        for demand in list(forecast)[:self.options.horizon_steps]:
            before = physical_inventory(s, cfg)
            interval = run_coupled_interval(s, control, demand, cfg, cost_observer=ledger.observe)
            s.time_sec += cfg.simulation.T_c_sec
            total += interval.freeway_ttt+interval.urban_ttt
            row = interval.diagnostics
            np_service += float(row["inbound_service_veh"])-float(row["outbound_service_veh"])
            # 外部 demand는 대기 포함 전량 inventory 유입, true exit만 한 번 차감한다.
            arrivals = cfg.simulation.T_c_h*(sum(demand.freeway_mainline.values())+sum(demand.ramp_arrival.values())
                       +sum(demand.urban_boundary.get(g, 0.) for g in cfg.network.boundary_in_links))
            exits = cfg.simulation.T_c_h*float(row.get("mainline_exit_flow_total", 0.))+float(row.get("boundary_out_sink_veh", 0.))
            exits_total += exits
            conservation.append(physical_inventory(s, cfg)-before-arrivals+exits)
            numbers = (list(s.ramp_queue.values())+list(s.urban_movement_queue.values())+list(s.mainline_origin_queue.values())
                       +[v for x in s.freeway_density.values() for v in x]+freeway_buffer_state_values(s)[0])
            all_numbers = numbers+list(s.urban_link_storage.values())+[v for x in s.freeway_speed.values() for v in x]+freeway_buffer_state_values(s)[1]
            physical &= bool(np.all(np.isfinite(all_numbers)) and min([0.]+numbers) >= -1e-7)
            physical &= all(-1e-7 <= s.urban_link_storage.get(k, cap) <= cap+1e-7 for k, cap in cfg.network.urban_link_storage_veh.items())
            rows.append(row)
            states.append(s.copy())
        loss = sum(float(r.get("movement_queue_projection_veh", 0.))+float(r.get("coupling_offramp_arrivals_rejected_veh", 0.)) for r in rows)
        accounting = abs(ledger.total_ttt-total)
        physical &= bool(math.isfinite(total) and np.all(np.isfinite(ledger.cost_vector())) and loss <= 1e-6
                         and max(abs(r) for r in conservation) <= 1e-6 and accounting <= 1e-8)
        validation = validate_player_control(control, previous or control, cfg)
        diagnostics = {"intervals": rows, "first_interval": rows[0], "horizon_steps": len(rows),
                       "max_conservation_residual_veh": max(abs(r) for r in conservation),
                       "conservation_residuals_veh": conservation, "external_exit_veh": exits_total,
                       "numerical_vehicle_loss_veh": loss, "accounting_residual_veh_h": accounting,
                       "control_validation": validation, "raw_costs": dict(ledger.raw_costs),
                       "passive_cost_owners": dict(self.ownership.passive_cost_owners),
                       "first_np_stock_change_veh": states[0].protected_accumulation_veh(cfg.network)-state.protected_accumulation_veh(cfg.network)}
        return PlayerRolloutEvaluation(total, ledger.player_costs, dict(ledger.regional_costs),
                   {k:v for k,v in ledger.raw_costs.items() if k.startswith("PASSIVE:")}, np_service,
                   float(sum(control.ramp_metering.get(r,0.) for r in cfg.network.ramps)), float(rows[0]["total_metering_flow"]),
                   states[0], states, diagnostics, validation["valid"], physical, time.perf_counter()-start)

    def _scales(self) -> np.ndarray:
        return np.array([self.options.budget_scale_np_veh, self.options.budget_scale_nuf_veh_h])

    def _feasible(self, ev: PlayerRolloutEvaluation, budget: PlayerBudget) -> bool:
        c = ev.budget_vector-budget.vector()
        return bool(ev.control_valid and ev.physical_valid and math.isfinite(ev.total_ttt) and np.all(np.isfinite(c))
                    and c[0] <= self.options.tolerance_np_veh and abs(c[1]) <= self.options.tolerance_nuf_veh_h)

    def solve_fixed_budget(self, state: TrafficState, forecast: Sequence[DemandStep], previous: ControlAction | None,
                           budget: PlayerBudget, initial_control: ControlAction | None = None) -> PlayerSDMPCResult:
        if not np.all(np.isfinite(budget.vector())):
            raise ValueError("budget must be finite")
        opts, cfg = self.options, self.cfg
        previous = (previous or ControlAction.uncontrolled(cfg)).copy()
        coords = PlayerControlCoordinates(cfg, opts, previous)
        seed = initial_control or previous
        y, vsl = coords.encode(seed), coords.initial_vsl(seed)
        cache = self._decision_cache if self._decision_cache is not None else {}
        evaluations = 0
        def evaluate(z, velocities=None):
            nonlocal evaluations
            vv = vsl if velocities is None else velocities
            key = (np.asarray(z, dtype=float).tobytes(), tuple(sorted(vv.items())))
            if key not in cache:
                cache[key] = self.evaluate_control(state, forecast, coords.decode(z, vv), previous)
                evaluations += 1
            return cache[key]
        def violation(ev):
            return mixed_violation((ev.budget_vector-budget.vector())/self._scales())
        impossible = budget.nuf_veh_h < 0. or budget.nuf_veh_h > cfg.network.total_ramp_capacity+1e-9
        current = evaluate(y)
        best = (y.copy(), dict(vsl), current) if not impossible and self._feasible(current, budget) else None
        rows, sensitivities, locals_log = [], [], []
        prices, trust = np.zeros(2), opts.trust_radius
        stationarity, reason, converged = math.inf, "iteration_limit", False
        price_stationarity, dual_step_norm, complementarity = math.inf, math.inf, math.inf
        local_resource_violation = math.inf

        def derivatives(z, vv):
            gradients, Araw = player_finite_difference_model(lambda x: evaluate(x, vv), z, coords.lower, coords.upper, coords.fd)
            # UF는 명령 합: capacity로 정규화된 metering 축의 analytic 미분.
            Araw[1] = np.array([a[2] if a[0] == "meter" else 0. for a in coords.axes])
            return gradients, Araw, Araw/self._scales()[:, None]

        def restore(z, vv, ev, A):
            z, A = z.copy(), A.copy()
            used = 0
            for used in range(1, opts.restoration_iterations+1):
                if self._feasible(ev, budget):
                    break
                c = (ev.budget_vector-budget.vector())/self._scales()
                qp = mixed_resource_step(np.zeros_like(z), A, c, np.maximum(coords.lower-z, -trust), np.minimum(coords.upper-z, trust))
                found = False
                for alpha in (1., .5, .25):
                    trial = np.clip(z+alpha*qp.step, coords.lower, coords.upper)
                    test = evaluate(trial, vv)
                    if test.control_valid and test.physical_valid and violation(test) < violation(ev)-1e-10:
                        delta = trial-z
                        dc = (test.budget_vector-ev.budget_vector)/self._scales()
                        if delta @ delta > 1e-16:
                            A += np.outer(dc-A @ delta, delta)/(delta @ delta)
                        z, ev, found = trial, test, True
                        break
                if not found:
                    break
            return z, ev, used

        def record(iteration, accepted=False, step=0., qp=None, local_ok=False, poll=False):
            c = current.budget_vector-budget.vector()
            rows.append({"iteration": iteration, "objective_ttt": current.total_ttt, **{f"cost_{k}":v for k,v in current.player_costs.items()},
              "requested_np_cap_veh": budget.np_cap_veh, "requested_nuf_veh_h": budget.nuf_veh_h,
              "achieved_np_veh": current.achieved_np_veh, "commanded_nuf_veh_h": current.commanded_nuf_veh_h,
              "actual_nuf_veh_h": current.actual_nuf_veh_h, "residual_np_veh": float(c[0]), "residual_nuf_veh_h": float(c[1]),
              "np_violation_veh": max(0., float(c[0])), "residual_scaled_inf": violation(current),
              "feasible": not impossible and self._feasible(current, budget), "accepted": accepted,
              "stationarity_norm": stationarity, "step_norm": step, "trust_radius": trust,
              "price_kkt_mapping":price_stationarity,"dual_step_norm":dual_step_norm,
              "np_complementarity":complementarity,"raw_local_resource_violation":local_resource_violation,
              "mu_np_h": float(prices[0]/self._scales()[0]), "lambda_nuf_h2": float(prices[1]/self._scales()[1]),
              "resource_qp_success": bool(qp and qp.success), "linear_residual_scaled_inf": qp.linear_residual if qp else 0.,
              "projection_mu_np": float(qp.multiplier[0]) if qp else 0., "projection_lambda_uf": float(qp.multiplier[1]) if qp else 0.,
              "local_solvers_converged": local_ok, "discrete_poll_complete": poll, "rollout_evaluations": evaluations,
              "local_model": "exact_nonlinear_own_TTT_plus_frozen_externality_and_budget_price"})
        record(0, True)
        if impossible:
            reason = "physical_command_capacity_bound"
        else:
            for iteration in range(1, opts.max_iterations+1):
                old_y = y.copy()
                gradients, Araw, A = derivatives(y, vsl)
                own, external = np.zeros(len(y)), np.zeros(len(y))
                for j, owner in enumerate(coords.owners):
                    own[j] = gradients[self.player_ids.index(owner), j]
                    external[j] = sum(gradients[:, j])-own[j] if opts.externality_enabled else 0.
                price_used = prices.copy()
                price_gradient = A.T @ price_used
                proposal = np.zeros_like(y)
                local_ok = True
                # Jacobi의 같은 y/vsl을 유지한 채 각 실제 player의 nonlinear NLP를 푼다.
                for pid in self.player_ids:
                    indices = np.array([j for j,p in enumerate(coords.owners) if p == pid], dtype=int)
                    def own_callback(s, ix=indices, player=pid):
                        trial = y.copy()
                        trial[ix] += s
                        return evaluate(trial).player_costs[player]
                    local = solve_nonlinear_local(own_callback, (external+price_gradient)[indices],
                         np.maximum(coords.lower[indices]-y[indices], -trust), np.minimum(coords.upper[indices]-y[indices], trust),
                         coords.fd[indices], opts.proximal, solver=opts.local_solver, max_iterations=opts.local_max_iterations,
                         max_evaluations=opts.local_max_evaluations, tolerance=opts.local_tolerance, mesh_min=opts.local_mesh_min)
                    proposal[indices] = local.step
                    local_ok &= local.success
                    locals_log.append({"iteration":iteration, "player":pid, "solver":opts.local_solver,
                      "status":local.status, "success":local.success, "nonlinear_own_evaluations":local.evaluations,
                      "local_iterations":local.iterations, "own_initial":local.own_initial, "own_final":local.own_final,
                      "local_model_initial":local.model_initial, "local_model_final":local.model_final,
                      "local_stationarity":local.stationarity, "mesh_size":local.mesh_size,
                      "response_norm":float(np.max(np.abs(local.step), initial=0.)),
                      "response":local.step.tolist(), "externality_norm":float(np.linalg.norm(external[indices])),
                      "budget_gradient_norm":float(np.linalg.norm(price_gradient[indices]))})
                c = (current.budget_vector-budget.vector())/self._scales()
                raw_local_residual = c+A @ proposal
                prices = update_budget_prices(price_used, raw_local_residual, opts.dual_step)
                dual_step_norm = float(np.max(np.abs(prices-price_used)))
                local_resource_violation = mixed_violation(raw_local_residual)
                complementarity = abs(float(prices[0]*c[0]))
                lag_gradient = own+external+A.T @ prices
                price_stationarity = opts.proximal*float(np.max(np.abs(y-np.clip(y-lag_gradient/opts.proximal,coords.lower,coords.upper))))
                qp = mixed_resource_step(proposal, A, c, np.maximum(coords.lower-y, -trust), np.minimum(coords.upper-y, trust))
                for j,(kind,key,scale,*_) in enumerate(coords.axes):
                    sensitivities.append({"iteration":iteration,"player":coords.owners[j],"kind":kind,"control":key,
                      "physical_scale":scale,"own_gradient":own[j]/scale,"externality_gradient":external[j]/scale,
                      "budget_gradient":price_gradient[j]/scale,"d_np_d_control":Araw[0,j]/scale,"d_nuf_d_control":Araw[1,j]/scale,
                      "mu_np_used_h":price_used[0]/self._scales()[0],"lambda_nuf_used_h2":price_used[1]/self._scales()[1],
                      "mu_np_next_h":prices[0]/self._scales()[0],"lambda_nuf_next_h2":prices[1]/self._scales()[1],
                      "raw_local_np_scaled":raw_local_residual[0],"raw_local_nuf_scaled":raw_local_residual[1]})
                # trust 축소로 가짜 stationary가 되지 않도록 원 box의 전체 gradient mapping.
                tangent = mixed_resource_step(-(own+external)/opts.proximal, A, np.array([min(c[0],0.),0.]), coords.lower-y, coords.upper-y, opts.proximal)
                stationarity = opts.proximal*float(np.max(np.abs(tangent.step))) if tangent.success else math.inf
                accepted = False
                for ls in range(opts.line_search_steps):
                    trial = np.clip(y+(0.5**ls)*qp.step, coords.lower, coords.upper)
                    trial_ev = evaluate(trial)
                    if not self._feasible(trial_ev, budget):
                        trial, trial_ev, _ = restore(trial, vsl, trial_ev, A)
                    if self._feasible(trial_ev, budget) and (best is None or trial_ev.total_ttt < best[2].total_ttt-opts.objective_improvement_tolerance):
                        y, current, accepted = trial, trial_ev, True
                        best = (y.copy(), dict(vsl), current)
                        break
                    if best is None and trial_ev.control_valid and trial_ev.physical_valid and violation(trial_ev) < violation(current)-1e-10:
                        y, current, accepted = trial, trial_ev, True
                        break
                poll_complete, discrete_improved = not opts.discrete_poll, False
                if opts.discrete_poll and self._feasible(current,budget) and (iteration%opts.discrete_poll_every == 0 or iteration == opts.max_iterations or stationarity <= opts.stationarity_tolerance):
                    poll_complete = True
                    anchor_y, anchor_v, anchor_ev = y.copy(), dict(vsl), current
                    winner = None
                    for link,index in coords.active_vsl:
                        key = f"{link}__seg{index}"
                        allowed = coords.allowed_vsl(link,index)
                        at = allowed.index(anchor_v[key])
                        for next_index in (at-1,at+1):
                            if not 0 <= next_index < len(allowed):
                                continue
                            vv = dict(anchor_v)
                            vv[key] = allowed[next_index]
                            vv[link] = min(vv[f"{link}__seg{i}"] for i in range(cfg.network.freeway_segments_per_link))
                            trial, ev = anchor_y.copy(), evaluate(anchor_y,vv)
                            pid = f"FW:{link}"
                            own_delta = ev.player_costs[pid]-anchor_ev.player_costs[pid]
                            external_delta = ev.total_ttt-anchor_ev.total_ttt-own_delta
                            if not self._feasible(ev,budget):
                                _,_,fresh_A = derivatives(trial,vv)
                                trial,ev,_ = restore(trial,vv,ev,fresh_A)
                            feasible = self._feasible(ev,budget)
                            poll_complete &= feasible
                            locals_log.append({"iteration":iteration,"player":pid,"solver":"discrete_vsl_poll",
                              "control":key,"value":vv[key],"own_delta_before_restoration":own_delta,
                              "external_delta_before_restoration":external_delta,"success":feasible,
                              "status":"feasible_poll" if feasible else "unresolved_discrete_restoration","objective_after_restoration":ev.total_ttt})
                            if feasible and ev.total_ttt < anchor_ev.total_ttt-opts.objective_improvement_tolerance and (winner is None or ev.total_ttt < winner[2].total_ttt):
                                winner = (trial.copy(),vv,ev)
                    if winner is not None:
                        y,vsl,current = winner
                        best,accepted,discrete_improved = (y.copy(),dict(vsl),current),True,True
                step_norm = float(np.max(np.abs(y-old_y)))
                record(iteration,accepted,step_norm,qp,local_ok,poll_complete)
                # Pattern의 local mesh 통과는 global feasible mesh 통과가 아니므로 승격하지 않는다.
                dual_ok = (max(price_stationarity,dual_step_norm,complementarity) <= opts.stationarity_tolerance
                           and raw_local_residual[0] <= opts.tolerance_np_veh/self._scales()[0]
                           and abs(raw_local_residual[1]) <= opts.tolerance_nuf_veh_h/self._scales()[1])
                if opts.local_solver == "slsqp" and local_ok and dual_ok and qp.success and not discrete_improved and poll_complete and self._feasible(current,budget) and step_norm <= 1e-8 and stationarity <= opts.stationarity_tolerance:
                    converged,reason = True,"feasible_numerical_projected_stationarity_local_nlp_and_discrete_poll"
                    break
                if not accepted:
                    trust *= .5
                    if trust < opts.min_trust_radius:
                        reason = "trust_region_stall_not_convergence"
                        break
        if best is not None:
            y,vsl,current = best
        feasible = not impossible and self._feasible(current,budget)
        c = current.budget_vector-budget.vector()
        status = "infeasible_proven" if impossible else "converged_feasible" if feasible and converged else "feasible_but_not_converged" if feasible else "unresolved_algorithm_failure"
        control = coords.decode(y,vsl)
        control.N_P_star,control.N_UF_star = budget.np_cap_veh,budget.nuf_veh_h
        control.diagnostics.update({"player_sdmpc_active":True,"player_sdmpc_feasible":feasible,
          "player_sdmpc_converged":feasible and converged,"budget_contract_failed":not feasible,"fallback":not feasible,
          "player_sdmpc_np_service_veh":current.achieved_np_veh,"player_sdmpc_command_nuf_veh_h":current.commanded_nuf_veh_h,
          "player_sdmpc_np_residual_veh":float(c[0]),"player_sdmpc_nuf_residual_veh_h":float(c[1]),
          "player_sdmpc_rollout_evaluations":evaluations,"player_sdmpc_stationarity_norm":stationarity,
          "player_sdmpc_mu_np_h":prices[0]/self._scales()[0],"player_sdmpc_lambda_nuf_h2":prices[1]/self._scales()[1],
          "player_sdmpc_constant_move_block":True,"player_sdmpc_np_horizon_cap":True})
        return PlayerSDMPCResult(control,current.total_ttt,current,float(c[0]),float(c[1]),feasible,feasible and converged,
                                 status,len(rows)-1,rows,sensitivities,locals_log,reason)

    def decide(self, state: TrafficState, forecast: Sequence[DemandStep], previous: ControlAction | None = None,
               cfg: ExperimentConfig | None = None) -> ControlAction:
        previous = (previous or self.previous_control or ControlAction.uncontrolled(self.cfg)).copy()
        coords = PlayerControlCoordinates(self.cfg,self.options,previous)
        hint = self.warm_start_control or previous
        warm = coords.decode(coords.encode(hint),coords.initial_vsl(hint))
        self.warm_start_control = None
        witness = self.evaluate_control(state,forecast,warm,previous)
        b0 = PlayerBudget(witness.achieved_np_veh,witness.commanded_nuf_veh_h)
        candidates = [(b0,"warm_start_rollout_witness")]
        # 후보 생성과 exact service cap은 같은 H를 사용한다. 원 cfg는 수정하지 않는다.
        leader_cfg = self.cfg.with_updates({"mpc":{"horizon_steps":self.options.horizon_steps}})
        raw = Leader(leader_cfg).candidates(state,previous,forecast=list(forecast)[:self.options.horizon_steps])
        raw = sorted(raw,key=lambda b:(abs(b.N_P_star-b0.np_cap_veh)/self._scales()[0]+abs(b.N_UF_star-b0.nuf_veh_h)/self._scales()[1],b.N_P_star,b.N_UF_star))
        for action in raw:
            if len(candidates) >= self.options.max_candidates:
                break
            budget = PlayerBudget(action.N_P_star,action.N_UF_star)
            if budget not in [b for b,_ in candidates]:
                candidates.append((budget,"existing_leader_raw_pair"))
        self.last_results,self.candidate_rows,self.last_result = [],[],None
        self._decision_cache = {}
        try:
            for index,(budget,origin) in enumerate(candidates):
                started = time.perf_counter()
                result = self.solve_fixed_budget(state,forecast,previous,budget,warm)
                self.last_results.append(result)
                self.candidate_rows.append({"candidate":index,"origin":origin,"requested_np_cap_veh":budget.np_cap_veh,
                  "requested_nuf_veh_h":budget.nuf_veh_h,"achieved_np_veh":result.evaluation.achieved_np_veh,
                  "commanded_nuf_veh_h":result.evaluation.commanded_nuf_veh_h,"actual_nuf_veh_h":result.evaluation.actual_nuf_veh_h,
                  "residual_np_veh":result.residual_np_veh,"residual_nuf_veh_h":result.residual_nuf_veh_h,
                  "objective_ttt":result.objective,"status":result.status,"feasible":result.feasible,"converged":result.converged,
                  "iterations":result.iterations,"termination_reason":result.termination_reason,
                  "rollout_evaluations":result.control.diagnostics["player_sdmpc_rollout_evaluations"],
                  "computation_time_sec":time.perf_counter()-started,"selected":False})
        finally:
            self._decision_cache = None
        eligible = [r for r in self.last_results if r.feasible
                    and self._feasible(r.evaluation,PlayerBudget(r.control.N_P_star,r.control.N_UF_star))
                    and validate_player_control(r.control,previous,self.cfg)["valid"]]
        if not eligible:
            raise NoExecutableControlError("No player-SDMPC candidate passed original cap/equality and physical/control checks")
        self.last_result = min(eligible,key=lambda r:r.objective)
        chosen = next(i for i,r in enumerate(self.last_results) if r is self.last_result)
        self.candidate_rows[chosen]["selected"] = True
        self.previous_control = self.last_result.control.copy()
        return self.last_result.control.copy()
