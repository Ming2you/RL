"""구간 고정 budget 가격 실험: 원 historical controller/plant 파일은 불변.

API: FrozenPriceSDMPC(cfg, options=None, mode='interval'|'inner').
begin_interval(state, forecast, previous=None) -> 정수 token; 후보 solve는 가격 불변.
select_candidate(results=None) -> 결정적 feasible 최적 후보.
commit_selected(result) -> 선택 후보 raw c+A*d로 한 번 갱신하고 구간 종료.
abort_interval() -> 갱신 없이 종료. decide()는 전체 절차를 자동 수행한다.
result.converged는 원 전체 gate만 의미한다. 별도 conditional 진단은 PASS 대체가 아니다.
instrumentation 시간은 중첩 구간이다(NLP/FD 시간 안에 rollout 시간이 포함됨).
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import asdict, is_dataclass
from typing import Sequence

import numpy as np

from src.controllers.player_sensitivity_dmpc import (
    PlayerSensitivityDMPC as OriginalPlayerSensitivityDMPC,
    PlayerBudget, PlayerSDMPCOptions, PlayerSDMPCResult, PlayerControlCoordinates,
    PlayerRolloutEvaluation, mixed_violation, mixed_resource_step,
    player_finite_difference_model, solve_nonlinear_local, update_budget_prices,
    validate_player_control, Leader, NoExecutableControlError,
)
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.models.demand import DemandStep


def _frozen(value):
    """동적 cfg 속성과 tuple-key buffer를 포함한 정확한 문맥 fingerprint."""
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return _frozen(value.item())
    if is_dataclass(value):
        return (type(value).__name__, _frozen(vars(value)))
    if isinstance(value, dict):
        return tuple(sorted(((_frozen(k), _frozen(v)) for k,v in value.items()), key=repr))
    if isinstance(value, (tuple, list)):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    raise TypeError(f'Unsupported context value: {type(value).__name__}')


def _digest(value):
    return hashlib.sha256(repr(_frozen(value)).encode('utf-8')).hexdigest()


def _commands(control):
    return {k:v for k,v in vars(control).items() if k != 'diagnostics'}


class FrozenPriceSDMPC(OriginalPlayerSensitivityDMPC):
    coordinate_type = PlayerControlCoordinates

    def __init__(self, cfg, options=None, *, mode='interval'):
        super().__init__(cfg, options)
        if mode not in {'inner', 'interval'}:
            raise ValueError('mode must be inner or interval')
        self.mode = mode
        self._persistent_prices = np.zeros(2)
        self._active = None
        self._next_token = 0
        self.price_rows = []
        self.interval_metrics = self._new_metrics()
        self.profile_rows = []
        self._registered = {}
        self._selected = None
        self._derivative_cache = {}
        self._rollout_cache = {}

    @staticmethod
    def _new_metrics():
        return {key:0 for key in ('rollout_calls','rollout_cache_hits','rollout_seconds',
            'rollout_lookup_calls','coordinate_cache_hits','fd_calls','fd_cache_hits','fd_seconds',
            'local_nlp_calls','local_nlp_seconds','local_own_evaluations',
            'price_gradient_calls','price_gradient_seconds',
            'price_update_calls','price_update_seconds','boundary_price_updates',
            'candidate_calls','candidate_seconds','decision_seconds')}

    @property
    def persistent_prices(self):
        return self._persistent_prices.copy()

    def _context(self, state, forecast, previous):
        return _digest((self.mode, self.cfg, self.options, state, list(forecast), previous))

    def begin_interval(self, state, forecast, previous=None):
        if self._active is not None:
            raise RuntimeError('Interval already open; commit or abort first')
        source_previous = previous or self.previous_control or ControlAction.uncontrolled(self.cfg)
        source_forecast = forecast
        previous = source_previous.copy()
        forecast = list(forecast)
        if len(forecast) < self.options.horizon_steps:
            raise ValueError('Forecast does not cover H')
        self._next_token += 1
        self._active = {'token':self._next_token,'state':state,'forecast':forecast,'previous':previous,
            'context':self._context(state,forecast,previous),'prices':self._persistent_prices.copy(),
            'source_previous':source_previous,'source_forecast':source_forecast,
            'started':time.perf_counter()}
        self.interval_metrics = self._new_metrics()
        self._decision_cache, self._derivative_cache, self._rollout_cache = {}, {}, {}
        self._registered, self._selected = {}, None
        self.last_results, self.candidate_rows, self.last_result = [], [], None
        return self._next_token

    def _check_context(self, state=None, forecast=None, previous=None):
        if self._active is None:
            raise RuntimeError('No open interval')
        a = self._active
        if self._context(a['state'],a['forecast'],a['previous']) != a['context']:
            raise RuntimeError('Frozen interval input mutated')
        if self._context(a['state'],a['source_forecast'],a['source_previous']) != a['context']:
            raise RuntimeError('Caller interval input mutated')
        if not np.array_equal(self._persistent_prices,a['prices']):
            raise RuntimeError('Persistent price mutated inside interval')
        if state is not None and self._context(state,forecast,previous) != a['context']:
            raise RuntimeError('Candidate belongs to a different interval context')

    def abort_interval(self):
        self._active = None
        self._decision_cache = None
        self._derivative_cache, self._rollout_cache = {}, {}
        self._registered, self._selected = {}, None

    def evaluate_control(self, state, forecast, control, previous=None):
        # 同一 구간의 원 control 입력이 같은 경우만 rollout 재사용한다.
        if self._active is not None:
            self._check_context(state,forecast,previous or self._active['previous'])
        key = _digest(_commands(control)) if self._active is not None else None
        if key is not None and key in self._rollout_cache:
            self.interval_metrics['rollout_cache_hits'] += 1
            return self._rollout_cache[key]
        started = time.perf_counter()
        result = super().evaluate_control(state,forecast,control,previous)
        self.interval_metrics['rollout_calls'] += 1
        self.interval_metrics['rollout_seconds'] += time.perf_counter()-started
        if key is not None:
            self._rollout_cache[key] = result
        return result

    def _local(self, *args, **kwargs):
        started = time.perf_counter()
        self.interval_metrics['local_nlp_calls'] += 1
        try:
            result = solve_nonlinear_local(*args, **kwargs)
            self.interval_metrics['local_own_evaluations'] += result.evaluations
            return result
        finally:
            self.interval_metrics['local_nlp_seconds'] += time.perf_counter()-started

    def _price_update(self, prices, residual):
        started = time.perf_counter()
        self.interval_metrics['price_update_calls'] += 1
        try:
            return update_budget_prices(prices,residual,self.options.dual_step)
        finally:
            self.interval_metrics['price_update_seconds'] += time.perf_counter()-started

    def _price_gradient(self, A, prices):
        started = time.perf_counter()
        self.interval_metrics['price_gradient_calls'] += 1
        try:
            return A.T @ prices
        finally:
            self.interval_metrics['price_gradient_seconds'] += time.perf_counter()-started

    def _derivatives(self, evaluate, z, vv, coords):
        # budget/가격과 무관한 물리 총미분: 정확히 같은 anchor/VSL/좌표에서만 재사용.
        key = (np.asarray(z,dtype=float).tobytes(),tuple(sorted(vv.items())))
        self.interval_metrics['fd_calls'] += 1
        if key in self._derivative_cache:
            self.interval_metrics['fd_cache_hits'] += 1
            return tuple(x.copy() for x in self._derivative_cache[key])
        started = time.perf_counter()
        gradients,Araw = player_finite_difference_model(lambda x:evaluate(x,vv),z,coords.lower,coords.upper,coords.fd)
        Araw[1] = np.array([axis[2] if axis[0]=='meter' else 0. for axis in coords.axes])
        value = gradients,Araw,Araw/self._scales()[:,None]
        self.interval_metrics['fd_seconds'] += time.perf_counter()-started
        self._derivative_cache[key] = tuple(x.copy() for x in value)
        return value

    def solve_fixed_budget(self, state, forecast, previous, budget, initial_control=None):
        previous = (previous or self.previous_control or ControlAction.uncontrolled(self.cfg)).copy()
        forecast = list(forecast)
        if self._active is None:
            self.begin_interval(state,forecast,previous)
        self._check_context(state,forecast,previous)
        metrics_before = dict(self.interval_metrics)
        started = time.perf_counter()
        self.interval_metrics['candidate_calls'] += 1
        try:
            result = self._solve_impl(state,forecast,previous,budget,initial_control)
        finally:
            self.interval_metrics['candidate_seconds'] += time.perf_counter()-started
        self._check_context(state,forecast,previous)
        result.instrumentation = {key:self.interval_metrics[key]-metrics_before[key] for key in self.interval_metrics}
        result.control.diagnostics['frozen_price'] = {
            'mode':self.mode,'interval_token':self._active['token'],
            'prices_frozen_normalized':self._active['prices'].tolist(),
            'original_converged':result.converged,
            'conditional_primal_local':result.conditional_primal_local,
            'price_update_signal_available':result.price_update_signal is not None}
        fingerprint = _digest((asdict(result),result.price_update_signal))
        self._registered[id(result)] = (result,fingerprint)
        self.last_results.append(result)
        return result

    def _eligible(self, result):
        return bool(result.feasible and self._feasible(result.evaluation,
            PlayerBudget(result.control.N_P_star,result.control.N_UF_star)) and
            validate_player_control(result.control,self._active['previous'],self.cfg)['valid'])

    def select_candidate(self, results=None):
        self._check_context()
        candidates = list(self.last_results if results is None else results)
        for result in candidates:
            record = self._registered.get(id(result))
            if record is None or record[0] is not result or record[1] != _digest((asdict(result),result.price_update_signal)):
                raise RuntimeError('Foreign or mutated candidate')
        eligible = [r for r in candidates if self._eligible(r)]
        if not eligible:
            raise NoExecutableControlError('No candidate passed original execution checks')
        # 동률 시 budget과 물리 제어의 결정적 표현 사용: 평가 순서를 의미에 넣지 않는다.
        self._selected = min(eligible,key=lambda r:(r.objective,r.control.N_P_star,r.control.N_UF_star,
                                                   repr(_frozen(_commands(r.control)))))
        return self._selected

    def commit_selected(self, result):
        self._check_context()
        if self.select_candidate() is not result:
            raise RuntimeError('Only the deterministic selected candidate may commit')
        signal = result.price_update_signal
        if signal is None:
            raise NoExecutableControlError('Selected result has no consistent raw proposal evidence')
        if signal['context'] != self._active['context'] or signal['interval_token'] != self._active['token']:
            raise RuntimeError('Price signal belongs to another interval')
        if self.mode=='interval' and not np.array_equal(np.asarray(signal['prices_used_normalized']),self._active['prices']):
            raise RuntimeError('Price signal does not use frozen interval prices')
        if signal['selected_control'] != _commands(result.control):
            raise RuntimeError('Price signal does not identify selected control')
        c,A,d = (np.asarray(signal[k],dtype=float) for k in ('c_scaled','A_scaled','proposal'))
        raw = np.asarray(signal['raw_local_residual_scaled'],dtype=float)
        target = np.array([result.control.N_P_star,result.control.N_UF_star])
        if c.shape!=(2,) or A.shape!=(2,len(d)) or raw.shape!=(2,) or not all(np.all(np.isfinite(x)) for x in (c,A,d,raw)):
            raise RuntimeError('Invalid raw proposal signal')
        if not np.array_equal(raw,c+A@d) or not np.array_equal(c,(np.asarray(signal['anchor_budget_vector'])-target)/self._scales()):
            raise RuntimeError('Mixed anchor/target in raw proposal signal')
        before = self._persistent_prices.copy()
        after = self._price_update(before,raw) if self.mode=='interval' else before.copy()
        if not np.all(np.isfinite(after)) or after[0]<0:
            raise RuntimeError('Nonfinite boundary price update')
        row = {'interval_token':self._active['token'],'mode':self.mode,
            'before_normalized':before.tolist(),'after_normalized':after.tolist(),
            'raw_local_residual_scaled':raw.tolist(),'selected_budget':target.tolist(),
            'signal_iteration':signal['iteration'],'relation':signal['relation'],
            'updated':self.mode=='interval','price_update_count':1 if self.mode=='interval' else 0}
        # 모든 실행 gate와 일관성 검사가 끝난 뒤에만 영속 가격에 쓴다.
        self._persistent_prices = after
        self.interval_metrics['boundary_price_updates'] += int(self.mode=='interval')
        self.interval_metrics['decision_seconds'] = time.perf_counter()-self._active['started']
        row['instrumentation'] = dict(self.interval_metrics)
        self.price_rows.append(row)
        self.profile_rows.append(dict(self.interval_metrics))
        self.last_result = result
        self.previous_control = result.control.copy()
        self.abort_interval()
        return row

    def decide(self, state, forecast, previous=None, cfg=None):
        previous = (previous or self.previous_control or ControlAction.uncontrolled(self.cfg)).copy()
        forecast = list(forecast)
        self.begin_interval(state,forecast,previous)
        try:
            coords = self.coordinate_type(self.cfg,self.options,previous)
            hint = self.warm_start_control or previous
            warm = coords.decode(coords.encode(hint),coords.initial_vsl(hint))
            self.warm_start_control = None
            witness = self.evaluate_control(state,forecast,warm,previous)
            b0 = PlayerBudget(witness.achieved_np_veh,witness.commanded_nuf_veh_h)
            candidates = [(b0,'warm_start_rollout_witness')]
            leader_cfg = self.cfg.with_updates({'mpc':{'horizon_steps':self.options.horizon_steps}})
            raw = Leader(leader_cfg).candidates(state,previous,forecast=forecast[:self.options.horizon_steps])
            raw = sorted(raw,key=lambda b:(abs(b.N_P_star-b0.np_cap_veh)/self._scales()[0]+abs(b.N_UF_star-b0.nuf_veh_h)/self._scales()[1],b.N_P_star,b.N_UF_star))
            for action in raw:
                if len(candidates)>=self.options.max_candidates:
                    break
                budget = PlayerBudget(action.N_P_star,action.N_UF_star)
                if budget not in [b for b,_ in candidates]:
                    candidates.append((budget,'existing_leader_raw_pair'))
            for index,(budget,origin) in enumerate(candidates):
                result = self.solve_fixed_budget(state,forecast,previous,budget,warm)
                self.candidate_rows.append({'candidate':index,'origin':origin,
                    'requested_np_cap_veh':budget.np_cap_veh,'requested_nuf_veh_h':budget.nuf_veh_h,
                    'achieved_np_veh':result.evaluation.achieved_np_veh,
                    'commanded_nuf_veh_h':result.evaluation.commanded_nuf_veh_h,
                    'actual_nuf_veh_h':result.evaluation.actual_nuf_veh_h,
                    'residual_np_veh':result.residual_np_veh,'residual_nuf_veh_h':result.residual_nuf_veh_h,
                    'computation_time_sec':result.instrumentation['candidate_seconds'],
                    'objective_ttt':result.objective,'feasible':result.feasible,'converged':result.converged,
                    'conditional_primal_local':result.conditional_primal_local,'selected':False,
                    'iterations':result.iterations,'status':result.status,'termination_reason':result.termination_reason,
                    **result.instrumentation})
            selected = self.select_candidate()
            self.candidate_rows[next(i for i,r in enumerate(self.last_results) if r is selected)]['selected'] = True
            self.commit_selected(selected)
            return selected.control.copy()
        except BaseException:
            self.abort_interval()
            raise

    # BEGIN GENERATED ORIGINAL SOLVE BODY
    def _solve_impl(self, state: TrafficState, forecast: Sequence[DemandStep], previous: ControlAction | None,
                           budget: PlayerBudget, initial_control: ControlAction | None = None) -> PlayerSDMPCResult:
        if not np.all(np.isfinite(budget.vector())):
            raise ValueError("budget must be finite")
        opts, cfg = self.options, self.cfg
        previous = (previous or ControlAction.uncontrolled(cfg)).copy()
        coords = self.coordinate_type(cfg, opts, previous)
        seed = initial_control or previous
        y, vsl = coords.encode(seed), coords.initial_vsl(seed)
        cache = self._decision_cache if self._decision_cache is not None else {}
        evaluations = 0
        def evaluate(z, velocities=None):
            nonlocal evaluations
            vv = vsl if velocities is None else velocities
            key = (np.asarray(z, dtype=float).tobytes(), tuple(sorted(vv.items())))
            self.interval_metrics['rollout_lookup_calls'] += 1
            if key in cache:
                self.interval_metrics['coordinate_cache_hits'] += 1
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
        prices = self._active['prices'].copy() if self.mode=='interval' else np.zeros(2)
        trust = opts.trust_radius
        best_signal = None
        conditional = False
        stationarity, reason, converged = math.inf, "iteration_limit", False
        price_stationarity, dual_step_norm, complementarity = math.inf, math.inf, math.inf
        local_resource_violation = math.inf

        def derivatives(z, vv):
            return self._derivatives(evaluate,z,vv,coords)

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
                old_vsl = dict(vsl)
                anchor_budget_vector = current.budget_vector.copy()
                gradients, Araw, A = derivatives(y, vsl)
                own, external = np.zeros(len(y)), np.zeros(len(y))
                for j, owner in enumerate(coords.owners):
                    own[j] = gradients[self.player_ids.index(owner), j]
                    external[j] = sum(gradients[:, j])-own[j] if opts.externality_enabled else 0.
                price_used = prices.copy()
                price_gradient = self._price_gradient(A,price_used)
                proposal = np.zeros_like(y)
                local_ok = True
                # Jacobi의 같은 y/vsl을 유지한 채 각 실제 player의 nonlinear NLP를 푼다.
                for pid in self.player_ids:
                    indices = np.array([j for j,p in enumerate(coords.owners) if p == pid], dtype=int)
                    def own_callback(s, ix=indices, player=pid):
                        trial = y.copy()
                        trial[ix] += s
                        return evaluate(trial).player_costs[player]
                    local = self._local(own_callback, (external+price_gradient)[indices],
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
                # 동일 반복의 원 c/A/지역 제안만 기록한다. QP 투영 뒤 잔차로 대체하지 않는다.
                signal = {'interval_token':self._active['token'],'context':self._active['context'],
                    'iteration':iteration,'anchor_y':old_y.tolist(),'anchor_vsl':old_vsl,
                    'anchor_budget_vector':anchor_budget_vector.tolist(),'c_scaled':c.tolist(),
                    'A_scaled':A.tolist(),'proposal':proposal.tolist(),
                    'raw_local_residual_scaled':raw_local_residual.tolist(),
                    'prices_used_normalized':price_used.tolist()}
                prices = self._price_update(price_used,raw_local_residual) if self.mode=='inner' else price_used.copy()
                dual_step_norm = float(np.max(np.abs(prices-price_used)))
                local_resource_violation = mixed_violation(raw_local_residual)
                complementarity = abs(float(prices[0]*c[0]))
                lag_gradient = own+external+self._price_gradient(A,prices)
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
                # 원 best를 만든 반복의 pre/post 관계 보존. 후보 평가 중 가격은 커밋하지 않는다.
                if best is not None and np.array_equal(y,best[0]) and vsl==best[1] and (
                        accepted or (np.array_equal(old_y,y) and old_vsl==vsl)):
                    best_signal = copy.deepcopy(signal)
                    best_signal.update(selected_y=y.tolist(),selected_vsl=dict(vsl),
                        relation='accepted_iteration_post_line_restore_or_vsl' if accepted else 'unchanged_best_identical_anchor')
                conditional = bool(opts.local_solver=='slsqp' and local_ok and qp.success and not discrete_improved
                    and poll_complete and self._feasible(current,budget) and step_norm<=1e-8
                    and stationarity<=opts.stationarity_tolerance)
                rows[-1]['conditional_primal_local'] = conditional
                rows[-1]['price_mode'] = self.mode
                # Pattern의 local mesh 통과는 global feasible mesh 통과가 아니므로 승격하지 않는다.
                dual_ok = (max(price_stationarity,dual_step_norm,complementarity) <= opts.stationarity_tolerance
                           and raw_local_residual[0] <= opts.tolerance_np_veh/self._scales()[0]
                           and abs(raw_local_residual[1]) <= opts.tolerance_nuf_veh_h/self._scales()[1])
                if opts.local_solver == "slsqp" and local_ok and dual_ok and qp.success and not discrete_improved and poll_complete and self._feasible(current,budget) and step_norm <= 1e-8 and stationarity <= opts.stationarity_tolerance:
                    converged,reason = True,"feasible_numerical_projected_stationarity_local_nlp_and_discrete_poll"
                    break
                # 구간 고정 가격의 조건부 primal/local 정지: 원 수렴 PASS와 분리한다.
                if self.mode=='interval' and conditional:
                    reason = 'conditional_primal_local_frozen_price'
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
        result = PlayerSDMPCResult(control,current.total_ttt,current,float(c[0]),float(c[1]),feasible,feasible and converged,
                                 status,len(rows)-1,rows,sensitivities,locals_log,reason)
        if best_signal is not None:
            best_signal['selected_control'] = copy.deepcopy(_commands(control))
        result.price_update_signal = best_signal
        result.conditional_primal_local = bool(feasible and conditional)
        return result

