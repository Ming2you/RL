"""C 실험: 연속 VSL 최적화 → 허용값 양자화 → 원 물리·예산 재검사.

A/B의 frozen_price_controller.py 및 historical 원본은 수정하지 않는다.
연속 VSL FD는 0.1 km/h, 좌표는 speed/100이며 기존 scalar proximal을 유지한다.
양자화 후 수렴성은 재인증하지 않으므로 result.converged/conditional은 False다.
역사 METANET의 vsl_active 문턱(max(vsl_set)-0.5)과 모든 물리는 그대로다.
허용 집합이 단일 값이면 해당 연속축은 고정되며 VSL 자유도를 새로 만들지 않는다.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import replace

import numpy as np

from frozen_price_controller import (
    FrozenPriceSDMPC, PlayerControlCoordinates, PlayerBudget,
    validate_player_control, mixed_resource_step, mixed_violation, _commands,
)
from src.models.state import segment_vsl


class RelaxedVSLCoordinates(PlayerControlCoordinates):
    """원 12축 뒤에 원 권한의 VSL축만 추가하고 격자 poll은 별도 단계로 옮긴다."""
    vsl_scale = 100.0
    fd_vsl_km_h = 0.1

    def __init__(self,cfg,options,previous):
        self.base = PlayerControlCoordinates(cfg,options,previous)
        self.cfg,self.options,self.previous = cfg,options,previous.copy()
        self.continuous_count = len(self.base.axes)
        self.relaxed_vsl_indices = tuple(self.base.active_vsl)
        self.axes = list(self.base.axes)
        for link,index in self.relaxed_vsl_indices:
            allowed = self.base.allowed_vsl(link,index)
            if not allowed:
                raise ValueError('No executable VSL value under previous-step bounds')
            self.axes.append(('vsl',f'{link}__seg{index}',self.vsl_scale,
                min(allowed),max(allowed),self.fd_vsl_km_h,f'FW:{link}'))
        self.scales = np.array([axis[2] for axis in self.axes])
        self.lower = np.array([axis[3]/axis[2] for axis in self.axes])
        self.upper = np.array([axis[4]/axis[2] for axis in self.axes])
        self.fd = np.array([axis[5]/axis[2] for axis in self.axes])
        self.owners = tuple(axis[6] for axis in self.axes)
        self.active_vsl = ()

    def encode(self,control):
        first = self.base.encode(control)
        speeds = [segment_vsl(control,link,index,self.cfg)/self.vsl_scale
                  for link,index in self.relaxed_vsl_indices]
        return np.clip(np.concatenate((first,np.asarray(speeds))),self.lower,self.upper)

    def initial_vsl(self,control):
        return self.base.initial_vsl(control)

    def decode(self,y,vsl):
        control = self.base.decode(np.asarray(y)[:self.continuous_count],vsl)
        for value,(link,index) in zip(np.asarray(y)[self.continuous_count:],self.relaxed_vsl_indices):
            control.vsl[f'{link}__seg{index}'] = float(value*self.vsl_scale)
        for link in self.cfg.network.freeway_links:
            control.vsl[link] = min(control.vsl[f'{link}__seg{i}'] for i in range(self.cfg.network.freeway_segments_per_link))
        return control


class ContinuousVSLPriceSDMPC(FrozenPriceSDMPC):
    coordinate_type = RelaxedVSLCoordinates

    def __init__(self,cfg,options=None,*,mode='interval'):
        super().__init__(cfg,options,mode=mode)
        self._relaxed_phase = False
        self.quantization_rows = []

    def begin_interval(self,state,forecast,previous=None):
        token = super().begin_interval(state,forecast,previous)
        self.quantization_rows = []
        return token

    @staticmethod
    def _with_speeds(control,speeds,cfg):
        clone = control.copy()
        clone.vsl.update(speeds)
        for link in cfg.network.freeway_links:
            clone.vsl[link] = min(clone.vsl[f'{link}__seg{i}'] for i in range(cfg.network.freeway_segments_per_link))
        return clone

    def _nearest(self,control,previous):
        coords = PlayerControlCoordinates(self.cfg,self.options,previous)
        values = {}
        for link,index in coords.active_vsl:
            desired = segment_vsl(control,link,index,self.cfg)
            allowed = coords.allowed_vsl(link,index)
            if not allowed or not math.isfinite(desired):
                raise ValueError('Invalid VSL quantization input')
            values[f'{link}__seg{index}'] = min(allowed,key=lambda speed:(abs(speed-desired),-speed))
        return self._with_speeds(control,values,self.cfg)

    def _relaxed_validation(self,control,previous):
        coords = self.coordinate_type(self.cfg,self.options,previous)
        errors = []
        for link,index in coords.relaxed_vsl_indices:
            value = segment_vsl(control,link,index,self.cfg)
            allowed = coords.allowed_vsl(link,index)
            if not math.isfinite(value) or not min(allowed)-1e-9 <= value <= max(allowed)+1e-9:
                errors.append(f'relaxed_vsl_bounds:{link}__seg{index}')
        for link in self.cfg.network.freeway_links:
            actual = [control.vsl.get(f'{link}__seg{i}',float('nan')) for i in range(self.cfg.network.freeway_segments_per_link)]
            if not all(math.isfinite(x) for x in actual) or control.vsl.get(link) != min(actual):
                errors.append(f'relaxed_vsl_summary:{link}')
        # 검증용 clone에서 active VSL만 양자화한다. 나머지 권한/제어 gate는 원본 그대로다.
        try:
            quantized = self._nearest(control,previous)
            checked = validate_player_control(quantized,previous,self.cfg)
            errors.extend(checked['violations'])
        except (ValueError,KeyError):
            errors.append('relaxed_vsl_quantization_input')
        return {'valid':not errors,'violations':errors,'domain':'continuous_active_vsl_only'}

    def evaluate_control(self,state,forecast,control,previous=None):
        raw = super().evaluate_control(state,forecast,control,previous)
        if not self._relaxed_phase:
            return raw
        previous = previous or self._active['previous']
        relaxed = self._relaxed_validation(control,previous)
        # 원 cache의 control_valid=False와 원 검증 내역은 변경하지 않는다.
        diagnostics = copy.deepcopy(raw.diagnostics)
        diagnostics['original_discrete_control_validation'] = copy.deepcopy(raw.diagnostics['control_validation'])
        diagnostics['relaxed_control_validation'] = relaxed
        diagnostics['execution_authorized'] = False
        return replace(raw,control_valid=relaxed['valid'],diagnostics=diagnostics)

    def _solve_impl(self,state,forecast,previous,budget,initial_control=None):
        started = time.perf_counter()
        rollout_start = self.interval_metrics['rollout_calls']
        self._relaxed_phase = True
        try:
            result = super()._solve_impl(state,forecast,previous,budget,initial_control)
        finally:
            self._relaxed_phase = False
        postprocess_started = time.perf_counter()
        proven_impossible = result.status == 'infeasible_proven'
        original_termination = result.termination_reason
        relaxed_control = result.control.copy()
        relaxed_diagnostics = {
            'status':result.status,
            'objective':result.objective,'feasible':result.feasible,'original_gate_in_relaxed_domain':result.converged,
            'conditional_primal_local_in_relaxed_domain':result.conditional_primal_local,
            'termination_reason':result.termination_reason,
            'stationarity':result.control.diagnostics.get('player_sdmpc_stationarity_norm'),
            'control':copy.deepcopy(_commands(relaxed_control)),
            'continuous_vsl_fd_km_h':RelaxedVSLCoordinates.fd_vsl_km_h,
            'vsl_normalization_scale':RelaxedVSLCoordinates.vsl_scale,
            'vsl_proximal':self.options.proximal,
            'solve_seconds':postprocess_started-started,
            'historical_vsl_activation_threshold_km_h':max(self.cfg.freeway_follower.vsl_set)-0.5,
            'fixed_vsl_axes':[f'{link}__seg{index}' for link,index in
                self.coordinate_type(self.cfg,self.options,previous).relaxed_vsl_indices
                if len(PlayerControlCoordinates(self.cfg,self.options,previous).allowed_vsl(link,index))==1],
        }
        for row in result.iteration_rows:
            row['optimization_domain'] = 'continuous_active_vsl_relaxation'
        for row in result.local_rows:
            row['optimization_domain'] = 'continuous_active_vsl_relaxation'
        options = self.options
        coords = PlayerControlCoordinates(self.cfg,options,previous)
        trust = float(result.iteration_rows[-1].get('trust_radius',options.trust_radius))
        trials = []

        def evaluate(z,vv):
            # 실행 도메인: 원 12 연속축만 복원하며 모든 VSL은 격자값으로 고정한다.
            candidate = coords.decode(z,vv)
            candidate.N_P_star,candidate.N_UF_star = budget.np_cap_veh,budget.nuf_veh_h
            return self.evaluate_control(state,forecast,candidate,previous)

        def violation(ev):
            return mixed_violation((ev.budget_vector-budget.vector())/self._scales())

        def trial(control,label):
            z,vv = coords.encode(control),dict(control.vsl)
            before_count = self.interval_metrics['rollout_calls']
            ev = evaluate(z,vv)
            initial_obj = ev.total_ttt
            initial_residual = (ev.budget_vector-budget.vector()).tolist()
            restoration = []
            if not proven_impossible and not self._feasible(ev,budget) and options.restoration_iterations:
                _,_,A = self._derivatives(evaluate,z,vv,coords)
                A = A.copy()
                # 원 restoration의 QP·수락·Broyden 식과 반복/step 경계를 그대로 사용한다.
                for iteration in range(1,options.restoration_iterations+1):
                    if self._feasible(ev,budget):
                        break
                    c = (ev.budget_vector-budget.vector())/self._scales()
                    qp = mixed_resource_step(np.zeros_like(z),A,c,
                        np.maximum(coords.lower-z,-trust),np.minimum(coords.upper-z,trust))
                    found = False
                    for alpha in (1.,.5,.25):
                        point = np.clip(z+alpha*qp.step,coords.lower,coords.upper)
                        test = evaluate(point,vv)
                        accepted = bool(test.control_valid and test.physical_valid and violation(test)<violation(ev)-1e-10)
                        restoration.append({'iteration':iteration,'alpha':alpha,'qp_success':qp.success,
                            'objective':test.total_ttt,'residual':(test.budget_vector-budget.vector()).tolist(),
                            'physical_valid':test.physical_valid,'control_valid':test.control_valid,'accepted':accepted})
                        if accepted:
                            delta = point-z
                            dc = (test.budget_vector-ev.budget_vector)/self._scales()
                            if delta@delta>1e-16:
                                A += np.outer(dc-A@delta,delta)/(delta@delta)
                            z,ev,found = point,test,True
                            break
                    if not found:
                        break
            candidate = coords.decode(z,vv)
            candidate.N_P_star,candidate.N_UF_star = budget.np_cap_veh,budget.nuf_veh_h
            # 원 실행 검사를 별도로 재확인한다. relaxed_valid는 여기서 사용하지 않는다.
            exact_check = validate_player_control(candidate,previous,self.cfg)
            feasible = bool(self._feasible(ev,budget) and exact_check['valid'])
            row = {'label':label,'initial_objective':initial_obj,'initial_residual':initial_residual,
                'objective':ev.total_ttt,'residual':(ev.budget_vector-budget.vector()).tolist(),
                'feasible':feasible,'physical_valid':ev.physical_valid,'control_validation':exact_check,
                'vsl':dict(candidate.vsl),'restoration':restoration,'restored':any(r['accepted'] for r in restoration),
                'new_rollout_calls':self.interval_metrics['rollout_calls']-before_count,
                'candidate_budget':budget.vector().tolist()}
            trials.append(row)
            return candidate,ev,feasible,row

        nearest = self._nearest(relaxed_control,previous)
        first = trial(nearest,'nearest_allowed')
        tested = [first]
        if not proven_impossible and not first[2]:
            # 필요할 때만 한 축의 인접 허용값을 시험한다. 전체 Cartesian 격자는 만들지 않는다.
            for link,index in coords.active_vsl:
                key = f'{link}__seg{index}'
                desired = segment_vsl(relaxed_control,link,index,self.cfg)
                allowed = coords.allowed_vsl(link,index)
                lower = [value for value in allowed if value<desired]
                upper = [value for value in allowed if value>desired]
                neighbors = ([max(lower)] if lower else [])+([min(upper)] if upper else [])
                for speed in sorted(set(neighbors)):
                    if speed == nearest.vsl[key]:
                        continue
                    candidate = self._with_speeds(nearest,{key:speed},self.cfg)
                    tested.append(trial(candidate,f'neighbor:{key}:{speed:g}'))
        eligible = [item for item in tested if item[2]]
        selected = min(eligible,key=lambda item:(item[1].total_ttt,repr(sorted(item[0].vsl.items())))) if eligible else first
        final_control,final_ev,physical_feasible,chosen = selected
        # 가격 proposal이 없으면 실행 선택으로 승격하지 않는다. 새 proposal을 예산 밖에서 만들지 않는다.
        signal_available = result.price_update_signal is not None
        executable = bool(not proven_impossible and physical_feasible and signal_available)
        result.control,result.evaluation,result.objective = final_control,final_ev,final_ev.total_ttt
        residual = final_ev.budget_vector-budget.vector()
        result.residual_np_veh,result.residual_nuf_veh_h = float(residual[0]),float(residual[1])
        result.feasible,result.converged,result.conditional_primal_local = executable,False,False
        result.status = 'infeasible_proven' if proven_impossible else 'feasible_but_not_converged' if executable else 'unresolved_algorithm_failure'
        result.termination_reason = original_termination if proven_impossible else 'quantized_execution_checked_stationarity_unchecked' if executable else 'quantized_execution_failclosed'
        if signal_available:
            signal = result.price_update_signal
            signal['relaxed_selected_y'] = signal['selected_y']
            signal['relaxed_selected_vsl'] = signal['selected_vsl']
            signal['selected_y'] = self.coordinate_type(self.cfg,options,previous).encode(final_control).tolist()
            signal['selected_vsl'] = dict(final_control.vsl)
            signal['selected_control'] = copy.deepcopy(_commands(final_control))
            signal['relation'] += '+quantized_execution:'+chosen['label']+(':continuous_budget_restored' if chosen['restored'] else '')
            signal['quantization_provenance'] = {'chosen_label':chosen['label'],
                'relaxed_objective':relaxed_diagnostics['objective'],'executed_objective':final_ev.total_ttt,
                'no_fresh_local_price_proposal':True}
        result.relaxed_diagnostics = relaxed_diagnostics
        result.quantization_rows = trials
        self.quantization_rows.extend(copy.deepcopy(trials))
        final_control.diagnostics = copy.deepcopy(relaxed_control.diagnostics)
        final_control.diagnostics.update({
            'player_sdmpc_active':True,'player_sdmpc_feasible':executable,'player_sdmpc_converged':False,
            'budget_contract_failed':not physical_feasible,'fallback':not executable,
            'player_sdmpc_np_service_veh':final_ev.achieved_np_veh,
            'player_sdmpc_command_nuf_veh_h':final_ev.commanded_nuf_veh_h,
            'player_sdmpc_np_residual_veh':float(residual[0]),'player_sdmpc_nuf_residual_veh_h':float(residual[1]),
            'player_sdmpc_stationarity_norm':float('nan'),'post_quantization_stationarity_checked':False,
            'player_sdmpc_rollout_evaluations':self.interval_metrics['rollout_calls']-rollout_start,
            'post_quantization_original_control_valid':bool(validate_player_control(final_control,previous,self.cfg)['valid']),
            'continuous_vsl_relaxation':relaxed_diagnostics,
            'vsl_quantization':{'chosen':chosen['label'],'trial_count':len(trials),'feasible_count':len(eligible),
                'execution_gate_pass':executable,'price_signal_available':signal_available,
                'postprocessing_seconds':time.perf_counter()-postprocess_started},
        })
        return result
