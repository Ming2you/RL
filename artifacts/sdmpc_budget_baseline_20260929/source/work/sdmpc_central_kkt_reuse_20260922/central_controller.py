"""완전한 현재 condensed 중앙 식에서 최종 제어를 고정한 승수 회수."""
import copy
import time
import numpy as np
from fixed_controller import FixedNPBandSDMPC
from fixed_policy import halfwidth
from controller import SelectedDualSDMPC
from group_block_engine import SparseRolloutEngine
from sparse.dual import primal,derivative
from central import physical_rows,assemble,recover


class RecordedEngine(SparseRolloutEngine):
    def evaluate(self,*args,**kwargs):
        self.last_result=None
        result=super().evaluate(*args,**kwargs)
        self.last_result=result
        return result


class CentralKKTSDMPC(SelectedDualSDMPC):
    def __init__(self,cfg,options):
        super().__init__(cfg,options)
        self.engine=RecordedEngine(cfg,options.horizon_steps)
        self.engine.ownership=self.ownership

    def begin(self,*args):
        super().begin(*args)
        self.tangent_cache={};self.central_cache={}

    def derivatives(self,y):
        key=np.asarray(y,dtype=float).tobytes();fresh=key not in self.dcache
        if getattr(self,"record_lower_model_calls",False):self.lower_model_references.append(np.asarray(y).copy())
        result=super().derivatives(y)
        if fresh:self.tangent_cache[key]=getattr(self.engine,'last_result',None)
        return result

    def central_system(self,y,budget):
        from reused_model import build
        return build(self,y,budget)

    def solve(self,budget,seed,initial_dual):
        # 기존 하위 계산을 그대로 실행하고 이전 부분 승수 추정만 건너뛴다.
        self.lower_model_references=[];self.record_lower_model_calls=True
        try:result=FixedNPBandSDMPC.solve(self,budget,seed,initial_dual)
        finally:self.record_lower_model_calls=False
        started=time.perf_counter();cpu=time.process_time();before=copy.deepcopy(self.counts)
        y=np.asarray(result['point']).copy();b=np.asarray(result['budget']);ev=self.evaluate(y)
        try:g,h,C,names,check=self.central_system(y,b)
        except RuntimeError as exc:
            diagnostic=dict(fit_success=False,stationarity_inf=None,accepted_for_leader_direction=False,
                gradient_estimate=None,reference_point=y.tolist(),budget=b.tolist(),
                rejection_reasons=[str(exc)],control_optimized=False,
                wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                counts_delta={k:self.counts[k]-before[k] for k in before})
            result.update(iteration_budget_gradient_hint=result['budget_gradient_hint'],
                iteration_direction=result['direction'],leader_multiplier=diagnostic,
                budget_gradient_hint=None,direction=[0.,0.])
            return result
        fit_started=time.perf_counter()
        full=recover(g,h,C,names,b,self.scales,tolerance=self.options.stationarity_tolerance)
        # 이산 VSL 조건부 문제도 따로 계산해, 기존 14축 진단과 비교 가능하게 한다.
        continuous=np.array([j for j,a in enumerate(self.coords.axes) if a[0]!='vsl_group'])
        conditional=recover(g[continuous],h,C[:,continuous],names,b,self.scales,
                            tolerance=self.options.stationarity_tolerance)
        multiplier_seconds=time.perf_counter()-fit_started
        executable=self.feasible(y,b,True)
        # 엄격한 최적성 인증과 근사 budget 방향의 사용을 구분한다. 새 budget은 다시 풀고 검사한다.
        usable=bool(executable and full['fit_success'] and np.all(np.isfinite(full['gradient_estimate']))
                    and b[1]!=0)
        gradient=full.get('gradient_estimate');direction=np.zeros(2)
        if usable:direction=np.where(abs(np.asarray(gradient))>1e-8,-np.sign(gradient),0.)
        quality=bool(full.get('stationarity_pass',False) and full.get('complementarity_pass',False)
                     and check['derivative_check_pass'] and not check['exact_tie_axes']
                     and executable and full.get('multiplier_identifiability_sufficient_check',False))
        diagnostic=dict(full,reference_point=y.tolist(),budget=b.tolist(),
            original_residual=result['original_residual'],selected_control=copy.deepcopy(result['control']),
            accepted_for_leader_direction=usable,leader_signal_kind='approximate_central_multiplier',
            first_order_diagnostics_pass=quality,optimality_certified=False,
            linear_model_max_violation=float(max(0.,np.max(check['affine_constraint_values']))),
            actual_max_violation=float(max(0.,np.max(check['actual_constraint_values']))),
            actual_complementarity=(max((abs(row['multiplier']*check['actual_constraint_values'][row['index']])
                                        for row in full.get('active_rows',[])),default=0.)),
            scope='fixed final original constraints with reused first-order central coefficients',
            conditional_fixed_VSL=conditional,derivative_evidence=check,
            physical_validation_passed=ev.physical_valid,executable=executable,
            accounting_residual=abs(sum(ev.player_costs.values())-ev.total_ttt),
            constraint_names=names,constraint_values=h.tolist(),constraint_jacobian=C.tolist(),
            total_TTT_gradient=g.tolist(),constraint_coverage='all inequalities of existing condensed gate; numerical audits classified separately',
            eliminated_equalities=['coupled dynamics','green sum','four-segment common VSL','offset periodic coordinate'],
            original_execution_audits=dict(conservation=ev.diagnostics['max_conservation_residual_veh'],
                numerical_loss=ev.diagnostics['numerical_vehicle_loss_veh'],
                accounting=ev.diagnostics['accounting_residual_veh_h'],
                control=self.coords.validate(result['control'],True)),
            rejection_reasons=([] if usable else ['selected_action_not_executable_or_fit_failed']),
            nnls_two_systems_wall_seconds=multiplier_seconds,
            wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
            counts_delta={k:self.counts[k]-before[k] for k in before})
        result.update(iteration_budget_gradient_hint=result['budget_gradient_hint'],
            iteration_direction=result['direction'],leader_multiplier=diagnostic,
            leader_dual_source='reused_first_order_central_model_at_fixed_selected_control',
            leader_dual_reference=y.tolist(),leader_dual_matches_selected=True,
            leader_model_reference=check['model_reference'],
            leader_derivative_reference_matches_selected=check['model_matches_selection'],
            budget_gradient_hint=gradient if usable else None,direction=direction.tolist(),
            leader_direction_source='approximate_central_multiplier' if usable else 'explicit_axis_search',
            elapsed_wall=result['elapsed_wall']+diagnostic['wall_seconds'])
        np.testing.assert_array_equal(y,result['point'])
        assert self.counts['scalar_rollouts']==before['scalar_rollouts']
        assert self.counts['tangent_rollouts']==before['tangent_rollouts']
        return result
