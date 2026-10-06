"""매 interval PFO 달성 budget에서 탐색하고 실행 가능한 PFO TTT를 보존한다.

명세04/05/10/12/15 및 사용자 지정 S-DMPC 계약 적용.
하위 계산·가격 갱신·band·기존 예외 기록은 상속하며 실제 선택만 재검사한다.
"""
import numpy as np
from exception_controller import BudgetExceptionSDMPC
from fixed_controller import GridCoordinates
from fixed_policy import mask


class PFOAnchorSDMPC(BudgetExceptionSDMPC):
    def decide(self,state,forecast,previous,warm):
        # 제거한 제약의 가격은 carry-over/warm start에도 남기지 않는다.
        self.dual=np.where(mask(),self.dual,0.)
        incoming_dual=self.dual.copy()
        prior_budget=None if self.last_budget is None else self.last_budget.copy()
        # 그룹별 VSL/이산 격자/이전 제어 대비 제한을 먼저 적용한다.
        # budget witness와 하위 초기값은 반드시 이 동일한 제어를 사용한다.
        coords=GridCoordinates(self.cfg,self.options,previous)
        mapped=coords.decode(coords.quantize(coords.encode(warm)))
        self.last_budget=None
        selected,results=super().decide(state,forecast,previous,mapped)
        seed=self.coords.encode(mapped)
        witness=self.evaluate(seed)
        budget=witness.budget_vector.copy()
        check=self.execution_check(seed,budget)
        # 첫 요청은 새로운 interval의 witness다. 후보 실패 목표를 사후 변경하지 않는다.
        if not results or not np.array_equal(results[0]['budget'],budget):
            raise RuntimeError('PFO witness and first requested budget differ')
        audit=dict(center_source='current_interval_mapped_PFO_rollout',
            previous_selected_budget=None if prior_budget is None else prior_budget.tolist(),
            reference_budget=budget.tolist(),reference_point=seed.tolist(),
            reference_TTT=witness.total_ttt,reference_check=check,
            price_before=incoming_dual.tolist(),lower_candidate_count=len(results),
            lower_selected_TTT=None if selected is None else selected['evaluation'].total_ttt,
            lower_selected_budget=None if selected is None else list(selected['budget']),
            guarantee_scope='same_state_same_H3_prediction_only',long_run_guarantee=False)
        if not check['physical_control_valid'] or not check['budget_feasible']:
            # 기준이 원 물리/제어 검사에 실패하면 guard를 통과했다고 표시하지 않는다.
            self.dual=incoming_dual;self.last_budget=prior_budget
            audit.update(status='invalid_PFO_reference_fail_closed',selected_source=None)
            self.anchor_audit=audit
            return None,results
        keep_lower=(selected is not None and selected['feasible'] and
                    selected['execution_check']['physical_control_valid'] and
                    selected['execution_check']['budget_feasible'] and
                    selected['evaluation'].total_ttt<=witness.total_ttt)
        if not keep_lower:
            # 최적화하지 않은 PFO 기준점에 다른 최종점의 승수/수렴 진단을 붙이지 않는다.
            control=self.coords.decode(seed)
            control.N_P_star,control.N_UF_star=budget
            selected=dict(budget=budget.tolist(),point=seed.copy(),control=control,evaluation=witness,
                feasible=True,converged=False,execution_allowed=True,execution_exception=False,
                execution_check=check,status='PFO_reference_selected_uncertified',
                reason='PFO_reference_TTT_guard',original_residual=check['original_residual'],
                stationarity=None,stationarity_point=None,local_solvers_converged=False,
                dual=incoming_dual.copy(),leader_multiplier=None,selection_source='PFO_reference')
            self.dual=incoming_dual.copy()
        else:
            selected['selection_source']='lower_solution'
        self.last_budget=np.asarray(selected['budget']).copy()
        audit.update(status='passed',selected_source=selected['selection_source'],
            selected_TTT=selected['evaluation'].total_ttt,
            predicted_TTT_nonworsening=selected['evaluation'].total_ttt<=witness.total_ttt,
            price_after=self.dual.tolist(),
            price_policy='hold_incoming_for_PFO_reference_else_inherited_selected_policy')
        self.anchor_audit=audit
        return selected,results
