"""기존 하위 반복은 보존하고 최종 선택점의 승수만 leader 방향 후보로 사용한다."""
import copy
import time
import numpy as np
from fixed_controller import FixedNPBandSDMPC
from fixed_policy import halfwidth
from multiplier import recover,direction_from_recovery


class SelectedDualSDMPC(FixedNPBandSDMPC):
    def solve(self,budget,seed,initial_dual):
        result=super().solve(budget,seed,initial_dual)
        started=time.perf_counter();cpu=time.process_time();before=copy.deepcopy(self.counts)
        y=np.asarray(result['point']).copy();b=np.asarray(result['budget'])
        ev=self.evaluate(y);G,A=self.derivatives(y)
        # キャッシュ hit のときも「最後の行」ではなく選択点と一致する診断を参照する。
        evidence=next(row for row in reversed(self.derivative_rows)
                      if np.array_equal(row['anchor'],y))
        ix=np.array([j for j,axis in enumerate(self.coords.axes) if axis[0]!='vsl_group'])
        h=np.concatenate(((ev.budget_vector-b-halfwidth(b))/self.scales,
                          (-ev.budget_vector+b-halfwidth(b))/self.scales))
        diagnostic=recover(G.sum(axis=0)[ix],A[:,ix],h,y[ix],
            self.coords.lower[ix],self.coords.upper[ix],b,self.scales,
            executable=self.feasible(y,b,True),
            derivatives_verified=bool(evidence.get('derivatives_certified',False)) and
                not set(evidence['fd_columns']).intersection(ix),
            # 既存評価器は物理有効性 bool だけを返す。活性制約の完全性とは別である。
            physical_constraints_verified=False,
            stationarity_tolerance=self.options.stationarity_tolerance)
        diagnostic.update(reference_point=y.tolist(),budget=b.tolist(),
            original_residual=(ev.budget_vector-b).tolist(),normalized_inequalities=h.tolist(),
            selected_control=copy.deepcopy(result['control']),
            continuous_coordinate_indices=ix.tolist(),derivative_evidence=copy.deepcopy(evidence),
            physical_validation_passed=ev.physical_valid,
            accounting_residual=abs(sum(ev.player_costs.values())-ev.total_ttt),
            physical_evidence_limitation='rollout boolean gate supplies no certified active physical Jacobian',
            wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
            counts_delta={k:self.counts[k]-before[k] for k in before})
        # 内部価格の warm start は変更しない。leader では旧 hint を一切参照しない。
        result.update(iteration_budget_gradient_hint=result['budget_gradient_hint'],
            iteration_direction=result['direction'],leader_multiplier=diagnostic,
            leader_dual_source='recovered_at_final_selected_action',
            leader_dual_reference=y.tolist(),leader_dual_matches_selected=True,
            budget_gradient_hint=(diagnostic['gradient_estimate']
                if diagnostic['accepted_for_leader_direction'] else None),
            direction=direction_from_recovery(diagnostic).tolist(),
            leader_direction_source=('selected_action_multiplier' if diagnostic['accepted_for_leader_direction']
                else 'multiplier_rejected_explicit_candidate_search'),
            elapsed_wall=result['elapsed_wall']+diagnostic['wall_seconds'])
        return result

    def decide(self,state,forecast,previous,warm):
        self.begin(state,forecast,previous)
        seed=self.coords.encode(warm);witness=self.evaluate(seed)
        center=witness.budget_vector.copy() if self.last_budget is None else self.last_budget.copy()
        requests=[];results=[];initial_dual=self.dual.copy();radius=np.array([50.,1000.])
        request=center.copy();origin='PFO_witness' if self.last_budget is None else 'previous_selected_budget'
        for m in range(self.options.max_candidates):
            requests.append(request.copy())
            result=self.solve(request,seed,initial_dual.copy());result['request_origin']=origin;results.append(result)
            if m+1>=self.options.max_candidates:break
            hint=direction_from_recovery(result['leader_multiplier']) if result['feasible'] else np.zeros(2)
            origin='selected_action_multiplier'
            if not np.any(hint):
                hint=np.array([1.,0.]) if m==0 else np.array([0.,1.])
                origin='explicit_axis_search_no_valid_multiplier_direction'
            trials=[(center+radius*hint,origin),(center-radius*hint,origin+'_opposite'),
                    (center+radius*np.array([0.,1.]),'explicit_NUF_axis'),
                    (center-radius*np.array([0.,1.]),'explicit_NUF_axis_opposite')]
            request=None
            for trial,trial_origin in trials:
                trial[1]=np.clip(trial[1],0.,self.cfg.network.total_ramp_capacity)
                if not any(np.array_equal(trial,r) for r in requests):
                    request=trial;origin=trial_origin;break
            if request is None:break
        eligible=[r for r in results if r['feasible']]
        if not eligible:return None,results
        selected=min(eligible,key=lambda r:(r['evaluation'].total_ttt,tuple(r['budget'])))
        self.dual=selected['dual'].copy();self.last_budget=np.asarray(selected['budget']).copy()
        return selected,results
