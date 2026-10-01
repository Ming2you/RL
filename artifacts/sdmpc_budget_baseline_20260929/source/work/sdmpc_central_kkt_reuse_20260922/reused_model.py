"""하위에서 이미 계산한 1차식을 중앙 승수 회수에 재사용한다. 추가 예측 없음."""
import copy
import numpy as np
from central import physical_rows,assemble
from fixed_policy import halfwidth
from sparse.dual import primal,derivative


def build(solver,selected,budget):
    # 동일 후보의 실제 미분 호출 중 최종 선택점과 가장 가까운 식을 사용한다.
    # 거리가 같으면 마지막 호출을 택한다. 다른 budget 후보의 모델을 가져오지 않는다.
    refs=solver.lower_model_references
    if not refs:raise RuntimeError('No existing lower model available for multiplier recovery')
    _,reference=min(enumerate(refs),key=lambda item:(np.linalg.norm(item[1]-selected),-item[0]))
    key=np.asarray(reference,dtype=float).tobytes()
    ev=solver.cache[key];G,A=solver.dcache[key];tangent=solver.tangent_cache[key]
    if tangent is None:raise RuntimeError('Original model has no reusable state tangent')
    names,values=physical_rows(ev.states,solver.cfg)
    ad_names,ad_values=physical_rows(tangent.states,solver.cfg)
    if names!=ad_names or np.max(abs(np.asarray(values)-[primal(v) for v in ad_values]))>1e-8:
        raise RuntimeError('Reusable state tangent primal mismatch')
    n=len(reference)
    J=np.array([[derivative(v).get(j,0.) for j in range(n)] for v in ad_values])
    evidence=next(row for row in reversed(solver.derivative_rows) if np.array_equal(row['anchor'],reference))
    risks=set(evidence['fd_columns'])
    for v in ad_values:
        if hasattr(v,'risks'):
            risks.update(j for j in range(n) if (v.risks[0]|v.risks[1])&(1<<j))
    reused_fd=[];unverified=[]
    # FD용 상태가 하위 cache에 있을 때만 사용한다. 없으면 AD 근사와 경고를 남긴다.
    for j in sorted(risks):
        high,low=reference.copy(),reference.copy()
        high[j]=min(solver.coords.upper[j],reference[j]+solver.coords.fd[j])
        low[j]=max(solver.coords.lower[j],reference[j]-solver.coords.fd[j])
        hk,lk=high.tobytes(),low.tobytes()
        if high[j]>low[j] and hk in solver.cache and lk in solver.cache:
            hn,hv=physical_rows(solver.cache[hk].states,solver.cfg)
            ln,lv=physical_rows(solver.cache[lk].states,solver.cfg)
            if hn!=names or ln!=names:raise RuntimeError('Cached physical rows inconsistent')
            J[:,j]=(np.asarray(hv)-np.asarray(lv))/(high[j]-low[j]);reused_fd.append(j)
        else:unverified.append(j)
    g,h0,C,labels=assemble(reference,budget,ev,G,A,values,J,names,solver.coords,solver.scales,halfwidth(budget))
    delta=selected-reference
    h=h0+C@delta
    actual=solver.cache[np.asarray(selected,dtype=float).tobytes()]
    actual_names,actual_values=physical_rows(actual.states,solver.cfg)
    if actual_names!=names:raise RuntimeError('Final physical rows inconsistent')
    _,h_actual,_,_=assemble(selected,budget,actual,G,A,actual_values,J,names,
                          solver.coords,solver.scales,halfwidth(budget))
    check=dict(model_reference=reference.tolist(),selection_point=selected.tolist(),
        model_matches_selection=bool(np.array_equal(reference,selected)),
        reference_distance_inf=float(np.max(abs(delta))),physical_primal_match=True,
        same_candidate_model=True,physical_fd_columns=reused_fd,unverified_physical_columns=unverified,
        exact_tie_axes=tangent.diagnostics['trace'].get('output_connected_exact_tie_axes',[]),
        derivative_check_pass=bool(evidence.get('derivatives_certified',False)),
        derivative_audit=[],lower_derivative_evidence=copy.deepcopy(evidence),
        actual_constraint_values=h_actual.tolist(),affine_constraint_values=h.tolist(),
        actual_minus_affine_max=float(np.max(abs(h_actual-h))),
        predicted_TTT=float(ev.total_ttt+g@delta),actual_TTT=actual.total_ttt,
        scalar_rollouts_for_recovery=0,tangent_rollouts_for_recovery=0,
        sensitivity_warning='reused lower linear model; not re-differentiated at the final selected point')
    # 활성 여부와 상보성은 이미 계산된 최종 원 제약값으로 판정한다.
    # 미분 계수만 재사용하며, 선형 예측 잔차를 실제 제약 잔차로 대체 보고하지 않는다.
    return g,h_actual,C,labels,check
