"""선택된 동일 점의 budget/box KKT 승수 추정과 leader 사용 검증."""
import numpy as np
from scipy.optimize import nnls


def recover(g, A, h, y, lower, upper, budget, scales, *, executable,
            derivatives_verified, physical_constraints_verified,
            stationarity_tolerance=1e-3, active_tolerance=1e-8):
    """모든 입력은 같은 선택점. h=[상한 2개, 하한 2개] <= 0, A는 정규화 Jacobian.

    g는 순수 TTT의 무차원 제어좌표 미분이다. proximal/반복 가격을 넣지 않는다.
    이 추정기는 budget/box 부분을 다룬다. 누락된 활성 물리 제약이 있거나
    그 부재를 검증하지 못한 호출자는 physical_constraints_verified=False여야 한다.
    """
    g,A,h,y,lower,upper,budget,scales = map(
        lambda x: np.asarray(x,dtype=float), (g,A,h,y,lower,upper,budget,scales))
    reasons=[]
    if not executable: reasons.append('selected_action_not_executable')
    if not derivatives_verified: reasons.append('selected_point_derivatives_unverified')
    if not physical_constraints_verified: reasons.append('physical_active_constraints_unverified')
    if budget[1]==0: reasons.append('NUF_band_width_nondifferentiable_at_zero')
    if not all(np.all(np.isfinite(x)) for x in (g,A,h,y,lower,upper,budget,scales)):
        return dict(accepted_for_leader_direction=False,rejection_reasons=reasons+['nonfinite_input'],
                    budget_dual=None,gradient_estimate=None,stationarity_inf=None)
    if np.any(scales<=0): raise ValueError('Budget scales must be positive')
    if np.any(h>0) or np.any(y<lower) or np.any(y>upper):
        reasons.append('original_constraints_violated')
    rows=[];labels=[];slacks=[]
    # 비활성 제약의 승수는 정확히 0. 활성 판별 오차는 실행 가능 범위를 넓히지 않는다.
    for j in range(4):
        if abs(h[j])<=active_tolerance:
            rows.append(A[j] if j<2 else -A[j-2]);labels.append(('budget',j));slacks.append(h[j])
    for j in range(len(y)):
        e=np.eye(len(y))[j]
        if abs(y[j]-lower[j])<=active_tolerance:
            rows.append(-e);labels.append(('lower',j));slacks.append(lower[j]-y[j])
        if abs(y[j]-upper[j])<=active_tolerance:
            rows.append(e);labels.append(('upper',j));slacks.append(y[j]-upper[j])
    C=np.asarray(rows).reshape(-1,len(y))
    try:
        # 열 정규화는 NNLS conditioning만 변경하며 반환 승수는 원 단위로 환산한다.
        norms=np.linalg.norm(C,axis=1) if len(C) else np.zeros(0)
        norm_scale=np.where(norms>0,norms,1.)
        lam=nnls((C/norm_scale[:,None]).T,-g)[0]/norm_scale if len(C) else np.zeros(0)
    except (RuntimeError,ValueError) as exc:
        return dict(accepted_for_leader_direction=False,rejection_reasons=reasons+['multiplier_fit_failed'],
                    error=str(exc),budget_dual=None,gradient_estimate=None,stationarity_inf=None)
    residual=g+C.T@lam
    stationarity=float(np.max(abs(residual)))
    complementarity=float(np.max(abs(lam*np.asarray(slacks)))) if len(lam) else 0.
    rank=int(np.linalg.matrix_rank(C)) if len(C) else 0
    if stationarity>stationarity_tolerance: reasons.append('stationarity_failed')
    if complementarity>stationarity_tolerance: reasons.append('complementarity_failed')
    if rank<len(C): reasons.append('active_gradient_rank_deficient')
    dual=np.zeros(4)
    for label,value in zip(labels,lam):
        if label[0]=='budget':dual[label[1]]=value
    dual=dual.reshape(2,2)
    # NP 고정 ±10veh, NUF ±0.05|B|. λ는 normalized constraint 승수 [veh*h].
    gradient=(dual[1]-dual[0]-np.array([0.,.05])*np.sign(budget)*(dual[0]+dual[1]))/scales
    return dict(accepted_for_leader_direction=not reasons,rejection_reasons=reasons,
        budget_dual=dual.tolist(),gradient_estimate=gradient.tolist(),
        stationarity_inf=stationarity,stationarity_vector=residual.tolist(),
        complementarity=complementarity,active_constraints=labels,
        active_multipliers=lam.tolist(),active_gradient_rank=rank,
        stationarity_tolerance=stationarity_tolerance,active_detection_tolerance=active_tolerance,
        executable=bool(executable),derivatives_verified=bool(derivatives_verified),
        physical_constraints_verified=bool(physical_constraints_verified),
        scope='conditional_continuous_problem_at_fixed_selected_discrete_VSL',
        optimality_certified=False,dual_units='veh*h (normalized constraint)',
        gradient_units=['h','h^2'])


def direction_from_recovery(recovery):
    """미검증 추정값을 0 gradient로 해석하지 않고 방향 사용 자체를 차단한다."""
    if not recovery['accepted_for_leader_direction']:return np.zeros(2)
    g=np.asarray(recovery['gradient_estimate'])
    return np.where(abs(g)>1e-8,-np.sign(g),0.)
