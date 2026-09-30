"""band/상한/제거 정책을 지역가격·자원QP·원 잔차에 일관되게 적용한다."""
import numpy as np
from scipy.optimize import minimize
from fixed_policy import constraint_values,mask,VARIANT

def band_violation(residual,halfwidth):
    return float(np.max(np.maximum(constraint_values(residual,halfwidth),0.),initial=0.))

def update_duals(dual,raw_residual,halfwidth,eta):
    return np.where(mask(),np.maximum(0.,np.asarray(dual)+eta*constraint_values(raw_residual,halfwidth)),0.)

def signed_price(dual):
    d=np.where(mask(),np.asarray(dual),0.)
    return d[0]-d[1]

def budget_direction(dual,scales,threshold=1e-8):
    gradient=-signed_price(dual)/np.asarray(scales)
    return gradient,np.where(abs(gradient)>threshold,-np.sign(gradient),0.)

def resource_step(proposal,A,residual,halfwidth,lower,upper,metric=1.):
    p,A,r,eps,lo,hi=map(np.asarray,(proposal,A,residual,halfwidth,lower,upper))
    if VARIANT=='none':
        # budget 제거 시 원 box만 남으므로 유클리드 사영은 clip으로 정확히 계산한다.
        d=np.clip(p,lo,hi)
        return d,dict(success=True,message='box_projection_only_no_budget',budget_projection_skipped=True,
            linear_residual=(r+A@d).tolist(),projection_is_not_original_feasibility=True)
    active=mask().ravel()
    def inequalities(d):return -constraint_values(r+A@d,eps).ravel()[active]
    out=minimize(lambda d:.5*metric*float((d-p)@(d-p)),np.clip(p,lo,hi),
        jac=lambda d:metric*(d-p),bounds=list(zip(lo,hi)),
        constraints=[dict(type='ineq',fun=inequalities,jac=lambda d:np.vstack((-A,A))[active])],
        method='SLSQP',options=dict(maxiter=60,ftol=1e-11))
    d=np.clip(out.x,lo,hi)
    return d,dict(success=bool(out.success and band_violation(r+A@d,eps)<=1e-6),
        message=str(out.message),linear_residual=(r+A@d).tolist(),
        projection_is_not_original_feasibility=True)
