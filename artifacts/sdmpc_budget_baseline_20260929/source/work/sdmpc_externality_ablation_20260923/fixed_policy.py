"""프로세스당 고정된 budget 실험 정책. 원 물리/제어 gate는 변경하지 않는다."""
import os
import numpy as np
from policy import decode_grid_value

VARIANTS=('band1','band5','band10','none','upper')
VARIANT=os.environ['SDMPC_BUDGET_VARIANT']
if VARIANT not in VARIANTS:raise ValueError(VARIANT)
FACTOR={'band1':1.,'band5':5.,'band10':10.}.get(VARIANT,0.)

def halfwidth(budget):
    return FACTOR*np.array([10.,.05*abs(float(budget[1]))])

def mask():
    return np.array([[VARIANT!='none']*2,[FACTOR>0]*2],dtype=bool)

def constraint_values(residual,width):
    # 비활성 행은 고정 -1 및 영 Jacobian을 갖는 로그용 placeholder다.
    r=np.asarray(residual)
    return np.where(mask(),np.stack((r-width,-r-width)),-1.)

def excess(achieved,budget):
    h=constraint_values(np.asarray(achieved)-np.asarray(budget),halfwidth(budget))
    return np.maximum(h,0.).max(axis=0)

def budget_gradient(dual,budget,scales):
    dual=np.where(mask(),np.asarray(dual),0.)
    return (dual[1]-dual[0]-np.array([0.,.05*FACTOR])*np.sign(budget)*(dual[0]+dual[1]))/np.asarray(scales)

def impossible_command(budget,capacity):
    if VARIANT=='none':return False
    width=halfwidth(budget)
    return bool(budget[1]+width[1]<0 or (FACTOR>0 and budget[1]-width[1]>capacity))

def contract():
    return dict(variant=VARIANT,band_factor=FACTOR,budget_active_mask=mask().tolist(),
        NP_halfwidth_veh=10*FACTOR,NUF_halfwidth_fraction=.05*FACTOR,
        definition='G <= B, no extra margin' if VARIANT=='upper' else
                   'no budget constraints/prices/resource projection; B recorded as diagnostic reference only' if VARIANT=='none' else 'abs(G-B) <= width(B)',
        budget_candidate_solves=1 if VARIANT=='none' else 3)

def bounds(budget):
    b=np.asarray(budget);w=halfwidth(budget)
    return (b-w).tolist() if FACTOR>0 else [None,None], (b+w).tolist() if VARIANT!='none' else [None,None]
