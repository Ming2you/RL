"""최종 제어를 고정한 condensed 중앙 문제의 제약 목록과 승수 회수."""
import numpy as np
from scipy.optimize import nnls
from fixed_policy import constraint_values,mask,budget_gradient


def physical_rows(states,cfg):
    """원 evaluator가 각 horizon 끝에서 검사하는 수학적 상태 부등식 h<=0.

    실제 실행 gate의 1e-7 수치 여유는 변경하지 않는다. 수학적 원 경계는 0이며
    중복 queue view나 원 gate에 없는 jam-density/속도 상한을 추가하지 않는다.
    반환 값은 Dual도 허용하므로 동일 결합 동역학의 상태 총미분을 보존한다.
    """
    names=[];values=[]
    for k,state in enumerate(states):
        for attr in ('ramp_queue','urban_movement_queue','mainline_origin_queue',
                     'freeway_density','freeway_buffer_up_density','freeway_buffer_down_density'):
            for key,value in sorted(getattr(state,attr).items()):
                seq=value if isinstance(value,(list,tuple)) else [value]
                for j,x in enumerate(seq):
                    names.append(f'physical/{k}/{attr}/{key}/{j}/lower');values.append(-x)
        for key,cap in sorted(cfg.network.urban_link_storage_veh.items()):
            x=state.urban_link_storage.get(key,cap)
            names.extend([f'physical/{k}/storage/{key}/lower',f'physical/{k}/storage/{key}/upper'])
            values.extend([-x,x-cap])
    return names,values


def assemble(y,budget,ev,G,A,physical_values,physical_jacobian,physical_names,coords,scales,width):
    """모든 player TTT와 모든 제어축을 결합한다. G/A는 같은 선택점의 총미분이다."""
    y,budget=np.asarray(y),np.asarray(budget)
    n=len(y);eye=np.eye(n)
    h=np.concatenate((np.where(mask(),constraint_values(ev.budget_vector-budget,width)/scales,-1.).ravel(),
                      coords.lower-y,y-coords.upper,np.asarray(physical_values)))
    budget_rows=np.where(mask().ravel()[:,None],np.vstack((A,-A)),0.)
    C=np.vstack((budget_rows,-eye,eye,physical_jacobian))
    names=['budget/NP/upper','budget/NUF/upper','budget/NP/lower','budget/NUF/lower']
    names=[name if active else 'inactive_'+name for name,active in zip(names,mask().ravel())]
    names += [f'control/{j}/{axis[0]}/{axis[1]}/lower' for j,axis in enumerate(coords.axes)]
    names += [f'control/{j}/{axis[0]}/{axis[1]}/upper' for j,axis in enumerate(coords.axes)]
    return G.sum(axis=0),h,C,names+physical_names


def recover(g,h,C,names,budget,scales,*,tolerance=1e-3,active_tolerance=1e-8):
    """제어를 변경하지 않고 전체 중앙 stationarity의 비음수 승수만 회수한다."""
    g,h,C,budget,scales=map(lambda v:np.asarray(v,dtype=float),(g,h,C,budget,scales))
    if not all(np.all(np.isfinite(v)) for v in (g,h,C,budget,scales)):
        return dict(fit_success=False,error='nonfinite_central_system',stationarity_inf=None)
    active=np.flatnonzero(abs(h)<=active_tolerance)
    norms=np.linalg.norm(C,axis=1)
    # 구조적으로 영인 행은 stationarity에 기여하지 않으므로 승수를 0으로 둔다.
    usable=[int(i) for i in active if norms[i]>1e-12]
    # 같은 방향의 중복 행은 하나로 묶고 원래 행과의 대응을 로그에 남긴다.
    unique=[];groups=[];lookup={}
    for i in usable:
        key=tuple(np.round(C[i]/norms[i],12))
        if key not in lookup:
            lookup[key]=len(unique);unique.append(i);groups.append([])
        groups[lookup[key]].append(i)
    M=(C[unique]/norms[unique,None]).T if unique else np.zeros((len(g),0))
    try:
        weights=nnls(M,-g,maxiter=max(1000,10*len(unique)))[0] if unique else np.zeros(0)
    except (RuntimeError,ValueError) as exc:
        return dict(fit_success=False,error=repr(exc),stationarity_inf=None)
    multipliers=np.zeros(len(h))
    if unique:multipliers[unique]=weights/norms[unique]
    residual=g+C.T@multipliers
    dual=multipliers[:4].reshape(2,2)
    gradient=budget_gradient(dual,budget,scales)
    stat=float(np.max(abs(residual)));comp=float(np.max(abs(h*multipliers)))
    # 중복에 budget 행이 포함되면 해당 budget 가격의 유일성을 주장하지 않는다.
    rank=int(np.linalg.matrix_rank(M)) if unique else 0
    identifiable=(rank==len(unique) and not any(len(group)>1 and min(group)<4 for group in groups))
    by_family={}
    for family in ('budget','control','physical'):
        ids=[i for i,name in enumerate(names) if name.startswith(family+'/')]
        by_family[family]=dict(total=len(ids),active=sum(i in active for i in ids),
            nonzero_multipliers=sum(multipliers[i]>1e-12 for i in ids),
            stationarity_contribution=(C[ids].T@multipliers[ids]).tolist())
    return dict(fit_success=True,budget_dual=dual.tolist(),gradient_estimate=gradient.tolist(),
        stationarity_inf=stat,stationarity_vector=residual.tolist(),complementarity=comp,
        stationarity_pass=stat<=tolerance,complementarity_pass=comp<=tolerance,
        multiplier_identifiability_sufficient_check=identifiable,active_generator_rank=rank,
        active_nonzero_generator_count=len(unique),active_total=len(active),
        active_zero_jacobian_count=sum(norms[i]<=1e-12 for i in active),
        active_rows=[dict(index=int(i),name=names[i],value=float(h[i]),
                          jacobian_norm=float(norms[i]),multiplier=float(multipliers[i])) for i in active],
        duplicate_groups=[[names[i] for i in group] for group in groups if len(group)>1],
        family_summary=by_family,stationarity_tolerance=tolerance,
        active_detection_tolerance=active_tolerance,control_optimized=False,
        dual_units='budget: veh*h (normalized constraints); physical: TTT/state unit',
        budget_gradient_units=['h','h^2'])
