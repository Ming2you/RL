"""공통 기준점의 TTT 총미분으로 만드는 지역 box QP. 교통 예측 의존성 없음."""
import numpy as np


def response(own,external,budget,lower,upper,proximal):
    if proximal<=0:raise ValueError('Positive proximal required')
    gradient=np.asarray(own)+np.asarray(external)+np.asarray(budget)
    lo,hi=np.asarray(lower),np.asarray(upper)
    if np.any(lo>hi):raise ValueError('Inconsistent box')
    d=np.clip(-gradient/proximal,lo,hi)
    mapping=d-np.clip(d-(gradient+proximal*d)/proximal,lo,hi)
    stationarity=proximal*float(np.max(abs(mapping),initial=0.))
    return d,stationarity
