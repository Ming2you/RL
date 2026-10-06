"""Specs 03/04/12/15: fused local Jacobians; unchanged primal and nonsmooth gates.

Only three smooth METANET expressions are fused. The surrounding receiving,
signal, capacity and min/max branches still use the original active-path AD.
This is an incremental block derivative implementation, not a full adjoint.
"""
import builtins
import math
from sparse.dual import Dual, primal, derivative, extremum


def _extreme(is_max, args, kwargs):
    if len(args)!=2 or kwargs:
        return extremum(is_max,*args,**kwargs)
    a,b=args
    if not isinstance(a,Dual) and not isinstance(b,Dual):
        return builtins.max(a,b) if is_max else builtins.min(a,b)
    trace=a.trace if isinstance(a,Dual) else b.trace
    pick_b=primal(b)>primal(a) if is_max else primal(b)<primal(a)
    selected,other=(b,a) if pick_b else (a,b)
    risks=trace.branch(other,selected,output_connected=True) if other is not selected else (0,0)
    return trace.scalar(primal(selected),derivative(selected),parents=(selected,),risks=risks)


def minimum(*args,**kwargs): return _extreme(False,args,kwargs)
def maximum(*args,**kwargs): return _extreme(True,args,kwargs)


def jvp(value,inputs,partials):
    """One sparse accumulation per physical output, preserving upstream risk masks."""
    active=[(x,p) for x,p in zip(inputs,partials) if isinstance(x,Dual)]
    if not active: return float(value)
    tangent={}
    for x,p in active:
        for k,v in x.tangent.items(): tangent[k]=tangent.get(k,0.0)+p*v
    return active[0][0].trace.scalar(value,tangent,parents=inputs)


def ordered_jvp(value,inputs,expression):
    """Explicit tangent recurrence, retaining original floating-point operation order."""
    active=[x for x in inputs if isinstance(x,Dual)]
    if not active:return float(value)
    tangents=[derivative(x) for x in inputs]
    keys=set().union(*(d.keys() for d in tangents))
    result={k:expression(*(d.get(k,0.) for d in tangents)) for k in keys}
    return active[0].trace.scalar(value,result,parents=inputs)


def install_metanet_blocks(module):
    old_flow=module.segment_flow_veh_h
    old_desired=module.desired_speed_kmh
    old_speed=module.metanet_speed_update_kmh

    def flow(rho_veh_km_lane,speed_km_h,lanes):
        r,v,l=maximum(rho_veh_km_lane,0.),maximum(speed_km_h,0.),maximum(lanes,0.)
        rp,vp,lp=primal(r),primal(v),primal(l)
        return ordered_jvp(rp*vp*lp,(r,v,l),lambda dr,dv,dl:lp*(vp*dr+rp*dv)+(rp*vp)*dl)

    def desired(rho,v_free,rho_crit,a=1.867):
        if any(isinstance(v,Dual) for v in (v_free,rho_crit,a)):
            return old_desired(rho,v_free,rho_crit,a)
        r=maximum(rho,0.); denom=max(rho_crit,1e-9)
        ratio=primal(r)/denom
        value=v_free*math.exp(-(1./a)*(ratio**a))
        # At ratio=0 and a<1 the original derivative can be singular: preserve it.
        if ratio==0 and a<1: return old_desired(rho,v_free,rho_crit,a)
        exp_value=math.exp(-(1./a)*(ratio**a))
        power_partial=a*ratio**(a-1)
        return ordered_jvp(value,(r,),lambda dr:v_free*(exp_value*((-1./a)*(power_partial*((1./denom)*dr)))))

    def speed(speed,upstream_speed,rho,downstream_rho,v_eff,dt_h,length_km,tau_h,
              nu_km2_h,kappa_veh_km_lane,v_min):
        constants=(dt_h,length_km,tau_h,nu_km2_h,kappa_veh_km_lane,v_min)
        if any(isinstance(v,Dual) for v in constants):
            return old_speed(speed,upstream_speed,rho,downstream_rho,v_eff,*constants)
        # The denominator clamp is retained as an AD gate, including stencil risks.
        den=maximum(rho+kappa_veh_km_lane,1e-9)
        v,u,r,rd,ve,d=map(primal,(speed,upstream_speed,rho,downstream_rho,v_eff,den))
        a=dt_h/max(tau_h,1e-9)
        b=dt_h/max(length_km,1e-9)
        c=-nu_km2_h*dt_h/(max(tau_h,1e-9)*max(length_km,1e-9))
        relaxation=a*(ve-v)
        convection=b*v*(u-v)
        anticipation=c*(rd-r)/d
        def tangent(dv,du,dr,drd,dve,dd):
            d_relax=a*(dve-dv)
            d_conv=(u-v)*(b*dv)+(b*v)*(du-dv)
            d_anti=(1./d)*(c*(drd-dr))+(-c*(rd-r)/d**2)*dd
            return ((dv+d_relax)+d_conv)+d_anti
        raw=ordered_jvp(v+relaxation+convection+anticipation,
            (speed,upstream_speed,rho,downstream_rho,v_eff,den),tangent)
        return maximum(v_min,raw)

    module.segment_flow_veh_h=flow
    module.desired_speed_kmh=desired
    module.metanet_speed_update_kmh=speed
