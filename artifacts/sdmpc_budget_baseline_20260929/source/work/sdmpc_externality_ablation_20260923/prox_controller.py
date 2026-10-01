"""명세03/04/10/12/15 및 사용자 지시: 매 조정 반복의 공통 민감도와 9-player 근사 QP.

own과 외부효과 모두 같은 기준점에서 1차화하고 원 TTT는 결합 제어에서 평가한다.
전 player가 같은 기준점/승수를 사용하고 응답 결합 후 k->k+1에서 갱신한다.
전역 최적성·미분 가능성·비선형 수렴을 가정하지 않는다.
"""
from dataclasses import asdict, replace
import copy
import time
import numpy as np
from band_math import band_violation,update_duals,signed_price,budget_direction,resource_step
from fixed_policy import constraint_values,impossible_command,VARIANT
from src.controllers.player_sensitivity_dmpc import PlayerSensitivityDMPC
from local_qp import response
from externality_policy import ENABLED, local_externality
from src.simulation.player_cost_accounting import PlayerCostOwnership
from src.models.state import ControlAction,segment_vsl


def ownership(cfg):
    own=PlayerCostOwnership.from_config(cfg,freeway_segments=True)
    if cfg.network.freeway_segments_per_link!=8:
        raise ValueError('Explicit four-segment groups require eight segments per direction')
    # 사용자 지시: 방향별 전반0..3/후반4..7을 각각 하나의 player로 구성한다.
    def grouped(owner):
        if owner.startswith('FW:') and ':seg' in owner:
            link,index=owner.rsplit(':seg',1)
            return link+':group'+str(int(index)//4)
        return owner
    for mapping in (own.segment_owner,own.origin_owner,own.ramp_owner,own.offramp_storage_owner):
        for key in mapping:mapping[key]=grouped(mapping[key])
    own.active_player_ids=tuple(dict.fromkeys(grouped(p) for p in own.active_player_ids))
    # off-ramp storage는 도시 수신 교차로에 귀속. 물리 F/U 회계는 별도 유지한다.
    for ramp,storage in cfg.network.off_ramp_storage_link.items():
        nodes={str(cfg.network.urban_movements[m].get('intersection',''))
               for m in cfg.network.off_ramp_to_movement[ramp]}
        if len(nodes)!=1:
            raise ValueError(f'Ambiguous urban off-ramp owner: {ramp}, {storage}, {nodes}')
        node=next(iter(nodes))
        own.offramp_storage_owner[storage]='URBAN:'+node if node in cfg.network.signals else 'PASSIVE:'+node
    return own


class Coordinates:
    """4 group VSL,4 metering,5 green,5 offset. 시간축은 기존 H3 상수 move block."""
    def __init__(self,cfg,opts,previous):
        self.cfg,self.options,self.previous=cfg,opts,previous.copy()
        n=cfg.network;self.axes=[]
        for link in n.freeway_links:
            for i in range(2):
                allowed=self.allowed_vsl(link,i)
                if not allowed:raise ValueError('No common VSL under all four previous-segment step limits')
                self.axes.append(('vsl_group',f'{link}__group{i}',100.,min(allowed),max(allowed),.1,f'FW:{link}:group{i}'))
        for ramp in n.ramps:
            link=n.ramp_to_freeway[ramp];i=n.ramp_merge_segment_index[ramp]
            cap=n.ramp_capacity_veh_h[ramp]
            self.axes.append(('meter',ramp,cap,0.,cap,opts.fd_metering_veh_h,f'FW:{link}:group{i//4}'))
        for signal in n.signals:
            lo=max(n.green_min,n.effective_green_total-n.green_max)
            hi=min(n.green_max,n.effective_green_total-n.green_min)
            self.axes.append(('green',signal,n.cycle_length,lo,hi,opts.fd_green_sec,'URBAN:'+signal))
            step=cfg.urban_follower.max_offset_step
            self.axes.append(('offset',signal,n.cycle_length,-step,step,opts.fd_offset_sec,'URBAN:'+signal))
        self.scales=np.array([a[2] for a in self.axes])
        self.lower=np.array([a[3]/a[2] for a in self.axes]);self.upper=np.array([a[4]/a[2] for a in self.axes])
        self.fd=np.array([a[5]/a[2] for a in self.axes]);self.owners=tuple(a[6] for a in self.axes)

    def allowed_vsl(self,link,index):
        return [float(v) for v in self.cfg.freeway_follower.vsl_set
                if all(abs(v-segment_vsl(self.previous,link,i,self.cfg))<=self.cfg.freeway_follower.max_vsl_step+1e-9
                       for i in range(index*4,(index+1)*4))]

    def encode(self,c):
        values=[];cycle=self.cfg.network.cycle_length
        for kind,key,scale,*_ in self.axes:
            if kind=='vsl_group':
                link,index=key.split('__group');v=np.mean([segment_vsl(c,link,i,self.cfg) for i in range(4*int(index),4*(int(index)+1))])
            elif kind=='meter':v=c.ramp_metering.get(key,scale)
            elif kind=='green':v=c.green_times[key+'_p1']
            else:v=(c.offsets.get(key,0.)-self.previous.offsets.get(key,0.)+cycle/2)%cycle-cycle/2
            values.append(v/scale)
        return np.clip(values,self.lower,self.upper)

    def decode(self,y):
        c=ControlAction.uncontrolled(self.cfg);n=self.cfg.network
        c.inflow_outflow_allocation={}
        for value,(kind,key,scale,*_) in zip(y,self.axes):
            v=float(value*scale)
            if kind=='vsl_group':
                link,index=key.split('__group')
                for i in range(4*int(index),4*(int(index)+1)):c.vsl[f'{link}__seg{i}']=v
            elif kind=='meter':c.ramp_metering[key]=v
            elif kind=='green':
                c.green_times[key+'_p1']=v;c.green_times[key+'_p2']=n.effective_green_total-v
            else:
                offset=(self.previous.offsets.get(key,0.)+v)%n.cycle_length
                c.offsets[key]=0. if offset>=n.cycle_length else offset
        for link in n.freeway_links:c.vsl[link]=min(c.vsl[f'{link}__seg{i}'] for i in range(n.freeway_segments_per_link))
        return c

    def quantize(self,y):
        z=np.array(y,copy=True)
        for j,(kind,key,scale,*_) in enumerate(self.axes):
            if kind=='vsl_group':
                link,index=key.split('__group');allowed=self.allowed_vsl(link,int(index))
                z[j]=min(allowed,key=lambda v:(abs(v-y[j]*scale),-v))/scale
        return z

    def validate(self,c,discrete=False):
        n=self.cfg.network;errors=[]
        if c.inflow_outflow_allocation:errors.append('allocation_authority')
        if set(c.ramp_metering)!=set(n.ramps):errors.append('ramp_keys')
        for kind,key,scale,lo,hi,fd,owner in self.axes:
            if kind=='vsl_group':
                link,index=key.split('__group');values=[c.vsl.get(f'{link}__seg{i}',float('nan')) for i in range(4*int(index),4*(int(index)+1))]
                v=values[0]
                if any(value!=v for value in values):errors.append('group_vsl_equality:'+key)
            elif kind=='meter':v=c.ramp_metering.get(key,float('nan'))
            elif kind=='green':v=c.green_times.get(key+'_p1',float('nan'))
            else:
                offset=c.offsets.get(key,float('nan'))
                if not 0<=offset<n.cycle_length:errors.append('offset_range:'+key)
                v=(offset-self.previous.offsets.get(key,0.)+n.cycle_length/2)%n.cycle_length-n.cycle_length/2
            if not np.isfinite(v) or v<lo-1e-9 or v>hi+1e-9:errors.append('bounds:'+key)
            if kind=='green' and abs(v+c.green_times.get(key+'_p2',float('inf'))-n.effective_green_total)>1e-8:errors.append('green_sum:'+key)
            if kind=='vsl_group' and discrete and v not in self.cfg.freeway_follower.vsl_set:errors.append('vsl_discrete:'+key)
        for link in n.freeway_links:
            if c.vsl.get(link)!=min(c.vsl[f'{link}__seg{i}'] for i in range(n.freeway_segments_per_link)):errors.append('vsl_summary:'+link)
        return dict(valid=not errors,violations=errors,domain='discrete_execution' if discrete else 'continuous_relaxation')


class GroupProxSDMPC:
    def __init__(self,cfg,options):
        from group_block_engine import SparseRolloutEngine
        self.cfg,self.options=cfg,options
        self.evaluator=PlayerSensitivityDMPC(cfg,options)
        self.ownership=ownership(cfg)
        self.evaluator.ownership=self.ownership;self.evaluator.player_ids=self.ownership.active_player_ids
        self.engine=SparseRolloutEngine(cfg,options.horizon_steps);self.engine.ownership=self.ownership
        self.player_ids=self.ownership.active_player_ids
        self.scales=np.array([options.budget_scale_np_veh,options.budget_scale_nuf_veh_h])
        self.width=np.array([options.tolerance_np_veh,options.tolerance_nuf_veh_h])
        self.dual=np.zeros((2,2));self.last_budget=None

    def begin(self,state,forecast,previous):
        self.state,self.forecast,self.previous=state.copy(),copy.deepcopy(forecast),previous.copy()
        self.coords=Coordinates(self.cfg,self.options,previous)
        self.cache={};self.dcache={};self.derivative_rows=[]
        self.counts=dict(scalar_rollouts=0,tangent_rollouts=0,fd_columns=0,local_nlp_calls=0,
                         local_qp_calls=0,local_trial_rollouts=0,
                         cache_hits=0,scalar_wall=0.,tangent_wall=0.,local_wall=0.,dual_updates=0)

    def evaluate(self,y):
        key=np.asarray(y,dtype=float).tobytes()
        if key in self.cache:
            self.counts['cache_hits']+=1;return self.cache[key]
        started=time.perf_counter();c=self.coords.decode(y)
        ev=self.evaluator.evaluate_control(self.state,self.forecast,c,self.previous)
        check=self.coords.validate(c)
        # 확대된 제어 권한 검사는 새 좌표 계약에서 수행하고 원 물리 검사는 유지한다.
        ev=replace(ev,control_valid=check['valid'],diagnostics={**ev.diagnostics,'slide_control_validation':check})
        if abs(sum(ev.player_costs.values())-ev.total_ttt)>1e-8:raise RuntimeError('Player accounting mismatch')
        self.counts['scalar_rollouts']+=1;self.counts['scalar_wall']+=time.perf_counter()-started
        self.cache[key]=ev;return ev

    def derivatives(self,y):
        key=np.asarray(y,dtype=float).tobytes()
        if key in self.dcache:return self.dcache[key]
        ev=self.evaluate(y);coords=self.coords;n=len(y);used=set();unsafe=set();error=None
        g=np.zeros((len(self.player_ids),n));a=np.zeros((2,n));start=time.perf_counter()
        try:
            tangent=self.engine.evaluate(self.state,self.forecast,coords.decode(y),coords)
            self.counts['tangent_rollouts']+=1
            if max(np.max(abs(tangent.cost_vector-ev.cost_vector)),np.max(abs(tangent.budget_vector-ev.budget_vector)))>1e-8:
                raise ValueError('Tangent primal mismatch')
            trace=tangent.diagnostics['trace']
            unsafe=set(trace['output_connected_exact_tie_axes'])|set(trace['output_connected_stencil_crossing_axes'])
            for j in range(n):
                if j not in unsafe and np.all(np.isfinite(tangent.gradients[:,j])) and np.all(np.isfinite(tangent.budget_jacobian[:,j])):
                    g[:,j]=tangent.gradients[:,j];a[:,j]=tangent.budget_jacobian[:,j];used.add(j)
        except Exception as exc:error=repr(exc)
        self.counts['tangent_wall']+=time.perf_counter()-start
        for j in set(range(n))-used:
            high,low=y.copy(),y.copy();high[j]=min(coords.upper[j],y[j]+coords.fd[j]);low[j]=max(coords.lower[j],y[j]-coords.fd[j])
            if high[j]>low[j]:
                h,l=self.evaluate(high),self.evaluate(low)
                g[:,j]=(h.cost_vector-l.cost_vector)/(high[j]-low[j]);a[:,j]=(h.budget_vector-l.budget_vector)/(high[j]-low[j])
        a[1]=[axis[2] if axis[0]=='meter' else 0. for axis in coords.axes]
        self.counts['fd_columns']+=n-len(used)
        self.derivative_rows.append(dict(anchor=y.tolist(),ad_columns=sorted(used),unsafe_columns=sorted(unsafe),
            fd_columns=sorted(set(range(n))-used),fallback_error=error,derivatives_certified=False,
            coupled_total_derivatives=True,total_gradient=g.sum(axis=0).tolist()))
        self.dcache[key]=(g,a/self.scales[:,None]);return self.dcache[key]

    def feasible(self,y,budget,discrete=False):
        ev=self.evaluate(y)
        # band 자체가 허용 범위다. 별도 feasibility 허용치를 추가하지 않는다.
        return bool(ev.physical_valid and ev.control_valid and
                    np.all(abs(ev.budget_vector-budget)<=self.width) and
                    self.coords.validate(self.coords.decode(y),discrete)['valid'])

    def solve(self,budget,seed,initial_dual):
        start=time.perf_counter();o=self.options;coords=self.coords
        budget=np.asarray(budget,dtype=float).copy();y=np.asarray(seed).copy();dual=np.array(initial_dual,copy=True)
        eps=self.width/self.scales;rows=[];local_rows=[];trust=o.trust_radius
        best=y.copy() if self.feasible(y,budget) else None
        impossible=impossible_command(budget,self.cfg.network.total_ramp_capacity)
        reason='proven_command_capacity_infeasible' if impossible else 'iteration_limit'
        stationarity=float('inf');local_ok=False;stationarity_point=None

        def residual(z):return (self.evaluate(z).budget_vector-budget)/self.scales

        def restore(z,A,freeze_vsl=False):
            z=z.copy();A=A.copy()
            for _ in range(o.restoration_iterations):
                if self.feasible(z,budget,freeze_vsl):break
                r=residual(z);lo=np.maximum(coords.lower-z,-trust);hi=np.minimum(coords.upper-z,trust)
                if freeze_vsl:
                    for j,ax in enumerate(coords.axes):
                        if ax[0]=='vsl_group':lo[j]=hi[j]=0.
                d,qp=resource_step(np.zeros(len(z)),A,r,eps,lo,hi)
                found=False
                for alpha in (1.,.5,.25):
                    trial=np.clip(z+alpha*d,coords.lower,coords.upper);ev=self.evaluate(trial)
                    if ev.physical_valid and ev.control_valid and band_violation(residual(trial),eps)<band_violation(r,eps)-1e-10:
                        delta=trial-z
                        if delta@delta>1e-16:A+=np.outer(residual(trial)-r-A@delta,delta)/(delta@delta)
                        z=trial;found=True;break
                if not found:break
            return z

        for k in range(0 if impossible else o.max_iterations):
            anchor=y.copy();ev=self.evaluate(y);g,A=self.derivatives(y);r=residual(y)
            own=np.array([g[self.player_ids.index(pid),j] for j,pid in enumerate(coords.owners)])
            # 전체 연결 미분은 유지하고 지역 목적에 투입하는 pi_i만 전환한다.
            external_computed=g.sum(axis=0)-own
            external=local_externality(external_computed);price=A.T@signed_price(dual)
            proposal=np.zeros(len(y));local_ok=True;used=dual.copy()
            for pid in self.player_ids:
                ix=np.array([j for j,p in enumerate(coords.owners) if p==pid],dtype=int)
                local_start=time.perf_counter()
                before=(self.counts['scalar_rollouts'],self.counts['tangent_rollouts'])
                lo=np.maximum(coords.lower[ix]-y[ix],-trust);hi=np.minimum(coords.upper[ix]-y[ix],trust)
                step,local_stationarity=response(own[ix],external[ix],price[ix],lo,hi,o.proximal)
                valid=bool(np.all(np.isfinite(step)) and local_stationarity<=o.local_tolerance)
                # 지역 내부의 물리 rollout·재미분 호출이 0임을 실제 카운터로 검사한다.
                if before!=(self.counts['scalar_rollouts'],self.counts['tangent_rollouts']):
                    raise RuntimeError('Local surrogate unexpectedly performed traffic prediction')
                self.counts['local_qp_calls']+=1;self.counts['local_wall']+=time.perf_counter()-local_start
                proposal[ix]=step;local_ok &= valid
                local_rows.append(dict(k=k,player=pid,indices=ix.tolist(),step=step.tolist(),
                    success=valid,local_qp_stationarity=local_stationarity,
                    true_own_at_anchor=ev.player_costs[pid],
                    approximate_own_at_proposal=ev.player_costs[pid]+float(own[ix]@step),
                    own_gradient=own[ix].tolist(),externality=external[ix].tolist(),budget_gradient=price[ix].tolist(),
                    externality_computed=external_computed[ix].tolist(),externality_enabled=ENABLED,
                    model_reference=anchor.tolist(),dual_used=used.tolist(),
                    own_model='first_order_own_with_proximal',local_trial_rollouts=0))
            # Jacobi 전 player가 완료된 뒤에만 승수 갱신. 후보별 상태는 독립 복사다.
            raw=r+A@proposal;dual=update_duals(used,raw,eps,o.dual_step);self.counts['dual_updates']+=int(VARIANT!='none')
            d,qp=resource_step(proposal,A,r,eps,np.maximum(coords.lower-y,-trust),np.minimum(coords.upper-y,trust))
            pg,pq=resource_step(-g.sum(axis=0)/o.proximal,A,r,eps,coords.lower-y,coords.upper-y,o.proximal)
            stationarity=o.proximal*float(np.max(abs(pg))) if pq['success'] else float('inf')
            stationarity_point=y.copy();accepted=False
            for ls in range(o.line_search_steps):
                trial=np.clip(y+(.5**ls)*d,coords.lower,coords.upper)
                if not self.feasible(trial,budget):trial=restore(trial,A)
                if self.feasible(trial,budget) and (best is None or self.evaluate(trial).total_ttt<self.evaluate(best).total_ttt-o.objective_improvement_tolerance):
                    y=trial;best=trial.copy();accepted=True;break
                if best is None and self.evaluate(trial).physical_valid and self.evaluate(trial).control_valid and band_violation(residual(trial),eps)<band_violation(r,eps)-1e-10:
                    y=trial;accepted=True;break
            comp=float(np.max(abs(dual*constraint_values(r,eps))))
            rows.append(dict(k=k,anchor=anchor.tolist(),objective=self.evaluate(anchor).total_ttt,
                residual_physical=(r*self.scales).tolist(),raw_proposal_residual_scaled=raw.tolist(),
                dual_before=used.tolist(),dual_after=dual.tolist(),dual_updates=int(VARIANT!='none'),
                primal_stationarity=stationarity,stationarity_anchor=anchor.tolist(),
                complementarity=comp,local_surrogate_solvers_converged=bool(local_ok),
                local_nonlinear_solvers_converged=False,resource=qp,
                accepted=accepted,next_objective=self.evaluate(y).total_ttt,next_residual_physical=(residual(y)*self.scales).tolist(),
                own_gradient=own.tolist(),externality=external.tolist(),budget_gradient=price.tolist(),
                externality_computed=external_computed.tolist(),externality_enabled=ENABLED))
            if not accepted:trust=max(o.min_trust_radius,trust*.5)
            # 미분/활성 물리 제약 인증이 없으므로 작은 이동만으로 수렴 선언하지 않는다.
            if local_ok and self.feasible(anchor,budget) and stationarity<=o.stationarity_tolerance and comp<=o.stationarity_tolerance and np.max(abs(dual-used))<=o.stationarity_tolerance:
                reason='surrogate_first_order_diagnostics_pass_uncertified';break
        z=best.copy() if best is not None else y.copy()
        _,A=self.derivatives(z) if not impossible else (None,np.zeros((2,len(z))))
        trials=[]
        quantized=coords.quantize(z)
        q=restore(quantized,A,True) if not self.feasible(quantized,budget,True) else quantized
        trials.append(q)
        if not self.feasible(q,budget,True) and not impossible:
            # 최근접 격자가 실패한 경우에만 한 축의 위·아래 값으로 원 모델 재검사.
            for j,(kind,key,scale,*_) in enumerate(coords.axes):
                if kind!='vsl_group':continue
                link,index=key.split('__group');allowed=coords.allowed_vsl(link,int(index));v=z[j]*scale
                neighbors=([max(x for x in allowed if x<v)] if any(x<v for x in allowed) else [])+([min(x for x in allowed if x>v)] if any(x>v for x in allowed) else [])
                for value in neighbors:
                    test=quantized.copy();test[j]=value/scale
                    trials.append(restore(test,A,True))
        eligible=[p for p in trials if not impossible and self.feasible(p,budget,True)]
        chosen=min(eligible,key=lambda p:self.evaluate(p).total_ttt) if eligible else q
        ev=self.evaluate(chosen);control=coords.decode(chosen);control.N_P_star,control.N_UF_star=budget.tolist()
        # 未수렴 반복 승수는 상위 방향 힌트일 뿐 value-function gradient 인증이 아니다.
        gb,direction=budget_direction(dual,self.scales)
        return dict(budget=budget.tolist(),control=control,point=chosen,evaluation=ev,
            feasible=bool(eligible),converged=False,status='feasible_uncertified' if eligible else reason if impossible else 'algorithm_failed_no_feasible',
            reason=reason,rows=rows,local_rows=local_rows,dual=dual,budget_gradient_hint=gb.tolist(),direction=direction.tolist(),
            dual_source='last_inner_iterate_uncertified',dual_reference=rows[-1]['anchor'] if rows else None,
            selected_differs_from_dual_reference=not rows or not np.array_equal(chosen,np.asarray(rows[-1]['anchor'])),
            original_residual=(ev.budget_vector-budget).tolist(),quantization_trials=len(trials),
            stationarity=stationarity,stationarity_point=stationarity_point,local_solvers_converged=False,
            local_surrogate_solvers_converged=bool(local_ok),original_nonlinear_own_preserved=False,
            local_trial_rollouts=0,algorithm='group_player_proxlinear_refresh_each_anchor',
            elapsed_wall=time.perf_counter()-start)

    def decide(self,state,forecast,previous,warm):
        self.begin(state,forecast,previous)
        seed=self.coords.encode(warm);witness=self.evaluate(seed)
        center=witness.budget_vector.copy() if self.last_budget is None else self.last_budget.copy()
        requests=[];results=[];initial_dual=self.dual.copy();radius=np.array([50.,1000.])
        request=center.copy()
        for m in range(self.options.max_candidates):
            requests.append(request.copy());result=self.solve(request,seed,initial_dual);results.append(result)
            if m+1>=self.options.max_candidates:break
            hint=np.asarray(result['direction']) if result['feasible'] else np.zeros(2)
            # 0승수/실패 때 감소를 강제하지 않고 명시한 탐색 후보를 시험한다.
            if not np.any(hint):hint=np.array([1.,0.]) if m==0 else np.array([0.,1.])
            trials=[center+radius*hint,center-radius*hint,center+radius*np.array([0.,1.]),center-radius*np.array([0.,1.])]
            request=None
            for trial in trials:
                trial[1]=np.clip(trial[1],0.,self.cfg.network.total_ramp_capacity)
                if not any(np.array_equal(trial,r) for r in requests):request=trial;break
            if request is None:break
        eligible=[r for r in results if r['feasible']]
        if not eligible:return None,results
        selected=min(eligible,key=lambda r:(r['evaluation'].total_ttt,tuple(r['budget'])))
        self.dual=selected['dual'].copy();self.last_budget=np.asarray(selected['budget']).copy()
        return selected,results
