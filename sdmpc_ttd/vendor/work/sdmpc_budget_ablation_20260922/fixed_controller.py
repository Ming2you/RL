"""N_P고정±10대,N_UF±5%별도실험.물리/요청/6회상한유지."""
import numpy as np
from prox_controller import GroupProxSDMPC,Coordinates
from fixed_policy import halfwidth,budget_gradient,decode_grid_value,excess,contract,bounds


class GridCoordinates(Coordinates):
    def decode(self,y):
        c=super().decode(y)
        for value,(kind,key,scale,*_) in zip(y,self.axes):
            if kind!='vsl_group':continue
            link,index=key.split('__group');index=int(index)
            speed=decode_grid_value(value,scale,self.allowed_vsl(link,index))
            for i in range(4*index,4*(index+1)):c.vsl[f'{link}__seg{i}']=speed
        for link in self.cfg.network.freeway_links:
            c.vsl[link]=min(c.vsl[f'{link}__seg{i}'] for i in range(self.cfg.network.freeway_segments_per_link))
        return c


class FixedNPBandSDMPC(GroupProxSDMPC):
    def begin(self,state,forecast,previous):
        super().begin(state,forecast,previous)
        self.coords=GridCoordinates(self.cfg,self.options,previous)

    def feasible(self,y,budget,discrete=False):
        # 評가순서와현재후보에독립적으로해당요청의band를계산한다.
        ev=self.evaluate(y)
        return bool(ev.physical_valid and ev.control_valid and
            np.all(excess(ev.budget_vector,budget)==0.) and
            self.coords.validate(self.coords.decode(y),discrete)['valid'])

    def solve(self,budget,seed,initial_dual):
        # 하위문제에서는요청B가고정이므로폭도상수.지역QP/복원/dual모두같은폭을사용한다.
        self.width=halfwidth(budget)
        result=super().solve(budget,seed,initial_dual)
        g=budget_gradient(result['dual'],budget,self.scales)
        result.update(budget_gradient_hint=g.tolist(),direction=np.where(abs(g)>1e-8,-np.sign(g),0.).tolist(),
            budget_np_halfwidth_veh=contract()['NP_halfwidth_veh'],budget_nuf_fraction=contract()['NUF_halfwidth_fraction'],budget_halfwidth=self.width.tolist(),
            budget_lower=bounds(budget)[0],budget_upper=bounds(budget)[1],
            zero_budget_nonsmooth_axes=([1] if budget[1]==0 else []),
            band_policy=contract(),old_absolute_tolerances_active=False)
        return result

