"""저장 peak 상태에서 원 물리 모델의 고정 budget 및 짧은 폐루프 검증."""
import argparse
from dataclasses import asdict,replace,is_dataclass
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from runtime import ROOT,HERE,load,read


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if is_dataclass(x):return {k:plain(v) for k,v in asdict(x).items()}
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def write(path,value):
    path.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--outer',type=int,default=10)
    p.add_argument('--cpu-mask',type=int,default=2);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rc,cfg,opts,state,forecast,previous,warm=load(30)
    sys.path.insert(0,str(ROOT/'work/extended_matrix_no_slsqp_20260920'))
    from matrix_common import pin
    affinity=pin(args.cpu_mask)
    from slide_controller import SlideSDMPC,Coordinates
    opts=replace(opts,max_iterations=args.outer)
    solver=SlideSDMPC(cfg,opts)
    source_hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in HERE.glob('*.py')}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for file in source_hashes:(snapshot/Path(file).name).write_bytes(Path(file).read_bytes())
    write(out/'source_hashes.json',source_hashes);write(out/'command.json',sys.argv)
    write(out/'config.json',cfg);write(out/'options.json',opts)
    write(out/'contract.json',dict(players=9,freeway_players=4,urban_players=5,
        segment_groups=[[0,1,2,3],[4,5,6,7]],group_vsl_equal=True,
        objective='pure_TTT',nonlinear_own=True,shared_budgets='two_sided_bands',halfwidth=[.1,2.],
        NP='horizon inbound_service-outbound_service veh',NUF='sum metering commands veh/h',
        price_update='after all Jacobi local responses at k to k+1',initial_dual=np.zeros((2,2)),
        upper='uncertified candidate-specific dual direction and actual TTT selection',
        time_control_block='constant action over H3; independent future controls not implemented',
        authority_change='four grouped VSL including downstream; D/F offset now optimized',
        architecture='central coupled rollout, tangent and resource QP; serial local NLPs',
        timing_comparable=False,concurrent_reference_batch='sdmpc_upper_ttt_20260922',affinity=affinity))
    profile=rc.FrozenProfile(read(ROOT/'outputs/extended_matrix_20260919/protocols_0/sweet_190_skew15_w/forecast.json'),cfg.simulation.T_c_sec)
    sim=rc.MixedTrafficSimulator(cfg);sim.state=state.copy()
    nc=rc.MixedTrafficSimulator(cfg);nc.state=state.copy()
    warm_controller=rc.make_controller('WU-FAITHFUL-FOLLOWER',cfg)
    logs=[];start=time.perf_counter();cpu=time.process_time()
    try:
        # 同일 입력에서 원 모델은 그대로이며 비용 소유권만 다르게 합산한다.
        coords=Coordinates(cfg,opts,previous)
        write(out/'player_mapping.json',dict(players=solver.player_ids,ownership={
            k:plain(v) for k,v in vars(solver.ownership).items()},axes=coords.axes))
        for k in range(args.steps):
            forecast=profile.horizon(sim.state.time_sec,cfg.mpc.horizon_steps)
            initial=warm if k==0 else warm_controller.solve(sim.state.copy(),None,forecast,previous).control
            s0=sim.state.copy();w=time.perf_counter()
            selected,results=solver.decide(s0,forecast,previous,initial)
            write(out/f'decision_{k:03d}.json',dict(candidates=results,counts=solver.counts,
                derivatives=solver.derivative_rows,wall_seconds=time.perf_counter()-w))
            if selected is None:
                write(out/'failure.json',dict(reason='no_feasible_candidate',plant_advanced=False,
                    requested_budgets=[r['budget'] for r in results],algorithm_failure_not_physical_proof=True))
                raise RuntimeError('Fail-closed: no executable original-band candidate')
            c=selected['control']
            if not solver.feasible(selected['point'],np.array(selected['budget']),True):
                raise RuntimeError('Final original-band/control/physical gate failed')
            if any(hashlib.sha256(Path(f).read_bytes()).hexdigest()!=digest for f,digest in source_hashes.items()):
                raise RuntimeError('Run source changed')
            interval=sim.step(c,forecast[0],k)
            nc_interval=nc.step(rc.ControlAction.uncontrolled(cfg),forecast[0],k)
            log=dict(step=k,time_sec=sim.state.time_sec,ttt=interval.freeway_ttt+interval.urban_ttt,
                nc_ttt=nc_interval.freeway_ttt+nc_interval.urban_ttt,selected_budget=selected['budget'],
                budget_residual=selected['original_residual'],feasible=selected['feasible'],converged=selected['converged'],
                control=asdict(c),state=asdict(sim.state),counts=solver.counts,prices=solver.dual.tolist())
            logs.append(log);write(out/'run_log.json',logs);previous=c.copy()
            print(json.dumps(dict(step=k+1,ttt=log['ttt'],nc_ttt=log['nc_ttt'],feasible=True,converged=False)),flush=True)
        proposed=sum(l['ttt'] for l in logs);baseline=sum(l['nc_ttt'] for l in logs)
        write(out/'completion.json',dict(experiment_complete=True,steps=len(logs),new_simulation_seconds=len(logs)*180,
            ttt=proposed,nc_ttt=baseline,improvement_pct=100*(baseline-proposed)/baseline,
            controller_acceptance=False,convergence_certified=False,wall_seconds=time.perf_counter()-start,
            parent_cpu_seconds=time.process_time()-cpu,timing_comparable=False))
    finally:
        if hasattr(warm_controller,'close'):warm_controller.close()


if __name__=='__main__':main()
