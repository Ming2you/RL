"""명세05/08/12/15: 매 interval PFO budget anchor/TTT guard 역사190 실행."""
import argparse
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/sdmpc_upper_ttt_20260922'))
from matrix_common import pin, environment, read
PROTOCOLS=ROOT/'outputs/sdmpc_budget_exception_all_20260922/protocols_0'
SCENARIOS=('sweet_155_w','sweet_170_w','sweet_170_incident_w','sweet_170_skew15_w','sweet_190_w','sweet_190_skew15_w','sweet_190_incident_w','sweet_220_w','sweet_220_skew15_w','sweet_220_incident_w')
sys.path.insert(0,str(ROOT/'work/sdmpc_slide_alignment_20260922'))
from runtime import load
from run_validation import plain


def save(path,value):
    import json
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',choices=SCENARIOS,required=True)
    p.add_argument('--variant',choices=('band1','band5','band10','none','upper'),required=True)
    p.add_argument('--externality',choices=('on','off'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,choices=[7,11,80],default=80)
    p.add_argument('--cpu-mask',type=int,default=8);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    affinity=pin(args.cpu_mask);environment('SDMPC')
    rc,cfg,opts,*_=load(30);environment('SDMPC');opts=replace(opts,max_iterations=6)
    sys.path.insert(0,str(ROOT/'work/sdmpc_group_proxlinear_20260922'))
    sys.path[:0]=[str(ROOT/'work/sdmpc_central_kkt_reuse_20260922'),str(ROOT/'work/sdmpc_selected_dual_20260922'),str(ROOT/'work/sdmpc_np10_20260922'),str(ROOT/'work/sdmpc_relative_band_20260922')]
    sys.path.insert(0,str(HERE))
    sys.path.insert(0,str(ROOT/'work/sdmpc_budget_exception_20260922'))
    os.environ['SDMPC_BUDGET_VARIANT']=args.variant
    os.environ['SDMPC_EXTERNALITY']=args.externality
    sys.path.insert(0,str(HERE))
    from externality_policy import ARMS, ENABLED, audit_rows
    arm=next(k for k,v in ARMS.items() if v==(args.variant,args.externality))
    from fixed_policy import contract
    if args.variant=='none':opts=replace(opts,max_candidates=1)
    from anchor_controller import PFOAnchorSDMPC as GroupProxSDMPC
    from fixed_controller import GridCoordinates as Coordinates
    import reused_model
    import numpy as np
    for module_name in ('fixed_policy','band_math','prox_controller','fixed_controller','central','exception_controller','anchor_controller','externality_policy'):
        if Path(sys.modules[module_name].__file__).resolve().parent!=HERE:raise RuntimeError('Wrong variant module: '+module_name)
    protocol=PROTOCOLS/args.scenario
    if rc.to_plain_dict(cfg)!=read(protocol/'config.json'):
        raise RuntimeError('Historical factory config mismatch')
    shutil.copytree(protocol,out/'protocol_snapshot')
    inputs={str(q):sha(q) for q in protocol.glob('*.json')}
    # 실제 import된 역사 물리 코드와 새 controller를 함께 보존한다.
    paths=set(q.resolve() for folder in (HERE,ROOT/'work/sdmpc_group_proxlinear_20260922',ROOT/'work/sdmpc_slide_alignment_20260922') for q in folder.glob('*.py'))
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT/'work'))
    sources={str(q):sha(q) for q in paths};snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,q in enumerate(sorted(paths)):shutil.copy2(q,snapshot/f'{i:03d}_{q.name}')
    save(out/'source_hashes.json',sources);save(out/'input_hashes.json',inputs)
    save(out/'command.json',sys.argv);save(out/'factory_config.json',rc.to_plain_dict(cfg));save(out/'solver_options.json',opts)
    save(out/'contract.json',dict(players=9,freeway_players=4,urban_players=5,outer_limit=6,
        horizon_steps=opts.horizon_steps,local_model='own_and_externality_first_order_plus_proximal' if ENABLED else 'own_first_order_plus_proximal',
        ablation_arm=arm,externality_enabled=ENABLED,full_derivatives_still_computed=True,
        central_TTT_guard_and_full_gradient_diagnostics_retained=True,
        sensitivity='refresh_at_each_changed_anchor_exact_key_cache',upper='current_interval_PFO_witness_then_reused_central_multiplier',
        upper_objective='pure_TTT',budget=contract(),NP_halfwidth_veh=contract()['NP_halfwidth_veh'],NUF_halfwidth_fraction=contract()['NUF_halfwidth_fraction'],
        NP='horizon net service veh',NUF='meter command sum veh/h',price_update='k_to_k_plus_1',
        warmup_intervals=5,seed=42,target_seconds=args.steps*cfg.simulation.T_c_sec,
        concurrent_reference_runs='see launch_process_inventory.json',timing_comparable=False,affinity=affinity,
        execution_policy='current_interval_discrete_PFO_budget_witness_and_H3_TTT_guard; original_gates_required',
        exception_norm='Linf using unchanged budget scales; TTT tie break; prices held',
        architecture='single_process_serial_players_central_prediction_and_resource_QP',controller_acceptance=False))
    sim=rc.MixedTrafficSimulator(cfg);nc=rc.MixedTrafficSimulator(cfg)
    if rc.to_plain_dict(sim.state)!=read(protocol/'initial_state.json'):raise RuntimeError('Historical initial state mismatch')
    if opts.horizon_steps!=cfg.mpc.horizon_steps:raise RuntimeError('Prediction horizon mismatch')
    solver=GroupProxSDMPC(cfg,opts);previous=rc.ControlAction.uncontrolled(cfg)
    save(out/'player_mapping.json',dict(players=solver.player_ids,ownership=vars(solver.ownership),axes=Coordinates(cfg,opts,previous).axes))
    profile=rc.FrozenProfile(read(protocol/'forecast.json'),cfg.simulation.T_c_sec)
    warm=rc.make_controller('WU-FAITHFUL-FOLLOWER',cfg)
    logs=[];start=time.perf_counter();cpu=time.process_time()
    def verify():
        for q,h in {**sources,**inputs}.items():
            if sha(q)!=h:raise RuntimeError('Source/input changed: '+q)
    try:
        for k in range(args.steps):
            ff=profile.horizon(sim.state.time_sec,opts.horizon_steps)
            selected=None;results=[];counts={};wwall=wcpu=dwall=dcpu=0.
            before=solver.dual.copy()
            if k<5:control=rc.ControlAction.uncontrolled(cfg)
            else:
                tick=time.perf_counter();ctick=time.process_time()
                initial=warm.solve(sim.state.copy(),None,ff,previous).control
                wwall=time.perf_counter()-tick;wcpu=time.process_time()-ctick
                tick=time.perf_counter();ctick=time.process_time()
                selected,results=solver.decide(sim.state.copy(),ff,previous,initial)
                audit_rows(results)
                dwall=time.perf_counter()-tick;dcpu=time.process_time()-ctick;counts=solver.counts.copy()
                save(out/f'decision_{k:03d}.json',dict(candidates=results,derivatives=solver.derivative_rows,
                    counts=counts,wall_seconds=dwall,cpu_seconds=dcpu,initial_prices=before,
                    committed_prices=solver.dual,requested_budgets=[r['budget'] for r in results],
                    execution_audit=solver.execution_audit,anchor_audit=solver.anchor_audit,selected_execution=None if selected is None else
                    {key:value for key,value in selected.items() if key not in ('evaluation','rows','local_rows')}))
                if selected is None:raise RuntimeError('Fail-closed: no physically valid discrete control in candidate archive')
                gate=solver.execution_check(selected['point'],selected['budget'])
                if not gate['physical_control_valid']:raise RuntimeError('Original physical/control execution gate failed')
                if not gate['budget_feasible'] and not selected['execution_exception']:raise RuntimeError('Unrecorded budget violation')
                if selected['execution_exception'] and not np.array_equal(solver.dual,before):raise RuntimeError('Failed candidate changed prices')
                if counts['local_trial_rollouts'] or counts['local_nlp_calls']:raise RuntimeError('Unexpected local rollout/NLP')
                for candidate in results:
                    if len(candidate['rows'])>6:raise RuntimeError('Iteration cap exceeded')
                    for row in candidate['local_rows']:
                        anchor=candidate['rows'][row['k']]
                        if row['model_reference']!=anchor['anchor'] or row['dual_used']!=anchor['dual_before']:
                            raise RuntimeError('Mixed reference or price')
                if not solver.anchor_audit['predicted_TTT_nonworsening']:raise RuntimeError('PFO TTT guard failed')
                if not gate['budget_feasible']:raise RuntimeError('PFO anchor policy executed budget violation')
                control=selected['control']
            verify()
            # 물리/제어는 필수 통과, budget 예외는 별도 기록한다. 무제어는 동일 수요의 독립 상태다.
            log=sim.step(control,ff[0],k);nlog=nc.step(rc.ControlAction.uncontrolled(cfg),ff[0],k)
            row=dict(step=k,time_sec=sim.state.time_sec,ttt_veh_h=log.freeway_ttt+log.urban_ttt,
                nc_ttt_veh_h=nlog.freeway_ttt+nlog.urban_ttt,cumulative_ttt_veh_h=sim.total_ttt,
                cumulative_nc_ttt_veh_h=nc.total_ttt,decision_wall_seconds=wwall+dwall,
                warm_wall_seconds=wwall,solve_wall_seconds=dwall,parent_cpu_seconds=wcpu+dcpu,
                feasible=None if selected is None else selected['feasible'],
                budget_exception=False if selected is None else selected['execution_exception'],
                selection_source=None if selected is None else selected['selection_source'],
                PFO_reference_TTT=None if selected is None else solver.anchor_audit['reference_TTT'],
                PFO_reference_budget=None if selected is None else solver.anchor_audit['reference_budget'],
                execution_check=None if selected is None else selected['execution_check'],
                converged=None if selected is None else selected['converged'],
                budget=None if selected is None else selected['budget'],residual=None if selected is None else selected['original_residual'],
                termination=None if selected is None else selected['reason'],counts=counts)
            logs.append(row);save(out/'run_log.json',logs)
            save(out/f'plant_{k:03d}.json',dict(control=control,state=sim.state,log=log,nc_state=nc.state,nc_log=nlog,
                persistent_dual=solver.dual,last_budget=solver.last_budget))
            previous=control.copy()
            save(out/'progress.json',dict(pid=os.getpid(),step=k+1,steps=args.steps,time_sec=sim.state.time_sec,
                ttt=sim.total_ttt,nc_ttt=nc.total_ttt,status='running'))
            print(f'{args.scenario}: {k+1}/{args.steps}, TTT={sim.total_ttt:.6f}, NC={nc.total_ttt:.6f}',flush=True)
        verify();controlled=logs[5:]
        save(out/'progress.json',dict(pid=os.getpid(),status='completed',step=len(logs),steps=args.steps,time_sec=sim.state.time_sec))
        save(out/'completion.json',dict(experiment_complete=True,source_input_integrity=True,steps=len(logs),
            elapsed_seconds=sim.state.time_sec,ttt=sim.total_ttt,nc_ttt=nc.total_ttt,
            improvement_pct=100*(nc.total_ttt-sim.total_ttt)/nc.total_ttt,
            mean_decision_seconds=sum(r['decision_wall_seconds'] for r in controlled)/len(controlled),
            wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
            feasible_count=sum(bool(r['feasible']) for r in controlled),
            budget_exception_count=sum(bool(r['budget_exception']) for r in controlled),
            max_scaled_band_excess=max((max(r['execution_check']['scaled_excess']) for r in controlled),default=0.),
            converged_count=sum(bool(r['converged']) for r in controlled),
            PFO_reference_selected_count=sum(r['selection_source']=='PFO_reference' for r in controlled),
            TTT_guard_pass_count=sum(r['execution_check']['ttt']<=r['PFO_reference_TTT'] for r in controlled),
            budget_variant=args.variant,ablation_arm=arm,externality_enabled=ENABLED,
            controller_acceptance=False,timing_comparable=False))
    except Exception:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),completed_steps=len(logs),
            fail_closed=True,budget_exception_count=sum(bool(r['budget_exception']) for r in logs),
            original_targets_preserved=True,thresholds_unchanged=True,
            elapsed_seconds=sim.state.time_sec,ttt=sim.total_ttt,nc_ttt=nc.total_ttt,
            scenario=args.scenario,target_seconds=args.steps*cfg.simulation.T_c_sec,
            wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu))
        save(out/'progress.json',dict(pid=os.getpid(),status='failed',step=len(logs),steps=args.steps,time_sec=sim.state.time_sec))
        raise
    finally:
        if hasattr(warm,'close'):warm.close()


if __name__=='__main__':main()

