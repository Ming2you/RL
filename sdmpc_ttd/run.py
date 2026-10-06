"""명세03/04/05/08/12/15: TTT-alpha*TTD inequality S-DMPC 배포 실행기.

기존 5회 warm-up, H3, 원 물리/제어 gate와 하위 6회를 보존하고 budget 후보를
최대 3개(기준) 또는 10개(추가 실험)로 둔다. 실제 TTT와 TTD를 별도 기록한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from .runtime import VENDOR as ROOT
HERE = ROOT/'work/ttd_upper10_20261006'
PROTOCOLS = None  # 실행 폴더 안에 재생성하며 기존 결과 파일을 요구하지 않는다.
SCENARIOS = ('sweet_155_w', 'sweet_170_w', 'sweet_170_skew15_w', 'sweet_170_incident_w', 'sweet_190_w')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    from .runtime import plain
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--objective', choices=('ttd',), default='ttd')
    parser.add_argument('--alpha', type=float, help='Optional assertion against regenerated full-run NC alpha')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--max-iterations', type=int, choices=(6,), default=6)
    parser.add_argument('--max-candidates', type=int, choices=(3, 10), default=3)
    parser.add_argument('--cpu-mask', type=int, default=None)
    parser.add_argument('--output', type=Path, required=True)
    # 저장 peak checkpoint에 의존하던 --validate-fixed는 배포 CLI에서 제외한다.
    args = parser.parse_args()
    if not 1 <= args.steps <= 80 or (args.alpha is not None and args.alpha < 0) or (args.objective == 'ctg' and args.alpha != 0.):
        parser.error('steps must be 1..80 and alpha nonnegative; CTG requires alpha=0')
    global PROTOCOLS
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from .runtime import prepare_inputs, environment, pin, bootstrap
    affinity = pin(args.cpu_mask)
    environment('SDMPC')
    bootstrap()
    PROTOCOLS, args.normalization_file, calibrated = prepare_inputs(out/'inputs', args.scenario)
    if args.alpha is not None and abs(args.alpha-calibrated) > 1e-15:
        raise ValueError('Requested alpha differs from full-horizon NC normalization')
    args.alpha = calibrated
    from sdmpc_objective import bootstrap, create_controller_class
    rc, cfg, options, saved, original_class = bootstrap()
    # 하위 6회와 기존 조기 종료 조건을 유지하고 상위 후보 평가 상한만 늘린다.
    from dataclasses import replace
    options = replace(options, max_iterations=args.max_iterations, max_candidates=args.max_candidates)
    from fixed_controller import GridCoordinates
    from fixed_policy import contract
    import numpy as np
    from ttd_accounting import DistanceCapture, install_scalar
    install_scalar()
    solver = create_controller_class(args.alpha, args.objective, args.max_candidates)(cfg, options)
    from terminal_cost import metadata as terminal_metadata
    save(out/'command.json', sys.argv)
    save(out/'adapter_manifest.json', solver.objective_adapter_manifest)
    save(out/'factory_config.json', rc.to_plain_dict(cfg))
    save(out/'solver_options.json', options)
    save(out/'contract.json', dict(objective='TTT-alpha*TTD' if args.objective == 'ttd' else 'TTT+V_terminal',
        objective_arm=args.objective, alpha_h_per_km=args.alpha,
        equivalent_objective='TTT/TTT_NC - TTD/TTD_NC' if args.objective == 'ttd' else None,
        normalization_fixed_for_scenario=args.objective == 'ttd',
        normalization_scope='saved full 14400s no-control trajectory' if args.objective == 'ttd' else None,
        warm_start_objective='unchanged historical PFO; proposal only; final guard uses current objective',
        terminal_settings=solver.terminal_settings if args.objective == 'ctg' else None,
        terminal_metadata=terminal_metadata(solver.evaluator.terminal_cfg) if args.objective == 'ctg' else None,
        terminal_scope='once at H3 terminal; no extra rollout' if args.objective == 'ctg' else None,
        ttd_metric='completed-link distance production proxy; not continuous vehicle odometry',
        raw_ttt_fields_preserved=True, players=9, lower_iterations=args.max_iterations, horizon_steps=3,
        max_candidates=options.max_candidates,
        upper_search_policy='legacy_first_then_fixed_center_interior_grid',
        budget=dict(contract(), budget_candidate_solves=args.max_candidates), seed=42, warmup_intervals=5,
        target_seconds=args.steps*cfg.simulation.T_c_sec, affinity=affinity,
        architecture='single_process_serial_players_central_prediction_and_resource_QP',
        own_model='existing first-order own and externality with proximal retained',
        controller_acceptance=False))
    reference = None
    if args.normalization_file:
        reference = read(args.normalization_file)
        if (not reference.get('experiment_complete') or reference['scenario'] != args.scenario
                or reference['elapsed_seconds'] != 14400 or reference.get('seed') != 42):
            raise RuntimeError('Normalization scenario or duration mismatch')
        if (not np.isfinite(reference['ttt']) or not np.isfinite(reference['ttd'])
                or reference['ttt'] <= 0. or reference['ttd'] <= 0.):
            raise RuntimeError('Nonfinite or nonpositive NC normalization')
        exact_alpha = reference['ttt']/reference['ttd']
        if abs(reference['alpha_h_per_km']-exact_alpha) > 1e-15:
            raise RuntimeError('Stored NC alpha differs from TTT_NC/TTD_NC')
        if args.objective == 'ttd' and abs(reference['alpha_h_per_km']-args.alpha) > 1e-15:
            raise RuntimeError('alpha does not match fixed full-horizon NC normalization')
        shutil.copy2(args.normalization_file, out/'normalization_input.json')

    protocol = PROTOCOLS/args.scenario
    if rc.to_plain_dict(cfg) != read(protocol/'config.json'):
        raise RuntimeError('Historical factory config mismatch')
    shutil.copytree(protocol, out/'protocol_snapshot')
    # 사용자 축소 범위: 기존 PFO는 warm proposal 생성기로 그대로 둔다.
    # 제안된 제어안의 실제 composite 값은 SDMPC의 원 rollout에서 다시 평가한다.
    warm = rc.make_controller('WU-FAITHFUL-FOLLOWER', cfg)
    sim, nc = rc.MixedTrafficSimulator(cfg), rc.MixedTrafficSimulator(cfg)
    if rc.to_plain_dict(sim.state) != read(protocol/'initial_state.json'):
        raise RuntimeError('Historical initial state mismatch')
    profile = rc.FrozenProfile(read(protocol/'forecast.json'), cfg.simulation.T_c_sec)
    previous = rc.ControlAction.uncontrolled(cfg)
    sources = {str(Path(module.__file__).resolve()): sha(module.__file__)
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).resolve().is_relative_to(ROOT/'work')
        and Path(module.__file__).suffix == '.py'}
    sources.update({str(path): sha(path) for path in (Path(__file__), Path(__file__).with_name('runtime.py'), HERE/'sdmpc_objective.py',
        HERE/'ttd_accounting.py', HERE/'terminal_cost.py') if path.exists()})
    inputs = {str(path): sha(path) for path in protocol.glob('*.json')}
    inputs.update({str(path): sha(path) for path in (Path(__file__).parent/'config').glob('*.json')})
    if args.normalization_file:
        inputs[str(args.normalization_file.resolve())] = sha(args.normalization_file)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for i, path in enumerate(sorted(sources)):
        shutil.copy2(path, snapshot/f'{i:03d}_{Path(path).name}')
    save(out/'source_hashes.json', sources)
    save(out/'input_hashes.json', inputs)
    save(out/'player_mapping.json', dict(players=solver.player_ids, ownership=vars(solver.ownership),
        axes=GridCoordinates(cfg, options, previous).axes))
    logs = []
    cumulative_ttd = cumulative_nc_ttd = 0.
    prior_state = sim.state.copy()
    start, cpu = time.perf_counter(), time.process_time()

    def verify():
        for path, digest in {**sources, **inputs}.items():
            if sha(path) != digest:
                raise RuntimeError('Source/input changed: '+path)

    try:
        for k in range(args.steps):
            forecast = profile.horizon(sim.state.time_sec, options.horizon_steps)
            selected = None
            counts = {}
            warm_wall = solve_wall = warm_cpu = solve_cpu = 0.
            before = solver.dual.copy()
            if k < 5:
                control = rc.ControlAction.uncontrolled(cfg)
            else:
                tick, ctick = time.perf_counter(), time.process_time()
                initial = warm.solve(sim.state.copy(), None, forecast, previous).control
                warm_wall, warm_cpu = time.perf_counter()-tick, time.process_time()-ctick
                tick, ctick = time.perf_counter(), time.process_time()
                selected, results = solver.decide(sim.state.copy(), forecast, previous, initial)
                solve_wall, solve_cpu = time.perf_counter()-tick, time.process_time()-ctick
                counts = solver.counts.copy()
                save(out/f'decision_{k:03d}.json', dict(candidates=results,
                    derivatives=solver.derivative_rows, counts=counts,
                    wall_seconds=solve_wall, cpu_seconds=solve_cpu, initial_prices=before,
                    committed_prices=solver.dual, execution_audit=solver.execution_audit,
                    upper_search_audit=solver.upper_search_audit,
                    anchor_audit=solver.anchor_audit, selected_execution=None if selected is None else
                    {key: value for key, value in selected.items() if key not in ('evaluation', 'rows', 'local_rows')}))
                if selected is None:
                    raise RuntimeError('Fail-closed: no physically valid discrete control')
                gate = solver.execution_check(selected['point'], selected['budget'])
                if not gate['physical_control_valid'] or not gate['budget_feasible']:
                    raise RuntimeError('Original physical/control/budget execution gate failed')
                if counts['local_trial_rollouts'] or counts['local_nlp_calls']:
                    raise RuntimeError('Unexpected local rollout/NLP')
                if not solver.anchor_audit['predicted_objective_nonworsening']:
                    raise RuntimeError('PFO composite objective guard failed')
                for candidate in results:
                    if len(candidate['rows']) > args.max_iterations:
                        raise RuntimeError('Iteration cap exceeded')
                    for row in candidate['local_rows']:
                        anchor = candidate['rows'][row['k']]
                        if row['model_reference'] != anchor['anchor'] or row['dual_used'] != anchor['dual_before']:
                            raise RuntimeError('Mixed reference or price')
                control = selected['control']
            verify()
            # plant와 무제어는 독립 capture를 사용해 어떤 예측 TTD도 실제 성과에 더하지 않는다.
            prior_state = sim.state.copy()
            with DistanceCapture(cfg) as distance:
                log = sim.step(control, forecast[0], k)
            with DistanceCapture(cfg) as nc_distance:
                nc_log = nc.step(rc.ControlAction.uncontrolled(cfg), forecast[0], k)
            ttd, nc_ttd = float(distance.total_ttd), float(nc_distance.total_ttd)
            cumulative_ttd += ttd
            cumulative_nc_ttd += nc_ttd
            raw_ttt, raw_nc_ttt = log.freeway_ttt+log.urban_ttt, nc_log.freeway_ttt+nc_log.urban_ttt
            row = dict(step=k, time_sec=sim.state.time_sec, ttt_veh_h=raw_ttt,
                nc_ttt_veh_h=raw_nc_ttt, ttd_veh_km=ttd, nc_ttd_veh_km=nc_ttd,
                executed_stage_objective_veh_h=raw_ttt-args.alpha*ttd,
                nc_executed_stage_objective_veh_h=raw_nc_ttt-args.alpha*nc_ttd,
                cumulative_ttt_veh_h=sim.total_ttt, cumulative_nc_ttt_veh_h=nc.total_ttt,
                cumulative_ttd_veh_km=cumulative_ttd, cumulative_nc_ttd_veh_km=cumulative_nc_ttd,
                decision_wall_seconds=warm_wall+solve_wall, warm_wall_seconds=warm_wall,
                solve_wall_seconds=solve_wall, parent_cpu_seconds=warm_cpu+solve_cpu,
                feasible=None if selected is None else selected['feasible'],
                converged=None if selected is None else selected['converged'],
                budget_exception=False if selected is None else selected['execution_exception'],
                selection_source=None if selected is None else selected['selection_source'],
                PFO_reference_objective=None if selected is None else solver.anchor_audit['reference_objective'],
                execution_check=None if selected is None else selected['execution_check'],
                predicted_ttt_veh_h=None if selected is None else selected['evaluation'].total_ttt,
                predicted_ttd_veh_km=None if selected is None else selected['evaluation'].total_ttd_veh_km,
                predicted_terminal_cost_veh_h=None if selected is None else selected['evaluation'].terminal_cost_veh_h,
                predicted_objective_value_veh_h=None if selected is None else selected['evaluation'].objective_value,
                budget=None if selected is None else selected['budget'],
                residual=None if selected is None else selected['original_residual'],
                termination=None if selected is None else selected['reason'], counts=counts)
            logs.append(row)
            save(out/'run_log.json', logs)
            save(out/f'plant_{k:03d}.json', dict(control=control, state=sim.state, log=log,
                nc_state=nc.state, nc_log=nc_log, persistent_dual=solver.dual, last_budget=solver.last_budget,
                ttd=dict(total=ttd, freeway=float(distance.freeway_ttd), urban=float(distance.urban_ttd)),
                nc_ttd=dict(total=nc_ttd, freeway=float(nc_distance.freeway_ttd), urban=float(nc_distance.urban_ttd))))
            previous = control.copy()
            save(out/'progress.json', dict(pid=os.getpid(), step=k+1, steps=args.steps,
                time_sec=sim.state.time_sec, ttt=sim.total_ttt, ttd=cumulative_ttd, status='running'))
            print(f'{args.scenario}: {k+1}/{args.steps}, TTT={sim.total_ttt:.6f}, TTD={cumulative_ttd:.6f}', flush=True)
        verify()
        controlled = logs[5:]
        normalization_replay_check = None
        if args.steps == 80:
            normalization_replay_check = dict(nc_ttt_error=nc.total_ttt-reference['ttt'],
                nc_ttd_error=cumulative_nc_ttd-reference['ttd'], absolute_tolerance=1e-8)
            normalization_replay_check['passed'] = max(abs(normalization_replay_check['nc_ttt_error']),
                abs(normalization_replay_check['nc_ttd_error'])) <= 1e-8
            if not normalization_replay_check['passed']:
                raise RuntimeError('NC plant replay does not match the fixed normalization trajectory')
        final_terminal = None
        if args.objective == 'ctg':
            from terminal_cost import evaluate_terminal
            # 겹치는 H3 말단비용을 interval마다 실현 TTT에 누적하지 않는다.
            final_terminal = sum(evaluate_terminal(solver.evaluator.terminal_cfg, sim.state,
                solver.ownership, prev_state=prior_state).values())
        save(out/'completion.json', dict(experiment_complete=True, source_input_integrity=True,
            steps=len(logs), elapsed_seconds=sim.state.time_sec, ttt=sim.total_ttt, nc_ttt=nc.total_ttt,
            ttd=cumulative_ttd, ttd_veh_km=cumulative_ttd, nc_ttd=cumulative_nc_ttd, nc_ttd_veh_km=cumulative_nc_ttd,
            executed_stage_objective_veh_h=sim.total_ttt-args.alpha*cumulative_ttd,
            objective_value_veh_h=sim.total_ttt-args.alpha*cumulative_ttd if args.objective == 'ttd' else None,
            objective_arm=args.objective,
            normalization_replay_check=normalization_replay_check,
            final_state_terminal_cost_veh_h=final_terminal,
            retrospective_ttt_plus_final_terminal_veh_h=None if final_terminal is None else sim.total_ttt+final_terminal,
            alpha_h_per_km=args.alpha, improvement_pct=100*(nc.total_ttt-sim.total_ttt)/nc.total_ttt,
            mean_decision_seconds=sum(r['decision_wall_seconds'] for r in controlled)/max(1, len(controlled)),
            feasible_count=sum(bool(r['feasible']) for r in controlled),
            converged_count=sum(bool(r['converged']) for r in controlled),
            budget_exception_count=sum(bool(r['budget_exception']) for r in controlled),
            wall_seconds=time.perf_counter()-start, parent_cpu_seconds=time.process_time()-cpu,
            controller_acceptance=False, convergence_certified=False))
        save(out/'progress.json', dict(pid=os.getpid(), status='completed', step=len(logs),
            steps=args.steps, time_sec=sim.state.time_sec))
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), completed_steps=len(logs),
            fail_closed=True, original_targets_preserved=True, thresholds_unchanged=True,
            elapsed_seconds=sim.state.time_sec, ttt=sim.total_ttt, ttd=cumulative_ttd,
            scenario=args.scenario, target_seconds=args.steps*cfg.simulation.T_c_sec,
            wall_seconds=time.perf_counter()-start, parent_cpu_seconds=time.process_time()-cpu))
        save(out/'progress.json', dict(pid=os.getpid(), status='failed', step=len(logs),
            steps=args.steps, time_sec=sim.state.time_sec))
        raise
    finally:
        if hasattr(warm, 'close'):
            warm.close()


def fixed_validation(old, zero, changed, state, forecast, previous, warm, out):
    """같은 저장 기준점에서 alpha=0 호환성과 목적 회계를 검사하고 실패도 보존한다."""
    import numpy as np
    from fixed_controller import GridCoordinates
    coords = GridCoordinates(old.cfg, old.options, previous)
    mapped = coords.decode(coords.quantize(coords.encode(warm)))
    models = [old, zero, changed]
    values = []
    for model in models:
        model.begin(state, forecast, previous)
        z = model.coords.encode(mapped)
        ev = model.evaluate(z)
        g, a = model.derivatives(z)
        values.append((z, ev, g, a))
    z, base, g0, a0 = values[0]
    _, compat, gz, az = values[1]
    _, composite, gc, ac = values[2]
    alpha0 = dict(state_match=all(vars(x)==vars(y) for x, y in zip(base.states, compat.states)),
        raw_ttt_difference=compat.total_ttt-base.total_ttt,
        objective_difference=compat.objective_value-base.total_ttt,
        gradient_max_difference=float(np.max(abs(gz-g0))),
        budget_jacobian_max_difference=float(np.max(abs(az-a0))))
    report = dict(alpha0=alpha0,
        composite=dict(ttt= composite.total_ttt, ttd=composite.total_ttd_veh_km,
            objective=composite.objective_value, terminal_cost_veh_h=composite.terminal_cost_veh_h,
            accounting_error=abs(sum(composite.objective_player_costs.values())-composite.objective_value),
            budget_jacobian_max_difference=float(np.max(abs(ac-a0))),
            derivative_evidence=changed.derivative_rows, derivatives_certified=False))
    save(out/'fixed_validation.json', report)
    if not alpha0['state_match'] or any(abs(alpha0[key]) > 1e-8 for key in alpha0 if key != 'state_match'):
        raise RuntimeError('Alpha-zero scalar or derivative compatibility failed')
    # budget과 seed가 같은 6회 하위 결과를 비교한다. 최적성 PASS로 읽어 바꾸지 않는다.
    results = [model.solve(base.budget_vector.copy(), z.copy(), model.dual.copy()) for model in (old, zero)]
    report['alpha0_fixed_budget'] = dict(control_match=vars(results[0]['control'])==vars(results[1]['control']),
        objective_difference=results[1]['evaluation'].objective_value-results[0]['evaluation'].total_ttt,
        residual_max_difference=float(np.max(abs(np.asarray(results[0]['original_residual'])-results[1]['original_residual']))),
        original_status=results[0]['status'], adapted_status=results[1]['status'],
        original_converged=results[0]['converged'], adapted_converged=results[1]['converged'])
    save(out/'fixed_validation.json', report)
    if not report['alpha0_fixed_budget']['control_match']:
        raise RuntimeError('Alpha-zero six-iteration fixed-budget result mismatch')
    print(json.dumps(report['alpha0_fixed_budget']), flush=True)


if __name__ == '__main__':
    main()
