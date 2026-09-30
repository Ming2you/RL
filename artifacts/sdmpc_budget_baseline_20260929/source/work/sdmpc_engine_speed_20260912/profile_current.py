"""Unchanged saved C problem, explicit wall/CPU scopes and true forecast attribution.

Specs: 03/04/05/10/12/15. Source physics and controller options remain unchanged.
Instrumentation wraps entry points only. Timers are inclusive unless named exclusive.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
from dataclasses import asdict
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HIST = ROOT / 'work/sdmpc_matrix_14400_20260912/historical_tree'
HELPERS = ROOT / 'work/sdmpc_matrix_14400_20260912'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bootstrap(cell):
    env = read(cell / 'protocol_snapshot/protocol.json')['frozen_environment']
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    sys.path[:0] = [str(HERE / 'baseline_controller'), str(HIST), str(HELPERS), str(ROOT)]
    import run_cell as rc
    rc.load_runtime(HIST)
    from continuous_vsl_controller import ContinuousVSLPriceSDMPC
    return rc, ContinuousVSLPriceSDMPC


def restore_state(rc, data):
    data = dict(data)
    for name in ('urban_arrival_buffer', 'urban_storage_release_buffer'):
        data[name] = {key: {int(t): v for t, v in buf.items()}
                      for key, buf in data.get(name, {}).items()}
    from src.models.state import TrafficState
    return TrafficState(**data)


def load_decision(rc, cell, step):
    source = read(cell / f'candidate_signals/decision_inputs_{step:03d}.json')
    cfg = rc.restore_historical_config(read(cell / 'factory_config_used.json'))
    options = rc.PlayerSDMPCOptions(**read(cell / 'solver_options.json'))
    state = restore_state(rc, source['state'])
    forecast = []
    for d in source['forecast']:
        d['freeway_lane_loss'] = {link: {int(k): v for k, v in losses.items()}
                                for link, losses in d.get('freeway_lane_loss', {}).items()}
        forecast.append(rc.DemandStep(**d))
    return cfg, options, state, forecast, rc.ControlAction(**source['previous']), rc.ControlAction(**source['warm_control']), source['price_before']


class Timings:
    def __init__(self):
        self.rows = {}
        self.candidates = []
        self.rollouts = []
        self.local = []
        self.phase = 'setup'

    @contextmanager
    def scope(self, name):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            row = self.rows.setdefault(name, dict(calls=0, wall_seconds=0., cpu_seconds=0.))
            row['calls'] += 1
            row['wall_seconds'] += time.perf_counter() - wall
            row['cpu_seconds'] += time.process_time() - cpu

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.scope(label):
                return original(*args, **kwargs)
        setattr(obj, name, measured)

    def install(self, rc, cls):
        from src.controllers import player_sensitivity_dmpc as core
        from src.simulation import coupling
        import frozen_price_controller as frozen
        original = core.PlayerSensitivityDMPC.evaluate_control
        @functools.wraps(original)
        def rollout(solver, *args, **kwargs):
            names = []
            frame = sys._getframe(1)
            while frame:
                names.append(frame.f_code.co_name)
                frame = frame.f_back
            purpose = ('sensitivity' if '_derivatives' in names else
                       'own_cost' if 'own_callback' in names else
                       'restoration' if any('restor' in n for n in names) else
                       'quantization' if '_quantize' in ' '.join(names) else 'other')
            wall, cpu = time.perf_counter(), time.process_time()
            with self.scope('full_forecast/' + purpose):
                result = original(solver, *args, **kwargs)
            self.rollouts.append(dict(purpose=purpose, phase=self.phase,
                wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                horizon_steps=len(result.states), total_ttt=result.total_ttt))
            return result
        core.PlayerSensitivityDMPC.evaluate_control = rollout
        # Calls made by local PFO predictors are recorded separately, never called full forecasts.
        self.wrap(core, 'run_coupled_interval', 'player_coupled_interval')
        self.wrap(core, 'validate_player_control', 'original_control_check')
        self.wrap(core.PlayerSensitivityDMPC, '_feasible', 'original_budget_boolean_check')
        self.wrap(frozen.FrozenPriceSDMPC, '_derivatives', 'sensitivity_requests_inclusive')
        self.wrap(frozen.FrozenPriceSDMPC, '_check_context', 'context_hash_check')
        self.wrap(frozen.FrozenPriceSDMPC, '_price_update', 'price_two_number_update')
        self.wrap(frozen, 'mixed_resource_step', 'central_resource_qp')
        old_local = frozen.FrozenPriceSDMPC._local
        def local(controller, *args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            result = old_local(controller, *args, **kwargs)
            self.local.append(dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                status=result.status, success=result.success, evaluations=result.evaluations,
                iterations=result.iterations, stationarity=result.stationarity))
            return result
        frozen.FrozenPriceSDMPC._local = local
        old_candidate = cls.solve_fixed_budget
        def candidate(solver, *args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            count = len(self.rollouts)
            result = old_candidate(solver, *args, **kwargs)
            self.candidates.append(dict(index=len(self.candidates), wall_seconds=time.perf_counter()-wall,
                cpu_seconds=time.process_time()-cpu, full_forecasts=len(self.rollouts)-count,
                budget=asdict(args[3]), objective=result.objective, feasible=result.feasible,
                converged=result.converged, termination_reason=result.termination_reason))
            return result
        cls.solve_fixed_budget = candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=Path, default=ROOT / 'outputs/sdmpc_frozen_price_20260912/matrix_C_attempt_0/interval_vsl/sweet_170_w')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--mode', choices=['rollout', 'decision'], default='rollout')
    parser.add_argument('--engine', choices=['baseline', 'static'], default='baseline')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cprofile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rc, cls = bootstrap(args.cell)
    cfg, options, state, forecast, previous, warm_saved, prices = load_decision(rc, args.cell, args.step)
    if args.engine != 'baseline':
        from static_engine import install
        install(cfg)
    timer = Timings()
    timer.install(rc, cls)
    solver = cls(cfg, options)
    import numpy as np
    solver._persistent_prices = np.array(prices, dtype=float)
    before = digest([rc.to_plain_dict(cfg), asdict(state), [asdict(d) for d in forecast], asdict(previous), asdict(warm_saved)])
    profile = cProfile.Profile() if args.cprofile else None
    results = []
    if profile:
        profile.enable()
    if args.mode == 'rollout':
        coords = solver.coordinate_type(cfg, options, previous)
        control = coords.decode(coords.encode(warm_saved), coords.initial_vsl(warm_saved))
        # No open interval, so no physical rollout cache. Every repetition executes H coupled intervals.
        timer.phase = 'rollout_benchmark'
        for i in range(args.repeats):
            wall, cpu = time.perf_counter(), time.process_time()
            ev = solver.evaluate_control(state, forecast, control, previous)
            results.append(dict(repeat=i, wall_seconds=time.perf_counter()-wall,
                                cpu_seconds=time.process_time()-cpu, total_ttt=ev.total_ttt))
        value = asdict(ev)
        value.pop('evaluation_seconds', None)
        write(args.output / 'evaluation.json', value)
        write(args.output / 'control.json', asdict(control))
    else:
        timer.phase = 'warm_start'
        warm = rc.make_controller('WU-FAITHFUL-FOLLOWER', cfg)
        with timer.scope('decision_total'):
            with timer.scope('warm_start'):
                warm_control = warm.solve(state.copy(), None, forecast, previous).control
            solver.warm_start_control = warm_control
            timer.phase = 'controller'
            with timer.scope('controller_without_warm_start'):
                action = solver.decide(state.copy(), forecast, previous, cfg)
            with timer.scope('final_scalar_execution_gate'):
                result = solver.last_result
                budget = rc.PlayerBudget(action.N_P_star, action.N_UF_star)
                if not (result and result.feasible and result.evaluation.physical_valid
                        and solver._feasible(result.evaluation, budget)
                        and rc.validate_player_control(action, previous, cfg)['valid']):
                    raise RuntimeError('Original fail-closed execution check failed')
        write(args.output / 'control.json', asdict(action))
        write(args.output / 'warm_control.json', asdict(warm_control))
        write(args.output / 'warm_saved.json', asdict(warm_saved))
        write(args.output / 'selected_result.json', asdict(result))
        write(args.output / 'price_rows.json', solver.price_rows)
        write(args.output / 'candidate_rows.json', solver.candidate_rows)
        write(args.output / 'solver_metrics.json', solver.interval_metrics)
    if profile:
        profile.disable()
        profile.dump_stats(str(args.output / 'profile.pstats'))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('tottime').print_stats(80)
        (args.output / 'profile_self.txt').write_text(stream.getvalue(), encoding='utf-8')
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumtime').print_stats(60)
        (args.output / 'profile_cumulative.txt').write_text(stream.getvalue(), encoding='utf-8')
    after = digest([rc.to_plain_dict(cfg), asdict(state), [asdict(d) for d in forecast], asdict(previous), asdict(warm_saved)])
    assert before == after, 'Input mutation'
    write(args.output / 'measurements.json', dict(scopes=timer.rows, full_forecasts=timer.rollouts,
        local_nlps=timer.local, candidates=timer.candidates, repeated_rollouts=results,
        engine=args.engine, cprofile_enabled=args.cprofile, step=args.step, input_digest=before,
        prices_initial=prices, mode=args.mode, input_unchanged=True,
        note='Inclusive scopes overlap; do not sum. decision_total excludes plant/logging; no plant executed here.'))
    write(args.output / 'options.json', asdict(options))
    write(args.output / 'config.json', rc.to_plain_dict(cfg))
    sources = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for n, m in sys.modules.items() if getattr(m, '__file__', None)
               and (n.startswith('src.') or n in ['frozen_price_controller', 'continuous_vsl_controller', 'static_engine'])}
    write(args.output / 'source_hashes.json', sources)
    write(args.output / 'command.json', sys.argv)
    print(json.dumps({'complete': True, 'output': str(args.output), 'scopes': timer.rows,
                      'full_forecasts': len(timer.rollouts)}), flush=True)


if __name__ == '__main__':
    main()
