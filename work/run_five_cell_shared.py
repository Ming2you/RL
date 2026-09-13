"""Balanced five-scenario shared DDQN experiment, with pinned resumable stages.

Every scenario contributes the same number of complete 75-step episodes. The
old single-scenario data and paused experiment are never altered or resumed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from rl_leader.env import make_cfg, configure_pstack_b13_follower_contract, CANONICAL_PSTACK_OPTIONS
from rl_leader.experiment_contract import ExperimentContract
from rl_leader.response_continuation_state import FIVE_CELL_CONTRACT_SHA256
from src.controllers.coordination import CoordinationActionSchema
from rl_leader.response_dqn import ResponseDQNConfig, train_response_dqn_member
from rl_leader.response_dqn_catalog import StructuredActionCatalog, build_structured_action_catalog
from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.five_cell_data import FIVE_CELL_SCENARIOS, merge_five_cell_replays, validate_five_cell_replay
from rl_leader.train_response_dqn import _configure_torch_threads
from work.five_cell_worker import collect_five_cell_episode

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def pin_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'pinned artifact changed: {path}')
    else:
        write_json(path, value)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'pinned file changed: {path}')


def stop_requested(root):
    if (root / 'STOP').exists():
        raise InterruptedError('five-cell STOP requested')


@contextmanager
def runner_lock(root):
    """An OS lock prevents duplicate runners; stale lock files are harmless."""
    import msvcrt
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def validate_plan(plan):
    if tuple(plan['scenarios']) != FIVE_CELL_SCENARIOS or plan['scenario_weights'] != [.2] * 5:
        raise ValueError('require the five specified scenarios with equal weights')
    if (plan['t_total'], plan['warmup_steps'], plan['control_steps']) != (14400., 5, 75):
        raise ValueError('the physical horizon and accounting must stay fixed')
    if any(type(plan[k]) is not int or plan[k] < 1 for k in ('actors', 'response_workers')):
        raise ValueError('actor and preview worker counts must be positive integers')
    if plan['maximum_response_workers'] != 8 or not 1 <= plan['actors'] * plan['response_workers'] <= 8:
        raise ValueError('total preview worker budget must be within 8')
    for key in ('rounds', 'initial_exploration_episodes_per_scenario',
                'onpolicy_episodes_per_scenario_per_round', 'max_episode_attempts'):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f'{key} must be positive')
    if plan['training']['backup_horizon'] != 1:
        raise ValueError('the first shared-policy study uses the validated 1-step control')
    if (plan['training']['ensemble_size'] != 3 or plan['training']['gamma'] != 1.
            or plan['training']['value_parameterization'] != 'finite_horizon_cost_v1'):
        raise ValueError('require three members with undiscounted finite-horizon cost heads')
    if plan['evaluation']['epsilon'] != 0 or plan['evaluation']['forced_first_action'] is not None:
        raise ValueError('evaluations must be ungated greedy policies')
    if plan['evaluation']['intervention_gate'] is not False or plan['evaluation']['lcb_fallback'] is not False:
        raise ValueError('evaluation gates and fallback are forbidden')
    if any(not 0 <= plan[key] <= 1 for key in ('initial_epsilon', 'onpolicy_epsilon')):
        raise ValueError('exploration epsilon must lie in [0,1]')
    if not np.isfinite(plan['max_episode_wall_seconds']) or plan['max_episode_wall_seconds'] <= 0:
        raise ValueError('episode wall budget must be finite and positive')
    if plan['response_equivalence_mode'] != 'post_commit_continuation_five_cell_v1':
        raise ValueError('require audited five-cell continuation semantics')
    ResponseDQNConfig(**plan['training']).validate()
    root = (ROOT / plan['output_dir']).resolve()
    if not root.is_relative_to(ROOT / 'results'):
        raise ValueError('experiment outputs must stay within repository results')
    return root


def prepare(plan, config_path):
    root = validate_plan(plan)
    stop_requested(root)
    contracts, catalog = {}, None
    for scenario in plan['scenarios']:
        # Construct metadata only; RLLeaderEnv.__init__ would run warmup physics.
        cfg, scenario_config = make_cfg(scenario)
        cfg.simulation.T_total = plan['t_total']
        configure_pstack_b13_follower_contract(cfg)
        context = SimpleNamespace(cfg=cfg, scenario=scenario_config, scenario_name=scenario,
                    pstack_options=CANONICAL_PSTACK_OPTIONS, warmup=5, pfo_supervisor_enabled=False,
                    T_total=14400., dt=float(cfg.simulation.control_interval))
        contract = ExperimentContract.from_env(context)
        if contract.sha256 != FIVE_CELL_CONTRACT_SHA256[scenario]:
            raise ValueError('prepared contract differs from the audited five-cell allowlist')
        contracts[scenario] = contract.payload
        action_schema = CoordinationActionSchema(cfg)
        source = build_structured_action_catalog(
            action_schema.names, magnitudes=(plan['catalog']['magnitude'],),
            families=('linear', 'quadratic', 'cross', 'combo'),
            domains=('freeway',), owners=tuple(plan['catalog']['owners']))
        selected = [source.actions[0]] + [a for a in source.actions[1:]
                                        if a.template in plan['catalog']['templates']]
        current = StructuredActionCatalog(source.action_names,
                    [replace(action, action_id=i) for i, action in enumerate(selected)])
        if catalog is not None and current.fingerprint != catalog.fingerprint:
            raise ValueError('scenarios expose different action catalogs')
        catalog = current
    if catalog.size != 17:
        raise ValueError('expected anchor plus 8 symmetric operators on each freeway owner')
    files = sorted({p for directory in ('src', 'rl_leader', 'work')
                    for p in (ROOT / directory).rglob('*.py')
                    if 'tests' not in p.parts and '__pycache__' not in p.parts}
                   | set((ROOT / 'work').glob('*.ps1'))
                   | set((ROOT / 'src' / 'config').glob('*.yaml')) | {config_path.resolve()})
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    runtime = {'python': sys.version, 'executable': sys.executable,
               'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform()}
    pin_json(root / 'plan.json', plan)
    pin_json(root / 'contracts.json', contracts)
    pin_json(root / 'catalog.json', catalog.as_manifest())
    pin_json(root / 'source_hashes.json', hashes)
    pin_json(root / 'runtime.json', runtime)
    return root, contracts, catalog, hashes


def job(plan, root, contracts, catalog, sources, *, scenario, purpose, index, models=()):
    from rl_leader.experiment_contract import ExperimentContract
    scene_id = plan['scenarios'].index(scenario)
    purpose_id = {'baseline': 0, 'collection': 1, 'evaluation': 2}[purpose]
    episode = purpose_id * 100000 + index * 10 + scene_id
    epsilon = 0. if purpose != 'collection' else (plan['onpolicy_epsilon'] if models else plan['initial_epsilon'])
    return {
        'directory': (root / purpose / f'{index:03d}' / scenario).relative_to(ROOT).as_posix(),
        'scenario': scenario, 'purpose': purpose,
        'experiment_contract_sha256': ExperimentContract.from_payload(contracts[scenario]).sha256,
        'catalog': catalog.as_manifest(), 'seed': plan['seed'] + episode, 'episode': episode,
        'epsilon': epsilon, 'checkpoints': list(models), 'response_workers': plan['response_workers'],
        'max_wall_seconds': plan['max_episode_wall_seconds'], 'stop_file': str(root / 'STOP'),
        'expected_source_hashes': sources, 'expected_model_hashes': {p: digest(p) for p in models},
        't_total': plan['t_total'], 'response_equivalence_mode': plan['response_equivalence_mode'],
    }


def run_jobs(plan, root, jobs):
    pending = list(jobs)
    for attempt in range(plan['max_episode_attempts']):
        stop_requested(root)
        retries = []
        with ProcessPoolExecutor(max_workers=plan['actors']) as pool:
            futures = {pool.submit(collect_five_cell_episode, payload): payload for payload in pending}
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    result = future.result()
                    print(json.dumps({'event': 'episode_complete', 'directory': payload['directory'],
                                      'ttt': result['total_ttt']}, sort_keys=True), flush=True)
                except InterruptedError:
                    retries.append(payload)
                except BaseException:
                    # Let other actors persist at their next complete transition.
                    (root / 'STOP').touch(exist_ok=True)
                    raise
        stop_requested(root)
        if not retries:
            return
        pending = retries
        write_json(root / 'retry_status.json', {'attempt': attempt + 1,
                      'pending': [p['directory'] for p in pending]})
    raise RuntimeError('episode wall budget exhausted; checkpoints retained for inspected resume')


def train_member(payload):
    _configure_torch_threads(1)
    verify_hashes(payload['source_hashes'])
    replay = load_frozen_response_replay(payload['data'])
    validate_five_cell_replay(replay)
    catalog = StructuredActionCatalog.from_manifest(replay.manifest['catalog'])
    model = train_response_dqn_member(replay, catalog.feature_matrix,
                catalog_fingerprint=catalog.fingerprint, config=ResponseDQNConfig(**payload['training']),
                seed=payload['seed'], bootstrap=False, stop_files=(payload['stop_file'],))
    path = Path(payload['output'])
    temporary = path.with_suffix('.tmp.pt')
    model.save(temporary); temporary.replace(path)
    result = {'checkpoint': str(path), 'sha256': digest(path), 'seed': model.seed,
              'updates': len(model.training_losses), 'last_loss': model.training_losses[-1],
              'training_input_signature': training_input_signature(payload)}
    write_json(path.with_suffix('.json'), result)
    return result


def training_input_signature(payload):
    identity = {'data_sha256': digest(payload['data']), 'training': payload['training'],
                'seed': payload['seed'], 'source_hashes': payload['source_hashes']}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def train_round(plan, root, catalog, contracts, sources, round_index, training_jobs):
    directory = root / f'round_{round_index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / 'training_replay.npz'
    replays = [load_frozen_response_replay(Path(p['directory']) / 'replay.npz') for p in training_jobs]
    merged = merge_five_cell_replays(replays, contracts=contracts, source=__name__)
    summary = validate_five_cell_replay(merged)
    if data.exists():
        # Check semantic arrays as compressed ZIP timestamps need not be stable.
        previous = load_frozen_response_replay(data)
        if previous.manifest != merged.manifest:
            raise ValueError('pinned merged replay manifest changed')
        for field in previous.__dataclass_fields__:
            if field != 'manifest' and not np.array_equal(getattr(previous, field), getattr(merged, field)):
                raise ValueError(f'pinned merged replay array changed: {field}')
    else:
        merged.save(data)
    pin_json(directory / 'dataset_audit.json', {'summary': summary, 'sha256': digest(data)})
    payloads = []
    model_paths = []
    for member in range(plan['training']['ensemble_size']):
        path = directory / f'shared_member_{member:02d}.pt'
        model_paths.append(path.relative_to(ROOT).as_posix())
        payload = {'data': str(data), 'training': plan['training'], 'seed': plan['seed'] + member,
                   'stop_file': str(root / 'STOP'), 'source_hashes': sources, 'output': str(path)}
        if path.exists():
            metadata = read_json(path.with_suffix('.json')) if path.with_suffix('.json').exists() else {}
            if (digest(path) != metadata.get('sha256')
                    or metadata.get('training_input_signature') != training_input_signature(payload)
                    or metadata.get('seed') != payload['seed']
                    or metadata.get('updates') != plan['training']['gradient_steps']):
                raise ValueError(f'partial or changed model output: {path}')
        else:
            payloads.append(payload)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=plan['training']['ensemble_size']) as pool:
        for result in pool.map(train_member, payloads):
            print(json.dumps({'event': 'shared_member_trained', **result}), flush=True)
    manifest = {'policy': 'one common greedy ensemble across all five scenarios',
                'checkpoints': model_paths, 'model_sha256': {p: digest(p) for p in model_paths},
                'dataset_sha256': digest(data), 'training': plan['training'],
                'contracts': {key: value for key, value in merged.manifest.items() if 'contract' in key}}
    pin_json(directory / 'ensemble_manifest.json', manifest)
    if not (directory / 'training_time.json').exists():
        write_json(directory / 'training_time.json', {'wall_seconds': time.perf_counter() - started})
    return model_paths


def compare_round(plan, root, baseline_jobs, eval_jobs, round_index):
    rows = []
    for base, evaluation in zip(baseline_jobs, eval_jobs):
        b, e = (read_json(Path(p['directory']) / 'summary.json') for p in (base, evaluation))
        if b['experiment_contract_sha256'] != e['experiment_contract_sha256']:
            raise ValueError('baseline and policy evaluation contract differ')
        if b['transitions'] != 75 or e['transitions'] != 75:
            raise ValueError('partial trajectories cannot be compared')
        gain = 100. * (1. - e['total_ttt'] / b['total_ttt'])
        rows.append({'scenario': base['scenario'], 'pstack_ttt': b['total_ttt'],
                     'shared_policy_ttt': e['total_ttt'], 'improvement_percent': gain,
                     'terminal_inventory': e['terminal_inventory'],
                     'evaluation_wall_seconds': e['wall_seconds']})
    gains = [row['improvement_percent'] for row in rows]
    result = {'round': round_index, 'scenarios': rows,
              'macro_improvement_percent': float(np.mean(gains)),
              'worst_scenario_improvement_percent': min(gains),
              'improved_scenarios': sum(value > 0 for value in gains),
              'generalization_claim': False, 'all_evaluations_complete': True,
              'policy': 'same frozen three-member ensemble in every scenario'}
    pin_json(root / f'round_{round_index:02d}' / 'comparison.json', result)
    return result


def run(plan, config_path, *, prepare_only=False):
    root, contracts, catalog, sources = prepare(plan, config_path)
    if prepare_only:
        print(json.dumps({'prepared': True, 'scenarios': plan['scenarios'], 'catalog_size': catalog.size,
                          'output_dir': str(root), 'rl_started': False}), flush=True)
        return
    with runner_lock(root):
        process = {'pid': os.getpid(), 'python': sys.executable, 'started_at': time.time(),
                   'config': str(config_path), 'state': 'running'}
        write_json(root / 'process.json', process)
        try:
            baseline_jobs = [job(plan, root, contracts, catalog, sources, scenario=s,
                                purpose='baseline', index=0) for s in plan['scenarios']]
            write_json(root / 'status.json', {'phase': 'baseline_collection', 'scenarios': plan['scenarios']})
            run_jobs(plan, root, baseline_jobs)
            training_jobs = list(baseline_jobs)
            initial_jobs = [job(plan, root, contracts, catalog, sources, scenario=s,
                                purpose='collection', index=i)
                            for i in range(plan['initial_exploration_episodes_per_scenario'])
                            for s in plan['scenarios']]
            write_json(root / 'status.json', {'phase': 'balanced_initial_collection'})
            run_jobs(plan, root, initial_jobs)
            training_jobs.extend(initial_jobs)
            comparisons, models = [], []
            for round_index in range(plan['rounds']):
                stop_requested(root); verify_hashes(sources)
                if round_index:
                    count = plan['onpolicy_episodes_per_scenario_per_round']
                    offset = plan['initial_exploration_episodes_per_scenario'] + (round_index - 1) * count
                    additional = [job(plan, root, contracts, catalog, sources, scenario=s,
                                      purpose='collection', index=offset+i, models=models)
                                  for i in range(count) for s in plan['scenarios']]
                    write_json(root / 'status.json', {'phase': 'balanced_onpolicy_collection', 'round': round_index})
                    run_jobs(plan, root, additional); training_jobs.extend(additional)
                write_json(root / 'status.json', {'phase': 'shared_policy_training', 'round': round_index})
                models = train_round(plan, root, catalog, contracts, sources, round_index, training_jobs)
                evaluation_jobs = [job(plan, root, contracts, catalog, sources, scenario=s,
                                       purpose='evaluation', index=round_index, models=models) for s in plan['scenarios']]
                write_json(root / 'status.json', {'phase': 'five_cell_evaluation', 'round': round_index})
                run_jobs(plan, root, evaluation_jobs)
                comparisons.append(compare_round(plan, root, baseline_jobs, evaluation_jobs, round_index))
            write_json(root / 'comparison.json', {'rounds': comparisons, 'research_complete_for_this_plan': True})
            write_json(root / 'status.json', {'phase': 'complete', 'rounds': plan['rounds']})
            process['state'] = 'completed'
        except InterruptedError as exc:
            write_json(root / 'status.json', {'phase': 'paused', 'reason': str(exc)})
            process['state'] = 'paused'
        except BaseException as exc:
            write_json(root / 'status.json', {'phase': 'failed', 'reason': str(exc)})
            process['state'] = 'failed'
            raise
        finally:
            process['finished_at'] = time.time()
            write_json(root / 'process.json', process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('work/five_cell_shared_v1.json'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    _configure_torch_threads(1)
    run(read_json(args.config), args.config, prepare_only=args.prepare_only)


if __name__ == '__main__':
    main()
