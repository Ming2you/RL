"""Target two unobserved nonlinear responses, retaining sequential Bellman data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_leader.response_dqn_catalog import StructuredActionCatalog
from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.run_sequential_response_ddqn import _atomic_json, _collect_actor_worker, validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.response_policy_prefix import capture_matched_prefix
from work.run_response_cql_ablation import _pinned_json, _read_json, main as evaluate_comparison
from work.run_response_onpolicy_refit import _models, replay_fit
from work.run_response_reference_coverage import persist_merged_replay, train_variant
from work.run_response_value_head_ablation import cost_head_checks


def branch_opportunity(training, policy, branch):
    step, action = branch['step'], branch['first_action']
    if action not in (6, 9) or not 0 <= step < policy.size or not policy.action_mask[step, action]:
        raise ValueError('requested nonlinear response is not an executable representative')
    same = np.all(training.observation == policy.observation[step], axis=1)
    if np.any(training.action_id[same] == action):
        raise ValueError('targeted state-action transition is already observed')
    response = policy.response_features[step, action]
    if any(np.allclose(response, other, rtol=0, atol=1e-6) for other in policy.response_features[step, :6]):
        raise ValueError('target nonlinear response equals a linear response')
    return {'step': step, 'action': action, 'matched_state_behavior_actions': training.action_id[same].tolist(),
            'global_action_support': int(training.action_support_counts()[action]),
            'physically_distinct_from_linear_candidates': True}


def verify_branch(summary, branch_replay, policy, policy_summary, branch):
    validate_complete_episode(branch_replay)
    step, action = branch['step'], branch['first_action']
    np.testing.assert_array_equal(branch_replay.control_step, np.arange(step, 75))
    np.testing.assert_array_equal(branch_replay.observation[0], policy.observation[step])
    np.testing.assert_array_equal(branch_replay.response_features[0], policy.response_features[step])
    np.testing.assert_array_equal(branch_replay.action_mask[0], policy.action_mask[step])
    if branch_replay.action_id[0] != action:
        raise ValueError('forced exploratory response was aliased to another action')
    for field, expected in {'scope': 'fixed_prefix_continuation', 'epsilon': 0.0,
                            'first_action': action, 'lcb_guard': False, 'response_preview': True,
                            'start_control_step': step, 'end_control_step': 74,
                            'transitions': 75 - step}.items():
        if summary.get(field) != expected:
            raise ValueError(f'branch summary mismatch: {field}')
    for field in ('experiment_contract_sha256', 'catalog_fingerprint', 'scenario', 't_total_sec', 'response_equivalence_mode'):
        if branch_replay.manifest.get(field) != policy.manifest.get(field):
            raise ValueError(f'branch replay contract mismatch: {field}')
    prefix_ttt = policy_summary['prefix_ttt'] - policy.reward[:step].astype(float).sum()
    np.testing.assert_allclose(summary['prefix_ttt'], prefix_ttt, rtol=1e-6)
    np.testing.assert_allclose(summary['prefix_ttt'] - branch_replay.reward.astype(float).sum(), summary['total_ttt'], rtol=1e-6)
    return {'step': step, 'first_action': action, 'transitions': branch_replay.size,
            'total_ttt_with_fixed_prefix': summary['total_ttt'],
            'delta_ttt_vs_frozen_cost_policy': summary['total_ttt'] - policy_summary['total_ttt'],
            'terminal_inventory': summary['terminal_inventory'], 'wall_seconds': summary['wall_seconds'],
            'interpretation': 'Matched-prefix exploratory first action, then frozen learned-policy recourse. Not ungated policy performance, an optimal return, or a fixed training label.'}


def run(plan):
    root, previous = Path(plan['output_dir']), Path(plan['start_only_after_output_dir'])
    stops = [str(root / 'STOP'), str(previous / 'STOP'), *plan['additional_stop_files']]
    def check_stop():
        if any(Path(path).exists() for path in stops):
            raise InterruptedError('STOP requested')
    check_stop()
    if _read_json(previous / 'status.json')['phase'] != 'ablation_complete' or _read_json(previous / 'process.json')['state'] != 'exited':
        raise ValueError('predecessor has not completed')
    _pinned_json(root / 'coverage_plan.json', plan)
    _pinned_json(root / 'implementation_hashes.json', {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in plan['implementation_files']
    })
    for path, digest in plan['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest.lower():
            raise ValueError(f'input hash changed: {path}')
    replays = [load_frozen_response_replay(path) for path in plan['refit_replays']]
    for replay in replays:
        validate_sequential_td_replay(replay, gamma=1)
    policy = load_frozen_response_replay(plan['policy_replay'])
    validate_complete_episode(policy)
    np.testing.assert_array_equal(policy.control_step, np.arange(75))
    summary = _read_json(Path(plan['control_evaluation_dir']) / 'summary.json')
    if summary['scope'] != 'ungated_full_run' or summary['epsilon'] != 0 or summary['first_action'] is not None or summary['lcb_guard']:
        raise ValueError('source must be the unforced greedy full policy')
    if Path(summary['replay']).resolve() != Path(plan['policy_replay']).resolve():
        raise ValueError('policy summary points to a different replay')
    np.testing.assert_allclose(summary['prefix_ttt'] - policy.reward.astype(float).sum(), summary['total_ttt'], rtol=1e-6)
    refit = merge_frozen_response_replays(replays, source=__name__)
    if refit.size != plan['expected_refit_transitions']:
        raise ValueError('unexpected refit row count')
    branches = plan['branches']
    if len(branches) != 2 or len({b['name'] for b in branches}) != 2 or plan['response_workers_per_actor'] != 4:
        raise ValueError('require two branch actors with four workers each')
    if any(Path(b['name']).name != b['name'] or b['name'] in {'.', '..'} for b in branches):
        raise ValueError('branch name must be a directory name')
    _pinned_json(root / 'coverage_opportunities.json', [branch_opportunity(refit, policy, b) for b in branches])
    environment = _read_json(plan['environment_config'])
    catalog = StructuredActionCatalog.from_manifest(policy.manifest['catalog'])
    _atomic_json(root / 'status.json', {'phase': 'capturing_matched_prefix'})
    snapshots = capture_matched_prefix(environment, catalog, policy, summary['trace_segments'],
                                       [b['step'] for b in branches], root / 'prefix', stop_files=stops)
    check_stop()
    model_manifest = _read_json(Path(plan['control_model_dir']) / 'ensemble_manifest.json')
    payloads = [{
        'config': {**environment, 'snapshot': str(snapshots[b['step']]), 'reference_trace': summary['trace_segments'][0]},
        'catalog': policy.manifest['catalog'], 'directory': str(root / 'collection' / b['name']),
        'seed': plan['collection_seed'] + i, 'episode': plan['collection_episode_base'] + i,
        'response_workers': 4, 'checkpoints': model_manifest['checkpoints'], 'epsilon': 0.0,
        'first_action': b['first_action'], 'full_run': False,
        'response_equivalence_mode': policy.manifest['response_equivalence_mode'],
        'stop_file': stops[0], 'additional_stop_files': stops[1:],
    } for i, b in enumerate(branches)]
    _atomic_json(root / 'status.json', {'phase': 'collecting_targeted_branches', 'episodes': 2,
                                      'response_workers_per_actor': 4})
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_collect_actor_worker, payload): b for payload, b in zip(payloads, branches)}
        for future in as_completed(futures):
            result, branch = future.result(), futures[future]
            replay = load_frozen_response_replay(root / 'collection' / branch['name'] / 'replay.npz')
            diagnostic = verify_branch(result, replay, policy, summary, branch)
            _pinned_json(root / 'collection' / branch['name'] / 'matched_branch_diagnostic.json', diagnostic)
            print(json.dumps({'event': 'targeted_branch_complete', 'name': branch['name'], **diagnostic}), flush=True)
    check_stop()
    tails = [load_frozen_response_replay(root / 'collection' / b['name'] / 'replay.npz') for b in branches]
    targeted = merge_frozen_response_replays([refit, *tails], source=__name__)
    if targeted.size != plan['expected_targeted_transitions']:
        raise ValueError('unexpected targeted row count')
    spec = {**_read_json(plan['training_spec']), 'output_dir': str(root),
            'start_only_after_output_dir': str(previous), 'required_previous_phase': 'ablation_complete',
            'additional_lock_dirs': plan['additional_lock_dirs'], 'additional_stop_files': stops[2:],
            'control_evaluation_dir': plan['control_evaluation_dir'], 'control_response_workers': 4,
            'evaluation_episode_base': plan['evaluation_episode_base'], 'hypothesis': plan['hypothesis'],
            'response_equivalence_mode': policy.manifest['response_equivalence_mode'],
            'do_not_add_new_replay_to_this_controlled_comparison': False}
    spec['common_training'] = {**spec['common_training'], 'value_parameterization': 'finite_horizon_cost_v1'}
    spec.pop('pre_evaluation_fit_gate', None)
    variants = []
    for name, replay in (('refit_only', refit), ('targeted', targeted)):
        check_stop()
        validate_sequential_td_replay(replay, gamma=1)
        data = root / name / 'training_replay.npz'
        digest = persist_merged_replay(replay, data)
        variant_spec = {**spec, 'data': str(data), 'data_sha256': digest}
        _atomic_json(root / 'status.json', {'phase': 'training', 'variant': name, 'transitions': replay.size})
        model_dir = train_variant(data, root / name / 'model', variant_spec, .1, stop_files=stops)
        check_stop()
        models = _models(model_dir, policy.manifest['catalog_fingerprint'])
        _pinned_json(root / name / 'fit_diagnostics.json', {
            'training': replay_fit(models, replay), 'source_policy_in_sample': replay_fit(models, policy),
            'structural_checks': cost_head_checks(models, replay),
            'interpretation': 'In-sample Bellman diagnostics, not a calibration selection gate or TTT evidence.'})
        variants.append({'name': name, 'data': str(data), 'data_sha256': digest,
                         'model_dir': str(model_dir), 'conservative_alpha': .1})
    check_stop()
    spec.update(data=variants[0]['data'], data_sha256=variants[0]['data_sha256'], variants=variants)
    spec_path = root / 'evaluation_spec.json'
    _pinned_json(spec_path, spec)
    evaluate_comparison(['--config', str(spec_path)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    _configure_torch_threads(1)
    plan = _read_json(args.config)
    try:
        run(plan)
    except InterruptedError as exc:
        _atomic_json(Path(plan['output_dir']) / 'status.json', {'phase': 'paused', 'error': str(exc)})
    except Exception as exc:
        _atomic_json(Path(plan['output_dir']) / 'status.json', {'phase': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
