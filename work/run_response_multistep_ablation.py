"""Compare one-step and greedy-path DDQN backups on the same reached-state batch."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.response_dqn import build_sequential_links
from rl_leader.run_sequential_response_ddqn import _atomic_json, validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.run_response_cost_refinement import path_rankings
from work.run_response_cql_ablation import _pinned_json, _read_json, main as evaluate_comparison, variant_training_spec
from work.run_response_onpolicy_refit import _models, replay_fit
from work.run_response_reference_coverage import persist_merged_replay, train_variant
from work.run_response_value_head_ablation import cost_head_checks


def verify_policy_source(replay, summary, path):
    validate_complete_episode(replay)
    if replay.size != 75 or not np.array_equal(replay.control_step, np.arange(75)):
        raise ValueError('new source must contain all 75 control steps')
    for key, expected in {'scope': 'ungated_full_run', 'epsilon': 0., 'first_action': None,
                          'lcb_guard': False, 'response_preview': True}.items():
        if summary.get(key) != expected:
            raise ValueError(f'new source is not an unforced full evaluation: {key}')
    if Path(summary['replay']).resolve() != Path(path).resolve():
        raise ValueError('source summary points to another replay')
    np.testing.assert_allclose(summary['prefix_ttt'] - replay.reward.astype(float).sum(),
                               summary['total_ttt'], rtol=1e-6)


def greedy_path_coverage(models, replay, horizon=5):
    links = build_sequential_links(replay)
    result = []
    for model in models:
        q = model.q_values(replay.observation, replay.response_features)
        mask = replay.action_mask & (model.action_support_counts >= model.config.min_action_support)[None, :]
        if not np.isfinite(q).all() or not mask.any(axis=1).all():
            raise ValueError('invalid greedy path diagnostics')
        greedy = np.where(mask, q, -np.inf).argmax(1)
        lengths = []
        for index in range(replay.size):
            length, following = 1, links[index]
            while length < horizon and following >= 0 and replay.action_id[following] == greedy[following]:
                length += 1
                following = links[following]
            lengths.append(length)
        result.append({'seed': model.seed, 'length_counts': np.bincount(lengths, minlength=horizon + 1).tolist()})
    return {'maximum_horizon': horizon, 'verified_links': int((links >= 0).sum()), 'members': result,
            'interpretation': 'Endpoint online-member greedy agreement on unweighted replay rows, not actual sampled training horizons or a performance gate.'}


def run(plan):
    root, previous = Path(plan['output_dir']), Path(plan['start_only_after_output_dir'])
    stops = [str(root / 'STOP'), str(previous / 'STOP'), *plan['additional_stop_files']]
    def check_stop():
        if any(Path(path).exists() for path in stops):
            raise InterruptedError('STOP requested')
    check_stop()
    if _read_json(previous / 'status.json')['phase'] != 'ablation_complete' or _read_json(previous / 'process.json')['state'] != 'exited':
        raise ValueError('predecessor has not completed')
    _pinned_json(root / 'multistep_plan.json', plan)
    _pinned_json(root / 'implementation_hashes.json', {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in plan['implementation_files']
    })
    for path, digest in plan['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest.lower():
            raise ValueError(f'input hash changed: {path}')
    base = load_frozen_response_replay(plan['base_replay'])
    validate_sequential_td_replay(base, gamma=1)
    policies = []
    for folder in plan['policy_evaluation_dirs']:
        path = Path(folder) / 'replay.npz'
        replay = load_frozen_response_replay(path)
        verify_policy_source(replay, _read_json(Path(folder) / 'summary.json'), path)
        policies.append(replay)
    if len(policies) != 2:
        raise ValueError('require exactly the two completed policy evaluations')
    merged = merge_frozen_response_replays([base, *policies], source=__name__)
    validate_sequential_td_replay(merged, gamma=1)
    if merged.size != plan['expected_transitions']:
        raise ValueError('unexpected merged row count')
    links = build_sequential_links(merged)
    data = root / 'training_replay.npz'
    digest = persist_merged_replay(merged, data)
    _pinned_json(root / 'data_summary.json', {'base_rows': base.size, 'new_rows': [r.size for r in policies],
                 'total_rows': merged.size, 'terminals': int(merged.done.sum()), 'verified_links': int((links >= 0).sum()),
                 'action_counts': merged.action_support_counts().tolist(), 'data_sha256': digest,
                 'new_simulator_collection_episodes': 0})
    spec = {**_read_json(plan['training_spec']), 'output_dir': str(root),
            'start_only_after_output_dir': str(previous), 'required_previous_phase': 'ablation_complete',
            'additional_lock_dirs': plan['additional_lock_dirs'], 'additional_stop_files': stops[2:],
            'data': str(data), 'data_sha256': digest, 'control_evaluation_dir': plan['control_evaluation_dir'],
            'control_response_workers': 4, 'evaluation_episode_base': plan['evaluation_episode_base'],
            'hypothesis': plan['hypothesis'], 'response_equivalence_mode': merged.manifest['response_equivalence_mode'],
            'do_not_add_new_replay_to_this_controlled_comparison': True}
    spec['common_training'] = {**spec['common_training'], 'backup_horizon': 1}
    common = spec['common_training']
    if common['value_parameterization'] != 'finite_horizon_cost_v1' or common['gradient_steps'] != 24000 or common['group_resampling']:
        raise ValueError('unexpected paired cost-head training recipe')
    variants = plan['variants']
    if len(variants) != 2 or {v['backup_horizon'] for v in variants} != {1, 5} or any(v['conservative_alpha'] != .01 for v in variants):
        raise ValueError('require the prespecified horizons with fixed CQL .01')
    if len({v['name'] for v in variants}) != 2:
        raise ValueError('variant names must be unique')
    spec.pop('pre_evaluation_fit_gate', None)
    source = load_frozen_response_replay(plan['source_policy_replay'])
    branch = load_frozen_response_replay(plan['quadratic_branch_replay'])
    for diagnostic in (source, branch):
        validate_complete_episode(diagnostic)
        if any(diagnostic.manifest.get(k) != merged.manifest.get(k) for k in
               ('catalog_fingerprint', 'experiment_contract_sha256', 'response_equivalence_mode', 'scenario', 't_total_sec')):
            raise ValueError('diagnostic contract mismatch')
    trained = []
    for variant in variants:
        check_stop()
        name = variant['name']
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('variant must be a directory name')
        _atomic_json(root / 'status.json', {'phase': 'training', 'variant': name, 'transitions': merged.size,
                                          'backup_horizon': variant['backup_horizon'], 'gradient_steps_per_member': 24000})
        started = time.perf_counter()
        model_dir = train_variant(data, root / name / 'model', variant_training_spec(spec, variant), .01, stop_files=stops)
        elapsed = time.perf_counter() - started
        check_stop()
        models = _models(model_dir, merged.manifest['catalog_fingerprint'])
        report = {label: replay_fit(models, replay) for label, replay in
                  [('base_706', base), ('latest_high_cql_75', policies[0]), ('latest_low_cql_75', policies[1]), ('merged_856', merged)]}
        report['path_rankings'] = path_rankings(models, source, branch)
        report['greedy_path_coverage'] = greedy_path_coverage(models, merged)
        report['structural_checks'] = cost_head_checks(models, merged)
        report['loss_blocks'] = [{'seed': m.seed, 'block_size': 1000,
             'means': [float(np.mean(m.training_losses[k:k + 1000])) for k in range(0, len(m.training_losses), 1000)]} for m in models]
        _pinned_json(root / name / 'fit_diagnostics.json', report)
        timing = root / name / 'training_call_timing.json'
        if not timing.exists():
            _atomic_json(timing, {'wall_seconds': elapsed, 'model_dir': str(model_dir),
                                 'note': 'Measured training-helper call; resumed completed checkpoints may be reused.'})
        trained.append({**variant, 'model_dir': str(model_dir)})
    check_stop()
    spec['variants'] = trained
    path = root / 'evaluation_spec.json'
    _pinned_json(path, spec)
    evaluate_comparison(['--config', str(path)])


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
