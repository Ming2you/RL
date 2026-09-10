"""Compare free and finite-horizon cost Q heads on identical sequential rows."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.run_sequential_response_ddqn import _atomic_json, validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.run_response_cql_ablation import (
    _pinned_json, _read_json, main as evaluate_comparison,
    validate_checkpoint_value_heads, variant_training_spec,
)
from work.run_response_onpolicy_refit import _models, replay_fit
from work.run_response_reference_coverage import persist_merged_replay, train_variant


def cost_head_checks(models, replay):
    """Correctness checks, not a calibration-based policy selection gate."""
    current = np.stack([model.q_values(replay.observation, replay.response_features) for model in models])
    following = np.stack([model.q_values(replay.next_observation, replay.next_response_features) for model in models])
    if not np.isfinite(current).all() or not np.isfinite(following).all():
        raise ValueError('nonfinite value-head Q')
    terminal = replay.done == 1
    if not terminal.any():
        raise ValueError('value-head check requires terminal rows')
    if current.max() > 0 or following.max() > 0 or np.any(following[:, terminal] != 0):
        raise ValueError('finite-horizon cost head violates nonpositive/terminal-zero contract')
    return {'maximum_current_q_scaled': float(current.max()),
            'maximum_next_q_scaled': float(following.max()),
            'terminal_successor_q_exactly_zero': True,
            'interpretation': 'Guaranteed by the head, not evidence of improved action ranking.'}


def run(plan):
    root, previous = Path(plan['output_dir']), Path(plan['start_only_after_output_dir'])
    stops = [str(root / 'STOP'), str(previous / 'STOP'), *plan['additional_stop_files']]
    def check_stop():
        if any(Path(path).exists() for path in stops):
            raise InterruptedError('STOP requested')
    check_stop()
    if _read_json(previous / 'status.json')['phase'] != 'ablation_complete' or _read_json(previous / 'process.json')['state'] != 'exited':
        raise ValueError('predecessor has not completed')
    _pinned_json(root / 'ablation_plan.json', plan)
    _pinned_json(root / 'implementation_hashes.json', {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in plan['implementation_files']
    })
    for path, digest in plan['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest.lower():
            raise ValueError(f'input hash changed: {path}')
    base = load_frozen_response_replay(plan['base_replay'])
    policy = load_frozen_response_replay(plan['policy_replay'])
    validate_sequential_td_replay(base, gamma=1)
    validate_complete_episode(policy)
    np.testing.assert_array_equal(policy.control_step, np.arange(75))
    summary = _read_json(Path(plan['control_evaluation_dir']) / 'summary.json')
    for field, expected in {'scope': 'ungated_full_run', 'epsilon': 0,
                            'first_action': None, 'lcb_guard': False, 'response_preview': True,
                            'start_control_step': 0, 'end_control_step': 74, 'transitions': 75}.items():
        if summary.get(field) != expected:
            raise ValueError(f'policy replay is not an unforced full evaluation: {field}')
    if Path(summary['replay']).resolve() != Path(plan['policy_replay']).resolve():
        raise ValueError('policy summary points to a different replay')
    np.testing.assert_allclose(summary['prefix_ttt'] - policy.reward.astype(float).sum(), summary['total_ttt'], rtol=1e-6)
    merged = merge_frozen_response_replays([base, policy], source=__name__)
    if merged.size != plan['expected_transitions']:
        raise ValueError('unexpected merged transition count')
    validate_sequential_td_replay(merged, gamma=1)
    data = root / 'training_replay.npz'
    digest = persist_merged_replay(merged, data)
    spec = {**_read_json(plan['training_spec']), 'output_dir': str(root),
            'start_only_after_output_dir': str(previous), 'required_previous_phase': 'ablation_complete',
            'data': str(data), 'data_sha256': digest, 'additional_stop_files': stops[2:],
            'additional_lock_dirs': plan['additional_lock_dirs'],
            'control_evaluation_dir': plan['control_evaluation_dir'], 'control_response_workers': 8,
            'evaluation_episode_base': plan['evaluation_episode_base'], 'hypothesis': plan['hypothesis'],
            'response_equivalence_mode': merged.manifest['response_equivalence_mode'],
            'do_not_add_new_replay_to_this_controlled_comparison': True,
            'reuse_control_without_retraining': False}
    spec.pop('pre_evaluation_fit_gate', None)
    if spec['common_training']['gamma'] != 1 or not spec['common_training']['mask_constant_features']:
        raise ValueError('unexpected common training contract')
    variants = plan['variants']
    if len(variants) != 2 or {v['value_parameterization'] for v in variants} != {'free_q', 'finite_horizon_cost_v1'}:
        raise ValueError('comparison requires exactly the two declared value heads')
    if len({v['name'] for v in variants}) != 2 or any(v['conservative_alpha'] != .1 for v in variants):
        raise ValueError('comparison must isolate value head, with unique variant names')
    reports, trained = {}, []
    for variant in variants:
        check_stop()
        name = variant['name']
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('variant must be a directory name')
        variant_spec = variant_training_spec(spec, variant)
        _atomic_json(root / 'status.json', {'phase': 'training', 'variant': name,
                                          'transitions': merged.size, 'new_simulator_episodes': 0})
        model_dir = train_variant(data, root / name / 'model', variant_spec, .1, stop_files=stops)
        check_stop()
        models = _models(model_dir, merged.manifest['catalog_fingerprint'])
        validate_checkpoint_value_heads(models, variant_spec)
        report = {label: replay_fit(models, replay) for label, replay in
                  (('previous_training', base), ('latest_policy_75', policy), ('merged_training', merged))}
        if variant['value_parameterization'] == 'finite_horizon_cost_v1':
            report['structural_checks'] = cost_head_checks(models, merged)
        reports[name] = report
        trained.append({**variant, 'model_dir': str(model_dir)})
        _pinned_json(root / name / 'fit_diagnostics.json', report)
    _pinned_json(root / 'fit_comparison.json', {
        'interpretation': 'Both heads use identical frozen sequential rows and paired seeds. These are in-sample ensemble-greedy TD diagnostics, not per-member targets, fixed-return labels, a policy selection gate, or TTT evidence.',
        'fits': reports,
    })
    check_stop()
    spec['variants'] = trained
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
