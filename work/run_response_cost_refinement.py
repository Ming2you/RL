"""Separate fitting duration and CQL strength on the same frozen cost-head data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.run_sequential_response_ddqn import _atomic_json, validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.run_response_cql_ablation import _pinned_json, _read_json, main as evaluate_comparison
from work.run_response_onpolicy_refit import _models, replay_fit
from work.run_response_reference_coverage import train_variant
from work.run_response_value_head_ablation import cost_head_checks


def verify_long_control_prefix(before, after):
    if len(before) != len(after):
        raise ValueError('control ensemble size changed')
    for old, new in zip(before, after):
        if old.seed != new.seed or asdict(new.config) != {**asdict(old.config), 'gradient_steps': 24000}:
            raise ValueError('long control changed more than optimization duration')
        if len(old.training_losses) != 6000 or len(new.training_losses) != 24000:
            raise ValueError('unexpected completed training update count')
        np.testing.assert_array_equal(new.training_losses[:6000], old.training_losses)
        for name, value in vars(old.normalizer).items():
            np.testing.assert_array_equal(getattr(new.normalizer, name), value)
        np.testing.assert_array_equal(new.candidate_features, old.candidate_features)
        np.testing.assert_array_equal(new.action_support_counts, old.action_support_counts)
        if new.value_contract != old.value_contract:
            raise ValueError('long control value contract changed')
    return {'matched_loss_prefix_steps': 6000, 'paired_seeds': [m.seed for m in before],
            'interpretation': 'Exact stored loss-history prefix, inputs, normalization and recipe agreement. No intermediate weight checkpoint was captured, so this is not a direct 6k weight-equality check.'}


def path_rankings(models, source, branch):
    if source.control_step[0] != 0 or branch.control_step[0] != 1 or branch.action_id[0] != 6:
        raise ValueError('diagnostic requires the step1 quadratic branch and its source policy')
    np.testing.assert_array_equal(source.observation[1], branch.observation[0])
    np.testing.assert_array_equal(source.response_features[1], branch.response_features[0])
    scale = models[0].config.reward_scale
    if any(model.config.gamma != 1 or model.config.reward_scale != scale for model in models):
        raise ValueError('incompatible ranking reward contract')
    q = np.mean([m.q_values(source.observation, source.response_features) for m in models], axis=0, dtype=np.float64)
    next_q = np.mean([m.q_values(branch.next_observation[:1], branch.next_response_features[:1]) for m in models], axis=0, dtype=np.float64)
    if not np.isfinite(q).all() or not np.isfinite(next_q).all():
        raise ValueError('nonfinite ranking Q')
    support = np.min([m.action_support_counts for m in models], axis=0) >= models[0].config.min_action_support
    mask = branch.next_action_mask[0] & support
    if not mask.any():
        raise ValueError('empty supported successor set')
    return {'source_step0_q5_minus_q0_scaled': float(q[0, 5] - q[0, 0]),
            'source_step1_q6_minus_q0_scaled': float(q[1, 6] - q[1, 0]),
            'quadratic_first_q6_scaled': float(q[1, 6]),
            'quadratic_first_ensemble_greedy_td_target_scaled': float(scale * branch.reward[0] + (1 - branch.done[0]) * next_q[0, mask].max()),
            'quadratic_recorded_tail_return_scaled': float(scale * branch.reward.astype(float).sum()),
            'interpretation': 'Frozen-state ranking and ensemble-greedy residual diagnostics, not per-member DDQN targets. Recorded tail return belongs to the older recourse policy, is not optimal Q and is never a training label.'}


def run(plan):
    root, previous = Path(plan['output_dir']), Path(plan['start_only_after_output_dir'])
    stops = [str(root / 'STOP'), str(previous / 'STOP'), *plan['additional_stop_files']]
    def check_stop():
        if any(Path(path).exists() for path in stops):
            raise InterruptedError('STOP requested')
    check_stop()
    if _read_json(previous / 'status.json')['phase'] != 'ablation_complete' or _read_json(previous / 'process.json')['state'] != 'exited':
        raise ValueError('predecessor has not completed')
    _pinned_json(root / 'refinement_plan.json', plan)
    _pinned_json(root / 'implementation_hashes.json', {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in plan['implementation_files']
    })
    for path, digest in plan['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest.lower():
            raise ValueError(f'input hash changed: {path}')
    data = Path(plan['data'])
    training = load_frozen_response_replay(data)
    validate_sequential_td_replay(training, gamma=1)
    if training.size != plan['expected_transitions']:
        raise ValueError('frozen training row count changed')
    diagnostics = {name: load_frozen_response_replay(path) for name, path in plan['diagnostic_replays'].items()}
    for replay in diagnostics.values():
        validate_complete_episode(replay)
        for field in ('catalog_fingerprint', 'experiment_contract_sha256', 'response_equivalence_mode', 'scenario', 't_total_sec'):
            if replay.manifest.get(field) != training.manifest.get(field):
                raise ValueError(f'diagnostic replay contract mismatch: {field}')
    old_models = _models(plan['control_model_dir'], training.manifest['catalog_fingerprint'])
    if any(m.config.value_parameterization != 'finite_horizon_cost_v1' or m.config.gradient_steps != 6000
           or m.config.conservative_alpha != .1 for m in old_models):
        raise ValueError('unexpected short-training control recipe')
    spec = {**_read_json(plan['training_spec']), 'output_dir': str(root),
            'start_only_after_output_dir': str(previous), 'required_previous_phase': 'ablation_complete',
            'additional_lock_dirs': plan['additional_lock_dirs'], 'additional_stop_files': stops[2:],
            'data': str(data), 'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
            'control_evaluation_dir': plan['control_evaluation_dir'], 'control_response_workers': 4,
            'evaluation_episode_base': plan['evaluation_episode_base'], 'hypothesis': plan['hypothesis'],
            'response_equivalence_mode': training.manifest['response_equivalence_mode'],
            'do_not_add_new_replay_to_this_controlled_comparison': True,
            'reuse_control_without_retraining': True}
    spec['common_training'] = {**spec['common_training'], 'gradient_steps': plan['gradient_steps']}
    if spec['common_training']['value_parameterization'] != 'finite_horizon_cost_v1' or plan['gradient_steps'] != 24000:
        raise ValueError('unexpected long-training cost-head recipe')
    variants = plan['variants']
    if len(variants) != 2 or {v['conservative_alpha'] for v in variants} != {.1, .01} or len({v['name'] for v in variants}) != 2:
        raise ValueError('require the prespecified paired CQL strengths')
    spec.pop('pre_evaluation_fit_gate', None)
    def report(models):
        result = {name: replay_fit(models, replay) for name, replay in {'training_in_sample': training, **diagnostics}.items()}
        result['path_rankings'] = path_rankings(models, diagnostics['source_policy_in_training'], diagnostics['quadratic_branch_in_training'])
        result['structural_checks'] = cost_head_checks(models, training)
        result['loss_blocks'] = [{'seed': m.seed, 'block_size': 1000,
                                 'means': [float(np.mean(m.training_losses[k:k + 1000])) for k in range(0, len(m.training_losses), 1000)]}
                                for m in models]
        return result
    _pinned_json(root / 'short_control_diagnostics.json', report(old_models))
    trained = []
    for variant in variants:
        check_stop()
        name = variant['name']
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('variant must be a directory name')
        _atomic_json(root / 'status.json', {'phase': 'training', 'variant': name, 'transitions': training.size,
                                          'new_simulator_collection_episodes': 0, 'gradient_steps_per_member': plan['gradient_steps']})
        folder = train_variant(data, root / name / 'model', spec, variant['conservative_alpha'], stop_files=stops)
        check_stop()
        models = _models(folder, training.manifest['catalog_fingerprint'])
        if variant['conservative_alpha'] == .1:
            _pinned_json(root / name / 'loss_prefix_verification.json', verify_long_control_prefix(old_models, models))
        _pinned_json(root / name / 'fit_diagnostics.json', report(models))
        trained.append({**variant, 'model_dir': str(folder)})
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
