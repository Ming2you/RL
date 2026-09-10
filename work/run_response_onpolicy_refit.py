"""Reuse a completed policy trajectory for one versioned sequential TD refit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.run_sequential_response_ddqn import _atomic_json, validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.run_response_cql_ablation import _pinned_json, _read_json, main as evaluate_comparison
from work.run_response_reference_coverage import persist_merged_replay, train_variant


def replay_fit(models, replay):
    """Same frozen states for both models; no fixed-trajectory return labels."""
    mode = replay.manifest.get('response_equivalence_mode', 'legacy_follower_runtime_v1')
    scale = models[0].config.reward_scale
    if any(model.response_equivalence_mode != mode or model.config.gamma != 1
           or model.config.reward_scale != scale for model in models):
        raise ValueError('incompatible diagnostic model or reward contract')
    mean = np.stack([model.q_values(replay.observation, replay.response_features)
                     for model in models]).mean(axis=0, dtype=np.float64)
    following = np.stack([model.q_values(replay.next_observation, replay.next_response_features)
                          for model in models]).mean(axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(following).all():
        raise ValueError('nonfinite diagnostic Q')
    support = np.min([model.action_support_counts for model in models], axis=0)
    mask = replay.next_action_mask & (support >= models[0].config.min_action_support)[None, :]
    if not mask.any(axis=1).all():
        raise ValueError('empty supported successor mask')
    selected = mean[np.arange(replay.size), replay.action_id]
    immediate = scale * replay.reward
    residual = immediate + (1 - replay.done) * np.where(mask, following, -np.inf).max(1) - selected
    terminal = replay.done == 1
    if not terminal.any():
        raise ValueError('diagnostic requires a true terminal row')
    return {
        'rows': replay.size, 'terminal_rows': int(terminal.sum()),
        'terminal_behavior_mae_scaled': float(abs(residual[terminal]).mean()),
        'all_behavior_td_mae_scaled': float(abs(residual).mean()),
        'observed_action_immediate_reward_bound_violations': int((selected > immediate + 1e-6).sum()),
        'positive_behavior_q_rows': int((selected > 1e-6).sum()),
        'terminal_mean_q_scaled': selected[terminal].tolist(),
        'terminal_exact_target_scaled': immediate[terminal].tolist(),
    }


def fit_gate_passed(before, after, gate):
    for field, ratio in (
        ('terminal_behavior_mae_scaled', gate['terminal_mae_max_ratio']),
        ('all_behavior_td_mae_scaled', gate['all_td_mae_max_ratio']),
    ):
        if not np.isfinite([before[field], after[field], ratio]).all() or ratio <= 0:
            raise ValueError('invalid calibration gate inputs')
        if after[field] > ratio * max(before[field], 1e-6):
            return False
    return True


def _models(folder, fingerprint):
    manifest = _read_json(Path(folder) / 'ensemble_manifest.json')
    return [load_trained_response_dqn(path, expected_catalog_fingerprint=fingerprint)
            for path in manifest['checkpoints']]


def run(plan):
    root, previous = Path(plan['output_dir']), Path(plan['start_only_after_output_dir'])
    stops = [str(root / 'STOP'), str(previous / 'STOP'), *plan['additional_stop_files']]
    def check_stop():
        if any(Path(path).exists() for path in stops):
            raise InterruptedError('STOP requested')
    check_stop()
    if _read_json(previous / 'status.json')['phase'] != 'ablation_complete' or _read_json(previous / 'process.json')['state'] != 'exited':
        raise ValueError('predecessor has not completed')
    _pinned_json(root / 'refit_plan.json', plan)
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
    summary = _read_json(Path(plan['control_evaluation_dir']) / 'summary.json')
    if summary['scope'] != 'ungated_full_run' or summary['epsilon'] != 0 or summary['first_action'] is not None or summary['lcb_guard']:
        raise ValueError('new replay must come from the unforced greedy full evaluation')
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
            'control_evaluation_dir': plan['control_evaluation_dir'], 'control_response_workers': 8,
            'evaluation_episode_base': plan['evaluation_episode_base'], 'hypothesis': plan['hypothesis'],
            'response_equivalence_mode': merged.manifest['response_equivalence_mode'],
            'do_not_add_new_replay_to_this_controlled_comparison': False}
    spec['common_training'] = {**spec['common_training'], 'mask_constant_features': plan['mask_constant_features']}
    spec.pop('pre_evaluation_fit_gate', None)
    check_stop()
    _atomic_json(root / 'status.json', {'phase': 'training', 'transitions': merged.size, 'new_simulator_episodes': 0})
    model_dir = train_variant(data, root / 'onpolicy/model', spec, .1, stop_files=stops)
    check_stop()
    old_models = _models(plan['control_model_dir'], merged.manifest['catalog_fingerprint'])
    new_models = _models(model_dir, merged.manifest['catalog_fingerprint'])
    fits = {name: {'before': replay_fit(old_models, replay), 'after': replay_fit(new_models, replay)}
            for name, replay in (('base_264', base), ('latest_policy_75', policy), ('merged_training', merged))}
    passed = fit_gate_passed(fits['latest_policy_75']['before'], fits['latest_policy_75']['after'], plan['fit_gate'])
    _pinned_json(root / 'fit_comparison.json', {
        'interpretation': 'Identical frozen states for before/after. Terminal targets are exact immediate rewards; nonterminal residual uses each frozen mean-ensemble greedy successor, not the per-member training target. Post-refit states are in-sample, not held-out evidence or fixed long-horizon training labels.',
        'fits': fits, 'fit_gate_passed': passed,
    })
    if not passed:
        _atomic_json(root / 'status.json', {'phase': 'refit_calibration_failed_requires_diagnosis'})
        return
    check_stop()
    spec['variants'] = [{'name': 'onpolicy', 'conservative_alpha': .1, 'model_dir': str(model_dir)}]
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
