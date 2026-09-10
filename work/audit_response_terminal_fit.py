"""Diagnose the exact terminal boundary without installing trajectory labels."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.run_response_cql_ablation import _pinned_json, _read_json


def terminal_fit(model_dir):
    _configure_torch_threads(1)
    manifest_path = Path(model_dir) / 'ensemble_manifest.json'
    manifest = _read_json(manifest_path)
    data = Path(manifest['source_dataset'])
    replay = load_frozen_response_replay(data)
    validate_sequential_td_replay(replay, gamma=1)
    paths = [Path(path) for path in manifest['checkpoints']]
    models = [load_trained_response_dqn(path, expected_catalog_fingerprint=replay.manifest['catalog_fingerprint'])
              for path in paths]
    scale = models[0].config.reward_scale
    if any(model.config.gamma != 1 or model.config.reward_scale != scale for model in models):
        raise ValueError('inconsistent reward objective')
    q = np.stack([model.q_values(replay.observation, replay.response_features) for model in models])
    next_q = np.stack([model.q_values(replay.next_observation, replay.next_response_features) for model in models])
    if not np.isfinite(q).all() or not np.isfinite(next_q).all():
        raise ValueError('nonfinite Q')
    mean = q.mean(axis=0, dtype=np.float64)
    next_mean = next_q.mean(axis=0, dtype=np.float64)
    support = np.min([model.action_support_counts for model in models], axis=0)
    next_mask = replay.next_action_mask & (support >= models[0].config.min_action_support)[None, :]
    if not next_mask.any(axis=1).all():
        raise ValueError('empty successor mask')
    selected = mean[np.arange(replay.size), replay.action_id]
    residual = scale * replay.reward + (1 - replay.done) * np.where(next_mask, next_mean, -np.inf).max(1) - selected
    terminal = replay.done == 1
    if not terminal.any():
        raise ValueError('no true terminal observations')
    chosen = np.where(replay.action_mask, mean, -np.inf).argmax(axis=1)
    return {
        'format': 'response_terminal_boundary_fit_v1',
        'interpretation': 'Terminal behavior Q has an exact observed immediate-reward target. Nonterminal residual uses the frozen mean-ensemble greedy successor, not the per-member training target or a fixed long-horizon label. Training fit is not held-out performance.',
        'model_dir': str(model_dir), 'training_rows': replay.size,
        'terminal_rows': int(terminal.sum()), 'reward_scale': scale,
        'terminal_batch_fraction': models[0].config.terminal_batch_fraction,
        'terminal_behavior_mae_scaled': float(abs(residual[terminal]).mean()),
        'all_behavior_td_mae_scaled': float(abs(residual).mean()),
        'nonterminal_behavior_td_mae_scaled': float(abs(residual[~terminal]).mean()),
        'greedy_training_action_counts': np.bincount(chosen, minlength=replay.action_count).tolist(),
        'terminal_rows_detail': [
            {'episode': int(replay.episode[i]), 'step': int(replay.control_step[i]),
             'action': int(replay.action_id[i]), 'expected_q': float(scale * replay.reward[i]),
             'mean_q': float(selected[i]), 'member_q': q[:, i, replay.action_id[i]].astype(float).tolist()}
            for i in np.flatnonzero(terminal)
        ],
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), data, manifest_path, *paths]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = terminal_fit(args.model_dir)
    _pinned_json(args.out, result)
    print({key: result[key] for key in ('training_rows', 'terminal_rows', 'terminal_behavior_mae_scaled',
                                      'all_behavior_td_mae_scaled', 'nonterminal_behavior_td_mae_scaled')})


if __name__ == '__main__':
    main()
