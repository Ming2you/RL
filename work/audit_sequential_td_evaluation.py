"""Read-only policy calibration audit of one completed sequential evaluation.

Realized returns describe this frozen ensemble policy, not every possible
future recovery policy. They are diagnostic observations, not training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.run_sequential_response_ddqn import validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay


def audit(evaluation_dir: Path, model_dir: Path, baseline_path: Path) -> dict:
    _configure_torch_threads(1)
    summary_path = evaluation_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest_path = model_dir / "ensemble_manifest.json"
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_path = Path(model_manifest["source_dataset"])
    replay_path = evaluation_dir / "replay.npz"
    replay = load_frozen_response_replay(replay_path)
    training = load_frozen_response_replay(training_path)
    validate_complete_episode(replay)
    validate_sequential_td_replay(training, gamma=1.0)
    for field in ("experiment_contract_sha256", "scenario", "t_total_sec"):
        if not baseline.get(field) or replay.manifest[field] != baseline[field]:
            raise ValueError(f"baseline evaluation contract mismatch: {field}")
        if training.manifest[field] != replay.manifest[field]:
            raise ValueError(f"training evaluation contract mismatch: {field}")
    if summary["epsilon"] != 0 or summary["first_action"] is not None or summary["lcb_guard"]:
        raise ValueError("audit requires unforced greedy-policy evaluation")
    if summary["catalog_fingerprint"] != training.manifest["catalog_fingerprint"]:
        raise ValueError("training evaluation catalog mismatch")
    np.testing.assert_allclose(
        summary["prefix_ttt"] - replay.reward.astype(np.float64).sum(),
        summary["total_ttt"], rtol=1e-6,
    )
    paths = [Path(path) for path in model_manifest["checkpoints"]]
    models = [load_trained_response_dqn(
        path, expected_catalog_fingerprint=replay.manifest["catalog_fingerprint"],
    ) for path in paths]
    scale = models[0].config.reward_scale
    if any(model.config.gamma != 1.0 or model.config.reward_scale != scale for model in models):
        raise ValueError("audit requires common reward scale and gamma=1")
    q = np.stack([model.q_values(replay.observation, replay.response_features) for model in models])
    if not np.isfinite(q).all():
        raise ValueError("nonfinite Q output")
    mean_q = q.mean(axis=0, dtype=np.float64)
    support = np.min(np.stack([model.action_support_counts for model in models]), axis=0)
    valid = replay.action_mask & (support >= models[0].config.min_action_support)[None, :]
    valid[:, 0] = True
    greedy = np.where(valid, mean_q, -np.inf).argmax(axis=1)
    np.testing.assert_array_equal(greedy, replay.action_id)
    selected_q = mean_q[np.arange(replay.size), replay.action_id]
    # Later resume segments supersede trace tails not committed to checkpoint.
    traces = {}
    trace_paths = [Path(path) for path in summary["trace_segments"]]
    for path in trace_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            traces[int(row["control_step"])] = row
    ordered = [traces[int(step)] for step in replay.control_step]
    np.testing.assert_array_equal([row["selected_action_id"] for row in ordered], replay.action_id)
    np.testing.assert_allclose([row["q_selected"] for row in ordered], selected_q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose([row["interval_reward"] for row in ordered], replay.reward, rtol=1e-6)
    remaining_return = np.cumsum(replay.reward[::-1], dtype=np.float64)[::-1]
    cost_underprediction = (selected_q / scale) - remaining_return
    if np.any(replay.reward > 1e-6):
        raise ValueError("negative-TTT reward must not be positive")
    immediate_bound_violation = selected_q > scale * replay.reward + 1e-6
    next_policy_q = np.concatenate((selected_q[1:], [0.0]))
    policy_td_error = scale * replay.reward + next_policy_q - selected_q
    np.testing.assert_allclose(
        policy_td_error.sum(), scale * remaining_return[0] - selected_q[0], atol=1e-5,
    )
    observation_scale = models[0].normalizer.observation_scale.astype(np.float64)
    actions = []
    for action in replay.manifest["catalog"]["actions"]:
        action_id = int(action["action_id"])
        training_steps = training.control_step[training.action_id == action_id]
        evaluation_steps = replay.control_step[replay.action_id == action_id]
        actions.append({
            "action_id": action_id, "key": action["key"],
            "training_count": int(training_steps.size),
            "training_control_steps": training_steps.astype(int).tolist(),
            "evaluation_count": int(evaluation_steps.size),
            "evaluation_control_steps": evaluation_steps.astype(int).tolist(),
        })
    rows = []
    for index, step in enumerate(replay.control_step):
        nearby = training.action_id[abs(training.control_step - step) <= 3]
        same_step = training.observation[training.control_step == step]
        nearest_rms = None
        if len(same_step):
            distances = ((same_step - replay.observation[index]) / observation_scale) ** 2
            nearest_rms = float(np.sqrt(distances.mean(axis=1)).min())
        rows.append({
            "control_step": int(step), "action_id": int(replay.action_id[index]),
            "selected_q_scaled": float(selected_q[index]),
            "realized_remaining_return_scaled": float(remaining_return[index] * scale),
            "remaining_ttt_underprediction": float(cost_underprediction[index]),
            "violates_nonpositive_future_reward_bound": bool(immediate_bound_violation[index]),
            "frozen_policy_td_error_scaled": float(policy_td_error[index]),
            "selected_q_ensemble_std_scaled": float(q[:, index, replay.action_id[index]].std()),
            "same_action_training_count_within_3_steps": int((nearby == replay.action_id[index]).sum()),
            "nearest_same_step_observation_standardized_rms": nearest_rms,
        })
    source_paths = [Path(__file__), summary_path, replay_path, training_path, manifest_path, baseline_path, *paths, *trace_paths]
    return {
        "format": "sequential_td_frozen_policy_calibration_v1",
        "scope": summary["scope"], "total_ttt": summary["total_ttt"],
        "pstack_total_ttt": baseline["total_ttt"],
        "pstack_improvement_percent": 100 * (1 - summary["total_ttt"] / baseline["total_ttt"]),
        "terminal_inventory": summary["terminal_inventory"],
        "pstack_terminal_inventory": baseline["terminal_inventory"],
        "evaluation_wall_seconds": summary["wall_seconds"],
        "training_transitions": training.size, "evaluation_transitions": replay.size,
        "catalog_fingerprint": replay.manifest["catalog_fingerprint"],
        "experiment_contract_sha256": replay.manifest["experiment_contract_sha256"],
        "reward_scale": scale, "gamma": 1.0,
        "initial_predicted_remaining_ttt": float(-selected_q[0] / scale),
        "initial_realized_remaining_ttt": float(-remaining_return[0]),
        "mean_remaining_ttt_underprediction": float(cost_underprediction.mean()),
        "remaining_ttt_underprediction_positive_rows": int((cost_underprediction > 0).sum()),
        "positive_selected_q_rows": int((selected_q > 1e-6).sum()),
        "nonpositive_future_reward_bound_violations": int(immediate_bound_violation.sum()),
        "mean_absolute_frozen_policy_td_error_scaled": float(np.abs(policy_td_error).mean()),
        "validity_gate_failed_rows": sum(not row["validity_gate_pass"] for row in ordered),
        "minimum_distinct_responses": min(row["unique_follower_response_count"] for row in ordered),
        "actions": actions, "rows": rows,
        "interpretation": (
            "Positive underprediction means mean Q forecasts less remaining TTT than this "
            "frozen greedy ensemble actually incurs. This flags calibration risk, but is not "
            "a proof that every alternative future recovery policy fails. Temporal support "
            "and nearest observed states are descriptive coverage proxies, not causal tests. "
            "For nonpositive rewards, true Q cannot exceed the selected action's immediate "
            "reward regardless of the future recovery policy; bound violations flag invalid "
            "value estimates, but do not alone identify why they arose. "
            "No replay is changed and no return is installed as a training label."
        ),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite an existing diagnostic")
    result = audit(args.evaluation_dir, args.model_dir, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "actions", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
