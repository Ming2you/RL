"""Read-only diagnostics of a completed training batch; never changes the policy."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(ROOT / ".deps-budget"), str(ROOT / "work/sdmpc_rl_budget_20260929")]

import numpy as np
import torch
from budget_runtime import DEFAULT_SNAPSHOT, read, save
from compare_runs import load_completed_run
from run_budget import file_hash, verify_pins
from td3 import TD3


def stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite diagnostic values")
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()))


def behavior_returns(rewards, terminal):
    if len(rewards) != len(terminal) or not terminal[-1]:
        raise ValueError("Need complete episodes for behavior returns")
    out = np.zeros(len(rewards))
    total = 0.
    for i in range(len(rewards) - 1, -1, -1):
        total = float(rewards[i]) + (0. if terminal[i] else total)
        out[i] = total
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    folder = args.pilot / "train"
    completed, settings, traces, schema = load_completed_run(folder, "train", [6101, 6102])
    verify_pins(DEFAULT_SNAPSHOT, settings["source_pins"])
    path = folder / "model_final.pt"
    before = file_hash(path)
    model = torch.load(path, map_location="cpu", weights_only=False)
    learner = TD3(len(schema["names"]), settings["policy_seed"])
    learner.load_state_dict(model["learner"])
    data = model["learner"]["replay"]
    obs, action = data["observations"], data["actions"]
    rewards, terminal = data["rewards"].numpy(), data["terminated"].numpy()
    rows = [row for trace in traces for row in trace]
    if len(rows) != len(rewards):
        raise ValueError("Replay and trajectory counts differ")
    np.testing.assert_allclose(rewards, [r["reward"] for r in rows], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(terminal, [r["terminated"] for r in rows])
    np.testing.assert_array_equal(action.numpy(), np.asarray([r["action_requested"] for r in rows], dtype=np.float32))
    with torch.no_grad():
        q1, q2 = [q(torch.cat((obs, action), dim=1)).ravel() for q in learner.critics]
        target = learner._target_values(data["rewards"].reshape(-1, 1),
                                       data["next_observations"], data["terminated"].reshape(-1, 1)).ravel()
        greedy = learner.actor(obs).numpy()
    actual_returns = behavior_returns(rewards, terminal)
    q = torch.minimum(q1, q2).numpy()
    episodes = []
    for summary, trace in zip(completed["episodes"], traces):
        episodes.append(dict(seed=summary["training_seed"], intervals=len(trace),
            fallback_count=summary["fallback_count"],
            fallback_reasons=dict(Counter(reason for r in trace for reason in r["fallback_reasons"])),
            nuf_clipped_count=sum(abs(r["B_raw"][1] - r["B_requested"][0][1]) > 1e-9 for r in trace),
            requested_actions={name: stats([r["action_requested"][i] for r in trace])
                               for i, name in enumerate(("NP", "NUF"))}))
    result = dict(scope="completed_training_batch_only_not_full_run_RL_acceptance",
        model_sha256=before, analysis_source_sha256=file_hash(__file__),
        replay_transitions=len(rows), true_terminals=int(terminal.sum()),
        critic_updates=learner.updates, actor_updates=learner.updates // learner.policy_delay,
        episodes=episodes, replay_rewards=stats(rewards),
        q1=stats(q1.numpy()), q2=stats(q2.numpy()),
        bellman_residual_abs=stats(abs(q - target.numpy())),
        realized_behavior_returns=stats(actual_returns),
        q_minus_behavior_return=stats(q - actual_returns),
        behavior_return_caveat="Logged future actions are exploratory and changing; these returns are NOT current-policy Q ground truth.",
        greedy_actions_at_logged_states={name: dict(**stats(greedy[:, i]),
            fraction_abs_ge_095=float((abs(greedy[:, i]) >= .95).mean()))
            for i, name in enumerate(("NP", "NUF"))},
        action_caveat="Logged training states are not the frozen policy's closed-loop state distribution.",
        performance_or_causal_claim=False)
    if file_hash(path) != before:
        raise RuntimeError("Model changed during read-only analysis")
    verify_pins(DEFAULT_SNAPSHOT, settings["source_pins"])
    save(args.output, result)
    print("TRAINING_DIAGNOSTICS_SAVED", len(rows), learner.updates)


if __name__ == "__main__":
    main()
