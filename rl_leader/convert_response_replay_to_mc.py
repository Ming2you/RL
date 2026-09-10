"""Convert sequential response-DQN replay rows to terminal Monte Carlo targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from rl_leader.response_dqn_data import FrozenResponseReplay, load_frozen_response_replay


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in str(value).split(",") if item)


def _return_to_go(
    rewards: np.ndarray,
    done: np.ndarray,
    groups: np.ndarray,
    episodes: np.ndarray,
    *,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    returns = np.zeros_like(rewards, dtype=np.float32)
    option_steps = np.ones(rewards.shape[0], dtype=np.int64)
    keys = np.asarray([
        f"{group}:{int(episode)}" for group, episode in zip(groups, episodes)
    ])
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        running = 0.0
        steps = 0
        for index in indices[::-1]:
            if float(done[index]) >= 0.5:
                running = 0.0
                steps = 0
            running = float(rewards[index]) + float(gamma) * running
            steps += 1
            returns[index] = running
            option_steps[index] = steps
    return returns, option_steps


def convert_replay_to_mc_targets(
    replay: FrozenResponseReplay,
    *,
    gamma: float = 1.0,
    control_steps: Iterable[int] | None = None,
    source: str = "rl_leader.convert_response_replay_to_mc",
) -> FrozenResponseReplay:
    """Return a replay whose rewards are terminal Monte Carlo return-to-go values."""
    replay.validate()
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    rewards, option_steps = _return_to_go(
        np.asarray(replay.reward, dtype=np.float32),
        np.asarray(replay.done, dtype=np.float32),
        np.asarray(replay.event_group).astype(str),
        np.asarray(replay.episode, dtype=np.int64),
        gamma=float(gamma),
    )
    keep = np.ones(replay.size, dtype=bool)
    if control_steps is not None:
        allowed = {int(step) for step in control_steps}
        keep = np.asarray([
            int(step) in allowed for step in replay.control_step
        ], dtype=bool)
    if not np.any(keep):
        raise ValueError("control-step filter removed every replay row")

    next_action_mask = np.zeros_like(replay.next_action_mask, dtype=bool)
    next_action_mask[:, 0] = True
    next_response_features = np.zeros_like(replay.next_response_features)
    manifest = dict(replay.manifest)
    manifest.update({
        "transition_count": int(np.count_nonzero(keep)),
        "source": str(source),
        "reward_semantics": "terminal_monte_carlo_return_to_go",
        "mc_gamma": float(gamma),
        "selected_control_steps": (
            None if control_steps is None else sorted({int(step) for step in control_steps})
        ),
        "done_semantics": "monte_carlo_targets_are_terminal",
    })
    return FrozenResponseReplay(
        observation=np.asarray(replay.observation[keep], dtype=np.float32),
        action_id=np.asarray(replay.action_id[keep], dtype=np.int64),
        reward=np.asarray(rewards[keep], dtype=np.float32),
        next_observation=np.asarray(replay.next_observation[keep], dtype=np.float32),
        done=np.ones(int(np.count_nonzero(keep)), dtype=np.float32),
        option_steps=np.asarray(option_steps[keep], dtype=np.int64),
        action_mask=np.asarray(replay.action_mask[keep], dtype=bool),
        next_action_mask=np.asarray(next_action_mask[keep], dtype=bool),
        response_features=np.asarray(replay.response_features[keep], dtype=np.float32),
        next_response_features=np.asarray(next_response_features[keep], dtype=np.float32),
        event_group=np.asarray(replay.event_group[keep]).astype(str),
        episode=np.asarray(replay.episode[keep], dtype=np.int64),
        control_step=np.asarray(replay.control_step[keep], dtype=np.int64),
        manifest=manifest,
    ).validate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--control-steps", default="")
    args = parser.parse_args()
    replay = load_frozen_response_replay(args.input)
    steps = _parse_csv(args.control_steps, int) if args.control_steps else None
    converted = convert_replay_to_mc_targets(
        replay,
        gamma=args.gamma,
        control_steps=steps,
    )
    converted.save(args.out)
    print(json.dumps({
        "output": str(args.out),
        "input": str(args.input),
        "gamma": args.gamma,
        "control_steps": (
            None if steps is None else list(map(int, steps))
        ),
        "transitions": converted.size,
        "action_support_counts": converted.action_support_counts().tolist(),
        "reward_min": float(np.min(converted.reward)),
        "reward_max": float(np.max(converted.reward)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
