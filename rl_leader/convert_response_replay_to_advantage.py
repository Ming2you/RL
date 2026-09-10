"""Convert paired response-DQN replay rows to action0-relative advantages."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_leader.response_dqn_data import FrozenResponseReplay, load_frozen_response_replay


ADVANTAGE_REPLAY_FORMAT = "response_aware_action0_advantage_replay_v1"


def _paired_group_keys(replay: FrozenResponseReplay) -> list[tuple[str, int]]:
    return [
        (str(group), int(step))
        for group, step in zip(replay.event_group, replay.control_step)
    ]


def convert_replay_to_action0_advantage(
    replay: FrozenResponseReplay,
    *,
    source: str,
    require_non_anchor: bool = True,
) -> FrozenResponseReplay:
    """Use paired action0 rows as the same-state baseline.

    The input reward is assumed to be a larger-is-better return, such as
    negative TTT. The converted reward is therefore:

        reward(action) - mean reward(action0 from the same state)

    Positive values mean the action improved over action0.
    """
    replay.validate()
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(_paired_group_keys(replay)):
        grouped[key].append(index)

    keep_indices = []
    converted_rewards = []
    skipped_without_anchor = 0
    skipped_anchor_only = 0
    for _, indices in grouped.items():
        actions = replay.action_id[indices].astype(np.int64)
        anchor_local = np.flatnonzero(actions == 0)
        if anchor_local.size == 0:
            skipped_without_anchor += len(indices)
            continue
        if require_non_anchor and np.count_nonzero(actions != 0) == 0:
            skipped_anchor_only += len(indices)
            continue
        anchor_reward = float(np.mean(replay.reward[np.asarray(indices)[anchor_local]]))
        for index in indices:
            keep_indices.append(index)
            converted_rewards.append(float(replay.reward[index] - anchor_reward))

    if not keep_indices:
        raise ValueError("no paired action0 groups were available for advantage conversion")
    idx = np.asarray(keep_indices, dtype=np.int64)
    reward = np.asarray(converted_rewards, dtype=np.float32)
    manifest = dict(replay.manifest)
    manifest.update({
        "format_version": replay.manifest["format_version"],
        "advantage_format_version": ADVANTAGE_REPLAY_FORMAT,
        "source": source,
        "source_dataset": replay.manifest.get("source", ""),
        "transition_count": int(idx.size),
        "reward_mode": "action0_advantage",
        "reward_semantics": "paired_reward_minus_same_state_action0_mean_reward",
        "paired_group_key": "event_group+control_step",
        "skipped_rows_without_action0": int(skipped_without_anchor),
        "skipped_anchor_only_rows": int(skipped_anchor_only),
    })
    return FrozenResponseReplay(
        observation=replay.observation[idx],
        action_id=replay.action_id[idx],
        reward=reward,
        next_observation=replay.next_observation[idx],
        done=replay.done[idx],
        option_steps=replay.option_steps[idx],
        action_mask=replay.action_mask[idx],
        next_action_mask=replay.next_action_mask[idx],
        response_features=replay.response_features[idx],
        next_response_features=replay.next_response_features[idx],
        event_group=replay.event_group[idx],
        episode=replay.episode[idx],
        control_step=replay.control_step[idx],
        manifest=manifest,
    ).validate()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keep-anchor-only",
        action="store_true",
        help="keep groups that only contain action0; by default they are skipped",
    )
    args = parser.parse_args(argv)

    replay = load_frozen_response_replay(args.input)
    converted = convert_replay_to_action0_advantage(
        replay,
        source="rl_leader.convert_response_replay_to_advantage",
        require_non_anchor=not args.keep_anchor_only,
    )
    converted.save(args.out)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.out),
        "transitions": converted.size,
        "action_support_counts": converted.action_support_counts().tolist(),
        "reward_min": float(np.min(converted.reward)),
        "reward_max": float(np.max(converted.reward)),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
