"""Merge contract-compatible frozen response-DQN collection batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_leader.response_dqn_data import (
    load_frozen_response_replay,
    merge_frozen_response_replays,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replays = [load_frozen_response_replay(path) for path in args.inputs]
    merged = merge_frozen_response_replays(
        replays, source="rl_leader.merge_response_dqn_replay",
    )
    merged.save(args.out)
    print(json.dumps({
        "output": str(args.out),
        "batches": len(replays),
        "transitions": merged.size,
        "event_groups": merged.unique_event_groups().tolist(),
        "action_support_counts": merged.action_support_counts().tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
