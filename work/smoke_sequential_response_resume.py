"""Exercise real follower execution, checkpoint restore, and trace parity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from rl_leader.run_sequential_response_ddqn import (
    _atomic_json, build_pilot_catalog, collect_actor, load_verified_snapshot,
)
from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.response_dqn_collect import _shutdown_response_process_pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path("work/sequential_ddqn_170_incident_v1.json").read_text(encoding="utf-8"))
    config["action_keys"] = config["action_keys"][:2]
    snapshot = load_verified_snapshot(config)
    catalog = build_pilot_catalog(config, snapshot)
    payload = {
        "config": config, "catalog": catalog.as_manifest(),
        "directory": str(args.output_dir), "seed": 17, "episode": 90000,
        "checkpoints": [], "first_action": 1, "epsilon": 0.0,
        "response_workers": 2, "stop_file": str(args.output_dir / "STOP"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Path(config["reference_trace"]).open(encoding="utf-8") as handle:
        reference = {int(row["control_step"]): row for row in map(json.loads, handle)}
    checks = []
    try:
        for expected_count in (1, 2):
            try:
                collect_actor(payload, max_new_transitions=1)
            except InterruptedError as exc:
                if "diagnostic pause" not in str(exc):
                    raise
            replay = load_frozen_response_replay(args.output_dir / "replay.npz")
            with (args.output_dir / "checkpoint.pkl").open("rb") as handle:
                saved = pickle.load(handle)
            assert replay.size == expected_count
            assert np.all(replay.done == 0), "diagnostic stop must not become terminal"
            step = 18 + expected_count
            actual = saved["env"]._anchor_context_state_fingerprint()
            assert actual == reference[step]["state_id"], "resumed state differs from reference"
            np.testing.assert_array_equal(saved["env"]._observe(), replay.next_observation[-1])
            assert np.isclose(float(replay.reward[-1]), reference[step - 1]["interval_reward"])
            if expected_count > 1:
                np.testing.assert_array_equal(replay.next_observation[0], replay.observation[1])
            checks.append({"transitions": replay.size, "next_control_step": step,
                           "reference_state_match": True, "terminal_count": int(replay.done.sum())})
            print(json.dumps(checks[-1]), flush=True)
    finally:
        _shutdown_response_process_pools()
    _atomic_json(args.output_dir / "smoke_result.json", {"passed": True, "checks": checks})


if __name__ == "__main__":
    main()
