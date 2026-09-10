"""Replay frozen-policy overrides and generate paired P-Stack-relative labels."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.generate_long_horizon_labels import _evaluate_event


def action_from_trace_row(row: dict, names: tuple[str, ...]) -> np.ndarray:
    raw = row.get("raw_action")
    if not isinstance(raw, dict):
        raise ValueError("trace row is missing its raw_action mapping")
    missing = [name for name in names if name not in raw]
    extra = sorted(set(raw) - set(names))
    if missing or extra:
        raise ValueError(
            f"trace action schema mismatch: missing={missing}, extra={extra}"
        )
    return np.asarray([raw[name] for name in names], dtype=np.float32)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _observation_from_trace(row: dict, names: tuple[str, ...]) -> np.ndarray:
    values = row.get("observation")
    if not isinstance(values, dict):
        raise ValueError("trace row is missing its observation mapping")
    missing = [name for name in names if name not in values]
    extra = sorted(set(values) - set(names))
    if missing or extra:
        raise ValueError(
            f"trace observation schema mismatch: missing={missing}, extra={extra}"
        )
    return np.asarray([values[name] for name in names], dtype=np.float32)


def replay_accepted_events(
    trace_path: Path,
    *,
    max_rollout_steps: int,
    horizons: tuple[int, ...],
    parity_tolerance: float,
    output_path: Path,
) -> dict:
    rows = _load_jsonl(trace_path)
    if not rows:
        raise ValueError(f"empty trace: {trace_path}")
    meta_path = trace_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scenario = str(meta["scenario"])
    mask = str(meta["mask"])
    expected_pick_steps = [
        int(row["step"])
        for row in rows
        if float(row["info"].get("leader_rl_pstack_anchor_pick_rl", 0.0)) > 0.5
    ]
    if not expected_pick_steps:
        raise ValueError(f"trace has no accepted RL events: {trace_path}")

    env = RLLeaderEnv(
        scenario_name=scenario,
        mask=mask,
        pstack_anchor=True,
    )
    env.action_parameterization = str(meta["action_parameterization"])
    observation = env.reset()
    if meta.get("experiment_contract_sha256") != env.experiment_contract_fingerprint:
        raise ValueError("trace experiment contract does not match replay environment")

    event_snapshots = []
    expected_set = set(expected_pick_steps)
    last_expected = max(expected_pick_steps)
    replayed_pick_steps = []
    for row in rows:
        step = int(row["step"])
        if step > last_expected:
            break
        saved_observation = _observation_from_trace(
            row, env.observation_schema.names
        )
        observation_error = float(np.max(np.abs(observation - saved_observation)))
        if observation_error > parity_tolerance:
            raise RuntimeError(
                f"trace observation replay error at step {step}: {observation_error}"
            )
        action = action_from_trace_row(row, env.action_schema.names)
        before = copy.deepcopy(env) if step in expected_set else None
        observation, reward, _, info = env.step(action)
        replay_pick = float(
            info.get("leader_rl_pstack_anchor_pick_rl", 0.0)
        ) > 0.5
        saved_pick = step in expected_set
        reward_error = abs(float(-reward) - float(row["info"]["step_ttt"]))
        if replay_pick != saved_pick or reward_error > parity_tolerance:
            raise RuntimeError(
                f"trace transition replay mismatch at step {step}: "
                f"saved_pick={saved_pick}, replay_pick={replay_pick}, "
                f"reward_error={reward_error}"
            )
        if replay_pick:
            replayed_pick_steps.append(step)
            event_snapshots.append({
                "step": step,
                "simulation_time_sec": float(row["state_before"]["time_sec"]),
                "env": before,
                "action": action,
                "observation_error": observation_error,
                "reward_error": reward_error,
                "h3_gate_gain": float(
                    info["leader_rl_pstack_anchor_gain"]
                ),
                "h3_required_gain": float(
                    info["leader_rl_pstack_anchor_required_gain"]
                ),
            })

    if replayed_pick_steps != expected_pick_steps:
        raise RuntimeError(
            f"accepted-event replay mismatch: expected={expected_pick_steps}, "
            f"actual={replayed_pick_steps}"
        )

    output = {
        "format_version": "frozen_event_attribution_v1",
        **env.experiment_contract.artifact_fields(),
        "scenario": scenario,
        "mask": mask,
        "source_trace": str(trace_path),
        "source_trace_meta": str(meta_path),
        "max_rollout_steps": int(max_rollout_steps),
        "requested_horizons": list(horizons),
        "expected_pick_steps": expected_pick_steps,
        "events": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for snapshot in event_snapshots:
        result = _evaluate_event(
            snapshot["env"],
            snapshot["action"],
            max_rollout_steps=max_rollout_steps,
            horizons=horizons,
        )
        event = {
            key: value for key, value in snapshot.items() if key not in {"env", "action"}
        }
        event.update({
            "residual_l2": float(np.linalg.norm(snapshot["action"])),
            "residual_abs_max": float(np.max(np.abs(snapshot["action"]))),
            **result,
        })
        output["events"].append(event)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(
            f"attribution scenario={scenario} step={event['step']} "
            f"rollout={event['rollout_steps']} gain={event['ttt_gain']:+.3f} "
            f"positive={int(event['long_horizon_positive'])}",
            flush=True,
        )
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--max-rollout-steps", type=int, default=12)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--output-dir",
        default="results/rl_phase0_implementation_20260827/frozen_event_attribution",
    )
    args = parser.parse_args(argv)
    horizons = tuple(sorted({
        int(value) for value in args.horizons.split(",") if value.strip()
    }))
    output_dir = Path(args.output_dir)
    summary = []
    for value in args.traces:
        trace_path = Path(value)
        output_path = output_dir / f"{trace_path.stem}.json"
        result = replay_accepted_events(
            trace_path,
            max_rollout_steps=args.max_rollout_steps,
            horizons=horizons,
            parity_tolerance=args.parity_tolerance,
            output_path=output_path,
        )
        summary.append({
            "scenario": result["scenario"],
            "output": str(output_path),
            "event_count": len(result["events"]),
            "positive_count": sum(
                int(event["long_horizon_positive"])
                for event in result["events"]
            ),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
