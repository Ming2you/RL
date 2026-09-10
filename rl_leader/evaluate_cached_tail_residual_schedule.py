"""Evaluate a residual schedule from a cached P-Stack prefix state."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from rl_leader.evaluate_fixed_residual_schedule import (
    ScheduledResidual,
    load_sampled_residual_schedule,
    load_oracle_choice_schedule,
    merge_schedules,
    parse_sparse_schedule_items,
)


EVAL_FORMAT = "cached_tail_residual_schedule_eval_v1"


def load_cached_prefix(path: Path):
    """Load an ``(env, anchor_context)`` tuple written by tail-label generation."""
    with Path(path).open("rb") as handle:
        env, anchor_context = pickle.load(handle)
    if int(env.step_idx) != int(anchor_context.step_idx):
        raise ValueError("cached env/context step_idx mismatch")
    actual = env._anchor_context_state_fingerprint()
    if str(actual) != str(anchor_context.state_fingerprint):
        raise ValueError("cached env/context state fingerprint mismatch")
    return env, anchor_context


def run_cached_tail_residual_schedule(
    *,
    replay_cache: Path,
    schedule: dict[int, ScheduledResidual],
    output_dir: Path,
    pstack_summary: dict | None = None,
    force_without_h3_gate: bool = True,
    verbose: bool = True,
    max_tail_steps: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "trace.jsonl"
    collisions = [path for path in (summary_path, trace_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing cached-tail outputs: "
            + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    env, cached_context = load_cached_prefix(Path(replay_cache))
    initial_control_step = int(env.step_idx - env.warmup)
    initial_total_ttt = float(env.sim.total_ttt)
    forced_steps: list[int] = []
    applied_nonzero_steps: list[int] = []
    coordination_zero_steps: list[int] = []
    labels: list[str] = []
    rewards: list[float] = []
    first_iteration = True
    done = False

    tail_step_limit = None if max_tail_steps is None else int(max_tail_steps)
    if tail_step_limit is not None and tail_step_limit <= 0:
        raise ValueError("max_tail_steps must be positive when provided")

    with trace_path.open("w", encoding="utf-8") as trace:
        while not done and env.step_idx < env.n_steps:
            step = int(env.step_idx - env.warmup)
            anchor_context = cached_context if first_iteration else env.prepare_pstack_anchor_context()
            first_iteration = False
            scheduled = schedule.get(step)
            if scheduled is None:
                _next_observation, reward, done, info, _ = env.step_prepared_optimizer_anchor(
                    anchor_context,
                    sync_follower_state=True,
                )
                residual = np.zeros(env.action_dim, dtype=np.float32)
                label = "anchor"
            else:
                forced_steps.append(step)
                residual = scheduled.residual_array()
                if np.linalg.norm(residual, ord=2) <= 1.0e-9:
                    coordination_zero_steps.append(step)
                if force_without_h3_gate:
                    _next_observation, reward, done, info = env.step_anchored_candidate(
                        residual,
                        anchor_context,
                    )
                else:
                    _next_observation, reward, done, info = env._step(
                        residual,
                        anchor_context=anchor_context,
                        apply_anchor_gate=True,
                    )
                deployed = np.asarray(env.last_deployed_residual, dtype=np.float32)
                if np.linalg.norm(deployed, ord=2) > 1.0e-9:
                    applied_nonzero_steps.append(step)
                label = scheduled.label

            rewards.append(float(reward))
            labels.append(label)
            payload = {
                "format_version": EVAL_FORMAT,
                "control_step": int(step),
                "scheduled": scheduled is not None,
                "label": label,
                "source": None if scheduled is None else scheduled.source,
                "step_ttt": float(-reward),
                "total_ttt_so_far": float(env.sim.total_ttt),
                "terminal_inventory": float(info.get("inventory_after", env._inventory())),
                "validity_gate_pass": bool(info.get("validity_gate_pass", False)),
                "force_without_h3_gate": bool(force_without_h3_gate),
                "residual_l2": float(np.linalg.norm(residual)),
                "residual_nonzero_count": int(np.count_nonzero(np.abs(residual) > 1.0e-9)),
            }
            for key in (
                "leader_rl_pstack_anchor_pick_rl",
                "leader_rl_pstack_anchor_pick_pstack",
                "leader_rl_pstack_anchor_gain",
                "leader_rl_pstack_anchor_required_gain",
            ):
                if key in info:
                    payload[key] = float(info[key])
            trace.write(json.dumps(payload, sort_keys=True) + "\n")
            trace.flush()
            if verbose:
                print(json.dumps({
                    "control_step": int(step),
                    "done": bool(done),
                    "label": label,
                    "scheduled": scheduled is not None,
                    "step_ttt": round(float(-reward), 6),
                }, sort_keys=True), flush=True)
            if tail_step_limit is not None and len(labels) >= tail_step_limit:
                break

    rewards_array = np.asarray(rewards, dtype=np.float64)
    total_ttt = float(env.sim.total_ttt)
    partial_tail_evaluation = bool(not done and env.step_idx < env.n_steps)
    summary = {
        "format_version": EVAL_FORMAT,
        "replay_cache": str(replay_cache),
        "scenario": env.scenario_name,
        "t_total_sec": float(env.T_total),
        "warmup_steps": int(env.warmup),
        "initial_control_step": int(initial_control_step),
        "initial_total_ttt": float(initial_total_ttt),
        "tail_steps": int(len(labels)),
        "control_steps": int(env.step_idx - env.warmup),
        "total_ttt": total_ttt,
        "tail_ttt": float(np.sum(-rewards_array)),
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(env._inventory()),
        "forced_steps": list(map(int, forced_steps)),
        "applied_nonzero_steps": list(map(int, applied_nonzero_steps)),
        "coordination_zero_steps": list(map(int, coordination_zero_steps)),
        "scheduled_labels": {
            str(step): item.label for step, item in sorted(schedule.items())
        },
        "force_without_h3_gate": bool(force_without_h3_gate),
        "trace_path": str(trace_path),
        "tail_step_limit": tail_step_limit,
        "partial_tail_evaluation": partial_tail_evaluation,
        "wall_seconds": float(time.perf_counter() - started),
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
        if partial_tail_evaluation:
            summary.update({
                "pstack_total_ttt": pstack_total,
                "target_5pct_total_ttt": float(0.95 * pstack_total),
                "beats_pstack": False,
                "meets_5pct_target": False,
                "comparison_is_partial": True,
            })
        else:
            gap = total_ttt - pstack_total
            summary.update({
                "pstack_total_ttt": pstack_total,
                "vs_pstack_ttt_gap": float(gap),
                "vs_pstack_percent": float(
                    100.0 * gap / max(abs(pstack_total), 1.0e-9)
                ),
                "beats_pstack": bool(gap < 0.0),
                "target_5pct_total_ttt": float(0.95 * pstack_total),
                "meets_5pct_target": bool(total_ttt <= 0.95 * pstack_total),
                "comparison_is_partial": False,
            })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--schedule", action="append", default=[])
    parser.add_argument("--oracle-artifact", action="append", default=[], type=Path)
    parser.add_argument("--oracle-selection", default="oracle_choice")
    parser.add_argument("--sample-artifact", action="append", default=[], type=Path)
    parser.add_argument("--sample-selection", default="best_h12")
    parser.add_argument("--use-h3-anchor-gate", action="store_true")
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tail-steps", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env, _context = load_cached_prefix(args.replay_cache)
    action_names = tuple(env.action_schema.names)
    schedules = [
        parse_sparse_schedule_items(args.schedule, action_names),
        *[
            load_oracle_choice_schedule(
                path,
                action_names,
                selection=args.oracle_selection,
            )
            for path in args.oracle_artifact
        ],
        *[
            load_sampled_residual_schedule(
                path,
                action_names,
                selection=args.sample_selection,
            )
            for path in args.sample_artifact
        ],
    ]
    schedule = merge_schedules(*schedules)
    pstack_summary = None
    if args.pstack_summary is not None:
        pstack_summary = json.loads(args.pstack_summary.read_text(encoding="utf-8"))
    run_cached_tail_residual_schedule(
        replay_cache=args.replay_cache,
        schedule=schedule,
        output_dir=args.output_dir,
        pstack_summary=pstack_summary,
        force_without_h3_gate=not bool(args.use_h3_anchor_gate),
        verbose=not bool(args.quiet),
        max_tail_steps=args.max_tail_steps,
    )


if __name__ == "__main__":
    main()
