"""Evaluate the native P-Stack anchor through the RL environment contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rl_leader.env import RLLeaderEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "trace.jsonl"
    if summary_path.exists() or trace_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    env = RLLeaderEnv(
        scenario_name=args.scenario,
        T_total=args.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    rows: list[dict[str, float | int | str | bool]] = []
    control_step = 0
    done = False
    while not done:
        step_started = time.perf_counter()
        context = env.prepare_pstack_anchor_context()
        _, reward, done, info, _ = env.step_prepared_optimizer_anchor(
            context,
            sync_follower_state=True,
        )
        row = {
            "control_step": control_step,
            "simulation_step": int(env.step_idx - 1),
            "simulation_time_sec": float(env.sim.state.time_sec),
            "interval_ttt": float(-reward),
            "cumulative_total_ttt": float(env.sim.total_ttt),
            "cumulative_urban_ttt": float(env.sim.urban_ttt),
            "cumulative_freeway_ttt": float(env.sim.freeway_ttt),
            "terminal_inventory": float(info.get("inventory_after", env._inventory())),
            "evaluation_seconds": float(time.perf_counter() - step_started),
            "anchor_fingerprint": context.anchor_fingerprint,
        }
        rows.append(row)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"anchor step {control_step + 1}/{env.n_steps - env.warmup} "
            f"total_ttt={env.sim.total_ttt:.6f} "
            f"interval_ttt={-reward:.6f} "
            f"elapsed={row['evaluation_seconds']:.2f}s",
            flush=True,
        )
        control_step += 1

    summary = {
        "scenario": args.scenario,
        "t_total_sec": float(args.t_total),
        "warmup_steps": int(env.warmup),
        "control_steps": len(rows),
        "experiment_contract_sha256": env.experiment_contract_fingerprint,
        "total_ttt": float(env.sim.total_ttt),
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(rows[-1]["terminal_inventory"]),
        "wall_seconds": float(time.perf_counter() - started),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
