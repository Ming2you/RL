"""Evaluate an explicit P-Stack-relative action schedule in closed loop."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn_catalog import (
    StructuredActionCatalog,
    build_structured_action_catalog,
    load_extra_action_specs,
)


EVAL_FORMAT = "fixed_response_action_schedule_eval_v1"


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in str(value).split(",") if item)


def parse_schedule(items: list[str] | tuple[str, ...]) -> dict[int, str]:
    schedule: dict[int, str] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"schedule item must be STEP=ACTION_KEY: {item}")
        raw_step, key = str(item).split("=", 1)
        step = int(raw_step)
        if step < 0:
            raise ValueError(f"schedule step must be nonnegative: {step}")
        key = key.strip()
        if not key:
            raise ValueError("schedule action key must not be empty")
        if step in schedule:
            raise ValueError(f"duplicate schedule step: {step}")
        schedule[step] = key
    return dict(sorted(schedule.items()))


def resolve_schedule(
    catalog: StructuredActionCatalog,
    schedule_keys: dict[int, str],
) -> dict[int, int]:
    by_key = {action.key: action.action_id for action in catalog.actions}
    missing = sorted(set(schedule_keys.values()) - set(by_key))
    if missing:
        known = ", ".join(action.key for action in catalog.actions[:12])
        raise ValueError(
            "schedule references unknown action key(s): "
            + ", ".join(missing)
            + f". Known prefix: {known}"
        )
    resolved = {step: int(by_key[key]) for step, key in schedule_keys.items()}
    anchor_steps = [step for step, action_id in resolved.items() if action_id == 0]
    if anchor_steps:
        raise ValueError(f"schedule should omit anchor-only steps: {anchor_steps}")
    return resolved


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_fixed_action_schedule(
    *,
    scenario: str,
    t_total: float,
    catalog: StructuredActionCatalog,
    schedule_actions: dict[int, int],
    output_dir: Path,
    pstack_summary: dict | None = None,
    force_without_h3_gate: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "trace.jsonl"
    collisions = [path for path in (summary_path, trace_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing schedule outputs: "
            + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    observation = np.asarray(env.reset(), dtype=np.float32)
    rewards: list[float] = []
    action_ids: list[int] = []
    forced_steps: list[int] = []
    applied_nonanchor_steps: list[int] = []

    with trace_path.open("w", encoding="utf-8") as trace:
        step = 0
        done = False
        while not done:
            anchor_context = env.prepare_pstack_anchor_context()
            action_id = int(schedule_actions.get(step, 0))
            action = catalog.action(action_id)
            if action_id == 0:
                next_observation, reward, done, info, _ = env.step_prepared_optimizer_anchor(
                    anchor_context,
                )
            else:
                forced_steps.append(step)
                if force_without_h3_gate:
                    next_observation, reward, done, info = env.step_anchored_candidate(
                        catalog.residual(action_id),
                        anchor_context,
                    )
                else:
                    next_observation, reward, done, info = env._step(
                        catalog.residual(action_id),
                        anchor_context=anchor_context,
                        apply_anchor_gate=True,
                    )
                deployed = np.asarray(env.last_deployed_residual, dtype=np.float32)
                if np.linalg.norm(deployed, ord=2) > 1.0e-9:
                    applied_nonanchor_steps.append(step)

            rewards.append(float(reward))
            action_ids.append(action_id)
            payload = {
                "format_version": EVAL_FORMAT,
                "control_step": int(step),
                "scheduled": bool(action_id != 0),
                "selected_action_id": int(action_id),
                "action_key": action.key,
                "owner": action.owner,
                "template": action.template,
                "magnitude": float(action.magnitude),
                "step_ttt": float(-reward),
                "total_ttt_so_far": float(env.sim.total_ttt),
                "terminal_inventory": float(info.get("inventory_after", env._inventory())),
                "validity_gate_pass": bool(info.get("validity_gate_pass", False)),
                "force_without_h3_gate": bool(force_without_h3_gate),
                "observation_norm": float(np.linalg.norm(observation)),
                "next_observation_norm": float(np.linalg.norm(next_observation)),
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
            print(json.dumps({
                "control_step": int(step),
                "action_key": action.key,
                "scheduled": bool(action_id != 0),
                "step_ttt": round(float(-reward), 6),
                "done": bool(done),
            }, sort_keys=True), flush=True)
            observation = np.asarray(next_observation, dtype=np.float32)
            step += 1

    rewards_array = np.asarray(rewards, dtype=np.float64)
    action_counts = Counter(action_ids)
    total_ttt = float(env.sim.total_ttt)
    summary = {
        "format_version": EVAL_FORMAT,
        "scenario": scenario,
        "t_total_sec": float(t_total),
        "control_steps": int(len(action_ids)),
        "total_ttt": total_ttt,
        "control_ttt": float(np.sum(-rewards_array)),
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(env._inventory()),
        "action_counts": {
            str(action_id): int(count)
            for action_id, count in sorted(action_counts.items())
        },
        "forced_steps": list(map(int, forced_steps)),
        "applied_nonanchor_steps": list(map(int, applied_nonanchor_steps)),
        "non_anchor_steps": int(sum(1 for action_id in action_ids if action_id != 0)),
        "force_without_h3_gate": bool(force_without_h3_gate),
        "catalog_fingerprint": catalog.fingerprint,
        "catalog_size": int(catalog.size),
        "trace_path": str(trace_path),
        "wall_seconds": float(time.perf_counter() - started),
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
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
        })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--magnitudes", default="0.25,0.5")
    parser.add_argument("--families", default="linear")
    parser.add_argument("--domains", default="freeway")
    parser.add_argument("--owners", default="")
    parser.add_argument(
        "--extra-actions",
        action="append",
        default=[],
        type=Path,
        help="manual or artifact-derived residual action manifest to append",
    )
    parser.add_argument(
        "--schedule",
        action="append",
        default=[],
        help="repeatable STEP=ACTION_KEY entry",
    )
    parser.add_argument(
        "--use-h3-anchor-gate",
        action="store_true",
        help="apply the existing H3 P-Stack anchor gate instead of forcing the residual",
    )
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.schedule:
        raise SystemExit("at least one --schedule STEP=ACTION_KEY is required")
    env = RLLeaderEnv(
        scenario_name=args.scenario,
        T_total=args.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    catalog = build_structured_action_catalog(
        env.action_schema.names,
        magnitudes=_parse_csv(args.magnitudes, float),
        families=_parse_csv(args.families),
        domains=_parse_csv(args.domains),
        owners=_parse_csv(args.owners) or None,
        extra_actions=load_extra_action_specs(args.extra_actions),
    )
    schedule_keys = parse_schedule(args.schedule)
    schedule_actions = resolve_schedule(catalog, schedule_keys)
    run_fixed_action_schedule(
        scenario=args.scenario,
        t_total=args.t_total,
        catalog=catalog,
        schedule_actions=schedule_actions,
        output_dir=args.output_dir,
        pstack_summary=_read_json(args.pstack_summary),
        force_without_h3_gate=not bool(args.use_h3_anchor_gate),
    )


if __name__ == "__main__":
    main()
