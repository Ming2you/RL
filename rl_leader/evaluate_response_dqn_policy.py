"""Evaluate a response-aware Double DQN policy against the exact RL contract."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_catalog import (
    build_structured_action_catalog,
    load_extra_action_specs,
)
from rl_leader.response_dqn_collect import (
    CollectionDecision,
    collect_sequential_episode,
    ensemble_lcb_policy,
    rows_to_replay,
    _save_replay_atomic,
)


EVAL_FORMAT = "response_dqn_exact_policy_eval_v1"


def summarize_policy_run(
    *,
    env: RLLeaderEnv,
    rows: dict[str, list],
    pstack_summary: dict | None = None,
) -> dict:
    rewards = np.asarray(rows["reward"], dtype=np.float64)
    action_ids = np.asarray(rows["action_id"], dtype=np.int64)
    control_ttt = float(np.sum(-rewards))
    total_ttt = float(env.sim.total_ttt)
    warmup_ttt = float(total_ttt - control_ttt)
    summary = {
        "format_version": EVAL_FORMAT,
        "scenario": env.scenario_name,
        "t_total_sec": float(env.T_total),
        "warmup_steps": int(env.warmup),
        "control_steps": int(action_ids.size),
        "experiment_contract_sha256": env.experiment_contract_fingerprint,
        "total_ttt": total_ttt,
        "control_ttt": control_ttt,
        "warmup_ttt": warmup_ttt,
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(env._inventory()),
        "action_counts": {
            str(action_id): int(count)
            for action_id, count in Counter(action_ids.tolist()).items()
        },
        "non_anchor_steps": int(np.count_nonzero(action_ids != 0)),
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
        pstack_control = pstack_total - warmup_ttt
        gap = total_ttt - pstack_total
        summary.update({
            "pstack_total_ttt": pstack_total,
            "pstack_control_ttt": float(pstack_control),
            "vs_pstack_ttt_gap": float(gap),
            "vs_pstack_percent": float(100.0 * gap / max(abs(pstack_total), 1.0e-9)),
            "beats_pstack": bool(gap < 0.0),
        })
    return summary


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in str(value).split(",") if item)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _step_gated_policy(policy, allowed_steps: set[int] | None):
    if not allowed_steps:
        return policy
    counter = {"step": 0}

    def choose(evaluated, catalog, rng):
        step = int(counter["step"])
        counter["step"] = step + 1
        if step not in allowed_steps:
            return CollectionDecision(action_id=0, diagnostics={
                "policy": "ensemble_lcb_step_gate",
                "exploratory": False,
                "q_anchor": 0.0,
                "q_selected": 0.0,
                "delta_q": 0.0,
                "ensemble_std": 0.0,
                "lcb": 0.0,
                "fallback_reason": "outside_allowed_control_steps",
                "step_gate_allowed": False,
            })
        decision = policy(evaluated, catalog, rng)
        decision.diagnostics["step_gate_allowed"] = True
        return decision

    return choose


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--magnitudes", default="0.25")
    parser.add_argument("--families", default="linear,quadratic,cross")
    parser.add_argument("--domains", default="urban,freeway")
    parser.add_argument("--owners", default="R_F_E")
    parser.add_argument(
        "--extra-actions",
        action="append",
        default=[],
        type=Path,
        help="manual or artifact-derived residual action manifest to append",
    )
    parser.add_argument("--z-value", type=float, default=1.96)
    parser.add_argument("--material-margin", type=float, default=0.0)
    parser.add_argument("--exploration-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--allow-control-steps",
        default="",
        help="comma-separated control-step indices where non-anchor actions are allowed",
    )
    parser.add_argument(
        "--preview-actions",
        default="",
        help=(
            "comma-separated action IDs to preview at evaluated steps; "
            "empty preserves full-catalog preview"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend", choices=("serial", "thread", "process"), default="process",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.output_dir / "summary.json"
    trace_path = args.output_dir / "trace.jsonl"
    replay_path = args.output_dir / "replay.npz"
    collisions = [
        path for path in (summary_path, trace_path, replay_path) if path.exists()
    ]
    if collisions:
        raise SystemExit(
            "refusing to overwrite existing evaluation outputs: "
            + ", ".join(map(str, collisions))
        )
    pstack_summary = (
        json.loads(args.pstack_summary.read_text(encoding="utf-8"))
        if args.pstack_summary is not None else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
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
    paths = sorted(args.model_dir.glob("response_dqn_member_*.pt"))
    if not paths:
        raise SystemExit(f"no response DQN checkpoints in {args.model_dir}")
    ensemble = [
        load_trained_response_dqn(
            path,
            expected_catalog_fingerprint=catalog.fingerprint,
        )
        for path in paths
    ]
    policy = ensemble_lcb_policy(
        ensemble,
        z_value=args.z_value,
        material_margin=args.material_margin,
        exploration_epsilon=args.exploration_epsilon,
    )
    allowed_control_steps = set(_parse_csv(args.allow_control_steps, int))
    preview_action_ids = _parse_csv(args.preview_actions, int)
    policy = _step_gated_policy(policy, allowed_control_steps)
    rows = collect_sequential_episode(
        env,
        catalog,
        policy,
        episode=args.episode,
        rng=np.random.default_rng(args.seed),
        event_group=f"{args.scenario}:eval:{args.episode}:seed:{args.seed}",
        log_path=trace_path,
        response_workers=args.workers,
        response_backend=args.backend,
        checkpoint_every=0,
        evaluate_response_steps=(
            (lambda step: int(step) in allowed_control_steps)
            if allowed_control_steps else None
        ),
        anchor_only_response_feature_dim=(
            int(ensemble[0].normalizer.response_mean.size)
            if allowed_control_steps else None
        ),
        preview_action_ids=preview_action_ids or None,
    )
    replay = rows_to_replay(
        rows,
        env=env,
        catalog=catalog,
        source="rl_leader.evaluate_response_dqn_policy",
    )
    _save_replay_atomic(replay, replay_path)
    summary = summarize_policy_run(
        env=env,
        rows=rows,
        pstack_summary=pstack_summary,
    )
    summary.update({
        "model_dir": str(args.model_dir),
        "catalog_fingerprint": catalog.fingerprint,
        "replay_path": str(replay_path),
        "trace_path": str(trace_path),
        "wall_seconds": float(time.perf_counter() - started),
        "arguments": {
            key: _json_ready(value)
            for key, value in vars(args).items()
        },
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
