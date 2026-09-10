"""Evaluate response-DQN checkpoints without candidate response previews.

The response-aware policy normally evaluates every candidate follower response
before scoring actions.  This runner measures the cheaper deployment-style
path: score the fixed action catalog from the current observation plus a
surrogate response tensor, then execute only the selected residual through the
actual follower MPC.
"""
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


EVAL_FORMAT = "response_dqn_nominal_no_preview_eval_v1"


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


def _surrogate_response_features(
    *,
    env: RLLeaderEnv,
    anchor_context,
    action_count: int,
    response_dim: int,
    mode: str,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((int(action_count), int(response_dim)), dtype=np.float32)
    if mode != "anchor":
        raise ValueError("response surrogate must be 'anchor' or 'zero'")
    if int(response_dim) % 2 != 0:
        raise ValueError("anchor surrogate expects raw+delta response features")
    raw_dim = int(response_dim) // 2
    anchor = np.asarray(
        env.response_vector(anchor_context.result.control),
        dtype=np.float32,
    ).reshape(-1)
    if anchor.size != raw_dim:
        raise ValueError(
            "anchor response dimension does not match checkpoint response feature size"
        )
    raw = np.repeat(anchor[None, :], int(action_count), axis=0)
    delta = np.zeros_like(raw)
    return np.concatenate((raw, delta), axis=1).astype(np.float32)


def _select_action(
    *,
    ensemble,
    observation: np.ndarray,
    response_features: np.ndarray,
    support_mask: np.ndarray,
    mode: str,
    z_value: float,
) -> tuple[int, dict]:
    values = np.stack([
        model.q_values(observation, response_features)[0]
        for model in ensemble
    ]).astype(np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    mask = np.asarray(support_mask, dtype=bool).copy()
    if not np.any(mask):
        raise ValueError("nominal DDQN selection received an empty support mask")
    if mode == "mean":
        score = mean
    elif mode == "lcb":
        score = mean - float(z_value) * std
    else:
        raise ValueError("selection mode must be 'mean' or 'lcb'")
    selected = int(np.flatnonzero(mask)[np.argmax(score[mask])])
    anchor_q = float(mean[0])
    return selected, {
        "selection_mode": mode,
        "selected_action_id": selected,
        "q_anchor_mean": anchor_q,
        "q_selected_mean": float(mean[selected]),
        "q_selected_std": float(std[selected]),
        "delta_q_mean": float(mean[selected] - anchor_q),
        "score_selected": float(score[selected]),
        "supported_action_count": int(mask.sum()),
    }


def run_nominal_policy(args: argparse.Namespace) -> dict:
    summary_path = args.output_dir / "summary.json"
    trace_path = args.output_dir / "trace.jsonl"
    replay_collision = args.output_dir / "replay.npz"
    collisions = [
        path for path in (summary_path, trace_path, replay_collision) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing nominal DDQN outputs: "
            + ", ".join(map(str, collisions))
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
    model_paths = sorted(args.model_dir.glob("response_dqn_member_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"no response DQN checkpoints in {args.model_dir}")
    ensemble = [
        load_trained_response_dqn(
            path,
            expected_catalog_fingerprint=catalog.fingerprint,
        )
        for path in model_paths
    ]
    support = np.min(
        np.stack([model.action_support_counts for model in ensemble]),
        axis=0,
    )
    support_mask = support >= int(args.min_action_support)
    support_mask[0] = True
    response_dim = int(ensemble[0].normalizer.response_mean.size)
    allowed_steps = set(_parse_csv(args.allow_control_steps, int))

    rng = np.random.default_rng(args.seed)
    del rng
    observation = np.asarray(env.reset(), dtype=np.float32)
    rewards: list[float] = []
    action_ids: list[int] = []
    valid_gate_failures = 0
    anchor_seconds = 0.0
    q_seconds = 0.0
    commit_seconds = 0.0
    done = False

    with trace_path.open("w", encoding="utf-8") as trace:
        while not done and env.step_idx < env.n_steps:
            control_step = int(env.step_idx - env.warmup)
            t0 = time.perf_counter()
            anchor_context = env.prepare_pstack_anchor_context()
            anchor_elapsed = time.perf_counter() - t0
            features = _surrogate_response_features(
                env=env,
                anchor_context=anchor_context,
                action_count=catalog.size,
                response_dim=response_dim,
                mode=args.response_surrogate,
            )
            t1 = time.perf_counter()
            if allowed_steps and control_step not in allowed_steps:
                action_id = 0
                decision = {
                    "selection_mode": args.selection,
                    "selected_action_id": 0,
                    "q_anchor_mean": 0.0,
                    "q_selected_mean": 0.0,
                    "q_selected_std": 0.0,
                    "delta_q_mean": 0.0,
                    "score_selected": 0.0,
                    "supported_action_count": int(support_mask.sum()),
                    "step_gate_allowed": False,
                }
            else:
                action_id, decision = _select_action(
                    ensemble=ensemble,
                    observation=observation,
                    response_features=features,
                    support_mask=support_mask,
                    mode=args.selection,
                    z_value=args.z_value,
                )
                decision["step_gate_allowed"] = True
            q_elapsed = time.perf_counter() - t1
            t2 = time.perf_counter()
            if action_id == 0:
                next_observation, reward, done, info, _ = (
                    env.step_prepared_optimizer_anchor(
                        anchor_context,
                        sync_follower_state=True,
                    )
                )
            else:
                next_observation, reward, done, info = env.step_anchored_candidate(
                    catalog.residual(action_id),
                    anchor_context,
                )
            commit_elapsed = time.perf_counter() - t2

            action = catalog.action(action_id)
            validity = bool(info.get("validity_gate_pass", False))
            if not validity:
                valid_gate_failures += 1
            rewards.append(float(reward))
            action_ids.append(int(action_id))
            observation = np.asarray(next_observation, dtype=np.float32)
            anchor_seconds += float(anchor_elapsed)
            q_seconds += float(q_elapsed)
            commit_seconds += float(commit_elapsed)
            payload = {
                "format_version": EVAL_FORMAT,
                "control_step": int(control_step),
                "selected_action_id": int(action_id),
                "owner": action.owner,
                "template": action.template,
                "family": action.family,
                "magnitude": float(action.magnitude),
                "interval_ttt": float(-reward),
                "total_ttt_so_far": float(env.sim.total_ttt),
                "validity_gate_pass": validity,
                "anchor_seconds": float(anchor_elapsed),
                "q_seconds": float(q_elapsed),
                "commit_seconds": float(commit_elapsed),
                "response_surrogate": args.response_surrogate,
                **decision,
            }
            trace.write(json.dumps(payload, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps({
                "control_step": int(control_step),
                "action_id": int(action_id),
                "interval_ttt": round(float(-reward), 6),
                "anchor_seconds": round(float(anchor_elapsed), 3),
                "q_seconds": round(float(q_elapsed), 3),
                "commit_seconds": round(float(commit_elapsed), 3),
                "done": bool(done),
            }, sort_keys=True), flush=True)

    total_ttt = float(env.sim.total_ttt)
    control_ttt = float(np.sum(-np.asarray(rewards, dtype=np.float64)))
    warmup_ttt = float(total_ttt - control_ttt)
    counts = Counter(action_ids)
    summary = {
        "format_version": EVAL_FORMAT,
        "scenario": env.scenario_name,
        "t_total_sec": float(env.T_total),
        "warmup_steps": int(env.warmup),
        "control_steps": int(len(action_ids)),
        "total_ttt": total_ttt,
        "control_ttt": control_ttt,
        "warmup_ttt": warmup_ttt,
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(env._inventory()),
        "action_counts": {str(key): int(value) for key, value in counts.items()},
        "non_anchor_steps": int(sum(1 for action_id in action_ids if action_id != 0)),
        "validity_gate_failures": int(valid_gate_failures),
        "anchor_seconds": float(anchor_seconds),
        "q_seconds": float(q_seconds),
        "commit_seconds": float(commit_seconds),
        "wall_seconds": float(time.perf_counter() - started),
        "model_dir": str(args.model_dir),
        "catalog_fingerprint": catalog.fingerprint,
        "selection": args.selection,
        "response_surrogate": args.response_surrogate,
        "allow_control_steps": sorted(map(int, allowed_steps)),
        "support_counts": support.astype(int).tolist(),
        "arguments": {
            key: _json_ready(value)
            for key, value in vars(args).items()
        },
    }
    if args.pstack_summary is not None:
        pstack = json.loads(args.pstack_summary.read_text(encoding="utf-8"))
        pstack_total = float(pstack["total_ttt"])
        gap = float(total_ttt - pstack_total)
        summary.update({
            "pstack_total_ttt": pstack_total,
            "vs_pstack_ttt_gap": gap,
            "vs_pstack_percent": float(100.0 * gap / max(abs(pstack_total), 1.0e-9)),
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
    parser.add_argument("--selection", choices=("mean", "lcb"), default="mean")
    parser.add_argument("--z-value", type=float, default=1.96)
    parser.add_argument("--min-action-support", type=int, default=1)
    parser.add_argument(
        "--response-surrogate",
        choices=("anchor", "zero"),
        default="anchor",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-control-steps",
        default="",
        help="comma-separated control-step indices where DDQN may choose non-anchor",
    )
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_nominal_policy(_parse_args())


if __name__ == "__main__":
    main()
