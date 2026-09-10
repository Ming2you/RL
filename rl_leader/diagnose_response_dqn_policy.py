"""Diagnose response-aware DQN action ranking on a frozen replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_leader.response_dqn import (
    conservative_ensemble_selection,
    load_trained_response_dqn,
)
from rl_leader.response_dqn_data import load_frozen_response_replay


DEFAULT_LABELS = {
    0: "anchor",
    1: "linear_first_negative",
    2: "linear_first_positive",
    3: "linear_second_negative",
    4: "linear_second_positive",
    5: "linear_corner_nn",
    6: "linear_corner_np",
    7: "linear_corner_pn",
    8: "linear_corner_pp",
    9: "quadratic_first_decrease",
    10: "quadratic_first_increase",
    11: "quadratic_second_decrease",
    12: "quadratic_second_increase",
    13: "cross_negative",
    14: "cross_positive",
}


def _labels_from_manifest(manifest: dict) -> dict[int, str]:
    catalog = manifest.get("catalog") or {}
    labels: dict[int, str] = {}
    for action in catalog.get("actions", []):
        action_id = int(action["action_id"])
        owner = str(action.get("owner", ""))
        template = str(action.get("template", f"action_{action_id}"))
        if owner and owner not in {"P-Stack", "anchor"}:
            labels[action_id] = f"{owner}:{template}"
        else:
            labels[action_id] = template
    return labels


def _label(action_id: int, labels: dict[int, str] | None = None) -> str:
    action_id = int(action_id)
    if labels and action_id in labels:
        return labels[action_id]
    return DEFAULT_LABELS.get(action_id, f"action_{action_id}")


def _parse_steps(value: str) -> set[int] | None:
    if not value:
        return None
    return {int(item) for item in value.split(",") if item}


def _rank_row(
    q_values: np.ndarray,
    valid: np.ndarray,
    top_k: int,
    labels: dict[int, str] | None,
) -> list[dict]:
    values = np.asarray(q_values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    delta_mean = mean - mean[0]
    delta_std = (values - values[:, [0]]).std(axis=0)
    lcb = delta_mean - 1.96 * delta_std
    ranked = [
        {
            "action_id": int(action_id),
            "label": _label(action_id, labels),
            "q_mean": float(mean[action_id]),
            "q_std": float(std[action_id]),
            "delta_mean": float(delta_mean[action_id]),
            "delta_std": float(delta_std[action_id]),
            "lcb_z196": float(lcb[action_id]),
        }
        for action_id in np.flatnonzero(mask)
    ]
    ranked.sort(key=lambda item: item["lcb_z196"], reverse=True)
    return ranked[: int(top_k)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-steps", default="")
    parser.add_argument("--z-value", type=float, default=1.96)
    parser.add_argument("--material-margin", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    replay = load_frozen_response_replay(args.data)
    labels = _labels_from_manifest(replay.manifest)
    steps = _parse_steps(args.control_steps)
    model_paths = sorted(args.model_dir.glob("response_dqn_member_*.pt"))
    if not model_paths:
        raise SystemExit(f"no response DQN checkpoints found in {args.model_dir}")
    ensemble = [
        load_trained_response_dqn(
            path,
            expected_catalog_fingerprint=replay.manifest["catalog_fingerprint"],
        )
        for path in model_paths
    ]
    support = np.min(
        np.stack([model.action_support_counts for model in ensemble]),
        axis=0,
    )
    min_support = int(ensemble[0].config.min_action_support)

    rows = []
    for index in range(replay.size):
        step = int(replay.control_step[index])
        if steps is not None and step not in steps:
            continue
        q_values = np.stack([
            model.q_values(
                replay.observation[index],
                replay.response_features[index],
            )[0]
            for model in ensemble
        ])
        valid = replay.action_mask[index].copy()
        decision = conservative_ensemble_selection(
            q_values,
            valid,
            action_support_counts=support,
            min_action_support=min_support,
            z_value=args.z_value,
            material_margin=args.material_margin,
        )
        rows.append({
            "index": int(index),
            "event_group": str(replay.event_group[index]),
            "episode": int(replay.episode[index]),
            "control_step": step,
            "behavior_action_id": int(replay.action_id[index]),
            "behavior_label": _label(int(replay.action_id[index]), labels),
            "reward_target": float(replay.reward[index]),
            "selected_action_id": int(decision.action_id),
            "selected_label": _label(int(decision.action_id), labels),
            "fallback": bool(decision.fallback),
            "fallback_reason": str(decision.fallback_reason),
            "anchor_q_mean": float(decision.anchor_q_mean),
            "selected_q_mean": float(decision.selected_q_mean),
            "delta_q_mean": float(decision.delta_q_mean),
            "delta_q_std": float(decision.delta_q_std),
            "lcb": float(decision.lcb),
            "valid_action_count": int(decision.valid_action_count),
            "top_actions": _rank_row(q_values, valid, args.top_k, labels),
        })

    summary = {
        "format_version": "response_dqn_policy_replay_diagnostic_v1",
        "data": str(args.data),
        "model_dir": str(args.model_dir),
        "z_value": float(args.z_value),
        "material_margin": float(args.material_margin),
        "min_action_support": min_support,
        "action_support_counts": support.astype(int).tolist(),
        "rows": rows,
        "selected_action_counts": {
            str(action_id): int(count)
            for action_id, count in zip(
                *np.unique(
                    np.asarray([row["selected_action_id"] for row in rows], dtype=np.int64),
                    return_counts=True,
                )
            )
        } if rows else {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "rows": len(rows),
        "selected_action_counts": summary["selected_action_counts"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
