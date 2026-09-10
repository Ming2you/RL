"""Select only tail-positive residual candidates over a P-Stack fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_leader.build_tail_pairwise_dataset import TAIL_PAIRWISE_FORMAT


LONG_HORIZON_SELECTOR_FORMAT = "pstack_anchored_tail_selector_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("format_version") != TAIL_PAIRWISE_FORMAT:
        raise ValueError("selector input is not a tail-pairwise dataset")
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ValueError("selector input has no row list")
    return dataset


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _residual_norm(row: dict) -> float:
    residual = np.asarray(row.get("candidate_residual", []), dtype=float)
    if residual.ndim != 1 or not np.all(np.isfinite(residual)):
        raise ValueError("candidate_residual must be a finite vector")
    return float(np.linalg.norm(residual))


def _passes_tail_gate(
    row: dict,
    *,
    min_margin_ratio: float,
    min_gain_veh_h: float,
    max_terminal_inventory_delta_veh: float | None,
) -> tuple[bool, str]:
    if row.get("target_valid") is not True:
        return False, "invalid_or_quarantined_tail_label"
    if int(row.get("target", 0)) != 1:
        return False, "tail_label_not_positive"
    gain = _finite_float(row.get("gain_veh_h"), "gain_veh_h")
    required = _finite_float(row.get("required_gain_veh_h"), "required_gain_veh_h")
    margin_ratio = _finite_float(row.get("margin_ratio"), "margin_ratio")
    if gain <= max(float(min_gain_veh_h), required):
        return False, "gain_below_required_margin"
    if margin_ratio < float(min_margin_ratio):
        return False, "margin_ratio_below_threshold"
    if max_terminal_inventory_delta_veh is not None:
        delta = row.get("terminal_inventory_delta_veh")
        if delta is None:
            return False, "missing_terminal_inventory_delta"
        if _finite_float(delta, "terminal_inventory_delta_veh") > float(
            max_terminal_inventory_delta_veh
        ):
            return False, "terminal_inventory_delta_above_threshold"
    return True, "tail_positive"


def _candidate_sort_key(row: dict) -> tuple[float, float, float, str]:
    return (
        _finite_float(row.get("gain_veh_h"), "gain_veh_h"),
        _finite_float(row.get("margin_ratio"), "margin_ratio"),
        -_residual_norm(row),
        str(row.get("candidate_id", "")),
    )


def _selection_row(
    group_id: str,
    rows: list[dict],
    *,
    min_margin_ratio: float,
    min_gain_veh_h: float,
    max_terminal_inventory_delta_veh: float | None,
) -> dict:
    accepted = []
    rejected = []
    for row in rows:
        ok, reason = _passes_tail_gate(
            row,
            min_margin_ratio=min_margin_ratio,
            min_gain_veh_h=min_gain_veh_h,
            max_terminal_inventory_delta_veh=max_terminal_inventory_delta_veh,
        )
        compact = {
            "candidate_id": row.get("candidate_id"),
            "tail_status": row.get("tail_status"),
            "target_valid": bool(row.get("target_valid")),
            "target": row.get("target"),
            "gain_veh_h": row.get("gain_veh_h"),
            "required_gain_veh_h": row.get("required_gain_veh_h"),
            "margin_ratio": row.get("margin_ratio"),
            "terminal_inventory_delta_veh": row.get(
                "terminal_inventory_delta_veh"
            ),
            "reason": reason,
        }
        if ok:
            accepted.append(row)
        else:
            rejected.append(compact)
    best = max(accepted, key=_candidate_sort_key) if accepted else None
    reference = rows[0]
    if best is None:
        return {
            "event_group_id": group_id,
            "scenario": reference.get("scenario"),
            "stratum": reference.get("stratum"),
            "policy_step": reference.get("policy_step"),
            "decision": "pstack_fallback",
            "selected_candidate_id": None,
            "selected_candidate_residual": None,
            "selected_gain_veh_h": None,
            "selected_required_gain_veh_h": None,
            "selected_margin_ratio": None,
            "selected_terminal_inventory_delta_veh": None,
            "accepted_candidate_count": 0,
            "rejected_candidates": rejected,
        }
    return {
        "event_group_id": group_id,
        "scenario": best.get("scenario"),
        "stratum": best.get("stratum"),
        "policy_step": best.get("policy_step"),
        "decision": "select_candidate",
        "selected_candidate_id": best.get("candidate_id"),
        "selected_candidate_residual": best.get("candidate_residual"),
        "selected_gain_veh_h": best.get("gain_veh_h"),
        "selected_required_gain_veh_h": best.get("required_gain_veh_h"),
        "selected_margin_ratio": best.get("margin_ratio"),
        "selected_terminal_inventory_delta_veh": best.get(
            "terminal_inventory_delta_veh"
        ),
        "accepted_candidate_count": len(accepted),
        "rejected_candidates": rejected,
    }


def select_long_horizon_candidates(
    dataset_path: Path,
    output_path: Path,
    *,
    min_margin_ratio: float = 0.0,
    min_gain_veh_h: float = 0.0,
    max_terminal_inventory_delta_veh: float | None = None,
) -> dict:
    dataset = _load_dataset(dataset_path)
    grouped: dict[str, list[dict]] = {}
    for row in dataset["rows"]:
        group_id = str(row.get("event_group_id", ""))
        if not group_id:
            raise ValueError("tail-pairwise row is missing event_group_id")
        grouped.setdefault(group_id, []).append(row)

    selections = [
        _selection_row(
            group_id,
            rows,
            min_margin_ratio=min_margin_ratio,
            min_gain_veh_h=min_gain_veh_h,
            max_terminal_inventory_delta_veh=max_terminal_inventory_delta_veh,
        )
        for group_id, rows in sorted(grouped.items())
    ]
    selected = [row for row in selections if row["decision"] == "select_candidate"]
    result = {
        "format_version": LONG_HORIZON_SELECTOR_FORMAT,
        "selector_contract": (
            "oracle teacher selector over completed tail-pairwise labels; "
            "not a deployable learned model"
        ),
        "input_dataset": str(dataset_path),
        "input_dataset_sha256": _sha256_file(dataset_path),
        "input_dataset_format": dataset.get("format_version"),
        "parameters": {
            "min_margin_ratio": float(min_margin_ratio),
            "min_gain_veh_h": float(min_gain_veh_h),
            "max_terminal_inventory_delta_veh": max_terminal_inventory_delta_veh,
        },
        "selections": selections,
        "summary": {
            "event_groups": len(selections),
            "selected_event_groups": len(selected),
            "pstack_fallback_event_groups": len(selections) - len(selected),
            "valid_input_rows": sum(row.get("target_valid") is True for row in dataset["rows"]),
            "positive_input_rows": sum(row.get("target") == 1 for row in dataset["rows"]),
            "selected_candidate_ids": [
                row["selected_candidate_id"] for row in selected
            ],
        },
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-margin-ratio", type=float, default=0.0)
    parser.add_argument("--min-gain-veh-h", type=float, default=0.0)
    parser.add_argument("--max-terminal-inventory-delta-veh", type=float)
    args = parser.parse_args(argv)
    result = select_long_horizon_candidates(
        Path(args.dataset),
        Path(args.output),
        min_margin_ratio=args.min_margin_ratio,
        min_gain_veh_h=args.min_gain_veh_h,
        max_terminal_inventory_delta_veh=args.max_terminal_inventory_delta_veh,
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
