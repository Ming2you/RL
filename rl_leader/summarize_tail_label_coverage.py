"""Summarize H12-to-drain label coverage for the tail selector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BALANCED_HORIZON_FORMAT = "balanced_owner_block_h3_selective_h12_v1"
DRAIN_OUT_FORMAT = "balanced_positive_zero_demand_drain_out_v1"
COVERAGE_FORMAT = "pstack_anchored_tail_label_coverage_v1"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _path_key(path: str | Path) -> str:
    return str(Path(path)).replace("/", "\\").lower()


def _residual_key(row: dict[str, Any]) -> str | None:
    residual = row.get("continuous_residual")
    if residual is None:
        return None
    try:
        payload = [float(value) for value in residual]
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _iter_json_files(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
        else:
            yield from root.rglob("*.json")


def _h12_candidates(path: Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in artifact.get("outcomes", []):
        label = row.get("h12_label")
        if not isinstance(label, dict):
            continue
        if label.get("validity_gate_pass", True) is not True:
            continue
        rows.append({
            "source_artifact": str(path),
            "source_key": _path_key(path),
            "scenario": artifact.get("scenario"),
            "stratum": artifact.get("stratum"),
            "policy_step": artifact.get("policy_step"),
            "candidate_domain": artifact.get("candidate_domain") or "urban",
            "candidate_id": row.get("candidate_id"),
            "residual_sha256": _residual_key(row),
            "h12_positive": bool(label.get("positive")),
            "h12_gain_veh_h": label.get("ttt_gain"),
            "h12_required_gain_veh_h": label.get("required_gain"),
        })
    return rows


def discover_h12_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    candidates = []
    for path in _iter_json_files(roots):
        artifact = _load_json(path)
        if artifact is None:
            continue
        if (
            artifact.get("format_version") == BALANCED_HORIZON_FORMAT
            and artifact.get("passed") is True
        ):
            candidates.extend(_h12_candidates(path, artifact))
    return sorted(
        candidates,
        key=lambda row: (
            str(row["scenario"]),
            str(row["stratum"]),
            str(row["candidate_domain"]),
            str(row["candidate_id"]),
            str(row["source_artifact"]),
        ),
    )


def _drain_rows(path: Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    source = artifact.get("source_artifact")
    source_key = _path_key(source or "")
    requested = artifact.get("parameters", {}).get("requested_candidate_ids")
    if requested is None:
        requested_ids = None
    else:
        requested_ids = set(map(str, requested))
    outcome_ids = {
        str(row.get("candidate_id")): row for row in artifact.get("outcomes", [])
    }
    rows = []
    for candidate_id, row in outcome_ids.items():
        verdict = row.get("verdict", {})
        rows.append({
            "source_key": source_key,
            "candidate_id": candidate_id,
            "residual_sha256": _residual_key(row),
            "drain_artifact": str(path),
            "drain_status": artifact.get("status"),
            "drain_passed": artifact.get("passed") is True,
            "tail_status": verdict.get("status"),
            "tail_gain_veh_h": verdict.get("ttt_gain"),
            "tail_required_gain_veh_h": verdict.get("required_gain"),
        })
    if artifact.get("status") in {"running", "initializing"}:
        ids = requested_ids or set()
        completed_ids = set(outcome_ids)
        for candidate_id in sorted(ids - completed_ids):
            rows.append({
                "source_key": source_key,
                "candidate_id": candidate_id,
                "residual_sha256": None,
                "drain_artifact": str(path),
                "drain_status": artifact.get("status"),
                "drain_passed": False,
                "tail_status": None,
                "tail_gain_veh_h": None,
                "tail_required_gain_veh_h": None,
            })
        if requested_ids is None:
            rows.append({
                "source_key": source_key,
                "candidate_id": "*",
                "residual_sha256": None,
                "drain_artifact": str(path),
                "drain_status": artifact.get("status"),
                "drain_passed": False,
                "tail_status": None,
                "tail_gain_veh_h": None,
                "tail_required_gain_veh_h": None,
            })
    return rows


def discover_drain_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in _iter_json_files(roots):
        artifact = _load_json(path)
        if artifact is None:
            continue
        if artifact.get("format_version") == DRAIN_OUT_FORMAT:
            rows.extend(_drain_rows(path, artifact))
    return sorted(
        rows,
        key=lambda row: (
            str(row["source_key"]),
            str(row["candidate_id"]),
            str(row["drain_artifact"]),
        ),
    )


def summarize_tail_label_coverage(roots: list[Path], output_path: Path) -> dict:
    h12_rows = discover_h12_candidates(roots)
    drain_rows = discover_drain_rows(roots)
    drain_by_key = {
        (row["source_key"], str(row["candidate_id"])): row for row in drain_rows
    }
    completed_residual_keys = {
        (
            row.get("scenario"),
            row.get("stratum"),
            row.get("policy_step"),
            row.get("residual_sha256"),
        )
        for row in h12_rows
        for drain in [drain_by_key.get((row["source_key"], str(row["candidate_id"])))]
        if (
            drain is not None
            and drain.get("drain_status") == "complete"
            and drain.get("drain_passed") is True
            and row.get("residual_sha256") is not None
        )
    }

    coverage_rows = []
    for row in h12_rows:
        key = (row["source_key"], str(row["candidate_id"]))
        drain = drain_by_key.get(key)
        if drain is None and row["h12_positive"]:
            drain = drain_by_key.get((row["source_key"], "*"))
        if drain is None:
            status = "not_drained"
        elif drain["drain_status"] == "complete" and drain["drain_passed"]:
            status = "complete"
        elif drain["drain_status"] in {"running", "initializing"}:
            status = "running"
        else:
            status = "failed_or_invalid"
        residual_key = (
            row.get("scenario"),
            row.get("stratum"),
            row.get("policy_step"),
            row.get("residual_sha256"),
        )
        if (
            status == "not_drained"
            and row["h12_positive"]
            and row.get("residual_sha256") is not None
            and residual_key in completed_residual_keys
        ):
            status = "duplicate_complete"
        coverage_rows.append({
            **{k: v for k, v in row.items() if k != "source_key"},
            "coverage_status": status,
            "tail_status": None if drain is None else drain.get("tail_status"),
            "tail_gain_veh_h": None if drain is None else drain.get("tail_gain_veh_h"),
            "tail_required_gain_veh_h": (
                None if drain is None else drain.get("tail_required_gain_veh_h")
            ),
            "drain_artifact": None if drain is None else drain.get("drain_artifact"),
        })

    h12_positive = [row for row in coverage_rows if row["h12_positive"]]
    completed = [row for row in coverage_rows if row["coverage_status"] == "complete"]
    running = [row for row in coverage_rows if row["coverage_status"] == "running"]
    duplicate_complete = [
        row for row in coverage_rows if row["coverage_status"] == "duplicate_complete"
    ]
    undrained_positive = [
        row for row in h12_positive if row["coverage_status"] == "not_drained"
    ]
    result = {
        "format_version": COVERAGE_FORMAT,
        "roots": [str(root) for root in roots],
        "summary": {
            "h12_candidate_rows": len(coverage_rows),
            "h12_positive_candidate_rows": len(h12_positive),
            "strict_completed_candidate_rows": len(completed),
            "strict_positive_rows": sum(
                row.get("tail_status") == "positive" for row in completed
            ),
            "strict_negative_rows": sum(
                row.get("tail_status") == "negative" for row in completed
            ),
            "running_candidate_rows": len(running),
            "duplicate_complete_candidate_rows": len(duplicate_complete),
            "undrained_h12_positive_candidate_rows": len(undrained_positive),
            "event_groups_with_complete_strict_labels": len({
                (
                    row.get("scenario"),
                    row.get("stratum"),
                    row.get("policy_step"),
                )
                for row in completed
            }),
        },
        "undrained_h12_positive_candidates": undrained_positive,
        "coverage_rows": coverage_rows,
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "results/rl_phase0_implementation_20260829",
            "results/rl_phase0_implementation_20260830",
        ],
    )
    parser.add_argument(
        "--output",
        default=(
            "results/rl_phase0_implementation_20260830/"
            "tail_label_coverage_v1/current.json"
        ),
    )
    args = parser.parse_args(argv)
    result = summarize_tail_label_coverage(
        [Path(value) for value in args.roots], Path(args.output)
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
