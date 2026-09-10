"""Build fail-closed realized-response pairs from completed drain-out artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_balanced_remaining_horizon import _recorded_path
from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.oracle_candidate_ablation import residual_sha256
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


TAIL_PAIRWISE_FORMAT = "tail_pairwise_realized_response_v1"
MODEL_FEATURE_FIELDS = (
    "observation",
    "anchor_envelope",
    "native_anchor_branch",
    "candidate_execution_branch",
    "candidate_residual",
)
PROHIBITED_MODEL_FIELDS = (
    "scenario",
    "stratum",
    "policy_step",
    "simulation_time_sec",
    "candidate_id",
    "candidate_aliases",
    "h3_evidence",
    "h12_evidence",
    "tail_status",
    "target",
    "gain_veh_h",
    "required_gain_veh_h",
    "margin_ratio",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_pool(h1: dict) -> dict:
    pools = h1.get("candidate_pools", [])
    if len(pools) != 1:
        raise ValueError("tail pair source H1 must contain exactly one pool")
    return pools[0]


def _event_context(manifest: dict, drain: dict, pool: dict) -> tuple[dict, dict]:
    scenario = next(
        (row for row in manifest["scenarios"] if row["scenario"] == drain["scenario"]),
        None,
    )
    if scenario is None:
        raise ValueError("tail pair scenario is absent from frozen manifest")
    events = [
        row for row in scenario["events"]
        if int(row["policy_step"]) == int(drain["policy_step"])
    ]
    if len(events) != 1:
        raise ValueError("tail pair drain does not identify one frozen event")
    event = events[0]
    checks = (
        pool.get("scenario") == drain.get("scenario"),
        int(pool.get("policy_step", -1)) == int(drain.get("policy_step", -2)),
        pool.get("anchor_fingerprint") == drain.get("anchor_fingerprint"),
        event.get("stratum") == drain.get("stratum"),
        event.get("coordination_eligible") is True,
        pool.get("observation") == event.get("observation"),
        pool.get("anchor_envelope") == event.get("anchor_envelope"),
        _digest(pool.get("observation")) == event.get("observation_sha256"),
        _digest(pool.get("anchor_envelope")) == event.get("anchor_envelope_sha256"),
    )
    if not all(checks):
        raise ValueError("tail pair H1/event/drain binding mismatch")
    return scenario, event


def _source_outcome(source: dict, candidate_id: str) -> dict:
    rows = [
        row for row in source.get("outcomes", [])
        if str(row.get("candidate_id")) == str(candidate_id)
    ]
    if len(rows) != 1:
        raise ValueError("drain candidate does not identify one source outcome")
    return rows[0]


def _realized_key(source_row: dict, pool: dict, residual: list[float]) -> tuple[str, str]:
    direct = (
        source_row.get("response_memory_outcome_sha256"),
        source_row.get("post_physical_sha256"),
    )
    if all(direct):
        return str(direct[0]), str(direct[1])
    aliases = set(map(str, source_row.get("candidate_aliases", [])))
    aliases.add(str(source_row["candidate_id"]))
    matching = [
        row for row in pool.get("rows", [])
        if str(row.get("candidate_id")) in aliases
    ]
    keys = {
        (
            str(row["response_memory_outcome_sha256"]),
            str(row["post_physical_sha256"]),
        )
        for row in matching
    }
    if len(keys) != 1:
        raise ValueError("source aliases do not identify one H1 realized outcome")
    residual_digest = residual_sha256(residual)
    if not any(
        residual_sha256(row["continuous_residual"]) == residual_digest
        for row in matching
    ):
        raise ValueError("drain residual is absent from its H1 realized outcome")
    return next(iter(keys))


def _margin_ratio(gain: float, required: float) -> float:
    if not np.isfinite(gain) or not np.isfinite(required) or required <= 0.0:
        raise ValueError("tail pair margin is invalid")
    return float((gain - required) / required)


def _build_rows(drain_path: Path) -> list[dict]:
    drain = _load_json(drain_path)
    if drain.get("status") != "complete" or drain.get("passed") is not True:
        raise ValueError("tail pair drain artifact is not complete and passed")
    source_path = _recorded_path(drain["source_artifact"])
    if _sha256_file(source_path) != drain.get("source_artifact_sha256"):
        raise ValueError("tail pair balanced source SHA mismatch")
    source = _load_json(source_path)
    h1_path = _recorded_path(drain["source_h1_artifact"])
    manifest_path = _recorded_path(drain["source_manifest"])
    if _sha256_file(h1_path) != drain.get("source_h1_sha256"):
        raise ValueError("tail pair H1 SHA mismatch")
    if _sha256_file(manifest_path) != drain.get("source_manifest_sha256"):
        raise ValueError("tail pair manifest SHA mismatch")
    if (
        source.get("source_h1_sha256") != drain.get("source_h1_sha256")
        or source.get("source_manifest_sha256") != drain.get("source_manifest_sha256")
    ):
        raise ValueError("tail pair drain/source dependency mismatch")
    h1 = _load_json(h1_path)
    manifest = _load_json(manifest_path)
    validate_oracle_label_artifact(h1, require_decision_horizon=False)
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    pool = _single_pool(h1)
    scenario, event = _event_context(manifest, drain, pool)
    event_group_id = _stable_id({
        "experiment_contract_sha256": scenario["experiment_contract_sha256"],
        "scenario": drain["scenario"],
        "stratum": drain["stratum"],
        "policy_step": int(drain["policy_step"]),
        "anchor_fingerprint": drain["anchor_fingerprint"],
    })

    rows = []
    for outcome in drain.get("outcomes", []):
        candidate_id = str(outcome["candidate_id"])
        source_row = _source_outcome(source, candidate_id)
        residual = list(map(float, outcome["continuous_residual"]))
        if residual_sha256(residual) != residual_sha256(source_row["continuous_residual"]):
            raise ValueError("tail pair drain/source residual mismatch")
        realized_key = _realized_key(source_row, pool, residual)
        verdict = outcome["verdict"]
        status = str(verdict["status"])
        if status not in ("positive", "negative", "invalid", "quarantine"):
            raise ValueError(f"unknown tail verdict status: {status}")
        replay_passed = outcome.get("h12_replay", {}).get("passed") is True
        target_valid = bool(replay_passed and status in ("positive", "negative"))
        gain = float(verdict["ttt_gain"]) if target_valid else None
        required = float(verdict["required_gain"]) if target_valid else None
        target = (1 if status == "positive" else 0) if target_valid else None
        candidate_rollout = outcome.get("rollout", {})
        native_rollout = drain.get("native_rollout", {})
        inventory_delta = None
        if candidate_rollout.get("terminal_inventory") and native_rollout.get("terminal_inventory"):
            inventory_delta = float(
                candidate_rollout["terminal_inventory"]["total"]
                - native_rollout["terminal_inventory"]["total"]
            )
        rows.append({
            "event_group_id": event_group_id,
            "realized_outcome_id": _stable_id(realized_key),
            "realized_outcome_key": list(realized_key),
            "observation": pool["observation"],
            "anchor_envelope": pool["anchor_envelope"],
            "native_anchor_branch": pool["native_anchor_branch"],
            "candidate_execution_branch": "coordination",
            "candidate_residual": residual,
            "scenario": drain["scenario"],
            "stratum": drain["stratum"],
            "policy_step": int(drain["policy_step"]),
            "simulation_time_sec": float(event["simulation_time_sec"]),
            "candidate_id": candidate_id,
            "candidate_aliases": list(map(str, outcome.get("candidate_aliases", []))),
            "residual_aliases": [{
                "candidate_id": candidate_id,
                "candidate_residual": residual,
            }],
            "h3_evidence": source_row.get("h3_label"),
            "h12_evidence": outcome.get("source_h12_label"),
            "tail_status": status,
            "target_valid": target_valid,
            "target": target,
            "gain_veh_h": gain,
            "required_gain_veh_h": required,
            "margin_ratio": (
                _margin_ratio(gain, required) if target_valid else None
            ),
            "terminal_inventory_delta_veh": inventory_delta,
            "sample_weight": 0.0,
            "source_provenance": {
                "drain_artifact": str(drain_path),
                "drain_artifact_sha256": _sha256_file(drain_path),
                "balanced_artifact": str(source_path),
                "balanced_artifact_sha256": _sha256_file(source_path),
                "h1_artifact": str(h1_path),
                "h1_artifact_sha256": _sha256_file(h1_path),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "drain_implementation_sha256": drain.get("implementation_sha256"),
                "source_implementation_sha256": drain.get(
                    "source_implementation_sha256"
                ),
            },
        })
    return rows


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    merged = {}
    for row in rows:
        key = (row["event_group_id"], row["realized_outcome_id"])
        current = merged.get(key)
        if current is None:
            merged[key] = row
            continue
        comparable = (
            "target_valid", "target", "tail_status", "gain_veh_h",
            "required_gain_veh_h", "margin_ratio",
        )
        if any(current[field] != row[field] for field in comparable):
            raise ValueError("conflicting labels for one realized outcome")
        current["candidate_aliases"] = sorted(set(
            current["candidate_aliases"] + row["candidate_aliases"]
        ))
        current["residual_aliases"].extend(row["residual_aliases"])
        current["source_provenance"] = {
            "merged_sources": [current["source_provenance"], row["source_provenance"]]
        }
        left = np.asarray(current["candidate_residual"], dtype=float)
        right = np.asarray(row["candidate_residual"], dtype=float)
        left_priority = (np.count_nonzero(left), float(np.linalg.norm(left)))
        right_priority = (np.count_nonzero(right), float(np.linalg.norm(right)))
        if right_priority < left_priority:
            current["candidate_id"] = row["candidate_id"]
            current["candidate_residual"] = row["candidate_residual"]
    return [merged[key] for key in sorted(merged)]


def _assign_group_weights(rows: list[dict]) -> None:
    groups = {}
    for row in rows:
        if row["target_valid"]:
            groups.setdefault(row["event_group_id"], []).append(row)
    for grouped in groups.values():
        weight = 1.0 / len(grouped)
        for row in grouped:
            row["sample_weight"] = weight


def build_tail_pairwise_dataset(inputs: list[Path], output: Path) -> dict:
    if not inputs:
        raise ValueError("tail pairwise builder requires at least one drain artifact")
    rows = _deduplicate_rows([
        row for path in inputs for row in _build_rows(path)
    ])
    _assign_group_weights(rows)
    scenarios = sorted({row["scenario"] for row in rows})
    folds = []
    for held_out in scenarios:
        folds.append({
            "held_out_scenario": held_out,
            "train_event_group_ids": sorted({
                row["event_group_id"] for row in rows
                if row["scenario"] != held_out
            }),
            "test_event_group_ids": sorted({
                row["event_group_id"] for row in rows
                if row["scenario"] == held_out
            }),
        })
    result = {
        "format_version": TAIL_PAIRWISE_FORMAT,
        "model_feature_fields": list(MODEL_FEATURE_FIELDS),
        "prohibited_model_fields": list(PROHIBITED_MODEL_FIELDS),
        "split_contract": "leave_one_scenario_out_v1",
        "rows": rows,
        "folds": folds,
        "summary": {
            "rows": len(rows),
            "valid_rows": sum(row["target_valid"] for row in rows),
            "positive_rows": sum(row["target"] == 1 for row in rows),
            "negative_rows": sum(row["target"] == 0 for row in rows),
            "event_groups": len({row["event_group_id"] for row in rows}),
            "scenarios": scenarios,
        },
        "passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_tail_pairwise_dataset(
        [Path(value) for value in args.inputs], Path(args.output)
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
