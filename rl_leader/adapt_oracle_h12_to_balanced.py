"""Adapt a validated branch-aware oracle H12 artifact for balanced drain-out."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_candidate_ablation import _implementation_fingerprints
from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.evaluate_balanced_horizons import (
    BALANCED_HORIZON_FORMAT,
    _replay_event,
)
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


ADAPTATION_CONTRACT = "validated_oracle_h12_to_balanced_v1"
ALLOWED_CANDIDATE_MODES = {"owner_block_v2_urban", "owner_block_v2_all"}
PROVENANCE_FIELDS = (
    "simulation_step",
    "simulation_time_sec",
    "native_anchor_branch",
    "pre_runtime_sha256",
    "forecast_sha256",
    "anchor_fingerprint",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_domain(candidate_id: str) -> str:
    parts = str(candidate_id).split(":")
    if len(parts) < 3 or parts[1] not in {"urban", "freeway"}:
        raise ValueError(f"candidate has no supported owner domain: {candidate_id}")
    return parts[1]


def _source_alias_set(row: dict) -> frozenset[str]:
    aliases = row.get("candidate_aliases")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("candidate has invalid source aliases")
    normalized = frozenset(str(candidate_id) for candidate_id in aliases)
    if len(normalized) != len(aliases):
        raise ValueError("candidate source aliases are not a set")
    return normalized


def _realized_outcome_key(row: dict) -> tuple[str, str]:
    try:
        response = str(row["response_memory_outcome_sha256"])
        physical = str(row["post_physical_sha256"])
    except KeyError as exc:
        raise ValueError("coordination row lacks realized-outcome provenance") from exc
    if not response or not physical:
        raise ValueError("coordination row lacks realized-outcome provenance")
    return response, physical


def _validate_alias_provenance(pool: dict) -> dict[str, list[str]]:
    """Require source aliases to be the exact realized-outcome equivalence class."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in pool["rows"]:
        if row.get("execution_branch") == "coordination":
            groups.setdefault(_realized_outcome_key(row), []).append(row)

    aliases_by_candidate: dict[str, list[str]] = {}
    for rows in groups.values():
        source_aliases = {_source_alias_set(row) for row in rows}
        if len(source_aliases) != 1:
            raise ValueError("realized outcome has inconsistent source aliases")
        aliases = next(iter(source_aliases))
        grouped_ids = {str(row["candidate_id"]) for row in rows}
        if aliases != grouped_ids:
            raise ValueError("source aliases do not exactly match realized-outcome group")
        for candidate_id in grouped_ids:
            aliases_by_candidate[candidate_id] = sorted(aliases)
    return aliases_by_candidate


def _source_pool_and_event(source: dict, manifest: dict) -> tuple[dict, dict, dict]:
    if source.get("passed") is not True:
        raise ValueError("source oracle artifact did not pass")
    if source.get("implementation_sha256") != _implementation_fingerprints():
        raise ValueError("source oracle artifact implementation drift")
    pools = source.get("candidate_pools", [])
    if len(pools) != 1:
        raise ValueError("adapter requires exactly one candidate pool")
    pool = pools[0]
    if pool.get("candidate_mode") not in ALLOWED_CANDIDATE_MODES:
        raise ValueError("adapter requires an owner_block_v2 source candidate mode")

    frozen_matches = [
        row for row in manifest["scenarios"] if row.get("scenario") == pool.get("scenario")
    ]
    if len(frozen_matches) != 1:
        raise ValueError("source pool does not identify exactly one frozen scenario")
    frozen = frozen_matches[0]
    event_matches = [
        row for row in frozen["events"]
        if int(row.get("policy_step", -1)) == int(pool.get("policy_step", -2))
    ]
    if len(event_matches) != 1:
        raise ValueError("source pool does not identify exactly one frozen event")
    event = event_matches[0]
    if event.get("coordination_eligible") is not True:
        raise ValueError("frozen event is not coordination eligible")

    mismatches = [
        field for field in PROVENANCE_FIELDS
        if pool.get(field) != event.get(field)
    ]
    for payload_field, event_hash_field in (
        ("observation", "observation_sha256"),
        ("anchor_envelope", "anchor_envelope_sha256"),
    ):
        if payload_field not in pool or _digest(pool[payload_field]) != event.get(event_hash_field):
            mismatches.append(event_hash_field)
    if mismatches:
        raise ValueError(f"source pool frozen event mismatch: {sorted(mismatches)}")
    return pool, frozen, event


def _positive_outcomes(pool: dict, aliases_by_candidate: dict[str, list[str]]) -> tuple[list[dict], str]:
    outcomes = []
    domains = set()
    for row in pool["rows"]:
        label = row.get("horizon_labels", {}).get("12")
        if label is None or label.get("positive") is not True:
            continue
        if row.get("execution_branch") != "coordination":
            raise ValueError("positive H12 row is not a coordination candidate")
        candidate_id = str(row.get("candidate_id", ""))
        aliases = aliases_by_candidate.get(candidate_id)
        if aliases is None:
            raise ValueError("positive H12 row lacks validated alias provenance")
        domains.add(_candidate_domain(candidate_id))
        outcomes.append({
            "candidate_id": candidate_id,
            "representative_candidate_id": candidate_id,
            "candidate_aliases": aliases,
            "continuous_residual": copy.deepcopy(row["continuous_residual"]),
            "h12_label": copy.deepcopy(label),
            "h3_label": copy.deepcopy(row.get("horizon_labels", {}).get("3")),
            "h1_label": copy.deepcopy(row.get("horizon_labels", {}).get("1")),
            "h1_replay_evidence": copy.deepcopy(row.get("h1_replay_evidence")),
            "h3_selection_replay_evidence": copy.deepcopy(
                row.get("h3_selection_replay_evidence")
            ),
        })
    if not outcomes:
        raise ValueError("source oracle artifact contains no positive H12 outcome")
    if len(domains) != 1:
        raise ValueError(f"positive H12 rows span mixed candidate domains: {sorted(domains)}")
    domain = next(iter(domains))
    if pool["candidate_mode"] == "owner_block_v2_urban" and domain != "urban":
        raise ValueError("urban owner-block source selected a non-urban candidate")
    return outcomes, domain


def adapt_oracle_h12_to_balanced(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict:
    """Adapt existing H12 labels after an exact frozen-event replay."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(source, require_decision_horizon=True)
    validate_frozen_manifest(manifest, require_all_scenarios=True)
    pool, frozen, event = _source_pool_and_event(source, manifest)
    _, context = _replay_event(frozen, event)
    if context.anchor_fingerprint != pool["anchor_fingerprint"]:
        raise ValueError("frozen event replay reached a different anchor")
    aliases_by_candidate = _validate_alias_provenance(pool)
    outcomes, candidate_domain = _positive_outcomes(pool, aliases_by_candidate)
    current_implementation = _implementation_fingerprints()
    result = {
        "format_version": BALANCED_HORIZON_FORMAT,
        "contract": ADAPTATION_CONTRACT,
        "candidate_domain": candidate_domain,
        "source_h1_artifact": str(source_path),
        "source_h1_sha256": _sha256_file(source_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "implementation_sha256": current_implementation,
        "core_implementation_sha256": current_implementation,
        "adapter_sha256": _sha256_file(Path(__file__)),
        "scenario": pool["scenario"],
        "stratum": event["stratum"],
        "policy_step": int(pool["policy_step"]),
        "anchor_fingerprint": pool["anchor_fingerprint"],
        "frozen_event_replay": {
            "passed": True,
            "frozen_event_sha256": _digest(event),
            "replayed_frozen_event_sha256": _digest(event),
            "anchor_fingerprint_exact": True,
        },
        "selection_policy": {
            "selected_positive_h12_candidate_ids": [
                row["candidate_id"] for row in outcomes
            ],
            "source_candidate_mode": pool["candidate_mode"],
        },
        "adaptation_provenance": {
            "source_passed": True,
            "source_h12_labels_validated": True,
            "frozen_event_exact": True,
            "execution": "no candidate rollout; frozen event replayed exactly",
            "no_candidate_rollout": True,
            "frozen_event_replayed_exactly": True,
        },
        "outcomes": outcomes,
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = adapt_oracle_h12_to_balanced(
        Path(args.source), Path(args.manifest), Path(args.output)
    )
    print(json.dumps({
        "passed": result["passed"],
        "scenario": result["scenario"],
        "outcomes": len(result["outcomes"]),
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
