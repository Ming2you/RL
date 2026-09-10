"""Promote frozen all-mode H1 freeway outcomes through H3 and owner-wise H12."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_candidate_ablation import (
    _h3_selection_replay_evidence,
    _horizon_label,
    _implementation_fingerprints,
)
from rl_leader.diagnose_pcent_guided_oracle import _require_rollout_coverage
from rl_leader.diagnose_pcent_reachability import (
    _response_scales_and_families,
    response_distance,
)
from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.diagnose_reachable_candidate_attribution import (
    _rollout_price_candidate,
    _rollout_pstack,
)
from rl_leader.evaluate_balanced_horizons import (
    BALANCED_HORIZON_FORMAT,
    _h1_exact,
    _h1_gain,
    _replay_event,
)
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


FREEWAY_CANDIDATE_MODE = "owner_block_v2_all"
FREEWAY_DOMAIN = "freeway"
REQUIRED_FREEWAY_OWNERS = ("R_D_W", "R_F_W", "R_D_E", "R_F_E")


def _progress(message: str) -> None:
    print(f"[balanced-freeway] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_freeway_candidate(row: dict) -> bool:
    return ":freeway:" in str(row.get("candidate_id", ""))


def _realized_outcome_key(row: dict) -> tuple[str, str]:
    return (
        str(row["response_memory_outcome_sha256"]),
        str(row["post_physical_sha256"]),
    )


def _source_alias_set(row: dict) -> frozenset[str]:
    aliases = row.get("candidate_aliases")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("freeway candidate has invalid source aliases")
    normalized = frozenset(str(candidate_id) for candidate_id in aliases)
    if len(normalized) != len(aliases):
        raise ValueError("freeway candidate source aliases are not a set")
    return normalized


def _freeway_owner(candidate_id: str) -> str | None:
    parts = str(candidate_id).split(":")
    if len(parts) < 3 or parts[1] != FREEWAY_DOMAIN:
        return None
    return parts[2]


def freeway_outcome_representatives(pool: dict) -> list[dict]:
    """Filter first, then preserve every freeway alias of each realized response."""
    logical_rows = [
        row for row in pool["rows"]
        if row.get("execution_branch") == "coordination"
    ]
    by_outcome: dict[tuple[str, str], list[dict]] = {}
    for row in logical_rows:
        by_outcome.setdefault(_realized_outcome_key(row), []).append(row)

    representatives = []
    for aliases in by_outcome.values():
        freeway_rows = [row for row in aliases if _is_freeway_candidate(row)]
        if not freeway_rows:
            continue
        source_aliases = {_source_alias_set(row) for row in aliases}
        if len(source_aliases) != 1:
            raise ValueError("freeway realized outcome has inconsistent source aliases")
        source_aliases = next(iter(source_aliases))
        grouped_ids = {str(row["candidate_id"]) for row in aliases}
        if grouped_ids != source_aliases:
            raise ValueError("freeway source aliases do not exactly match grouped candidates")
        representative = min(freeway_rows, key=lambda row: str(row["candidate_id"]))
        copied = dict(representative)
        copied["candidate_aliases"] = sorted(source_aliases)
        representatives.append(copied)
    return sorted(representatives, key=lambda row: str(row["candidate_id"]))


def select_minimal_freeway_execution_alias(pool: dict, representative: dict) -> dict:
    rows = {
        str(row["candidate_id"]): row
        for row in pool["rows"]
        if row.get("execution_branch") == "coordination" and _is_freeway_candidate(row)
    }
    aliases = [
        rows[candidate_id]
        for candidate_id in representative["candidate_aliases"]
        if candidate_id in rows
    ]
    if not aliases:
        raise ValueError("freeway realized outcome has no executable freeway alias")
    outcome = _realized_outcome_key(representative)
    if any(_realized_outcome_key(row) != outcome for row in aliases):
        raise ValueError("freeway candidate aliases do not share one realized outcome")

    def priority(row: dict) -> tuple:
        residual = np.asarray(row["continuous_residual"], dtype=float)
        return (
            bool(row.get("uses_pcent_target", False)),
            int(np.count_nonzero(residual)),
            float(np.linalg.norm(residual)),
            str(row["candidate_id"]),
        )

    return min(aliases, key=priority)


def _freeway_candidate_owners(row: dict) -> set[str]:
    return {
        owner
        for candidate_id in row.get("candidate_aliases", [])
        if (owner := _freeway_owner(candidate_id)) is not None
    }


def _prefilter_freeway_representatives(
    pool: dict,
    representatives: list[dict],
    *,
    anchor_response: np.ndarray,
    response_scales,
    response_families,
    max_per_owner: int | None,
) -> list[tuple[dict, dict, dict]]:
    prepared = []
    for representative in representatives:
        execution = select_minimal_freeway_execution_alias(pool, representative)
        distance = response_distance(
            np.asarray(execution["candidate_response"], dtype=float),
            anchor_response,
            response_scales,
            response_families,
        )
        if distance["overall_rmse"] <= 0.0:
            continue
        prepared.append((representative, execution, distance))
    if max_per_owner is None or int(max_per_owner) <= 0:
        return prepared

    by_owner: dict[str, list[tuple[dict, dict, dict]]] = {}
    for item in prepared:
        for owner in _freeway_candidate_owners(item[0]):
            by_owner.setdefault(owner, []).append(item)

    selected = {}
    for rows in by_owner.values():
        ranked = sorted(
            rows,
            key=lambda item: (
                -_h1_gain(item[1]),
                -float(item[2].get("overall_rmse", 0.0)),
                str(item[1]["candidate_id"]),
            ),
        )
        for item in ranked[:int(max_per_owner)]:
            selected[str(item[1]["candidate_id"])] = item
    return [selected[key] for key in sorted(selected)]


def select_h12_freeway_representatives(rows: list[dict]) -> tuple[list[dict], dict]:
    """Apply selector-v2 H3 ranking to every freeway owner represented by aliases."""
    by_owner: dict[str, list[dict]] = {}
    required = set(REQUIRED_FREEWAY_OWNERS)
    for row in rows:
        owners = {
            owner
            for candidate_id in row["candidate_aliases"]
            if (owner := _freeway_owner(candidate_id)) is not None
        }
        unexpected = owners - required
        if unexpected:
            raise ValueError(f"unexpected freeway owners: {sorted(unexpected)}")
        if (
            row["h3_label"]["validity_gate_pass"]
            and row["native_response_distance"]["overall_rmse"] > 0.0
        ):
            for owner in owners:
                by_owner.setdefault(owner, []).append(row)
    if set(by_owner) != required:
        raise ValueError(
            "freeway H12 selection does not cover exactly the required owners"
        )

    selected_by: dict[str, list[str]] = {}
    selected: dict[str, dict] = {}
    for owner in REQUIRED_FREEWAY_OWNERS:
        winner = max(
            by_owner[owner],
            key=lambda row: (row["h3_label"]["ttt_gain"], row["candidate_id"]),
        )
        candidate_id = winner["candidate_id"]
        selected[candidate_id] = winner
        selected_by.setdefault(candidate_id, []).append(f"owner_h3_best:{owner}")
    return [selected[candidate_id] for candidate_id in sorted(selected)], {
        "selected_by": selected_by,
        "owner_coverage": list(REQUIRED_FREEWAY_OWNERS),
    }


def _source_pool_and_event(
    source: dict,
    manifest: dict,
    *,
    allow_implementation_drift: bool = False,
) -> tuple[dict, dict, dict]:
    if (
        not allow_implementation_drift
        and source.get("implementation_sha256") != _implementation_fingerprints()
    ):
        raise ValueError("H1 source implementation drift")
    if len(source.get("candidate_pools", [])) != 1:
        raise ValueError("freeway horizon evaluator requires one H1 pool")
    pool = source["candidate_pools"][0]
    if pool.get("candidate_mode") != FREEWAY_CANDIDATE_MODE:
        raise ValueError("freeway horizon evaluator requires owner_block_v2_all source")
    frozen = next(
        (row for row in manifest["scenarios"] if row["scenario"] == pool["scenario"]),
        None,
    )
    if frozen is None:
        raise ValueError("H1 pool scenario is absent from frozen manifest")
    matches = [
        row for row in frozen["events"]
        if int(row["policy_step"]) == int(pool["policy_step"])
    ]
    if len(matches) != 1:
        raise ValueError("H1 pool does not identify exactly one frozen event")
    event = matches[0]
    differing = [
        field
        for field in (
            "simulation_step",
            "simulation_time_sec",
            "native_anchor_branch",
            "pre_runtime_sha256",
            "forecast_sha256",
            "anchor_fingerprint",
        )
        if pool.get(field) != event.get(field)
    ]
    for payload_field, event_hash_field in (
        ("observation", "observation_sha256"),
        ("anchor_envelope", "anchor_envelope_sha256"),
    ):
        if payload_field not in pool:
            differing.append(payload_field)
        elif _digest(pool[payload_field]) != event.get(event_hash_field):
            differing.append(event_hash_field)
    if differing:
        raise ValueError(f"H1 pool frozen event mismatch: {sorted(differing)}")
    if not event["coordination_eligible"]:
        raise ValueError("frozen event is not coordination eligible")
    return pool, frozen, event


def evaluate_freeway_horizons(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    max_h3_candidates_per_owner: int | None = None,
    replayed_env_context: tuple | None = None,
    allow_source_implementation_drift: bool = False,
) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(source, require_decision_horizon=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    pool, frozen, event = _source_pool_and_event(
        source,
        manifest,
        allow_implementation_drift=allow_source_implementation_drift,
    )
    if replayed_env_context is None:
        _progress(f"replay-start scenario={pool['scenario']} step={pool['policy_step']}")
        env, context = _replay_event(frozen, event)
    else:
        _progress(f"replay-reuse scenario={pool['scenario']} step={pool['policy_step']}")
        env, context = replayed_env_context
    if context.anchor_fingerprint != pool["anchor_fingerprint"]:
        raise ValueError("replayed frozen event anchor fingerprint drift")
    _progress(f"native-rollout-start scenario={pool['scenario']} step={pool['policy_step']}")
    native = _rollout_pstack(env, context, rollout_steps=12, horizons=(1, 3, 12))
    _require_rollout_coverage(native, (1, 3, 12), label="balanced freeway native")
    scales, families = _response_scales_and_families(env)
    anchor_response = np.asarray(native["first_step_response"], dtype=float)

    evaluated = []
    prepared = _prefilter_freeway_representatives(
        pool,
        freeway_outcome_representatives(pool),
        anchor_response=anchor_response,
        response_scales=scales,
        response_families=families,
        max_per_owner=max_h3_candidates_per_owner,
    )
    fast_single_h12 = (
        max_h3_candidates_per_owner is not None
        and int(max_h3_candidates_per_owner) == 1
    )
    _progress(
        "candidate-rollout-start "
        f"scenario={pool['scenario']} step={pool['policy_step']} "
        f"prepared={len(prepared)} fast_single_h12={fast_single_h12}"
    )
    for representative, execution, native_response_distance in prepared:
        rollout_steps = 12 if fast_single_h12 else 3
        horizons = (1, 3, 12) if fast_single_h12 else (1, 3)
        rollout = _rollout_price_candidate(
            env,
            np.asarray(execution["continuous_residual"], dtype=np.float32),
            context,
            rollout_steps=rollout_steps,
            horizons=horizons,
        )
        _require_rollout_coverage(rollout, horizons, label=execution["candidate_id"])
        if not _h1_exact(execution, rollout):
            raise ValueError(f"H3 rollout failed H1 replay: {execution['candidate_id']}")
        row = {
            "candidate_id": execution["candidate_id"],
            "representative_candidate_id": representative["candidate_id"],
            "candidate_aliases": representative["candidate_aliases"],
            "continuous_residual": execution["continuous_residual"],
            "native_response_distance": native_response_distance,
            "h3_label": _horizon_label(
                rollout["checkpoints"]["3"], native["checkpoints"]["3"], horizon=3
            ),
            "h12_label": None,
            "h1_replay_exact": True,
        }
        if fast_single_h12:
            row["h12_label"] = _horizon_label(
                rollout["checkpoints"]["12"], native["checkpoints"]["12"], horizon=12
            )
            row["h3_to_h12_replay"] = {
                "passed": True,
                "mode": "single_h12_rollout",
            }
        else:
            row["_h3_rollout"] = rollout
        evaluated.append(row)
    _progress(
        f"candidate-rollout-done scenario={pool['scenario']} "
        f"step={pool['policy_step']} evaluated={len(evaluated)}"
    )
    selected, selection = select_h12_freeway_representatives(evaluated)
    if fast_single_h12:
        selected_ids = {row["candidate_id"] for row in selected}
        for row in evaluated:
            if row["candidate_id"] not in selected_ids:
                row["h12_label"] = None
                row.pop("h3_to_h12_replay", None)
    else:
        for row in selected:
            rollout = _rollout_price_candidate(
                env,
                np.asarray(row["continuous_residual"], dtype=np.float32),
                context,
                rollout_steps=12,
                horizons=(1, 3, 12),
            )
            evidence = _h3_selection_replay_evidence(row["_h3_rollout"], rollout)
            if not evidence["passed"]:
                raise ValueError(f"H12 rollout failed H3 replay: {row['candidate_id']}")
            row["h12_label"] = _horizon_label(
                rollout["checkpoints"]["12"], native["checkpoints"]["12"], horizon=12
            )
            row["h3_to_h12_replay"] = evidence
    _progress(
        f"h12-selection-done scenario={pool['scenario']} "
        f"step={pool['policy_step']} selected={len(selected)}"
    )
    for row in evaluated:
        row.pop("_h3_rollout")

    result = {
        "format_version": BALANCED_HORIZON_FORMAT,
        "candidate_domain": FREEWAY_DOMAIN,
        "source_h1_artifact": str(source_path),
        "source_h1_sha256": _sha256_file(source_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "implementation_sha256": _implementation_fingerprints(),
        "core_implementation_sha256": _implementation_fingerprints(),
        "sidecar_sha256": _sha256_file(Path(__file__)),
        "scenario": pool["scenario"],
        "stratum": event["stratum"],
        "policy_step": pool["policy_step"],
        "anchor_fingerprint": pool["anchor_fingerprint"],
        "selection_policy": selection,
        "selector_version": "v2",
        "max_h3_candidates_per_owner": max_h3_candidates_per_owner,
        "outcomes": evaluated,
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
    parser.add_argument("--max-h3-candidates-per-owner", type=int)
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = evaluate_freeway_horizons(
        Path(args.source),
        Path(args.manifest),
        Path(args.output),
        max_h3_candidates_per_owner=args.max_h3_candidates_per_owner,
    )
    print(json.dumps({
        "passed": result["passed"],
        "outcomes": len(result["outcomes"]),
        "h12_selected": sum(row["h12_label"] is not None for row in result["outcomes"]),
        "elapsed_sec": time.monotonic() - started,
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
