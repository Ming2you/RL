"""Evaluate a small H1-selected tail-label candidate set through H12."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_candidate_ablation import (
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
from rl_leader.oracle_candidate_ablation import residual_sha256
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


BUDGETED_TAIL_HORIZON_CONTRACT = "budgeted_tail_h12_v1"
BUDGETED_TAIL_DOMAIN = "budgeted_tail"
BUDGETED_SELECTOR_VERSION = "h1_budgeted_owner_diverse_singles_first_v2"


def _progress(message: str) -> None:
    print(f"[budgeted-tail] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_domain(row: dict[str, Any]) -> str | None:
    candidate_id = str(row.get("candidate_id", ""))
    if ":urban:" in candidate_id:
        return "urban"
    if ":freeway:" in candidate_id:
        return "freeway"
    return None


def _candidate_owner(row: dict[str, Any]) -> str:
    metadata = row.get("generator_metadata", {})
    if isinstance(metadata, dict) and metadata.get("owner"):
        return str(metadata["owner"])
    parts = str(row.get("candidate_id", "")).split(":")
    if len(parts) >= 3 and parts[1] in {"urban", "freeway"}:
        return parts[2]
    return str(row.get("candidate_id", "unknown"))


def _coordination_rows(pool: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in pool.get("rows", [])
        if row.get("execution_branch") == "coordination"
        and row.get("continuous_residual_valid", True) is True
        and _candidate_domain(row) in {"urban", "freeway"}
    ]


def _realized_outcome_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row["response_memory_outcome_sha256"]),
        str(row["post_physical_sha256"]),
    )


def _h1_valid(row: dict[str, Any]) -> bool:
    label = row.get("horizon_labels", {}).get("1", {})
    return bool(
        isinstance(label, dict)
        and label.get("label_valid", True) is True
        and label.get("validity_gate_pass", True) is True
    )


def _residual_array(row: dict[str, Any]) -> np.ndarray:
    residual = np.asarray(row["continuous_residual"], dtype=np.float32)
    if residual.ndim != 1 or residual.size == 0 or not np.all(np.isfinite(residual)):
        raise ValueError("budgeted tail candidate has invalid residual")
    return residual


def _single_priority(row: dict[str, Any]) -> tuple:
    residual = _residual_array(row)
    return (
        _h1_valid(row),
        row.get("horizon_labels", {}).get("1", {}).get("positive") is True,
        _h1_gain(row),
        -max(
            0.0,
            float(
                row.get("horizon_labels", {})
                .get("1", {})
                .get("terminal_inventory_delta", 0.0)
            ),
        ),
        -int(np.count_nonzero(residual)),
        -float(np.linalg.norm(residual)),
        str(row["candidate_id"]),
    )


def _representatives(pool: dict[str, Any]) -> list[dict[str, Any]]:
    by_outcome: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _coordination_rows(pool):
        key = _realized_outcome_key(row)
        current = by_outcome.get(key)
        if current is None or _single_priority(row) > _single_priority(current):
            by_outcome[key] = row
    return sorted(
        by_outcome.values(),
        key=lambda row: _single_priority(row),
        reverse=True,
    )


def _domain_top(
    representatives: list[dict[str, Any]],
    domain: str,
    limit: int,
) -> list[dict[str, Any]]:
    by_owner: dict[str, dict[str, Any]] = {}
    for row in representatives:
        if _candidate_domain(row) != domain or not _h1_valid(row):
            continue
        owner = _candidate_owner(row)
        current = by_owner.get(owner)
        if current is None or _single_priority(row) > _single_priority(current):
            by_owner[owner] = row
    rows = sorted(by_owner.values(), key=lambda row: _single_priority(row), reverse=True)
    return rows[:max(0, int(limit))]


def _joint_spec(
    urban: dict[str, Any],
    freeway: dict[str, Any],
) -> dict[str, Any] | None:
    urban_residual = _residual_array(urban)
    freeway_residual = _residual_array(freeway)
    if urban_residual.shape != freeway_residual.shape:
        raise ValueError("budgeted joint residual shapes differ")
    overlap = np.flatnonzero((urban_residual != 0.0) & (freeway_residual != 0.0))
    if overlap.size:
        return None
    residual = urban_residual + freeway_residual
    if np.max(np.abs(residual)) > 1.0 + 1.0e-7:
        return None
    digest = residual_sha256(residual)
    score = _h1_gain(urban) + _h1_gain(freeway)
    return {
        "candidate_id": f"budgeted:urban+freeway:{digest[:12]}",
        "representative_candidate_id": f"budgeted:urban+freeway:{digest[:12]}",
        "candidate_aliases": [f"budgeted:urban+freeway:{digest[:12]}"],
        "component_candidate_ids": [
            str(urban["candidate_id"]),
            str(freeway["candidate_id"]),
        ],
        "component_aliases": [
            list(map(str, urban.get("candidate_aliases", [urban["candidate_id"]]))),
            list(map(str, freeway.get("candidate_aliases", [freeway["candidate_id"]]))),
        ],
        "continuous_residual": residual.astype(float).tolist(),
        "selection_score": float(score),
        "selection_source": "joint_h1_sum",
        "source_h1_row": None,
    }


def _single_spec(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "representative_candidate_id": str(row["candidate_id"]),
        "candidate_aliases": list(map(str, row.get("candidate_aliases", [row["candidate_id"]]))),
        "component_candidate_ids": [str(row["candidate_id"])],
        "component_aliases": [
            list(map(str, row.get("candidate_aliases", [row["candidate_id"]])))
        ],
        "continuous_residual": _residual_array(row).astype(float).tolist(),
        "selection_score": float(_h1_gain(row)),
        "selection_source": f"single_{_candidate_domain(row)}_h1",
        "source_h1_row": row,
    }


def budgeted_candidate_specs(
    pool: dict[str, Any],
    *,
    max_h12_candidates_per_event: int = 2,
    domain_pool_size: int = 2,
) -> list[dict[str, Any]]:
    representatives = _representatives(pool)
    urban = _domain_top(representatives, "urban", domain_pool_size)
    freeway = _domain_top(representatives, "freeway", domain_pool_size)
    specs = [_single_spec(row) for row in urban + freeway]
    specs.extend(
        spec
        for left in urban
        for right in freeway
        if (spec := _joint_spec(left, right)) is not None
    )
    if not specs:
        raise ValueError("budgeted tail selector found no H1-valid candidates")

    by_residual: dict[str, dict[str, Any]] = {}
    for spec in specs:
        digest = residual_sha256(spec["continuous_residual"])
        current = by_residual.get(digest)
        if current is None or (
            float(spec["selection_score"]),
            str(spec["candidate_id"]),
        ) > (
            float(current["selection_score"]),
            str(current["candidate_id"]),
        ):
            by_residual[digest] = spec
    ranked = sorted(
        by_residual.values(),
        key=lambda spec: (
            str(spec["selection_source"]).startswith("single_"),
            float(spec["selection_score"]),
            str(spec["candidate_id"]),
        ),
        reverse=True,
    )
    if int(max_h12_candidates_per_event) > 0:
        ranked = ranked[:int(max_h12_candidates_per_event)]
    return ranked


def _outcome_digest(response: np.ndarray, follower_sha256: str) -> str:
    return _digest({
        "response": np.round(np.asarray(response, dtype=float), 6).tolist(),
        "post_follower_sha256": str(follower_sha256),
    })


def _h1_replay_status(spec: dict[str, Any], rollout: dict[str, Any]) -> bool | None:
    source_row = spec.get("source_h1_row")
    if source_row is None:
        return None
    return bool(_h1_exact(source_row, rollout))


def _source_pool_event(
    source: dict[str, Any],
    manifest: dict[str, Any],
    *,
    allow_implementation_drift: bool = False,
) -> tuple[dict, dict, dict]:
    if (
        not allow_implementation_drift
        and source.get("implementation_sha256") != _implementation_fingerprints()
    ):
        raise ValueError("H1 source implementation drift")
    if len(source.get("candidate_pools", [])) != 1:
        raise ValueError("budgeted tail evaluator requires one H1 pool")
    pool = source["candidate_pools"][0]
    if pool.get("candidate_mode") != "owner_block_v2_all":
        raise ValueError("budgeted tail evaluator requires owner_block_v2_all H1 source")
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


def evaluate_budgeted_tail_horizons(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    max_h12_candidates_per_event: int = 2,
    domain_pool_size: int = 2,
    replayed_env_context: tuple | None = None,
    allow_source_implementation_drift: bool = False,
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(source, require_decision_horizon=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    pool, frozen, event = _source_pool_event(
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
        raise ValueError("budgeted replay anchor fingerprint drift")

    _progress(f"native-rollout-start scenario={pool['scenario']} step={pool['policy_step']}")
    native = _rollout_pstack(env, context, rollout_steps=12, horizons=(1, 3, 12))
    _require_rollout_coverage(native, (1, 3, 12), label="budgeted native")
    scales, families = _response_scales_and_families(env)
    anchor_response = np.asarray(native["first_step_response"], dtype=float)

    specs = budgeted_candidate_specs(
        pool,
        max_h12_candidates_per_event=max_h12_candidates_per_event,
        domain_pool_size=domain_pool_size,
    )
    _progress(
        f"candidate-rollout-start scenario={pool['scenario']} "
        f"step={pool['policy_step']} specs={len(specs)}"
    )
    outcomes = []
    for spec in specs:
        residual = np.asarray(spec["continuous_residual"], dtype=np.float32)
        rollout = _rollout_price_candidate(
            env,
            residual,
            context,
            rollout_steps=12,
            horizons=(1, 3, 12),
        )
        _require_rollout_coverage(rollout, (1, 3, 12), label=spec["candidate_id"])
        h1_exact = _h1_replay_status(spec, rollout)
        if h1_exact is False:
            raise ValueError(f"budgeted single H12 failed H1 replay: {spec['candidate_id']}")
        checkpoint = rollout["checkpoints"]["1"]
        outcomes.append({
            "candidate_id": spec["candidate_id"],
            "representative_candidate_id": spec["representative_candidate_id"],
            "candidate_aliases": spec["candidate_aliases"],
            "component_candidate_ids": spec["component_candidate_ids"],
            "component_aliases": spec["component_aliases"],
            "continuous_residual": spec["continuous_residual"],
            "selection_score": float(spec["selection_score"]),
            "selection_source": spec["selection_source"],
            "response_memory_outcome_sha256": _outcome_digest(
                np.asarray(rollout["first_step_response"], dtype=float),
                str(checkpoint["follower_memory_sha256"]),
            ),
            "post_follower_sha256": str(checkpoint["follower_memory_sha256"]),
            "post_physical_sha256": str(checkpoint["physical_state_sha256"]),
            "candidate_response": list(map(float, rollout["first_step_response"])),
            "native_response_distance": response_distance(
                np.asarray(rollout["first_step_response"], dtype=float),
                anchor_response,
                scales,
                families,
            ),
            "h1_label": _horizon_label(
                checkpoint, native["checkpoints"]["1"], horizon=1
            ),
            "h3_label": _horizon_label(
                rollout["checkpoints"]["3"], native["checkpoints"]["3"], horizon=3
            ),
            "h12_label": _horizon_label(
                rollout["checkpoints"]["12"], native["checkpoints"]["12"], horizon=12
            ),
            "h1_replay_exact": h1_exact,
            "h3_to_h12_replay": {"passed": True, "mode": "single_h12_rollout"},
            "h1_step_ttt": float(checkpoint["ttt"]),
            "h1_terminal_inventory": float(checkpoint["terminal_inventory"]),
            "validity_gate_pass": bool(checkpoint["validity_gate_pass"]),
        })
    _progress(
        f"candidate-rollout-done scenario={pool['scenario']} "
        f"step={pool['policy_step']} outcomes={len(outcomes)}"
    )

    result = {
        "format_version": BALANCED_HORIZON_FORMAT,
        "contract": BUDGETED_TAIL_HORIZON_CONTRACT,
        "candidate_domain": BUDGETED_TAIL_DOMAIN,
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
        "selection_policy": {
            "version": BUDGETED_SELECTOR_VERSION,
            "max_h12_candidates_per_event": int(max_h12_candidates_per_event),
            "domain_pool_size": int(domain_pool_size),
            "selected_candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "selection_source": row["selection_source"],
                    "selection_score": row["selection_score"],
                }
                for row in outcomes
            ],
        },
        "selector_version": BUDGETED_SELECTOR_VERSION,
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
    parser.add_argument("--max-h12-candidates-per-event", type=int, default=2)
    parser.add_argument("--domain-pool-size", type=int, default=2)
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = evaluate_budgeted_tail_horizons(
        Path(args.source),
        Path(args.manifest),
        Path(args.output),
        max_h12_candidates_per_event=args.max_h12_candidates_per_event,
        domain_pool_size=args.domain_pool_size,
    )
    print(json.dumps({
        "passed": result["passed"],
        "scenario": result["scenario"],
        "policy_step": result["policy_step"],
        "outcomes": len(result["outcomes"]),
        "h12_positive": sum(
            row["h12_label"]["positive"] for row in result["outcomes"]
        ),
        "elapsed_sec": time.monotonic() - started,
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
