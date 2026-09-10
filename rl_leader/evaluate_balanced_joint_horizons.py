"""Evaluate sparse urban/freeway joint price interventions from one frozen state."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_balanced_remaining_horizon import _recorded_path
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
from rl_leader.diagnose_reachable_candidate_attribution import (
    _rollout_price_candidate,
    _rollout_pstack,
)
from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.evaluate_balanced_horizons import (
    BALANCED_HORIZON_FORMAT,
    _replay_event,
)
from rl_leader.oracle_candidate_ablation import residual_sha256
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


JOINT_DOMAIN = "urban+freeway"
JOINT_SELECTOR_VERSION = "exhaustive_cross_of_single_h12_representatives_v1"


def _progress(message: str) -> None:
    print(f"[balanced-joint] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_balanced_source(
    path: Path,
    domain: str,
    *,
    allow_implementation_drift: bool = False,
) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("format_version") != BALANCED_HORIZON_FORMAT:
        raise ValueError(f"{domain} source is not a balanced horizon artifact")
    if source.get("passed") is not True:
        raise ValueError(f"{domain} source did not pass")
    if (
        not allow_implementation_drift
        and source.get("implementation_sha256") != _implementation_fingerprints()
    ):
        raise ValueError(f"{domain} source implementation drift")
    expected_sidecar = Path(__file__).with_name(
        "evaluate_balanced_horizons.py"
        if domain == "urban"
        else "evaluate_balanced_freeway_horizons.py"
    )
    if source.get("sidecar_sha256") != _sha256_file(expected_sidecar):
        raise ValueError(f"{domain} source sidecar drift")
    if source.get("selector_version") != "v2":
        raise ValueError(f"{domain} source selector mismatch")
    expected_candidate_domain = None if domain == "urban" else "freeway"
    if source.get("candidate_domain") != expected_candidate_domain:
        raise ValueError(f"{domain} source candidate domain mismatch")
    for row in source.get("outcomes", []):
        residual = np.asarray(row.get("continuous_residual", []), dtype=np.float32)
        if residual.size == 0 or not str(row.get("candidate_id", "")).endswith(
            residual_sha256(residual)[:12]
        ):
            raise ValueError(f"{domain} source candidate residual mismatch")
        if row.get("h1_replay_exact") is not True:
            raise ValueError(f"{domain} source H1 replay mismatch")
        if row.get("h12_label") is not None and row.get(
            "h3_to_h12_replay", {}
        ).get("passed") is not True:
            raise ValueError(f"{domain} source H3 replay mismatch")
    return source


def _selected_single_rows(source: dict, domain: str) -> list[dict]:
    marker = f":{domain}:"
    rows = [
        row for row in source.get("outcomes", [])
        if row.get("h12_label") is not None
        and marker in str(row.get("candidate_id", ""))
        and float(row.get("native_response_distance", {}).get("overall_rmse", 0.0)) > 0.0
    ]
    if not rows:
        raise ValueError(f"{domain} source has no nonidentity H12 representative")
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def joint_candidate_specs(urban_source: dict, freeway_source: dict) -> list[dict]:
    """Cross only already-selected single-domain H12 representatives."""
    specs = []
    for urban in _selected_single_rows(urban_source, "urban"):
        urban_residual = np.asarray(urban["continuous_residual"], dtype=np.float32)
        for freeway in _selected_single_rows(freeway_source, "freeway"):
            freeway_residual = np.asarray(freeway["continuous_residual"], dtype=np.float32)
            if urban_residual.shape != freeway_residual.shape:
                raise ValueError("joint component action shapes differ")
            overlap = np.flatnonzero((urban_residual != 0.0) & (freeway_residual != 0.0))
            if overlap.size:
                raise ValueError(f"joint component price blocks overlap: {overlap.tolist()}")
            residual = urban_residual + freeway_residual
            if np.max(np.abs(residual)) > 1.0 + 1.0e-7:
                raise ValueError("joint residual exceeds normalized action box")
            digest = residual_sha256(residual)
            specs.append({
                "candidate_id": f"joint:urban+freeway:{digest[:12]}",
                "component_candidate_ids": [
                    str(urban["candidate_id"]), str(freeway["candidate_id"]),
                ],
                "continuous_residual": residual.astype(float).tolist(),
            })
    if not specs:
        raise ValueError("joint candidate cross is empty")
    return sorted(specs, key=lambda row: row["candidate_id"])


def _merge_residual_specs(specs: list[dict]) -> list[dict]:
    """Reuse identical executions while preserving every logical component pair."""
    merged = {}
    for spec in specs:
        candidate_id = spec["candidate_id"]
        existing = merged.get(candidate_id)
        if existing is None:
            merged[candidate_id] = {
                **spec,
                "component_aliases": [spec["component_candidate_ids"]],
            }
        else:
            if existing["continuous_residual"] != spec["continuous_residual"]:
                raise ValueError("joint candidate ID collision")
            existing["component_aliases"].append(spec["component_candidate_ids"])
    return [merged[candidate_id] for candidate_id in sorted(merged)]


def _assert_matching_sources(urban: dict, freeway: dict) -> None:
    fields = (
        "scenario", "stratum", "policy_step", "anchor_fingerprint",
        "source_manifest_sha256",
    )
    differing = [field for field in fields if urban.get(field) != freeway.get(field)]
    if differing:
        raise ValueError(f"joint source provenance mismatch: {differing}")


def _validate_recorded_dependencies(source: dict, domain: str) -> tuple[Path, Path]:
    h1_path = _recorded_path(source["source_h1_artifact"])
    manifest_path = _recorded_path(source["source_manifest"])
    if _sha256_file(h1_path) != source["source_h1_sha256"]:
        raise ValueError(f"{domain} source H1 SHA mismatch")
    if _sha256_file(manifest_path) != source["source_manifest_sha256"]:
        raise ValueError(f"{domain} source manifest SHA mismatch")
    return h1_path, manifest_path


def _source_pool_and_event(
    h1: dict,
    manifest: dict,
    source: dict,
    domain: str,
    *,
    allow_implementation_drift: bool = False,
) -> tuple[dict, dict, dict]:
    if (
        not allow_implementation_drift
        and h1.get("implementation_sha256") != _implementation_fingerprints()
    ):
        raise ValueError(f"{domain} H1 source implementation drift")
    if len(h1.get("candidate_pools", [])) != 1:
        raise ValueError(f"{domain} H1 source must contain one pool")
    pool = h1["candidate_pools"][0]
    frozen = next(
        (row for row in manifest["scenarios"] if row["scenario"] == pool["scenario"]),
        None,
    )
    if frozen is None:
        raise ValueError(f"{domain} H1 scenario is absent from manifest")
    events = [
        row for row in frozen["events"]
        if int(row["policy_step"]) == int(pool["policy_step"])
    ]
    if len(events) != 1:
        raise ValueError(f"{domain} H1 pool does not identify one frozen event")
    event = events[0]
    differing = [
        field for field in (
            "simulation_step", "simulation_time_sec", "native_anchor_branch",
            "pre_runtime_sha256", "forecast_sha256", "anchor_fingerprint",
        )
        if pool.get(field) != event.get(field)
    ]
    for payload_field, hash_field in (
        ("observation", "observation_sha256"),
        ("anchor_envelope", "anchor_envelope_sha256"),
    ):
        if payload_field not in pool or _digest(pool.get(payload_field)) != event.get(hash_field):
            differing.append(hash_field)
    if (
        pool.get("scenario") != source.get("scenario")
        or int(pool.get("policy_step", -1)) != int(source.get("policy_step", -2))
        or pool.get("anchor_fingerprint") != source.get("anchor_fingerprint")
        or event.get("stratum") != source.get("stratum")
        or event.get("coordination_eligible") is not True
    ):
        differing.append("balanced_source_binding")
    if differing:
        raise ValueError(f"{domain} H1 frozen event mismatch: {sorted(set(differing))}")
    return pool, frozen, event


def _outcome_key(rollout: dict) -> tuple[str, str]:
    checkpoint = rollout["checkpoints"]["1"]
    return (
        str(checkpoint["follower_memory_sha256"]),
        str(checkpoint["physical_state_sha256"]),
    )


def evaluate_joint_horizons(
    urban_path: Path,
    freeway_path: Path,
    output_path: Path,
    *,
    replayed_env_context: tuple | None = None,
    allow_source_implementation_drift: bool = False,
) -> dict:
    urban = _load_balanced_source(
        urban_path,
        "urban",
        allow_implementation_drift=allow_source_implementation_drift,
    )
    freeway = _load_balanced_source(
        freeway_path,
        "freeway",
        allow_implementation_drift=allow_source_implementation_drift,
    )
    _assert_matching_sources(urban, freeway)

    urban_h1_path, urban_manifest_path = _validate_recorded_dependencies(
        urban, "urban"
    )
    h1_path, manifest_path = _validate_recorded_dependencies(freeway, "freeway")
    if urban_manifest_path.resolve() != manifest_path.resolve():
        raise ValueError("joint sources record different frozen manifests")
    urban_h1 = json.loads(urban_h1_path.read_text(encoding="utf-8"))
    h1 = json.loads(h1_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(urban_h1, require_decision_horizon=False)
    validate_oracle_label_artifact(h1, require_decision_horizon=False)
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    _, urban_frozen, urban_event = _source_pool_and_event(
        urban_h1,
        manifest,
        urban,
        "urban",
        allow_implementation_drift=allow_source_implementation_drift,
    )
    _, frozen, event = _source_pool_and_event(
        h1,
        manifest,
        freeway,
        "freeway",
        allow_implementation_drift=allow_source_implementation_drift,
    )
    if urban_frozen is not frozen or urban_event is not event:
        raise ValueError("joint H1 sources do not bind to the same frozen event")

    if replayed_env_context is None:
        _progress(f"replay-start scenario={freeway['scenario']} step={freeway['policy_step']}")
        env, context = _replay_event(frozen, event)
    else:
        _progress(f"replay-reuse scenario={freeway['scenario']} step={freeway['policy_step']}")
        env, context = replayed_env_context
    if context.anchor_fingerprint != freeway["anchor_fingerprint"]:
        raise ValueError("joint replay anchor fingerprint drift")
    _progress(f"native-rollout-start scenario={freeway['scenario']} step={freeway['policy_step']}")
    native = _rollout_pstack(env, context, rollout_steps=12, horizons=(1, 3, 12))
    _require_rollout_coverage(native, (1, 3, 12), label="balanced joint native")
    scales, families = _response_scales_and_families(env)
    anchor_response = np.asarray(native["first_step_response"], dtype=float)

    logical_specs = joint_candidate_specs(urban, freeway)
    specs = _merge_residual_specs(logical_specs)
    _progress(
        f"candidate-rollout-start scenario={freeway['scenario']} "
        f"step={freeway['policy_step']} specs={len(specs)}"
    )
    by_outcome: dict[tuple[str, str], dict] = {}
    for spec in specs:
        residual = np.asarray(spec["continuous_residual"], dtype=np.float32)
        rollout = _rollout_price_candidate(
            env, residual, context, rollout_steps=3, horizons=(1, 3)
        )
        _require_rollout_coverage(rollout, (1, 3), label=spec["candidate_id"])
        key = _outcome_key(rollout)
        existing = by_outcome.get(key)
        if existing is not None:
            existing["candidate_aliases"].append(spec["candidate_id"])
            existing["component_aliases"].extend(spec["component_aliases"])
            continue
        by_outcome[key] = {
            **spec,
            "representative_candidate_id": spec["candidate_id"],
            "candidate_aliases": [spec["candidate_id"]],
            "response_memory_outcome_sha256": key[0],
            "post_physical_sha256": key[1],
            "native_response_distance": response_distance(
                np.asarray(rollout["first_step_response"], dtype=float),
                anchor_response,
                scales,
                families,
            ),
            "h3_label": _horizon_label(
                rollout["checkpoints"]["3"], native["checkpoints"]["3"], horizon=3
            ),
            "h12_label": None,
            "_h3_rollout": rollout,
        }

    evaluated = sorted(by_outcome.values(), key=lambda row: row["candidate_id"])
    _progress(
        f"candidate-rollout-done scenario={freeway['scenario']} "
        f"step={freeway['policy_step']} evaluated={len(evaluated)}"
    )
    for row in evaluated:
        rollout = _rollout_price_candidate(
            env,
            np.asarray(row["continuous_residual"], dtype=np.float32),
            context,
            rollout_steps=12,
            horizons=(1, 3, 12),
        )
        evidence = _h3_selection_replay_evidence(row["_h3_rollout"], rollout)
        if not evidence["passed"]:
            raise ValueError(f"joint H12 rollout failed H3 replay: {row['candidate_id']}")
        row["h12_label"] = _horizon_label(
            rollout["checkpoints"]["12"], native["checkpoints"]["12"], horizon=12
        )
        row["h3_to_h12_replay"] = evidence
        row.pop("_h3_rollout")
    _progress(
        f"h12-rollout-done scenario={freeway['scenario']} "
        f"step={freeway['policy_step']} evaluated={len(evaluated)}"
    )

    result = {
        "format_version": BALANCED_HORIZON_FORMAT,
        "candidate_domain": JOINT_DOMAIN,
        "source_urban_artifact": str(urban_path),
        "source_urban_sha256": _sha256_file(urban_path),
        "source_freeway_artifact": str(freeway_path),
        "source_freeway_sha256": _sha256_file(freeway_path),
        "source_h1_artifact": str(h1_path),
        "source_h1_sha256": _sha256_file(h1_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "implementation_sha256": _implementation_fingerprints(),
        "sidecar_sha256": _sha256_file(Path(__file__)),
        "scenario": freeway["scenario"],
        "stratum": freeway["stratum"],
        "policy_step": freeway["policy_step"],
        "anchor_fingerprint": freeway["anchor_fingerprint"],
        "selection_policy": {
            "version": JOINT_SELECTOR_VERSION,
            "logical_cross_count": len(logical_specs),
            "unique_residual_count": len(specs),
            "unique_realized_outcomes": len(evaluated),
        },
        "selector_version": JOINT_SELECTOR_VERSION,
        "outcomes": evaluated,
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urban", required=True)
    parser.add_argument("--freeway", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = evaluate_joint_horizons(
        Path(args.urban), Path(args.freeway), Path(args.output)
    )
    print(json.dumps({
        "passed": result["passed"],
        "outcomes": len(result["outcomes"]),
        "h12_positive": sum(row["h12_label"]["positive"] for row in result["outcomes"]),
        "elapsed_sec": time.monotonic() - started,
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
