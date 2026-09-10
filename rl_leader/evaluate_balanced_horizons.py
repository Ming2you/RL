"""Promote frozen H1 owner-block outcomes through H3 and selective H12."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.balanced_oracle_manifest import (
    _freeze_event,
    validate_frozen_manifest,
)
from rl_leader.diagnose_candidate_ablation import (
    _h3_selection_replay_evidence,
    _horizon_label,
    _implementation_fingerprints,
)
from rl_leader.diagnose_oracle_remaining_horizon import (
    select_minimal_execution_alias,
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
from rl_leader.env import RLLeaderEnv
from rl_leader.oracle_label_contract import validate_oracle_label_artifact
from rl_leader.run_balanced_owner_block_pilot import _frozen_event_differences


BALANCED_HORIZON_FORMAT = "balanced_owner_block_h3_selective_h12_v1"
URBAN_DOMAIN = "urban"


def _progress(message: str) -> None:
    print(f"[balanced-urban] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_h12_representatives(rows: list[dict]) -> tuple[list[dict], dict]:
    eligible = [row for row in rows if row["h3_label"]["validity_gate_pass"]]
    if not eligible:
        return [], {"selected_by": {}}
    criteria = {
        "h3_ttt_best": max(
            eligible,
            key=lambda row: (row["h3_label"]["ttt_gain"], row["candidate_id"]),
        ),
        "h3_inventory_best": min(
            eligible,
            key=lambda row: (
                row["h3_label"]["terminal_inventory_delta"], row["candidate_id"]
            ),
        ),
        "native_response_diversity": max(
            eligible,
            key=lambda row: (row["native_response_distance"]["overall_rmse"], row["candidate_id"]),
        ),
    }
    selected_by: dict[str, list[str]] = {}
    by_id = {}
    for criterion, row in criteria.items():
        candidate_id = row["candidate_id"]
        by_id[candidate_id] = row
        selected_by.setdefault(candidate_id, []).append(criterion)
    selected = [by_id[candidate_id] for candidate_id in sorted(by_id)]
    return selected, {"selected_by": selected_by}


def _candidate_owner(candidate_id: str) -> str | None:
    parts = str(candidate_id).split(":")
    if len(parts) >= 3 and parts[1] in ("urban", "freeway"):
        return f"{parts[1]}:{parts[2]}"
    return None


def _is_urban_candidate(row: dict) -> bool:
    return f":{URBAN_DOMAIN}:" in str(row.get("candidate_id", ""))


def _h1_gain(row: dict) -> float:
    label = row.get("horizon_labels", {}).get("1", {})
    return float(label.get("ttt_gain", 0.0)) if isinstance(label, dict) else 0.0


def _prefilter_h1_representatives(
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
        execution = select_minimal_execution_alias(pool, representative)
        owner = _candidate_owner(execution["candidate_id"])
        if owner is None or not owner.startswith(f"{URBAN_DOMAIN}:"):
            continue
        distance = response_distance(
            np.asarray(execution["candidate_response"], dtype=float),
            anchor_response,
            response_scales,
            response_families,
        )
        prepared.append((representative, execution, distance))
    if max_per_owner is None or int(max_per_owner) <= 0:
        return prepared

    by_owner: dict[str, list[tuple[dict, dict, dict]]] = {}
    for item in prepared:
        owner = _candidate_owner(item[1]["candidate_id"])
        by_owner.setdefault(str(owner), []).append(item)
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


def select_h12_representatives_v2(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep the H3-best non-identity response for every realized owner."""
    by_owner: dict[str, list[dict]] = {}
    for row in rows:
        owner = _candidate_owner(row["candidate_id"])
        if (
            owner is not None
            and row["h3_label"]["validity_gate_pass"]
            and row["native_response_distance"]["overall_rmse"] > 0.0
        ):
            by_owner.setdefault(owner, []).append(row)
    selected_by = {}
    selected = []
    for owner in sorted(by_owner):
        winner = max(
            by_owner[owner],
            key=lambda row: (row["h3_label"]["ttt_gain"], row["candidate_id"]),
        )
        selected.append(winner)
        selected_by[winner["candidate_id"]] = [f"owner_h3_best:{owner}"]
    return selected, {"selected_by": selected_by, "owner_coverage": sorted(by_owner)}


def _replay_event(frozen_scenario: dict, event: dict):
    env = RLLeaderEnv(
        scenario_name=frozen_scenario["scenario"],
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    target = int(event["policy_step"])
    while int(env.step_idx - env.warmup) < target:
        env.step_optimizer_anchor(sync_follower_state=True)
    context = env.prepare_pstack_anchor_context()
    actual = _freeze_event(env, event["stratum"], target, context)
    differing = _frozen_event_differences(actual, event)
    if differing:
        raise ValueError(f"frozen event replay drift: {differing}")
    return env, context


def _h1_exact(row: dict, rollout: dict) -> bool:
    checkpoint = rollout["checkpoints"]["1"]
    return all((
        np.max(np.abs(
            np.asarray(rollout["first_step_response"], dtype=float)
            - np.asarray(row["candidate_response"], dtype=float)
        )) <= 1.0e-6,
        checkpoint["follower_memory_sha256"] == row["post_follower_sha256"],
        checkpoint["physical_state_sha256"] == row["post_physical_sha256"],
        float(checkpoint["ttt"]) == float(row["h1_step_ttt"]),
        float(checkpoint["terminal_inventory"]) == float(row["h1_terminal_inventory"]),
        bool(checkpoint["validity_gate_pass"]) == bool(row["validity_gate_pass"]),
    ))


def evaluate_horizons(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    selector_version: str = "v1",
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
    if (
        not allow_source_implementation_drift
        and source["implementation_sha256"] != _implementation_fingerprints()
    ):
        raise ValueError("H1 source implementation drift")
    if len(source["candidate_pools"]) != 1:
        raise ValueError("balanced horizon evaluator requires one H1 pool")
    pool = source["candidate_pools"][0]
    frozen = next(
        row for row in manifest["scenarios"] if row["scenario"] == pool["scenario"]
    )
    event = next(
        row for row in frozen["events"] if row["policy_step"] == pool["policy_step"]
    )
    if not event["coordination_eligible"]:
        raise ValueError("frozen event is not coordination eligible")
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
    _require_rollout_coverage(native, (1, 3, 12), label="balanced native")
    scales, families = _response_scales_and_families(env)
    anchor_response = np.asarray(native["first_step_response"], dtype=float)

    logical = [
        row for row in pool["rows"]
        if row.get("execution_branch") == "coordination"
        and _is_urban_candidate(row)
    ]
    by_outcome = {}
    for row in logical:
        key = (row["response_memory_outcome_sha256"], row["post_physical_sha256"])
        by_outcome.setdefault(key, row)
    evaluated = []
    prepared = _prefilter_h1_representatives(
        pool,
        list(by_outcome.values()),
        anchor_response=anchor_response,
        response_scales=scales,
        response_families=families,
        max_per_owner=max_h3_candidates_per_owner,
    )
    fast_single_h12 = (
        selector_version == "v2"
        and max_h3_candidates_per_owner is not None
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
    if selector_version == "v1":
        selected, selection = select_h12_representatives(evaluated)
    elif selector_version == "v2":
        selected, selection = select_h12_representatives_v2(evaluated)
    else:
        raise ValueError(f"unknown selector version: {selector_version}")
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
        "selector_version": selector_version,
        "max_h3_candidates_per_owner": max_h3_candidates_per_owner,
        "outcomes": evaluated,
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selector-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--max-h3-candidates-per-owner", type=int)
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = evaluate_horizons(
        Path(args.source),
        Path(args.manifest),
        Path(args.output),
        selector_version=args.selector_version,
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
