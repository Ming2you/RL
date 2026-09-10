"""Generate balanced long-horizon tail labels from frozen P-Stack states."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from rl_leader.balanced_oracle_manifest import (
    BALANCED_SCENARIOS,
    _freeze_event,
    validate_frozen_manifest,
)
from rl_leader.diagnose_balanced_drain_out import run_balanced_drain_out
from rl_leader.evaluate_balanced_freeway_horizons import (
    _source_pool_and_event,
    evaluate_freeway_horizons,
)
from rl_leader.evaluate_balanced_horizons import _replay_event, evaluate_horizons
from rl_leader.evaluate_balanced_joint_horizons import evaluate_joint_horizons
from rl_leader.evaluate_budgeted_tail_horizons import (
    BUDGETED_SELECTOR_VERSION,
    evaluate_budgeted_tail_horizons,
)
from rl_leader.oracle_label_contract import (
    ORACLE_LABEL_DATASET_FORMAT,
    validate_oracle_label_artifact,
)
from rl_leader.run_balanced_owner_block_pilot import (
    _frozen_event_differences,
    run_balanced_pilot,
)


TAIL_LABEL_GENERATION_FORMAT = "pstack_tail_label_generation_pipeline_v1"
BALANCED_HORIZON_FORMAT = "balanced_owner_block_h3_selective_h12_v1"
DRAIN_OUT_FORMAT = "balanced_positive_zero_demand_drain_out_v1"
REPLAY_CACHE_FORMAT = "tail_label_generation_replay_cache_v1"


def _progress(message: str) -> None:
    print(f"[tail-label] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_complete_h1(
    path: Path,
    scenario: str,
    policy_step: int,
    *,
    candidate_mode: str | None = None,
) -> bool:
    payload = _load_json(path)
    if payload is None:
        return False
    pools = payload.get("candidate_pools", [])
    return bool(
        payload.get("format_version") == ORACLE_LABEL_DATASET_FORMAT
        and payload.get("passed") is True
        and len(pools) == 1
        and pools[0].get("scenario") == scenario
        and int(pools[0].get("policy_step", -1)) == int(policy_step)
        and (
            candidate_mode is None
            or pools[0].get("candidate_mode") == candidate_mode
        )
    )


def _is_complete_horizon(
    path: Path,
    scenario: str,
    policy_step: int,
    *,
    selector_version: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> bool:
    payload = _load_json(path)
    if not bool(
        payload is not None
        and payload.get("format_version") == BALANCED_HORIZON_FORMAT
        and payload.get("passed") is True
        and payload.get("scenario") == scenario
        and int(payload.get("policy_step", -1)) == int(policy_step)
    ):
        return False
    if selector_version is not None and payload.get("selector_version") != selector_version:
        return False
    recorded_parameters = {
        **payload.get("selection_policy", {}),
        **payload.get("parameters", {}),
    }
    for key, expected in (parameters or {}).items():
        if recorded_parameters.get(key) != expected:
            return False
    return True


def _is_complete_drain(path: Path, scenario: str, policy_step: int) -> bool:
    payload = _load_json(path)
    return bool(
        payload is not None
        and payload.get("format_version") == DRAIN_OUT_FORMAT
        and payload.get("status") == "complete"
        and payload.get("passed") is True
        and payload.get("scenario") == scenario
        and int(payload.get("policy_step", -1)) == int(policy_step)
    )


def _is_skipped_drain(path: Path, scenario: str, policy_step: int) -> bool:
    payload = _load_json(path)
    return bool(
        payload is not None
        and payload.get("format_version") == DRAIN_OUT_FORMAT
        and payload.get("status") in {
            "skipped_no_h12_positive",
            "skipped_no_drain_candidate",
        }
        and payload.get("scenario") == scenario
        and int(payload.get("policy_step", -1)) == int(policy_step)
    )


def _is_terminal_drain(path: Path, scenario: str, policy_step: int) -> bool:
    return _is_complete_drain(path, scenario, policy_step) or _is_skipped_drain(
        path, scenario, policy_step
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def eligible_manifest_events(
    manifest: dict[str, Any],
    *,
    scenarios: tuple[str, ...] = (),
    policy_steps: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    scenario_filter = set(scenarios)
    step_filter = set(policy_steps)
    rows = []
    for scenario in manifest.get("scenarios", []):
        name = str(scenario["scenario"])
        if scenario_filter and name not in scenario_filter:
            continue
        for event in scenario.get("events", []):
            step = int(event["policy_step"])
            if step_filter and step not in step_filter:
                continue
            if event.get("coordination_eligible") is True:
                rows.append({
                    "scenario": name,
                    "stratum": str(event["stratum"]),
                    "policy_step": step,
                    "native_anchor_branch": str(event["native_anchor_branch"]),
                    "anchor_fingerprint": str(event["anchor_fingerprint"]),
                })
    order = {name: index for index, name in enumerate(BALANCED_SCENARIOS)}
    return sorted(rows, key=lambda row: (order[row["scenario"]], row["policy_step"]))


def _h12_label(row: dict[str, Any]) -> dict[str, Any] | None:
    label = row.get("h12_label")
    if not isinstance(label, dict):
        return None
    if label.get("validity_gate_pass", True) is not True:
        return None
    return label


def _h12_gain(row: dict[str, Any]) -> float:
    label = _h12_label(row)
    if label is None:
        return float("-inf")
    return float(label.get("ttt_gain", 0.0))


def select_drain_candidate_ids(
    source: dict[str, Any],
    *,
    limit: int = 4,
    min_h12_gain_for_drain: float = 0.0,
) -> list[str]:
    rows = [row for row in source.get("outcomes", []) if _h12_label(row)]
    if not rows:
        raise ValueError("joint H12 source has no valid H12-labeled outcomes")
    min_gain = float(min_h12_gain_for_drain)
    positives = [
        row for row in rows
        if _h12_label(row).get("positive") is True and _h12_gain(row) >= min_gain
    ]
    promising = [
        row
        for row in rows
        if _h12_label(row).get("positive") is not True
        and _h12_gain(row) > 0.0
        and _h12_gain(row) >= min_gain
    ]
    positives = sorted(positives, key=lambda row: (-_h12_gain(row), row["candidate_id"]))
    promising = sorted(promising, key=lambda row: (-_h12_gain(row), row["candidate_id"]))
    if int(limit) <= 0:
        selected = positives + promising
    else:
        selected = positives + promising[:max(0, int(limit) - len(positives))]
    return [str(row["candidate_id"]) for row in selected]


def _event_dir(output_dir: Path, scenario: str, policy_step: int) -> Path:
    return output_dir / f"{scenario}_step{int(policy_step):02d}"


def _record_key(event: dict[str, Any]) -> str:
    return f"{event['scenario']}:{int(event['policy_step'])}"


def _record_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [records[key] for key in sorted(records)]


def _load_index(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload and payload.get("format_version") == TAIL_LABEL_GENERATION_FORMAT:
        return payload
    return {"format_version": TAIL_LABEL_GENERATION_FORMAT, "events": [], "passed": False}


def _replay_cache_paths(root: Path) -> tuple[Path, Path]:
    return root / "replay_cache.json", root / "replay_cache.pkl"


def _replay_cache_metadata(
    manifest_path: Path,
    h1_path: Path,
    event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": REPLAY_CACHE_FORMAT,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_h1_artifact": str(h1_path),
        "source_h1_sha256": _sha256_file(h1_path),
        "scenario": str(event["scenario"]),
        "policy_step": int(event["policy_step"]),
        "anchor_fingerprint": str(event["anchor_fingerprint"]),
    }


def _load_replay_cache(
    root: Path,
    manifest_path: Path,
    h1_path: Path,
    event: dict[str, Any],
    frozen_event: dict[str, Any],
) -> tuple | None:
    meta_path, payload_path = _replay_cache_paths(root)
    metadata = _load_json(meta_path)
    expected = _replay_cache_metadata(manifest_path, h1_path, event)
    if metadata != expected or not payload_path.is_file():
        return None
    try:
        env, context = pickle.loads(payload_path.read_bytes())
        actual = _freeze_event(
            env,
            frozen_event["stratum"],
            int(frozen_event["policy_step"]),
            context,
        )
        differing = _frozen_event_differences(actual, frozen_event)
    except Exception as exc:
        _progress(f"replay-cache-load-failed reason={type(exc).__name__}")
        return None
    if differing:
        _progress(f"replay-cache-drift fields={','.join(differing)}")
        return None
    return env, context


def _write_replay_cache(
    root: Path,
    manifest_path: Path,
    h1_path: Path,
    event: dict[str, Any],
    replayed_env_context: tuple,
) -> None:
    meta_path, payload_path = _replay_cache_paths(root)
    metadata = _replay_cache_metadata(manifest_path, h1_path, event)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload_tmp = payload_path.with_suffix(".tmp.pkl")
    meta_tmp = meta_path.with_suffix(".tmp.json")
    payload_tmp.write_bytes(pickle.dumps(replayed_env_context))
    meta_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    payload_tmp.replace(payload_path)
    meta_tmp.replace(meta_path)


def _write_replay_cache_metadata(
    root: Path,
    manifest_path: Path,
    h1_path: Path,
    event: dict[str, Any],
) -> None:
    meta_path, payload_path = _replay_cache_paths(root)
    if not payload_path.is_file():
        return
    metadata = _replay_cache_metadata(manifest_path, h1_path, event)
    meta_tmp = meta_path.with_suffix(".tmp.json")
    meta_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)


def _h1_paths(output_dir: Path, event: dict[str, Any]) -> tuple[Path, Path]:
    root = _event_dir(output_dir, event["scenario"], int(event["policy_step"]))
    scenario = str(event["scenario"])
    return root / "h1_all" / f"{scenario}.json", root / "h1" / f"{scenario}.json"


def _needs_h1(output_dir: Path, event: dict[str, Any]) -> bool:
    scenario = str(event["scenario"])
    policy_step = int(event["policy_step"])
    all_h1_path, legacy_h1_path = _h1_paths(output_dir, event)
    return not (
        _is_complete_h1(
            all_h1_path,
            scenario,
            policy_step,
            candidate_mode="owner_block_v2_all",
        )
        or _is_complete_h1(
            legacy_h1_path,
            scenario,
            policy_step,
            candidate_mode="owner_block_v2_all",
        )
    )


def _split_batch_h1_artifact(
    artifact: dict[str, Any],
    output_dir: Path,
    manifest_path: Path,
    events: list[dict[str, Any]],
) -> None:
    by_step = {
        int(pool["policy_step"]): pool
        for pool in artifact.get("candidate_pools", [])
    }
    for event in events:
        scenario = str(event["scenario"])
        step = int(event["policy_step"])
        if step not in by_step:
            raise ValueError(f"batch H1 artifact is missing step {step}")
        root = _event_dir(output_dir, scenario, step)
        split_path = root / "h1_all" / f"{scenario}.json"
        split = dict(artifact)
        split["candidate_pools"] = [by_step[step]]
        split["passed"] = True
        split.pop("pending_event", None)
        validate_oracle_label_artifact(split, require_decision_horizon=False)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        _write_replay_cache_metadata(root, manifest_path, split_path, event)


def populate_missing_h1_batches(
    manifest_path: Path,
    output_dir: Path,
    events: list[dict[str, Any]],
) -> None:
    missing = [event for event in events if _needs_h1(output_dir, event)]
    if not missing:
        return
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in missing:
        grouped.setdefault(str(event["scenario"]), []).append(event)
    for scenario, rows in sorted(grouped.items()):
        steps = tuple(sorted(int(row["policy_step"]) for row in rows))
        batch_dir = (
            output_dir
            / "_h1_batches"
            / f"{scenario}_steps_{'-'.join(map(str, steps))}"
        )
        _progress(
            f"h1-batch-start scenario={scenario} steps={','.join(map(str, steps))}"
        )
        run_balanced_pilot(
            manifest_path,
            batch_dir,
            scenarios=(scenario,),
            strata=("all",),
            policy_steps=steps,
            candidate_mode="owner_block_v2_all",
            run_h12=False,
            replay_cache_payloads={
                int(event["policy_step"]): _replay_cache_paths(
                    _event_dir(
                        output_dir,
                        str(event["scenario"]),
                        int(event["policy_step"]),
                    )
                )[1]
                for event in rows
            },
        )
        artifact_path = batch_dir / f"{scenario}.json"
        artifact = _load_json(artifact_path)
        if artifact is None:
            raise ValueError("batch H1 artifact was not written")
        _split_batch_h1_artifact(artifact, output_dir, manifest_path, rows)
        _progress(
            f"h1-batch-done scenario={scenario} steps={','.join(map(str, steps))}"
        )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in records if row.get("status") == "complete"]
    h1_complete = [row for row in records if row.get("status") == "h1_complete"]
    no_drain_candidate = [
        row
        for row in records
        if row.get("status") in {"no_h12_positive", "no_drain_candidate"}
    ]
    failed = [row for row in records if row.get("status") == "failed"]
    terminal = complete + h1_complete + no_drain_candidate + failed
    statuses = {}
    for row in complete:
        for status in row.get("drain_statuses", []):
            statuses[status] = statuses.get(status, 0) + 1
    return {
        "events": len(records),
        "complete_events": len(complete),
        "h1_complete_events": len(h1_complete),
        "no_drain_candidate_events": len(no_drain_candidate),
        "no_h12_positive_events": len(no_drain_candidate),
        "failed_events": len(failed),
        "terminal_events": len(terminal),
        "drain_outcomes": sum(len(row.get("drain_statuses", [])) for row in complete),
        "drain_status_counts": statuses,
    }


def _write_index(path: Path, result: dict[str, Any]) -> None:
    result["summary"] = _summarize(result.get("events", []))
    result["passed"] = (
        result["summary"]["events"] > 0
        and result["summary"]["events"] == result["summary"]["terminal_events"]
        and result["summary"]["failed_events"] == 0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _write_no_drain_candidate(
    path: Path,
    *,
    source_path: Path,
    source: dict[str, Any],
    drain_candidates_per_event: int,
    min_h12_gain_for_drain: float,
) -> dict[str, Any]:
    result = {
        "format_version": DRAIN_OUT_FORMAT,
        "status": "skipped_no_drain_candidate",
        "passed": False,
        "skip_reason": (
            "H12 source has no positive or positive-TTT-gain candidate; "
            "strict drain-out is reserved for promising H12 rows."
        ),
        "source_artifact": str(source_path),
        "source_artifact_sha256": _sha256_file(source_path),
        "scenario": source.get("scenario"),
        "stratum": source.get("stratum"),
        "policy_step": int(source.get("policy_step", -1)),
        "anchor_fingerprint": source.get("anchor_fingerprint"),
        "parameters": {
            "candidate_selection": "h12_positive_or_positive_gain",
            "drain_candidates_per_event": int(drain_candidates_per_event),
            "min_h12_gain_for_drain": float(min_h12_gain_for_drain),
            "requested_candidate_ids": [],
        },
        "h12_outcomes": len(source.get("outcomes", [])),
        "h12_positive_outcomes": sum(
            1
            for row in source.get("outcomes", [])
            if _h12_label(row) and _h12_label(row).get("positive") is True
        ),
        "h12_promising_outcomes": 0,
        "outcomes": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _run_event(
    manifest_path: Path,
    output_dir: Path,
    event: dict[str, Any],
    *,
    drain_candidates_per_event: int,
    max_h3_candidates_per_owner: int | None,
    max_h12_candidates_per_event: int,
    domain_pool_size: int,
    horizon_strategy: str,
    min_h12_gain_for_drain: float,
) -> dict[str, Any]:
    scenario = event["scenario"]
    policy_step = int(event["policy_step"])
    root = _event_dir(output_dir, scenario, policy_step)
    all_h1_dir = root / "h1_all"
    all_h1_path = all_h1_dir / f"{scenario}.json"
    legacy_all_h1_path = root / "h1" / f"{scenario}.json"
    urban_path = root / "urban_h12" / f"{scenario}.json"
    freeway_path = root / "freeway_h12" / f"{scenario}.json"
    joint_path = root / "joint_h12" / f"{scenario}.json"
    budgeted_path = root / "budgeted_h12" / f"{scenario}.json"
    drain_path = root / "drain" / f"{scenario}.json"

    started = time.monotonic()
    _progress(
        f"event-start scenario={scenario} step={policy_step} "
        f"max_h3_per_owner={max_h3_candidates_per_owner}"
    )
    if not _is_complete_h1(
        all_h1_path,
        scenario,
        policy_step,
        candidate_mode="owner_block_v2_all",
    ):
        if _is_complete_h1(
            legacy_all_h1_path,
            scenario,
            policy_step,
            candidate_mode="owner_block_v2_all",
        ):
            all_h1_path = legacy_all_h1_path
        else:
            _progress(f"h1-start scenario={scenario} step={policy_step}")
            run_balanced_pilot(
                manifest_path,
                all_h1_dir,
                scenarios=(scenario,),
                strata=("all",),
                policy_steps=(policy_step,),
                candidate_mode="owner_block_v2_all",
                run_h12=False,
                replay_cache_payloads={
                    policy_step: _replay_cache_paths(root)[1],
                },
            )
            _write_replay_cache_metadata(
                root, manifest_path, all_h1_path, event
            )
            _progress(f"h1-done scenario={scenario} step={policy_step}")
    if horizon_strategy not in ("budgeted", "component_joint"):
        raise ValueError(f"unknown horizon strategy: {horizon_strategy}")
    source_h12_path = budgeted_path if horizon_strategy == "budgeted" else joint_path
    budgeted_parameters = {
        "max_h12_candidates_per_event": int(max_h12_candidates_per_event),
        "domain_pool_size": int(domain_pool_size),
    }
    if horizon_strategy == "budgeted":
        horizon_ready = _is_complete_horizon(
            budgeted_path,
            scenario,
            policy_step,
            selector_version=BUDGETED_SELECTOR_VERSION,
            parameters=budgeted_parameters,
        )
    else:
        horizon_ready = (
            _is_complete_horizon(urban_path, scenario, policy_step)
            and _is_complete_horizon(freeway_path, scenario, policy_step)
            and _is_complete_horizon(joint_path, scenario, policy_step)
        )
    replayed_env_context = None

    def ensure_replay() -> tuple:
        nonlocal replayed_env_context
        if replayed_env_context is not None:
            return replayed_env_context
        source = _load_json(all_h1_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if source is None:
            raise ValueError("complete H1 source could not be loaded")
        _, frozen, frozen_event = _source_pool_and_event(
            source,
            manifest,
            allow_implementation_drift=True,
        )
        replayed_env_context = _load_replay_cache(
            root, manifest_path, all_h1_path, event, frozen_event
        )
        if replayed_env_context is None:
            _progress(f"replay-cache-start scenario={scenario} step={policy_step}")
            replayed_env_context = _replay_event(frozen, frozen_event)
            _write_replay_cache(
                root, manifest_path, all_h1_path, event, replayed_env_context
            )
        else:
            _progress(f"replay-cache-hit scenario={scenario} step={policy_step}")
        if replayed_env_context[1].anchor_fingerprint != event["anchor_fingerprint"]:
            raise ValueError("runner replay cache anchor fingerprint drift")
        _progress(f"replay-cache-done scenario={scenario} step={policy_step}")
        return replayed_env_context

    if horizon_strategy == "budgeted":
        if not _is_complete_horizon(
            budgeted_path,
            scenario,
            policy_step,
            selector_version=BUDGETED_SELECTOR_VERSION,
            parameters=budgeted_parameters,
        ):
            ensure_replay()
            _progress(f"budgeted-h12-start scenario={scenario} step={policy_step}")
            evaluate_budgeted_tail_horizons(
                all_h1_path,
                manifest_path,
                budgeted_path,
                max_h12_candidates_per_event=max_h12_candidates_per_event,
                domain_pool_size=domain_pool_size,
                replayed_env_context=replayed_env_context,
                allow_source_implementation_drift=True,
            )
            _progress(f"budgeted-h12-done scenario={scenario} step={policy_step}")
    elif not _is_complete_horizon(urban_path, scenario, policy_step):
        ensure_replay()
        _progress(f"urban-h12-start scenario={scenario} step={policy_step}")
        evaluate_horizons(
            all_h1_path,
            manifest_path,
            urban_path,
            selector_version="v2",
            max_h3_candidates_per_owner=max_h3_candidates_per_owner,
            replayed_env_context=replayed_env_context,
            allow_source_implementation_drift=True,
        )
        _progress(f"urban-h12-done scenario={scenario} step={policy_step}")
    if horizon_strategy == "component_joint" and not _is_complete_horizon(
        freeway_path, scenario, policy_step
    ):
        ensure_replay()
        _progress(f"freeway-h12-start scenario={scenario} step={policy_step}")
        evaluate_freeway_horizons(
            all_h1_path,
            manifest_path,
            freeway_path,
            max_h3_candidates_per_owner=max_h3_candidates_per_owner,
            replayed_env_context=replayed_env_context,
            allow_source_implementation_drift=True,
        )
        _progress(f"freeway-h12-done scenario={scenario} step={policy_step}")
    if horizon_strategy == "component_joint" and not _is_complete_horizon(
        joint_path, scenario, policy_step
    ):
        ensure_replay()
        _progress(f"joint-h12-start scenario={scenario} step={policy_step}")
        evaluate_joint_horizons(
            urban_path,
            freeway_path,
            joint_path,
            replayed_env_context=replayed_env_context,
            allow_source_implementation_drift=True,
        )
        _progress(f"joint-h12-done scenario={scenario} step={policy_step}")

    joint = _load_json(source_h12_path)
    if joint is None:
        raise ValueError("H12 source was not written")
    candidate_ids = select_drain_candidate_ids(
        joint,
        limit=drain_candidates_per_event,
        min_h12_gain_for_drain=min_h12_gain_for_drain,
    )
    if not candidate_ids:
        _progress(f"drain-skip-no-candidate scenario={scenario} step={policy_step}")
        drain = _write_no_drain_candidate(
            drain_path,
            source_path=source_h12_path,
            source=joint,
            drain_candidates_per_event=drain_candidates_per_event,
            min_h12_gain_for_drain=min_h12_gain_for_drain,
        )
    elif not _is_complete_drain(drain_path, scenario, policy_step):
        ensure_replay()
        _progress(
            f"drain-start scenario={scenario} step={policy_step} "
            f"candidates={len(candidate_ids)}"
        )
        run_balanced_drain_out(
            source_h12_path,
            drain_path,
            candidate_ids=tuple(candidate_ids),
            replayed_env_context=replayed_env_context,
        )
        _progress(f"drain-done scenario={scenario} step={policy_step}")
        drain = _load_json(drain_path)
    else:
        drain = _load_json(drain_path)
    if drain is None:
        raise ValueError("drain artifact was not written")
    event_status = "complete" if candidate_ids else "no_drain_candidate"
    return {
        **event,
        "status": event_status,
        "root": str(root),
        "h1_artifact": str(all_h1_path),
        "h1_sha256": _sha256_file(all_h1_path),
        "all_h1_artifact": str(all_h1_path),
        "all_h1_sha256": _sha256_file(all_h1_path),
        "urban_artifact": str(urban_path),
        "urban_sha256": (
            _sha256_file(urban_path)
            if urban_path.is_file() else None
        ),
        "freeway_artifact": str(freeway_path),
        "freeway_sha256": (
            _sha256_file(freeway_path)
            if freeway_path.is_file() else None
        ),
        "joint_artifact": str(joint_path),
        "joint_sha256": (
            _sha256_file(joint_path)
            if joint_path.is_file() else None
        ),
        "budgeted_artifact": str(budgeted_path),
        "budgeted_sha256": (
            _sha256_file(budgeted_path)
            if budgeted_path.is_file() else None
        ),
        "source_h12_artifact": str(source_h12_path),
        "source_h12_sha256": _sha256_file(source_h12_path),
        "drain_artifact": str(drain_path),
        "drain_sha256": _sha256_file(drain_path),
        "horizon_strategy": horizon_strategy,
        "max_h3_candidates_per_owner": max_h3_candidates_per_owner,
        "max_h12_candidates_per_event": int(max_h12_candidates_per_event),
        "domain_pool_size": int(domain_pool_size),
        "min_h12_gain_for_drain": float(min_h12_gain_for_drain),
        "budgeted_selector_version": (
            BUDGETED_SELECTOR_VERSION
            if horizon_strategy == "budgeted" else None
        ),
        "selected_candidate_ids": candidate_ids,
        "joint_h12_outcomes": len(joint.get("outcomes", [])),
        "joint_h12_positive": sum(
            1
            for row in joint.get("outcomes", [])
            if _h12_label(row) and _h12_label(row).get("positive") is True
        ),
        "drain_statuses": [
            row.get("verdict", {}).get("status")
            for row in drain.get("outcomes", [])
        ],
        "elapsed_sec": float(time.monotonic() - started),
    }


def run_tail_label_generation(
    manifest_path: Path,
    output_dir: Path,
    *,
    scenarios: tuple[str, ...] = (),
    policy_steps: tuple[int, ...] = (),
    max_events: int | None = None,
    drain_candidates_per_event: int = 4,
    max_h3_candidates_per_owner: int | None = 1,
    max_h12_candidates_per_event: int = 2,
    domain_pool_size: int = 2,
    horizon_strategy: str = "budgeted",
    min_h12_gain_for_drain: float = 0.0,
    batch_h1: bool = True,
    h1_only: bool = False,
    keep_going: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    events = eligible_manifest_events(
        manifest, scenarios=scenarios, policy_steps=policy_steps
    )
    if max_events is not None:
        events = events[:int(max_events)]
    if not events:
        raise ValueError("tail label generation selected no eligible events")
    if batch_h1:
        populate_missing_h1_batches(manifest_path, output_dir, events)
    index_path = output_dir / "pipeline_index.json"
    result = _load_index(index_path)
    result.update({
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "output_dir": str(output_dir),
        "drain_candidates_per_event": int(drain_candidates_per_event),
        "max_h3_candidates_per_owner": max_h3_candidates_per_owner,
        "max_h12_candidates_per_event": int(max_h12_candidates_per_event),
        "domain_pool_size": int(domain_pool_size),
        "horizon_strategy": horizon_strategy,
        "min_h12_gain_for_drain": float(min_h12_gain_for_drain),
        "batch_h1": bool(batch_h1),
        "h1_only": bool(h1_only),
        "requested_scenarios": list(scenarios),
        "requested_policy_steps": list(policy_steps),
        "max_events": max_events,
    })
    if h1_only:
        records = {}
        for event in events:
            root = _event_dir(
                output_dir,
                str(event["scenario"]),
                int(event["policy_step"]),
            )
            all_h1_path, _legacy_h1_path = _h1_paths(output_dir, event)
            meta_path, payload_path = _replay_cache_paths(root)
            if not _is_complete_h1(
                all_h1_path,
                event["scenario"],
                int(event["policy_step"]),
                candidate_mode="owner_block_v2_all",
            ):
                raise ValueError(
                    "H1-only generation did not produce a complete H1 artifact "
                    f"for {event['scenario']} step {event['policy_step']}"
                )
            records[_record_key(event)] = {
                **event,
                "status": "h1_complete",
                "root": str(root),
                "h1_artifact": str(all_h1_path),
                "h1_sha256": _sha256_file(all_h1_path),
                "all_h1_artifact": str(all_h1_path),
                "all_h1_sha256": _sha256_file(all_h1_path),
                "replay_cache_metadata": str(meta_path) if meta_path.is_file() else None,
                "replay_cache_payload": str(payload_path) if payload_path.is_file() else None,
            }
        result["events"] = _record_rows(records)
        _write_index(index_path, result)
        return result
    records = {
        _record_key(row): row for row in result.get("events", [])
    }
    started = time.monotonic()
    for event in events:
        key = _record_key(event)
        existing = records.get(key)
        if (
            existing is not None
            and existing.get("status") in {
                "complete",
                "no_h12_positive",
                "no_drain_candidate",
            }
            and _is_terminal_drain(
                Path(existing["drain_artifact"]),
                event["scenario"],
                int(event["policy_step"]),
            )
        ):
            continue
        records[key] = {**event, "status": "running"}
        result["events"] = _record_rows(records)
        _write_index(index_path, result)
        try:
            records[key] = _run_event(
                manifest_path,
                output_dir,
                event,
                drain_candidates_per_event=drain_candidates_per_event,
                max_h3_candidates_per_owner=max_h3_candidates_per_owner,
                max_h12_candidates_per_event=max_h12_candidates_per_event,
                domain_pool_size=domain_pool_size,
                horizon_strategy=horizon_strategy,
                min_h12_gain_for_drain=min_h12_gain_for_drain,
            )
        except Exception as exc:
            records[key] = {
                **event,
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            result["events"] = _record_rows(records)
            result["elapsed_sec"] = float(time.monotonic() - started)
            _write_index(index_path, result)
            if not keep_going:
                raise
        result["events"] = _record_rows(records)
        result["elapsed_sec"] = float(time.monotonic() - started)
        _write_index(index_path, result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--policy-steps", default="")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--drain-candidates-per-event", type=int, default=4)
    parser.add_argument("--max-h3-candidates-per-owner", type=int, default=1)
    parser.add_argument("--max-h12-candidates-per-event", type=int, default=2)
    parser.add_argument("--domain-pool-size", type=int, default=2)
    parser.add_argument("--min-h12-gain-for-drain", type=float, default=0.0)
    parser.add_argument(
        "--horizon-strategy",
        choices=("budgeted", "component_joint"),
        default="budgeted",
    )
    parser.add_argument("--no-batch-h1", action="store_true")
    parser.add_argument("--h1-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    result = run_tail_label_generation(
        Path(args.manifest),
        Path(args.output_dir),
        scenarios=_parse_csv(args.scenarios),
        policy_steps=_parse_int_csv(args.policy_steps),
        max_events=args.max_events,
        drain_candidates_per_event=args.drain_candidates_per_event,
        max_h3_candidates_per_owner=args.max_h3_candidates_per_owner,
        max_h12_candidates_per_event=args.max_h12_candidates_per_event,
        domain_pool_size=args.domain_pool_size,
        horizon_strategy=args.horizon_strategy,
        min_h12_gain_for_drain=args.min_h12_gain_for_drain,
        batch_h1=not bool(args.no_batch_h1),
        h1_only=bool(args.h1_only),
        keep_going=bool(args.keep_going),
    )
    print(json.dumps({
        "passed": result["passed"],
        "summary": result["summary"],
        "output": str(Path(args.output_dir) / "pipeline_index.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
