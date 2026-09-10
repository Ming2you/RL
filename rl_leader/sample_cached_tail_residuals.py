"""Sample current-contract residuals from a cached P-Stack prefix state."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rl_leader.diagnose_candidate_ablation import _horizon_label
from rl_leader.diagnose_pcent_reachability import (
    _response_scales_and_families,
    response_distance,
)
from rl_leader.diagnose_phase0_parity import _digest
from rl_leader.diagnose_reachable_candidate_attribution import (
    _rollout_price_candidate,
    _rollout_pstack,
)
from rl_leader.evaluate_cached_tail_residual_schedule import load_cached_prefix
from rl_leader.oracle_candidate_ablation import residual_sha256
from rl_leader.response_dqn_catalog import build_structured_action_catalog


SAMPLE_FORMAT = "cached_tail_residual_sampler_v1"
NATIVE_ROLLOUT_CACHE_FORMAT = "native_pstack_rollout_cache_v1"


@dataclass(frozen=True)
class SampledResidualSpec:
    candidate_id: str
    label: str
    generator: str
    family: str
    residual: tuple[float, ...]
    metadata: dict[str, Any]

    def residual_array(self) -> np.ndarray:
        return np.asarray(self.residual, dtype=np.float32)


def parse_magnitudes(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    else:
        values = tuple(float(value) for value in raw)
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("magnitudes must be non-empty and in (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("magnitudes must be unique")
    return values


def native_rollout_cache_path(
    cache_dir: Path,
    *,
    scenario: str,
    control_step: int,
    horizon_steps: int,
) -> Path:
    return (
        Path(cache_dir)
        / f"{scenario}_step{int(control_step):02d}_h{int(horizon_steps)}.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_native_rollout_cache(
    path: Path,
    *,
    scenario: str,
    control_step: int,
    horizon_steps: int,
    horizons: tuple[int, ...],
    context,
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version": NATIVE_ROLLOUT_CACHE_FORMAT,
        "scenario": str(scenario),
        "control_step": int(control_step),
        "horizon_steps": int(horizon_steps),
        "horizons": list(map(int, horizons)),
        "state_fingerprint": str(context.state_fingerprint),
        "anchor_fingerprint": str(context.anchor_fingerprint),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"native rollout cache mismatch for {path}: {key}")
    return dict(payload["rollout"])


def _write_native_rollout_cache(
    path: Path,
    *,
    scenario: str,
    control_step: int,
    horizon_steps: int,
    horizons: tuple[int, ...],
    context,
    rollout: dict[str, Any],
) -> None:
    _write_json_atomic(
        path,
        {
            "format_version": NATIVE_ROLLOUT_CACHE_FORMAT,
            "scenario": str(scenario),
            "control_step": int(control_step),
            "horizon_steps": int(horizon_steps),
            "horizons": list(map(int, horizons)),
            "state_fingerprint": str(context.state_fingerprint),
            "anchor_fingerprint": str(context.anchor_fingerprint),
            "rollout": rollout,
        },
    )


def _sparse_residual(
    action_names: Sequence[str],
    sparse: dict[str, float],
) -> np.ndarray:
    by_name = {name: index for index, name in enumerate(action_names)}
    missing = sorted(set(sparse) - set(by_name))
    if missing:
        raise ValueError(f"unknown residual fields: {missing}")
    residual = np.zeros(len(action_names), dtype=np.float32)
    for name, value in sparse.items():
        residual[by_name[name]] = float(value)
    if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1.0 + 1.0e-7:
        raise ValueError("residual must be finite and inside [-1, 1]")
    return residual


def _spec(
    action_names: Sequence[str],
    *,
    label: str,
    generator: str,
    family: str,
    sparse: dict[str, float] | None = None,
    residual: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> SampledResidualSpec:
    values = (
        _sparse_residual(action_names, dict(sparse or {}))
        if residual is None
        else np.asarray(residual, dtype=np.float32).reshape(-1)
    )
    if values.shape != (len(action_names),):
        raise ValueError("sampled residual has the wrong dimension")
    if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1.0 + 1.0e-7:
        raise ValueError("sampled residual must stay inside [-1, 1]")
    values = values.copy()
    values[values == 0.0] = 0.0
    digest = residual_sha256(values)
    return SampledResidualSpec(
        candidate_id=f"{generator}:{label}:{digest[:12]}",
        label=label,
        generator=generator,
        family=family,
        residual=tuple(map(float, values)),
        metadata=dict(metadata or {}),
    )


def _append_unique(
    specs: list[SampledResidualSpec],
    seen: set[str],
    spec: SampledResidualSpec,
) -> None:
    digest = residual_sha256(spec.residual)
    if digest not in seen:
        seen.add(digest)
        specs.append(spec)


def _targeted_specs(
    action_names: Sequence[str],
    magnitudes: tuple[float, ...],
) -> list[SampledResidualSpec]:
    names = tuple(map(str, action_names))
    available = set(names)
    specs: list[SampledResidualSpec] = []
    seen: set[str] = set()

    def add(label: str, sparse: dict[str, float], family: str = "targeted") -> None:
        if set(sparse).issubset(available):
            _append_unique(
                specs,
                seen,
                _spec(
                    names,
                    label=label,
                    generator="targeted",
                    family=family,
                    sparse=sparse,
                    metadata={"source": "hand_seeded_current_contract"},
                ),
            )

    west_meter = "freeway.R_F_W.g_meter"
    west_vsl = "freeway.R_F_W.g_vsl"
    diamond_meter = "freeway.R_D_W.g_meter"
    freeway_curvature = [
        "freeway.R_F_W.l11",
        "freeway.R_F_W.l21",
        "freeway.R_F_W.l22",
        "freeway.R_D_W.l11",
        "freeway.R_D_W.l21",
        "freeway.R_D_W.l22",
    ]
    west_segment_terms = [
        name for name in names
        if name.startswith("vsl.FW_W__seg") and (
            name.endswith(".g_vsl") or name.endswith(".sqrt_h")
        )
    ]
    urban_terms = [
        name for name in names
        if name.startswith("urban.") and (
            name.endswith(".g_green") or name.endswith(".g_offset")
        )
    ]

    for magnitude in magnitudes:
        add(
            f"rfw_meter_positive_m{magnitude:g}",
            {west_meter: magnitude},
            family="linear",
        )
        for secondary in magnitudes:
            add(
                f"rfw_meter_urban_B_g_green_negative_vsl_FW_W_seg4_g_vsl_negative_m{magnitude:g}_s{secondary:g}",
                {
                    west_meter: magnitude,
                    "urban.B.g_green": -secondary,
                    "vsl.FW_W__seg4.g_vsl": -secondary,
                },
                family="discovered_step21_neighborhood",
            )
        add(
            f"west_pair_meter_positive_m{magnitude:g}",
            {west_meter: magnitude, diamond_meter: magnitude},
            family="multi_linear",
        )
        for sign, suffix in ((1.0, "positive"), (-1.0, "negative")):
            add(
                f"rfw_meter_vsl_{suffix}_m{magnitude:g}",
                {west_meter: magnitude, west_vsl: sign * magnitude},
                family="multi_linear",
            )
            for term in freeway_curvature:
                add(
                    f"rfw_meter_{term.replace('.', '_')}_{suffix}_m{magnitude:g}",
                    {west_meter: magnitude, term: sign * magnitude},
                    family="nonlinear_combo",
                )
            for term in west_segment_terms:
                add(
                    f"rfw_meter_{term.replace('.', '_')}_{suffix}_m{magnitude:g}",
                    {west_meter: magnitude, term: sign * magnitude},
                    family="vsl_combo",
                )
            for term in urban_terms:
                add(
                    f"rfw_meter_{term.replace('.', '_')}_{suffix}_m{magnitude:g}",
                    {west_meter: magnitude, term: sign * magnitude},
                    family="urban_freeway_combo",
                )
    return specs


def _catalog_specs(
    action_names: Sequence[str],
    *,
    seed: int,
    limit: int,
    magnitudes: tuple[float, ...],
) -> list[SampledResidualSpec]:
    if limit <= 0:
        return []
    catalog_magnitudes = tuple(value for value in magnitudes if value <= 0.5) or (0.5,)
    catalog = build_structured_action_catalog(
        action_names,
        magnitudes=catalog_magnitudes,
        families=("linear", "quadratic", "cross", "combo", "hybrid"),
        domains=("urban", "freeway"),
    )
    actions = list(catalog.actions[1:])
    priority_owners = {"R_F_W", "R_D_W"}
    priority = [action for action in actions if action.owner in priority_owners]
    rest = [action for action in actions if action.owner not in priority_owners]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(rest)
    specs: list[SampledResidualSpec] = []
    seen: set[str] = set()
    for action in [*priority, *rest]:
        _append_unique(
            specs,
            seen,
            _spec(
                action_names,
                label=action.key.replace(":", "_"),
                generator="catalog",
                family=action.family,
                residual=action.residual_array(),
                metadata={
                    "catalog_action_id": int(action.action_id),
                    "domain": action.domain,
                    "owner": action.owner,
                    "template": action.template,
                    "magnitude": float(action.magnitude),
                },
            ),
        )
        if len(specs) >= limit:
            break
    return specs


def _random_specs(
    action_names: Sequence[str],
    *,
    seed: int,
    count: int,
    magnitudes: tuple[float, ...],
) -> list[SampledResidualSpec]:
    if count <= 0:
        return []
    names = tuple(map(str, action_names))
    linear = [
        name for name in names
        if name.endswith((".g_green", ".g_offset", ".g_meter", ".g_vsl"))
    ]
    nonlinear = [
        name for name in names
        if name.endswith((".l11", ".l21", ".l22", ".sqrt_h"))
    ]
    budget = [name for name in ("budget.N_P", "budget.N_UF") if name in names]
    active_pool = tuple(linear + nonlinear + budget)
    if not active_pool:
        return []
    rng = np.random.default_rng(int(seed))
    specs: list[SampledResidualSpec] = []
    seen: set[str] = set()
    attempts = 0
    west_meter = "freeway.R_F_W.g_meter"
    while len(specs) < count and attempts < max(1000, count * 80):
        attempts += 1
        residual = np.zeros(len(names), dtype=np.float32)
        sparse: dict[str, float] = {}
        if west_meter in names and rng.random() < 0.55:
            sparse[west_meter] = float(rng.choice(magnitudes))
        component_count = int(rng.integers(2, 7))
        pool = list(active_pool)
        rng.shuffle(pool)
        for name in pool[:component_count]:
            if name in sparse:
                continue
            sign = float(rng.choice((-1.0, 1.0)))
            sparse[name] = sign * float(rng.choice(magnitudes))
        for name, value in sparse.items():
            residual[names.index(name)] = float(value)
        residual = np.clip(residual, -1.0, 1.0)
        if np.count_nonzero(np.abs(residual) > 1.0e-9) < 2:
            continue
        family = "random_multiblock"
        if any(name in nonlinear for name in sparse):
            family = "random_nonlinear_multiblock"
        _append_unique(
            specs,
            seen,
            _spec(
                names,
                label=f"seed{int(seed)}_draw{attempts}",
                generator="random",
                family=family,
                residual=residual,
                metadata={
                    "seed": int(seed),
                    "draw": int(attempts),
                    "nonzero": dict(sorted(sparse.items())),
                },
            ),
        )
    return specs


def build_sampled_residual_specs(
    action_names: Sequence[str],
    *,
    seed: int,
    max_h1_candidates: int,
    magnitudes: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> list[SampledResidualSpec]:
    """Build a bounded deterministic mix of targeted, catalog, and random samples."""
    if max_h1_candidates <= 0:
        raise ValueError("max_h1_candidates must be positive")
    names = tuple(map(str, action_names))
    targeted = _targeted_specs(names, magnitudes)
    catalog_quota = max(1, int(round(max_h1_candidates * 0.25)))
    catalog = _catalog_specs(
        names,
        seed=seed,
        limit=catalog_quota,
        magnitudes=magnitudes,
    )
    random = _random_specs(
        names,
        seed=seed + 7919,
        count=max_h1_candidates + catalog_quota,
        magnitudes=magnitudes,
    )

    targeted_quota = max(1, int(round(max_h1_candidates * 0.40)))
    selected_sources = [
        targeted[:targeted_quota],
        catalog[:catalog_quota],
        random,
        targeted[targeted_quota:],
        catalog[catalog_quota:],
    ]
    selected: list[SampledResidualSpec] = []
    seen: set[str] = set()
    for source in selected_sources:
        for spec in source:
            _append_unique(selected, seen, spec)
            if len(selected) >= max_h1_candidates:
                return selected
    return selected


def _rollout_checkpoint(rollout: dict[str, Any], horizon: int) -> dict[str, Any]:
    checkpoint = rollout.get("checkpoints", {}).get(str(int(horizon)))
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"rollout is missing H{horizon} checkpoint")
    return checkpoint


def evaluation_horizons(horizon_steps: int) -> tuple[int, ...]:
    horizon_steps = int(horizon_steps)
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    return tuple(
        sorted({horizon for horizon in (1, 3, horizon_steps) if horizon <= horizon_steps})
    )


def _response_memory_outcome_sha256(rollout: dict[str, Any]) -> str:
    checkpoint = _rollout_checkpoint(rollout, 1)
    return _digest({
        "response": np.round(
            np.asarray(rollout["first_step_response"], dtype=float), 6
        ).tolist(),
        "post_follower_sha256": str(checkpoint["follower_memory_sha256"]),
    })


def _record_from_rollout(
    *,
    spec: SampledResidualSpec,
    rollout: dict[str, Any],
    native: dict[str, Any],
    horizons: Iterable[int],
    action_names: Sequence[str],
    anchor_response: np.ndarray,
    response_scales: np.ndarray,
    response_families: dict[str, list[int]],
    control_step: int,
    selected_for_h12: bool,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    residual = spec.residual_array()
    first = _rollout_checkpoint(rollout, 1)
    horizon_labels = {
        str(int(horizon)): _horizon_label(
            _rollout_checkpoint(rollout, int(horizon)),
            _rollout_checkpoint(native, int(horizon)),
            horizon=int(horizon),
        )
        for horizon in sorted(set(int(value) for value in horizons))
    }
    nonzero = {
        str(name): float(residual[index])
        for index, name in enumerate(action_names)
        if abs(float(residual[index])) > 1.0e-9
    }
    distance = response_distance(
        np.asarray(rollout["first_step_response"], dtype=float),
        anchor_response,
        response_scales,
        response_families,
    )
    return {
        "format_version": SAMPLE_FORMAT,
        "control_step": int(control_step),
        "candidate_id": spec.candidate_id,
        "label": spec.label,
        "generator": spec.generator,
        "family": spec.family,
        "metadata": spec.metadata,
        "continuous_residual": list(map(float, residual)),
        "residual_nonzero": nonzero,
        "residual_sha256": residual_sha256(residual),
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_nonzero_count": int(np.count_nonzero(np.abs(residual) > 1.0e-9)),
        "first_step_response": list(map(float, rollout["first_step_response"])),
        "native_response_distance": distance,
        "native_response_distance_score": float(distance["overall_rmse"]),
        "response_memory_outcome_sha256": _response_memory_outcome_sha256(rollout),
        "post_follower_sha256": str(first["follower_memory_sha256"]),
        "post_physical_sha256": str(first["physical_state_sha256"]),
        "horizon_labels": horizon_labels,
        "selected_for_h12": bool(selected_for_h12),
        "selection_reason": selection_reason,
    }


def _evaluate_record_task(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    rollout = _rollout_price_candidate(
        payload["env"],
        spec.residual_array(),
        payload["context"],
        rollout_steps=int(payload["rollout_steps"]),
        horizons=tuple(payload["horizons"]),
    )
    return _record_from_rollout(
        spec=spec,
        rollout=rollout,
        native=payload["native"],
        horizons=tuple(payload["horizons"]),
        action_names=tuple(payload["action_names"]),
        anchor_response=np.asarray(payload["anchor_response"], dtype=float),
        response_scales=np.asarray(payload["response_scales"], dtype=float),
        response_families=dict(payload["response_families"]),
        control_step=int(payload["control_step"]),
        selected_for_h12=bool(payload["selected_for_h12"]),
        selection_reason=payload.get("selection_reason"),
    )


def _chunked(values: Sequence[Any], chunk_count: int) -> list[list[Any]]:
    count = int(chunk_count)
    if count <= 0:
        raise ValueError("chunk_count must be positive")
    chunks = [list(values[index::count]) for index in range(count)]
    return [chunk for chunk in chunks if chunk]


def _evaluate_record_chunk_task(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for spec in payload["specs"]:
        item = dict(payload)
        item["spec"] = spec
        item.pop("specs", None)
        records.append(_evaluate_record_task(item))
    return records


def _evaluate_records(
    *,
    env,
    context,
    specs: Sequence[SampledResidualSpec],
    native: dict[str, Any],
    rollout_steps: int,
    horizons: tuple[int, ...],
    action_names: Sequence[str],
    anchor_response: np.ndarray,
    response_scales: np.ndarray,
    response_families: dict[str, list[int]],
    control_step: int,
    selected_for_h12: bool,
    selection_reasons: dict[str, str] | None,
    workers: int,
    chunks_per_worker: int = 1,
) -> Iterable[dict[str, Any]]:
    payloads = [
        {
            "env": env,
            "context": context,
            "spec": spec,
            "native": native,
            "rollout_steps": int(rollout_steps),
            "horizons": tuple(horizons),
            "action_names": tuple(action_names),
            "anchor_response": np.asarray(anchor_response, dtype=float),
            "response_scales": np.asarray(response_scales, dtype=float),
            "response_families": dict(response_families),
            "control_step": int(control_step),
            "selected_for_h12": bool(selected_for_h12),
            "selection_reason": None if selection_reasons is None else selection_reasons.get(spec.candidate_id),
        }
        for spec in specs
    ]
    if int(workers) <= 1:
        for payload in payloads:
            yield _evaluate_record_task(payload)
        return
    with futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
        chunk_payloads = []
        chunk_count = min(
            len(payloads),
            int(workers) * max(1, int(chunks_per_worker)),
        )
        for chunk in _chunked(payloads, chunk_count):
            first = dict(chunk[0])
            first["specs"] = [payload["spec"] for payload in chunk]
            first.pop("spec", None)
            chunk_payloads.append(first)
        pending = [
            executor.submit(_evaluate_record_chunk_task, payload)
            for payload in chunk_payloads
        ]
        for future in futures.as_completed(pending):
            try:
                records = future.result()
            except Exception:
                continue
            for record in records:
                yield record


def _h1_gain(record: dict[str, Any]) -> float:
    return float(record.get("horizon_labels", {}).get("1", {}).get("ttt_gain", 0.0))


def _inventory_delta(record: dict[str, Any]) -> float:
    return float(
        record.get("horizon_labels", {})
        .get("1", {})
        .get("terminal_inventory_delta", 0.0)
    )


def _valid_h1(record: dict[str, Any]) -> bool:
    label = record.get("horizon_labels", {}).get("1", {})
    return bool(
        isinstance(label, dict)
        and label.get("label_valid", False)
        and label.get("validity_gate_pass", False)
    )


def _distance_score(record: dict[str, Any]) -> float:
    value = record.get("native_response_distance_score")
    if value is not None:
        return float(value)
    value = record.get("native_response_distance", 0.0)
    if isinstance(value, dict):
        return float(value.get("overall_rmse", value.get("overall_linf", 0.0)))
    return float(value)


def select_h12_records(
    h1_records: Sequence[dict[str, Any]],
    *,
    max_h12_candidates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select diverse executable H1 representatives for H12 evaluation."""
    if max_h12_candidates <= 0:
        return []
    representatives: dict[tuple[str, str], dict[str, Any]] = {}
    for record in h1_records:
        if not _valid_h1(record):
            continue
        key = (
            str(record["response_memory_outcome_sha256"]),
            str(record["post_physical_sha256"]),
        )
        current = representatives.get(key)
        priority = (
            _h1_gain(record),
            -_inventory_delta(record),
            _distance_score(record),
            -int(record.get("residual_nonzero_count", 0)),
            str(record["candidate_id"]),
        )
        if current is None:
            representatives[key] = record
            continue
        current_priority = (
            _h1_gain(current),
            -_inventory_delta(current),
            _distance_score(current),
            -int(current.get("residual_nonzero_count", 0)),
            str(current["candidate_id"]),
        )
        if priority > current_priority:
            representatives[key] = record
    reps = list(representatives.values())
    if not reps:
        return []

    buckets = [
        (
            "h1_gain",
            sorted(reps, key=lambda row: (_h1_gain(row), str(row["candidate_id"])), reverse=True),
        ),
        (
            "inventory_drain",
            sorted(reps, key=lambda row: (_inventory_delta(row), str(row["candidate_id"]))),
        ),
        (
            "response_distance",
            sorted(
                reps,
                key=lambda row: (
                    _distance_score(row),
                    str(row["candidate_id"]),
                ),
                reverse=True,
            ),
        ),
        (
            "multi_nonlinear",
            sorted(
                [
                    row for row in reps
                    if "nonlinear" in str(row.get("family", ""))
                    or int(row.get("residual_nonzero_count", 0)) >= 3
                ],
                key=lambda row: (_h1_gain(row), str(row["candidate_id"])),
                reverse=True,
            ),
        ),
    ]
    rng = np.random.default_rng(int(seed))
    shuffled = list(reps)
    rng.shuffle(shuffled)
    buckets.append(("random_representative", shuffled))

    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    bucket_positions = [0] * len(buckets)
    while len(chosen) < max_h12_candidates:
        progressed = False
        for bucket_index, (reason, rows) in enumerate(buckets):
            position = bucket_positions[bucket_index]
            while position < len(rows):
                row = rows[position]
                position += 1
                if row["candidate_id"] in chosen_ids:
                    continue
                copy_row = dict(row)
                copy_row["selection_reason"] = reason
                chosen.append(copy_row)
                chosen_ids.add(row["candidate_id"])
                progressed = True
                break
            bucket_positions[bucket_index] = position
            if len(chosen) >= max_h12_candidates:
                break
        if not progressed:
            break
    return chosen


def run_cached_tail_residual_sampler(
    *,
    replay_cache: Path,
    output_dir: Path,
    seed: int,
    max_h1_candidates: int,
    max_h12_candidates: int,
    horizon_steps: int,
    magnitudes: tuple[float, ...],
    pstack_summary: dict[str, Any] | None = None,
    workers: int = 1,
    h1_source_artifact: Path | None = None,
    native_rollout_cache_dir: Path | None = None,
    h1_chunks_per_worker: int = 1,
    h12_chunks_per_worker: int = 1,
    verbose: bool = True,
) -> dict[str, Any]:
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    h1_path = output_dir / "h1_records.jsonl"
    h12_path = output_dir / "h12_records.jsonl"
    collisions = [path for path in (summary_path, h1_path, h12_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite residual sampler outputs: "
            + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    env, context = load_cached_prefix(Path(replay_cache))
    control_step = int(env.step_idx - env.warmup)
    h1_native = _rollout_pstack(
        env,
        context,
        rollout_steps=1,
        horizons=(1,),
    )
    action_names = tuple(env.action_schema.names)
    scales, families = _response_scales_and_families(env)
    anchor_response = np.asarray(h1_native["first_step_response"], dtype=float)

    h1_records: list[dict[str, Any]] = []
    if h1_source_artifact is not None:
        source = json.loads(Path(h1_source_artifact).read_text(encoding="utf-8"))
        if source.get("format_version") != SAMPLE_FORMAT:
            raise ValueError("H1 source artifact has the wrong format")
        if int(source.get("control_step", -1)) != int(control_step):
            raise ValueError("H1 source artifact control_step does not match replay cache")
        source_names = tuple(map(str, source.get("action_names", [])))
        if source_names and source_names != action_names:
            raise ValueError("H1 source artifact action schema does not match replay cache")
        h1_records = [dict(row) for row in source.get("h1_records", [])]
        if not h1_records:
            raise ValueError("H1 source artifact has no h1_records")
        with h1_path.open("w", encoding="utf-8") as handle:
            for record in h1_records:
                record = dict(record)
                record["selected_for_h12"] = False
                record["selection_reason"] = None
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    else:
        specs = build_sampled_residual_specs(
            action_names,
            seed=int(seed),
            max_h1_candidates=int(max_h1_candidates),
            magnitudes=magnitudes,
        )
        with h1_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(
                _evaluate_records(
                    env=env,
                    context=context,
                    specs=specs,
                    native=h1_native,
                    rollout_steps=1,
                    horizons=(1,),
                    action_names=action_names,
                    anchor_response=anchor_response,
                    response_scales=scales,
                    response_families=families,
                    control_step=control_step,
                    selected_for_h12=False,
                    selection_reasons=None,
                    workers=int(workers),
                    chunks_per_worker=int(h1_chunks_per_worker),
                ),
                start=1,
            ):
                h1_records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if verbose and (
                    index == 1 or index == len(specs) or index % 10 == 0
                ):
                    print(json.dumps({
                        "phase": "h1",
                        "control_step": int(control_step),
                        "candidate": int(index),
                        "total": int(len(specs)),
                        "best_h1_gain": round(
                            max(_h1_gain(row) for row in h1_records), 6
                        ),
                    }, sort_keys=True), flush=True)

    selected = select_h12_records(
        h1_records,
        max_h12_candidates=int(max_h12_candidates),
        seed=int(seed) + 104729,
    )
    by_candidate_id = {row["candidate_id"]: row for row in h1_records}
    for row in selected:
        by_candidate_id[row["candidate_id"]]["selected_for_h12"] = True
        by_candidate_id[row["candidate_id"]]["selection_reason"] = row.get("selection_reason")

    horizons = evaluation_horizons(int(horizon_steps))
    native = h1_native
    native_cache_path = None
    if selected and int(horizon_steps) > 1:
        if native_rollout_cache_dir is not None:
            native_cache_path = native_rollout_cache_path(
                Path(native_rollout_cache_dir),
                scenario=str(env.scenario_name),
                control_step=int(control_step),
                horizon_steps=int(horizon_steps),
            )
            native = _load_native_rollout_cache(
                native_cache_path,
                scenario=str(env.scenario_name),
                control_step=int(control_step),
                horizon_steps=int(horizon_steps),
                horizons=horizons,
                context=context,
            )
        if native is None or native is h1_native:
            native = _rollout_pstack(
                env,
                context,
                rollout_steps=int(horizon_steps),
                horizons=horizons,
            )
            if native_cache_path is not None:
                _write_native_rollout_cache(
                    native_cache_path,
                    scenario=str(env.scenario_name),
                    control_step=int(control_step),
                    horizon_steps=int(horizon_steps),
                    horizons=horizons,
                    context=context,
                    rollout=native,
                )

    h12_records: list[dict[str, Any]] = []
    selection_reasons = {
        str(row["candidate_id"]): str(row.get("selection_reason", ""))
        for row in selected
    }
    with h12_path.open("w", encoding="utf-8") as handle:
        if int(horizon_steps) == 1:
            for index, selected_h1 in enumerate(selected, start=1):
                record = dict(selected_h1)
                record["selected_for_h12"] = True
                record["selection_reason"] = selection_reasons.get(
                    str(record["candidate_id"]),
                    str(record.get("selection_reason", "")),
                )
                h12_records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                horizon_label = record["horizon_labels"]["1"]
                if verbose:
                    print(json.dumps({
                        "phase": "h1_promoted",
                        "control_step": int(control_step),
                        "candidate": int(index),
                        "total": int(len(selected)),
                        "gain": round(float(horizon_label["ttt_gain"]), 6),
                        "positive": bool(horizon_label["positive"]),
                        "candidate_id": record["candidate_id"],
                    }, sort_keys=True), flush=True)
        else:
            selected_specs = [
                SampledResidualSpec(
                    candidate_id=str(selected_h1["candidate_id"]),
                    label=str(selected_h1["label"]),
                    generator=str(selected_h1["generator"]),
                    family=str(selected_h1["family"]),
                    residual=tuple(map(float, selected_h1["continuous_residual"])),
                    metadata=dict(selected_h1.get("metadata", {})),
                )
                for selected_h1 in selected
            ]
            for index, record in enumerate(
                _evaluate_records(
                    env=env,
                    context=context,
                    specs=selected_specs,
                    native=native,
                    rollout_steps=int(horizon_steps),
                    horizons=horizons,
                    action_names=action_names,
                    anchor_response=anchor_response,
                    response_scales=scales,
                    response_families=families,
                    control_step=control_step,
                    selected_for_h12=True,
                    selection_reasons=selection_reasons,
                    workers=int(workers),
                    chunks_per_worker=int(h12_chunks_per_worker),
                ),
                start=1,
            ):
                h12_records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                horizon_label = record["horizon_labels"][str(int(horizon_steps))]
                if verbose:
                    print(json.dumps({
                        "phase": f"h{int(horizon_steps)}",
                        "control_step": int(control_step),
                        "candidate": int(index),
                        "total": int(len(selected)),
                        "gain": round(float(horizon_label["ttt_gain"]), 6),
                        "positive": bool(horizon_label["positive"]),
                        "candidate_id": record["candidate_id"],
                    }, sort_keys=True), flush=True)

    best_h12 = None
    if h12_records:
        best_h12 = max(
            h12_records,
            key=lambda row: (
                float(row["horizon_labels"][str(int(horizon_steps))]["ttt_gain"]),
                bool(row["horizon_labels"][str(int(horizon_steps))]["positive"]),
                str(row["candidate_id"]),
            ),
        )
    summary = {
        "format_version": SAMPLE_FORMAT,
        "replay_cache": str(replay_cache),
        "scenario": str(env.scenario_name),
        "control_step": int(control_step),
        "seed": int(seed),
        "max_h1_candidates": int(max_h1_candidates),
        "max_h12_candidates": int(max_h12_candidates),
        "horizon_steps": int(horizon_steps),
        "magnitudes": list(map(float, magnitudes)),
        "workers": int(workers),
        "h1_chunks_per_worker": int(h1_chunks_per_worker),
        "h12_chunks_per_worker": int(h12_chunks_per_worker),
        "h1_source_artifact": None if h1_source_artifact is None else str(h1_source_artifact),
        "action_names": list(action_names),
        "native_h1_rollout": h1_native,
        "native_rollout": native,
        "native_rollout_cache": None if native_cache_path is None else str(native_cache_path),
        "h1_records_path": str(h1_path),
        "h12_records_path": str(h12_path),
        "h1_records": h1_records,
        "h12_records": h12_records,
        "h1_record_count": int(len(h1_records)),
        "h12_record_count": int(len(h12_records)),
        "h1_unique_response_count": int(len({
            (
                str(row["response_memory_outcome_sha256"]),
                str(row["post_physical_sha256"]),
            )
            for row in h1_records
        })),
        "h12_positive_count": int(sum(
            bool(row["horizon_labels"][str(int(horizon_steps))]["positive"])
            for row in h12_records
        )),
        "best_h12": best_h12,
        "wall_seconds": float(time.perf_counter() - started),
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
        summary.update({
            "pstack_total_ttt": pstack_total,
            "target_5pct_total_ttt": float(0.95 * pstack_total),
            "required_full_run_gain_5pct": float(0.05 * pstack_total),
        })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(json.dumps({
            "passed": True,
            "scenario": env.scenario_name,
            "control_step": int(control_step),
            "h1_records": int(len(h1_records)),
            "h12_records": int(len(h12_records)),
            "h12_positive": int(summary["h12_positive_count"]),
            "best_h12_gain": (
                None if best_h12 is None
                else float(best_h12["horizon_labels"][str(int(horizon_steps))]["ttt_gain"])
            ),
            "output": str(summary_path),
        }, indent=2), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--max-h1-candidates", type=int, default=160)
    parser.add_argument("--max-h12-candidates", type=int, default=32)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--magnitudes", default="0.25,0.5,0.75")
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--h1-source-artifact", type=Path)
    parser.add_argument("--native-rollout-cache-dir", type=Path)
    parser.add_argument("--h1-chunks-per-worker", type=int, default=1)
    parser.add_argument("--h12-chunks-per-worker", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pstack_summary = None
    if args.pstack_summary is not None:
        pstack_summary = json.loads(args.pstack_summary.read_text(encoding="utf-8"))
    run_cached_tail_residual_sampler(
        replay_cache=args.replay_cache,
        output_dir=args.output_dir,
        seed=int(args.seed),
        max_h1_candidates=int(args.max_h1_candidates),
        max_h12_candidates=int(args.max_h12_candidates),
        horizon_steps=int(args.horizon_steps),
        magnitudes=parse_magnitudes(args.magnitudes),
        pstack_summary=pstack_summary,
        workers=int(args.workers),
        h1_source_artifact=args.h1_source_artifact,
        native_rollout_cache_dir=args.native_rollout_cache_dir,
        h1_chunks_per_worker=int(args.h1_chunks_per_worker),
        h12_chunks_per_worker=int(args.h12_chunks_per_worker),
        verbose=not bool(args.quiet),
    )


if __name__ == "__main__":
    main()
