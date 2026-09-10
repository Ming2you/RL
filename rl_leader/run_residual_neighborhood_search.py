"""Full-tail search around residuals that already beat P-Stack locally.

The 5% loop can spend a lot of time sampling nominal prices whose follower
responses collapse to unhelpful actions.  This runner takes an observed
sample artifact, treats its selected residual as an executable response seed,
then evaluates nearby schedules directly on the cached full-tail simulator.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl_leader.evaluate_cached_tail_residual_schedule import (
    load_cached_prefix,
    run_cached_tail_residual_schedule,
)
from rl_leader.evaluate_fixed_residual_schedule import ScheduledResidual
from rl_leader.run_residual_5pct_loop import (
    ensure_replay_caches,
    parse_int_csv,
    step_cache_path,
)


NEIGHBORHOOD_FORMAT = "residual_neighborhood_search_v1"


@dataclass(frozen=True)
class ResidualSeed:
    source_path: Path
    source_step: int
    source_label: str
    action_names: tuple[str, ...]
    residual: tuple[float, ...]


@dataclass(frozen=True)
class ScheduleCandidate:
    label: str
    schedule: tuple[ScheduledResidual, ...]
    metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")
    return safe[:120] or "candidate"


def _horizon_key(artifact: dict[str, Any]) -> str:
    return str(int(artifact.get("horizon_steps", 12)))


def _record_gain(record: dict[str, Any], horizon_key: str) -> float:
    return float(
        record.get("horizon_labels", {})
        .get(str(horizon_key), {})
        .get("ttt_gain", 0.0)
    )


def _record_positive(record: dict[str, Any], horizon_key: str) -> bool:
    label = record.get("horizon_labels", {}).get(str(horizon_key), {})
    return bool(label.get("positive", False) and label.get("validity_gate_pass", False))


def select_seed_record(
    artifact: dict[str, Any],
    *,
    selection: str = "best_h12",
) -> dict[str, Any]:
    if selection in artifact and isinstance(artifact[selection], dict):
        return dict(artifact[selection])
    records = list(artifact.get("h12_records", []))
    if not records:
        raise ValueError("source artifact has no h12_records")
    horizon_key = _horizon_key(artifact)
    if selection == "best_h12":
        return dict(
            max(
                records,
                key=lambda row: (
                    _record_gain(row, horizon_key),
                    _record_positive(row, horizon_key),
                    str(row.get("candidate_id", "")),
                ),
            )
        )
    if selection == "best_positive_h12":
        positives = [row for row in records if _record_positive(row, horizon_key)]
        if not positives:
            raise ValueError("source artifact has no positive h12 record")
        return dict(
            max(
                positives,
                key=lambda row: (
                    _record_gain(row, horizon_key),
                    str(row.get("candidate_id", "")),
                ),
            )
        )
    for row in records:
        if selection in {str(row.get("candidate_id", "")), str(row.get("label", ""))}:
            return dict(row)
    raise ValueError(f"source artifact has no selected record {selection!r}")


def load_residual_seed(
    artifact_path: Path,
    *,
    selection: str = "best_h12",
    fallback_replay_cache: Path | None = None,
) -> ResidualSeed:
    artifact_path = Path(artifact_path)
    artifact = _read_json(artifact_path)
    if artifact.get("format_version") != "cached_tail_residual_sampler_v1":
        raise ValueError("source artifact has the wrong format")
    names = tuple(map(str, artifact.get("action_names", ())))
    if not names:
        replay_cache = artifact.get("replay_cache")
        schema_cache = Path(replay_cache) if replay_cache else fallback_replay_cache
        if schema_cache is None:
            raise ValueError("source artifact is missing action_names")
        env, _context = load_cached_prefix(Path(schema_cache))
        names = tuple(map(str, env.action_schema.names))
    record = select_seed_record(artifact, selection=selection)
    residual = tuple(map(float, record["continuous_residual"]))
    if len(residual) != len(names):
        raise ValueError("source residual length does not match action_names")
    return ResidualSeed(
        source_path=artifact_path,
        source_step=int(record.get("control_step", artifact["control_step"])),
        source_label=str(record.get("candidate_id", selection)),
        action_names=names,
        residual=residual,
    )


def _scaled(values: Sequence[float], scale: float) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float32) * float(scale)
    return tuple(map(float, np.clip(arr, -1.0, 1.0)))


def _subset(
    values: Sequence[float],
    indices: Sequence[int],
    *,
    scale: float,
) -> tuple[float, ...]:
    arr = np.zeros(len(values), dtype=np.float32)
    base = np.asarray(values, dtype=np.float32)
    for index in indices:
        arr[int(index)] = base[int(index)] * float(scale)
    return tuple(map(float, np.clip(arr, -1.0, 1.0)))


def _flip_one(
    values: Sequence[float],
    index: int,
    *,
    scale: float,
) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float32) * float(scale)
    arr[int(index)] *= -1.0
    return tuple(map(float, np.clip(arr, -1.0, 1.0)))


def _nonzero_indices(values: Sequence[float], *, tol: float = 1.0e-9) -> tuple[int, ...]:
    return tuple(
        int(index)
        for index, value in enumerate(values)
        if abs(float(value)) > float(tol)
    )


def _steps_by_distance(steps: Sequence[int], *, center: int) -> tuple[int, ...]:
    unique = sorted(set(int(step) for step in steps))
    return tuple(sorted(unique, key=lambda step: (abs(step - int(center)), step)))


def _residual_signature(schedule: Sequence[ScheduledResidual]) -> tuple[tuple[int, tuple[float, ...]], ...]:
    return tuple((int(item.step), tuple(map(float, item.residual))) for item in schedule)


def make_candidate(
    *,
    label: str,
    source: ResidualSeed,
    step_residuals: Sequence[tuple[int, Sequence[float]]],
    metadata: dict[str, Any],
) -> ScheduleCandidate:
    schedule = tuple(
        ScheduledResidual(
            step=int(step),
            label=label,
            residual=tuple(map(float, residual)),
            source=str(source.source_path),
        )
        for step, residual in sorted(step_residuals, key=lambda row: int(row[0]))
    )
    return ScheduleCandidate(label=label, schedule=schedule, metadata=dict(metadata))


def generate_neighborhood_candidates(
    source: ResidualSeed,
    *,
    placement_steps: Sequence[int],
    scales: Sequence[float],
    repeat_sizes: Sequence[int] = (2,),
    subset_pool_limit: int = 6,
    max_subset_size: int = 3,
    include_sign_flips: bool = True,
    max_candidates: int = 64,
) -> list[ScheduleCandidate]:
    if max_candidates <= 0:
        return []
    nonzero = _nonzero_indices(source.residual)
    if not nonzero:
        raise ValueError("source residual has no nonzero entries")
    ranked_indices = tuple(
        index
        for index, _value in sorted(
            ((idx, abs(source.residual[idx])) for idx in nonzero),
            key=lambda row: (row[1], -row[0]),
            reverse=True,
        )
    )
    subset_pool = ranked_indices[: max(1, int(subset_pool_limit))]
    placement_steps = _steps_by_distance(placement_steps, center=int(source.source_step))
    if not placement_steps:
        placement_steps = (int(source.source_step),)
    scales = tuple(float(scale) for scale in scales)
    if not scales:
        scales = (1.0,)

    candidates: list[ScheduleCandidate] = []
    seen: set[tuple[tuple[int, tuple[float, ...]], ...]] = set()

    def add(candidate: ScheduleCandidate) -> None:
        if len(candidates) >= int(max_candidates):
            return
        signature = _residual_signature(candidate.schedule)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(candidate)

    for step in placement_steps:
        for scale in scales:
            add(
                make_candidate(
                    label=f"all_s{scale:g}_at{step:02d}",
                    source=source,
                    step_residuals=[(step, _scaled(source.residual, scale))],
                    metadata={"family": "scaled_all", "scale": float(scale)},
                )
            )

    for size in range(1, min(int(max_subset_size), len(subset_pool)) + 1):
        for indices in itertools.combinations(subset_pool, size):
            field_tag = "-".join(source.action_names[index].replace(".", "_") for index in indices)
            for step in placement_steps:
                for scale in scales:
                    add(
                        make_candidate(
                            label=f"subset{size}_s{scale:g}_at{step:02d}_{field_tag}",
                            source=source,
                            step_residuals=[(step, _subset(source.residual, indices, scale=scale))],
                            metadata={
                                "family": "field_subset",
                                "scale": float(scale),
                                "fields": [source.action_names[index] for index in indices],
                            },
                        )
                    )

    if include_sign_flips:
        for index in subset_pool:
            field_tag = source.action_names[index].replace(".", "_")
            for step in placement_steps:
                for scale in scales:
                    add(
                        make_candidate(
                            label=f"flip1_s{scale:g}_at{step:02d}_{field_tag}",
                            source=source,
                            step_residuals=[(step, _flip_one(source.residual, index, scale=scale))],
                            metadata={
                                "family": "single_field_flip",
                                "scale": float(scale),
                                "flipped_field": source.action_names[index],
                            },
                        )
                    )

    for repeat_size in sorted(set(int(size) for size in repeat_sizes)):
        if repeat_size <= 1:
            continue
        for steps in itertools.combinations(placement_steps, repeat_size):
            for scale in scales:
                residual = _scaled(source.residual, scale)
                step_label = "-".join(f"{step:02d}" for step in sorted(steps))
                add(
                    make_candidate(
                        label=f"repeat{repeat_size}_s{scale:g}_steps{step_label}",
                        source=source,
                        step_residuals=[(step, residual) for step in steps],
                        metadata={
                            "family": "repeated_all",
                            "scale": float(scale),
                            "repeat_size": int(repeat_size),
                        },
                    )
                )
    return candidates


def _candidate_output_dir(base: Path, label: str, *, subdir: str = "tail_evals") -> Path:
    base_dir = Path(base) / str(subdir) / _safe_label(label)
    if (base_dir / "summary.json").is_file() or not (base_dir / "trace.jsonl").exists():
        return base_dir
    for index in range(1, 1000):
        candidate = base_dir.with_name(f"{base_dir.name}_retry{index:02d}")
        if (candidate / "summary.json").is_file() or not (candidate / "trace.jsonl").exists():
            return candidate
    raise RuntimeError(f"too many retry eval directories for {base_dir}")


def _schedule_dict(candidate: ScheduleCandidate) -> dict[int, ScheduledResidual]:
    schedule: dict[int, ScheduledResidual] = {}
    for item in candidate.schedule:
        if int(item.step) in schedule:
            raise ValueError(f"duplicate schedule step in {candidate.label}: {item.step}")
        schedule[int(item.step)] = item
    return schedule


def _run_candidate_task(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = ScheduleCandidate(
        label=str(payload["label"]),
        schedule=tuple(
            ScheduledResidual(
                step=int(item["step"]),
                label=str(item["label"]),
                residual=tuple(map(float, item["residual"])),
                source=str(item["source"]),
            )
            for item in payload["schedule"]
        ),
        metadata=dict(payload["metadata"]),
    )
    output_dir = Path(payload["output_dir"])
    eval_dir = _candidate_output_dir(output_dir, candidate.label)
    summary_path = eval_dir / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
    else:
        summary = run_cached_tail_residual_schedule(
            replay_cache=Path(payload["replay_cache"]),
            schedule=_schedule_dict(candidate),
            output_dir=eval_dir,
            pstack_summary=dict(payload["pstack_summary"]),
            force_without_h3_gate=True,
            verbose=bool(payload["verbose"]),
            max_tail_steps=payload.get("tail_step_limit"),
        )
    return {
        "label": candidate.label,
        "metadata": candidate.metadata,
        "schedule_steps": [int(item.step) for item in candidate.schedule],
        "earliest_step": int(min(item.step for item in candidate.schedule)),
        "output_dir": str(eval_dir),
        "summary": summary,
    }


def _run_anchor_screen_task(payload: dict[str, Any]) -> dict[str, Any]:
    step = int(payload["step"])
    output_dir = Path(payload["output_dir"])
    eval_dir = output_dir / "anchor_tail_evals" / f"anchor_step{step:02d}_h{int(payload['tail_step_limit']):02d}"
    summary_path = eval_dir / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
    else:
        summary = run_cached_tail_residual_schedule(
            replay_cache=Path(payload["replay_cache"]),
            schedule={},
            output_dir=eval_dir,
            pstack_summary=dict(payload["pstack_summary"]),
            force_without_h3_gate=True,
            verbose=bool(payload["verbose"]),
            max_tail_steps=int(payload["tail_step_limit"]),
        )
    return {"step": step, "output_dir": str(eval_dir), "summary": summary}


def _serialize_candidate(candidate: ScheduleCandidate) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "metadata": candidate.metadata,
        "schedule": [
            {
                "step": int(item.step),
                "label": item.label,
                "residual": list(map(float, item.residual)),
                "source": item.source,
                "nonzero_count": int(np.count_nonzero(np.abs(np.asarray(item.residual)) > 1.0e-9)),
            }
            for item in candidate.schedule
        ],
    }


def run_residual_neighborhood_search(
    *,
    source_artifact: Path,
    output_dir: Path,
    cache_output_dir: Path,
    pstack_summary_path: Path,
    scenario: str = "sweet_170_incident_w60",
    selection: str = "best_h12",
    placement_steps: tuple[int, ...] = (),
    scales: tuple[float, ...] = (0.5, 0.75, 1.0),
    repeat_sizes: tuple[int, ...] = (2,),
    subset_pool_limit: int = 6,
    max_subset_size: int = 3,
    include_sign_flips: bool = True,
    max_candidates: int = 64,
    eval_parallel_workers: int = 1,
    tail_step_limit: int | None = None,
    full_validate_top_k: int = 0,
    full_eval_parallel_workers: int | None = None,
    cache_mode: str = "direct",
    verbose: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    pstack_summary = _read_json(pstack_summary_path)
    pstack_total = float(pstack_summary["total_ttt"])
    source_artifact_payload = _read_json(source_artifact)
    source_step_hint = int(source_artifact_payload["control_step"])
    source_cache = ensure_replay_caches(
        manifest_path=None,
        cache_output_dir=cache_output_dir,
        scenario=scenario,
        steps=(source_step_hint,),
        t_total=float(pstack_summary.get("t_total_sec", 14400.0)),
        cache_mode=cache_mode,
    )[source_step_hint]
    source = load_residual_seed(
        source_artifact,
        selection=selection,
        fallback_replay_cache=source_cache,
    )
    if not placement_steps:
        placement_steps = (source.source_step,)
    candidates = generate_neighborhood_candidates(
        source,
        placement_steps=placement_steps,
        scales=scales,
        repeat_sizes=repeat_sizes,
        subset_pool_limit=int(subset_pool_limit),
        max_subset_size=int(max_subset_size),
        include_sign_flips=bool(include_sign_flips),
        max_candidates=int(max_candidates),
    )
    cache_steps = sorted(
        {
            min(int(item.step) for item in candidate.schedule)
            for candidate in candidates
        }
    )
    caches = ensure_replay_caches(
        manifest_path=None,
        cache_output_dir=cache_output_dir,
        scenario=scenario,
        steps=cache_steps,
        t_total=float(pstack_summary.get("t_total_sec", 14400.0)),
        cache_mode=cache_mode,
    )
    _write_json(
        output_dir / "candidate_manifest.json",
        {
            "format_version": NEIGHBORHOOD_FORMAT,
            "scenario": scenario,
            "source_artifact": str(source_artifact),
            "source_step": int(source.source_step),
            "source_label": source.source_label,
            "placement_steps": list(map(int, placement_steps)),
            "scales": list(map(float, scales)),
            "repeat_sizes": list(map(int, repeat_sizes)),
            "candidate_count": int(len(candidates)),
            "candidates": [_serialize_candidate(candidate) for candidate in candidates],
        },
    )

    anchor_summaries: dict[int, dict[str, Any]] = {}
    if tail_step_limit is not None:
        anchor_payloads = [
            {
                "step": int(step),
                "output_dir": str(output_dir),
                "replay_cache": str(caches[int(step)]),
                "pstack_summary": dict(pstack_summary),
                "verbose": bool(verbose),
                "tail_step_limit": int(tail_step_limit),
            }
            for step in cache_steps
        ]
        anchor_workers = min(max(1, int(eval_parallel_workers)), len(anchor_payloads))
        if anchor_workers <= 1:
            anchor_results = [_run_anchor_screen_task(payload) for payload in anchor_payloads]
        else:
            with futures.ProcessPoolExecutor(max_workers=anchor_workers) as executor:
                anchor_results = list(
                    executor.map(_run_anchor_screen_task, anchor_payloads)
                )
        for row in anchor_results:
            anchor_summaries[int(row["step"])] = dict(row["summary"])
            print(
                json.dumps(
                    {
                        "phase": "neighborhood_anchor_done",
                        "step": int(row["step"]),
                        "total_ttt": float(row["summary"].get("total_ttt", np.inf)),
                        "tail_step_limit": int(tail_step_limit),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    eval_summaries: list[dict[str, Any]] = []
    payloads = []
    for candidate in candidates:
        earliest = min(int(item.step) for item in candidate.schedule)
        payloads.append(
            {
                **_serialize_candidate(candidate),
                "output_dir": str(output_dir),
                "replay_cache": str(caches[earliest]),
                "pstack_summary": dict(pstack_summary),
                "verbose": bool(verbose),
                "tail_step_limit": tail_step_limit,
            }
        )

    def append_result(
        result: dict[str, Any],
        sink: list[dict[str, Any]],
        *,
        phase: str,
    ) -> None:
        if tail_step_limit is not None and phase == "neighborhood_eval_done":
            anchor = anchor_summaries.get(int(result["earliest_step"]))
            if anchor is not None:
                result["screen_anchor_total_ttt"] = float(anchor["total_ttt"])
                result["screen_ttt_gain"] = float(
                    anchor["total_ttt"] - result["summary"]["total_ttt"]
                )
        sink.append(result)
        summary = dict(result["summary"])
        raw_percent = summary.get("vs_pstack_percent")
        print(
            json.dumps(
                {
                    "phase": phase,
                    "label": result["label"],
                    "steps": result["schedule_steps"],
                    "total_ttt": float(summary.get("total_ttt", np.inf)),
                    "screen_ttt_gain": result.get("screen_ttt_gain"),
                    "vs_pstack_percent": None if raw_percent is None else float(raw_percent),
                    "partial_tail_evaluation": bool(summary.get("partial_tail_evaluation", False)),
                    "meets_5pct_target": bool(summary.get("meets_5pct_target", False)),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if int(eval_parallel_workers) <= 1:
        for payload in payloads:
            append_result(
                _run_candidate_task(payload),
                eval_summaries,
                phase="neighborhood_eval_done",
            )
    else:
        with futures.ProcessPoolExecutor(max_workers=int(eval_parallel_workers)) as executor:
            pending = {executor.submit(_run_candidate_task, payload): payload for payload in payloads}
            for future in futures.as_completed(pending):
                append_result(
                    future.result(),
                    eval_summaries,
                    phase="neighborhood_eval_done",
                )

    full_eval_summaries: list[dict[str, Any]] = []
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    if tail_step_limit is not None and int(full_validate_top_k) > 0 and eval_summaries:
        top_screened = sorted(
            eval_summaries,
            key=lambda row: (
                float(row.get("screen_ttt_gain", -np.inf)),
                -float(row["summary"].get("total_ttt", np.inf)),
                str(row["label"]),
            ),
            reverse=True,
        )[: int(full_validate_top_k)]
        full_payloads = []
        for row in top_screened:
            candidate = candidate_by_label[str(row["label"])]
            earliest = min(int(item.step) for item in candidate.schedule)
            full_payloads.append(
                {
                    **_serialize_candidate(candidate),
                    "output_dir": str(output_dir / "full_validation"),
                    "replay_cache": str(caches[earliest]),
                    "pstack_summary": dict(pstack_summary),
                    "verbose": bool(verbose),
                    "tail_step_limit": None,
                }
            )
        workers = int(full_eval_parallel_workers or eval_parallel_workers)
        if workers <= 1:
            for payload in full_payloads:
                append_result(
                    _run_candidate_task(payload),
                    full_eval_summaries,
                    phase="neighborhood_full_eval_done",
                )
        else:
            with futures.ProcessPoolExecutor(max_workers=workers) as executor:
                pending = {
                    executor.submit(_run_candidate_task, payload): payload
                    for payload in full_payloads
                }
                for future in futures.as_completed(pending):
                    append_result(
                        future.result(),
                        full_eval_summaries,
                        phase="neighborhood_full_eval_done",
                    )

    final_summaries = full_eval_summaries or eval_summaries
    best_eval = None
    if final_summaries:
        best_eval = min(
            final_summaries,
            key=lambda row: float(row["summary"].get("total_ttt", np.inf)),
        )
    result = {
        "format_version": NEIGHBORHOOD_FORMAT,
        "scenario": scenario,
        "source_artifact": str(source_artifact),
        "selection": selection,
        "output_dir": str(output_dir),
        "cache_output_dir": str(cache_output_dir),
        "pstack_summary": str(pstack_summary_path),
        "pstack_total_ttt": float(pstack_total),
        "target_5pct_total_ttt": float(0.95 * pstack_total),
        "candidate_count": int(len(candidates)),
        "eval_parallel_workers": int(eval_parallel_workers),
        "tail_step_limit": tail_step_limit,
        "full_validate_top_k": int(full_validate_top_k),
        "full_eval_parallel_workers": int(full_eval_parallel_workers or eval_parallel_workers),
        "placement_steps": list(map(int, placement_steps)),
        "scales": list(map(float, scales)),
        "repeat_sizes": list(map(int, repeat_sizes)),
        "anchor_summaries": {
            str(step): summary for step, summary in sorted(anchor_summaries.items())
        },
        "eval_summaries": eval_summaries,
        "full_eval_summaries": full_eval_summaries,
        "best_eval": best_eval,
        "meets_5pct_target": bool(
            best_eval is not None
            and float(best_eval["summary"].get("total_ttt", np.inf)) <= 0.95 * pstack_total
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "neighborhood_summary.json", result)
    print(
        json.dumps(
            {
                "passed": bool(result["meets_5pct_target"]),
                "best_total_ttt": (
                    None if best_eval is None else float(best_eval["summary"]["total_ttt"])
                ),
                "target_5pct_total_ttt": float(0.95 * pstack_total),
                "output": str(output_dir / "neighborhood_summary.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _parse_float_csv(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        return tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(float(value) for value in raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-output-dir", type=Path, required=True)
    parser.add_argument("--pstack-summary", type=Path, required=True)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--selection", default="best_h12")
    parser.add_argument("--placement-steps", default="")
    parser.add_argument("--scales", default="0.5,0.75,1.0")
    parser.add_argument("--repeat-sizes", default="2")
    parser.add_argument("--subset-pool-limit", type=int, default=6)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--no-sign-flips", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--eval-parallel-workers", type=int, default=1)
    parser.add_argument("--tail-step-limit", type=int)
    parser.add_argument("--full-validate-top-k", type=int, default=0)
    parser.add_argument("--full-eval-parallel-workers", type=int)
    parser.add_argument("--cache-mode", choices=("direct", "manifest"), default="direct")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_residual_neighborhood_search(
        source_artifact=args.source_artifact,
        output_dir=args.output_dir,
        cache_output_dir=args.cache_output_dir,
        pstack_summary_path=args.pstack_summary,
        scenario=args.scenario,
        selection=args.selection,
        placement_steps=parse_int_csv(args.placement_steps),
        scales=_parse_float_csv(args.scales),
        repeat_sizes=parse_int_csv(args.repeat_sizes),
        subset_pool_limit=int(args.subset_pool_limit),
        max_subset_size=int(args.max_subset_size),
        include_sign_flips=not bool(args.no_sign_flips),
        max_candidates=int(args.max_candidates),
        eval_parallel_workers=int(args.eval_parallel_workers),
        tail_step_limit=args.tail_step_limit,
        full_validate_top_k=int(args.full_validate_top_k),
        full_eval_parallel_workers=args.full_eval_parallel_workers,
        cache_mode=args.cache_mode,
        verbose=not bool(args.quiet),
    )


if __name__ == "__main__":
    main()
