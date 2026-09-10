"""Iterate current-contract residual schedules toward a 5% P-Stack improvement.

This runner stays deliberately small.  It reuses the existing cached-prefix,
residual sampler, and cached-tail evaluator, then records which executable
response schedules were actually tested against the 14400 s P-Stack baseline.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl_leader.evaluate_cached_tail_residual_schedule import (
    load_cached_prefix,
    run_cached_tail_residual_schedule,
)
from rl_leader.evaluate_fixed_residual_schedule import (
    load_sampled_residual_schedule,
    merge_schedules,
)
from rl_leader.run_tail_label_generation import run_tail_label_generation
from rl_leader.sample_cached_tail_residuals import (
    SAMPLE_FORMAT,
    parse_magnitudes,
    run_cached_tail_residual_sampler,
    select_h12_records,
)
from rl_leader.env import RLLeaderEnv


LOOP_FORMAT = "residual_5pct_loop_v1"


@dataclass(frozen=True)
class StepArtifact:
    step: int
    seed: int
    path: Path
    gain: float
    positive: bool
    candidate_id: str


@dataclass(frozen=True)
class SamplerJob:
    step: int
    seed: int
    replay_cache: Path
    output_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int_csv(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    else:
        values = tuple(int(value) for value in raw)
    if len(set(values)) != len(values):
        raise ValueError("integer CSV values must be unique")
    return values


def step_cache_path(cache_output_dir: Path, scenario: str, step: int) -> Path:
    return Path(cache_output_dir) / f"{scenario}_step{int(step):02d}" / "replay_cache.pkl"


def _write_direct_cache_payload(
    *,
    cache_path: Path,
    scenario: str,
    t_total: float,
    step: int,
    env,
    anchor_context,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump((env, anchor_context), handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(
        cache_path.with_suffix(".json"),
        {
            "format_version": "direct_pstack_replay_cache_v1",
            "scenario": scenario,
            "t_total_sec": float(t_total),
            "control_step": int(step),
            "env_step_idx": int(env.step_idx),
            "warmup_steps": int(env.warmup),
            "state_fingerprint": str(anchor_context.state_fingerprint),
        },
    )


def write_direct_pstack_replay_caches(
    *,
    cache_paths: dict[int, Path],
    scenario: str,
    t_total: float,
) -> dict[int, Path]:
    missing = {
        int(step): Path(path)
        for step, path in cache_paths.items()
        if not Path(path).is_file() or not Path(path).with_suffix(".json").is_file()
    }
    if not missing:
        return {int(step): Path(path) for step, path in cache_paths.items()}
    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=float(t_total),
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    pending = set(missing)
    max_step = max(pending)
    done = False
    while not done and pending:
        actual_step = int(env.step_idx - env.warmup)
        print(json.dumps({
            "phase": "direct_cache_replay",
            "scenario": scenario,
            "control_step": int(actual_step),
            "pending": list(map(int, sorted(pending))),
        }, sort_keys=True), flush=True)
        if actual_step in pending:
            anchor_context = env.prepare_pstack_anchor_context()
            _write_direct_cache_payload(
                cache_path=missing[actual_step],
                scenario=scenario,
                t_total=float(t_total),
                step=actual_step,
                env=env,
                anchor_context=anchor_context,
            )
            print(json.dumps({
                "phase": "direct_cache",
                "scenario": scenario,
                "control_step": int(actual_step),
                "path": str(missing[actual_step]),
            }, sort_keys=True), flush=True)
            pending.remove(actual_step)
            if not pending:
                break
        if actual_step > max_step:
            break
        context = env.prepare_pstack_anchor_context()
        _obs, _reward, done, _info, _extra = env.step_prepared_optimizer_anchor(
            context,
            sync_follower_state=True,
        )
    if pending:
        raise RuntimeError(
            "could not replay P-Stack to control step(s): "
            + ",".join(map(str, sorted(pending)))
        )
    return {int(step): Path(path) for step, path in cache_paths.items()}


def write_direct_pstack_replay_cache(
    *,
    cache_path: Path,
    scenario: str,
    t_total: float,
    step: int,
) -> Path:
    return write_direct_pstack_replay_caches(
        cache_paths={int(step): Path(cache_path)},
        scenario=scenario,
        t_total=float(t_total),
    )[int(step)]


def peak_steps_from_trace(
    trace_path: Path,
    *,
    min_step: int,
    max_step: int,
    limit: int,
) -> tuple[int, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows = []
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["control_step"])
        if min_step <= step <= max_step:
            rows.append((float(row["interval_ttt"]), step))
    if not rows:
        raise ValueError("P-Stack trace has no rows in the requested step window")
    top = sorted(rows, key=lambda item: (item[0], -item[1]), reverse=True)[:limit]
    return tuple(sorted(step for _ttt, step in top))


def ensure_replay_caches(
    *,
    manifest_path: Path | None,
    cache_output_dir: Path,
    scenario: str,
    steps: Sequence[int],
    t_total: float,
    cache_mode: str,
) -> dict[int, Path]:
    missing = [
        int(step)
        for step in steps
        if not step_cache_path(cache_output_dir, scenario, int(step)).is_file()
    ]
    if missing:
        if cache_mode == "manifest":
            if manifest_path is None:
                raise ValueError("manifest cache mode requires --manifest")
            run_tail_label_generation(
                manifest_path,
                cache_output_dir,
                scenarios=(scenario,),
                policy_steps=tuple(sorted(missing)),
                h1_only=True,
                keep_going=True,
            )
        elif cache_mode == "direct":
            write_direct_pstack_replay_caches(
                cache_paths={
                    int(step): step_cache_path(cache_output_dir, scenario, int(step))
                    for step in missing
                },
                scenario=scenario,
                t_total=float(t_total),
            )
        else:
            raise ValueError(f"unknown cache_mode: {cache_mode}")
    caches = {
        int(step): step_cache_path(cache_output_dir, scenario, int(step))
        for step in steps
    }
    absent = [str(path) for path in caches.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("missing replay cache(s): " + ", ".join(absent))
    return caches


def _horizon_key(summary: dict[str, Any]) -> str:
    return str(int(summary.get("horizon_steps", 12)))


def _best_record(summary: dict[str, Any]) -> dict[str, Any] | None:
    record = summary.get("best_h12")
    return record if isinstance(record, dict) else None


def sampler_gain(summary: dict[str, Any]) -> float:
    record = _best_record(summary)
    if record is None:
        return float("-inf")
    label = record.get("horizon_labels", {}).get(_horizon_key(summary), {})
    return float(label.get("ttt_gain", float("-inf")))


def sampler_positive(summary: dict[str, Any]) -> bool:
    record = _best_record(summary)
    if record is None:
        return False
    label = record.get("horizon_labels", {}).get(_horizon_key(summary), {})
    return bool(label.get("positive", False) and label.get("validity_gate_pass", False))


def sampler_candidate_id(summary: dict[str, Any]) -> str:
    record = _best_record(summary)
    return "" if record is None else str(record.get("candidate_id", ""))


def sampler_dir(
    output_dir: Path,
    *,
    step: int,
    seed: int,
    max_h1_candidates: int,
    max_h12_candidates: int,
    horizon_steps: int,
) -> Path:
    return (
        Path(output_dir)
        / "samplers"
        / (
            f"step{int(step):02d}_seed{int(seed)}"
            f"_h1{int(max_h1_candidates)}_h{int(horizon_steps)}x{int(max_h12_candidates)}"
        )
    )


def run_or_load_sampler(
    *,
    replay_cache: Path,
    output_dir: Path,
    seed: int,
    max_h1_candidates: int,
    max_h12_candidates: int,
    horizon_steps: int,
    magnitudes: tuple[float, ...],
    pstack_summary: dict[str, Any],
    workers: int,
    native_rollout_cache_dir: Path | None = None,
    h1_chunks_per_worker: int = 1,
    h12_chunks_per_worker: int = 1,
    verbose: bool = True,
) -> dict[str, Any]:
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.is_file():
        return _read_json(summary_path)
    partial = recover_partial_sampler_summary(
        output_dir=output_dir,
        seed=int(seed),
        max_h1_candidates=int(max_h1_candidates),
        max_h12_candidates=int(max_h12_candidates),
        horizon_steps=int(horizon_steps),
        magnitudes=magnitudes,
        pstack_summary=pstack_summary,
        workers=int(workers),
        native_rollout_cache_dir=native_rollout_cache_dir,
        h1_chunks_per_worker=int(h1_chunks_per_worker),
        h12_chunks_per_worker=int(h12_chunks_per_worker),
    )
    if partial is not None:
        return partial
    return run_cached_tail_residual_sampler(
        replay_cache=replay_cache,
        output_dir=output_dir,
        seed=int(seed),
        max_h1_candidates=int(max_h1_candidates),
        max_h12_candidates=int(max_h12_candidates),
        horizon_steps=int(horizon_steps),
        magnitudes=magnitudes,
        pstack_summary=pstack_summary,
        workers=int(workers),
        native_rollout_cache_dir=native_rollout_cache_dir,
        h1_chunks_per_worker=int(h1_chunks_per_worker),
        h12_chunks_per_worker=int(h12_chunks_per_worker),
        verbose=bool(verbose),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def recover_partial_sampler_summary(
    *,
    output_dir: Path,
    seed: int,
    max_h1_candidates: int,
    max_h12_candidates: int,
    horizon_steps: int,
    magnitudes: tuple[float, ...],
    pstack_summary: dict[str, Any],
    workers: int,
    native_rollout_cache_dir: Path | None,
    h1_chunks_per_worker: int,
    h12_chunks_per_worker: int,
) -> dict[str, Any] | None:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    h1_path = output_dir / "h1_records.jsonl"
    h12_path = output_dir / "h12_records.jsonl"
    h12_records = _read_jsonl(h12_path)
    if summary_path.is_file():
        return None
    h1_records = _read_jsonl(h1_path)
    if not h12_records and int(horizon_steps) == 1 and h1_records:
        h12_records = select_h12_records(
            h1_records,
            max_h12_candidates=int(max_h12_candidates),
            seed=int(seed) + 104729,
        )
        for record in h12_records:
            record["selected_for_h12"] = True
        h12_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in h12_records)
            + ("\n" if h12_records else ""),
            encoding="utf-8",
        )
    if not h12_records:
        return None
    control_step = int(h12_records[0]["control_step"])
    horizon_key = str(int(horizon_steps))
    best_h12 = max(
        h12_records,
        key=lambda row: (
            float(row["horizon_labels"][horizon_key]["ttt_gain"]),
            bool(row["horizon_labels"][horizon_key].get("positive", False)),
            str(row.get("candidate_id", "")),
        ),
    )
    summary = {
        "format_version": SAMPLE_FORMAT,
        "recovered_partial": True,
        "output": str(summary_path),
        "scenario": None,
        "control_step": int(control_step),
        "seed": int(seed),
        "max_h1_candidates": int(max_h1_candidates),
        "max_h12_candidates": int(max_h12_candidates),
        "horizon_steps": int(horizon_steps),
        "magnitudes": list(map(float, magnitudes)),
        "workers": int(workers),
        "h1_chunks_per_worker": int(h1_chunks_per_worker),
        "native_rollout_cache_dir": (
            None
            if native_rollout_cache_dir is None
            else str(Path(native_rollout_cache_dir))
        ),
        "h12_chunks_per_worker": int(h12_chunks_per_worker),
        "h1_records_path": str(h1_path),
        "h12_records_path": str(h12_path),
        "h1_records": h1_records,
        "h12_records": h12_records,
        "h1_record_count": int(len(h1_records)),
        "h12_record_count": int(len(h12_records)),
        "h12_positive_count": int(sum(
            bool(row.get("horizon_labels", {}).get(horizon_key, {}).get("positive", False))
            for row in h12_records
        )),
        "best_h12": best_h12,
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
        summary.update({
            "pstack_total_ttt": pstack_total,
            "target_5pct_total_ttt": float(0.95 * pstack_total),
            "required_full_run_gain_5pct": float(0.05 * pstack_total),
        })
    _write_json(summary_path, summary)
    return summary


def build_sampler_jobs(
    *,
    caches: dict[int, Path],
    output_dir: Path,
    steps: Sequence[int],
    seeds: Sequence[int],
    max_h1_candidates: int,
    max_h12_candidates: int,
    horizon_steps: int,
) -> tuple[SamplerJob, ...]:
    jobs: list[SamplerJob] = []
    for step in steps:
        for seed in seeds:
            jobs.append(
                SamplerJob(
                    step=int(step),
                    seed=int(seed),
                    replay_cache=Path(caches[int(step)]),
                    output_dir=sampler_dir(
                        output_dir,
                        step=int(step),
                        seed=int(seed),
                        max_h1_candidates=int(max_h1_candidates),
                        max_h12_candidates=int(max_h12_candidates),
                        horizon_steps=int(horizon_steps),
                    ),
                )
            )
    return tuple(jobs)


def _run_sampler_job_task(payload: dict[str, Any]) -> dict[str, Any]:
    job = payload["job"]
    return run_or_load_sampler(
        replay_cache=Path(job.replay_cache),
        output_dir=Path(job.output_dir),
        seed=int(job.seed),
        max_h1_candidates=int(payload["max_h1_candidates"]),
        max_h12_candidates=int(payload["max_h12_candidates"]),
        horizon_steps=int(payload["horizon_steps"]),
        magnitudes=tuple(float(value) for value in payload["magnitudes"]),
        pstack_summary=dict(payload["pstack_summary"]),
        workers=int(payload["sampler_workers"]),
        native_rollout_cache_dir=(
            None
            if payload["native_rollout_cache_dir"] is None
            else Path(payload["native_rollout_cache_dir"])
        ),
        h1_chunks_per_worker=int(payload["h1_chunks_per_worker"]),
        h12_chunks_per_worker=int(payload["h12_chunks_per_worker"]),
        verbose=bool(payload["sampler_verbose"]),
    )


def run_sampler_jobs(
    *,
    jobs: Sequence[SamplerJob],
    max_h1_candidates: int,
    max_h12_candidates: int,
    horizon_steps: int,
    magnitudes: tuple[float, ...],
    pstack_summary: dict[str, Any],
    sampler_workers: int,
    native_rollout_cache_dir: Path | None,
    h1_chunks_per_worker: int,
    h12_chunks_per_worker: int,
    sampler_verbose: bool,
    step_parallel_workers: int,
) -> list[dict[str, Any]]:
    payloads = [
        {
            "job": job,
            "max_h1_candidates": int(max_h1_candidates),
            "max_h12_candidates": int(max_h12_candidates),
            "horizon_steps": int(horizon_steps),
            "magnitudes": tuple(float(value) for value in magnitudes),
            "pstack_summary": dict(pstack_summary),
            "sampler_workers": int(sampler_workers),
            "native_rollout_cache_dir": (
                None
                if native_rollout_cache_dir is None
                else str(Path(native_rollout_cache_dir))
            ),
            "h1_chunks_per_worker": int(h1_chunks_per_worker),
            "h12_chunks_per_worker": int(h12_chunks_per_worker),
            "sampler_verbose": bool(sampler_verbose),
        }
        for job in jobs
    ]
    summaries: list[dict[str, Any]] = []
    if int(step_parallel_workers) <= 1:
        for payload in payloads:
            summary = _run_sampler_job_task(payload)
            summaries.append(summary)
            print(json.dumps({
                "phase": "sampler_done",
                "control_step": int(summary["control_step"]),
                "seed": int(summary["seed"]),
                "best_gain": sampler_gain(summary),
                "positive": sampler_positive(summary),
            }, sort_keys=True), flush=True)
        return summaries

    with futures.ProcessPoolExecutor(max_workers=int(step_parallel_workers)) as executor:
        pending = {
            executor.submit(_run_sampler_job_task, payload): payload["job"]
            for payload in payloads
        }
        for future in futures.as_completed(pending):
            job = pending[future]
            summary = future.result()
            summaries.append(summary)
            print(json.dumps({
                "phase": "sampler_done",
                "control_step": int(summary["control_step"]),
                "seed": int(summary["seed"]),
                "best_gain": sampler_gain(summary),
                "positive": sampler_positive(summary),
                "output": str(job.output_dir / "summary.json"),
            }, sort_keys=True), flush=True)
    return sorted(
        summaries,
        key=lambda row: (int(row["control_step"]), int(row["seed"])),
    )


def select_best_step_artifacts(
    summaries: Sequence[dict[str, Any]],
    *,
    require_positive: bool = True,
) -> list[StepArtifact]:
    by_step: dict[int, StepArtifact] = {}
    for summary in summaries:
        positive = sampler_positive(summary)
        if require_positive and not positive:
            continue
        artifact = StepArtifact(
            step=int(summary["control_step"]),
            seed=int(summary["seed"]),
            path=Path(summary["output"] if "output" in summary else Path(summary["h12_records_path"]).parent / "summary.json"),
            gain=sampler_gain(summary),
            positive=positive,
            candidate_id=sampler_candidate_id(summary),
        )
        current = by_step.get(artifact.step)
        if current is None or (artifact.positive, artifact.gain, artifact.seed) > (
            current.positive,
            current.gain,
            current.seed,
        ):
            by_step[artifact.step] = artifact
    return [by_step[step] for step in sorted(by_step)]


def candidate_artifact_sets(
    artifacts: Sequence[StepArtifact],
    *,
    max_schedules: int,
    max_combo_size: int = 4,
    combo_pool_limit: int = 8,
) -> list[tuple[str, list[StepArtifact]]]:
    if max_schedules <= 0 or not artifacts:
        return []
    by_gain = sorted(artifacts, key=lambda item: (item.gain, -item.step), reverse=True)
    sets: list[tuple[str, list[StepArtifact]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(label: str, subset: Sequence[StepArtifact]) -> None:
        if len(sets) >= max_schedules or not subset:
            return
        subset = sorted(subset, key=lambda item: item.step)
        key = tuple(item.step for item in subset)
        if key not in seen:
            seen.add(key)
            sets.append((label, list(subset)))

    max_combo_size = max(1, int(max_combo_size))
    combo_pool_limit = max(1, int(combo_pool_limit))
    for count in range(1, min(len(by_gain), max_combo_size, max_schedules) + 1):
        add(f"top{count}_by_horizon_gain", by_gain[:count])

    for artifact in by_gain:
        add(f"single_step_{artifact.step:02d}", [artifact])

    pool = by_gain[: min(len(by_gain), combo_pool_limit)]
    scored_combos: list[
        tuple[float, int, int, tuple[int, ...], tuple[StepArtifact, ...]]
    ] = []
    for size in range(2, min(max_combo_size, len(pool)) + 1):
        for combo in itertools.combinations(pool, size):
            steps = [item.step for item in combo]
            scored_combos.append((
                float(sum(item.gain for item in combo)),
                -int(max(steps) - min(steps)),
                -int(min(steps)),
                tuple(sorted(steps)),
                combo,
            ))
    for rank, (_score, _spread, _first_step, _steps, combo) in enumerate(
        sorted(
            scored_combos,
            key=lambda row: (row[0], row[1], row[2], row[3]),
            reverse=True,
        ),
        start=1,
    ):
        add(f"combo_rank{rank:02d}_horizon_gain", combo)

    if len(artifacts) > 1:
        add("all_positive_steps", sorted(artifacts, key=lambda item: item.step))
    return sets[:max_schedules]


def _safe_eval_name(label: str, artifacts: Sequence[StepArtifact]) -> str:
    steps = "-".join(f"{item.step:02d}" for item in artifacts)
    return f"{label}_steps_{steps}"


def _retryable_eval_dir(
    *,
    output_dir: Path,
    label: str,
    artifacts: Sequence[StepArtifact],
) -> Path:
    base = Path(output_dir) / "tail_evals" / _safe_eval_name(label, artifacts)
    if (base / "summary.json").is_file() or not (base / "trace.jsonl").exists():
        return base
    for index in range(1, 1000):
        candidate = base.with_name(f"{base.name}_retry{index:02d}")
        if (candidate / "summary.json").is_file() or not (
            candidate / "trace.jsonl"
        ).exists():
            return candidate
    raise RuntimeError(f"too many retry eval directories for {base}")


def evaluate_artifact_set(
    *,
    caches: dict[int, Path],
    artifacts: Sequence[StepArtifact],
    output_dir: Path,
    pstack_summary: dict[str, Any],
    label: str,
    eval_verbose: bool = True,
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("cannot evaluate an empty artifact set")
    earliest = min(item.step for item in artifacts)
    eval_dir = _retryable_eval_dir(
        output_dir=output_dir,
        label=label,
        artifacts=artifacts,
    )
    summary_path = eval_dir / "summary.json"
    if summary_path.is_file():
        return _read_json(summary_path)
    env, _context = load_cached_prefix(caches[earliest])
    action_names = tuple(env.action_schema.names)
    schedules = [
        load_sampled_residual_schedule(
            item.path,
            action_names,
            selection="best_positive_h12" if item.positive else "best_h12",
        )
        for item in artifacts
    ]
    schedule = merge_schedules(*schedules)
    return run_cached_tail_residual_schedule(
        replay_cache=caches[earliest],
        schedule=schedule,
        output_dir=eval_dir,
        pstack_summary=pstack_summary,
        force_without_h3_gate=True,
        verbose=bool(eval_verbose),
    )


def _run_eval_job_task(payload: dict[str, Any]) -> dict[str, Any]:
    summary = evaluate_artifact_set(
        caches={int(step): Path(path) for step, path in payload["caches"].items()},
        artifacts=tuple(payload["artifacts"]),
        output_dir=Path(payload["output_dir"]),
        pstack_summary=dict(payload["pstack_summary"]),
        label=str(payload["label"]),
        eval_verbose=bool(payload["eval_verbose"]),
    )
    return {
        "label": str(payload["label"]),
        "artifacts": list(payload["artifacts"]),
        "summary": summary,
    }


def run_residual_5pct_loop(
    *,
    manifest_path: Path | None,
    output_dir: Path,
    cache_output_dir: Path,
    pstack_summary_path: Path,
    pstack_trace_path: Path,
    scenario: str = "sweet_170_incident_w60",
    policy_steps: tuple[int, ...] = (),
    peak_min_step: int = 17,
    peak_max_step: int = 30,
    peak_limit: int = 8,
    seeds: tuple[int, ...] = (2026090301,),
    max_h1_candidates: int = 80,
    max_h12_candidates: int = 16,
    horizon_steps: int = 3,
    magnitudes: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_schedules: int = 4,
    max_combo_size: int = 4,
    combo_pool_limit: int = 8,
    sampler_workers: int = 1,
    native_rollout_cache_dir: Path | None = None,
    h1_chunks_per_worker: int = 1,
    h12_chunks_per_worker: int = 1,
    sampler_verbose: bool = True,
    eval_verbose: bool = True,
    stop_after_target: bool = True,
    require_positive_artifacts: bool = True,
    eval_parallel_workers: int = 1,
    step_parallel_workers: int = 1,
    cache_mode: str = "direct",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    cache_output_dir = Path(cache_output_dir)
    pstack_summary = _read_json(pstack_summary_path)
    pstack_total = float(pstack_summary["total_ttt"])
    target_total = 0.95 * pstack_total
    selected_steps = policy_steps or peak_steps_from_trace(
        pstack_trace_path,
        min_step=int(peak_min_step),
        max_step=int(peak_max_step),
        limit=int(peak_limit),
    )

    caches = ensure_replay_caches(
        manifest_path=manifest_path,
        cache_output_dir=cache_output_dir,
        scenario=scenario,
        steps=selected_steps,
        t_total=float(pstack_summary.get("t_total_sec", 14400.0)),
        cache_mode=cache_mode,
    )

    sampler_summaries = run_sampler_jobs(
        jobs=build_sampler_jobs(
            caches=caches,
            output_dir=output_dir,
            steps=selected_steps,
            seeds=seeds,
            max_h1_candidates=int(max_h1_candidates),
            max_h12_candidates=int(max_h12_candidates),
            horizon_steps=int(horizon_steps),
        ),
        max_h1_candidates=int(max_h1_candidates),
        max_h12_candidates=int(max_h12_candidates),
        horizon_steps=int(horizon_steps),
        magnitudes=magnitudes,
        pstack_summary=pstack_summary,
        sampler_workers=int(sampler_workers),
        native_rollout_cache_dir=native_rollout_cache_dir,
        h1_chunks_per_worker=int(h1_chunks_per_worker),
        h12_chunks_per_worker=int(h12_chunks_per_worker),
        sampler_verbose=bool(sampler_verbose),
        step_parallel_workers=int(step_parallel_workers),
    )

    artifacts = select_best_step_artifacts(
        sampler_summaries,
        require_positive=bool(require_positive_artifacts),
    )
    eval_jobs = candidate_artifact_sets(
        artifacts,
        max_schedules=int(max_schedules),
        max_combo_size=int(max_combo_size),
        combo_pool_limit=int(combo_pool_limit),
    )
    eval_summaries = []

    def append_eval(label: str, group: Sequence[StepArtifact], summary: dict[str, Any]) -> bool:
        eval_summaries.append(
            {
                "label": label,
                "steps": [int(item.step) for item in group],
                "artifacts": [str(item.path) for item in group],
                "summary": summary,
            }
        )
        print(json.dumps({
            "phase": "eval_done",
            "label": label,
            "steps": [int(item.step) for item in group],
            "total_ttt": float(summary.get("total_ttt", np.inf)),
            "vs_pstack_percent": (
                None
                if "vs_pstack_percent" not in summary
                else float(summary["vs_pstack_percent"])
            ),
            "meets_5pct_target": bool(summary.get("meets_5pct_target", False)),
        }, sort_keys=True), flush=True)
        return bool(summary.get("meets_5pct_target", False))

    if int(eval_parallel_workers) <= 1:
        for label, group in eval_jobs:
            summary = evaluate_artifact_set(
                caches=caches,
                artifacts=group,
                output_dir=output_dir,
                pstack_summary=pstack_summary,
                label=label,
                eval_verbose=bool(eval_verbose),
            )
            if append_eval(label, group, summary) and stop_after_target:
                break
    else:
        payloads = [
            {
                "label": label,
                "artifacts": tuple(group),
                "caches": {int(step): str(path) for step, path in caches.items()},
                "output_dir": str(output_dir),
                "pstack_summary": dict(pstack_summary),
                "eval_verbose": bool(eval_verbose),
            }
            for label, group in eval_jobs
        ]
        with futures.ProcessPoolExecutor(max_workers=int(eval_parallel_workers)) as executor:
            pending = {
                executor.submit(_run_eval_job_task, payload): payload
                for payload in payloads
            }
            for future in futures.as_completed(pending):
                payload = pending[future]
                result = future.result()
                group = tuple(result["artifacts"])
                if append_eval(str(result["label"]), group, dict(result["summary"])):
                    if stop_after_target:
                        for remaining in pending:
                            remaining.cancel()
                        break

    best_eval = None
    if eval_summaries:
        best_eval = min(
            eval_summaries,
            key=lambda row: float(row["summary"].get("total_ttt", np.inf)),
        )

    result = {
        "format_version": LOOP_FORMAT,
        "scenario": scenario,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "cache_output_dir": str(cache_output_dir),
        "pstack_summary": str(pstack_summary_path),
        "pstack_trace": str(pstack_trace_path),
        "pstack_total_ttt": pstack_total,
        "target_5pct_total_ttt": float(target_total),
        "required_gain_5pct": float(pstack_total - target_total),
        "policy_steps": list(map(int, selected_steps)),
        "seeds": list(map(int, seeds)),
        "max_h1_candidates": int(max_h1_candidates),
        "max_h12_candidates": int(max_h12_candidates),
        "horizon_steps": int(horizon_steps),
        "max_combo_size": int(max_combo_size),
        "combo_pool_limit": int(combo_pool_limit),
        "sampler_workers": int(sampler_workers),
        "native_rollout_cache_dir": (
            None
            if native_rollout_cache_dir is None
            else str(Path(native_rollout_cache_dir))
        ),
        "h1_chunks_per_worker": int(h1_chunks_per_worker),
        "h12_chunks_per_worker": int(h12_chunks_per_worker),
        "sampler_verbose": bool(sampler_verbose),
        "eval_verbose": bool(eval_verbose),
        "stop_after_target": bool(stop_after_target),
        "require_positive_artifacts": bool(require_positive_artifacts),
        "eval_parallel_workers": int(eval_parallel_workers),
        "step_parallel_workers": int(step_parallel_workers),
        "cache_mode": cache_mode,
        "magnitudes": list(map(float, magnitudes)),
        "sampler_summaries": [
            {
                "step": int(row["control_step"]),
                "seed": int(row["seed"]),
                "path": str(Path(row.get("h12_records_path", "")).parent / "summary.json"),
                "h1_unique_response_count": int(row.get("h1_unique_response_count", 0)),
                "h12_positive_count": int(row.get("h12_positive_count", 0)),
                "best_gain": sampler_gain(row),
                "best_positive": sampler_positive(row),
                "best_candidate_id": sampler_candidate_id(row),
            }
            for row in sampler_summaries
        ],
        "selected_artifacts": [
            {
                "step": int(item.step),
                "seed": int(item.seed),
                "path": str(item.path),
                "gain": float(item.gain),
                "positive": bool(item.positive),
                "candidate_id": item.candidate_id,
            }
            for item in artifacts
        ],
        "eval_summaries": eval_summaries,
        "best_eval": best_eval,
        "meets_5pct_target": bool(
            best_eval is not None
            and float(best_eval["summary"].get("total_ttt", np.inf)) <= target_total
        ),
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "loop_summary.json", result)
    print(json.dumps({
        "passed": bool(result["meets_5pct_target"]),
        "policy_steps": result["policy_steps"],
        "selected_positive_steps": [int(item.step) for item in artifacts],
        "best_total_ttt": (
            None if best_eval is None else float(best_eval["summary"]["total_ttt"])
        ),
        "target_5pct_total_ttt": float(target_total),
        "output": str(output_dir / "loop_summary.json"),
    }, indent=2, sort_keys=True), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-output-dir", type=Path, required=True)
    parser.add_argument("--pstack-summary", type=Path, required=True)
    parser.add_argument("--pstack-trace", type=Path, required=True)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--policy-steps", default="")
    parser.add_argument("--peak-min-step", type=int, default=17)
    parser.add_argument("--peak-max-step", type=int, default=30)
    parser.add_argument("--peak-limit", type=int, default=8)
    parser.add_argument("--seeds", default="2026090301")
    parser.add_argument("--max-h1-candidates", type=int, default=80)
    parser.add_argument("--max-h12-candidates", type=int, default=16)
    parser.add_argument("--horizon-steps", type=int, default=3)
    parser.add_argument("--magnitudes", default="0.25,0.5,0.75")
    parser.add_argument("--max-schedules", type=int, default=4)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--combo-pool-limit", type=int, default=8)
    parser.add_argument("--sampler-workers", type=int, default=1)
    parser.add_argument("--native-rollout-cache-dir", type=Path)
    parser.add_argument("--h1-chunks-per-worker", type=int, default=1)
    parser.add_argument("--h12-chunks-per-worker", type=int, default=1)
    parser.add_argument("--quiet-samplers", action="store_true")
    parser.add_argument("--quiet-evals", action="store_true")
    parser.add_argument("--continue-after-target", action="store_true")
    parser.add_argument("--allow-nonpositive-artifacts", action="store_true")
    parser.add_argument("--eval-parallel-workers", type=int, default=1)
    parser.add_argument("--step-parallel-workers", type=int, default=1)
    parser.add_argument(
        "--cache-mode",
        choices=("direct", "manifest"),
        default="direct",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_residual_5pct_loop(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        cache_output_dir=args.cache_output_dir,
        pstack_summary_path=args.pstack_summary,
        pstack_trace_path=args.pstack_trace,
        scenario=args.scenario,
        policy_steps=parse_int_csv(args.policy_steps),
        peak_min_step=int(args.peak_min_step),
        peak_max_step=int(args.peak_max_step),
        peak_limit=int(args.peak_limit),
        seeds=parse_int_csv(args.seeds),
        max_h1_candidates=int(args.max_h1_candidates),
        max_h12_candidates=int(args.max_h12_candidates),
        horizon_steps=int(args.horizon_steps),
        magnitudes=parse_magnitudes(args.magnitudes),
        max_schedules=int(args.max_schedules),
        max_combo_size=int(args.max_combo_size),
        combo_pool_limit=int(args.combo_pool_limit),
        sampler_workers=int(args.sampler_workers),
        native_rollout_cache_dir=args.native_rollout_cache_dir,
        h1_chunks_per_worker=int(args.h1_chunks_per_worker),
        h12_chunks_per_worker=int(args.h12_chunks_per_worker),
        sampler_verbose=not bool(args.quiet_samplers),
        eval_verbose=not bool(args.quiet_evals),
        stop_after_target=not bool(args.continue_after_target),
        require_positive_artifacts=not bool(args.allow_nonpositive_artifacts),
        eval_parallel_workers=int(args.eval_parallel_workers),
        step_parallel_workers=int(args.step_parallel_workers),
        cache_mode=args.cache_mode,
    )


if __name__ == "__main__":
    main()
