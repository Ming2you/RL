"""Evaluate P-CENT-guided executable residuals from cached P-Stack states.

This runner keeps P-CENT as a privileged proposal source only.  Each generated
candidate is still a coordination residual that must pass through the existing
urban/freeway followers before it receives H1/H12 labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl_leader.diagnose_pcent_reachability import (
    _response_scales_and_families,
    structured_residual_candidates,
)
from rl_leader.diagnose_reachable_candidate_attribution import _rollout_pstack
from rl_leader.evaluate_cached_tail_residual_schedule import load_cached_prefix
from rl_leader.oracle_candidate_ablation import residual_sha256
from rl_leader.run_residual_5pct_loop import (
    ensure_replay_caches,
    parse_int_csv,
)
from rl_leader.sample_cached_tail_residuals import (
    SAMPLE_FORMAT,
    SampledResidualSpec,
    _evaluate_records,
    _h1_gain,
    _load_native_rollout_cache,
    _spec,
    _write_native_rollout_cache,
    evaluation_horizons,
    native_rollout_cache_path,
    parse_magnitudes,
    select_h12_records,
)
from src.controllers.centralized_mpc import CentralizedMPC


PERCENT_SEARCH_FORMAT = "cached_pcent_guided_residual_search_v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pcent_guided_specs_from_candidates(
    action_names: Sequence[str],
    candidates: Sequence[tuple[str, np.ndarray]],
) -> list[SampledResidualSpec]:
    """Convert raw P-CENT residual candidates to sampler-compatible specs."""
    specs: list[SampledResidualSpec] = []
    seen: set[str] = set()
    names = tuple(map(str, action_names))
    for label, residual in candidates:
        values = np.asarray(residual, dtype=np.float32).reshape(-1)
        if values.shape != (len(names),):
            raise ValueError("candidate residual has the wrong dimension")
        if np.linalg.norm(values, ord=2) <= 1.0e-9:
            continue
        digest = residual_sha256(values)
        if digest in seen:
            continue
        seen.add(digest)
        parts = str(label).split(":")
        family = "pcent_guided"
        if len(parts) >= 2:
            family = f"pcent_{parts[1]}"
        specs.append(_spec(
            names,
            label=str(label),
            generator="pcent_cached",
            family=family,
            residual=values,
            metadata={"proposal_source": "centralized_mpc_direction"},
        ))
    return specs


def build_pcent_guided_specs(
    env,
    context,
    magnitudes: tuple[float, ...],
) -> tuple[list[SampledResidualSpec], dict[str, Any]]:
    """Build executable residual proposals oriented by same-state P-CENT."""
    target_env = copy.deepcopy(env)
    forecast = list(copy.deepcopy(context.forecast))
    target_env._update_rl_far_gate(forecast)
    decision = CentralizedMPC(target_env.cfg, mode="proposed").decide_with_info(
        target_env.sim.state.copy(),
        forecast,
        target_env.previous,
    )
    raw_candidates = structured_residual_candidates(
        env,
        context.result.control,
        decision.control,
        np.zeros(env.action_dim, dtype=np.float32),
        magnitudes,
    )
    specs = pcent_guided_specs_from_candidates(env.action_schema.names, raw_candidates)
    metadata = {
        "pcent_objective": float(decision.objective),
        "pcent_converged": bool(decision.converged),
        "pcent_solver_evaluations": int(decision.solver_evaluations),
        "raw_candidate_count": int(len(raw_candidates)),
        "nonzero_candidate_count": int(len(specs)),
    }
    return specs, metadata


def _record_priority(record: dict[str, Any], horizon_steps: int) -> tuple:
    label = record.get("horizon_labels", {}).get(str(int(horizon_steps)), {})
    return (
        float(label.get("ttt_gain", 0.0)),
        bool(label.get("positive", False)),
        -float(record.get("native_response_distance_score", 0.0)),
        str(record.get("candidate_id", "")),
    )


def run_cached_pcent_guided_search(
    *,
    replay_cache: Path,
    output_dir: Path,
    magnitudes: tuple[float, ...],
    max_h12_candidates: int,
    horizon_steps: int,
    pstack_summary: dict[str, Any] | None = None,
    workers: int = 1,
    h1_chunks_per_worker: int = 1,
    h12_chunks_per_worker: int = 1,
    evaluate_all_h12: bool = False,
    native_rollout_cache_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if int(horizon_steps) < 1:
        raise ValueError("horizon_steps must be positive")
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    h1_path = output_dir / "h1_records.jsonl"
    h12_path = output_dir / "h12_records.jsonl"
    collisions = [path for path in (summary_path, h1_path, h12_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite cached P-CENT search outputs: "
            + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    env, context = load_cached_prefix(Path(replay_cache))
    control_step = int(env.step_idx - env.warmup)
    action_names = tuple(env.action_schema.names)
    specs, pcent_metadata = build_pcent_guided_specs(env, context, magnitudes)
    h1_native = _rollout_pstack(env, context, rollout_steps=1, horizons=(1,))
    scales, families = _response_scales_and_families(env)
    anchor_response = np.asarray(h1_native["first_step_response"], dtype=float)

    h1_records: list[dict[str, Any]] = []
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
            if verbose and (index == 1 or index == len(specs) or index % 10 == 0):
                print(json.dumps({
                    "phase": "pcent_h1",
                    "control_step": int(control_step),
                    "candidate": int(index),
                    "total": int(len(specs)),
                    "best_h1_gain": round(max(_h1_gain(row) for row in h1_records), 6),
                }, sort_keys=True), flush=True)
    if specs and not h1_records:
        raise RuntimeError("P-CENT H1 evaluation produced no records")

    if evaluate_all_h12:
        selected = list(h1_records)
        for row in selected:
            row["selection_reason"] = "all_pcent_guided"
    else:
        selected = select_h12_records(
            h1_records,
            max_h12_candidates=int(max_h12_candidates),
            seed=20260904 + int(control_step),
        )
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for row in h1_records:
        row["selected_for_h12"] = str(row["candidate_id"]) in selected_ids

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

    selected_specs = [
        SampledResidualSpec(
            candidate_id=str(row["candidate_id"]),
            label=str(row["label"]),
            generator=str(row["generator"]),
            family=str(row["family"]),
            residual=tuple(map(float, row["continuous_residual"])),
            metadata=dict(row.get("metadata", {})),
        )
        for row in selected
    ]
    selection_reasons = {
        str(row["candidate_id"]): str(row.get("selection_reason", ""))
        for row in selected
    }
    h12_records: list[dict[str, Any]] = []
    with h12_path.open("w", encoding="utf-8") as handle:
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
            label = record["horizon_labels"][str(int(horizon_steps))]
            if verbose:
                print(json.dumps({
                    "phase": f"pcent_h{int(horizon_steps)}",
                    "control_step": int(control_step),
                    "candidate": int(index),
                    "total": int(len(selected_specs)),
                    "gain": round(float(label["ttt_gain"]), 6),
                    "positive": bool(label["positive"]),
                    "candidate_id": record["candidate_id"],
                }, sort_keys=True), flush=True)
    if selected_specs and not h12_records:
        raise RuntimeError("P-CENT H12 evaluation produced no records")

    best_h12 = None
    if h12_records:
        best_h12 = max(h12_records, key=lambda row: _record_priority(row, horizon_steps))
    summary = {
        "format_version": SAMPLE_FORMAT,
        "search_format_version": PERCENT_SEARCH_FORMAT,
        "replay_cache": str(replay_cache),
        "scenario": str(env.scenario_name),
        "control_step": int(control_step),
        "magnitudes": list(map(float, magnitudes)),
        "max_h1_candidates": int(len(specs)),
        "max_h12_candidates": int(max_h12_candidates),
        "horizon_steps": int(horizon_steps),
        "workers": int(workers),
        "h1_chunks_per_worker": int(h1_chunks_per_worker),
        "h12_chunks_per_worker": int(h12_chunks_per_worker),
        "evaluate_all_h12": bool(evaluate_all_h12),
        "action_names": list(action_names),
        "pcent_metadata": pcent_metadata,
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
    _write_json(summary_path, summary)
    if verbose:
        best_gain = None
        if best_h12 is not None:
            best_gain = float(
                best_h12["horizon_labels"][str(int(horizon_steps))]["ttt_gain"]
            )
        print(json.dumps({
            "passed": True,
            "scenario": env.scenario_name,
            "control_step": int(control_step),
            "h1_records": int(len(h1_records)),
            "h12_records": int(len(h12_records)),
            "h12_positive": int(summary["h12_positive_count"]),
            "best_h12_gain": best_gain,
            "output": str(summary_path),
        }, indent=2, sort_keys=True), flush=True)
    return summary


def run_cached_pcent_guided_multi_step_search(
    *,
    output_dir: Path,
    cache_output_dir: Path,
    pstack_summary_path: Path,
    scenario: str,
    steps: tuple[int, ...],
    magnitudes: tuple[float, ...],
    max_h12_candidates: int,
    horizon_steps: int,
    workers: int,
    h1_chunks_per_worker: int,
    h12_chunks_per_worker: int,
    evaluate_all_h12: bool,
    native_rollout_cache_dir: Path | None,
    cache_mode: str,
    verbose: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    pstack_summary = json.loads(Path(pstack_summary_path).read_text(encoding="utf-8"))
    caches = ensure_replay_caches(
        manifest_path=None,
        cache_output_dir=cache_output_dir,
        scenario=scenario,
        steps=steps,
        t_total=float(pstack_summary.get("t_total_sec", 14400.0)),
        cache_mode=cache_mode,
    )
    summaries = []
    for step in steps:
        step_dir = output_dir / f"step{int(step):02d}"
        summary_path = step_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = run_cached_pcent_guided_search(
                replay_cache=Path(caches[int(step)]),
                output_dir=step_dir,
                magnitudes=magnitudes,
                max_h12_candidates=max_h12_candidates,
                horizon_steps=horizon_steps,
                pstack_summary=pstack_summary,
                workers=workers,
                h1_chunks_per_worker=h1_chunks_per_worker,
                h12_chunks_per_worker=h12_chunks_per_worker,
                evaluate_all_h12=evaluate_all_h12,
                native_rollout_cache_dir=native_rollout_cache_dir,
                verbose=verbose,
            )
        best = summary.get("best_h12")
        best_gain = None
        if isinstance(best, dict):
            best_gain = (
                best.get("horizon_labels", {})
                .get(str(int(horizon_steps)), {})
                .get("ttt_gain")
            )
        summaries.append({
            "step": int(step),
            "summary": str(summary_path),
            "h1_records": int(summary.get("h1_record_count", 0)),
            "h12_records": int(summary.get("h12_record_count", 0)),
            "h12_positive": int(summary.get("h12_positive_count", 0)),
            "best_h12_gain": None if best_gain is None else float(best_gain),
            "best_candidate_id": None if not isinstance(best, dict) else best.get("candidate_id"),
        })
    result = {
        "format_version": PERCENT_SEARCH_FORMAT,
        "scenario": scenario,
        "steps": list(map(int, steps)),
        "output_dir": str(output_dir),
        "cache_output_dir": str(cache_output_dir),
        "pstack_summary": str(pstack_summary_path),
        "magnitudes": list(map(float, magnitudes)),
        "max_h12_candidates": int(max_h12_candidates),
        "horizon_steps": int(horizon_steps),
        "workers": int(workers),
        "evaluate_all_h12": bool(evaluate_all_h12),
        "step_summaries": summaries,
    }
    _write_json(output_dir / "pcent_search_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-output-dir", type=Path)
    parser.add_argument("--pstack-summary", type=Path, required=True)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--steps", default="")
    parser.add_argument("--magnitudes", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--max-h12-candidates", type=int, default=32)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--h1-chunks-per-worker", type=int, default=1)
    parser.add_argument("--h12-chunks-per-worker", type=int, default=1)
    parser.add_argument("--evaluate-all-h12", action="store_true")
    parser.add_argument("--native-rollout-cache-dir", type=Path)
    parser.add_argument("--cache-mode", choices=("direct", "manifest"), default="direct")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pstack_summary = json.loads(args.pstack_summary.read_text(encoding="utf-8"))
    magnitudes = parse_magnitudes(args.magnitudes)
    steps = parse_int_csv(args.steps)
    if args.replay_cache is not None:
        if steps:
            raise ValueError("--steps cannot be combined with --replay-cache")
        run_cached_pcent_guided_search(
            replay_cache=args.replay_cache,
            output_dir=args.output_dir,
            magnitudes=magnitudes,
            max_h12_candidates=int(args.max_h12_candidates),
            horizon_steps=int(args.horizon_steps),
            pstack_summary=pstack_summary,
            workers=int(args.workers),
            h1_chunks_per_worker=int(args.h1_chunks_per_worker),
            h12_chunks_per_worker=int(args.h12_chunks_per_worker),
            evaluate_all_h12=bool(args.evaluate_all_h12),
            native_rollout_cache_dir=args.native_rollout_cache_dir,
            verbose=not bool(args.quiet),
        )
        return
    if not steps:
        raise ValueError("provide either --replay-cache or --steps")
    if args.cache_output_dir is None:
        raise ValueError("--cache-output-dir is required with --steps")
    run_cached_pcent_guided_multi_step_search(
        output_dir=args.output_dir,
        cache_output_dir=args.cache_output_dir,
        pstack_summary_path=args.pstack_summary,
        scenario=args.scenario,
        steps=steps,
        magnitudes=magnitudes,
        max_h12_candidates=int(args.max_h12_candidates),
        horizon_steps=int(args.horizon_steps),
        workers=int(args.workers),
        h1_chunks_per_worker=int(args.h1_chunks_per_worker),
        h12_chunks_per_worker=int(args.h12_chunks_per_worker),
        evaluate_all_h12=bool(args.evaluate_all_h12),
        native_rollout_cache_dir=args.native_rollout_cache_dir,
        cache_mode=args.cache_mode,
        verbose=not bool(args.quiet),
    )


if __name__ == "__main__":
    main()
