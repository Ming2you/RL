"""Evaluate same-state one-step upper bounds from a cached P-Stack prefix."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rl_leader.diagnose_reachable_candidate_attribution import (
    _paired_metrics,
    _rollout_direct_pcent,
    _rollout_price_candidate,
    _rollout_pstack,
    _require_rollout_horizon_coverage,
)
from rl_leader.evaluate_cached_tail_residual_schedule import load_cached_prefix
from src.controllers.centralized_mpc import CentralizedMPC


UPPER_BOUND_FORMAT = "cached_same_state_step_upper_bound_v1"


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash_residual(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _candidate_gain(record: dict[str, Any], horizon_key: str) -> float:
    return float(
        record.get("horizon_labels", {})
        .get(horizon_key, {})
        .get("ttt_gain", float("-inf"))
    )


def _candidate_positive(record: dict[str, Any], horizon_key: str) -> bool:
    label = record.get("horizon_labels", {}).get(horizon_key, {})
    return bool(label.get("positive", False) and label.get("validity_gate_pass", False))


def load_sampler_candidates(
    paths: Iterable[Path],
    *,
    selection: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Load deduplicated residuals from cached sampler summaries."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if artifact.get("format_version") != "cached_tail_residual_sampler_v1":
            raise ValueError(f"sampler artifact has the wrong format: {path}")
        horizon_key = str(int(artifact.get("horizon_steps", 12)))
        records = list(artifact.get("h12_records", []))
        if selection == "best":
            if artifact.get("best_h12"):
                records = [artifact["best_h12"]]
            elif records:
                records = [max(records, key=lambda row: _candidate_gain(row, horizon_key))]
        elif selection == "positive":
            records = [row for row in records if _candidate_positive(row, horizon_key)]
        elif selection == "all":
            pass
        else:
            raise ValueError("selection must be best, positive, or all")
        for record in records:
            residual = np.asarray(record["continuous_residual"], dtype=np.float32)
            if residual.ndim != 1 or not np.all(np.isfinite(residual)):
                raise ValueError(f"candidate residual is invalid in {path}")
            rows.append({
                "source_artifact": str(path),
                "candidate_id": str(record.get("candidate_id", "")),
                "residual": residual.astype(float).tolist(),
                "source_horizon_key": horizon_key,
                "source_horizon_gain": _candidate_gain(record, horizon_key),
                "source_horizon_positive": _candidate_positive(record, horizon_key),
                "residual_sha256": str(
                    record.get("residual_sha256") or _hash_residual(residual)
                ),
                "residual_nonzero": dict(record.get("residual_nonzero", {})),
            })

    by_residual: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = str(row["residual_sha256"])
        current = by_residual.get(digest)
        if current is None or (
            bool(row["source_horizon_positive"]),
            float(row["source_horizon_gain"]),
            str(row["candidate_id"]),
        ) > (
            bool(current["source_horizon_positive"]),
            float(current["source_horizon_gain"]),
            str(current["candidate_id"]),
        ):
            by_residual[digest] = row
    ranked = sorted(
        by_residual.values(),
        key=lambda row: (
            bool(row["source_horizon_positive"]),
            float(row["source_horizon_gain"]),
            str(row["candidate_id"]),
        ),
        reverse=True,
    )
    if max_candidates > 0:
        ranked = ranked[:max_candidates]
    return ranked


def _rollout_record(
    *,
    label: str,
    rollout: dict[str, Any],
    native: dict[str, Any] | None,
    initial_total_ttt: float,
    pstack_total_ttt: float | None,
) -> dict[str, Any]:
    full_total = float(initial_total_ttt + rollout["ttt"])
    row = {
        "label": label,
        "tail_ttt": float(rollout["ttt"]),
        "full_total_ttt": full_total,
        "rollout_steps": int(rollout["steps"]),
        "terminal_inventory": float(rollout["terminal_inventory"]),
        "paired": None if native is None else _paired_metrics(rollout, native),
    }
    if pstack_total_ttt is not None:
        gap = full_total - float(pstack_total_ttt)
        row.update({
            "pstack_total_ttt": float(pstack_total_ttt),
            "vs_pstack_ttt_gap": float(gap),
            "vs_pstack_percent": float(
                100.0 * gap / max(abs(float(pstack_total_ttt)), 1.0e-9)
            ),
            "meets_5pct_target": bool(full_total <= 0.95 * float(pstack_total_ttt)),
        })
    return row


def _checkpoint(
    checkpoints: dict[str, dict[str, float]],
    completed: int,
    cumulative_ttt: float,
    env,
    requested: set[int],
    validity_gate_pass: bool,
) -> None:
    if completed in requested:
        checkpoints[str(completed)] = {
            "ttt": float(cumulative_ttt),
            "terminal_inventory": float(env._inventory()),
            "validity_gate_pass": bool(validity_gate_pass),
        }


def _continue_with_pstack_progress(
    env,
    *,
    first_step_ttt: float,
    first_step_response: np.ndarray,
    first_step_valid: bool,
    done: bool,
    rollout_steps: int,
    horizons: tuple[int, ...],
    label: str,
    progress_every: int,
) -> dict[str, Any]:
    cumulative_ttt = float(first_step_ttt)
    completed = 1
    validity = bool(first_step_valid)
    checkpoints: dict[str, dict[str, float]] = {}
    requested = set(horizons) | {int(rollout_steps)}
    _checkpoint(checkpoints, completed, cumulative_ttt, env, requested, validity)
    print(json.dumps({
        "phase": "branch_step",
        "label": label,
        "completed": completed,
        "rollout_steps": int(rollout_steps),
        "cumulative_ttt": round(cumulative_ttt, 6),
    }, sort_keys=True), flush=True)
    while not done and completed < int(rollout_steps):
        context = env.prepare_pstack_anchor_context()
        _, reward, done, info, _ = env.step_prepared_optimizer_anchor(
            context,
            sync_follower_state=True,
        )
        cumulative_ttt += -float(reward)
        completed += 1
        validity = validity and bool(info.get("validity_gate_pass", False))
        _checkpoint(checkpoints, completed, cumulative_ttt, env, requested, validity)
        if progress_every > 0 and (
            completed % int(progress_every) == 0 or done or completed == int(rollout_steps)
        ):
            print(json.dumps({
                "phase": "branch_step",
                "label": label,
                "completed": completed,
                "rollout_steps": int(rollout_steps),
                "cumulative_ttt": round(cumulative_ttt, 6),
                "done": bool(done),
            }, sort_keys=True), flush=True)
    return {
        "steps": int(completed),
        "ttt": float(cumulative_ttt),
        "terminal_inventory": float(env._inventory()),
        "final_simulation_time_sec": float(env.sim.state.time_sec),
        "first_step_ttt": float(first_step_ttt),
        "first_step_response": np.asarray(first_step_response, dtype=float).tolist(),
        "first_step_validity_gate_pass": bool(first_step_valid),
        "validity_gate_pass": bool(validity),
        "checkpoints": checkpoints,
    }


def _rollout_direct_pcent_progress(
    source_env,
    *,
    rollout_steps: int,
    horizons: tuple[int, ...],
    progress_every: int,
) -> dict[str, Any]:
    env = copy.deepcopy(source_env)
    forecast = env._forecast()
    env._update_rl_far_gate(forecast)
    decision = CentralizedMPC(env.cfg, mode="proposed").decide_with_info(
        env.sim.state.copy(),
        forecast,
        env.previous,
    )
    _, step_ttt, done = env.step_with_control(decision.control)
    result = _continue_with_pstack_progress(
        env,
        first_step_ttt=float(step_ttt),
        first_step_response=env.response_vector(env.previous),
        first_step_valid=True,
        done=done,
        rollout_steps=rollout_steps,
        horizons=horizons,
        label="direct_pcent",
        progress_every=progress_every,
    )
    result["solver_evaluations"] = int(decision.solver_evaluations)
    return result


def _rollout_price_candidate_progress(
    source_env,
    residual: np.ndarray,
    anchor_context,
    *,
    rollout_steps: int,
    horizons: tuple[int, ...],
    label: str,
    progress_every: int,
) -> dict[str, Any]:
    env, anchor_context = copy.deepcopy((source_env, anchor_context))
    _, reward, done, info = env.step_anchored_candidate(residual, anchor_context)
    return _continue_with_pstack_progress(
        env,
        first_step_ttt=-float(reward),
        first_step_response=env.response_vector(env.previous),
        first_step_valid=bool(info.get("validity_gate_pass", False)),
        done=done,
        rollout_steps=rollout_steps,
        horizons=horizons,
        label=label,
        progress_every=progress_every,
    )


def run_cached_step_upper_bound(
    *,
    replay_cache: Path,
    output: Path,
    pstack_summary: dict[str, Any] | None = None,
    sampler_artifacts: Iterable[Path] = (),
    sampler_selection: str = "best",
    max_sampler_candidates: int = 0,
    max_rollout_steps: int = 0,
    include_direct_pcent: bool = True,
    skip_native_tail: bool = False,
    progress_every: int = 1,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    started = time.perf_counter()
    env, context = load_cached_prefix(Path(replay_cache))
    initial_total_ttt = float(env.sim.total_ttt)
    control_step = int(env.step_idx - env.warmup)
    remaining = int(env.n_steps - env.step_idx)
    rollout_steps = remaining if max_rollout_steps <= 0 else min(remaining, int(max_rollout_steps))
    if rollout_steps <= 0:
        raise ValueError("cached prefix has no remaining rollout steps")
    horizons = tuple(sorted({h for h in (1, 3, 6, 12, rollout_steps) if h <= rollout_steps}))
    pstack_total = None if pstack_summary is None else float(pstack_summary["total_ttt"])

    result: dict[str, Any] = {
        "format_version": UPPER_BOUND_FORMAT,
        "replay_cache": str(replay_cache),
        "scenario": str(env.scenario_name),
        "control_step": control_step,
        "initial_total_ttt": initial_total_ttt,
        "rollout_steps": int(rollout_steps),
        "horizons": list(map(int, horizons)),
        "pstack_total_ttt": pstack_total,
        "target_5pct_total_ttt": None if pstack_total is None else 0.95 * pstack_total,
        "native": None,
        "direct_pcent": None,
        "sampler_candidates": [],
        "best": None,
        "wall_seconds": None,
    }
    _write_json(output, result)

    native = None
    if not skip_native_tail:
        native = _rollout_pstack(
            env,
            context,
            rollout_steps=rollout_steps,
            horizons=horizons,
        )
        _require_rollout_horizon_coverage(
            native, horizons, label="cached upper-bound P-Stack"
        )
        result["native"] = _rollout_record(
            label="native_pstack_tail",
            rollout=native,
            native=native,
            initial_total_ttt=initial_total_ttt,
            pstack_total_ttt=pstack_total,
        )
        _write_json(output, result)
        print(json.dumps({
            "phase": "native",
            "control_step": control_step,
            "full_total_ttt": result["native"]["full_total_ttt"],
        }, sort_keys=True), flush=True)

    if include_direct_pcent:
        direct = (
            _rollout_direct_pcent_progress(
                env,
                rollout_steps=rollout_steps,
                horizons=horizons,
                progress_every=progress_every,
            )
            if skip_native_tail else
            _rollout_direct_pcent(
                env,
                rollout_steps=rollout_steps,
                horizons=horizons,
            )
        )
        _require_rollout_horizon_coverage(
            direct, horizons, label="cached upper-bound direct P-CENT"
        )
        result["direct_pcent"] = _rollout_record(
            label="direct_pcent_one_step_tail",
            rollout=direct,
            native=native,
            initial_total_ttt=initial_total_ttt,
            pstack_total_ttt=pstack_total,
        )
        _write_json(output, result)
        print(json.dumps({
            "phase": "direct_pcent",
            "control_step": control_step,
            "tail_gain": (
                None if result["direct_pcent"]["paired"] is None
                else result["direct_pcent"]["paired"]["ttt_gain"]
            ),
            "vs_pstack_percent": result["direct_pcent"].get("vs_pstack_percent"),
        }, sort_keys=True), flush=True)

    candidates = load_sampler_candidates(
        sampler_artifacts,
        selection=sampler_selection,
        max_candidates=max_sampler_candidates,
    )
    for index, candidate in enumerate(candidates, start=1):
        rollout = (
            _rollout_price_candidate_progress(
                env,
                np.asarray(candidate["residual"], dtype=np.float32),
                context,
                rollout_steps=rollout_steps,
                horizons=horizons,
                label=str(candidate["candidate_id"]),
                progress_every=progress_every,
            )
            if skip_native_tail else
            _rollout_price_candidate(
                env,
                np.asarray(candidate["residual"], dtype=np.float32),
                context,
                rollout_steps=rollout_steps,
                horizons=horizons,
            )
        )
        _require_rollout_horizon_coverage(
            rollout, horizons, label=f"cached upper-bound {candidate['candidate_id']}"
        )
        row = {
            **candidate,
            **_rollout_record(
                label=str(candidate["candidate_id"]),
                rollout=rollout,
                native=native,
                initial_total_ttt=initial_total_ttt,
                pstack_total_ttt=pstack_total,
            ),
        }
        result["sampler_candidates"].append(row)
        _write_json(output, result)
        print(json.dumps({
            "phase": "sampler_candidate",
            "index": index,
            "total": len(candidates),
            "candidate_id": candidate["candidate_id"],
            "tail_gain": None if row["paired"] is None else row["paired"]["ttt_gain"],
            "vs_pstack_percent": row.get("vs_pstack_percent"),
        }, sort_keys=True), flush=True)

    rollouts = [
        row for row in (
            result.get("direct_pcent"),
            *result.get("sampler_candidates", []),
        )
        if isinstance(row, dict)
    ]
    result["best"] = (
        None if not rollouts else min(
            rollouts,
            key=lambda row: float(row["full_total_ttt"]),
        )
    )
    result["wall_seconds"] = float(time.perf_counter() - started)
    _write_json(output, result)
    print(json.dumps({
        "phase": "complete",
        "control_step": control_step,
        "best_label": None if result["best"] is None else result["best"]["label"],
        "best_total_ttt": None if result["best"] is None else result["best"]["full_total_ttt"],
        "best_vs_pstack_percent": (
            None if result["best"] is None else result["best"].get("vs_pstack_percent")
        ),
        "output": str(output),
    }, indent=2, sort_keys=True), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--sample-artifact", action="append", default=[], type=Path)
    parser.add_argument(
        "--sample-selection",
        choices=("best", "positive", "all"),
        default="best",
    )
    parser.add_argument("--max-sample-candidates", type=int, default=0)
    parser.add_argument("--max-rollout-steps", type=int, default=0)
    parser.add_argument("--skip-direct-pcent", action="store_true")
    parser.add_argument("--skip-native-tail", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_cached_step_upper_bound(
        replay_cache=args.replay_cache,
        output=args.output,
        pstack_summary=_read_json(args.pstack_summary),
        sampler_artifacts=args.sample_artifact,
        sampler_selection=args.sample_selection,
        max_sampler_candidates=int(args.max_sample_candidates),
        max_rollout_steps=int(args.max_rollout_steps),
        include_direct_pcent=not bool(args.skip_direct_pcent),
        skip_native_tail=bool(args.skip_native_tail),
        progress_every=int(args.progress_every),
    )


if __name__ == "__main__":
    main()
