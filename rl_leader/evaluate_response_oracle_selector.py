"""Closed-loop oracle selector over executable follower responses."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.diagnose_pcent_reachability import _response_scales_and_families
from rl_leader.diagnose_reachable_candidate_attribution import _rollout_pstack
from rl_leader.sample_cached_tail_residuals import (
    SampledResidualSpec,
    _evaluate_records,
    build_sampled_residual_specs,
    evaluation_horizons,
    parse_magnitudes,
    select_h12_records,
)
from rl_leader.response_dqn_mask import (
    CandidateResponse,
    build_response_mask,
    response_feature_matrix,
)


EVAL_FORMAT = "response_level_oracle_selector_eval_v1"


def parse_int_csv(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    else:
        values = tuple(int(value) for value in raw)
    if len(set(values)) != len(values):
        raise ValueError("control-step values must be unique")
    return values


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


def _record_label(record: dict[str, Any], horizon_steps: int) -> dict[str, Any]:
    return record["horizon_labels"][str(int(horizon_steps))]


def _record_positive(record: dict[str, Any], horizon_steps: int) -> bool:
    label = _record_label(record, horizon_steps)
    return bool(label.get("positive", False) and label.get("validity_gate_pass", False))


def _record_gain(record: dict[str, Any], horizon_steps: int) -> float:
    return float(_record_label(record, horizon_steps).get("ttt_gain", float("-inf")))


def _candidate_response(record: dict[str, Any]) -> CandidateResponse:
    return CandidateResponse(
        action_id=int(record["action_id"]),
        response=tuple(map(float, record["first_step_response"])),
        follower_memory_fingerprint=str(record["post_follower_sha256"]),
        valid=bool(
            record.get("horizon_labels", {})
            .get("1", {})
            .get("validity_gate_pass", False)
        ),
        invalid_reason="",
    )


def _evaluate_response_level_step(
    env: RLLeaderEnv,
    *,
    control_step: int,
    seed: int,
    max_h1_candidates: int,
    max_horizon_candidates: int,
    horizon_steps: int,
    magnitudes: tuple[float, ...],
    workers: int,
) -> dict[str, Any]:
    context = env.prepare_pstack_anchor_context()
    action_names = tuple(env.action_schema.names)
    scales, families = _response_scales_and_families(env)
    horizons = evaluation_horizons(horizon_steps)
    native = _rollout_pstack(
        env,
        context,
        rollout_steps=int(horizon_steps),
        horizons=horizons,
    )
    anchor_response = np.asarray(native["first_step_response"], dtype=float)
    specs = build_sampled_residual_specs(
        action_names,
        seed=int(seed) + int(control_step),
        max_h1_candidates=int(max_h1_candidates),
        magnitudes=magnitudes,
    )
    h1_records = []
    for index, record in enumerate(_evaluate_records(
        env=env,
        context=context,
        specs=specs,
        native=native,
        rollout_steps=1,
        horizons=(1,),
        action_names=action_names,
        anchor_response=anchor_response,
        response_scales=scales,
        response_families=families,
        control_step=int(control_step),
        selected_for_h12=False,
        selection_reasons=None,
        workers=int(workers),
    ), start=1):
        h1_records.append(record)
        if index == 1 or index == len(specs) or index % 10 == 0:
            print(json.dumps({
                "phase": "response_h1",
                "control_step": int(control_step),
                "candidate": int(index),
                "total": int(len(specs)),
                "gain": round(float(
                    record["horizon_labels"]["1"]["ttt_gain"]
                ), 6),
            }, sort_keys=True), flush=True)
    selected = select_h12_records(
        h1_records,
        max_h12_candidates=int(max_horizon_candidates),
        seed=int(seed) + 1009 * int(control_step),
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
    h_records = []
    for index, record in enumerate(_evaluate_records(
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
        control_step=int(control_step),
        selected_for_h12=True,
        selection_reasons=selection_reasons,
        workers=int(workers),
    ), start=1):
        h_records.append(record)
        label = _record_label(record, horizon_steps)
        print(json.dumps({
            "phase": f"response_h{int(horizon_steps)}",
            "control_step": int(control_step),
            "candidate": int(index),
            "total": int(len(selected_specs)),
            "gain": round(float(label["ttt_gain"]), 6),
            "positive": bool(label["positive"]),
            "candidate_id": record["candidate_id"],
        }, sort_keys=True), flush=True)
    positive = [row for row in h_records if _record_positive(row, horizon_steps)]
    choice = (
        max(
            positive,
            key=lambda row: (
                _record_gain(row, horizon_steps),
                str(row.get("candidate_id", "")),
            ),
        )
        if positive else None
    )
    candidates_for_mask = [
        CandidateResponse(
            action_id=0,
            response=tuple(map(float, native["first_step_response"])),
            follower_memory_fingerprint=str(
                native["checkpoints"]["1"]["follower_memory_sha256"]
            ),
            valid=True,
            invalid_reason="",
        )
    ]
    for index, record in enumerate(h1_records, start=1):
        record["action_id"] = index
        candidates_for_mask.append(_candidate_response(record))
    response_mask = build_response_mask(
        candidates_for_mask,
        catalog_size=len(candidates_for_mask),
    )
    response_features = response_feature_matrix(
        candidates_for_mask,
        catalog_size=len(candidates_for_mask),
    )
    del response_features
    if choice is None:
        next_obs, reward, done, info, _ = env.step_prepared_optimizer_anchor(context)
        selected_residual = np.zeros(env.action_dim, dtype=np.float32)
        selected_candidate_id = "anchor"
        selected_gain = 0.0
    else:
        selected_residual = np.asarray(choice["continuous_residual"], dtype=np.float32)
        next_obs, reward, done, info = env.step_anchored_candidate(
            selected_residual,
            context,
        )
        selected_candidate_id = str(choice["candidate_id"])
        selected_gain = _record_gain(choice, horizon_steps)
    return {
        "control_step": int(control_step),
        "selected_candidate_id": selected_candidate_id,
        "selected_horizon_gain": float(selected_gain),
        "selected_nonzero_residual": {
            str(name): float(selected_residual[index])
            for index, name in enumerate(action_names)
            if abs(float(selected_residual[index])) > 1.0e-9
        },
        "committed_step_ttt": float(-reward),
        "done": bool(done),
        "terminal_inventory": float(info.get("inventory_after", env._inventory())),
        "raw_candidate_count": int(len(specs)),
        "h1_records": int(len(h1_records)),
        "horizon_records": int(len(h_records)),
        "horizon_positive": int(len(positive)),
        "unique_response_count": int(len(response_mask.groups)),
        "valid_response_representatives": int(response_mask.valid_action_mask.sum()),
        "best_horizon_gain": (
            None if not h_records else float(max(
                _record_gain(row, horizon_steps) for row in h_records
            ))
        ),
        "choice_mode": "response_horizon_positive" if choice is not None else "anchor",
        "next_observation_norm": float(np.linalg.norm(next_obs)),
    }


def run_response_oracle_selector(
    *,
    scenario: str,
    t_total: float,
    allowed_steps: tuple[int, ...],
    seed: int,
    max_h1_candidates: int,
    max_horizon_candidates: int,
    horizon_steps: int,
    magnitudes: tuple[float, ...],
    workers: int,
    output_dir: Path,
    pstack_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "trace.jsonl"
    collisions = [path for path in (summary_path, trace_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite response-oracle outputs: "
            + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    observation = np.asarray(env.reset(), dtype=np.float32)
    allowed = set(int(step) for step in allowed_steps)
    rows = []
    rewards = []
    with trace_path.open("w", encoding="utf-8") as handle:
        step = 0
        done = False
        while not done:
            if step in allowed:
                row = _evaluate_response_level_step(
                    env,
                    control_step=step,
                    seed=seed,
                    max_h1_candidates=max_h1_candidates,
                    max_horizon_candidates=max_horizon_candidates,
                    horizon_steps=horizon_steps,
                    magnitudes=magnitudes,
                    workers=workers,
                )
                reward = -float(row["committed_step_ttt"])
                done = bool(row["done"])
            else:
                context = env.prepare_pstack_anchor_context()
                observation, reward, done, info, _ = env.step_prepared_optimizer_anchor(
                    context,
                )
                row = {
                    "control_step": int(step),
                    "choice_mode": "anchor_outside_allowed_steps",
                    "selected_candidate_id": "anchor",
                    "selected_horizon_gain": 0.0,
                    "selected_nonzero_residual": {},
                    "committed_step_ttt": float(-reward),
                    "done": bool(done),
                    "terminal_inventory": float(info.get("inventory_after", env._inventory())),
                    "raw_candidate_count": 0,
                    "h1_records": 0,
                    "horizon_records": 0,
                    "horizon_positive": 0,
                    "unique_response_count": 1,
                    "valid_response_representatives": 1,
                    "best_horizon_gain": None,
                    "next_observation_norm": float(np.linalg.norm(observation)),
                }
            rewards.append(float(reward))
            rows.append(row)
            handle.write(json.dumps({
                "format_version": EVAL_FORMAT,
                **row,
            }, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({
                "control_step": int(step),
                "choice_mode": row["choice_mode"],
                "selected_candidate_id": row["selected_candidate_id"],
                "step_ttt": round(float(row["committed_step_ttt"]), 6),
                "horizon_positive": int(row["horizon_positive"]),
                "done": bool(done),
            }, sort_keys=True), flush=True)
            step += 1

    total_ttt = float(env.sim.total_ttt)
    summary = {
        "format_version": EVAL_FORMAT,
        "scenario": str(scenario),
        "t_total_sec": float(t_total),
        "control_steps": int(len(rows)),
        "allowed_steps": list(map(int, allowed_steps)),
        "seed": int(seed),
        "max_h1_candidates": int(max_h1_candidates),
        "max_horizon_candidates": int(max_horizon_candidates),
        "horizon_steps": int(horizon_steps),
        "magnitudes": list(map(float, magnitudes)),
        "workers": int(workers),
        "total_ttt": total_ttt,
        "control_ttt": float(np.sum(-np.asarray(rewards, dtype=np.float64))),
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(env._inventory()),
        "non_anchor_steps": int(sum(row["selected_candidate_id"] != "anchor" for row in rows)),
        "selected_steps": [
            {
                "control_step": int(row["control_step"]),
                "selected_candidate_id": row["selected_candidate_id"],
                "selected_horizon_gain": row["selected_horizon_gain"],
                "selected_nonzero_residual": row["selected_nonzero_residual"],
                "horizon_positive": row["horizon_positive"],
                "unique_response_count": row["unique_response_count"],
            }
            for row in rows
            if row["selected_candidate_id"] != "anchor"
        ],
        "trace_path": str(trace_path),
        "wall_seconds": float(time.perf_counter() - started),
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
        gap = total_ttt - pstack_total
        summary.update({
            "pstack_total_ttt": pstack_total,
            "vs_pstack_ttt_gap": float(gap),
            "vs_pstack_percent": float(100.0 * gap / max(abs(pstack_total), 1.0e-9)),
            "beats_pstack": bool(gap < 0.0),
            "target_5pct_total_ttt": float(0.95 * pstack_total),
            "meets_5pct_target": bool(total_ttt <= 0.95 * pstack_total),
        })
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--allowed-steps", default="21,24")
    parser.add_argument("--seed", type=int, default=2026090308)
    parser.add_argument("--max-h1-candidates", type=int, default=80)
    parser.add_argument("--max-horizon-candidates", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--magnitudes", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_response_oracle_selector(
        scenario=args.scenario,
        t_total=float(args.t_total),
        allowed_steps=parse_int_csv(args.allowed_steps),
        seed=int(args.seed),
        max_h1_candidates=int(args.max_h1_candidates),
        max_horizon_candidates=int(args.max_horizon_candidates),
        horizon_steps=int(args.horizon_steps),
        magnitudes=parse_magnitudes(args.magnitudes),
        workers=int(args.workers),
        output_dir=args.output_dir,
        pstack_summary=_read_json(args.pstack_summary),
    )


if __name__ == "__main__":
    main()
