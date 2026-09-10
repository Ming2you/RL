"""Measure whether structured price candidates can reach P-CENT controls."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rl_leader.diagnose_anchor_context_parity import _dynamic_follower_payload
from rl_leader.diagnose_frozen_event_attribution import (
    _load_jsonl,
    _observation_from_trace,
    action_from_trace_row,
)
from rl_leader.diagnose_phase0_parity import _digest, _physical_snapshot
from rl_leader.env import RLLeaderEnv
from src.controllers.centralized_mpc import CentralizedMPC
from src.models.state import segment_vsl


def _shortest_offset_delta(target: float, reference: float, cycle: float) -> float:
    return float((target - reference + 0.5 * cycle) % cycle - 0.5 * cycle)


def _response_scales_and_families(env: RLLeaderEnv):
    scales = []
    families = {name: [] for name in ("green", "offset", "meter", "vsl")}
    index = 0
    for _ in env.action_schema.signals:
        scales.extend((6.0, max(float(env.net.cycle_length) / 8.0, 1.0)))
        families["green"].append(index)
        families["offset"].append(index + 1)
        index += 2
    for ramp in env.action_schema.ramps:
        scales.extend((
            0.2 * float(env.net.ramp_capacity_veh_h[ramp]),
            max(float(env.cfg.freeway_follower.max_vsl_step), 1.0),
        ))
        families["meter"].append(index)
        families["vsl"].append(index + 1)
        index += 2
    for _ in env.action_schema.nonmerge_vsl_keys:
        scales.append(max(float(env.cfg.freeway_follower.max_vsl_step), 1.0))
        families["vsl"].append(index)
        index += 1
    return np.asarray(scales, dtype=float), families


def response_distance(
    response: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    families: dict[str, list[int]],
) -> dict[str, float]:
    normalized = (np.asarray(response, dtype=float) - target) / scales
    result = {
        "overall_rmse": float(np.sqrt(np.mean(np.square(normalized)))),
        "overall_linf": float(np.max(np.abs(normalized))),
    }
    for family, indices in families.items():
        values = normalized[np.asarray(indices, dtype=int)]
        result[f"{family}_rmse"] = float(np.sqrt(np.mean(np.square(values))))
    return result


def _directed_groups(env: RLLeaderEnv, anchor_control, target_control):
    dimension = env.action_dim
    action_index = {
        name: index for index, name in enumerate(env.action_schema.names)
    }
    groups: dict[str, np.ndarray] = {}

    def group(name: str) -> np.ndarray:
        return groups.setdefault(name, np.zeros(dimension, dtype=np.float32))

    for signal in env.action_schema.signals:
        green_delta = float(
            target_control.green_times.get(f"{signal}_p1", 0.0)
            - anchor_control.green_times.get(f"{signal}_p1", 0.0)
        )
        offset_delta = _shortest_offset_delta(
            float(target_control.offsets.get(signal, 0.0)),
            float(anchor_control.offsets.get(signal, 0.0)),
            float(env.net.cycle_length),
        )
        if abs(green_delta) > 1.0e-9:
            value = -float(np.sign(green_delta))
            index = action_index[f"urban.{signal}.g_green"]
            group("green")[index] = value
            group(f"signal:{signal}")[index] = value
        if abs(offset_delta) > 1.0e-9:
            value = -float(np.sign(offset_delta))
            index = action_index[f"urban.{signal}.g_offset"]
            group("offset")[index] = value
            group(f"signal:{signal}")[index] = value

    for ramp in env.action_schema.ramps:
        link = env.net.ramp_to_freeway[ramp]
        segment = int(env.net.ramp_merge_segment_index.get(ramp, 0))
        meter_delta = float(
            target_control.ramp_metering.get(ramp, 0.0)
            - anchor_control.ramp_metering.get(ramp, 0.0)
        )
        vsl_delta = float(
            segment_vsl(target_control, link, segment, env.cfg)
            - segment_vsl(anchor_control, link, segment, env.cfg)
        )
        if abs(meter_delta) > 1.0e-9:
            value = -float(np.sign(meter_delta))
            index = action_index[f"freeway.{ramp}.g_meter"]
            group("meter")[index] = value
            group(f"ramp:{ramp}")[index] = value
        if abs(vsl_delta) > 1.0e-9:
            value = -float(np.sign(vsl_delta))
            index = action_index[f"freeway.{ramp}.g_vsl"]
            group("vsl")[index] = value
            group(f"ramp:{ramp}")[index] = value
            group(f"vsl_link:{link}")[index] = value

    for key in env.action_schema.nonmerge_vsl_keys:
        link, segment_text = key.rsplit("__seg", 1)
        vsl_delta = float(
            segment_vsl(target_control, link, int(segment_text), env.cfg)
            - segment_vsl(anchor_control, link, int(segment_text), env.cfg)
        )
        if abs(vsl_delta) > 1.0e-9:
            value = -float(np.sign(vsl_delta))
            index = action_index[f"vsl.{key}.g_vsl"]
            group("vsl")[index] = value
            group(f"vsl_link:{link}")[index] = value

    nonempty = {
        name: values for name, values in groups.items() if np.any(values != 0.0)
    }
    if nonempty:
        combined = np.zeros(dimension, dtype=np.float32)
        for family in ("green", "offset", "meter", "vsl"):
            if family in nonempty:
                combined += nonempty[family]
        nonempty["all"] = np.clip(combined, -1.0, 1.0)
    return nonempty


def structured_residual_candidates(
    env: RLLeaderEnv,
    anchor_control,
    target_control,
    actor_residual: np.ndarray,
    magnitudes: tuple[float, ...],
) -> list[tuple[str, np.ndarray]]:
    candidates = [
        ("anchor", np.zeros(env.action_dim, dtype=np.float32)),
        ("frozen_actor", np.asarray(actor_residual, dtype=np.float32).copy()),
    ]
    for name, direction in sorted(
        _directed_groups(env, anchor_control, target_control).items()
    ):
        for magnitude in magnitudes:
            candidates.append((
                f"toward:{name}:{magnitude:g}",
                np.asarray(direction * magnitude, dtype=np.float32),
            ))
            candidates.append((
                f"away:{name}:{magnitude:g}",
                np.asarray(-direction * magnitude, dtype=np.float32),
            ))
    return candidates


def _replay_event_snapshots(trace_path: Path, parity_tolerance: float):
    rows = _load_jsonl(trace_path)
    meta = json.loads(trace_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    expected = [
        int(row["step"])
        for row in rows
        if float(row["info"].get("leader_rl_pstack_anchor_pick_rl", 0.0)) > 0.5
    ]
    if not expected:
        raise ValueError(f"trace has no accepted events: {trace_path}")
    env = RLLeaderEnv(
        scenario_name=str(meta["scenario"]),
        mask=str(meta["mask"]),
        pstack_anchor=True,
    )
    env.action_parameterization = str(meta["action_parameterization"])
    observation = env.reset()
    if meta.get("experiment_contract_sha256") != env.experiment_contract_fingerprint:
        raise ValueError("trace experiment contract does not match replay environment")
    snapshots = []
    expected_set = set(expected)
    for row in rows:
        step = int(row["step"])
        if step > max(expected):
            break
        saved_observation = _observation_from_trace(row, env.observation_schema.names)
        error = float(np.max(np.abs(observation - saved_observation)))
        if error > parity_tolerance:
            raise RuntimeError(f"observation replay error at step {step}: {error}")
        action = action_from_trace_row(row, env.action_schema.names)
        before = copy.deepcopy(env) if step in expected_set else None
        observation, reward, _, info = env.step(action)
        replay_pick = float(
            info.get("leader_rl_pstack_anchor_pick_rl", 0.0)
        ) > 0.5
        saved_pick = step in expected_set
        reward_error = abs(float(-reward) - float(row["info"]["step_ttt"]))
        if replay_pick != saved_pick or reward_error > parity_tolerance:
            raise RuntimeError(
                f"transition replay mismatch at step {step}: "
                f"pick={replay_pick}/{saved_pick}, reward_error={reward_error}"
            )
        if replay_pick:
            snapshots.append({
                "step": step,
                "simulation_time_sec": float(row["state_before"]["time_sec"]),
                "env": before,
                "actor_residual": action,
            })
    if [item["step"] for item in snapshots] != expected:
        raise RuntimeError("accepted-event snapshot replay is incomplete")
    return meta, snapshots


def _effective_rank(
    responses: list[np.ndarray], anchor: np.ndarray, scales: np.ndarray
) -> dict:
    if not responses:
        return {"rank": 0, "singular_values": [], "participation_ratio": 0.0}
    matrix = np.vstack([
        (np.asarray(response, dtype=float) - anchor) / scales
        for response in responses
    ])
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    threshold = max(float(singular[0]) * 1.0e-6, 1.0e-9) if singular.size else 1.0e-9
    squared = np.square(singular)
    participation = (
        float(np.square(np.sum(squared)) / np.sum(np.square(squared)))
        if np.sum(np.square(squared)) > 0.0 else 0.0
    )
    return {
        "rank": int(np.count_nonzero(singular > threshold)),
        "singular_values": singular.astype(float).tolist(),
        "participation_ratio": participation,
    }


def evaluate_event(snapshot: dict, magnitudes: tuple[float, ...]) -> dict:
    env = snapshot["env"]
    anchor_context = env.prepare_pstack_anchor_context()
    anchor_control = anchor_context.result.control

    target_env = copy.deepcopy(env)
    target_forecast = target_env._forecast()
    target_env._update_rl_far_gate(target_forecast)
    target_decision = CentralizedMPC(
        target_env.cfg, mode="proposed"
    ).decide_with_info(
        target_env.sim.state.copy(), target_forecast, target_env.previous
    )
    target_control = target_decision.control

    target_response = env.response_vector(target_control)
    anchor_response = env.response_vector(anchor_control)
    scales, families = _response_scales_and_families(env)
    candidates = structured_residual_candidates(
        env,
        anchor_control,
        target_control,
        snapshot["actor_residual"],
        magnitudes,
    )
    unique: dict[tuple[float, ...], dict] = {}
    for label, residual in candidates:
        trial = copy.deepcopy(env)
        _, _, _, info = trial.step_anchored_candidate(residual, anchor_context)
        response = trial.response_vector(trial.previous)
        post_follower_sha256 = _digest(
            _dynamic_follower_payload(trial.controller.nash_solver)
        )
        post_physical_sha256 = _digest(_physical_snapshot(trial))
        signature = (
            tuple(np.round(response.astype(float), 6)),
            post_follower_sha256,
            post_physical_sha256,
        )
        distance = response_distance(response, target_response, scales, families)
        row = {
            "representative_label": label,
            "aliases": [label],
            "residual_nonzero": {
                name: float(value)
                for name, value in zip(env.action_schema.names, residual)
                if abs(float(value)) > 1.0e-12
            },
            "response": response.astype(float).tolist(),
            "distance_to_pcent": distance,
            "step_ttt": float(info["step_ttt"]),
            "validity_gate_pass": bool(info["validity_gate_pass"]),
            "post_follower_sha256": post_follower_sha256,
            "post_physical_sha256": post_physical_sha256,
        }
        existing = unique.get(signature)
        if existing is None:
            unique[signature] = row
        else:
            existing["aliases"].append(label)

    unique_rows = sorted(
        unique.values(),
        key=lambda row: row["distance_to_pcent"]["overall_rmse"],
    )
    pstack_metrics = env._fixed_control_metrics(
        anchor_control, anchor_context.forecast
    )
    pcent_metrics = env._fixed_control_metrics(target_control, target_forecast)
    return {
        "step": int(snapshot["step"]),
        "simulation_time_sec": float(snapshot["simulation_time_sec"]),
        "anchor_context_contract": anchor_context.contract_version,
        "anchor_fingerprint": anchor_context.anchor_fingerprint,
        "anchor_selected_branch": anchor_context.coordination.selected_branch,
        "candidate_count": len(candidates),
        "unique_response_count": len(unique_rows),
        "response_dedupe_fraction": float(
            1.0 - len(unique_rows) / max(len(candidates), 1)
        ),
        "pstack_distance_to_pcent": response_distance(
            anchor_response, target_response, scales, families
        ),
        "pstack_h3_metrics": pstack_metrics,
        "pcent_h3_metrics": pcent_metrics,
        "pcent_h3_ttt_gain": float(pstack_metrics["ttt"] - pcent_metrics["ttt"]),
        "pcent_solver_evaluations": int(target_decision.solver_evaluations),
        "pcent_response": target_response.astype(float).tolist(),
        "pstack_response": anchor_response.astype(float).tolist(),
        "response_effective_rank": _effective_rank(
            [np.asarray(row["response"], dtype=float) for row in unique_rows],
            anchor_response,
            scales,
        ),
        "best_candidates": unique_rows[:10],
        "all_unique_candidates": unique_rows,
    }


def run_trace(
    trace_path: Path,
    *,
    magnitudes: tuple[float, ...],
    parity_tolerance: float,
    output_path: Path,
) -> dict:
    meta, snapshots = _replay_event_snapshots(trace_path, parity_tolerance)
    output = {
        "format_version": "pcent_price_reachability_v2_native_anchor",
        "source_trace": str(trace_path),
        "scenario": str(meta["scenario"]),
        "mask": str(meta["mask"]),
        "experiment_contract_sha256": str(meta["experiment_contract_sha256"]),
        "magnitudes": list(magnitudes),
        "events": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for snapshot in snapshots:
        event = evaluate_event(snapshot, magnitudes)
        output["events"].append(event)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        best = event["best_candidates"][0]
        print(
            f"reachability scenario={output['scenario']} step={event['step']} "
            f"unique={event['unique_response_count']}/{event['candidate_count']} "
            f"rank={event['response_effective_rank']['rank']} "
            f"best_rmse={best['distance_to_pcent']['overall_rmse']:.4f}",
            flush=True,
        )
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--magnitudes", default="0.25,0.5,1.0")
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--output-dir",
        default="results/rl_phase0_implementation_20260827/pcent_reachability",
    )
    args = parser.parse_args(argv)
    magnitudes = tuple(sorted({
        float(value) for value in args.magnitudes.split(",") if value.strip()
    }))
    output_dir = Path(args.output_dir)
    summary = []
    for value in args.traces:
        trace_path = Path(value)
        output_path = output_dir / f"{trace_path.stem}.json"
        result = run_trace(
            trace_path,
            magnitudes=magnitudes,
            parity_tolerance=args.parity_tolerance,
            output_path=output_path,
        )
        summary.append({
            "scenario": result["scenario"],
            "event_count": len(result["events"]),
            "output": str(output_path),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
