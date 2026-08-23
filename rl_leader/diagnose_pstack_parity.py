"""Replay native P-Stack coordination through the RL adapter on identical states."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from rl_leader.env import RLLeaderEnv
from src.controllers.coordination import CoordinationMask
from src.models.state import ControlAction, segment_vsl


ROOT = Path(__file__).resolve().parents[1]
PRICE_FIELDS = (
    "signal_marginal_price",
    "offset_marginal_price",
    "metering_marginal_price",
    "vsl_marginal_price",
    "signal_marginal_price_ref",
    "offset_marginal_price_ref",
    "metering_marginal_price_ref",
    "vsl_marginal_price_ref",
    "metering_release_certified",
)


def _mapping(value) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _price_snapshot(follower) -> dict[str, dict[str, float]]:
    return {name: _mapping(getattr(follower, name, None)) for name in PRICE_FIELDS}


def _follower_state_snapshot(follower) -> dict:
    scalar_fields = (
        "_lambda_P",
        "_lambda_UF",
        "_np_last_sum_nin",
        "_np_bias_ratio",
        "_np_prev_accum",
        "_np_last_real_q",
    )
    result = {
        name: (None if getattr(follower, name, None) is None else float(getattr(follower, name)))
        for name in scalar_fields
    }
    result["_prev_coupling"] = _mapping(getattr(follower, "_prev_coupling", None))
    pending = getattr(follower, "_np_corrector_pending", None)
    result["_np_corrector_pending"] = (
        [float(value) for value in pending] if pending is not None else None
    )
    return result


def _all_vsl_keys(env: RLLeaderEnv) -> tuple[str, ...]:
    return tuple(
        f"{link}__seg{index}"
        for link in env.net.freeway_links
        for index in range(len(env.sim.state.freeway_density.get(link, [])))
    )


def _control_vectors(env: RLLeaderEnv, control: ControlAction) -> dict[str, np.ndarray]:
    return {
        "budget": np.asarray([control.N_P_star, control.N_UF_star], dtype=float),
        "green": np.asarray([
            control.green_times.get(f"{signal}_p1", 0.0)
            for signal in env.action_schema.signals
        ], dtype=float),
        "offset": np.asarray([
            control.offsets.get(signal, 0.0)
            for signal in env.action_schema.signals
        ], dtype=float),
        "metering": np.asarray([
            control.ramp_metering.get(ramp, 0.0)
            for ramp in env.action_schema.ramps
        ], dtype=float),
        "vsl_all_segments": np.asarray([
            segment_vsl(control, link, index, env.cfg)
            for link in env.net.freeway_links
            for index in range(len(env.sim.state.freeway_density.get(link, [])))
        ], dtype=float),
    }


def _control_snapshot(env: RLLeaderEnv, control: ControlAction) -> dict[str, list[float]]:
    return {
        name: values.astype(float).tolist()
        for name, values in _control_vectors(env, control).items()
    }


def _vector_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    absolute = np.abs(delta)
    return {
        "count": int(delta.size),
        "changed_count": int(np.count_nonzero(absolute > 1.0e-9)),
        "mean_abs": float(absolute.mean()) if absolute.size else 0.0,
        "max_abs": float(absolute.max()) if absolute.size else 0.0,
        "l2": float(np.linalg.norm(delta)),
    }


def compare_controls(
    env: RLLeaderEnv,
    native: ControlAction,
    replay: ControlAction,
) -> dict[str, dict[str, float | int]]:
    native_vectors = _control_vectors(env, native)
    replay_vectors = _control_vectors(env, replay)
    return {
        name: _vector_difference(native_vectors[name], replay_vectors[name])
        for name in native_vectors
    }


def _price_difference(
    native: Mapping[str, float],
    replay: Mapping[str, float],
    keys: set[str] | None = None,
) -> dict[str, float | int]:
    compared_keys = sorted(keys if keys is not None else set(native) | set(replay))
    delta = np.asarray([
        float(replay.get(key, 0.0)) - float(native.get(key, 0.0))
        for key in compared_keys
    ], dtype=float)
    absolute = np.abs(delta)
    native_values = np.asarray([float(native.get(key, 0.0)) for key in compared_keys], dtype=float)
    return {
        "key_count": len(compared_keys),
        "native_nonzero_count": int(np.count_nonzero(np.abs(native_values) > 1.0e-9)),
        "changed_count": int(np.count_nonzero(absolute > 1.0e-9)),
        "mean_abs": float(absolute.mean()) if absolute.size else 0.0,
        "max_abs": float(absolute.max()) if absolute.size else 0.0,
        "l2": float(np.linalg.norm(delta)),
    }


def compare_prices(
    env: RLLeaderEnv,
    native: dict[str, dict[str, float]],
    replay: dict[str, dict[str, float]],
) -> dict:
    merge_vsl_keys = set(env.action_schema.freeway_vsl_keys.values())
    all_vsl_keys = set(_all_vsl_keys(env))
    nonmerge_vsl_keys = all_vsl_keys - merge_vsl_keys
    fields = {
        name: _price_difference(native[name], replay[name])
        for name in PRICE_FIELDS
    }
    fields["vsl_marginal_price_merge"] = _price_difference(
        native["vsl_marginal_price"], replay["vsl_marginal_price"], merge_vsl_keys,
    )
    fields["vsl_marginal_price_nonmerge"] = _price_difference(
        native["vsl_marginal_price"], replay["vsl_marginal_price"], nonmerge_vsl_keys,
    )
    fields["vsl_marginal_price_all_segments"] = _price_difference(
        native["vsl_marginal_price"], replay["vsl_marginal_price"], all_vsl_keys,
    )
    return fields


def _one_step_ttt(env: RLLeaderEnv, control: ControlAction, forecast) -> dict[str, float]:
    simulator = copy.deepcopy(env.sim)
    log = simulator.step(control.copy(), forecast[0], env.step_idx)
    return {
        "total": float(log.urban_ttt + log.freeway_ttt),
        "urban": float(log.urban_ttt),
        "freeway": float(log.freeway_ttt),
    }


def _summarize(rows: list[dict], scenario: str, elapsed_sec: float) -> dict:
    def values(path: tuple[str, ...]) -> np.ndarray:
        result = []
        for row in rows:
            current = row
            for key in path:
                current = current[key]
            result.append(float(current))
        return np.asarray(result, dtype=float)

    summary = {
        "scenario": scenario,
        "steps": len(rows),
        "elapsed_sec": float(elapsed_sec),
        "time_sec": [float(row["time_sec"]) for row in rows],
    }
    if not rows:
        return summary
    summary["native_ttt_veh_h"] = float(values(("one_step_ttt", "native", "total")).sum())
    summary["replay_ttt_veh_h"] = float(values(("one_step_ttt", "replay", "total")).sum())
    summary["replay_minus_native_ttt_veh_h"] = float(
        summary["replay_ttt_veh_h"] - summary["native_ttt_veh_h"]
    )
    summary["native_leader_objective"] = float(values(("leader_objective", "native")).sum())
    summary["replay_leader_objective"] = float(values(("leader_objective", "replay")).sum())
    summary["control_max_abs"] = {
        name: float(values(("control_difference", name, "max_abs")).max())
        for name in rows[0]["control_difference"]
    }
    summary["control_changed_step_count"] = {
        name: int(np.count_nonzero(values(("control_difference", name, "changed_count")) > 0.0))
        for name in rows[0]["control_difference"]
    }
    summary["price_changed_step_count"] = {
        name: int(np.count_nonzero(values(("price_difference", name, "changed_count")) > 0.0))
        for name in (
            "signal_marginal_price",
            "offset_marginal_price",
            "metering_marginal_price",
            "vsl_marginal_price_merge",
            "vsl_marginal_price_nonmerge",
        )
    }
    summary["nonmerge_vsl_native_nonzero_step_count"] = int(np.count_nonzero(
        values(("price_difference", "vsl_marginal_price_nonmerge", "native_nonzero_count")) > 0.0
    ))
    summary["encoded_saturated_step_count"] = int(np.count_nonzero(
        values(("encoded_saturated_count",)) > 0.0
    ))
    summary["encoded_budget_saturated_step_count"] = int(np.count_nonzero(
        values(("encoded_budget_saturated_count",)) > 0.0
    ))
    summary["encoded_price_saturated_step_count"] = int(np.count_nonzero(
        values(("encoded_price_saturated_count",)) > 0.0
    ))
    return summary


def diagnose_scenario(
    scenario: str,
    output_dir: Path,
    max_policy_steps: int,
    max_sec: float,
) -> dict:
    env = RLLeaderEnv(scenario_name=scenario)
    rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{scenario}.jsonl"
    started = time.monotonic()
    with trace_path.open("w", encoding="utf-8") as trace:
        while len(rows) < max_policy_steps and env.step_idx < env.n_steps:
            if time.monotonic() - started >= max_sec:
                break
            time_sec = float(env.sim.state.time_sec)
            state = env.sim.state.copy()
            previous = env.previous.copy()
            follower_state_before = {
                "native": _follower_state_snapshot(
                    env._ensure_optimizer_controller().nash_solver,
                ),
                "replay": _follower_state_snapshot(env.controller.nash_solver),
            }
            native_result, forecast, _, _, encoded = env._optimizer_decision()
            native_prices = _price_snapshot(env.optimizer_controller.nash_solver)

            replay_coordination = env.action_schema.decode(
                encoded,
                previous,
                CoordinationMask.named("RL-LINEAR"),
            )
            env.provider.action = replay_coordination
            replay_result = env.controller.decide_with_info(
                state.copy(), forecast, previous.copy(),
            )
            replay_prices = _price_snapshot(env.controller.nash_solver)
            native_ttt = _one_step_ttt(env, native_result.control, forecast)
            replay_ttt = _one_step_ttt(env, replay_result.control, forecast)
            all_endpoint_indices = np.flatnonzero(np.abs(encoded) >= 1.0 - 1.0e-7)
            certificate_start = env.action_dim - len(env.action_schema.certificate_ramps)
            saturated_indices = all_endpoint_indices[all_endpoint_indices < certificate_start]
            certificate_endpoint_indices = all_endpoint_indices[
                all_endpoint_indices >= certificate_start
            ]
            row = {
                "policy_step": len(rows),
                "simulation_step": int(env.step_idx),
                "time_sec": time_sec,
                "phase": "peak" if time_sec < 5220.0 else "recovery",
                "encoded_saturated_count": int(saturated_indices.size),
                "encoded_budget_saturated_count": int(np.count_nonzero(saturated_indices < 2)),
                "encoded_price_saturated_count": int(np.count_nonzero(saturated_indices >= 2)),
                "encoded_certificate_endpoint_count": int(certificate_endpoint_indices.size),
                "encoded_saturated_names": [
                    env.action_schema.names[int(index)] for index in saturated_indices
                ],
                "encoded_action": encoded.astype(float).tolist(),
                "leader_objective": {
                    "native": float(native_result.leader_objective),
                    "replay": float(replay_result.leader_objective),
                },
                "nash": {
                    "native_converged": bool(native_result.nash.converged),
                    "replay_converged": bool(replay_result.nash.converged),
                    "native_residual_objective": float(native_result.nash.residual_objective),
                    "replay_residual_objective": float(replay_result.nash.residual_objective),
                    "native_residual_control": float(native_result.nash.residual_control),
                    "replay_residual_control": float(replay_result.nash.residual_control),
                },
                "follower_state_before": follower_state_before,
                "follower_state_after": {
                    "native": _follower_state_snapshot(env.optimizer_controller.nash_solver),
                    "replay": _follower_state_snapshot(env.controller.nash_solver),
                },
                "control_difference": compare_controls(
                    env, native_result.control, replay_result.control,
                ),
                "native_control": _control_snapshot(env, native_result.control),
                "replay_control": _control_snapshot(env, replay_result.control),
                "price_difference": compare_prices(env, native_prices, replay_prices),
                "native_prices": native_prices,
                "replay_prices": replay_prices,
                "one_step_ttt": {"native": native_ttt, "replay": replay_ttt},
            }
            rows.append(row)
            trace.write(json.dumps(row, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps({
                "scenario": scenario,
                "policy_step": len(rows),
                "time_sec": time_sec,
                "native_ttt": native_ttt["total"],
                "replay_ttt": replay_ttt["total"],
                "vsl_max_abs": row["control_difference"]["vsl_all_segments"]["max_abs"],
                "nonmerge_price_nonzero": row["price_difference"]["vsl_marginal_price_nonmerge"]["native_nonzero_count"],
                "saturated": row["encoded_saturated_count"],
            }), flush=True)
            env.step_with_control(native_result.control)

    summary = _summarize(rows, scenario, time.monotonic() - started)
    (output_dir / f"{scenario}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        default="sweet_170_incident_w60,sweet_190_w60",
    )
    parser.add_argument("--output-dir", default="results/pstack_gap_diagnosis_v1/parity")
    parser.add_argument("--max-policy-steps", type=int, default=75)
    parser.add_argument("--max-sec", type=float, default=14400.0)
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    summaries = []
    for scenario in (item.strip() for item in args.scenarios.split(",") if item.strip()):
        summaries.append(diagnose_scenario(
            scenario,
            output_dir,
            max(0, int(args.max_policy_steps)),
            max(0.0, float(args.max_sec)),
        ))
    aggregate = {"format_version": "pstack_adapter_parity_v1", "scenarios": summaries}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
