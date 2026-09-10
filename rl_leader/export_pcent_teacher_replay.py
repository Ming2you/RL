"""Replay matched P-CENT controls into a contract-verified physical teacher set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.experiment_contract import ExperimentContract
from rl_leader.run_pcent_matched import _step_validity
from src.models.state import ControlAction


def control_from_payload(payload: dict) -> ControlAction:
    def mapping(name: str) -> dict[str, float]:
        values = payload.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"control field {name} must be a mapping")
        return {str(key): float(value) for key, value in values.items()}

    return ControlAction(
        N_P_star=float(payload["N_P_star"]),
        N_UF_star=float(payload["N_UF_star"]),
        green_times=mapping("green_times"),
        offsets=mapping("offsets"),
        vsl=mapping("vsl"),
        ramp_metering=mapping("ramp_metering"),
        inflow_outflow_allocation=mapping("inflow_outflow_allocation"),
    )


def physical_action_names(env: RLLeaderEnv) -> tuple[str, ...]:
    names = []
    for signal in env.action_schema.signals:
        names.extend((f"green.{signal}.p1", f"offset.{signal}"))
    for ramp in env.action_schema.ramps:
        link = env.net.ramp_to_freeway[ramp]
        segment = int(env.net.ramp_merge_segment_index.get(ramp, 0))
        names.extend((f"meter.{ramp}", f"vsl.{link}.segment.{segment}"))
    for key in env.action_schema.nonmerge_vsl_keys:
        link, segment_text = key.rsplit("__seg", 1)
        names.append(f"vsl.{link}.segment.{int(segment_text)}")
    return tuple(names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_teacher(
    source_path: Path,
    output_path: Path,
    *,
    parity_tolerance: float,
) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    contract = ExperimentContract.from_artifact(
        source["experiment_contract"], source["experiment_contract_sha256"]
    )
    env = RLLeaderEnv(experiment_contract=contract)
    observation = env.reset()
    if env.experiment_contract_fingerprint != source["experiment_contract_sha256"]:
        raise ValueError("P-CENT source contract does not match replay environment")
    if str(source["scenario"]) != str(env.scenario_name):
        raise ValueError("P-CENT source scenario does not match replay environment")

    observations = []
    next_observations = []
    physical_actions = []
    budgets = []
    step_ttt = []
    inventories = []
    dones = []
    max_errors = {
        "time_before_sec": 0.0,
        "time_after_sec": 0.0,
        "inventory_before": 0.0,
        "inventory_after": 0.0,
        "step_ttt": 0.0,
        "step_urban_ttt": 0.0,
        "step_freeway_ttt": 0.0,
    }
    replay_valid = True
    rows = list(source["rows"])
    for index, row in enumerate(rows):
        if int(row["policy_step"]) != index or int(row["simulation_step"]) != env.step_idx:
            raise RuntimeError(f"P-CENT row order mismatch at policy step {index}")
        control = control_from_payload(row["control"])
        forecast = env._forecast()
        inventory_before = float(env._inventory())
        time_before = float(env.sim.state.time_sec)
        observations.append(np.asarray(observation, dtype=np.float32))
        physical_actions.append(env.response_vector(control))
        budgets.append([float(control.N_P_star), float(control.N_UF_star)])

        log = env.sim.step(control, forecast[0], env.step_idx)
        current_ttt = float(log.urban_ttt + log.freeway_ttt)
        inventory_after = float(env._inventory())
        env.previous = control.copy()
        env.step_idx += 1
        observation = env._observe()
        done = env.step_idx >= env.n_steps

        actual = {
            "time_before_sec": time_before,
            "time_after_sec": float(env.sim.state.time_sec),
            "inventory_before": inventory_before,
            "inventory_after": inventory_after,
            "step_ttt": current_ttt,
            "step_urban_ttt": float(log.urban_ttt),
            "step_freeway_ttt": float(log.freeway_ttt),
        }
        expected = {
            "time_before_sec": float(row["simulation_time_before_sec"]),
            "time_after_sec": float(row["simulation_time_after_sec"]),
            "inventory_before": float(row["inventory_before"]),
            "inventory_after": float(row["inventory_after"]),
            "step_ttt": float(row["step_ttt"]),
            "step_urban_ttt": float(row["step_urban_ttt"]),
            "step_freeway_ttt": float(row["step_freeway_ttt"]),
        }
        for key in max_errors:
            max_errors[key] = max(max_errors[key], abs(actual[key] - expected[key]))
        replay_valid = replay_valid and _step_validity(
            env, forecast[0], log.diagnostics, inventory_before, inventory_after
        )
        next_observations.append(np.asarray(observation, dtype=np.float32))
        step_ttt.append(current_ttt)
        inventories.append([inventory_before, inventory_after])
        dones.append(done)

    worst_error = max(max_errors.values(), default=0.0)
    if worst_error > parity_tolerance:
        raise RuntimeError(
            f"P-CENT teacher replay parity failed: worst_error={worst_error}, "
            f"errors={max_errors}"
        )
    if not replay_valid:
        raise RuntimeError("P-CENT teacher replay failed the conservation validity gate")

    observation_names = tuple(env.observation_schema.names)
    physical_names = physical_action_names(env)
    physical_array = np.asarray(physical_actions, dtype=np.float32)
    budget_array = np.asarray(budgets, dtype=np.float32)
    physical_std = np.std(physical_array, axis=0)
    varying_physical_names = [
        name for name, std in zip(physical_names, physical_std)
        if float(std) > 1.0e-6
    ]
    state_indices = np.asarray([
        index for index, name in enumerate(observation_names)
        if not name.startswith("dual.") and not name.startswith("follower.")
    ], dtype=np.int64)
    observation_array = np.asarray(observations, dtype=np.float32)
    next_observation_array = np.asarray(next_observations, dtype=np.float32)
    teacher_provenance_complete = bool(
        source.get("teacher_contract_sha256") and source.get("teacher_contract")
    )
    manifest = {
        "format_version": "pcent_physical_teacher_v1",
        **env.experiment_contract.artifact_fields(),
        "scenario": str(source["scenario"]),
        "source_result": str(source_path),
        "source_result_sha256": _sha256(source_path),
        "source_policy_steps": len(rows),
        "observation_schema": env.observation_schema.metadata(),
        "state_observation_names": [observation_names[index] for index in state_indices],
        "physical_action_names": list(physical_names),
        "physical_action_unique_count": int(
            np.unique(np.round(physical_array, 6), axis=0).shape[0]
        ),
        "physical_action_varying_names": varying_physical_names,
        "budget_names": ["N_P_star", "N_UF_star"],
        "budget_unique_count": int(
            np.unique(np.round(budget_array, 6), axis=0).shape[0]
        ),
        "teacher_semantics": "centralized_physical_control_on_pcent_trajectory",
        "budget_teacher_semantics": "not_identified_by_centralized_controller",
        "teacher_contract_sha256": source.get("teacher_contract_sha256"),
        "teacher_contract": source.get("teacher_contract"),
        "teacher_implementation_provenance_complete": teacher_provenance_complete,
        "teacher_implementation_provenance_limit": (
            None if teacher_provenance_complete else
            "source P-CENT result predates a persisted teacher-code digest"
        ),
        "deployable_follower_memory_aligned": False,
        "eligible_for_direct_physical_bc": True,
        "eligible_for_budget_bc": False,
        "eligible_for_price_bc": False,
        "replay_max_abs_errors": max_errors,
        "validity_gate_pass": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observation_array,
        next_observations=next_observation_array,
        state_observations=observation_array[:, state_indices],
        next_state_observations=next_observation_array[:, state_indices],
        physical_actions=physical_array,
        budgets=budget_array,
        step_ttt=np.asarray(step_ttt, dtype=np.float32),
        inventories=np.asarray(inventories, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        state_observation_indices=state_indices,
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    summary = []
    for value in args.sources:
        source_path = Path(value)
        output_path = output_dir / f"{source_path.stem}.npz"
        manifest = export_teacher(
            source_path,
            output_path,
            parity_tolerance=float(args.parity_tolerance),
        )
        summary.append({
            "scenario": manifest["scenario"],
            "steps": manifest["source_policy_steps"],
            "physical_action_unique_count": manifest["physical_action_unique_count"],
            "physical_action_varying_dimension_count": len(
                manifest["physical_action_varying_names"]
            ),
            "eligible_for_direct_physical_bc": manifest[
                "eligible_for_direct_physical_bc"
            ],
            "eligible_for_budget_bc": manifest["eligible_for_budget_bc"],
            "eligible_for_price_bc": manifest["eligible_for_price_bc"],
            "output": str(output_path),
            "experiment_contract_sha256": manifest["experiment_contract_sha256"],
        })
        print(
            f"P-CENT teacher replay scenario={manifest['scenario']} "
            f"steps={manifest['source_policy_steps']} parity=exact",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
