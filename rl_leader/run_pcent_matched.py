"""Run P-CENT under the same experiment contract and warm-up as RL evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from rl_leader.env import RLLeaderEnv
from src.controllers.centralized_mpc import CentralizedMPC


TEACHER_CONTRACT_VERSION = "pcent_teacher_contract_v1"
ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_contract(env: RLLeaderEnv) -> tuple[dict, str]:
    payload = {
        "version": TEACHER_CONTRACT_VERSION,
        "controller": "CentralizedMPC",
        "mode": "proposed",
        "horizon_steps": int(env.cfg.mpc.horizon_steps),
        "centralized_mpc_source_sha256": _sha256_file(
            ROOT / "src" / "controllers" / "centralized_mpc.py"
        ),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "scope_note": (
            "plant/config semantics are fingerprinted separately by ExperimentContract"
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return payload, hashlib.sha256(encoded).hexdigest()


def _numeric_mapping(values) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _control_payload(control) -> dict:
    return {
        "N_P_star": float(control.N_P_star),
        "N_UF_star": float(control.N_UF_star),
        "green_times": _numeric_mapping(control.green_times),
        "offsets": _numeric_mapping(control.offsets),
        "vsl": _numeric_mapping(control.vsl),
        "ramp_metering": _numeric_mapping(control.ramp_metering),
        "inflow_outflow_allocation": _numeric_mapping(
            control.inflow_outflow_allocation
        ),
    }


def _step_validity(
    env: RLLeaderEnv,
    demand,
    diagnostics: dict,
    inventory_before: float,
    inventory_after: float,
) -> bool:
    mainline_arrivals = (
        sum(float(value) for value in demand.freeway_mainline.values())
        * env.cfg.simulation.T_c_h
    )
    external_arrivals = (
        mainline_arrivals
        + float(diagnostics.get("urban_demand_arrivals_veh", 0.0))
        + float(diagnostics.get("onramp_arrivals_veh", 0.0))
    )
    completed = (
        float(diagnostics.get("boundary_out_sink_veh", 0.0))
        + float(diagnostics.get("mainline_exit_flow_total", 0.0))
        * env.cfg.simulation.T_c_h
    )
    residual = inventory_after - inventory_before - external_arrivals + completed
    projection = float(diagnostics.get("movement_queue_projection_veh", 0.0))
    rejected = float(
        diagnostics.get("coupling_offramp_arrivals_rejected_veh", 0.0)
    )
    overflow = float(diagnostics.get("ramp_queue_overflow_count", 0.0)) + float(
        diagnostics.get("queue_overflow_count", 0.0)
    )
    return bool(
        projection <= 1.0e-9
        and rejected <= 1.0e-9
        and abs(residual) <= 1.0e-3
        and overflow <= 0.0
    )


def _compact_result(row: dict) -> dict:
    profile_id = row.get("experiment_profile_id")
    if profile_id is None:
        profile_id = row.get("experiment_contract", {}).get("profile_id")
    if not profile_id:
        raise ValueError("P-CENT result is missing its experiment profile id")
    return {
        "experiment_contract_sha256": row["experiment_contract_sha256"],
        "teacher_contract_sha256": row.get("teacher_contract_sha256"),
        "experiment_profile_id": str(profile_id),
        **{
            key: row[key]
            for key in (
                "scenario",
                "policy_steps",
                "completed",
                "controlled_ttt",
                "terminal_inventory",
                "validity_gate_pass",
                "elapsed_sec",
                "trace_path",
            )
        },
    }


def run_scenario(
    scenario: str,
    *,
    max_steps: int,
    max_sec: float,
    output_dir: Path,
) -> dict:
    env = RLLeaderEnv(scenario_name=scenario)
    env.reset()
    teacher_contract, teacher_contract_sha256 = _teacher_contract(env)
    controller = CentralizedMPC(env.cfg, mode="proposed")
    controlled_ttt = 0.0
    validity = True
    rows = []
    started = time.monotonic()
    trace_path = output_dir / f"{scenario}.jsonl"
    result_path = output_dir / f"{scenario}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with trace_path.open("w", encoding="utf-8") as trace:
        while (
            env.step_idx < env.n_steps
            and len(rows) < max_steps
            and time.monotonic() - started < max_sec
        ):
            forecast = env._forecast()
            env._update_rl_far_gate(forecast)
            inventory_before = float(env._inventory())
            simulation_time_before = float(env.sim.state.time_sec)
            decision = controller.decide_with_info(
                env.sim.state.copy(), forecast, env.previous
            )
            log = env.sim.step(decision.control, forecast[0], env.step_idx)
            step_ttt = float(log.urban_ttt + log.freeway_ttt)
            inventory_after = float(env._inventory())
            step_validity = _step_validity(
                env,
                forecast[0],
                log.diagnostics,
                inventory_before,
                inventory_after,
            )
            validity = validity and step_validity
            controlled_ttt += step_ttt
            env.previous = decision.control.copy()
            env.step_idx += 1
            row = {
                "policy_step": int(len(rows)),
                "simulation_step": int(env.step_idx - 1),
                "simulation_time_before_sec": simulation_time_before,
                "simulation_time_after_sec": float(env.sim.state.time_sec),
                "step_ttt": step_ttt,
                "step_urban_ttt": float(log.urban_ttt),
                "step_freeway_ttt": float(log.freeway_ttt),
                "inventory_before": inventory_before,
                "inventory_after": inventory_after,
                "validity_gate_pass": bool(step_validity),
                "centralized_objective": float(decision.objective),
                "centralized_solver_evaluations": int(decision.solver_evaluations),
                "centralized_converged": bool(decision.converged),
                "centralized_computation_time_sec": float(
                    decision.computation_time_sec
                ),
                "control": _control_payload(decision.control),
            }
            rows.append(row)
            trace.write(json.dumps(row, separators=(",", ":")) + "\n")
            trace.flush()
            if len(rows) == 1 or len(rows) % 5 == 0 or env.step_idx >= env.n_steps:
                print(
                    f"P-CENT scenario={scenario} step={len(rows)}/{max_steps} "
                    f"ttt={controlled_ttt:.3f} valid={int(validity)}",
                    flush=True,
                )

    completed = env.step_idx >= env.n_steps or len(rows) >= max_steps
    result = {
        **env.experiment_contract.artifact_fields(),
        "teacher_contract_version": TEACHER_CONTRACT_VERSION,
        "teacher_contract_sha256": teacher_contract_sha256,
        "teacher_contract": teacher_contract,
        "experiment_profile_id": env.experiment_contract_payload["profile_id"],
        "scenario": scenario,
        "warmup_steps": int(env.warmup),
        "policy_steps": int(len(rows)),
        "requested_steps": int(max_steps),
        "completed": bool(completed),
        "controlled_ttt": float(controlled_ttt),
        "total_ttt_including_warmup": float(env.sim.total_ttt),
        "terminal_inventory": float(env._inventory()),
        "validity_gate_pass": bool(validity),
        "elapsed_sec": float(time.monotonic() - started),
        "trace_path": str(trace_path),
        "rows": rows,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"P-CENT complete scenario={scenario} steps={len(rows)} "
        f"ttt={controlled_ttt:.3f} valid={int(validity)}",
        flush=True,
    )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", default="sweet_170_w60,sweet_190_w60"
    )
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--max-sec", type=float, default=7200.0)
    parser.add_argument(
        "--output-dir",
        default="results/rl_phase0_implementation_20260827/pcent_matched",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    summary = [
        run_scenario(
            scenario,
            max_steps=args.max_steps,
            max_sec=args.max_sec,
            output_dir=output_dir,
        )
        for scenario in scenarios
    ]
    compact = [_compact_result(row) for row in summary]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
