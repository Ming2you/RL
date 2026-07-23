from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.authority import PRIMARY_CONTROLLERS
from src.analysis.free_flow_reference import FreeFlowReference, compute_free_flow_reference
from src.experiments.six_controller_comparison import _write_csv, run_controller, summarize_controller
from src.models.demand import ScenarioConfig, apply_scenario_network_overrides, load_scenarios
from src.models.state import ExperimentConfig
from src.simulation.closed_loop_runner import run_closed_loop


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _boundary_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result["run_rows"]
    b_in = [float(r.get("B_in", 0.0)) for r in rows]
    b_out = [float(r.get("B_out", 0.0)) for r in rows]
    controllable = [float(r.get("boundary_balance_controllable", 1.0)) for r in rows]
    degenerate = [float(r.get("boundary_balance_degenerate", 0.0)) for r in rows]
    return {
        "mean_B_in": round(_mean(b_in), 6),
        "mean_B_out": round(_mean(b_out), 6),
        "mean_B_sum": round(_mean([a + b for a, b in zip(b_in, b_out)]), 6),
        "mean_CV_boundary": round(_mean([float(r.get("CV_boundary", 0.0)) for r in rows]), 6),
        "max_OverflowRatio_boundary": round(
            max([float(r.get("OverflowRatio_boundary", 0.0)) for r in rows] or [0.0]), 6
        ),
        "boundary_controllable_fraction": round(_mean(controllable), 6),
        "boundary_degenerate_fraction": round(_mean(degenerate), 6),
    }


def _summarize_no_control(
    cfg: ExperimentConfig,
    scenario_name: str,
    result: Dict[str, Any],
    reference: FreeFlowReference,
) -> Dict[str, Any]:
    net = cfg.network
    rows = result["run_rows"]
    horizon_h = cfg.simulation.T_total / 3600.0
    t_c_h = cfg.simulation.T_c_h
    completed_urban = sum(float(r.get("boundary_out_sink_veh", 0.0)) for r in rows)
    completed_freeway = sum(float(r.get("mainline_exit_flow_total", 0.0)) * t_c_h for r in rows)
    completed = completed_urban + completed_freeway
    final = result["final_state"]
    onramp_terminal = sum(final.ramp_queue.values()) + sum(
        max(0.0, final.urban_movement_queue.get(m, 0.0))
        for movements in net.on_ramp_to_movement.values()
        for m in movements
    )
    total_delay = result["total_ttt"] - reference.total_ttt
    summary = {
        "scenario": scenario_name,
        "controller_id": "NO-CONTROL",
        "authority_group": "",
        "leader_enabled": "",
        "centralized": "",
        "authority_ok": "",
        "authority_violations": "",
        "total_ttt": round(result["total_ttt"], 3),
        "urban_ttt": round(result["urban_ttt"], 3),
        "freeway_ttt": round(result["freeway_ttt"], 3),
        "free_flow_reference_total_ttt": round(reference.total_ttt, 3),
        "free_flow_reference_urban_ttt": round(reference.urban_ttt, 3),
        "free_flow_reference_freeway_ttt": round(reference.freeway_ttt, 3),
        "total_delay": round(total_delay, 3),
        "urban_delay": round(result["urban_ttt"] - reference.urban_ttt, 3),
        "freeway_delay": round(result["freeway_ttt"] - reference.freeway_ttt, 3),
        "completed_vehicles": round(completed, 1),
        "network_throughput_veh_h": round(completed / max(horizon_h, 1e-9), 1),
        "average_delay_per_completed_vehicle_h": round(total_delay / max(completed, 1.0), 6),
        "terminal_total_vehicles": round(
            final.total_urban_vehicles(net) + final.total_freeway_vehicles(net), 1
        ),
        "terminal_urban_vehicles": round(final.total_urban_vehicles(net), 1),
        "terminal_onramp_vehicles": round(onramp_terminal, 1),
        "terminal_freeway_vehicles": round(
            final.total_freeway_vehicles(net) - sum(final.ramp_queue.values()), 1
        ),
        "computation_time_sec": 0.0,
        "solver_evaluations": 0.0,
        "solver_converged_rate": "",
    }
    summary.update(_boundary_summary(result))
    return summary


def _add_no_control_comparison(row: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    base_ttt = float(baseline["total_ttt"])
    base_delay = float(baseline["total_delay"])
    base_urban = float(baseline["urban_ttt"])
    base_freeway = float(baseline["freeway_ttt"])
    base_balance = float(baseline.get("mean_B_sum", 0.0))
    out["total_ttt_improvement_vs_no_control_pct"] = round(
        100.0 * (base_ttt - float(row["total_ttt"])) / max(base_ttt, 1e-9), 3
    )
    out["total_ttt_delta_vs_no_control"] = round(base_ttt - float(row["total_ttt"]), 3)
    out["total_delay_improvement_vs_no_control_pct"] = round(
        100.0 * (base_delay - float(row["total_delay"])) / max(abs(base_delay), 1e-9), 3
    )
    out["total_delay_delta_vs_no_control"] = round(base_delay - float(row["total_delay"]), 3)
    out["urban_ttt_improvement_vs_no_control_pct"] = round(
        100.0 * (base_urban - float(row["urban_ttt"])) / max(base_urban, 1e-9), 3
    )
    out["urban_ttt_delta_vs_no_control"] = round(base_urban - float(row["urban_ttt"]), 3)
    out["freeway_ttt_improvement_vs_no_control_pct"] = round(
        100.0 * (base_freeway - float(row["freeway_ttt"])) / max(base_freeway, 1e-9), 3
    )
    out["freeway_ttt_delta_vs_no_control"] = round(base_freeway - float(row["freeway_ttt"]), 3)
    out["throughput_delta_vs_no_control"] = round(
        float(row["network_throughput_veh_h"]) - float(baseline["network_throughput_veh_h"]), 3
    )
    out["terminal_delta_vs_no_control"] = round(
        float(row["terminal_total_vehicles"]) - float(baseline["terminal_total_vehicles"]), 3
    )
    out["mean_B_sum_delta_vs_no_control"] = round(float(row.get("mean_B_sum", 0.0)) - base_balance, 6)
    out["boundary_balance_non_degraded_vs_no_control"] = bool(
        float(row.get("mean_B_sum", 0.0)) <= base_balance + 1.0e-9
        and float(row.get("max_OverflowRatio_boundary", 0.0))
        <= float(baseline.get("max_OverflowRatio_boundary", 0.0)) + 1.0e-9
    )
    return out


def _parse_updates(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"simulation": {"T_total": args.T_total}}
    if args.seed is not None:
        overrides["simulation"]["random_seed"] = args.seed
    overrides["mpc"] = {"relaxed_quantized_controls": True}
    if args.leader_search_mode is not None:
        overrides["mpc"]["leader_search_mode"] = args.leader_search_mode
    if args.leader_candidate_count is not None:
        overrides["mpc"]["leader_candidate_count"] = args.leader_candidate_count
    if args.leader_refinement_candidate_count is not None:
        overrides["mpc"]["leader_refinement_candidate_count"] = args.leader_refinement_candidate_count
    if args.leader_continuous_max_evals is not None:
        overrides["mpc"]["leader_continuous_max_evals"] = args.leader_continuous_max_evals
    if args.leader_continuous_seed_count is not None:
        overrides["mpc"]["leader_continuous_seed_count"] = args.leader_continuous_seed_count
    if args.leader_continuous_prefilter_samples is not None:
        overrides["mpc"]["leader_continuous_prefilter_samples"] = args.leader_continuous_prefilter_samples
    if args.leader_continuous_prefilter_top_k is not None:
        overrides["mpc"]["leader_continuous_prefilter_top_k"] = args.leader_continuous_prefilter_top_k
    if args.disable_leader_continuous_hard_precheck:
        overrides["mpc"]["leader_continuous_hard_precheck"] = False
    if args.leader_continuous_precheck_spillback_tolerance_veh is not None:
        overrides["mpc"]["leader_continuous_precheck_spillback_tolerance_veh"] = (
            args.leader_continuous_precheck_spillback_tolerance_veh
        )
    if args.disable_leader_continuous_parallel_multistart:
        overrides["mpc"]["leader_continuous_parallel_multistart"] = False
    if args.disable_leader_continuous_sensitivity_directions:
        overrides["mpc"]["leader_continuous_use_sensitivity_directions"] = False
    if args.leader_continuous_local_iterations is not None:
        overrides["mpc"]["leader_continuous_local_iterations"] = args.leader_continuous_local_iterations
    if args.leader_continuous_initial_step_fraction is not None:
        overrides["mpc"]["leader_continuous_initial_step_fraction"] = (
            args.leader_continuous_initial_step_fraction
        )
    if args.leader_continuous_shrink_factor is not None:
        overrides["mpc"]["leader_continuous_shrink_factor"] = args.leader_continuous_shrink_factor
    if args.leader_continuous_min_np_step_veh is not None:
        overrides["mpc"]["leader_continuous_min_np_step_veh"] = args.leader_continuous_min_np_step_veh
    if args.leader_continuous_min_nuf_step_veh_h is not None:
        overrides["mpc"]["leader_continuous_min_nuf_step_veh_h"] = args.leader_continuous_min_nuf_step_veh_h
    if args.max_nash_iter is not None:
        overrides["mpc"]["max_nash_iter"] = args.max_nash_iter
    if args.optimizer_maxiter is not None:
        overrides["mpc"]["optimizer_maxiter"] = args.optimizer_maxiter
    if args.optimizer_n_starts is not None:
        overrides["mpc"]["optimizer_n_starts"] = args.optimizer_n_starts
    if args.grid_parallel_backend is not None:
        overrides["mpc"]["grid_parallel_backend"] = args.grid_parallel_backend
    if args.grid_parallel_max_workers is not None:
        overrides["mpc"]["grid_parallel_max_workers"] = args.grid_parallel_max_workers
    if args.grid_parallel_chunk_size is not None:
        overrides["mpc"]["grid_parallel_chunk_size"] = args.grid_parallel_chunk_size
    if args.grid_global_refresh_sec is not None:
        overrides["mpc"]["grid_global_refresh_sec"] = args.grid_global_refresh_sec
    if args.disable_grid_process_pool_reuse:
        overrides["mpc"]["grid_reuse_process_pool"] = False
    if args.stackelberg_prefilter_top_k is not None:
        overrides["mpc"]["stackelberg_prefilter_top_k"] = args.stackelberg_prefilter_top_k
    if args.stackelberg_prefilter_local_top_k is not None:
        overrides["mpc"]["stackelberg_prefilter_local_top_k"] = args.stackelberg_prefilter_local_top_k
    if args.stackelberg_fallback_full_refresh_sec is not None:
        overrides["mpc"]["stackelberg_fallback_full_refresh_sec"] = args.stackelberg_fallback_full_refresh_sec
    if args.disable_stackelberg_fallback:
        overrides["mpc"]["stackelberg_enable_fallback"] = False
    if args.disable_stackelberg_pfo_fallback_cache:
        overrides["mpc"]["stackelberg_fallback_use_cached_pfo"] = False
    if args.disable_stackelberg_process_pool_reuse:
        overrides["mpc"]["stackelberg_reuse_process_pool"] = False
    if args.stackelberg_leader_parallel_backend is not None:
        overrides["mpc"]["stackelberg_leader_parallel_backend"] = args.stackelberg_leader_parallel_backend
    if args.stackelberg_leader_parallel_max_workers is not None:
        overrides["mpc"]["stackelberg_leader_parallel_max_workers"] = args.stackelberg_leader_parallel_max_workers
    if args.stackelberg_inner_backend_when_outer_process is not None:
        overrides["mpc"]["stackelberg_inner_backend_when_outer_process"] = (
            args.stackelberg_inner_backend_when_outer_process
        )
    if args.stackelberg_allocation_mode is not None:
        overrides["mpc"]["stackelberg_allocation_mode"] = args.stackelberg_allocation_mode
    if args.freeway_prediction_horizon_steps is not None:
        overrides["freeway_follower"] = {
            "freeway_prediction_horizon_steps": args.freeway_prediction_horizon_steps
        }
    return overrides


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default="src/config/scenarios.yaml")
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--T-total", type=float, default=7200.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controllers", default=",".join(PRIMARY_CONTROLLERS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--leader-search-mode", choices=["grid", "continuous"], default=None)
    parser.add_argument("--leader-candidate-count", type=int, default=None)
    parser.add_argument("--leader-refinement-candidate-count", type=int, default=None)
    parser.add_argument("--leader-continuous-max-evals", type=int, default=None)
    parser.add_argument("--leader-continuous-seed-count", type=int, default=None)
    parser.add_argument("--leader-continuous-prefilter-samples", type=int, default=None)
    parser.add_argument("--leader-continuous-prefilter-top-k", type=int, default=None)
    parser.add_argument("--disable-leader-continuous-hard-precheck", action="store_true")
    parser.add_argument("--leader-continuous-precheck-spillback-tolerance-veh", type=float, default=None)
    parser.add_argument("--disable-leader-continuous-parallel-multistart", action="store_true")
    parser.add_argument("--disable-leader-continuous-sensitivity-directions", action="store_true")
    parser.add_argument("--leader-continuous-local-iterations", type=int, default=None)
    parser.add_argument("--leader-continuous-initial-step-fraction", type=float, default=None)
    parser.add_argument("--leader-continuous-shrink-factor", type=float, default=None)
    parser.add_argument("--leader-continuous-min-np-step-veh", type=float, default=None)
    parser.add_argument("--leader-continuous-min-nuf-step-veh-h", type=float, default=None)
    parser.add_argument("--max-nash-iter", type=int, default=None)
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    parser.add_argument("--optimizer-n-starts", type=int, default=None)
    parser.add_argument("--grid-parallel-backend", choices=["serial", "thread", "process"], default=None)
    parser.add_argument("--grid-parallel-max-workers", type=int, default=None)
    parser.add_argument("--grid-parallel-chunk-size", type=int, default=None)
    parser.add_argument("--grid-global-refresh-sec", type=float, default=None)
    parser.add_argument("--disable-grid-process-pool-reuse", action="store_true")
    parser.add_argument("--stackelberg-prefilter-top-k", type=int, default=None)
    parser.add_argument("--stackelberg-prefilter-local-top-k", type=int, default=None)
    parser.add_argument("--stackelberg-fallback-full-refresh-sec", type=float, default=None)
    parser.add_argument("--disable-stackelberg-fallback", action="store_true")
    parser.add_argument("--disable-stackelberg-pfo-fallback-cache", action="store_true")
    parser.add_argument("--disable-stackelberg-process-pool-reuse", action="store_true")
    parser.add_argument("--stackelberg-leader-parallel-backend", choices=["serial", "thread", "process"], default=None)
    parser.add_argument("--stackelberg-leader-parallel-max-workers", type=int, default=None)
    parser.add_argument("--stackelberg-inner-backend-when-outer-process", choices=["serial", "thread"], default=None)
    parser.add_argument("--stackelberg-allocation-mode", choices=["direct", "simplified", "pso"], default=None)
    parser.add_argument("--freeway-prediction-horizon-steps", type=int, default=None)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    overrides = _parse_updates(args)
    base_cfg = ExperimentConfig.from_file(args.config, overrides)
    scenarios = load_scenarios(args.scenarios_config)
    if args.scenario == "all":
        scenario_names = list(scenarios)
    else:
        scenario_names = [name.strip() for name in args.scenario.split(",") if name.strip()]
    requested = [name.strip() for name in args.controllers.split(",") if name.strip()]

    all_controller_rows: List[Dict[str, Any]] = []
    all_no_control_rows: List[Dict[str, Any]] = []
    all_comparison_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "overrides": overrides,
        "controllers": requested,
        "scenarios": scenario_names,
        "results": {},
    }

    for scenario_name in scenario_names:
        if scenario_name not in scenarios:
            raise SystemExit(f"Unknown scenario: {scenario_name}")
        scenario: ScenarioConfig = scenarios[scenario_name]
        cfg = apply_scenario_network_overrides(base_cfg.with_updates({}), scenario)
        reference = compute_free_flow_reference(cfg, scenario)
        scenario_dir = output / "runs" / scenario_name

        t0 = time.perf_counter()
        no_control = run_closed_loop(
            base_cfg.with_updates({}),
            scenario,
            mode="baseline",
            baseline_mode="no_control",
            output_dir=scenario_dir / "NO-CONTROL",
        )
        no_control_row = _summarize_no_control(cfg, scenario_name, no_control, reference)
        no_control_row["wall_time_sec"] = round(time.perf_counter() - t0, 3)
        all_no_control_rows.append(no_control_row)
        all_comparison_rows.append(_add_no_control_comparison(no_control_row, no_control_row))

        scenario_results: Dict[str, Any] = {"NO-CONTROL": no_control_row}
        print(
            f"{scenario_name} NO-CONTROL total_ttt={no_control_row['total_ttt']:.3f} "
            f"B_sum={no_control_row['mean_B_sum']:.6f}"
        )
        for controller_id in requested:
            t0 = time.perf_counter()
            result = run_controller(cfg, scenario, controller_id, scenario_dir / controller_id)
            row = summarize_controller(cfg, controller_id, result, reference)
            row["scenario"] = scenario_name
            row["wall_time_sec"] = round(time.perf_counter() - t0, 3)
            row.update(_boundary_summary(result))
            all_controller_rows.append(row)
            comparison = _add_no_control_comparison(row, no_control_row)
            all_comparison_rows.append(comparison)
            scenario_results[controller_id] = comparison
            print(
                f"{scenario_name} {controller_id} total_ttt={row['total_ttt']:.3f} "
                f"improvement={comparison['total_ttt_improvement_vs_no_control_pct']:.3f}% "
                f"B_sum={row['mean_B_sum']:.6f}"
            )
        summary_json["results"][scenario_name] = scenario_results

    _write_csv(output / "all_controller_summary.csv", all_controller_rows)
    _write_csv(output / "all_no_control_summary.csv", all_no_control_rows)
    analysis = output / "analysis"
    _write_csv(analysis / "summary_with_no_control.csv", all_comparison_rows)
    _write_csv(analysis / "controller_vs_no_control.csv", [
        row for row in all_comparison_rows if row["controller_id"] != "NO-CONTROL"
    ])
    (output / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"ALL_SCENARIOS output={output}")


if __name__ == "__main__":
    main()
