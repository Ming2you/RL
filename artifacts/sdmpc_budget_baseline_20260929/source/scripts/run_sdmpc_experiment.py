"""S-DMPC 실험: 동일 plant/수요 비교, 고정 budget 검증, 원래 잔차와 그림 보존.

Spec 03/04/05/06/08/10/11/12/15 및 docs/sdmpc_formulation.md를 사용한다.
기존 정본 factory는 비교군 생성에만 사용하며 기본 동작을 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = Path(__file__).read_bytes()
RUNNER_SHA256 = hashlib.sha256(RUNNER_SOURCE).hexdigest()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.metrics import evaluate, summarize_run
from src.models.demand import DemandProfile
from src.models.state import ControlAction, TrafficState, ExperimentConfig
from src.models.urban_queue_model import movement_balance_summary
from src.simulation.simulator import MixedTrafficSimulator, control_row, state_row
from work.run_claude_style_five_controller import build_cfg, make_controller, decide


IDS = {
    "NC": "NO-CONTROL",
    "PFO": "WU-FAITHFUL-FOLLOWER",
    "CANONICAL": "P-STACK-WU-FAITHFUL-APJOINT-FINAL",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=lambda x: x.item() if hasattr(x, "item") else str(x)),
                          encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inventory(state, cfg):
    # 버퍼/집계 queue는 더하지 않음. off-ramp transit은 freeway 총량 API에 별도 추가.
    return (state.total_urban_vehicles(cfg.network)
            + state.total_freeway_vehicles(cfg.network)
            + state.off_ramp_storage_occupancy_veh(cfg.network))


def config_for(args):
    cfg, scenario = build_cfg(args.scenario, args.demand_horizon,
                              wu_faithful_offset_enabled=True)
    cfg.simulation.T_total = args.duration
    cfg.simulation.random_seed = args.seed
    cfg.mpc.horizon_steps = args.horizon
    # 수요파의 길이와 실행 길이를 분리한다. 짧은 smoke가 수요파를 압축하지 않음.
    demand_cfg = cfg.with_updates({"simulation": {"T_total": args.demand_horizon}})
    return cfg, scenario, DemandProfile(demand_cfg, scenario)


def options_for(args):
    from src.controllers.sensitivity_dmpc import SDMPCOptions
    return SDMPCOptions(horizon_steps=args.horizon, max_iterations=args.iterations,
                        tolerance_np_veh=args.np_tolerance,
                        tolerance_nuf_veh_h=args.nuf_tolerance,
                        externality_enabled=not args.own_only, max_candidates=args.candidates,
                        trust_radius=args.trust_radius)


def common_demand_rows(args, cfg, profile):
    return [{"step": k, "time_sec": k * cfg.simulation.T_c_sec,
             **asdict(profile.at(k * cfg.simulation.T_c_sec))}
            for k in range(int(round(args.duration / cfg.simulation.T_c_sec)))]


def run_controller(name, args):
    cfg, scenario, profile = config_for(args)
    out = args.output / name
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner_source.py.txt").write_bytes(RUNNER_SOURCE)
    np.random.seed(args.seed)
    if name in {"SDMPC", "SDMPC-OWN"}:
        from src.controllers.sensitivity_dmpc import NoExecutableControlError, SensitivityDMPC
        (out / "core_source.py.txt").write_bytes((ROOT / "src/controllers/sensitivity_dmpc.py").read_bytes())
        options = options_for(args)
        options.externality_enabled = name != "SDMPC-OWN" and not args.own_only
        controller = SensitivityDMPC(cfg, options)
        warm_controller = make_controller(IDS["PFO"], cfg) if args.warm_start == "pfo" else None
        write_json(out / "solver_options.json", asdict(options))
    else:
        controller = make_controller(IDS[name], cfg)
    sim = MixedTrafficSimulator(cfg)
    previous = ControlAction.uncontrolled(cfg)
    initial_inventory = inventory(sim.state, cfg)
    n_steps = int(round(args.duration / cfg.simulation.T_c_sec))
    rows, controls, states, candidates, iterations, prices = [], [], [], [], [], []
    demand_rows = common_demand_rows(args, cfg, profile)
    demand_hash = hashlib.sha256(json.dumps(demand_rows, sort_keys=True).encode()).hexdigest()
    write_json(out / "demand.json", demand_rows)
    write_json(out / "config_used.yaml", cfg.to_dict())
    write_json(out / "metadata.json", {"scenario": args.scenario, "seed": args.seed,
               "duration_sec": args.duration, "demand_horizon_sec": args.demand_horizon,
               "warmup_steps": args.warmup_steps, "demand_sha256": demand_hash,
               "controller": name, "initial_inventory_veh": initial_inventory,
               "warm_start": args.warm_start if name.startswith("SDMPC") else None,
               "runner_sha256": RUNNER_SHA256,
               "core_sha256": hashlib.sha256((ROOT / "src/controllers/sensitivity_dmpc.py").read_bytes()).hexdigest()
                   if name.startswith("SDMPC") else None})
    for k in range(n_steps):
        forecast_depth = args.horizon + (max(0, cfg.mpc.leader_value_depth) if name == "CANONICAL" else 0)
        forecast = profile.horizon(sim.state.time_sec, forecast_depth)
        before_inventory = inventory(sim.state, cfg)
        before_np = sim.state.protected_accumulation_veh(cfg.network)
        started = time.perf_counter()
        if k < args.warmup_steps:
            action = ControlAction.uncontrolled(cfg)
        elif name in {"SDMPC", "SDMPC-OWN"}:
            if warm_controller is not None:
                # 기존 follower의 제어안은 초기화일 뿐, budget는 이후 새로 공개 생성한다.
                # 최종 선택은 하위 solver의 원 equality 검사를 통과한 후보만 사용한다.
                controller.warm_start_control = warm_controller.solve(sim.state.copy(), None, forecast, previous).control
            decision_error = None
            try:
                action = controller.decide(sim.state.copy(), forecast, previous, cfg)
            except NoExecutableControlError as error:
                decision_error = error
            for row in controller.candidate_rows:
                candidates.append({"step": k, **row})
            for candidate_id, result in enumerate(controller.last_results):
                iterations.extend({"step": k, "candidate_index": candidate_id, **r}
                                  for r in result.iteration_rows)
                prices.extend({"step": k, "candidate_index": candidate_id, **r}
                              for r in result.sensitivity_rows)
            if decision_error is not None:
                # 실패 step의 후보/반복까지 먼저 보존한다. 유효한 실행안이 없으면
                # sim.step을 호출하지 않고 exception을 재발생시켜 CLI를 nonzero로 종료한다.
                for filename, records in [("run_log.csv", rows), ("control_timeseries.csv", controls),
                                          ("state_timeseries.csv", states), ("budget_candidates.csv", candidates),
                                          ("iterations.csv", iterations), ("sensitivities.csv", prices)]:
                    write_csv(out / filename, records)
                write_json(out / "abort_state.json", asdict(sim.state))
                write_json(out / "abort.json", {
                    "status": "no_executable_control", "reason": str(decision_error),
                    "step": k, "time_sec": sim.state.time_sec, "executed_steps": len(rows),
                    "aborted_before_sim_step": True, "budget_contract_failed": True,
                    "fallback_executed": False, "computation_time_sec": time.perf_counter() - started,
                    "requested_candidates": controller.candidate_rows,
                    "previous_control": asdict(previous), "demand_forecast": [asdict(d) for d in forecast],
                })
                raise decision_error
        else:
            action = decide(IDS[name], controller, sim, forecast, previous, cfg, k)
        seconds = time.perf_counter() - started
        actuator_valid = True
        prediction_valid = True
        if name.startswith("SDMPC") and k >= args.warmup_steps:
            from src.controllers.sensitivity_dmpc import validate_control
            actuator_valid = bool(validate_control(action, previous, cfg)["valid"])
            prediction_valid = bool(controller.last_result and controller.last_result.feasible
                                    and controller.last_result.evaluation.physical_valid)
        log = sim.step(action, forecast[0], k)
        actual_np = sim.state.protected_accumulation_veh(cfg.network) - before_np
        actual_nuf = float(log.diagnostics["total_metering_flow"])
        # source queue에서 대기한 수요도 시스템 유입이다. 내부 ramp/offramp 전이는 상쇄됨.
        arrivals = cfg.simulation.T_c_h * (sum(forecast[0].freeway_mainline.values())
                                          + sum(forecast[0].urban_boundary.get(g, 0.0)
                                                for g in cfg.network.boundary_in_links)
                                          + sum(forecast[0].ramp_arrival.values()))
        exits = (cfg.simulation.T_c_h * float(log.diagnostics.get("mainline_exit_flow_total", 0.0))
                 + float(log.diagnostics.get("boundary_out_sink_veh", 0.0)))
        balance = movement_balance_summary(sim.state, cfg,
                       saturation_fraction=cfg.evaluation.boundary_degenerate_saturation_fraction,
                       degenerate_ratio=cfg.evaluation.boundary_degenerate_ratio, eps=cfg.evaluation.eps)
        row = {"step": k, "time_sec": sim.state.time_sec,
               "total_ttt": log.freeway_ttt + log.urban_ttt,
               "freeway_ttt": log.freeway_ttt, "urban_ttt": log.urban_ttt,
               "cumulative_total_ttt": sim.total_ttt, "computation_time_sec": seconds,
               **{key: value for key, value in log.diagnostics.items()
                  if isinstance(value, (int, float, bool))}, **balance,
               "actual_np_change_veh": actual_np, "actual_nuf_veh_h": actual_nuf,
               "commanded_meter_sum_veh_h": sum(action.ramp_metering.values()),
               "requested_np_change_veh": action.N_P_star,
               "requested_nuf_veh_h": action.N_UF_star,
               "raw_np_residual_veh": actual_np - action.N_P_star,
               "raw_nuf_residual_veh_h": actual_nuf - action.N_UF_star,
               "external_arrivals_veh": arrivals, "external_exits_veh": exits,
               "inventory_veh": inventory(sim.state, cfg),
               "conservation_residual_veh": inventory(sim.state, cfg) - before_inventory - arrivals + exits,
               "budget_checked": name in {"SDMPC", "SDMPC-OWN"} and k >= args.warmup_steps}
        row["executed_actuator_valid"] = actuator_valid
        row["selected_prediction_valid"] = prediction_valid
        rows.append(row)
        controls.append(control_row(action, cfg, k, sim.state.time_sec))
        sr = state_row(sim.state, cfg, k)
        sr["urban_vehicles"] = sim.state.total_urban_vehicles(cfg.network)
        sr["total_inventory_veh"] = inventory(sim.state, cfg)
        for link in cfg.network.freeway_links:
            sr[f"origin_queue_{link}"] = sim.state.mainline_origin_queue.get(link, 0.0)
            for idx, value in enumerate(sim.state.freeway_density[link]):
                sr[f"rho_{link}_seg{idx}"] = value
        states.append(sr)
        previous = action.copy()
        # 실행 중 중단되어도 후보별 원래 budget과 실패 iteration을 잃지 않는다.
        for filename, records in [("run_log.csv", rows), ("control_timeseries.csv", controls),
                                  ("state_timeseries.csv", states), ("budget_candidates.csv", candidates),
                                  ("iterations.csv", iterations), ("sensitivities.csv", prices)]:
            write_csv(out / filename, records)
        print(f"{name} step={k + 1}/{n_steps} TTT={sim.total_ttt:.3f} "
              f"budget_residual=({row['raw_np_residual_veh']:.4g},{row['raw_nuf_residual_veh_h']:.4g}) "
              f"seconds={seconds:.2f}", flush=True)
    result = {"run_rows": rows, "control_rows": controls, "state_rows": states,
              "final_state": sim.state, "total_ttt": sim.total_ttt,
              "freeway_ttt": sim.freeway_ttt, "urban_ttt": sim.urban_ttt}
    summary = summarize_run(result, cfg)
    checked = [r for r in rows if r["budget_checked"]]
    summary.update({"controller": name, "demand_sha256": demand_hash,
                    "completed_vehicles": sum(r["external_exits_veh"] for r in rows),
                    "terminal_inventory_veh": inventory(sim.state, cfg),
                    "max_conservation_residual_veh": max(abs(r["conservation_residual_veh"]) for r in rows),
                    "computation_time_sec": sum(r["computation_time_sec"] for r in rows),
                    "max_np_residual_veh": max([abs(r["raw_np_residual_veh"]) for r in checked], default=0.0),
                    "max_nuf_residual_veh_h": max([abs(r["raw_nuf_residual_veh_h"]) for r in checked], default=0.0),
                    "executed_actuator_pass": all(r["executed_actuator_valid"] for r in rows),
                    "selected_prediction_pass": all(r["selected_prediction_valid"] for r in rows),
                    "budget_contract_pass": bool(checked) and all(abs(r["raw_np_residual_veh"]) <= args.np_tolerance
                         and abs(r["raw_nuf_residual_veh_h"]) <= args.nuf_tolerance for r in checked)})
    write_json(out / "metrics_summary.json", summary)
    write_json(out / "final_state.json", asdict(sim.state))
    write_json(out / "diagnostics.json", {"summary": summary, "candidate_count": len(candidates),
               "equality_contract": "first interval actual protected stock change and actual ramp release rate",
               "scope": "two regions; constant move-blocked horizon; numerical distributed emulation"})
    (out / "report.md").write_text(f"# {name} 실행\n\nTTT={sim.total_ttt:.6f} veh*h. "
        f"공유 budget 검사={summary['budget_contract_pass']}.\n\n최종 연구 수용 판정은 비교 보고서를 확인한다.\n",
        encoding="utf-8")
    if hasattr(controller, "close"):
        controller.close()
    return result


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_saved(args):
    run_dirs = [p for p in args.output.iterdir() if p.is_dir() and (p / "metrics_summary.json").exists()]
    runs = {}
    for directory in run_dirs:
        summary = json.loads((directory / "metrics_summary.json").read_text(encoding="utf-8"))
        numeric = lambda rows: [{k: float(v) if v not in {"True", "False", ""} else
                                (float(v == "True") if v else 0.0) for k, v in r.items()} for r in rows]
        runs[directory.name] = {"run_rows": numeric(read_rows(directory / "run_log.csv")),
                                "control_rows": numeric(read_rows(directory / "control_timeseries.csv")),
                                "final_state": TrafficState(**json.loads((directory / "final_state.json").read_text(encoding="utf-8"))),
                                "total_ttt": summary["total_ttt"], "freeway_ttt": summary["freeway_ttt"],
                                "urban_ttt": summary["urban_ttt"], "summary": summary}
        runs[directory.name]["metadata"] = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        runs[directory.name]["config"] = ExperimentConfig.from_file(directory / "config_used.yaml")
    comparisons = {}
    if "NC" in runs:
        cfg = runs["NC"]["config"]
        for name, result in runs.items():
            if name == "NC":
                continue
            ev = evaluate(runs["NC"], result, cfg)
            ns, bs = result["summary"], runs["NC"]["summary"]
            boundary_ok = ns["B_in"] <= bs["B_in"] + 1e-6 and ns["B_out"] <= bs["B_out"] + 1e-6
            same_demand = ns["demand_sha256"] == bs["demand_sha256"]
            provenance_keys = ("scenario", "seed", "duration_sec", "demand_horizon_sec", "warmup_steps")
            same_provenance = (all(result["metadata"][k] == runs["NC"]["metadata"][k] for k in provenance_keys)
                               and asdict(result["config"].network) == asdict(cfg.network)
                               and asdict(result["config"].simulation) == asdict(cfg.simulation))
            if not same_provenance or not same_demand:
                raise ValueError(f"{name}/NC scenario, seed, demand, horizon or plant differs")
            comparisons[name] = {"improvement_pct": ev.improvement_pct,
                "legacy_evaluation_pass": ev.passed, "legacy_control_validation": ev.control_validation,
                "same_demand": same_demand, "same_provenance": same_provenance, "boundary_not_degraded": boundary_ok,
                "exact_budget_pass": ns["budget_contract_pass"] if name.startswith("SDMPC") else None,
                "simulation_acceptance_pass": bool(ev.passed and boundary_ok and same_demand and same_provenance
                     and ns.get("executed_actuator_pass", True) and ns.get("selected_prediction_pass", True)
                     and ns["max_conservation_residual_veh"] <= 1e-5
                     and (ns["budget_contract_pass"] if name.startswith("SDMPC") else True)),
                "full_acceptance_pass": False,
                "claude_review_current": False}
    write_json(args.output / "comparison.json", comparisons)
    table = [{"controller": name, **r["summary"], **{k: v for k, v in comparisons.get(name, {}).items()
                                                    if not isinstance(v, dict)}} for name, r in runs.items()]
    write_csv(args.output / "comparison.csv", table)
    plot_saved(args.output, runs)
    return comparisons


def plot_saved(output, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 125, "savefig.dpi": 160})
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    specs = [("cumulative_total_ttt", "Cumulative TTT [veh h]"),
             ("inventory_veh", "Total inventory [veh]"),
             ("actual_nuf_veh_h", "Actual ramp release [veh/h]"),
             ("computation_time_sec", "Decision time [s]")]
    for ax, (key, label) in zip(axes.flat, specs):
        for name, run in runs.items():
            rows = run["run_rows"]
            ax.plot([float(r["time_sec"]) / 60 for r in rows], [float(r[key]) for r in rows], label=name)
        ax.set(xlabel="Time [min]", ylabel=label)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(plots / "traffic_comparison.png")
    plt.close(fig)
    for name, run in runs.items():
        if not name.startswith("SDMPC"):
            continue
        controls = run["control_rows"]
        states = read_rows(output / name / "state_timeseries.csv")
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
        families = [(controls, "ramp_metering_", "Ramp command [veh/h]"),
                    (controls, "vsl_FW_", "VSL [km/h]"),
                    (controls, "green_", "Green [s]"),
                    (controls, "offset_", "Offset [s]"),
                    (states, "rho_FW_", "Density [veh/km/lane]"),
                    (controls, "allocation_", "Allocation cap [veh/h]")]
        for ax, (rows, prefix, label) in zip(axes.flat, families):
            keys = [k for k in rows[0] if k.startswith(prefix)] if rows else []
            if prefix == "rho_FW_":
                keys = [k for k in keys if k.endswith("_mean")]
            if prefix == "allocation_":
                # legacy link 합계의 '미지정=0'을 실제 saturation cap로 그리지 않는다.
                net = run["config"].network
                active_names = {*net.urban_movements, *net.boundary_in_links}
                keys = [k for k in keys if k[len(prefix):] in active_names]
            # 동일한 제어열은 한 번 그려 legend가 실험 신호를 가리지 않도록 한다.
            seen = set()
            for key in keys:
                values = tuple(float(r[key]) for r in rows)
                if values in seen:
                    continue
                seen.add(values)
                ax.plot([float(r["time_sec"]) / 60 for r in rows], values, label=key[len(prefix):])
            ax.set(xlabel="Time [min]", ylabel=label)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            if prefix == "ramp_metering_":
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(fontsize=6, ncol=2, loc="best")
        fig.suptitle(name + " | Actuators and traffic states")
        fig.savefig(plots / f"{name}_controls_states.png")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, prefix, label in zip(axes.flat,
                ("speed_FW_", "ramp_queue_", "origin_queue_", "boundary_queue_in_"),
                ("Mean speed [km/h]", "Ramp reservoir [veh]", "Freeway origin queue [veh]", "Entry gate queue [veh]")):
            keys = [k for k in states[0] if k.startswith(prefix)]
            for key in keys:
                ax.plot([float(r["time_sec"]) / 60 for r in states],
                        [float(r[key]) for r in states], label=key[len(prefix):])
            ax.set(xlabel="Time [min]", ylabel=label)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7, ncol=2)
        fig.savefig(plots / f"{name}_queues_speed.png")
        plt.close(fig)
        trace_path = output / name / "iterations.csv"
        if trace_path.exists():
            saved_options = json.loads((output / name / "solver_options.json").read_text(encoding="utf-8"))
            trace_rows = read_rows(trace_path)
            steps = sorted({int(r["step"]) for r in trace_rows})
            representative = {steps[0], steps[len(steps) // 2], steps[-1]} if steps else set()
            plot_convergence([dict(r, case=f"step {r['step']} / budget {r['candidate_index']}")
                              for r in trace_rows if int(r["step"]) in representative],
                             plots / f"{name}_iterations.png", saved_options["tolerance_np_veh"],
                             saved_options["tolerance_nuf_veh_h"])


def plot_convergence(rows, path, np_tolerance, nuf_tolerance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    families = [("objective_ttt", "True TTT [veh h]", None),
                ("residual_np_veh", "|Original N_P residual| [veh]", np_tolerance),
                ("residual_nuf_veh_h", "|Original N_UF residual| [veh/h]", nuf_tolerance),
                ("stationarity_norm", "Gradient mapping [veh h / normalized control]", None)]
    for ax, (key, label, tolerance) in zip(axes.flat, families):
        for case in dict.fromkeys(r["case"] for r in rows):
            selected = [r for r in rows if r["case"] == case]
            values = [float(r[key]) for r in selected]
            if key != "objective_ttt":
                values = [max(abs(x), 1e-10) if np.isfinite(x) else np.nan for x in values]
            ax.plot([int(r["iteration"]) for r in selected], values, marker=".", label=case)
        if key != "objective_ttt":
            ax.set_yscale("log")
        if tolerance is not None:
            ax.axhline(tolerance, color="black", ls="--", lw=.8)
        ax.set(xlabel="Coordination iteration", ylabel=label)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=6)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fixed_validation(args):
    from src.controllers.sensitivity_dmpc import Budget, SensitivityDMPC
    cfg, _, profile = config_for(args)
    out = args.output / "fixed_validation"
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner_source.py.txt").write_bytes(RUNNER_SOURCE)
    (out / "core_source.py.txt").write_bytes((ROOT / "src/controllers/sensitivity_dmpc.py").read_bytes())
    sim = MixedTrafficSimulator(cfg)
    previous = ControlAction.uncontrolled(cfg)
    for k in range(args.snapshot_steps):
        sim.step(previous, profile.at(sim.state.time_sec), k)
    # feasible witness는 budget 생성 이전에 명시적 제어안으로 마련한다.
    seed_control = previous.copy()
    for ramp in cfg.network.ramps:
        seed_control.ramp_metering[ramp] = .6 * cfg.network.ramp_capacity_veh_h[ramp]
    controller = SensitivityDMPC(cfg, options_for(args))
    forecast = profile.horizon(sim.state.time_sec, args.horizon)
    witness = controller.evaluate_control(sim.state, forecast, seed_control)
    write_json(out / "seed_control.json", asdict(seed_control))
    write_json(out / "forecast.json", [asdict(d) for d in forecast])
    targets = [
        ("witness", Budget(witness.achieved_np_change_veh, witness.achieved_nuf_veh_h)),
        ("nearby", Budget(witness.achieved_np_change_veh + 5.0, max(0.0, witness.achieved_nuf_veh_h - 100.0))),
        ("capacity_impossible", Budget(0.0, cfg.network.total_ramp_capacity + 1.0)),
    ]
    rows, traces, price_rows = [], [], []
    for label, budget in targets:
        t0 = time.perf_counter()
        result = controller.solve_fixed_budget(sim.state, forecast, previous, budget, initial_control=seed_control)
        write_json(out / f"{label}_control.json", asdict(result.control))
        rows.append({"case": label, **asdict(budget), "objective": result.objective,
                     "witness_objective": witness.total_ttt, "status": result.status,
                     "feasible": result.feasible, "converged": result.converged,
                     "residual_np_veh": result.residual_np_veh,
                     "residual_nuf_veh_h": result.residual_nuf_veh_h,
                     "iterations": result.iterations, "computation_time_sec": time.perf_counter() - t0})
        traces.extend({"case": label, **r} for r in result.iteration_rows)
        price_rows.extend({"case": label, **r} for r in result.sensitivity_rows)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        write_csv(out / "cases.csv", rows)
        write_csv(out / "iterations.csv", traces)
        write_csv(out / "sensitivities.csv", price_rows)
    write_json(out / "config_used.yaml", cfg.to_dict())
    write_json(out / "solver_options.json", asdict(controller.options))
    write_json(out / "snapshot_state.json", asdict(sim.state))
    write_json(out / "results.json", rows)
    plot_convergence(traces, out / "convergence.png", args.np_tolerance, args.nuf_tolerance)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "closed-loop", "plot"], default="closed-loop")
    parser.add_argument("--controllers", default="NC,PFO,SDMPC")
    parser.add_argument("--scenario", default="sweet_170_w")
    parser.add_argument("--duration", type=float, default=3600.)
    parser.add_argument("--demand-horizon", type=float, default=7200.)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--trust-radius", type=float, default=.15)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--warm-start", choices=["previous", "pfo"], default="previous")
    parser.add_argument("--np-tolerance", type=float, default=.1)
    parser.add_argument("--nuf-tolerance", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--snapshot-steps", type=int, default=4)
    parser.add_argument("--own-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or abs(args.duration / 180 - round(args.duration / 180)) > 1e-9:
        parser.error("duration은 현재 plant control interval 180초의 양의 배수여야 함")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "fixed":
        fixed_validation(args)
    elif args.mode == "closed-loop":
        for name in args.controllers.split(","):
            if name not in {*IDS, "SDMPC", "SDMPC-OWN"}:
                parser.error(f"unknown controller {name}")
            run_controller(name, args)
        compare_saved(args)
    else:
        compare_saved(args)


if __name__ == "__main__":
    main()
