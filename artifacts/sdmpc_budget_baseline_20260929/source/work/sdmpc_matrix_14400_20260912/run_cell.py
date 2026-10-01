"""고정 historical runtime/profile로 실행하는 14400초 matrix 단일 worker.

Spec 05/10/12와 worker_brief.md: plant/controller/factory 변경 없이 입력·gate를 연결한다.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys
import subprocess
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYER_OPTIONS = "outputs/sdmpc_speed_20260912/attempt_0/PLAYER_SDMPC/solver_options.json"
FU_OPTIONS = "outputs/sdmpc_20260911/attempt_1/SDMPC/solver_options.json"
CONTRACTS = {
    "PLAYER_SDMPC": "forecast-plan H-step actual net-service cap [veh]; command-sum equality [veh/h]",
    "SDMPC": "first interval protected stock change equality [veh]; actual ramp release equality [veh/h]",
    "NC": "no shared budget contract",
    "CANONICAL": "historical canonical Stackelberg contract; not an SDMPC budget",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(args):
    """src를 불러오기 전에 historical root와 명시된 환경을 선택한다."""
    protocol = read_json(args.protocol_dir / "protocol.json")
    root = Path(args.runtime_root or protocol["runtime_root"]).resolve()
    if not (root / "src" / "models" / "state.py").is_file():
        raise ValueError("Invalid historical runtime root")
    frozen_env = protocol["frozen_environment"]
    for name, value in frozen_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    sys.path[:0] = [str(root), str(REPO_ROOT), str(Path(__file__).resolve().parent)]
    return root, frozen_env


def load_runtime(root):
    """모든 src와 factory가 선택한 tree에서 왔는지 검사한다."""
    global np, inventory, write_csv, write_json, verify_validation_provenance, ValidationProvenanceError
    global PlayerBudget, PlayerSDMPCOptions, PlayerSensitivityDMPC, validate_player_control
    global Budget, SDMPCOptions, SensitivityDMPC, validate_control, summarize_run
    global DemandStep, ControlAction, ExperimentConfig, movement_balance_summary
    global MixedTrafficSimulator, control_row, state_row, make_controller, decide
    global restore_historical_config, to_plain_dict, object_sha256
    import numpy as np
    from scripts.run_sdmpc_experiment import write_csv, write_json
    from scripts.player_sdmpc_validation_provenance import ValidationProvenanceError, verify_validation_provenance
    from src.controllers.player_sensitivity_dmpc import (
        PlayerBudget, PlayerSDMPCOptions, PlayerSensitivityDMPC, validate_player_control,
    )
    from src.controllers.sensitivity_dmpc import Budget, SDMPCOptions, SensitivityDMPC, validate_control, physical_inventory as inventory
    from src.evaluation.metrics import summarize_run
    from src.models.demand import DemandStep
    from src.models.state import ControlAction, ExperimentConfig
    from src.models.urban_queue_model import movement_balance_summary
    from src.simulation.simulator import MixedTrafficSimulator, control_row, state_row
    from work.run_claude_style_five_controller import make_controller, decide
    from historical_config import restore_historical_config, to_plain_dict, object_sha256
    paths = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
             if getattr(module, "__file__", None) and (name == "src" or name.startswith("src.")
                 or name == "work.run_claude_style_five_controller")}
    if any(not Path(path).is_relative_to(root) for path in paths.values()):
        raise ValueError(f"Mixed current/historical src or factory imports: {paths}")
    return paths


class FrozenProfile:
    """저장 table의 정확한 시점만 조회하며 외삽·수요 재생성을 금지한다."""

    def __init__(self, rows, dt):
        self.dt, self.table = dt, {}
        for row in rows:
            value = copy.deepcopy(row)
            timestamp = float(value.pop("time_sec"))
            if not math.isfinite(timestamp) or timestamp in self.table:
                raise ValueError("Invalid or duplicate frozen forecast time")
            value["freeway_lane_loss"] = {
                link: {int(segment): loss for segment, loss in losses.items()}
                for link, losses in value.get("freeway_lane_loss", {}).items()}
            self.table[timestamp] = DemandStep(**value)
        if sorted(self.table) != [index * dt for index in range(len(self.table))]:
            raise ValueError("Frozen forecast is not contiguous from zero")

    def at(self, timestamp):
        if float(timestamp) not in self.table:
            raise ValueError(f"Frozen forecast has no time {timestamp}; extrapolation forbidden")
        return copy.deepcopy(self.table[float(timestamp)])

    def horizon(self, timestamp, depth):
        return [self.at(timestamp + index * self.dt) for index in range(depth)]


class FrozenInputs:
    """protocol·source manifest를 실행 전/후 byte hash로 고정한다."""

    def __init__(self, directory, out, root, frozen_env, imported):
        self.directory, self.out, self.root = directory.resolve(), out, root
        self.frozen_env = frozen_env
        self.inputs = {}
        snapshot = out / "protocol_snapshot"
        snapshot.mkdir()
        for name in ("config.json", "protocol.json", "forecast.json", "source_manifest.json",
                     "scenario.json", "initial_state.json", "historical_provenance.json"):
            path = self.directory / name
            raw = path.read_bytes()
            (snapshot / name).write_bytes(raw)
            self.inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        self.protocol = read_json(snapshot / "protocol.json")
        required_inputs = {"config.json", "forecast.json", "scenario.json", "initial_state.json", "historical_provenance.json"}
        if set(self.protocol["input_sha256"]) != required_inputs:
            raise ValueError("Protocol declared input manifest differs from required frozen inputs")
        for name, expected in self.protocol["input_sha256"].items():
            if sha(snapshot / name) != expected:
                raise ValueError(f"Protocol declared input hash differs: {name}")
        self.base_config = read_json(snapshot / "config.json")
        self.cfg = restore_historical_config(self.base_config)
        if to_plain_dict(self.cfg) != self.base_config:
            raise ValueError("Protocol config changed on restoration")
        self.initial_state = read_json(snapshot / "initial_state.json")
        self.scenario = read_json(snapshot / "scenario.json")
        provenance = read_json(snapshot / "historical_provenance.json")
        if (provenance["effective_config_sha256"] != object_sha256(self.cfg)
                or provenance["initial_state_sha256"] != object_sha256(self.initial_state)):
            raise ValueError("Historical provenance differs from effective config/initial state")
        p, cfg = self.protocol, self.cfg
        if self.scenario["name"] != p["scenario"] or provenance["scenario_id"] != p["scenario"]:
            raise ValueError("Frozen scenario metadata differs")
        if (p["duration_sec"] != 14400 or p["warmup_steps"] != 5 or p["expected_steps"] != 80
                or p["demand_horizon_sec"] != 14400 or p["seed"] != 42
                or cfg.simulation.T_total != p["duration_sec"] or cfg.simulation.random_seed != p["seed"]
                or cfg.mpc.horizon_steps != 3 or cfg.simulation.T_c_sec * p["expected_steps"] != p["duration_sec"]):
            raise ValueError("Protocol/config differs from bounded matrix contract")
        manifest = read_json(snapshot / "source_manifest.json")
        self.sources = manifest["sources"]
        if Path(manifest["source_root"]).resolve() != root:
            raise ValueError("Manifest source_root differs from selected runtime")
        source_files = {}
        for index, (relative, expected) in enumerate(self.sources.items()):
            path = (root / relative).resolve()
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Frozen source differs: {relative}")
            target = out / "source_snapshot" / (f"external_{index}_{path.name}" if Path(relative).is_absolute() else relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            source_files[relative] = str(target.relative_to(out))
        # 실제 import한 source와 options/worker 모두 manifest로 연결되어야 한다.
        required_paths = [Path(path) for path in imported.values()] + [REPO_ROOT / PLAYER_OPTIONS,
            REPO_ROOT / FU_OPTIONS, Path(__file__).resolve()]
        covered = {(root / path).resolve() for path in self.sources}
        missing = [str(path) for path in required_paths if path not in covered]
        if missing:
            raise ValueError(f"Source manifest missing imported/input files: {missing}")
        self.profile = FrozenProfile(read_json(snapshot / "forecast.json"), cfg.simulation.T_c_sec)
        if max(self.profile.table) < (p["expected_steps"] + 5) * cfg.simulation.T_c_sec:
            raise ValueError("Frozen forecast does not cover canonical terminal horizon")
        write_json(out / "environment.json", {"frozen_environment": frozen_env,
            "runtime_root": str(root), "imported_modules": imported,
            "runtime": {name: os.environ.get(name) for name in
                ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")}})
        write_json(out / "source_hashes.json", self.sources)
        write_json(out / "source_snapshot_manifest.json", source_files)
        self.simulator()
        self.verify()

    def simulator(self, cfg=None):
        """생성한 초기 상태가 저장 state와 정확히 같을 때만 실행한다."""
        sim = MixedTrafficSimulator(cfg or self.cfg)
        if to_plain_dict(sim.state) != self.initial_state:
            raise ValueError("Generated simulator initial state differs from frozen initial_state.json")
        return sim

    def track(self, path):
        path = Path(path).resolve()
        self.inputs[str(path)] = sha(path)

    def verify(self):
        changed = [path for path, expected in self.inputs.items() if sha(path) != expected]
        changed += [relative for relative, expected in self.sources.items() if sha(self.root / relative) != expected]
        changed += [f"environment:{name}" for name, value in self.frozen_env.items()
                    if os.environ.get(name) != (None if value is None else str(value))]
        report = {"pass": not changed, "changed": changed, "input_hashes": self.inputs, "source_hashes": self.sources}
        write_json(self.out / "integrity.json", report)
        if changed:
            raise ValueError(f"Frozen source/input changed: {changed}")


def saved_options(name, fixed=False):
    """보존 options 전체를 복원하며 fixed outer iteration 5만 명시적으로 바꾼다."""
    options = (PlayerSDMPCOptions(**read_json(REPO_ROOT / PLAYER_OPTIONS)) if name == "PLAYER_SDMPC"
               else SDMPCOptions(**read_json(REPO_ROOT / FU_OPTIONS)))
    return replace(options, max_iterations=5) if fixed else options


def player_startup(args, frozen, options):
    """기존 player gate를 유지하고 진단 허용 두 항목만 별도 기록한다."""
    out, p, failures = frozen.out, frozen.protocol, []
    try:
        if args.fixed_validation is None or args.sensitivity_validation is None:
            raise ValueError("PLAYER requires --fixed-validation and --sensitivity-validation")
        fixed, evidence = read_json(args.fixed_validation), read_json(args.sensitivity_validation)
        for directory in (args.fixed_validation.parent, args.sensitivity_validation.parent):
            for path in directory.iterdir():
                if path.is_file():
                    frozen.track(path)
        for key, value in (("scenario", p["scenario"]), ("seed", p["seed"]),
                ("horizon_steps", options.horizon_steps), ("demand_horizon_sec", p["demand_horizon_sec"])):
            if fixed[key] != value or evidence[key] != value:
                raise ValueError(f"Fixed/sensitivity conditions differ: {key}")
        if fixed["source_hashes"] != frozen.sources:
            raise ValueError("Fixed validation does not cover complete frozen source manifest")
        if any(evidence["source_hashes"].get(path) != expected for path, expected in frozen.sources.items()):
            raise ValueError("Sensitivity source differs from complete frozen manifest")
        if not (fixed["fixed_feasibility_pass"] and fixed["impossibility_classification_pass"] and fixed["cost_partition_pass"]):
            raise ValueError("Fixed-budget feasibility/accounting/classification gate did not pass")
        if not fixed["fixed_optimization_converged"]:
            failures.append("fixed_optimizer_not_converged")
        if not (abs(evidence["player_cost_sum_residual"]) <= 1e-8 and evidence["physical_price_scaling_pass"]
                and evidence["input_immutable"] and evidence["all_exchange_command_equality"]
                and evidence["direct_exchange_feasible_count"] > 0):
            raise ValueError("Independent accounting/price/resource-exchange validation did not pass")
        if not evidence["all_independent_directional_agreement"]:
            failures.append("independent_directional_derivative_mismatch")
        if failures and not args.diagnostic_after_fixed_failure:
            raise ValueError(f"Explicit --diagnostic-after-fixed-failure required: {failures}")
        provenance = verify_validation_provenance(args.fixed_validation, args.sensitivity_validation, to_plain_dict(frozen.cfg), options)
        write_json(out / "validation_provenance.json", provenance)
        write_json(out / "fixed_validation_used.json", fixed)
        write_json(out / "sensitivity_validation_used.json", evidence)
        frozen.verify()
    except Exception as exc:
        if isinstance(exc, ValidationProvenanceError):
            write_json(out / "validation_provenance.json", exc.report)
        write_json(out / "startup_gate.json", {"pass": False, "reason": str(exc), "retained_failures": failures})
        raise
    write_json(out / "startup_gate.json", {"pass": True, "retained_failures": failures,
        "diagnostic_after_fixed_failure": bool(failures), "derivative_pass_is_not_overridden": True})
    return failures


def fixed_validation(frozen):
    """동일 profile의 NC 5구간 뒤 세 원 budget를 그대로 검증한다."""
    out, cfg, p = frozen.out, frozen.cfg, frozen.protocol
    options = saved_options("PLAYER_SDMPC", fixed=True)
    solver = PlayerSensitivityDMPC(cfg, options)
    sim = frozen.simulator(cfg)
    previous = ControlAction.uncontrolled(cfg)
    for step in range(p["warmup_steps"]):
        sim.step(previous, frozen.profile.at(sim.state.time_sec), step)
        print(f"FIXED NC warmup {step+1}/{p['warmup_steps']} time={sim.state.time_sec}", flush=True)
    seed = previous.copy()
    for ramp in cfg.network.ramps:
        seed.ramp_metering[ramp] = .6 * cfg.network.ramp_capacity_veh_h[ramp]
    forecast = frozen.profile.horizon(sim.state.time_sec, options.horizon_steps)
    for name, value in (("config_used.yaml", to_plain_dict(cfg)), ("solver_options.json", asdict(options)),
            ("snapshot_state.json", asdict(sim.state)), ("seed_control.json", asdict(seed)),
            ("previous_control.json", asdict(previous)), ("forecast.json", [asdict(d) for d in forecast])):
        write_json(out / name, value)
    write_json(out / "metadata.json", {**p, "runtime_root": str(frozen.root),
        "mode": "fixed", "source_hashes": frozen.sources, "snapshot_time_sec": sim.state.time_sec})
    write_json(out / "demand.json", [{"time_sec": timestamp, **asdict(frozen.profile.at(timestamp))}
        for timestamp in sorted(frozen.profile.table)])
    witness = solver.evaluate_control(sim.state, forecast, seed, previous)
    targets = [("witness", PlayerBudget(witness.achieved_np_veh, witness.commanded_nuf_veh_h)),
        ("nearby", PlayerBudget(witness.achieved_np_veh + 5., witness.commanded_nuf_veh_h - 100.)),
        ("capacity_impossible", PlayerBudget(0., cfg.network.total_ramp_capacity + 1.))]
    cases, iterations, sensitivities, locals_ = [], [], [], []
    for name, budget in targets:
        started = time.perf_counter()
        write_json(out / "active_case.json", {"case": name, "budget": asdict(budget)})
        result = solver.solve_fixed_budget(sim.state, forecast, previous, budget, seed)
        row = {"case": name, "requested_np_cap_veh": budget.np_cap_veh,
            "requested_nuf_veh_h": budget.nuf_veh_h, "objective_ttt": result.objective,
            "initial_objective_ttt": witness.total_ttt, "feasible": result.feasible,
            "converged": result.converged, "status": result.status,
            "achieved_np_veh": result.evaluation.achieved_np_veh,
            "commanded_nuf_veh_h": result.evaluation.commanded_nuf_veh_h,
            "actual_first_nuf_veh_h": result.evaluation.actual_nuf_veh_h,
            "residual_np_veh": result.residual_np_veh, "residual_nuf_veh_h": result.residual_nuf_veh_h,
            "iterations": result.iterations, "termination_reason": result.termination_reason,
            "computation_time_sec": time.perf_counter() - started,
            "player_cost_sum_residual": sum(result.evaluation.player_costs.values()) - result.objective}
        cases.append(row)
        iterations.extend({"case": name, **r} for r in result.iteration_rows)
        sensitivities.extend({"case": name, **r} for r in result.sensitivity_rows)
        locals_.extend({"case": name, **r} for r in result.local_rows)
        write_json(out / f"{name}_control.json", asdict(result.control))
        write_json(out / f"{name}_costs.json", result.evaluation.player_costs)
        for filename, values in (("cases.csv", cases), ("iterations.csv", iterations),
                ("sensitivities.csv", sensitivities), ("local_problems.csv", locals_)):
            write_csv(out / filename, values)
        frozen.verify()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    gate = {"fixed_feasibility_pass": all(r["feasible"] for r in cases[:-1]),
        "fixed_optimization_converged": all(r["converged"] for r in cases[:-1]),
        "impossibility_classification_pass": cases[-1]["status"] == "infeasible_proven",
        "cost_partition_pass": max(abs(r["player_cost_sum_residual"]) for r in cases) <= 1e-8,
        "cases": cases, "source_hashes": frozen.sources, "scenario": p["scenario"], "seed": p["seed"],
        "runtime_root": str(frozen.root),
        "duration_sec": p["duration_sec"], "horizon_steps": options.horizon_steps,
        "demand_horizon_sec": p["demand_horizon_sec"], "snapshot_time_sec": sim.state.time_sec,
        "warmup_steps": p["warmup_steps"], "contract": CONTRACTS["PLAYER_SDMPC"]}
    frozen.verify()
    write_json(out / "validation_gate.json", gate)
    write_json(out / "diagnostics.json", gate)
    (out / "report.md").write_text(f"# PLAYER 고정 검증\n\nNC 5구간 뒤 900초 snapshot의 원 budget 3건. "
        f"feasibility={gate['fixed_feasibility_pass']}, optimization_converged={gate['fixed_optimization_converged']}. "
        "수렴 실패는 닫힌루프 진단 허용과 별도로 보존한다.\n", encoding="utf-8")
    return gate


def run_native_canonical(args, frozen):
    """원 historical CLI 전체(supervisor/latch 포함)를 실행하고 원 checkpoint를 변환한다."""
    out, cfg, p = frozen.out, frozen.cfg, frozen.protocol
    canonical_id = p["canonical_controller_id"]
    native_root = out / "native_raw"
    native_root.mkdir()
    # 원 CLI가 생성할 profile와 frozen 표의 모든 horizon entry가 같은지 독립 확인한다.
    from src.models.demand import DemandProfile, load_scenarios
    scenario = load_scenarios(frozen.root / "src/config/scenarios.yaml")[p["scenario"]]
    if to_plain_dict(scenario) != frozen.scenario:
        raise ValueError("Native scenario differs from frozen scenario.json")
    native_profile = DemandProfile(cfg, scenario)
    for timestamp in frozen.profile.table:
        if asdict(native_profile.at(timestamp)) != asdict(frozen.profile.at(timestamp)):
            raise ValueError(f"Native demand differs from frozen profile at {timestamp}")
    write_json(out / "native_profile_validation.json", {"pass": True,
        "entries": len(frozen.profile.table), "scope": "same historical DemandProfile/config/scenario as native run_one"})
    factory_cfg = copy.deepcopy(cfg)
    factory = make_controller(canonical_id, factory_cfg)
    if hasattr(factory, "close"):
        factory.close()
    write_json(out / "factory_config_used.json", to_plain_dict(factory_cfg))
    write_json(out / "config_used.yaml", frozen.base_config)
    if any(to_plain_dict(factory_cfg)[key] != frozen.base_config[key] for key in ("network", "simulation")):
        raise ValueError("Native factory changes frozen physical config")
    environment = os.environ.copy()
    # 부모에서 적용한 고정 historical 환경을 원 subprocess에도 정확히 넘긴다.
    environment["PYTHONPATH"] = os.pathsep.join([str(frozen.root),
        str(REPO_ROOT / "work/sdmpc_20260911/deps")])
    command = [sys.executable, "-B", "-u", str(frozen.root / "work/run_claude_style_five_controller.py"),
        "--scenario", p["scenario"], "--T-total", str(p["duration_sec"]),
        "--controllers", canonical_id, "--output", str(native_root.resolve())]
    write_json(out / "native_invocation.json", {"command": command, "cwd": str(frozen.root),
        "frozen_environment": p["frozen_environment"], "runtime_root": str(frozen.root),
        "native_runner_unmodified": True, "factory_config_scope": "independently reconstructed initial factory config"})
    frozen.verify()
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=frozen.root, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        with (out / "native.log").open("w", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if "step " in line and "cum_ttt=" in line:
                    print(line.rstrip(), flush=True)
                    frozen.verify()
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    write_json(out / "native_process_result.json", {"returncode": code,
        "wall_time_sec": time.perf_counter() - started})
    if code:
        raise RuntimeError(f"Native canonical runner failed with exit code {code}; raw checkpoints preserved")
    frozen.verify()
    return convert_native_canonical(frozen, native_root / canonical_id)


def convert_native_canonical(frozen, native_dir):
    """동일 실행이 만든 trusted local pickle checkpoint만 읽어 표준 로그로 변환한다."""
    out, cfg, p = frozen.out, frozen.cfg, frozen.protocol
    checkpoints = []
    with (native_dir / "checkpoints.pkl").open("rb") as stream:
        while True:
            try:
                checkpoints.append(pickle.load(stream))
            except EOFError:
                break
    with (native_dir / "run_log.csv").open(encoding="utf-8", newline="") as stream:
        original_rows = [{key: (value == "True" if value in ("True", "False") else float(value))
                          for key, value in row.items() if value != ""} for row in csv.DictReader(stream)]
    if (len(checkpoints) != 80 or len(original_rows) != 80
            or [item["step"] for item in checkpoints] != list(range(80))
            or [row["step"] for row in original_rows] != list(range(80))
            or any(item["time_sec"] != (k+1)*cfg.simulation.T_c_sec for k,item in enumerate(checkpoints))
            or checkpoints[-1]["state"].time_sec != 14400):
        raise ValueError("Native output lacks exact 80 contiguous post-step checkpoints/rows")
    initial = frozen.simulator(cfg).state
    previous_inventory = inventory(initial, cfg)
    rows, controls, states = [], [], []
    for k, (snapshot, original) in enumerate(zip(checkpoints, original_rows)):
        state, action = snapshot["state"], snapshot["previous_control"]
        if original["time_sec"] != state.time_sec:
            raise ValueError("Native row/checkpoint time differs")
        demand = frozen.profile.at(k * cfg.simulation.T_c_sec)
        arrivals = cfg.simulation.T_c_h * (sum(demand.freeway_mainline.values())
            + sum(demand.urban_boundary.get(g, 0.) for g in cfg.network.boundary_in_links)
            + sum(demand.ramp_arrival.values()))
        exits = cfg.simulation.T_c_h * original["mainline_exit_flow_total"] + original["boundary_out_sink_veh"]
        current_inventory = inventory(state, cfg)
        balance = movement_balance_summary(state, cfg,
            saturation_fraction=cfg.evaluation.boundary_degenerate_saturation_fraction,
            degenerate_ratio=cfg.evaluation.boundary_degenerate_ratio, eps=cfg.evaluation.eps)
        rows.append({**original, **balance, "step": k, "total_ttt": original["step_total_ttt"],
            "freeway_ttt": original["step_freeway_ttt"], "urban_ttt": original["step_urban_ttt"],
            "inventory_veh": current_inventory, "external_arrivals_veh": arrivals, "external_exits_veh": exits,
            "conservation_residual_veh": current_inventory-previous_inventory-arrivals+exits,
            "budget_checked": False, "warmup": k < 5,
            "actual_nuf_veh_h": original["total_metering_flow"],
            "commanded_meter_sum_veh_h": sum(action.ramp_metering.values())})
        controls.append(control_row(action, cfg, k, state.time_sec))
        sr = state_row(state, cfg, k)
        sr.update(urban_vehicles=state.total_urban_vehicles(cfg.network), total_inventory_veh=current_inventory)
        for link in cfg.network.freeway_links:
            sr[f"origin_queue_{link}"] = state.mainline_origin_queue.get(link, 0.)
            for seg, density in enumerate(state.freeway_density[link]):
                sr[f"rho_{link}_seg{seg}"] = density
        states.append(sr)
        previous_inventory = current_inventory
    final = checkpoints[-1]
    # total_ttt는 simulator property여서 원 sim_num pickle에는 두 component만 들어간다.
    totals = {key: final["sim_num"][key] for key in ("freeway_ttt", "urban_ttt")}
    totals["total_ttt"] = totals["freeway_ttt"] + totals["urban_ttt"]
    for key in ("total_ttt", "freeway_ttt", "urban_ttt"):
        if not math.isclose(totals[key], sum(row[key] for row in rows), rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError(f"Native checkpoint and row TTT differ: {key}")
    demand_rows = [{"step": k, "time_sec": k*cfg.simulation.T_c_sec,
        **asdict(frozen.profile.at(k*cfg.simulation.T_c_sec))} for k in range(80)]
    demand_hash = hashlib.sha256(json.dumps(demand_rows, sort_keys=True).encode()).hexdigest()
    result = {"run_rows": rows, "control_rows": controls, "state_rows": states,
        "final_state": final["state"], **totals}
    summary = summarize_run(result, cfg)
    summary.update(controller="CANONICAL", demand_sha256=demand_hash, executed_steps=80,
        controlled_steps=75, warmup_steps=5, budget_checked_steps=0,
        post_warmup_total_ttt=sum(r["total_ttt"] for r in rows[5:]), warmup_total_ttt=sum(r["total_ttt"] for r in rows[:5]),
        completed_vehicles=sum(r["external_exits_veh"] for r in rows), terminal_inventory_veh=previous_inventory,
        max_conservation_residual_veh=max(abs(r["conservation_residual_veh"]) for r in rows),
        computation_time_sec=sum(r["computation_time_sec"] for r in rows), full_acceptance_pass=False)
    for filename, values in (("run_log.csv", rows), ("control_timeseries.csv", controls),
            ("state_timeseries.csv", states), ("budget_candidates.csv", []), ("iterations.csv", []),
            ("sensitivities.csv", []), ("local_problems.csv", [])):
        write_csv(out / filename, values)
    write_json(out / "demand.json", demand_rows)
    write_json(out / "metadata.json", {**p, "controller": "CANONICAL", "demand_sha256": demand_hash,
        "source_hashes": frozen.sources, "initial_inventory_veh": inventory(initial,cfg),
        "native_original_runner": True, "native_raw": str(native_dir), "full_acceptance_pass": False,
        "partition": "historical PFO_SPLIT=2: four freeway half-link regions plus five urban regions (9); differs from proposed 7 players",
        "budget_contract": CONTRACTS["CANONICAL"]})
    write_json(out / "metrics_summary.json", summary)
    write_json(out / "final_state.json", asdict(final["state"]))
    write_json(out / "diagnostics.json", {"summary": summary, "native_original_runner": True,
        "contract": CONTRACTS["CANONICAL"], "source_hashes": frozen.sources})
    (out / "report.md").write_text(f"# CANONICAL 원본 실행\n\n과거 CLI 전체의 80구간/14400초 실행을 변환했다. "
        f"TTT={summary['total_ttt']:.6f} veh*h. supervisor/latch 원본 실행이며 연구 acceptance는 별도다.\n", encoding="utf-8")
    frozen.verify()
    write_json(out / "completion.json", {"simulation_complete": True, "executed_steps": 80,
        "final_time_sec": 14400, "full_acceptance_pass": False, "source_input_integrity_pass": True})
    return summary


def run_controller(args, frozen):
    out, cfg, p = frozen.out, frozen.cfg, frozen.protocol
    name = args.controller
    if name == "CANONICAL":
        return run_native_canonical(args, frozen)
    options = saved_options(name) if name in {"SDMPC", "PLAYER_SDMPC"} else None
    failures = player_startup(args, frozen, options) if name == "PLAYER_SDMPC" else []
    # Factory의 leader depth 변경은 보존하되 plant/simulation 값 변경은 중단한다.
    if name in {"SDMPC", "PLAYER_SDMPC"}:
        controller = (PlayerSensitivityDMPC(cfg, options) if name == "PLAYER_SDMPC" else SensitivityDMPC(cfg, options))
        warm = make_controller("WU-FAITHFUL-FOLLOWER", cfg)
    else:
        controller, warm = None, None
    actual_config = to_plain_dict(cfg)
    write_json(out / "config_used.yaml", frozen.base_config)
    write_json(out / "factory_config_used.json", actual_config)
    if any(actual_config[key] != frozen.base_config[key] for key in ("network", "simulation")):
        raise ValueError("Factory changed frozen physical network/simulation")
    if options:
        write_json(out / "solver_options.json", asdict(options))
    sim = frozen.simulator(cfg)
    previous = ControlAction.uncontrolled(cfg)
    initial_inventory = inventory(sim.state, cfg)
    depth = cfg.mpc.horizon_steps
    demand_rows = [{"step": k, "time_sec": k * cfg.simulation.T_c_sec,
        **asdict(frozen.profile.at(k * cfg.simulation.T_c_sec))} for k in range(p["expected_steps"])]
    demand_hash = hashlib.sha256(json.dumps(demand_rows, sort_keys=True).encode()).hexdigest()
    metadata = {**p, "controller": name, "demand_sha256": demand_hash, "source_hashes": frozen.sources,
        "partition": "two whole-freeway players plus five urban players (7)" if name == "PLAYER_SDMPC" else "two regional agents F/U" if name == "SDMPC" else "uncontrolled",
        "initial_inventory_veh": initial_inventory, "warm_start": "pfo" if warm else None,
        "forecast_depth": depth, "budget_contract": CONTRACTS[name], "full_acceptance_pass": False,
        "fixed_validation_failures": failures, "diagnostic_after_fixed_failure": bool(failures),
        "parallel_execution": False, "full_plant_evaluation_centralized": name in {"SDMPC", "PLAYER_SDMPC"},
        "validation_provenance_pass": True if name == "PLAYER_SDMPC" else None,
        "rolling_realization": "diagnostic; receding-horizon replanning does not commit previous full plans"}
    write_json(out / "metadata.json", metadata)
    write_json(out / "demand.json", demand_rows)
    rows, controls, states, candidates, iterations, sensitivities, locals_ = [], [], [], [], [], [], []

    def save_logs():
        for filename, values in (("run_log.csv", rows), ("control_timeseries.csv", controls),
                ("state_timeseries.csv", states), ("budget_candidates.csv", candidates),
                ("iterations.csv", iterations), ("sensitivities.csv", sensitivities), ("local_problems.csv", locals_)):
            write_csv(out / filename, values)

    def collect(step):
        candidates.extend({"step": step, **r} for r in controller.candidate_rows)
        for index, result in enumerate(controller.last_results):
            iterations.extend({"step": step, "candidate_index": index, **r} for r in result.iteration_rows)
            sensitivities.extend({"step": step, "candidate_index": index, **r} for r in result.sensitivity_rows)
            locals_.extend({"step": step, "candidate_index": index, **r} for r in getattr(result, "local_rows", []))

    save_logs()
    try:
        for k in range(p["expected_steps"]):
            forecast = frozen.profile.horizon(sim.state.time_sec, depth)
            before, before_np = inventory(sim.state, cfg), sim.state.protected_accumulation_veh(cfg.network)
            started, stage, selected = time.perf_counter(), "decision", None
            checked = name in {"SDMPC", "PLAYER_SDMPC"} and k >= p["warmup_steps"]
            try:
                if k < p["warmup_steps"] or name == "NC":
                    action = ControlAction.uncontrolled(cfg)
                elif checked:
                    controller.warm_start_control = warm.solve(sim.state.copy(), None, forecast, previous).control
                    prior_results = controller.last_results
                    try:
                        action = controller.decide(sim.state.copy(), forecast, previous, cfg)
                    finally:
                        # witness 평가 전에 예외가 나면 직전 구간의 결과를 새 실패로 오표기하지 않는다.
                        if controller.last_results is not prior_results:
                            collect(k)
                    selected = controller.last_result
                    validation = validate_player_control if name == "PLAYER_SDMPC" else validate_control
                    budget = (PlayerBudget(action.N_P_star, action.N_UF_star) if name == "PLAYER_SDMPC"
                              else Budget(action.N_P_star, action.N_UF_star))
                    if not (selected and selected.feasible and selected.evaluation.control_valid
                            and selected.evaluation.physical_valid and controller._feasible(selected.evaluation, budget)
                            and validation(action, previous, cfg)["valid"]):
                        raise RuntimeError("Final nonlinear budget/control/physical execution gate failed")
                seconds = time.perf_counter() - started
                # 입력/코드 변경 역시 plant 실행 전에 실패시킨다.
                frozen.verify()
                stage = "plant_step"
                log = sim.step(action, forecast[0], k)
                stage = "logging"
                service = float(log.diagnostics["inbound_service_veh"]) - float(log.diagnostics["outbound_service_veh"])
                arrivals = cfg.simulation.T_c_h * (sum(forecast[0].freeway_mainline.values())
                    + sum(forecast[0].urban_boundary.get(g, 0.) for g in cfg.network.boundary_in_links)
                    + sum(forecast[0].ramp_arrival.values()))
                exits = cfg.simulation.T_c_h * float(log.diagnostics.get("mainline_exit_flow_total", 0.)) + float(log.diagnostics.get("boundary_out_sink_veh", 0.))
                balance = movement_balance_summary(sim.state, cfg,
                    saturation_fraction=cfg.evaluation.boundary_degenerate_saturation_fraction,
                    degenerate_ratio=cfg.evaluation.boundary_degenerate_ratio, eps=cfg.evaluation.eps)
                row = {"step": k, "time_sec": sim.state.time_sec,
                    "total_ttt": log.freeway_ttt + log.urban_ttt, "freeway_ttt": log.freeway_ttt,
                    "urban_ttt": log.urban_ttt, "cumulative_total_ttt": sim.total_ttt,
                    "computation_time_sec": seconds,
                    **{a: b for a, b in log.diagnostics.items() if isinstance(b, (int, float, bool))}, **balance,
                    "inventory_veh": inventory(sim.state, cfg), "external_arrivals_veh": arrivals,
                    "external_exits_veh": exits, "conservation_residual_veh": inventory(sim.state, cfg) - before - arrivals + exits,
                    "actual_interval_net_service_veh": service, "actual_np_change_veh": sim.state.protected_accumulation_veh(cfg.network) - before_np,
                    "actual_nuf_veh_h": float(log.diagnostics["total_metering_flow"]),
                    "commanded_meter_sum_veh_h": sum(action.ramp_metering.values()),
                    "budget_checked": checked, "warmup": k < p["warmup_steps"],
                    "selected_feasible": selected.feasible if checked else None,
                    "selected_converged": selected.converged if checked else None,
                    "selected_control_valid": selected.evaluation.control_valid if checked else None,
                    "selected_physical_valid": selected.evaluation.physical_valid if checked else None}
                if checked:
                    ev = selected.evaluation
                    row.update(requested_nuf_veh_h=action.N_UF_star,
                        planned_np_signed_residual_veh=selected.residual_np_veh,
                        planned_nuf_residual_veh_h=selected.residual_nuf_veh_h)
                    if name == "PLAYER_SDMPC":
                        row.update(requested_np_cap_veh=action.N_P_star, planned_horizon_net_service_veh=ev.achieved_np_veh,
                            planned_np_violation_veh=max(0., selected.residual_np_veh),
                            commanded_nuf_veh_h=sum(action.ramp_metering.values()),
                            command_nuf_residual_veh_h=sum(action.ramp_metering.values()) - action.N_UF_star,
                            actual_release_minus_command_budget_veh_h=row["actual_nuf_veh_h"] - action.N_UF_star,
                            planned_player_cost_sum_residual=sum(ev.player_costs.values()) - ev.total_ttt,
                            **{f"planned_J_{player}": value for player, value in ev.player_costs.items()})
                    else:
                        row.update(requested_np_change_veh=action.N_P_star,
                            raw_np_residual_veh=row["actual_np_change_veh"] - action.N_P_star,
                            raw_nuf_residual_veh_h=row["actual_nuf_veh_h"] - action.N_UF_star,
                            planned_first_np_change_veh=ev.achieved_np_change_veh,
                            planned_first_actual_nuf_veh_h=ev.achieved_nuf_veh_h)
                rows.append(row)
                controls.append(control_row(action, cfg, k, sim.state.time_sec))
                sr = state_row(sim.state, cfg, k)
                sr.update(urban_vehicles=sim.state.total_urban_vehicles(cfg.network), total_inventory_veh=inventory(sim.state, cfg))
                for link in cfg.network.freeway_links:
                    sr[f"origin_queue_{link}"] = sim.state.mainline_origin_queue.get(link, 0.)
                    for seg, density in enumerate(sim.state.freeway_density[link]):
                        sr[f"rho_{link}_seg{seg}"] = density
                states.append(sr)
                previous = action.copy()
                save_logs()
                print(f"{name} step={k+1}/{p['expected_steps']} TTT={sim.total_ttt:.6f} budget_checked={checked} seconds={seconds:.2f}", flush=True)
            except Exception as exc:
                save_logs()
                write_json(out / "abort_state.json", asdict(sim.state))
                write_json(out / "abort.json", {"status": "aborted", "reason": str(exc), "type": type(exc).__name__,
                    "step": k, "executed_steps": len(rows), "time_sec": sim.state.time_sec, "stage": stage,
                    "aborted_before_sim_step": stage == "decision", "fallback_executed": False,
                    "requested_candidates": [row for row in candidates if row["step"] == k],
                    "previous_control": asdict(previous), "forecast": [asdict(d) for d in forecast],
                    "selected_result": asdict(selected) if selected else None, "traceback": traceback.format_exc()})
                raise
        frozen.verify()
        if (len(rows) != 80 or len(controls) != 80 or len(states) != 80
                or [r["step"] for r in rows] != list(range(80))
                or any(r["time_sec"] != (k + 1) * cfg.simulation.T_c_sec for k, r in enumerate(rows))
                or sim.state.time_sec != 14400):
            raise RuntimeError("Incomplete/noncontiguous matrix run")
        result = {"run_rows": rows, "control_rows": controls, "state_rows": states,
            "final_state": sim.state, "total_ttt": sim.total_ttt, "freeway_ttt": sim.freeway_ttt, "urban_ttt": sim.urban_ttt}
        summary = summarize_run(result, cfg)
        checked_rows = [r for r in rows if r["budget_checked"]]
        summary.update(controller=name, demand_sha256=demand_hash, executed_steps=len(rows),
            controlled_steps=75 if name != "NC" else 0, warmup_steps=5,
            post_warmup_total_ttt=sum(r["total_ttt"] for r in rows[5:]), warmup_total_ttt=sum(r["total_ttt"] for r in rows[:5]),
            completed_vehicles=sum(r["external_exits_veh"] for r in rows), terminal_inventory_veh=inventory(sim.state, cfg),
            max_conservation_residual_veh=max(abs(r["conservation_residual_veh"]) for r in rows),
            computation_time_sec=sum(r["computation_time_sec"] for r in rows), budget_checked_steps=len(checked_rows),
            selected_converged_steps=sum(bool(r["selected_converged"]) for r in checked_rows),
            execution_gate_pass=all(r["selected_feasible"] and r["selected_control_valid"] and r["selected_physical_valid"] for r in checked_rows) if checked_rows else None,
            fixed_validation_failures=failures, diagnostic_after_fixed_failure=bool(failures), full_acceptance_pass=False)
        if name == "PLAYER_SDMPC":
            windows = []
            for k in range(5, len(rows) - options.horizon_steps + 1):
                achieved = sum(r["actual_interval_net_service_veh"] for r in rows[k:k+options.horizon_steps])
                residual = achieved - rows[k]["requested_np_cap_veh"]
                windows.append({"start_step": k, "end_step": k + options.horizon_steps - 1,
                    "requested_at_start_np_cap_veh": rows[k]["requested_np_cap_veh"],
                    "actual_replanned_horizon_net_service_veh": achieved, "signed_residual_veh": residual,
                    "violation_veh": max(0., residual), "was_committed_execution_contract": False})
            write_csv(out / "actual_rolling_budget_windows.csv", windows)
            summary.update(max_planned_np_violation_veh=max(r["planned_np_violation_veh"] for r in checked_rows),
                max_command_nuf_residual_veh_h=max(abs(r["command_nuf_residual_veh_h"]) for r in checked_rows),
                horizon_plan_contract_pass=all(r["planned_np_violation_veh"] <= options.tolerance_np_veh
                    and abs(r["command_nuf_residual_veh_h"]) <= options.tolerance_nuf_veh_h for r in checked_rows),
                actual_rolling_window_count=len(windows),
                actual_replanned_horizon_np_violations=sum(r["violation_veh"] > options.tolerance_np_veh for r in windows),
                max_actual_replanned_horizon_np_violation_veh=max(r["violation_veh"] for r in windows))
        elif name == "SDMPC":
            summary.update(max_np_residual_veh=max(abs(r["raw_np_residual_veh"]) for r in checked_rows),
                max_nuf_residual_veh_h=max(abs(r["raw_nuf_residual_veh_h"]) for r in checked_rows),
                budget_contract_pass=all(abs(r["raw_np_residual_veh"]) <= options.tolerance_np_veh
                    and abs(r["raw_nuf_residual_veh_h"]) <= options.tolerance_nuf_veh_h for r in checked_rows))
        write_json(out / "metrics_summary.json", summary)
        write_json(out / "final_state.json", asdict(sim.state))
        write_json(out / "diagnostics.json", {"summary": summary, "contract": CONTRACTS[name], "source_hashes": frozen.sources})
        (out / "report.md").write_text(f"# {name} 실행\n\n80구간/14400초 실행 완료. 전체 TTT={sim.total_ttt:.6f} veh*h, "
            f"warmup 제외 TTT={summary['post_warmup_total_ttt']:.6f} veh*h.\n\n"
            f"원 budget 의미: {CONTRACTS[name]}. 보존된 검증 실패: {failures}.\n\n"
            "실행 완료와 연구 수용 판정은 별개이며 full_acceptance_pass=False이다.\n", encoding="utf-8")
        frozen.verify()
        write_json(out / "completion.json", {"simulation_complete": True, "executed_steps": 80,
            "final_time_sec": 14400, "full_acceptance_pass": False, "source_input_integrity_pass": True})
        return summary
    finally:
        if hasattr(controller, "close"):
            controller.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixed", "run"), required=True)
    parser.add_argument("--controller", choices=("NC", "CANONICAL", "SDMPC", "PLAYER_SDMPC"), required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-validation", type=Path)
    parser.add_argument("--sensitivity-validation", type=Path)
    parser.add_argument("--diagnostic-after-fixed-failure", action="store_true")
    args = parser.parse_args()
    if args.mode == "fixed" and args.controller != "PLAYER_SDMPC":
        parser.error("fixed mode requires PLAYER_SDMPC")
    out = args.output / ("fixed_validation" if args.mode == "fixed" else args.controller)
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner_source.py.txt").write_bytes(Path(__file__).read_bytes())
    try:
        root, frozen_env = bootstrap(args)
        imported = load_runtime(root)
        frozen = FrozenInputs(args.protocol_dir, out, root, frozen_env, imported)
        np.random.seed(frozen.protocol["seed"])
        if args.mode == "fixed":
            fixed_validation(frozen)
        else:
            run_controller(args, frozen)
    except Exception as exc:
        # import/bootstrap 실패도 helper에 의존하지 않고 원인과 nonzero를 보존한다.
        if not (out / "abort.json").exists():
            (out / "abort.json").write_text(json.dumps({"status": "aborted", "reason": str(exc),
                "type": type(exc).__name__, "simulation_complete": False, "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        (out / "completion.json").write_text(json.dumps({"simulation_complete": False,
            "full_acceptance_pass": False, "reason": str(exc)}, indent=2), encoding="utf-8")
        print(f"CELL_ABORT {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
