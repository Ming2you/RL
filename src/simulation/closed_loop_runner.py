from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.controllers.stackelberg_mpc import StackelbergMPCController
from src.models.demand import (
    DemandProfile,
    ScenarioConfig,
    apply_scenario_network_overrides,
)
from src.models.state import ControlAction, ExperimentConfig, TrafficState
from src.simulation.baseline import baseline_control
from src.simulation.simulator import MixedTrafficSimulator, control_row, state_row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _apply_ablation(control: ControlAction, cfg: ExperimentConfig, modes: Iterable[str]) -> ControlAction:
    modes = set(modes or [])
    fixed = ControlAction.fixed(cfg)
    if "proposed_without_ramp_metering" in modes:
        control.ramp_metering = fixed.ramp_metering
    if "proposed_without_vsl" in modes:
        control.vsl = fixed.vsl
    if "proposed_without_offset_control" in modes:
        control.offsets = fixed.offsets
    if "proposed_without_green_time_allocation" in modes:
        control.green_times = fixed.green_times
    if "proposed_without_inflow_outflow_allocation" in modes:
        control.inflow_outflow_allocation = fixed.inflow_outflow_allocation
    return control


def run_closed_loop(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
    mode: str,
    output_dir: Optional[str | Path] = None,
    baseline_mode: str = "fixed_signal_fixed_speed",
    ablation_modes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    # 시나리오 한정 off-ramp split override를 cfg.network에 단일 주입(plant+controller 공유).
    cfg = apply_scenario_network_overrides(cfg, scenario)
    demand = DemandProfile(cfg, scenario)
    sim = MixedTrafficSimulator(cfg)
    controller = StackelbergMPCController(cfg) if mode == "stackelberg_mpc" else None
    previous: ControlAction | None = None
    state_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    progress_rows: List[Dict[str, Any]] = []
    out = Path(output_dir) if output_dir is not None else None

    for step in range(cfg.simulation.n_control_steps):
        current_demand = demand.at(sim.state.time_sec)
        if controller:
            forecast = demand.horizon(sim.state.time_sec, cfg.mpc.horizon_steps)
            control = controller.decide(sim.state.copy(), forecast, previous, cfg)
            control = _apply_ablation(control, cfg, ablation_modes or [])
        else:
            control = baseline_control(baseline_mode, cfg, sim.state.copy(), current_demand, previous)
        log = sim.step(control, current_demand, step)
        previous = control
        state_rows.append(state_row(sim.state, cfg, step))
        control_rows.append(control_row(control, cfg, step, sim.state.time_sec))
        run_row = {
            "step": step,
            "time_sec": sim.state.time_sec,
            "freeway_ttt": log.freeway_ttt,
            "urban_ttt": log.urban_ttt,
            "total_ttt": log.freeway_ttt + log.urban_ttt,
        }
        run_row.update({k: v for k, v in log.diagnostics.items() if isinstance(v, (int, float, bool))})
        run_rows.append(run_row)
        progress_rows.append({
            "step": step,
            "time_sec": sim.state.time_sec,
            "controller_id": "STACKELBERG-MPC" if controller else "NO-CONTROL",
            "step_total_ttt": log.freeway_ttt + log.urban_ttt,
            "step_urban_ttt": log.urban_ttt,
            "step_freeway_ttt": log.freeway_ttt,
            "cumulative_total_ttt": sim.total_ttt,
            "cumulative_urban_ttt": sim.urban_ttt,
            "cumulative_freeway_ttt": sim.freeway_ttt,
            "N_P_star": control.N_P_star,
            "N_UF_star": control.N_UF_star,
            "mean_B_sum_step": float(run_row.get("B_in", 0.0)) + float(run_row.get("B_out", 0.0)),
            "terminal_total_vehicles": (
                sim.state.total_urban_vehicles(cfg.network) + sim.state.total_freeway_vehicles(cfg.network)
            ),
        })
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            _write_csv(out / "state_timeseries.csv", state_rows)
            _write_csv(out / "control_timeseries.csv", control_rows)
            _write_csv(out / "run_log.csv", run_rows)
            _write_csv(out / "progress_summary.csv", progress_rows)
            print(
                f"{progress_rows[-1]['controller_id']} step {step + 1}/{cfg.simulation.n_control_steps} "
                f"cum_ttt={sim.total_ttt:.3f} step_ttt={log.freeway_ttt + log.urban_ttt:.3f}",
                flush=True,
            )

    result = {
        "mode": mode,
        "scenario": scenario.name,
        "baseline_mode": baseline_mode,
        "freeway_ttt": sim.freeway_ttt,
        "urban_ttt": sim.urban_ttt,
        "total_ttt": sim.total_ttt,
        "state_rows": state_rows,
        "control_rows": control_rows,
        "run_rows": run_rows,
        "progress_rows": progress_rows,
        "final_state": sim.state,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "state_timeseries.csv", state_rows)
        _write_csv(out / "control_timeseries.csv", control_rows)
        _write_csv(out / "run_log.csv", run_rows)
        _write_csv(out / "progress_summary.csv", progress_rows)
        summary = {
            "mode": mode,
            "scenario": scenario.name,
            "baseline_mode": baseline_mode,
            "freeway_ttt": sim.freeway_ttt,
            "urban_ttt": sim.urban_ttt,
            "total_ttt": sim.total_ttt,
            "config": cfg.to_dict(),
        }
        (out / "metrics_raw.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result
