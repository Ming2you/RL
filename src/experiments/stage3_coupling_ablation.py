# Stage 3 runner — player·coupling-정보 ablation과 한계가치/synergy 계산 (plan §9~§13)
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.controllers.distributed_coordinator import ABLATION_MODES, DistributedCoordinator
from src.controllers.stackelberg_mpc import StackelbergMPCController
from src.models.demand import DemandProfile, ScenarioConfig, load_scenarios
from src.models.state import ExperimentConfig
from src.simulation.simulator import MixedTrafficSimulator


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_ablation_case(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
    ablation: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """PROPOSED-STACKELBERG를 지정 ablation으로 closed-loop 실행한다.

    leader는 매 interval 후보를 ablated coordinator의 follower 응답으로 평가하므로
    '잔여 player + leader 재최적화' 요건(plan §9.1)이 구조적으로 충족된다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.mpc.follower_solver_mode = "distributed"
    controller = StackelbergMPCController(cfg)
    coordinator = DistributedCoordinator(cfg, ablation=ablation)
    controller.nash_solver = coordinator  # ablated follower game으로 교체.
    profile = DemandProfile(cfg, scenario)
    sim = MixedTrafficSimulator(cfg)
    steps = max(1, int(round(cfg.simulation.T_total / cfg.simulation.control_interval)))
    start = time.perf_counter()
    iterations_total = 0
    run_rows: List[Dict[str, Any]] = []
    for step in range(steps):
        t = step * cfg.simulation.control_interval
        forecast = profile.horizon(t, cfg.mpc.horizon_steps)
        control = controller.decide(sim.state.copy(), forecast)
        log = sim.step(control, forecast[0], step)
        iterations_total += int(control.diagnostics.get("nash_iterations", 0))
        run_rows.append({
            "step": step,
            "urban_ttt": log.urban_ttt,
            "freeway_ttt": log.freeway_ttt,
            "N_P_star": control.N_P_star,
            "N_UF_star": control.N_UF_star,
            "offramp_accepted_veh": float(log.diagnostics.get("coupling_offramp_arrivals_accepted_veh", 0.0)),
            "offramp_blocked_veh": float(log.diagnostics.get("offramp_blocked_flow_total", 0.0)),
            "onramp_transfer_veh": float(log.diagnostics.get("onramp_green_releases_veh", 0.0)),
            "ramp_queue_veh": float(log.diagnostics.get("ramp_queue_veh", 0.0)),
            "onramp_approach_queue_veh": float(log.diagnostics.get("onramp_approach_queue_veh", 0.0)),
        })
    _write_csv(output_dir / "run_log.csv", run_rows)
    net = cfg.network
    final = sim.state
    summary = {
        "ablation": ablation,
        "total_ttt": round(sim.total_ttt, 3),
        "urban_ttt": round(sim.urban_ttt, 3),
        "freeway_ttt": round(sim.freeway_ttt, 3),
        "terminal_total_vehicles": round(
            final.total_urban_vehicles(net) + final.total_freeway_vehicles(net), 1
        ),
        "offramp_accepted_veh": round(sum(r["offramp_accepted_veh"] for r in run_rows), 1),
        "onramp_transfer_veh": round(sum(r["onramp_transfer_veh"] for r in run_rows), 1),
        "x_on_peak_veh": round(max(r["onramp_approach_queue_veh"] for r in run_rows), 1),
        "w_r_peak_veh": round(max(r["ramp_queue_veh"] for r in run_rows), 1),
        "computation_time_sec": round(time.perf_counter() - start, 2),
        "nash_iterations_total": iterations_total,
        "physical_coupling_active": True,   # plant는 어떤 ablation에서도 그대로(plan §9.1).
        "remaining_players_reoptimized": True,
        "leader_reoptimized": True,
    }
    return summary


def coupling_values(j: Dict[str, float]) -> List[Dict[str, Any]]:
    """plan §12 — directional marginal value, synergy, order-averaged contribution."""
    rows = [
        {"metric": "Value_U_to_F_given_F_to_U", "value": j["NO_U_TO_F_INFO"] - j["FULL_COUPLING"]},
        {"metric": "Value_F_to_U_given_U_to_F", "value": j["NO_F_TO_U_INFO"] - j["FULL_COUPLING"]},
        {"metric": "BidirectionalSynergy", "value": (
            j["NO_CROSS_NETWORK_INFO"] - j["NO_U_TO_F_INFO"] - j["NO_F_TO_U_INFO"] + j["FULL_COUPLING"]
        )},
        {"metric": "Phi_U_to_F", "value": 0.5 * (
            (j["NO_CROSS_NETWORK_INFO"] - j["NO_F_TO_U_INFO"]) + (j["NO_U_TO_F_INFO"] - j["FULL_COUPLING"])
        )},
        {"metric": "Phi_F_to_U", "value": 0.5 * (
            (j["NO_CROSS_NETWORK_INFO"] - j["NO_U_TO_F_INFO"]) + (j["NO_F_TO_U_INFO"] - j["FULL_COUPLING"])
        )},
        {"metric": "UrbanCouplingPlayerValue", "value": j["FIXED_URBAN_COUPLING_PLAYERS"] - j["FULL_COUPLING"]},
        {"metric": "FreewayCouplingPlayerValue", "value": j["FIXED_FREEWAY_COUPLING_PLAYERS"] - j["FULL_COUPLING"]},
        {"metric": "AllCouplingPlayersValue", "value": j["FIXED_ALL_COUPLING_PLAYERS"] - j["FULL_COUPLING"]},
    ]
    for row in rows:
        row["value"] = round(row["value"], 3)
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default="src/config/scenarios.yaml")
    parser.add_argument("--scenario", default="peak_demand")
    parser.add_argument("--T-total", type=float, default=None)
    parser.add_argument("--output", default="post_analysis/stage3")
    parser.add_argument("--cases", default=",".join(ABLATION_MODES))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    scenario = load_scenarios(args.scenarios_config)[args.scenario]

    summaries: List[Dict[str, Any]] = []
    for ablation in requested:
        overrides: Dict[str, Any] = {}
        if args.T_total is not None:
            overrides["simulation"] = {"T_total": args.T_total}
        if args.seed is not None:
            overrides.setdefault("simulation", {})["random_seed"] = args.seed
        cfg = ExperimentConfig.from_file(args.config, overrides)  # case마다 새 cfg(상태 격리).
        summary = run_ablation_case(cfg, scenario, ablation, out / "runs" / args.scenario / ablation)
        summaries.append(summary)
        print(f"{ablation}: ttt={summary['total_ttt']:.1f} urban={summary['urban_ttt']:.1f} "
              f"freeway={summary['freeway_ttt']:.1f}")
    info_rows = [s for s in summaries if not s["ablation"].startswith("FIXED_")]
    player_rows = [s for s in summaries if s["ablation"].startswith("FIXED_") or s["ablation"] == "FULL_COUPLING"]
    _write_csv(out / "information_ablation_summary.csv", info_rows)
    _write_csv(out / "player_ablation_summary.csv", player_rows)
    j = {s["ablation"]: float(s["total_ttt"]) for s in summaries}
    if all(m in j for m in ABLATION_MODES):
        values = coupling_values(j)
        _write_csv(out / "directional_coupling_value.csv", [v for v in values if "Value_" in v["metric"] or "Phi_" in v["metric"] or "Player" in v["metric"]])
        _write_csv(out / "coupling_synergy.csv", [v for v in values if v["metric"] == "BidirectionalSynergy"])
        for v in values:
            print(f"{v['metric']}: {v['value']}")
    (out / "summary.json").write_text(
        json.dumps({"scenario": args.scenario, "summaries": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"STAGE3 scenario={args.scenario} cases={len(summaries)} output={out}")


if __name__ == "__main__":
    main()
