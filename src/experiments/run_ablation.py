from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import evaluate
from src.models.demand import load_scenarios
from src.models.state import ExperimentConfig
from src.simulation.closed_loop_runner import run_closed_loop


ABLATION_MODES = [
    "proposed_without_ramp_metering",
    "proposed_without_vsl",
    "proposed_without_offset_control",
    "proposed_without_green_time_allocation",
    "proposed_without_inflow_outflow_allocation",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default=None)
    parser.add_argument("--scenario", default="peak_demand")
    parser.add_argument("--baseline", default="fixed_signal_fixed_speed")
    parser.add_argument("--output", default="outputs/ablation_peak_demand")
    parser.add_argument("--T-total", type=float, default=None)
    args = parser.parse_args(argv)

    overrides = {"simulation": {"T_total": args.T_total}} if args.T_total is not None else None
    cfg = ExperimentConfig.from_file(args.config, overrides)
    scenarios_path = Path(args.scenarios_config) if args.scenarios_config else Path(args.config).resolve().parent / "scenarios.yaml"
    scenario = load_scenarios(scenarios_path)[args.scenario]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    baseline = run_closed_loop(cfg, scenario, "baseline", out / "baseline", baseline_mode=args.baseline)
    proposed = run_closed_loop(cfg, scenario, "stackelberg_mpc", out / "proposed", baseline_mode=args.baseline)
    rows = {
        "proposed": {
            "total_ttt": proposed["total_ttt"],
            "improvement_pct": evaluate(baseline, proposed, cfg).improvement_pct,
        }
    }
    for mode in ABLATION_MODES:
        result = run_closed_loop(
            cfg,
            scenario,
            "stackelberg_mpc",
            out / mode,
            baseline_mode=args.baseline,
            ablation_modes=[mode],
        )
        rows[mode] = {
            "total_ttt": result["total_ttt"],
            "improvement_pct": evaluate(baseline, result, cfg).improvement_pct,
        }
    (out / "ablation_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved ablation results: {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
