from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.models.demand import ScenarioConfig
from src.models.state import ExperimentConfig
from src.simulation.closed_loop_runner import run_closed_loop


def tuned_capacity_drop_config(base_path: str, t_total: float) -> ExperimentConfig:
    """Capacity drop과 VSL activation을 동시에 관측하기 위한 stress tuning config를 만든다."""
    return ExperimentConfig.from_file(base_path).with_updates({
        "simulation": {"T_total": t_total},
        "mpc": {
            "horizon_steps": 3,
            "leader_candidate_count": 15,
            "max_nash_iter": 10,
        },
        "network": {
            # W/E freeway 각각 D/F 두 off-ramp가 있으므로 split ratio 합이 1을 넘지 않게 둔다.
            "off_ramp_split_ratio": {
                "OR_D_W": 0.45,
                "OR_F_W": 0.45,
                "OR_D_E": 0.45,
                "OR_F_E": 0.45,
            },
            "urban_link_storage_veh": {
                "OR_D_W_storage": 20.0,
                "OR_F_W_storage": 20.0,
                "OR_D_E_storage": 20.0,
                "OR_F_E_storage": 20.0,
            },
            "urban_avg_speed_km_h": 3.0,
            "urban_avg_vehicle_length_m": 15.0,
        },
        "freeway_offramp_capacity_drop": {
            "enabled": True,
            "lane_reduction": 0.75,
            "gamma": 0.2,
            "b": 2.0,
        },
        "freeway_follower": {
            "density_penalty": 10.0,
            "vsl_smoothness_weight": 0.0,
            "horizon_beam_width": 2,
            "horizon_vsl_candidate_limit_per_link": 3,
            "horizon_ramp_candidate_limit": 3,
        },
    })


def stress_scenario() -> ScenarioConfig:
    """Off-ramp spill-back을 빠르게 재현하기 위한 demand scenario를 반환한다."""
    return ScenarioConfig(
        "capacity_drop_stress_delay",
        urban_scale=1.55,
        freeway_scale=1.45,
        ramp_scale=1.60,
        incident_capacity_factor=0.92,
    )


def _vsl_active_steps(control_rows: Iterable[Mapping[str, Any]], cfg: ExperimentConfig) -> int:
    threshold = max(cfg.freeway_follower.vsl_set) - 0.5
    return sum(
        any(float(row.get(f"vsl_{link}", max(cfg.freeway_follower.vsl_set))) < threshold for link in cfg.network.freeway_links)
        for row in control_rows
    )


def _capacity_summary(result: Mapping[str, Any], cfg: ExperimentConfig) -> Dict[str, Any]:
    rows = list(result.get("run_rows", []))
    controls = list(result.get("control_rows", []))
    lambda_values = [
        float(row.get(f"lambda_eff_{link}_last", cfg.network.freeway_lanes))
        for row in rows
        for link in cfg.network.freeway_links
    ]
    return {
        "total_ttt": float(result.get("total_ttt", 0.0)),
        "freeway_ttt": float(result.get("freeway_ttt", 0.0)),
        "urban_ttt": float(result.get("urban_ttt", 0.0)),
        "capacity_drop_active_steps": sum(1 for row in rows if float(row.get("capacity_drop_active", 0.0)) > 0.0),
        "lambda_min": min(lambda_values) if lambda_values else float(cfg.network.freeway_lanes),
        "vsl_active_steps": _vsl_active_steps(controls, cfg),
        "overlap_steps": sum(
            1
            for row, control in zip(rows, controls)
            if float(row.get("capacity_drop_active", 0.0)) > 0.0
            and any(
                float(control.get(f"vsl_{link}", max(cfg.freeway_follower.vsl_set)))
                < max(cfg.freeway_follower.vsl_set) - 0.5
                for link in cfg.network.freeway_links
            )
        ),
        "controls": [
            {
                "step": row.get("step"),
                "N_UF_star": row.get("N_UF_star"),
                **{f"vsl_{link}": row.get(f"vsl_{link}") for link in cfg.network.freeway_links},
            }
            for row in controls
        ],
        "capacity_diagnostics": [
            {
                key: value
                for key, value in row.items()
                if "capacity_drop" in key
                or "lambda_eff" in key
                or "offramp_occupancy" in key
                or "density_exceedance" in key
            }
            for row in rows
        ],
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    proposed = summary["proposed"]
    baseline = summary["baseline"]
    without_vsl = summary["proposed_without_vsl"]
    text = f"""# Capacity Drop VSL Probe

## 목적

이 probe는 기본 `peak_demand`가 아니라 off-ramp spill-back을 의도적으로 악화시킨 stress tuning에서 VSL이 실제로 activate되는지 확인한다.

## Tuning

- `off_ramp_split_ratio`: each off-ramp `0.45`
- `OR_*_storage`: `20 veh`
- `urban_avg_speed_km_h`: `3.0`
- `urban_avg_vehicle_length_m`: `15.0`
- `lane_reduction`: `0.75`
- `gamma`: `0.2`
- `vsl_smoothness_weight`: `0.0`
- `horizon_steps`: `3`

## 결과

| run | total TTT | freeway TTT | urban TTT | capacity drop active | lambda min | VSL active | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | {baseline['total_ttt']:.3f} | {baseline['freeway_ttt']:.3f} | {baseline['urban_ttt']:.3f} | {baseline['capacity_drop_active_steps']} | {baseline['lambda_min']:.6f} | {baseline['vsl_active_steps']} | {baseline['overlap_steps']} |
| proposed | {proposed['total_ttt']:.3f} | {proposed['freeway_ttt']:.3f} | {proposed['urban_ttt']:.3f} | {proposed['capacity_drop_active_steps']} | {proposed['lambda_min']:.6f} | {proposed['vsl_active_steps']} | {proposed['overlap_steps']} |
| proposed_without_vsl | {without_vsl['total_ttt']:.3f} | {without_vsl['freeway_ttt']:.3f} | {without_vsl['urban_ttt']:.3f} | {without_vsl['capacity_drop_active_steps']} | {without_vsl['lambda_min']:.6f} | {without_vsl['vsl_active_steps']} | {without_vsl['overlap_steps']} |

## 해석

- 이 stress tuning에서는 capacity drop과 VSL activation이 같은 control step에서 함께 관측되는지를 확인한다.
- 다만 proposed total TTT가 `proposed_without_vsl`보다 항상 낮다는 보장은 없다. 따라서 이 결과는 "VSL이 켜지는가"에 대한 positive check이며, "VSL이 항상 성능을 개선하는가"에 대한 positive proof는 아니다.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--T-total", type=float, default=720.0)
    parser.add_argument("--output", default="outputs/capacity_drop_vsl_probe")
    args = parser.parse_args(argv)

    cfg = tuned_capacity_drop_config(args.config, args.T_total)
    scenario = stress_scenario()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    baseline = run_closed_loop(
        cfg,
        scenario,
        mode="baseline",
        output_dir=out / "baseline",
        baseline_mode="fixed_signal_fixed_speed",
    )
    proposed = run_closed_loop(
        cfg,
        scenario,
        mode="stackelberg_mpc",
        output_dir=out / "proposed",
        baseline_mode="fixed_signal_fixed_speed",
    )
    proposed_without_vsl = run_closed_loop(
        cfg,
        scenario,
        mode="stackelberg_mpc",
        output_dir=out / "proposed_without_vsl",
        baseline_mode="fixed_signal_fixed_speed",
        ablation_modes=["proposed_without_vsl"],
    )

    summary = {
        "scenario": scenario.name,
        "baseline": _capacity_summary(baseline, cfg),
        "proposed": _capacity_summary(proposed, cfg),
        "proposed_without_vsl": _capacity_summary(proposed_without_vsl, cfg),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out / "report.md", summary)

    proposed_summary = summary["proposed"]
    print(
        "capacity_drop_active_steps="
        f"{proposed_summary['capacity_drop_active_steps']} "
        f"vsl_active_steps={proposed_summary['vsl_active_steps']} "
        f"overlap_steps={proposed_summary['overlap_steps']} "
        f"lambda_min={proposed_summary['lambda_min']:.6f} "
        f"report={out / 'report.md'}"
    )


if __name__ == "__main__":
    main()
