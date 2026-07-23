from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.models.demand import ScenarioConfig, load_scenarios
from src.models.state import ExperimentConfig
from src.simulation.closed_loop_runner import run_closed_loop


def _parse_scales(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one urban scale must be provided.")
    return values


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scenario_with_urban_scale(base: ScenarioConfig, scale: float) -> ScenarioConfig:
    return replace(
        base,
        name=f"{base.name}_urban_scale_{scale:g}",
        urban_scale=float(scale),
    )


def _extract_mfd_points(
    result: Dict[str, Any],
    cfg: ExperimentConfig,
    scenario_name: str,
    urban_scale: float,
) -> list[Dict[str, Any]]:
    points: list[Dict[str, Any]] = []
    state_rows = result.get("state_rows", [])
    run_rows = result.get("run_rows", [])
    for state_row, run_row in zip(state_rows, run_rows):
        production_veh = float(
            run_row.get(
                "urban_total_departures_veh",
                run_row.get("outbound_service_veh", 0.0),
            )
        )
        production_veh_h = production_veh / max(cfg.simulation.T_c_h, 1.0e-9)
        if "urban_protected_accumulation_veh" not in state_row:
            raise KeyError(
                "state_timeseries row must include urban_protected_accumulation_veh "
                "for N_P_crit calibration."
            )
        points.append({
            "scenario": scenario_name,
            "urban_scale": float(urban_scale),
            "step": int(run_row.get("step", state_row.get("step", 0))),
            "time_sec": float(run_row.get("time_sec", state_row.get("time_sec", 0.0))),
            "urban_accumulation_veh": float(state_row["urban_protected_accumulation_veh"]),
            "urban_production_veh_h": production_veh_h,
            "net_inflow_veh_h": float(run_row.get("net_inflow", 0.0)),
            "urban_total_departures_veh": production_veh,
            "outbound_service_veh": float(run_row.get("outbound_service_veh", 0.0)),
            "CV_boundary": float(run_row.get("CV_boundary", 0.0)),
            "MaxMin_boundary": float(run_row.get("MaxMin_boundary", 0.0)),
        })
    return points


def _estimate_n_crit(points: list[Dict[str, Any]]) -> Dict[str, float]:
    if not points:
        return {
            "n_crit_veh": 0.0,
            "max_production_veh_h": 0.0,
            "source_urban_scale": 0.0,
            "source_time_sec": 0.0,
        }
    best = max(points, key=lambda row: float(row.get("urban_production_veh_h", 0.0)))
    return {
        "n_crit_veh": float(best.get("urban_accumulation_veh", 0.0)),
        "max_production_veh_h": float(best.get("urban_production_veh_h", 0.0)),
        "source_urban_scale": float(best.get("urban_scale", 0.0)),
        "source_time_sec": float(best.get("time_sec", 0.0)),
    }


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    estimate = summary["estimate"]
    lines = [
        "# Setpoint Calibration Report",
        "",
        "## 요약",
        "",
        f"- 추정 `n_crit`: `{estimate['n_crit_veh']:.3f} veh`",
        f"- 관측 최대 urban production: `{estimate['max_production_veh_h']:.3f} veh/h`",
        f"- 해당 urban scale: `{estimate['source_urban_scale']:.3f}`",
        f"- 해당 시각: `{estimate['source_time_sec']:.1f} s`",
        "",
        "## 방법",
        "",
        "- 여러 urban demand scale에 대해 baseline closed-loop simulation을 실행했습니다.",
        "- `state_timeseries.csv`의 `urban_protected_accumulation_veh`를 urban accumulation으로 기록했습니다.",
        "- 이 값은 `TrafficState.protected_accumulation_veh(net)`이며, off-ramp storage는 제외하고 보호영역 내부 storage와 movement queue를 포함합니다.",
        "- run diagnostics의 `urban_total_departures_veh / T_c_h`를 production으로 기록했습니다.",
        "- 관측 production이 최대인 지점의 accumulation을 1차 `n_crit` 추정치로 선택했습니다.",
        "",
        "## 주의",
        "",
        "- 이 결과는 deterministic scaffold이며, 최종 통계적 calibration은 아닙니다.",
        "- config 값을 확정하기 전에는 더 긴 horizon과 더 촘촘한 scale point를 사용해야 합니다.",
        "- Freeway q-rho calibration은 P1 단계로 분리했습니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default="src/config/scenarios.yaml")
    parser.add_argument("--scenario", default="peak_demand")
    parser.add_argument("--baseline", default="fixed_signal_fixed_speed")
    parser.add_argument("--urban-scales", default="0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0")
    parser.add_argument("--T-total", type=float, default=None)
    parser.add_argument("--output", default="outputs/setpoint_calibration")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.T_total is not None:
        overrides.setdefault("simulation", {})["T_total"] = args.T_total
    cfg = ExperimentConfig.from_file(args.config, overrides)
    scenarios = load_scenarios(args.scenarios_config)
    if args.scenario not in scenarios:
        raise SystemExit(f"Unknown scenario {args.scenario}. Available: {', '.join(sorted(scenarios))}")
    base = scenarios[args.scenario]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_points: list[Dict[str, Any]] = []
    runs: list[Dict[str, Any]] = []
    for scale in _parse_scales(args.urban_scales):
        scenario = _scenario_with_urban_scale(base, scale)
        run_out = out / f"urban_scale_{scale:g}"
        result = run_closed_loop(
            cfg,
            scenario,
            mode="baseline",
            output_dir=run_out,
            baseline_mode=args.baseline,
        )
        points = _extract_mfd_points(result, cfg, scenario.name, scale)
        all_points.extend(points)
        runs.append({
            "scenario": scenario.name,
            "urban_scale": scale,
            "total_ttt": float(result.get("total_ttt", 0.0)),
            "urban_ttt": float(result.get("urban_ttt", 0.0)),
            "points": len(points),
            "path": str(run_out),
        })

    estimate = _estimate_n_crit(all_points)
    summary = {
        "config": args.config,
        "scenario": args.scenario,
        "baseline": args.baseline,
        "urban_scales": _parse_scales(args.urban_scales),
        "estimate": estimate,
        "runs": runs,
    }
    _write_csv(out / "mfd_points.csv", all_points)
    (out / "setpoint_calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_report(out / "setpoint_calibration_report.md", summary)
    print(
        "CALIBRATION "
        f"n_crit={estimate['n_crit_veh']:.3f} "
        f"max_production={estimate['max_production_veh_h']:.3f} "
        f"output={out}"
    )


if __name__ == "__main__":
    main()
