# Stage 2 runner — traced closed loop + event 검출 + frozen replay counterfactual (plan §3~§8)
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.stage2_mechanism import (
    detect_and_evaluate_events,
    run_traced_closed_loop,
    summarize_events,
)
from src.models.demand import load_scenarios
from src.models.state import ExperimentConfig


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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default="src/config/scenarios.yaml")
    parser.add_argument("--scenario", default="peak_demand")
    parser.add_argument("--T-total", type=float, default=None)
    parser.add_argument("--output", default="post_analysis/stage2")
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.T_total is not None:
        overrides["simulation"] = {"T_total": args.T_total}
    if args.seed is not None:
        overrides.setdefault("simulation", {})["random_seed"] = args.seed
    cfg = ExperimentConfig.from_file(args.config, overrides)
    scenario = load_scenarios(args.scenarios_config)[args.scenario]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    traces = run_traced_closed_loop(cfg, scenario)
    events = detect_and_evaluate_events(cfg, scenario, traces, max_events_per_control=args.max_events)

    event_rows = []
    for e in events:
        row = asdict(e)
        row["mediator_change"] = json.dumps(row["mediator_change"], ensure_ascii=False)
        event_rows.append(row)
    _write_csv(out / "control_event_catalog.csv", event_rows)
    for control_name, filename in (
        ("allocation_green", "allocation_green_events.csv"),
        ("offset", "offset_events.csv"),
        ("vsl", "vsl_events.csv"),
        ("metering", "metering_events.csv"),
    ):
        _write_csv(out / filename, [r for r in event_rows if r["control"] == control_name])
    summary = summarize_events(events)
    _write_csv(out / "mechanism_summary.csv", summary)
    (out / "summary.json").write_text(
        json.dumps({"scenario": args.scenario, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for row in summary:
        print(
            f"{row['control']}: challenged={row['challenged_event_count']} "
            f"mech={row['mechanism_success_rate']} dir={row['directional_accuracy']} "
            f"gain={row['mean_outcome_gain_veh_h']}"
        )
    print(f"STAGE2 scenario={args.scenario} output={out}")


if __name__ == "__main__":
    main()
