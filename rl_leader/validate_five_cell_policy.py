"""Apply the frozen five-cell acceptance criteria to a three-seed comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_SCENARIOS = {
    "sweet_155_w60",
    "sweet_170_w60",
    "sweet_170_incident_w60",
    "sweet_170_skew15_w60",
    "sweet_190_w60",
}
RECOVERY_GATED = {"sweet_170_incident_w60", "sweet_190_w60"}


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate(comparison: list[dict], phases: list[dict], support_limit: float) -> list[str]:
    failures = []
    by_scenario = {row["scenario"]: row for row in comparison}
    phase_by_scenario = {row["scenario"]: row for row in phases}
    missing = sorted(REQUIRED_SCENARIOS - set(by_scenario))
    if missing:
        failures.append(f"missing comparison scenarios: {missing}")
    for scenario in sorted(REQUIRED_SCENARIOS & set(by_scenario)):
        row = by_scenario[scenario]
        if row.get("all_complete") != "True" or row.get("all_valid") != "True":
            failures.append(f"{scenario} has incomplete or invalid evaluation seeds")
        gap = float(row["vs_pstack_percent"])
        if gap < -1.0:
            failures.append(f"{scenario} mean TTT is {gap:.3f}% versus P-Stack")
        support = float(row["support_out_mean"])
        if support > support_limit:
            failures.append(
                f"{scenario} support-out fraction {support:.4f} exceeds {support_limit:.4f}"
            )
    for scenario in sorted(RECOVERY_GATED):
        row = phase_by_scenario.get(scenario)
        if row is None:
            failures.append(f"{scenario} recovery comparison is missing")
            continue
        gap = float(row["rl_recovery_vs_pstack_percent"])
        if gap < -2.0:
            failures.append(f"{scenario} recovery TTT is {gap:.3f}% versus P-Stack")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--support-limit", type=float, default=0.10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    failures = validate(
        _rows(Path(args.comparison)),
        _rows(Path(args.phase)),
        float(args.support_limit),
    )
    result = {
        "format_version": "five_cell_policy_gate_v1",
        "passed": not failures,
        "criteria": {
            "mean_ttt_vs_pstack_min_percent": -1.0,
            "incident_190_recovery_vs_pstack_min_percent": -2.0,
            "support_out_fraction_max": float(args.support_limit),
            "all_complete": True,
            "all_valid": True,
        },
        "failures": failures,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
