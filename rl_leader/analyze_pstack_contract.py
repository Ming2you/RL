"""Audit P-Stack baseline channels that are absent or different in the RL contract."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "sweet_155_w60",
    "sweet_170_w60",
    "sweet_170_incident_w60",
    "sweet_170_skew15_w60",
    "sweet_190_w60",
)
MERGE_VSL = {
    "FW_E__seg3",
    "FW_E__seg5",
    "FW_W__seg3",
    "FW_W__seg5",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _phase(step: int) -> str:
    return "peak" if float(step) * 180.0 < 5220.0 else "recovery"


def _count_values(
    rows: list[dict[str, str]],
    keys: list[str],
    *,
    predicate,
) -> dict:
    total = nonzero = 0
    by_phase = {
        "peak": {"total": 0, "nonzero": 0},
        "recovery": {"total": 0, "nonzero": 0},
    }
    for row in rows:
        step = int(float(row["step"]))
        phase = _phase(step)
        for key in keys:
            value = _number(row, key)
            if value is None:
                continue
            total += 1
            by_phase[phase]["total"] += 1
            if predicate(value):
                nonzero += 1
                by_phase[phase]["nonzero"] += 1
    result = {
        "opportunities": total,
        "active": nonzero,
        "active_fraction": float(nonzero / total) if total else 0.0,
        "by_phase": by_phase,
    }
    for values in by_phase.values():
        values["active_fraction"] = (
            float(values["nonzero"] / values["total"]) if values["total"] else 0.0
        )
    return result


def _scenario_files(root: Path, scenario: str) -> tuple[Path, Path]:
    directory = root / f"pstack_{scenario}"
    decisions = list(directory.rglob("decision_diagnostics.csv"))
    controls = list(directory.rglob("control_timeseries.csv"))
    if len(decisions) != 1 or len(controls) != 1:
        raise RuntimeError(f"expected one P-Stack trace for {scenario}")
    return decisions[0], controls[0]


def analyze_scenario(root: Path, scenario: str) -> dict:
    decision_path, control_path = _scenario_files(root, scenario)
    decision_rows = [
        row for row in _read_csv(decision_path)
        if int(float(row["step"])) >= 5
    ]
    control_rows = [
        row for row in _read_csv(control_path)
        if int(float(row["step"])) >= 5
    ]
    columns = set(decision_rows[0]) if decision_rows else set()
    vsl_price_columns = sorted(
        key for key in columns if key.startswith("wu_b3_vsl_price_FW_")
    )
    merge_vsl_price_columns = [
        key for key in vsl_price_columns
        if key.removeprefix("wu_b3_vsl_price_") in MERGE_VSL
    ]
    nonmerge_vsl_price_columns = [
        key for key in vsl_price_columns
        if key.removeprefix("wu_b3_vsl_price_") not in MERGE_VSL
    ]
    signal_price_columns = sorted(
        key for key in columns
        if key.startswith("wu_b2_price_")
        and key.removeprefix("wu_b2_price_") in {"A", "B", "C", "D", "F"}
    )
    offset_price_columns = sorted(
        key for key in columns if key.startswith("wu_f3_offset_price_")
    )
    metering_price_columns = sorted(
        key for key in columns if key.startswith("wu_b3_meter_price_R_")
    )
    vsl_control_columns = sorted(
        key for key in control_rows[0]
        if key.startswith("vsl_FW_") and "_seg" in key
    ) if control_rows else []
    merge_vsl_control_columns = [
        key for key in vsl_control_columns
        if key.replace("vsl_", "").replace("_seg", "__seg") in MERGE_VSL
    ]
    nonmerge_vsl_control_columns = [
        key for key in vsl_control_columns if key not in merge_vsl_control_columns
    ]

    supervisor_rows = [row for row in decision_rows if _number(row, "sup_pick_pfo") is not None]
    supervisor = _count_values(
        supervisor_rows,
        ["sup_pick_pfo"],
        predicate=lambda value: value >= 0.5,
    )
    d_f_offset = _count_values(
        control_rows,
        ["offset_D", "offset_F"],
        predicate=lambda value: abs(value) > 1.0e-9,
    )
    d_f_moves = 0
    previous = None
    for row in control_rows:
        current = (_number(row, "offset_D") or 0.0, _number(row, "offset_F") or 0.0)
        if previous is not None:
            d_f_moves += sum(abs(a - b) > 1.0e-9 for a, b in zip(current, previous))
        previous = current

    return {
        "scenario": scenario,
        "policy_steps": len(decision_rows),
        "supervisor_pfo": supervisor,
        "signal_price": _count_values(
            decision_rows, signal_price_columns, predicate=lambda value: abs(value) > 1.0e-9,
        ),
        "offset_price": _count_values(
            decision_rows, offset_price_columns, predicate=lambda value: abs(value) > 1.0e-9,
        ),
        "offset_price_owners": [key.removeprefix("wu_f3_offset_price_") for key in offset_price_columns],
        "metering_price": _count_values(
            decision_rows, metering_price_columns, predicate=lambda value: abs(value) > 1.0e-9,
        ),
        "merge_vsl_price": _count_values(
            decision_rows, merge_vsl_price_columns, predicate=lambda value: abs(value) > 1.0e-9,
        ),
        "nonmerge_vsl_price": _count_values(
            decision_rows, nonmerge_vsl_price_columns, predicate=lambda value: abs(value) > 1.0e-9,
        ),
        "merge_vsl_below_115": _count_values(
            control_rows, merge_vsl_control_columns, predicate=lambda value: value < 115.0 - 1.0e-9,
        ),
        "nonmerge_vsl_below_115": _count_values(
            control_rows, nonmerge_vsl_control_columns, predicate=lambda value: value < 115.0 - 1.0e-9,
        ),
        "d_f_offset_nonzero": d_f_offset,
        "d_f_offset_move_count": int(d_f_moves),
        "source": {
            "decision_diagnostics": decision_path.relative_to(ROOT).as_posix(),
            "control_timeseries": control_path.relative_to(ROOT).as_posix(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="results/five_cell_baselines")
    parser.add_argument(
        "--output",
        default="results/pstack_gap_diagnosis_v1/baseline_contract_attribution.json",
    )
    args = parser.parse_args()
    rows = [analyze_scenario(ROOT / args.baseline_root, scenario) for scenario in SCENARIOS]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "format_version": "pstack_baseline_contract_attribution_v1",
        "warmup_policy_step": 5,
        "scenarios": rows,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        row["scenario"]: {
            "supervisor_pfo": row["supervisor_pfo"]["active"],
            "nonmerge_vsl_price": row["nonmerge_vsl_price"]["active"],
            "nonmerge_vsl_below_115": row["nonmerge_vsl_below_115"]["active"],
            "d_f_offset_move_count": row["d_f_offset_move_count"],
        }
        for row in rows
    }, indent=2))


if __name__ == "__main__":
    main()
