"""Aggregate three-seed RL results against warmup-aligned five-cell baselines."""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


CELLS = (
    ("sweet_155_w60", "155"),
    ("sweet_170_w60", "170"),
    ("sweet_170_incident_w60", "170 incident"),
    ("sweet_170_skew15_w60", "170 skew"),
    ("sweet_190_w60", "190"),
)
BASELINE_CONTROLLERS = {
    "nc": "NO-CONTROL",
    "pfo": "WU-FAITHFUL-FOLLOWER",
    "pstack": "P-STACK-WU-FAITHFUL-ALLPRICE-JOINT",
}


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def baseline_metrics(root: Path, kind: str, scenario: str) -> dict:
    path = root / f"{kind}_{scenario}" / BASELINE_CONTROLLERS[kind] / "run_log.csv"
    rows = read_rows(path)
    policy = [row for row in rows if int(row["step"]) >= 5]
    peak = sum(float(row["step_total_ttt"]) for row in policy if int(row["step"]) <= 28)
    recovery = sum(float(row["step_total_ttt"]) for row in policy if int(row["step"]) >= 29)
    return {"ttt": peak + recovery, "peak": peak, "recovery": recovery}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Comma-separated CSV paths or glob patterns")
    parser.add_argument("--baseline-root", default="results/five_cell_baselines")
    parser.add_argument("--out", default="results/five_cell_targeted_v1/comparison.csv")
    args = parser.parse_args(argv)

    patterns = [value.strip() for value in args.eval.split(",") if value.strip()]
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise SystemExit("no evaluation files matched")
    evaluations = [row for path in files for row in read_rows(path)]
    output_rows = []
    phase_rows = []
    for scenario, label in CELLS:
        rl = [row for row in evaluations if row["scenario"] == scenario and row["mask"] == "RL-FULL"]
        if len(rl) != 3:
            raise ValueError(f"expected 3 RL seeds for {scenario}, got {len(rl)}")
        if not all(row["complete"] == "True" and row["validity_gate_pass"] == "True" for row in rl):
            raise ValueError(f"incomplete or invalid RL evaluation for {scenario}")
        baseline = {
            kind: baseline_metrics(Path(args.baseline_root), kind, scenario)
            for kind in BASELINE_CONTROLLERS
        }
        ttt = np.asarray([float(row["total_ttt"]) for row in rl])
        peak = np.asarray([float(row["peak_ttt"]) for row in rl])
        recovery = np.asarray([float(row["recovery_ttt"]) for row in rl])
        support = np.asarray([float(row["support_out_fraction"]) for row in rl])
        row = {
            "scenario": scenario,
            "label": label,
            **{f"{kind}_ttt": values["ttt"] for kind, values in baseline.items()},
            "rl_mean_ttt": float(ttt.mean()),
            "rl_std_ttt": float(ttt.std()),
            "rl_min_ttt": float(ttt.min()),
            "rl_max_ttt": float(ttt.max()),
            **{f"rl_seed{int(item['seed'])}_ttt": float(item["total_ttt"]) for item in rl},
            **{
                f"vs_{kind}_percent": 100.0 * (values["ttt"] - ttt.mean()) / values["ttt"]
                for kind, values in baseline.items()
            },
            "support_out_mean": float(support.mean()),
            "throughput_mean": float(np.mean([float(item["throughput_veh"]) for item in rl])),
            "all_complete": True,
            "all_valid": True,
        }
        output_rows.append(row)
        phase_rows.append({
            "scenario": scenario,
            "label": label,
            "pfo_peak_ttt": baseline["pfo"]["peak"],
            "pfo_recovery_ttt": baseline["pfo"]["recovery"],
            "pstack_peak_ttt": baseline["pstack"]["peak"],
            "pstack_recovery_ttt": baseline["pstack"]["recovery"],
            "rl_peak_mean": float(peak.mean()),
            "rl_recovery_mean": float(recovery.mean()),
            "rl_peak_vs_pstack_percent": 100.0 * (baseline["pstack"]["peak"] - peak.mean()) / baseline["pstack"]["peak"],
            "rl_recovery_vs_pstack_percent": 100.0 * (baseline["pstack"]["recovery"] - recovery.mean()) / baseline["pstack"]["recovery"],
        })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with output.with_name("phase_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    output.with_suffix(".json").write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    print(f"saved five-cell comparison -> {output}")


if __name__ == "__main__":
    main()
