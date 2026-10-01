"""Read-only export of every machine-B result into small, Git-friendly CSV/JSON tables.

Output: docs/machine_b_results/{centers,canonical,carries,branches}.csv and summary.json.
Never runs physics; only reads completed result files under results/sdmpc_rl_machine_b_20260930.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MB = REPO / "results/sdmpc_rl_machine_b_20260930"
MA = REPO / "results/sdmpc_rl_multi_20260929/pilot_v1/center"
OUT = REPO / "docs/machine_b_results"
SCEN = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def multipliers(seed):
    rng = np.random.default_rng(seed)
    return [round(float(rng.uniform(.98, 1.02)), 6) for _ in range(3)]  # freeway_mainline, urban_boundary, ramp_arrival


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    centers = []
    for s in SCEN:
        b = read(MB / "center_repro_v1" / s / "episode_00_summary.json")
        a = read(MA / s / "episode_00_summary.json")
        centers.append(dict(scenario=s, machine_b_ttt=b["ttt"], machine_a_ttt=a["ttt"], b_minus_a=b["ttt"] - a["ttt"],
                            b_minus_a_pct=100 * (b["ttt"] - a["ttt"]) / a["ttt"], fallbacks=b["fallback_count"],
                            converged_intervals=b["converged_count"], terminal_inventory=b["terminal_inventory"]))
    canonical = []
    for run, label in (("return_canonical_b2", "return_mc_v1 (Codex model)"), ("canonical_b1_run1", "B1 latched-return perimeter bind"),
                       ("canonical_null_check2", "null actor (evaluator validation)")):
        for s in SCEN:
            folder = MB / run / s
            if run == "return_canonical_b2":
                f = folder / "summary.json"
                if not f.exists():
                    continue
                x = read(f)
                base = read(MB / "center_repro_v1" / s / "episode_00_summary.json")["ttt"]
                canonical.append(dict(run=run, policy=label, scenario=s, ttt=x["ttt"], center_ttt=base,
                                      improvement=base - x["ttt"], improvement_pct=100 * (base - x["ttt"]) / base,
                                      passes=(base - x["ttt"]) > max(1e-6, 1e-8 * base), fallbacks=x["fallback_count"]))
            else:
                f = folder / "summary.json"
                if not f.exists():
                    continue
                x = read(f)
                canonical.append(dict(run=run, policy=label, scenario=s, ttt=x["ttt"], center_ttt=x["center_ttt"],
                                      improvement=x["improvement"], improvement_pct=x["improvement_pct"],
                                      passes=x["passes"], fallbacks=len(x["fallback_steps"])))
    carries, branches = {}, []
    for probe in sorted(MB.glob("probe_*")):
        for slot in sorted(p for p in probe.iterdir() if p.is_dir()):
            cf = slot / "carry.json"
            if not cf.exists():
                continue
            c = read(cf)
            if c.get("truncated"):
                continue
            r = c["rows"]
            key = (c["scenario"], c["seed"])
            carries[key] = dict(scenario=c["scenario"], seed=c["seed"], multipliers=";".join(map(str, multipliers(c["seed"]))),
                                carry_ttt=c["ttt"], freeway_ttt=r[-1]["freeway_ttt"], urban_ttt=r[-1]["urban_ttt"],
                                fallback_steps=";".join(str(x["control_step"] + 1) for x in r if x["source"] == "reference_fallback"),
                                peak_urban_queue=max(x["urban_queue"] for x in r), peak_boundary_queue=max(x["boundary_queue"] for x in r),
                                peak_ramp_queue=max(x["ramp_queue"] for x in r))
            for bf in sorted((slot / "branches").glob("*.json")) if (slot / "branches").exists() else []:
                b = read(bf)
                rows = b["rows"]
                if b["option"]["name"] == "carry" or len(rows) + b["branch_step"] - 1 != 75:
                    continue
                opt = {k: v for k, v in b["option"].items() if k not in ("name",)}
                branches.append(dict(wave=probe.name, scenario=c["scenario"], seed=c["seed"], label=b["label"],
                                     option=b["option"]["name"], option_params=json.dumps(opt, sort_keys=True),
                                     branch_step=b["branch_step"], carry_ttt=c["ttt"], ttt=b["ttt"], delta=b["ttt"] - c["ttt"],
                                     delta_pct=100 * (b["ttt"] - c["ttt"]) / c["ttt"], fallbacks=b["fallbacks"],
                                     d_freeway=rows[-1]["freeway_ttt"] - r[-1]["freeway_ttt"],
                                     d_urban=rows[-1]["urban_ttt"] - r[-1]["urban_ttt"],
                                     peak_urban_queue=max(x["urban_queue"] for x in rows),
                                     peak_boundary_queue=max(x["boundary_queue"] for x in rows),
                                     peak_ramp_queue=max(x["ramp_queue"] for x in rows),
                                     source_file=bf.relative_to(REPO).as_posix(),
                                     source_sha256=hashlib.sha256(bf.read_bytes()).hexdigest()))
    n = dict(centers=write("centers.csv", centers), canonical=write("canonical.csv", canonical),
             carries=write("carries.csv", sorted(carries.values(), key=lambda d: (d["scenario"], d["seed"]))),
             branches=write("branches.csv", branches))
    (OUT / "summary.json").write_text(json.dumps(dict(rows=n, note="machine-B results; negative delta_pct = better than carry"),
                                                 indent=1), encoding="utf-8")
    print(n)


if __name__ == "__main__":
    main()
