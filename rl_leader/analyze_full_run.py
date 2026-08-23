"""Summarize full-horizon RL traces for TTT and lever-response diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINE_FILES = {
    "sweet_190_skew15_w60": {
        "NC": "ho_nc_skew.csv",
        "P-CENT": "ho_pcent_skew.csv",
        "PFO": "ho_pfo_skew.csv",
        "P-STACK": "ho_pstack_skew.csv",
    },
    "sweet_190_incident_w60": {
        "NC": "ho_nc_inc.csv",
        "P-CENT": "ho_pcent_inc.csv",
        "PFO": "ho_pfo_inc.csv",
        "P-STACK": "ho_pstack_inc.csv",
    },
}


def _spread(values):
    values = [float(value) for value in values]
    return max(values) - min(values) if values else 0.0


def _summary(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    mapping = meta["action_schema"]["freeway_vsl_keys"]
    linear_errors = []
    quadratic_errors = []
    phase = {name: {"steps": 0, "ttt": 0.0} for name in ("peak", "recovery")}
    support_count = 0
    validity = True
    offset_values = {owner: [] for owner in ("D", "F")}
    offset_skips = {owner: {} for owner in offset_values}
    vsl_values = {owner: [] for owner in mapping}
    vsl_density_ratio = {owner: [] for owner in mapping}
    vsl_candidate = {
        "invocations": 0,
        "trust_fallbacks": 0,
        "raw_candidate_count": [],
        "kept_candidate_count": [],
        "base_cost_spread": [],
        "smoothness_cost_spread": [],
        "price_cost_spread": [],
        "total_cost_spread": [],
    }
    candidate_variation = {key: [] for key in mapping.values()}
    for row in rows:
        time_sec = float(row["state_before"]["time_sec"])
        phase_name = "peak" if time_sec < 5220.0 else "recovery"
        phase[phase_name]["steps"] += 1
        phase[phase_name]["ttt"] += float(row["info"]["step_ttt"])
        support_count += len(row["support_out_names"])
        validity = validity and bool(row["info"]["validity_gate_pass"])
        for receipt in row["adapter_receipts"]:
            linear_errors.extend(
                abs(float(expected) - float(actual))
                for expected, actual in zip(receipt["expected_linear"], receipt["actual_linear"])
            )
            quadratic_errors.extend(
                abs(float(expected) - float(actual))
                for expected, actual in zip(receipt["expected_quadratic"], receipt["actual_quadratic"])
            )
        control = row["control_after"]
        for owner in offset_values:
            offset_values[owner].append(float(control["offsets"].get(owner, 0.0)))
            trace = row.get("candidate_trace", {}).get("offset", {}).get(owner, {})
            reason = trace.get("skipped_reason") or ("active" if trace else "missing_trace")
            offset_skips[owner][reason] = offset_skips[owner].get(reason, 0) + 1
        rho_crit = float(meta["rho_crit"])
        for owner, key in mapping.items():
            link, segment_text = key.split("__seg")
            segment = int(segment_text)
            value = control["vsl"].get(key, control["vsl"].get(link, 115.0))
            vsl_values[owner].append(float(value))
            densities = row["state_before"]["freeway_density"].get(link, [])
            density = float(densities[segment]) if segment < len(densities) else 0.0
            vsl_density_ratio[owner].append(density / rho_crit)
        for link, invocations in row.get("candidate_trace", {}).get("vsl", {}).items():
            for invocation in invocations:
                vsl_candidate["invocations"] += 1
                vsl_candidate["trust_fallbacks"] += int(invocation["trust_fallback_used"])
                vsl_candidate["raw_candidate_count"].append(invocation["raw_candidate_count"])
                kept = invocation.get("trust_kept_count")
                if kept is not None:
                    vsl_candidate["kept_candidate_count"].append(kept)
                candidates = invocation.get("candidates", [])
                vsl_candidate["base_cost_spread"].append(_spread(
                    candidate["base_cost"] for candidate in candidates
                ))
                vsl_candidate["smoothness_cost_spread"].append(_spread(
                    candidate["smoothness_cost"] for candidate in candidates
                ))
                vsl_candidate["price_cost_spread"].append(_spread(
                    candidate["linear_cost"]
                    + candidate["quadratic_cost"]
                    + candidate["cross_cost"]
                    for candidate in candidates
                ))
                vsl_candidate["total_cost_spread"].append(_spread(
                    candidate["total_cost"] for candidate in candidates
                ))
                for key in candidate_variation:
                    if not key.startswith(f"{link}__seg"):
                        continue
                    segment = int(key.split("__seg", 1)[1])
                    candidate_variation[key].append(_spread(
                        candidate["first_vsl"][segment]
                        for candidate in candidates
                        if segment < len(candidate["first_vsl"])
                    ) > 1.0e-9)

    def response_stats(values, density=None):
        arr = np.asarray(values, dtype=float)
        result = {
            "unique": sorted(float(value) for value in np.unique(arr)),
            "changes": int(np.sum(np.abs(np.diff(arr)) > 1.0e-9)) if arr.size else 0,
        }
        if density is not None:
            rho = np.asarray(density, dtype=float)
            active = rho >= 0.95
            result["rho_ratio_max"] = float(rho.max()) if rho.size else 0.0
            result["rho_ge_0_95_steps"] = int(active.sum())
            result["unique_when_rho_ge_0_95"] = sorted(
                float(value) for value in np.unique(arr[active])
            ) if active.any() else []
        return result

    invocations = max(vsl_candidate.pop("invocations"), 0)
    fallbacks = vsl_candidate.pop("trust_fallbacks")
    candidate_summary = {
        "invocations": invocations,
        "trust_fallback_fraction": float(fallbacks / invocations) if invocations else 0.0,
    }
    for name, values in vsl_candidate.items():
        candidate_summary[name] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
        }
    candidate_summary["priced_segment_variation_fraction"] = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in candidate_variation.items()
    }
    return {
        "trace": str(path),
        "checkpoint": meta["checkpoint"],
        "response_contract": meta.get("response_contract", "legacy"),
        "checkpoint_response_contract": meta.get("checkpoint_response_contract", "legacy"),
        "seed": meta["seed"],
        "scenario": meta["scenario"],
        "steps": len(rows),
        "final_time_sec": (
            float(rows[-1]["state_before"]["time_sec"] + meta["control_interval_sec"])
            if rows else 0.0
        ),
        "total_ttt": float(sum(float(row["info"]["step_ttt"]) for row in rows)),
        "throughput_veh": float(sum(float(row["info"]["throughput_veh"]) for row in rows)),
        "phase": phase,
        "validity_pass": bool(validity),
        "support_out_fraction": float(support_count / max(
            len(rows) * int(meta.get("action_schema", {}).get("dimension", 47)), 1,
        )),
        "adapter": {
            "max_linear_abs_error": max(linear_errors, default=0.0),
            "max_quadratic_abs_error": max(quadratic_errors, default=0.0),
        },
        "offset_response": {
            owner: {**response_stats(values), "candidate_skip_reasons": offset_skips[owner]}
            for owner, values in offset_values.items()
        },
        "vsl_response": {
            owner: response_stats(vsl_values[owner], vsl_density_ratio[owner])
            for owner in mapping
        },
        "vsl_candidates": candidate_summary,
    }


def _load_baselines(directory: Path, scenarios, skip_steps: int):
    result = {}
    for scenario in scenarios:
        scenario_result = {}
        for controller, filename in BASELINE_FILES.get(scenario, {}).items():
            path = directory / filename
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))[skip_steps:]
            scenario_result[controller] = {
                "steps": len(rows),
                "total_ttt": float(sum(float(row["step_total_ttt"]) for row in rows)),
            }
        if scenario_result:
            result[scenario] = scenario_result
    return result


def _aggregate(summaries, baselines):
    grouped = defaultdict(list)
    for item in summaries:
        grouped[item["scenario"]].append(item)
    result = {}
    for scenario, items in grouped.items():
        ttt = np.asarray([item["total_ttt"] for item in items], dtype=float)
        peak = np.asarray([item["phase"]["peak"]["ttt"] for item in items], dtype=float)
        recovery = np.asarray([item["phase"]["recovery"]["ttt"] for item in items], dtype=float)
        throughput = np.asarray([item["throughput_veh"] for item in items], dtype=float)
        mean_ttt = float(ttt.mean())
        result[scenario] = {
            "seeds": sorted(int(item["seed"]) for item in items),
            "runs": len(items),
            "total_ttt_mean": mean_ttt,
            "total_ttt_std": float(ttt.std()),
            "total_ttt_min": float(ttt.min()),
            "total_ttt_max": float(ttt.max()),
            "peak_ttt_mean": float(peak.mean()),
            "recovery_ttt_mean": float(recovery.mean()),
            "throughput_veh_mean": float(throughput.mean()),
            "throughput_veh_std": float(throughput.std()),
            "validity_all_pass": all(item["validity_pass"] for item in items),
            "support_out_fraction_mean": float(np.mean([
                item["support_out_fraction"] for item in items
            ])),
            "baseline_improvement_percent": {
                controller: 100.0 * (values["total_ttt"] - mean_ttt) / values["total_ttt"]
                for controller, values in baselines.get(scenario, {}).items()
                if values["steps"] == items[0]["steps"]
            },
        }
    return result


def _write_csv(path: Path, summaries):
    rows = []
    for item in summaries:
        variations = item["vsl_candidates"]["priced_segment_variation_fraction"]
        rows.append({
            "seed": item["seed"],
            "scenario": item["scenario"],
            "response_contract": item["response_contract"],
            "checkpoint_response_contract": item["checkpoint_response_contract"],
            "steps": item["steps"],
            "final_time_sec": item["final_time_sec"],
            "total_ttt": item["total_ttt"],
            "throughput_veh": item["throughput_veh"],
            "peak_ttt": item["phase"]["peak"]["ttt"],
            "recovery_ttt": item["phase"]["recovery"]["ttt"],
            "validity_pass": item["validity_pass"],
            "support_out_fraction": item["support_out_fraction"],
            "adapter_max_linear_abs_error": item["adapter"]["max_linear_abs_error"],
            "adapter_max_quadratic_abs_error": item["adapter"]["max_quadratic_abs_error"],
            "D_offset_disabled_steps": item["offset_response"]["D"][
                "candidate_skip_reasons"
            ].get("ramp_offset_disabled", 0),
            "F_offset_disabled_steps": item["offset_response"]["F"][
                "candidate_skip_reasons"
            ].get("ramp_offset_disabled", 0),
            "vsl_price_cost_spread_mean": item["vsl_candidates"][
                "price_cost_spread"
            ]["mean"],
            "vsl_smoothness_cost_spread_mean": item["vsl_candidates"][
                "smoothness_cost_spread"
            ]["mean"],
            "vsl_trust_fallback_fraction": item["vsl_candidates"][
                "trust_fallback_fraction"
            ],
            "priced_segment_variation_max": max(variations.values(), default=0.0),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--out", default="results/full_run_diagnostic_summary.json")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--baseline-dir", default="data/holdout")
    parser.add_argument("--baseline-skip-steps", type=int, default=5)
    args = parser.parse_args(argv)
    summaries = [_summary(Path(path)) for path in args.traces]
    baselines = _load_baselines(
        Path(args.baseline_dir),
        {item["scenario"] for item in summaries},
        args.baseline_skip_steps,
    )
    report = {
        "runs": summaries,
        "aggregate": _aggregate(summaries, baselines),
        "baselines": baselines,
        "baseline_skip_steps": args.baseline_skip_steps,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.csv_out:
        _write_csv(Path(args.csv_out), summaries)
    for item in summaries:
        print(
            f"seed={item['seed']} scenario={item['scenario']} steps={item['steps']} "
            f"ttt={item['total_ttt']:.3f} support={item['support_out_fraction']:.3f}",
            flush=True,
        )
    print(f"saved diagnostic summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
