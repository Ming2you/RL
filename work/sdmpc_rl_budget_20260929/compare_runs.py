"""Reconcile completed full episodes without treating development probes as wins."""
import argparse
import math
from pathlib import Path
from budget_runtime import read, save
from run_budget import RUN_FORMAT, POLICY_FORMAT, file_hash, training_profiles


def finite(value, label, nonnegative=True):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            (nonnegative and value < 0)):
        raise ValueError("Invalid finite value: " + label)
    return value


def same_cost(actual, expected, label):
    if abs(finite(actual, label, False) - finite(expected, label, False)) > 1e-8:
        raise ValueError("Accounting mismatch: " + label)


def validate_episode(episode, trace, settings, index):
    seed = settings["seeds"][index]
    profile = settings["profile_sha256"][index]
    training = settings["mode"] == "train"
    if (len(trace) != 75 or episode["episode"] != index or
            episode["simulation_seconds"] != 14400 or episode["control_steps"] != 75 or
            episode["full_run"] is not True or episode["evaluation"] is not (not training) or
            episode["exploration"] is not training or episode["training_seed"] != seed or
            episode["profile_sha256"] != profile):
        raise ValueError("Episode identity or full-run contract differs")
    total = finite(episode["warmup_ttt"], "warmup_ttt")
    times = []
    for i, row in enumerate(trace):
        if (row["episode"] != index or row["profile_sha256"] != profile or
                row["step"] != i + 5 or row["control_step"] != i or row["time_sec"] != (i + 6) * 180 or
                row["terminated"] is not (i == 74) or row["truncated"] is not False):
            raise ValueError("Trace identity, timeline, or terminal contract differs")
        total += finite(row["interval_ttt"], "interval_ttt")
        same_cost(row["total_ttt"], total, "cumulative TTT")
        same_cost(finite(row["freeway_ttt"], "freeway_ttt") + finite(row["urban_ttt"], "urban_ttt"),
                  total, "area TTT")
        same_cost(row["reward"], -row["interval_ttt"] / settings["reward_divisor"], "reward")
        if (row["execution_check"]["physical_control_valid"] is not True or
                row["execution_check"]["budget_feasible"] is not True or
                finite(row["selected_TTT"], "selected_TTT") > finite(row["reference_TTT"], "reference_TTT")):
            raise ValueError("Original execution/PFO guard failed")
        duration = finite(row["decision_wall_seconds"], "decision time")
        same_cost(duration, sum(finite(row[k], k) for k in (
            "forecast_wall_seconds", "observation_wall_seconds",
            "pfo_wall_seconds", "reference_wall_seconds", "lower_wall_seconds",
            "guard_wall_seconds", "actor_wall_seconds")), "decision components")
        same_cost(row["decision_cpu_seconds"], sum(finite(row[p + "_cpu_seconds"], p) for p in
                  ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor")),
                  "decision CPU components")
        times.append(duration)
    same_cost(finite(episode["ttt"], "ttt"), total, "full TTT")
    for area in ("freeway_ttt", "urban_ttt"):
        same_cost(episode[area], trace[-1][area], "final " + area)
    same_cost(episode["decision_wall_seconds"], sum(times), "decision sum")
    same_cost(episode["decision_cpu_seconds"], sum(r["decision_cpu_seconds"] for r in trace), "decision CPU sum")
    for key, expected in (
        ("fallback_count", sum(r["selection_source"] == "PFO_reference" for r in trace)),
        ("lower_candidate_solves", sum(r["lower_candidate_count"] for r in trace)),
        ("converged_count", sum(r["converged"] for r in trace)),
    ):
        same_cost(episode[key], expected, key)


def load_completed_run(folder, mode, seeds, source_pins=None, runtime=None, model_sha256=None):
    """One acceptance path for comparison and new/skipped pilot children."""
    folder = Path(folder)
    try:
        result, settings = read(folder / "completion.json"), read(folder / "settings.json")
        versions = read(folder / "runtime_versions.json")
        if (result["format"] != RUN_FORMAT or settings["format"] != RUN_FORMAT or
                result["status"] != "completed" or result["mode"] != mode or settings["mode"] != mode or
                not settings["run_id"] or result["run_id"] != settings["run_id"] or
                settings["seeds"] != seeds or len(result["episodes"]) != len(seeds) or
                len(settings["profile_sha256"]) != len(seeds)):
            raise ValueError("Completed run identity differs")
        if (settings["source_pins"] != result["source_pins"] or
                (source_pins is not None and source_pins != settings["source_pins"]) or
                not versions or versions != settings["runtime_versions"] or versions != result["runtime_versions"] or
                (runtime is not None and runtime != versions)):
            raise ValueError("Completed source/runtime contracts differ")
        if (settings["reward_divisor"], settings["gamma"], settings["total_seconds"],
                settings["warmup_steps"], settings["controlled_steps"]) != (100., 1., 14400, 5, 75):
            raise ValueError("Completed experiment contract differs")
        model_hash = settings["model_sha256"]
        if mode == "rl" and (not isinstance(model_hash, str) or len(model_hash) != 64 or
                             any(c not in "0123456789abcdef" for c in model_hash)):
            raise ValueError("RL model identity missing")
        if model_sha256 is not None and model_hash != model_sha256:
            raise ValueError("RL model identity differs")
        finite(result["elapsed_wall_seconds"], "elapsed wall time")
        traces = []
        for index, episode in enumerate(result["episodes"]):
            trace = read(folder / f"episode_{index:02d}_trace.json")
            validate_episode(episode, trace, settings, index)
            if episode != read(folder / f"episode_{index:02d}_summary.json"):
                raise ValueError("Completion and episode summary differ")
            traces.append(trace)
        schema = read(folder / "observation_schema.json")
        if mode == "train":
            manifest = read(folder / "model_hash.json")
            expected_contract = dict(source_pins=settings["source_pins"], runtime_versions=versions,
                environment_contract=settings["environment_contract"], observation_schema=schema,
                reward_divisor=settings["reward_divisor"], gamma=settings["gamma"])
            if (manifest["format"] != POLICY_FORMAT or manifest["frozen"] is not True or
                    manifest["training_run_id"] != settings["run_id"] or
                    manifest["train_seeds"] != seeds or
                    manifest["training_profiles"] != training_profiles(result["episodes"]) or
                    manifest["contract"] != expected_contract or
                    manifest["sha256"] != file_hash(folder / "model_final.pt")):
                raise ValueError("Training policy provenance differs")
        return result, settings, traces, schema
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Missing or malformed completed run: " + str(folder)) from exc


def queue_exposure(trace):
    totals, capacities = {}, {}
    for row in trace:
        estimate = row.get("queue_near_capacity_estimate")
        if (not estimate or estimate.get("method") != "interval_endpoint_sampled" or
                estimate.get("exact_substep_exposure") is not False or
                estimate.get("threshold_fraction") != .9 or estimate.get("interval_seconds") != 180 or
                estimate.get("duration_units") != "s" or estimate.get("capacity_units") != "veh"):
            raise ValueError("Queue exposure estimate missing or contract differs")
        current_capacities = {}
        for kind in ("ramp_queue", "boundary_queue"):
            values = estimate[kind]
            if set(values) != set(row["queue_state"][kind]):
                raise ValueError("Queue exposure identities differ")
            for key, item in values.items():
                queue = finite(item["queue_veh"], "queue vehicles")
                capacity = finite(item["capacity_veh"], "queue capacity")
                if capacity <= 0 or queue != row["queue_state"][kind][key]:
                    raise ValueError("Queue exposure endpoint differs")
                same_cost(item["near_capacity_seconds"], 180. * (queue >= .9 * capacity), "queue exposure")
                current_capacities[kind + "/" + key] = capacity
                durations = totals.setdefault(kind, {})
                durations[key] = durations.get(key, 0.) + item["near_capacity_seconds"]
        if capacities and capacities != current_capacities:
            raise ValueError("Queue exposure capacities differ")
        capacities = current_capacities
    return dict(method="interval_endpoint_sampled", exact_substep_exposure=False,
                threshold_fraction=.9, per_queue_seconds=totals, capacity_veh=capacities,
                total_queue_seconds=sum(sum(v.values()) for v in totals.values()),
                limitation="Endpoint estimate; summed queue-seconds are not elapsed time or exact substep exposure")


def compare(folders):
    if set(folders) != {"native", "center", "rl"} or len({Path(p).resolve() for p in folders.values()}) != 3:
        raise ValueError("Expected three distinct native/center/RL run folders")
    rows, common, run_ids = [], None, set()
    for mode, folder in folders.items():
        result, settings, traces, schema = load_completed_run(folder, mode, [None])
        if settings["run_id"] in run_ids:
            raise ValueError("Duplicate run identity")
        run_ids.add(settings["run_id"])
        episode = result["episodes"][0]
        trace = traces[0]
        if episode["ttt"] <= 0:
            raise ValueError("Comparison requires positive finite TTT")
        key = (episode["profile_sha256"], result["source_pins"], settings["runtime_versions"],
               settings["environment_contract"], schema, episode["warmup_ttt"])
        if common is None:
            common = key
        elif common != key:
            raise ValueError("Comparison contracts differ")
        queues = {}
        for row in trace:
            for kind, values in row["queue_state"].items():
                if not values:
                    continue
                for value in values.values():
                    finite(value, "queue state")
                q = queues.setdefault(kind, dict(max_queue_veh=0., max_spread_veh=0.))
                q["max_queue_veh"] = max(q["max_queue_veh"], *values.values())
                q["max_spread_veh"] = max(q["max_spread_veh"], max(values.values()) - min(values.values()))
        rows.append(dict(mode=mode, folder=str(folder), **episode,
                         queues_at_control_boundaries=queues,
                         queue_near_capacity_estimate=queue_exposure(trace),
                         model_sha256=settings["model_sha256"], run_id=settings["run_id"],
                         elapsed_wall_seconds=result["elapsed_wall_seconds"], cpu_mask=settings["cpu_mask"]))
    baseline = next(row for row in rows if row["mode"] == "native")["ttt"]
    center = next(row for row in rows if row["mode"] == "center")["ttt"]
    for row in rows:
        row["improvement_vs_native_pct"] = 100 * (1 - row["ttt"] / baseline)
        row["improvement_vs_center_pct"] = 100 * (1 - row["ttt"] / center)
    return dict(status="reconciled", rows=rows, goal_claim=False, controller_acceptance=False,
                scope="single_170_incident_development_benchmark_not_generalization",
                timing_scope="includes_PFO_actor_lower_guard; concurrent_single_CPU_runs_are_not_an_isolated_speedup_benchmark",
                legacy_DDQN_contract_reused=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--center", type=Path, required=True)
    p.add_argument("--rl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    save(args.output, compare({name: getattr(args, name) for name in ("native", "center", "rl")}))


if __name__ == "__main__":
    main()
