import copy
import tempfile
import unittest
from pathlib import Path
from budget_runtime import read, save
from compare_runs import compare


PINS = {"snapshot": "frozen"}
VERSIONS = {"python": "test", "numpy": "test", "scipy": "test", "torch": "test", "PyYAML": "test"}
MODEL_HASH = "a" * 64
SCHEMA = dict(names=["state/time"], scale="fixed_physical_constants_no_eval_fitting", delay_bins=3)
ENVIRONMENT = dict(config_sha256="config", protocol_sha256="protocol")


def fixture(root, name, mode=None, seeds=None, model_hash=MODEL_HASH):
    mode = mode or name
    seeds = seeds if seeds is not None else [None]
    folder = root / name
    profiles = [f"profile-{seed}" for seed in seeds]
    settings = dict(format="sdmpc-budget-run-v2", run_id=name, mode=mode, seeds=seeds,
                    profile_sha256=profiles, policy_seed=6100,
                    model_sha256=model_hash if mode == "rl" else None,
                    reward_divisor=100., gamma=1., cpu_mask=1, total_seconds=14400,
                    warmup_steps=5, controlled_steps=75, source_pins=PINS,
                    runtime_versions=VERSIONS, environment_contract=ENVIRONMENT)
    episodes = []
    for episode, seed in enumerate(seeds):
        rows = [dict(episode=episode, profile_sha256=profiles[episode], step=i+5,
                     control_step=i, time_sec=(i+6)*180, terminated=i == 74,
                     truncated=False, interval_ttt=1., total_ttt=i+6., reward=-.01,
                     freeway_ttt=(i+6.)*.6, urban_ttt=(i+6.)*.4,
                     selected_TTT=2., reference_TTT=3., decision_wall_seconds=5.,
                     pfo_wall_seconds=1., reference_wall_seconds=1., actor_wall_seconds=1.,
                     lower_wall_seconds=1., guard_wall_seconds=1.,
                     forecast_wall_seconds=0., observation_wall_seconds=0.,
                     forecast_cpu_seconds=0., observation_cpu_seconds=0.,
                     pfo_cpu_seconds=1., reference_cpu_seconds=1., lower_cpu_seconds=1.,
                     guard_cpu_seconds=1., actor_cpu_seconds=1., decision_cpu_seconds=5.,
                     selection_source="lower_solution", lower_candidate_count=1, converged=True,
                     execution_check=dict(physical_control_valid=True, budget_feasible=True),
                     queue_state={"ramp_queue": {"a": 1., "b": 2.}, "boundary_queue": {"c": 9.}},
                     queue_near_capacity_estimate=dict(method="interval_endpoint_sampled",
                         exact_substep_exposure=False, threshold_fraction=.9, interval_seconds=180.,
                         duration_units="s", capacity_units="veh",
                         ramp_queue={"a": dict(queue_veh=1., capacity_veh=10., near_capacity_seconds=0.),
                                     "b": dict(queue_veh=2., capacity_veh=10., near_capacity_seconds=0.)},
                         boundary_queue={"c": dict(queue_veh=9., capacity_veh=10., near_capacity_seconds=180.)}))
                for i in range(75)]
        summary = dict(episode=episode, full_run=True, simulation_seconds=14400, control_steps=75,
                       ttt=80., warmup_ttt=5., freeway_ttt=48., urban_ttt=32.,
                       evaluation=mode != "train", exploration=mode == "train",
                       training_seed=seed, profile_sha256=profiles[episode], decision_wall_seconds=375.,
                       decision_cpu_seconds=375., fallback_count=0, lower_candidate_solves=75, converged_count=75)
        episodes.append(summary)
        save(folder / f"episode_{episode:02d}_trace.json", rows)
        save(folder / f"episode_{episode:02d}_summary.json", summary)
    result = dict(format=settings["format"], run_id=name, status="completed", mode=mode,
                  episodes=episodes, source_pins=PINS, runtime_versions=VERSIONS,
                  elapsed_wall_seconds=400.)
    save(folder / "completion.json", result)
    save(folder / "settings.json", settings)
    save(folder / "runtime_versions.json", VERSIONS)
    save(folder / "observation_schema.json", SCHEMA)
    return folder


class RunContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folders = {mode: fixture(self.root, mode) for mode in ("native", "center", "rl")}

    def change(self, filename, change):
        path = self.folders["rl"] / filename
        data = read(path)
        change(data)
        save(path, data)

    def test_valid_comparison_and_no_goal_claim(self):
        result = compare(self.folders)
        self.assertEqual(result["status"], "reconciled")
        self.assertFalse(result["goal_claim"])
        self.assertEqual(result["rows"][-1]["improvement_vs_native_pct"], 0.)
        self.assertEqual(result["rows"][-1]["freeway_ttt"], 48.)
        self.assertEqual(result["rows"][-1]["urban_ttt"], 32.)
        exposure = result["rows"][-1]["queue_near_capacity_estimate"]
        self.assertFalse(exposure["exact_substep_exposure"])
        self.assertEqual(exposure["per_queue_seconds"]["boundary_queue"]["c"], 13500.)
        self.assertEqual(exposure["total_queue_seconds"], 13500.)

    def test_reject_incorrect_exposure_estimate(self):
        self.change("episode_00_trace.json", lambda rows:
            rows[0]["queue_near_capacity_estimate"]["boundary_queue"]["c"].update(near_capacity_seconds=0.))
        with self.assertRaises(ValueError):
            compare(self.folders)

    def test_reject_duplicate_folders(self):
        with self.assertRaises(ValueError):
            compare(dict.fromkeys(self.folders, self.folders["native"]))

    def test_reject_swapped_modes(self):
        self.folders["native"], self.folders["rl"] = self.folders["rl"], self.folders["native"]
        with self.assertRaises(ValueError):
            compare(self.folders)

    def test_reject_duplicate_run_id(self):
        for name in ("settings.json", "completion.json"):
            self.change(name, lambda d: d.update(run_id="native"))
        with self.assertRaises(ValueError):
            compare(self.folders)

    def test_reject_missing_model_identity(self):
        self.change("settings.json", lambda d: d.update(model_sha256=None))
        with self.assertRaises(ValueError):
            compare(self.folders)

    def test_reject_runtime_drift(self):
        versions = dict(VERSIONS, scipy="changed")
        for name in ("settings.json", "completion.json"):
            self.change(name, lambda d: d.update(runtime_versions=versions))
        save(self.folders["rl"] / "runtime_versions.json", versions)
        with self.assertRaises(ValueError):
            compare(self.folders)

    def test_reject_invalid_trace_and_summary(self):
        trace_path = self.folders["rl"] / "episode_00_trace.json"
        result_path = self.folders["rl"] / "completion.json"
        original_trace, original_result = read(trace_path), read(result_path)
        changes = {
            "partial": lambda rows, result: rows.pop(),
            "step": lambda rows, result: rows[1].update(step=5),
            "control_step": lambda rows, result: rows[1].update(control_step=0),
            "time": lambda rows, result: rows[1].update(time_sec=1080),
            "episode": lambda rows, result: rows[1].update(episode=9),
            "profile": lambda rows, result: rows[1].update(profile_sha256="other"),
            "cumulative": lambda rows, result: rows[1].update(total_ttt=99.),
            "reward": lambda rows, result: rows[1].update(reward=0.),
            "area_cost": lambda rows, result: rows[1].update(urban_ttt=99.),
            "early_terminal": lambda rows, result: rows[1].update(terminated=True),
            "truncation": lambda rows, result: rows[1].update(truncated=True),
            "nan_total": lambda rows, result: result["episodes"][0].update(ttt=float("nan")),
            "nan_guard": lambda rows, result: rows[1].update(selected_TTT=float("nan")),
            "negative_cost": lambda rows, result: rows[1].update(interval_ttt=-1.),
            "physical": lambda rows, result: rows[1]["execution_check"].update(physical_control_valid=False),
            "guard": lambda rows, result: rows[1].update(selected_TTT=4.),
            "timing": lambda rows, result: rows[1].update(decision_wall_seconds=99.),
            "exploration": lambda rows, result: result["episodes"][0].update(exploration=True),
            "pins": lambda rows, result: result.update(source_pins={"changed": "pins"}),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                rows, result = copy.deepcopy(original_trace), copy.deepcopy(original_result)
                change(rows, result)
                save(trace_path, rows)
                save(result_path, result)
                with self.assertRaises(ValueError):
                    compare(self.folders)

    def test_reject_zero_baseline_cost(self):
        folder = self.folders["native"]
        result = read(folder / "completion.json")
        result["episodes"][0].update(ttt=0., warmup_ttt=0., freeway_ttt=0., urban_ttt=0.)
        rows = read(folder / "episode_00_trace.json")
        for row in rows:
            row.update(interval_ttt=0., total_ttt=0., freeway_ttt=0., urban_ttt=0., reward=0.)
        save(folder / "completion.json", result)
        save(folder / "episode_00_summary.json", result["episodes"][0])
        save(folder / "episode_00_trace.json", rows)
        with self.assertRaises(ValueError):
            compare(self.folders)


if __name__ == "__main__":
    unittest.main()
