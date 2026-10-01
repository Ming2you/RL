"""Synthetic completed-run fixtures; no runtime boot, solvers, or simulations."""
import copy
import tempfile
import unittest
from pathlib import Path

from budget_runtime import read, save
from compare_runs import compare, load_completed_run, validate_episode
import run_budget


PINS = {"snapshot": "frozen"}
VERSIONS = {"python": "test", "numpy": "test", "scipy": "test", "torch": "test", "PyYAML": "test"}
MODEL_HASH = "a" * 64
SCHEMA = dict(names=["state/time"], scale="fixed_physical_constants_no_eval_fitting", delay_bins=3)
ENVIRONMENT = dict(config_sha256="config", options_sha256="options", protocol_sha256="protocol",
                   coordinator=dict(version=3, guard_mode="physical",
                                    action_anchor="previous_executed_budget"))


def fixture(root, name, mode=None, seeds=None, model_hash=MODEL_HASH, guard_mode="physical"):
    mode = mode or name
    seeds = seeds if seeds is not None else [None]
    folder = root / name
    profiles = [f"profile-{seed}" for seed in seeds]
    environment = copy.deepcopy(ENVIRONMENT)
    environment["coordinator"]["guard_mode"] = guard_mode
    settings = dict(format=run_budget.RUN_FORMAT, run_id=name, mode=mode, seeds=seeds,
                    profile_sha256=profiles, policy_seed=6200,
                    model_sha256=model_hash if mode == "rl" else None,
                    reward_divisor=100., gamma=1., cpu_mask=1, total_seconds=14400,
                    warmup_steps=5, controlled_steps=75, source_pins=PINS,
                    runtime_versions=VERSIONS, environment_contract=environment,
                    guard_mode=guard_mode, updates_per_interval=20, batch_size=32)
    episodes = []
    for episode, seed in enumerate(seeds):
        rows = []
        for i in range(75):
            source = "pfo_initial" if i == 0 else "pfo_recovery" if i == 30 else "previous"
            pfo = float(source != "previous")
            rows.append(dict(episode=episode, profile_sha256=profiles[episode], step=i + 5,
                control_step=i, time_sec=(i + 6) * 180, terminated=i == 74,
                truncated=False, interval_ttt=1., total_ttt=i + 6., reward=-.01,
                freeway_ttt=(i + 6.) * .6, urban_ttt=(i + 6.) * .4,
                selected_TTT=2., reference_TTT=3., decision_wall_seconds=4. + pfo,
                pfo_wall_seconds=pfo, reference_wall_seconds=1., actor_wall_seconds=1.,
                lower_wall_seconds=1., guard_wall_seconds=1.,
                forecast_wall_seconds=0., observation_wall_seconds=0.,
                forecast_cpu_seconds=0., observation_cpu_seconds=0.,
                pfo_cpu_seconds=pfo, reference_cpu_seconds=1., lower_cpu_seconds=1.,
                guard_cpu_seconds=1., actor_cpu_seconds=1., decision_cpu_seconds=4. + pfo,
                reference_source=source, pfo_calls=int(pfo), reference_recovery=source == "pfo_recovery",
                guard_mode=guard_mode, h3_guard_enabled=guard_mode == "h3",
                selection_source="lower_solution", lower_candidate_count=1, converged=True,
                B_requested=[[-20., 1500.]], B_executed=[-20., 1500.], G_achieved=[-25., 1000.],
                execution_check=dict(physical_control_valid=True, budget_feasible=True),
                queue_state={"ramp_queue": {"a": 1., "b": 2.}, "boundary_queue": {"c": 9.}},
                queue_near_capacity_estimate=dict(method="interval_endpoint_sampled",
                    exact_substep_exposure=False, threshold_fraction=.9, interval_seconds=180.,
                    duration_units="s", capacity_units="veh",
                    ramp_queue={"a": dict(queue_veh=1., capacity_veh=10., near_capacity_seconds=0.),
                                "b": dict(queue_veh=2., capacity_veh=10., near_capacity_seconds=0.)},
                    boundary_queue={"c": dict(queue_veh=9., capacity_veh=10., near_capacity_seconds=180.)})))
        summary = dict(episode=episode, full_run=True, simulation_seconds=14400, control_steps=75,
                       ttt=80., warmup_ttt=5., freeway_ttt=48., urban_ttt=32.,
                       evaluation=mode != "train", exploration=mode == "train",
                       training_seed=seed, profile_sha256=profiles[episode], decision_wall_seconds=302.,
                       decision_cpu_seconds=302., fallback_count=0, lower_candidate_solves=75,
                       converged_count=75, pfo_calls=2, pfo_recovery_count=1)
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
        temp = tempfile.TemporaryDirectory(prefix="carry-contract-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folders = {mode: fixture(self.root, mode) for mode in ("center", "rl")}

    def change(self, filename, change):
        path = self.folders["rl"] / filename
        data = read(path)
        change(data)
        save(path, data)

    def test_carry_formats_are_separate_from_v1(self):
        self.assertEqual(run_budget.RUN_FORMAT, "sdmpc-carry-run-v1")
        self.assertEqual(run_budget.POLICY_FORMAT, "sdmpc-carry-policy-v1")
        self.assertEqual(run_budget.CHECKPOINT_FORMAT, "sdmpc-carry-checkpoint-v1")

    def test_valid_center_rl_comparison_and_no_goal_claim(self):
        result = compare(self.folders)
        self.assertEqual(result["status"], "reconciled")
        self.assertFalse(result["goal_claim"])
        self.assertFalse(result["controller_acceptance"])
        self.assertEqual([row["mode"] for row in result["rows"]], ["center", "rl"])
        self.assertEqual(result["rows"][-1]["improvement_vs_center_pct"], 0.)
        self.assertEqual(result["rows"][-1]["pfo_calls"], 2)
        exposure = result["rows"][-1]["queue_near_capacity_estimate"]
        self.assertFalse(exposure["exact_substep_exposure"])
        self.assertEqual(exposure["total_queue_seconds"], 13500.)

    def test_physical_guard_accepts_higher_h3_ttt_but_h3_rejects(self):
        for guard in ("physical", "h3"):
            with self.subTest(guard=guard):
                folder = fixture(self.root, "guard-" + guard, "rl", guard_mode=guard)
                rows = read(folder / "episode_00_trace.json")
                rows[1]["selected_TTT"] = 4.
                save(folder / "episode_00_trace.json", rows)
                if guard == "physical":
                    load_completed_run(folder, "rl", [None])
                else:
                    with self.assertRaisesRegex(ValueError, "guard"):
                        load_completed_run(folder, "rl", [None])

    def test_fallback_keeps_failed_requested_budget_distinct(self):
        folder = self.folders["rl"]
        rows = read(folder / "episode_00_trace.json")
        rows[1].update(selection_source="reference_fallback", converged=False,
                       B_requested=[[-50., 500.]], B_executed=[-20., 1500.])
        summary = read(folder / "episode_00_summary.json")
        summary.update(fallback_count=1, converged_count=74)
        settings = read(folder / "settings.json")
        validate_episode(summary, rows, settings, 0)
        self.assertNotEqual(rows[1]["B_requested"][0], rows[1]["B_executed"])

    def test_reject_native_missing_duplicate_or_swapped_folders(self):
        alternatives = [dict(self.folders, native=self.folders["center"]),
                        {"rl": self.folders["rl"]},
                        dict.fromkeys(self.folders, self.folders["center"]),
                        {"center": self.folders["rl"], "rl": self.folders["center"]}]
        for folders in alternatives:
            with self.subTest(folders=folders), self.assertRaises(ValueError):
                compare(folders)

    def test_reject_reused_run_identity(self):
        for name in ("settings.json", "completion.json"):
            self.change(name, lambda data: data.update(run_id="center"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare(self.folders)

    def test_reject_training_and_guard_contract_drift(self):
        path = self.folders["rl"] / "settings.json"
        original = read(path)
        for key, value in (("updates_per_interval", 1), ("batch_size", 64),
                           ("guard_mode", "invalid"), ("guard_mode", "h3"),
                           ("gamma", .99), ("reward_divisor", 1.),
                           ("format", "sdmpc-budget-run-v2")):
            with self.subTest(key=key, value=value):
                save(path, dict(original, **{key: value}))
                with self.assertRaises(ValueError):
                    load_completed_run(self.folders["rl"], "rl", [None])

    def test_reject_missing_invalid_or_unexpected_model_identity(self):
        for value in (None, "", "f" * 63, "z" * 64, "b" * 64):
            with self.subTest(value=value):
                self.change("settings.json", lambda data: data.update(model_sha256=value))
                with self.assertRaisesRegex(ValueError, "model identity"):
                    load_completed_run(self.folders["rl"], "rl", [None], model_sha256=MODEL_HASH)

    def test_reject_source_runtime_and_profile_drift(self):
        folder = self.folders["rl"]
        with self.assertRaisesRegex(ValueError, "source/runtime"):
            load_completed_run(folder, "rl", [None], source_pins={"snapshot": "other"})
        with self.assertRaisesRegex(ValueError, "source/runtime"):
            load_completed_run(folder, "rl", [None], runtime=dict(VERSIONS, torch="other"))
        self.change("settings.json", lambda data: data.update(profile_sha256=["other"]))
        with self.assertRaisesRegex(ValueError, "identity"):
            compare(self.folders)

    def test_reject_comparison_contract_drift_even_when_each_run_is_consistent(self):
        for field in ("runtime", "source", "profile", "environment", "schema"):
            with self.subTest(field=field):
                folder = fixture(self.root, "rl")
                settings, result = read(folder / "settings.json"), read(folder / "completion.json")
                rows = read(folder / "episode_00_trace.json")
                if field == "runtime":
                    versions = dict(VERSIONS, scipy="changed")
                    settings["runtime_versions"] = result["runtime_versions"] = versions
                    save(folder / "runtime_versions.json", versions)
                elif field == "source":
                    settings["source_pins"] = result["source_pins"] = {"snapshot": "changed"}
                elif field == "profile":
                    settings["profile_sha256"] = ["changed"]
                    result["episodes"][0]["profile_sha256"] = "changed"
                    for row in rows:
                        row["profile_sha256"] = "changed"
                elif field == "environment":
                    settings["environment_contract"]["options_sha256"] = "changed"
                else:
                    save(folder / "observation_schema.json", dict(SCHEMA, names=["different"]))
                save(folder / "settings.json", settings)
                save(folder / "completion.json", result)
                save(folder / "episode_00_summary.json", result["episodes"][0])
                save(folder / "episode_00_trace.json", rows)
                with self.assertRaisesRegex(ValueError, "Comparison contracts"):
                    compare(self.folders)

    def test_reject_invalid_trace_accounting_sources_and_summary(self):
        folder = self.folders["rl"]
        original_rows = read(folder / "episode_00_trace.json")
        original_summary = read(folder / "episode_00_summary.json")
        settings = read(folder / "settings.json")
        changes = {
            "partial": lambda rows, summary: rows.pop(),
            "step": lambda rows, summary: rows[1].update(step=5),
            "control_step": lambda rows, summary: rows[1].update(control_step=0),
            "time": lambda rows, summary: rows[1].update(time_sec=1080),
            "episode": lambda rows, summary: rows[1].update(episode=9),
            "profile": lambda rows, summary: rows[1].update(profile_sha256="other"),
            "cumulative": lambda rows, summary: rows[1].update(total_ttt=99.),
            "reward": lambda rows, summary: rows[1].update(reward=0.),
            "area_cost": lambda rows, summary: rows[1].update(urban_ttt=99.),
            "early_terminal": lambda rows, summary: rows[1].update(terminated=True),
            "missing_terminal": lambda rows, summary: rows[-1].update(terminated=False),
            "truncation": lambda rows, summary: rows[1].update(truncated=True),
            "negative_cost": lambda rows, summary: rows[1].update(interval_ttt=-1.),
            "physical": lambda rows, summary: rows[1]["execution_check"].update(physical_control_valid=False),
            "budget": lambda rows, summary: rows[1]["execution_check"].update(budget_feasible=False),
            "guard_flag": lambda rows, summary: rows[1].update(h3_guard_enabled=True),
            "old_source": lambda rows, summary: rows[1].update(reference_source="PFO_reference"),
            "initial_source": lambda rows, summary: rows[0].update(reference_source="previous", pfo_calls=0),
            "late_initial": lambda rows, summary: rows[1].update(reference_source="pfo_initial", pfo_calls=1),
            "pfo_calls": lambda rows, summary: rows[1].update(pfo_calls=1),
            "recovery_flag": lambda rows, summary: rows[30].update(reference_recovery=False),
            "selection": lambda rows, summary: rows[1].update(selection_source="PFO_reference"),
            "candidate_count": lambda rows, summary: rows[1].update(lower_candidate_count=2),
            "wall_time": lambda rows, summary: rows[1].update(decision_wall_seconds=99.),
            "cpu_time": lambda rows, summary: rows[1].update(decision_cpu_seconds=99.),
            "pfo_wall_cost": lambda rows, summary: rows[0].update(pfo_wall_seconds=0.),
            "pfo_cpu_cost": lambda rows, summary: rows[0].update(pfo_cpu_seconds=0.),
            "exploration": lambda rows, summary: summary.update(exploration=True),
            "partial_summary": lambda rows, summary: summary.update(full_run=False),
        }
        for key in ("fallback_count", "pfo_calls", "pfo_recovery_count", "lower_candidate_solves", "converged_count"):
            changes["summary_" + key] = lambda rows, summary, key=key: summary.update({key: 999})
        for label, change in changes.items():
            with self.subTest(label=label):
                rows, summary = copy.deepcopy(original_rows), copy.deepcopy(original_summary)
                change(rows, summary)
                with self.assertRaises(ValueError):
                    validate_episode(summary, rows, settings, 0)

    def test_reject_nonfinite_cost_reward_guard_and_timing(self):
        folder = self.folders["rl"]
        rows = read(folder / "episode_00_trace.json")
        summary = read(folder / "episode_00_summary.json")
        settings = read(folder / "settings.json")
        for value in (float("nan"), float("inf"), float("-inf")):
            for key in ("interval_ttt", "total_ttt", "freeway_ttt", "urban_ttt", "reward",
                        "selected_TTT", "reference_TTT", "decision_wall_seconds", "decision_cpu_seconds",
                        "pfo_wall_seconds", "pfo_cpu_seconds"):
                with self.subTest(value=value, key=key):
                    changed = copy.deepcopy(rows)
                    changed[1][key] = value
                    with self.assertRaises(ValueError):
                        validate_episode(summary, changed, settings, 0)
            for key in ("ttt", "warmup_ttt", "decision_wall_seconds", "decision_cpu_seconds", "pfo_calls"):
                with self.subTest(value=value, summary_key=key), self.assertRaises(ValueError):
                    validate_episode(dict(summary, **{key: value}), rows, settings, 0)

    def test_reject_partial_status_and_missing_trace(self):
        self.change("completion.json", lambda data: data.update(status="paused"))
        with self.assertRaises(ValueError):
            compare(self.folders)
        fixture(self.root, "rl")
        (self.folders["rl"] / "episode_00_trace.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or malformed"):
            compare(self.folders)

    def test_reject_incorrect_queue_exposure(self):
        self.change("episode_00_trace.json", lambda rows:
            rows[0]["queue_near_capacity_estimate"]["boundary_queue"]["c"].update(near_capacity_seconds=0.))
        with self.assertRaises(ValueError):
            compare(self.folders)

    def test_reject_zero_center_cost(self):
        folder = self.folders["center"]
        result = read(folder / "completion.json")
        result["episodes"][0].update(ttt=0., warmup_ttt=0., freeway_ttt=0., urban_ttt=0.)
        rows = read(folder / "episode_00_trace.json")
        for row in rows:
            row.update(interval_ttt=0., total_ttt=0., freeway_ttt=0., urban_ttt=0., reward=0.)
        save(folder / "completion.json", result)
        save(folder / "episode_00_summary.json", result["episodes"][0])
        save(folder / "episode_00_trace.json", rows)
        with self.assertRaisesRegex(ValueError, "positive finite TTT"):
            compare(self.folders)


if __name__ == "__main__":
    unittest.main()
