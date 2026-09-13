"""Mocked orchestration checks; never train a model or advance a simulator."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_leader.five_cell_data import FIVE_CELL_SCENARIOS
from rl_leader.response_dqn_catalog import StructuredActionCatalog
from src.tests.test_five_cell_data import fixture
from work import run_five_cell_shared as runner


def plan():
    return json.loads((runner.ROOT / "work/five_cell_shared_v1.json").read_text(encoding="utf-8"))


class NoWorkPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, payloads):
        assert not payloads, "the test must not launch training"
        return iter(())


class FiveCellRunnerTest(unittest.TestCase):
    def test_plan_requires_actual_three_member_ungated_finite_horizon_recipe(self):
        runner.validate_plan(plan())
        mutations = [
            ("training", "ensemble_size", 2),
            ("training", "gamma", .99),
            ("training", "value_parameterization", "free_q"),
            ("evaluation", "intervention_gate", True),
            ("evaluation", "lcb_fallback", True),
            (None, "actors", -2),
            (None, "response_workers", 0),
            (None, "response_equivalence_mode", "post_commit_continuation_v1"),
        ]
        for section, key, value in mutations:
            candidate = plan()
            (candidate[section] if section else candidate)[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                runner.validate_plan(candidate)

    def test_prepare_only_does_not_initialize_simulator_and_builds_exact_17_actions(self):
        candidate = plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "work/plan.json"
            config.parent.mkdir()
            config.write_text(json.dumps(candidate), encoding="utf-8")
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "RLLeaderEnv", create=True, side_effect=AssertionError("prepare must not construct/reset an environment")), \
                 patch.object(runner, "ProcessPoolExecutor", side_effect=AssertionError("prepare must not spawn work")), \
                 redirect_stdout(io.StringIO()):
                runner.run(candidate, config, prepare_only=True)
            output = root / candidate["output_dir"]
            catalog = StructuredActionCatalog.from_manifest(runner.read_json(output / "catalog.json"))
            self.assertEqual(catalog.size, 17)
            self.assertEqual(catalog.actions[0].key, "anchor")
            actual = {(action.owner, action.template) for action in catalog.actions[1:]}
            expected = {(owner, template) for owner in candidate["catalog"]["owners"]
                        for template in candidate["catalog"]["templates"]}
            self.assertEqual(actual, expected)
            self.assertEqual(set(runner.read_json(output / "contracts.json")), set(FIVE_CELL_SCENARIOS))

    def test_job_ids_and_common_models_are_unique_across_cells_rounds_and_purposes(self):
        candidate = plan()
        replays, contracts = fixture()
        catalog = StructuredActionCatalog.from_manifest(replays[0].manifest["catalog"])
        root = runner.ROOT / candidate["output_dir"]
        models = [f"shared-{i}.pt" for i in range(3)]
        jobs = []
        with patch.object(runner, "digest", return_value="a" * 64):
            for purpose in ("baseline", "collection", "evaluation"):
                for index in range(4):
                    for scenario in FIVE_CELL_SCENARIOS:
                        job = runner.job(candidate, root, contracts, catalog, {"source.py": "b" * 64},
                                         scenario=scenario, purpose=purpose, index=index,
                                         models=models if purpose != "baseline" else ())
                        jobs.append(job)
                        self.assertEqual(job["scenario"], scenario)
                        if purpose == "evaluation":
                            self.assertEqual(job["checkpoints"], models)
                            self.assertEqual(job["epsilon"], 0)
                        if purpose == "baseline":
                            self.assertEqual(job["checkpoints"], [])
                            self.assertEqual(job["epsilon"], 0)
        self.assertEqual(len({job["episode"] for job in jobs}), len(jobs))
        self.assertEqual(len({job["directory"] for job in jobs}), len(jobs))

    def test_round_data_sizes_are_1125_then_1500_and_eval_uses_same_ensemble(self):
        candidate = plan()
        replays, contracts = fixture()
        catalog = StructuredActionCatalog.from_manifest(replays[0].manifest["catalog"])
        collections, training = [], []
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = repo / candidate["output_dir"]
            root.mkdir(parents=True)

            def train(plan, root, catalog, contracts, sources, round_index, jobs):
                training.append((round_index, copy.deepcopy(jobs)))
                return [f"models/round_{round_index}/member_{i}.pt" for i in range(3)]

            with patch.object(runner, "ROOT", repo), \
                 patch.object(runner, "prepare", return_value=(root, contracts, catalog, {})), \
                 patch.object(runner, "runner_lock", return_value=nullcontext()), \
                 patch.object(runner, "run_jobs", side_effect=lambda p, r, jobs: collections.append(copy.deepcopy(jobs))), \
                 patch.object(runner, "train_round", side_effect=train), \
                 patch.object(runner, "compare_round", side_effect=lambda *args: {"round": args[-1]}), \
                 patch.object(runner, "digest", return_value="a" * 64):
                runner.run(candidate, repo / "plan.json")
        self.assertEqual([75 * len(jobs) for _, jobs in training], [1125, 1500])
        for round_index, jobs in training:
            counts = {scene: sum(job["scenario"] == scene for job in jobs) for scene in FIVE_CELL_SCENARIOS}
            self.assertEqual(set(counts.values()), {3 + round_index})
            self.assertTrue(all(job["purpose"] != "evaluation" for job in jobs))
        evaluations = [jobs for jobs in collections if jobs[0]["purpose"] == "evaluation"]
        self.assertEqual(len(evaluations), 2)
        for round_index, jobs in enumerate(evaluations):
            self.assertEqual(len(jobs), 5)
            self.assertEqual(len({tuple(job["checkpoints"]) for job in jobs}), 1)
            self.assertEqual(jobs[0]["checkpoints"], [f"models/round_{round_index}/member_{i}.pt" for i in range(3)])

    def test_model_sidecar_hash_alone_cannot_reuse_wrong_seed_or_training_recipe(self):
        candidate = plan()
        replays, contracts = fixture()
        catalog = StructuredActionCatalog.from_manifest(replays[0].manifest["catalog"])
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = repo / candidate["output_dir"]
            round_dir = root / "round_00"
            round_dir.mkdir(parents=True)
            for member in range(3):
                path = round_dir / f"shared_member_{member:02d}.pt"
                path.write_bytes(b"validly-hashed-but-wrong-training-run")
                runner.write_json(path.with_suffix(".json"), {
                    "checkpoint": str(path), "sha256": runner.digest(path),
                    "seed": -100, "updates": 1, "last_loss": 0.,
                })
            jobs = [{"directory": f"source_{i}"} for i in range(5)]
            by_path = {str(Path(job["directory"]) / "replay.npz"): replay
                       for job, replay in zip(jobs, replays)}
            with patch.object(runner, "ROOT", repo), \
                 patch.object(runner, "load_frozen_response_replay", side_effect=lambda path: by_path[str(path)]), \
                 patch.object(runner, "ProcessPoolExecutor", NoWorkPool), \
                 self.assertRaisesRegex(ValueError, "model|signature|seed|training"):
                runner.train_round(candidate, root, catalog, contracts, {}, 0, jobs)

    def test_stop_blocks_prepare_before_any_environment_work(self):
        candidate = plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / candidate["output_dir"]
            output.mkdir(parents=True)
            (output / "STOP").touch()
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "RLLeaderEnv", create=True, side_effect=AssertionError("must not reset")), \
                 self.assertRaises(InterruptedError):
                runner.prepare(candidate, root / "plan.json")

    def test_macro_comparison_is_equal_weight_and_requires_matching_full_contracts(self):
        candidate = plan()
        gains = [10., -10., 0., 20., 5.]
        baseline_ttt = [100., 1000., 10000., 5000., 200.]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baselines, evaluations = [], []
            for index, scenario in enumerate(FIVE_CELL_SCENARIOS):
                contract = f"{index:064x}"
                for purpose, jobs, ttt in (
                    ("baseline", baselines, baseline_ttt[index]),
                    ("evaluation", evaluations, baseline_ttt[index] * (1 - gains[index] / 100)),
                ):
                    output = root / purpose / scenario
                    jobs.append({"directory": str(output), "scenario": scenario,
                                 "purpose": purpose, "experiment_contract_sha256": contract,
                                 "checkpoints": ["a", "b", "c"] if purpose == "evaluation" else []})
                    runner.write_json(output / "summary.json", {
                        "scenario": scenario, "purpose": purpose, "status": "complete",
                        "experiment_contract_sha256": contract, "transitions": 75,
                        "terminal": True, "start_control_step": 0, "end_control_step": 74,
                        "total_ttt": ttt, "terminal_inventory": 0., "wall_seconds": 10.,
                        "epsilon": 0., "scope": "ungated_full_run", "lcb_guard": False,
                    })
            result = runner.compare_round(candidate, root, baselines, evaluations, 0)
            self.assertAlmostEqual(result["macro_improvement_percent"], 5.)
            self.assertAlmostEqual(result["worst_scenario_improvement_percent"], -10.)
            self.assertEqual(result["improved_scenarios"], 3)
            path = Path(evaluations[0]["directory"]) / "summary.json"
            summary = runner.read_json(path)
            for field, value, message in (("experiment_contract_sha256", "f" * 64, "contract"),
                                          ("transitions", 74, "partial")):
                runner.write_json(path, {**summary, field: value})
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                    runner.compare_round(candidate, root, baselines, evaluations, 0)


if __name__ == "__main__":
    unittest.main()
