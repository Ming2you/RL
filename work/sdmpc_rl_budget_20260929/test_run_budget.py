import copy
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass, field
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_budget as runner
from budget_runtime import read, save
from test_run_contracts import fixture, PINS, VERSIONS


class FakeEnv:
    templates = None
    output = None
    pause_at = None
    actions = []

    def __init__(self, runtime, training_seed=None):
        self.training_seed = training_seed
        self.profile_hash = f"profile-{training_seed}"
        self.protocol = {"scenario": "synthetic"}
        self.observer = SimpleNamespace(names=["state/time"], delay=2,
                                       contract=lambda: {"scale": 80., "names": ["state/time"]})

    def reset(self):
        self.k = 5
        self.warmup_ttt = 5.
        self.sim = SimpleNamespace(total_ttt=5., state=SimpleNamespace(time_sec=900.))
        return np.array([self.k / 80], dtype=np.float32)

    def step(self, action, mode, actor_seconds, actor_cpu_seconds=None):
        row = copy.deepcopy(self.templates[self.k - 5])
        self.actions.append((self.training_seed, self.k, action.copy()))
        self.k += 1
        self.sim.total_ttt += 1.
        self.sim.state.time_sec += 180.
        row.update(action_requested=action.copy(), inventory={"vehicles": 1.},
                   selection_source="PFO_reference" if self.k % 2 else "lower_solution",
                   actor_wall_seconds=actor_seconds, actor_cpu_seconds=actor_cpu_seconds,
                   decision_wall_seconds=4. + actor_seconds, decision_cpu_seconds=4. + actor_cpu_seconds)
        if self.pause_at == (self.training_seed, self.k):
            (self.output / "STOP").touch()
        return np.array([0. if self.k == 80 else self.k / 80], dtype=np.float32), -.01, self.k == 80, row

    def checkpoint(self):
        return dict(k=self.k, sim=copy.deepcopy(self.sim), profile_hash=self.profile_hash)

    def restore(self, state):
        self.k, self.sim = state["k"], copy.deepcopy(state["sim"])
        self.warmup_ttt = 5.
        return np.array([0. if self.k == 80 else self.k / 80], dtype=np.float32)

    def close(self):
        pass


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        template = fixture(self.root, "templates", "native")
        FakeEnv.templates = read(template / "episode_00_trace.json")
        FakeEnv.pause_at = None
        FakeEnv.actions = []
        spec = importlib.util.spec_from_file_location("runner_test_historical_config",
            runner.DEFAULT_SNAPSHOT / "work/sdmpc_matrix_14400_20260912/historical_config.py")
        self.historical_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.historical_config)

    def run_main(self, name, mode="train", resume=False, extra=(), save_hook=None, runtime=None):
        folder = self.root / name
        FakeEnv.output = folder
        argv = ["run_budget.py", "--output", str(folder), "--mode", mode, *extra]
        if resume:
            argv.append("--resume")
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            previous_env = sys.modules.get("budget_env")
            sys.modules["budget_env"] = SimpleNamespace(BudgetEnv=FakeEnv)
            stack.callback(lambda: sys.modules.pop("budget_env", None) if previous_env is None else
                           sys.modules.__setitem__("budget_env", previous_env))
            stack.enter_context(patch.object(runner, "pins", return_value=PINS))
            stack.enter_context(patch.object(runner, "verify_pins"))
            stack.enter_context(patch.object(runner, "boot", return_value=runtime if runtime is not None else
                                             {"cfg": {}, "options": {}, "rc": self.historical_config}))
            stack.enter_context(patch.object(runner, "runtime_versions", return_value=VERSIONS))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if save_hook:
                stack.enter_context(patch.object(runner, "save", side_effect=save_hook))
            runner.main()
        return folder

    def equal_state(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_state(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_state(a, b)
        else:
            self.assertEqual(left, right)

    def load(self, folder, name="model_final.pt"):
        return torch.load(folder / name, weights_only=False)

    def test_two_episodes_requested_actions_and_frozen_evaluation(self):
        folder = self.run_main("train")
        model = self.load(folder)
        state = model["learner"]
        self.assertEqual(len(FakeEnv.actions), 150)
        self.assertEqual(state["updates"], 119)
        np.testing.assert_array_equal(state["replay"]["actions"].numpy(), np.stack([r[2] for r in FakeEnv.actions]))
        self.assertEqual(torch.where(state["replay"]["terminated"])[0].tolist(), [74, 149])
        self.assertEqual(len(model["training_profiles"]), 2)
        evaluated = self.run_main("rl", "rl", extra=["--model", str(folder / "model_final.pt")])
        self.equal_state(state, self.load(evaluated, "checkpoint.pt")["learner"])
        result = read(evaluated / "completion.json")
        self.assertFalse(result["episodes"][0]["exploration"])
        self.assertIsNone(result["episodes"][0]["training_seed"])
        self.assertEqual(len(result["episodes"]), 1)

    def test_exact_resume_after_odd_update_and_terminal_checkpoint(self):
        baseline = self.load(self.run_main("baseline"))["learner"]
        for boundary in ("odd_update", "terminal"):
            with self.subTest(boundary=boundary):
                FakeEnv.actions = []
                if boundary == "odd_update":
                    FakeEnv.pause_at = (6101, 37)
                    folder = self.run_main(boundary)
                    self.assertEqual(self.load(folder, "checkpoint.pt")["learner"]["updates"], 1)
                    self.assertFalse((folder / "completion.json").exists())
                    (folder / "STOP").unlink()
                    FakeEnv.pause_at = None
                else:
                    def fail_at_summary(path, value):
                        if path.name == "episode_00_summary.json":
                            raise RuntimeError("synthetic finalization interruption")
                        save(path, value)
                    with self.assertRaisesRegex(RuntimeError, "finalization interruption"):
                        self.run_main(boundary, save_hook=fail_at_summary)
                    folder = self.root / boundary
                    self.assertEqual(self.load(folder, "checkpoint.pt")["environment"]["k"], 80)
                self.run_main(boundary, resume=True)
                self.equal_state(baseline, self.load(folder)["learner"])
                self.assertEqual(len(FakeEnv.actions), 150)
                self.assertEqual(len(read(folder / "completion.json")["episodes"]), 2)

    def test_runtime_mismatch_preserves_original_metadata(self):
        folder = self.root / "metadata"
        old = dict(run_id="run", runtime_versions=VERSIONS)
        runner.write_run_settings(folder, old, False)
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        changed = dict(old, runtime_versions=dict(VERSIONS, scipy="different"))
        with self.assertRaisesRegex(RuntimeError, "runtime differs"):
            runner.write_run_settings(folder, changed, True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
        save(folder / "runtime_versions.json", dict(VERSIONS, torch="drift"))
        with self.assertRaisesRegex(RuntimeError, "runtime record"):
            runner.write_run_settings(folder, old, True)

    def test_policy_contract_rejected_before_learner_mutation(self):
        env = FakeEnv({}, None)
        settings = dict(source_pins=PINS, runtime_versions=VERSIONS, environment_contract={"cfg": "frozen"},
                        reward_divisor=100., gamma=1., run_id="trainer")
        completed = [dict(training_seed=6101, profile_sha256="training-profile")]
        learner = SimpleNamespace(state_dict=lambda: {"weight": 1})
        payload = runner.export_policy(learner, env, settings, completed)
        loaded = []
        target = SimpleNamespace(load_state_dict=loaded.append)
        runner.load_policy(target, payload, env, settings)
        self.assertEqual(loaded, [{"weight": 1}])
        for field in ("source_pins", "runtime_versions", "environment_contract", "observation_schema", "gamma"):
            changed = copy.deepcopy(payload)
            changed["contract"][field] = "drift"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.load_policy(target, changed, env, settings)
        for field in ("format", "training_run_id", "training_profiles"):
            changed = copy.deepcopy(payload)
            changed.pop(field)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.load_policy(target, changed, env, settings)
        self.assertEqual(loaded, [{"weight": 1}])

    def test_checkpoint_contract_and_boundary_rejected(self):
        settings = dict(seeds=[6101, 6102], profile_sha256=["one", "two"])
        ck = dict(format=runner.CHECKPOINT_FORMAT, settings=settings, episode=0,
                  completed=[], trace=[{}], environment=dict(k=6, profile_hash="one"))
        runner.validate_checkpoint(ck, settings)
        for field, value in (("settings", {}), ("episode", 2), ("completed", [{}]),
                             ("trace", []), ("environment", dict(k=6, profile_hash="two"))):
            changed = dict(ck, **{field: value})
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.validate_checkpoint(changed, settings)

    def capture_settings(self, runtime, name):
        class SettingsCaptured(Exception):
            pass

        def capture(_output, settings, _resume):
            raise SettingsCaptured(settings)

        with patch.object(runner, "write_run_settings", side_effect=capture), \
                self.assertRaises(SettingsCaptured) as captured:
            self.run_main(name, runtime=runtime)
        return captured.exception.args[0]

    def dynamic_runtime(self):
        @dataclass
        class Section:
            declared_value: int = 1

        @dataclass
        class Config:
            network: Section = field(default_factory=Section)
            mpc: Section = field(default_factory=Section)

        cfg, options = Config(), Section()
        cfg.network.terminal_zero_gradient = True
        cfg.mpc.leader_rollout_box_walk = True
        options.runtime_flag = True
        return {"cfg": cfg, "options": options, "rc": self.historical_config}

    def test_runner_hashes_include_public_dynamic_config_and_options(self):
        runtime = self.dynamic_runtime()
        baseline = self.capture_settings(runtime, "dynamic_baseline")["environment_contract"]
        self.assertEqual(baseline, self.capture_settings(runtime, "dynamic_same")["environment_contract"])
        cases = [(runtime["cfg"].network, "terminal_zero_gradient", "config_sha256"),
                 (runtime["cfg"].mpc, "leader_rollout_box_walk", "config_sha256"),
                 (runtime["options"], "runtime_flag", "options_sha256")]
        for index, (obj, attribute, key) in enumerate(cases):
            with self.subTest(attribute=attribute):
                setattr(obj, attribute, False)
                changed = self.capture_settings(runtime, f"dynamic_{index}")["environment_contract"]
                self.assertNotEqual(baseline[key], changed[key])
                other = "options_sha256" if key == "config_sha256" else "config_sha256"
                self.assertEqual(baseline[other], changed[other])
                setattr(obj, attribute, True)
        self.assertEqual(FakeEnv.actions, [])

    def test_dynamic_config_hash_drift_refuses_resume_and_policy_before_mutation(self):
        runtime = self.dynamic_runtime()
        baseline = self.capture_settings(runtime, "hash_baseline")
        runtime["cfg"].network.terminal_zero_gradient = False
        changed = self.capture_settings(runtime, "hash_changed")
        changed["run_id"] = baseline["run_id"]
        folder = self.root / "resume_metadata"
        runner.write_run_settings(folder, baseline, False)
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with self.assertRaises(RuntimeError):
            runner.write_run_settings(folder, changed, True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
        env = FakeEnv({}, None)
        policy = runner.export_policy(SimpleNamespace(state_dict=lambda: {"weight": 1}), env,
            baseline, [dict(training_seed=6101, profile_sha256="training-profile")])
        loaded = []
        with self.assertRaises(RuntimeError):
            runner.load_policy(SimpleNamespace(load_state_dict=loaded.append), policy, env, changed)
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
