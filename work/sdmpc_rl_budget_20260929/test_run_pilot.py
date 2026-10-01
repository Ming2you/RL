from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from budget_runtime import read, save
import run_budget
import run_pilot
from test_run_contracts import fixture, PINS, VERSIONS, SCHEMA, ENVIRONMENT


def completed_child(root, name):
    mode = "rl" if name == "validation" else name
    seeds = [6101, 6102] if name == "train" else [6103] if name == "validation" else [None]
    model_hash = run_budget.file_hash(root / "train/model_final.pt") if mode == "rl" else None
    folder = fixture(root, name, mode, seeds, model_hash)
    if name == "train":
        (folder / "model_final.pt").write_bytes(b"synthetic policy")
        result = read(folder / "completion.json")
        save(folder / "model_hash.json", dict(format=run_budget.POLICY_FORMAT, frozen=True,
             sha256=run_budget.file_hash(folder / "model_final.pt"), training_run_id="train",
             train_seeds=seeds, training_profiles=run_budget.training_profiles(result["episodes"]),
             contract=dict(source_pins=PINS, runtime_versions=VERSIONS, environment_contract=ENVIRONMENT,
                           observation_schema=SCHEMA, reward_divisor=100., gamma=1.)))


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "pilot"
        self.gate = self.root / "gate.json"
        save(self.gate, dict(ready_for_bounded_pilot=True, source_pins=PINS))
        self.commands = []
        self.live = self.maximum_live = 0
        self.behavior = {}
        self.waited = []

    def popen(self, command, **kwargs):
        folder = Path(command[command.index("--output") + 1])
        name = folder.name
        self.commands.append(command)
        if self.behavior.get(name) == "launch_error":
            raise OSError("synthetic launch failure")
        self.live += 1
        self.maximum_live = max(self.maximum_live, self.live)
        owner = self

        class Child:
            pid = 12345
            polls = 0
            done = False

            def finish(self, waiting=False):
                if self.done:
                    return 0
                self.done = True
                owner.live -= 1
                behavior = owner.behavior.get(name)
                if waiting or behavior == "paused":
                    save(folder / "status.json", dict(status="paused"))
                    (folder / "checkpoint.pt").write_bytes(b"synthetic checkpoint")
                    (folder / "STOP").touch()
                elif behavior == "failure":
                    return 1
                elif behavior != "missing_completion":
                    completed_child(owner.output, name)
                return 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else self.finish()

            def wait(self):
                owner.waited.append(name)
                return self.finish(waiting=True)

        return Child()

    def run_main(self, resume=False):
        argv = ["run_pilot.py", "--output", str(self.output), "--gate", str(self.gate)]
        if resume:
            argv.append("--resume")
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(run_budget, "pins", return_value=PINS))
            stack.enter_context(patch.object(run_budget, "verify_pins"))
            stack.enter_context(patch.object(run_budget, "runtime_versions", return_value=VERSIONS))
            stack.enter_context(patch.object(run_pilot.subprocess, "Popen", side_effect=self.popen))
            stack.enter_context(patch.object(run_pilot.time, "sleep"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_pilot.main()

    def test_bounded_workflow_and_validation_complete(self):
        self.run_main()
        self.assertEqual(self.maximum_live, 3)
        self.assertEqual(self.live, 0)
        self.assertEqual([Path(c[c.index("--output")+1]).name for c in self.commands],
                         ["train", "native", "center", "validation", "rl"])
        self.assertEqual([c[c.index("--cpu-mask")+1] for c in self.commands], ["1", "4", "16", "1", "4"])
        self.assertIn("--training-seeds", self.commands[0])
        self.assertEqual(read(self.output / "completion.json")["evaluation_runs"], 4)
        self.assertEqual(read(self.output / "validation/completion.json")["episodes"][0]["training_seed"], 6103)

    def test_paused_validation_never_completes_and_can_resume(self):
        self.behavior["validation"] = "paused"
        self.run_main()
        self.assertFalse((self.output / "completion.json").exists())
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(self.live, 0)
        (self.output / "STOP").unlink()
        (self.output / "validation/STOP").unlink()
        self.behavior.clear()
        self.commands.clear()
        self.run_main(resume=True)
        self.assertEqual(len(self.commands), 1)
        self.assertIn("--resume", self.commands[0])
        self.assertTrue((self.output / "completion.json").exists())

    def test_zero_exit_without_completion_fails(self):
        self.behavior["validation"] = "missing_completion"
        self.run_main()
        self.assertFalse((self.output / "completion.json").exists())
        self.assertEqual(read(self.output / "failure.json")["child"], "validation")
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(self.live, 0)

    def test_failed_child_stops_before_second_stage(self):
        self.behavior["native"] = "failure"
        self.run_main()
        self.assertEqual(len(self.commands), 3)
        self.assertFalse((self.output / "completion.json").exists())
        self.assertEqual(self.live, 0)

    def test_launch_exception_drains_active_children(self):
        self.behavior["center"] = "launch_error"
        with self.assertRaisesRegex(OSError, "launch failure"):
            self.run_main()
        self.assertEqual(set(self.waited), {"train", "native"})
        self.assertEqual(self.live, 0)
        self.assertTrue((self.output / "STOP").exists())

    def test_skipped_completion_is_validated(self):
        self.run_main()
        (self.output / "completion.json").unlink()
        result = read(self.output / "validation/completion.json")
        result["episodes"][0]["training_seed"] = None
        save(self.output / "validation/completion.json", result)
        self.commands.clear()
        with self.assertRaises(ValueError):
            self.run_main(resume=True)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.output / "completion.json").exists())

    def test_training_model_provenance_and_evaluation_model_must_match(self):
        self.run_main()
        plan = read(self.output / "plan.json")
        manifest_path = self.output / "train/model_hash.json"
        original = read(manifest_path)
        for field, value in (("training_run_id", "other"), ("train_seeds", [6103]),
                             ("training_profiles", []), ("sha256", "0"*64)):
            save(manifest_path, dict(original, **{field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_pilot.validate_child(self.output, "train", plan)
        save(manifest_path, original)
        settings_path = self.output / "validation/settings.json"
        settings = read(settings_path)
        settings["model_sha256"] = "f"*64
        save(settings_path, settings)
        with self.assertRaisesRegex(ValueError, "model identity"):
            run_pilot.validate_child(self.output, "validation", plan)


if __name__ == "__main__":
    unittest.main()
