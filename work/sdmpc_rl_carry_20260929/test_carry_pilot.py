"""Pilot lifecycle tests with synthetic child processes; never launches a solver."""
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
from test_carry_run_contracts import fixture, PINS, VERSIONS, SCHEMA


def completed_child(root, name, guard_mode="physical"):
    mode = name
    seeds = [6201, 6202] if name == "train" else [None]
    model_hash = run_budget.file_hash(root / "train/model_final.pt") if mode == "rl" else None
    folder = fixture(root, name, mode, seeds, model_hash, guard_mode=guard_mode)
    if name == "train":
        (folder / "model_final.pt").write_bytes(b"synthetic policy")
        result = read(folder / "completion.json")
        save(folder / "model_hash.json", dict(format=run_budget.POLICY_FORMAT, frozen=True,
             sha256=run_budget.file_hash(folder / "model_final.pt"), training_run_id="train",
             train_seeds=seeds, training_profiles=run_budget.training_profiles(result["episodes"]),
             contract=dict(source_pins=PINS, runtime_versions=VERSIONS,
                           environment_contract=read(folder / "settings.json")["environment_contract"],
                           observation_schema=SCHEMA, reward_divisor=100., gamma=1.)))


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "pilot"
        self.gate = self.root / "gate.json"
        save(self.gate, dict(ready_for_bounded_pilot=True, source_pins=PINS, evidence_sha256={}))
        self.commands = []
        self.live = self.maximum_live = 0
        self.behavior = {}
        self.waited = []
        self.streams = []

    def popen(self, command, **kwargs):
        folder = Path(command[command.index("--output") + 1])
        name = folder.name
        self.commands.append(command)
        self.streams.extend([kwargs["stdout"], kwargs["stderr"]])
        self.assertEqual(kwargs["creationflags"], run_pilot.subprocess.CREATE_NO_WINDOW)
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        if name == "rl":
            self.assertEqual(self.live, 0)
            self.assertTrue((self.output / "train/completion.json").exists())
            self.assertTrue((self.output / "center/completion.json").exists())
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
                    completed_child(owner.output, name, guard_mode="h3" if behavior == "h3" else "physical")
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
            stack.enter_context(patch.object(sys, "path", list(sys.path)))
            stack.enter_context(patch.object(run_budget, "pins", return_value=PINS))
            stack.enter_context(patch.object(run_budget, "verify_pins"))
            stack.enter_context(patch.object(run_budget, "runtime_versions", return_value=VERSIONS))
            stack.enter_context(patch.object(run_pilot.subprocess, "Popen", side_effect=self.popen))
            stack.enter_context(patch.object(run_pilot.time, "sleep"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_pilot.main()

    def test_bounded_train_center_then_rl_complete(self):
        self.run_main()
        self.assertEqual(self.maximum_live, 2)
        self.assertEqual(self.live, 0)
        self.assertEqual([Path(c[c.index("--output")+1]).name for c in self.commands],
                         ["train", "center", "rl"])
        self.assertEqual([c[c.index("--cpu-mask")+1] for c in self.commands], ["1", "4", "1"])
        self.assertIn("--training-seeds", self.commands[0])
        self.assertEqual(read(self.output / "completion.json")["evaluation_runs"], 2)
        self.assertTrue(all(stream.closed for stream in self.streams))
        plan = read(self.output / "plan.json")
        self.assertEqual(plan["training_seeds"], [6201, 6202])
        self.assertEqual(plan["maximum_child_processes"], 2)
        self.assertEqual(plan["numerical_threads_per_child"], 1)
        self.assertEqual(plan["guard_mode"], "physical")
        self.assertEqual(plan["updates_per_interval"], 20)
        self.assertEqual([c[c.index("--guard-mode") + 1] for c in self.commands], ["physical"] * 3)
        self.assertFalse(plan["scheduled_automation"])
        position = self.commands[0].index("--training-seeds")
        self.assertEqual(self.commands[0][position + 1:position + 3], ["6201", "6202"])
        self.assertIsNone(read(self.output / "rl/completion.json")["episodes"][0]["training_seed"])

    def test_paused_rl_never_completes_and_can_resume(self):
        self.behavior["rl"] = "paused"
        self.run_main()
        self.assertFalse((self.output / "completion.json").exists())
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(self.live, 0)
        (self.output / "STOP").unlink()
        (self.output / "rl/STOP").unlink()
        self.behavior.clear()
        self.commands.clear()
        self.run_main(resume=True)
        self.assertEqual(len(self.commands), 1)
        self.assertIn("--resume", self.commands[0])
        self.assertTrue((self.output / "completion.json").exists())

    def test_zero_exit_without_completion_fails(self):
        self.behavior["rl"] = "missing_completion"
        self.run_main()
        self.assertFalse((self.output / "completion.json").exists())
        self.assertEqual(read(self.output / "failure.json")["child"], "rl")
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(self.live, 0)

    def test_failed_child_stops_before_second_stage(self):
        self.behavior["center"] = "failure"
        self.run_main()
        self.assertEqual(len(self.commands), 2)
        self.assertFalse((self.output / "completion.json").exists())
        self.assertEqual(self.live, 0)

    def test_launch_exception_drains_active_children(self):
        self.behavior["center"] = "launch_error"
        with self.assertRaisesRegex(OSError, "launch failure"):
            self.run_main()
        self.assertEqual(self.waited, ["train"])
        self.assertTrue(all(stream.closed for stream in self.streams))
        self.assertEqual(self.live, 0)
        self.assertTrue((self.output / "STOP").exists())

    def test_skipped_completion_is_validated(self):
        self.run_main()
        (self.output / "completion.json").unlink()
        result = read(self.output / "rl/completion.json")
        result["episodes"][0]["training_seed"] = 6203
        save(self.output / "rl/completion.json", result)
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
        for field, value in (("training_run_id", "other"), ("train_seeds", [6203]),
                             ("training_profiles", []), ("sha256", "0"*64)):
            save(manifest_path, dict(original, **{field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_pilot.validate_child(self.output, "train", plan)
        save(manifest_path, original)
        settings_path = self.output / "rl/settings.json"
        settings = read(settings_path)
        settings["model_sha256"] = "f"*64
        save(settings_path, settings)
        with self.assertRaisesRegex(ValueError, "model identity"):
            run_pilot.validate_child(self.output, "rl", plan)

    def test_changed_admission_evidence_rejected_before_launch(self):
        evidence = self.root / "evidence.json"
        save(evidence, {"admitted": True})
        gate = read(self.gate)
        gate["evidence_sha256"] = {str(evidence): run_budget.file_hash(evidence)}
        save(self.gate, gate)
        save(evidence, {"admitted": False})
        with self.assertRaisesRegex(RuntimeError, "Admission evidence changed"):
            self.run_main()
        self.assertEqual(self.commands, [])
        self.assertFalse(self.output.exists())

    def test_matching_admission_evidence_is_verified(self):
        evidence = self.root / "evidence.json"
        save(evidence, {"synthetic": True})
        gate = read(self.gate)
        gate["evidence_sha256"] = {str(evidence): run_budget.file_hash(evidence)}
        save(self.gate, gate)
        self.run_main()
        self.assertTrue((self.output / "completion.json").exists())

    def test_missing_gate_or_source_mismatch_rejects_before_launch(self):
        for changes in ({"ready_for_bounded_pilot": False}, {"source_pins": {"snapshot": "drift"}}):
            with self.subTest(changes=changes):
                save(self.gate, dict(ready_for_bounded_pilot=True, source_pins=PINS,
                                     evidence_sha256={}))
                gate = read(self.gate)
                gate.update(changes)
                save(self.gate, gate)
                with self.assertRaisesRegex(RuntimeError, "Preflight gate"):
                    self.run_main()
                self.assertEqual(self.commands, [])

    def test_child_stop_before_launch_drains_sibling(self):
        self.output.mkdir()
        (self.output / "center").mkdir()
        (self.output / "center/STOP").touch()
        save(self.output / "plan.json", {})
        # Obtain the actual declared plan in a separate mocked pilot run.
        stopped_output = self.output
        self.output = self.root / "plan_template"
        self.run_main()
        self.expected_plan = read(self.output / "plan.json")
        self.output = stopped_output
        save(self.output / "plan.json", self.expected_plan)
        self.commands.clear()
        self.waited.clear()
        self.run_main(resume=True)
        self.assertEqual(self.waited, ["train"])
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.live, 0)
        self.assertTrue((self.output / "STOP").exists())
        self.assertFalse((self.output / "completion.json").exists())

    def test_completed_pilot_refuses_duplicate_work(self):
        self.run_main()
        self.commands.clear()
        with self.assertRaisesRegex(RuntimeError, "already complete"):
            self.run_main(resume=True)
        self.assertEqual(self.commands, [])

    def test_consistent_h3_children_rejected_by_physical_plan(self):
        self.run_main()
        plan = read(self.output / "plan.json")
        for name in ("center", "rl", "train"):
            with self.subTest(name=name):
                completed_child(self.output, name, guard_mode="h3")
                with self.assertRaisesRegex(ValueError, "guard"):
                    run_pilot.validate_child(self.output, name, plan)
                completed_child(self.output, name)

    def test_guard_required_in_settings_and_coordinator_contract(self):
        self.run_main()
        plan = read(self.output / "plan.json")
        path = self.output / "center/settings.json"
        original = read(path)
        for location in ("settings", "coordinator"):
            with self.subTest(location=location):
                settings = read(path)
                target = settings if location == "settings" else settings["environment_contract"]["coordinator"]
                target["guard_mode"] = "h3"
                save(path, settings)
                with self.assertRaisesRegex(ValueError, "guard"):
                    run_pilot.validate_child(self.output, "center", plan)
                save(path, original)

    def test_completed_h3_sibling_refused_before_any_resume_launch(self):
        self.run_main()
        plan = read(self.output / "plan.json")
        for name in ("center", "rl"):
            with self.subTest(name=name):
                self.output = self.root / ("resume-" + name)
                save(self.output / "plan.json", plan)
                if name == "rl":
                    (self.output / "train").mkdir()
                    (self.output / "train/model_final.pt").write_bytes(b"synthetic policy")
                completed_child(self.output, name, guard_mode="h3")
                folder = self.output / name
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                self.commands.clear()
                with self.assertRaisesRegex(ValueError, "guard"):
                    self.run_main(resume=True)
                self.assertEqual(self.commands, [])
                self.assertEqual(self.live, 0)
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
                self.assertFalse((self.output / "completion.json").exists())

    def test_newly_exited_h3_child_stops_before_rl_stage(self):
        self.behavior["center"] = "h3"
        self.run_main()
        self.assertEqual([Path(c[c.index("--output") + 1]).name for c in self.commands], ["train", "center"])
        self.assertEqual(self.live, 0)
        self.assertTrue((self.output / "STOP").exists())
        self.assertFalse((self.output / "completion.json").exists())
        self.assertEqual(read(self.output / "failure.json")["child"], "center")


if __name__ == "__main__":
    unittest.main()
