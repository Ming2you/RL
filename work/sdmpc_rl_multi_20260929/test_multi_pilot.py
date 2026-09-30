"""Controlled child-process lifecycle tests; Popen never starts a real process."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import build_preflight
import run_budget
import run_pilot
from budget_runtime import read, save
from td3 import SCENARIOS
from test_multi_contracts import (PINS, VERSIONS, change_experience, change_json,
                                  completed_run, synthetic_training, admission_fixture)


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    admission = admission_fixture(tmp_path / "admission", monkeypatch)
    admission.build()
    output, gate = tmp_path / "pilot", admission.gate
    state = SimpleNamespace(output=output, gate=gate, commands=[], live={}, maximum=0, streams=[],
                            behaviors={}, waited=[], launches=[], on_launch=None, on_poll=None)
    monkeypatch.setattr(build_preflight, "pins", lambda _: PINS)
    monkeypatch.setattr(run_pilot, "runtime_versions", lambda: VERSIONS)
    monkeypatch.setattr(run_pilot, "verify_pins", lambda *_: None)
    monkeypatch.setattr(run_pilot.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def complete(job):
        if job["kind"] == "train":
            synthetic_training(output, job, dict(source_pins=PINS, runtime_versions=VERSIONS))
        else:
            completed_run(output / job["key"], job["scenario"], job["kind"], job["round"],
                None if job["model"] is None else run_budget.file_hash(job["model"]), job["cpu_mask"])

    def popen(command, **kwargs):
        folder = Path(command[command.index("--output")+1])
        key = folder.relative_to(output).as_posix()
        job = next(j for j in run_pilot.jobs_for(output, SCENARIOS) if j["key"] == key)
        state.commands.append(command)
        state.streams.extend((kwargs["stdout"], kwargs["stderr"]))
        state.launches.append((job, tuple(state.live)))
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert kwargs["env"][name] == "1"
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert Path(kwargs["cwd"]) == run_pilot.REPO
        if job["kind"] == "train":
            assert not state.live, "Shared learner must run alone"
        else:
            assert all(j["kind"] != "train" for j in state.live.values())
            mask = int(command[command.index("--cpu-mask")+1])
            assert mask > 0 and mask & (mask-1) == 0
        if state.behaviors.get(key) == "launch_error":
            raise OSError("synthetic Popen failure")
        state.live[key] = job
        state.maximum = max(state.maximum, len(state.live))
        if state.on_launch:
            state.on_launch(job)

        class Child:
            pid = 1000 + len(state.commands)
            polls = 0
            code = None

            def finish(self, waiting=False):
                if self.code is not None:
                    return self.code
                state.live.pop(key)
                behavior = state.behaviors.get(key)
                self.code = 9 if behavior == "failure" else 0
                if waiting or behavior == "paused" or any(
                        (p / "STOP").exists() for p in (output, folder.parent, folder)):
                    save(folder / "status.json", dict(status="paused"))
                    (folder / "checkpoint.pt").write_bytes(b"synthetic child checkpoint")
                elif behavior == "missing_completion" or behavior == "failure":
                    pass
                else:
                    complete(job)
                    if behavior == "malformed":
                        change_json(folder / "completion.json", lambda r: r.update(status="paused"))
                return self.code

            def poll(self):
                self.polls += 1
                if state.on_poll:
                    state.on_poll(job, self.polls)
                return None if self.polls == 1 else self.finish()

            def wait(self):
                state.waited.append(key)
                return self.finish(waiting=True)

        return Child()

    monkeypatch.setattr(run_pilot.subprocess, "Popen", popen)

    def run(resume=False):
        argv = ["run_pilot.py", "--output", str(output), "--gate", str(gate)]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", argv)
        run_pilot.main()

    state.run = run
    yield state
    assert not state.live, "Fake children were orphaned"
    assert all(stream.closed for stream in state.streams), "Child log stream leaked"


def test_full_pilot_collects_two_balanced_rounds_trains_alone_and_evaluates_one_policy(pilot):
    pilot.run()
    jobs = [row[0] for row in pilot.launches]
    assert len(jobs) == 22
    assert [j["stage"] for j in jobs] == [0]*5 + [1] + [2]*5 + [3] + [4]*5 + [5]*5
    assert pilot.maximum == 5
    for index, seeds in ((0, range(6301, 6306)), (1, range(6401, 6406))):
        collectors = [j for j in jobs if j["kind"] == "collect" and j["round"] == index]
        assert [j["scenario"] for j in collectors] == list(SCENARIOS)
        assert [j["seed"] for j in collectors] == list(seeds)
        assert len({j["cpu_mask"] for j in collectors}) == 5
    for command, job in zip(pilot.commands, jobs):
        if job["kind"] in ("center", "rl"):
            assert "--training-seed" not in command
        if job["kind"] == "rl":
            assert command[command.index("--model")+1] == str(pilot.output / "train_round1/model_final.pt")
    final = read(pilot.output / "completion.json")
    assert final["training_episodes"] == final["evaluation_episodes"] == 10
    assert final["goal_claim"] is final["controller_acceptance"] is False
    comparison = read(pilot.output / "comparison.json")
    assert len(comparison["comparisons"]) == 5
    assert not comparison["generalization_claim"]
    model = torch.load(pilot.output / "train_round1/model_final.pt", weights_only=False)
    assert {s: len(v["rewards"]) for s, v in model["learner"]["replay"].items()} == dict.fromkeys(SCENARIOS, 150)


@pytest.mark.parametrize("behavior", ["failure", "paused", "missing_completion", "malformed"])
def test_unsuccessful_collector_drains_workers_and_never_starts_training(pilot, behavior):
    key = f"collect_round0/{SCENARIOS[1]}"
    pilot.behaviors[key] = behavior
    if behavior == "malformed":
        with pytest.raises(ValueError):
            pilot.run()
    else:
        pilot.run()
    assert all(j[0]["stage"] == 0 for j in pilot.launches)
    assert (pilot.output / "STOP").exists()
    assert not (pilot.output / "completion.json").exists()
    assert not (pilot.output / "train_round0/completion.json").exists()
    if behavior == "failure":
        assert read(pilot.output / "failure.json")["returncode"] == 9
        assert read(pilot.output / "status.json")["status"] in ("failed", "paused_or_failed")


def test_launch_exception_waits_all_already_started_children(pilot):
    pilot.behaviors[f"collect_round0/{SCENARIOS[2]}"] = "launch_error"
    with pytest.raises(OSError, match="synthetic Popen failure"):
        pilot.run()
    assert pilot.waited == [f"collect_round0/{s}" for s in SCENARIOS[:2]]
    assert len(pilot.commands) == 3
    assert (pilot.output / "STOP").exists()


@pytest.mark.parametrize("location", ["parent", "stage", "child"])
def test_preexisting_stop_anywhere_blocks_all_startup_work(pilot, location):
    # Pause first to obtain the exact path-dependent resume plan.
    pilot.behaviors[f"collect_round0/{SCENARIOS[0]}"] = "paused"
    pilot.run()
    (pilot.output / "STOP").unlink()
    target = {"parent": pilot.output, "stage": pilot.output / "rl",
              "child": pilot.output / "rl" / SCENARIOS[-1]}[location]
    target.mkdir(parents=True, exist_ok=True)
    (target / "STOP").touch()
    pilot.commands.clear()
    pilot.run(resume=True)
    assert pilot.commands == []
    assert read(pilot.output / "status.json")["status"] == "paused"


@pytest.mark.parametrize("location", ["parent", "stage", "child"])
def test_stop_appearing_during_launch_prevents_new_siblings_and_drains(pilot, location):
    first = f"collect_round0/{SCENARIOS[0]}"

    def stop_on_first(job):
        if job["key"] != first:
            return
        target = {"parent": pilot.output, "stage": pilot.output / "collect_round0",
                  "child": pilot.output / first}[location]
        (target / "STOP").touch()

    pilot.on_launch = stop_on_first
    pilot.run()
    assert len(pilot.commands) == 1, "STOP must prevent launching another numerical worker"
    assert not (pilot.output / "completion.json").exists()
    assert (pilot.output / "STOP").exists()


def test_active_orphan_lock_blocks_even_an_earlier_stage(pilot):
    pilot.output.mkdir()
    with run_budget.exclusive_run(pilot.output / "rl" / SCENARIOS[-1]):
        # A fresh existing directory is normally rejected, so use the stage boundary directly.
        jobs = run_pilot.jobs_for(pilot.output, SCENARIOS)
        with pytest.raises(OSError):
            run_pilot.ensure_idle([pilot.output / j["key"] for j in jobs])
    assert pilot.commands == []


def test_busy_sibling_lock_blocks_whole_collection_wave(pilot):
    jobs = [j for j in run_pilot.jobs_for(pilot.output, SCENARIOS) if j["stage"] == 0]
    with run_budget.exclusive_run(pilot.output / jobs[-1]["key"]):
        with pytest.raises(OSError):
            run_pilot.run_stage(pilot.output, 0, jobs,
                                dict(source_pins=PINS, runtime_versions=VERSIONS), SCENARIOS)
    assert pilot.commands == []


def test_completed_collection_is_fully_validated_before_missing_sibling_launch(pilot):
    jobs = [j for j in run_pilot.jobs_for(pilot.output, SCENARIOS) if j["stage"] == 0]
    job = jobs[-1]
    folder = completed_run(pilot.output / job["key"], job["scenario"], cpu_mask=job["cpu_mask"])
    change_json(folder / "episode_00_trace.json", lambda rows: rows[30].update(terminated=True))
    with pytest.raises(ValueError):
        run_pilot.run_stage(pilot.output, 0, jobs, dict(source_pins=PINS, runtime_versions=VERSIONS), SCENARIOS)
    assert pilot.commands == []


def test_completed_collection_experience_is_validated_before_sibling_launch(pilot):
    jobs = [j for j in run_pilot.jobs_for(pilot.output, SCENARIOS) if j["stage"] == 0]
    job = jobs[-1]
    folder = completed_run(pilot.output / job["key"], job["scenario"], cpu_mask=job["cpu_mask"])
    # Keep the completion hash current: the payload itself must be validated before skipping.
    change_experience(folder, lambda payload: payload["transitions"].pop())
    with pytest.raises(ValueError):
        run_pilot.run_stage(pilot.output, 0, jobs, dict(source_pins=PINS, runtime_versions=VERSIONS), SCENARIOS)
    assert pilot.commands == []


def test_valid_completed_collector_is_skipped_and_checkpoint_sibling_resumes(pilot):
    jobs = [j for j in run_pilot.jobs_for(pilot.output, SCENARIOS) if j["stage"] == 0]
    completed_run(pilot.output / jobs[0]["key"], jobs[0]["scenario"], cpu_mask=jobs[0]["cpu_mask"])
    resume_folder = pilot.output / jobs[1]["key"]
    resume_folder.mkdir(parents=True)
    (resume_folder / "checkpoint.pt").write_bytes(b"synthetic checkpoint")
    assert run_pilot.run_stage(pilot.output, 0, jobs, dict(source_pins=PINS, runtime_versions=VERSIONS), SCENARIOS)
    assert len(pilot.commands) == 4
    assert "--resume" in pilot.commands[0]
    assert all("--resume" not in c for c in pilot.commands[1:])


@pytest.mark.parametrize("field", ["seed", "sample_counts", "provenance"])
def test_completed_trainer_checks_actual_checkpoint_identity_counters_and_provenance(pilot, field):
    pilot.run()
    plan = read(pilot.output / "plan.json")
    job = next(j for j in plan["jobs"] if j["key"] == "train_round1")
    folder = pilot.output / job["key"]
    path = folder / "model_final.pt"
    model = torch.load(path, weights_only=False)
    if field == "seed":
        model["learner"]["spec"]["seed"] = 99
    elif field == "sample_counts":
        model["learner"]["sample_counts"] = dict.fromkeys(SCENARIOS, 5999)
    else:
        model["training_profiles"][-1]["experience_sha256"] = "wrong-collector"
    torch.save(model, path)
    change_json(folder / "completion.json", lambda r: r.update(model_sha256=run_budget.file_hash(path)))
    with pytest.raises(ValueError):
        run_pilot.validate_job(pilot.output, job, plan)


def test_changed_gate_evidence_blocks_all_children(pilot):
    evidence = Path(read(pilot.gate)["evidence"]["review_report"]["path"])
    evidence.write_text("changed review", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        pilot.run()
    assert pilot.commands == []
    assert not pilot.output.exists()


def test_completed_pilot_and_changed_plan_refuse_duplicate_work(pilot):
    pilot.run()
    pilot.commands.clear()
    with pytest.raises(ValueError, match="already complete"):
        pilot.run(resume=True)
    assert pilot.commands == []
    (pilot.output / "completion.json").unlink()
    change_json(pilot.output / "plan.json", lambda p: p.update(batch_size=32))
    with pytest.raises(ValueError, match="plan changed"):
        pilot.run(resume=True)
    assert pilot.commands == []


@pytest.mark.parametrize("location", ["stage", "child", "parent"])
def test_stop_during_poll_propagates_before_another_worker_can_progress(pilot, location):
    first = f"collect_round0/{SCENARIOS[0]}"
    created = []

    def on_poll(job, polls):
        if job["key"] == first and polls == 1:
            target = {"stage": pilot.output / "collect_round0", "child": pilot.output / first,
                      "parent": pilot.output}[location]
            (target / "STOP").write_text("user stop: preserve this text", encoding="utf-8")
            created.append(target / "STOP")
        elif created:
            assert (pilot.output / "STOP").exists(), "Stop must propagate before the next child poll"

    pilot.on_poll = on_poll
    pilot.run()
    assert len(pilot.commands) == 5
    assert len(created) == 1
    assert created[0].read_text(encoding="utf-8") == "user stop: preserve this text"
    assert not (pilot.output / "completion.json").exists()
    assert not (pilot.output / "train_round0/completion.json").exists()


def test_parent_sets_numerical_threads_before_scenario_torch_import(pilot, monkeypatch):
    import builtins
    import os
    original = builtins.__import__
    seen = []
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "8")

    def check_import(name, *args, **kwargs):
        if name == "td3":
            seen.append(name)
            assert all(os.environ[k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    pilot.behaviors[f"collect_round0/{SCENARIOS[0]}"] = "paused"
    pilot.run()
    assert seen
