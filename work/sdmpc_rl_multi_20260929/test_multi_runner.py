"""Real TD3 integration with synthetic environments, experiences, and resume points."""
import copy
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_budget
import run_pilot
import train_round
from budget_runtime import read, save
from td3 import SCENARIOS, TD3
from test_multi_contracts import (COORDINATOR, ENVIRONMENT, MANIFEST, PINS, SCHEMA, VERSIONS,
    collection_group, contract_for, settings_for, trajectory, synthetic_training, change_json)


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def frozen_policy(path):
    learner = TD3(len(SCHEMA["names"]), 6300)
    profiles = []
    for scenario in SCENARIOS:
        settings = settings_for("model-source-"+scenario, scenario)
        for transition in trajectory(settings)[2]:
            learner.add(*transition, scenario=scenario)
        profiles.append(dict(scenario=scenario, seed=settings["seeds"][0], run_id=settings["run_id"],
            profile_sha256=settings["profile_sha256"][0], experience_sha256="synthetic-source-hash"))
    # Distinguish loading a frozen actor from silently rebuilding the zero actor.
    with torch.no_grad():
        learner.actor[-2].bias.copy_(torch.tensor([.4, -.25]))
    torch.save(dict(format=run_budget.POLICY_FORMAT, learner=learner.state_dict(),
        contract=contract_for(settings), training_run_id="synthetic-training-round0",
        training_profiles=profiles, training_round=0), path)
    return learner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    state = SimpleNamespace(output=None, pause_at=None, actions=[], instances=[], profile_drift=False,
                            restored=[], fail_reset=False)

    class FakeEnv:
        def __init__(self, runtime, scenario, training_seed=None, guard_mode="physical"):
            assert guard_mode == "physical"
            self.settings = settings_for("environment", scenario, "collect" if training_seed else "center")
            self.profile_hash = ("drift" if state.profile_drift else
                settings_for("profile", scenario, "collect" if training_seed else "center",
                             int(training_seed is not None and training_seed >= 6401))["profile_sha256"][0])
            self.observer = SimpleNamespace(names=SCHEMA["names"], delay=2,
                                           contract=lambda: SCHEMA["normalization"])
            self.rng = np.random.default_rng(100 if training_seed is None else training_seed+100)
            self.closed = False
            state.instances.append(self)

        def contract(self):
            return copy.deepcopy(COORDINATOR)

        def observation(self):
            return np.array([0., 0.] if self.k == 80 else [self.k/80., self.memory], dtype=np.float32)

        def reset(self):
            if state.fail_reset:
                raise RuntimeError("synthetic reset failure")
            self.k, self.memory, self.warmup_ttt = 5, float(self.rng.normal()), 5.
            self.sim = SimpleNamespace(total_ttt=5., state=SimpleNamespace(time_sec=900.))
            self.rows = trajectory(self.settings)[1]
            return self.observation()

        def step(self, action, mode, actor_seconds, actor_cpu_seconds=None):
            assert mode in ("center", "rl")
            before = self.observation()
            row = copy.deepcopy(self.rows[self.k-5])
            row["decision_wall_seconds"] += actor_seconds-row["actor_wall_seconds"]
            row["decision_cpu_seconds"] += actor_cpu_seconds-row["actor_cpu_seconds"]
            row.update(actor_wall_seconds=actor_seconds, actor_cpu_seconds=actor_cpu_seconds,
                       action_requested=action.copy())
            self.k += 1
            self.memory += float(self.rng.normal(scale=.02))
            self.sim.total_ttt += 1.
            self.sim.state.time_sec += 180.
            after = self.observation()
            terminal = self.k == 80
            state.actions.append((before, action.copy(), -.01, after, terminal))
            if self.k == state.pause_at:
                (state.output / "STOP").touch()
            return after, -.01, terminal, row

        def checkpoint(self):
            return dict(k=self.k, memory=self.memory, profile_hash=self.profile_hash,
                        sim=copy.deepcopy(self.sim), rng=copy.deepcopy(self.rng.bit_generator.state))

        def restore(self, checkpoint):
            state.restored.append(copy.deepcopy(checkpoint))
            self.k, self.memory = checkpoint["k"], checkpoint["memory"]
            self.sim, self.warmup_ttt = copy.deepcopy(checkpoint["sim"]), 5.
            self.rng.bit_generator.state = checkpoint["rng"]
            self.rows = trajectory(self.settings)[1]
            return self.observation()

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "budget_env", SimpleNamespace(BudgetEnv=FakeEnv))
    monkeypatch.setattr(run_budget, "pins", lambda _: copy.deepcopy(PINS))
    monkeypatch.setattr(run_budget, "verify_pins", lambda *_: None)
    monkeypatch.setattr(run_budget, "runtime_versions", lambda: copy.deepcopy(VERSIONS))
    monkeypatch.setattr(run_budget, "scenario_manifest", lambda _: copy.deepcopy(MANIFEST))
    monkeypatch.setattr(run_budget, "boot", lambda *_: dict(
        cfg={}, options={}, rc=SimpleNamespace(to_plain_dict=lambda value: value)))

    def run(name, mode="collect", round_index=0, model=None, resume=False, scenario=SCENARIOS[0], extra=()):
        folder = tmp_path / name
        state.output = folder
        argv = ["run_budget.py", "--output", str(folder), "--mode", mode,
                "--scenario", scenario, "--round", str(round_index)]
        if mode == "collect":
            argv += ["--training-seed", str((6301 if round_index == 0 else 6401)+SCENARIOS.index(scenario))]
        if model is not None:
            argv += ["--model", str(model)]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        run_budget.main()
        return folder

    state.run = run
    return state


@pytest.mark.parametrize("round_index", [0, 1])
def test_collector_persists_exact_requested_experience_with_frozen_policy(runner, tmp_path, monkeypatch, round_index):
    model = tmp_path / "frozen.pt" if round_index else None
    learner = frozen_policy(model) if model else TD3(2, 6300)
    original_hash = run_budget.file_hash(model) if model else None
    monkeypatch.setattr(TD3, "update", lambda *_: pytest.fail("Collector called TD3.update"))
    monkeypatch.setattr(TD3, "add", lambda *_args, **_kwargs: pytest.fail("Collector mutated learner replay"))
    folder = runner.run("collect", round_index=round_index, model=model)
    payload = load(folder / "experience.pt")
    assert_state_equal(payload["transitions"], runner.actions)
    assert len(runner.actions) == 75
    rng = np.random.default_rng(6301 if round_index == 0 else 6401)
    rows = read(folder / "episode_00_trace.json")
    for index, (obs, action, reward, next_obs, terminal) in enumerate(runner.actions):
        expected = (rng.uniform(-1., 1., 2).astype(np.float32) if round_index == 0 and index < 32
            else np.clip(learner.act(obs)+rng.normal(0., .3, 2), -1., 1.).astype(np.float32))
        np.testing.assert_array_equal(action, expected)
        with torch.no_grad():
            inputs = torch.from_numpy(np.concatenate((obs, action))).unsqueeze(0)
            expected_q = [float(critic(inputs).item()) for critic in learner.critics]
        assert rows[index]["policy_q"] == expected_q
        assert reward == rows[index]["reward"]
        assert terminal is (index == 74)
    assert not (folder / "model_final.pt").exists()
    if model:
        assert run_budget.file_hash(model) == original_hash
    assert all(env.closed for env in runner.instances)


@pytest.mark.parametrize("mode", ["center", "rl"])
def test_evaluation_has_no_training_seed_exploration_replay_or_updates(runner, tmp_path, monkeypatch, mode):
    model = tmp_path / "frozen.pt" if mode == "rl" else None
    learner = frozen_policy(model) if model else None
    monkeypatch.setattr(TD3, "update", lambda *_: pytest.fail("Evaluation trained the actor"))
    monkeypatch.setattr(TD3, "add", lambda *_args, **_kwargs: pytest.fail("Evaluation admitted replay"))
    folder = runner.run(mode, mode=mode, model=model)
    settings = read(folder / "settings.json")
    checkpoint = load(folder / "checkpoint.pt")
    assert settings["seeds"] == [None]
    assert settings["profile_sha256"] == [MANIFEST[0]["profile_sha256"]]
    assert checkpoint["experience"] == []
    assert "learner" not in checkpoint
    assert not (folder / "experience.pt").exists()
    for obs, action, *_ in runner.actions:
        np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32) if learner is None else learner.act(obs))
    if mode == "center":
        assert all(row["policy_q"] is None for row in checkpoint["trace"])


@pytest.mark.parametrize("mode,round_index", [("collect", 0), ("collect", 1), ("rl", 1)])
def test_resume_restores_environment_and_exploration_rng_and_reloads_immutable_policy(
        runner, tmp_path, monkeypatch, mode, round_index):
    model = tmp_path / "frozen.pt" if round_index else None
    if model:
        frozen_policy(model)
    baseline = runner.run("baseline", mode, round_index, model)
    expected_actions = copy.deepcopy(runner.actions)
    expected_checkpoint = load(baseline / "checkpoint.pt")
    runner.actions.clear()
    runner.pause_at = 37
    paused = runner.run("resumed", mode, round_index, model)
    checkpoint = load(paused / "checkpoint.pt")
    assert checkpoint["environment"]["k"] == 37
    assert "learner" not in checkpoint
    assert len(checkpoint["trace"]) == 32
    assert not (paused / "completion.json").exists()
    (paused / "STOP").unlink()
    runner.pause_at = None
    restored_policies = []
    original_load = TD3.load_state_dict

    def record_load(self, payload):
        restored_policies.append(copy.deepcopy(payload))
        return original_load(self, payload)

    monkeypatch.setattr(TD3, "load_state_dict", record_load)
    runner.run("resumed", mode, round_index, model, resume=True)
    assert_state_equal(runner.actions, expected_actions)
    actual = load(paused / "checkpoint.pt")
    assert_state_equal(actual["environment"], expected_checkpoint["environment"])
    assert_state_equal(actual["exploration_rng"], expected_checkpoint["exploration_rng"])
    assert_state_equal(actual["experience"], expected_checkpoint["experience"])
    assert_state_equal(runner.restored[-1], checkpoint["environment"])
    assert len(restored_policies) == int(model is not None)
    if model:
        assert_state_equal(restored_policies[0], load(model)["learner"])


def test_terminal_checkpoint_resume_finalizes_without_duplicate_transition(runner, monkeypatch):
    original_save = run_budget.save

    def interrupt(path, value):
        if path.name == "episode_00_summary.json":
            raise RuntimeError("synthetic finalization interruption")
        original_save(path, value)

    monkeypatch.setattr(run_budget, "save", interrupt)
    with pytest.raises(RuntimeError, match="finalization interruption"):
        runner.run("terminal")
    folder = runner.output
    assert load(folder / "checkpoint.pt")["environment"]["k"] == 80
    assert not (folder / "completion.json").exists()
    monkeypatch.setattr(run_budget, "save", original_save)
    runner.run("terminal", resume=True)
    assert len(runner.actions) == 75
    assert len(load(folder / "experience.pt")["transitions"]) == 75


@pytest.mark.parametrize("fault", ["source", "runtime", "scenario", "profile", "model"])
def test_resume_rejects_drift_before_any_new_step_and_preserves_settings(runner, tmp_path, monkeypatch, fault):
    model = tmp_path / "frozen.pt"
    frozen_policy(model)
    folder = tmp_path / "paused"
    folder.mkdir()
    (folder / "STOP").touch()
    runner.run("paused", "collect", 1, model)
    before = (folder / "settings.json").read_bytes()
    (folder / "STOP").unlink()
    scenario = SCENARIOS[0]
    if fault == "source":
        monkeypatch.setattr(run_budget, "pins", lambda _: {"snapshot": "drift"})
    elif fault == "runtime":
        monkeypatch.setattr(run_budget, "runtime_versions", lambda: dict(VERSIONS, torch="drift"))
    elif fault == "scenario":
        scenario = SCENARIOS[1]
    elif fault == "profile":
        runner.profile_drift = True
    else:
        with model.open("ab") as stream:
            stream.write(b"changed policy")
    with pytest.raises(ValueError):
        runner.run("paused", "collect", 1, model, resume=True, scenario=scenario)
    assert runner.actions == []
    assert (folder / "settings.json").read_bytes() == before
    assert not (folder / "completion.json").exists()


def test_startup_failure_without_checkpoint_preserves_identity_and_checks_settings(runner, monkeypatch):
    runner.fail_reset = True
    with pytest.raises(RuntimeError, match="reset failure"):
        runner.run("startup")
    folder = runner.output
    settings = read(folder / "settings.json")
    assert not (folder / "checkpoint.pt").exists()
    runner.fail_reset = False
    monkeypatch.setattr(run_budget, "runtime_versions", lambda: dict(VERSIONS, torch="drift"))
    with pytest.raises(ValueError, match="settings/runtime"):
        runner.run("startup")
    assert read(folder / "settings.json") == settings
    monkeypatch.setattr(run_budget, "runtime_versions", lambda: copy.deepcopy(VERSIONS))
    runner.run("startup")
    assert read(folder / "completion.json")["run_id"] == settings["run_id"]


@pytest.mark.parametrize("location", ["child", "stage", "parent"])
def test_runner_stop_checkpoints_without_new_transitions(runner, tmp_path, location):
    folder = tmp_path / "pilot" / "stage" / "child"
    folder.mkdir(parents=True)
    target = {"child": folder, "stage": folder.parent, "parent": folder.parent.parent}[location]
    (target / "STOP").touch()
    runner.run("pilot/stage/child")
    assert runner.actions == []
    assert load(folder / "checkpoint.pt")["environment"]["k"] == 5
    assert read(folder / "status.json")["status"] == "paused"
    assert not (folder / "completion.json").exists()


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(train_round, "boot", lambda *_: None)
    monkeypatch.setattr(train_round, "pins", lambda _: copy.deepcopy(PINS))
    monkeypatch.setattr(train_round, "runtime_versions", lambda: copy.deepcopy(VERSIONS))
    monkeypatch.setattr(train_round, "verify_pins", lambda *_: None)

    def run(output, collections, round_index=0, model=None, resume=False):
        argv = ["train_round.py", "--output", str(output), "--round", str(round_index),
                "--collections", *map(str, collections)]
        if model is not None:
            argv += ["--model", str(model)]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", argv)
        train_round.main()
        return output

    return run


def test_actual_shared_training_two_rounds_preserves_episode_boundaries_and_balance(tmp_path, trainer):
    first = collection_group(tmp_path / "collect_round0")
    output0 = trainer(tmp_path / "train_round0", first)
    model0 = load(output0 / "model_final.pt")
    second = collection_group(tmp_path / "collect_round1", 1, run_budget.file_hash(output0 / "model_final.pt"))
    output1 = trainer(tmp_path / "train_round1", second, 1, output0 / "model_final.pt")
    model1 = load(output1 / "model_final.pt")
    initial = TD3(2, 6300).state_dict()
    assert any(not torch.equal(initial["actor"][k], model0["learner"]["actor"][k]) for k in initial["actor"])
    assert any(not torch.equal(model0["learner"]["actor"][k], model1["learner"]["actor"][k]) for k in initial["actor"])
    for index, model, output in ((0, model0, output0), (1, model1, output1)):
        learner = model["learner"]
        assert learner["updates"] == 375*(index+1)
        assert learner["sample_counts"] == dict.fromkeys(SCENARIOS, 3000*(index+1))
        assert len(model["training_profiles"]) == 5*(index+1)
        metrics = read(output / "metrics.json")
        assert len(metrics) == 375
        assert all(row["sampled_per_scenario"] == dict.fromkeys(SCENARIOS, 8) for row in metrics)
        diagnostics = read(output / "q_diagnostics.json")["per_scenario"]
        for s in SCENARIOS:
            replay = learner["replay"][s]
            assert len(replay["rewards"]) == 75*(index+1)
            assert torch.where(replay["terminated"])[0].tolist() == [74+75*r for r in range(index+1)]
            assert diagnostics[s]["true_terminals"] == index+1
            for r in range(index+1):
                assert torch.equal(replay["next_observations"][75*r:75*r+74],
                                   replay["observations"][75*r+1:75*r+75])
            if index:
                assert not torch.equal(replay["next_observations"][74], replay["observations"][75])
            assert all(np.isfinite(diagnostics[s][key]) for key in ("q_mean", "terminal_abs_error_mean"))
        job = next(j for j in run_pilot.jobs_for(tmp_path, SCENARIOS) if j["key"] == f"train_round{index}")
        run_pilot.validate_training(output, job, dict(source_pins=PINS, runtime_versions=VERSIONS), tmp_path)


def test_training_resume_matches_all_weights_optimizers_rng_replay_and_metrics(tmp_path, trainer, monkeypatch):
    collections = collection_group(tmp_path / "collections")
    baseline = trainer(tmp_path / "baseline", collections)
    stopped = tmp_path / "resumed"
    update = TD3.update

    def pause(self, batch_size=40):
        row = update(self, batch_size)
        if self.updates == 25:
            (stopped / "STOP").touch()
        return row

    monkeypatch.setattr(TD3, "update", pause)
    trainer(stopped, collections)
    checkpoint = load(stopped / "checkpoint.pt")
    assert checkpoint["completed"] == 25
    assert not (stopped / "completion.json").exists()
    monkeypatch.setattr(TD3, "update", update)
    (stopped / "STOP").unlink()
    trainer(stopped, collections, resume=True)
    assert_state_equal(load(baseline / "model_final.pt")["learner"], load(stopped / "model_final.pt")["learner"])
    assert read(baseline / "metrics.json") == read(stopped / "metrics.json")
    assert load(baseline / "model_final.pt")["training_profiles"] == load(stopped / "model_final.pt")["training_profiles"]


@pytest.mark.parametrize("fault", ["settings", "completed", "metrics", "updates", "seed", "sample_counts", "replay"])
def test_training_resume_rejects_corrupt_checkpoint_before_update(tmp_path, trainer, monkeypatch, fault):
    collections = collection_group(tmp_path / "collections")
    output = tmp_path / "train"
    output.mkdir()
    (output / "STOP").touch()
    trainer(output, collections)
    path = output / "checkpoint.pt"
    checkpoint = load(path)
    if fault == "settings":
        checkpoint["settings"]["batch_size"] = 32
    elif fault == "completed":
        checkpoint["completed"] = True
    elif fault == "metrics":
        checkpoint["metrics"] = [{"updates": 1}]
    elif fault == "updates":
        checkpoint["completed"] = 1
        checkpoint["metrics"] = [{"updates": 1}]
    elif fault == "seed":
        checkpoint["learner"]["spec"]["seed"] = 99
    elif fault == "sample_counts":
        learner = TD3(2, 6300)
        learner.load_state_dict(checkpoint["learner"])
        checkpoint["metrics"] = [learner.update(40)]
        checkpoint["learner"] = learner.state_dict()
        checkpoint["completed"] = 1
        checkpoint["learner"]["sample_counts"] = dict.fromkeys(SCENARIOS, 7)
    else:
        group = checkpoint["learner"]["replay"][SCENARIOS[0]]
        for key in group:
            group[key] = group[key][:-1]
    torch.save(checkpoint, path)
    (output / "STOP").unlink()

    def unexpected_update(*_):
        pytest.fail("Corrupt trainer checkpoint reached a new gradient update")

    monkeypatch.setattr(TD3, "update", unexpected_update)
    with pytest.raises(ValueError):
        trainer(output, collections, resume=True)
    assert not (output / "completion.json").exists()


def test_training_changed_collection_hash_rejected_on_resume(tmp_path, trainer):
    collections = collection_group(tmp_path / "collections")
    output = tmp_path / "train"
    output.mkdir()
    (output / "STOP").touch()
    trainer(output, collections)
    original = (output / "settings.json").read_bytes()
    completion = read(collections[0] / "completion.json")
    completion["elapsed_wall_seconds"] += 1.
    save(collections[0] / "completion.json", completion)
    (output / "STOP").unlink()
    with pytest.raises(ValueError, match="settings changed"):
        trainer(output, collections, resume=True)
    assert (output / "settings.json").read_bytes() == original


def test_wrong_predecessor_round_rejected_before_training(tmp_path, trainer, monkeypatch):
    path = tmp_path / "wrong-predecessor.pt"
    frozen_policy(path)
    model = load(path)
    model["training_round"] = 1
    torch.save(model, path)
    collections = collection_group(tmp_path / "collections", 1, run_budget.file_hash(path))
    monkeypatch.setattr(TD3, "update", lambda *_: pytest.fail("Wrong predecessor reached training"))
    with pytest.raises(ValueError, match="predecessor training round"):
        trainer(tmp_path / "train", collections, 1, path)


@pytest.mark.parametrize("fault", ["observations", "actions", "rewards", "next_observations", "terminated",
    "evaluation_data", "provenance_hash", "provenance_order", "training_run_id",
    "metric_update", "metric_samples", "metric_phase", "metric_loss"])
def test_resume_reconciles_equal_size_replay_provenance_and_metric_phase(tmp_path, trainer, monkeypatch, fault):
    collections = collection_group(tmp_path / "collections")
    output = tmp_path / "train"
    output.mkdir()
    (output / "STOP").touch()
    trainer(output, collections)
    checkpoint = load(output / "checkpoint.pt")
    replay = checkpoint["learner"]["replay"][SCENARIOS[0]]
    if fault in ("observations", "actions", "rewards", "next_observations"):
        replay[fault][0] += .01
    elif fault == "terminated":
        replay["terminated"][-1] = False
    elif fault == "evaluation_data":
        # Equal shape and legal values cannot establish training-only provenance.
        replay["actions"].zero_()
        replay["rewards"].fill_(-999.)
    elif fault == "provenance_hash":
        checkpoint["training_profiles"][0]["experience_sha256"] = "wrong"
    elif fault == "provenance_order":
        checkpoint["training_profiles"].reverse()
    elif fault == "training_run_id":
        checkpoint["training_run_id"] = "another-training-run"
    else:
        learner = TD3(2, 6300)
        learner.load_state_dict(checkpoint["learner"])
        checkpoint["metrics"] = [learner.update(40), learner.update(40)]
        checkpoint["learner"], checkpoint["completed"] = learner.state_dict(), 2
        if fault == "metric_update":
            checkpoint["metrics"][0]["updates"] = 2
        elif fault == "metric_samples":
            checkpoint["metrics"][0]["sampled_per_scenario"] = dict.fromkeys(SCENARIOS, 7)
        elif fault == "metric_phase":
            checkpoint["metrics"][0]["actor_loss"] = checkpoint["metrics"][1].pop("actor_loss")
        else:
            checkpoint["metrics"][1]["actor_loss"] = float("nan")
    torch.save(checkpoint, output / "checkpoint.pt")
    (output / "STOP").unlink()
    monkeypatch.setattr(TD3, "update", lambda *_: pytest.fail("Corruption reached another gradient update"))
    with pytest.raises(ValueError):
        trainer(output, collections, resume=True)
    assert not (output / "completion.json").exists()


@pytest.fixture(scope="module")
def admitted_round0(tmp_path_factory):
    output = tmp_path_factory.mktemp("multi-admitted-training")
    collection_group(output / "collect_round0")
    job = next(j for j in run_pilot.jobs_for(output, SCENARIOS) if j["key"] == "train_round0")
    synthetic_training(output, job, dict(source_pins=PINS, runtime_versions=VERSIONS))
    train_round.validate_training_output(output / "train_round0", 0, PINS, VERSIONS)
    return output / "train_round0"


@pytest.mark.parametrize("fault", ["observations", "actions", "rewards", "next_observations", "terminated",
    "provenance_hash", "profile_hash", "seed_schedule", "provenance_order", "training_run_id",
    "metric_update", "metric_samples", "metric_phase", "metric_loss"])
def test_completed_training_reconciles_actual_input_columns_and_all_provenance(tmp_path, admitted_round0, fault):
    output = tmp_path / "completed"
    shutil.copytree(admitted_round0, output)
    path = output / "model_final.pt"
    model = load(path)
    replay = model["learner"]["replay"][SCENARIOS[0]]
    if fault in ("observations", "actions", "rewards", "next_observations"):
        replay[fault][0] += .01
    elif fault == "terminated":
        replay["terminated"][-1] = False
    elif fault in ("provenance_hash", "profile_hash", "seed_schedule"):
        key, value = {"provenance_hash": ("experience_sha256", "wrong"),
            "profile_hash": ("profile_sha256", "wrong"), "seed_schedule": ("seed", 9999)}[fault]
        model["training_profiles"][0][key] = value
    elif fault == "provenance_order":
        model["training_profiles"].reverse()
    elif fault == "training_run_id":
        model["training_run_id"] = "other"
    else:
        metrics = read(output / "metrics.json")
        if fault == "metric_update":
            metrics[0]["updates"] = 376
        elif fault == "metric_samples":
            metrics[0]["sampled_per_scenario"] = dict.fromkeys(SCENARIOS, 7)
        elif fault == "metric_phase":
            metrics[0]["actor_loss"] = metrics[1].pop("actor_loss")
        else:
            metrics[1]["actor_loss"] = float("nan")
        save(output / "metrics.json", metrics)
    torch.save(model, path)
    change_json(output / "completion.json", lambda r: r.update(model_sha256=run_budget.file_hash(path)))
    with pytest.raises(ValueError):
        train_round.validate_training_output(output, 0, PINS, VERSIONS)


def test_round1_resume_preserves_predecessor_replay_and_odd_update_phase(tmp_path, admitted_round0, trainer, monkeypatch):
    model_path = admitted_round0 / "model_final.pt"
    collections = collection_group(tmp_path / "collections", 1, run_budget.file_hash(model_path))
    baseline = trainer(tmp_path / "baseline", collections, 1, model_path)
    output = tmp_path / "resumed"
    original = TD3.update

    def pause(self, batch_size=40):
        row = original(self, batch_size)
        if self.updates == 376:
            (output / "STOP").touch()
        return row

    monkeypatch.setattr(TD3, "update", pause)
    trainer(output, collections, 1, model_path)
    checkpoint = load(output / "checkpoint.pt")
    assert checkpoint["completed"] == 1
    assert "actor_loss" in checkpoint["metrics"][0]
    assert len(checkpoint["training_profiles"]) == 10
    monkeypatch.setattr(TD3, "update", original)
    (output / "STOP").unlink()
    trainer(output, collections, 1, model_path, resume=True)
    assert_state_equal(load(baseline / "model_final.pt")["learner"], load(output / "model_final.pt")["learner"])
    assert read(baseline / "metrics.json") == read(output / "metrics.json")
