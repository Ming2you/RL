"""Synthetic-only terminal fitting tests; production loading is prohibited."""

from copy import deepcopy
import json
from pathlib import Path
import random

import terminal_fit as diagnostic
import numpy as np
import pytest

torch = diagnostic.torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def prohibit_production(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Production model/source loading or model export forbidden in tests")
    monkeypatch.setattr(diagnostic, "load_base", forbidden)
    monkeypatch.setattr(diagnostic.common, "pins", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    diagnostic.single_thread()


@pytest.fixture
def learner():
    value = diagnostic.TD3(3, seed=17, hidden=8)
    rng = np.random.default_rng(23)
    for number, scenario in enumerate(diagnostic.SCENARIOS):
        for index in range(150):
            reward = -0.25 * (number + 1) - index / 200 if index in (74, 149) else -99.0
            value.add(rng.uniform(0.1, 1, 3), rng.uniform(-0.5, 0.5, 2), reward,
                      rng.uniform(2, 3, 3), index in (74, 149), scenario)
    value.update()
    value.update()
    return value


@pytest.fixture
def sandbox(tmp_path, monkeypatch, learner):
    repository = tmp_path / "repository"
    goal = repository / "goal"
    goal.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "REPO", repository)
    monkeypatch.setattr(diagnostic, "GOAL_ROOT", goal)
    source = tmp_path / "synthetic-source.txt"
    model = tmp_path / "synthetic-input.txt"
    source.write_text("synthetic source", encoding="utf-8")
    model.write_text("synthetic input; never deserialized", encoding="utf-8")
    state = learner.state_dict()
    calls = []

    def load():
        calls.append("load")
        return {"learner": state}, learner

    def identity(stop):
        stop()
        return dict(input_sha256={str(model): diagnostic.file_hash(model)},
                    source_sha256={str(source): diagnostic.file_hash(source)},
                    runtime_sha256="synthetic")

    monkeypatch.setattr(diagnostic, "load_base", load)
    monkeypatch.setattr(diagnostic, "capture_identity", identity)
    return dict(repository=repository, goal=goal, source=source, model=model,
                state=state, calls=calls, output=tmp_path / "output")


def run(output, **kwargs):
    return diagnostic.run_diagnostic(output, updates=3, log_updates=(0, 1, 3), **kwargs)


def fit(learner, state=None, updates=3, logs=(0, 1, 3)):
    result = dict(folds=[])
    diagnostic.fit(learner, learner.state_dict() if state is None else state, result,
                   lambda: None, lambda: None, updates=updates, log_updates=logs)
    return result


def test_terminal_only_exact_immediate_rewards_and_disjoint_balanced_folds(learner):
    state = learner.state_dict()
    data = diagnostic.extract_terminals(state)
    assert data["inputs"].shape == (10, 5) and data["targets"].shape == (10, 1)
    assert [row["row_index"] for row in data["rows"]] == [74, 149] * 5
    for train_round in (0, 1):
        train, holdout = diagnostic.fold_data(data, train_round)
        assert len(train["rows"]) == len(holdout["rows"]) == 5
        assert [row["scenario"] for row in train["rows"]] == list(diagnostic.SCENARIOS)
        assert [row["scenario"] for row in holdout["rows"]] == list(diagnostic.SCENARIOS)
        assert {r["row_index"] for r in train["rows"]} == {diagnostic.TERMINALS[train_round]}
        assert {r["row_index"] for r in holdout["rows"]} == {diagnostic.TERMINALS[1 - train_round]}
    for i, row in enumerate(data["rows"]):
        group = state["replay"][row["scenario"]]
        index = row["row_index"]
        assert torch.equal(data["inputs"][i], torch.cat((group["observations"][index], group["actions"][index])))
        assert torch.equal(data["targets"][i, 0], group["rewards"][index])
    before = diagnostic.fingerprint(state)
    data["inputs"].zero_()
    data["targets"].zero_()
    assert diagnostic.fingerprint(state) == before


@pytest.mark.parametrize("invalid", ["missing_scenario", "extra_scenario", "missing_terminal",
                                    "misplaced_terminal", "extra_terminal", "terminal_dtype",
                                    "short_replay", "observation_shape", "action_shape",
                                    "reward_shape", "next_shape", "nonfinite_observation",
                                    "nonfinite_action", "nonfinite_reward", "nonfinite_next",
                                    "reward_dtype"])
def test_rejects_invalid_terminal_positions_shapes_and_finiteness(learner, invalid):
    state = learner.state_dict()
    group = state["replay"][diagnostic.SCENARIOS[0]]
    if invalid == "missing_scenario":
        state["replay"].pop(diagnostic.SCENARIOS[-1])
    elif invalid == "extra_scenario":
        state["replay"]["unexpected"] = deepcopy(group)
    elif invalid == "missing_terminal":
        group["terminated"][149] = False
    elif invalid == "misplaced_terminal":
        group["terminated"][74], group["terminated"][73] = False, True
    elif invalid == "extra_terminal":
        group["terminated"][20] = True
    elif invalid == "terminal_dtype":
        group["terminated"] = group["terminated"].float()
    elif invalid == "short_replay":
        group["rewards"] = group["rewards"][:-1]
    elif invalid in ("observation_shape", "action_shape", "next_shape"):
        key = {"observation_shape": "observations", "action_shape": "actions", "next_shape": "next_observations"}[invalid]
        group[key] = group[key][:, :-1]
    elif invalid == "reward_shape":
        group["rewards"] = group["rewards"].unsqueeze(1)
    elif invalid.startswith("nonfinite_"):
        key = {"nonfinite_observation": "observations", "nonfinite_action": "actions",
               "nonfinite_reward": "rewards", "nonfinite_next": "next_observations"}[invalid]
        group[key][74] = float("nan")
    else:
        group["rewards"] = group["rewards"].double()
    with pytest.raises(ValueError):
        diagnostic.extract_terminals(state)


def test_all_layers_eligible_fresh_adam_and_no_original_mutation(learner):
    learner.critics.requires_grad_(False).eval()
    before = diagnostic.learner_fingerprint(learner)
    critics, optimizer = diagnostic.critic_copy(learner.critics)
    assert len(optimizer.state) == 0
    assert optimizer.param_groups[0]["lr"] == 0.0003
    assert all(p.requires_grad for p in critics.parameters())
    assert {id(p) for p in critics.parameters()} == {id(p) for p in optimizer.param_groups[0]["params"]}
    assert {p.data_ptr() for p in critics.parameters()}.isdisjoint(p.data_ptr() for p in learner.critics.parameters())
    with torch.no_grad():
        for p in critics.parameters():
            p.fill_(0.2)
    train, _ = diagnostic.fold_data(diagnostic.extract_terminals(learner.state_dict()), 0)
    parameters_before = [p.clone() for p in critics.parameters()]
    expected_loss = sum((q(train["inputs"]) - train["targets"]).square().mean() for q in critics)
    assert diagnostic.critic_step(critics, optimizer, train) == float(expected_loss.detach())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critics.parameters())
    assert all(not torch.equal(old, p) for old, p in zip(parameters_before, critics.parameters()))
    assert diagnostic.learner_fingerprint(learner) == before


def test_metrics_errors_minimum_and_aggregate_are_exact(learner):
    data = diagnostic.extract_terminals(learner.state_dict())
    critics, _ = diagnostic.critic_copy(learner.critics)
    with torch.no_grad():
        for critic, bias in zip(critics, (2.0, -1.0)):
            for p in critic.parameters():
                p.zero_()
            critic[-1].bias.fill_(bias)
    result = diagnostic.metrics(critics, data, 0, [])
    for name, prediction in (("q1", 2.0), ("q2", -1.0), ("min_q", -1.0)):
        errors = [prediction - float(target) for target in data["targets"].flatten()]
        assert result["aggregate"][name] == dict(mse=np.mean(np.square(errors)),
                                                 mae=np.mean(np.abs(errors)), max_abs_error=max(map(abs, errors)))
        for i, row in enumerate(result["rows"]):
            assert row["predictions"][name] == prediction
            assert row["errors"][name] == errors[i]
            assert row["abs_errors"][name] == abs(errors[i])


def test_fresh_initialization_matches_seed8100_td3_and_restores_rng(learner):
    before = diagnostic.global_rng_fingerprint()
    first = diagnostic.fresh_critics(learner)
    random.random()
    np.random.random()
    torch.rand(1)
    after_draw = diagnostic.global_rng_fingerprint()
    second = diagnostic.fresh_critics(learner)
    assert diagnostic.global_rng_fingerprint() == after_draw
    expected = diagnostic.TD3(3, 8100, 8).critics
    assert diagnostic.fingerprint(first.state_dict()) == diagnostic.fingerprint(second.state_dict())
    assert diagnostic.fingerprint(first.state_dict()) == diagnostic.fingerprint(expected.state_dict())
    assert before != after_draw


def test_separation_balance_determinism_labels_and_no_td3_updates(learner, monkeypatch):
    calls = []
    original = diagnostic.critic_step

    def forbidden(*args, **kwargs):
        raise AssertionError("TD3 target/actor/joint updates must never run")

    def observe(critics, optimizer, train):
        calls.append(deepcopy(train))
        return original(critics, optimizer, train)

    monkeypatch.setattr(diagnostic.TD3, "update", forbidden)
    monkeypatch.setattr(diagnostic.TD3, "_target_values", forbidden)
    monkeypatch.setattr(diagnostic, "critic_step", observe)
    before, rng_before = diagnostic.learner_fingerprint(learner), diagnostic.global_rng_fingerprint()
    result = fit(learner)
    assert diagnostic.learner_fingerprint(learner) == before
    assert diagnostic.global_rng_fingerprint() == rng_before
    assert len(calls) == 12
    for i, call in enumerate(calls):
        assert [r["scenario"] for r in call["rows"]] == list(diagnostic.SCENARIOS)
        assert {r["row_index"] for r in call["rows"]} == {74 if i < 6 else 149}
    for fold in result["folds"]:
        for name, arm in fold["arms"].items():
            assert arm["holdout_note"] == diagnostic.HOLDOUT_NOTES[name]
            assert arm["training_draws_by_scenario"] == dict.fromkeys(diagnostic.SCENARIOS, 3)
            assert [c["update"] for c in arm["checkpoints"]] == [0, 1, 3]
            for checkpoint in arm["checkpoints"]:
                for split in ("train", "holdout"):
                    metric = checkpoint[split]
                    assert metric["sample_count"] == 5
                    assert metric["training_draws_by_scenario"] == dict.fromkeys(diagnostic.SCENARIOS, checkpoint["update"])
                    assert metric["diagnostic_training_rows"] == fold["training_rows"]
            for head in ("q1", "q2"):
                assert arm["descriptive_train_max_abs_error_le_0_01"][head] == (
                    arm["checkpoints"][-1]["train"]["aggregate"][head]["max_abs_error"] <= 0.01)
    assert "previously saw both folds" in diagnostic.HOLDOUT_NOTES["original_critics"]
    assert "no out-of-sample generalization claim" in diagnostic.HOLDOUT_NOTES["fresh_critics"]
    for name in diagnostic.ARMS:
        assert result["folds"][0]["arms"][name]["initialization_sha256"] == result["folds"][1]["arms"][name]["initialization_sha256"]
    assert result == fit(learner)


def test_holdout_targets_and_logging_do_not_affect_training(learner):
    first = fit(learner)
    changed = learner.state_dict()
    for group in changed["replay"].values():
        group["rewards"][149] = 1000
        group["next_observations"].fill_(10000)
        group["rewards"][:74] = -10000
    second = fit(learner, changed, logs=(0, 3))
    for name in diagnostic.ARMS:
        a, b = [r["folds"][0]["arms"][name]["checkpoints"][-1] for r in (first, second)]
        assert a["train"] == b["train"]
        assert a["holdout"] != b["holdout"]


def test_runner_json_only_preserves_every_input_and_rng(sandbox, learner):
    before = diagnostic.learner_fingerprint(learner)
    rng_before = diagnostic.global_rng_fingerprint()
    result = run(sandbox["output"])
    assert result["status"] == "completed"
    assert sandbox["calls"] == ["load"]
    assert diagnostic.learner_fingerprint(learner) == before
    assert diagnostic.global_rng_fingerprint() == rng_before
    assert all(result["immutability"][key] is True for key in
               ("input_source_runtime_unchanged", "original_learner_unchanged",
                "original_model_state_unchanged", "global_rng_unchanged"))
    output = sandbox["output"]
    assert {p.name for p in output.iterdir()} == {"runner.lock", "settings.json", "status.json", "metrics.json", "completion.json"}
    assert read(output / "completion.json") == result == read(output / "status.json")
    settings = read(output / "settings.json")
    assert settings["scenario_contribution"] == dict.fromkeys(diagnostic.SCENARIOS, 0.2)
    assert settings["fresh_seed"] == 8100 and settings["learning_rate"] == 0.0003
    assert settings["learned_weights_export"] is False
    assert settings["actor_updates"] == settings["target_updates"] == 0
    assert all(not settings[key] for key in ("noise", "bootstrap", "reward_normalization", "reward_clipping", "augmentation"))
    assert settings["identity_before"] == result["identity_after"]
    assert result["pid"] > 0 and result["started_at"] <= result["ended_at"]
    assert len(result["progress"]) == 4


@pytest.mark.parametrize("where", ["output", "goal", "repository"])
def test_early_stop_preserved_before_loading_or_hashing(sandbox, where, monkeypatch):
    folder = sandbox[where]
    folder.mkdir(parents=True, exist_ok=True)
    stop = folder / "STOP"
    stop.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "capture_identity", lambda _: pytest.fail("No early hashing"))
    result = run(sandbox["output"])
    assert result["status"] == "stopped" and result["output_created"] is False
    assert sandbox["calls"] == [] and stop.read_text() == "keep"
    assert not (sandbox["output"] / "completion.json").exists()


@pytest.mark.parametrize("where", ["output", "goal", "repository"])
def test_loop_stop_writes_honest_partial_without_completion(sandbox, where, monkeypatch):
    original = diagnostic.critic_step
    count = []

    def stop_after_step(*args):
        loss = original(*args)
        count.append(1)
        (sandbox[where] / "STOP").touch()
        return loss

    monkeypatch.setattr(diagnostic, "critic_step", stop_after_step)
    result = run(sandbox["output"])
    assert result["status"] == "stopped" and len(count) == 1
    assert result["progress"][0]["updates_completed"] == 1
    assert result["progress"][0]["training_draws_by_scenario"] == dict.fromkeys(diagnostic.SCENARIOS, 1)
    assert (sandbox[where] / "STOP").exists()
    assert not (sandbox["output"] / "completion.json").exists()
    assert read(sandbox["output"] / "status.json")["status"] == "stopped"
    assert result["immutability"]["original_learner_unchanged"] is True


def test_stop_arriving_during_loading_prevents_fitting(sandbox, monkeypatch):
    load = diagnostic.load_base

    def stop_after_load():
        result = load()
        (sandbox["goal"] / "STOP").touch()
        return result

    monkeypatch.setattr(diagnostic, "load_base", stop_after_load)
    monkeypatch.setattr(diagnostic, "critic_step", lambda *args: pytest.fail("No fitting after STOP"))
    result = run(sandbox["output"])
    assert result["status"] == "stopped" and result["progress"] == []


def test_late_stop_after_status_write_prevents_completion(sandbox, monkeypatch):
    writer = diagnostic.json_save

    def stop_on_final(path, value):
        writer(path, value)
        if path.name == "status.json" and value.get("status") == "completed":
            (sandbox["goal"] / "STOP").touch()

    monkeypatch.setattr(diagnostic, "json_save", stop_on_final)
    assert run(sandbox["output"])["status"] == "stopped"
    assert not (sandbox["output"] / "completion.json").exists()
    assert read(sandbox["output"] / "status.json")["status"] == "stopped"


@pytest.mark.parametrize("kind", ["empty_directory", "occupied_directory", "file"])
def test_refuses_all_existing_outputs_without_writes(sandbox, kind):
    output = sandbox["output"]
    if kind == "file":
        output.write_text("keep", encoding="utf-8")
    else:
        output.mkdir()
        if kind == "occupied_directory":
            (output / "old.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(output)
    assert sandbox["calls"] == []
    if kind == "file":
        assert output.read_text() == "keep"
    else:
        assert {p.name for p in output.iterdir()} == ({"old.json"} if kind == "occupied_directory" else set())


def test_existing_helper_enforces_exclusive_lock(tmp_path):
    with diagnostic.exclusive_run(tmp_path):
        with pytest.raises(OSError):
            with diagnostic.exclusive_run(tmp_path):
                pytest.fail("Second lock admitted")


@pytest.mark.parametrize("changed", ["source", "model"])
def test_changed_source_or_input_cannot_complete(sandbox, changed, monkeypatch):
    original = diagnostic.critic_step

    def mutate(*args):
        loss = original(*args)
        sandbox[changed].write_text("changed synthetic input", encoding="utf-8")
        return loss

    monkeypatch.setattr(diagnostic, "critic_step", mutate)
    with pytest.raises(ValueError, match="identity changed"):
        run(sandbox["output"])
    assert read(sandbox["output"] / "status.json")["status"] == "failed"
    assert not (sandbox["output"] / "completion.json").exists()


@pytest.mark.parametrize("bad", ["prediction", "loss", "gradient", "parameter", "adam"])
def test_finite_training_guards(learner, bad, monkeypatch):
    train, _ = diagnostic.fold_data(diagnostic.extract_terminals(learner.state_dict()), 0)
    critics, optimizer = diagnostic.critic_copy(learner.critics)
    if bad == "prediction":
        with torch.no_grad():
            critics[0][-1].bias.fill_(float("nan"))
    elif bad == "loss":
        train["targets"].fill_(1e30)
    elif bad == "gradient":
        next(critics.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
    else:
        step = optimizer.step

        def poison():
            step()
            with torch.no_grad():
                parameter = next(critics.parameters())
                if bad == "parameter":
                    parameter.fill_(float("inf"))
                else:
                    optimizer.state[parameter]["exp_avg"].fill_(float("inf"))

        monkeypatch.setattr(optimizer, "step", poison)
    with pytest.raises(ValueError, match="[Nn]onfinite"):
        diagnostic.critic_step(critics, optimizer, train)


def test_failed_fit_preserves_rng_and_no_completion(sandbox, monkeypatch):
    before = diagnostic.global_rng_fingerprint()

    def failure(*args):
        random.random()
        np.random.random()
        torch.rand(1)
        raise ValueError("injected nonfinite failure")

    monkeypatch.setattr(diagnostic, "critic_step", failure)
    with pytest.raises(ValueError, match="nonfinite"):
        run(sandbox["output"])
    assert diagnostic.global_rng_fingerprint() == before
    assert read(sandbox["output"] / "status.json")["status"] == "failed"
    assert not (sandbox["output"] / "completion.json").exists()


def test_json_guard_leaves_existing_file_untouched(tmp_path):
    path = tmp_path / "metric.json"
    diagnostic.json_save(path, {"finite": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        diagnostic.json_save(path, {"bad": float("nan")})
    assert path.read_bytes() == before and not path.with_suffix(".json.tmp").exists()


def test_real_provenance_helper_only_with_synthetic_input(tmp_path, monkeypatch):
    base = tmp_path / "base"
    (base / "train_round1").mkdir(parents=True)
    model = base / "train_round1/model_final.pt"
    model.write_bytes(b"not a model; synthetic hash input only")
    (base / "completion.json").write_text('{"status":"completed"}', encoding="utf-8")
    monkeypatch.setattr(diagnostic, "BASE", base)
    monkeypatch.setattr(diagnostic, "BASE_HASH", diagnostic.file_hash(model))
    monkeypatch.setattr(diagnostic.common, "pins", lambda _: {"synthetic_source_pins": "test"})
    before = diagnostic.capture_identity(lambda: None)
    assert before == diagnostic.capture_identity(lambda: None)
    assert before["runtime"]["torch_threads"] == before["runtime"]["torch_interop_threads"] == 1
    assert set(before["runtime"]["thread_environment"].values()) == {"1"}
    assert before["runtime_sha256"] == diagnostic.fingerprint(before["runtime"])
    assert before["input_sha256"][str(model)] == diagnostic.file_hash(model)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        diagnostic.capture_identity(lambda: None)


def test_production_cli_is_fixed_and_has_no_training_budget_switch(tmp_path, monkeypatch, capsys):
    assert diagnostic.UPDATES == 1000 and diagnostic.LOG_UPDATES == (0, 50, 250, 1000)
    assert diagnostic.FRESH_SEED == 8100 and diagnostic.LEARNING_RATE == 0.0003
    calls = []

    def inspect(output, **kwargs):
        calls.append((output, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr(diagnostic, "run_diagnostic", inspect)
    output = tmp_path / "new"
    assert diagnostic.main(["--output", str(output)]) == 0
    assert calls[0][0] == output and set(calls[0][1]) == {"command"}
    assert calls[0][1]["command"][-2:] == ["--output", str(output)]
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(["--output", str(output), "--updates", "1"])
    assert exc.value.code == 2
    assert len(calls) == 1
    capsys.readouterr()
