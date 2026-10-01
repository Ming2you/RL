"""Synthetic CPU tests only: production loading, simulation and export prohibited."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import terminal_quota as diagnostic

torch = diagnostic.torch
terminal = diagnostic.terminal
critic = diagnostic.critic


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Production load, snapshot access, simulator or export forbidden")

    for module, name in ((diagnostic, "load_base"), (terminal, "load_base"),
                         (terminal.common, "load_base"), (terminal.common, "pins"),
                         (terminal, "capture_identity"), (torch, "load"), (torch, "save"),
                         (critic, "load_frozen_model")):
        monkeypatch.setattr(module, name, forbidden)
    # These imports are already present via common; no physical runtime is loaded.
    import budget_runtime
    monkeypatch.setattr(budget_runtime, "boot", forbidden)
    terminal.single_thread()
    with terminal.preserve_rng():
        yield


@pytest.fixture
def learner():
    value = terminal.TD3(3, seed=17, hidden=8)
    rng = np.random.default_rng(23)
    for number, scenario in enumerate(diagnostic.SCENARIOS):
        observations = rng.uniform(0.1, 1, (150, 3))
        for index in range(150):
            done = index in (74, 149)
            value.add(observations[index], rng.uniform(-0.5, 0.5, 2),
                      -0.25 * (number + 1) - index / 200,
                      np.zeros(3) if done else observations[index + 1], done, scenario)
    # Populate both Adam histories and original target lag using synthetic replay.
    value.update()
    value.update()
    return value


@pytest.fixture
def sandbox(tmp_path, monkeypatch, learner):
    repo = tmp_path / "repository"
    goal = repo / "goal"
    goal.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "REPO", repo)
    monkeypatch.setattr(diagnostic, "GOAL_ROOT", goal)
    files = {}
    for name in ("source", "input", "projection"):
        files[name] = tmp_path / (name + ".txt")
        files[name].write_text("synthetic " + name, encoding="utf-8")
    model = dict(learner=learner.state_dict(), contract={"synthetic": True}, metadata=["untouched"])
    calls = []

    def load():
        calls.append("load")
        return model, learner

    def capture(stop):
        stop()
        calls.append("identity")
        return {name: diagnostic.file_hash(path) for name, path in files.items()}

    monkeypatch.setattr(diagnostic, "load_base", load)
    monkeypatch.setattr(diagnostic, "capture_identity", capture)
    return dict(repo=repo, goal=goal, output=tmp_path / "output", files=files, model=model, calls=calls)


def run(output):
    return diagnostic.run_diagnostic(output, updates=3, log_updates=(0, 1, 2, 3))


def fit(learner, *, logs=(0, 1, 2, 3), seed=6529, record=lambda _: None):
    return diagnostic.fit_seed(learner, learner.state_dict(), seed, lambda: None, record,
                               updates=3, log_updates=logs)


def test_balanced_pairing_incidental_terminals_and_replacement_stream():
    uniform_rng, replacement_rng = np.random.default_rng(6529), np.random.default_rng(106529)
    reference = np.random.default_rng(6529)
    repeated = (np.random.default_rng(6529), np.random.default_rng(106529))
    seen = set()
    incidental = False
    for _ in range(100):
        indices = diagnostic.paired_indices(uniform_rng, replacement_rng)
        again = diagnostic.paired_indices(*repeated)
        expected = np.concatenate([reference.integers(150, size=8) + 150 * i for i in range(5)])
        np.testing.assert_array_equal(indices["uniform"].numpy(), expected)
        for name in diagnostic.ARMS:
            assert torch.equal(indices[name], again[name])
            assert torch.bincount(indices[name] // 150).tolist() == [8] * 5
        uniform, quota = (indices[name].reshape(5, 8) % 150 for name in diagnostic.ARMS)
        assert torch.equal(uniform[:, 1:], quota[:, 1:])
        assert set(quota[:, 0].tolist()) <= {74, 149}
        seen.update(quota[:, 0].tolist())
        terminals = (quota == 74) | (quota == 149)
        assert torch.all(terminals.sum(dim=1) >= 1)
        incidental |= bool(torch.any(terminals.sum(dim=1) > 1))
    assert incidental and seen == {74, 149}
    a = diagnostic.paired_indices(np.random.default_rng(99), np.random.default_rng(1))
    b = diagnostic.paired_indices(np.random.default_rng(99), np.random.default_rng(2))
    assert torch.equal(a["uniform"], b["uniform"])
    assert not torch.equal(a["terminal_quota"], b["terminal_quota"])


def test_full_independent_copies_preserve_adam_targets_and_freeze_online_actor(learner):
    original = terminal.learner_fingerprint(learner)
    arms = diagnostic.clone_arms(learner)
    pointers = [{p.data_ptr() for m in (value.actor, value.actor_target, value.critics, value.critic_targets)
                 for p in m.parameters()} for value in (learner, *arms.values())]
    assert all(pointers[i].isdisjoint(pointers[j]) for i in range(3) for j in range(i))
    for value in arms.values():
        for name in ("critics", "critic_targets", "critic_optimizer", "actor_optimizer"):
            assert diagnostic.fingerprint(getattr(value, name).state_dict()) == diagnostic.fingerprint(
                getattr(learner, name).state_dict())
        for module in (value.actor, value.actor_target):
            assert not module.training
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
            assert diagnostic.fingerprint(module.state_dict()) == diagnostic.fingerprint(learner.actor.state_dict())
        params = {id(p) for p in value.critic_optimizer.param_groups[0]["params"]}
        assert params == {id(p) for p in value.critics.parameters()}
    assert terminal.learner_fingerprint(learner) == original


def test_gamma_one_and_true_terminal_mask(learner):
    value = diagnostic.clone_arms(learner)["uniform"]
    with torch.no_grad():
        for q, bias in zip(value.critic_targets, (2., 3.)):
            for p in q.parameters():
                p.zero_()
            q[-1].bias.fill_(bias)
    batch = dict(next_observations=torch.zeros(4, 3), rewards=torch.tensor([-1., -4., -5., 2.]),
                 terminated=torch.tensor([True, False, True, False]))
    actual = critic.target_values(value, batch, torch.zeros(4, 2))
    torch.testing.assert_close(actual, torch.tensor([-1., -2., -5., 4.]), rtol=0, atol=0)
    assert critic.TD3.gamma == 1.


def test_only_critics_update_at_exact_target_cadence(learner, monkeypatch):
    value = diagnostic.clone_arms(learner)["uniform"]
    before = diagnostic.frozen_fingerprints(value)
    monkeypatch.setattr(value.actor_optimizer, "step", lambda *a, **k: pytest.fail("Actor optimizer used"))
    data = critic.replay_tensors(learner.state_dict())
    batch = {k: v[:40] for k, v in data.items()}
    old = [p.clone() for p in value.critic_targets.parameters()]
    original_online = diagnostic.fingerprint(value.critics.state_dict())
    diagnostic.checked_step(value, batch, torch.zeros(40, 2), 1, lambda: None)
    assert all(torch.equal(a, b) for a, b in zip(old, value.critic_targets.parameters()))
    assert diagnostic.fingerprint(value.critics.state_dict()) != original_online
    diagnostic.checked_step(value, batch, torch.zeros(40, 2), 2, lambda: None)
    for prev, online, target in zip(old, value.critics.parameters(), value.critic_targets.parameters()):
        torch.testing.assert_close(target, prev.mul(0.995).add(online, alpha=0.005), rtol=0, atol=0)
    assert diagnostic.frozen_fingerprints(value) == before
    assert all(entry["step"] == 4 for entry in value.critic_optimizer.state.values())


def test_exact_terminal_metrics_and_original_diagnostic_slices(learner):
    terminals = terminal.extract_terminals(learner.state_dict())
    value = diagnostic.clone_arms(learner)["uniform"]
    with torch.no_grad():
        for q, bias in zip(value.critics, (2., -1.)):
            for p in q.parameters():
                p.zero_()
            q[-1].bias.fill_(bias)
    metrics = diagnostic.terminal_metrics(value, terminals, 3)
    assert len(metrics["rows"]) == 10
    assert {row["round"] for row in metrics["rows"]} == {0, 1}
    for name, prediction in (("q1", 2.), ("q2", -1.), ("min_q", -1.)):
        for row in metrics["rows"]:
            assert row["predictions"][name] == prediction
            assert row["errors"][name] == prediction - row["target"]
            assert row["abs_errors"][name] == abs(prediction - row["target"])
        assert metrics["aggregate"][name]["mae"] == np.mean([r["abs_errors"][name] for r in metrics["rows"]])
        for scenario, entry in metrics["per_scenario"].items():
            expected = np.mean([r["abs_errors"][name] for r in metrics["rows"] if r["scenario"] == scenario])
            assert entry[name]["mae"] == expected
    result = fit(learner)
    original = result["arms"]["uniform"]["checkpoints"][0]["diagnostics"]
    assert original["total"]["transitions"] == 750
    for row in original["per_scenario"].values():
        assert row["transitions"] == 150 and row["true_terminals"] == 2
        assert len(row["time_slices"]) == 3 and len(row["episodes"]) == 2
        assert "total_td_loss" in row and "behavior_return_gap_rmse" in row["critics"]["minimum_q"]


def test_training_noise_identical_eval_separate_counts_and_no_joint_update(learner, monkeypatch):
    steps, evaluations = [], []
    original_step, original_metrics = critic.critic_step, critic.diagnostics

    def observe_step(value, batch, noise, tau, update):
        steps.append((noise, noise.clone(), batch["terminated"].clone()))
        return original_step(value, batch, noise, tau, update)

    def observe_metrics(value, data, noise):
        evaluations.append((noise, noise.clone()))
        return original_metrics(value, data, noise)

    monkeypatch.setattr(critic, "critic_step", observe_step)
    monkeypatch.setattr(critic, "diagnostics", observe_metrics)
    monkeypatch.setattr(terminal.TD3, "update", lambda *a: pytest.fail("Joint TD3 update forbidden"))
    original, rng = terminal.learner_fingerprint(learner), terminal.global_rng_fingerprint()
    result = fit(learner)
    assert original == terminal.learner_fingerprint(learner)
    assert rng == terminal.global_rng_fingerprint()
    for i in range(0, len(steps), 2):
        assert steps[i][0] is steps[i + 1][0]
        assert torch.equal(steps[i][1], steps[i + 1][1])
    assert all(pair[0] is evaluations[0][0] and torch.equal(pair[1], evaluations[0][1]) for pair in evaluations)
    for index, name in enumerate(diagnostic.ARMS):
        arm = result["arms"][name]
        assert arm["online_critic_updates"] == 3 and arm["target_critic_updates"] == 1
        assert arm["sample_counts"] == dict.fromkeys(diagnostic.SCENARIOS, 24)
        expected = torch.stack([row[2].reshape(5, 8).sum(dim=1) for row in steps[index::2]]).sum(dim=0)
        assert arm["terminal_counts"] == dict(zip(diagnostic.SCENARIOS, expected.tolist()))
        assert arm["frozen_initial"] == arm["frozen_final"]
    assert result["arms"]["uniform"]["noise_stream_sha256"] == result["arms"]["terminal_quota"]["noise_stream_sha256"]
    second = fit(learner, logs=(0, 3))
    for name in diagnostic.ARMS:
        first_arm, second_arm = result["arms"][name], second["arms"][name]
        for key in ("final_critic_sha256", "final_critic_target_sha256", "final_critic_optimizer_sha256",
                    "index_stream_sha256", "noise_stream_sha256"):
            assert first_arm[key] == second_arm[key]
        assert first_arm["checkpoints"][-1] == second_arm["checkpoints"][-1]


@pytest.mark.parametrize("bad", ["continuity", "terminal", "replay_nan", "gamma"])
def test_replay_and_spec_validation(learner, bad):
    state = learner.state_dict()
    group = state["replay"][diagnostic.SCENARIOS[0]]
    if bad == "continuity":
        group["next_observations"][0, 0] += 1
    elif bad == "terminal":
        group["terminated"][74] = False
    elif bad == "replay_nan":
        group["actions"][0, 0] = float("nan")
    else:
        state["spec"]["gamma"] = .9
    with pytest.raises(ValueError):
        diagnostic.fit_seed(learner, state, 6529, lambda: None, lambda _: None, updates=1, log_updates=(0, 1))


def test_runner_json_only_fixed_seeds_metadata_and_immutability(sandbox, learner):
    original = terminal.learner_fingerprint(learner)
    model = diagnostic.fingerprint(sandbox["model"])
    rng = terminal.global_rng_fingerprint()
    result = run(sandbox["output"])
    assert result["status"] == "completed"
    assert sandbox["calls"] == ["identity", "load", "identity", "identity"]
    assert terminal.learner_fingerprint(learner) == original
    assert diagnostic.fingerprint(sandbox["model"]) == model
    assert terminal.global_rng_fingerprint() == rng
    assert result["original_model_before"] == result["original_model_after"] == model
    assert result["original_learner_before"] == result["original_learner_after"] == original
    assert result["global_rng_before"] == result["global_rng_after"] == rng
    assert result["pid"] > 0 and result["elapsed_seconds"] > 0
    output = sandbox["output"]
    assert {p.name for p in output.iterdir()} == {"runner.lock", "settings.json", "metrics.json", "status.json", "completion.json"}
    assert read(output / "completion.json") == result
    settings = read(output / "settings.json")
    assert not settings["learned_weights_export"] and settings["actor_optimizer_updates"] == 0
    assert settings["identity_before"] == result["identity_after"]
    assert settings["gamma"] == 1 and settings["critic_target_tau"] == .005
    seeds = read(output / "metrics.json")["seeds"]
    assert [seed["seed"] for seed in seeds] == [6529, 6530]
    assert seeds[0]["initial_state_sha256"] == seeds[1]["initial_state_sha256"]
    assert seeds[0]["evaluation_noise_sha256"] == seeds[1]["evaluation_noise_sha256"]
    for seed in seeds:
        for name in diagnostic.ARMS:
            assert [c["update"] for c in seed["arms"][name]["checkpoints"]] == [0, 1, 2, 3]


@pytest.mark.parametrize("where", ["repo", "goal", "output"])
def test_stop_before_lock_load_or_write(sandbox, where):
    sandbox[where].mkdir(parents=True, exist_ok=True)
    marker = sandbox[where] / "STOP"
    marker.write_text("preserve", encoding="utf-8")
    result = run(sandbox["output"])
    assert result["status"] == "stopped" and sandbox["calls"] == []
    assert marker.read_text() == "preserve"
    assert not (sandbox["output"] / "runner.lock").exists()


@pytest.mark.parametrize("race", ["stop", "occupied"])
def test_lock_races_rechecked_before_load_or_write(sandbox, monkeypatch, race):
    lock = diagnostic.exclusive_run

    @contextmanager
    def raced(output):
        with lock(output):
            (output / ("STOP" if race == "stop" else "old.json")).write_text("keep", encoding="utf-8")
            yield

    monkeypatch.setattr(diagnostic, "exclusive_run", raced)
    if race == "stop":
        assert run(sandbox["output"])["status"] == "stopped"
    else:
        with pytest.raises(FileExistsError):
            run(sandbox["output"])
    assert sandbox["calls"] == []
    assert not (sandbox["output"] / "settings.json").exists()


@pytest.mark.parametrize("where", ["repo", "goal", "output"])
def test_stop_between_arms_prevents_later_steps_and_all_writes(sandbox, monkeypatch, where):
    step = critic.critic_step
    calls = []
    written_before_stop = {}

    def stopping(*args):
        result = step(*args)
        calls.append(1)
        written_before_stop.update({p.name: p.read_bytes() for p in sandbox["output"].glob("*.json")})
        (sandbox[where] / "STOP").touch()
        return result

    monkeypatch.setattr(critic, "critic_step", stopping)
    assert run(sandbox["output"])["status"] == "stopped"
    assert len(calls) == 1
    assert {p.name: p.read_bytes() for p in sandbox["output"].glob("*.json")} == written_before_stop
    assert not (sandbox["output"] / "completion.json").exists()


@pytest.mark.parametrize("when", ["load", "final_status", "final_identity"])
def test_stop_during_load_and_before_completion(sandbox, monkeypatch, when):
    if when == "load":
        load = diagnostic.load_base

        def changed():
            value = load()
            (sandbox["goal"] / "STOP").touch()
            return value

        monkeypatch.setattr(diagnostic, "load_base", changed)
    elif when == "final_status":
        writer = diagnostic.json_save

        def changed(path, value):
            writer(path, value)
            if path.name == "status.json" and value.get("status") == "verifying":
                (sandbox["goal"] / "STOP").touch()

        monkeypatch.setattr(diagnostic, "json_save", changed)
    else:
        capture = diagnostic.capture_identity
        calls = []

        def changed(stop):
            value = capture(stop)
            calls.append(1)
            if len(calls) == 3:
                (sandbox["goal"] / "STOP").touch()
            return value

        monkeypatch.setattr(diagnostic, "capture_identity", changed)
    assert run(sandbox["output"])["status"] == "stopped"
    assert not (sandbox["output"] / "completion.json").exists()
    if when == "load":
        assert not (sandbox["output"] / "settings.json").exists()


@pytest.mark.parametrize("entry", ["old.json", "completion.json", "runner.lock", "file"])
def test_nonempty_output_never_overwritten(sandbox, entry):
    output = sandbox["output"]
    if entry == "file":
        output.write_text("keep", encoding="utf-8")
        path = output
    else:
        output.mkdir()
        path = output / entry
        path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(output)
    assert sandbox["calls"] == [] and path.read_text() == "keep"


def test_empty_output_allowed_and_existing_lock_is_exclusive(sandbox, tmp_path):
    sandbox["output"].mkdir()
    assert run(sandbox["output"])["status"] == "completed"
    with diagnostic.exclusive_run(tmp_path):
        with pytest.raises(OSError):
            with diagnostic.exclusive_run(tmp_path):
                pytest.fail("Second lock admitted")


@pytest.mark.parametrize("which", ["source", "input", "projection"])
@pytest.mark.parametrize("when", ["load", "step"])
def test_source_input_completion_changes_rejected(sandbox, monkeypatch, which, when):
    module, name = (diagnostic, "load_base") if when == "load" else (critic, "critic_step")
    original = getattr(module, name)

    def changed(*args):
        result = original(*args)
        sandbox["files"][which].write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(module, name, changed)
    with pytest.raises(ValueError, match="identity changed"):
        run(sandbox["output"])
    assert read(sandbox["output"] / "status.json")["status"] == "failed"
    assert not (sandbox["output"] / "completion.json").exists()


@pytest.mark.parametrize("which", ["model", "learner"])
def test_final_verification_rechecks_original_state(sandbox, learner, monkeypatch, which):
    capture = diagnostic.capture_identity
    calls = []

    def changed(stop):
        result = capture(stop)
        calls.append(1)
        if len(calls) == 3:
            if which == "model":
                sandbox["model"]["metadata"].append("late change")
            else:
                with torch.no_grad():
                    next(learner.critics.parameters()).add_(1)
        return result

    monkeypatch.setattr(diagnostic, "capture_identity", changed)
    with pytest.raises(ValueError, match="Original learner/model changed"):
        run(sandbox["output"])
    assert not (sandbox["output"] / "completion.json").exists()


def test_final_identity_rng_work_is_also_preserved(sandbox, monkeypatch):
    capture = diagnostic.capture_identity
    before = terminal.global_rng_fingerprint()

    def consuming(stop):
        result = capture(stop)
        random.random()
        np.random.random()
        torch.rand(1)
        return result

    monkeypatch.setattr(diagnostic, "capture_identity", consuming)
    result = run(sandbox["output"])
    assert result["status"] == "completed"
    assert terminal.global_rng_fingerprint() == before == result["global_rng_after"]


@pytest.mark.parametrize("bad", ["parameter", "gradient", "post_parameter", "adam", "target", "loss"])
def test_nonfinite_checks_prevent_completion_and_restore_rng(sandbox, monkeypatch, bad):
    clone = diagnostic.clone_arms
    rng = terminal.global_rng_fingerprint()
    before_steps = []

    def poison(original):
        arms = clone(original)
        value = arms["uniform"]
        parameter = next(value.critics.parameters())
        if bad == "parameter":
            with torch.no_grad():
                parameter.fill_(float("nan"))
        elif bad == "gradient":
            parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
            value.critic_optimizer.register_step_pre_hook(lambda *args: before_steps.append(1))
            value.critic_optimizer.register_step_post_hook(lambda *args: pytest.fail("Bad gradient reached Adam"))
        elif bad == "loss":
            for q in value.critics:
                with torch.no_grad():
                    q[-1].bias.fill_(1e30)
        else:
            def after_step(*args):
                with torch.no_grad():
                    if bad == "post_parameter":
                        parameter.fill_(float("inf"))
                    elif bad == "target":
                        next(value.critic_targets.parameters()).fill_(float("inf"))
                    else:
                        value.critic_optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
                random.random()
                np.random.random()
                torch.rand(1)
            value.critic_optimizer.register_step_post_hook(after_step)
        return arms

    monkeypatch.setattr(diagnostic, "clone_arms", poison)
    with pytest.raises(ValueError, match="[Nn]onfinite|Out of range float"):
        run(sandbox["output"])
    assert terminal.global_rng_fingerprint() == rng
    assert not (sandbox["output"] / "completion.json").exists()
    assert not list(sandbox["output"].glob("*.pt"))
    assert read(sandbox["output"] / "status.json")["status"] == "failed"
    if bad == "gradient":
        assert before_steps == [1]


@pytest.mark.parametrize("which", ["actor", "actor_target", "actor_optimizer", "original_model", "original_learner"])
def test_mutation_guards(sandbox, learner, monkeypatch, which):
    step = critic.critic_step

    def changed(value, *args):
        result = step(value, *args)
        with torch.no_grad():
            if which == "actor_optimizer":
                next(iter(value.actor_optimizer.state.values()))["exp_avg"].add_(1)
            elif which == "original_model":
                sandbox["model"]["metadata"].append("changed")
            elif which == "original_learner":
                next(learner.actor.parameters()).add_(1)
            else:
                next(getattr(value, which).parameters()).add_(1)
        return result

    monkeypatch.setattr(critic, "critic_step", changed)
    with pytest.raises(ValueError, match="changed"):
        run(sandbox["output"])
    assert not (sandbox["output"] / "completion.json").exists()


@pytest.fixture
def provenance(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    base = repo / "base"
    here = repo / "work/audit"
    common = repo / "work/recovery/common.py"
    monkeypatch.setattr(diagnostic, "REPO", repo)
    monkeypatch.setattr(diagnostic, "BASE", base)
    monkeypatch.setattr(diagnostic, "HERE", here)
    monkeypatch.setattr(terminal.common, "__file__", str(common))
    completion = tmp_path / "projection.json"
    monkeypatch.setattr(diagnostic, "PROJECTION", completion)
    for path in diagnostic.input_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic bytes only")
    source_paths = [here / "terminal_quota_brief.md", here / "projection_audit.py", common]
    source_paths += [repo / "work/sdmpc_rl_multi_20260929" / name for name in
                     ("td3.py", "run_budget.py", "budget_runtime.py", "freeze_runtime.py")]
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source", encoding="utf-8")
    digest = diagnostic.file_hash(base / "train_round1/model_final.pt")
    monkeypatch.setattr(diagnostic, "BASE_HASH", digest)
    identity = dict(source_sha256={}, base_source_pins={"synthetic": "pins"}, runtime={"versions": {"test": "1"}})
    monkeypatch.setattr(terminal, "capture_identity", lambda stop: deepcopy(identity))
    payload = dict(status="completed", settings=dict(base_sha256=digest,
                   input_sha256={str(p.relative_to(repo)): diagnostic.file_hash(p) for p in diagnostic.input_paths()},
                   source_sha256={str(p.relative_to(repo)): diagnostic.file_hash(p) for p in (here / "projection_audit.py", common)},
                   contract=dict(source_pins=identity["base_source_pins"], runtime_versions=identity["runtime"]["versions"])))
    completion.write_text(json.dumps(payload), encoding="utf-8")
    return dict(completion=completion, payload=payload, sources=source_paths, base=base)


def test_real_manifest_verifier_with_73_synthetic_files(provenance):
    before = diagnostic.capture_identity(lambda: None)
    assert len(before["projection_input_sha256"]) == 73
    assert before == diagnostic.capture_identity(lambda: None)
    assert before["projection_completion"]["sha256"] == diagnostic.file_hash(provenance["completion"])
    assert str(Path(critic.__file__)) in before["source_sha256"]
    assert str(Path(terminal.__file__)) in before["source_sha256"]
    assert str(Path(diagnostic.__file__)) in before["source_sha256"]


@pytest.mark.parametrize("bad", ["status", "base", "missing", "extra", "substituted_path", "input",
                                "projection_source", "source_manifest", "contract", "completion_race"])
def test_projection_authentication_fails_closed(provenance, monkeypatch, bad):
    payload = provenance["payload"]
    settings = payload["settings"]
    if bad == "status":
        payload["status"] = "failed"
    elif bad == "base":
        settings["base_sha256"] = "wrong"
    elif bad == "missing":
        settings["input_sha256"].pop(next(iter(settings["input_sha256"])))
    elif bad in ("extra", "substituted_path"):
        if bad == "substituted_path":
            settings["input_sha256"].pop(next(iter(settings["input_sha256"])))
        settings["input_sha256"]["unexpected"] = "wrong"
    elif bad == "input":
        diagnostic.input_paths()[5].write_bytes(b"changed")
    elif bad == "projection_source":
        provenance["sources"][1].write_text("changed", encoding="utf-8")
    elif bad == "source_manifest":
        settings["source_sha256"] = {}
    elif bad == "contract":
        settings["contract"]["runtime_versions"] = {}
    else:
        original = diagnostic.file_hash

        def changed(path):
            if path == provenance["completion"]:
                path.write_text("{}", encoding="utf-8")
            return original(path)

        monkeypatch.setattr(diagnostic, "file_hash", changed)
    provenance["completion"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        diagnostic.capture_identity(lambda: None)


def test_json_nan_rejected_before_atomic_writer(tmp_path):
    path = tmp_path / "output.json"
    diagnostic.json_save(path, dict(value=1))
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        diagnostic.json_save(path, dict(value=float("nan")))
    assert path.read_bytes() == previous
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("uniform,quota,passes", [([10., 12.], [4., 5.], True),
                                               ([10., 12.], [5., 6.], False),
                                               ([2., 20.], [3., 4.], False),
                                               ([5., 6.], [4., 5.], False)])
def test_predeclared_criterion(uniform, quota, passes):
    seeds = [dict(arms={name: dict(checkpoints=[dict(terminals=dict(aggregate=dict(min_q=dict(mae=values[i]))))])
                       for name, values in zip(diagnostic.ARMS, (uniform, quota))}) for i in range(2)]
    result = diagnostic.interpretation(seeds)
    assert result["supports_sampling_hypothesis"] is passes
    assert result["initial_threshold"] == 5.701837813854217


def test_fixed_cli_only_output(tmp_path, monkeypatch, capsys):
    assert diagnostic.SEEDS == (6529, 6530)
    assert diagnostic.UPDATES == 3750 and diagnostic.LOG_UPDATES == (0, 375, 750, 1500, 3750)
    calls = []
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda output, **kwargs:
                        calls.append((output, kwargs)) or dict(status="completed"))
    output = tmp_path / "output"
    assert diagnostic.main(["--output", str(output)]) == 0
    assert calls == [(output, dict(command=[sys.executable, str(Path(diagnostic.__file__).resolve()),
                                          "--output", str(output)]))]
    for option in ("--updates", "--seed", "--seeds", "--log-updates", "--resume", "--tau", "--model", "--out"):
        with pytest.raises(SystemExit) as exc:
            diagnostic.main(["--output", str(output), option, "1"])
        assert exc.value.code == 2
    assert len(calls) == 1
    capsys.readouterr()
