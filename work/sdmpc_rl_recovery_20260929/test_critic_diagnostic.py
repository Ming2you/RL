"""Focused synthetic tests only: no production model, plant, or numerical run."""

from copy import deepcopy
import json

import critic_diagnostic as diagnostic
import numpy as np
import pytest
import torch


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.fixture(scope="module")
def state():
    learner = diagnostic.TD3(3, seed=23, hidden=8)
    rng = np.random.default_rng(17)
    for scenario in diagnostic.SCENARIOS:
        for episode in range(2):
            observations = rng.normal(size=(76, 3))
            for i in range(75):
                learner.add(observations[i], rng.uniform(-1, 1, 2), -float(episode + 1),
                            observations[i + 1], i == 74, scenario)
    learner.update()
    learner.update()
    return learner.state_dict()


def sample(state, seed=72):
    return diagnostic.draw_batch(diagnostic.replay_tensors(state), np.random.default_rng(seed),
                                 torch.Generator().manual_seed(seed))


def records(output):
    return [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]


def test_clones_preserve_critics_optimizer_and_replace_stale_target_actor(state):
    assert any(not torch.equal(value, state["actor_target"][key]) for key, value in state["actor"].items())
    arms = diagnostic.clone_arms(state)
    for learner in arms.values():
        equal(learner.critics.state_dict(), state["critics"])
        equal(learner.critic_targets.state_dict(), state["critic_targets"])
        equal(learner.critic_optimizer.state_dict(), state["critic_optimizer"])
        equal(learner.actor.state_dict(), state["actor"])
        equal(learner.actor_target.state_dict(), state["actor"])
        assert all(not p.requires_grad for p in learner.actor.parameters())
        assert all(not p.requires_grad for p in learner.actor_target.parameters())
    a, b = arms.values()
    assert {p.data_ptr() for p in a.critics.parameters()}.isdisjoint(p.data_ptr() for p in b.critics.parameters())
    a_moments = a.critic_optimizer.state_dict()["state"]
    b_moments = b.critic_optimizer.state_dict()["state"]
    assert all(a_moments[k]["exp_avg"].data_ptr() != b_moments[k]["exp_avg"].data_ptr() for k in a_moments)


def test_true_terminal_targets_and_nonterminal_bootstrap(state):
    learner = next(iter(diagnostic.clone_arms(state).values()))
    with torch.no_grad():
        for critic, bias in zip(learner.critic_targets, (7.0, 9.0)):
            for parameter in critic.parameters():
                parameter.zero_()
            critic[-1].bias.fill_(bias)
    _, batch, noise = sample(state)
    batch["terminated"][::2] = True
    batch["terminated"][1::2] = False
    targets = diagnostic.target_values(learner, batch, noise)
    assert torch.equal(targets[::2], batch["rewards"][::2])
    assert torch.equal(targets[1::2], batch["rewards"][1::2] + 7)


def test_exact_balance_and_reproducible_shared_draws(state):
    data = diagnostic.replay_tensors(state)
    rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(42)
    noise_a, noise_b = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    for _ in range(7):
        indices, batch, noise = diagnostic.draw_batch(data, rng_a, noise_a)
        equal((indices, batch, noise), diagnostic.draw_batch(data, rng_b, noise_b))
        assert torch.bincount(indices // 150, minlength=5).tolist() == [8] * 5
        assert len(indices) == 40 and noise.shape == (40, 2)
        assert float(noise.abs().max()) <= diagnostic.TD3.noise_clip


def test_smoothing_is_applied_to_target_actor_and_actions_are_clipped(state):
    learner = next(iter(diagnostic.clone_arms(state).values()))
    _, batch, _ = sample(state)
    with torch.no_grad():
        learner.actor_target[-2].bias.copy_(torch.tensor([2.0, -2.0]))
    noise = torch.tensor([[0.5, -0.5]]).repeat(40, 1)
    captured = []
    hook = learner.critic_targets[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].clone()))
    try:
        diagnostic.target_values(learner, batch, noise)
    finally:
        hook.remove()
    expected_actions = (learner.actor_target(batch["next_observations"]) + noise).clamp(-1, 1)
    equal(captured[0], torch.cat((batch["next_observations"], expected_actions), dim=1))
    assert torch.all(expected_actions[:, 0] == 1) and torch.all(expected_actions[:, 1] == -1)


def test_noise_and_diagnostic_remain_cpu_float32_with_changed_torch_default(state, tmp_path):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, noise = sample(state)
        assert noise.dtype == torch.float32 and noise.device.type == "cpu"
        result = diagnostic.run_diagnostic(state, tmp_path / "float32", updates=1)
        assert result["updates"] == 1
    finally:
        torch.set_default_dtype(previous)


def test_actor_optimizer_replay_and_input_unchanged_after_critic_steps(state):
    before = deepcopy(state)
    _, batch, noise = sample(state)
    batch_before, noise_before = deepcopy(batch), noise.clone()
    arms = diagnostic.clone_arms(state)
    for name, learner in arms.items():
        for update in range(1, 5):
            diagnostic.critic_step(learner, batch, noise, diagnostic.ARMS[name], update)
        equal(learner.actor.state_dict(), before["actor"])
        equal(learner.actor_target.state_dict(), before["actor"])
        equal(learner.actor_optimizer.state_dict(), before["actor_optimizer"])
        equal(learner.state_dict()["replay"], before["replay"])
        assert any(not torch.equal(value, before["critics"][key]) for key, value in learner.critics.state_dict().items())
    equal(state, before)
    equal(batch, batch_before)
    equal(noise, noise_before)


@pytest.mark.parametrize("tau", [0.005, 1.0])
def test_critic_target_schedule_and_exact_rate(state, tau):
    learner = next(iter(diagnostic.clone_arms(state).values()))
    _, batch, noise = sample(state)
    for update in range(1, 5):
        before = [p.clone() for p in learner.critic_targets.parameters()]
        diagnostic.critic_step(learner, batch, noise, tau, update)
        for old, online, target in zip(before, learner.critics.parameters(), learner.critic_targets.parameters()):
            expected = old.mul(1.0 - tau).add(online, alpha=tau) if update % 2 == 0 else old
            assert torch.equal(target, expected)


def test_behavior_returns_reset_at_both_true_terminals(state):
    data = diagnostic.replay_tensors(state)
    expected = torch.cat((torch.arange(75, 0, -1) * -1.0, torch.arange(75, 0, -1) * -2.0))
    assert torch.equal(data["behavior_returns"], expected.repeat(5))


@pytest.mark.parametrize("invalid", ["missing_terminal", "misplaced_terminal", "short_replay", "discontinuous"])
def test_rejects_incomplete_or_misordered_training_episodes(state, invalid):
    broken = deepcopy(state)
    group = broken["replay"][diagnostic.SCENARIOS[0]]
    if invalid == "missing_terminal":
        group["terminated"][149] = False
    elif invalid == "misplaced_terminal":
        group["terminated"][74] = False
        group["terminated"][73] = True
    elif invalid == "short_replay":
        group["rewards"] = group["rewards"][:-1]
    else:
        group["next_observations"][0] += 1
    with pytest.raises(ValueError):
        diagnostic.replay_tensors(broken)


def test_runner_passes_identical_samples_and_noise_and_logs_metrics(state, tmp_path, monkeypatch):
    original = diagnostic.critic_step
    calls = []

    def observe(learner, batch, noise, tau, update):
        calls.append((update, tau, deepcopy(batch), noise.clone(), id(batch), id(noise)))
        return original(learner, batch, noise, tau, update)

    monkeypatch.setattr(diagnostic, "critic_step", observe)
    output = tmp_path / "paired"
    result = diagnostic.run_diagnostic(state, output, updates=4, log_updates=(0, 2, 4))
    assert result["status"] == "completed" and result["target_updates"] == 2
    assert result["sample_counts"] == dict.fromkeys(diagnostic.SCENARIOS, 32)
    for a, b in zip(calls[::2], calls[1::2]):
        assert a[0] == b[0] and (a[1], b[1]) == (0.005, 1.0)
        equal(a[2:], b[2:])
    rows = records(output)
    assert rows[0]["runtime"]["torch_threads"] == rows[0]["runtime"]["torch_interop_threads"] == 1
    assert "not current-policy Q truth" in rows[0]["caveat"]
    checkpoints = [r for r in rows if r["kind"] == "checkpoint"]
    assert [r["updates"] for r in checkpoints] == [0, 2, 4]
    equal(*checkpoints[0]["arms"].values())
    for row in checkpoints:
        for arm in row["arms"].values():
            assert arm["total"]["total_td_loss"] >= 0
            for scenario in arm["per_scenario"].values():
                assert scenario["transitions"] == 150 and scenario["true_terminals"] == 2
                assert [s["transitions"] for s in scenario["time_slices"].values()] == [50] * 3
                assert scenario["time_slices"]["early_0_24"]["critics"]["q1"]["terminal_abs_error_mean"] is None
                assert scenario["time_slices"]["late_50_74"]["true_terminals"] == 2
                assert all(ep["true_terminals"] == 1 for ep in scenario["episodes"].values())
    assert {p.name for p in output.iterdir()} == {"diagnostics.jsonl", "summary.json"}


def test_stop_between_paired_batches_saves_partial_and_refuses_overwrite(state, tmp_path, monkeypatch):
    original = diagnostic.critic_step
    stop = tmp_path / "STOP"

    def stop_after_pair(learner, batch, noise, tau, update):
        loss = original(learner, batch, noise, tau, update)
        if update == 3 and tau == 1.0:
            stop.touch()
        return loss

    monkeypatch.setattr(diagnostic, "critic_step", stop_after_pair)
    output = tmp_path / "partial"
    result = diagnostic.run_diagnostic(state, output, updates=6, log_updates=(0, 6))
    assert result["status"] == "stopped" and result["updates"] == 3
    assert result["sample_counts"] == dict.fromkeys(diagnostic.SCENARIOS, 24)
    assert [r["updates"] for r in records(output) if r["kind"] == "checkpoint"] == [0, 3]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(state, output, updates=1)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_preexisting_stop_exits_without_updates(state, tmp_path):
    stop = tmp_path / "manual.stop"
    stop.touch()
    result = diagnostic.run_diagnostic(state, tmp_path / "stopped", stop_paths=(stop,), updates=2)
    assert result["status"] == "stopped" and result["updates"] == 0


def test_reproducibility_and_evaluation_does_not_consume_training_rng(state, tmp_path):
    outputs = [tmp_path / "first", tmp_path / "second"]
    results = [diagnostic.run_diagnostic(state, output, seed=81, updates=4, log_updates=logs)
               for output, logs in zip(outputs, ((0, 4), (0, 1, 2, 3, 4)))]
    assert results[0]["paired_batch_sha256"] == results[1]["paired_batch_sha256"]
    final = [[r for r in records(output) if r["kind"] == "checkpoint"][-1] for output in outputs]
    equal(*final)


def test_input_hash_pin_and_no_file_mutation(state, tmp_path, monkeypatch):
    path = tmp_path / "synthetic.pt"
    torch.save({"learner": state}, path)
    before = path.read_bytes()
    digest = diagnostic.file_hash(path)
    with pytest.raises(ValueError, match="SHA-256"):
        diagnostic.load_frozen_model(path)
    monkeypatch.setattr(diagnostic, "MODEL_SHA256", digest)
    loaded, actual = diagnostic.load_frozen_model(path)
    equal(loaded, state)
    result = diagnostic.run_diagnostic(loaded, tmp_path / "input_check", updates=2,
                                       provenance=dict(path=str(path), sha256=actual))
    assert result["input_unchanged"] is True and path.read_bytes() == before


def test_production_budget_and_checkpoint_constants():
    assert diagnostic.UPDATES == 3750
    assert diagnostic.LOG_UPDATES == (0, 375, 750, 1500, 3750)
    assert diagnostic.TD3.gamma == 1.0
    assert diagnostic.ARMS == {"A_tau_0.005": 0.005, "B_tau_1": 1.0}
