"""Synthetic shared TD3 contracts. No traffic, environment, or runner imports."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from td3 import SCENARIOS, TD3


def _fill(learner, count=12, scenario=None):
    rng = np.random.default_rng(731)
    for name in SCENARIOS if scenario is None else (scenario,):
        for index in range(count):
            learner.add(
                rng.normal(size=learner.observation_dim), rng.uniform(-1, 1, size=2),
                -float(index + 1), rng.normal(size=learner.observation_dim), index % 5 == 0,
                scenario=name,
            )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _equal(first, second)
    else:
        assert left == right


def _parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))


def test_zero_actor_shared_architecture_and_fixed_scenario_order():
    assert SCENARIOS == (
        "sweet_155_w", "sweet_170_w", "sweet_170_incident_w",
        "sweet_170_skew15_w", "sweet_190_w",
    )
    learner = TD3(5, seed=17)
    layers = [layer for layer in learner.actor if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (5, 64), (64, 64), (64, 2)
    ]
    assert isinstance(learner.actor[-1], nn.Tanh)
    assert torch.count_nonzero(layers[-1].weight) == 0
    assert torch.count_nonzero(layers[-1].bias) == 0
    for observation in (np.zeros(5), np.arange(5), np.full(5, -1e6)):
        np.testing.assert_array_equal(learner.act(observation), np.zeros(2))
    _equal(learner.actor.state_dict(), learner.actor_target.state_dict())
    _equal(learner.critics.state_dict(), learner.critic_targets.state_dict())
    first, second = learner.critics
    for critic in learner.critics:
        assert [(layer.in_features, layer.out_features) for layer in critic if isinstance(layer, nn.Linear)] == [
            (7, 64), (64, 64), (64, 1)
        ]
    assert {p.data_ptr() for p in first.parameters()}.isdisjoint(p.data_ptr() for p in second.parameters())
    assert not torch.equal(first[0].weight, second[0].weight)
    for module in (learner.actor, learner.critics, learner.actor_target, learner.critic_targets):
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in module.parameters())
    assert torch.get_num_threads() == 1
    assert learner.replay_counts() == dict.fromkeys(SCENARIOS, 0)
    assert learner.total_transition_count() == 0


def test_act_is_deterministic_bounded_copied_and_without_gradients():
    learner = TD3(3, seed=7)
    with torch.no_grad():
        learner.actor[-2].bias.copy_(torch.tensor([100.0, -100.0]))
    observation = np.array([1.0, 2.0, 3.0])
    observation.flags.writeable = False
    original_state = learner.state_dict()
    grad_enabled = []
    hook = learner.actor.register_forward_pre_hook(
        lambda *_: grad_enabled.append(torch.is_grad_enabled())
    )
    try:
        first, second = learner.act(observation), learner.act(observation)
    finally:
        hook.remove()
    assert first.shape == (2,) and first.dtype == np.float32
    np.testing.assert_array_equal(first, [1, -1])
    np.testing.assert_array_equal(first, second)
    assert not any(grad_enabled)
    first[:] = 0
    np.testing.assert_array_equal(learner.act(observation), second)
    assert all(p.grad is None for p in learner.actor.parameters())
    np.testing.assert_array_equal(observation, [1, 2, 3])
    _equal(original_state, learner.state_dict())


class _ConstantCritic(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32).reshape(-1, 1))

    def forward(self, inputs):
        return self.values.expand(len(inputs), 1)


def test_targets_bootstrap_only_continuing_and_take_elementwise_twin_min():
    learner = TD3(3, seed=9)
    learner.critic_targets = nn.ModuleList([
        _ConstantCritic([4, -7, 20, 8]), _ConstantCritic([9, 3, 2, -10])
    ])
    learner.critics = nn.ModuleList([_ConstantCritic([100]), _ConstantCritic([200])])
    rewards = torch.tensor([[-1000.0], [-2.0], [-3.0], [-4.0]], requires_grad=True)
    next_obs = torch.ones((4, 3), requires_grad=True)
    terminated = torch.tensor([[False], [False], [True], [True]])
    result = learner._target_values(rewards, next_obs, terminated)
    torch.testing.assert_close(result, torch.tensor([[-996.0], [-9.0], [-3.0], [-4.0]]))
    assert not result.requires_grad and result.grad_fn is None
    assert rewards.grad is None and next_obs.grad is None
    assert all(p.grad is None for p in learner.critic_targets.parameters())


def test_target_policy_noise_scale_clip_action_bounds_and_target_actor(monkeypatch):
    learner = TD3(3, seed=9)
    with torch.no_grad():
        learner.actor_target[-2].bias.copy_(torch.atanh(torch.tensor([0.8, -0.8])))
        learner.actor[-2].bias.zero_()
    captures = []
    hook = learner.critic_targets[0].register_forward_pre_hook(
        lambda _module, args: captures.append(args[0].detach().clone())
    )
    draws = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [0.25, -0.25]])

    def fixed_noise(shape, *, generator, device, dtype):
        assert shape == (3, 2)
        assert generator is learner._torch_rng and device == "cpu"
        assert dtype == torch.float32
        return draws.clone()

    monkeypatch.setattr(torch, "randn", fixed_noise)
    try:
        learner._target_values(torch.zeros(3, 1), torch.ones(3, 3), torch.zeros(3, 1, dtype=torch.bool))
    finally:
        hook.remove()
    torch.testing.assert_close(captures[0][:, -2:], torch.tensor([[1.0, -1.0], [0.3, -0.3], [0.85, -0.85]]))


@pytest.mark.parametrize("late_scenario", SCENARIOS)
def test_default_update_waits_for_eight_in_every_scenario_without_mutation(late_scenario):
    learner = TD3(3, seed=5, hidden=8)
    for scenario in SCENARIOS:
        _fill(learner, 7 if scenario == late_scenario else 17, scenario)
    before = learner.state_dict()
    assert learner.update() == {}
    _equal(before, learner.state_dict())
    _fill(learner, 1, late_scenario)
    metrics = learner.update()
    assert metrics["updates"] == 1
    assert metrics["sampled_per_scenario"] == dict.fromkeys(SCENARIOS, 8)
    assert learner.sample_counts == dict.fromkeys(SCENARIOS, 8)


def test_one_scenario_alone_cannot_update_even_for_small_batch():
    learner = TD3(3, seed=5, hidden=8)
    _fill(learner, 100, SCENARIOS[0])
    before = learner.state_dict()
    for batch_size in (5, 40, 100):
        assert learner.update(batch_size) == {}
    _equal(before, learner.state_dict())


def test_unequal_replay_counts_produce_equal_actual_batches_with_replacement():
    learner = TD3(3, seed=0, hidden=8)
    sizes = (8, 13, 47, 211, 997)
    for group, (scenario, count) in enumerate(zip(SCENARIOS, sizes)):
        for index in range(count):
            learner.add([group, index, 1], [-0.5, 0.5], -group - 1, [group, index + 1, 1], False, scenario)
    assert learner.replay_counts() == dict(zip(SCENARIOS, sizes))
    assert learner.total_transition_count() == sum(sizes)
    returned_counts = learner.replay_counts()
    returned_counts[SCENARIOS[0]] = 999
    assert learner.replay_counts()[SCENARIOS[0]] == 8
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = deepcopy(learner.state_dict()["numpy_rng"])
    captures = [[], []]
    hooks = [
        critic.register_forward_pre_hook(
            lambda _module, args, group=group: captures[group].append(args[0].detach().clone())
        )
        for group, critic in enumerate(learner.critics)
    ]
    try:
        for batch_size in (40, 10):
            per_scenario = batch_size // 5
            expected_rows = []
            for group, size in enumerate(sizes):
                draws = expected_rng.integers(size, size=per_scenario)
                expected_rows.extend([group, int(index), 1, -0.5, 0.5] for index in draws)
                if group == 0 and batch_size == 40:
                    assert len(set(draws)) < len(draws)
            for values in captures:
                values.clear()
            metrics = learner.update() if batch_size == 40 else learner.update(batch_size)
            expected = torch.tensor(expected_rows, dtype=torch.float32)
            assert metrics["sampled_per_scenario"] == dict.fromkeys(SCENARIOS, per_scenario)
            for values in captures:
                assert torch.equal(values[0], expected)
                assert torch.bincount(values[0][:, 0].long()).tolist() == [per_scenario] * 5
            assert len(captures[1]) == 1
            if batch_size == 10:
                # The delayed actor uses the same balanced observations.
                assert len(captures[0]) == 2
                assert torch.equal(captures[0][1][:, :3], expected[:, :3])
    finally:
        for hook in hooks:
            hook.remove()
    assert learner.updates == 2
    assert learner.sample_counts == dict.fromkeys(SCENARIOS, 10)
    metrics["sampled_per_scenario"][SCENARIOS[0]] = 999
    assert learner.sample_counts[SCENARIOS[0]] == 10
    for value in learner.critic_optimizer.state.values():
        assert value["step"].item() == 2


def test_critics_learn_actor_and_targets_delay_and_polyak_is_exact():
    learner = TD3(3, seed=5, hidden=16)
    _fill(learner)
    actor_before = _parameters(learner.actor)
    critics_before = [_parameters(critic) for critic in learner.critics]
    actor_target_before = _parameters(learner.actor_target)
    critic_targets_before = _parameters(learner.critic_targets)
    first = learner.update()
    assert first["updates"] == learner.updates == 1
    assert "actor_loss" not in first
    assert not _changed(actor_before, learner.actor)
    assert all(_changed(before, critic) for before, critic in zip(critics_before, learner.critics))
    assert not _changed(actor_target_before, learner.actor_target)
    assert not _changed(critic_targets_before, learner.critic_targets)
    assert not learner.actor_optimizer.state
    second = learner.update()
    assert second["updates"] == learner.updates == 2
    assert np.isfinite(second["actor_loss"]) and np.isfinite(second["critic_loss"])
    assert _changed(actor_before, learner.actor)
    assert _changed(actor_target_before, learner.actor_target)
    assert _changed(critic_targets_before, learner.critic_targets)
    for old_parameters, online, target in (
        (actor_target_before, learner.actor, learner.actor_target),
        (critic_targets_before, learner.critics, learner.critic_targets),
    ):
        for old, current, actual in zip(old_parameters, online.parameters(), target.parameters()):
            expected = old.mul(0.995).add(current.detach(), alpha=0.005)
            assert torch.equal(expected, actual)
    for target in (learner.actor_target, learner.critic_targets):
        assert all(not p.requires_grad and p.grad is None for p in target.parameters())
    for value in learner.actor_optimizer.state.values():
        assert value["step"].item() == 1
    for value in learner.critic_optimizer.state.values():
        assert value["step"].item() == 2
    before_third = [_parameters(m) for m in (learner.actor, learner.actor_target, learner.critic_targets)]
    assert "actor_loss" not in learner.update()
    for before, module in zip(before_third, (learner.actor, learner.actor_target, learner.critic_targets)):
        assert not _changed(before, module)


@pytest.mark.parametrize("terminated,expected_loss", [(False, 2 * 993.0**2), (True, 2_000_000.0)])
def test_critic_loss_uses_requested_actions_unclipped_rewards_and_bootstrap(terminated, expected_loss):
    learner = TD3(2, seed=3, hidden=8)
    for critic in learner.critics:
        for parameter in critic.parameters():
            nn.init.zeros_(parameter)
    learner.critic_targets = nn.ModuleList([_ConstantCritic([7]), _ConstantCritic([11])])
    for scenario in SCENARIOS:
        learner.add([1, 2], [-0.75, 0.25], -1000, [2, 3], terminated, scenario)
    captured = []
    hook = learner.critics[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].clone())
    )
    try:
        metrics = learner.update(batch_size=5)
    finally:
        hook.remove()
    assert metrics["critic_loss"] == expected_loss
    torch.testing.assert_close(captured[0], torch.tensor([[1.0, 2.0, -0.75, 0.25]]).expand(5, 4))


def test_terminal_transitions_do_not_bootstrap_into_next_episode():
    class NextObservationCritic(nn.Module):
        def forward(self, inputs):
            return inputs[:, :1]

    learner = TD3(2, seed=0, hidden=8)
    learner.critic_targets = nn.ModuleList([NextObservationCritic(), NextObservationCritic()])
    for critic in learner.critics:
        for parameter in critic.parameters():
            nn.init.zeros_(parameter)
    for scenario in SCENARIOS:
        for index in range(8):
            terminal = index % 2 == 0
            learner.add([index, 1], [0, 0], -2, [100000 if terminal else 7, 2], terminal, scenario)
    captured = []
    original = learner._target_values

    def capture(rewards, next_observations, terminated):
        targets = original(rewards, next_observations, terminated)
        assert terminated.any() and (~terminated).any()
        assert torch.equal(targets[terminated], rewards[terminated])
        assert torch.equal(targets[~terminated], rewards[~terminated] + 7)
        captured.append(targets)
        return targets

    learner._target_values = capture
    metrics = learner.update()
    assert metrics["critic_loss"] == float(2 * captured[0].square().mean())


def test_replay_copies_all_source_inputs():
    learner = TD3(2, seed=8, hidden=8)
    obs, action, reward, next_obs = (np.array(v) for v in ([1.0, 2.0], [0.2, -0.3], -2000.0, [3.0, 4.0]))
    expected = [item.copy() for item in (obs, action, reward, next_obs)]
    learner.add(obs, action, reward, next_obs, np.bool_(False), SCENARIOS[2])
    for actual, original in zip((obs, action, reward, next_obs), expected):
        np.testing.assert_array_equal(actual, original)
        actual[...] = 99
    stored = learner.state_dict()["replay"][SCENARIOS[2]]
    for name, original in zip(("observations", "actions", "rewards", "next_observations"), expected):
        np.testing.assert_array_equal(stored[name][0].numpy(), np.asarray(original, dtype=np.float32))
    assert stored["terminated"].dtype == torch.bool and stored["terminated"][0].item() is False


def test_each_scenario_has_equal_bounded_fifo_and_cannot_evict_a_minority():
    learner = TD3(2, seed=8, hidden=8)
    for scenario in SCENARIOS:
        learner.add([-1, 0], [0, 0], -1, [0, 0], False, scenario)
    minority_before = learner.state_dict()["replay"]
    for index in range(2003):
        learner.add([index, 0], [-1, 1], -index, [index + 1, 0], index % 2 == 0, SCENARIOS[0])
    current = learner.state_dict()["replay"]
    for scenario in SCENARIOS[1:]:
        _equal(minority_before[scenario], current[scenario])
        for index in range(2003):
            learner.add([index, 0], [-1, 1], -index, [index + 1, 0], index % 2 == 0, scenario)
    assert learner.replay_counts() == dict.fromkeys(SCENARIOS, 2000)
    assert learner.total_transition_count() == 10000
    for group in learner.state_dict()["replay"].values():
        np.testing.assert_array_equal(group["observations"][0].numpy(), [3, 0])
        np.testing.assert_array_equal(group["observations"][-1].numpy(), [2002, 0])
    restored = TD3(2, seed=99, hidden=8)
    restored.load_state_dict(learner.state_dict())
    for candidate in (learner, restored):
        for scenario in SCENARIOS:
            candidate.add([2003, 0], [0.5, -0.5], -2003, [2004, 0], True, scenario)
    _equal(learner.state_dict(), restored.state_dict())
    assert learner.update() == restored.update()
    _equal(learner.state_dict(), restored.state_dict())


@pytest.mark.parametrize("observation", [
    [1, 2], [[1, 2, 3]], [1, 2, np.nan], [1, np.inf, 3], [1, 2, 1e100],
    ["1", "2", "3"], [1j, 2, 3], [True, False, True],
])
def test_act_rejects_invalid_observations(observation):
    with pytest.raises(ValueError):
        TD3(3, seed=0, hidden=8).act(observation)


@pytest.mark.parametrize("field,value", [
    ("obs", [1, 2]), ("obs", [1, np.nan, 3]), ("obs", [1, 2, 1e100]),
    ("next_obs", [[1, 2, 3]]), ("next_obs", [1, 2, np.inf]),
    ("requested_action", [0]), ("requested_action", [[0, 0]]),
    ("requested_action", [0, np.nan]), ("requested_action", [-1.01, 0]),
    ("requested_action", [np.nextafter(1.0, 2.0), 0]),
    ("requested_action", ["0", "0"]), ("reward", [1]), ("reward", np.inf),
    ("reward", np.nan), ("reward", 1e100), ("reward", "-1"), ("reward", True),
    ("terminated", "False"), ("terminated", "True"), ("terminated", 0),
    ("terminated", 1), ("terminated", None), ("terminated", np.array(False)),
    ("scenario", "unknown"), ("scenario", "sweet_190_incident_w"),
    ("scenario", None), ("scenario", 0), ("scenario", True), ("scenario", []),
])
def test_add_validation_is_atomic(field, value):
    learner = TD3(3, seed=0, hidden=8)
    args = dict(obs=[1, 2, 3], requested_action=[0, 0], reward=-1, next_obs=[2, 3, 4],
                terminated=False, scenario=SCENARIOS[0])
    learner.add(**args)
    before = learner.state_dict()
    args[field] = value
    with pytest.raises(ValueError):
        learner.add(**args)
    _equal(before, learner.state_dict())


def test_scenario_is_required():
    with pytest.raises(TypeError):
        TD3(2, seed=0, hidden=8).add([0, 0], [0, 0], -1, [1, 1], False)


@pytest.mark.parametrize("kwargs", [
    {"observation_dim": 0}, {"observation_dim": True}, {"observation_dim": 2.5},
    {"hidden": 0}, {"hidden": False}, {"seed": -1}, {"seed": True}, {"seed": 2**64},
])
def test_constructor_validation(kwargs):
    args = dict(observation_dim=3, seed=0, hidden=8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        TD3(**args)


@pytest.mark.parametrize("batch_size", [0, -1, True, np.bool_(False), 2.5, "40", 1, 8, 32, 41])
def test_update_rejects_invalid_or_nondivisible_batch_size_before_warmup(batch_size):
    learner = TD3(3, seed=0, hidden=8)
    before = learner.state_dict()
    with pytest.raises(ValueError):
        learner.update(batch_size)
    _equal(before, learner.state_dict())


@pytest.mark.parametrize("completed_updates", [0, 1, 2, 3])
def test_checkpoint_exact_continuation_with_replay_optimizers_and_local_rng(completed_updates):
    learner = TD3(3, seed=42, hidden=16)
    _fill(learner, 17)
    for _ in range(completed_updates):
        learner.update(15)
    saved = learner.state_dict()
    for name, steps in (("critic_optimizer", completed_updates), ("actor_optimizer", completed_updates // 2)):
        history = saved[name]["state"]
        assert bool(history) == bool(steps)
        assert {entry["step"].item() for entry in history.values()} == ({steps} if steps else set())
    pristine = deepcopy(saved)
    serialized = BytesIO()
    torch.save(saved, serialized)
    serialized.seek(0)
    decoded = torch.load(serialized, map_location="cpu", weights_only=True)
    restored = TD3(3, seed=999, hidden=16)
    _fill(restored, 8)
    restored.update()
    assert restored.load_state_dict(decoded) is None
    _equal(learner.state_dict(), restored.state_dict())
    np.testing.assert_array_equal(learner.act([1, -2, 3]), restored.act([1, -2, 3]))
    assert restored.spec()["seed"] == 42
    for batch_size in (40, 10, 15):
        expected_metrics = learner.update(batch_size)
        torch.rand(51)
        np.random.random(51)
        actual_metrics = restored.update(batch_size)
        assert expected_metrics == actual_metrics
        _equal(learner.state_dict(), restored.state_dict())
        np.testing.assert_array_equal(learner.act([1, -2, 3]), restored.act([1, -2, 3]))
    _equal(saved, pristine)
    _equal(decoded, pristine)
    assert learner.sample_counts == dict.fromkeys(SCENARIOS, completed_updates * 3 + 13)
    for snapshot in (saved, decoded):
        snapshot["actor"]["4.bias"].fill_(99)
        snapshot["sample_counts"][SCENARIOS[0]] = 999
        snapshot["torch_rng"].zero_()
        snapshot["numpy_rng"]["state"]["state"] = 1
        for group in snapshot["replay"].values():
            for value in group.values():
                value.fill_(True if value.dtype == torch.bool else 99)
        for value in snapshot["critic_optimizer"]["state"].values():
            value["exp_avg"].fill_(99)
    _equal(learner.state_dict(), restored.state_dict())


def test_empty_checkpoint_roundtrip():
    learner, restored = TD3(2, 1, hidden=8), TD3(2, 2, hidden=8)
    restored.load_state_dict(learner.state_dict())
    _equal(learner.state_dict(), restored.state_dict())
    assert restored.update() == {}
    for group in restored.state_dict()["replay"].values():
        assert group["observations"].shape == (0, 2)
        assert group["actions"].shape == (0, 2)
        assert group["rewards"].shape == group["terminated"].shape == (0,)


@pytest.mark.parametrize("change", [
    "old_format", "dimension", "hidden", "gamma", "batch_size", "seed",
    "scenario_order", "scenario_missing", "scenario_unknown", "scenario_duplicate",
    "replay_missing", "replay_unknown", "replay_not_dict",
])
def test_rejects_incompatible_checkpoints_before_mutation(change):
    learner = TD3(3, 1, hidden=8)
    _fill(learner, 2)
    before = learner.state_dict()
    invalid = deepcopy(before)
    if change == "old_format":
        invalid["format"] = "sdmpc-budget-td3-v1"
    elif change == "dimension":
        invalid["observation_dim"] = 4
    elif change in ("hidden", "gamma"):
        invalid["spec"][change] = 0
    elif change == "batch_size":
        invalid["spec"]["default_batch_size"] = 32
    elif change == "seed":
        invalid["spec"]["seed"] = 2**64
    elif change == "scenario_order":
        invalid["spec"]["scenarios"].reverse()
    elif change == "scenario_missing":
        invalid["spec"]["scenarios"].pop()
    elif change == "scenario_unknown":
        invalid["spec"]["scenarios"][0] = "unknown"
    elif change == "scenario_duplicate":
        invalid["spec"]["scenarios"][0] = SCENARIOS[1]
    elif change == "replay_missing":
        del invalid["replay"][SCENARIOS[0]]
    elif change == "replay_unknown":
        invalid["replay"]["unknown"] = invalid["replay"][SCENARIOS[0]]
    else:
        invalid["replay"] = list(invalid["replay"].values())
    with pytest.raises(ValueError):
        learner.load_state_dict(invalid)
    _equal(before, learner.state_dict())


@pytest.mark.parametrize("change", [
    "observations", "actions", "rewards", "next_observations", "terminated",
    "missing_column", "extra_column", "non_tensor", "numeric_termination", "wrong_dtype",
    "capacity", "nan_obs", "inf_reward", "nan_next_obs", "unbounded_action",
])
def test_rejects_malformed_replay_before_mutation(change):
    learner = TD3(3, 1, hidden=8)
    _fill(learner, 2)
    before = learner.state_dict()
    invalid = deepcopy(before)
    group = invalid["replay"][SCENARIOS[-1]]
    if change in group:
        group[change] = group[change].unsqueeze(0)
    elif change == "missing_column":
        del group["actions"]
    elif change == "extra_column":
        group["unknown"] = torch.zeros(2)
    elif change == "non_tensor":
        group["observations"] = group["observations"].numpy()
    elif change == "numeric_termination":
        group["terminated"] = torch.tensor([0, 1])
    elif change == "wrong_dtype":
        group["observations"] = group["observations"].double()
    elif change == "capacity":
        group["rewards"] = torch.zeros(2001)
    elif change == "nan_obs":
        group["observations"][0, 0] = float("nan")
    elif change == "inf_reward":
        group["rewards"][0] = float("inf")
    elif change == "nan_next_obs":
        group["next_observations"][0, 0] = float("nan")
    else:
        group["actions"][0, 0] = 1.01
    with pytest.raises(ValueError):
        learner.load_state_dict(invalid)
    _equal(before, learner.state_dict())


@pytest.mark.parametrize("change", [
    "negative_updates", "bool_updates", "fractional_updates", "missing_updates",
    "negative_samples", "bool_samples", "fractional_samples", "missing_samples",
    "missing_scenario", "unknown_scenario", "unequal_samples", "zero_samples",
    "samples_without_updates", "too_many_samples",
])
def test_rejects_invalid_update_and_sampling_counters_before_mutation(change):
    learner = TD3(3, 1, hidden=8)
    _fill(learner, 8)
    learner.update()
    before = learner.state_dict()
    invalid = deepcopy(before)
    if change == "negative_updates":
        invalid["updates"] = -1
    elif change == "bool_updates":
        invalid["updates"] = True
    elif change == "fractional_updates":
        invalid["updates"] = 1.5
    elif change == "missing_updates":
        del invalid["updates"]
    elif change == "negative_samples":
        invalid["sample_counts"] = dict.fromkeys(SCENARIOS, -1)
    elif change == "bool_samples":
        invalid["sample_counts"] = dict.fromkeys(SCENARIOS, True)
    elif change == "fractional_samples":
        invalid["sample_counts"] = dict.fromkeys(SCENARIOS, 8.5)
    elif change == "missing_samples":
        del invalid["sample_counts"]
    elif change == "missing_scenario":
        del invalid["sample_counts"][SCENARIOS[0]]
    elif change == "unknown_scenario":
        invalid["sample_counts"]["unknown"] = 8
    elif change == "unequal_samples":
        invalid["sample_counts"][SCENARIOS[0]] = 7
    elif change == "zero_samples":
        invalid["sample_counts"] = dict.fromkeys(SCENARIOS, 0)
    elif change == "samples_without_updates":
        invalid["updates"] = 0
    else:
        invalid["sample_counts"] = dict.fromkeys(SCENARIOS, 2001)
    with pytest.raises(ValueError):
        learner.load_state_dict(invalid)
    _equal(before, learner.state_dict())


@pytest.fixture(scope="module")
def trained_checkpoint():
    learner = TD3(3, seed=19, hidden=8)
    _fill(learner, 8)
    learner.update()
    learner.update()
    return learner.state_dict()


def _assert_checkpoint_rejected_without_mutation(state, message):
    recipient = TD3(3, seed=999, hidden=8)
    before = recipient.state_dict()
    with pytest.raises(ValueError, match=message):
        recipient.load_state_dict(state)
    _equal(before, recipient.state_dict())


@pytest.mark.parametrize("optimizer", ["actor_optimizer", "critic_optimizer"])
@pytest.mark.parametrize("field,value", [
    ("lr", 0.0), ("lr", torch.tensor(3e-4)), ("betas", (0.8, 0.999)),
    ("eps", 1e-4), ("weight_decay", 0.1), ("amsgrad", True),
    ("maximize", True), ("foreach", True), ("capturable", True),
    ("differentiable", True), ("fused", True), ("decoupled_weight_decay", True),
])
def test_rejects_effective_adam_settings_despite_matching_spec(trained_checkpoint, optimizer, field, value):
    invalid = deepcopy(trained_checkpoint)
    invalid[optimizer]["param_groups"][0][field] = value
    _assert_checkpoint_rejected_without_mutation(invalid, optimizer)


@pytest.mark.parametrize("optimizer", ["actor_optimizer", "critic_optimizer"])
@pytest.mark.parametrize("change", [
    "missing_optimizer", "empty_history", "missing_history_entry", "extra_history_entry",
    "missing_moment", "missing_variance", "missing_step", "extra_moment",
    "moment_shape", "variance_shape", "moment_dtype", "moment_not_tensor",
    "moment_nan", "variance_inf", "negative_variance", "saved_parameter_shape",
    "wrong_step", "zero_step", "fractional_step", "step_nan", "step_shape",
    "step_bool", "step_not_tensor", "duplicate_parameter", "missing_parameter",
    "parameter_order", "extra_group", "missing_setting",
])
def test_rejects_incomplete_or_incoherent_adam_history(trained_checkpoint, optimizer, change):
    invalid = deepcopy(trained_checkpoint)
    payload = invalid[optimizer]
    group = payload["param_groups"][0]
    parameter_id = group["params"][0]
    entry = payload["state"][parameter_id]
    if change == "missing_optimizer":
        del invalid[optimizer]
    elif change == "empty_history":
        payload["state"] = {}
    elif change == "missing_history_entry":
        del payload["state"][parameter_id]
    elif change == "extra_history_entry":
        payload["state"][999] = deepcopy(entry)
    elif change == "missing_moment":
        del entry["exp_avg"]
    elif change == "missing_variance":
        del entry["exp_avg_sq"]
    elif change == "missing_step":
        del entry["step"]
    elif change == "extra_moment":
        entry["max_exp_avg_sq"] = entry["exp_avg_sq"].clone()
    elif change == "moment_shape":
        entry["exp_avg"] = entry["exp_avg"].flatten()
    elif change == "variance_shape":
        entry["exp_avg_sq"] = entry["exp_avg_sq"].flatten()
    elif change == "moment_dtype":
        entry["exp_avg"] = entry["exp_avg"].double()
    elif change == "moment_not_tensor":
        entry["exp_avg"] = entry["exp_avg"].numpy()
    elif change == "moment_nan":
        entry["exp_avg"].fill_(float("nan"))
    elif change == "variance_inf":
        entry["exp_avg_sq"].fill_(float("inf"))
    elif change == "negative_variance":
        entry["exp_avg_sq"].fill_(-1)
    elif change == "saved_parameter_shape":
        network = "actor" if optimizer == "actor_optimizer" else "critics"
        key = next(iter(invalid[network]))
        invalid[network][key] = invalid[network][key].flatten()
        entry["exp_avg"] = entry["exp_avg"].flatten()
        entry["exp_avg_sq"] = entry["exp_avg_sq"].flatten()
    elif change == "wrong_step":
        entry["step"].add_(1)
    elif change == "zero_step":
        entry["step"].zero_()
    elif change == "fractional_step":
        entry["step"].add_(0.5)
    elif change == "step_nan":
        entry["step"].fill_(float("nan"))
    elif change == "step_shape":
        entry["step"] = entry["step"].unsqueeze(0)
    elif change == "step_bool":
        entry["step"] = torch.tensor(True)
    elif change == "step_not_tensor":
        entry["step"] = entry["step"].item()
    elif change == "duplicate_parameter":
        group["params"][0] = group["params"][1]
    elif change == "missing_parameter":
        group["params"].pop()
    elif change == "parameter_order":
        group["params"].reverse()
    elif change == "extra_group":
        payload["param_groups"].append(deepcopy(group))
    else:
        del group["lr"]
    message = "invalid checkpoint (actor|critics)/" if change == "saved_parameter_shape" else optimizer
    _assert_checkpoint_rejected_without_mutation(invalid, message)


def test_rejects_changed_update_count_when_adam_steps_disagree(trained_checkpoint):
    invalid = deepcopy(trained_checkpoint)
    invalid["updates"] = 3
    _assert_checkpoint_rejected_without_mutation(invalid, "critic_optimizer step")


@pytest.mark.parametrize("updates,optimizer", [(0, "critic_optimizer"), (0, "actor_optimizer"), (1, "actor_optimizer")])
def test_rejects_adam_history_before_first_scheduled_step(trained_checkpoint, updates, optimizer):
    learner = TD3(3, seed=19, hidden=8)
    _fill(learner, 8)
    for _ in range(updates):
        learner.update()
    invalid = learner.state_dict()
    invalid[optimizer]["state"] = deepcopy(trained_checkpoint[optimizer]["state"])
    _assert_checkpoint_rejected_without_mutation(invalid, optimizer + " history")


@pytest.mark.parametrize("change", ["just_above_bound", "capacity_bound", "minority_group", "empty_group"])
def test_rejects_sampling_counts_impossible_for_restored_replay(trained_checkpoint, change):
    invalid = deepcopy(trained_checkpoint)
    if change in ("just_above_bound", "capacity_bound"):
        invalid["sample_counts"] = dict.fromkeys(SCENARIOS, 17 if change == "just_above_bound" else 4000)
    else:
        size = 4 if change == "minority_group" else 0
        group = invalid["replay"][SCENARIOS[-1]]
        invalid["replay"][SCENARIOS[-1]] = {key: value[:size].clone() for key, value in group.items()}
    _assert_checkpoint_rejected_without_mutation(invalid, "sample_counts exceed possible replay history")


def test_replay_history_bound_accepts_exact_limit_with_unequal_groups():
    learner = TD3(3, seed=19, hidden=8)
    _fill(learner, 8)
    for scenario in SCENARIOS[1:]:
        _fill(learner, 12, scenario)
    learner.update()
    learner.update()
    assert learner.sample_counts == dict.fromkeys(SCENARIOS, 16)
    restored = TD3(3, seed=999, hidden=8)
    restored.load_state_dict(learner.state_dict())
    _equal(learner.state_dict(), restored.state_dict())
    assert learner.update() == restored.update()
    _equal(learner.state_dict(), restored.state_dict())


def test_initialization_and_updates_do_not_consume_global_rng_or_set_interop_threads(monkeypatch):
    torch_state = torch.get_rng_state().clone()
    numpy_state = deepcopy(np.random.get_state())

    def forbidden(*_args):
        pytest.fail("TD3 must not change inter-op threads")

    monkeypatch.setattr(torch, "set_num_interop_threads", forbidden)
    learner = TD3(3, seed=91, hidden=8)
    _fill(learner)
    learner.act([1, 2, 3])
    learner.update()
    learner.update()
    restored = TD3(3, seed=37, hidden=8)
    restored.load_state_dict(learner.state_dict())
    assert torch.equal(torch_state, torch.get_rng_state())
    _equal(numpy_state, np.random.get_state())


def test_combined_updates_are_exactly_the_prior_carry_td3_math():
    reference_path = Path(__file__).parents[1] / "sdmpc_rl_carry_20260929" / "td3.py"
    module_spec = spec_from_file_location("carry_td3_reference", reference_path)
    reference = module_from_spec(module_spec)
    module_spec.loader.exec_module(reference)
    learner = TD3(3, seed=19, hidden=16)
    prior = reference.TD3(3, seed=19, hidden=16)
    _fill(learner, 19)

    class SequentialBatch:
        def integers(self, high, size):
            assert high == size
            return np.arange(size)

    prior._numpy_rng = SequentialBatch()
    for batch_size in (40, 15, 10):
        rng = np.random.default_rng()
        snapshot = learner.state_dict()
        rng.bit_generator.state = deepcopy(snapshot["numpy_rng"])
        prior.replay.clear()
        for scenario in SCENARIOS:
            group = snapshot["replay"][scenario]
            for index in rng.integers(len(group["rewards"]), size=batch_size // 5):
                prior.add(
                    group["observations"][index].numpy(), group["actions"][index].numpy(),
                    group["rewards"][index].item(), group["next_observations"][index].numpy(),
                    group["terminated"][index].item(),
                )
        actual = learner.update(batch_size)
        actual.pop("sampled_per_scenario")
        assert actual == prior.update(batch_size)
        for name in ("actor", "critics", "actor_target", "critic_targets", "actor_optimizer", "critic_optimizer"):
            _equal(getattr(learner, name).state_dict(), getattr(prior, name).state_dict())
        assert torch.equal(learner._torch_rng.get_state(), prior._torch_rng.get_state())
        np.testing.assert_array_equal(learner.act([1, -2, 3]), prior.act([1, -2, 3]))
