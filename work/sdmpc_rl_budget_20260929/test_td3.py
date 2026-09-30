"""Synthetic TD3 contracts. No traffic, environment, runtime, or legacy imports."""

from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch
from torch import nn

from td3 import TD3


def _fill(learner, count=40):
    rng = np.random.default_rng(731)
    for index in range(count):
        learner.add(
            rng.normal(size=learner.observation_dim), rng.uniform(-1, 1, size=2),
            -float(index + 1), rng.normal(size=learner.observation_dim), index % 5 == 0,
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


def test_zero_actor_architecture_and_spec():
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
    assert {p.data_ptr() for p in first.parameters()}.isdisjoint(
        p.data_ptr() for p in second.parameters()
    )
    assert not torch.equal(first[0].weight, second[0].weight)
    for module in (learner.actor, learner.critics, learner.actor_target, learner.critic_targets):
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in module.parameters())
    assert torch.get_num_threads() == 1
    spec = learner.spec()
    for key, value in {
        "observation_dim": 5, "hidden": 64, "action_dim": 2, "gamma": 1.0,
        "learning_rate": 3e-4, "tau": 0.005, "policy_noise": 0.2,
        "noise_clip": 0.5, "policy_delay": 2, "replay_capacity": 10000,
    }.items():
        assert spec[key] == value


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
    # Online predictions must not participate in target construction.
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


def test_update_warmup_is_a_complete_noop():
    learner = TD3(3, seed=5)
    _fill(learner, 31)
    before = learner.state_dict()
    assert learner.update() == {}
    _equal(before, learner.state_dict())


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
    learner.add([1, 2], [-0.75, 0.25], -1000, [2, 3], terminated)
    captured = []
    hook = learner.critics[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].clone())
    )
    try:
        metrics = learner.update(batch_size=1)
    finally:
        hook.remove()
    assert metrics["critic_loss"] == expected_loss
    torch.testing.assert_close(captured[0], torch.tensor([[1.0, 2.0, -0.75, 0.25]]))


def test_replay_copies_inputs_and_is_bounded_fifo():
    learner = TD3(2, seed=8, hidden=8)
    obs, action, reward, next_obs = (np.array(v) for v in ([1.0, 2.0], [0.2, -0.3], -2000.0, [3.0, 4.0]))
    expected = [item.copy() for item in (obs, action, reward, next_obs)]
    learner.add(obs, action, reward, next_obs, np.bool_(False))
    for actual, original in zip((obs, action, reward, next_obs), expected):
        np.testing.assert_array_equal(actual, original)
        actual[...] = 99
    stored = learner.replay[0]
    for actual, original in zip(stored[:4], expected):
        np.testing.assert_array_equal(actual, np.asarray(original, dtype=np.float32))
    assert stored[4] is False
    for index in range(10002):
        learner.add([index, 0], [-1, 1], -index, [index + 1, 0], index % 2 == 0)
    assert len(learner.replay) == 10000
    np.testing.assert_array_equal(learner.replay[0][0], [2, 0])
    np.testing.assert_array_equal(learner.replay[-1][0], [10001, 0])


@pytest.mark.parametrize("observation", [
    [1, 2], [[1, 2, 3]], [1, 2, np.nan], [1, np.inf, 3], [1, 2, 1e100],
    ["1", "2", "3"], [1j, 2, 3], [True, False, True],
])
def test_act_rejects_invalid_observations(observation):
    with pytest.raises(ValueError):
        TD3(3, seed=0, hidden=8).act(observation)


@pytest.mark.parametrize("field,value", [
    ("obs", [1, 2]), ("obs", [1, np.nan, 3]),
    ("next_obs", [[1, 2, 3]]), ("next_obs", [1, 2, np.inf]),
    ("requested_action", [0]), ("requested_action", [[0, 0]]),
    ("requested_action", [0, np.nan]), ("requested_action", [-1.01, 0]),
    ("requested_action", [np.nextafter(1.0, 2.0), 0]),
    ("requested_action", ["0", "0"]), ("reward", [1]), ("reward", np.inf),
    ("reward", np.nan), ("reward", 1e100), ("reward", "-1"),
    ("terminated", "False"), ("terminated", "True"), ("terminated", 0),
    ("terminated", 1), ("terminated", None), ("terminated", np.array(False)),
])
def test_add_validation_is_atomic(field, value):
    learner = TD3(3, seed=0, hidden=8)
    args = dict(obs=[1, 2, 3], requested_action=[0, 0], reward=-1, next_obs=[2, 3, 4], terminated=False)
    learner.add(**args)
    before = learner.state_dict()
    args[field] = value
    with pytest.raises(ValueError):
        learner.add(**args)
    _equal(before, learner.state_dict())


@pytest.mark.parametrize("kwargs", [
    {"observation_dim": 0}, {"observation_dim": True}, {"observation_dim": 2.5},
    {"hidden": 0}, {"hidden": False}, {"seed": -1}, {"seed": True}, {"seed": 2**64},
])
def test_constructor_validation(kwargs):
    args = dict(observation_dim=3, seed=0, hidden=8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        TD3(**args)


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.5, "32"])
def test_update_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        TD3(3, seed=0, hidden=8).update(batch_size)


@pytest.mark.parametrize("completed_updates", [0, 1, 2, 3])
def test_checkpoint_exact_continuation_with_replay_optimizers_and_local_rng(completed_updates):
    learner = TD3(3, seed=42, hidden=16)
    _fill(learner, 70)
    for _ in range(completed_updates):
        learner.update(16)
    saved = learner.state_dict()
    pristine = deepcopy(saved)
    serialized = BytesIO()
    torch.save(saved, serialized)
    serialized.seek(0)
    decoded = torch.load(serialized, map_location="cpu", weights_only=True)
    restored = TD3(3, seed=999, hidden=16)
    _fill(restored, 2)
    assert restored.load_state_dict(decoded) is None
    _equal(learner.state_dict(), restored.state_dict())
    assert restored.spec()["seed"] == 42
    for _ in range(3):
        expected_metrics = learner.update(16)
        torch.rand(51)
        np.random.random(51)
        actual_metrics = restored.update(16)
        assert expected_metrics == actual_metrics
        _equal(learner.state_dict(), restored.state_dict())
        np.testing.assert_array_equal(learner.act([1, -2, 3]), restored.act([1, -2, 3]))
    _equal(saved, pristine)
    _equal(decoded, pristine)
    # Both checkpoint export and import must own their arrays/tensors.
    saved["actor"]["4.bias"].fill_(99)
    decoded["replay"]["observations"].fill_(99)
    _equal(learner.state_dict(), restored.state_dict())


def test_empty_checkpoint_and_capacity_wrap_roundtrip():
    learner, restored = TD3(2, 1, hidden=8), TD3(2, 2, hidden=8)
    restored.load_state_dict(learner.state_dict())
    _equal(learner.state_dict(), restored.state_dict())
    for index in range(10003):
        learner.add([index, 1], [0, 0], -index, [index + 1, 1], False)
    restored.load_state_dict(learner.state_dict())
    for candidate in (learner, restored):
        candidate.add([10003, 1], [0.5, -0.5], -10003, [10004, 1], True)
    _equal(learner.state_dict(), restored.state_dict())
    assert learner.update() == restored.update()
    _equal(learner.state_dict(), restored.state_dict())


@pytest.mark.parametrize("change", ["format", "dimension", "hidden", "gamma", "termination", "capacity"])
def test_rejects_incompatible_checkpoints_before_mutation(change):
    learner = TD3(3, 1, hidden=8)
    _fill(learner, 2)
    before = learner.state_dict()
    invalid = deepcopy(before)
    if change == "format":
        invalid["format"] = "other"
    elif change == "dimension":
        invalid["observation_dim"] = 4
    elif change in ("hidden", "gamma"):
        invalid["spec"][change] = 0
    elif change == "termination":
        invalid["replay"]["terminated"] = torch.tensor([0, 1])
    else:
        invalid["replay"]["rewards"] = torch.zeros(10001)
    with pytest.raises(ValueError):
        learner.load_state_dict(invalid)
    _equal(before, learner.state_dict())


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
