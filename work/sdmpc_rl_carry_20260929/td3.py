"""CPU TD3 for two requested actions; no environment or runner dependencies."""

from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _array(value, shape, name, bounded=False):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    # Check the original precision so out-of-range actions cannot round to 1.
    if bounded and (np.any(array < -1) or np.any(array > 1)):
        raise ValueError(f"{name} must lie in [-1, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable as finite float32")
    return result


def _mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden, device="cpu", dtype=torch.float32), nn.ReLU(),
        nn.Linear(hidden, hidden, device="cpu", dtype=torch.float32), nn.ReLU(),
        nn.Linear(hidden, outputs, device="cpu", dtype=torch.float32),
    )


class TD3:
    """Bounded replay and complete, independent checkpoints for exact continuation.

    Rewards are already scaled interval rewards and are stored without clipping.
    Only true termination disables bootstrapping. Replay entries contain copies
    of (observation, requested_action, reward, next_observation, terminated).
    """

    FORMAT = "sdmpc-budget-td3-v1"
    gamma = 1.0
    learning_rate = 3e-4
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_delay = 2
    replay_capacity = 10000
    action_dim = 2

    def __init__(self, observation_dim: int, seed: int, hidden: int = 64):
        self.observation_dim = _integer(observation_dim, "observation_dim")
        self.hidden = _integer(hidden, "hidden")
        self.seed = _integer(seed, "seed", minimum=0)
        if self.seed >= 2**64:
            raise ValueError("seed must be smaller than 2**64")
        torch.set_num_threads(1)
        self._numpy_rng = np.random.default_rng(self.seed)
        self._torch_rng = torch.Generator(device="cpu").manual_seed(self.seed)
        # Linear initialization uses the global CPU RNG; restore it afterwards.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._torch_rng.get_state())
            self.actor = nn.Sequential(
                *_mlp(self.observation_dim, self.hidden, self.action_dim), nn.Tanh()
            )
            nn.init.zeros_(self.actor[-2].weight)
            nn.init.zeros_(self.actor[-2].bias)
            self.critics = nn.ModuleList([
                _mlp(self.observation_dim + self.action_dim, self.hidden, 1)
                for _ in range(2)
            ])
            self._torch_rng.set_state(torch.get_rng_state())
        self.actor_target = deepcopy(self.actor).requires_grad_(False)
        self.critic_targets = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=self.learning_rate
        )
        self.replay = deque(maxlen=self.replay_capacity)
        self.updates = 0

    def spec(self) -> dict:
        return {
            "algorithm": "TD3", "observation_dim": self.observation_dim,
            "action_dim": self.action_dim, "seed": self.seed, "hidden": self.hidden,
            "hidden_layers": 2, "activation": "relu", "actor_output": "tanh",
            "actor_final_init": "zeros", "action_bounds": [-1.0, 1.0],
            "gamma": self.gamma, "learning_rate": self.learning_rate,
            "tau": self.tau, "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip, "policy_delay": self.policy_delay,
            "replay_capacity": self.replay_capacity, "default_batch_size": 32,
            "replay_sampling": "uniform_with_replacement", "optimizer": "Adam",
            "device": "cpu", "dtype": "float32", "torch_num_threads": 1,
        }

    @torch.no_grad()
    def act(self, observation) -> np.ndarray:
        obs = _array(observation, (self.observation_dim,), "observation")
        return self.actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy().copy()

    def add(self, obs, requested_action, reward, next_obs, terminated: bool):
        observation = _array(obs, (self.observation_dim,), "obs")
        action = _array(requested_action, (self.action_dim,), "requested_action", True)
        scalar_reward = float(_array(reward, (), "reward"))
        next_observation = _array(next_obs, (self.observation_dim,), "next_obs")
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("terminated must be a boolean")
        self.replay.append((
            observation, action, scalar_reward, next_observation, bool(terminated)
        ))

    @torch.no_grad()
    def _target_values(self, rewards, next_observations, terminated):
        """Inputs have shapes (B, 1), (B, observation_dim), and boolean (B, 1)."""
        actions = self.actor_target(next_observations)
        noise = torch.randn(
            actions.shape, generator=self._torch_rng, device="cpu", dtype=torch.float32
        ).mul(self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (actions + noise).clamp(-1.0, 1.0)
        inputs = torch.cat((next_observations, next_actions), dim=1)
        minimum_q = torch.minimum(*(critic(inputs) for critic in self.critic_targets))
        bootstrap = torch.where(terminated, torch.zeros_like(minimum_q), minimum_q)
        return rewards + self.gamma * bootstrap

    def update(self, batch_size=32) -> dict:
        batch_size = _integer(batch_size, "batch_size")
        if len(self.replay) < batch_size:
            return {}
        indices = self._numpy_rng.integers(len(self.replay), size=batch_size)
        batch = [self.replay[int(index)] for index in indices]
        obs, actions, rewards, next_obs, terminated = zip(*batch)
        observations = torch.from_numpy(np.stack(obs))
        actions = torch.from_numpy(np.stack(actions))
        rewards = torch.tensor(rewards, dtype=torch.float32, device="cpu").unsqueeze(1)
        next_observations = torch.from_numpy(np.stack(next_obs))
        terminated = torch.tensor(terminated, dtype=torch.bool, device="cpu").unsqueeze(1)
        targets = self._target_values(rewards, next_observations, terminated)
        inputs = torch.cat((observations, actions), dim=1)
        critic_loss = sum(F.mse_loss(critic(inputs), targets) for critic in self.critics)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.updates += 1
        metrics = {"critic_loss": float(critic_loss.detach()), "updates": self.updates}
        if self.updates % self.policy_delay == 0:
            self.critics.requires_grad_(False)
            try:
                policy_inputs = torch.cat((observations, self.actor(observations)), dim=1)
                actor_loss = -self.critics[0](policy_inputs).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
            finally:
                self.critics.requires_grad_(True)
            metrics["actor_loss"] = float(actor_loss.detach())
            with torch.no_grad():
                for online, target in (
                    (self.actor, self.actor_target), (self.critics, self.critic_targets)
                ):
                    for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                        target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        return metrics

    def state_dict(self) -> dict:
        """Detached snapshot; replay tensors also permit weights-only torch.load."""
        obs, actions, rewards, next_obs, terminated = (
            zip(*self.replay) if self.replay else ((), (), (), (), ())
        )
        replay = {
            "observations": torch.from_numpy(
                np.asarray(obs, dtype=np.float32).reshape(-1, self.observation_dim)
            ),
            "actions": torch.from_numpy(np.asarray(actions, dtype=np.float32).reshape(-1, 2)),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device="cpu"),
            "next_observations": torch.from_numpy(
                np.asarray(next_obs, dtype=np.float32).reshape(-1, self.observation_dim)
            ),
            "terminated": torch.tensor(terminated, dtype=torch.bool, device="cpu"),
        }
        return deepcopy({
            "format": self.FORMAT, "observation_dim": self.observation_dim,
            "spec": self.spec(), "updates": self.updates, "replay": replay,
            "actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "numpy_rng": self._numpy_rng.bit_generator.state,
            "torch_rng": self._torch_rng.get_state(),
        })

    def load_state_dict(self, state) -> None:
        if state["format"] != self.FORMAT:
            raise ValueError("unsupported TD3 checkpoint format")
        if state["observation_dim"] != self.observation_dim:
            raise ValueError("checkpoint observation_dim does not match")
        expected_spec = self.spec()
        expected_spec["seed"] = state["spec"]["seed"]
        if state["spec"] != expected_spec:
            raise ValueError("checkpoint TD3 hyperparameters do not match")
        updates = _integer(state["updates"], "updates", minimum=0)
        seed = _integer(state["spec"]["seed"], "seed", minimum=0)
        replay_state = state["replay"]
        columns = []
        size = len(replay_state["rewards"])
        if size > self.replay_capacity:
            raise ValueError("checkpoint replay exceeds capacity")
        for name, shape in (
            ("observations", (size, self.observation_dim)), ("actions", (size, 2)),
            ("rewards", (size,)), ("next_observations", (size, self.observation_dim)),
            ("terminated", (size,)),
        ):
            column = replay_state[name]
            if not isinstance(column, torch.Tensor) or tuple(column.shape) != shape:
                raise ValueError(f"invalid checkpoint replay {name}")
            columns.append(column.detach().cpu().numpy())
        restored_replay = deque(maxlen=self.replay_capacity)
        for obs, action, reward, next_obs, terminated in zip(*columns):
            if not isinstance(terminated, np.bool_):
                raise ValueError("checkpoint terminated must be boolean")
            restored_replay.append((
                _array(obs, (self.observation_dim,), "obs"),
                _array(action, (2,), "requested_action", True),
                float(_array(reward, (), "reward")),
                _array(next_obs, (self.observation_dim,), "next_obs"), bool(terminated),
            ))
        for name in ("actor", "critics", "actor_target", "critic_targets"):
            getattr(self, name).load_state_dict(state[name])
            getattr(self, name).zero_grad(set_to_none=True)
        self.actor_optimizer.load_state_dict(deepcopy(state["actor_optimizer"]))
        self.critic_optimizer.load_state_dict(deepcopy(state["critic_optimizer"]))
        self._numpy_rng.bit_generator.state = deepcopy(state["numpy_rng"])
        self._torch_rng.set_state(state["torch_rng"].clone())
        self.replay = restored_replay
        self.updates = updates
        self.seed = seed
