"""CPU shared TD3 with balanced scenario replay; no environment dependencies."""

from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCENARIOS = (
    "sweet_155_w",
    "sweet_170_w",
    "sweet_170_incident_w",
    "sweet_170_skew15_w",
    "sweet_190_w",
)


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
    """One actor and twin critics trained on equal samples from every scenario.

    Rewards are already scaled interval rewards and are stored without clipping.
    Only true termination disables bootstrapping. Each scenario's replay contains
    copies of (observation, requested_action, reward, next_observation, terminated).
    Scenario IDs select replay groups only; they are never network inputs.
    """

    FORMAT = "sdmpc-multi-td3-v1"
    gamma = 1.0
    learning_rate = 3e-4
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_delay = 2
    replay_capacity = 2000
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
        self._replay = {scenario: deque(maxlen=self.replay_capacity) for scenario in SCENARIOS}
        self.updates = 0
        self.sample_counts = dict.fromkeys(SCENARIOS, 0)

    def spec(self) -> dict:
        return {
            "algorithm": "TD3", "observation_dim": self.observation_dim,
            "action_dim": self.action_dim, "seed": self.seed, "hidden": self.hidden,
            "hidden_layers": 2, "activation": "relu", "actor_output": "tanh",
            "actor_final_init": "zeros", "action_bounds": [-1.0, 1.0],
            "gamma": self.gamma, "learning_rate": self.learning_rate,
            "tau": self.tau, "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip, "policy_delay": self.policy_delay,
            "replay_capacity": self.replay_capacity, "default_batch_size": 40,
            "scenarios": list(SCENARIOS),
            "replay_sampling": "equal_per_scenario_with_replacement", "optimizer": "Adam",
            "device": "cpu", "dtype": "float32", "torch_num_threads": 1,
        }

    @torch.no_grad()
    def act(self, observation) -> np.ndarray:
        obs = _array(observation, (self.observation_dim,), "observation")
        return self.actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy().copy()

    def add(self, obs, requested_action, reward, next_obs, terminated: bool, scenario):
        if not isinstance(scenario, str) or scenario not in SCENARIOS:
            raise ValueError("scenario must be one of SCENARIOS")
        observation = _array(obs, (self.observation_dim,), "obs")
        action = _array(requested_action, (self.action_dim,), "requested_action", True)
        scalar_reward = float(_array(reward, (), "reward"))
        next_observation = _array(next_obs, (self.observation_dim,), "next_obs")
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("terminated must be a boolean")
        self._replay[scenario].append((
            observation, action, scalar_reward, next_observation, bool(terminated)
        ))

    def replay_counts(self) -> dict:
        return {scenario: len(self._replay[scenario]) for scenario in SCENARIOS}

    def total_transition_count(self) -> int:
        return sum(self.replay_counts().values())

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

    def update(self, batch_size=40) -> dict:
        batch_size = _integer(batch_size, "batch_size")
        if batch_size % len(SCENARIOS):
            raise ValueError("batch_size must be divisible by the number of scenarios (5)")
        per_scenario = batch_size // len(SCENARIOS)
        if any(len(self._replay[scenario]) < per_scenario for scenario in SCENARIOS):
            return {}
        batch = []
        for scenario in SCENARIOS:
            replay = self._replay[scenario]
            indices = self._numpy_rng.integers(len(replay), size=per_scenario)
            batch.extend(replay[int(index)] for index in indices)
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
        sampled_per_scenario = dict.fromkeys(SCENARIOS, per_scenario)
        for scenario in SCENARIOS:
            self.sample_counts[scenario] += per_scenario
        metrics = {
            "critic_loss": float(critic_loss.detach()), "updates": self.updates,
            "sampled_per_scenario": sampled_per_scenario,
        }
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
        replay = {}
        for scenario in SCENARIOS:
            entries = self._replay[scenario]
            obs, actions, rewards, next_obs, terminated = (
                zip(*entries) if entries else ((), (), (), (), ())
            )
            replay[scenario] = {
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
            "sample_counts": self.sample_counts,
            "actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "numpy_rng": self._numpy_rng.bit_generator.state,
            "torch_rng": self._torch_rng.get_state(),
        })

    def _validate_optimizer(self, state, name, network, steps):
        payload = state.get(name)
        if not isinstance(payload, dict) or set(payload) != {"state", "param_groups"}:
            raise ValueError(f"invalid checkpoint {name}")
        parameters = list(getattr(self, network).named_parameters())
        # A fresh Adam supplies the exact settings used by this learner's constructor.
        required = torch.optim.Adam(
            (parameter for _, parameter in parameters), lr=self.learning_rate
        ).state_dict()["param_groups"][0]
        groups = payload["param_groups"]
        if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
            raise ValueError(f"invalid checkpoint {name} parameter groups")
        group = groups[0]
        if set(group) != set(required):
            raise ValueError(f"checkpoint {name} settings do not match")
        for key, expected in required.items():
            actual = group[key]
            if isinstance(expected, (list, tuple)):
                if type(actual) is not type(expected) or len(actual) != len(expected):
                    raise ValueError(f"checkpoint {name} {key} does not match")
                pairs = zip(actual, expected)
            else:
                pairs = ((actual, expected),)
            if any(type(value) is not type(target) or value != target for value, target in pairs):
                raise ValueError(f"checkpoint {name} {key} does not match")
        history = payload["state"]
        expected_ids = set(required["params"]) if steps else set()
        if (not isinstance(history, dict) or set(history) != expected_ids
                or any(type(key) is not int for key in history)):
            raise ValueError(f"checkpoint {name} history does not match scheduled steps")
        saved_network = state.get(network)
        if not isinstance(saved_network, dict):
            raise ValueError(f"invalid checkpoint {network}")
        for parameter_id, (parameter_name, parameter) in zip(required["params"], parameters):
            saved_parameter = saved_network.get(parameter_name)
            if not isinstance(saved_parameter, torch.Tensor) or saved_parameter.shape != parameter.shape:
                raise ValueError(f"invalid checkpoint {network}/{parameter_name}")
            if not steps:
                continue
            entry = history[parameter_id]
            if not isinstance(entry, dict) or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"invalid checkpoint {name} moments")
            step = entry["step"]
            if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                    or step.device.type != "cpu" or step.layout != torch.strided
                    or step.dtype not in (torch.float32, torch.float64)
                    or not torch.isfinite(step).item() or step.item() != steps):
                raise ValueError(f"checkpoint {name} step does not match scheduled steps")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = entry[moment_name]
                if (not isinstance(moment, torch.Tensor) or moment.shape != saved_parameter.shape
                        or moment.dtype != parameter.dtype or moment.device.type != "cpu"
                        or moment.layout != torch.strided or not torch.isfinite(moment).all()
                        or (moment_name == "exp_avg_sq" and (moment < 0).any())):
                    raise ValueError(f"invalid checkpoint {name} {moment_name}")

    def load_state_dict(self, state) -> None:
        if not isinstance(state, dict) or state.get("format") != self.FORMAT:
            raise ValueError("unsupported TD3 checkpoint format")
        if _integer(state.get("observation_dim"), "observation_dim") != self.observation_dim:
            raise ValueError("checkpoint observation_dim does not match")
        checkpoint_spec = state.get("spec")
        if not isinstance(checkpoint_spec, dict):
            raise ValueError("invalid checkpoint TD3 specification")
        seed = _integer(checkpoint_spec.get("seed"), "seed", minimum=0)
        if seed >= 2**64:
            raise ValueError("seed must be smaller than 2**64")
        expected_spec = self.spec()
        expected_spec["seed"] = seed
        if checkpoint_spec != expected_spec:
            raise ValueError("checkpoint TD3 hyperparameters or scenario order do not match")
        updates = _integer(state.get("updates"), "updates", minimum=0)
        sample_counts = state.get("sample_counts")
        if not isinstance(sample_counts, dict) or set(sample_counts) != set(SCENARIOS):
            raise ValueError("checkpoint sample_counts must contain exactly SCENARIOS")
        sample_counts = {
            scenario: _integer(sample_counts[scenario], "sample_counts", minimum=0)
            for scenario in SCENARIOS
        }
        if len(set(sample_counts.values())) != 1:
            raise ValueError("checkpoint sample_counts must be equal across scenarios")
        if not updates <= sample_counts[SCENARIOS[0]] <= updates * self.replay_capacity:
            raise ValueError("checkpoint sample_counts are inconsistent with updates")
        replay_state = state.get("replay")
        if not isinstance(replay_state, dict) or set(replay_state) != set(SCENARIOS):
            raise ValueError("checkpoint replay must contain exactly SCENARIOS")
        restored_replay = {}
        for scenario in SCENARIOS:
            group = replay_state[scenario]
            names = {"observations", "actions", "rewards", "next_observations", "terminated"}
            if not isinstance(group, dict) or set(group) != names:
                raise ValueError(f"invalid checkpoint replay columns for {scenario}")
            rewards = group["rewards"]
            if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
                raise ValueError("invalid checkpoint replay rewards")
            size = len(rewards)
            if size > self.replay_capacity:
                raise ValueError("checkpoint replay exceeds per-scenario capacity")
            columns = []
            for name, shape in (
                ("observations", (size, self.observation_dim)), ("actions", (size, 2)),
                ("rewards", (size,)), ("next_observations", (size, self.observation_dim)),
                ("terminated", (size,)),
            ):
                column = group[name]
                dtype = torch.bool if name == "terminated" else torch.float32
                if (not isinstance(column, torch.Tensor) or tuple(column.shape) != shape
                        or column.dtype != dtype or column.device.type != "cpu"
                        or column.layout != torch.strided):
                    raise ValueError(f"invalid checkpoint replay {scenario}/{name}")
                columns.append(column.detach().numpy())
            entries = deque(maxlen=self.replay_capacity)
            for obs, action, reward, next_obs, terminated in zip(*columns):
                entries.append((
                    _array(obs, (self.observation_dim,), "obs"),
                    _array(action, (2,), "requested_action", True),
                    float(_array(reward, (), "reward")),
                    _array(next_obs, (self.observation_dim,), "next_obs"), bool(terminated),
                ))
            restored_replay[scenario] = entries
        minimum_replay_size = min(len(entries) for entries in restored_replay.values())
        if sample_counts[SCENARIOS[0]] > updates * minimum_replay_size:
            raise ValueError("checkpoint sample_counts exceed possible replay history")
        self._validate_optimizer(state, "critic_optimizer", "critics", updates)
        self._validate_optimizer(state, "actor_optimizer", "actor", updates // self.policy_delay)
        # Validate local RNG payloads before installing any checkpoint state.
        numpy_rng = np.random.default_rng(seed)
        numpy_rng.bit_generator.state = deepcopy(state["numpy_rng"])
        torch_rng = torch.Generator(device="cpu")
        torch_rng.set_state(state["torch_rng"].clone())
        for name in ("actor", "critics", "actor_target", "critic_targets"):
            getattr(self, name).load_state_dict(state[name])
            getattr(self, name).zero_grad(set_to_none=True)
        self.actor_optimizer.load_state_dict(deepcopy(state["actor_optimizer"]))
        self.critic_optimizer.load_state_dict(deepcopy(state["critic_optimizer"]))
        self._numpy_rng = numpy_rng
        self._torch_rng = torch_rng
        self._replay = restored_replay
        self.updates = updates
        self.sample_counts = sample_counts
        self.seed = seed
