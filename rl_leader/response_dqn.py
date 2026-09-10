"""Parametric masked Double DQN for executable follower responses."""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_leader.response_dqn_data import FrozenResponseReplay
from rl_leader.response_dqn_mask import LEGACY_EQUIVALENCE


MODEL_FORMAT = "response_aware_parametric_double_dqn_v1"
VALUE_PARAMETERIZATIONS = ("free_q", "finite_horizon_cost_v1")
PHASE_GRID_ATOL = 1.0e-5


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if q_values.shape != mask.shape:
        raise ValueError("Q values and action mask must have the same shape")
    mask = mask.bool()
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("masked argmax received a row with no valid action")
    floor = torch.finfo(q_values.dtype).min
    return q_values.masked_fill(~mask, floor).argmax(dim=1)


def masked_cql_penalty(
    q_values: torch.Tensor, action_ids: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Discrete CQL(H) logsumexp-minus-behavior-Q over executable responses."""
    if q_values.ndim != 2 or valid_mask.shape != q_values.shape:
        raise ValueError("CQL Q values and mask must have shape [B,A]")
    if action_ids.shape != (q_values.shape[0],):
        raise ValueError("CQL needs one behavior action per row")
    valid = valid_mask.bool()
    if not torch.all(valid.any(dim=1)) or not torch.isfinite(q_values[valid]).all():
        raise ValueError("CQL requires nonempty finite valid Q rows")
    if torch.any(action_ids < 0) or torch.any(action_ids >= q_values.shape[1]):
        raise ValueError("CQL behavior action is outside the catalog")
    if not torch.all(valid.gather(1, action_ids.unsqueeze(1))):
        raise ValueError("CQL behavior action must be valid")
    behavior_q = q_values.gather(1, action_ids.unsqueeze(1)).squeeze(1)
    return (torch.logsumexp(q_values.masked_fill(~valid, -torch.inf), dim=1) - behavior_q).mean()


@dataclass(frozen=True)
class ResponseDQNConfig:
    gamma: float = 0.99
    learning_rate: float = 3.0e-4
    batch_size: int = 128
    gradient_steps: int = 20_000
    target_update_interval: int = 250
    soft_target_tau: float = 0.0
    gradient_clip: float = 10.0
    reward_scale: float = 0.01
    hidden: tuple[int, ...] = (256, 256)
    ensemble_size: int = 5
    min_action_support: int = 1
    conservative_alpha: float = 0.0
    terminal_batch_fraction: float = 0.0
    mask_constant_features: bool = False
    value_parameterization: str = "free_q"
    backup_horizon: int = 1

    def validate(self) -> None:
        if type(self.backup_horizon) is not int or self.backup_horizon < 1:
            raise ValueError("backup_horizon must be a positive integer")
        if self.backup_horizon > 1 and self.gamma != 1.0:
            raise ValueError("multi-step backup requires gamma=1")
        if self.value_parameterization not in VALUE_PARAMETERIZATIONS:
            raise ValueError("unknown value parameterization")
        if self.value_parameterization == "finite_horizon_cost_v1" and self.gamma != 1.0:
            raise ValueError("finite-horizon cost requires gamma=1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0.0 or self.batch_size <= 0:
            raise ValueError("learning rate and batch size must be positive")
        if self.gradient_steps < 0 or self.target_update_interval <= 0:
            raise ValueError("invalid training-step configuration")
        if not 0.0 <= self.soft_target_tau <= 1.0:
            raise ValueError("soft_target_tau must be in [0, 1]")
        if self.gradient_clip <= 0.0 or self.reward_scale <= 0.0:
            raise ValueError("gradient clip and reward scale must be positive")
        if not self.hidden or min(self.hidden) <= 0:
            raise ValueError("hidden layer widths must be positive")
        if self.ensemble_size <= 0 or self.min_action_support < 0:
            raise ValueError("invalid ensemble or support configuration")
        if not np.isfinite(self.conservative_alpha) or self.conservative_alpha < 0:
            raise ValueError("conservative alpha must be finite and nonnegative")
        if not np.isfinite(self.terminal_batch_fraction) or not 0 <= self.terminal_batch_fraction < 1:
            raise ValueError("terminal batch fraction must be finite and in [0, 1)")


def sample_training_indices(rng, train_indices, done, batch_size, terminal_fraction):
    """Optional terminal-stratified replay, without relabeling nonterminal rows."""
    size = min(int(batch_size), train_indices.size)
    if terminal_fraction == 0:
        return rng.choice(train_indices, size=size, replace=train_indices.size < batch_size)
    terminal = train_indices[done[train_indices] == 1]
    continuing = train_indices[done[train_indices] == 0]
    if size < 2 or not terminal.size or not continuing.size:
        raise ValueError("terminal-stratified replay needs two batch slots and both strata")
    terminal_count = min(max(int(round(size * terminal_fraction)), 1), size - 1)
    # This deliberately reweights the fitting distribution; it is not unbiased PER.
    return np.concatenate((
        rng.choice(terminal, size=terminal_count, replace=terminal.size < terminal_count),
        rng.choice(continuing, size=size - terminal_count, replace=continuing.size < size - terminal_count),
    ))


@dataclass(frozen=True)
class FeatureNormalizer:
    observation_mean: np.ndarray
    observation_scale: np.ndarray
    response_mean: np.ndarray
    response_scale: np.ndarray
    observation_active_mask: np.ndarray | None = None
    response_active_mask: np.ndarray | None = None

    @classmethod
    def fit(
        cls, replay: FrozenResponseReplay, indices: np.ndarray, *,
        mask_constant_features: bool = False,
    ) -> "FeatureNormalizer":
        observations = np.concatenate((
            replay.observation[indices], replay.next_observation[indices],
        ), axis=0).astype(np.float64)
        current_response = replay.response_features[indices][replay.action_mask[indices]]
        next_response = replay.next_response_features[indices][replay.next_action_mask[indices]]
        responses = np.concatenate((current_response, next_response), axis=0).astype(np.float64)
        obs_mean = observations.mean(axis=0)
        obs_scale = observations.std(axis=0)
        response_mean = responses.mean(axis=0)
        response_scale = responses.std(axis=0)
        # Exact zero variance only, before the legacy small-scale fallback.
        observation_active_mask = obs_scale != 0.0 if mask_constant_features else None
        response_active_mask = response_scale != 0.0 if mask_constant_features else None
        obs_scale[obs_scale < 1.0e-6] = 1.0
        response_scale[response_scale < 1.0e-6] = 1.0
        return cls(
            observation_mean=obs_mean.astype(np.float32),
            observation_scale=obs_scale.astype(np.float32),
            response_mean=response_mean.astype(np.float32),
            response_scale=response_scale.astype(np.float32),
            observation_active_mask=observation_active_mask,
            response_active_mask=response_active_mask,
        )

    def normalize_observation(self, values: np.ndarray) -> np.ndarray:
        normalized = (values - self.observation_mean) / self.observation_scale
        if self.observation_active_mask is not None:
            normalized[..., ~self.observation_active_mask] = 0.0
        return normalized

    def normalize_response(self, values: np.ndarray) -> np.ndarray:
        normalized = (values - self.response_mean) / self.response_scale
        if self.response_active_mask is not None:
            normalized[..., ~self.response_active_mask] = 0.0
        return normalized

    def as_dict(self) -> dict:
        payload = {
            "observation_mean": self.observation_mean.tolist(),
            "observation_scale": self.observation_scale.tolist(),
            "response_mean": self.response_mean.tolist(),
            "response_scale": self.response_scale.tolist(),
        }
        if self.observation_active_mask is not None:
            payload["observation_active_mask"] = self.observation_active_mask.tolist()
        if self.response_active_mask is not None:
            payload["response_active_mask"] = self.response_active_mask.tolist()
        return payload


@dataclass(frozen=True)
class FiniteHorizonCostContract:
    phase_index: int
    horizon_intervals: int

    def validate(self, observation_dim: int) -> None:
        if type(self.phase_index) is not int or not 0 <= self.phase_index < observation_dim:
            raise ValueError("invalid value-contract phase index")
        if type(self.horizon_intervals) is not int or self.horizon_intervals < 1:
            raise ValueError("invalid value-contract horizon intervals")

    def remaining_intervals(self, raw_observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_observation)
        if raw.ndim not in (1, 2):
            raise ValueError("raw observation must be [O] or [B,O]")
        self.validate(raw.shape[-1])
        phase = raw[..., self.phase_index].astype(np.float64)
        if not np.all(np.isfinite(phase)) or np.any((phase < 0) | (phase > 1)):
            raise ValueError("raw time.phase must be finite and in [0, 1]")
        ticks = phase * self.horizon_intervals
        nearest = np.rint(ticks)
        if np.any(np.abs(ticks - nearest) > PHASE_GRID_ATOL):
            raise ValueError("raw time.phase is off the full-horizon interval grid")
        # Snap float32 phase roundoff to the validated grid, not to a new horizon.
        return (self.horizon_intervals - nearest).astype(np.float32)

    @classmethod
    def from_replay(cls, replay: FrozenResponseReplay) -> "FiniteHorizonCostContract":
        if replay.manifest.get("reward_semantics") != "interval_negative_ttt":
            raise ValueError("finite-horizon cost requires interval_negative_ttt rewards")
        if replay.manifest.get("done_semantics") != "environment_terminal":
            raise ValueError("finite-horizon cost requires environment_terminal done semantics")
        if not np.all(np.isfinite(replay.reward)) or np.any(replay.reward >= 0):
            raise ValueError("finite-horizon cost requires strictly negative interval rewards")
        if not np.all(replay.option_steps == 1):
            raise ValueError("finite-horizon cost requires one-step interval transitions")
        names = replay.manifest.get("observation_schema", {}).get("names", [])
        if len(names) != replay.observation_dim or names.count("time.phase") != 1:
            raise ValueError("observation schema must name exactly one time.phase column")
        phase_index = names.index("time.phase")
        phase = replay.observation[:, phase_index].astype(np.float64)
        next_phase = replay.next_observation[:, phase_index].astype(np.float64)
        increments = next_phase - phase
        if not np.all(np.isfinite(increments)) or np.any(increments <= 0):
            raise ValueError("time.phase must advance by one positive uniform interval")
        horizon = int(round(1.0 / float(np.median(increments))))
        contract = cls(phase_index, horizon)
        remaining = contract.remaining_intervals(replay.observation)
        next_remaining = contract.remaining_intervals(replay.next_observation)
        if (np.any(np.abs(increments * horizon - 1) > 2 * PHASE_GRID_ATOL)
                or np.any(remaining < 1) or np.any(remaining - next_remaining != 1)):
            raise ValueError("time.phase increments must be uniform one-step intervals")
        if not np.array_equal(replay.done, (next_remaining == 0).astype(np.float32)):
            raise ValueError("done must agree with next remaining intervals being zero")
        return contract

    def as_dict(self) -> dict:
        return {
            "format_version": "finite_horizon_cost_v1",
            "phase_name": "time.phase", "phase_index": self.phase_index,
            "horizon_intervals": self.horizon_intervals,
            "phase_grid_atol_intervals": PHASE_GRID_ATOL,
            "remaining_includes_current_interval": True, "end_phase": 1.0,
            "reward_semantics": "interval_negative_ttt",
            "done_semantics": "environment_terminal", "gamma": 1.0,
        }

    @classmethod
    def from_dict(cls, payload: dict, observation_dim: int) -> "FiniteHorizonCostContract":
        if not isinstance(payload, dict):
            raise ValueError("cost checkpoint requires an explicit value contract")
        contract = cls(payload.get("phase_index"), payload.get("horizon_intervals"))
        contract.validate(observation_dim)
        if payload != contract.as_dict():
            raise ValueError("inconsistent finite-horizon value contract")
        return contract


class CandidateQNetwork(nn.Module):
    """Score Q(o, structured action, realized follower response)."""

    def __init__(
        self,
        observation_dim: int,
        candidate_feature_dim: int,
        response_feature_dim: int,
        hidden: Sequence[int],
        *,
        value_parameterization: str = "free_q",
    ) -> None:
        super().__init__()
        if value_parameterization not in VALUE_PARAMETERIZATIONS:
            raise ValueError("unknown value parameterization")
        self.value_parameterization = value_parameterization
        input_dim = int(observation_dim + candidate_feature_dim + response_feature_dim)
        layers: list[nn.Module] = []
        width = input_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, int(next_width)), nn.ReLU()))
            width = int(next_width)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observation: torch.Tensor,
        candidate_features: torch.Tensor,
        response_features: torch.Tensor,
        *,
        remaining_intervals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observation.ndim != 2 or response_features.ndim != 3:
            raise ValueError("observation must be [B,O] and response features [B,A,R]")
        batch, actions, _ = response_features.shape
        if candidate_features.ndim == 2:
            if candidate_features.shape[0] != actions:
                raise ValueError("candidate and response action dimensions differ")
            candidates = candidate_features.unsqueeze(0).expand(batch, -1, -1)
        elif candidate_features.ndim == 3:
            candidates = candidate_features
        else:
            raise ValueError("candidate features must be [A,C] or [B,A,C]")
        if candidates.shape[:2] != (batch, actions):
            raise ValueError("candidate feature batch shape is invalid")
        context = observation.unsqueeze(1).expand(-1, actions, -1)
        values = self.network(torch.cat((context, candidates, response_features), dim=2))
        values = values.squeeze(2)
        if self.value_parameterization == "free_q":
            return values
        if remaining_intervals is None or remaining_intervals.shape != (batch,):
            raise ValueError("cost head requires raw-phase remaining intervals [B]")
        if not torch.isfinite(remaining_intervals).all() or torch.any(remaining_intervals < 0):
            raise ValueError("remaining intervals must be finite and nonnegative")
        remaining = remaining_intervals.unsqueeze(1)
        cost_q = -remaining * F.softplus(values)
        return torch.where(remaining == 0, torch.zeros_like(cost_q), cost_q)


@dataclass
class TrainedResponseDQN:
    online: CandidateQNetwork
    target: CandidateQNetwork
    normalizer: FeatureNormalizer
    candidate_features: np.ndarray
    action_support_counts: np.ndarray
    catalog_fingerprint: str
    config: ResponseDQNConfig
    seed: int
    training_losses: list[float]
    response_equivalence_mode: str = LEGACY_EQUIVALENCE
    value_contract: FiniteHorizonCostContract | None = None

    def q_values(
        self,
        observation: np.ndarray,
        response_features: np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        dev = torch.device(device)
        self.online.to(dev).eval()
        raw_observation = np.asarray(observation, dtype=np.float32)
        obs = self.normalizer.normalize_observation(raw_observation)
        response = self.normalizer.normalize_response(np.asarray(response_features, dtype=np.float32))
        if obs.ndim == 1:
            obs = obs[None, :]
        if response.ndim == 2:
            response = response[None, :, :]
        remaining = None
        if self.config.value_parameterization == "finite_horizon_cost_v1":
            if self.value_contract is None:
                raise ValueError("cost model is missing its value contract")
            remaining = torch.as_tensor(
                self.value_contract.remaining_intervals(raw_observation).reshape(-1),
                dtype=torch.float32, device=dev,
            )
        with torch.no_grad():
            values = self.online(
                torch.as_tensor(obs, dtype=torch.float32, device=dev),
                torch.as_tensor(self.candidate_features, dtype=torch.float32, device=dev),
                torch.as_tensor(response, dtype=torch.float32, device=dev),
                remaining_intervals=remaining,
            )
        return values.cpu().numpy()

    def checkpoint(self) -> dict:
        payload = {
            "format_version": MODEL_FORMAT,
            "catalog_fingerprint": self.catalog_fingerprint,
            "seed": int(self.seed),
            "config": asdict(self.config),
            "candidate_features": np.asarray(self.candidate_features, dtype=np.float32),
            "action_support_counts": np.asarray(self.action_support_counts, dtype=np.int64),
            "normalizer": self.normalizer.as_dict(),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "training_losses": list(map(float, self.training_losses)),
            "response_equivalence_mode": self.response_equivalence_mode,
        }
        if self.value_contract is not None:
            payload["value_contract"] = self.value_contract.as_dict()
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)


def load_trained_response_dqn(
    path: str | Path,
    *,
    expected_catalog_fingerprint: str | None = None,
    device: str | torch.device = "cpu",
) -> TrainedResponseDQN:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != MODEL_FORMAT:
        raise ValueError("checkpoint is not a response-aware DQN model")
    if (
        expected_catalog_fingerprint is not None
        and payload.get("catalog_fingerprint") != expected_catalog_fingerprint
    ):
        raise ValueError("checkpoint action catalog does not match the runtime")
    config = ResponseDQNConfig(**payload["config"])
    normalizer_payload = payload["normalizer"]
    normalizer = FeatureNormalizer(**{
        name: np.asarray(
            value, dtype=bool if name.endswith("_active_mask") else np.float32,
        )
        for name, value in normalizer_payload.items()
    })
    value_contract = None
    if config.value_parameterization != "free_q":
        config.validate()
        value_contract = FiniteHorizonCostContract.from_dict(
            payload.get("value_contract"), normalizer.observation_mean.size,
        )
    elif payload.get("value_contract") is not None:
        raise ValueError("free-Q checkpoint must not contain a cost value contract")
    candidates = np.asarray(payload["candidate_features"], dtype=np.float32)
    online = CandidateQNetwork(
        normalizer.observation_mean.size,
        candidates.shape[1],
        normalizer.response_mean.size,
        config.hidden,
        value_parameterization=config.value_parameterization,
    ).to(device)
    target = copy.deepcopy(online).to(device)
    online.load_state_dict(payload["online_state_dict"])
    target.load_state_dict(payload["target_state_dict"])
    return TrainedResponseDQN(
        online=online,
        target=target,
        normalizer=normalizer,
        candidate_features=candidates,
        action_support_counts=np.asarray(
            payload["action_support_counts"], dtype=np.int64,
        ),
        catalog_fingerprint=str(payload["catalog_fingerprint"]),
        config=config,
        seed=int(payload["seed"]),
        training_losses=list(map(float, payload.get("training_losses", []))),
        response_equivalence_mode=payload.get("response_equivalence_mode", LEGACY_EQUIVALENCE),
        value_contract=value_contract,
    )


def _normalized_tensors(
    replay: FrozenResponseReplay,
    normalizer: FeatureNormalizer,
    device: torch.device,
    value_contract: FiniteHorizonCostContract | None = None,
) -> dict[str, torch.Tensor]:
    tensors = {
        "observation": torch.as_tensor(
            normalizer.normalize_observation(replay.observation),
            dtype=torch.float32, device=device,
        ),
        "next_observation": torch.as_tensor(
            normalizer.normalize_observation(replay.next_observation),
            dtype=torch.float32, device=device,
        ),
        "response": torch.as_tensor(
            normalizer.normalize_response(replay.response_features),
            dtype=torch.float32, device=device,
        ),
        "next_response": torch.as_tensor(
            normalizer.normalize_response(replay.next_response_features),
            dtype=torch.float32, device=device,
        ),
        "action": torch.as_tensor(replay.action_id, dtype=torch.long, device=device),
        "reward": torch.as_tensor(replay.reward, dtype=torch.float32, device=device),
        "done": torch.as_tensor(replay.done, dtype=torch.float32, device=device),
        "option_steps": torch.as_tensor(replay.option_steps, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(replay.action_mask, dtype=torch.bool, device=device),
        "next_mask": torch.as_tensor(replay.next_action_mask, dtype=torch.bool, device=device),
    }
    if value_contract is not None:
        tensors["remaining_intervals"] = torch.as_tensor(
            value_contract.remaining_intervals(replay.observation), device=device,
        )
        tensors["next_remaining_intervals"] = torch.as_tensor(
            value_contract.remaining_intervals(replay.next_observation), device=device,
        )
    return tensors


def build_sequential_links(replay: FrozenResponseReplay) -> np.ndarray:
    """Verify raw interval chains; return successor row indices, or -1 at gaps/ends.

    Links depend on (episode, control_step), never array order or event group.
    A missing row is a replay boundary, not evidence of environment termination.
    """
    replay.validate()
    if replay.manifest.get("reward_semantics") != "interval_negative_ttt":
        raise ValueError("multi-step backup requires interval_negative_ttt rewards")
    if replay.manifest.get("done_semantics") != "environment_terminal":
        raise ValueError("multi-step backup requires environment_terminal done flags")
    if not np.all(replay.option_steps == 1):
        raise ValueError("multi-step backup requires option_steps=1")
    for name in ("episode", "control_step"):
        if np.asarray(getattr(replay, name)).dtype.kind not in "iu":
            raise ValueError(f"sequential links require integer {name} keys")

    by_key = {}
    last_step = {}
    keys = list(zip(map(int, replay.episode), map(int, replay.control_step)))
    for row, key in enumerate(keys):
        if key in by_key:
            raise ValueError(f"duplicate sequential key {key} at rows {by_key[key]} and {row}")
        by_key[key] = row
        episode, step = key
        last_step[episode] = max(step, last_step.get(episode, step))

    links = np.full(replay.size, -1, dtype=np.int64)
    for row, (episode, step) in enumerate(keys):
        if replay.done[row]:
            if last_step[episode] > step:
                raise ValueError(f"terminal row {(episode, step)} is followed by data in its episode")
            continue
        following = by_key.get((episode, step + 1))
        if following is None:
            continue
        for next_name, current_name in (
            ("next_observation", "observation"),
            ("next_response_features", "response_features"),
            ("next_action_mask", "action_mask"),
        ):
            if not np.array_equal(
                getattr(replay, next_name)[row], getattr(replay, current_name)[following],
            ):
                raise ValueError(
                    f"sequential chain mismatch in {next_name} at {(episode, step)} "
                    f"to {(episode, step + 1)}"
                )
        links[row] = following
    return links


@torch.no_grad()
def greedy_consistent_ddqn_targets(
    online: CandidateQNetwork,
    target: CandidateQNetwork,
    tensors: dict[str, torch.Tensor],
    candidate_features: torch.Tensor,
    support_mask: torch.Tensor,
    sequential_links: torch.Tensor,
    indices: torch.Tensor,
    *,
    backup_horizon: int,
    reward_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return capped greedy-path DDQN targets and included transition counts.

    Requires verified links, raw option_steps=1 replay and gamma=1. Each call
    uses this member's current online greedy action (including argmax ties).
    Stop BEFORE a nongreedy successor action, then evaluate the online-selected
    boundary action with the target network. This is a finite Watkins-style
    cutoff, not an uncorrected n-step return or a full lambda/Retrace algorithm.
    """
    if type(backup_horizon) is not int or backup_horizon < 1:
        raise ValueError("backup_horizon must be a positive integer")
    current = indices.clone()
    target_q = reward_scale * tensors["reward"][current]
    horizons = torch.ones_like(indices)
    active = torch.nonzero(tensors["done"][current] == 0, as_tuple=True)[0]
    for depth in range(1, backup_horizon + 1):
        if active.numel() == 0:
            break
        rows = current[active]
        remaining = (
            tensors["next_remaining_intervals"][rows]
            if "next_remaining_intervals" in tensors else None
        )
        next_online = online(
            tensors["next_observation"][rows], candidate_features,
            tensors["next_response"][rows], remaining_intervals=remaining,
        )
        next_action = masked_argmax(
            next_online, tensors["next_mask"][rows] & support_mask.unsqueeze(0),
        )
        following = sequential_links[rows]
        extend = (
            (depth < backup_horizon) & (following >= 0)
            & (tensors["action"][following.clamp(min=0)] == next_action)
        )
        boundary = ~extend
        if boundary.any():
            boundary_rows = rows[boundary]
            next_target = target(
                tensors["next_observation"][boundary_rows], candidate_features,
                tensors["next_response"][boundary_rows],
                remaining_intervals=remaining[boundary] if remaining is not None else None,
            ).gather(1, next_action[boundary].unsqueeze(1)).squeeze(1)
            target_q[active[boundary]] += next_target
        active = active[extend]
        following = following[extend]
        current[active] = following
        target_q[active] += reward_scale * tensors["reward"][following]
        horizons[active] += 1
        active = active[tensors["done"][following] == 0]
    return target_q, horizons


def _check_stop_files(stop_files: Sequence[str | Path]) -> None:
    for path in stop_files:
        if Path(path).exists():
            raise InterruptedError(f"training stopped by {path}")


def train_response_dqn_member(
    replay: FrozenResponseReplay,
    candidate_features: np.ndarray,
    *,
    catalog_fingerprint: str,
    config: ResponseDQNConfig | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    bootstrap: bool = True,
    stop_files: Sequence[str | Path] = (),
) -> TrainedResponseDQN:
    """Train one frozen-batch member; no simulator object is accepted or imported."""
    stop_files = tuple(stop_files)
    _check_stop_files(stop_files)
    replay.validate()
    config = config or ResponseDQNConfig()
    config.validate()
    sequential_links = None
    if config.backup_horizon > 1:
        if bootstrap:
            raise ValueError("multi-step backup requires bootstrap=False (--no-group-bootstrap)")
        sequential_links = build_sequential_links(replay)
    if replay.manifest["catalog_fingerprint"] != str(catalog_fingerprint):
        raise ValueError("replay and requested action catalog do not match")
    candidates = np.asarray(candidate_features, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[0] != replay.action_count:
        raise ValueError("candidate features must have shape [A, C]")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate features contain nonfinite values")
    value_contract = (
        FiniteHorizonCostContract.from_replay(replay)
        if config.value_parameterization == "finite_horizon_cost_v1" else None
    )

    set_seed(seed)
    rng = np.random.default_rng(seed)
    train_indices = replay.bootstrap_indices(rng) if bootstrap else np.arange(replay.size)
    global_support = replay.action_support_counts()
    required_support = global_support >= config.min_action_support
    member_support = np.bincount(
        replay.action_id[train_indices], minlength=replay.action_count,
    )
    additions = []
    for action_id in np.flatnonzero(required_support):
        missing = max(0, config.min_action_support - int(member_support[action_id]))
        if missing:
            available = np.flatnonzero(replay.action_id == action_id)
            additions.extend(rng.choice(available, size=missing, replace=True).tolist())
    if additions:
        train_indices = np.concatenate((
            train_indices, np.asarray(additions, dtype=np.int64),
        ))
    normalizer = FeatureNormalizer.fit(
        replay, train_indices, mask_constant_features=config.mask_constant_features,
    )
    dev = torch.device(device)
    online = CandidateQNetwork(
        replay.observation_dim, candidates.shape[1], replay.response_feature_dim,
        config.hidden,
        value_parameterization=config.value_parameterization,
    ).to(dev)
    target = copy.deepcopy(online).to(dev).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    tensors = _normalized_tensors(replay, normalizer, dev, value_contract)
    if sequential_links is not None:
        sequential_links = torch.as_tensor(sequential_links, dtype=torch.long, device=dev)
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32, device=dev)
    if global_support[0] < config.min_action_support:
        raise ValueError(
            "P-Stack anchor lacks the minimum behavior support required for Delta-Q"
        )
    support_counts = np.bincount(
        replay.action_id[train_indices], minlength=replay.action_count,
    ).astype(np.int64)
    support_mask = torch.as_tensor(
        support_counts >= config.min_action_support,
        dtype=torch.bool,
        device=dev,
    )
    losses: list[float] = []

    for step in range(config.gradient_steps):
        if stop_files and step % 100 == 0:
            _check_stop_files(stop_files)
        batch_indices = sample_training_indices(
            rng, train_indices, replay.done, config.batch_size, config.terminal_batch_fraction,
        )
        idx = torch.as_tensor(batch_indices, dtype=torch.long, device=dev)
        remaining = tensors["remaining_intervals"][idx] if value_contract is not None else None
        next_remaining = tensors["next_remaining_intervals"][idx] if value_contract is not None else None
        q_all = online(
            tensors["observation"][idx], candidate_tensor, tensors["response"][idx],
            remaining_intervals=remaining,
        )
        selected_q = q_all.gather(1, tensors["action"][idx].unsqueeze(1)).squeeze(1)
        if config.backup_horizon == 1:
            with torch.no_grad():
                next_online = online(
                    tensors["next_observation"][idx], candidate_tensor,
                    tensors["next_response"][idx],
                    remaining_intervals=next_remaining,
                )
                supported_next_mask = tensors["next_mask"][idx] & support_mask.unsqueeze(0)
                next_action = masked_argmax(next_online, supported_next_mask)
                next_target = target(
                    tensors["next_observation"][idx], candidate_tensor,
                    tensors["next_response"][idx],
                    remaining_intervals=next_remaining,
                ).gather(1, next_action.unsqueeze(1)).squeeze(1)
                discount = torch.pow(
                    torch.full_like(tensors["option_steps"][idx], config.gamma),
                    tensors["option_steps"][idx],
                )
                target_q = (
                    config.reward_scale * tensors["reward"][idx]
                    + discount * (1.0 - tensors["done"][idx]) * next_target
                )
        else:
            target_q, _ = greedy_consistent_ddqn_targets(
                online, target, tensors, candidate_tensor, support_mask,
                sequential_links, idx,
                backup_horizon=config.backup_horizon, reward_scale=config.reward_scale,
            )
        loss = F.smooth_l1_loss(selected_q, target_q)
        if config.conservative_alpha > 0:
            loss = loss + config.conservative_alpha * masked_cql_penalty(
                q_all, tensors["action"][idx], tensors["mask"][idx],
            )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), config.gradient_clip)
        optimizer.step()
        losses.append(float(loss.item()))

        if config.soft_target_tau > 0.0:
            tau = config.soft_target_tau
            with torch.no_grad():
                for target_param, online_param in zip(target.parameters(), online.parameters()):
                    target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)
        elif (step + 1) % config.target_update_interval == 0:
            target.load_state_dict(online.state_dict())

    _check_stop_files(stop_files)
    return TrainedResponseDQN(
        online=online,
        target=target,
        normalizer=normalizer,
        candidate_features=candidates,
        action_support_counts=support_counts,
        catalog_fingerprint=str(catalog_fingerprint),
        config=config,
        seed=int(seed),
        training_losses=losses,
        response_equivalence_mode=replay.manifest.get("response_equivalence_mode", LEGACY_EQUIVALENCE),
        value_contract=value_contract,
    )


def train_response_dqn_ensemble(
    replay: FrozenResponseReplay,
    candidate_features: np.ndarray,
    *,
    catalog_fingerprint: str,
    config: ResponseDQNConfig | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[TrainedResponseDQN]:
    config = config or ResponseDQNConfig()
    return [
        train_response_dqn_member(
            replay,
            candidate_features,
            catalog_fingerprint=catalog_fingerprint,
            config=config,
            seed=seed + member,
            device=device,
            bootstrap=True,
        )
        for member in range(config.ensemble_size)
    ]


@dataclass(frozen=True)
class ConservativeDecision:
    action_id: int
    fallback: bool
    fallback_reason: str
    anchor_q_mean: float
    selected_q_mean: float
    delta_q_mean: float
    delta_q_std: float
    lcb: float
    valid_action_count: int


def conservative_ensemble_selection(
    ensemble_q_values: np.ndarray,
    valid_action_mask: np.ndarray,
    *,
    action_support_counts: np.ndarray | None = None,
    min_action_support: int = 1,
    anchor_action_id: int = 0,
    z_value: float = 1.96,
    material_margin: float = 0.0,
) -> ConservativeDecision:
    values = np.asarray(ensemble_q_values, dtype=np.float64)
    mask = np.asarray(valid_action_mask, dtype=bool).copy()
    if values.ndim != 2 or mask.shape != (values.shape[1],):
        raise ValueError("ensemble Q values must be [E,A] with a matching mask")
    if not np.all(np.isfinite(values)):
        raise ValueError("ensemble Q values contain nonfinite values")
    if not 0 <= int(anchor_action_id) < values.shape[1] or not mask[int(anchor_action_id)]:
        raise ValueError("anchor action must be valid")
    if action_support_counts is not None:
        counts = np.asarray(action_support_counts, dtype=np.int64)
        if counts.shape != mask.shape:
            raise ValueError("action support count shape mismatch")
        if counts[int(anchor_action_id)] < int(min_action_support):
            return ConservativeDecision(
                action_id=int(anchor_action_id), fallback=True,
                fallback_reason="anchor_q_unsupported",
                anchor_q_mean=float(values[:, int(anchor_action_id)].mean()),
                selected_q_mean=float(values[:, int(anchor_action_id)].mean()),
                delta_q_mean=0.0, delta_q_std=0.0, lcb=0.0,
                valid_action_count=1,
            )
        mask &= counts >= int(min_action_support)
        mask[int(anchor_action_id)] = True
    delta = values - values[:, [int(anchor_action_id)]]
    means = delta.mean(axis=0)
    stds = delta.std(axis=0)
    lcb = means - float(z_value) * stds
    candidate_mask = mask.copy()
    candidate_mask[int(anchor_action_id)] = False
    if not np.any(candidate_mask):
        return ConservativeDecision(
            action_id=int(anchor_action_id), fallback=True,
            fallback_reason="no_supported_non_anchor_response",
            anchor_q_mean=float(values[:, int(anchor_action_id)].mean()),
            selected_q_mean=float(values[:, int(anchor_action_id)].mean()),
            delta_q_mean=0.0, delta_q_std=0.0, lcb=0.0,
            valid_action_count=int(mask.sum()),
        )
    ranked = np.where(candidate_mask, lcb, -np.inf)
    selected = int(np.argmax(ranked))
    if float(ranked[selected]) <= float(material_margin):
        return ConservativeDecision(
            action_id=int(anchor_action_id), fallback=True,
            fallback_reason="lcb_below_material_margin",
            anchor_q_mean=float(values[:, int(anchor_action_id)].mean()),
            selected_q_mean=float(values[:, int(anchor_action_id)].mean()),
            delta_q_mean=0.0, delta_q_std=0.0, lcb=0.0,
            valid_action_count=int(mask.sum()),
        )
    return ConservativeDecision(
        action_id=selected, fallback=False, fallback_reason="",
        anchor_q_mean=float(values[:, int(anchor_action_id)].mean()),
        selected_q_mean=float(values[:, selected].mean()),
        delta_q_mean=float(means[selected]),
        delta_q_std=float(stds[selected]),
        lcb=float(lcb[selected]),
        valid_action_count=int(mask.sum()),
    )


def masked_epsilon_greedy(
    q_values: np.ndarray,
    valid_action_mask: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape or not np.any(mask):
        raise ValueError("epsilon-greedy requires matching Q values and a nonempty mask")
    valid = np.flatnonzero(mask)
    if rng.random() < float(epsilon):
        return int(rng.choice(valid))
    return int(valid[np.argmax(values[valid])])
