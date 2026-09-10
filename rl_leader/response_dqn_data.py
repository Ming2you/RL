"""Frozen sequential replay contract for response-aware Double DQN."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPLAY_FORMAT = "response_aware_sequential_replay_v1"


@dataclass(frozen=True)
class FrozenResponseReplay:
    observation: np.ndarray
    action_id: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    done: np.ndarray
    option_steps: np.ndarray
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    response_features: np.ndarray
    next_response_features: np.ndarray
    event_group: np.ndarray
    episode: np.ndarray
    control_step: np.ndarray
    manifest: dict

    @property
    def size(self) -> int:
        return int(self.observation.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observation.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.action_mask.shape[1])

    @property
    def response_feature_dim(self) -> int:
        return int(self.response_features.shape[2])

    def validate(self) -> "FrozenResponseReplay":
        n = self.size
        if self.observation.ndim != 2 or n <= 0:
            raise ValueError("replay observation must be a non-empty rank-2 array")
        if self.next_observation.shape != self.observation.shape:
            raise ValueError("next observation shape does not match observation")
        if self.action_mask.ndim != 2 or self.action_mask.shape[0] != n:
            raise ValueError("action mask must have shape [N, A]")
        if self.next_action_mask.shape != self.action_mask.shape:
            raise ValueError("next action mask shape does not match current mask")
        expected_response_shape = (n, self.action_count, self.response_feature_dim)
        if self.response_features.shape != expected_response_shape:
            raise ValueError("response feature shape does not match [N, A, R]")
        if self.next_response_features.shape != expected_response_shape:
            raise ValueError("next response feature shape does not match current features")
        vectors = {
            "action_id": self.action_id,
            "reward": self.reward,
            "done": self.done,
            "option_steps": self.option_steps,
            "event_group": self.event_group,
            "episode": self.episode,
            "control_step": self.control_step,
        }
        for name, values in vectors.items():
            if np.asarray(values).shape != (n,):
                raise ValueError(f"{name} must contain one value per transition")
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("observation contains nonfinite values")
        if not np.all(np.isfinite(self.next_observation)):
            raise ValueError("next observation contains nonfinite values")
        if not np.all(np.isfinite(self.response_features)):
            raise ValueError("response features contain nonfinite values")
        if not np.all(np.isfinite(self.next_response_features)):
            raise ValueError("next response features contain nonfinite values")
        if not np.all(np.isfinite(self.reward)):
            raise ValueError("reward contains nonfinite values")

        action_ids = np.asarray(self.action_id, dtype=np.int64).reshape(-1)
        if np.any(action_ids < 0) or np.any(action_ids >= self.action_count):
            raise ValueError("selected action ID is outside the catalog")
        current_mask = np.asarray(self.action_mask, dtype=bool)
        next_mask = np.asarray(self.next_action_mask, dtype=bool)
        if not np.all(current_mask[:, 0]) or not np.all(next_mask[:, 0]):
            raise ValueError("P-Stack action 0 must always be valid")
        if np.any(np.sum(current_mask, axis=1) == 0) or np.any(np.sum(next_mask, axis=1) == 0):
            raise ValueError("every transition must expose at least one valid action")
        if not np.all(current_mask[np.arange(n), action_ids]):
            raise ValueError("replay contains an action that was masked at collection time")
        done = np.asarray(self.done, dtype=np.float32).reshape(-1)
        if np.any((done != 0.0) & (done != 1.0)):
            raise ValueError("done must be binary")
        if np.any(np.asarray(self.option_steps, dtype=np.int64).reshape(-1) < 1):
            raise ValueError("option_steps must be positive")
        if any(str(value) == "" for value in np.asarray(self.event_group).reshape(-1)):
            raise ValueError("event_group values must be non-empty")

        if self.manifest.get("format_version") != REPLAY_FORMAT:
            raise ValueError("replay manifest has the wrong format version")
        if int(self.manifest.get("transition_count", -1)) != n:
            raise ValueError("replay manifest transition count mismatch")
        if int(self.manifest.get("action_count", -1)) != self.action_count:
            raise ValueError("replay manifest action count mismatch")
        if not self.manifest.get("catalog_fingerprint"):
            raise ValueError("replay manifest is missing the action catalog fingerprint")
        if self.manifest.get("collection_phase") != "simulator_data_construction":
            raise ValueError("replay was not produced by an explicit data-construction phase")
        return self

    def action_support_counts(self) -> np.ndarray:
        return np.bincount(
            np.asarray(self.action_id, dtype=np.int64).reshape(-1),
            minlength=self.action_count,
        ).astype(np.int64)

    def unique_event_groups(self) -> np.ndarray:
        return np.unique(np.asarray(self.event_group).astype(str))

    def bootstrap_indices(self, rng: np.random.Generator) -> np.ndarray:
        """Bootstrap whole event groups so correlated rows remain together."""
        groups = self.unique_event_groups()
        sampled = rng.choice(groups, size=groups.size, replace=True)
        by_group = {
            group: np.flatnonzero(np.asarray(self.event_group).astype(str) == group)
            for group in groups
        }
        return np.concatenate([by_group[str(group)] for group in sampled])

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observation=np.asarray(self.observation, dtype=np.float32),
            action_id=np.asarray(self.action_id, dtype=np.int64),
            reward=np.asarray(self.reward, dtype=np.float32),
            next_observation=np.asarray(self.next_observation, dtype=np.float32),
            done=np.asarray(self.done, dtype=np.float32),
            option_steps=np.asarray(self.option_steps, dtype=np.int64),
            action_mask=np.asarray(self.action_mask, dtype=bool),
            next_action_mask=np.asarray(self.next_action_mask, dtype=bool),
            response_features=np.asarray(self.response_features, dtype=np.float32),
            next_response_features=np.asarray(self.next_response_features, dtype=np.float32),
            event_group=np.asarray(self.event_group).astype(str),
            episode=np.asarray(self.episode, dtype=np.int64),
            control_step=np.asarray(self.control_step, dtype=np.int64),
            manifest_json=np.asarray(json.dumps(
                self.manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )),
        )


def load_frozen_response_replay(path: str | Path) -> FrozenResponseReplay:
    """Load only frozen arrays; this module intentionally has no simulator import."""
    with np.load(Path(path), allow_pickle=False) as data:
        required = {
            "observation", "action_id", "reward", "next_observation", "done",
            "option_steps", "action_mask", "next_action_mask",
            "response_features", "next_response_features", "event_group",
            "episode", "control_step", "manifest_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"response replay is missing fields: {missing}")
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        replay = FrozenResponseReplay(
            observation=np.asarray(data["observation"], dtype=np.float32),
            action_id=np.asarray(data["action_id"], dtype=np.int64),
            reward=np.asarray(data["reward"], dtype=np.float32),
            next_observation=np.asarray(data["next_observation"], dtype=np.float32),
            done=np.asarray(data["done"], dtype=np.float32),
            option_steps=np.asarray(data["option_steps"], dtype=np.int64),
            action_mask=np.asarray(data["action_mask"], dtype=bool),
            next_action_mask=np.asarray(data["next_action_mask"], dtype=bool),
            response_features=np.asarray(data["response_features"], dtype=np.float32),
            next_response_features=np.asarray(data["next_response_features"], dtype=np.float32),
            event_group=np.asarray(data["event_group"]).astype(str),
            episode=np.asarray(data["episode"], dtype=np.int64),
            control_step=np.asarray(data["control_step"], dtype=np.int64),
            manifest=manifest,
        )
    return replay.validate()


def make_replay_manifest(
    *,
    transition_count: int,
    action_count: int,
    catalog_fingerprint: str,
    observation_schema: dict,
    response_contract: str,
    scenario: str,
    source: str,
) -> dict:
    return {
        "format_version": REPLAY_FORMAT,
        "collection_phase": "simulator_data_construction",
        "training_phase_simulator_interaction": False,
        "transition_count": int(transition_count),
        "action_count": int(action_count),
        "catalog_fingerprint": str(catalog_fingerprint),
        "observation_schema": dict(observation_schema),
        "response_contract": str(response_contract),
        "scenario": str(scenario),
        "source": str(source),
    }


def _effective_replay_semantics(manifest: dict) -> dict[str, str]:
    defaults = {
        "reward_semantics": "interval_negative_ttt",
        "done_semantics": "environment_terminal",
    }
    missing = [field for field in defaults if not manifest.get(field)]
    if missing:
        # Only unlabelled legacy interval data can inherit the raw defaults.
        label_clues = any(
            manifest.get(field) is not None
            for field in (
                "mc_gamma", "recovery_config", "reward_mode",
                "advantage_format_version", "anchor_tail_return_by_action",
                "best_recovery_sequence_by_action", "recovery_state_count",
            )
        ) or any(
            manifest.get(field) not in (None, "", default)
            for field, default in defaults.items()
        ) or any(
            marker in str(manifest.get(field, "")).lower()
            for field in ("collector_format", "response_evaluation_mode", "source")
            for marker in ("recovery", "monte_carlo", "terminal", "advantage", "replay_to_mc")
        )
        if label_clues:
            raise ValueError(
                f"cannot merge replay with missing {', '.join(missing)} "
                "when the manifest indicates terminal or transformed labels"
            )
    return {field: manifest.get(field) or default for field, default in defaults.items()}


def _replay_environment_metadata(manifest: dict) -> dict:
    states = [manifest, manifest.get("recovery_state") or {}]
    states.extend(manifest.get("recovery_states") or [])
    if any(not isinstance(state, dict) for state in states):
        raise ValueError("cannot merge replay with invalid recovery state metadata")
    metadata = {}
    for field in ("experiment_contract_sha256", "t_total_sec"):
        known = [state[field] for state in states if state.get(field) is not None]
        if known:
            if any(value != known[0] for value in known[1:]):
                raise ValueError(f"cannot merge replay with conflicting {field} metadata")
            metadata[field] = known[0]
    return metadata


def merge_frozen_response_replays(
    replays: list[FrozenResponseReplay],
    *,
    source: str,
) -> FrozenResponseReplay:
    if not replays:
        raise ValueError("at least one frozen replay is required")
    for replay in replays:
        replay.validate()
    semantics = [_effective_replay_semantics(replay.manifest) for replay in replays]
    environment_metadata = [_replay_environment_metadata(replay.manifest) for replay in replays]
    reference = replays[0]
    contract_fields = (
        "catalog_fingerprint", "action_count", "observation_schema",
        "response_contract", "scenario",
        "mc_gamma", "reward_mode",
    )
    recovery_configs = [replay.manifest.get("recovery_config") for replay in replays]
    if any(config is not None and not isinstance(config, dict) for config in recovery_configs):
        raise ValueError("cannot merge replay with invalid recovery_config")
    for index, replay in enumerate(replays[1:], start=1):
        if replay.manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1") != reference.manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1"):
            raise ValueError("cannot merge replay with different response_equivalence_mode")
        for field, value in semantics[0].items():
            if semantics[index][field] != value:
                raise ValueError(f"cannot merge replay with different {field}")
        for field in ("experiment_contract_sha256", "t_total_sec"):
            if environment_metadata[index].get(field) != environment_metadata[0].get(field):
                raise ValueError(f"cannot merge replay with different {field}")
        for field in contract_fields:
            # Known-vs-missing objectives/contracts are not provably compatible.
            if replay.manifest.get(field) != reference.manifest.get(field):
                raise ValueError(f"cannot merge replay with different {field}")
        for field in ("max_rollout_steps", "recovery_depth"):
            if (recovery_configs[index] or {}).get(field) != (recovery_configs[0] or {}).get(field):
                raise ValueError(f"cannot merge replay with different recovery_config.{field}")
        if replay.observation_dim != reference.observation_dim:
            raise ValueError("cannot merge replay observation dimensions")
        if replay.response_feature_dim != reference.response_feature_dim:
            raise ValueError("cannot merge replay response dimensions")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(replay, name)) for replay in replays])

    manifest = dict(reference.manifest)
    manifest.update(semantics[0])
    manifest.update(environment_metadata[0])
    manifest.update({
        "transition_count": int(sum(replay.size for replay in replays)),
        "source": str(source),
        "merged_batch_count": len(replays),
        "merged_sources": [str(replay.manifest.get("source", "")) for replay in replays],
    })
    return FrozenResponseReplay(
        observation=concatenate("observation").astype(np.float32),
        action_id=concatenate("action_id").astype(np.int64),
        reward=concatenate("reward").astype(np.float32),
        next_observation=concatenate("next_observation").astype(np.float32),
        done=concatenate("done").astype(np.float32),
        option_steps=concatenate("option_steps").astype(np.int64),
        action_mask=concatenate("action_mask").astype(bool),
        next_action_mask=concatenate("next_action_mask").astype(bool),
        response_features=concatenate("response_features").astype(np.float32),
        next_response_features=concatenate("next_response_features").astype(np.float32),
        event_group=concatenate("event_group").astype(str),
        episode=concatenate("episode").astype(np.int64),
        control_step=concatenate("control_step").astype(np.int64),
        manifest=manifest,
    ).validate()
