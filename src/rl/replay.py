from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class RLTransition:
    """DDQN-ready single-agent transition for leader and follower replay buffers."""

    episode: int
    step: int
    agent_id: str
    agent_family: str
    observation: Mapping[str, Any]
    action_index: int
    reward: float
    next_observation: Mapping[str, Any]
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode": int(self.episode),
            "step": int(self.step),
            "agent_id": str(self.agent_id),
            "agent_family": str(self.agent_family),
            "observation": _jsonable(self.observation),
            "action_index": int(self.action_index),
            "reward": float(self.reward),
            "next_observation": _jsonable(self.next_observation),
            "done": bool(self.done),
            "info": _jsonable(self.info),
        }


class ReplayDatasetWriter:
    """Append-only replay dataset writer for cheap offline RL data collection."""

    index_fields = (
        "episode",
        "step",
        "agent_id",
        "agent_family",
        "action_index",
        "reward",
        "done",
        "observation_dim",
        "next_observation_dim",
    )

    def __init__(
        self,
        output_dir: str | Path,
        metadata: Mapping[str, Any] | None = None,
        transitions_filename: str = "rl_transitions.jsonl",
        index_filename: str = "rl_transition_index.csv",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transitions_path = self.output_dir / transitions_filename
        self.index_path = self.output_dir / index_filename
        self.metadata_path = self.output_dir / "metadata.json"
        self._transition_handle = self.transitions_path.open("w", encoding="utf-8")
        self._index_handle = self.index_path.open("w", newline="", encoding="utf-8")
        self._index_writer = csv.DictWriter(self._index_handle, fieldnames=self.index_fields)
        self._index_writer.writeheader()
        self.count = 0
        self._write_metadata(metadata or {})

    def append(self, transition: RLTransition) -> None:
        payload = transition.as_dict()
        self._transition_handle.write(json.dumps(payload, sort_keys=True))
        self._transition_handle.write("\n")
        self._index_writer.writerow(_transition_index_row(payload))
        self.count += 1

    def append_many(self, transitions) -> None:
        for transition in transitions:
            self.append(transition)
        self.flush()

    def flush(self) -> None:
        self._transition_handle.flush()
        self._index_handle.flush()

    def close(self) -> None:
        self.flush()
        self._transition_handle.close()
        self._index_handle.close()
        self._write_metadata({"transition_count": self.count})

    def paths(self) -> Dict[str, Path]:
        return {
            "transitions": self.transitions_path,
            "index": self.index_path,
            "metadata": self.metadata_path,
        }

    def __enter__(self) -> "ReplayDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _write_metadata(self, updates: Mapping[str, Any]) -> None:
        existing = {}
        if self.metadata_path.exists():
            existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        existing.update(_jsonable(updates))
        existing.update({
            "format": "stackelberg_rl_replay_jsonl_v1",
            "files": {
                "transitions": self.transitions_path.name,
                "index": self.index_path.name,
            },
        })
        self.metadata_path.write_text(
            json.dumps(existing, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def transitions_from_env_step(episode: int, step_result) -> list[RLTransition]:
    """Build leader and follower replay transitions from one environment step."""

    record = step_result.record
    done = bool(step_result.done or step_result.truncated)
    transitions = [
        RLTransition(
            episode=episode,
            step=record.step_index,
            agent_id="leader",
            agent_family="leader",
            observation=observation_to_replay_payload(record.leader_observation),
            action_index=record.leader_action_index,
            reward=step_result.leader_reward,
            next_observation=observation_to_replay_payload(step_result.observation.leader),
            done=done,
            info={
                "leader_action": record.leader_action,
                "reward_terms": record.leader_reward_terms,
                "global_step_ttt": record.global_step_ttt,
                "follower_count": len(record.follower_rewards),
            },
        )
    ]
    for agent_id, observation in sorted(record.follower_observations.items()):
        transitions.append(
            RLTransition(
                episode=episode,
                step=record.step_index,
                agent_id=agent_id,
                agent_family=observation.family,
                observation=observation_to_replay_payload(observation),
                action_index=record.follower_action_indices[agent_id],
                reward=step_result.follower_rewards[agent_id],
                next_observation=observation_to_replay_payload(step_result.observation.followers[agent_id]),
                done=done,
                info={
                    "leader_action": record.leader_action,
                    "physical_action": record.physical_follower_actions.get(agent_id, {}),
                    "reward_terms": record.follower_reward_terms.get(agent_id, {}),
                },
            )
        )
    return transitions


def observation_to_replay_payload(observation) -> Dict[str, Any]:
    return {
        "agent_id": observation.agent_id,
        "family": observation.family,
        "features": [float(value) for value in observation.features.tolist()],
        "feature_names": list(observation.feature_names),
        "owned_links": list(observation.owned_links),
        "owned_ramps": list(observation.owned_ramps),
        "owned_movements": list(observation.owned_movements),
        "owned_signals": list(observation.owned_signals),
        "connected_coupling_links": list(observation.connected_coupling_links),
    }


def _transition_index_row(payload: Mapping[str, Any]) -> Dict[str, Any]:
    observation = payload["observation"]
    next_observation = payload["next_observation"]
    return {
        "episode": payload["episode"],
        "step": payload["step"],
        "agent_id": payload["agent_id"],
        "agent_family": payload["agent_family"],
        "action_index": payload["action_index"],
        "reward": payload["reward"],
        "done": payload["done"],
        "observation_dim": len(observation.get("features", ())),
        "next_observation_dim": len(next_observation.get("features", ())),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
