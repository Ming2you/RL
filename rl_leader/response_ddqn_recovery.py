"""Recovery-aware branch labels for response-aware Double DQN.

This module keeps the existing response-DQN learner intact.  It changes the
data-construction target: a row can now represent the long-horizon value of a
first action after allowing a short recovery tree before falling back to the
native P-Stack tail.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn import TrainedResponseDQN, load_trained_response_dqn
from rl_leader.response_dqn_catalog import (
    StructuredActionCatalog,
    build_structured_action_catalog,
    load_extra_action_specs,
)
from rl_leader.response_dqn_collect import (
    EvaluatedState,
    commit_action,
    evaluate_executable_responses,
)
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from src.controllers.coordination import RL_RESPONSE_CONTRACT_VERSION


RECOVERY_REPLAY_FORMAT = "response_ddqn_recovery_aware_replay_v1"
SNAPSHOT_CACHE_FORMAT = "response_ddqn_policy_snapshot_cache_v1"
RECOVERY_REWARD_MODES = ("return", "advantage")


@dataclass(frozen=True)
class EnvSnapshot:
    """In-process replay point for exact same-state counterfactual branching."""

    env: RLLeaderEnv
    observation: np.ndarray
    label: str
    step_idx: int
    control_step: int
    simulation_time_sec: float
    state_fingerprint: str
    experiment_contract_sha256: str


@dataclass(frozen=True)
class RecoveryBranchConfig:
    """Controls the size of the recovery tree and the P-Stack tail."""

    max_rollout_steps: int = 12
    recovery_depth: int = 1
    top_k: int = 3
    include_anchor_recovery: bool = True
    response_workers: int = 1
    response_backend: str = "serial"
    branch_workers: int = 1

    def validate(self) -> None:
        if self.max_rollout_steps < 1:
            raise ValueError("max_rollout_steps must be positive")
        if self.recovery_depth < 0:
            raise ValueError("recovery_depth cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.response_workers < 1:
            raise ValueError("response_workers must be positive")
        if self.branch_workers < 1:
            raise ValueError("branch_workers must be positive")
        if self.response_backend not in {"serial", "thread", "process"}:
            raise ValueError("response_backend must be serial, thread, or process")
        if (
            self.branch_workers > 1
            and self.response_backend == "process"
            and self.recovery_depth > 0
        ):
            raise ValueError(
                "branch process workers cannot be nested with process response workers"
            )


@dataclass(frozen=True)
class BranchOutcome:
    """One evaluated continuation branch."""

    total_return: float
    total_ttt: float
    steps: int
    terminal_inventory: float
    sequence: tuple[int, ...]
    terminal: bool


@dataclass(frozen=True)
class RecoveryActionLabel:
    """Long-horizon targets for one first action from one frozen state."""

    action_id: int
    valid: bool
    invalid_reason: str
    first_step_reward: float
    first_step_ttt: float
    anchor_tail_return: float
    anchor_tail_ttt: float
    anchor_tail_steps: int
    anchor_tail_inventory: float
    best_recovery_return: float
    best_recovery_ttt: float
    best_recovery_steps: int
    best_recovery_inventory: float
    best_recovery_sequence: tuple[int, ...]

    @property
    def recovery_gain_vs_anchor_tail(self) -> float:
        return float(self.best_recovery_return - self.anchor_tail_return)


@dataclass(frozen=True)
class RecoveryStateEvaluation:
    """All labels needed to turn one state into terminal DDQN targets."""

    snapshot: EnvSnapshot
    evaluated_state: EvaluatedState
    config: RecoveryBranchConfig
    labels: tuple[RecoveryActionLabel, ...]
    evaluation_seconds: float


def capture_env_snapshot(
    env: RLLeaderEnv,
    observation: np.ndarray | None = None,
    *,
    label: str = "",
) -> EnvSnapshot:
    """Capture an exact in-process simulator/controller state."""
    obs = (
        np.asarray(env._observe(), dtype=np.float32)
        if observation is None
        else np.asarray(observation, dtype=np.float32)
    )
    return EnvSnapshot(
        env=copy.deepcopy(env),
        observation=obs.copy(),
        label=str(label),
        step_idx=int(env.step_idx),
        control_step=int(env.step_idx - env.warmup),
        simulation_time_sec=float(env.sim.state.time_sec),
        state_fingerprint=str(env._anchor_context_state_fingerprint()),
        experiment_contract_sha256=str(env.experiment_contract_fingerprint),
    )


def restore_env_snapshot(snapshot: EnvSnapshot) -> tuple[RLLeaderEnv, np.ndarray]:
    """Return a fresh mutable environment copy and its saved observation."""
    env = copy.deepcopy(snapshot.env)
    observation = np.asarray(snapshot.observation, dtype=np.float32).copy()
    actual = str(env._anchor_context_state_fingerprint())
    if actual != snapshot.state_fingerprint:
        raise RuntimeError(
            "restored snapshot fingerprint mismatch: "
            f"expected {snapshot.state_fingerprint}, got {actual}"
        )
    return env, observation


def read_policy_trace(
    path: str | Path,
    *,
    catalog: StructuredActionCatalog | None = None,
    action_id_remap: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Read a policy prefix, resolving stable keys or explicit legacy-ID maps."""
    result: dict[int, int] = {}
    remap = {
        int(source): int(target)
        for source, target in (action_id_remap or {}).items()
    }
    key_to_id = (
        {str(action.key): int(action.action_id) for action in catalog.actions}
        if catalog is not None
        else {}
    )
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "control_step" not in row or "selected_action_id" not in row:
                continue
            source_action_id = int(row["selected_action_id"])
            action_key = row.get("selected_action_key")
            if catalog is not None and action_key is not None:
                if str(action_key) not in key_to_id:
                    raise ValueError(
                        f"policy trace action key is absent from target catalog: "
                        f"{action_key}"
                    )
                action_id = key_to_id[str(action_key)]
            else:
                source_fingerprint = row.get("catalog_fingerprint")
                if (
                    catalog is not None
                    and source_action_id != 0
                    and source_fingerprint is not None
                    and str(source_fingerprint) != str(catalog.fingerprint)
                    and source_action_id not in remap
                ):
                    raise ValueError(
                        "policy trace catalog differs from the target catalog; "
                        "provide stable action keys or --policy-action-remap"
                    )
                action_id = remap.get(source_action_id, source_action_id)
            if catalog is not None and not 0 <= action_id < catalog.size:
                raise ValueError(
                    f"policy trace action_id {source_action_id} resolves outside "
                    f"target catalog size {catalog.size}; provide "
                    "--policy-action-remap"
                )
            result[int(row["control_step"])] = int(action_id)
    return result


def _parse_action_id_remap(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in str(value).split(","):
        if not item:
            continue
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                "policy action remap entries must use SOURCE_ID:TARGET_ID"
            )
        source, target = map(int, parts)
        if source in result and result[source] != target:
            raise ValueError(f"conflicting remap for action {source}")
        result[source] = target
    return result


def select_problem_steps(
    policy_rows: Sequence[dict],
    pstack_rows: Sequence[dict] = (),
    *,
    explicit_steps: Iterable[int] = (),
    max_steps: int = 0,
) -> list[int]:
    """Choose replay states where recovery-aware labels are most informative."""
    explicit = sorted({int(step) for step in explicit_steps})
    if explicit:
        return explicit[:max_steps] if max_steps > 0 else explicit

    non_anchor = sorted({
        int(row["control_step"])
        for row in policy_rows
        if int(row.get("selected_action_id", 0)) != 0
    })
    if non_anchor:
        return non_anchor[:max_steps] if max_steps > 0 else non_anchor

    pstack_by_step = {int(row["control_step"]): row for row in pstack_rows}
    scored = []
    for row in policy_rows:
        step = int(row.get("control_step", -1))
        if step not in pstack_by_step:
            continue
        policy_ttt = _row_interval_ttt(row)
        pstack_ttt = _row_interval_ttt(pstack_by_step[step])
        scored.append((policy_ttt - pstack_ttt, step))
    scored.sort(reverse=True)
    steps = [step for _, step in scored if step >= 0]
    if max_steps > 0:
        steps = steps[:max_steps]
    return sorted(set(steps))


def replay_prefix_to_step(
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    target_control_step: int,
    policy_prefix: dict[int, int] | None = None,
    *,
    response_workers: int = 1,
    response_backend: str = "serial",
) -> tuple[EnvSnapshot, np.ndarray]:
    """Replay a policy-action prefix and capture the requested pre-action state."""
    snapshots = capture_policy_snapshots(
        env,
        catalog,
        (int(target_control_step),),
        policy_prefix,
        response_workers=response_workers,
        response_backend=response_backend,
    )
    snapshot = snapshots[int(target_control_step)]
    return snapshot, snapshot.observation.copy()


def capture_policy_snapshots(
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    target_control_steps: Iterable[int],
    policy_prefix: dict[int, int] | None = None,
    *,
    response_workers: int = 1,
    response_backend: str = "serial",
) -> dict[int, EnvSnapshot]:
    """Replay one policy prefix pass and capture each requested pre-action state."""
    del response_workers, response_backend
    targets = sorted({int(step) for step in target_control_steps})
    if not targets:
        raise ValueError("at least one target control step is required")
    if targets[0] < 0:
        raise ValueError("target control steps cannot be negative")
    prefix = dict(policy_prefix or {})
    observation = np.asarray(env.reset(), dtype=np.float32)
    snapshots: dict[int, EnvSnapshot] = {}
    max_target = targets[-1]
    for control_step in range(max_target + 1):
        if control_step in targets:
            snapshots[control_step] = capture_env_snapshot(
                env,
                observation,
                label=f"policy_step:{control_step}",
            )
            if control_step == max_target:
                break
        action_id = int(prefix.get(control_step, 0))
        observation, _, done, _ = commit_catalog_action_from_anchor(
            env,
            catalog,
            action_id,
        )
        if done and control_step < max_target:
            raise RuntimeError(
                f"simulation ended before target control step {max_target}"
            )
    missing = sorted(set(targets) - set(snapshots))
    if missing:
        raise RuntimeError(f"failed to capture target steps: {missing}")
    return snapshots


def _snapshot_cache_key(
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    target_control_step: int,
    policy_prefix: dict[int, int] | None,
) -> str:
    """Stable cache key for the pre-action state at one policy step."""
    target = int(target_control_step)
    prefix = {
        int(step): int(action_id)
        for step, action_id in (policy_prefix or {}).items()
        if int(step) < target
    }
    uses_catalog = any(action_id != 0 for action_id in prefix.values())
    payload = {
        "format_version": SNAPSHOT_CACHE_FORMAT,
        "scenario": str(env.scenario_name),
        "t_total": float(getattr(env, "T_total", 0.0)),
        "warmup": int(getattr(env, "warmup", 0)),
        "target_control_step": target,
        "experiment_contract_sha256": str(env.experiment_contract_fingerprint),
        "catalog_fingerprint": (
            str(getattr(catalog, "fingerprint", "")) if uses_catalog else "anchor-only"
        ),
        "policy_prefix": sorted(prefix.items()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_cache_path(cache_dir: Path, step: int, key: str) -> Path:
    return cache_dir / f"step_{int(step):04d}_{key[:16]}.pkl"


def _validate_cached_snapshot(
    snapshot: EnvSnapshot,
    env: RLLeaderEnv,
    target_control_step: int,
) -> None:
    if not isinstance(snapshot, EnvSnapshot):
        raise ValueError("cached object is not an EnvSnapshot")
    if int(snapshot.control_step) != int(target_control_step):
        raise ValueError(
            "cached snapshot control step mismatch: "
            f"expected {target_control_step}, got {snapshot.control_step}"
        )
    if str(snapshot.experiment_contract_sha256) != str(
        env.experiment_contract_fingerprint
    ):
        raise ValueError("cached snapshot experiment contract mismatch")
    if str(snapshot.env.scenario_name) != str(env.scenario_name):
        raise ValueError("cached snapshot scenario mismatch")
    if float(getattr(snapshot.env, "T_total", 0.0)) != float(
        getattr(env, "T_total", 0.0)
    ):
        raise ValueError("cached snapshot T_total mismatch")


def capture_policy_snapshots_cached(
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    target_control_steps: Iterable[int],
    policy_prefix: dict[int, int] | None = None,
    *,
    cache_dir: Path | None = None,
    response_workers: int = 1,
    response_backend: str = "serial",
) -> dict[int, EnvSnapshot]:
    """Capture policy snapshots with an optional on-disk cache."""
    if cache_dir is None:
        return capture_policy_snapshots(
            env,
            catalog,
            target_control_steps,
            policy_prefix,
            response_workers=response_workers,
            response_backend=response_backend,
        )

    targets = sorted({int(step) for step in target_control_steps})
    if not targets:
        raise ValueError("at least one target control step is required")
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[int, EnvSnapshot] = {}
    misses: list[int] = []
    keys = {
        step: _snapshot_cache_key(env, catalog, step, policy_prefix)
        for step in targets
    }
    for step in targets:
        path = _snapshot_cache_path(cache_dir, step, keys[step])
        if not path.exists():
            misses.append(step)
            continue
        try:
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
            _validate_cached_snapshot(snapshot, env, step)
        except (OSError, pickle.PickleError, EOFError, ValueError, AttributeError):
            misses.append(step)
            continue
        snapshots[step] = snapshot

    if misses:
        captured = capture_policy_snapshots(
            env,
            catalog,
            misses,
            policy_prefix,
            response_workers=response_workers,
            response_backend=response_backend,
        )
        for step, snapshot in captured.items():
            key = keys[int(step)]
            path = _snapshot_cache_path(cache_dir, int(step), key)
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("wb") as handle:
                pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
            snapshots[int(step)] = snapshot
    missing = sorted(set(targets) - set(snapshots))
    if missing:
        raise RuntimeError(f"failed to capture target steps: {missing}")
    return {step: snapshots[step] for step in targets}


def commit_catalog_action_from_anchor(
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    action_id: int,
):
    """Commit one catalog action using only its same-state P-Stack anchor."""
    action_id = int(action_id)
    if not 0 <= action_id < catalog.size:
        raise IndexError(f"action_id out of range: {action_id}")
    anchor_context = env.prepare_pstack_anchor_context()
    if action_id == 0:
        next_obs, reward, done, info, _ = env.step_prepared_optimizer_anchor(
            anchor_context,
            sync_follower_state=True,
        )
    else:
        next_obs, reward, done, info = env.step_anchored_candidate(
            catalog.residual(action_id),
            anchor_context,
        )
    return np.asarray(next_obs, dtype=np.float32), float(reward), bool(done), dict(info)


def evaluate_recovery_state(
    snapshot: EnvSnapshot,
    catalog: StructuredActionCatalog,
    config: RecoveryBranchConfig,
    *,
    first_action_ids: Iterable[int] | None = None,
    ensemble: Sequence[TrainedResponseDQN] = (),
) -> RecoveryStateEvaluation:
    """Evaluate first-action labels from one frozen state."""
    config.validate()
    started = time.perf_counter()
    env, observation = restore_env_snapshot(snapshot)
    preview_action_ids = None
    if first_action_ids is not None:
        preview_action_ids = sorted({0, *(int(value) for value in first_action_ids)})
    evaluated = evaluate_executable_responses(
        env,
        observation,
        catalog,
        action_ids=preview_action_ids,
        workers=config.response_workers,
        backend=config.response_backend,
    )
    if first_action_ids is None:
        action_ids = [int(value) for value in np.flatnonzero(
            evaluated.response_mask.valid_action_mask
        )]
    else:
        action_ids = sorted({0, *(int(value) for value in first_action_ids)})
    labels = _evaluate_first_actions(
        env,
        evaluated,
        catalog,
        action_ids,
        config,
        ensemble=ensemble,
    )
    return RecoveryStateEvaluation(
        snapshot=snapshot,
        evaluated_state=evaluated,
        config=config,
        labels=labels,
        evaluation_seconds=float(time.perf_counter() - started),
    )


def recovery_state_to_replay(
    state: RecoveryStateEvaluation,
    catalog: StructuredActionCatalog,
    *,
    source: str,
    reward_mode: str = "return",
) -> FrozenResponseReplay:
    """Convert recovery labels to terminal Monte-Carlo DDQN regression rows."""
    labels = [label for label in state.labels if label.valid]
    if not labels:
        raise ValueError("cannot build replay without valid recovery labels")
    if not any(label.action_id == 0 for label in labels):
        raise ValueError("recovery replay requires an anchor action label")
    if reward_mode not in RECOVERY_REWARD_MODES:
        raise ValueError(
            f"reward_mode must be one of {', '.join(RECOVERY_REWARD_MODES)}"
        )

    n = len(labels)
    observation = np.repeat(
        state.evaluated_state.observation[None, :], n, axis=0,
    ).astype(np.float32)
    response_features = np.repeat(
        state.evaluated_state.response_features[None, :, :], n, axis=0,
    ).astype(np.float32)
    action_mask = np.repeat(
        state.evaluated_state.response_mask.valid_action_mask[None, :], n, axis=0,
    ).astype(bool)
    next_action_mask = np.zeros_like(action_mask)
    next_action_mask[:, 0] = True
    next_features = np.zeros_like(response_features)
    action_id = np.asarray([label.action_id for label in labels], dtype=np.int64)
    anchor_label = next(label for label in labels if label.action_id == 0)
    if reward_mode == "advantage":
        anchor_return = float(anchor_label.best_recovery_return)
        reward_values = [
            float(label.best_recovery_return - anchor_return)
            for label in labels
        ]
        reward_semantics = "terminal_recovery_advantage_vs_action0_return"
    else:
        reward_values = [label.best_recovery_return for label in labels]
        reward_semantics = "terminal_best_recovery_return_negative_ttt"
    reward = np.asarray(reward_values, dtype=np.float32)
    option_steps = np.asarray(
        [max(1, label.best_recovery_steps) for label in labels], dtype=np.int64,
    )
    event_group = np.asarray([
        (
            f"{state.snapshot.experiment_contract_sha256}:"
            f"{state.snapshot.control_step}:"
            f"{state.snapshot.state_fingerprint[:12]}"
        )
        for _ in labels
    ])
    manifest = make_replay_manifest(
        transition_count=n,
        action_count=catalog.size,
        catalog_fingerprint=catalog.fingerprint,
        observation_schema=state.snapshot.env.observation_schema.metadata(),
        response_contract=RL_RESPONSE_CONTRACT_VERSION,
        scenario=state.snapshot.env.scenario_name,
        source=source,
    )
    manifest.update({
        "collector_format": RECOVERY_REPLAY_FORMAT,
        "response_evaluation_mode": "recovery_aware_counterfactual_tree",
        "reward_semantics": reward_semantics,
        "reward_mode": reward_mode,
        "done_semantics": "recovery_labels_are_terminal_mc_targets",
        "catalog": catalog.as_manifest(),
        "recovery_state": _snapshot_metadata(state.snapshot),
        "recovery_config": asdict(state.config),
        "label_count": n,
        "label_action_ids": action_id.astype(int).tolist(),
        "anchor_tail_return_by_action": {
            str(label.action_id): float(label.anchor_tail_return) for label in labels
        },
        "best_recovery_sequence_by_action": {
            str(label.action_id): list(map(int, label.best_recovery_sequence))
            for label in labels
        },
    })
    return FrozenResponseReplay(
        observation=observation,
        action_id=action_id,
        reward=reward,
        next_observation=observation.copy(),
        done=np.ones(n, dtype=np.float32),
        option_steps=option_steps,
        action_mask=action_mask,
        next_action_mask=next_action_mask,
        response_features=response_features,
        next_response_features=next_features,
        event_group=event_group.astype(str),
        episode=np.zeros(n, dtype=np.int64),
        control_step=np.full(n, state.snapshot.control_step, dtype=np.int64),
        manifest=manifest,
    ).validate()


def write_recovery_log(
    path: str | Path,
    state: RecoveryStateEvaluation,
    catalog: StructuredActionCatalog,
) -> None:
    """Append one JSON line per evaluated first action."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for label in state.labels:
            action = catalog.action(label.action_id)
            handle.write(json.dumps({
                "format_version": RECOVERY_REPLAY_FORMAT,
                "snapshot": _snapshot_metadata(state.snapshot),
                "action": action.as_dict(),
                "valid": bool(label.valid),
                "invalid_reason": label.invalid_reason,
                "first_step_ttt": label.first_step_ttt,
                "anchor_tail_ttt": label.anchor_tail_ttt,
                "best_recovery_ttt": label.best_recovery_ttt,
                "recovery_gain_vs_anchor_tail": (
                    label.recovery_gain_vs_anchor_tail
                ),
                "best_recovery_sequence": list(label.best_recovery_sequence),
                "best_recovery_steps": label.best_recovery_steps,
                "best_recovery_inventory": label.best_recovery_inventory,
                "state_evaluation_seconds": state.evaluation_seconds,
            }, sort_keys=True) + "\n")


def _evaluate_first_action(
    env: RLLeaderEnv,
    evaluated: EvaluatedState,
    catalog: StructuredActionCatalog,
    action_id: int,
    config: RecoveryBranchConfig,
    *,
    ensemble: Sequence[TrainedResponseDQN],
) -> RecoveryActionLabel:
    action_id = int(action_id)
    if not 0 <= action_id < catalog.size:
        raise IndexError(f"action_id out of range: {action_id}")
    if not evaluated.response_mask.valid_action_mask[action_id]:
        return RecoveryActionLabel(
            action_id=action_id,
            valid=False,
            invalid_reason=evaluated.response_mask.invalid_reasons[action_id],
            first_step_reward=0.0,
            first_step_ttt=0.0,
            anchor_tail_return=float("-inf"),
            anchor_tail_ttt=float("inf"),
            anchor_tail_steps=0,
            anchor_tail_inventory=float("nan"),
            best_recovery_return=float("-inf"),
            best_recovery_ttt=float("inf"),
            best_recovery_steps=0,
            best_recovery_inventory=float("nan"),
            best_recovery_sequence=(),
        )
    first_env, first_eval = copy.deepcopy((env, evaluated))
    next_observation, reward, done, info = commit_action(
        first_env,
        first_eval,
        catalog,
        action_id,
    )
    first_ttt = -float(reward)
    remaining = max(0, int(config.max_rollout_steps) - 1)
    anchor_tail = (
        BranchOutcome(
            total_return=0.0,
            total_ttt=0.0,
            steps=0,
            terminal_inventory=float(info.get("inventory_after", first_env._inventory())),
            sequence=(),
            terminal=bool(done),
        )
        if done or remaining == 0 else
        rollout_pstack_tail(copy.deepcopy(first_env), max_steps=remaining)
    )
    anchor_total_return = float(reward + anchor_tail.total_return)
    anchor_total_ttt = float(first_ttt + anchor_tail.total_ttt)

    if done or remaining == 0 or config.recovery_depth == 0:
        best = anchor_tail
    else:
        best = _best_recovery_continuation(
            first_env,
            np.asarray(next_observation, dtype=np.float32),
            catalog,
            remaining_steps=remaining,
            recovery_depth=config.recovery_depth,
            config=config,
            ensemble=ensemble,
        )
    best_total_return = float(reward + best.total_return)
    best_total_ttt = float(first_ttt + best.total_ttt)
    return RecoveryActionLabel(
        action_id=action_id,
        valid=True,
        invalid_reason="",
        first_step_reward=float(reward),
        first_step_ttt=first_ttt,
        anchor_tail_return=anchor_total_return,
        anchor_tail_ttt=anchor_total_ttt,
        anchor_tail_steps=1 + anchor_tail.steps,
        anchor_tail_inventory=anchor_tail.terminal_inventory,
        best_recovery_return=best_total_return,
        best_recovery_ttt=best_total_ttt,
        best_recovery_steps=1 + best.steps,
        best_recovery_inventory=best.terminal_inventory,
        best_recovery_sequence=(action_id, *best.sequence),
    )


def _evaluate_first_actions(
    env: RLLeaderEnv,
    evaluated: EvaluatedState,
    catalog: StructuredActionCatalog,
    action_ids: Sequence[int],
    config: RecoveryBranchConfig,
    *,
    ensemble: Sequence[TrainedResponseDQN],
) -> tuple[RecoveryActionLabel, ...]:
    """Evaluate first-action labels, optionally in independent processes."""
    ordered = tuple(int(action_id) for action_id in action_ids)
    if config.branch_workers <= 1 or len(ordered) <= 1:
        return tuple(
            _evaluate_first_action(
                env,
                evaluated,
                catalog,
                action_id,
                config,
                ensemble=ensemble,
            )
            for action_id in ordered
        )
    workers = min(int(config.branch_workers), len(ordered))
    payloads = [
        (env, evaluated, catalog, action_id, config, tuple(ensemble))
        for action_id in ordered
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        labels = list(executor.map(_evaluate_first_action_worker, payloads))
    return tuple(labels)


def _evaluate_first_action_worker(payload) -> RecoveryActionLabel:
    env, evaluated, catalog, action_id, config, ensemble = payload
    return _evaluate_first_action(
        env,
        evaluated,
        catalog,
        int(action_id),
        config,
        ensemble=ensemble,
    )


def _best_recovery_continuation(
    env: RLLeaderEnv,
    observation: np.ndarray,
    catalog: StructuredActionCatalog,
    *,
    remaining_steps: int,
    recovery_depth: int,
    config: RecoveryBranchConfig,
    ensemble: Sequence[TrainedResponseDQN],
) -> BranchOutcome:
    if remaining_steps <= 0:
        return BranchOutcome(
            total_return=0.0,
            total_ttt=0.0,
            steps=0,
            terminal_inventory=float(env._inventory()),
            sequence=(),
            terminal=False,
        )
    if recovery_depth <= 0:
        return rollout_pstack_tail(env, max_steps=remaining_steps)

    evaluated = evaluate_executable_responses(
        env,
        observation,
        catalog,
        workers=config.response_workers,
        backend=config.response_backend,
    )
    action_ids = rank_recovery_actions(
        evaluated,
        ensemble=ensemble,
        top_k=config.top_k,
        include_anchor=config.include_anchor_recovery,
    )
    best: BranchOutcome | None = None
    for action_id in action_ids:
        branch_env, branch_eval = copy.deepcopy((env, evaluated))
        next_observation, reward, done, info = commit_action(
            branch_env,
            branch_eval,
            catalog,
            int(action_id),
        )
        if done:
            child = BranchOutcome(
                total_return=0.0,
                total_ttt=0.0,
                steps=0,
                terminal_inventory=float(
                    info.get("inventory_after", branch_env._inventory())
                ),
                sequence=(),
                terminal=True,
            )
        else:
            child = _best_recovery_continuation(
                branch_env,
                np.asarray(next_observation, dtype=np.float32),
                catalog,
                remaining_steps=remaining_steps - 1,
                recovery_depth=recovery_depth - 1,
                config=config,
                ensemble=ensemble,
            )
        candidate = BranchOutcome(
            total_return=float(reward + child.total_return),
            total_ttt=float(-reward + child.total_ttt),
            steps=1 + child.steps,
            terminal_inventory=child.terminal_inventory,
            sequence=(int(action_id), *child.sequence),
            terminal=bool(done or child.terminal),
        )
        if best is None or candidate.total_return > best.total_return:
            best = candidate
    if best is None:
        return rollout_pstack_tail(env, max_steps=remaining_steps)
    return best


def rollout_pstack_tail(env: RLLeaderEnv, *, max_steps: int) -> BranchOutcome:
    """Roll P-Stack from the current state for at most max_steps intervals."""
    max_steps = int(max_steps)
    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    total_return = 0.0
    total_ttt = 0.0
    completed = 0
    done = False
    while completed < max_steps and not done and env.step_idx < env.n_steps:
        context = env.prepare_pstack_anchor_context()
        _, reward, done, info, _ = env.step_prepared_optimizer_anchor(
            context,
            sync_follower_state=True,
        )
        total_return += float(reward)
        total_ttt += -float(reward)
        completed += 1
    inventory = float(info.get("inventory_after", env._inventory())) if completed else float(env._inventory())
    return BranchOutcome(
        total_return=float(total_return),
        total_ttt=float(total_ttt),
        steps=int(completed),
        terminal_inventory=inventory,
        sequence=tuple(0 for _ in range(completed)),
        terminal=bool(done or env.step_idx >= env.n_steps),
    )


def rank_recovery_actions(
    evaluated: EvaluatedState,
    *,
    ensemble: Sequence[TrainedResponseDQN] = (),
    top_k: int = 3,
    include_anchor: bool = True,
) -> list[int]:
    """Rank executable actions for the recovery tree."""
    mask = np.asarray(evaluated.response_mask.valid_action_mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("cannot rank an empty action mask")
    if top_k == 0:
        selected = list(map(int, np.flatnonzero(mask)))
    elif ensemble:
        values = np.stack([
            model.q_values(evaluated.observation, evaluated.response_features)[0]
            for model in ensemble
        ])
        scores = values.mean(axis=0)
        valid = np.flatnonzero(mask)
        ordered = valid[np.argsort(scores[valid])[::-1]]
        selected = list(map(int, ordered[:top_k]))
    else:
        selected = list(map(int, np.flatnonzero(mask)[:top_k]))
    if include_anchor and mask[0] and 0 not in selected:
        selected.insert(0, 0)
    if not selected:
        selected = [0]
    return selected


def _snapshot_metadata(snapshot: EnvSnapshot) -> dict:
    return {
        "label": snapshot.label,
        "step_idx": int(snapshot.step_idx),
        "control_step": int(snapshot.control_step),
        "simulation_time_sec": float(snapshot.simulation_time_sec),
        "state_fingerprint": snapshot.state_fingerprint,
        "experiment_contract_sha256": snapshot.experiment_contract_sha256,
    }


def _row_interval_ttt(row: dict) -> float:
    if "interval_ttt" in row:
        return float(row["interval_ttt"])
    if "interval_reward" in row:
        return -float(row["interval_reward"])
    if "step_ttt" in row:
        return float(row["step_ttt"])
    raise KeyError("row does not contain an interval TTT or reward field")


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in str(value).split(",") if item)


def _load_ensemble(model_dir: Path | None, catalog: StructuredActionCatalog):
    if model_dir is None:
        return []
    paths = sorted(model_dir.glob("response_dqn_member_*.pt"))
    if not paths:
        raise SystemExit(f"no response DQN checkpoints in {model_dir}")
    return [
        load_trained_response_dqn(
            path,
            expected_catalog_fingerprint=catalog.fingerprint,
        )
        for path in paths
    ]


def _merge_recovery_states(
    states: Sequence[RecoveryStateEvaluation],
    catalog: StructuredActionCatalog,
    *,
    source: str,
    reward_mode: str = "return",
) -> FrozenResponseReplay:
    replays = [
        recovery_state_to_replay(
            state,
            catalog,
            source=source,
            reward_mode=reward_mode,
        )
        for state in states
    ]
    reference = replays[0]
    arrays = {
        "observation": np.concatenate([replay.observation for replay in replays]),
        "action_id": np.concatenate([replay.action_id for replay in replays]),
        "reward": np.concatenate([replay.reward for replay in replays]),
        "next_observation": np.concatenate([
            replay.next_observation for replay in replays
        ]),
        "done": np.concatenate([replay.done for replay in replays]),
        "option_steps": np.concatenate([replay.option_steps for replay in replays]),
        "action_mask": np.concatenate([replay.action_mask for replay in replays]),
        "next_action_mask": np.concatenate([
            replay.next_action_mask for replay in replays
        ]),
        "response_features": np.concatenate([
            replay.response_features for replay in replays
        ]),
        "next_response_features": np.concatenate([
            replay.next_response_features for replay in replays
        ]),
        "event_group": np.concatenate([replay.event_group for replay in replays]),
        "episode": np.concatenate([replay.episode for replay in replays]),
        "control_step": np.concatenate([replay.control_step for replay in replays]),
    }
    manifest = dict(reference.manifest)
    manifest.update({
        "transition_count": int(arrays["observation"].shape[0]),
        "source": source,
        "recovery_state_count": len(states),
        "recovery_states": [
            _snapshot_metadata(state.snapshot) for state in states
        ],
    })
    return FrozenResponseReplay(
        **arrays,
        manifest=manifest,
    ).validate()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--policy-trace", type=Path)
    parser.add_argument(
        "--policy-action-remap",
        default="",
        help=(
            "comma-separated SOURCE_ID:TARGET_ID mappings for legacy traces "
            "that do not contain stable action keys"
        ),
    )
    parser.add_argument("--pstack-trace", type=Path)
    parser.add_argument("--policy-steps", default="")
    parser.add_argument("--max-problem-steps", type=int, default=5)
    parser.add_argument("--first-actions", default="valid")
    parser.add_argument("--magnitudes", default="0.25")
    parser.add_argument("--families", default="linear,quadratic,cross")
    parser.add_argument("--domains", default="urban,freeway")
    parser.add_argument("--owners", default="R_F_E")
    parser.add_argument(
        "--extra-actions",
        action="append",
        default=[],
        type=Path,
        help="manual or artifact-derived residual action manifest to append",
    )
    parser.add_argument("--recovery-depth", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-rollout-steps", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--branch-workers", type=int, default=1)
    parser.add_argument(
        "--backend", choices=("serial", "thread", "process"), default="serial",
    )
    parser.add_argument("--snapshot-cache-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--reward-mode",
        choices=RECOVERY_REWARD_MODES,
        default="return",
        help=(
            "return stores negative tail TTT targets; advantage stores "
            "anchor_tail_TTT - action_tail_TTT so anchor is approximately zero"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if args.log is not None and args.log.exists():
        raise SystemExit(f"refusing to overwrite existing log: {args.log}")

    env = RLLeaderEnv(
        scenario_name=args.scenario,
        T_total=args.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    catalog = build_structured_action_catalog(
        env.action_schema.names,
        magnitudes=_parse_csv(args.magnitudes, float),
        families=_parse_csv(args.families),
        domains=_parse_csv(args.domains),
        owners=_parse_csv(args.owners) or None,
        extra_actions=load_extra_action_specs(args.extra_actions),
    )
    policy_prefix = (
        read_policy_trace(
            args.policy_trace,
            catalog=catalog,
            action_id_remap=_parse_action_id_remap(args.policy_action_remap),
        )
        if args.policy_trace
        else {}
    )
    explicit_steps = _parse_csv(args.policy_steps, int)
    if explicit_steps:
        steps = list(explicit_steps)
    elif args.policy_trace:
        policy_rows = _read_jsonl(args.policy_trace)
        pstack_rows = _read_jsonl(args.pstack_trace) if args.pstack_trace else []
        steps = select_problem_steps(
            policy_rows,
            pstack_rows,
            max_steps=args.max_problem_steps,
        )
    else:
        raise SystemExit("provide --policy-steps or --policy-trace")
    config = RecoveryBranchConfig(
        max_rollout_steps=args.max_rollout_steps,
        recovery_depth=args.recovery_depth,
        top_k=args.top_k,
        response_workers=args.workers,
        response_backend=args.backend,
        branch_workers=args.branch_workers,
    )
    config.validate()
    ensemble = _load_ensemble(args.model_dir, catalog)
    first_actions = None
    if args.first_actions != "valid":
        first_actions = _parse_csv(args.first_actions, int)

    started = time.perf_counter()
    env_at_steps = RLLeaderEnv(
        scenario_name=args.scenario,
        T_total=args.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    snapshots = capture_policy_snapshots_cached(
        env_at_steps,
        catalog,
        steps,
        policy_prefix,
        cache_dir=args.snapshot_cache_dir,
        response_workers=args.workers,
        response_backend=args.backend,
    )
    states = []
    for step in steps:
        state = evaluate_recovery_state(
            snapshots[int(step)],
            catalog,
            config,
            first_action_ids=first_actions,
            ensemble=ensemble,
        )
        states.append(state)
        if args.log is not None:
            write_recovery_log(args.log, state, catalog)
        print(json.dumps({
            "event": "recovery_state_completed",
            "control_step": int(step),
            "labels": len(state.labels),
            "valid_labels": sum(label.valid for label in state.labels),
            "seconds": state.evaluation_seconds,
        }, sort_keys=True), flush=True)
    replay = _merge_recovery_states(
        states,
        catalog,
        source="rl_leader.response_ddqn_recovery",
        reward_mode=args.reward_mode,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    replay.save(args.out)
    print(json.dumps({
        "format_version": RECOVERY_REPLAY_FORMAT,
        "output": str(args.out),
        "log": None if args.log is None else str(args.log),
        "scenario": args.scenario,
        "policy_steps": list(map(int, steps)),
        "snapshot_cache_dir": (
            None if args.snapshot_cache_dir is None else str(args.snapshot_cache_dir)
        ),
        "reward_mode": args.reward_mode,
        "transitions": replay.size,
        "action_support_counts": replay.action_support_counts().tolist(),
        "wall_seconds": time.perf_counter() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
