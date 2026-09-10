"""Oracle-response sequential data construction for response-aware DQN."""
from __future__ import annotations

import argparse
import atexit
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn import masked_epsilon_greedy
from rl_leader.response_dqn import (
    TrainedResponseDQN,
    conservative_ensemble_selection,
    load_trained_response_dqn,
)
from rl_leader.response_dqn_catalog import (
    StructuredActionCatalog,
    build_structured_action_catalog,
    load_extra_action_specs,
)
from rl_leader.response_dqn_data import (
    FrozenResponseReplay,
    make_replay_manifest,
)
from rl_leader.response_dqn_mask import (
    CandidateResponse,
    ResponseMask,
    build_response_mask,
    response_feature_matrix,
    LEGACY_EQUIVALENCE, CONTINUATION_EQUIVALENCE,
)
from src.controllers.coordination import RL_RESPONSE_CONTRACT_VERSION


COLLECTOR_FORMAT = "oracle_response_sequential_collector_v1"
_RESPONSE_PROCESS_POOLS: dict[int, ProcessPoolExecutor] = {}


def _shutdown_response_process_pools(*, wait: bool = False) -> None:
    for executor in list(_RESPONSE_PROCESS_POOLS.values()):
        executor.shutdown(wait=wait, cancel_futures=True)
    _RESPONSE_PROCESS_POOLS.clear()


def _response_process_pool(workers: int) -> ProcessPoolExecutor:
    executor = _RESPONSE_PROCESS_POOLS.get(workers)
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=workers)
        _RESPONSE_PROCESS_POOLS[workers] = executor
    return executor


atexit.register(_shutdown_response_process_pools)


@dataclass(frozen=True)
class EvaluatedState:
    observation: np.ndarray
    anchor_context: object
    candidate_responses: tuple[CandidateResponse, ...]
    response_mask: ResponseMask
    response_features: np.ndarray
    raw_candidate_count: int
    unique_response_count: int
    evaluation_seconds: float
    response_workers: int
    response_backend: str
    response_equivalence_mode: str = LEGACY_EQUIVALENCE


@dataclass(frozen=True)
class CollectionDecision:
    action_id: int
    diagnostics: dict


Policy = Callable[[EvaluatedState, StructuredActionCatalog, np.random.Generator], CollectionDecision]


def _active_follower(env: RLLeaderEnv):
    controller = getattr(env, "_active_controller", None) or env.controller
    return controller.nash_solver


def _equivalence_mode(env):
    mode = getattr(env, "response_equivalence_mode", LEGACY_EQUIVALENCE)
    if mode not in {LEGACY_EQUIVALENCE, CONTINUATION_EQUIVALENCE}:
        raise ValueError(f"unsupported response equivalence mode: {mode}")
    return mode


def _continuation_identity(env, *, interval_reward, terminal):
    from rl_leader.response_continuation_state import continuation_identity
    return continuation_identity(env, interval_reward=interval_reward, terminal=terminal)


def _encoded_continuation(env, reward, done, valid):
    return json.dumps({
        "validity_gate_pass": bool(valid),
        "continuation": _continuation_identity(env, interval_reward=float(reward), terminal=bool(done)),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _candidate_trial(
    source: RLLeaderEnv,
    anchor_context,
    catalog: StructuredActionCatalog,
    action_id: int,
) -> tuple[np.ndarray, str, bool, str]:
    mode = _equivalence_mode(source)
    trial, trial_context = copy.deepcopy((source, anchor_context))
    try:
        if int(action_id) == 0:
            _, reward, done, info, _ = trial.step_prepared_optimizer_anchor(trial_context)
        else:
            _, reward, done, info = trial.step_anchored_candidate(
                catalog.residual(action_id), trial_context,
            )
        response = trial.response_vector(trial.previous)
        validity = bool(info.get("validity_gate_pass", False))
        if mode == LEGACY_EQUIVALENCE:
            memory = trial._follower_runtime_fingerprint(_active_follower(trial))
    except Exception as exc:
        return np.empty(0, dtype=np.float32), "", False, type(exc).__name__
    # Identity failures are audit failures, not evidence that an action is infeasible.
    if mode == CONTINUATION_EQUIVALENCE:
        memory = _encoded_continuation(trial, reward, done, validity)
    return response, memory, validity, ""


def _candidate_trials(
    source: RLLeaderEnv,
    anchor_context,
    catalog: StructuredActionCatalog,
    action_ids: list[int],
    *,
    workers: int,
    backend: str = "thread",
) -> dict[int, tuple[np.ndarray, str, bool, str]]:
    """Evaluate independent deep-copied branches in deterministic action order."""
    workers = int(workers)
    if workers < 1:
        raise ValueError("response workers must be positive")
    if backend not in {"serial", "thread", "process"}:
        raise ValueError("response backend must be serial, thread, or process")

    def evaluate(action_id: int):
        return _candidate_trial(source, anchor_context, catalog, action_id)

    if backend == "serial" or workers == 1 or len(action_ids) <= 1:
        values = [evaluate(action_id) for action_id in action_ids]
    elif backend == "thread":
        with ThreadPoolExecutor(max_workers=min(workers, len(action_ids))) as executor:
            values = list(executor.map(evaluate, action_ids))
        return dict(zip(action_ids, values))
    else:
        process_workers = min(workers, len(action_ids))
        chunks = [
            action_ids[index::process_workers] for index in range(process_workers)
        ]
        payloads = [
            (source, anchor_context, catalog, chunk) for chunk in chunks if chunk
        ]
        executor = _response_process_pool(process_workers)
        chunk_results = list(executor.map(_candidate_trial_chunk, payloads))
        unordered = {
            action_id: result
            for chunk_result in chunk_results
            for action_id, result in chunk_result
        }
        return {action_id: unordered[action_id] for action_id in action_ids}
    return dict(zip(action_ids, values))


def _candidate_trial_chunk(payload):
    source, anchor_context, catalog, action_ids = payload
    return [
        (
            action_id,
            _candidate_trial(source, anchor_context, catalog, action_id),
        )
        for action_id in action_ids
    ]


def evaluate_executable_responses(
    env: RLLeaderEnv,
    observation: np.ndarray,
    catalog: StructuredActionCatalog,
    *,
    action_ids: Iterable[int] | None = None,
    response_atol: float = 1.0e-6,
    workers: int = 1,
    backend: str = "thread",
) -> EvaluatedState:
    """Preview candidates from one untouched state and common anchor."""
    mode = _equivalence_mode(env)
    started = time.perf_counter()
    anchor_context = env.prepare_pstack_anchor_context()
    anchor_response, anchor_memory, anchor_valid, anchor_reason = _candidate_trial(
        env, anchor_context, catalog, 0,
    )
    if anchor_response.size == 0:
        raise RuntimeError(f"P-Stack anchor preview failed: {anchor_reason}")
    if mode == CONTINUATION_EQUIVALENCE and not anchor_valid:
        raise ValueError("anchor preview failed the validity gate")
    if action_ids is None:
        preview_ids = [
            action.action_id for action in catalog.actions if action.action_id != 0
        ]
    else:
        preview_ids = sorted({
            int(action_id) for action_id in action_ids if int(action_id) != 0
        })
        bad_ids = [
            action_id for action_id in preview_ids
            if action_id < 0 or action_id >= catalog.size
        ]
        if bad_ids:
            raise ValueError(f"preview action_id out of range: {bad_ids[:8]}")
    trials = _candidate_trials(
        env,
        anchor_context,
        catalog,
        preview_ids,
        workers=workers,
        backend=backend,
    )
    previewed = set(preview_ids)
    responses = []
    for action in catalog.actions:
        if action.action_id == 0:
            response, memory, valid, reason = (
                anchor_response, anchor_memory, True, anchor_reason,
            )
        elif action.action_id not in previewed:
            response, memory, valid, reason = (
                anchor_response.copy(),
                f"preview_skipped:{action.action_id}",
                False,
                "preview_skipped_by_action_filter",
            )
        else:
            response, memory, valid, reason = trials[action.action_id]
            if response.size == 0:
                response = anchor_response.copy()
                memory = f"invalid:{action.action_id}:{reason}"
        responses.append(CandidateResponse(
            action_id=action.action_id,
            response=tuple(map(float, response)),
            follower_memory_fingerprint=memory,
            valid=valid,
            invalid_reason=reason,
        ))
    response_mask = build_response_mask(
        responses,
        catalog_size=catalog.size,
        response_atol=response_atol,
    )
    return EvaluatedState(
        observation=np.asarray(observation, dtype=np.float32).copy(),
        anchor_context=anchor_context,
        candidate_responses=tuple(responses),
        response_mask=response_mask,
        response_features=response_feature_matrix(
            responses, catalog_size=catalog.size,
        ),
        raw_candidate_count=1 + len(preview_ids),
        unique_response_count=len(response_mask.groups),
        evaluation_seconds=float(time.perf_counter() - started),
        response_workers=int(workers),
        response_backend=str(backend if workers > 1 else "serial"),
        response_equivalence_mode=mode,
    )


def evaluate_anchor_response(
    env: RLLeaderEnv,
    observation: np.ndarray,
    catalog: StructuredActionCatalog,
    *,
    response_feature_dim: int | None = None,
) -> EvaluatedState:
    """Build a cheap evaluated state when policy selection is forced to anchor."""
    if _equivalence_mode(env) == CONTINUATION_EQUIVALENCE:
        raise ValueError("continuation equivalence requires explicit executable previews")
    started = time.perf_counter()
    anchor_context = env.prepare_pstack_anchor_context()
    if response_feature_dim is not None:
        response_feature_dim = int(response_feature_dim)
        if response_feature_dim <= 0:
            raise ValueError("response_feature_dim must be positive")
        valid = np.zeros(catalog.size, dtype=bool)
        valid[0] = True
        representative = np.full(catalog.size, -1, dtype=np.int64)
        representative[0] = 0
        response_mask = ResponseMask(
            valid_action_mask=valid,
            representative_of=representative,
            groups=((0,),),
            invalid_reasons=tuple(
                "" if action.action_id == 0 else "preview_skipped_by_policy_gate"
                for action in catalog.actions
            ),
        )
        return EvaluatedState(
            observation=np.asarray(observation, dtype=np.float32).copy(),
            anchor_context=anchor_context,
            candidate_responses=(),
            response_mask=response_mask,
            response_features=np.zeros(
                (catalog.size, response_feature_dim), dtype=np.float32,
            ),
            raw_candidate_count=0,
            unique_response_count=1,
            evaluation_seconds=float(time.perf_counter() - started),
            response_workers=1,
            response_backend="anchor_commit_only",
        )
    anchor_response, anchor_memory, _, anchor_reason = _candidate_trial(
        env, anchor_context, catalog, 0,
    )
    if anchor_response.size == 0:
        raise RuntimeError(f"P-Stack anchor preview failed: {anchor_reason}")
    anchor_tuple = tuple(map(float, anchor_response))
    responses = [
        CandidateResponse(
            action.action_id,
            anchor_tuple,
            anchor_memory if action.action_id == 0 else f"anchor_only:{action.action_id}",
            action.action_id == 0,
            "" if action.action_id == 0 else "preview_skipped_by_policy_gate",
        )
        for action in catalog.actions
    ]
    response_mask = build_response_mask(responses, catalog_size=catalog.size)
    return EvaluatedState(
        observation=np.asarray(observation, dtype=np.float32).copy(),
        anchor_context=anchor_context,
        candidate_responses=tuple(responses),
        response_mask=response_mask,
        response_features=response_feature_matrix(
            responses, catalog_size=catalog.size,
        ),
        raw_candidate_count=1,
        unique_response_count=len(response_mask.groups),
        evaluation_seconds=float(time.perf_counter() - started),
        response_workers=1,
        response_backend="anchor_only",
    )


def commit_action(
    env: RLLeaderEnv,
    evaluated: EvaluatedState,
    catalog: StructuredActionCatalog,
    action_id: int,
):
    if evaluated.response_equivalence_mode != _equivalence_mode(env):
        raise ValueError("response equivalence changed between preview and commit")
    action_id = int(action_id)
    if not evaluated.response_mask.valid_action_mask[action_id]:
        raise ValueError(f"cannot commit masked action {action_id}")
    if action_id == 0:
        next_obs, reward, done, info, _ = env.step_prepared_optimizer_anchor(
            evaluated.anchor_context,
        )
    else:
        next_obs, reward, done, info = env.step_anchored_candidate(
            catalog.residual(action_id), evaluated.anchor_context,
        )
    if evaluated.response_equivalence_mode == CONTINUATION_EQUIVALENCE:
        expected = next(item for item in evaluated.candidate_responses if item.action_id == action_id)
        actual = _encoded_continuation(env, reward, done, info.get("validity_gate_pass", False))
        if actual != expected.follower_memory_fingerprint:
            raise ValueError("response preview and commit continuation differ")
    return np.asarray(next_obs, dtype=np.float32), float(reward), bool(done), dict(info)


def random_masked_policy(
    epsilon: float = 1.0,
    *,
    anchor_probability: float = 0.25,
) -> Policy:
    epsilon = float(epsilon)
    anchor_probability = float(anchor_probability)
    if not 0.0 <= anchor_probability <= 1.0:
        raise ValueError("anchor_probability must be in [0, 1]")

    def choose(
        evaluated: EvaluatedState,
        catalog: StructuredActionCatalog,
        rng: np.random.Generator,
    ) -> CollectionDecision:
        mask = evaluated.response_mask.valid_action_mask.copy()
        if rng.random() < anchor_probability or np.count_nonzero(mask) == 1:
            action_id = 0
        else:
            mask[0] = False
            action_id = masked_epsilon_greedy(
                np.zeros(catalog.size, dtype=np.float32),
                mask,
                epsilon=epsilon,
                rng=rng,
            )
        return CollectionDecision(
            action_id=action_id,
            diagnostics={
                "policy": "random_masked",
                "epsilon": epsilon,
                "anchor_probability": anchor_probability,
            },
        )

    return choose


def _validate_policy_equivalence(ensemble, evaluated):
    if any(getattr(model, "response_equivalence_mode", LEGACY_EQUIVALENCE) != getattr(evaluated, "response_equivalence_mode", LEGACY_EQUIVALENCE)
           for model in ensemble):
        raise ValueError("policy model and runtime response equivalence differ")


def ensemble_lcb_policy(
    ensemble: list[TrainedResponseDQN],
    *,
    z_value: float = 1.96,
    material_margin: float = 0.0,
    exploration_epsilon: float = 0.0,
    explore_unsupported: bool = False,
) -> Policy:
    exploration_epsilon = float(exploration_epsilon)
    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must be in [0, 1]")
    if not ensemble:
        raise ValueError("ensemble policy requires at least one model")
    fingerprint = ensemble[0].catalog_fingerprint
    if any(model.catalog_fingerprint != fingerprint for model in ensemble):
        raise ValueError("ensemble members use different action catalogs")
    support = np.min(np.stack([
        model.action_support_counts for model in ensemble
    ]), axis=0)

    def choose(
        evaluated: EvaluatedState,
        catalog: StructuredActionCatalog,
        rng: np.random.Generator,
    ) -> CollectionDecision:
        if catalog.fingerprint != fingerprint:
            raise ValueError("policy model and runtime action catalog differ")
        _validate_policy_equivalence(ensemble, evaluated)
        q_values = np.stack([
            model.q_values(
                evaluated.observation, evaluated.response_features,
            )[0]
            for model in ensemble
        ])
        valid = evaluated.response_mask.valid_action_mask.copy()
        supported = valid & (support >= ensemble[0].config.min_action_support)
        supported[0] = True
        exploration_mask = valid if explore_unsupported else supported
        exploratory = bool(
            exploration_epsilon > 0.0
            and np.count_nonzero(exploration_mask) > 1
            and rng.random() < exploration_epsilon
        )
        if exploratory:
            action_id = int(rng.choice(np.flatnonzero(exploration_mask)))
            return CollectionDecision(action_id=action_id, diagnostics={
                "policy": "ensemble_lcb",
                "exploratory": True,
                "explore_unsupported": bool(explore_unsupported),
                "fallback_reason": "",
            })
        decision = conservative_ensemble_selection(
            q_values,
            valid,
            action_support_counts=support,
            min_action_support=ensemble[0].config.min_action_support,
            z_value=z_value,
            material_margin=material_margin,
        )
        return CollectionDecision(action_id=decision.action_id, diagnostics={
            "policy": "ensemble_lcb",
            "exploratory": False,
            "explore_unsupported": bool(explore_unsupported),
            "q_anchor": decision.anchor_q_mean,
            "q_selected": decision.selected_q_mean,
            "delta_q": decision.delta_q_mean,
            "ensemble_std": decision.delta_q_std,
            "lcb": decision.lcb,
            "fallback_reason": decision.fallback_reason,
        })

    return choose


def ensemble_greedy_policy(
    ensemble: list[TrainedResponseDQN],
    *,
    exploration_epsilon: float = 0.0,
) -> Policy:
    """Collect sequential TD with mean-Q exploitation and all-valid exploration."""
    exploration_epsilon = float(exploration_epsilon)
    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must be in [0, 1]")
    if not ensemble:
        raise ValueError("ensemble policy requires at least one model")
    fingerprint = ensemble[0].catalog_fingerprint
    if any(model.catalog_fingerprint != fingerprint for model in ensemble):
        raise ValueError("ensemble members use different action catalogs")
    support = np.min(np.stack([
        model.action_support_counts for model in ensemble
    ]), axis=0)

    def choose(
        evaluated: EvaluatedState,
        catalog: StructuredActionCatalog,
        rng: np.random.Generator,
    ) -> CollectionDecision:
        if catalog.fingerprint != fingerprint:
            raise ValueError("policy model and runtime action catalog differ")
        _validate_policy_equivalence(ensemble, evaluated)
        q_values = np.stack([
            model.q_values(evaluated.observation, evaluated.response_features)[0]
            for model in ensemble
        ]).astype(np.float64)
        valid = evaluated.response_mask.valid_action_mask.copy()
        if q_values.shape != (len(ensemble), catalog.size) or valid.shape != (catalog.size,):
            raise ValueError("ensemble Q values and mask must match the action catalog")
        if not np.all(np.isfinite(q_values)):
            raise ValueError("ensemble Q values contain nonfinite values")
        if support.shape != valid.shape:
            raise ValueError("action support count shape mismatch")
        if not valid[0]:
            raise ValueError("anchor action must be valid")
        supported = valid & (support >= ensemble[0].config.min_action_support)
        supported[0] = True
        mean = q_values.mean(axis=0)
        exploratory = bool(
            exploration_epsilon > 0.0
            and np.count_nonzero(valid) > 1
            and rng.random() < exploration_epsilon
        )
        if exploratory:
            action_id = int(rng.choice(np.flatnonzero(valid)))
        else:
            action_id = int(np.flatnonzero(supported)[np.argmax(mean[supported])])
        return CollectionDecision(action_id=action_id, diagnostics={
            "policy": "ensemble_greedy",
            "exploratory": exploratory,
            "q_anchor": float(mean[0]),
            "q_selected": float(mean[action_id]),
            "delta_q": float(mean[action_id] - mean[0]),
            "ensemble_std": float(q_values[:, action_id].std()),
            "supported_action_count": int(supported.sum()),
            "fallback_reason": "",
        })

    return choose


def collect_sequential_episode(
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    policy: Policy,
    *,
    episode: int,
    rng: np.random.Generator,
    event_group: str,
    log_path: str | Path | None = None,
    response_workers: int = 1,
    response_backend: str = "thread",
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[dict[str, list]], None] | None = None,
    evaluate_response_steps: Callable[[int], bool] | None = None,
    anchor_only_response_feature_dim: int | None = None,
    preview_action_ids: Iterable[int] | None = None,
    initial_observation: np.ndarray | None = None,
    initial_rows: dict[str, list] | None = None,
) -> dict[str, list]:
    """Reset by default, or continue a supplied observation and append prior rows."""
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative")
    if initial_rows is not None and initial_observation is None:
        raise ValueError("initial_rows requires initial_observation")
    observation = np.asarray(
        env.reset() if initial_observation is None else initial_observation,
        dtype=np.float32,
    ).copy()

    def evaluate_current_step(step_index: int, obs: np.ndarray) -> EvaluatedState:
        if (
            evaluate_response_steps is not None
            and not bool(evaluate_response_steps(int(step_index)))
        ):
            return evaluate_anchor_response(
                env,
                obs,
                catalog,
                response_feature_dim=anchor_only_response_feature_dim,
            )
        return evaluate_executable_responses(
            env,
            obs,
            catalog,
            action_ids=preview_action_ids,
            workers=response_workers,
            backend=response_backend,
        )

    step = int(env.step_idx - env.warmup)
    rows: dict[str, list] = {
        name: [] for name in (
            "observation", "action_id", "reward", "next_observation", "done",
            "option_steps", "action_mask", "next_action_mask",
            "response_features", "next_response_features", "event_group",
            "episode", "control_step",
        )
    }
    if initial_rows is not None:
        if any(name not in initial_rows for name in rows):
            raise ValueError("initial_rows is missing collection fields")
        if len({len(initial_rows[name]) for name in rows}) != 1:
            raise ValueError("initial_rows fields must have equal lengths")
        rows = initial_rows
        if rows["action_id"]:
            if int(rows["control_step"][-1]) + 1 != step:
                raise ValueError("initial_rows do not end immediately before the resumed step")
            if not np.array_equal(rows["next_observation"][-1], observation):
                raise ValueError("initial_rows next observation differs from the resumed observation")
            if rows["done"][-1] and env.step_idx < env.n_steps:
                raise ValueError("cannot continue terminal rows in a nonterminal environment")
    if initial_observation is not None:
        if step < 0:
            raise ValueError("continuation must start after warmup")
        if env.step_idx >= env.n_steps:
            return rows
    current = evaluate_current_step(step, observation)
    if rows["action_id"]:
        if not np.array_equal(rows["next_action_mask"][-1], current.response_mask.valid_action_mask):
            raise ValueError("resumed response mask differs from the checkpoint")
        if not np.array_equal(rows["next_response_features"][-1], current.response_features):
            raise ValueError("resumed response features differ from the checkpoint")
    log_handle = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
    try:
        while True:
            decision = policy(current, catalog, rng)
            action_id = int(decision.action_id)
            next_observation, reward, done, info = commit_action(
                env, current, catalog, action_id,
            )
            if done:
                next_mask = np.zeros(catalog.size, dtype=bool)
                next_mask[0] = True
                next_features = np.zeros_like(current.response_features)
            else:
                following = evaluate_current_step(
                    int(env.step_idx - env.warmup), next_observation,
                )
                next_mask = following.response_mask.valid_action_mask.copy()
                next_features = following.response_features.copy()

            rows["observation"].append(current.observation.copy())
            rows["action_id"].append(action_id)
            rows["reward"].append(reward)
            rows["next_observation"].append(next_observation.copy())
            rows["done"].append(float(done))
            rows["option_steps"].append(1)
            rows["action_mask"].append(current.response_mask.valid_action_mask.copy())
            rows["next_action_mask"].append(next_mask)
            rows["response_features"].append(current.response_features.copy())
            rows["next_response_features"].append(next_features)
            rows["event_group"].append(str(event_group))
            rows["episode"].append(int(episode))
            rows["control_step"].append(int(step))

            if log_handle is not None:
                action = catalog.action(action_id)
                payload = {
                    "format_version": COLLECTOR_FORMAT,
                    "episode": int(episode),
                    "control_step": int(step),
                    "state_id": str(current.anchor_context.state_fingerprint),
                    "selected_action_id": action_id,
                    "selected_action_key": action.key,
                    "catalog_fingerprint": catalog.fingerprint,
                    "owner": action.owner,
                    "template": action.template,
                    "magnitude": action.magnitude,
                    "anchor": action_id == 0,
                    "raw_candidate_count": current.raw_candidate_count,
                    "unique_follower_response_count": current.unique_response_count,
                    "response_equivalence_mode": current.response_equivalence_mode,
                    "physical_control_group_count": len(build_response_mask(
                        [replace(item, follower_memory_fingerprint="") for item in current.candidate_responses],
                        catalog_size=catalog.size,
                    ).groups) if current.candidate_responses else None,
                    "continuation_groups": [list(group) for group in current.response_mask.groups],
                    "valid_action_mask_size": int(current.response_mask.valid_action_mask.sum()),
                    "interval_reward": reward,
                    "terminal_inventory": float(info.get("inventory_after", env._inventory())),
                    "validity_gate_pass": bool(info.get("validity_gate_pass", False)),
                    "response_workers": current.response_workers,
                    "response_backend": current.response_backend,
                    "response_evaluation_seconds": current.evaluation_seconds,
                    **decision.diagnostics,
                }
                if current.response_equivalence_mode == CONTINUATION_EQUIVALENCE:
                    payload["candidate_continuation_identities"] = {
                        str(item.action_id): json.loads(item.follower_memory_fingerprint)
                        for item in current.candidate_responses if item.valid
                    }
                log_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                log_handle.flush()
            transition_count = len(rows["action_id"])
            if (
                checkpoint_callback is not None
                and checkpoint_every > 0
                and (transition_count % checkpoint_every == 0 or done)
            ):
                checkpoint_callback(rows)
            print(json.dumps({
                "control_step": step,
                "response_workers": current.response_workers,
                "response_backend": current.response_backend,
                "response_evaluation_seconds": round(current.evaluation_seconds, 3),
                "transitions_checkpointed": transition_count if checkpoint_callback else 0,
                "done": done,
            }, sort_keys=True), flush=True)
            if done:
                break
            step = int(env.step_idx - env.warmup)
            current = following
    finally:
        if log_handle is not None:
            log_handle.close()
    return rows


def rows_to_replay(
    rows: dict[str, list],
    *,
    env: RLLeaderEnv,
    catalog: StructuredActionCatalog,
    source: str,
) -> FrozenResponseReplay:
    count = len(rows["action_id"])
    manifest = make_replay_manifest(
        transition_count=count,
        action_count=catalog.size,
        catalog_fingerprint=catalog.fingerprint,
        observation_schema=env.observation_schema.metadata(),
        response_contract=RL_RESPONSE_CONTRACT_VERSION,
        scenario=env.scenario_name,
        source=source,
    )
    manifest.update({
        "collector_format": COLLECTOR_FORMAT,
        "response_evaluation_mode": "oracle_all_actual_follower_responses",
        "reward_semantics": "interval_negative_ttt",
        "done_semantics": "environment_terminal",
        "experiment_contract_sha256": env.experiment_contract_fingerprint,
        "t_total_sec": float(env.T_total),
        "catalog": catalog.as_manifest(),
    })
    if _equivalence_mode(env) != LEGACY_EQUIVALENCE:
        manifest["response_equivalence_mode"] = _equivalence_mode(env)
    return FrozenResponseReplay(
        observation=np.asarray(rows["observation"], dtype=np.float32),
        action_id=np.asarray(rows["action_id"], dtype=np.int64),
        reward=np.asarray(rows["reward"], dtype=np.float32),
        next_observation=np.asarray(rows["next_observation"], dtype=np.float32),
        done=np.asarray(rows["done"], dtype=np.float32),
        option_steps=np.asarray(rows["option_steps"], dtype=np.int64),
        action_mask=np.asarray(rows["action_mask"], dtype=bool),
        next_action_mask=np.asarray(rows["next_action_mask"], dtype=bool),
        response_features=np.asarray(rows["response_features"], dtype=np.float32),
        next_response_features=np.asarray(rows["next_response_features"], dtype=np.float32),
        event_group=np.asarray(rows["event_group"]).astype(str),
        episode=np.asarray(rows["episode"], dtype=np.int64),
        control_step=np.asarray(rows["control_step"], dtype=np.int64),
        manifest=manifest,
    ).validate()


def _save_replay_atomic(replay: FrozenResponseReplay, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    replay.save(temporary)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--magnitudes", default="0.25")
    parser.add_argument("--families", default="linear,quadratic,cross")
    parser.add_argument("--domains", default="urban,freeway")
    parser.add_argument("--owners", default="")
    parser.add_argument(
        "--extra-actions",
        action="append",
        default=[],
        type=Path,
        help="manual or artifact-derived residual action manifest to append",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--anchor-probability", type=float, default=0.25)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--z-value", type=float, default=1.96)
    parser.add_argument("--material-margin", type=float, default=0.0)
    parser.add_argument("--exploration-epsilon", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend", choices=("serial", "thread", "process"), default="process",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = RLLeaderEnv(
        scenario_name=args.scenario,
        T_total=args.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    owners = tuple(value for value in args.owners.split(",") if value) or None
    catalog = build_structured_action_catalog(
        env.action_schema.names,
        magnitudes=tuple(float(value) for value in args.magnitudes.split(",")),
        families=tuple(value for value in args.families.split(",") if value),
        domains=tuple(value for value in args.domains.split(",") if value),
        owners=owners,
        extra_actions=load_extra_action_specs(args.extra_actions),
    )
    if args.model_dir is None:
        policy = random_masked_policy(
            args.epsilon, anchor_probability=args.anchor_probability,
        )
    else:
        paths = sorted(args.model_dir.glob("response_dqn_member_*.pt"))
        if not paths:
            raise SystemExit(f"no response DQN checkpoints in {args.model_dir}")
        ensemble = [load_trained_response_dqn(
            path,
            expected_catalog_fingerprint=catalog.fingerprint,
        ) for path in paths]
        policy = ensemble_lcb_policy(
            ensemble,
            z_value=args.z_value,
            material_margin=args.material_margin,
            exploration_epsilon=args.exploration_epsilon,
        )
    rows = collect_sequential_episode(
        env,
        catalog,
        policy,
        episode=args.episode,
        rng=np.random.default_rng(args.seed),
        event_group=f"{args.scenario}:episode:{args.episode}:seed:{args.seed}",
        log_path=args.log,
        response_workers=args.workers,
        response_backend=args.backend,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=lambda partial_rows: _save_replay_atomic(
            rows_to_replay(
                partial_rows, env=env, catalog=catalog, source=__file__,
            ),
            args.out,
        ),
    )
    replay = rows_to_replay(rows, env=env, catalog=catalog, source=__file__)
    _save_replay_atomic(replay, args.out)
    print(json.dumps({
        "output": str(args.out),
        "transitions": replay.size,
        "actions": catalog.size,
        "response_workers": args.workers,
        "response_backend": args.backend,
        "catalog_fingerprint": catalog.fingerprint,
        "action_support_counts": replay.action_support_counts().tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
