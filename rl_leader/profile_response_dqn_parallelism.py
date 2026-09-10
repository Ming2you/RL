"""Benchmark where a fixed CPU budget best accelerates response collection."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn_catalog import build_structured_action_catalog
from rl_leader.response_dqn_collect import (
    _active_follower,
    evaluate_executable_responses,
)


def _make_actor(scenario: str, owner: str):
    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=14400.0,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    catalog = build_structured_action_catalog(
        env.action_schema.names,
        magnitudes=(0.25,),
        families=("linear", "quadratic", "cross"),
        domains=("urban", "freeway"),
        owners=(owner,),
    )
    return env, catalog


def _profile_actor(payload: tuple[str, str, int, str, int]) -> dict:
    scenario, owner, inner_workers, backend, actor_id = payload
    started = time.perf_counter()
    env, catalog = _make_actor(scenario, owner)
    setup_seconds = time.perf_counter() - started
    observation = np.asarray(env._observe(), dtype=np.float32)
    evaluated = evaluate_executable_responses(
        env,
        observation,
        catalog,
        workers=inner_workers,
        backend=backend,
    )
    return {
        "actor_id": actor_id,
        "setup_seconds": setup_seconds,
        "evaluation_seconds": evaluated.evaluation_seconds,
        "total_seconds": time.perf_counter() - started,
        "unique_responses": evaluated.unique_response_count,
        "catalog_size": catalog.size,
    }


def profile_layout(
    *,
    scenario: str,
    owner: str,
    actors: int,
    inner_workers: int,
    backend: str,
) -> dict:
    payloads = [
        (scenario, owner, inner_workers, backend, actor_id)
        for actor_id in range(actors)
    ]
    started = time.perf_counter()
    if actors == 1:
        rows = [_profile_actor(payloads[0])]
    else:
        with ProcessPoolExecutor(max_workers=actors) as executor:
            rows = list(executor.map(_profile_actor, payloads))
    wall_seconds = time.perf_counter() - started
    return {
        "layout": f"{actors}_actors_x_{inner_workers}_{backend}_workers",
        "actors": actors,
        "inner_workers": inner_workers,
        "nominal_compute_workers": actors * inner_workers,
        "wall_seconds": wall_seconds,
        "states_evaluated": actors,
        "states_per_minute": 60.0 * actors / wall_seconds,
        "actor_results": rows,
    }


def profile_components(*, scenario: str, owner: str) -> dict:
    env, catalog = _make_actor(scenario, owner)
    started = time.perf_counter()
    anchor_context = env.prepare_pstack_anchor_context()
    anchor_prepare_seconds = time.perf_counter() - started

    started = time.perf_counter()
    serialized = pickle.dumps((env, anchor_context), protocol=pickle.HIGHEST_PROTOCOL)
    pickle_seconds = time.perf_counter() - started

    branches = []
    for action in catalog.actions:
        started = time.perf_counter()
        trial, trial_context = copy.deepcopy((env, anchor_context))
        deepcopy_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if action.action_id == 0:
            _, _, _, info, _ = trial.step_prepared_optimizer_anchor(trial_context)
        else:
            _, _, _, info = trial.step_anchored_candidate(
                catalog.residual(action.action_id), trial_context,
            )
        execute_seconds = time.perf_counter() - started
        response = trial.response_vector(trial.previous)
        branches.append({
            "action_id": action.action_id,
            "template": action.template,
            "deepcopy_seconds": deepcopy_seconds,
            "execute_seconds": execute_seconds,
            "valid": bool(info.get("validity_gate_pass", False)),
            "response_norm": float(np.linalg.norm(response)),
            "follower_fingerprint_chars": len(
                trial._follower_runtime_fingerprint(_active_follower(trial))
            ),
        })
    return {
        "scenario": scenario,
        "owner": owner,
        "catalog_size": catalog.size,
        "anchor_prepare_seconds": anchor_prepare_seconds,
        "pickle_seconds": pickle_seconds,
        "pickle_megabytes": len(serialized) / (1024.0 * 1024.0),
        "branch_deepcopy_seconds": sum(row["deepcopy_seconds"] for row in branches),
        "branch_execute_seconds": sum(row["execute_seconds"] for row in branches),
        "branches": branches,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--owner", default="R_F_E")
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--inner-workers", type=int, default=8)
    parser.add_argument(
        "--backend", choices=("serial", "thread", "process"), default="process",
    )
    parser.add_argument("--components", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.actors < 1 or args.inner_workers < 1:
        raise SystemExit("actors and inner-workers must be positive")
    if args.components:
        result = profile_components(scenario=args.scenario, owner=args.owner)
    else:
        result = profile_layout(
            scenario=args.scenario,
            owner=args.owner,
            actors=args.actors,
            inner_workers=args.inner_workers,
            backend=args.backend,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
