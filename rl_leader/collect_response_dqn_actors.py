"""Collect response-aware replay with independent simulator actors."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn_catalog import (
    build_structured_action_catalog,
    load_extra_action_specs,
)
from rl_leader.response_dqn_collect import (
    _save_replay_atomic,
    collect_sequential_episode,
    random_masked_policy,
    rows_to_replay,
)
from rl_leader.response_dqn_data import (
    load_frozen_response_replay,
    merge_frozen_response_replays,
)


@dataclass(frozen=True)
class ActorSpec:
    actor_id: int
    scenario: str
    t_total: float
    episode: int
    seed: int
    magnitudes: tuple[float, ...]
    families: tuple[str, ...]
    domains: tuple[str, ...]
    owners: tuple[str, ...]
    extra_action_paths: tuple[str, ...]
    epsilon: float
    anchor_probability: float
    checkpoint_every: int
    output_path: str
    log_path: str


def build_actor_specs(
    *,
    scenario: str,
    t_total: float,
    episodes: int,
    episode_start: int,
    seed_start: int,
    magnitudes: tuple[float, ...],
    families: tuple[str, ...],
    domains: tuple[str, ...],
    owners: tuple[str, ...],
    extra_action_paths: tuple[str, ...] = (),
    epsilon: float,
    anchor_probability: float,
    checkpoint_every: int,
    output_dir: Path,
    log_dir: Path,
) -> list[ActorSpec]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    specs = []
    for actor_id in range(episodes):
        episode = episode_start + actor_id
        seed = seed_start + actor_id
        stem = f"episode_{episode:03d}_seed_{seed:03d}"
        specs.append(ActorSpec(
            actor_id=actor_id,
            scenario=scenario,
            t_total=float(t_total),
            episode=episode,
            seed=seed,
            magnitudes=magnitudes,
            families=families,
            domains=domains,
            owners=owners,
            extra_action_paths=extra_action_paths,
            epsilon=float(epsilon),
            anchor_probability=float(anchor_probability),
            checkpoint_every=int(checkpoint_every),
            output_path=str(output_dir / f"{stem}.npz"),
            log_path=str(log_dir / f"{stem}.jsonl"),
        ))
    return specs


def _collect_actor(spec: ActorSpec) -> dict:
    started = time.perf_counter()
    output_path = Path(spec.output_path)
    log_path = Path(spec.log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or log_path.exists():
        raise FileExistsError(
            f"actor output already exists: {output_path} or {log_path}"
        )

    env = RLLeaderEnv(
        scenario_name=spec.scenario,
        T_total=spec.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    catalog = build_structured_action_catalog(
        env.action_schema.names,
        magnitudes=spec.magnitudes,
        families=spec.families,
        domains=spec.domains,
        owners=spec.owners or None,
        extra_actions=load_extra_action_specs(spec.extra_action_paths),
    )
    policy = random_masked_policy(
        spec.epsilon,
        anchor_probability=spec.anchor_probability,
    )

    def checkpoint(rows: dict[str, list]) -> None:
        replay = rows_to_replay(
            rows,
            env=env,
            catalog=catalog,
            source=__file__,
        )
        _save_replay_atomic(replay, output_path)

    rows = collect_sequential_episode(
        env,
        catalog,
        policy,
        episode=spec.episode,
        rng=np.random.default_rng(spec.seed),
        event_group=(
            f"{spec.scenario}:episode:{spec.episode}:seed:{spec.seed}"
        ),
        log_path=log_path,
        response_workers=1,
        response_backend="serial",
        checkpoint_every=spec.checkpoint_every,
        checkpoint_callback=checkpoint,
    )
    replay = rows_to_replay(
        rows,
        env=env,
        catalog=catalog,
        source=__file__,
    )
    _save_replay_atomic(replay, output_path)
    return {
        "actor_id": spec.actor_id,
        "episode": spec.episode,
        "seed": spec.seed,
        "output": str(output_path),
        "log": str(log_path),
        "transitions": replay.size,
        "catalog_fingerprint": catalog.fingerprint,
        "action_support_counts": replay.action_support_counts().tolist(),
        "wall_seconds": time.perf_counter() - started,
    }


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in value.split(",") if item)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=0)
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
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--anchor-probability", type=float, default=0.25)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--merged-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.actors < 1:
        raise SystemExit("actors must be positive")
    if args.episodes < 1:
        raise SystemExit("episodes must be positive")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be positive")
    specs = build_actor_specs(
        scenario=args.scenario,
        t_total=args.t_total,
        episodes=args.episodes,
        episode_start=args.episode_start,
        seed_start=args.seed_start,
        magnitudes=_parse_csv(args.magnitudes, float),
        families=_parse_csv(args.families),
        domains=_parse_csv(args.domains),
        owners=_parse_csv(args.owners),
        extra_action_paths=tuple(map(str, args.extra_actions)),
        epsilon=args.epsilon,
        anchor_probability=args.anchor_probability,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    collisions = [
        path
        for spec in specs
        for path in (Path(spec.output_path), Path(spec.log_path))
        if path.exists()
    ]
    if args.merged_out.exists():
        collisions.append(args.merged_out)
    if collisions:
        raise SystemExit(
            "refusing to overwrite existing actor outputs: "
            + ", ".join(map(str, collisions))
        )

    started = time.perf_counter()
    completed = []
    with ProcessPoolExecutor(max_workers=min(args.actors, args.episodes)) as executor:
        futures = {executor.submit(_collect_actor, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            completed.append(result)
            print(json.dumps({
                "event": "actor_completed",
                "completed_actors": len(completed),
                "total_actors": len(specs),
                **result,
            }, sort_keys=True), flush=True)

    completed.sort(key=lambda row: row["actor_id"])
    replays = [
        load_frozen_response_replay(row["output"]) for row in completed
    ]
    merged = merge_frozen_response_replays(
        replays,
        source="rl_leader.collect_response_dqn_actors",
    )
    _save_replay_atomic(merged, args.merged_out)
    summary = {
        "format_version": "response_dqn_actor_collection_v1",
        "parallelism": "independent_actor_processes",
        "inner_response_workers": 1,
        "scenario": args.scenario,
        "actors": args.actors,
        "episodes": args.episodes,
        "transitions": merged.size,
        "merged_output": str(args.merged_out),
        "action_support_counts": merged.action_support_counts().tolist(),
        "event_groups": merged.unique_event_groups().tolist(),
        "wall_seconds": time.perf_counter() - started,
        "actor_results": completed,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "actor_specs": [asdict(spec) for spec in specs],
    }
    summary_path = args.merged_out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
