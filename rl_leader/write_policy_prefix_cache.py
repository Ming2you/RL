"""Write cached-tail replay caches from an executed policy prefix.

The residual samplers operate on an ``(env, anchor_context)`` pickle captured at
one pre-action control step.  The original cache writer only replays the native
P-Stack prefix.  This helper replays an arbitrary response-DQN policy trace so
we can keep searching from states that the learned leader actually reaches.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

from rl_leader.env import RLLeaderEnv
from rl_leader.response_ddqn_recovery import (
    capture_policy_snapshots_cached,
    read_policy_trace,
    restore_env_snapshot,
)
from rl_leader.response_dqn_catalog import (
    build_structured_action_catalog,
    load_extra_action_specs,
)


CACHE_FORMAT = "policy_prefix_replay_cache_v1"


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in str(value).split(",") if item)


def policy_prefix_cache_path(output_dir: Path, scenario: str, step: int) -> Path:
    return Path(output_dir) / f"{scenario}_step{int(step):02d}" / "replay_cache.pkl"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_cache_payload(
    *,
    cache_path: Path,
    env: RLLeaderEnv,
    anchor_context,
    metadata: dict,
    overwrite: bool,
) -> None:
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".json")
    if not overwrite and (cache_path.exists() or meta_path.exists()):
        raise FileExistsError(
            "refusing to overwrite policy-prefix replay cache: "
            f"{cache_path} / {meta_path}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_payload = cache_path.with_suffix(".tmp.pkl")
    with tmp_payload.open("wb") as handle:
        pickle.dump((env, anchor_context), handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_payload.replace(cache_path)
    _write_json(meta_path, metadata)


def write_policy_prefix_caches(
    *,
    scenario: str,
    t_total: float,
    policy_trace: Path,
    policy_steps: Iterable[int],
    output_dir: Path,
    magnitudes: tuple[float, ...],
    families: tuple[str, ...],
    domains: tuple[str, ...],
    owners: tuple[str, ...] | None,
    extra_action_paths: Iterable[Path] = (),
    snapshot_cache_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[int, Path]:
    steps = sorted({int(step) for step in policy_steps})
    if not steps:
        raise ValueError("at least one policy step is required")
    if steps[0] < 0:
        raise ValueError("policy steps cannot be negative")

    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=float(t_total),
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    catalog = build_structured_action_catalog(
        env.action_schema.names,
        magnitudes=magnitudes,
        families=families,
        domains=domains,
        owners=owners,
        extra_actions=load_extra_action_specs(extra_action_paths),
    )
    policy_prefix = read_policy_trace(policy_trace)
    snapshots = capture_policy_snapshots_cached(
        env,
        catalog,
        steps,
        policy_prefix,
        cache_dir=snapshot_cache_dir,
    )

    written: dict[int, Path] = {}
    for step in steps:
        snapshot = snapshots[int(step)]
        replay_env, _observation = restore_env_snapshot(snapshot)
        anchor_context = replay_env.prepare_pstack_anchor_context()
        actual_fingerprint = str(replay_env._anchor_context_state_fingerprint())
        if actual_fingerprint != str(anchor_context.state_fingerprint):
            raise RuntimeError(
                "policy-prefix cache fingerprint mismatch: "
                f"{actual_fingerprint} != {anchor_context.state_fingerprint}"
            )
        if int(replay_env.step_idx) != int(anchor_context.step_idx):
            raise RuntimeError(
                "policy-prefix cache step mismatch: "
                f"{replay_env.step_idx} != {anchor_context.step_idx}"
            )
        cache_path = policy_prefix_cache_path(output_dir, scenario, step)
        prefix_actions = {
            str(prefix_step): int(action_id)
            for prefix_step, action_id in sorted(policy_prefix.items())
            if int(prefix_step) < int(step)
        }
        metadata = {
            "format_version": CACHE_FORMAT,
            "scenario": str(scenario),
            "t_total_sec": float(t_total),
            "control_step": int(step),
            "env_step_idx": int(replay_env.step_idx),
            "warmup_steps": int(replay_env.warmup),
            "simulation_time_sec": float(replay_env.sim.state.time_sec),
            "state_fingerprint": str(anchor_context.state_fingerprint),
            "experiment_contract_sha256": str(
                replay_env.experiment_contract_fingerprint
            ),
            "catalog_fingerprint": str(catalog.fingerprint),
            "policy_trace": str(policy_trace),
            "policy_prefix": prefix_actions,
            "extra_actions": [str(path) for path in extra_action_paths],
        }
        _write_cache_payload(
            cache_path=cache_path,
            env=replay_env,
            anchor_context=anchor_context,
            metadata=metadata,
            overwrite=bool(overwrite),
        )
        written[int(step)] = cache_path
        print(json.dumps({
            "phase": "policy_prefix_cache",
            "scenario": str(scenario),
            "control_step": int(step),
            "path": str(cache_path),
            "state_fingerprint": str(anchor_context.state_fingerprint),
        }, sort_keys=True), flush=True)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--policy-trace", type=Path, required=True)
    parser.add_argument("--policy-steps", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--snapshot-cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_policy_prefix_caches(
        scenario=args.scenario,
        t_total=float(args.t_total),
        policy_trace=args.policy_trace,
        policy_steps=_parse_csv(args.policy_steps, int),
        output_dir=args.output_dir,
        magnitudes=_parse_csv(args.magnitudes, float),
        families=_parse_csv(args.families),
        domains=_parse_csv(args.domains),
        owners=_parse_csv(args.owners) or None,
        extra_action_paths=tuple(args.extra_actions),
        snapshot_cache_dir=args.snapshot_cache_dir,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
