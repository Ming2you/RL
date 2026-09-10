"""Extend a cached P-Stack prefix to later control-step replay caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_leader.evaluate_cached_tail_residual_schedule import load_cached_prefix
from rl_leader.run_residual_5pct_loop import _write_direct_cache_payload


def parse_steps(raw: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not steps:
        raise ValueError("at least one step is required")
    if any(step < 0 for step in steps):
        raise ValueError("steps must be non-negative")
    return steps


def _cache_path(cache_output_dir: Path, scenario: str, step: int) -> Path:
    return Path(cache_output_dir) / f"{scenario}_step{int(step):02d}" / "replay_cache.pkl"


def extend_pstack_replay_caches(
    *,
    source_replay_cache: Path,
    cache_output_dir: Path,
    steps: tuple[int, ...],
    scenario: str | None = None,
    t_total: float | None = None,
) -> dict[int, Path]:
    env, _context = load_cached_prefix(Path(source_replay_cache))
    scenario_name = str(scenario or env.scenario_name)
    total_time = float(t_total if t_total is not None else env.T_total)
    current_step = int(env.step_idx - env.warmup)
    targets = set(map(int, steps))
    if any(step < current_step for step in targets):
        raise ValueError(
            f"source cache is at control_step {current_step}; cannot create earlier targets"
        )

    paths = {
        step: _cache_path(Path(cache_output_dir), scenario_name, step)
        for step in targets
    }
    pending = {
        step for step, path in paths.items()
        if not path.is_file() or not path.with_suffix(".json").is_file()
    }
    done = False
    while pending and not done:
        actual_step = int(env.step_idx - env.warmup)
        if actual_step in pending:
            anchor_context = env.prepare_pstack_anchor_context()
            _write_direct_cache_payload(
                cache_path=paths[actual_step],
                scenario=scenario_name,
                t_total=total_time,
                step=actual_step,
                env=env,
                anchor_context=anchor_context,
            )
            print(json.dumps({
                "phase": "extend_cache",
                "control_step": int(actual_step),
                "path": str(paths[actual_step]),
            }, sort_keys=True), flush=True)
            pending.remove(actual_step)
            if not pending:
                break

        if actual_step > max(targets):
            break
        context = env.prepare_pstack_anchor_context()
        _obs, _reward, done, _info, _extra = env.step_prepared_optimizer_anchor(
            context,
            sync_follower_state=True,
        )

    if pending:
        raise RuntimeError(
            "could not extend P-Stack replay cache to control step(s): "
            + ",".join(map(str, sorted(pending)))
        )
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay-cache", type=Path, required=True)
    parser.add_argument("--cache-output-dir", type=Path, required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--t-total", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = extend_pstack_replay_caches(
        source_replay_cache=args.source_replay_cache,
        cache_output_dir=args.cache_output_dir,
        steps=parse_steps(args.steps),
        scenario=args.scenario,
        t_total=args.t_total,
    )
    print(json.dumps({
        "passed": True,
        "steps": list(map(int, sorted(paths))),
        "paths": {str(step): str(path) for step, path in sorted(paths.items())},
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
