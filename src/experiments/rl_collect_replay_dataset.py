from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.models.demand import load_scenarios
from src.models.state import ExperimentConfig
from src.rl.env import StackelbergRLEnvironment
from src.rl.replay import ReplayDatasetWriter, transitions_from_env_step


def collect_replay_dataset(
    config_path: str | Path = "src/config/default.yaml",
    scenarios_config_path: str | Path = "src/config/scenarios.yaml",
    scenario_name: str = "medium_demand",
    episodes: int = 1,
    steps: int = 2,
    policy: str = "scripted",
    output_dir: str | Path = "outputs/rl_replay_dataset_medium",
    seed: int | None = None,
    t_total: float | None = None,
) -> Dict[str, Path]:
    """Collect replay transitions while rolling out cheap safe policies."""

    overrides: Dict[str, Any] = {}
    if t_total is not None:
        overrides["simulation"] = {"T_total": float(t_total)}
    cfg = ExperimentConfig.from_file(config_path, overrides or None)
    scenarios = load_scenarios(scenarios_config_path)
    if scenario_name not in scenarios:
        raise SystemExit(
            f"Unknown scenario {scenario_name}. Available: {', '.join(sorted(scenarios))}"
        )
    scenario = scenarios[scenario_name]
    base_seed = int(seed if seed is not None else cfg.simulation.random_seed)
    metadata = {
        "scenario": scenario_name,
        "policy": policy,
        "episodes_requested": int(episodes),
        "steps_requested": int(steps),
        "seed": base_seed,
        "config_path": str(config_path),
        "scenarios_config_path": str(scenarios_config_path),
    }

    episode_count = 0
    step_count = 0
    with ReplayDatasetWriter(output_dir, metadata=metadata) as writer:
        for episode in range(max(0, int(episodes))):
            env = StackelbergRLEnvironment(cfg, scenario, seed=base_seed + episode)
            env.reset()
            episode_count += 1
            for _ in range(max(0, int(steps))):
                if policy == "scripted":
                    leader_index, follower_indices = env.scripted_safe_action_indices()
                elif policy == "random":
                    leader_index = env.random_leader_action_index()
                    follower_indices = env.random_follower_action_indices()
                else:
                    raise ValueError("policy must be 'scripted' or 'random'.")
                step_result = env.step(leader_index, follower_indices)
                writer.append_many(transitions_from_env_step(episode, step_result))
                step_count += 1
                if step_result.done or step_result.truncated:
                    break
    summary = {
        "episodes_recorded": episode_count,
        "environment_steps_recorded": step_count,
        "transitions_recorded": writer.count,
        "files": {key: str(path.name) for key, path in writer.paths().items()},
    }
    summary_path = Path(output_dir) / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths = writer.paths()
    paths["summary"] = summary_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Stackelberg-RL replay transitions.")
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default="src/config/scenarios.yaml")
    parser.add_argument("--scenario", default="medium_demand")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--T-total", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = collect_replay_dataset(
        config_path=args.config,
        scenarios_config_path=args.scenarios_config,
        scenario_name=args.scenario,
        episodes=args.episodes,
        steps=args.steps,
        policy=args.policy,
        output_dir=args.output,
        seed=args.seed,
        t_total=args.T_total,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
