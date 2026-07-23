from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from src.models.demand import load_scenarios
from src.models.state import ExperimentConfig
from src.rl.env import RLStepRecord, StackelbergRLEnvironment
from src.rl.nash_probe import probe_follower_nash_residual
from src.rl.replay import ReplayDatasetWriter, transitions_from_env_step


def run_smoke_rollout(
    cfg: ExperimentConfig,
    scenario,
    steps: int = 2,
    seed: int | None = None,
    policy: str = "scripted",
) -> list[RLStepRecord]:
    """테스트와 재사용 스크립트를 위한 in-memory smoke rollout helper."""

    env = StackelbergRLEnvironment(cfg, scenario, seed=seed)
    env.reset()
    for _ in range(max(0, int(steps))):
        if policy == "scripted":
            leader_index, follower_indices = env.scripted_safe_action_indices()
        elif policy == "random":
            leader_index = env.random_leader_action_index()
            follower_indices = env.random_follower_action_indices()
        else:
            raise ValueError("policy must be 'scripted' or 'random'.")
        step_result = env.step(leader_index, follower_indices)
        if step_result.done or step_result.truncated:
            break
    return list(env.records)


def write_smoke_outputs(
    records: list[RLStepRecord],
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Path]:
    """이미 생성된 records를 요청된 CSV/metadata 파일 묶음으로 저장한다."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "step_records": output / "rl_step_records.csv",
        "rewards": output / "rl_rewards.csv",
        "actions": output / "rl_actions.csv",
        "observations_summary": output / "rl_observations_summary.csv",
        "metadata": output / "metadata.json",
    }
    _write_step_records(paths["step_records"], records, [])
    _write_rewards(paths["rewards"], records)
    _write_actions(paths["actions"], records)
    _write_observation_summary(paths["observations_summary"], records)
    payload = dict(metadata or {})
    payload.setdefault("steps_recorded", len(records))
    payload.setdefault("files", {key: path.name for key, path in paths.items() if key != "metadata"})
    paths["metadata"].write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def run_smoke_experiment(
    config_path: str | Path = "src/config/default.yaml",
    scenarios_config_path: str | Path = "src/config/scenarios.yaml",
    scenario_name: str = "medium_demand",
    steps: int = 2,
    policy: str = "scripted",
    output_dir: str | Path = "outputs/rl_stackelberg_smoke_medium",
    replay_output_dir: str | Path | None = None,
    seed: int | None = None,
    t_total: float | None = None,
) -> Dict[str, Path]:
    """StackelbergRLEnvironment safe rollout을 실행하고 DDQN-ready 로그 파일을 저장한다."""

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
    env = StackelbergRLEnvironment(cfg, scenario, seed=seed)
    env.reset()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nash_rows = []
    replay_writer = None
    replay_paths = None
    if replay_output_dir is not None:
        replay_writer = ReplayDatasetWriter(
            replay_output_dir,
            metadata={
                "scenario": scenario_name,
                "policy": policy,
                "seed": env.seed,
                "requested_steps": int(steps),
                "source": "rl_stackelberg_smoke",
            },
        )
    for _ in range(max(0, int(steps))):
        if policy == "scripted":
            leader_index, follower_indices = env.scripted_safe_action_indices()
        elif policy == "random":
            leader_index = env.random_leader_action_index()
            follower_indices = env.random_follower_action_indices()
        else:
            raise ValueError("policy must be 'scripted' or 'random'.")

        # 1-step probe는 실제 rollout 이전 state에서 unilateral deviation만 복제 평가한다.
        nash_result = probe_follower_nash_residual(env, leader_index, follower_indices)
        step_result = env.step(leader_index, follower_indices)
        if replay_writer is not None:
            replay_writer.append_many(transitions_from_env_step(0, step_result))
        nash_row = {
            "step_index": step_result.record.step_index,
            **nash_result.as_dict(),
        }
        nash_rows.append(nash_row)
        if step_result.done or step_result.truncated:
            break

    if replay_writer is not None:
        replay_writer.close()
        replay_paths = replay_writer.paths()

    paths = _write_requested_outputs(
        records=env.records,
        nash_rows=nash_rows,
        output_dir=output,
        cfg=cfg,
        scenario_name=scenario_name,
        policy=policy,
        seed=env.seed,
        requested_steps=steps,
    )
    if replay_paths is not None:
        paths.update({f"replay_{key}": path for key, path in replay_paths.items()})
    return paths


def _write_requested_outputs(
    records: list[RLStepRecord],
    nash_rows: list[Dict[str, Any]],
    output_dir: Path,
    cfg: ExperimentConfig,
    scenario_name: str,
    policy: str,
    seed: int,
    requested_steps: int,
) -> Dict[str, Path]:
    paths = {
        "step_records": output_dir / "rl_step_records.csv",
        "rewards": output_dir / "rl_rewards.csv",
        "actions": output_dir / "rl_actions.csv",
        "observations_summary": output_dir / "rl_observations_summary.csv",
        "metadata": output_dir / "metadata.json",
    }
    _write_step_records(paths["step_records"], records, nash_rows)
    _write_rewards(paths["rewards"], records)
    _write_actions(paths["actions"], records)
    _write_observation_summary(paths["observations_summary"], records)
    metadata = {
        "scenario": scenario_name,
        "policy": policy,
        "seed": int(seed),
        "requested_steps": int(requested_steps),
        "actual_steps": len(records),
        "config_path": "src/config/default.yaml",
        "control_interval_sec": float(cfg.simulation.control_interval),
        "T_total_sec": float(cfg.simulation.T_total),
        "n_control_steps": int(cfg.simulation.n_control_steps),
        "files": {key: path.name for key, path in paths.items() if key != "metadata"},
    }
    paths["metadata"].write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def _write_step_records(
    path: Path,
    records: list[RLStepRecord],
    nash_rows: list[Dict[str, Any]],
) -> None:
    nash_by_step = {int(row["step_index"]): row for row in nash_rows}
    rows = []
    for record in records:
        nash = nash_by_step.get(record.step_index, {})
        rows.append({
            "step_index": record.step_index,
            "time_sec_before": record.time_sec_before,
            "time_sec_after": record.time_sec_after,
            "leader_action_index": record.leader_action_index,
            "leader_N_P_star": record.leader_action.get("N_P_star", 0.0),
            "leader_N_UF_star": record.leader_action.get("N_UF_star", 0.0),
            "leader_reward": record.leader_reward,
            "global_step_ttt": record.global_step_ttt,
            "follower_count": len(record.follower_rewards),
            "epsilon_Nash_hat": nash.get("epsilon_Nash_hat", 0.0),
            "best_deviating_agent_id": nash.get("best_deviating_agent_id"),
            "best_unilateral_action_index": nash.get("best_unilateral_action_index"),
            "realized_reward": nash.get("realized_reward", 0.0),
            "best_unilateral_reward": nash.get("best_unilateral_reward", 0.0),
            "control_diagnostics": json.dumps(_jsonable(record.control.diagnostics), sort_keys=True),
            "sim_diagnostics": json.dumps(_jsonable(record.diagnostics), sort_keys=True),
        })
    _write_csv(path, rows)


def _write_rewards(path: Path, records: list[RLStepRecord]) -> None:
    rows = []
    for record in records:
        rows.append({
            "step_index": record.step_index,
            "agent_id": "leader",
            "agent_family": "leader",
            "reward": record.leader_reward,
            "terms": json.dumps(_jsonable(record.leader_reward_terms), sort_keys=True),
        })
        for agent_id, reward in sorted(record.follower_rewards.items()):
            rows.append({
                "step_index": record.step_index,
                "agent_id": agent_id,
                "agent_family": record.follower_observations[agent_id].family,
                "reward": reward,
                "terms": json.dumps(_jsonable(record.follower_reward_terms.get(agent_id, {})), sort_keys=True),
            })
    _write_csv(path, rows)


def _write_actions(path: Path, records: list[RLStepRecord]) -> None:
    rows = []
    for record in records:
        rows.append({
            "step_index": record.step_index,
            "agent_id": "leader",
            "agent_family": "leader",
            "action_index": record.leader_action_index,
            "action": json.dumps(_jsonable(record.leader_action), sort_keys=True),
            "physical_action": json.dumps(_control_to_jsonable(record.control), sort_keys=True),
        })
        for agent_id, action_index in sorted(record.follower_action_indices.items()):
            rows.append({
                "step_index": record.step_index,
                "agent_id": agent_id,
                "agent_family": record.follower_observations[agent_id].family,
                "action_index": action_index,
                "action": json.dumps(_jsonable(record.physical_follower_actions.get(agent_id, {})), sort_keys=True),
                "physical_action": json.dumps(_jsonable(record.physical_follower_actions.get(agent_id, {})), sort_keys=True),
            })
    _write_csv(path, rows)


def _write_observation_summary(path: Path, records: list[RLStepRecord]) -> None:
    rows = []
    for record in records:
        rows.append(_observation_row(record.step_index, record.leader_observation))
        for observation in sorted(record.follower_observations.values(), key=lambda obs: obs.agent_id):
            rows.append(_observation_row(record.step_index, observation))
    _write_csv(path, rows)


def _observation_row(step_index: int, observation) -> Dict[str, Any]:
    features = np.asarray(observation.features, dtype=float)
    return {
        "step_index": int(step_index),
        "agent_id": observation.agent_id,
        "agent_family": observation.family,
        "feature_count": int(features.size),
        "feature_mean": float(features.mean()) if features.size else 0.0,
        "feature_min": float(features.min()) if features.size else 0.0,
        "feature_max": float(features.max()) if features.size else 0.0,
        "feature_l2_norm": float(np.linalg.norm(features)) if features.size else 0.0,
        "features": json.dumps(_jsonable(features), sort_keys=True),
        "feature_names": json.dumps(list(observation.feature_names), sort_keys=True),
        "owned_links": json.dumps(list(observation.owned_links), sort_keys=True),
        "owned_ramps": json.dumps(list(observation.owned_ramps), sort_keys=True),
        "owned_movements": json.dumps(list(observation.owned_movements), sort_keys=True),
        "owned_signals": json.dumps(list(observation.owned_signals), sort_keys=True),
        "connected_coupling_links": json.dumps(list(observation.connected_coupling_links), sort_keys=True),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _control_to_jsonable(control) -> Dict[str, Any]:
    return {
        "N_P_star": float(control.N_P_star),
        "N_UF_star": float(control.N_UF_star),
        "ramp_metering": _jsonable(control.ramp_metering),
        "vsl": _jsonable(control.vsl),
        "green_times": _jsonable(control.green_times),
        "offsets": _jsonable(control.offsets),
        "inflow_outflow_allocation": _jsonable(control.inflow_outflow_allocation),
        "infeasibility": _jsonable(control.infeasibility),
        "diagnostics": _jsonable(control.diagnostics),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Stackelberg-RL safe smoke rollout.")
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default="src/config/scenarios.yaml")
    parser.add_argument("--scenario", default="medium_demand")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--T-total", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-output", default=None)
    args = parser.parse_args()

    paths = run_smoke_experiment(
        config_path=args.config,
        scenarios_config_path=args.scenarios_config,
        scenario_name=args.scenario,
        steps=args.steps,
        policy=args.policy,
        output_dir=args.output,
        replay_output_dir=args.replay_output,
        seed=args.seed,
        t_total=args.T_total,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
