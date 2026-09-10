"""Generate long-horizon P-Stack-relative labels for accepted residual actions."""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rl_leader.data_contract import validate_pstack_residual_rows
from rl_leader.env import RLLeaderEnv
from rl_leader.experiment_contract import ExperimentContract, verify_contract_map


LABEL_CONTRACT = "pstack_first_action_closed_loop_recovery_v2_anchor_reference"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_local_indices(dataset) -> np.ndarray:
    modes = np.asarray(dataset["behavior_mode"]).astype(str)
    picked = np.asarray(dataset["pstack_anchor_pick_rl"], dtype=float) > 0.5
    return np.flatnonzero((modes == "optimizer_local") & picked)


def counterfactual_verdict(
    *,
    candidate_ttt: float,
    pstack_ttt: float,
    candidate_terminal_inventory: float,
    pstack_terminal_inventory: float,
) -> dict[str, float | bool]:
    gain = float(pstack_ttt - candidate_ttt)
    required_gain = float(max(0.1, 1.0e-3 * max(abs(pstack_ttt), 1.0)))
    inventory_delta = float(candidate_terminal_inventory - pstack_terminal_inventory)
    inventory_blocked = inventory_delta > 1.0e-6
    return {
        "ttt_gain": gain,
        "required_gain": required_gain,
        "terminal_inventory_delta": inventory_delta,
        "inventory_blocked": bool(inventory_blocked),
        "long_horizon_positive": bool(gain > required_gain and not inventory_blocked),
    }


def replay_reward_parity(
    *,
    saved_reward: float,
    replay_reward: float,
    termination_reason: str,
    failure_cost: float,
    parity_tolerance: float,
) -> dict[str, float | bool]:
    """Compare environment rewards after removing collector-only abort penalties."""
    raw_error = abs(float(replay_reward) - float(saved_reward))
    adjustment = (
        float(failure_cost) if termination_reason == "wall_clock_abort" else 0.0
    )
    adjusted_saved = float(saved_reward) + adjustment
    adjusted_error = abs(float(replay_reward) - adjusted_saved)
    quantization_tolerance = (
        0.5 * abs(float(np.spacing(np.float32(saved_reward))))
        if adjustment else 0.0
    )
    allowed_error = float(parity_tolerance) + quantization_tolerance
    return {
        "raw_error": raw_error,
        "adjustment": adjustment,
        "adjusted_error": adjusted_error,
        "allowed_error": allowed_error,
        "passes": bool(adjusted_error <= allowed_error),
    }


def _manifest(dataset) -> dict:
    value = dataset["manifest_json"]
    value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return json.loads(str(value))


def _rollout_branch(
    env: RLLeaderEnv,
    first_action: np.ndarray | None,
    *,
    expect_rl_first: bool,
    rollout_steps: int,
    horizons: tuple[int, ...],
) -> dict:
    cumulative_ttt = 0.0
    checkpoints = {}
    if expect_rl_first:
        if first_action is None:
            raise ValueError("RL counterfactual branch requires a first action")
        _, reward, done, first_info = env.step(first_action)
        picked_rl = float(first_info.get("leader_rl_pstack_anchor_pick_rl", 0.0)) > 0.5
        if not picked_rl:
            raise RuntimeError("accepted residual no longer passes the replayed anchor gate")
    else:
        _, reward, done, first_info, _ = env.step_optimizer_anchor(
            sync_follower_state=True
        )
    cumulative_ttt += -float(reward)
    completed = 1
    requested = set(horizons) | {rollout_steps}
    if completed in requested:
        checkpoints[str(completed)] = {
            "ttt": float(cumulative_ttt),
            "terminal_inventory": float(env._inventory()),
        }
    while not done and completed < rollout_steps:
        _, reward, done, _, _ = env.step_optimizer_anchor(sync_follower_state=True)
        cumulative_ttt += -float(reward)
        completed += 1
        if completed in requested:
            checkpoints[str(completed)] = {
                "ttt": float(cumulative_ttt),
                "terminal_inventory": float(env._inventory()),
            }
    if str(completed) not in checkpoints:
        checkpoints[str(completed)] = {
            "ttt": float(cumulative_ttt),
            "terminal_inventory": float(env._inventory()),
        }
    return {
        "steps": int(completed),
        "ttt": float(cumulative_ttt),
        "terminal_inventory": float(env._inventory()),
        "final_simulation_time_sec": float(env.sim.state.time_sec),
        "first_step_ttt": float(first_info["step_ttt"]),
        "first_step_gate_gain": float(first_info.get("leader_rl_pstack_anchor_gain", 0.0)),
        "checkpoints": checkpoints,
    }


def _evaluate_event(
    env: RLLeaderEnv,
    residual: np.ndarray,
    *,
    max_rollout_steps: int,
    horizons: tuple[int, ...],
) -> dict:
    remaining = int(env.n_steps - env.step_idx)
    rollout_steps = remaining if max_rollout_steps <= 0 else min(remaining, max_rollout_steps)
    if rollout_steps <= 0:
        raise RuntimeError("counterfactual event has no remaining simulation steps")
    candidate = _rollout_branch(
        copy.deepcopy(env), residual,
        expect_rl_first=True,
        rollout_steps=rollout_steps,
        horizons=horizons,
    )
    pstack = _rollout_branch(
        copy.deepcopy(env), None,
        expect_rl_first=False,
        rollout_steps=rollout_steps,
        horizons=horizons,
    )
    if candidate["steps"] != pstack["steps"]:
        raise RuntimeError("candidate and P-Stack branches completed different horizons")
    verdict = counterfactual_verdict(
        candidate_ttt=candidate["ttt"],
        pstack_ttt=pstack["ttt"],
        candidate_terminal_inventory=candidate["terminal_inventory"],
        pstack_terminal_inventory=pstack["terminal_inventory"],
    )
    horizon_rows = {}
    common = sorted(set(candidate["checkpoints"]) & set(pstack["checkpoints"]), key=int)
    for key in common:
        candidate_point = candidate["checkpoints"][key]
        pstack_point = pstack["checkpoints"][key]
        horizon_rows[key] = {
            "candidate_ttt": float(candidate_point["ttt"]),
            "pstack_ttt": float(pstack_point["ttt"]),
            **counterfactual_verdict(
                candidate_ttt=candidate_point["ttt"],
                pstack_ttt=pstack_point["ttt"],
                candidate_terminal_inventory=candidate_point["terminal_inventory"],
                pstack_terminal_inventory=pstack_point["terminal_inventory"],
            ),
        }
    return {
        "rollout_steps": int(candidate["steps"]),
        "candidate": candidate,
        "pstack": pstack,
        "horizons": horizon_rows,
        **verdict,
    }


def _process_episode(
    dataset,
    path: str,
    episode: int,
    summary: dict,
    accepted_set: set[int],
    max_rollout_steps: int,
    horizons: tuple[int, ...],
    parity_tolerance: float,
    failure_cost: float,
    experiment_contract: ExperimentContract,
    source_sha256: str,
) -> list[dict]:
    episode_rows = np.flatnonzero(np.asarray(dataset["episode"], dtype=int) == episode)
    accepted_count = sum(int(index) in accepted_set for index in episode_rows)
    env = RLLeaderEnv(
        pstack_anchor=True,
        action_parameterization="pstack_residual",
        experiment_contract=experiment_contract,
    )
    obs = env.reset()
    labels = []
    for row_index in episode_rows:
        row_index = int(row_index)
        obs_error = float(np.max(np.abs(obs - dataset["obs"][row_index])))
        if obs_error > parity_tolerance:
            raise RuntimeError(
                f"{path} episode={episode} step={dataset['step'][row_index]} "
                f"observation replay error {obs_error} exceeds {parity_tolerance}"
            )
        residual = np.asarray(
            dataset["policy_residual"][row_index], dtype=np.float32
        )
        pending = None
        if row_index in accepted_set:
            pending = {
                "label_contract": LABEL_CONTRACT,
                "source_file": str(path),
                "source_file_sha256": source_sha256,
                "source_row_index": row_index,
                "episode": int(episode),
                "step": int(dataset["step"][row_index]),
                "simulation_time_sec": float(dataset["simulation_time_sec"][row_index]),
                "scenario": summary["scenario"],
                "target_scenario": str(summary["scenario"].get("target_scenario", "unknown")),
                "termination_reason": str(summary["termination_reason"]),
                "experiment_contract_sha256": experiment_contract.sha256,
                "original_h3_gate_gain": float(dataset["pstack_anchor_gain"][row_index]),
                "observation": np.asarray(dataset["obs"][row_index], dtype=float).tolist(),
                "residual_action": residual.astype(float).tolist(),
                "residual_l2": float(np.linalg.norm(residual)),
                "residual_abs_max": float(np.max(np.abs(residual))),
                "replay_observation_error": obs_error,
                **_evaluate_event(
                    env,
                    residual,
                    max_rollout_steps=max_rollout_steps,
                    horizons=horizons,
                ),
            }
            pending["h3_gate_false_positive"] = bool(
                pending["original_h3_gate_gain"] > 0.0
                and not pending["long_horizon_positive"]
            )
        next_obs, reward, _, info = env.step(residual)
        anchor_error = float(np.max(np.abs(
            env.last_anchor_raw_action - dataset["anchor_action"][row_index]
        )))
        termination_reason = (
            str(dataset["termination_reason"][row_index])
            if "termination_reason" in dataset else ""
        )
        reward_parity = replay_reward_parity(
            saved_reward=float(dataset["rew"][row_index]),
            replay_reward=float(reward),
            termination_reason=termination_reason,
            failure_cost=failure_cost,
            parity_tolerance=parity_tolerance,
        )
        reward_error = float(reward_parity["adjusted_error"])
        saved_pick = float(dataset["pstack_anchor_pick_rl"][row_index]) > 0.5
        actual_pick = float(info.get("leader_rl_pstack_anchor_pick_rl", 0.0)) > 0.5
        if (
            anchor_error > parity_tolerance
            or not reward_parity["passes"]
            or saved_pick != actual_pick
        ):
            raise RuntimeError(
                f"{path} episode={episode} step={dataset['step'][row_index]} replay mismatch: "
                f"anchor={anchor_error}, reward={reward_error}, "
                f"reward_raw={reward_parity['raw_error']}, "
                f"reward_allowed={reward_parity['allowed_error']}, "
                f"saved_pick={saved_pick}, actual_pick={actual_pick}"
            )
        if pending is not None:
            pending.update({
                "replay_anchor_error": anchor_error,
                "replay_reward_error": reward_error,
                "replay_reward_error_raw": float(reward_parity["raw_error"]),
                "replay_reward_external_adjustment": float(
                    reward_parity["adjustment"]
                ),
                "replay_pick_match": True,
            })
            labels.append(pending)
            print(
                f"label file={Path(path).name} episode={episode} "
                f"step={pending['step']} original_h3={pending['original_h3_gate_gain']:+.3f} "
                f"long_gain={pending['ttt_gain']:+.3f} "
                f"positive={int(pending['long_horizon_positive'])}",
                flush=True,
            )
        obs = next_obs
        if len(labels) >= accepted_count:
            break
    return labels


def _process_file(
    path: str,
    max_rollout_steps: int,
    horizons: tuple[int, ...],
    parity_tolerance: float,
    max_events: int,
    episode_filter: tuple[int, ...] | None = None,
) -> dict[str, list[dict]]:
    with np.load(path, allow_pickle=False) as dataset:
        accepted = accepted_local_indices(dataset)
        if max_events > 0:
            accepted = accepted[:max_events]
        if accepted.size == 0:
            return {"labels": [], "errors": []}
        manifest = _manifest(dataset)
        contracts = verify_contract_map(manifest.get("experiment_contracts", {}))
        if not contracts:
            raise ValueError(f"{path} is missing verified experiment contracts")
        validate_pstack_residual_rows(dataset, manifest)
        source_sha256 = _file_sha256(path)
        failure_cost = float(manifest.get("failure_cost", 5000.0))
        summaries = {
            int(summary["episode"]): summary for summary in manifest["episode_summaries"]
        }
        accepted_set = set(map(int, accepted))
        accepted_episodes = sorted({int(dataset["episode"][index]) for index in accepted})
        if episode_filter is not None:
            requested = set(map(int, episode_filter))
            accepted_episodes = [
                episode for episode in accepted_episodes if episode in requested
            ]
        labels = []
        errors = []
        for episode in accepted_episodes:
            try:
                contract_sha256 = str(
                    summaries[episode].get("experiment_contract_sha256", "")
                )
                if contract_sha256 not in contracts:
                    raise ValueError(
                        f"episode {episode} references unknown experiment contract "
                        f"{contract_sha256!r}"
                    )
                labels.extend(_process_episode(
                    dataset,
                    path,
                    episode,
                    summaries[episode],
                    accepted_set,
                    max_rollout_steps,
                    horizons,
                    parity_tolerance,
                    failure_cost,
                    contracts[contract_sha256],
                    source_sha256,
                ))
            except Exception as exc:
                error = {
                    "source_file": path,
                    "episode": int(episode),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(error)
                print(
                    f"failed {Path(path).name} episode={episode}: {error['error']}",
                    flush=True,
                )
        return {"labels": labels, "errors": errors}


def _summary(labels: list[dict]) -> dict:
    by_scenario = defaultdict(list)
    for label in labels:
        by_scenario[label["target_scenario"]].append(label)
    gains = np.asarray([label["ttt_gain"] for label in labels], dtype=float)
    return {
        "labels": len(labels),
        "long_horizon_positive": int(sum(label["long_horizon_positive"] for label in labels)),
        "h3_gate_false_positive": int(sum(label["h3_gate_false_positive"] for label in labels)),
        "termination_reasons": dict(Counter(label["termination_reason"] for label in labels)),
        "ttt_gain_mean": float(gains.mean()) if gains.size else None,
        "ttt_gain_min": float(gains.min()) if gains.size else None,
        "ttt_gain_max": float(gains.max()) if gains.size else None,
        "by_scenario": {
            scenario: {
                "labels": len(rows),
                "positive": int(sum(row["long_horizon_positive"] for row in rows)),
                "false_positive": int(sum(row["h3_gate_false_positive"] for row in rows)),
                "ttt_gain_mean": float(np.mean([row["ttt_gain"] for row in rows])),
            }
            for scenario, rows in sorted(by_scenario.items())
        },
    }


def _write_outputs(
    path: Path,
    labels: list[dict],
    inputs: list[str],
    args,
    errors: list[dict] | None = None,
    experiment_contracts: dict | None = None,
) -> None:
    result = {
        "format_version": "long_horizon_counterfactual_labels_v1",
        "label_contract": LABEL_CONTRACT,
        "experiment_contracts": dict(experiment_contracts or {}),
        "inputs": inputs,
        "max_rollout_steps": int(args.max_rollout_steps),
        "requested_horizons": list(args.horizons),
        "summary": _summary(labels),
        "errors": list(errors or []),
        "labels": labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    horizon_keys = sorted({key for label in labels for key in label["horizons"]}, key=int)
    csv_path = path.with_suffix(".csv")
    fields = [
        "source_file", "source_row_index", "episode", "step", "simulation_time_sec",
        "target_scenario", "termination_reason", "original_h3_gate_gain",
        "rollout_steps", "ttt_gain", "required_gain", "terminal_inventory_delta",
        "long_horizon_positive", "h3_gate_false_positive", "residual_l2",
        "residual_abs_max", "replay_observation_error", "replay_anchor_error",
        "replay_reward_error", "replay_reward_error_raw",
        "replay_reward_external_adjustment",
    ]
    for key in horizon_keys:
        fields.extend((f"gain_h{key}", f"inventory_delta_h{key}", f"positive_h{key}"))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            row = {key: label.get(key) for key in fields}
            for key, values in label["horizons"].items():
                row[f"gain_h{key}"] = values["ttt_gain"]
                row[f"inventory_delta_h{key}"] = values["terminal_inventory_delta"]
                row[f"positive_h{key}"] = values["long_horizon_positive"]
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-rollout-steps", type=int, default=0,
        help="0 rolls each branch to the end of the 14,400-second simulation",
    )
    parser.add_argument("--horizons", default="3,6,12,24")
    parser.add_argument("--max-events-per-file", type=int, default=0)
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--resume-errors", action="store_true",
        help="retry only file/episode errors recorded in the existing output",
    )
    args = parser.parse_args(argv)
    args.horizons = tuple(sorted({
        int(value) for value in args.horizons.split(",") if value.strip()
    }))
    patterns = [value.strip() for value in args.data.split(",") if value.strip()]
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no files matched: {args.data}")
    active = []
    experiment_contracts = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as dataset:
            if accepted_local_indices(dataset).size:
                manifest = _manifest(dataset)
                verified = verify_contract_map(
                    manifest.get("experiment_contracts", {})
                )
                if not verified:
                    raise SystemExit(
                        f"{path} is missing verified experiment contracts"
                    )
                for sha256, contract in verified.items():
                    existing = experiment_contracts.get(sha256)
                    if existing is not None and existing != contract.payload:
                        raise SystemExit(f"contract collision for {sha256}")
                    experiment_contracts[sha256] = contract.payload
                active.append(path)
    input_paths = list(active)
    labels = []
    errors = []
    retry_episodes: dict[str, tuple[int, ...] | None] = {}
    output = Path(args.out)
    if args.resume_errors:
        if not output.exists():
            raise SystemExit(f"cannot resume missing label output: {output}")
        previous = json.loads(output.read_text(encoding="utf-8"))
        labels = list(previous.get("labels", []))
        previous_errors = list(previous.get("errors", []))
        path_lookup = {str(Path(path).resolve()).casefold(): path for path in paths}
        unmatched_errors = []
        grouped: dict[str, set[int] | None] = {}
        for error in previous_errors:
            source = error.get("source_file")
            path = path_lookup.get(str(Path(source).resolve()).casefold()) if source else None
            if path is None:
                unmatched_errors.append(error)
                continue
            episode = error.get("episode")
            if episode is None:
                grouped[path] = None
            elif path not in grouped or grouped[path] is not None:
                grouped.setdefault(path, set()).add(int(episode))
        if not grouped:
            raise SystemExit("existing label output contains no retryable errors")
        errors = unmatched_errors
        retry_episodes = {
            path: None if episodes is None else tuple(sorted(episodes))
            for path, episodes in grouped.items()
        }
        active = [path for path in active if path in retry_episodes]
        retained = []
        for label in labels:
            source = path_lookup.get(
                str(Path(label["source_file"]).resolve()).casefold()
            )
            episodes = retry_episodes.get(source, ())
            retry_label = source in retry_episodes and (
                episodes is None or int(label["episode"]) in episodes
            )
            if not retry_label:
                retained.append(label)
        labels = retained
        print(
            f"resume errors: files={len(active)} retained_labels={len(labels)}",
            flush=True,
        )
    process_args = (
        int(args.max_rollout_steps), args.horizons,
        float(args.parity_tolerance), int(args.max_events_per_file),
    )
    if args.workers <= 1:
        for path in active:
            try:
                result = _process_file(
                    path, *process_args, retry_episodes.get(path)
                )
                labels.extend(result["labels"])
                errors.extend(result["errors"])
            except Exception as exc:
                errors.append({"source_file": path, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {Path(path).name}: {type(exc).__name__}: {exc}", flush=True)
            labels.sort(key=lambda row: (
                row["source_file"], row["episode"], row["step"]
            ))
            _write_outputs(
                Path(args.out), labels, input_paths, args, errors,
                experiment_contracts,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(active))) as executor:
            futures = {
                executor.submit(
                    _process_file, path, *process_args, retry_episodes.get(path)
                ): path for path in active
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    rows = result["labels"]
                    labels.extend(rows)
                    errors.extend(result["errors"])
                    print(
                        f"completed {Path(path).name}: labels={len(rows)} "
                        f"errors={len(result['errors'])}",
                        flush=True,
                    )
                except Exception as exc:
                    errors.append({
                        "source_file": path,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"failed {Path(path).name}: {type(exc).__name__}: {exc}", flush=True)
                labels.sort(key=lambda row: (
                    row["source_file"], row["episode"], row["step"]
                ))
                _write_outputs(
                    Path(args.out), labels, input_paths, args, errors,
                    experiment_contracts,
                )
    labels.sort(key=lambda row: (
        row["source_file"], row["episode"], row["step"]
    ))
    _write_outputs(
        output, labels, input_paths, args, errors, experiment_contracts
    )
    summary = _summary(labels)
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        print(json.dumps({"errors": errors}, indent=2), flush=True)
    print(f"saved long-horizon labels -> {output}", flush=True)


if __name__ == "__main__":
    main()
