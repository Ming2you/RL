"""Resumable collect -> sequential DDQN -> closed-loop evaluation pilot.

Snapshots shorten collection prefixes only. Every collected suffix reaches the
original environment terminal time and retains its one-step transitions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_ddqn_recovery import EnvSnapshot, restore_env_snapshot
from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_catalog import (
    StructuredActionCatalog, build_structured_action_catalog, load_extra_action_specs,
)
from rl_leader.response_dqn_collect import (
    CollectionDecision, collect_sequential_episode, ensemble_greedy_policy,
    random_masked_policy, rows_to_replay, _save_replay_atomic,
    _shutdown_response_process_pools,
)
from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay


class _SnapshotUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Historical `python -m` caches recorded the dataclass under __main__.
        if module == "__main__" and name == "EnvSnapshot":
            return EnvSnapshot
        return super().find_class(module, name)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_pickle(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp.pkl")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_verified_snapshot(config: dict) -> EnvSnapshot:
    with Path(config["snapshot"]).open("rb") as handle:
        snapshot = _SnapshotUnpickler(handle).load()
    if not isinstance(snapshot, EnvSnapshot):
        raise ValueError("input is not an environment snapshot")
    env, observation = restore_env_snapshot(snapshot)
    if env.scenario_name != config["scenario"] or env.T_total != config["t_total"]:
        raise ValueError("snapshot scenario or horizon mismatch")
    if snapshot.experiment_contract_sha256 != config["experiment_contract_sha256"]:
        raise ValueError("snapshot experiment contract mismatch")
    with Path(config["reference_trace"]).open(encoding="utf-8") as handle:
        expected = {int(row["control_step"]): row["state_id"] for row in map(json.loads, handle)}
    if expected.get(snapshot.control_step) != snapshot.state_fingerprint:
        raise ValueError("snapshot is not the requested reference policy state")
    np.testing.assert_array_equal(env._observe(), observation)
    return snapshot


def build_pilot_catalog(config: dict, snapshot: EnvSnapshot) -> StructuredActionCatalog:
    source = build_structured_action_catalog(
        snapshot.env.action_schema.names,
        magnitudes=(0.25, 0.5, 0.75),
        families=("linear", "quadratic", "cross", "combo"),
        domains=("freeway",), owners=("R_F_W",),
        extra_actions=load_extra_action_specs(config["extra_actions"]),
    )
    by_key = {action.key: action for action in source.actions}
    keys = config["action_keys"]
    if not keys or keys[0] != "anchor" or len(keys) != len(set(keys)):
        raise ValueError("pilot catalog requires unique keys starting with anchor")
    return StructuredActionCatalog(source.action_names, [
        replace(by_key[key], action_id=index) for index, key in enumerate(keys)
    ])


def validate_complete_episode(replay) -> None:
    validate_sequential_td_replay(replay, gamma=1.0)
    if int(replay.done.sum()) != 1 or replay.done[-1] != 1:
        raise ValueError("episode must reach exactly one real terminal transition")
    if not np.all(np.diff(replay.control_step) == 1):
        raise ValueError("episode control steps are not contiguous")
    np.testing.assert_array_equal(replay.next_observation[:-1], replay.observation[1:])
    np.testing.assert_array_equal(replay.next_action_mask[:-1], replay.action_mask[1:])
    np.testing.assert_array_equal(replay.next_response_features[:-1], replay.response_features[1:])


def _actor_stop_requested(payload: dict) -> bool:
    return any(Path(path).exists() for path in (
        payload["stop_file"], *payload.get("additional_stop_files", []),
    ))


def collect_actor(payload: dict, *, max_new_transitions: int | None = None) -> dict:
    _configure_torch_threads(1)
    directory = Path(payload["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    replay_path = directory / "replay.npz"
    checkpoint_path = directory / "checkpoint.pkl"
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") != signature:
            raise ValueError("completed episode configuration changed")
        validate_complete_episode(load_frozen_response_replay(replay_path))
        return summary
    config = payload["config"]
    catalog = StructuredActionCatalog.from_manifest(payload["catalog"])
    started = time.perf_counter()
    elapsed_before = 0.0
    rng = np.random.default_rng(payload["seed"])
    rows = None
    trace_path = directory / "trace.jsonl"
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["signature"] != signature:
            raise ValueError("checkpoint run configuration changed")
        env = saved["env"]
        observation = saved["observation"]
        rows = saved["rows"]
        rng.bit_generator.state = saved["rng"]
        elapsed_before = saved["wall_seconds"]
        start_step, prefix_ttt = saved["start_step"], saved["prefix_ttt"]
        # Preserve any trace tail written just before an interrupted checkpoint.
        segment = 1
        trace_path = directory / f"trace_resume_{segment:03d}.jsonl"
        while trace_path.exists():
            segment += 1
            trace_path = directory / f"trace_resume_{segment:03d}.jsonl"
    else:
        if replay_path.exists() or (directory / "trace.jsonl").exists():
            raise ValueError("partial episode has no resumable checkpoint")
        snapshot = load_verified_snapshot(config)
        if payload.get("full_run", False):
            env = RLLeaderEnv(
                scenario_name=config["scenario"], T_total=config["t_total"],
                action_mode="full", mask="RL-FULL", pstack_anchor=True,
                action_parameterization="pstack_residual",
            )
            observation = env.reset()
        else:
            env, observation = restore_env_snapshot(snapshot)
        start_step = int(env.step_idx - env.warmup)
        prefix_ttt = float(env.sim.total_ttt)
    rows_before = 0 if rows is None else len(rows["action_id"])
    mode = payload.get("response_equivalence_mode", "legacy_follower_runtime_v1")
    if rows is not None and getattr(env, "response_equivalence_mode", "legacy_follower_runtime_v1") != mode:
        raise ValueError("checkpoint response equivalence changed")
    if mode != "legacy_follower_runtime_v1":
        env.response_equivalence_mode = mode
    np.testing.assert_array_equal(env._observe(), observation)
    if env.experiment_contract_fingerprint != config["experiment_contract_sha256"]:
        raise ValueError("resumed environment contract mismatch")
    ensemble = [load_trained_response_dqn(path, expected_catalog_fingerprint=catalog.fingerprint)
                for path in payload["checkpoints"]]
    policy = (ensemble_greedy_policy(ensemble, exploration_epsilon=payload["epsilon"])
              if ensemble else random_masked_policy(anchor_probability=1.0 - payload["epsilon"]))

    def choose(evaluated, current_catalog, current_rng):
        forced = payload.get("first_action")
        if forced is not None and env.step_idx - env.warmup == start_step:
            representative = int(evaluated.response_mask.representative_of[int(forced)])
            if representative < 0:
                raise ValueError("initial exploratory action is infeasible")
            return CollectionDecision(representative, {
                "policy": "initial_branch_coverage", "exploratory": True,
                "requested_action_id": int(forced),
            })
        return policy(evaluated, current_catalog, current_rng)

    def checkpoint(current_rows):
        replay = rows_to_replay(current_rows, env=env, catalog=catalog, source=__name__)
        _save_replay_atomic(replay, replay_path)
        elapsed = elapsed_before + time.perf_counter() - started
        _atomic_pickle(checkpoint_path, {
            "signature": signature, "env": env, "rows": current_rows,
            "observation": current_rows["next_observation"][-1],
            "rng": rng.bit_generator.state, "start_step": start_step,
            "prefix_ttt": prefix_ttt, "wall_seconds": elapsed,
        })
        _atomic_json(directory / "progress.json", {
            "transitions": replay.size, "last_control_step": int(replay.control_step[-1]),
            "terminal": bool(replay.done[-1]), "total_ttt_so_far": float(env.sim.total_ttt),
            "wall_seconds": elapsed,
        })
        if _actor_stop_requested(payload) and not replay.done[-1]:
            raise InterruptedError("STOP requested; exact episode checkpoint saved")
        if max_new_transitions is not None and replay.size - rows_before >= max_new_transitions:
            raise InterruptedError("diagnostic pause; nonterminal checkpoint saved")

    if rows is None or not rows["done"][-1]:
        rows = collect_sequential_episode(
            env, catalog, choose, episode=payload["episode"], rng=rng,
            event_group=f"sequential:{payload['episode']}:seed:{payload['seed']}",
            log_path=trace_path, response_workers=payload["response_workers"],
            response_backend="process", checkpoint_every=1, checkpoint_callback=checkpoint,
            initial_observation=observation, initial_rows=rows,
        )
    replay = rows_to_replay(rows, env=env, catalog=catalog, source=__name__)
    validate_complete_episode(replay)
    _save_replay_atomic(replay, replay_path)
    total_ttt = float(env.sim.total_ttt)
    if not np.isclose(prefix_ttt - np.sum(replay.reward, dtype=np.float64), total_ttt, rtol=1e-6):
        raise ValueError("interval reward sum does not reconcile to simulator TTT")
    summary = {
        "signature": signature,
        "total_ttt": total_ttt, "prefix_ttt": prefix_ttt, "start_control_step": start_step,
        "end_control_step": int(replay.control_step[-1]), "transitions": replay.size,
        "scope": "ungated_full_run" if payload.get("full_run") else "fixed_prefix_continuation",
        "pstack_improvement_percent": 100 * (config["pstack_total_ttt"] - total_ttt) / config["pstack_total_ttt"],
        "wall_seconds": elapsed_before + time.perf_counter() - started,
        "response_preview": True, "lcb_guard": False, "epsilon": payload["epsilon"],
        "terminal_inventory": float(env._inventory()), "first_action": payload.get("first_action"),
        "catalog_fingerprint": catalog.fingerprint, "replay": str(replay_path),
        "trace_segments": sorted(str(path) for path in directory.glob("trace*.jsonl")),
        "action_support_counts": replay.action_support_counts().tolist(),
    }
    _atomic_json(summary_path, summary)
    return summary


def _collect_actor_worker(payload: dict) -> dict:
    try:
        return collect_actor(payload)
    finally:
        # multiprocessing joins children before ordinary atexit callbacks run.
        _shutdown_response_process_pools(wait=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if config["actors"] < 1 or config["response_workers"] < 1 or not 1 <= config["actors"] * config["response_workers"] <= 8:
        raise ValueError("total actor x response worker budget must be in [1,8]")
    if config["rounds"] < 1 or config["gradient_steps"] < 1 or not 0 <= config["epsilon"] <= 1:
        raise ValueError("invalid pilot round, training, or exploration configuration")
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    snapshot = load_verified_snapshot(config)
    catalog = build_pilot_catalog(config, snapshot)
    resolved = {**config, "catalog_fingerprint": catalog.fingerprint}
    config_path = root / "resolved_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != resolved:
        raise ValueError("output directory belongs to a different configuration")
    _atomic_json(config_path, resolved)
    _atomic_json(root / "catalog.json", catalog.as_manifest())
    if args.validate_only:
        print(json.dumps({"validated": True, "catalog_size": catalog.size,
                          "start_step": snapshot.control_step, "worker_budget": 8}), flush=True)
        return
    stop_file = root / "STOP"
    replay_paths = []
    checkpoints = []
    status_path = root / "status.json"
    for cycle in range(config["rounds"]):
        if stop_file.exists():
            _atomic_json(status_path, {"phase": "paused", "round": cycle})
            return
        directory = root / f"round_{cycle:02d}"
        payloads = [{
            "config": config, "catalog": catalog.as_manifest(),
            "directory": str(directory / f"actor_{actor:02d}"),
            "seed": config["seed"] + cycle * config["actors"] + actor,
            "episode": cycle * config["actors"] + actor,
            "response_workers": config["response_workers"], "checkpoints": checkpoints,
            "epsilon": config["epsilon"], "stop_file": str(stop_file),
            "first_action": (1 if actor == 0 else 0) if cycle == 0 else None,
            "full_run": bool(cycle > 0 and actor == 0),
        } for actor in range(config["actors"])]
        _shutdown_response_process_pools(wait=True)
        _atomic_json(status_path, {"phase": "collecting", "round": cycle})
        with ProcessPoolExecutor(max_workers=config["actors"]) as pool:
            for future in as_completed([pool.submit(_collect_actor_worker, payload) for payload in payloads]):
                try:
                    print(json.dumps({"event": "episode_complete", **future.result()}), flush=True)
                except InterruptedError:
                    _atomic_json(status_path, {"phase": "paused", "round": cycle})
                    return
        replay_paths.extend(Path(payload["directory"]) / "replay.npz" for payload in payloads)
        if stop_file.exists():
            _atomic_json(status_path, {"phase": "paused", "round": cycle})
            return
        merged = merge_frozen_response_replays(
            [load_frozen_response_replay(path) for path in replay_paths], source=__name__,
        )
        validate_sequential_td_replay(merged, gamma=1.0)
        data_path = directory / "training_replay.npz"
        _save_replay_atomic(merged, data_path)
        model_dir = directory / "model"
        _atomic_json(status_path, {"phase": "training", "round": cycle, "transitions": merged.size})
        if not (model_dir / "ensemble_manifest.json").exists():
            # A unique attempt directory preserves interrupted member outputs.
            attempt = 0
            while model_dir.exists():
                attempt += 1
                model_dir = directory / f"model_attempt_{attempt:02d}"
                if (model_dir / "ensemble_manifest.json").exists():
                    break
            if not (model_dir / "ensemble_manifest.json").exists():
                subprocess.run([
                    sys.executable, "-B", "-m", "rl_leader.train_response_dqn",
                    "--data", str(data_path), "--output-dir", str(model_dir),
                    "--ensemble-size", "3", "--workers", "3", "--member-threads", "1",
                    "--gradient-steps", str(config["gradient_steps"]), "--batch-size", "32",
                    "--hidden", "128,128", "--target-update-interval", "100",
                    "--learning-rate", "0.0001", "--gamma", "1", "--reward-scale", "0.01",
                    "--min-action-support", "1", "--require-sequential-td",
                    "--no-group-bootstrap", "--seed", str(config["seed"] + cycle * 10),
                ], check=True)
        checkpoints = sorted(str(path) for path in model_dir.glob("response_dqn_member_*.pt"))
        if len(checkpoints) != 3:
            raise ValueError("training did not produce all ensemble members")
        if stop_file.exists():
            _atomic_json(status_path, {"phase": "paused", "round": cycle})
            return
        _atomic_json(status_path, {"phase": "evaluating", "round": cycle, "model_dir": str(model_dir)})
        try:
            result = collect_actor({
                **payloads[0], "directory": str(directory / "evaluation"), "epsilon": 0.0,
                "episode": 10000 + cycle, "checkpoints": checkpoints, "first_action": None,
                "response_workers": 8, "full_run": False,
            })
        except InterruptedError:
            _atomic_json(status_path, {"phase": "paused", "round": cycle})
            return
        finally:
            _shutdown_response_process_pools(wait=True)
        _atomic_json(directory / "round_summary.json", result)
        print(json.dumps({"event": "round_complete", "round": cycle, **result}), flush=True)
    if config.get("final_full_run", True) and not stop_file.exists():
        _atomic_json(status_path, {"phase": "ungated_full_run", "model_dir": str(model_dir)})
        try:
            result = collect_actor({
                **payloads[0], "directory": str(root / "ungated_full_run"), "epsilon": 0.0,
                "episode": 20000, "checkpoints": checkpoints, "first_action": None,
                "response_workers": 8, "full_run": True,
            })
        except InterruptedError:
            _atomic_json(status_path, {"phase": "paused", "round": config["rounds"]})
            return
        finally:
            _shutdown_response_process_pools(wait=True)
        _atomic_json(root / "full_run_summary.json", result)
    _atomic_json(status_path, {"phase": "pilot_complete", "rounds": config["rounds"],
                              "model_dir": str(model_dir)})


if __name__ == "__main__":
    main()
