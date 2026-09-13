"""Full 75-transition, resumable five-scenario collection/evaluation worker.

Completed work returns a validated summary. STOP or an invocation wall budget
raises InterruptedError after the latest exact checkpoint; integrity/configuration
failures raise ValueError. Neither path is reported as a successful episode.
The historical experiment runtime is imported without modifying its files.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_catalog import StructuredActionCatalog
from rl_leader.response_dqn_collect import (
    COLLECTOR_FORMAT, collect_sequential_episode, ensemble_greedy_policy, random_masked_policy,
    rows_to_replay, _save_replay_atomic, _shutdown_response_process_pools,
)
from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.response_continuation_state import (
    FIVE_CELL_SHARED_CONTRACT_SHA256, validate_five_cell_contract,
)
from rl_leader.run_sequential_response_ddqn import _atomic_json, _atomic_pickle
from rl_leader.train_response_dqn import _configure_torch_threads


SCENARIOS = (
    "sweet_155_w60", "sweet_170_w60", "sweet_170_incident_w60",
    "sweet_170_skew15_w60", "sweet_190_w60",
)
FORMAT = "five_cell_full_episode_v1"
MODE = "post_commit_continuation_five_cell_v1"
TRANSITIONS = 75


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hashes(expected: dict, label: str) -> None:
    for name, digest in expected.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{label} hash mismatch: {path}")


def _payload(payload: dict) -> dict:
    value = dict(payload)
    for key in ("first_action", "snapshot", "reference_trace", "evaluate_response_steps",
                "step_gate", "preview_action_ids", "full_run"):
        if key in value:
            raise ValueError(f"full-run worker does not accept {key}")
    value.setdefault("purpose", "collection")
    value.setdefault("t_total", 14400.0)
    value.setdefault("response_equivalence_mode", MODE)
    value.setdefault("epsilon", 0.0)
    value.setdefault("checkpoints", [])
    value.setdefault("expected_model_hashes", {})
    for key in ("directory", "scenario", "experiment_contract_sha256", "catalog", "seed",
                "episode", "response_workers", "max_wall_seconds", "stop_file",
                "expected_source_hashes"):
        if key not in value:
            raise ValueError(f"missing worker payload field: {key}")
    if value["scenario"] not in SCENARIOS or float(value["t_total"]) != 14400.0:
        raise ValueError("worker requires a named five-cell scenario and T_total=14400")
    if value["purpose"] not in {"collection", "evaluation", "baseline"}:
        raise ValueError("invalid episode purpose")
    epsilon = float(value["epsilon"])
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be finite and in [0,1]")
    if value["purpose"] == "evaluation" and (epsilon != 0 or not value["checkpoints"]):
        raise ValueError("evaluation requires checkpoints and epsilon=0")
    if value["purpose"] == "baseline" and (epsilon != 0 or value["checkpoints"]):
        raise ValueError("baseline requires no checkpoints and epsilon=0")
    for key in ("seed", "episode", "response_workers"):
        if isinstance(value[key], bool) or int(value[key]) != value[key] or int(value[key]) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not 1 <= value["response_workers"] <= 8:
        raise ValueError("response_workers must be in [1,8]")
    budget = float(value["max_wall_seconds"])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("max_wall_seconds must be finite and positive")
    if value["response_equivalence_mode"] != MODE:
        raise ValueError("unsupported response equivalence mode")
    digest = str(value["experiment_contract_sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid experiment contract SHA256")
    if not isinstance(value["expected_source_hashes"], dict) or not value["expected_source_hashes"]:
        raise ValueError("nonempty expected_source_hashes are required")
    if not isinstance(value["expected_model_hashes"], dict):
        raise ValueError("expected_model_hashes must be a dictionary")
    models = [str(Path(path).resolve()) for path in value["checkpoints"]]
    pins = {str(Path(path).resolve()): digest
            for path, digest in value["expected_model_hashes"].items()}
    if len(models) != len(set(models)) or set(models) != set(pins):
        raise ValueError("model pins must cover exactly the unique checkpoints")
    value["checkpoints"] = models
    value["expected_model_hashes"] = pins
    value["expected_source_hashes"] = {
        str(Path(path).resolve()): digest
        for path, digest in value["expected_source_hashes"].items()
    }
    value["directory"] = str(Path(value["directory"]).resolve())
    value["stop_file"] = str(Path(value["stop_file"]).resolve())
    # JSON also rejects nonserializable payload fields and NaN configuration.
    json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def _signature(payload: dict) -> str:
    # The invocation budget may change on resume; it is not policy semantics.
    semantic = {key: value for key, value in payload.items()
                if key not in {"max_wall_seconds", "stop_file"}}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _assert_env(env, payload: dict, catalog) -> None:
    if (env.scenario_name != payload["scenario"] or float(env.T_total) != 14400.0
            or env.warmup != 5 or env.n_steps != 80):
        raise ValueError("environment scenario, horizon, or warmup mismatch")
    if env.experiment_contract_fingerprint != payload["experiment_contract_sha256"]:
        raise ValueError("environment experiment contract mismatch")
    if tuple(env.action_schema.names) != tuple(catalog.action_names):
        raise ValueError("catalog coordinates differ from the environment action schema")
    if getattr(env, "response_equivalence_mode", "legacy_follower_runtime_v1") != payload["response_equivalence_mode"]:
        raise ValueError("checkpoint response equivalence changed")
    if getattr(env, "response_expected_contract_sha256", None) != payload["experiment_contract_sha256"]:
        raise ValueError("checkpoint explicit scene contract changed")
    validate_five_cell_contract(env, expected_contract_sha256=payload["experiment_contract_sha256"])


def _validate_replay(replay, payload: dict, catalog, *, complete: bool) -> None:
    replay.validate()
    manifest = replay.manifest
    required = {
        "catalog_fingerprint": catalog.fingerprint,
        "scenario": payload["scenario"],
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "reward_semantics": "interval_negative_ttt",
        "done_semantics": "environment_terminal",
        "t_total_sec": 14400.0,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"replay {key} mismatch")
    if manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1") != payload["response_equivalence_mode"]:
        raise ValueError("replay response equivalence mismatch")
    expected_scope = {
        "version": MODE, "scenario": payload["scenario"],
        "expected_contract_sha256": payload["experiment_contract_sha256"],
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "shared_contract_sha256": FIVE_CELL_SHARED_CONTRACT_SHA256,
    }
    if manifest.get("continuation_contract_scope") != expected_scope:
        raise ValueError("replay continuation contract scope mismatch")
    if not 1 <= replay.size <= TRANSITIONS or not np.array_equal(replay.control_step, np.arange(replay.size)):
        raise ValueError("replay must be an unbroken full-run prefix starting at zero")
    if not np.all(replay.episode == payload["episode"]) or not np.all(replay.option_steps == 1):
        raise ValueError("replay episode or one-step transition mismatch")
    if np.any(replay.done[:-1]) or bool(replay.done[-1]) != (replay.size == TRANSITIONS):
        raise ValueError("only full-run transition 74 may be terminal")
    if complete and replay.size != TRANSITIONS:
        raise ValueError("completed episode must contain all 75 transitions")
    for following, current in ((replay.next_observation, replay.observation),
                               (replay.next_action_mask, replay.action_mask),
                               (replay.next_response_features, replay.response_features)):
        if not np.array_equal(following[:-1], current[1:]):
            raise ValueError("replay transition chain is disconnected")


def _validate_trace(directory: Path, rows, payload: dict, catalog) -> None:
    """Verify committed rows against this run's trace, including resumed duplicates."""
    count = 0 if rows is None else len(rows["action_id"])
    if not count:
        return
    seen = {}
    for path in sorted(directory.glob("trace*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                step = item["control_step"]
                if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < TRANSITIONS:
                    raise ValueError("invalid trace control step")
                if (item.get("format_version") != COLLECTOR_FORMAT
                        or item.get("episode") != payload["episode"]
                        or item.get("catalog_fingerprint") != catalog.fingerprint):
                    raise ValueError("trace episode or catalog mismatch")
                identity = (item["state_id"], item["selected_action_id"], float(item["interval_reward"]))
                if not isinstance(identity[0], str) or not identity[0] or not math.isfinite(identity[2]):
                    raise ValueError("invalid trace state or reward")
                if step in seen and seen[step] != identity:
                    raise ValueError("resumed trace duplicates disagree on state/action/reward")
                seen[step] = identity
                if item.get("validity_gate_pass") is not True:
                    raise ValueError(f"validity gate failed at controlled step {step}")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"malformed episode trace: {path}") from exc
    for step in range(count):
        if step not in seen:
            raise ValueError(f"missing committed trace step {step}")
        _, action, reward = seen[step]
        if action != rows["action_id"][step] or reward != rows["reward"][step]:
            raise ValueError(f"trace disagrees with checkpoint row {step}")


def collect_five_cell_episode(payload: dict, *, max_new_transitions: int | None = None) -> dict:
    """Return a verified completed summary, or raise on pause/integrity failure."""
    try:
        if max_new_transitions is not None and (
                isinstance(max_new_transitions, bool) or int(max_new_transitions) != max_new_transitions
                or max_new_transitions < 1):
            raise ValueError("max_new_transitions must be a positive integer")
        return _collect_five_cell_episode(payload, max_new_transitions=max_new_transitions)
    finally:
        _shutdown_response_process_pools(wait=True)


def _collect_five_cell_episode(original: dict, *, max_new_transitions=None) -> dict:
    payload = _payload(original)
    _verify_hashes(payload["expected_source_hashes"], "source")
    _verify_hashes(payload["expected_model_hashes"], "model")
    _configure_torch_threads(1)
    catalog = StructuredActionCatalog.from_manifest(payload["catalog"])
    signature = _signature(payload)
    directory = Path(payload["directory"])
    paths = {name: directory / filename for name, filename in {
        "summary": "summary.json", "replay": "replay.npz", "checkpoint": "checkpoint.pkl",
        "progress": "progress.json", "config": "run_config.json",
    }.items()}
    started = time.perf_counter()

    def pause_reason():
        if Path(payload["stop_file"]).exists():
            return "STOP requested"
        if time.perf_counter() - started >= payload["max_wall_seconds"]:
            return "invocation wall budget reached"
        return None

    if pause_reason():
        raise InterruptedError("STOP requested before environment initialization")
    directory.mkdir(parents=True, exist_ok=True)
    if paths["config"].exists():
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        if config.get("signature") != signature:
            raise ValueError("episode configuration changed")
    else:
        if any(paths[name].exists() for name in ("summary", "replay", "checkpoint", "progress")):
            raise ValueError("episode artifacts lack their original configuration")
        _atomic_json(paths["config"], {"format_version": FORMAT, "signature": signature, "payload": payload})

    def read_checkpoint():
        if not paths["progress"].is_file() or not paths["checkpoint"].is_file():
            raise ValueError("partial episode has no verified resumable checkpoint")
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if progress.get("signature") != signature:
            raise ValueError("checkpoint configuration changed")
        if sha256_file(paths["checkpoint"]) != progress.get("checkpoint_sha256"):
            raise ValueError("checkpoint hash mismatch")
        replay_sha = progress.get("replay_sha256")
        if replay_sha is not None:
            if not paths["replay"].is_file() or sha256_file(paths["replay"]) != replay_sha:
                raise ValueError("checkpoint replay hash mismatch")
        elif paths["replay"].exists():
            raise ValueError("unexpected replay beside an empty checkpoint")
        with paths["checkpoint"].open("rb") as handle:
            saved = pickle.load(handle)
        if saved.get("signature") != signature:
            raise ValueError("pickled checkpoint configuration changed")
        return saved, progress

    if paths["checkpoint"].exists() or paths["progress"].exists():
        saved, progress = read_checkpoint()
        env, rows = saved["env"], saved["rows"]
        observation = saved["observation"]
        prefix_ttt, elapsed_before = saved["prefix_ttt"], saved["wall_seconds"]
        if saved.get("start_step") != 0 or not math.isfinite(prefix_ttt) or not math.isfinite(elapsed_before):
            raise ValueError("checkpoint full-run prefix metadata mismatch")
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng"]
        _assert_env(env, payload, catalog)
        if not np.array_equal(env._observe(), observation):
            raise ValueError("checkpoint observation differs from restored environment")
        count = 0 if rows is None else len(rows["action_id"])
        if count != env.step_idx - env.warmup or progress["transitions"] != count:
            raise ValueError("checkpoint step and row count mismatch")
        if count:
            _validate_trace(directory, rows, payload, catalog)
            replay = load_frozen_response_replay(paths["replay"])
            _validate_replay(replay, payload, catalog, complete=paths["summary"].exists())
            expected = rows_to_replay(rows, env=env, catalog=catalog, source=__name__)
            for name in expected.__dataclass_fields__:
                if name != "manifest" and not np.array_equal(getattr(expected, name), getattr(replay, name)):
                    raise ValueError(f"checkpoint rows disagree with replay: {name}")
        if paths["summary"].exists():
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            expected_summary = {
                "format_version": FORMAT, "signature": signature, "status": "complete",
                "checkpoint_sha256": progress["checkpoint_sha256"], "replay_sha256": progress["replay_sha256"],
                "transitions": TRANSITIONS, "terminal": True, "scope": "ungated_full_run",
                "scenario": payload["scenario"], "purpose": payload["purpose"],
                "experiment_contract_sha256": payload["experiment_contract_sha256"],
                "catalog_fingerprint": catalog.fingerprint, "response_equivalence_mode": MODE,
                "continuation_contract_scope": replay.manifest["continuation_contract_scope"],
                "start_control_step": 0, "end_control_step": TRANSITIONS - 1,
                "total_ttt": float(env.sim.total_ttt), "prefix_ttt": prefix_ttt,
                "reward_sum": float(np.sum(replay.reward, dtype=np.float64)),
                "action_support_counts": replay.action_support_counts().tolist(), "ttt_reconciled": True,
                "all_valid": True, "validity_failures": 0,
            }
            if any(summary.get(key) != value for key, value in expected_summary.items()):
                raise ValueError("completed episode summary integrity mismatch")
            if not np.isclose(prefix_ttt - expected_summary["reward_sum"], env.sim.total_ttt, rtol=1e-6, atol=1e-6):
                raise ValueError("completed episode rewards do not reconcile")
            return summary
    else:
        if any(paths[name].exists() for name in ("summary", "replay")) or any(directory.glob("trace*.jsonl")):
            raise ValueError("partial episode has no resumable checkpoint")
        env = RLLeaderEnv(
            scenario_name=payload["scenario"], T_total=14400.0, warmup_nc_steps=5,
            action_mode="full", mask="RL-FULL", pstack_anchor=True,
            action_parameterization="pstack_residual",
        )
        env.response_equivalence_mode = payload["response_equivalence_mode"]
        env.response_expected_contract_sha256 = payload["experiment_contract_sha256"]
        _assert_env(env, payload, catalog)
        if pause_reason():
            raise InterruptedError("STOP or wall budget reached before reset")
        observation = env.reset()
        if env.step_idx - env.warmup != 0:
            raise ValueError("normal reset must begin at controlled step zero")
        prefix_ttt = float(env.sim.total_ttt)
        elapsed_before, rows = 0.0, None
        rng = np.random.default_rng(payload["seed"])

    rows_before = 0 if rows is None else len(rows["action_id"])

    def checkpoint(current_rows, *, check_pause=True):
        nonlocal observation
        _verify_hashes(payload["expected_source_hashes"], "source")
        _verify_hashes(payload["expected_model_hashes"], "model")
        count = 0 if current_rows is None else len(current_rows["action_id"])
        replay_sha, terminal = None, False
        if count:
            _validate_trace(directory, current_rows, payload, catalog)
            replay = rows_to_replay(current_rows, env=env, catalog=catalog, source=__name__)
            _validate_replay(replay, payload, catalog, complete=False)
            _save_replay_atomic(replay, paths["replay"])
            replay_sha = sha256_file(paths["replay"])
            observation = np.asarray(current_rows["next_observation"][-1]).copy()
            terminal = bool(replay.done[-1])
        elapsed = elapsed_before + time.perf_counter() - started
        _atomic_pickle(paths["checkpoint"], {
            "format_version": FORMAT, "signature": signature, "env": env,
            "rows": current_rows, "observation": observation, "rng": rng.bit_generator.state,
            "start_step": 0, "prefix_ttt": prefix_ttt, "wall_seconds": elapsed,
        })
        progress = {
            "format_version": FORMAT, "signature": signature, "transitions": count,
            "last_control_step": count - 1, "terminal": terminal,
            "total_ttt_so_far": float(env.sim.total_ttt), "wall_seconds": elapsed,
            "checkpoint_sha256": sha256_file(paths["checkpoint"]), "replay_sha256": replay_sha,
        }
        _atomic_json(paths["progress"], progress)
        reason = pause_reason()
        if check_pause and reason and not terminal:
            raise InterruptedError(f"{reason}; exact nonterminal episode checkpoint saved")
        if (check_pause and not terminal and max_new_transitions is not None
                and count - rows_before >= max_new_transitions):
            raise InterruptedError("diagnostic transition limit; exact nonterminal checkpoint saved")
        return progress

    if rows is None:
        checkpoint(None)
    ensemble = [load_trained_response_dqn(path, expected_catalog_fingerprint=catalog.fingerprint)
                for path in payload["checkpoints"]]
    if any(getattr(model, "response_equivalence_mode", "legacy_follower_runtime_v1")
           != payload["response_equivalence_mode"] for model in ensemble):
        raise ValueError("model response equivalence mismatch")
    policy = (ensemble_greedy_policy(ensemble, exploration_epsilon=payload["epsilon"])
              if ensemble else random_masked_policy(anchor_probability=1.0 - payload["epsilon"]))

    def choose(evaluated, current_catalog, current_rng):
        reason = pause_reason()
        if reason:
            raise InterruptedError(f"{reason}; previous exact checkpoint retained")
        return policy(evaluated, current_catalog, current_rng)

    if rows is None or not rows["done"][-1]:
        segment = len(list(directory.glob("trace*.jsonl")))
        trace = directory / ("trace.jsonl" if segment == 0 else f"trace_resume_{segment:03d}.jsonl")
        while trace.exists():
            segment += 1
            trace = directory / f"trace_resume_{segment:03d}.jsonl"
        rows = collect_sequential_episode(
            env, catalog, choose, episode=payload["episode"], rng=rng,
            event_group=f"five-cell:{payload['scenario']}:{payload['episode']}:seed:{payload['seed']}",
            log_path=trace, response_workers=payload["response_workers"],
            response_backend="process", checkpoint_every=1, checkpoint_callback=checkpoint,
            initial_observation=observation, initial_rows=rows,
        )
    replay = rows_to_replay(rows, env=env, catalog=catalog, source=__name__)
    _validate_replay(replay, payload, catalog, complete=True)
    total_ttt = float(env.sim.total_ttt)
    reward_sum = float(np.sum(replay.reward, dtype=np.float64))
    error = prefix_ttt - reward_sum - total_ttt
    if not np.isfinite(total_ttt) or not np.isclose(prefix_ttt - reward_sum, total_ttt, rtol=1e-6, atol=1e-6):
        raise ValueError("interval rewards do not reconcile to simulator TTT")
    progress = checkpoint(rows, check_pause=False)
    summary = {
        "format_version": FORMAT, "status": "complete", "signature": signature,
        "scenario": payload["scenario"], "purpose": payload["purpose"],
        "scope": "ungated_full_run", "terminal": True,
        "total_ttt": total_ttt, "prefix_ttt": prefix_ttt,
        "start_control_step": 0, "end_control_step": TRANSITIONS - 1, "transitions": replay.size,
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "catalog_fingerprint": catalog.fingerprint,
        "response_equivalence_mode": payload["response_equivalence_mode"],
        "continuation_contract_scope": replay.manifest["continuation_contract_scope"],
        "checkpoint": str(paths["checkpoint"]), "replay": str(paths["replay"]),
        "checkpoint_sha256": progress["checkpoint_sha256"], "replay_sha256": progress["replay_sha256"],
        "reward_sum": reward_sum, "ttt_reconciliation_error": error, "ttt_reconciled": True,
        "all_valid": True, "validity_failures": 0,
        "action_support_counts": replay.action_support_counts().tolist(),
        "wall_seconds": progress["wall_seconds"], "response_preview": True,
        "lcb_guard": False, "epsilon": payload["epsilon"],
        "terminal_inventory": float(env._inventory()),
        "trace_segments": sorted(str(path) for path in directory.glob("trace*.jsonl")),
    }
    _atomic_json(paths["summary"], summary)
    return summary
