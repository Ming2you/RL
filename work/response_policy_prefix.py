"""Resumable, trace-verified pre-action snapshots without candidate previews."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from uuid import uuid4

import numpy as np

from rl_leader.env import RLLeaderEnv
from rl_leader.response_ddqn_recovery import (
    EnvSnapshot, capture_env_snapshot, commit_catalog_action_from_anchor, restore_env_snapshot,
)
from rl_leader.response_dqn_collect import _encoded_continuation
from rl_leader.response_dqn_mask import CONTINUATION_EQUIVALENCE
from rl_leader.run_sequential_response_ddqn import _atomic_json, _atomic_pickle, validate_complete_episode


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _replay_digest(replay):
    digest = hashlib.sha256()
    for name, value in sorted(vars(replay).items()):
        if name == "manifest":
            digest.update(_canonical([name, value]).encode())
        else:
            array = np.asarray(value)
            digest.update(_canonical([name, array.dtype.str, array.shape]).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def capture_matched_prefix(environment_config, catalog, replay, trace_paths, steps, output_dir: Path, stop_files=()) -> dict[int, Path]:
    """Config uses scenario/t_total/experiment_contract_sha256; optional input_sha256 pins files.

    Ordered trace segments supersede earlier tails. Snapshots are plain EnvSnapshot
    pickles; inputs.json and prefix.pkl belong exclusively to this prefix root.
    """
    stops = tuple(map(Path, stop_files))
    def check_stop():
        if any(path.exists() for path in stops):
            raise InterruptedError("matched prefix STOP requested")
    check_stop()
    root = Path(output_dir)
    manifest_path, checkpoint_path = root / "inputs.json", root / "prefix.pkl"
    if root.exists() and not manifest_path.exists() and any(root.iterdir()):
        raise ValueError("refusing to adopt an unrecognized prefix directory")
    root.mkdir(parents=True, exist_ok=True)
    evidence = {}
    try:
        targets = sorted(set(steps))
        if not targets or any(type(step) is not int or not 0 <= step < replay.size for step in targets):
            raise ValueError("target steps must be nonempty valid integer control steps")
        replay.validate()
        validate_complete_episode(replay)
        np.testing.assert_array_equal(replay.control_step, np.arange(replay.size))
        if np.unique(replay.episode).size != 1:
            raise ValueError("prefix replay must describe one complete episode")
        contract = environment_config["experiment_contract_sha256"]
        for key, expected in (("catalog_fingerprint", catalog.fingerprint), ("experiment_contract_sha256", contract),
                              ("response_equivalence_mode", CONTINUATION_EQUIVALENCE),
                              ("scenario", environment_config["scenario"]), ("t_total_sec", environment_config["t_total"])):
            if replay.manifest.get(key) != expected:
                raise ValueError(f"replay contract mismatch: {key}")
        for path, expected in environment_config.get("input_sha256", {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected.lower():
                raise ValueError(f"pinned input changed: {path}")
        rows, hashes = {}, []
        for path in map(Path, trace_paths):
            data = path.read_bytes()
            hashes.append([str(path.resolve()), hashlib.sha256(data).hexdigest()])
            segment = {}
            for line in data.decode("utf-8").splitlines():
                row = json.loads(line)
                step = int(row["control_step"])
                if step in segment and row != segment[step]:
                    raise ValueError(f"conflicting duplicate within trace segment at {step}")
                segment[step] = row
            rows.update({step: row for step, row in segment.items() if 0 <= step <= targets[-1]})
        for step in range(targets[-1] + 1):
            if step not in rows:
                raise ValueError(f"trace has a hole at control step {step}")
            row = rows[step]
            action = int(replay.action_id[step])
            identity = row["candidate_continuation_identities"][str(action)]
            if (row["selected_action_id"] != action or row["catalog_fingerprint"] != catalog.fingerprint
                    or row["response_equivalence_mode"] != CONTINUATION_EQUIVALENCE or not row["state_id"]
                    or not identity["validity_gate_pass"] or identity["continuation"]["experiment_contract_sha256"] != contract):
                raise ValueError(f"trace/replay identity mismatch at {step}")
            np.testing.assert_allclose(row["interval_reward"], replay.reward[step], rtol=1e-6, atol=1e-6)
        inputs = {"format": "matched_policy_prefix_v1", "environment": environment_config,
                  "catalog": catalog.as_manifest(), "replay_sha256": _replay_digest(replay),
                  "traces": hashes, "steps": targets}
        signature = hashlib.sha256(_canonical(inputs).encode()).hexdigest()
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != inputs:
                raise ValueError("prefix inputs changed; use a new output directory")
        else:
            _atomic_json(manifest_path, inputs)
        label = f"matched_policy_prefix:{signature}"

        def check_state(env, observation, step):
            evidence.update(step=step, expected_state=rows[step]["state_id"],
                            actual_state=env._anchor_context_state_fingerprint())
            if (env.experiment_contract_fingerprint != contract or env.scenario_name != environment_config["scenario"]
                    or env.T_total != environment_config["t_total"] or env.step_idx - env.warmup != step
                    or env.response_equivalence_mode != CONTINUATION_EQUIVALENCE):
                raise ValueError("environment contract/step/equivalence mismatch")
            np.testing.assert_array_equal(env._observe(), observation)
            np.testing.assert_array_equal(observation, replay.observation[step])
            if evidence["actual_state"] != evidence["expected_state"]:
                raise ValueError("prefix state fingerprint differs from trace")

        def check_continuation(env, step, reward, done, valid):
            expected = rows[step]["candidate_continuation_identities"][str(int(replay.action_id[step]))]
            actual = _encoded_continuation(env, float(reward), bool(done), bool(valid))
            if actual != _canonical(expected):
                evidence.update(expected_continuation=expected, actual_continuation=json.loads(actual))
                raise ValueError(f"full post-commit continuation mismatch at {step}")

        def restore(snapshot, step):
            if (not isinstance(snapshot, EnvSnapshot) or snapshot.label != label or snapshot.control_step != step
                    or snapshot.experiment_contract_sha256 != contract):
                raise ValueError("incompatible prefix snapshot")
            env, observation = restore_env_snapshot(snapshot)
            check_state(env, observation, step)
            if step:
                check_continuation(env, step - 1, rows[step - 1]["interval_reward"], replay.done[step - 1], True)
            return env, observation

        paths = {step: root / f"step_{step:03d}.pkl" for step in targets}
        for step, path in paths.items():
            if path.exists():
                restore(pickle.loads(path.read_bytes()), step)
        if checkpoint_path.exists():
            saved = pickle.loads(checkpoint_path.read_bytes())
            if saved["signature"] != signature:
                raise ValueError("incompatible prefix checkpoint")
            step = saved["snapshot"].control_step
            env, observation = restore(saved["snapshot"], step)
            initial_ttt, prefix_reward = saved["initial_ttt"], saved["prefix_reward"]
            np.testing.assert_allclose(prefix_reward, replay.reward[:step].sum(dtype=np.float64), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(env.sim.total_ttt, initial_ttt - prefix_reward, rtol=1e-6, atol=1e-6)
            if any(target < step and not path.exists() for target, path in paths.items()):
                raise ValueError("checkpoint has passed a missing target snapshot")
            if step == targets[-1] and all(path.exists() for path in paths.values()):
                return paths
        else:
            if any(path.exists() for path in paths.values()):
                raise ValueError("snapshots exist without a prefix checkpoint")
            check_stop()
            env = RLLeaderEnv(scenario_name=environment_config["scenario"], T_total=environment_config["t_total"],
                              action_mode="full", mask="RL-FULL", pstack_anchor=True, action_parameterization="pstack_residual")
            observation = env.reset()
            env.response_equivalence_mode = CONTINUATION_EQUIVALENCE
            step, initial_ttt, prefix_reward = 0, float(env.sim.total_ttt), 0.0
        while True:
            check_state(env, observation, step)
            snapshot = capture_env_snapshot(env, observation, label=label)
            _atomic_pickle(checkpoint_path, {"signature": signature, "snapshot": snapshot,
                                           "initial_ttt": initial_ttt, "prefix_reward": prefix_reward})
            if step in paths and not paths[step].exists():
                _atomic_pickle(paths[step], snapshot)
            if step == targets[-1]:
                return paths
            check_stop()
            observation, reward, done, info = commit_catalog_action_from_anchor(env, catalog, int(replay.action_id[step]))
            evidence.update(committed_step=step, reward=float(reward), expected_reward=float(replay.reward[step]))
            np.testing.assert_allclose(reward, replay.reward[step], rtol=1e-6, atol=1e-6)
            if done != bool(replay.done[step]) or done:
                raise ValueError("unexpected prefix terminal flag")
            check_continuation(env, step, reward, done, info.get("validity_gate_pass", False))
            np.testing.assert_array_equal(observation, replay.next_observation[step])
            prefix_reward += float(reward)
            step += 1
            np.testing.assert_allclose(prefix_reward, replay.reward[:step].sum(dtype=np.float64), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(env.sim.total_ttt, initial_ttt - prefix_reward, rtol=1e-6, atol=1e-6)
    except InterruptedError:
        raise
    except Exception as exc:
        _atomic_json(root / f"failure_{uuid4().hex}.json", {"error": f"{type(exc).__name__}: {exc}", **evidence})
        raise
