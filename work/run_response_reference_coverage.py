"""Small reference-coverage ablation: collect, train two batches, then evaluate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl_leader.response_dqn_collect import _save_replay_atomic
from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.run_sequential_response_ddqn import _atomic_json, _collect_actor_worker, validate_complete_episode
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.run_response_cql_ablation import _pinned_json, _read_json, main as evaluate_comparison, validate_training_recipe


def verify_reference(summary, replay, reference, tolerance):
    validate_complete_episode(replay)
    if abs(summary["total_ttt"] - reference["expected_total_ttt"]) > tolerance:
        raise ValueError("reference TTT does not reproduce its recorded controller")
    expected = {}
    with Path(reference["expected_trace"]).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            expected[int(row["control_step"])] = reference["expected_interval_sign"] * row[reference["expected_interval_field"]]
    np.testing.assert_allclose(
        replay.reward, [expected[int(step)] for step in replay.control_step], rtol=1e-6, atol=1e-5,
    )
    actions = np.zeros(replay.size, dtype=np.int64)
    if reference["first_action"] is not None:
        actions[0] = reference["first_action"]
    np.testing.assert_array_equal(replay.action_id, actions)


def train_variant(data_path, output_dir, spec, alpha, *, stop_files=()):
    if any(Path(path).exists() for path in stop_files):
        raise InterruptedError("STOP requested before training")
    base_dir = output_dir
    attempt = 0
    while output_dir.exists() and not (output_dir / "ensemble_manifest.json").exists():
        attempt += 1
        output_dir = base_dir.with_name(f"{base_dir.name}_attempt_{attempt:02d}")
    manifest_path = output_dir / "ensemble_manifest.json"
    recipe = {**spec, "data": str(data_path)}
    if not manifest_path.exists():
        common = spec["common_training"]
        command = [sys.executable, "-B", "-m", "rl_leader.train_response_dqn",
                   "--data", str(data_path), "--output-dir", str(output_dir),
                   "--conservative-alpha", str(alpha), "--require-sequential-td"]
        for key, value in common.items():
            if key in {"group_resampling", "require_sequential_td", "mask_constant_features"}:
                continue
            command.extend(["--" + key.replace("_", "-"), str(value)])
        if not common["group_resampling"]:
            command.append("--no-group-bootstrap")
        if common.get("mask_constant_features", False):
            command.append("--mask-constant-features")
        for path in stop_files:
            command.extend(["--stop-file", str(path)])
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if any(Path(path).exists() for path in stop_files):
                raise InterruptedError("STOP requested during training") from None
            raise
    validate_training_recipe(_read_json(manifest_path), recipe, alpha)
    return output_dir


def persist_merged_replay(replay, path):
    if not path.exists():
        _save_replay_atomic(replay, path)
    saved = load_frozen_response_replay(path)
    for name in replay.__dataclass_fields__:
        if name == "manifest":
            if saved.manifest != replay.manifest:
                raise ValueError("saved merged replay manifest differs from its sources")
        else:
            np.testing.assert_array_equal(getattr(saved, name), getattr(replay, name))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    _configure_torch_threads(1)
    plan = _read_json(args.config)
    root = Path(plan["output_dir"])
    previous = Path(plan["start_only_after_output_dir"])
    stop_files = [root / "STOP", previous / "STOP", *[Path(path) / "STOP" for path in plan["additional_lock_dirs"]]]
    status_path = root / "status.json"
    def stopped():
        if any(path.exists() for path in stop_files):
            _atomic_json(status_path, {"phase": "paused"})
            return True
        return False
    if stopped():
        return
    if _read_json(previous / "status.json")["phase"] != "ablation_complete" or _read_json(previous / "process.json")["state"] != "exited":
        raise ValueError("CQL comparison must finish before reference collection")
    _pinned_json(root / "reference_plan.json", plan)
    for path_key, hash_key in (("base_replay", "base_sha256"), ("policy_replay", "policy_sha256")):
        if hashlib.sha256(Path(plan[path_key]).read_bytes()).hexdigest().lower() != plan[hash_key].lower():
            raise ValueError("source replay hash changed")
    base = load_frozen_response_replay(plan["base_replay"])
    policy = load_frozen_response_replay(plan["policy_replay"])
    validate_sequential_td_replay(base, gamma=1.0)
    validate_complete_episode(policy)
    spec = _read_json(plan["training_spec"])
    environment = _read_json(plan["environment_config"])
    workers = plan["response_workers_per_actor"]
    if workers < 1 or workers * len(plan["references"]) > 8:
        raise ValueError("reference worker budget exceeded")
    on_policy = merge_frozen_response_replays([base, policy], source=__name__)
    validate_sequential_td_replay(on_policy, gamma=1.0)
    payloads = [{
        "config": environment, "catalog": base.manifest["catalog"],
        "directory": str(root / "references" / reference["name"]),
        "seed": spec["common_training"]["seed"] + index,
        "episode": plan["reference_episode_base"] + index,
        "response_workers": workers, "checkpoints": [], "epsilon": 0.0,
        "first_action": reference["first_action"], "full_run": reference["full_run"],
        "stop_file": str(stop_files[0]), "additional_stop_files": [str(path) for path in stop_files[1:]],
    } for index, reference in enumerate(plan["references"])]
    if args.validate_only:
        print(json.dumps({"validated": True, "existing_rows_to_reuse": on_policy.size,
                          "new_reference_episodes": len(payloads), "total_preview_workers": workers * len(payloads)}))
        return
    _atomic_json(status_path, {"phase": "collecting_references"})
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        futures = {pool.submit(_collect_actor_worker, payload): index for index, payload in enumerate(payloads)}
        for future in as_completed(futures):
            try:
                summary = future.result()
            except InterruptedError:
                _atomic_json(status_path, {"phase": "paused"})
                return
            index = futures[future]
            replay = load_frozen_response_replay(Path(payloads[index]["directory"]) / "replay.npz")
            verify_reference(summary, replay, plan["references"][index], plan["reference_ttt_tolerance"])
            print(json.dumps({"event": "reference_verified", "name": plan["references"][index]["name"], **summary}), flush=True)
    if stopped():
        return
    references = [load_frozen_response_replay(Path(payload["directory"]) / "replay.npz") for payload in payloads]
    augmented = merge_frozen_response_replays([on_policy, *references], source=__name__)
    variants = []
    for name, replay in (("policy_only", on_policy), ("with_references", augmented)):
        if stopped():
            return
        validate_sequential_td_replay(replay, gamma=1.0)
        data_path = root / name / "training_replay.npz"
        data_hash = persist_merged_replay(replay, data_path)
        _pinned_json(root / name / "data_sources.json", {
            "source_hashes": [plan["base_sha256"], plan["policy_sha256"]] + (
                [hashlib.sha256((Path(payload["directory"]) / "replay.npz").read_bytes()).hexdigest() for payload in payloads]
                if name == "with_references" else []),
            "rows": replay.size, "data_sha256": data_hash,
        })
        _atomic_json(status_path, {"phase": "training", "variant": name, "transitions": replay.size})
        model_dir = train_variant(data_path, root / name / "model", spec, plan["conservative_alpha"])
        variants.append({"name": name, "data": str(data_path), "data_sha256": data_hash,
                         "model_dir": str(model_dir),
                         "conservative_alpha": plan["conservative_alpha"]})
    if stopped():
        return
    evaluation_spec = {
        **spec, "output_dir": str(root), "start_only_after_output_dir": str(previous),
        "required_previous_phase": "ablation_complete", "data": variants[0]["data"],
        "data_sha256": variants[0]["data_sha256"], "variants": variants,
        "control_evaluation_dir": plan["control_evaluation_dir"], "control_response_workers": workers,
        "evaluation_episode_base": plan["evaluation_episode_base"],
        "additional_stop_files": [str(path) for path in stop_files[2:]],
        "hypothesis": plan["hypothesis"], "status": "reference_collection_verified",
        "do_not_add_new_replay_to_this_controlled_comparison": False,
    }
    evaluation_spec_path = root / "evaluation_spec.json"
    _pinned_json(evaluation_spec_path, evaluation_spec)
    evaluate_comparison(["--config", str(evaluation_spec_path)])


if __name__ == "__main__":
    main()
