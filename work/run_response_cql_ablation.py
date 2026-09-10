"""Evaluate already-trained CQL variants on a pinned sequential DDQN experiment.

Launch through start_sequential_response_ddqn.ps1 to hold both experiment locks.
Frozen-state screening is diagnostic, not an estimate of candidate-policy TTT.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_leader.response_dqn import load_trained_response_dqn
from rl_leader.response_dqn_data import load_frozen_response_replay
from rl_leader.run_sequential_response_ddqn import (
    _atomic_json, _collect_actor_worker, validate_complete_episode,
)
from rl_leader.train_response_dqn import _configure_torch_threads, validate_sequential_td_replay
from work.audit_sequential_td_evaluation import audit


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _pinned_json(path, value):
    if path.exists() and _read_json(path) != value:
        raise ValueError(f"pinned artifact changed: {path}")
    if not path.exists():
        _atomic_json(path, value)


def validate_training_recipe(manifest, spec, alpha):
    expected = dict(spec["common_training"])
    expected["hidden"] = [int(value) for value in expected["hidden"].split(",")]
    expected["no_bootstrap"] = not expected.pop("group_resampling")
    expected["conservative_alpha"] = alpha
    actual = manifest["config"]
    for key, value in expected.items():
        default = {"value_parameterization": "free_q", "backup_horizon": 1}.get(key)
        if actual.get(key, default) != value:
            raise ValueError(f"controlled training recipe mismatch: {key}")
    if Path(manifest["source_dataset"]).resolve() != Path(spec["data"]).resolve():
        raise ValueError("model was trained on a different dataset")
    if manifest["catalog_fingerprint"] != spec["catalog_fingerprint"]:
        raise ValueError("model catalog mismatch")
    if len(manifest["checkpoints"]) != expected["ensemble_size"]:
        raise ValueError("incomplete ensemble")
    if manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1") != spec.get("response_equivalence_mode", "legacy_follower_runtime_v1"):
        raise ValueError("model response equivalence differs from evaluation recipe")


def variant_training_spec(spec, variant):
    result = {**spec, "data": variant.get("data", spec["data"])}
    common = dict(spec["common_training"])
    if "value_parameterization" in variant:
        head = variant["value_parameterization"]
        if head not in {"free_q", "finite_horizon_cost_v1"}:
            raise ValueError("unsupported variant value parameterization")
        common["value_parameterization"] = head
    if "backup_horizon" in variant:
        horizon = variant["backup_horizon"]
        if type(horizon) is not int or horizon < 1:
            raise ValueError("unsupported variant backup horizon")
        common["backup_horizon"] = horizon
    result["common_training"] = common
    return result


def validate_checkpoint_value_heads(models, spec):
    expected = spec["common_training"].get("value_parameterization", "free_q")
    if any(getattr(model.config, "value_parameterization", "free_q") != expected for model in models):
        raise ValueError("checkpoint value parameterization differs from its training recipe")
    horizon = spec["common_training"].get("backup_horizon", 1)
    if any(getattr(model.config, "backup_horizon", 1) != horizon for model in models):
        raise ValueError("checkpoint backup horizon differs from its training recipe")


def _screen(models, replays):
    support = np.min(np.stack([model.action_support_counts for model in models]), axis=0)
    reports = {}
    for name, replay in replays.items():
        mode = replay.manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1")
        if any(getattr(model, "response_equivalence_mode", "legacy_follower_runtime_v1") != mode for model in models):
            reports[name] = {"states": replay.size, "skipped_reason": "incompatible_response_equivalence_masks"}
            continue
        q = np.stack([model.q_values(replay.observation, replay.response_features) for model in models])
        if not np.isfinite(q).all():
            raise ValueError("nonfinite screening Q")
        mean = q.mean(axis=0, dtype=np.float64)
        mask = replay.action_mask & (support >= models[0].config.min_action_support)[None, :]
        mask[:, 0] = True
        chosen = np.where(mask, mean, -np.inf).argmax(axis=1)
        observed_q = mean[np.arange(replay.size), replay.action_id]
        scale = models[0].config.reward_scale
        reports[name] = {
            "states": replay.size,
            "selected_action_counts": np.bincount(chosen, minlength=replay.action_count).tolist(),
            "selected_action_support_at_most_3_rows": int((support[chosen] <= 3).sum()),
            "positive_selected_q_rows": int((mean[np.arange(replay.size), chosen] > 1e-6).sum()),
            "observed_action_immediate_reward_bound_violations": int((observed_q > scale * replay.reward + 1e-6).sum()),
            "initial_selected_action": int(chosen[0]),
            "initial_mean_q": mean[0].tolist(),
        }
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args(argv)
    _configure_torch_threads(1)
    spec = _read_json(args.config)
    root = Path(spec["output_dir"])
    previous = Path(spec["start_only_after_output_dir"])
    stop_files = [root / "STOP", previous / "STOP", *map(Path, spec.get("additional_stop_files", []))]
    status_path = root / "status.json"
    if any(path.exists() for path in stop_files):
        _atomic_json(status_path, {"phase": "paused"})
        return
    if _read_json(previous / "status.json")["phase"] != spec["required_previous_phase"]:
        raise ValueError("previous experiment is not complete")
    if _read_json(previous / "process.json")["state"] != "exited":
        raise ValueError("previous runner has not exited")
    baseline_path = Path("results/response_dqn_170_incident/pstack_rl_contract_v2/summary.json")
    target = _read_json(baseline_path)["total_ttt"] * 0.95
    control_dir = Path(spec.get("control_evaluation_dir", previous / "ungated_full_run"))
    control_summary = _read_json(control_dir / "summary.json")
    if control_summary["total_ttt"] <= target:
        _atomic_json(status_path, {"phase": "previous_goal_candidate_requires_confirmation"})
        return
    digest = hashlib.sha256(Path(spec["data"]).read_bytes()).hexdigest()
    if digest.lower() != spec["data_sha256"].lower():
        raise ValueError("frozen training replay hash changed")
    _pinned_json(root / "resolved_config.json", spec)
    training = load_frozen_response_replay(spec["data"])
    validate_sequential_td_replay(training, gamma=1.0)
    if training.manifest["experiment_contract_sha256"] != spec["experiment_contract_sha256"]:
        raise ValueError("training experiment contract changed")
    control = load_frozen_response_replay(control_dir / "replay.npz")
    validate_complete_episode(control)
    if control.manifest["catalog_fingerprint"] != training.manifest["catalog_fingerprint"]:
        raise ValueError("screening catalog mismatch")
    payloads, screening, model_record = [], {}, {}
    environment = _read_json(spec["environment_config"])
    for field, expected in {"scope": "ungated_full_run", "epsilon": 0.0,
                            "first_action": None, "lcb_guard": False, "response_preview": True}.items():
        if spec["evaluation"].get(field) != expected:
            raise ValueError(f"unsupported evaluation setting: {field}")
    workers = spec["evaluation"]["maximum_total_response_workers"] // len(spec["variants"])
    if workers < 1 or workers * len(spec["variants"]) > 8:
        raise ValueError("evaluation exceeds eight-worker budget")
    for index, variant in enumerate(spec["variants"]):
        name = variant["name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("variant must be a directory name")
        model_dir = Path(variant.get("model_dir", root / name / "model"))
        manifest = _read_json(model_dir / "ensemble_manifest.json")
        variant_spec = variant_training_spec(spec, variant)
        variant_training = training
        if "data" in variant:
            variant_digest = hashlib.sha256(Path(variant["data"]).read_bytes()).hexdigest()
            if variant_digest.lower() != variant["data_sha256"].lower():
                raise ValueError("variant training replay hash changed")
            variant_training = load_frozen_response_replay(variant["data"])
            validate_sequential_td_replay(variant_training, gamma=1.0)
            if any(variant_training.manifest[key] != training.manifest[key]
                   for key in ("catalog_fingerprint", "experiment_contract_sha256", "scenario", "t_total_sec")):
                raise ValueError("variant replay contract mismatch")
        validate_training_recipe(manifest, variant_spec, variant["conservative_alpha"])
        paths = manifest["checkpoints"]
        model_record[name] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
        models = [load_trained_response_dqn(path, expected_catalog_fingerprint=spec["catalog_fingerprint"])
                  for path in paths]
        validate_checkpoint_value_heads(models, variant_spec)
        if any(model.config.conservative_alpha != variant["conservative_alpha"] for model in models):
            raise ValueError("checkpoint alpha differs from its training manifest")
        mode = spec.get("response_equivalence_mode", "legacy_follower_runtime_v1")
        if variant_training.manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1") != mode:
            raise ValueError("training response equivalence differs from evaluation recipe")
        if any(model.response_equivalence_mode != mode for model in models):
            raise ValueError("checkpoint response equivalence differs from its training manifest")
        screening[name] = _screen(models, {"training": variant_training, "control_full_run_states": control})
        payloads.append({
            "config": environment, "catalog": training.manifest["catalog"],
            "directory": str(root / name / "evaluation"),
            "seed": spec["common_training"]["seed"] + index,
            "episode": spec.get("evaluation_episode_base", 21000) + index,
            "response_workers": workers, "checkpoints": paths, "epsilon": 0.0,
            "first_action": None, "full_run": True, "stop_file": str(stop_files[0]),
            "additional_stop_files": [str(path) for path in stop_files[1:]],
        })
        if mode != "legacy_follower_runtime_v1":
            payloads[-1]["response_equivalence_mode"] = mode
    _pinned_json(root / "model_hashes.json", model_record)
    _atomic_json(root / "screening.json", {
        "interpretation": "Frozen-state ranks only; new-policy TTT is not measured by screening.",
        "variants": screening,
    })
    if args.screen_only:
        _atomic_json(status_path, {"phase": "screened", "variants": list(screening)})
        print(json.dumps(screening), flush=True)
        return
    if any(path.exists() for path in stop_files):
        _atomic_json(status_path, {"phase": "paused"})
        return
    _atomic_json(status_path, {"phase": "evaluating", "variants": list(screening),
                              "response_workers_per_actor": workers})
    results = {}
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        futures = {pool.submit(_collect_actor_worker, payload): variant["name"]
                   for payload, variant in zip(payloads, spec["variants"])}
        for future in as_completed(futures):
            try:
                result = future.result()
            except InterruptedError:
                _atomic_json(status_path, {"phase": "paused"})
                return
            results[futures[future]] = result
            print(json.dumps({"event": "evaluation_complete", "variant": futures[future], **result}), flush=True)
    for variant in spec["variants"]:
        directory = root / variant["name"]
        diagnostic = audit(directory / "evaluation", Path(variant.get("model_dir", directory / "model")), baseline_path)
        _pinned_json(directory / "calibration_audit.json", diagnostic)
    _atomic_json(root / "comparison.json", {
        "control_full_run" if "control_evaluation_dir" in spec else "vanilla_full_run": control_summary,
        "variants": results, "target_ttt": target,
        "timing_note": f"Variants use {workers} preview workers each concurrently; control used {spec.get('control_response_workers', 8)} preview workers.",
    })
    candidate = min(results, key=lambda name: results[name]["total_ttt"])
    _atomic_json(status_path, {
        "phase": "goal_candidate_requires_confirmation" if results[candidate]["total_ttt"] <= target else "ablation_complete",
        "best_variant": candidate, "best_ttt": results[candidate]["total_ttt"],
        "goal_confirmed": False,
    })


if __name__ == "__main__":
    main()
