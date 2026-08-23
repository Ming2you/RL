"""Report full-action support, local response coverage, and censoring."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/full_action_v2/*.npz")
    parser.add_argument("--out", default="results/full_action_dataset_audit.json")
    args = parser.parse_args(argv)
    patterns = [value.strip() for value in str(args.data).split(",") if value.strip()]
    files = sorted({path for value in patterns for path in glob.glob(value)})
    if not files:
        raise SystemExit(f"no files matched: {args.data}")
    actions = []
    observations = []
    responses = []
    validity = []
    modes = []
    submodes = []
    reasons = []
    perturb_active = []
    anchor_distance = []
    episode_summaries = []
    dataset_names = set()
    anchor_transition_contracts = set()
    manifest = None
    for path in files:
        dataset = np.load(path, allow_pickle=False)
        actions.append(dataset["act"])
        observations.append(dataset["obs"])
        responses.append(dataset["response"])
        if "perturb_active" in dataset.files:
            perturb_active.extend(dataset["perturb_active"].tolist())
            anchor_distance.extend(dataset["anchor_distance"].tolist())
        else:
            perturb_active.extend([0.0] * int(dataset["obs"].shape[0]))
            anchor_distance.extend([0.0] * int(dataset["obs"].shape[0]))
        validity.extend(dataset["validity"].tolist())
        modes.extend(dataset["behavior_mode"].tolist())
        submodes.extend(
            dataset["behavior_submode"].tolist()
            if "behavior_submode" in dataset.files else dataset["behavior_mode"].tolist()
        )
        reasons.extend(value for value in dataset["termination_reason"].tolist() if value)
        current = json.loads(dataset["manifest_json"].item())
        dataset_names.add(current.get("dataset", "legacy_v1"))
        anchor_transition_contracts.add(
            current.get("optimizer_anchor_transition_contract", "missing")
        )
        episode_summaries.extend(current.get("episode_summaries", []))
        if manifest is None:
            manifest = current
        elif current["action_schema"] != manifest["action_schema"]:
            raise ValueError("action schema mismatch across dataset files")
        elif current.get("response_contract") != manifest.get("response_contract"):
            raise ValueError("response contracts do not match across dataset files")
    action = np.concatenate(actions)
    observation = np.concatenate(observations)
    response = np.concatenate(responses)
    schema = manifest["action_schema"]
    scenario_counts = Counter()
    target_counts = Counter()
    scenario_demands = []
    mode_phase_counts = Counter()
    mode_returns = {}
    for summary in episode_summaries:
        scenario = summary.get("scenario", {})
        if scenario.get("freeway_lane_closures"):
            stressor = "incident"
        elif scenario.get("urban_west_east_ratio") is not None:
            stressor = "skew"
        else:
            stressor = "none"
        scenario_counts[stressor] += 1
        target_counts[scenario.get("target_scenario", "broad_random")] += 1
        if "urban_scale" in scenario:
            scenario_demands.append(float(scenario["urban_scale"]))
    observation_names = manifest["observation_schema"]["names"]
    peak_index = observation_names.index("time.peak")
    recovery_index = observation_names.index("time.recovery")
    for mode, is_recovery in zip(modes, observation[:, recovery_index] > 0.5):
        mode_phase_counts[(str(mode), "recovery" if is_recovery else "peak")] += 1
    for path in files:
        dataset = np.load(path, allow_pickle=False)
        rewards = dataset["rew"]
        for episode in np.unique(dataset["episode"]):
            selected = dataset["episode"] == episode
            mode = str(dataset["behavior_mode"][selected][0])
            mode_returns.setdefault(mode, []).append(float(-np.sum(rewards[selected])))
    blocks = []
    response_index = 0
    for family, owners in (("urban", schema["signals"]), ("freeway", schema["ramps"])):
        for owner_index, owner in enumerate(owners):
            block_index = owner_index if family == "urban" else len(schema["signals"]) + owner_index
            values = action[:, 2 + 5 * block_index:2 + 5 * (block_index + 1)]
            local_response = response[:, response_index:response_index + 2]
            response_index += 2
            response_std = local_response.std(0)
            blocks.append({
                "family": family,
                "owner": owner,
                "action_min": values.min(0).tolist(),
                "action_max": values.max(0).tolist(),
                "action_std": values.std(0).tolist(),
                "response_min": local_response.min(0).tolist(),
                "response_max": local_response.max(0).tolist(),
                "response_std": response_std.tolist(),
                "response_unique_count": [
                    int(np.unique(local_response[:, index]).size)
                    for index in range(local_response.shape[1])
                ],
                "dead_response_fraction": float(np.mean(response_std < 1.0e-6)),
                "dead_action_fraction": float(np.mean(np.all(np.abs(values) < 1.0e-6, axis=0))),
            })
    scalar_offset = 2 + 5 * (len(schema["signals"]) + len(schema["ramps"]))
    for owner_index, owner in enumerate(schema.get("nonmerge_vsl_keys", [])):
        values = action[:, scalar_offset + 2 * owner_index:scalar_offset + 2 * (owner_index + 1)]
        local_response = response[:, response_index:response_index + 1]
        response_index += 1
        response_std = local_response.std(0)
        blocks.append({
            "family": "vsl",
            "owner": owner,
            "action_min": values.min(0).tolist(),
            "action_max": values.max(0).tolist(),
            "action_std": values.std(0).tolist(),
            "response_min": local_response.min(0).tolist(),
            "response_max": local_response.max(0).tolist(),
            "response_std": response_std.tolist(),
            "response_unique_count": [int(np.unique(local_response[:, 0]).size)],
            "dead_response_fraction": float(np.mean(response_std < 1.0e-6)),
            "dead_action_fraction": float(np.mean(np.all(np.abs(values) < 1.0e-6, axis=0))),
        })
    certificate_offset = scalar_offset + 2 * len(schema.get("nonmerge_vsl_keys", []))
    freeway_response_offset = 2 * len(schema["signals"])
    for owner_index, owner in enumerate(schema.get("certificate_ramps", [])):
        values = action[:, certificate_offset + owner_index:certificate_offset + owner_index + 1]
        local_response = response[
            :, freeway_response_offset + 2 * owner_index:freeway_response_offset + 2 * owner_index + 1
        ]
        response_std = local_response.std(0)
        blocks.append({
            "family": "certificate",
            "owner": owner,
            "action_min": values.min(0).tolist(),
            "action_max": values.max(0).tolist(),
            "action_std": values.std(0).tolist(),
            "release_true_fraction": float(np.mean(values[:, 0] > 0.0)),
            "response_min": local_response.min(0).tolist(),
            "response_max": local_response.max(0).tolist(),
            "response_std": response_std.tolist(),
            "response_unique_count": [int(np.unique(local_response[:, 0]).size)],
            "dead_response_fraction": float(np.mean(response_std < 1.0e-6)),
            "dead_action_fraction": float(np.mean(np.all(np.abs(values) < 1.0e-6, axis=0))),
        })
    dataset_names = sorted(dataset_names)
    report = {
        "dataset": "+".join(dataset_names),
        "dataset_components": dataset_names,
        "response_contract": manifest.get("response_contract", "legacy"),
        "action_schema_version": schema.get("version", "missing"),
        "observation_schema_version": manifest.get("observation_schema", {}).get(
            "version", "missing"
        ),
        "optimizer_anchor_transition_contracts": sorted(anchor_transition_contracts),
        "files": files,
        "transitions": int(action.shape[0]),
        "observation_dimension": int(manifest["observation_schema"]["dimension"]),
        "action_dimension": int(action.shape[1]),
        "action_support": {
            "min": action.min(0).tolist(),
            "max": action.max(0).tolist(),
            "std": action.std(0).tolist(),
            "dead_dimension_indices": np.flatnonzero(action.std(0) < 1.0e-6).astype(int).tolist(),
        },
        "observation_support": {
            "dead_dimension_indices": np.flatnonzero(
                observation.std(0) < 1.0e-6
            ).astype(int).tolist(),
        },
        "validity_pass_fraction": float(np.mean(validity)),
        "behavior_modes": dict(Counter(modes)),
        "behavior_submodes": dict(Counter(submodes)),
        "termination_reasons": dict(Counter(reasons)),
        "scenario_stressors": dict(scenario_counts),
        "target_scenarios": dict(target_counts),
        "scenario_demand": {
            "min": float(min(scenario_demands)) if scenario_demands else None,
            "mean": float(np.mean(scenario_demands)) if scenario_demands else None,
            "max": float(max(scenario_demands)) if scenario_demands else None,
        },
        "phase_transitions": {
            "peak": int(np.sum(observation[:, peak_index] > 0.5)),
            "recovery": int(np.sum(observation[:, recovery_index] > 0.5)),
        },
        "mode_phase_transitions": {
            f"{mode}.{phase}": int(count)
            for (mode, phase), count in sorted(mode_phase_counts.items())
        },
        "mode_episode_ttt": {
            mode: {
                "episodes": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for mode, values in sorted(mode_returns.items())
        },
        "anchor_perturbation": {
            "active_transitions": int(np.sum(np.asarray(perturb_active) > 0.5)),
            "active_fraction": float(np.mean(np.asarray(perturb_active) > 0.5)),
            "nonzero_distance_fraction": float(np.mean(np.asarray(anchor_distance) > 1.0e-9)),
            "active_distance_mean": float(np.mean(
                np.asarray(anchor_distance)[np.asarray(perturb_active) > 0.5]
            )) if np.any(np.asarray(perturb_active) > 0.5) else 0.0,
        },
        "blocks": blocks,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"transitions={report['transitions']} valid={report['validity_pass_fraction']:.3f} "
        f"modes={report['behavior_submodes']}",
        flush=True,
    )
    print(f"saved dataset audit -> {output}", flush=True)


if __name__ == "__main__":
    main()
