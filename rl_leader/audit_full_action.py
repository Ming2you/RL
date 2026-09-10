"""Report full-action support, local response coverage, and censoring."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

from rl_leader.data_contract import (
    TEACHER_REPLAY_TOLERANCE,
    continuous_actor_supervision_mask,
    pstack_residual_actor_supervision_mask,
    pstack_residual_trainable_dimension_mask,
    pstack_residual_targets,
    validate_pstack_residual_rows,
)
from rl_leader.experiment_contract import (
    EXPERIMENT_PROFILE_ID,
    verify_contract_map,
)


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
    teacher_pfo_selected = []
    teacher_outer_pfo_selected = []
    teacher_selected_stage = []
    teacher_far_enabled = []
    teacher_saturated = []
    teacher_response_finite = []
    teacher_responses = []
    anchor_actions = []
    policy_residuals = []
    deployed_residuals = []
    anchor_envelopes = []
    anchor_selected_branches = []
    anchor_fingerprints = []
    pstack_anchor_pick_rl = []
    long_horizon_label_valid = []
    long_horizon_positive = []
    long_horizon_presence = set()
    episode_summaries = []
    dataset_names = set()
    anchor_contracts = set()
    anchor_transition_contracts = set()
    optimizer_preview_contracts = set()
    warmup_control_contracts = set()
    pfo_supervisor_flags = set()
    pstack_anchor_flags = set()
    action_parameterization_supports = set()
    experiment_contract_sha256 = set()
    experiment_profiles = set()
    experiment_contract_errors = []
    manifest = None
    for path in files:
        dataset = np.load(path, allow_pickle=False)
        current = json.loads(dataset["manifest_json"].item())
        if bool(current.get("pstack_anchor", False)):
            validate_pstack_residual_rows(dataset, current)
        actions.append(dataset["act"])
        observations.append(dataset["obs"])
        responses.append(dataset["response"])
        anchor_actions.append(
            dataset["anchor_action"]
            if "anchor_action" in dataset.files
            else np.full_like(dataset["act"], np.nan)
        )
        if "policy_residual" in dataset.files:
            policy_residuals.append(dataset["policy_residual"])
            deployed_residuals.append(dataset["deployed_residual"])
            anchor_envelopes.append(dataset["anchor_envelope"])
            anchor_selected_branches.append(dataset["anchor_selected_branch"])
            anchor_fingerprints.append(dataset["anchor_fingerprint"])
        pstack_anchor_pick_rl.extend(
            dataset["pstack_anchor_pick_rl"].tolist()
            if "pstack_anchor_pick_rl" in dataset.files
            else [-1.0] * int(dataset["obs"].shape[0])
        )
        transition_count = int(dataset["obs"].shape[0])
        has_long_horizon = (
            "long_horizon_label_valid" in dataset.files
            and "long_horizon_positive" in dataset.files
        )
        long_horizon_presence.add(has_long_horizon)
        long_horizon_label_valid.extend(
            dataset["long_horizon_label_valid"].tolist()
            if has_long_horizon else [0.0] * transition_count
        )
        long_horizon_positive.extend(
            dataset["long_horizon_positive"].tolist()
            if has_long_horizon else [0.0] * transition_count
        )
        if "perturb_active" in dataset.files:
            perturb_active.extend(dataset["perturb_active"].tolist())
            anchor_distance.extend(dataset["anchor_distance"].tolist())
        else:
            perturb_active.extend([0.0] * int(dataset["obs"].shape[0]))
            anchor_distance.extend([0.0] * int(dataset["obs"].shape[0]))
        validity.extend(dataset["validity"].tolist())
        modes.extend(dataset["behavior_mode"].tolist())
        if "teacher_pfo_selected" in dataset.files:
            teacher_pfo_selected.extend(dataset["teacher_pfo_selected"].tolist())
            teacher_outer_pfo_selected.extend(
                dataset["teacher_outer_pfo_selected"].tolist()
                if "teacher_outer_pfo_selected" in dataset.files
                else [-1.0] * transition_count
            )
            teacher_selected_stage.extend(dataset["teacher_selected_stage"].tolist())
            teacher_far_enabled.extend(dataset["teacher_far_enabled"].tolist())
            teacher_saturated.extend(
                dataset["teacher_encoded_saturated_count"].tolist()
            )
            teacher_response_finite.extend(
                np.all(np.isfinite(dataset["teacher_response"]), axis=1).tolist()
            )
            teacher_responses.append(dataset["teacher_response"])
        else:
            teacher_pfo_selected.extend([-1.0] * transition_count)
            teacher_outer_pfo_selected.extend([-1.0] * transition_count)
            teacher_selected_stage.extend(["missing"] * transition_count)
            teacher_far_enabled.extend([-1.0] * transition_count)
            teacher_saturated.extend([-1.0] * transition_count)
            teacher_response_finite.extend([False] * transition_count)
            teacher_responses.append(np.full_like(dataset["response"], np.nan))
        submodes.extend(
            dataset["behavior_submode"].tolist()
            if "behavior_submode" in dataset.files else dataset["behavior_mode"].tolist()
        )
        reasons.extend(value for value in dataset["termination_reason"].tolist() if value)
        try:
            verified_contracts = verify_contract_map(
                current.get("experiment_contracts", {})
            )
            if not verified_contracts:
                raise ValueError("manifest contains no experiment contracts")
            for sha256, contract in verified_contracts.items():
                experiment_contract_sha256.add(sha256)
                experiment_profiles.add(str(contract.payload.get("profile_id", "")))
            summaries_by_episode = {
                int(summary["episode"]): summary
                for summary in current.get("episode_summaries", [])
            }
            present_episodes = set(map(int, np.unique(dataset["episode"])))
            if not present_episodes.issubset(summaries_by_episode):
                raise ValueError("episode summaries do not cover transition episodes")
            for episode, summary in summaries_by_episode.items():
                sha256 = str(summary.get("experiment_contract_sha256", ""))
                if sha256 not in verified_contracts:
                    raise ValueError(
                        f"episode {episode} references unknown contract {sha256!r}"
                    )
        except Exception as exc:
            experiment_contract_errors.append(
                {"source_file": path, "error": f"{type(exc).__name__}: {exc}"}
            )
        dataset_names.add(current.get("dataset", "legacy_v1"))
        anchor_contracts.add(current.get("optimizer_anchor_contract", "missing"))
        anchor_transition_contracts.add(
            current.get("optimizer_anchor_transition_contract", "missing")
        )
        optimizer_preview_contracts.add(
            current.get("optimizer_preview_contract", "missing")
        )
        warmup_control_contracts.add(current.get("warmup_control_contract", "missing"))
        pfo_supervisor_flags.add(bool(current.get("pfo_supervisor", False)))
        pstack_anchor_flags.add(bool(current.get("pstack_anchor", False)))
        action_parameterization_supports.add(
            current.get("action_parameterization_support", "missing")
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
    anchor_action = np.concatenate(anchor_actions)
    native_anchor_rows = bool(policy_residuals)
    if native_anchor_rows and len(policy_residuals) != len(files):
        raise ValueError("native-anchor row fields are missing from some dataset files")
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
    teacher_pfo_array = np.asarray(teacher_pfo_selected, dtype=float)
    teacher_outer_pfo_array = np.asarray(teacher_outer_pfo_selected, dtype=float)
    teacher_labeled = teacher_pfo_array >= 0.0
    teacher_pfo = teacher_labeled & (teacher_pfo_array > 0.5)
    teacher_leader = teacher_labeled & ~teacher_pfo
    native_anchor = np.asarray(modes) == "optimizer_anchor"
    native_anchor_labeled = teacher_labeled & native_anchor
    anchor_gate_pick_rl = np.asarray(pstack_anchor_pick_rl, dtype=float) > 0.5
    pstack_selected_native_anchor = native_anchor_labeled & ~anchor_gate_pick_rl
    teacher_far_array = np.asarray(teacher_far_enabled, dtype=float)
    teacher_saturated_array = np.asarray(teacher_saturated, dtype=float)
    teacher_response_finite_array = np.asarray(teacher_response_finite, dtype=bool)
    teacher_response = np.concatenate(teacher_responses)
    teacher_replay_linf = np.max(np.abs(response - teacher_response), axis=1)
    teacher_replay_finite = np.isfinite(teacher_replay_linf)
    replay_tolerance = TEACHER_REPLAY_TOLERANCE
    actor_supervision = continuous_actor_supervision_mask({
        "behavior_mode": np.asarray(modes),
        "teacher_pfo_selected": teacher_pfo_array,
        "teacher_response": teacher_response,
        "response": response,
    })
    residual_dataset = {
        "behavior_mode": np.asarray(modes),
        "teacher_pfo_selected": teacher_pfo_array,
        "teacher_response": teacher_response,
        "response": response,
        "act": action,
        "anchor_action": anchor_action,
        "pstack_anchor_pick_rl": np.asarray(pstack_anchor_pick_rl, dtype=float),
        "manifest_json": np.asarray(json.dumps({"action_schema": schema})),
    }
    if native_anchor_rows:
        residual_dataset.update({
            "policy_residual": np.concatenate(policy_residuals),
            "deployed_residual": np.concatenate(deployed_residuals),
            "anchor_envelope": np.concatenate(anchor_envelopes),
            "anchor_selected_branch": np.concatenate(anchor_selected_branches),
            "anchor_fingerprint": np.concatenate(anchor_fingerprints),
        })
    if long_horizon_presence == {True}:
        residual_dataset.update({
            "long_horizon_label_valid": np.asarray(
                long_horizon_label_valid, dtype=float
            ),
            "long_horizon_positive": np.asarray(long_horizon_positive, dtype=float),
        })
    elif long_horizon_presence == {False, True}:
        raise ValueError("long-horizon label arrays are missing from some dataset files")
    residual_supervision = pstack_residual_actor_supervision_mask(residual_dataset)
    residual_target = pstack_residual_targets(residual_dataset)
    residual_trainable = pstack_residual_trainable_dimension_mask(residual_dataset)
    residual_values = residual_target[residual_supervision]
    residual_std = (
        residual_values.std(0)
        if residual_values.size else np.zeros(action.shape[1], dtype=float)
    )
    action_names = list(schema.get("names", []))
    if len(action_names) != action.shape[1]:
        action_names = [f"action.{index}" for index in range(action.shape[1])]
    dead_action_indices = np.flatnonzero(action.std(0) < 1.0e-6).astype(int).tolist()
    all_dead_residual_indices = np.flatnonzero(
        residual_std < 1.0e-6
    ).astype(int).tolist()
    dead_residual_indices = np.flatnonzero(
        residual_trainable & (residual_std < 1.0e-6)
    ).astype(int).tolist()
    frozen_residual_indices = np.flatnonzero(~residual_trainable).astype(int).tolist()
    peak_mask = observation[:, peak_index] > 0.5
    recovery_mask = observation[:, recovery_index] > 0.5
    nonzero_residual = residual_supervision & np.any(
        np.abs(residual_target[:, residual_trainable]) > 1.0e-6, axis=1
    )
    perturb_block_episodes = Counter(
        str(owner)
        for summary in episode_summaries
        for owner in summary.get(
            "perturb_block_keys", summary.get("perturb_blocks", [])
        )
    )

    def replay_summary(mask):
        selected = np.asarray(mask, dtype=bool) & teacher_replay_finite
        if not np.any(selected):
            return {
                "transitions": 0,
                "exact_transitions": 0,
                "exact_fraction": 0.0,
                "linf_mean": None,
                "linf_max": None,
            }
        errors = teacher_replay_linf[selected]
        return {
            "transitions": int(np.count_nonzero(selected)),
            "exact_transitions": int(np.count_nonzero(errors <= replay_tolerance)),
            "exact_fraction": float(np.mean(errors <= replay_tolerance)),
            "linf_mean": float(np.mean(errors)),
            "linf_max": float(np.max(errors)),
        }

    stage_replay = {}
    for stage in sorted(set(map(str, teacher_selected_stage))):
        stage_mask = native_anchor_labeled & np.asarray([
            str(value) == stage for value in teacher_selected_stage
        ])
        if np.any(stage_mask):
            stage_replay[stage] = replay_summary(stage_mask)
    report = {
        "dataset": "+".join(dataset_names),
        "dataset_components": dataset_names,
        "response_contract": manifest.get("response_contract", "legacy"),
        "action_schema_version": schema.get("version", "missing"),
        "observation_schema_version": manifest.get("observation_schema", {}).get(
            "version", "missing"
        ),
        "optimizer_anchor_transition_contracts": sorted(anchor_transition_contracts),
        "optimizer_preview_contracts": sorted(optimizer_preview_contracts),
        "warmup_control_contracts": sorted(warmup_control_contracts),
        "optimizer_anchor_contracts": sorted(anchor_contracts),
        "pfo_supervisor_flags": sorted(pfo_supervisor_flags),
        "pstack_anchor_flags": sorted(pstack_anchor_flags),
        "action_parameterization_supports": sorted(action_parameterization_supports),
        "experiment_contract_valid": not experiment_contract_errors,
        "experiment_contract_errors": experiment_contract_errors,
        "experiment_contract_sha256": sorted(experiment_contract_sha256),
        "experiment_profiles": sorted(experiment_profiles),
        "required_experiment_profile": EXPERIMENT_PROFILE_ID,
        "files": files,
        "transitions": int(action.shape[0]),
        "observation_dimension": int(manifest["observation_schema"]["dimension"]),
        "action_dimension": int(action.shape[1]),
        "action_support": {
            "min": action.min(0).tolist(),
            "max": action.max(0).tolist(),
            "std": action.std(0).tolist(),
            "dead_dimension_indices": dead_action_indices,
            "dead_dimension_names": [action_names[index] for index in dead_action_indices],
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
            "active_peak_transitions": int(np.count_nonzero(
                (np.asarray(perturb_active) > 0.5) & peak_mask
            )),
            "active_recovery_transitions": int(np.count_nonzero(
                (np.asarray(perturb_active) > 0.5) & recovery_mask
            )),
            "active_fraction": float(np.mean(np.asarray(perturb_active) > 0.5)),
            "nonzero_distance_fraction": float(np.mean(np.asarray(anchor_distance) > 1.0e-9)),
            "active_distance_mean": float(np.mean(
                np.asarray(anchor_distance)[np.asarray(perturb_active) > 0.5]
            )) if np.any(np.asarray(perturb_active) > 0.5) else 0.0,
            "selected_block_episode_counts": dict(sorted(perturb_block_episodes.items())),
        },
        "optimizer_teacher_labels": {
            "labeled_transitions": int(np.count_nonzero(teacher_labeled)),
            "leader_transitions": int(np.count_nonzero(teacher_leader)),
            "pfo_transitions": int(np.count_nonzero(teacher_pfo)),
            "outer_pfo_transitions": int(np.count_nonzero(
                teacher_outer_pfo_array > 0.5
            )),
            "pfo_fraction": float(np.mean(teacher_pfo_array[teacher_labeled] > 0.5))
            if np.any(teacher_labeled) else 0.0,
            "far_enabled_fraction": float(np.mean(teacher_far_array[teacher_labeled] > 0.5))
            if np.any(teacher_labeled) else 0.0,
            "saturated_fraction": float(np.mean(teacher_saturated_array[teacher_labeled] > 0.0))
            if np.any(teacher_labeled) else 0.0,
            "native_response_finite_fraction": float(np.mean(
                teacher_response_finite_array[teacher_labeled]
            )) if np.any(teacher_labeled) else 0.0,
            "replay_exact_tolerance": replay_tolerance,
            "native_anchor_transitions": int(np.count_nonzero(native_anchor_labeled)),
            "replay_exact_fraction": replay_summary(native_anchor_labeled)["exact_fraction"],
            "replay_linf_mean": replay_summary(native_anchor_labeled)["linf_mean"],
            "replay_linf_max": replay_summary(native_anchor_labeled)["linf_max"],
            "leader_replay": replay_summary(teacher_leader & native_anchor),
            "pfo_replay": replay_summary(teacher_pfo & native_anchor),
            "pstack_selected_anchor_replay": replay_summary(
                pstack_selected_native_anchor
            ),
            "stage_replay": stage_replay,
            "selected_stages": dict(Counter(
                str(stage)
                for stage, labeled in zip(teacher_selected_stage, teacher_labeled)
                if labeled
            )),
        },
        "continuous_actor_supervision": {
            "eligible_transitions": int(np.count_nonzero(actor_supervision)),
            "eligible_fraction": float(np.mean(actor_supervision)),
            "excluded_optimizer_anchor_transitions": int(np.count_nonzero(
                (np.asarray(modes) == "optimizer_anchor") & ~actor_supervision
            )),
        },
        "pstack_residual_actor_supervision": {
            "eligible_transitions": int(np.count_nonzero(residual_supervision)),
            "eligible_fraction": float(np.mean(residual_supervision)),
            "eligible_peak_transitions": int(np.count_nonzero(
                residual_supervision & (observation[:, peak_index] > 0.5)
            )),
            "eligible_recovery_transitions": int(np.count_nonzero(
                residual_supervision & recovery_mask
            )),
            "nonzero_peak_transitions": int(np.count_nonzero(
                nonzero_residual & peak_mask
            )),
            "nonzero_recovery_transitions": int(np.count_nonzero(
                nonzero_residual & recovery_mask
            )),
            "eligible_by_mode": dict(Counter(
                mode for mode, eligible in zip(modes, residual_supervision) if eligible
            )),
            "eligible_leader_transitions": int(np.count_nonzero(
                residual_supervision & teacher_leader
            )),
            "eligible_pfo_transitions": int(np.count_nonzero(
                residual_supervision & teacher_pfo
            )),
            "anchor_gate_rl_pick_transitions": int(np.count_nonzero(
                np.asarray(pstack_anchor_pick_rl, dtype=float) > 0.5
            )),
            "long_horizon_label_valid_transitions": int(np.count_nonzero(
                np.asarray(long_horizon_label_valid, dtype=float) > 0.5
            )),
            "long_horizon_positive_transitions": int(np.count_nonzero(
                np.asarray(long_horizon_positive, dtype=float) > 0.5
            )),
            "anchor_action_finite_transitions": int(np.count_nonzero(
                np.all(np.isfinite(anchor_action), axis=1)
            )),
            "residual_abs_max": float(np.max(np.abs(residual_values)))
            if residual_values.size else None,
            "dead_dimension_indices": dead_residual_indices,
            "dead_dimension_names": [action_names[index] for index in dead_residual_indices],
            "all_dead_dimension_indices": all_dead_residual_indices,
            "all_dead_dimension_names": [
                action_names[index] for index in all_dead_residual_indices
            ],
            "frozen_dimension_indices": frozen_residual_indices,
            "frozen_dimension_names": [
                action_names[index] for index in frozen_residual_indices
            ],
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
