"""Dataset ownership rules shared by auditing and offline training."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    ANCHORED_RESIDUAL_CONTRACT_VERSION,
)


TEACHER_REPLAY_TOLERANCE = 1.0e-4
PSTACK_RESIDUAL_DATA_CONTRACT = "pstack_residual_exact_native_potential_v4"
PSTACK_RESIDUAL_CRITIC_ACTION_CONTRACT = "pstack_gate_applied_exact_native_residual_v3"
PSTACK_RESIDUAL_ROW_FORMAT = "pstack_exact_native_anchor_rows_v2"
PSTACK_DATASET_FORMAT = "rl_coordination_dataset_v4_exact_native"
RL_CHECKPOINT_FORMAT = "rl_coordination_checkpoint_v4_exact_native"
IQL_TERMINAL_CONTRACT = "external_collection_truncation_bootstrap_v2"
IQL_REWARD_CONTRACT = "wall_clock_failure_cost_removed_v1"
LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT = "pstack_h12_positive_exact_native_residual_v3"


def iql_bootstrap_done(dataset) -> np.ndarray:
    """Treat external collection limits as truncations, not MDP terminals."""
    done = np.asarray(dataset["done"], dtype=np.float32).copy()
    if "termination_reason" not in dataset:
        return done
    reasons = np.asarray(dataset["termination_reason"]).astype(str)
    done[np.isin(reasons, ("collection_time_limit", "wall_clock_abort"))] = 0.0
    return done


def iql_training_rewards(dataset, failure_cost: float = 5000.0) -> np.ndarray:
    """Remove collector-only wall-clock penalties from critic rewards."""
    rewards = np.asarray(dataset["rew"], dtype=np.float32).copy()
    if "termination_reason" not in dataset:
        return rewards
    reasons = np.asarray(dataset["termination_reason"]).astype(str)
    rewards[reasons == "wall_clock_abort"] += float(failure_cost)
    return rewards


def continuous_actor_supervision_mask(dataset) -> np.ndarray:
    """Keep real behavior actions, but only replayable leader teacher anchors."""
    modes = np.asarray(dataset["behavior_mode"]).astype(str)
    mask = np.ones(modes.shape[0], dtype=bool)
    anchors = modes == "optimizer_anchor"
    if not np.any(anchors):
        return mask
    required = ("teacher_pfo_selected", "teacher_response", "response")
    missing = [name for name in required if name not in dataset]
    if missing:
        mask[anchors] = False
        return mask
    labels = np.asarray(dataset["teacher_pfo_selected"], dtype=float)
    teacher_response = np.asarray(dataset["teacher_response"], dtype=float)
    response = np.asarray(dataset["response"], dtype=float)
    replay_error = np.max(np.abs(response - teacher_response), axis=1)
    replayable_leader = (
        (labels >= 0.0)
        & (labels < 0.5)
        & np.isfinite(replay_error)
        & (replay_error <= TEACHER_REPLAY_TOLERANCE)
    )
    mask[anchors] = replayable_leader[anchors]
    return mask


def pstack_residual_actor_supervision_mask(dataset) -> np.ndarray:
    """Select deployed zero residuals and gate-accepted local RL proposals."""
    modes = np.asarray(dataset["behavior_mode"]).astype(str)
    mask = np.zeros(modes.shape[0], dtype=bool)
    _require_pstack_residual_rows(dataset)
    anchor_envelope = np.asarray(dataset["anchor_envelope"], dtype=float)
    finite_anchor = np.all(np.isfinite(anchor_envelope), axis=1)
    native_anchor = modes == "optimizer_anchor"
    local_anchor = modes == "optimizer_local"
    gate_pick_rl = np.asarray(dataset["pstack_anchor_pick_rl"], dtype=float) > 0.5
    selected_local = local_anchor & gate_pick_rl
    if "long_horizon_label_valid" in dataset and "long_horizon_positive" in dataset:
        selected_local &= (
            (np.asarray(dataset["long_horizon_label_valid"], dtype=float) > 0.5)
            & (np.asarray(dataset["long_horizon_positive"], dtype=float) > 0.5)
        )
    mask = finite_anchor & (native_anchor | selected_local)
    return mask


def pstack_residual_targets(dataset) -> np.ndarray:
    """Return the exact residual requested in native anchor coordinates."""
    _require_pstack_residual_rows(dataset)
    targets = np.asarray(dataset["policy_residual"], dtype=np.float32).copy()
    targets[:, ~pstack_residual_trainable_dimension_mask(dataset)] = 0.0
    return targets


def pstack_residual_deployed_actions(dataset) -> np.ndarray:
    """Return the exact residual selected by the transactional anchor gate."""
    _require_pstack_residual_rows(dataset)
    deployed = np.asarray(dataset["deployed_residual"], dtype=np.float32).copy()
    deployed[:, ~pstack_residual_trainable_dimension_mask(dataset)] = 0.0
    return deployed


def _require_pstack_residual_rows(dataset) -> None:
    required = (
        "policy_residual",
        "deployed_residual",
        "anchor_envelope",
        "anchor_selected_branch",
        "anchor_fingerprint",
        "pstack_anchor_pick_rl",
    )
    missing = [name for name in required if name not in dataset]
    if missing:
        raise ValueError(
            "dataset lacks native-anchor residual fields: " + ", ".join(missing)
        )
    policy = np.asarray(dataset["policy_residual"])
    deployed = np.asarray(dataset["deployed_residual"])
    if policy.ndim != 2 or deployed.shape != policy.shape:
        raise ValueError("policy/deployed residual arrays have incompatible shapes")
    rows = policy.shape[0]
    for name in (
        "anchor_envelope", "anchor_selected_branch", "anchor_fingerprint",
        "pstack_anchor_pick_rl",
    ):
        if np.asarray(dataset[name]).shape[0] != rows:
            raise ValueError(f"{name} row count does not match residual rows")


def anchor_envelope_fingerprint(
    envelope,
    selected_branch: str,
    envelope_metadata: dict,
) -> str:
    values = np.asarray(envelope, dtype="<f8").reshape(-1)
    expected = int(envelope_metadata.get("dimension", -1))
    if values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError("cannot fingerprint an invalid anchor envelope")
    digest = hashlib.sha256()
    digest.update(json.dumps(
        envelope_metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(selected_branch).encode("utf-8"))
    digest.update(b"\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def validate_pstack_residual_rows(dataset, manifest: dict) -> dict[str, int]:
    _require_pstack_residual_rows(dataset)
    if manifest.get("format_version") != PSTACK_DATASET_FORMAT:
        raise ValueError("dataset format is not the native-anchor format")
    if manifest.get("action_parameterization_support") != PSTACK_RESIDUAL_DATA_CONTRACT:
        raise ValueError("dataset residual contract does not match the runtime")
    if manifest.get("pstack_residual_row_format") != PSTACK_RESIDUAL_ROW_FORMAT:
        raise ValueError("dataset row format does not preserve native anchors")
    action_schema = manifest.get("action_schema", {})
    if action_schema.get("version") != ACTION_SCHEMA_VERSION:
        raise ValueError("dataset action schema does not match the runtime")
    if (
        action_schema.get("anchored_residual_contract")
        != ANCHORED_RESIDUAL_CONTRACT_VERSION
    ):
        raise ValueError("dataset action coordinates are not native-anchor deltas")
    envelope_metadata = manifest.get("anchor_envelope_schema", {})
    if envelope_metadata != action_schema.get("anchor_envelope"):
        raise ValueError("dataset anchor envelope metadata is inconsistent")

    policy = np.asarray(dataset["policy_residual"], dtype=np.float32)
    deployed = np.asarray(dataset["deployed_residual"], dtype=np.float32)
    envelopes = np.asarray(dataset["anchor_envelope"], dtype=np.float64)
    branches = np.asarray(dataset["anchor_selected_branch"]).astype(str)
    fingerprints = np.asarray(dataset["anchor_fingerprint"]).astype(str)
    picks = np.asarray(dataset["pstack_anchor_pick_rl"], dtype=float)
    expected_envelope_shape = (
        policy.shape[0], int(envelope_metadata.get("dimension", -1))
    )
    if envelopes.shape != expected_envelope_shape:
        raise ValueError(
            "anchor envelope shape does not match its schema: "
            f"{envelopes.shape} != {expected_envelope_shape}"
        )
    if not np.all(np.isfinite(policy)) or not np.all(np.isfinite(deployed)):
        raise ValueError("residual rows contain nonfinite values")
    if not np.all(np.isfinite(envelopes)):
        raise ValueError("anchor envelope rows contain nonfinite values")
    if np.any(~np.isin(picks, (0.0, 1.0))):
        raise ValueError("anchor gate selections must be binary")
    expected_deployed = np.where(picks[:, None] > 0.5, policy, 0.0)
    if not np.array_equal(deployed, expected_deployed):
        raise ValueError("deployed residual does not match the anchor gate selection")
    for index, (envelope, branch, fingerprint) in enumerate(
        zip(envelopes, branches, fingerprints)
    ):
        if not branch or branch == "none":
            raise ValueError(f"anchor row {index} is missing selected branch identity")
        expected_fingerprint = anchor_envelope_fingerprint(
            envelope, branch, envelope_metadata
        )
        if fingerprint != expected_fingerprint:
            raise ValueError(f"anchor row {index} fingerprint mismatch")
    return {
        "rows": int(policy.shape[0]),
        "rl_selected_rows": int(np.count_nonzero(picks > 0.5)),
    }


def pstack_residual_trainable_dimension_mask(dataset) -> np.ndarray:
    """Keep continuous budget/price deltas while freezing binary certificates."""
    action_dimension = int(np.asarray(dataset["act"]).shape[-1])
    mask = np.ones(action_dimension, dtype=bool)
    if "manifest_json" not in dataset:
        return mask
    value = dataset["manifest_json"]
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    try:
        manifest = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return mask
    names = manifest.get("action_schema", {}).get("names", [])
    if len(names) != action_dimension:
        return mask
    return np.asarray([
        not str(name).startswith("certificate.") for name in names
    ], dtype=bool)


def scenario_phase_cells(dataset, manifest: dict) -> np.ndarray:
    """Build stable scenario x traffic-phase cells for balanced minibatches."""
    episode_to_scenario = {
        int(summary.get("episode", -1)): str(
            summary.get("scenario", {}).get("target_scenario", "unknown")
        )
        for summary in manifest.get("episode_summaries", [])
    }
    episodes = np.asarray(dataset["episode"], dtype=int)
    times = np.asarray(dataset["simulation_time_sec"], dtype=float)
    return np.asarray([
        f"{episode_to_scenario.get(int(episode), 'unknown')}|"
        f"{'peak' if time_sec < 5220.0 else 'recovery'}"
        for episode, time_sec in zip(episodes, times)
    ])


def balanced_sampling_probabilities(cells, eligible=None) -> np.ndarray:
    """Give every represented cell equal probability mass."""
    values = np.asarray(cells).astype(str)
    selected = (
        np.ones(values.shape[0], dtype=bool)
        if eligible is None else np.asarray(eligible, dtype=bool)
    )
    probabilities = np.zeros(values.shape[0], dtype=np.float64)
    selected_values = values[selected]
    if selected_values.size == 0:
        return probabilities
    unique, counts = np.unique(selected_values, return_counts=True)
    mass = 1.0 / float(unique.size)
    for value, count in zip(unique, counts):
        probabilities[selected & (values == value)] = mass / float(count)
    return probabilities
