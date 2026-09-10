"""Fail-closed contract for branch-aware follower-oracle candidate labels."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np

from src.controllers.coordination import ACTION_SCHEMA_VERSION


ORACLE_LABEL_DATASET_FORMAT = "pcent_guided_oracle_labels_v1_branch_aware"
ORACLE_CANDIDATE_ROW_FORMAT = "oracle_candidate_rows_v1"
ORACLE_RANKER_CHECKPOINT_FORMAT = (
    "rl_coordination_checkpoint_v5_branch_aware_oracle"
)
ORACLE_EXECUTION_BRANCHES = {"native_anchor", "coordination"}
ORACLE_NATIVE_BRANCHES = {
    "coarse", "refined", "fallback_pfo", "supervisor_pfo", "other",
}
ORACLE_CANDIDATE_FAMILIES = {
    "native_anchor",
    "coordination_zero",
    "pcent_guided",
    "schema_structured",
    "random_orthogonal",
    "local_jacobian",
}
DEPLOYMENT_RANKER_FEATURE_FIELDS = (
    "observation",
    "anchor_envelope",
    "native_anchor_branch",
    "execution_branch",
    "continuous_residual",
)
PRIVILEGED_PROVENANCE_FIELDS = {
    "pcent_control",
    "pcent_objective",
    "pcent_response",
    "pcent_solver_evaluations",
    "distance_to_pcent",
    "forecast",
    "forecast_sha256",
}
INVENTORY_GUARD_TOLERANCE_VEH = 1.0e-6


def _is_sha256(value) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _residual_sha256(values) -> str:
    residual = np.asarray(values, dtype="<f4").reshape(-1).copy()
    residual[residual == 0.0] = 0.0
    return hashlib.sha256(residual.tobytes()).hexdigest()


def _validate_h1_replay_evidence(
    evidence: Mapping,
    candidate_id: str,
    *,
    expected_response: np.ndarray,
    expected_follower_memory_sha256: str,
    expected_h1_follower_memory_sha256: str,
    expected_physical_sha256: str,
    expected_h1_ttt: float,
    expected_h1_terminal_inventory: float,
    expected_h1_validity_gate_pass: bool,
) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"candidate {candidate_id} lacks H1 replay evidence")
    try:
        probe_response = np.asarray(evidence.get("probe_response"), dtype=float)
        rollout_response = np.asarray(evidence.get("rollout_response"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"candidate {candidate_id} H1 replay response is invalid"
        ) from exc
    if (
        probe_response.shape != expected_response.shape
        or rollout_response.shape != expected_response.shape
        or not np.all(np.isfinite(probe_response))
        or not np.all(np.isfinite(rollout_response))
        or not np.array_equal(probe_response, expected_response)
    ):
        raise ValueError(f"candidate {candidate_id} H1 replay response is unbound")
    error = evidence.get("response_linf")
    tolerance = evidence.get("response_tolerance")
    if not _finite_number(error) or not _finite_number(tolerance):
        raise ValueError(f"candidate {candidate_id} H1 replay metrics are invalid")
    if float(error) < 0.0 or float(tolerance) < 0.0:
        raise ValueError(f"candidate {candidate_id} H1 replay metrics are invalid")
    probe_memory = evidence.get("probe_follower_memory_sha256")
    rollout_memory = evidence.get("rollout_follower_memory_sha256")
    memory_exact = (
        _is_sha256(probe_memory)
        and _is_sha256(rollout_memory)
        and probe_memory == expected_follower_memory_sha256
        and probe_memory == expected_h1_follower_memory_sha256
        and probe_memory == rollout_memory
    )
    probe_physical = evidence.get("probe_physical_sha256")
    rollout_physical = evidence.get("rollout_physical_sha256")
    physical_exact = (
        _is_sha256(probe_physical)
        and _is_sha256(rollout_physical)
        and probe_physical == expected_physical_sha256
        and probe_physical == rollout_physical
    )
    measured_error = float(np.max(np.abs(rollout_response - probe_response)))
    response_exact = measured_error <= float(tolerance)
    metric_fields = (
        "probe_step_ttt",
        "rollout_step_ttt",
        "probe_terminal_inventory",
        "rollout_terminal_inventory",
    )
    if not all(_finite_number(evidence.get(field)) for field in metric_fields):
        raise ValueError(f"candidate {candidate_id} H1 replay metrics are invalid")
    ttt_exact = (
        float(evidence["probe_step_ttt"]) == float(expected_h1_ttt)
        and float(evidence["rollout_step_ttt"]) == float(expected_h1_ttt)
    )
    inventory_exact = (
        float(evidence["probe_terminal_inventory"])
        == float(expected_h1_terminal_inventory)
        and float(evidence["rollout_terminal_inventory"])
        == float(expected_h1_terminal_inventory)
    )
    validity_exact = (
        evidence.get("probe_validity_gate_pass")
        is bool(expected_h1_validity_gate_pass)
        and evidence.get("rollout_validity_gate_pass")
        is bool(expected_h1_validity_gate_pass)
    )
    if (
        evidence.get("response_exact") is not True
        or evidence.get("follower_memory_exact") is not True
        or evidence.get("physical_exact") is not True
        or evidence.get("step_ttt_exact") is not True
        or evidence.get("terminal_inventory_exact") is not True
        or evidence.get("validity_gate_exact") is not True
        or evidence.get("passed") is not True
        or not response_exact
        or not memory_exact
        or not physical_exact
        or not ttt_exact
        or not inventory_exact
        or not validity_exact
        or not math.isclose(float(error), measured_error, abs_tol=1.0e-12)
    ):
        raise ValueError(f"candidate {candidate_id} H1 replay evidence failed")


def _validate_h3_selection_replay_evidence(
    evidence: Mapping,
    candidate_id: str,
    *,
    expected_h3_label: Mapping,
) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"candidate {candidate_id} lacks H3 selection replay evidence")
    numeric_pairs = (
        ("selector_ttt", "rollout_ttt", "candidate_ttt"),
        (
            "selector_terminal_inventory",
            "rollout_terminal_inventory",
            "candidate_terminal_inventory",
        ),
    )
    for selector_field, rollout_field, label_field in numeric_pairs:
        if not _finite_number(evidence.get(selector_field)) or not _finite_number(
            evidence.get(rollout_field)
        ):
            raise ValueError(
                f"candidate {candidate_id} H3 selection replay metrics are invalid"
            )
        expected = float(expected_h3_label[label_field])
        if (
            float(evidence[selector_field]) != expected
            or float(evidence[rollout_field]) != expected
        ):
            raise ValueError(
                f"candidate {candidate_id} H3 selection replay metrics are unbound"
            )
    for selector_field, rollout_field, label_field in (
        (
            "selector_follower_memory_sha256",
            "rollout_follower_memory_sha256",
            "candidate_follower_memory_sha256",
        ),
        (
            "selector_physical_sha256",
            "rollout_physical_sha256",
            "candidate_physical_state_sha256",
        ),
    ):
        expected = expected_h3_label.get(label_field)
        if (
            not _is_sha256(expected)
            or evidence.get(selector_field) != expected
            or evidence.get(rollout_field) != expected
        ):
            raise ValueError(
                f"candidate {candidate_id} H3 selection replay hashes are unbound"
            )
    expected_validity = bool(expected_h3_label["validity_gate_pass"])
    if (
        evidence.get("selector_validity_gate_pass") is not expected_validity
        or evidence.get("rollout_validity_gate_pass") is not expected_validity
        or any(evidence.get(field) is not True for field in (
            "ttt_exact",
            "terminal_inventory_exact",
            "follower_memory_exact",
            "physical_exact",
            "validity_gate_exact",
            "passed",
        ))
    ):
        raise ValueError(f"candidate {candidate_id} H3 selection replay evidence failed")


def _validate_horizon_label(
    label: Mapping,
    *,
    horizon: int,
    tolerance: float,
) -> None:
    if not isinstance(label, Mapping):
        raise ValueError(f"H{horizon} label must be a mapping")
    valid = bool(label.get("label_valid", False))
    positive = bool(label.get("positive", False))
    if not valid:
        if positive:
            raise ValueError(f"invalid H{horizon} label cannot be positive")
        return

    required = (
        "candidate_steps", "native_steps", "candidate_ttt", "native_ttt",
        "ttt_gain", "required_gain", "candidate_terminal_inventory",
        "native_terminal_inventory", "terminal_inventory_delta",
        "inventory_guard_pass", "validity_gate_pass",
        "candidate_follower_memory_sha256",
        "native_follower_memory_sha256",
    )
    missing = [name for name in required if name not in label]
    if missing:
        raise ValueError(f"H{horizon} label is missing fields: {missing}")
    if int(label["candidate_steps"]) != horizon or int(label["native_steps"]) != horizon:
        raise ValueError(f"H{horizon} label did not complete the requested horizon")
    numeric = required[2:9]
    if not all(_finite_number(label[name]) for name in numeric):
        raise ValueError(f"H{horizon} label contains nonfinite metrics")
    if not _is_sha256(label["candidate_follower_memory_sha256"]):
        raise ValueError(f"H{horizon} candidate follower memory hash is invalid")
    if not _is_sha256(label["native_follower_memory_sha256"]):
        raise ValueError(f"H{horizon} native follower memory hash is invalid")
    for field in (
        "candidate_physical_state_sha256", "native_physical_state_sha256",
    ):
        if field in label and not _is_sha256(label[field]):
            raise ValueError(f"H{horizon} physical state hash is invalid")

    candidate_ttt = float(label["candidate_ttt"])
    native_ttt = float(label["native_ttt"])
    gain = float(label["ttt_gain"])
    required_gain = float(label["required_gain"])
    candidate_inventory = float(label["candidate_terminal_inventory"])
    native_inventory = float(label["native_terminal_inventory"])
    inventory_delta = float(label["terminal_inventory_delta"])
    expected_required = max(0.1, 1.0e-3 * max(abs(native_ttt), 1.0))
    if not math.isclose(gain, native_ttt - candidate_ttt, abs_tol=1.0e-9):
        raise ValueError(f"H{horizon} TTT gain is inconsistent")
    if not math.isclose(required_gain, expected_required, abs_tol=1.0e-9):
        raise ValueError(f"H{horizon} required gain is inconsistent")
    if not math.isclose(
        inventory_delta, candidate_inventory - native_inventory, abs_tol=1.0e-9
    ):
        raise ValueError(f"H{horizon} inventory delta is inconsistent")
    guard_pass = inventory_delta <= tolerance
    if bool(label["inventory_guard_pass"]) != guard_pass:
        raise ValueError(f"H{horizon} inventory guard is inconsistent")
    expected_positive = bool(
        label["validity_gate_pass"] is True
        and gain > required_gain and guard_pass
    )
    if positive != expected_positive:
        raise ValueError(f"H{horizon} positive label is inconsistent")


def validate_oracle_label_artifact(
    artifact: Mapping,
    *,
    require_decision_horizon: bool = False,
) -> dict[str, int]:
    """Validate candidate pools without treating native anchors as zero residuals."""
    if artifact.get("format_version") != ORACLE_LABEL_DATASET_FORMAT:
        raise ValueError("oracle label dataset format mismatch")
    if artifact.get("row_format") != ORACLE_CANDIDATE_ROW_FORMAT:
        raise ValueError("oracle candidate row format mismatch")
    action_schema = artifact.get("action_schema", {})
    if action_schema.get("version") != ACTION_SCHEMA_VERSION:
        raise ValueError("oracle action schema does not match the runtime")
    action_dim = int(action_schema.get("dimension", -1))
    action_names = action_schema.get("names", [])
    if action_dim <= 0 or len(action_names) != action_dim:
        raise ValueError("oracle action schema dimension is invalid")

    horizons = tuple(int(value) for value in artifact.get("label_horizons", ()))
    if not horizons or tuple(sorted(set(horizons))) != horizons or min(horizons) < 1:
        raise ValueError("oracle label horizons are invalid")
    decision_horizon = int(artifact.get("decision_horizon", 12))
    if decision_horizon not in horizons:
        raise ValueError("decision horizon is absent from label horizons")
    tolerance = float(
        artifact.get(
            "inventory_guard_tolerance_veh", INVENTORY_GUARD_TOLERANCE_VEH
        )
    )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("inventory guard tolerance is invalid")

    features = tuple(artifact.get("ranker_feature_fields", ()))
    if features != DEPLOYMENT_RANKER_FEATURE_FIELDS:
        raise ValueError("ranker feature contract mismatch")
    leaked = sorted(set(features) & PRIVILEGED_PROVENANCE_FIELDS)
    if leaked:
        raise ValueError(f"privileged P-CENT fields leaked into ranker inputs: {leaked}")
    if bool(artifact.get("legacy_migration_allowed", True)):
        raise ValueError("legacy residual datasets cannot be migrated into oracle labels")

    implementation = artifact.get("implementation_sha256", {})
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("oracle implementation provenance is missing")
    if any(not _is_sha256(value) for value in implementation.values()):
        raise ValueError("oracle implementation provenance contains an invalid hash")

    pools = artifact.get("candidate_pools", [])
    if not isinstance(pools, Sequence) or not pools:
        raise ValueError("oracle artifact contains no candidate pools")
    pool_ids: set[str] = set()
    row_count = 0
    positive_count = 0
    for pool in pools:
        pool_id = str(pool.get("candidate_pool_id", ""))
        if not pool_id or pool_id in pool_ids:
            raise ValueError("candidate pool ids must be nonempty and unique")
        pool_ids.add(pool_id)
        if pool.get("native_anchor_branch") not in ORACLE_NATIVE_BRANCHES:
            raise ValueError(f"candidate pool {pool_id} has an unknown native branch")
        if not _is_sha256(pool.get("anchor_fingerprint", "")):
            raise ValueError(f"candidate pool {pool_id} has an invalid anchor fingerprint")
        for field in (
            "pre_runtime_sha256", "forecast_sha256", "candidate_manifest_sha256",
        ):
            if not _is_sha256(pool.get(field, "")):
                raise ValueError(
                    f"candidate pool {pool_id} provenance hash {field} is invalid"
                )
        if pool.get("probe_order_verified") is not True:
            raise ValueError(f"candidate pool {pool_id} lacks probe-order verification")
        if pool.get("source_isolation_verified") is not True:
            raise ValueError(f"candidate pool {pool_id} lacks source isolation")
        for field in ("identity_parity", "zero_round_trip"):
            evidence = pool.get(field)
            if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
                raise ValueError(f"candidate pool {pool_id} lacks {field} parity")
        selected_ids = pool.get("selected_h12_candidate_ids")
        if (
            not isinstance(selected_ids, Sequence)
            or isinstance(selected_ids, (str, bytes))
            or len(set(map(str, selected_ids))) != len(selected_ids)
        ):
            raise ValueError(f"candidate pool {pool_id} selected H12 ids are invalid")
        selected_ids = set(map(str, selected_ids))
        rows = pool.get("rows", [])
        if not isinstance(rows, Sequence) or not rows:
            raise ValueError(f"candidate pool {pool_id} contains no rows")
        native_rows = [row for row in rows if row.get("execution_branch") == "native_anchor"]
        winners = [row for row in rows if bool(row.get("oracle_is_winner", False))]
        if len(native_rows) != 1:
            raise ValueError(f"candidate pool {pool_id} must contain one native anchor")
        if len(winners) != 1:
            raise ValueError(f"candidate pool {pool_id} must contain one oracle winner")

        candidate_ids: set[str] = set()
        decision_positive_rows = []
        eligible_coordination_ids: set[str] = set()
        for row in rows:
            row_count += 1
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"candidate ids in pool {pool_id} must be unique")
            candidate_ids.add(candidate_id)
            branch = row.get("execution_branch")
            family = row.get("candidate_family")
            if branch not in ORACLE_EXECUTION_BRANCHES:
                raise ValueError(f"candidate {candidate_id} has an invalid branch")
            if family not in ORACLE_CANDIDATE_FAMILIES:
                raise ValueError(f"candidate {candidate_id} has an invalid family")
            if family == "direct_pcent":
                raise ValueError("direct P-CENT cannot be an executable oracle label")

            residual_valid = bool(row.get("continuous_residual_valid", False))
            residual = row.get("continuous_residual")
            if branch == "native_anchor":
                if family != "native_anchor" or residual_valid or residual is not None:
                    raise ValueError(
                        "native anchor must use a null, invalid continuous residual"
                    )
            else:
                if family == "native_anchor" or not residual_valid:
                    raise ValueError("coordination rows require a valid residual")
                values = np.asarray(residual, dtype=float)
                if values.shape != (action_dim,) or not np.all(np.isfinite(values)):
                    raise ValueError("coordination residual shape or values are invalid")
                if family == "coordination_zero" and not np.array_equal(
                    values, np.zeros(action_dim, dtype=float)
                ):
                    raise ValueError("coordination_zero must use an exact zero residual")
                if not _is_sha256(row.get("residual_sha256", "")):
                    raise ValueError(f"candidate {candidate_id} residual hash is invalid")
                if row["residual_sha256"] != _residual_sha256(values):
                    raise ValueError(f"candidate {candidate_id} residual hash is inconsistent")
                for field in ("post_follower_sha256", "post_physical_sha256"):
                    if not _is_sha256(row.get(field, "")):
                        raise ValueError(
                            f"candidate {candidate_id} provenance hash {field} is invalid"
                        )
                try:
                    candidate_response = np.asarray(
                        row.get("candidate_response"), dtype=float
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"candidate {candidate_id} response provenance is invalid"
                    ) from exc
                if (
                    candidate_response.ndim != 1
                    or candidate_response.size == 0
                    or not np.all(np.isfinite(candidate_response))
                ):
                    raise ValueError(
                        f"candidate {candidate_id} response provenance is invalid"
                    )
                budget = row.get("requested_budget")
                if not isinstance(budget, Mapping) or not all(
                    _finite_number(budget.get(field))
                    for field in ("N_P_star", "N_UF_star")
                ):
                    raise ValueError(f"candidate {candidate_id} requested budget is invalid")
                raw_budget = budget.get("raw_budget")
                if raw_budget is not None and (
                    not isinstance(raw_budget, Sequence)
                    or isinstance(raw_budget, (str, bytes))
                    or len(raw_budget) != 2
                    or not all(_finite_number(value) for value in raw_budget)
                ):
                    raise ValueError(f"candidate {candidate_id} raw budget is invalid")
                checks = row.get("budget_intent_checks")
                required_checks = (
                    "N_P_star_exact", "N_UF_star_exact", "raw_budget_exact",
                    "certificate_exact",
                )
                if not isinstance(checks, Mapping) or any(
                    checks.get(field) is not True for field in required_checks
                ):
                    raise ValueError(f"candidate {candidate_id} budget intent failed")

            labels = row.get("horizon_labels", {})
            if set(labels) != {str(horizon) for horizon in horizons}:
                raise ValueError(f"candidate {candidate_id} horizon labels are incomplete")
            for horizon in horizons:
                _validate_horizon_label(
                    labels[str(horizon)],
                    horizon=horizon,
                    tolerance=tolerance,
                )
            decision_label = labels[str(decision_horizon)]
            if bool(decision_label.get("positive", False)):
                positive_count += 1
                decision_positive_rows.append(row)
            if bool(row.get("ranker_eligible", False)) != bool(
                decision_label.get("label_valid", False)
            ):
                raise ValueError(f"candidate {candidate_id} ranker eligibility is inconsistent")
            if branch == "coordination" and bool(row.get("ranker_eligible", False)):
                eligible_coordination_ids.add(candidate_id)
                _validate_h1_replay_evidence(
                    row.get("h1_replay_evidence"),
                    candidate_id,
                    expected_response=candidate_response,
                    expected_follower_memory_sha256=str(
                        row["post_follower_sha256"]
                    ),
                    expected_h1_follower_memory_sha256=str(
                        labels["1"]["candidate_follower_memory_sha256"]
                    ),
                    expected_physical_sha256=str(row["post_physical_sha256"]),
                    expected_h1_ttt=float(labels["1"]["candidate_ttt"]),
                    expected_h1_terminal_inventory=float(
                        labels["1"]["candidate_terminal_inventory"]
                    ),
                    expected_h1_validity_gate_pass=bool(
                        labels["1"]["validity_gate_pass"]
                    ),
                )
                if str(pool.get("candidate_mode", "")).startswith("owner_block_v2_"):
                    _validate_h3_selection_replay_evidence(
                        row.get("h3_selection_replay_evidence"),
                        candidate_id,
                        expected_h3_label=labels["3"],
                    )

        if selected_ids != eligible_coordination_ids:
            raise ValueError(
                f"candidate pool {pool_id} selected H12 ids do not match eligible rows"
            )

        winner = winners[0]
        if decision_positive_rows:
            expected = min(
                decision_positive_rows,
                key=lambda row: (
                    float(row["horizon_labels"][str(decision_horizon)]["candidate_ttt"]),
                    str(row["candidate_id"]),
                ),
            )
            if winner["candidate_id"] != expected["candidate_id"]:
                raise ValueError(f"candidate pool {pool_id} winner is not the safest best TTT")
        elif winner.get("execution_branch") != "native_anchor":
            raise ValueError(
                f"candidate pool {pool_id} must fall back to native anchor"
            )
        if require_decision_horizon and not eligible_coordination_ids:
            raise ValueError(
                f"candidate pool {pool_id} lacks coordination decision-horizon labels"
            )

    return {
        "candidate_pools": len(pools),
        "rows": row_count,
        "positive_rows": positive_count,
    }
