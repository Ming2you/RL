"""Run a same-state, equal-budget candidate-generator ablation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from rl_leader.diagnose_anchor_context_parity import _dynamic_follower_payload
from rl_leader.diagnose_pcent_guided_oracle import (
    OracleGateError,
    _long_zero_parity,
    _require_rollout_coverage,
    _runtime_fingerprint,
    _zero_round_trip,
)
from rl_leader.diagnose_pcent_reachability import (
    _directed_groups,
    _response_scales_and_families,
    response_distance,
)
from rl_leader.diagnose_phase0_parity import (
    _control_payload,
    _digest,
    _physical_snapshot,
)
from rl_leader.diagnose_reachable_candidate_attribution import (
    _paired_metrics,
    _rollout_price_candidate,
    _rollout_pstack,
)
from rl_leader.env import PSTACK_ANCHOR_CONTEXT_CONTRACT, RLLeaderEnv
from rl_leader.generate_long_horizon_labels import counterfactual_verdict
from rl_leader.oracle_candidate_ablation import (
    ABLATION_CONTRACT,
    ABLATION_GENERATORS,
    OWNER_BLOCK_ABLATION_CONTRACT,
    AblationConfig,
    CandidateSpec,
    candidate_manifest_sha256,
    generate_known_positive_canary,
    generate_owner_block_pcent_candidates,
    generate_owner_block_structured_candidates,
    generate_jacobian_secant_candidates,
    generate_orthogonal_random_candidates,
    generate_pcent_sign_candidates,
    generate_structured_candidates,
    residual_sha256,
    select_h12_candidates_without_teacher,
    synthesize_jacobian_inverse_candidates,
    validate_candidate_specs,
)
from rl_leader.oracle_label_contract import (
    DEPLOYMENT_RANKER_FEATURE_FIELDS,
    INVENTORY_GUARD_TOLERANCE_VEH,
    ORACLE_CANDIDATE_ROW_FORMAT,
    ORACLE_LABEL_DATASET_FORMAT,
    validate_oracle_label_artifact,
)
from rl_leader.run_pcent_matched import _teacher_contract
from src.controllers.centralized_mpc import CentralizedMPC


ROOT = Path(__file__).resolve().parents[1]
LABEL_HORIZONS = (1, 3, 6, 12)
IMPLEMENTATION_SOURCES = (
    "rl_leader/diagnose_candidate_ablation.py",
    "rl_leader/oracle_candidate_ablation.py",
    "rl_leader/oracle_label_contract.py",
    "rl_leader/diagnose_pcent_guided_oracle.py",
    "rl_leader/diagnose_reachable_candidate_attribution.py",
    "rl_leader/env.py",
    "src/controllers/centralized_mpc.py",
    "src/controllers/coordination.py",
    "src/controllers/rl_stackelberg.py",
    "src/controllers/stackelberg_wu_metered.py",
    "src/controllers/wu_faithful_follower.py",
    "src/simulation/coupling.py",
    "src/simulation/simulator.py",
)
FAMILY_MAP = {
    "pcent_sign": "pcent_guided",
    "structured": "schema_structured",
    "pcent_jacobian": "local_jacobian",
    "orthogonal_random": "random_orthogonal",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_fingerprints() -> dict[str, str]:
    return {
        relative: _sha256_file(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _probe_candidate(
    source: RLLeaderEnv,
    anchor_context,
    spec: CandidateSpec,
    *,
    target_response: np.ndarray,
    response_scales: np.ndarray,
    response_families: dict[str, list[int]],
) -> dict:
    trial, trial_context = copy.deepcopy((source, anchor_context))
    _, reward, _, info = trial.step_anchored_candidate(
        spec.residual_array(), trial_context
    )
    requested = trial.last_requested_coordination
    if requested is None:
        raise OracleGateError(
            "ablation candidate did not preserve its requested coordination",
            {"candidate_id": spec.candidate_id},
        )
    anchor = anchor_context.coordination
    budget_checks = {
        "N_P_star_exact": requested.N_P_star == anchor.N_P_star,
        "N_UF_star_exact": requested.N_UF_star == anchor.N_UF_star,
        "raw_budget_exact": requested.raw_budget == anchor.raw_budget,
        "certificate_exact": (
            requested.metering_release_certified
            == anchor.metering_release_certified
        ),
    }
    if not all(budget_checks.values()):
        raise OracleGateError(
            "linear-price candidate changed budget or certificate intent",
            {"candidate_id": spec.candidate_id, "checks": budget_checks},
        )
    response = trial.response_vector(trial.previous)
    post_memory = _digest(_dynamic_follower_payload(trial.controller.nash_solver))
    return {
        "candidate_id": spec.candidate_id,
        "generator": spec.generator,
        "phase": spec.phase,
        "candidate_label": spec.label,
        "residual": spec.residual_array().astype(float).tolist(),
        "residual_sha256": residual_sha256(spec.residual),
        "residual_l2": float(np.linalg.norm(spec.residual_array())),
        "uses_pcent_target": bool(spec.uses_pcent_target),
        "teacher_signal_missing": bool(spec.teacher_signal_missing),
        "generator_metadata": copy.deepcopy(spec.metadata),
        "response": response.astype(float).tolist(),
        "distance_to_pcent": response_distance(
            response, target_response, response_scales, response_families
        ),
        "step_ttt": -float(reward),
        "terminal_inventory": float(trial._inventory()),
        "validity_gate_pass": bool(info["validity_gate_pass"]),
        "post_follower_sha256": post_memory,
        "post_physical_sha256": _digest(_physical_snapshot(trial)),
        "control": _control_payload(trial.previous),
        "requested_budget": {
            "N_P_star": float(requested.N_P_star),
            "N_UF_star": float(requested.N_UF_star),
            "raw_budget": (
                None if requested.raw_budget is None
                else list(map(float, requested.raw_budget))
            ),
        },
        "budget_intent_checks": budget_checks,
    }


def _probe_unique_specs(
    source,
    anchor_context,
    specs: list[CandidateSpec],
    physical_records: dict[str, dict],
    *,
    target_response,
    response_scales,
    response_families,
) -> None:
    representative: dict[str, CandidateSpec] = {}
    for spec in specs:
        representative.setdefault(residual_sha256(spec.residual), spec)
    for digest, spec in sorted(representative.items()):
        if digest in physical_records:
            continue
        physical_records[digest] = _probe_candidate(
            source,
            anchor_context,
            spec,
            target_response=target_response,
            response_scales=response_scales,
            response_families=response_families,
        )


def _logical_records(
    specs: list[CandidateSpec], physical_records: dict[str, dict]
) -> list[dict]:
    records = []
    for spec in specs:
        digest = residual_sha256(spec.residual)
        physical = copy.deepcopy(physical_records[digest])
        physical.update({
            "candidate_id": spec.candidate_id,
            "generator": spec.generator,
            "phase": spec.phase,
            "candidate_label": spec.label,
            "uses_pcent_target": bool(spec.uses_pcent_target),
            "teacher_signal_missing": bool(spec.teacher_signal_missing),
            "generator_metadata": copy.deepcopy(spec.metadata),
        })
        outcome_payload = {
            "response": np.round(
                np.asarray(physical["response"], dtype=float), 6
            ).tolist(),
            "post_follower_sha256": physical["post_follower_sha256"],
        }
        physical["response_memory_outcome_sha256"] = _digest(outcome_payload)
        records.append(physical)
    return records


def _verify_probe_order(
    source,
    anchor_context,
    physical_records: dict[str, dict],
    specs: list[CandidateSpec],
    *,
    target_response,
    response_scales,
    response_families,
) -> dict:
    by_digest = {
        residual_sha256(spec.residual): spec for spec in specs
    }
    checks = []
    for digest in sorted(by_digest, reverse=True):
        replay = _probe_candidate(
            source,
            anchor_context,
            by_digest[digest],
            target_response=target_response,
            response_scales=response_scales,
            response_families=response_families,
        )
        original = physical_records[digest]
        checks.append({
            "residual_sha256": digest,
            "response_exact": replay["response"] == original["response"],
            "step_ttt_exact": replay["step_ttt"] == original["step_ttt"],
            "post_follower_exact": (
                replay["post_follower_sha256"]
                == original["post_follower_sha256"]
            ),
            "post_physical_exact": (
                replay["post_physical_sha256"]
                == original["post_physical_sha256"]
            ),
            "budget_intent_exact": (
                replay["requested_budget"] == original["requested_budget"]
            ),
        })
    passed = all(all(
        value for key, value in row.items() if key != "residual_sha256"
    ) for row in checks)
    if not passed:
        raise OracleGateError(
            "ablation H1 probes depend on evaluation order",
            {"checks": checks},
        )
    return {"performed": True, "passed": True, "checks": checks}


def _invalid_horizon_label() -> dict:
    return {"label_valid": False, "positive": False}


def _horizon_label(
    candidate: dict,
    native: dict,
    *,
    horizon: int,
) -> dict:
    verdict = counterfactual_verdict(
        candidate_ttt=float(candidate["ttt"]),
        pstack_ttt=float(native["ttt"]),
        candidate_terminal_inventory=float(candidate["terminal_inventory"]),
        pstack_terminal_inventory=float(native["terminal_inventory"]),
    )
    valid = bool(
        candidate.get("validity_gate_pass", False)
        and native.get("validity_gate_pass", False)
    )
    return {
        "label_valid": True,
        "candidate_steps": int(horizon),
        "native_steps": int(horizon),
        "candidate_ttt": float(candidate["ttt"]),
        "native_ttt": float(native["ttt"]),
        "ttt_gain": float(verdict["ttt_gain"]),
        "required_gain": float(verdict["required_gain"]),
        "candidate_terminal_inventory": float(candidate["terminal_inventory"]),
        "native_terminal_inventory": float(native["terminal_inventory"]),
        "terminal_inventory_delta": float(verdict["terminal_inventory_delta"]),
        "inventory_guard_pass": not bool(verdict["inventory_blocked"]),
        "validity_gate_pass": valid,
        "positive": bool(verdict["long_horizon_positive"] and valid),
        "candidate_follower_memory_sha256": str(
            candidate["follower_memory_sha256"]
        ),
        "native_follower_memory_sha256": str(
            native["follower_memory_sha256"]
        ),
        "candidate_physical_state_sha256": str(
            candidate["physical_state_sha256"]
        ),
        "native_physical_state_sha256": str(native["physical_state_sha256"]),
    }


def _checkpoint_from_probe(record: dict) -> dict:
    return {
        "ttt": float(record["step_ttt"]),
        "terminal_inventory": float(record["terminal_inventory"]),
        "follower_memory_sha256": str(record["post_follower_sha256"]),
        "physical_state_sha256": str(record["post_physical_sha256"]),
        "validity_gate_pass": bool(record["validity_gate_pass"]),
    }


def _dedupe_for_selector(records: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str, str], dict] = {}
    for record in records:
        key = (
            str(record["generator"]),
            str(record["response_memory_outcome_sha256"]),
            str(record["post_physical_sha256"]),
        )
        existing = unique.get(key)
        if existing is None or (
            str(record["residual_sha256"]), str(record["candidate_id"])
        ) < (
            str(existing["residual_sha256"]), str(existing["candidate_id"])
        ):
            unique[key] = record
    return list(unique.values())


def _realized_outcome_key(record: dict) -> tuple[str, str]:
    return (
        str(record["response_memory_outcome_sha256"]),
        str(record["post_physical_sha256"]),
    )


def _dedupe_h3_representatives(records: list[dict]) -> list[dict]:
    representatives: dict[tuple[str, str], dict] = {}
    for record in records:
        key = _realized_outcome_key(record)
        priority = (
            bool(record.get("uses_pcent_target", False)),
            str(record["candidate_id"]),
        )
        existing = representatives.get(key)
        if existing is None:
            representatives[key] = record
            continue
        existing_priority = (
            bool(existing.get("uses_pcent_target", False)),
            str(existing["candidate_id"]),
        )
        if priority < existing_priority:
            representatives[key] = record
    return list(representatives.values())


def _h1_replay_evidence(record: dict, rollout: dict) -> dict:
    checkpoint = rollout["checkpoints"]["1"]
    response_error = float(np.max(np.abs(
        np.asarray(rollout["first_step_response"], dtype=float)
        - np.asarray(record["response"], dtype=float)
    )))
    memory_exact = (
        rollout["checkpoints"]["1"]["follower_memory_sha256"]
        == record["post_follower_sha256"]
    )
    physical_exact = (
        checkpoint["physical_state_sha256"] == record["post_physical_sha256"]
    )
    step_ttt_exact = float(checkpoint["ttt"]) == float(record["step_ttt"])
    inventory_exact = float(checkpoint["terminal_inventory"]) == float(
        record["terminal_inventory"]
    )
    validity_exact = bool(checkpoint["validity_gate_pass"]) == bool(
        record["validity_gate_pass"]
    )
    evidence = {
        "probe_response": list(map(float, record["response"])),
        "rollout_response": list(map(float, rollout["first_step_response"])),
        "response_linf": response_error,
        "response_tolerance": 1.0e-6,
        "response_exact": response_error <= 1.0e-6,
        "probe_follower_memory_sha256": str(record["post_follower_sha256"]),
        "rollout_follower_memory_sha256": str(
            rollout["checkpoints"]["1"]["follower_memory_sha256"]
        ),
        "follower_memory_exact": memory_exact,
        "probe_physical_sha256": str(record["post_physical_sha256"]),
        "rollout_physical_sha256": str(
            rollout["checkpoints"]["1"]["physical_state_sha256"]
        ),
        "physical_exact": physical_exact,
        "probe_step_ttt": float(record["step_ttt"]),
        "rollout_step_ttt": float(checkpoint["ttt"]),
        "step_ttt_exact": step_ttt_exact,
        "probe_terminal_inventory": float(record["terminal_inventory"]),
        "rollout_terminal_inventory": float(checkpoint["terminal_inventory"]),
        "terminal_inventory_exact": inventory_exact,
        "probe_validity_gate_pass": bool(record["validity_gate_pass"]),
        "rollout_validity_gate_pass": bool(checkpoint["validity_gate_pass"]),
        "validity_gate_exact": validity_exact,
        "passed": (
            response_error <= 1.0e-6
            and memory_exact
            and physical_exact
            and step_ttt_exact
            and inventory_exact
            and validity_exact
        ),
    }
    return evidence


def _h3_selection_replay_evidence(selector_rollout: dict, rollout: dict) -> dict:
    selector = selector_rollout["checkpoints"]["3"]
    replay = rollout["checkpoints"]["3"]
    evidence = {
        "selector_ttt": float(selector["ttt"]),
        "rollout_ttt": float(replay["ttt"]),
        "ttt_exact": float(selector["ttt"]) == float(replay["ttt"]),
        "selector_terminal_inventory": float(selector["terminal_inventory"]),
        "rollout_terminal_inventory": float(replay["terminal_inventory"]),
        "terminal_inventory_exact": (
            float(selector["terminal_inventory"])
            == float(replay["terminal_inventory"])
        ),
        "selector_follower_memory_sha256": str(selector["follower_memory_sha256"]),
        "rollout_follower_memory_sha256": str(replay["follower_memory_sha256"]),
        "follower_memory_exact": (
            selector["follower_memory_sha256"] == replay["follower_memory_sha256"]
        ),
        "selector_physical_sha256": str(selector["physical_state_sha256"]),
        "rollout_physical_sha256": str(replay["physical_state_sha256"]),
        "physical_exact": (
            selector["physical_state_sha256"] == replay["physical_state_sha256"]
        ),
        "selector_validity_gate_pass": bool(selector["validity_gate_pass"]),
        "rollout_validity_gate_pass": bool(replay["validity_gate_pass"]),
        "validity_gate_exact": (
            bool(selector["validity_gate_pass"])
            == bool(replay["validity_gate_pass"])
        ),
    }
    evidence["passed"] = all(
        evidence[field]
        for field in (
            "ttt_exact",
            "terminal_inventory_exact",
            "follower_memory_exact",
            "physical_exact",
            "validity_gate_exact",
        )
    )
    return evidence


def evaluate_ablation_event(
    source: RLLeaderEnv,
    anchor_context,
    *,
    scenario: str,
    policy_step: int,
    cfg: AblationConfig,
    run_h12: bool,
    candidate_mode: str = "v1",
) -> dict:
    started = time.monotonic()
    cfg.validate()
    branch = str(anchor_context.coordination.selected_branch)
    if branch not in cfg.allowed_anchor_branches:
        raise OracleGateError(
            "linear-price ablation requires a coarse/refined native anchor",
            {"policy_step": policy_step, "native_anchor_branch": branch},
        )
    source_before = _runtime_fingerprint(source)
    zero = _zero_round_trip(source, anchor_context)
    if not zero["coordination_zero_is_native_identity"]:
        raise OracleGateError(
            "linear-price ablation cannot mix an explicit branch switch",
            zero,
        )

    target_env = copy.deepcopy(source)
    forecast = list(copy.deepcopy(anchor_context.forecast))
    target_env._update_rl_far_gate(forecast)
    pcent = CentralizedMPC(target_env.cfg, mode="proposed").decide_with_info(
        target_env.sim.state.copy(), forecast, target_env.previous
    )
    target_response = source.response_vector(pcent.control)
    response_scales, response_families = _response_scales_and_families(source)

    physical_records: dict[str, dict] = {}
    names = tuple(source.action_schema.names)
    if candidate_mode == "known_positive_canary":
        all_specs = [generate_known_positive_canary(names)]
        generator_counts = validate_candidate_specs(
            all_specs, names, cfg, require_all_generators=False
        )
        jacobian_diagnostics = {
            "performed": False,
            "reason": "known-positive regression canary",
        }
    elif candidate_mode in ("owner_block_v2_urban", "owner_block_v2_all"):
        include_freeway = candidate_mode == "owner_block_v2_all"
        directed = _directed_groups(
            source, anchor_context.result.control, pcent.control
        )
        structured_specs = generate_owner_block_structured_candidates(
            names, include_freeway=include_freeway
        )
        guided_specs = generate_owner_block_pcent_candidates(
            names, directed, include_freeway=include_freeway
        )
        all_specs = structured_specs + guided_specs
        generator_counts = validate_candidate_specs(
            all_specs,
            names,
            cfg,
            require_all_generators=False,
            allowed_radii=(0.25, 0.5, 2.0 ** -1.5, 2.0 ** -0.5),
        )
        jacobian_diagnostics = {
            "performed": False,
            "reason": "sparse owner-block v2",
        }
    elif candidate_mode == "v1":
        directed = _directed_groups(
            source, anchor_context.result.control, pcent.control
        )
        pcent_specs = generate_pcent_sign_candidates(names, directed, cfg)
        structured_specs = generate_structured_candidates(names, cfg)
        random_specs = generate_orthogonal_random_candidates(
            names, cfg, pre_runtime_sha256=source_before
        )
        secant_specs = generate_jacobian_secant_candidates(names, cfg)
        initial_specs = pcent_specs + structured_specs + random_specs + secant_specs
        _probe_unique_specs(
            source,
            anchor_context,
            initial_specs,
            physical_records,
            target_response=target_response,
            response_scales=response_scales,
            response_families=response_families,
        )
        initial_logical = _logical_records(initial_specs, physical_records)
        by_candidate = {row["candidate_id"]: row for row in initial_logical}
        inverse_specs, jacobian_diagnostics = synthesize_jacobian_inverse_candidates(
            secant_specs,
            by_candidate,
            anchor_response=source.response_vector(anchor_context.result.control),
            target_response=target_response,
            response_scales=response_scales,
            cfg=cfg,
        )
        all_specs = (
            pcent_specs + structured_specs + secant_specs
            + inverse_specs + random_specs
        )
        generator_counts = validate_candidate_specs(all_specs, names, cfg)
    else:
        raise ValueError(f"unknown candidate mode: {candidate_mode}")
    _probe_unique_specs(
        source,
        anchor_context,
        all_specs,
        physical_records,
        target_response=target_response,
        response_scales=response_scales,
        response_families=response_families,
    )
    logical_records = _logical_records(all_specs, physical_records)
    order = _verify_probe_order(
        source,
        anchor_context,
        physical_records,
        all_specs,
        target_response=target_response,
        response_scales=response_scales,
        response_families=response_families,
    )
    source_after = _runtime_fingerprint(source)
    if source_after != source_before:
        raise OracleGateError(
            "ablation probes mutated the source runtime",
            {"before": source_before, "after": source_after},
        )

    rollout_steps = 12 if run_h12 else 1
    rollout_horizons = LABEL_HORIZONS if run_h12 else (1,)
    pstack = _rollout_pstack(
        source,
        anchor_context,
        rollout_steps=rollout_steps,
        horizons=rollout_horizons,
    )
    identity = _rollout_price_candidate(
        source,
        np.zeros(source.action_dim, dtype=np.float32),
        anchor_context,
        rollout_steps=rollout_steps,
        horizons=rollout_horizons,
    )
    identity_parity = _long_zero_parity(identity, pstack, rollout_horizons)
    _require_rollout_coverage(pstack, rollout_horizons, label="ablation P-Stack")
    anchor_h1 = pstack["checkpoints"]["1"]

    selector_records = _dedupe_for_selector(logical_records)
    h3_rollout_by_outcome: dict[tuple[str, str], dict] = {}
    owner_block_mode = candidate_mode.startswith("owner_block_v2_")
    if run_h12 and owner_block_mode:
        h3_selector_records = []
        for record in selector_records:
            outcome_key = _realized_outcome_key(record)
            if outcome_key not in h3_rollout_by_outcome:
                rollout = _rollout_price_candidate(
                    source,
                    np.asarray(record["residual"], dtype=np.float32),
                    anchor_context,
                    rollout_steps=3,
                    horizons=(1, 3),
                )
                _require_rollout_coverage(
                    rollout, (1, 3), label=f"owner-block H3 {record['candidate_id']}"
                )
                replay = _h1_replay_evidence(record, rollout)
                if not replay["passed"]:
                    raise OracleGateError(
                        "owner-block H3 rollout did not replay its H1 probe",
                        {"candidate_id": record["candidate_id"], **replay},
                    )
                h3_rollout_by_outcome[outcome_key] = rollout
            h3_record = copy.deepcopy(record)
            checkpoint = h3_rollout_by_outcome[outcome_key]["checkpoints"]["3"]
            h3_record["h3_ttt"] = float(checkpoint["ttt"])
            h3_record["h3_terminal_inventory"] = float(
                checkpoint["terminal_inventory"]
            )
            h3_record["validity_gate_pass"] = bool(
                checkpoint["validity_gate_pass"]
            )
            h3_selector_records.append(h3_record)
        representatives = _dedupe_h3_representatives(h3_selector_records)
        selected = [
            record for record in representatives
            if bool(record["validity_gate_pass"])
        ]
    elif not run_h12:
        selected = []
    elif candidate_mode == "known_positive_canary":
        selected = selector_records
    else:
        selected = select_h12_candidates_without_teacher(
            selector_records,
            anchor_response=np.asarray(pstack["first_step_response"], dtype=float),
            anchor_follower_memory_sha256=str(
                anchor_h1["follower_memory_sha256"]
            ),
            cfg=cfg,
        )
    selected_ids = {row["candidate_id"] for row in selected}
    rollout_by_residual: dict[str, dict] = {}
    replay_by_residual: dict[str, dict] = {}
    h3_selection_replay_by_candidate: dict[str, dict] = {}
    for record in selected:
        digest = str(record["residual_sha256"])
        if digest in rollout_by_residual:
            continue
        residual = np.asarray(record["residual"], dtype=np.float32)
        rollout = _rollout_price_candidate(
            source,
            residual,
            anchor_context,
            rollout_steps=12,
            horizons=LABEL_HORIZONS,
        )
        _require_rollout_coverage(
            rollout, LABEL_HORIZONS, label=f"ablation candidate {record['candidate_id']}"
        )
        replay_evidence = _h1_replay_evidence(record, rollout)
        if not replay_evidence["passed"]:
            raise OracleGateError(
                "H12 rollout did not replay its H1 probe",
                {
                    "candidate_id": record["candidate_id"],
                    **replay_evidence,
                },
            )
        if owner_block_mode:
            h3_evidence = _h3_selection_replay_evidence(
                h3_rollout_by_outcome[_realized_outcome_key(record)], rollout
            )
            if not h3_evidence["passed"]:
                raise OracleGateError(
                    "owner-block H12 rollout did not replay its H3 selection rollout",
                    {"candidate_id": record["candidate_id"], **h3_evidence},
                )
            h3_selection_replay_by_candidate[str(record["candidate_id"])] = h3_evidence
        rollout["paired"] = _paired_metrics(rollout, pstack)
        rollout_by_residual[digest] = rollout
        replay_by_residual[digest] = replay_evidence

    native_labels = {str(horizon): _invalid_horizon_label() for horizon in LABEL_HORIZONS}
    for horizon in rollout_horizons:
        checkpoint = pstack["checkpoints"][str(horizon)]
        native_labels[str(horizon)] = _horizon_label(
            checkpoint, checkpoint, horizon=horizon
        )
    aliases_by_outcome: dict[tuple[str, str], list[str]] = {}
    for record in logical_records:
        aliases_by_outcome.setdefault(_realized_outcome_key(record), []).append(
            str(record["candidate_id"])
        )
    rows = [{
        "candidate_id": "native_anchor",
        "execution_branch": "native_anchor",
        "candidate_family": "native_anchor",
        "candidate_label": "native_anchor",
        "candidate_aliases": ["native_anchor"],
        "continuous_residual_valid": False,
        "continuous_residual": None,
        "validity_gate_pass": bool(pstack["validity_gate_pass"]),
        "horizon_labels": native_labels,
        "ranker_eligible": bool(native_labels["12"]["label_valid"]),
        "oracle_is_winner": False,
    }]
    for record in logical_records:
        labels = {str(horizon): _invalid_horizon_label() for horizon in LABEL_HORIZONS}
        labels["1"] = _horizon_label(
            _checkpoint_from_probe(record), anchor_h1, horizon=1
        )
        h3_rollout = h3_rollout_by_outcome.get(_realized_outcome_key(record))
        if h3_rollout is not None:
            labels["3"] = _horizon_label(
                h3_rollout["checkpoints"]["3"],
                pstack["checkpoints"]["3"],
                horizon=3,
            )
        rollout = rollout_by_residual.get(str(record["residual_sha256"]))
        replay_evidence = None
        if rollout is not None and record["candidate_id"] in selected_ids:
            replay_evidence = copy.deepcopy(
                replay_by_residual[str(record["residual_sha256"])]
            )
            for horizon in LABEL_HORIZONS:
                labels[str(horizon)] = _horizon_label(
                    rollout["checkpoints"][str(horizon)],
                    pstack["checkpoints"][str(horizon)],
                    horizon=horizon,
                )
        rows.append({
            "candidate_id": str(record["candidate_id"]),
            "execution_branch": "coordination",
            "candidate_family": FAMILY_MAP[str(record["generator"])],
            "candidate_label": str(record["candidate_label"]),
            "candidate_aliases": sorted(
                aliases_by_outcome[_realized_outcome_key(record)]
            ),
            "continuous_residual_valid": True,
            "continuous_residual": list(map(float, record["residual"])),
            "validity_gate_pass": bool(record["validity_gate_pass"]),
            "horizon_labels": labels,
            "ranker_eligible": bool(labels["12"]["label_valid"]),
            "oracle_is_winner": False,
            "candidate_generator": str(record["generator"]),
            "generator_phase": str(record["phase"]),
            "uses_pcent_target": bool(record["uses_pcent_target"]),
            "teacher_signal_missing": bool(record["teacher_signal_missing"]),
            "generator_metadata": copy.deepcopy(record["generator_metadata"]),
            "candidate_response": list(map(float, record["response"])),
            "post_follower_sha256": str(record["post_follower_sha256"]),
            "post_physical_sha256": str(record["post_physical_sha256"]),
            "response_memory_outcome_sha256": str(
                record["response_memory_outcome_sha256"]
            ),
            "residual_sha256": str(record["residual_sha256"]),
            "residual_l2": float(record["residual_l2"]),
            "distance_to_pcent": copy.deepcopy(record["distance_to_pcent"]),
            "h1_step_ttt": float(record["step_ttt"]),
            "h1_terminal_inventory": float(record["terminal_inventory"]),
            "requested_budget": copy.deepcopy(record["requested_budget"]),
            "budget_intent_checks": copy.deepcopy(record["budget_intent_checks"]),
            "h1_replay_evidence": replay_evidence,
            "h3_selection_replay_evidence": copy.deepcopy(
                h3_selection_replay_by_candidate.get(str(record["candidate_id"]))
            ),
        })

    positive = [
        row for row in rows
        if bool(row["horizon_labels"]["12"].get("positive", False))
    ]
    if positive:
        winner = min(positive, key=lambda row: (
            float(row["horizon_labels"]["12"]["candidate_ttt"]),
            str(row["candidate_id"]),
        ))
    else:
        winner = rows[0]
    winner["oracle_is_winner"] = True

    outcome_counts = {
        generator: len({
            row["response_memory_outcome_sha256"]
            for row in logical_records if row["generator"] == generator
        })
        for generator in ABLATION_GENERATORS
    }
    return {
        "candidate_pool_id": (
            f"{scenario}:{policy_step}:{anchor_context.anchor_fingerprint[:12]}"
        ),
        "scenario": scenario,
        "simulation_step": int(source.step_idx),
        "policy_step": int(policy_step),
        "simulation_time_sec": float(source.sim.state.time_sec),
        "observation": source._observe().astype(float).tolist(),
        "anchor_envelope": source.action_schema.serialize_anchor(
            anchor_context.coordination
        ).astype(float).tolist(),
        "anchor_fingerprint": anchor_context.anchor_fingerprint,
        "anchor_context_contract": PSTACK_ANCHOR_CONTEXT_CONTRACT,
        "native_anchor_branch": branch,
        "pre_runtime_sha256": source_before,
        "forecast_sha256": _digest(anchor_context.forecast),
        "probe_order_verified": bool(order["passed"]),
        "source_isolation_verified": source_after == source_before,
        "identity_parity": identity_parity,
        "zero_round_trip": zero,
        "candidate_manifest_sha256": candidate_manifest_sha256(all_specs),
        "candidate_mode": candidate_mode,
        "owner_h12_policy": "exhaustive" if owner_block_mode else None,
        "candidate_generator_counts": generator_counts,
        "unique_physical_residual_count": len(physical_records),
        "unique_response_memory_counts": outcome_counts,
        "selected_h12_candidate_ids": sorted(selected_ids),
        "jacobian_diagnostics": jacobian_diagnostics,
        "pcent_provenance": {
            "objective": float(pcent.objective),
            "solver_evaluations": int(pcent.solver_evaluations),
            "converged": bool(pcent.converged),
            "control": _control_payload(pcent.control),
            "response": target_response.astype(float).tolist(),
        },
        "order_check": order,
        "rows": rows,
        "elapsed_sec": float(time.monotonic() - started),
    }


def run_scenario(
    scenario: str,
    event_steps: tuple[int, ...],
    *,
    cfg: AblationConfig,
    run_h12: bool,
    output_path: Path,
    candidate_mode: str = "v1",
    replay_cache_payloads: dict[int, Path] | None = None,
) -> dict:
    env = RLLeaderEnv(
        scenario_name=scenario,
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    teacher_contract, teacher_sha256 = _teacher_contract(env)
    result = {
        "format_version": ORACLE_LABEL_DATASET_FORMAT,
        "row_format": ORACLE_CANDIDATE_ROW_FORMAT,
        "ablation_contract": (
            OWNER_BLOCK_ABLATION_CONTRACT
            if candidate_mode.startswith("owner_block_v2_")
            else ABLATION_CONTRACT
        ),
        "label_horizons": list(LABEL_HORIZONS),
        "decision_horizon": 12,
        "inventory_guard_tolerance_veh": INVENTORY_GUARD_TOLERANCE_VEH,
        "action_schema": env.action_schema.metadata(),
        "observation_schema": env.observation_schema.metadata(),
        "ranker_feature_fields": list(DEPLOYMENT_RANKER_FEATURE_FIELDS),
        "legacy_migration_allowed": False,
        "implementation_sha256": _implementation_fingerprints(),
        "experiment_contract_sha256": env.experiment_contract_fingerprint,
        "teacher_contract": teacher_contract,
        "teacher_contract_sha256": teacher_sha256,
        "candidate_budget": {
            "radius_l2": float(cfg.radius),
            "radius_l2_levels": (
                [0.25, 2.0 ** -1.5, 0.5, 2.0 ** -0.5]
                if candidate_mode.startswith("owner_block_v2_") else [float(cfg.radius)]
            ),
            "h1_candidates_per_generator": (
                None if candidate_mode.startswith("owner_block_v2_")
                else cfg.h1_candidates_per_generator
            ),
            "h12_slots_per_generator": (
                None if candidate_mode.startswith("owner_block_v2_")
                else cfg.h12_slots_per_generator
            ),
            "generators": (
                ["structured"]
                if candidate_mode == "known_positive_canary"
                else ["structured", "pcent_sign"]
                if candidate_mode.startswith("owner_block_v2_")
                else list(ABLATION_GENERATORS)
            ),
            "subspace": "30-D linear prices; budget/quadratic/certificate fixed",
            "candidate_mode": candidate_mode,
            "templates": (
                ["signed_axis", "signed_corner"]
                if candidate_mode.startswith("owner_block_v2_") else None
            ),
        },
        "pcent_feature_policy": (
            "P-CENT is privileged provenance and is excluded from deployment ranker inputs"
        ),
        "h12_selection_policy": (
            "all unique H1 realized outcomes with cumulative H3 validity"
            if candidate_mode.startswith("owner_block_v2_")
            else "known canary forced"
            if candidate_mode == "known_positive_canary"
            else "one H1 TTT-best candidate per generator"
        ),
        "candidate_pools": [],
        "passed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    event_set = set(event_steps)
    max_event = max(event_steps)
    try:
        while int(env.step_idx - env.warmup) <= max_event:
            step = int(env.step_idx - env.warmup)
            if step in event_set:
                context = env.prepare_pstack_anchor_context()
                replay_cache_path = (
                    None if replay_cache_payloads is None
                    else replay_cache_payloads.get(int(step))
                )
                if replay_cache_path is not None:
                    replay_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = replay_cache_path.with_suffix(".tmp.pkl")
                    temporary.write_bytes(pickle.dumps((env, context)))
                    temporary.replace(replay_cache_path)
                result["pending_event"] = {
                    "policy_step": step,
                    "simulation_step": int(env.step_idx),
                    "native_anchor_branch": context.coordination.selected_branch,
                    "anchor_fingerprint": context.anchor_fingerprint,
                }
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(
                    f"ablation-event scenario={scenario} step={step} "
                    f"branch={context.coordination.selected_branch}",
                    flush=True,
                )
                pool = evaluate_ablation_event(
                    env,
                    context,
                    scenario=scenario,
                    policy_step=step,
                    cfg=cfg,
                    run_h12=run_h12,
                    candidate_mode=candidate_mode,
                )
                result.pop("pending_event", None)
                result["candidate_pools"].append(pool)
                result["elapsed_sec"] = float(time.monotonic() - started)
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(
                    f"ablation scenario={scenario} step={step} "
                    f"physical={pool['unique_physical_residual_count']} "
                    f"h12={len(pool['selected_h12_candidate_ids'])}",
                    flush=True,
                )
                if step == max_event:
                    break
                env.step_prepared_optimizer_anchor(context)
            else:
                env.step_optimizer_anchor(sync_follower_state=True)
            print(
                f"source-PStack scenario={scenario} "
                f"reached_policy_step={env.step_idx - env.warmup}/{max_event}",
                flush=True,
            )
        validate_oracle_label_artifact(
            result, require_decision_horizon=run_h12
        )
        result["passed"] = True
    except Exception as exc:
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, OracleGateError):
            result["error_diagnostics"] = exc.diagnostics
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    result["elapsed_sec"] = float(time.monotonic() - started)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not steps or min(steps) < 0:
        raise ValueError("event steps must be nonnegative")
    return steps


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="sweet_170_w60")
    parser.add_argument("--events", default="9")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--master-seed", type=int, default=20260828)
    parser.add_argument("--run-h12", action="store_true")
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "v1",
            "known_positive_canary",
            "owner_block_v2_urban",
            "owner_block_v2_all",
        ),
        default="v1",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/rl_phase0_implementation_20260828/"
            "linear_price_candidate_ablation_v1"
        ),
    )
    args = parser.parse_args(argv)
    radius = (
        2.0 ** -0.5
        if args.candidate_mode != "v1"
        else args.radius
    )
    cfg = AblationConfig(radius=radius, master_seed=args.master_seed)
    output_path = Path(args.output_dir) / f"{args.scenario}.json"
    result = run_scenario(
        args.scenario,
        _parse_steps(args.events),
        cfg=cfg,
        run_h12=bool(args.run_h12),
        output_path=output_path,
        candidate_mode=args.candidate_mode,
    )
    print(json.dumps({
        "scenario": args.scenario,
        "passed": result["passed"],
        "candidate_pools": len(result["candidate_pools"]),
        "elapsed_sec": result["elapsed_sec"],
        "output": str(output_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
