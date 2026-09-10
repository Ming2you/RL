"""Finalize balanced H12 positives with a paired zero-demand drain-out."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_anchor_context_parity import _dynamic_follower_payload
from rl_leader.diagnose_balanced_remaining_horizon import (
    _recorded_path,
    exact_balanced_h12_replay,
)
from rl_leader.diagnose_candidate_ablation import _implementation_fingerprints
from rl_leader.diagnose_phase0_parity import _digest, _physical_snapshot
from rl_leader.evaluate_balanced_horizons import (
    BALANCED_HORIZON_FORMAT,
    _replay_event,
)
from rl_leader.oracle_label_contract import validate_oracle_label_artifact
from src.models.demand import DemandStep


BALANCED_DRAIN_OUT_FORMAT = "balanced_positive_zero_demand_drain_out_v1"
BALANCED_DRAIN_OUT_CONTRACT = "raw_global_ttt_paired_zero_demand_drain_out_v1"
OBJECTIVE = "raw_global_ttt_v1"
CUTOFF_TIME_SEC = 14400.0
INVENTORY_PARITY_ATOL_VEH = 1.0e-9


def _progress(message: str) -> None:
    print(f"[balanced-drain] {message}", flush=True)


class ZeroDemandAfterCutoff:
    """Delegate demand before cutoff and suppress only external inflow after it."""

    def __init__(self, profile: Any, cutoff_time_sec: float):
        self._profile = profile
        self.cutoff_time_sec = float(cutoff_time_sec)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._profile, name)

    def at(self, time_sec: float) -> DemandStep:
        delegated = self._profile.at(time_sec)
        if float(time_sec) < self.cutoff_time_sec:
            return delegated
        return DemandStep(
            freeway_mainline={key: 0.0 for key in delegated.freeway_mainline},
            urban_boundary={key: 0.0 for key in delegated.urban_boundary},
            ramp_arrival={key: 0.0 for key in delegated.ramp_arrival},
            incident_capacity_factor=float(delegated.incident_capacity_factor),
            freeway_lane_loss=copy.deepcopy(delegated.freeway_lane_loss),
        )

    def horizon(self, start_time_sec: float, steps: int) -> list[DemandStep]:
        dt = float(self._profile.cfg.simulation.control_interval)
        return [
            self.at(float(start_time_sec) + index * dt)
            for index in range(max(1, int(steps)))
        ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_inventory(state: Any, net: Any) -> dict[str, float]:
    """Expose components whose sum exactly matches RLLeaderEnv._state_inventory."""
    freeway_buffers = 0.0
    for density_map in (
        state.freeway_buffer_up_density,
        state.freeway_buffer_down_density,
    ):
        freeway_buffers += sum(
            max(0.0, float(density))
            * float(net.freeway_segment_length_km)
            * float(net.freeway_lanes)
            for values in density_map.values()
            for density in values
        )
    components = {
        "urban": float(state.total_urban_vehicles(net)),
        "freeway": float(state.total_freeway_vehicles(net)),
        "off_ramp_storage": float(state.off_ramp_storage_occupancy_veh(net)),
        "freeway_buffers": float(freeway_buffers),
    }
    components["total"] = float(sum(components.values()))
    return components


def update_clearance_stability(
    inventory: float,
    *,
    threshold: float,
    stable_count: int,
    required_stable_steps: int,
) -> tuple[int, bool]:
    count = int(stable_count) + 1 if float(inventory) <= float(threshold) else 0
    return count, count >= int(required_stable_steps)


def drain_out_verdict(candidate: dict, native: dict) -> dict:
    candidate_main_arrivals = candidate["main_external_arrivals_sequence_veh"]
    native_main_arrivals = native["main_external_arrivals_sequence_veh"]
    checks = {
        "candidate_drain_complete": bool(candidate["drain_complete"]),
        "native_drain_complete": bool(native["drain_complete"]),
        "candidate_all_steps_valid": bool(candidate["validity_all_steps"]),
        "native_all_steps_valid": bool(native["validity_all_steps"]),
        "candidate_cutoff_contract_exact": bool(candidate["cutoff_check"]["passed"]),
        "native_cutoff_contract_exact": bool(native["cutoff_check"]["passed"]),
        "candidate_main_boundary_inventory_parity_exact": bool(
            candidate["inventory_parity"]["main_boundary"]["exact"]
        ),
        "native_main_boundary_inventory_parity_exact": bool(
            native["inventory_parity"]["main_boundary"]["exact"]
        ),
        "candidate_terminal_inventory_parity_exact": bool(
            candidate["inventory_parity"]["terminal"]["exact"]
        ),
        "native_terminal_inventory_parity_exact": bool(
            native["inventory_parity"]["terminal"]["exact"]
        ),
        "candidate_tail_external_arrivals_zero": (
            float(candidate["tail_external_arrivals_veh"]) == 0.0
        ),
        "native_tail_external_arrivals_zero": (
            float(native["tail_external_arrivals_veh"]) == 0.0
        ),
        "candidate_max_tail_step_external_arrivals_zero": (
            float(candidate["max_abs_tail_step_external_arrivals_veh"]) == 0.0
        ),
        "native_max_tail_step_external_arrivals_zero": (
            float(native["max_abs_tail_step_external_arrivals_veh"]) == 0.0
        ),
        "paired_main_external_arrival_sequence_exact": (
            candidate_main_arrivals == native_main_arrivals
        ),
        "paired_main_external_arrival_hash_exact": (
            candidate["main_external_arrivals_sha256"]
            == native["main_external_arrivals_sha256"]
        ),
        "candidate_conservation_bounded": (
            float(candidate["max_abs_conservation_residual_veh"]) <= 1.0e-3
        ),
        "native_conservation_bounded": (
            float(native["max_abs_conservation_residual_veh"]) <= 1.0e-3
        ),
        "candidate_no_projection": (
            abs(float(candidate["cumulative_projection_veh"])) <= 1.0e-9
        ),
        "native_no_projection": (
            abs(float(native["cumulative_projection_veh"])) <= 1.0e-9
        ),
        "candidate_no_rejected_flow": (
            abs(float(candidate["cumulative_rejected_flow_veh"])) <= 1.0e-9
        ),
        "native_no_rejected_flow": (
            abs(float(native["cumulative_rejected_flow_veh"])) <= 1.0e-9
        ),
        "candidate_no_overflow": (
            abs(float(candidate["cumulative_overflow_count"])) <= 1.0e-9
        ),
        "native_no_overflow": (
            abs(float(native["cumulative_overflow_count"])) <= 1.0e-9
        ),
    }
    ttt_gain = float(native["total_ttt"]) - float(candidate["total_ttt"])
    required_gain = max(0.1, 0.001 * float(native["total_ttt"]))
    drained = checks["candidate_drain_complete"] and checks["native_drain_complete"]
    quality = all(value for key, value in checks.items() if "drain_complete" not in key)
    if not drained:
        status = "quarantine"
    elif not quality:
        status = "invalid"
    elif ttt_gain > required_gain:
        status = "positive"
    else:
        status = "negative"
    return {
        "status": status,
        "final_positive": status == "positive",
        "final_negative": status == "negative",
        "invalid": status in {"invalid", "quarantine"},
        "quarantine": status == "quarantine",
        "ttt_gain": float(ttt_gain),
        "required_gain": float(required_gain),
        "checks": checks,
    }


def _terminal_follower_hash(env: Any) -> str:
    return _digest(_dynamic_follower_payload(env.controller.nash_solver))


def _terminal_physical_hash(env: Any) -> str:
    return _digest(_physical_snapshot(env))


def _new_metrics(tail_cap_steps: int) -> dict:
    return {
        "main_ttt": 0.0,
        "tail_ttt": 0.0,
        "total_ttt": 0.0,
        "main_steps": 0,
        "tail_steps": 0,
        "tail_cap_steps": int(tail_cap_steps),
        "main_boundary_inventory": None,
        "terminal_inventory": None,
        "tail_clearance_time_sec": None,
        "drain_complete": False,
        "validity_all_steps": True,
        "cumulative_external_arrivals_veh": 0.0,
        "main_external_arrivals_veh": 0.0,
        "tail_external_arrivals_veh": 0.0,
        "max_abs_tail_step_external_arrivals_veh": 0.0,
        "main_external_arrivals_sequence_veh": [],
        "tail_external_arrivals_sequence_veh": [],
        "external_arrivals_sequence_veh": [],
        "cumulative_completed_departures_veh": 0.0,
        "max_abs_conservation_residual_veh": 0.0,
        "cumulative_projection_veh": 0.0,
        "cumulative_rejected_flow_veh": 0.0,
        "cumulative_overflow_count": 0.0,
        "checkpoints": {},
    }


def _record_step(metrics: dict, info: dict, step_ttt: float, *, main: bool) -> None:
    phase = "main" if main else "tail"
    metrics[f"{phase}_ttt"] += float(step_ttt)
    metrics[f"{phase}_steps"] += 1
    metrics["total_ttt"] += float(step_ttt)
    metrics["validity_all_steps"] = bool(
        metrics["validity_all_steps"] and bool(info["validity_gate_pass"])
    )
    external_arrivals = float(info["external_arrivals_veh"])
    metrics["cumulative_external_arrivals_veh"] += external_arrivals
    metrics[f"{phase}_external_arrivals_veh"] += external_arrivals
    metrics[f"{phase}_external_arrivals_sequence_veh"].append(external_arrivals)
    metrics["external_arrivals_sequence_veh"].append(external_arrivals)
    if not main:
        metrics["max_abs_tail_step_external_arrivals_veh"] = max(
            float(metrics["max_abs_tail_step_external_arrivals_veh"]),
            abs(external_arrivals),
        )
    metrics["cumulative_completed_departures_veh"] += float(
        info["completed_departures_veh"]
    )
    metrics["max_abs_conservation_residual_veh"] = max(
        float(metrics["max_abs_conservation_residual_veh"]),
        abs(float(info["conservation_residual_veh"])),
    )
    metrics["cumulative_projection_veh"] += float(
        info["movement_queue_projection_veh"]
    )
    metrics["cumulative_rejected_flow_veh"] += float(
        info["coupling_offramp_arrivals_rejected_veh"]
    )
    metrics["cumulative_overflow_count"] += float(info["queue_overflow_count"])


def _record_h12_checkpoint(metrics: dict, env: Any) -> None:
    completed = int(metrics["main_steps"] + metrics["tail_steps"])
    if completed != 12:
        return
    metrics["checkpoints"]["12"] = {
        "ttt": float(metrics["total_ttt"]),
        "terminal_inventory": float(env._inventory()),
        "follower_memory_sha256": _terminal_follower_hash(env),
        "physical_state_sha256": _terminal_physical_hash(env),
        "validity_gate_pass": bool(metrics["validity_all_steps"]),
    }


def _cutoff_check(env: Any, original_n_steps: int) -> dict:
    check = {
        "required_cutoff_time_sec": CUTOFF_TIME_SEC,
        "env_T_total_sec": float(env.T_total),
        "config_T_total_sec": float(env.cfg.simulation.T_total),
        "n_steps_times_dt_sec": float(original_n_steps * env.dt),
    }
    check["passed"] = all(
        value == CUTOFF_TIME_SEC
        for key, value in check.items()
        if key != "passed"
    )
    if not check["passed"]:
        raise RuntimeError(f"v1 drain-out cutoff contract mismatch: {check}")
    return check


def _inventory_parity(env: Any, components: dict[str, float], label: str) -> dict:
    env_total = float(env._inventory())
    component_total = float(components["total"])
    absolute_error = abs(component_total - env_total)
    evidence = {
        "component_total": component_total,
        "environment_total": env_total,
        "absolute_error_veh": absolute_error,
        "tolerance_veh": INVENTORY_PARITY_ATOL_VEH,
        "exact": math.isclose(
            component_total,
            env_total,
            rel_tol=0.0,
            abs_tol=INVENTORY_PARITY_ATOL_VEH,
        ),
    }
    if not evidence["exact"]:
        raise RuntimeError(f"{label} component inventory parity failure: {evidence}")
    return evidence


def _finalize_arrival_diagnostics(metrics: dict) -> None:
    metrics["main_external_arrivals_sha256"] = _digest(
        metrics["main_external_arrivals_sequence_veh"]
    )
    metrics["tail_external_arrivals_sha256"] = _digest(
        metrics["tail_external_arrivals_sequence_veh"]
    )
    metrics["external_arrivals_sha256"] = _digest(
        metrics["external_arrivals_sequence_veh"]
    )


def _rollout_drain_out(
    source_env: Any,
    anchor_context: Any,
    *,
    residual: np.ndarray | None,
    tail_cap_steps: int,
    inventory_threshold: float,
    stable_steps: int,
) -> dict:
    env, context = copy.deepcopy((source_env, anchor_context))
    original_n_steps = int(env.n_steps)
    initial_step_idx = int(env.step_idx)
    cutoff_check = _cutoff_check(env, original_n_steps)
    cutoff_time_sec = CUTOFF_TIME_SEC
    env.profile = ZeroDemandAfterCutoff(env.profile, cutoff_time_sec)
    metrics = _new_metrics(tail_cap_steps)
    metrics["cutoff_check"] = cutoff_check
    metrics["first_action"] = "native_pstack" if residual is None else "candidate"
    metrics["n_steps_before_first_action"] = int(env.n_steps)

    if residual is None:
        _, reward, _, info, _ = env.step_prepared_optimizer_anchor(
            context, sync_follower_state=True
        )
    else:
        _, reward, _, info = env.step_anchored_candidate(residual, context)
    _record_step(metrics, info, -float(reward), main=initial_step_idx < original_n_steps)
    _record_h12_checkpoint(metrics, env)

    metrics["n_steps_after_first_action_before_extension"] = int(env.n_steps)
    env.n_steps = original_n_steps + int(tail_cap_steps)
    metrics["extended_n_steps"] = int(env.n_steps)
    metrics["n_steps_extension_applied_after_first_action"] = True
    metrics["pstack_steps_after_first_action"] = 0
    stable_count = 0
    while int(env.step_idx) < original_n_steps:
        _, reward, _, info, _ = env.step_optimizer_anchor(sync_follower_state=True)
        metrics["pstack_steps_after_first_action"] += 1
        _record_step(metrics, info, -float(reward), main=True)
        _record_h12_checkpoint(metrics, env)
    main_boundary_inventory = component_inventory(env.sim.state, env.net)
    metrics["main_boundary_inventory"] = main_boundary_inventory
    metrics["inventory_parity"] = {
        "main_boundary": _inventory_parity(
            env, main_boundary_inventory, "main-boundary"
        )
    }

    while int(env.step_idx) < int(env.n_steps):
        _, reward, _, info, _ = env.step_optimizer_anchor(sync_follower_state=True)
        metrics["pstack_steps_after_first_action"] += 1
        _record_step(metrics, info, -float(reward), main=False)
        _record_h12_checkpoint(metrics, env)
        stable_count, complete = update_clearance_stability(
            env._inventory(),
            threshold=inventory_threshold,
            stable_count=stable_count,
            required_stable_steps=stable_steps,
        )
        if complete:
            metrics["drain_complete"] = True
            metrics["tail_clearance_time_sec"] = float(metrics["tail_steps"] * env.dt)
            break

    if "12" not in metrics["checkpoints"]:
        raise RuntimeError("drain-out rollout did not reach H12")
    terminal_inventory = component_inventory(env.sim.state, env.net)
    metrics["terminal_inventory"] = terminal_inventory
    metrics["inventory_parity"]["terminal"] = _inventory_parity(
        env, terminal_inventory, "terminal"
    )
    _finalize_arrival_diagnostics(metrics)
    metrics["terminal_follower_memory_sha256"] = _terminal_follower_hash(env)
    metrics["terminal_physical_state_sha256"] = _terminal_physical_hash(env)
    metrics["final_simulation_time_sec"] = float(env.sim.state.time_sec)
    metrics["cutoff_time_sec"] = cutoff_time_sec
    metrics["initial_policy_step"] = initial_step_idx
    metrics["original_n_steps"] = original_n_steps
    metrics["stable_steps_observed"] = int(stable_count)
    return metrics


def select_drain_candidates(
    source: dict,
    candidate_ids: tuple[str, ...] | None = None,
) -> list[dict]:
    rows = [
        row for row in source.get("outcomes", [])
        if row.get("h12_label") is not None
        and row["h12_label"].get("validity_gate_pass", True) is True
    ]
    if candidate_ids is None:
        selected = [row for row in rows if row["h12_label"].get("positive") is True]
        if not selected:
            raise ValueError("source balanced artifact contains no positive H12 outcome")
        return selected

    requested = tuple(map(str, candidate_ids))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("explicit drain candidate IDs must be a nonempty unique set")
    by_id = {str(row["candidate_id"]): row for row in rows}
    missing = [candidate_id for candidate_id in requested if candidate_id not in by_id]
    if missing:
        raise ValueError(f"explicit drain candidates are absent or invalid: {missing}")
    return [by_id[candidate_id] for candidate_id in requested]


def _validate_source(
    source_path: Path,
    candidate_ids: tuple[str, ...] | None = None,
) -> tuple[dict, dict, dict, dict, dict, list[dict]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format_version") != BALANCED_HORIZON_FORMAT:
        raise ValueError("source is not a balanced H3/H12 artifact")
    if source.get("passed") is not True:
        raise ValueError("source balanced artifact did not pass")
    current_implementation = _implementation_fingerprints()
    if source.get("implementation_sha256") != current_implementation:
        raise ValueError("source balanced artifact implementation drift")

    candidate_domain = source.get("candidate_domain")
    source_contract = source.get("contract")
    if source_contract == "validated_oracle_h12_to_balanced_v1":
        adapter_sidecar = Path(__file__).with_name(
            "adapt_oracle_h12_to_balanced.py"
        )
        if source.get("adapter_sha256") != _sha256_file(adapter_sidecar):
            raise ValueError("adapted source sidecar drift")
        if candidate_domain not in ("urban", "freeway"):
            raise ValueError("adapted source candidate domain mismatch")
    elif source_contract == "budgeted_tail_h12_v1":
        budgeted_sidecar = Path(__file__).with_name(
            "evaluate_budgeted_tail_horizons.py"
        )
        if source.get("sidecar_sha256") != _sha256_file(budgeted_sidecar):
            raise ValueError("budgeted source sidecar drift")
        if candidate_domain != "budgeted_tail":
            raise ValueError("budgeted source candidate domain mismatch")
    elif source_contract is not None:
        raise ValueError(f"unknown balanced source contract: {source_contract}")
    elif candidate_domain == "urban+freeway":
        from rl_leader.evaluate_balanced_joint_horizons import (
            _assert_matching_sources,
            _load_balanced_source,
        )

        joint_sidecar = Path(__file__).with_name(
            "evaluate_balanced_joint_horizons.py"
        )
        if source.get("sidecar_sha256") != _sha256_file(joint_sidecar):
            raise ValueError("joint source sidecar drift")
        component_sources = {}
        for domain in ("urban", "freeway"):
            path_field = f"source_{domain}_artifact"
            sha_field = f"source_{domain}_sha256"
            component_path = _recorded_path(source[path_field])
            if _sha256_file(component_path) != source.get(sha_field):
                raise ValueError(f"joint {domain} source SHA mismatch")
            component_sources[domain] = _load_balanced_source(
                component_path, domain
            )
        _assert_matching_sources(
            component_sources["urban"], component_sources["freeway"]
        )
        for field in ("scenario", "stratum", "policy_step", "anchor_fingerprint"):
            if any(
                component_sources[domain].get(field) != source.get(field)
                for domain in ("urban", "freeway")
            ):
                raise ValueError(f"joint source {field} mismatch")
    elif candidate_domain == "freeway":
        freeway_sidecar = Path(__file__).with_name(
            "evaluate_balanced_freeway_horizons.py"
        )
        if source.get("sidecar_sha256") != _sha256_file(freeway_sidecar):
            raise ValueError("freeway source sidecar drift")
    elif candidate_domain is None:
        urban_sidecar = Path(__file__).with_name(
            "evaluate_balanced_horizons.py"
        )
        if source.get("sidecar_sha256") != _sha256_file(urban_sidecar):
            raise ValueError("urban source sidecar drift")
    else:
        raise ValueError(f"unknown balanced candidate domain: {candidate_domain}")

    h1_path = _recorded_path(source["source_h1_artifact"])
    manifest_path = _recorded_path(source["source_manifest"])
    if _sha256_file(h1_path) != source["source_h1_sha256"]:
        raise ValueError("source H1 artifact SHA mismatch")
    if _sha256_file(manifest_path) != source["source_manifest_sha256"]:
        raise ValueError("source manifest SHA mismatch")
    h1 = json.loads(h1_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(h1, require_decision_horizon=False)
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    if len(h1["candidate_pools"]) != 1:
        raise ValueError("balanced drain-out gate requires one H1 pool")
    pool = h1["candidate_pools"][0]
    if (
        pool["scenario"] != source["scenario"]
        or int(pool["policy_step"]) != int(source["policy_step"])
        or pool["anchor_fingerprint"] != source["anchor_fingerprint"]
    ):
        raise ValueError("balanced drain-out sidecar does not match its H1 source pool")

    selected = select_drain_candidates(source, candidate_ids)
    frozen = next(
        row for row in manifest["scenarios"] if row["scenario"] == source["scenario"]
    )
    event = next(
        row
        for row in frozen["events"]
        if int(row["policy_step"]) == int(source["policy_step"])
    )
    if event["stratum"] != source["stratum"] or not event["coordination_eligible"]:
        raise ValueError("balanced drain-out frozen event mismatch")
    return source, h1, manifest, frozen, event, selected


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_balanced_drain_out(
    source_path: Path,
    output_path: Path,
    *,
    tail_cap_steps: int = 80,
    inventory_threshold: float = 0.1,
    stable_steps: int = 2,
    candidate_ids: tuple[str, ...] | None = None,
    replayed_env_context: tuple | None = None,
) -> dict:
    result = {
        "format_version": BALANCED_DRAIN_OUT_FORMAT,
        "contract_version": BALANCED_DRAIN_OUT_CONTRACT,
        "objective": OBJECTIVE,
        "source_artifact": str(source_path),
        "source_artifact_sha256": (
            _sha256_file(source_path) if source_path.is_file() else None
        ),
        "parameters": {
            "tail_cap_steps": int(tail_cap_steps),
            "inventory_threshold_veh": float(inventory_threshold),
            "stable_steps": int(stable_steps),
            "cutoff_time_sec": CUTOFF_TIME_SEC,
            "external_demand_after_cutoff": "zero",
            "inventory_parity_tolerance_veh": INVENTORY_PARITY_ATOL_VEH,
            "candidate_selection": (
                "h12_positive_default"
                if candidate_ids is None
                else "explicit_h12_candidates"
            ),
            "requested_candidate_ids": (
                None if candidate_ids is None else list(map(str, candidate_ids))
            ),
        },
        "outcomes": [],
        "status": "initializing",
        "passed": False,
    }
    _write_result(output_path, result)
    started = time.monotonic()
    try:
        if int(tail_cap_steps) < 1:
            raise ValueError("tail_cap_steps must be positive")
        if float(inventory_threshold) < 0.0:
            raise ValueError("inventory_threshold must be nonnegative")
        if int(stable_steps) < 1:
            raise ValueError("stable_steps must be positive")
        source, _, _, frozen, event, selected = _validate_source(
            source_path, candidate_ids
        )
        current_implementation = _implementation_fingerprints()
        h1_path = _recorded_path(source["source_h1_artifact"])
        manifest_path = _recorded_path(source["source_manifest"])
        result.update({
            "source_h1_artifact": str(h1_path),
            "source_h1_sha256": source["source_h1_sha256"],
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_implementation_sha256": current_implementation,
            "implementation_sha256": {
                "rl_leader/diagnose_balanced_drain_out.py": _sha256_file(Path(__file__)),
                **current_implementation,
            },
            "scenario": source["scenario"],
            "stratum": source["stratum"],
            "policy_step": int(source["policy_step"]),
            "anchor_fingerprint": source["anchor_fingerprint"],
            "oracle_semantics": (
                "one balanced owner-block price intervention followed by native "
                "P-Stack through simulation end and paired zero-demand drain-out"
            ),
            "status": "running",
        })
        _write_result(output_path, result)

        if replayed_env_context is None:
            _progress(f"replay-start scenario={source['scenario']} step={source['policy_step']}")
            env, context = _replay_event(frozen, event)
        else:
            _progress(f"replay-reuse scenario={source['scenario']} step={source['policy_step']}")
            env, context = replayed_env_context
        anchor_exact = context.anchor_fingerprint == source["anchor_fingerprint"]
        if not anchor_exact:
            raise RuntimeError("replayed frozen event anchor fingerprint drift")
        result["replay_evidence"] = {
            "frozen_event_exact": True,
            "frozen_event_sha256": _digest(event),
            "anchor_fingerprint_exact": True,
            "anchor_fingerprint": context.anchor_fingerprint,
        }
        _write_result(output_path, result)
        _progress(f"native-drain-start scenario={source['scenario']} step={source['policy_step']}")
        native = _rollout_drain_out(
            env,
            context,
            residual=None,
            tail_cap_steps=tail_cap_steps,
            inventory_threshold=inventory_threshold,
            stable_steps=stable_steps,
        )
        result["native_rollout"] = native
        _write_result(output_path, result)

        for index, row in enumerate(selected, start=1):
            _progress(
                f"candidate-drain-start scenario={source['scenario']} "
                f"step={source['policy_step']} index={index}/{len(selected)}"
            )
            candidate = _rollout_drain_out(
                env,
                context,
                residual=np.asarray(row["continuous_residual"], dtype=np.float32),
                tail_cap_steps=tail_cap_steps,
                inventory_threshold=inventory_threshold,
                stable_steps=stable_steps,
            )
            replay = exact_balanced_h12_replay(candidate, native, row["h12_label"])
            if not replay["passed"]:
                raise RuntimeError(
                    f"drain-out rollout failed exact H12 replay: {row['candidate_id']}"
                )
            verdict = drain_out_verdict(candidate, native)
            result["outcomes"].append({
                "candidate_id": row["candidate_id"],
                "representative_candidate_id": row["representative_candidate_id"],
                "candidate_aliases": row["candidate_aliases"],
                "continuous_residual": row["continuous_residual"],
                "source_h12_label": row["h12_label"],
                "h12_replay": replay,
                "rollout": candidate,
                "verdict": verdict,
            })
            _write_result(output_path, result)
        _progress(
            f"candidate-drain-done scenario={source['scenario']} "
            f"step={source['policy_step']} outcomes={len(result['outcomes'])}"
        )

        result["elapsed_sec"] = float(time.monotonic() - started)
        result["status"] = "complete"
        result["passed"] = True
        _write_result(output_path, result)
        return result
    except Exception as exc:
        result["elapsed_sec"] = float(time.monotonic() - started)
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["passed"] = False
        _write_result(output_path, result)
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tail-cap-steps", type=int, default=80)
    parser.add_argument("--inventory-threshold", type=float, default=0.1)
    parser.add_argument("--stable-steps", type=int, default=2)
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    args = parser.parse_args(argv)
    result = run_balanced_drain_out(
        Path(args.source),
        Path(args.output),
        tail_cap_steps=args.tail_cap_steps,
        inventory_threshold=args.inventory_threshold,
        stable_steps=args.stable_steps,
        candidate_ids=(
            None if args.candidate_ids is None else tuple(args.candidate_ids)
        ),
    )
    print(json.dumps({
        "passed": result["passed"],
        "scenario": result["scenario"],
        "stratum": result["stratum"],
        "outcomes": len(result["outcomes"]),
        "statuses": [row["verdict"]["status"] for row in result["outcomes"]],
        "elapsed_sec": result["elapsed_sec"],
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
