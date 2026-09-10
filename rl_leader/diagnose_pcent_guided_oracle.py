"""Build same-state P-CENT-guided, follower-executable oracle labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.diagnose_anchor_context_parity import _dynamic_follower_payload
from rl_leader.diagnose_pcent_reachability import (
    _response_scales_and_families,
    response_distance,
    structured_residual_candidates,
)
from rl_leader.diagnose_phase0_parity import (
    _control_payload,
    _digest,
    _physical_snapshot,
    follower_behavior_payload,
)
from rl_leader.diagnose_reachable_candidate_attribution import (
    _paired_metrics,
    _rollout_price_candidate,
    _rollout_pstack,
    _validate_h1_probe_replay,
    residual_from_record,
)
from rl_leader.env import PSTACK_ANCHOR_CONTEXT_CONTRACT, RLLeaderEnv
from rl_leader.run_pcent_matched import _step_validity, _teacher_contract
from src.controllers.centralized_mpc import CentralizedMPC
from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    RL_RESPONSE_CONTRACT_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE_FORMAT_VERSION = "pcent_guided_follower_oracle_v2_branch_complete"
KNOWN_EXECUTION_BRANCHES = {"native_anchor", "coordination"}
KNOWN_ANCHOR_BRANCHES = {
    "coarse", "refined", "fallback_pfo", "supervisor_pfo", "other",
}
DEFAULT_EVENT_STEPS = {
    "sweet_170_w60": (9, 13, 25),
    "sweet_190_w60": (25, 30),
}
IMPLEMENTATION_SOURCES = (
    "rl_leader/diagnose_pcent_guided_oracle.py",
    "rl_leader/env.py",
    "src/controllers/centralized_mpc.py",
    "src/controllers/coordination.py",
    "src/controllers/rl_stackelberg.py",
    "src/controllers/stackelberg_wu_metered.py",
    "src/controllers/wu_faithful_follower.py",
    "src/simulation/coupling.py",
    "src/simulation/simulator.py",
)


class OracleGateError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def parse_event_steps(value: str, scenarios: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Parse ``scenario:step,step;scenario:step`` with canonical defaults."""
    requested = tuple(map(str, scenarios))
    if not value.strip():
        missing = sorted(set(requested) - set(DEFAULT_EVENT_STEPS))
        if missing:
            raise ValueError(f"no default event steps for scenarios: {missing}")
        return {name: DEFAULT_EVENT_STEPS[name] for name in requested}

    parsed: dict[str, tuple[int, ...]] = {}
    for group in value.split(";"):
        if not group.strip():
            continue
        if ":" not in group:
            raise ValueError(f"invalid event group: {group!r}")
        scenario, raw_steps = group.split(":", 1)
        scenario = scenario.strip()
        steps = tuple(sorted({
            int(item) for item in raw_steps.split(",") if item.strip()
        }))
        if not steps or min(steps) < 0:
            raise ValueError(f"invalid event steps for {scenario!r}: {raw_steps!r}")
        parsed[scenario] = steps
    missing = sorted(set(requested) - set(parsed))
    extra = sorted(set(parsed) - set(requested))
    if missing or extra:
        raise ValueError(f"event scenario mismatch: missing={missing}, extra={extra}")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprints() -> dict[str, str]:
    return {
        relative: _sha256_file(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _runtime_fingerprint(env: RLLeaderEnv) -> str:
    optimizer = getattr(env, "optimizer_controller", None)
    payload = {
        "context_state": env._anchor_context_state_fingerprint(),
        "physical": _physical_snapshot(env),
        "rl_follower": follower_behavior_payload(env.controller.nash_solver),
        "optimizer_follower": (
            None if optimizer is None
            else follower_behavior_payload(optimizer.nash_solver)
        ),
        "rl_far_stress": bool(getattr(env, "_rl_fargate_stress", False)),
        "optimizer_far_stress": bool(
            getattr(env, "_optimizer_fargate_stress", False)
        ),
    }
    return _digest(payload)


def _zero_round_trip(source: RLLeaderEnv, anchor_context) -> dict:
    native, native_context = copy.deepcopy((source, anchor_context))
    native_replay, native_replay_context = copy.deepcopy((source, anchor_context))
    candidate, candidate_context = copy.deepcopy((source, anchor_context))
    native_obs, native_reward, native_done, native_info, _ = (
        native.step_prepared_optimizer_anchor(native_context)
    )
    candidate_obs, candidate_reward, candidate_done, candidate_info = (
        candidate.step_anchored_candidate(
            np.zeros(candidate.action_dim, dtype=np.float32),
            candidate_context,
        )
    )
    replay_obs, replay_reward, replay_done, replay_info, _ = (
        native_replay.step_prepared_optimizer_anchor(native_replay_context)
    )
    native_dynamic = _dynamic_follower_payload(native.optimizer_controller.nash_solver)
    replay_dynamic = _dynamic_follower_payload(
        native_replay.optimizer_controller.nash_solver
    )
    candidate_dynamic = _dynamic_follower_payload(candidate.controller.nash_solver)
    coordination_identity = (
        anchor_context.coordination.selected_branch in {"coarse", "refined"}
    )
    identity = candidate if coordination_identity else native_replay
    identity_obs = candidate_obs if coordination_identity else replay_obs
    identity_reward = candidate_reward if coordination_identity else replay_reward
    identity_done = candidate_done if coordination_identity else replay_done
    identity_info = candidate_info if coordination_identity else replay_info
    identity_dynamic = candidate_dynamic if coordination_identity else replay_dynamic
    checks = {
        "control_exact": (
            _control_payload(native.previous) == _control_payload(identity.previous)
        ),
        "physical_state_exact": (
            _physical_snapshot(native) == _physical_snapshot(identity)
        ),
        "observation_exact": np.array_equal(native_obs, identity_obs),
        "reward_exact": native_reward == identity_reward,
        "done_exact": native_done == identity_done,
        "follower_post_memory_exact": (
            _digest(native_dynamic) == _digest(identity_dynamic)
        ),
        "validity_exact": (
            bool(native_info["validity_gate_pass"])
            == bool(identity_info["validity_gate_pass"])
        ),
    }
    coordination_zero_checks = {
        "control_exact": (
            _control_payload(native.previous) == _control_payload(candidate.previous)
        ),
        "physical_state_exact": (
            _physical_snapshot(native) == _physical_snapshot(candidate)
        ),
        "observation_exact": np.array_equal(native_obs, candidate_obs),
        "reward_exact": native_reward == candidate_reward,
        "follower_post_memory_exact": (
            _digest(native_dynamic) == _digest(candidate_dynamic)
        ),
    }
    result = {
        "anchor_selected_branch": anchor_context.coordination.selected_branch,
        "identity_execution_branch": (
            "coordination_zero" if coordination_identity else "native_anchor"
        ),
        "coordination_zero_is_native_identity": bool(coordination_identity),
        "checks": checks,
        "passed": all(checks.values()),
        "native_step_ttt": -float(native_reward),
        "candidate_step_ttt": -float(candidate_reward),
        "native_response": native.response_vector(native.previous).astype(float).tolist(),
        "candidate_response": candidate.response_vector(candidate.previous).astype(float).tolist(),
        "candidate_post_follower_sha256": _digest(candidate_dynamic),
        "observation_linf": float(np.max(np.abs(native_obs - candidate_obs))),
        "native_control": _control_payload(native.previous),
        "candidate_control": _control_payload(candidate.previous),
        "dynamic_follower_differing_fields": [
            name
            for name in sorted(set(native_dynamic) | set(candidate_dynamic))
            if _digest(native_dynamic.get(name)) != _digest(candidate_dynamic.get(name))
        ],
        "coordination_zero_comparison": {
            "checks": coordination_zero_checks,
            "matches_native": all(coordination_zero_checks.values()),
        },
    }
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise OracleGateError(
            f"zero-residual native parity failed: {failed}", result
        )
    return result


def _probe_candidate(
    source: RLLeaderEnv,
    anchor_context,
    label: str,
    residual: np.ndarray,
    target_response: np.ndarray,
    scales: np.ndarray,
    families: dict[str, list[int]],
) -> dict:
    trial, trial_context = copy.deepcopy((source, anchor_context))
    _, reward, _, info = trial.step_anchored_candidate(
        residual,
        trial_context,
    )
    response = trial.response_vector(trial.previous)
    post_memory_sha256 = _digest(
        _dynamic_follower_payload(trial.controller.nash_solver)
    )
    return {
        "representative_label": str(label),
        "aliases": [str(label)],
        "execution_branch": "coordination",
        "residual_nonzero": {
            name: float(value)
            for name, value in zip(source.action_schema.names, residual)
            if abs(float(value)) > 1.0e-12
        },
        "response": response.astype(float).tolist(),
        "distance_to_pcent": response_distance(
            response, target_response, scales, families
        ),
        "step_ttt": -float(reward),
        "terminal_inventory": float(trial._inventory()),
        "validity_gate_pass": bool(info["validity_gate_pass"]),
        "post_follower_sha256": post_memory_sha256,
        "post_physical_sha256": _digest(_physical_snapshot(trial)),
        "control": _control_payload(trial.previous),
    }


def _candidate_signature(record: dict) -> tuple:
    response = tuple(np.round(np.asarray(record["response"], dtype=float), 6))
    return (
        response,
        str(record["post_follower_sha256"]),
        str(record["post_physical_sha256"]),
    )


def _dedupe_candidates(records: list[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    for record in records:
        signature = _candidate_signature(record)
        existing = unique.get(signature)
        if existing is None:
            unique[signature] = record
        else:
            for alias in record["aliases"]:
                if alias not in existing["aliases"]:
                    existing["aliases"].append(alias)
    return sorted(
        unique.values(),
        key=lambda row: (
            row["distance_to_pcent"]["overall_rmse"],
            row["step_ttt"],
            row["representative_label"],
        ),
    )


def select_probe_candidates(
    records: list[dict], top_k: int, *, include_coordination_zero: bool = False
) -> list[dict]:
    eligible = [
        record for record in records
        if record["validity_gate_pass"]
        and (
            bool(record["residual_nonzero"])
            or (
                include_coordination_zero
                and record["representative_label"] == "coordination_zero"
            )
        )
    ]
    return eligible[: max(1, int(top_k))]


def choose_oracle_candidate(candidates: list[dict]) -> dict:
    positive = [
        row for row in candidates
        if row["rollout"]["validity_gate_pass"]
        and row["paired"]["long_horizon_positive"]
    ]
    if not positive:
        return {
            "selected": "anchor",
            "execution_branch": "native_anchor",
            "residual_nonzero": {},
            "ttt_gain": 0.0,
            "reason": "no executable candidate cleared the P-Stack long-horizon gate",
        }
    winner = min(positive, key=lambda row: row["rollout"]["ttt"])
    return {
        "selected": winner["representative_label"],
        "execution_branch": "coordination",
        "residual_nonzero": dict(winner["residual_nonzero"]),
        "ttt_gain": float(winner["paired"]["ttt_gain"]),
        "reason": "lowest-TTT executable candidate among long-horizon positives",
    }


def _direct_pcent_one_step(
    source: RLLeaderEnv,
    control,
    *,
    forecast,
    solver_evaluations: int,
) -> dict:
    env = copy.deepcopy(source)
    inventory_before = float(env._inventory())
    log = env.sim.step(control, forecast[0], env.step_idx)
    inventory_after = float(env._inventory())
    validity = _step_validity(
        env,
        forecast[0],
        log.diagnostics,
        inventory_before,
        inventory_after,
    )
    env.previous = control.copy()
    env.step_idx += 1
    return {
        "step_ttt": float(log.urban_ttt + log.freeway_ttt),
        "terminal_inventory": inventory_after,
        "response": env.response_vector(control).astype(float).tolist(),
        "validity_gate_pass": bool(validity),
        "solver_evaluations": int(solver_evaluations),
        "eligible_for_long_horizon_pairing": False,
        "reason": (
            "direct physical P-CENT has no deployable follower-memory transition"
        ),
    }


def _require_rollout_coverage(
    rollout: dict,
    horizons: tuple[int, ...],
    *,
    label: str,
) -> dict:
    required_steps = max(horizons)
    missing = [
        int(horizon)
        for horizon in horizons
        if str(horizon) not in rollout.get("checkpoints", {})
    ]
    missing_follower_memory = [
        int(horizon)
        for horizon in horizons
        if not str(
            rollout.get("checkpoints", {})
            .get(str(horizon), {})
            .get("follower_memory_sha256", "")
        )
    ]
    missing_physical_state = [
        int(horizon)
        for horizon in horizons
        if not str(
            rollout.get("checkpoints", {})
            .get(str(horizon), {})
            .get("physical_state_sha256", "")
        )
    ]
    diagnostics = {
        "label": str(label),
        "required_steps": int(required_steps),
        "completed_steps": int(rollout.get("steps", -1)),
        "missing_horizons": missing,
        "missing_follower_memory_horizons": missing_follower_memory,
        "missing_physical_state_horizons": missing_physical_state,
        "terminal_follower_memory_present": bool(
            rollout.get("terminal_follower_memory_sha256")
        ),
        "terminal_physical_state_present": bool(
            rollout.get("terminal_physical_state_sha256")
        ),
    }
    diagnostics["passed"] = bool(
        diagnostics["completed_steps"] == required_steps
        and not missing
        and not missing_follower_memory
        and not missing_physical_state
        and diagnostics["terminal_follower_memory_present"]
        and diagnostics["terminal_physical_state_present"]
    )
    if not diagnostics["passed"]:
        raise OracleGateError(
            f"{label} did not complete every requested horizon", diagnostics
        )
    return diagnostics


def _long_zero_parity(zero: dict, pstack: dict, horizons: tuple[int, ...]) -> dict:
    zero_coverage = _require_rollout_coverage(
        zero, horizons, label="identity rollout"
    )
    pstack_coverage = _require_rollout_coverage(
        pstack, horizons, label="P-Stack rollout"
    )
    checks = {
        "steps_exact": zero["steps"] == pstack["steps"],
        "ttt_exact": zero["ttt"] == pstack["ttt"],
        "terminal_inventory_exact": (
            zero["terminal_inventory"] == pstack["terminal_inventory"]
        ),
        "first_step_response_exact": (
            zero["first_step_response"] == pstack["first_step_response"]
        ),
        "validity_exact": (
            zero["validity_gate_pass"] == pstack["validity_gate_pass"]
        ),
        "terminal_follower_memory_exact": (
            zero["terminal_follower_memory_sha256"]
            == pstack["terminal_follower_memory_sha256"]
        ),
    }
    for horizon in horizons:
        key = str(horizon)
        checks[f"h{horizon}_exact"] = (
            zero["checkpoints"].get(key) == pstack["checkpoints"].get(key)
        )
    result = {
        "coverage": {
            "identity": zero_coverage,
            "pstack": pstack_coverage,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise OracleGateError(
            f"long-horizon zero-residual parity failed: {failed}", result
        )
    return result


def _validate_branch_contract(
    zero_parity: dict,
    evaluated: list[dict],
    oracle_choice: dict,
) -> dict:
    coordination_identity = bool(
        zero_parity["coordination_zero_is_native_identity"]
    )
    expected_identity = (
        "coordination_zero" if coordination_identity else "native_anchor"
    )
    selected = str(oracle_choice.get("selected", ""))
    choice_branch = str(oracle_choice.get("execution_branch", ""))
    selected_rows = [
        row for row in evaluated
        if row["representative_label"] == selected
    ]
    checks = {
        "known_anchor_branch": (
            zero_parity["anchor_selected_branch"] in KNOWN_ANCHOR_BRANCHES
        ),
        "identity_branch_explicit": (
            zero_parity["identity_execution_branch"] == expected_identity
        ),
        "candidate_branches_explicit": all(
            row.get("execution_branch") == "coordination" for row in evaluated
        ),
        "choice_branch_known": choice_branch in KNOWN_EXECUTION_BRANCHES,
        "native_anchor_choice_consistent": (
            choice_branch != "native_anchor"
            or (
                selected == "anchor"
                and not oracle_choice.get("residual_nonzero", {})
            )
        ),
        "coordination_choice_resolves": (
            choice_branch != "coordination" or len(selected_rows) == 1
        ),
        "coordination_choice_explicit": (
            choice_branch != "coordination"
            or selected_rows[0].get("execution_branch") == "coordination"
        ) if selected_rows else choice_branch != "coordination",
        "coordination_zero_is_branch_switch_only": (
            selected != "coordination_zero" or not coordination_identity
        ),
    }
    result = {
        "requires_explicit_branch_selector": not coordination_identity,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise OracleGateError(
            f"branch contract failed: {failed}", result
        )
    return result


def _event_integrity_passed(event: dict) -> bool:
    return bool(
        event["zero_residual_parity"]["passed"] is True
        and event["long_horizon_identity_parity"]["passed"] is True
        and event["branch_contract"]["passed"] is True
        and event["source_query_isolation"]["passed"] is True
        and event["candidate_order_invariance"]["passed"] is True
    )


def _verify_probe_order(
    source: RLLeaderEnv,
    anchor_context,
    selected: list[dict],
    target_response: np.ndarray,
    scales: np.ndarray,
    families: dict[str, list[int]],
) -> dict:
    checks = []
    for record in reversed(selected):
        residual = residual_from_record(record, source.action_schema.names)
        replay = _probe_candidate(
            source,
            anchor_context,
            record["representative_label"],
            residual,
            target_response,
            scales,
            families,
        )
        checks.append({
            "label": record["representative_label"],
            "response_exact": replay["response"] == record["response"],
            "step_ttt_exact": replay["step_ttt"] == record["step_ttt"],
            "post_follower_exact": (
                replay["post_follower_sha256"]
                == record["post_follower_sha256"]
            ),
        })
    passed = all(
        row["response_exact"]
        and row["step_ttt_exact"]
        and row["post_follower_exact"]
        for row in checks
    )
    if not passed:
        raise RuntimeError("candidate probe results depend on evaluation order")
    return {"performed": True, "passed": passed, "checks": checks}


def evaluate_event(
    source: RLLeaderEnv,
    anchor_context,
    *,
    policy_step: int,
    magnitudes: tuple[float, ...],
    top_k: int,
    horizons: tuple[int, ...],
    verify_order: bool,
) -> dict:
    event_started = time.monotonic()
    source_before = _runtime_fingerprint(source)
    zero_parity = _zero_round_trip(source, anchor_context)
    anchor_control = anchor_context.result.control
    rollout_steps = max(horizons)
    remaining_steps = int(source.n_steps - source.step_idx)
    if remaining_steps < rollout_steps:
        raise OracleGateError(
            "event does not have enough simulation horizon for requested labels",
            {
                "policy_step": int(policy_step),
                "remaining_steps": remaining_steps,
                "required_steps": int(rollout_steps),
                "horizons": list(horizons),
            },
        )

    target_env = copy.deepcopy(source)
    target_forecast = list(copy.deepcopy(anchor_context.forecast))
    target_env._update_rl_far_gate(target_forecast)
    pcent_decision = CentralizedMPC(
        target_env.cfg, mode="proposed"
    ).decide_with_info(
        target_env.sim.state.copy(), target_forecast, target_env.previous
    )
    target_control = pcent_decision.control
    target_response = source.response_vector(target_control)
    scales, families = _response_scales_and_families(source)

    candidates = structured_residual_candidates(
        source,
        anchor_control,
        target_control,
        np.zeros(source.action_dim, dtype=np.float32),
        magnitudes,
    )
    probes = []
    for label, residual in candidates:
        if (
            not zero_parity["coordination_zero_is_native_identity"]
            and not np.any(np.asarray(residual) != 0.0)
        ):
            label = "coordination_zero"
        probes.append(_probe_candidate(
            source,
            anchor_context,
            label,
            residual,
            target_response,
            scales,
            families,
        ))
    unique = _dedupe_candidates(probes)
    selected = select_probe_candidates(
        unique,
        top_k,
        include_coordination_zero=(
            not zero_parity["coordination_zero_is_native_identity"]
        ),
    )
    order_check = (
        _verify_probe_order(
            source,
            anchor_context,
            selected,
            target_response,
            scales,
            families,
        )
        if verify_order else {"performed": False, "passed": None, "checks": []}
    )
    source_after_queries = _runtime_fingerprint(source)
    if source_after_queries != source_before:
        raise RuntimeError("oracle queries mutated the source runtime")

    pstack = _rollout_pstack(
        source,
        anchor_context,
        rollout_steps=rollout_steps,
        horizons=horizons,
    )
    if zero_parity["coordination_zero_is_native_identity"]:
        identity_rollout = _rollout_price_candidate(
            source,
            np.zeros(source.action_dim, dtype=np.float32),
            anchor_context,
            rollout_steps=rollout_steps,
            horizons=horizons,
        )
    else:
        identity_rollout = _rollout_pstack(
            source,
            anchor_context,
            rollout_steps=rollout_steps,
            horizons=horizons,
        )
    long_identity_parity = _long_zero_parity(
        identity_rollout, pstack, horizons
    )
    direct_pcent = _direct_pcent_one_step(
        source,
        target_control,
        forecast=target_forecast,
        solver_evaluations=int(pcent_decision.solver_evaluations),
    )

    evaluated = []
    for record in selected:
        residual = residual_from_record(record, source.action_schema.names)
        rollout = _rollout_price_candidate(
            source,
            residual,
            anchor_context,
            rollout_steps=rollout_steps,
            horizons=horizons,
        )
        _require_rollout_coverage(
            rollout,
            horizons,
            label=f"candidate {record['representative_label']}",
        )
        replay_evidence = _validate_h1_probe_replay(record, rollout, 1.0e-6)
        evaluated.append({
            **record,
            "response_replay_linf": replay_evidence["response_linf"],
            "h1_replay_evidence": replay_evidence,
            "rollout": rollout,
            "paired": _paired_metrics(rollout, pstack),
        })
    evaluated.sort(key=lambda row: row["rollout"]["ttt"])
    oracle_choice = choose_oracle_candidate(evaluated)
    branch_contract = _validate_branch_contract(
        zero_parity, evaluated, oracle_choice
    )

    return {
        "step": int(policy_step),
        "simulation_step": int(source.step_idx),
        "simulation_time_sec": float(source.sim.state.time_sec),
        "pre_runtime_sha256": source_before,
        "forecast_sha256": _digest(anchor_context.forecast),
        "anchor_context_contract": anchor_context.contract_version,
        "anchor_fingerprint": anchor_context.anchor_fingerprint,
        "anchor_selected_branch": anchor_context.coordination.selected_branch,
        "zero_residual_parity": zero_parity,
        "long_horizon_identity_parity": long_identity_parity,
        "branch_contract": branch_contract,
        "requires_explicit_branch_selector": bool(
            not zero_parity["coordination_zero_is_native_identity"]
        ),
        "source_query_isolation": {
            "before_sha256": source_before,
            "after_sha256": source_after_queries,
            "passed": source_before == source_after_queries,
        },
        "candidate_order_invariance": order_check,
        "pcent": {
            "objective": float(pcent_decision.objective),
            "solver_evaluations": int(pcent_decision.solver_evaluations),
            "converged": bool(pcent_decision.converged),
            "control": _control_payload(target_control),
            "response": target_response.astype(float).tolist(),
            "anchor_distance": response_distance(
                source.response_vector(anchor_control),
                target_response,
                scales,
                families,
            ),
        },
        "candidate_generation": {
            "raw_count": len(candidates),
            "unique_response_memory_count": len(unique),
            "selected_for_rollout_count": len(selected),
            "selection_rule": "nearest valid same-state follower responses to P-CENT",
            "all_unique_probes": unique,
        },
        "runtime_budget": {
            "pcent_queries": 1,
            "candidate_first_step_follower_solves": len(candidates),
            "candidate_order_rechecks": len(selected) if verify_order else 0,
            "long_rollout_branches": 2 + len(selected),
            "pstack_continuation_decisions": (
                (2 + len(selected)) * max(0, rollout_steps - 1)
            ),
        },
        "pstack": pstack,
        "direct_pcent_one_step_reference": direct_pcent,
        "candidates_by_ttt": evaluated,
        "oracle_choice": oracle_choice,
        "elapsed_sec": float(time.monotonic() - event_started),
    }


def run_scenario(
    scenario: str,
    event_steps: tuple[int, ...],
    *,
    magnitudes: tuple[float, ...],
    top_k: int,
    horizons: tuple[int, ...],
    verify_order: bool,
    output_path: Path,
) -> dict:
    env = RLLeaderEnv(
        scenario_name=scenario,
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    teacher_contract, teacher_sha256 = _teacher_contract(env)
    result = {
        "format_version": ORACLE_FORMAT_VERSION,
        **env.experiment_contract.artifact_fields(),
        "scenario": scenario,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "response_contract": RL_RESPONSE_CONTRACT_VERSION,
        "anchor_context_contract": PSTACK_ANCHOR_CONTEXT_CONTRACT,
        "teacher_contract": teacher_contract,
        "teacher_contract_sha256": teacher_sha256,
        "implementation_sha256": implementation_fingerprints(),
        "event_steps": list(event_steps),
        "event_coordinate": "controlled_policy_step_after_warmup",
        "magnitudes": list(magnitudes),
        "top_k": int(top_k),
        "horizons": list(horizons),
        "candidate_semantics": (
            "training-time privileged P-CENT direction; every label is an exact "
            "follower-executable residual and deployment defaults to P-Stack"
        ),
        "oracle_semantics": "one-step residual intervention then native P-Stack recovery",
        "branch_semantics": (
            "native_anchor is the safe default; coordination is an explicit candidate "
            "branch whose continuous residual is defined relative to the rich anchor"
        ),
        "policy_value_claim": False,
        "order_check_required_for_pass": True,
        "events": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    event_set = set(event_steps)
    max_event = max(event_steps)
    while int(env.step_idx - env.warmup) <= max_event:
        step = int(env.step_idx - env.warmup)
        if step in event_set:
            anchor_context = env.prepare_pstack_anchor_context()
            result["pending_event"] = {
                "step": step,
                "simulation_step": int(env.step_idx),
                "simulation_time_sec": float(env.sim.state.time_sec),
                "anchor_selected_branch": (
                    anchor_context.coordination.selected_branch
                ),
                "anchor_fingerprint": anchor_context.anchor_fingerprint,
                "anchor_metadata": {
                    key: value
                    for key, value in anchor_context.result.metadata.items()
                    if "selected" in key.lower() or "pfo" in key.lower()
                },
            }
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"oracle-event scenario={scenario} step={step} "
                f"branch={anchor_context.coordination.selected_branch}",
                flush=True,
            )
            try:
                event = evaluate_event(
                    env,
                    anchor_context,
                    policy_step=step,
                    magnitudes=magnitudes,
                    top_k=top_k,
                    horizons=horizons,
                    verify_order=verify_order,
                )
            except OracleGateError as exc:
                result["passed"] = False
                result["failed_event"] = {
                    **result["pending_event"],
                    "error": str(exc),
                    "diagnostics": exc.diagnostics,
                }
                output_path.write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                raise
            result.pop("pending_event", None)
            result["events"].append(event)
            result["elapsed_sec"] = float(time.monotonic() - started)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"oracle scenario={scenario} step={step} "
                f"choice={event['oracle_choice']['selected']} "
                f"gain={event['oracle_choice']['ttt_gain']:+.3f}",
                flush=True,
            )
            if step == max_event:
                break
            env.step_prepared_optimizer_anchor(anchor_context)
        else:
            env.step_optimizer_anchor(sync_follower_state=True)
        print(
            f"source-PStack scenario={scenario} "
            f"reached_policy_step={env.step_idx - env.warmup}/{max_event}",
            flush=True,
        )
    result["elapsed_sec"] = float(time.monotonic() - started)
    result["passed"] = bool(
        len(result["events"]) == len(event_steps)
        and all(_event_integrity_passed(event) for event in result["events"])
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(
            f"oracle scenario {scenario} did not complete every requested event"
        )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="sweet_170_w60,sweet_190_w60")
    parser.add_argument(
        "--events",
        default="",
        help="scenario:step,step;scenario:step (defaults to diagnosed event steps)",
    )
    parser.add_argument("--magnitudes", default="0.25,0.5,1.0")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--skip-order-check", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/rl_phase0_implementation_20260828/pcent_guided_oracle",
    )
    args = parser.parse_args(argv)
    scenarios = tuple(
        item.strip() for item in args.scenarios.split(",") if item.strip()
    )
    event_steps = parse_event_steps(args.events, scenarios)
    magnitudes = tuple(sorted({
        float(item) for item in args.magnitudes.split(",") if item.strip()
    }))
    horizons = tuple(sorted({
        int(item) for item in args.horizons.split(",") if item.strip()
    }))
    if not magnitudes or min(magnitudes) <= 0.0 or max(magnitudes) > 1.0:
        raise ValueError("magnitudes must be in (0, 1]")
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons must be positive")

    output_dir = Path(args.output_dir)
    summaries = []
    for scenario in scenarios:
        output_path = output_dir / f"{scenario}.json"
        result = run_scenario(
            scenario,
            event_steps[scenario],
            magnitudes=magnitudes,
            top_k=max(1, int(args.top_k)),
            horizons=horizons,
            verify_order=not args.skip_order_check,
            output_path=output_path,
        )
        summaries.append({
            "scenario": scenario,
            "passed": result["passed"],
            "events": len(result["events"]),
            "elapsed_sec": result["elapsed_sec"],
            "output": str(output_path),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
