"""Evaluate P-CENT-guided reachable price responses on paired H-step rollouts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rl_leader.diagnose_anchor_context_parity import _dynamic_follower_payload
from rl_leader.diagnose_phase0_parity import _digest, _physical_snapshot
from rl_leader.diagnose_pcent_reachability import _replay_event_snapshots
from rl_leader.generate_long_horizon_labels import counterfactual_verdict
from src.controllers.centralized_mpc import CentralizedMPC


PRICE_ATTRIBUTION_ANCHOR_BRANCHES = frozenset({"coarse", "refined"})


def _require_price_anchor_branch(branch: str) -> None:
    if str(branch) not in PRICE_ATTRIBUTION_ANCHOR_BRANCHES:
        raise RuntimeError(
            "price attribution requires a coarse/refined native anchor; "
            f"got {branch!r}"
        )


def _require_requested_horizon_capacity(
    rollout_steps: int, horizons: tuple[int, ...]
) -> None:
    if not horizons or min(horizons) < 1:
        raise RuntimeError("requested attribution horizons are invalid")
    required = max(horizons)
    if int(rollout_steps) < required:
        raise RuntimeError(
            f"requested H{required} attribution has only {rollout_steps} rollout steps"
        )


def _require_rollout_horizon_coverage(
    rollout: dict, horizons: tuple[int, ...], *, label: str
) -> None:
    missing = [horizon for horizon in horizons if str(horizon) not in rollout["checkpoints"]]
    if int(rollout.get("steps", 0)) < max(horizons) or missing:
        raise RuntimeError(
            f"{label} did not cover requested horizons: missing={missing}"
        )


def _validate_h1_probe_replay(
    record: dict, rollout: dict, parity_tolerance: float
) -> dict:
    expected_memory = record.get("post_follower_sha256")
    if not expected_memory:
        raise RuntimeError("candidate probe is missing follower memory provenance")
    checkpoint = rollout.get("checkpoints", {}).get("1")
    if not isinstance(checkpoint, dict) or not checkpoint.get("follower_memory_sha256"):
        raise RuntimeError("H1 rollout is missing follower memory provenance")
    response_error = float(np.max(np.abs(
        np.asarray(rollout["first_step_response"], dtype=float)
        - np.asarray(record["response"], dtype=float)
    )))
    if response_error > parity_tolerance:
        raise RuntimeError(f"H1 response replay mismatch: {response_error}")
    actual_memory = str(checkpoint["follower_memory_sha256"])
    if actual_memory != str(expected_memory):
        raise RuntimeError("H1 follower memory replay mismatch")
    expected_physical = record.get("post_physical_sha256")
    if not expected_physical:
        raise RuntimeError("candidate probe is missing physical state provenance")
    actual_physical = checkpoint.get("physical_state_sha256")
    if not actual_physical:
        raise RuntimeError("H1 rollout is missing physical state provenance")
    if str(actual_physical) != str(expected_physical):
        raise RuntimeError("H1 physical state replay mismatch")
    return {
        "probe_response": list(map(float, record["response"])),
        "rollout_response": list(map(float, rollout["first_step_response"])),
        "response_linf": response_error,
        "response_tolerance": float(parity_tolerance),
        "response_exact": True,
        "probe_follower_memory_sha256": str(expected_memory),
        "rollout_follower_memory_sha256": actual_memory,
        "follower_memory_exact": True,
        "probe_physical_sha256": str(expected_physical),
        "rollout_physical_sha256": str(actual_physical),
        "physical_exact": True,
        "passed": True,
    }


def residual_from_record(record: dict, action_names: tuple[str, ...]) -> np.ndarray:
    values = record.get("residual_nonzero", {})
    if not isinstance(values, dict):
        raise ValueError("candidate residual_nonzero must be a mapping")
    extra = sorted(set(values) - set(action_names))
    if extra:
        raise ValueError(f"candidate residual contains unknown fields: {extra}")
    return np.asarray([float(values.get(name, 0.0)) for name in action_names], dtype=np.float32)


def _follower_memory_fingerprint(env) -> str:
    """Hash persistent follower state, excluding the current price potential."""
    return _digest(_dynamic_follower_payload(env.controller.nash_solver))


def _checkpoint(
    checkpoints: dict[str, dict[str, float]],
    completed: int,
    cumulative_ttt: float,
    env,
    requested: set[int],
    validity_gate_pass: bool,
) -> None:
    if completed in requested:
        checkpoints[str(completed)] = {
            "ttt": float(cumulative_ttt),
            "terminal_inventory": float(env._inventory()),
            "follower_memory_sha256": _follower_memory_fingerprint(env),
            "physical_state_sha256": _digest(_physical_snapshot(env)),
            "validity_gate_pass": bool(validity_gate_pass),
        }


def _continue_with_pstack(
    env,
    *,
    first_step_ttt: float,
    first_step_response: np.ndarray,
    first_step_valid: bool,
    done: bool,
    rollout_steps: int,
    horizons: tuple[int, ...],
) -> dict:
    cumulative_ttt = float(first_step_ttt)
    completed = 1
    validity = bool(first_step_valid)
    checkpoints: dict[str, dict[str, float]] = {}
    requested = set(horizons) | {rollout_steps}
    _checkpoint(
        checkpoints, completed, cumulative_ttt, env, requested, validity
    )
    while not done and completed < rollout_steps:
        _, reward, done, info, _ = env.step_optimizer_anchor(sync_follower_state=True)
        cumulative_ttt += -float(reward)
        completed += 1
        validity = validity and bool(info["validity_gate_pass"])
        _checkpoint(
            checkpoints, completed, cumulative_ttt, env, requested, validity
        )
    _checkpoint(
        checkpoints, completed, cumulative_ttt, env, {completed}, validity
    )
    return {
        "steps": int(completed),
        "ttt": float(cumulative_ttt),
        "terminal_inventory": float(env._inventory()),
        "final_simulation_time_sec": float(env.sim.state.time_sec),
        "first_step_ttt": float(first_step_ttt),
        "first_step_response": np.asarray(first_step_response, dtype=float).tolist(),
        "first_step_validity_gate_pass": bool(first_step_valid),
        "validity_gate_pass": bool(validity),
        "terminal_follower_memory_sha256": _follower_memory_fingerprint(env),
        "terminal_physical_state_sha256": _digest(_physical_snapshot(env)),
        "checkpoints": checkpoints,
    }


def _rollout_price_candidate(
    source_env,
    residual: np.ndarray,
    anchor_context,
    *,
    rollout_steps: int,
    horizons: tuple[int, ...],
) -> dict:
    env, anchor_context = copy.deepcopy((source_env, anchor_context))
    _, reward, done, info = env.step_anchored_candidate(
        residual,
        anchor_context,
    )
    return _continue_with_pstack(
        env,
        first_step_ttt=-float(reward),
        first_step_response=env.response_vector(env.previous),
        first_step_valid=bool(info["validity_gate_pass"]),
        done=done,
        rollout_steps=rollout_steps,
        horizons=horizons,
    )


def _rollout_pstack(
    source_env,
    anchor_context,
    *,
    rollout_steps: int,
    horizons: tuple[int, ...],
) -> dict:
    env, anchor_context = copy.deepcopy((source_env, anchor_context))
    _, reward, done, info, _ = env.step_prepared_optimizer_anchor(
        anchor_context,
        sync_follower_state=True,
    )
    return _continue_with_pstack(
        env,
        first_step_ttt=-float(reward),
        first_step_response=env.response_vector(env.previous),
        first_step_valid=bool(info["validity_gate_pass"]),
        done=done,
        rollout_steps=rollout_steps,
        horizons=horizons,
    )


def _rollout_direct_pcent(
    source_env, *, rollout_steps: int, horizons: tuple[int, ...]
) -> dict:
    env = copy.deepcopy(source_env)
    forecast = env._forecast()
    env._update_rl_far_gate(forecast)
    decision = CentralizedMPC(env.cfg, mode="proposed").decide_with_info(
        env.sim.state.copy(), forecast, env.previous
    )
    _, step_ttt, done = env.step_with_control(decision.control)
    result = _continue_with_pstack(
        env,
        first_step_ttt=float(step_ttt),
        first_step_response=env.response_vector(env.previous),
        first_step_valid=True,
        done=done,
        rollout_steps=rollout_steps,
        horizons=horizons,
    )
    result["solver_evaluations"] = int(decision.solver_evaluations)
    return result


def _paired_metrics(candidate: dict, pstack: dict) -> dict:
    if candidate["steps"] != pstack["steps"]:
        raise RuntimeError("candidate and P-Stack completed different horizons")
    horizons = {}
    common = sorted(set(candidate["checkpoints"]) & set(pstack["checkpoints"]), key=int)
    for key in common:
        cand = candidate["checkpoints"][key]
        base = pstack["checkpoints"][key]
        horizons[key] = {
            "candidate_ttt": float(cand["ttt"]),
            "pstack_ttt": float(base["ttt"]),
            **counterfactual_verdict(
                candidate_ttt=cand["ttt"],
                pstack_ttt=base["ttt"],
                candidate_terminal_inventory=cand["terminal_inventory"],
                pstack_terminal_inventory=base["terminal_inventory"],
            ),
        }
    return {
        **counterfactual_verdict(
            candidate_ttt=candidate["ttt"],
            pstack_ttt=pstack["ttt"],
            candidate_terminal_inventory=candidate["terminal_inventory"],
            pstack_terminal_inventory=pstack["terminal_inventory"],
        ),
        "horizons": horizons,
    }


def evaluate_trace(
    trace_path: Path,
    reachability_path: Path,
    *,
    top_k: int,
    max_rollout_steps: int,
    horizons: tuple[int, ...],
    parity_tolerance: float,
    output_path: Path,
) -> dict:
    meta, snapshots = _replay_event_snapshots(trace_path, parity_tolerance)
    reachability = json.loads(reachability_path.read_text(encoding="utf-8"))
    if reachability.get("format_version") != "pcent_price_reachability_v2_native_anchor":
        raise ValueError("reachability artifact does not use the native anchor context")
    reach_events = {int(event["step"]): event for event in reachability["events"]}
    expected = [int(snapshot["step"]) for snapshot in snapshots]
    if sorted(reach_events) != sorted(expected):
        raise ValueError(
            f"reachability event mismatch: trace={expected}, reach={sorted(reach_events)}"
        )

    output = {
        "format_version": "reachable_candidate_attribution_v2_native_anchor",
        "source_trace": str(trace_path),
        "source_reachability": str(reachability_path),
        "scenario": str(meta["scenario"]),
        "mask": str(meta["mask"]),
        "experiment_contract_sha256": str(meta["experiment_contract_sha256"]),
        "top_k": int(top_k),
        "max_rollout_steps": int(max_rollout_steps),
        "requested_horizons": list(horizons),
        "events": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for snapshot in snapshots:
        env = snapshot["env"]
        remaining = int(env.n_steps - env.step_idx)
        rollout_steps = remaining if max_rollout_steps <= 0 else min(remaining, max_rollout_steps)
        if rollout_steps <= 0:
            raise RuntimeError("candidate event has no remaining rollout steps")
        _require_requested_horizon_capacity(rollout_steps, horizons)

        anchor_context = env.prepare_pstack_anchor_context()
        _require_price_anchor_branch(anchor_context.coordination.selected_branch)
        pstack = _rollout_pstack(
            env,
            anchor_context,
            rollout_steps=rollout_steps,
            horizons=horizons,
        )
        direct_pcent = _rollout_direct_pcent(
            env, rollout_steps=rollout_steps, horizons=horizons
        )
        _require_rollout_horizon_coverage(pstack, horizons, label="P-Stack")
        _require_rollout_horizon_coverage(
            direct_pcent, horizons, label="direct P-CENT"
        )
        direct_pcent["paired"] = _paired_metrics(direct_pcent, pstack)

        reach_event = reach_events[int(snapshot["step"])]
        if reach_event.get("anchor_fingerprint") != anchor_context.anchor_fingerprint:
            raise ValueError(
                f"anchor fingerprint mismatch at step {snapshot['step']}"
            )
        pstack_response = np.asarray(reach_event["pstack_response"], dtype=float)
        selected = []
        for record in reach_event["best_candidates"]:
            response = np.asarray(record["response"], dtype=float)
            if np.max(np.abs(response - pstack_response)) <= 1.0e-6:
                continue
            selected.append(record)
            if len(selected) >= top_k:
                break

        candidates = []
        for record in selected:
            residual = residual_from_record(record, env.action_schema.names)
            rollout = _rollout_price_candidate(
                env,
                residual,
                anchor_context,
                rollout_steps=rollout_steps,
                horizons=horizons,
            )
            _require_rollout_horizon_coverage(
                rollout, horizons, label=f"candidate {record['representative_label']}"
            )
            replay = _validate_h1_probe_replay(
                record, rollout, parity_tolerance
            )
            candidates.append({
                "representative_label": str(record["representative_label"]),
                "aliases": list(record["aliases"]),
                "distance_to_pcent": dict(record["distance_to_pcent"]),
                "residual_nonzero": dict(record["residual_nonzero"]),
                "h1_replay_evidence": replay,
                "response_replay_linf": replay["response_linf"],
                "rollout": rollout,
                "paired": _paired_metrics(rollout, pstack),
            })

        candidates.sort(key=lambda row: row["rollout"]["ttt"])
        event = {
            "step": int(snapshot["step"]),
            "simulation_time_sec": float(snapshot["simulation_time_sec"]),
            "anchor_context_contract": anchor_context.contract_version,
            "anchor_fingerprint": anchor_context.anchor_fingerprint,
            "anchor_selected_branch": anchor_context.coordination.selected_branch,
            "rollout_steps": int(rollout_steps),
            "evaluated_candidate_count": len(candidates),
            "pstack": pstack,
            "direct_pcent": direct_pcent,
            "candidates_by_ttt": candidates,
        }
        output["events"].append(event)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        best_gain = candidates[0]["paired"]["ttt_gain"] if candidates else float("nan")
        print(
            f"candidate-attribution scenario={output['scenario']} step={event['step']} "
            f"tested={len(candidates)} best_gain={best_gain:+.3f} "
            f"pcent_gain={direct_pcent['paired']['ttt_gain']:+.3f}",
            flush=True,
        )
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("reachability")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rollout-steps", type=int, default=12)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    horizons = tuple(sorted({
        int(value) for value in args.horizons.split(",") if value.strip()
    }))
    evaluate_trace(
        Path(args.trace),
        Path(args.reachability),
        top_k=max(1, int(args.top_k)),
        max_rollout_steps=int(args.max_rollout_steps),
        horizons=horizons,
        parity_tolerance=float(args.parity_tolerance),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
