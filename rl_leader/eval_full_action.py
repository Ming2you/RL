"""Evaluate full coordination checkpoints under fixed-shape channel masks."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from rl_leader.env import RLLeaderEnv
from rl_leader.nets import Actor, UnifiedCoordinationActor
from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    RL_RESPONSE_CONTRACT_VERSION,
)


DEFAULT_MASKS = (
    "RL-FULL",
    "RL-LINEAR",
    "RL-BUDGET",
    "RL-URBAN",
    "RL-FREEWAY",
    "RL-NO-CROSS",
)


def load_actor(path: str | Path, env: RLLeaderEnv):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "rl_coordination_checkpoint_v2":
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format_version')!r}")
    action_schema = checkpoint.get("action_schema", {})
    observation_schema = checkpoint.get("observation_schema", {})
    if action_schema.get("version") != ACTION_SCHEMA_VERSION:
        raise ValueError("checkpoint action schema version does not match the controller")
    if observation_schema.get("version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("checkpoint observation schema version does not match the controller")
    if action_schema != env.action_schema.metadata():
        raise ValueError("checkpoint action schema layout does not match the environment")
    if observation_schema != env.observation_schema.metadata():
        raise ValueError("checkpoint observation schema layout does not match the environment")
    if checkpoint["actor_class"] == "UnifiedCoordinationActor":
        actor = UnifiedCoordinationActor(
            env.obs_dim,
            env.action_schema.urban_block_count,
            env.action_schema.freeway_block_count,
            env.action_schema.vsl_block_count,
            len(env.action_schema.certificate_ramps),
        )
    else:
        actor = Actor(env.obs_dim, env.action_dim)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, checkpoint


def _state_snapshot(env):
    state = env.sim.state
    return {
        "time_sec": float(state.time_sec),
        "urban_total_veh": float(state.total_urban_vehicles(env.net)),
        "freeway_density": {
            link: [float(value) for value in state.freeway_density.get(link, [])]
            for link in env.net.freeway_links
        },
        "freeway_speed": {
            link: [float(value) for value in state.freeway_speed.get(link, [])]
            for link in env.net.freeway_links
        },
        "freeway_effective_lanes": {
            link: [float(value) for value in state.freeway_effective_lanes.get(link, [])]
            for link in env.net.freeway_links
        },
        "ramp_queue": {
            ramp: float(state.ramp_queue.get(ramp, 0.0)) for ramp in env.net.ramps
        },
    }


def _control_snapshot(env):
    control = env.previous
    return {
        "N_P_star": float(control.N_P_star),
        "N_UF_star": float(control.N_UF_star),
        "green_times": {key: float(value) for key, value in control.green_times.items()},
        "offsets": {key: float(value) for key, value in control.offsets.items()},
        "ramp_metering": {key: float(value) for key, value in control.ramp_metering.items()},
        "vsl": {key: float(value) for key, value in control.vsl.items()},
    }


def _coordination_snapshot(coordination):
    def block_row(block):
        return {
            "owner": block.owner,
            "lever_keys": list(block.lever_keys),
            "reference": [float(value) for value in block.reference],
            "trust_radius": [float(value) for value in block.trust_radius],
            "linear": [float(value) for value in block.linear],
            "hessian": block.hessian().astype(float).tolist(),
        }

    return {
        "N_P_star": float(coordination.N_P_star),
        "N_UF_star": float(coordination.N_UF_star),
        "mask": coordination.mask.as_array().astype(int).tolist(),
        "urban_blocks": [block_row(block) for block in coordination.urban_blocks],
        "freeway_blocks": [block_row(block) for block in coordination.freeway_blocks],
        "vsl_blocks": [{
            "owner": block.owner,
            "lever_key": block.lever_key,
            "reference": float(block.reference),
            "trust_radius": float(block.trust_radius),
            "linear": float(block.linear),
            "curvature": float(block.curvature()),
        } for block in coordination.vsl_blocks],
        "metering_release_certified": (
            list(coordination.metering_release_certified)
            if coordination.metering_release_certified is not None
            else None
        ),
    }


def _price_snapshot(follower):
    names = (
        "signal_marginal_price", "offset_marginal_price",
        "metering_marginal_price", "vsl_marginal_price",
        "signal_quadratic_price", "offset_quadratic_price",
        "metering_quadratic_price", "vsl_quadratic_price",
        "green_offset_cross_price", "vsl_meter_cross_price",
        "signal_marginal_price_ref", "offset_marginal_price_ref",
        "metering_marginal_price_ref", "vsl_marginal_price_ref",
    )
    result = {}
    for name in names:
        values = getattr(follower, name, None)
        result[name] = (
            {str(key): float(value) for key, value in values.items()}
            if isinstance(values, dict) else None
        )
    return result


def _adapter_receipts(coordination, follower):
    receipts = []
    mask = coordination.mask
    for block in coordination.urban_blocks:
        tg, to = block.trust_radius
        h = block.hessian()
        receipts.append({
            "family": "urban",
            "owner": block.owner,
            "lever_key": block.owner,
            "expected_linear": [
                float(block.linear[0] / tg) if mask.linear and mask.green else 0.0,
                float(block.linear[1] / to) if mask.linear and mask.offset else 0.0,
            ],
            "actual_linear": [
                float((follower.signal_marginal_price or {}).get(block.owner, 0.0)),
                float((follower.offset_marginal_price or {}).get(block.owner, 0.0)),
            ],
            "expected_quadratic": [
                float(h[0, 0] / (tg * tg)) if mask.quadratic and mask.green else 0.0,
                float(h[1, 1] / (to * to)) if mask.quadratic and mask.offset else 0.0,
                float(h[0, 1] / (tg * to))
                if mask.quadratic and mask.cross and mask.green and mask.offset else 0.0,
            ],
            "actual_quadratic": [
                float((follower.signal_quadratic_price or {}).get(block.owner, 0.0)),
                float((follower.offset_quadratic_price or {}).get(block.owner, 0.0)),
                float((follower.green_offset_cross_price or {}).get(block.owner, 0.0)),
            ],
        })
    for block in coordination.freeway_blocks:
        tm, tv = block.trust_radius
        h = block.hessian()
        vsl_key = block.lever_keys[1]
        receipts.append({
            "family": "freeway",
            "owner": block.owner,
            "lever_key": vsl_key,
            "expected_linear": [
                float(block.linear[0] / tm) if mask.linear and mask.metering else 0.0,
                float(block.linear[1] / tv) if mask.linear and mask.vsl else 0.0,
            ],
            "actual_linear": [
                float((follower.metering_marginal_price or {}).get(block.owner, 0.0)),
                float((follower.vsl_marginal_price or {}).get(vsl_key, 0.0)),
            ],
            "expected_quadratic": [
                float(h[0, 0] / (tm * tm)) if mask.quadratic and mask.metering else 0.0,
                float(h[1, 1] / (tv * tv)) if mask.quadratic and mask.vsl else 0.0,
                float(h[0, 1] / (tm * tv))
                if mask.quadratic and mask.cross and mask.metering and mask.vsl else 0.0,
            ],
            "actual_quadratic": [
                float((follower.metering_quadratic_price or {}).get(block.owner, 0.0)),
                float((follower.vsl_quadratic_price or {}).get(vsl_key, 0.0)),
                float((follower.vsl_meter_cross_price or {}).get(block.owner, 0.0)),
            ],
        })
    for block in coordination.vsl_blocks:
        trust = float(block.trust_radius)
        receipts.append({
            "family": "vsl",
            "owner": block.owner,
            "lever_key": block.owner,
            "expected_linear": [
                float(block.linear / trust) if mask.linear and mask.vsl else 0.0,
            ],
            "actual_linear": [
                float((follower.vsl_marginal_price or {}).get(block.owner, 0.0)),
            ],
            "expected_quadratic": [
                float(block.curvature() / (trust * trust))
                if mask.quadratic and mask.vsl else 0.0,
            ],
            "actual_quadratic": [
                float((follower.vsl_quadratic_price or {}).get(block.owner, 0.0)),
            ],
        })
    return receipts


def rollout(
    checkpoint_path, scenario, mask, max_steps, max_sec,
    trace_path: str | Path | None = None, diagnostic_candidates: bool = False,
):
    env = RLLeaderEnv(scenario_name=scenario, mask=mask)
    actor, checkpoint = load_actor(checkpoint_path, env)
    obs = env.reset()
    env.follower.diagnostic_trace_enabled = bool(diagnostic_candidates)
    start = time.monotonic()
    total_ttt = throughput = 0.0
    phase_ttt = {"peak": 0.0, "recovery": 0.0}
    validity = True
    raw_nuf = []
    projected_nuf = []
    realized_meter = []
    support_out = []
    steps = 0
    done = False
    trace_output = Path(trace_path) if trace_path else None
    trace_handle = None
    if trace_output is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_output.open("w", encoding="utf-8")
        trace_output.with_suffix(".meta.json").write_text(json.dumps({
            "checkpoint": str(checkpoint_path),
            "response_contract": RL_RESPONSE_CONTRACT_VERSION,
            "checkpoint_response_contract": checkpoint.get("dataset", {}).get(
                "response_contract", "legacy"
            ),
            "seed": int(checkpoint["training"]["seed"]),
            "scenario": scenario,
            "mask": mask,
            "simulation_T_total_sec": float(env.T_total),
            "control_interval_sec": float(env.dt),
            "rho_crit": float(env.net.rho_crit),
            "warmup_steps": int(env.warmup),
            "policy_max_steps": int(max_steps),
            "action_schema": env.action_schema.metadata(),
            "observation_schema": env.observation_schema.metadata(),
            "diagnostic_candidates": bool(diagnostic_candidates),
        }, indent=2), encoding="utf-8")
    try:
        while not done and steps < max_steps and time.monotonic() - start < max_sec:
            time_sec_before = float(env.sim.state.time_sec)
            state_before = _state_snapshot(env) if trace_handle else None
            obs_before = obs
            action = actor.act(obs, deterministic=True)
            low = checkpoint["action_support_low"].cpu().numpy()
            high = checkpoint["action_support_high"].cpu().numpy()
            action_support_out = (action < low) | (action > high)
            support_out.append(float(np.mean(action_support_out)))
            requested = env.action_schema.decode(action, env.previous, env.mask)
            obs, _, done, info = env.step(action)
            coordination = env.controller.last_coordination_action or requested
            total_ttt += info["step_ttt"]
            throughput += info["throughput_veh"]
            phase = "peak" if time_sec_before < 5220.0 else "recovery"
            phase_ttt[phase] += float(info["step_ttt"])
            validity = validity and bool(info["validity_gate_pass"])
            raw_nuf.append(info["raw_N_UF"])
            projected_nuf.append(info["projected_N_UF"])
            realized_meter.append(info["realized_metering_total"])
            if info["native_price_refresh_count"] != 0.0:
                raise RuntimeError("RL evaluation unexpectedly enabled the native price generator")
            if trace_handle:
                relevant_info = {
                    key: float(value)
                    for key, value in info.items()
                    if isinstance(value, (int, float, bool))
                }
                trace_handle.write(json.dumps({
                    "step": int(steps),
                    "state_before": state_before,
                    "observation": {
                        name: float(value)
                        for name, value in zip(env.observation_schema.names, obs_before)
                    },
                    "raw_action": {
                        name: float(value)
                        for name, value in zip(env.action_schema.names, action)
                    },
                    "support_out_names": [
                        name for name, outside in zip(env.action_schema.names, action_support_out)
                        if outside
                    ],
                    "requested_coordination": _coordination_snapshot(requested),
                    "bounded_coordination": _coordination_snapshot(coordination),
                    "adapter_receipts": _adapter_receipts(coordination, env.follower),
                    "follower_prices": _price_snapshot(env.follower),
                    "candidate_trace": env.follower.last_candidate_trace,
                    "control_after": _control_snapshot(env),
                    "info": relevant_info,
                }, separators=(",", ":")) + "\n")
                trace_handle.flush()
            steps += 1
            if steps == 1 or steps % 5 == 0 or done:
                print(
                    f"progress seed={checkpoint['training']['seed']} scenario={scenario} "
                    f"mask={mask} step={steps}/{max_steps} ttt={total_ttt:.3f} "
                    f"valid={int(validity)}",
                    flush=True,
                )
    finally:
        if trace_handle:
            trace_handle.close()
    return {
        "checkpoint": str(checkpoint_path),
        "response_contract": RL_RESPONSE_CONTRACT_VERSION,
        "checkpoint_response_contract": checkpoint.get("dataset", {}).get(
            "response_contract", "legacy"
        ),
        "seed": int(checkpoint["training"]["seed"]),
        "scenario": scenario,
        "mask": mask,
        "steps": steps,
        "complete": bool(done),
        "final_simulation_time_sec": float(env.sim.state.time_sec),
        "wall_clock_truncated": bool(not done and time.monotonic() - start >= max_sec),
        "total_ttt": float(total_ttt),
        "peak_ttt": float(phase_ttt["peak"]),
        "recovery_ttt": float(phase_ttt["recovery"]),
        "throughput_veh": float(throughput),
        "validity_gate_pass": bool(validity),
        "raw_N_UF_mean": float(np.mean(raw_nuf)) if raw_nuf else 0.0,
        "projected_N_UF_mean": float(np.mean(projected_nuf)) if projected_nuf else 0.0,
        "realized_metering_mean": float(np.mean(realized_meter)) if realized_meter else 0.0,
        "support_out_fraction": float(np.mean(support_out)) if support_out else 0.0,
        "elapsed_sec": float(time.monotonic() - start),
        "trace_path": str(trace_output) if trace_output is not None else "",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--scenarios", default="sweet_190_skew15_w60,sweet_190_incident_w60")
    parser.add_argument("--masks", default=",".join(DEFAULT_MASKS))
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--max-sec", type=float, default=3600.0)
    parser.add_argument("--out", default="results/full_action_validation.csv")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--diagnostic-candidates", action="store_true")
    args = parser.parse_args(argv)
    checkpoints = []
    for pattern in args.checkpoints:
        matches = sorted(glob.glob(pattern))
        checkpoints.extend(matches or [pattern])
    scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    masks = [value.strip() for value in args.masks.split(",") if value.strip()]
    rows = []
    for checkpoint in checkpoints:
        for scenario in scenarios:
            for mask in masks:
                trace_path = None
                if args.trace_dir:
                    trace_path = Path(args.trace_dir) / (
                        f"{Path(checkpoint).stem}__{scenario}__{mask}.jsonl"
                    )
                row = rollout(
                    checkpoint, scenario, mask, args.max_steps, args.max_sec,
                    trace_path=trace_path,
                    diagnostic_candidates=args.diagnostic_candidates,
                )
                rows.append(row)
                print(
                    f"seed={row['seed']} scenario={scenario} mask={mask} "
                    f"steps={row['steps']} ttt={row['total_ttt']:.3f} "
                    f"valid={int(row['validity_gate_pass'])}",
                    flush=True,
                )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved evaluation -> {output}", flush=True)


if __name__ == "__main__":
    main()
