"""Produce deterministic counterexamples for observation and action contract aliasing."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv
from src.controllers.coordination import CoordinationMask


ROOT = Path(__file__).resolve().parents[1]


def _block_references(action) -> dict[str, list[float]]:
    return {
        f"urban.{block.owner}": [float(value) for value in block.reference]
        for block in action.urban_blocks
    } | {
        f"freeway.{block.owner}": [float(value) for value in block.reference]
        for block in action.freeway_blocks
    }


def _reference_difference(left: dict[str, list[float]], right: dict[str, list[float]]) -> dict:
    result = {}
    for owner in sorted(left):
        delta = np.asarray(right[owner], dtype=float) - np.asarray(left[owner], dtype=float)
        result[owner] = {
            "delta": delta.tolist(),
            "max_abs": float(np.abs(delta).max()),
        }
    return result


def _previous_control_alias(env: RLLeaderEnv) -> dict:
    state = env.sim.state.copy()
    forecast = env._forecast()
    previous_a = env.previous.copy()
    previous_b = env.previous.copy()
    green_total = float(env.net.effective_green_total)
    for signal in env.action_schema.signals:
        key_p1 = f"{signal}_p1"
        key_p2 = f"{signal}_p2"
        shifted = min(green_total - 6.0, float(previous_b.green_times[key_p1]) + 6.0)
        previous_b.green_times[key_p1] = shifted
        previous_b.green_times[key_p2] = green_total - shifted
        previous_b.offsets[signal] = (
            float(previous_b.offsets.get(signal, 0.0)) + env.net.cycle_length / 8.0
        ) % env.net.cycle_length
    for link in env.net.freeway_links:
        for index in range(len(state.freeway_density.get(link, []))):
            previous_b.vsl[f"{link}__seg{index}"] = 100.0

    obs_a = env.observation_schema.observe(
        state, forecast, previous_a, env.controller, env.n_steps,
    )
    obs_b = env.observation_schema.observe(
        state, forecast, previous_b, env.controller, env.n_steps,
    )
    raw = np.zeros(env.action_dim, dtype=np.float32)
    action_a = env.action_schema.decode(raw, previous_a, CoordinationMask.named("RL-FULL"))
    action_b = env.action_schema.decode(raw, previous_b, CoordinationMask.named("RL-FULL"))
    refs_a = _block_references(action_a)
    refs_b = _block_references(action_b)
    return {
        "observation_equal": bool(np.array_equal(obs_a, obs_b)),
        "observation_max_abs_difference": float(np.abs(obs_a - obs_b).max()),
        "same_raw_action": bool(np.array_equal(raw, raw.copy())),
        "decoded_reference_difference": _reference_difference(refs_a, refs_b),
        "changed_physical_previous_channels": {
            "green_count": len(env.action_schema.signals),
            "offset_count": len(env.action_schema.signals),
            "vsl_segment_count": sum(
                len(state.freeway_density.get(link, [])) for link in env.net.freeway_links
            ),
        },
    }


def _segment_location_alias(env: RLLeaderEnv) -> dict:
    state_a = env.sim.state.copy()
    state_b = env.sim.state.copy()
    link = env.net.freeway_links[0]
    count = len(state_a.freeway_density[link])
    density = np.linspace(0.5, 1.5, count) * float(env.net.rho_crit)
    speed = np.linspace(0.95, 0.45, count) * float(env.net.v_free)
    lanes = np.full(count, float(env.net.freeway_lanes))
    lanes[max(0, count - 3)] -= 1.0
    state_a.freeway_density[link] = density.tolist()
    state_a.freeway_speed[link] = speed.tolist()
    state_a.freeway_effective_lanes[link] = lanes.tolist()
    state_b.freeway_density[link] = density[::-1].tolist()
    state_b.freeway_speed[link] = speed[::-1].tolist()
    state_b.freeway_effective_lanes[link] = lanes[::-1].tolist()

    forecast_a = copy.deepcopy(env._forecast())
    forecast_b = copy.deepcopy(forecast_a)
    segment_a = max(0, count - 3)
    segment_b = max(0, count - 2)
    for demand in forecast_a:
        demand.freeway_lane_loss = {link: {segment_a: 1.0}}
    for demand in forecast_b:
        demand.freeway_lane_loss = {link: {segment_b: 1.0}}
    obs_a = env.observation_schema.observe(
        state_a, forecast_a, env.previous, env.controller, env.n_steps,
    )
    obs_b = env.observation_schema.observe(
        state_b, forecast_b, env.previous, env.controller, env.n_steps,
    )
    return {
        "link": link,
        "segment_a": segment_a,
        "segment_b": segment_b,
        "density_vectors_equal": bool(np.array_equal(density, density[::-1])),
        "lane_vectors_equal": bool(np.array_equal(lanes, lanes[::-1])),
        "observation_equal": bool(np.array_equal(obs_a, obs_b)),
        "observation_max_abs_difference": float(np.abs(obs_a - obs_b).max()),
        "observed_link_aggregates": ["rho_mean", "rho_max", "speed_mean", "max_lane_loss"],
    }


def _follower_state_alias(env: RLLeaderEnv) -> dict:
    follower = env.controller.nash_solver
    follower._prev_coupling = {"arr_A_p1": 100.0}
    follower._np_bias_ratio = 0.8
    observation_a = env._observe()
    follower._prev_coupling = {"arr_A_p1": 200.0}
    follower._np_bias_ratio = 1.2
    observation_b = env._observe()
    return {
        "observation_equal": bool(np.array_equal(observation_a, observation_b)),
        "observation_max_abs_difference": float(
            np.abs(observation_a - observation_b).max()
        ),
        "coupling_key_count": len(env.observation_schema.coupling_keys),
        "memory_field_count": len(env.observation_schema.memory_names),
    }


def audit() -> dict:
    env = RLLeaderEnv(scenario_name="sweet_170_incident_w60")
    optimizer = env._ensure_optimizer_controller()
    rl_follower = env.controller.nash_solver
    optimizer_follower = optimizer.nash_solver
    replay_action = env.action_schema.decode(
        np.zeros(env.action_dim, dtype=np.float32),
        env.previous,
        CoordinationMask.named("RL-FULL"),
    )
    env.controller.potential_adapter.apply(replay_action, rl_follower)
    follower_fields = (
        "segment_agents",
        "joint_green_offset_enabled",
        "ramp_offset_enabled",
        "priced_vsl_segment_candidates_enabled",
        "metering_price_split",
        "seg13_budget_inequality",
        "seg13_release_floor_frac",
        "seg13_release_floor_adaptive",
        "nuf_link_share_mode",
    )
    cfg_fields = (
        "seg13_meter_box_veh_h",
        "seg13_meter_box_up_veh_h",
        "seg13_vsl_box_kmh",
        "baseline_move_box",
        "leader_rollout_box_walk",
        "leader_rollout_box_walk_vg",
    )
    all_vsl_keys = [
        f"{link}__seg{index}"
        for link in env.net.freeway_links
        for index in range(len(env.sim.state.freeway_density.get(link, [])))
    ]
    represented_vsl = sorted(set(env.action_schema.represented_vsl_keys))
    missing_vsl = sorted(set(all_vsl_keys) - set(represented_vsl))
    return {
        "format_version": "coordination_contract_audit_v1",
        "observation_dimension": env.obs_dim,
        "action_dimension": env.action_dim,
        "action_schema_version": env.action_schema.metadata()["version"],
        "observation_schema_version": env.observation_schema.metadata()["version"],
        "previous_control_alias": _previous_control_alias(env),
        "segment_location_alias": _segment_location_alias(env),
        "follower_state_alias": _follower_state_alias(env),
        "vsl_action_coverage": {
            "all_segment_count": len(all_vsl_keys),
            "represented_segment_count": len(represented_vsl),
            "missing_segment_count": len(missing_vsl),
            "represented": represented_vsl,
            "missing": missing_vsl,
        },
        "metering_release_certificate_coverage": {
            "ramp_count": len(env.action_schema.certificate_ramps),
            "ramps": list(env.action_schema.certificate_ramps),
        },
        "follower_runtime_parity": {
            "same_controller_class": type(env.controller) is type(optimizer),
            "same_follower_class": type(rl_follower) is type(optimizer_follower),
            "rl": {name: getattr(rl_follower, name, None) for name in follower_fields},
            "optimizer": {
                name: getattr(optimizer_follower, name, None) for name in follower_fields
            },
            "differences": {
                name: {
                    "rl": getattr(rl_follower, name, None),
                    "optimizer": getattr(optimizer_follower, name, None),
                }
                for name in follower_fields
                if getattr(rl_follower, name, None) != getattr(optimizer_follower, name, None)
            },
        },
        "follower_config_parity": {
            "rl": {name: getattr(env.cfg.mpc, name, None) for name in cfg_fields},
            "optimizer": {
                name: getattr(env.optimizer_cfg.mpc, name, None) for name in cfg_fields
            },
            "differences": {
                name: {
                    "rl": getattr(env.cfg.mpc, name, None),
                    "optimizer": getattr(env.optimizer_cfg.mpc, name, None),
                }
                for name in cfg_fields
                if getattr(env.cfg.mpc, name, None) != getattr(env.optimizer_cfg.mpc, name, None)
            },
        },
        "observation_contract_coverage": {
            "previous_green": any(
                name.startswith("previous.green.") for name in env.observation_schema.names
            ),
            "previous_offset": any(
                name.startswith("previous.offset.") for name in env.observation_schema.names
            ),
            "previous_vsl": any(
                name.startswith("previous.vsl.") for name in env.observation_schema.names
            ),
            "segment_state": any(
                ".segment." in name and name.startswith("freeway.")
                for name in env.observation_schema.names
            ),
            "segment_lane_loss": any(
                ".segment." in name and name.endswith(".lane_loss")
                for name in env.observation_schema.names
            ),
            "follower_coupling": any(
                name.startswith("follower.coupling.")
                for name in env.observation_schema.names
            ),
            "follower_memory": any(
                name.startswith("follower.memory.")
                for name in env.observation_schema.names
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/pstack_gap_diagnosis_v1/coordination_contract_audit.json",
    )
    args = parser.parse_args()
    result = audit()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
