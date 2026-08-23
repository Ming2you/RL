"""Incremental structured collection for the full coordination action space."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rl_leader.env import (
    OPTIMIZER_ANCHOR_CONTRACT,
    RLLeaderEnv,
    make_random_scenario,
    make_targeted_scenario,
)
from src.controllers.coordination import CoordinationMask, RL_RESPONSE_CONTRACT_VERSION


DEFAULT_MODES = (
    "optimizer_anchor",
    "budget",
    "linear",
    "quadratic",
    "single_block",
    "correlated_block",
    "full",
    "epsilon_mixture",
)

TARGETED_MODES = (
    "optimizer_local",
    "loose_anchor",
    "loose_local",
)


def action_block_layout(action_schema):
    layout = []
    offset = 2
    for family, owners in (
        ("urban", action_schema.signals),
        ("freeway", action_schema.ramps),
    ):
        for owner in owners:
            layout.append((family, owner, slice(offset, offset + 5), (offset + 2, offset + 4)))
            offset += 5
    for key in action_schema.nonmerge_vsl_keys:
        layout.append(("vsl", key, slice(offset, offset + 2), (offset + 1,)))
        offset += 2
    for ramp in action_schema.certificate_ramps:
        layout.append(("certificate", ramp, slice(offset, offset + 1), ()))
        offset += 1
    if offset != action_schema.dimension:
        raise RuntimeError("action block layout does not match schema dimension")
    return tuple(layout)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def structured_action(
    env: RLLeaderEnv,
    rng: np.random.Generator,
    mode: str,
    priority_blocks: tuple[str, ...] = (),
    priority_probability: float = 0.0,
):
    action = np.zeros(env.action_schema.dimension, dtype=np.float32)
    certificate_count = len(env.action_schema.certificate_ramps)
    if certificate_count:
        action[-certificate_count:] = -1.0
    action[0] = np.clip(rng.normal(0.0, 0.25), -1.0, 1.0)
    action[1] = np.clip(rng.normal(0.75, 0.12), -1.0, 1.0)
    layout = action_block_layout(env.action_schema)
    owners = tuple(owner for _, owner, _, _ in layout)
    priority_indices = [index for index, owner in enumerate(owners) if owner in priority_blocks]

    def choose_block():
        candidates = priority_indices if priority_indices and rng.random() < priority_probability else range(len(layout))
        return layout[int(rng.choice(tuple(candidates)))]

    if mode == "budget":
        return action, CoordinationMask.named("RL-BUDGET")
    if mode == "linear":
        for family, _, block, positive in layout:
            if family == "certificate":
                action[block.start] = float(rng.choice((-1.0, 1.0)))
                continue
            linear_count = 2 if block.stop - block.start == 5 else 1
            action[block.start:block.start + linear_count] = rng.uniform(-0.8, 0.8, linear_count)
        return action, CoordinationMask.named("RL-LINEAR")
    if mode == "quadratic":
        for _, _, block, positive in layout:
            for index in range(block.start, block.stop):
                if index in positive:
                    action[index] = rng.uniform(0.05, 1.0)
                elif block.stop - block.start == 5 and index == block.start + 3:
                    action[index] = rng.uniform(-0.7, 0.7)
        return action, CoordinationMask(linear=False)
    if mode == "single_block":
        family, _, block, positive = choose_block()
        if family == "certificate":
            action[block.start] = float(rng.choice((-1.0, 1.0)))
            return action, CoordinationMask()
        action[block] = rng.uniform(-0.8, 0.8, block.stop - block.start)
        for index in positive:
            action[index] = abs(action[index])
        return action, CoordinationMask()
    if mode == "correlated_block":
        family, _, block, positive = choose_block()
        if family == "certificate":
            action[block.start] = float(rng.choice((-1.0, 1.0)))
            return action, CoordinationMask()
        if block.stop - block.start == 2:
            action[block.start] = rng.uniform(-0.8, 0.8)
            action[block.start + 1] = rng.uniform(0.05, 1.0)
            return action, CoordinationMask()
        direction = rng.uniform(-0.8, 0.8)
        action[block.start:block.start + 2] = (direction, -direction)
        action[block.start + 2] = rng.uniform(0.05, 1.0)
        action[block.start + 3] = rng.uniform(-0.8, 0.8)
        action[block.start + 4] = rng.uniform(0.05, 1.0)
        return action, CoordinationMask()
    if mode == "full":
        action[2:] = rng.uniform(-0.7, 0.7, action.size - 2)
        for family, _, block, positive in layout:
            if family == "certificate":
                action[block.start] = float(rng.choice((-1.0, 1.0)))
            for index in positive:
                action[index] = abs(action[index])
        return action, CoordinationMask()
    raise ValueError(f"unknown behavior mode: {mode}")


def loose_anchor_action(env: RLLeaderEnv) -> np.ndarray:
    """Represent the least restrictive zero-price action available to the RL follower."""
    action = np.zeros(env.action_schema.dimension, dtype=np.float32)
    certificate_count = len(env.action_schema.certificate_ramps)
    if certificate_count:
        action[-certificate_count:] = -1.0
    action[:2] = env.budget_to_action(
        env.previous.N_P_star,
        env.cfg.leader.N_UF_star_range[1],
    )
    return action


def optimizer_replay_anchor_step(env: RLLeaderEnv):
    """Execute a native leader intent through the deployable RL adapter path."""
    action = env.optimizer_anchor_action()
    mask = CoordinationMask.named("RL-LINEAR")
    env.mask = mask
    next_obs, reward, done, info = env.step(action)
    return next_obs, reward, done, info, action, mask


def temporally_correlated_action(
    base_action: np.ndarray,
    previous_noise: np.ndarray,
    rng: np.random.Generator,
    block_slices: tuple[slice, ...],
    positive_indices: tuple[int, ...],
    *,
    rho: float,
    budget_scale: float,
    block_scale: float,
    active: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an AR(1) local perturbation without leaving the PSD action contract."""
    rho = float(np.clip(rho, 0.0, 0.9999))
    innovation = rng.normal(0.0, 1.0, previous_noise.size)
    noise = rho * previous_noise + np.sqrt(1.0 - rho * rho) * innovation
    if not active:
        return np.asarray(base_action, dtype=np.float32).copy(), noise.astype(np.float32)
    scale = np.zeros(previous_noise.size, dtype=np.float32)
    scale[:2] = float(budget_scale)
    for block in block_slices:
        scale[block] = float(block_scale)
    action = np.clip(np.asarray(base_action, dtype=float) + noise * scale, -1.0, 1.0)
    for index in positive_indices:
        action[index] = max(0.0, action[index])
    return action.astype(np.float32), noise.astype(np.float32)


def _save(path: Path, rows: dict, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "obs": np.asarray(rows["obs"], dtype=np.float32),
        "act": np.asarray(rows["act"], dtype=np.float32),
        "rew": np.asarray(rows["rew"], dtype=np.float32),
        "next_obs": np.asarray(rows["next_obs"], dtype=np.float32),
        "done": np.asarray(rows["done"], dtype=np.float32),
        "episode": np.asarray(rows["episode"], dtype=np.int32),
        "step": np.asarray(rows["step"], dtype=np.int32),
        "behavior_mode": np.asarray(rows["behavior_mode"], dtype="U32"),
        "behavior_submode": np.asarray(rows["behavior_submode"], dtype="U32"),
        "behavior_probability": np.asarray(rows["behavior_probability"], dtype=np.float32),
        "epsilon": np.asarray(rows["epsilon"], dtype=np.float32),
        "exploratory": np.asarray(rows["exploratory"], dtype=np.float32),
        "termination_reason": np.asarray(rows["termination_reason"], dtype="U32"),
        "mask": np.asarray(rows["mask"], dtype=np.float32),
        "response": np.asarray(rows["response"], dtype=np.float32),
        "simulation_time_sec": np.asarray(rows["simulation_time_sec"], dtype=np.float32),
        "perturb_active": np.asarray(rows["perturb_active"], dtype=np.float32),
        "anchor_distance": np.asarray(rows["anchor_distance"], dtype=np.float32),
        "validity": np.asarray(rows["validity"], dtype=np.float32),
        "conservation_residual": np.asarray(rows["conservation_residual"], dtype=np.float32),
        "projection_veh": np.asarray(rows["projection_veh"], dtype=np.float32),
        "rejected_veh": np.asarray(rows["rejected_veh"], dtype=np.float32),
        "throughput_veh": np.asarray(rows["throughput_veh"], dtype=np.float32),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def collect(args) -> Path:
    rng = np.random.default_rng(args.seed)
    collection_start = time.monotonic()
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if not modes:
        raise ValueError("at least one behavior mode is required")
    unknown = set(modes) - set(DEFAULT_MODES) - set(TARGETED_MODES)
    if unknown:
        raise ValueError(f"unknown behavior modes: {sorted(unknown)}")
    priority_blocks = tuple(part.strip() for part in args.priority_blocks.split(",") if part.strip())
    rows = {key: [] for key in (
        "obs", "act", "rew", "next_obs", "done", "episode", "step",
        "behavior_mode", "behavior_probability", "termination_reason", "mask",
        "behavior_submode", "epsilon", "exploratory",
        "response", "simulation_time_sec", "perturb_active", "anchor_distance",
        "validity", "conservation_residual", "projection_veh",
        "rejected_veh", "throughput_veh",
    )}
    output = Path(args.out)
    episode_summaries = []
    action_schema = observation_schema = None
    for episode in range(args.episodes):
        mode = modes[episode % len(modes)]
        scenario = (
            make_targeted_scenario(rng)
            if args.scenario_profile == "targeted"
            else make_random_scenario(rng, holdout_demand=args.stressor_demand_cap)
        )
        env = RLLeaderEnv(
            scenario_dict=scenario,
            T_total=args.t_total,
            warmup_nc_steps=args.warmup,
        )
        action_schema = env.action_schema.metadata()
        observation_schema = env.observation_schema.metadata()
        obs = env.reset()
        start = time.monotonic()
        episode_start = len(rows["obs"])
        layout = action_block_layout(env.action_schema)
        owners = tuple(owner for _, owner, _, _ in layout)
        eligible = [
            index for index, owner in enumerate(owners)
            if not priority_blocks or owner in priority_blocks
        ]
        selected_count = min(max(int(args.perturb_block_count), 1), len(eligible))
        selected_indices = tuple(sorted(int(value) for value in rng.choice(
            eligible, size=selected_count, replace=False
        )))
        selected_owners = tuple(owners[index] for index in selected_indices)
        selected_slices = tuple(layout[index][2] for index in selected_indices)
        positive_indices = tuple(
            index for _, _, _, positive in layout for index in positive
        )
        temporal_noise = np.zeros(env.action_schema.dimension, dtype=np.float32)
        reason = "time_limit"
        error_message = ""
        try:
            for local_step in range(args.max_steps):
                if args.max_total_sec > 0.0:
                    progress = min(
                        (time.monotonic() - collection_start) / args.max_total_sec,
                        1.0,
                    )
                else:
                    progress = (episode * args.max_steps + local_step) / max(
                        args.episodes * args.max_steps - 1, 1
                    )
                epsilon = args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)
                simulation_time_sec = float(env.sim.state.time_sec)
                behavior_submode = mode
                exploratory = mode not in {"optimizer_anchor", "budget", "loose_anchor"}
                perturb_active = False
                anchor_distance = 0.0
                if mode == "optimizer_anchor":
                    next_obs, reward, natural_done, info, action, mask = (
                        optimizer_replay_anchor_step(env)
                    )
                elif mode in {"optimizer_local", "loose_anchor", "loose_local"}:
                    base_action = (
                        env.optimizer_anchor_action()
                        if mode == "optimizer_local"
                        else loose_anchor_action(env)
                    )
                    if mode.endswith("_local"):
                        perturb_active = local_step >= args.perturb_start_step
                        action, temporal_noise = temporally_correlated_action(
                            base_action,
                            temporal_noise,
                            rng,
                            selected_slices,
                            positive_indices,
                            rho=args.temporal_rho,
                            budget_scale=args.budget_perturb_scale,
                            block_scale=args.block_perturb_scale,
                            active=perturb_active,
                        )
                        anchor_distance = float(np.linalg.norm(action - base_action))
                    else:
                        action = base_action
                    mask = CoordinationMask.named("RL-FULL")
                    env.mask = mask
                    next_obs, reward, natural_done, info = env.step(action)
                else:
                    if mode == "epsilon_mixture":
                        exploratory = bool(rng.random() < epsilon)
                        behavior_submode = (
                            str(rng.choice(("linear", "quadratic", "single_block", "correlated_block", "full")))
                            if exploratory else "budget"
                        )
                    action, mask = structured_action(
                        env,
                        rng,
                        behavior_submode,
                        priority_blocks=priority_blocks,
                        priority_probability=args.priority_probability,
                    )
                    env.mask = mask
                    next_obs, reward, natural_done, info = env.step(action)
                invalid = not bool(info["validity_gate_pass"])
                done = bool(natural_done or invalid)
                if invalid:
                    reward -= args.failure_cost
                    reason = "safety_abort"
                elif natural_done:
                    reason = "natural"
                rows["obs"].append(obs)
                rows["act"].append(action)
                rows["rew"].append(reward)
                rows["next_obs"].append(next_obs)
                rows["done"].append(float(done))
                rows["episode"].append(episode)
                rows["step"].append(local_step)
                rows["behavior_mode"].append(mode)
                rows["behavior_submode"].append(behavior_submode)
                if mode == "epsilon_mixture":
                    probability = epsilon / 5.0 if exploratory else 1.0 - epsilon
                else:
                    probability = 1.0 / len(modes)
                rows["behavior_probability"].append(probability)
                rows["epsilon"].append(epsilon if mode == "epsilon_mixture" else 0.0)
                rows["exploratory"].append(float(exploratory))
                rows["termination_reason"].append("")
                rows["mask"].append(mask.as_array())
                rows["response"].append(env.response_vector())
                rows["simulation_time_sec"].append(simulation_time_sec)
                rows["perturb_active"].append(float(perturb_active))
                rows["anchor_distance"].append(anchor_distance)
                rows["validity"].append(float(info["validity_gate_pass"]))
                rows["conservation_residual"].append(info["conservation_residual_veh"])
                rows["projection_veh"].append(info["movement_queue_projection_veh"])
                rows["rejected_veh"].append(info["coupling_offramp_arrivals_rejected_veh"])
                rows["throughput_veh"].append(info["throughput_veh"])
                obs = next_obs
                if (local_step + 1) % 10 == 0:
                    print(
                        f"episode={episode} mode={mode} step={local_step + 1}/{args.max_steps} "
                        f"ttt={env.sim.total_ttt:.3f}",
                        flush=True,
                    )
                collection_limit = (
                    args.max_total_sec > 0.0
                    and time.monotonic() - collection_start >= args.max_total_sec
                )
                if collection_limit:
                    rows["done"][-1] = 1.0
                    reason = "collection_time_limit"
                    break
                if done:
                    break
                if time.monotonic() - start >= args.max_episode_sec:
                    rows["rew"][-1] -= args.failure_cost
                    rows["done"][-1] = 1.0
                    reason = "wall_clock_abort"
                    break
        except Exception as exc:
            reason = "solver_error"
            error_message = f"{type(exc).__name__}: {exc}"
            if len(rows["obs"]) > episode_start:
                rows["rew"][-1] -= args.failure_cost
                rows["done"][-1] = 1.0
        if len(rows["obs"]) > episode_start:
            rows["termination_reason"][-1] = reason
        episode_summaries.append({
            "episode": episode,
            "mode": mode,
            "transitions": len(rows["obs"]) - episode_start,
            "termination_reason": reason,
            "elapsed_sec": time.monotonic() - start,
            "error": error_message,
            "scenario": scenario,
            "perturb_blocks": list(selected_owners) if mode.endswith("_local") else [],
        })
        manifest = {
            "dataset": args.dataset_name,
            "response_contract": RL_RESPONSE_CONTRACT_VERSION,
            "seed": args.seed,
            "git_commit": _git_commit(),
            "action_schema": action_schema,
            "observation_schema": observation_schema,
            "episodes_requested": args.episodes,
            "episodes_completed": episode + 1,
            "transition_count": len(rows["obs"]),
            "modes": list(modes),
            "priority_blocks": list(priority_blocks),
            "priority_probability": float(args.priority_probability),
            "scenario_profile": args.scenario_profile,
            "optimizer_anchor_contract": OPTIMIZER_ANCHOR_CONTRACT,
            "optimizer_anchor_transition_contract": "rl_adapter_replay_v1",
            "perturb_start_step": int(args.perturb_start_step),
            "temporal_rho": float(args.temporal_rho),
            "budget_perturb_scale": float(args.budget_perturb_scale),
            "block_perturb_scale": float(args.block_perturb_scale),
            "perturb_block_count": int(args.perturb_block_count),
            "max_total_sec": float(args.max_total_sec),
            "collection_elapsed_sec": float(time.monotonic() - collection_start),
            "episode_summaries": episode_summaries,
        }
        _save(output, rows, manifest)
        print(
            f"episode={episode} mode={mode} transitions={episode_summaries[-1]['transitions']} "
            f"reason={reason} total={len(rows['obs'])}",
            flush=True,
        )
        if reason == "collection_time_limit":
            break
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=28)
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--max-episode-sec", type=float, default=1800.0)
    parser.add_argument("--max-total-sec", type=float, default=0.0)
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--failure-cost", type=float, default=5000.0)
    parser.add_argument("--epsilon-start", type=float, default=0.6)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--priority-blocks", default="B,R_D_E,R_D_W,A,F")
    parser.add_argument("--priority-probability", type=float, default=0.7)
    parser.add_argument("--stressor-demand-cap", type=float, default=1.8)
    parser.add_argument("--scenario-profile", choices=("broad", "targeted"), default="broad")
    parser.add_argument("--perturb-start-step", type=int, default=24)
    parser.add_argument("--temporal-rho", type=float, default=0.95)
    parser.add_argument("--budget-perturb-scale", type=float, default=0.05)
    parser.add_argument("--block-perturb-scale", type=float, default=0.10)
    parser.add_argument("--perturb-block-count", type=int, default=2)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--dataset-name", default="full_action_v2")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default="data/full_action_v2/worker_100.npz")
    args = parser.parse_args(argv)
    output = collect(args)
    print(f"saved full-action dataset -> {output}", flush=True)


if __name__ == "__main__":
    main()
