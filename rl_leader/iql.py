"""Offline IQL trainer for legacy budget and full coordination datasets."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_leader.nets import Actor, Critic, UnifiedCoordinationActor, mlp
from rl_leader.env import (
    OPTIMIZER_ANCHOR_CONTRACT,
    OPTIMIZER_ANCHOR_TRANSITION_CONTRACT,
    OPTIMIZER_PREVIEW_CONTRACT,
    WARMUP_CONTROL_CONTRACT,
)
from rl_leader.data_contract import (
    IQL_REWARD_CONTRACT,
    IQL_TERMINAL_CONTRACT,
    LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT,
    RL_CHECKPOINT_FORMAT,
    PSTACK_RESIDUAL_DATA_CONTRACT,
    PSTACK_RESIDUAL_CRITIC_ACTION_CONTRACT,
    balanced_sampling_probabilities,
    continuous_actor_supervision_mask,
    iql_bootstrap_done,
    iql_training_rewards,
    pstack_residual_actor_supervision_mask,
    pstack_residual_deployed_actions,
    pstack_residual_trainable_dimension_mask,
    pstack_residual_targets,
    scenario_phase_cells,
    validate_pstack_residual_rows,
)
from rl_leader.experiment_contract import verify_contract_map
from src.controllers.coordination import ACTION_SCHEMA_VERSION


def _manifest_from_npz(dataset) -> dict:
    if "manifest_json" not in dataset.files:
        return {}
    value = dataset["manifest_json"]
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return json.loads(str(value))


def load_data(pattern, action_parameterization="absolute"):
    obs_parts, action_parts, reward_parts, next_parts, done_parts = [], [], [], [], []
    actor_supervision_parts = []
    actor_target_parts = []
    sampling_cell_parts = []
    actor_sampling_cell_parts = []
    manifests = []
    timeout_bootstrap_count = 0
    wall_clock_reward_adjustment_count = 0
    patterns = [value.strip() for value in str(pattern).split(",") if value.strip()]
    files = sorted({path for value in patterns for path in glob.glob(value)})
    if not files:
        raise SystemExit(f"no dataset files matched: {pattern}")
    for path in files:
        try:
            dataset = np.load(path, allow_pickle=False)
            if int(dataset["obs"].shape[0]) == 0:
                if action_parameterization == "pstack_residual":
                    raise ValueError("native-anchor dataset is empty")
                continue
            manifest = _manifest_from_npz(dataset)
            if action_parameterization == "pstack_residual":
                if not manifest:
                    raise ValueError("native-anchor dataset is missing its manifest")
                validate_pstack_residual_rows(dataset, manifest)
            if action_parameterization == "pstack_residual":
                actor_supervision = pstack_residual_actor_supervision_mask(dataset)
                actor_target = pstack_residual_targets(dataset)
                critic_action = pstack_residual_deployed_actions(dataset)
            else:
                actor_supervision = continuous_actor_supervision_mask(dataset)
                actor_target = np.asarray(dataset["act"], dtype=np.float32)
                critic_action = np.asarray(dataset["act"], dtype=np.float32)
            if np.any(actor_supervision) and not np.all(np.isfinite(
                actor_target[actor_supervision]
            )):
                raise ValueError("actor supervision contains nonfinite targets")
            if manifest:
                verified_contracts = verify_contract_map(
                    manifest.get("experiment_contracts", {})
                )
                if not verified_contracts:
                    raise ValueError("dataset is missing verified experiment contracts")
                summaries = manifest.get("episode_summaries", [])
                episode_contracts = {
                    int(summary["episode"]): str(
                        summary.get("experiment_contract_sha256", "")
                    )
                    for summary in summaries
                }
                present_episodes = set(map(int, np.unique(dataset["episode"])))
                if not present_episodes.issubset(episode_contracts):
                    raise ValueError(
                        "dataset episode summaries do not cover every transition episode"
                    )
                unknown = set(episode_contracts.values()) - set(verified_contracts)
                if unknown:
                    raise ValueError(
                        f"dataset episodes reference unknown contracts: {sorted(unknown)}"
                    )
                if (
                    action_parameterization == "pstack_residual"
                    and manifest.get("action_parameterization_support")
                    != PSTACK_RESIDUAL_DATA_CONTRACT
                ):
                    raise ValueError("dataset does not contain P-Stack residual anchors")
                if action_parameterization == "pstack_residual" and (
                    manifest.get("optimizer_anchor_contract")
                    != OPTIMIZER_ANCHOR_CONTRACT
                    or bool(manifest.get("pfo_supervisor", False))
                    or not bool(manifest.get("pstack_anchor", False))
                    or manifest.get("warmup_control_contract")
                    != WARMUP_CONTROL_CONTRACT
                ):
                    raise ValueError(
                        "P-Stack residual data does not use the native anchor contract"
                    )
                if (
                    "optimizer_anchor" in manifest.get("modes", [])
                    and manifest.get("optimizer_anchor_transition_contract")
                    != OPTIMIZER_ANCHOR_TRANSITION_CONTRACT
                ):
                    raise ValueError(
                        "optimizer_anchor transitions bypassed the deployable RL adapter"
                    )
                if (
                    manifest.get("actor_supervision_contract")
                    == LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT
                    and manifest.get("optimizer_preview_contract")
                    != OPTIMIZER_PREVIEW_CONTRACT
                ):
                    raise ValueError(
                        "long-horizon residual data lacks the side-effect-free preview contract"
                    )
                manifests.append(manifest)
            obs_parts.append(dataset["obs"])
            action_parts.append(critic_action)
            failure_cost = float(manifest.get("failure_cost", 5000.0))
            reward_parts.append(iql_training_rewards(dataset, failure_cost))
            if "termination_reason" in dataset:
                wall_clock_reward_adjustment_count += int(np.count_nonzero(
                    np.asarray(dataset["termination_reason"]).astype(str)
                    == "wall_clock_abort"
                ))
            next_parts.append(dataset["next_obs"])
            bootstrap_done = iql_bootstrap_done(dataset)
            timeout_bootstrap_count += int(np.count_nonzero(
                (np.asarray(dataset["done"], dtype=float) > 0.5)
                & (bootstrap_done < 0.5)
            ))
            done_parts.append(bootstrap_done)
            actor_supervision_parts.append(actor_supervision)
            actor_target_parts.append(actor_target)
            sampling_cells = scenario_phase_cells(dataset, manifest)
            sampling_cell_parts.append(sampling_cells)
            if action_parameterization == "pstack_residual":
                modes = np.asarray(dataset["behavior_mode"]).astype(str)
                actor_sampling_cell_parts.append(np.asarray([
                    f"{cell}|{'accepted_local' if mode == 'optimizer_local' else 'zero_anchor'}"
                    for cell, mode in zip(sampling_cells, modes)
                ]))
            else:
                actor_sampling_cell_parts.append(sampling_cells)
        except Exception as exc:
            if action_parameterization == "pstack_residual":
                raise ValueError(f"invalid native-anchor dataset {path}: {exc}") from exc
            print(f"skip {path}: {exc}", flush=True)
    if not obs_parts:
        raise SystemExit("all matched dataset files were empty or invalid")
    obs = np.concatenate(obs_parts).astype(np.float32)
    action = np.concatenate(action_parts).astype(np.float32)
    reward = np.concatenate(reward_parts).astype(np.float32).reshape(-1, 1)
    next_obs = np.concatenate(next_parts).astype(np.float32)
    done = np.concatenate(done_parts).astype(np.float32).reshape(-1, 1)
    actor_supervision = np.concatenate(actor_supervision_parts).astype(bool)
    actor_target = np.concatenate(actor_target_parts).astype(np.float32)
    sampling_cells = np.concatenate(sampling_cell_parts)
    actor_sampling_cells = np.concatenate(actor_sampling_cell_parts)
    for manifest in manifests[1:]:
        if manifest.get("action_schema") != manifests[0].get("action_schema"):
            raise ValueError("dataset action schemas do not match")
        if manifest.get("observation_schema") != manifests[0].get("observation_schema"):
            raise ValueError("dataset observation schemas do not match")
        if manifest.get("response_contract") != manifests[0].get("response_contract"):
            raise ValueError("dataset response contracts do not match")
        if manifest.get("optimizer_anchor_transition_contract") != manifests[0].get(
            "optimizer_anchor_transition_contract"
        ):
            raise ValueError("optimizer anchor transition contracts do not match")
    manifest = dict(manifests[0]) if manifests else {}
    source_contracts = {}
    source_profiles = set()
    for item in manifests:
        for sha256, payload in item.get("experiment_contracts", {}).items():
            existing = source_contracts.get(sha256)
            if existing is not None and existing != payload:
                raise ValueError(f"experiment contract collision for {sha256}")
            source_contracts[sha256] = payload
            source_profiles.add(str(payload.get("profile_id", "")))
    if len(source_profiles) != 1:
        raise ValueError(
            f"datasets must use one experiment profile, got {sorted(source_profiles)}"
        )
    manifest["experiment_contracts"] = source_contracts
    manifest["source_experiment_contract_sha256"] = sorted(source_contracts)
    manifest["source_experiment_profiles"] = sorted(source_profiles)
    dataset_names = sorted({item.get("dataset", "legacy_v1") for item in manifests})
    if dataset_names:
        manifest["dataset"] = "+".join(dataset_names)
        manifest["dataset_components"] = dataset_names
    manifest["transition_count"] = int(obs.shape[0])
    manifest["actor_supervision_count"] = int(np.count_nonzero(actor_supervision))
    source_actor_contracts = {
        item.get("actor_supervision_contract") for item in manifests
        if item.get("actor_supervision_contract")
    }
    if len(source_actor_contracts) > 1:
        raise ValueError("dataset actor supervision contracts do not match")
    if action_parameterization == "pstack_residual":
        manifest["actor_supervision_contract"] = (
            next(iter(source_actor_contracts))
            if source_actor_contracts
            else "pstack_anchor_gate_accepted_residual_v3"
        )
        if (
            manifest["actor_supervision_contract"]
            == LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT
            and not source_actor_contracts
        ):
            raise ValueError("long-horizon actor contract is missing from source manifests")
    else:
        manifest["actor_supervision_contract"] = (
            "all_behavior_plus_replayable_optimizer_leader_v1"
        )
    manifest["actor_sampling_contract"] = (
        "scenario_phase_x_zero_or_accepted_local_v1"
        if action_parameterization == "pstack_residual"
        else "scenario_phase_v1"
    )
    if action_parameterization == "pstack_residual" and manifests:
        trainable = pstack_residual_trainable_dimension_mask({
            "act": action[:1],
            "manifest_json": np.asarray(json.dumps(manifests[0])),
        })
        names = manifests[0].get("action_schema", {}).get("names", [])
        manifest["actor_frozen_action_dimensions"] = [
            str(name) for name, keep in zip(names, trainable) if not keep
        ]
    manifest["action_parameterization"] = str(action_parameterization)
    manifest["critic_action_contract"] = (
        PSTACK_RESIDUAL_CRITIC_ACTION_CONTRACT
        if action_parameterization == "pstack_residual"
        else "behavior_action_v1"
    )
    manifest["critic_terminal_contract"] = IQL_TERMINAL_CONTRACT
    manifest["critic_timeout_bootstrap_count"] = int(timeout_bootstrap_count)
    manifest["critic_reward_contract"] = IQL_REWARD_CONTRACT
    manifest["critic_wall_clock_reward_adjustment_count"] = int(
        wall_clock_reward_adjustment_count
    )
    manifest["source_files"] = files
    print(
        f"dataset {len(obs_parts)} files -> {obs.shape[0]} transitions, "
        f"actor_supervision={np.count_nonzero(actor_supervision)}, "
        f"obs_dim={obs.shape[1]}, act_dim={action.shape[1]}",
        flush=True,
    )
    return (
        obs, action, reward, next_obs, done, actor_supervision, actor_target,
        sampling_cells, actor_sampling_cells, manifest,
    )


def _make_actor(obs_dim: int, action_dim: int, manifest: dict):
    action_schema = manifest.get("action_schema", {})
    if action_schema.get("version") == ACTION_SCHEMA_VERSION:
        actor = UnifiedCoordinationActor(
            obs_dim,
            len(action_schema.get("signals", [])),
            len(action_schema.get("ramps", [])),
            len(action_schema.get("nonmerge_vsl_keys", [])),
            len(action_schema.get("certificate_ramps", [])),
        )
        if actor.action_dim != action_dim:
            raise ValueError(
                f"checkpoint action layout expects {actor.action_dim} values, dataset has {action_dim}"
            )
        return actor, "UnifiedCoordinationActor"
    if action_schema:
        raise ValueError(
            f"unsupported action schema version: {action_schema.get('version')!r}"
        )
    return Actor(obs_dim, action_dim), "Actor"


def _channel_keep_mask(action_dim: int, manifest: dict, probability: float) -> torch.Tensor:
    keep = np.ones(action_dim, dtype=np.float32)
    if probability <= 0.0 or not manifest.get("action_schema"):
        return torch.as_tensor(keep)
    schema = manifest["action_schema"]
    offset = 2
    block_layout = (
        (len(schema.get("signals", [])), 5),
        (len(schema.get("ramps", [])), 5),
        (len(schema.get("nonmerge_vsl_keys", [])), 2),
        (len(schema.get("certificate_ramps", [])), 1),
    )
    for count, width in block_layout:
        for _ in range(count):
            if np.random.random() < probability:
                keep[offset:offset + width] = 0.0
            offset += width
    return torch.as_tensor(keep)


def _checkpoint(actor, actor_class: str, manifest: dict, args, step: int, support_lo, support_hi):
    return {
        "format_version": RL_CHECKPOINT_FORMAT,
        "actor_class": actor_class,
        "actor_state_dict": actor.state_dict(),
        "git_commit": manifest.get("git_commit", "unknown"),
        "dataset": {
            "name": manifest.get("dataset", "legacy_v1"),
            "seed": manifest.get("seed"),
            "transition_count": manifest.get("transition_count"),
            "actor_supervision_count": manifest.get("actor_supervision_count"),
            "response_contract": manifest.get("response_contract", "legacy"),
            "experiment_contract_sha256": manifest.get(
                "source_experiment_contract_sha256", []
            ),
            "experiment_profiles": manifest.get(
                "source_experiment_profiles", []
            ),
        },
        "observation_schema": manifest.get("observation_schema", {}),
        "action_schema": manifest.get("action_schema", {}),
        "training": {
            "algorithm": "IQL",
            "step": int(step),
            "seed": int(args.seed),
            "gamma": float(args.gamma),
            "expectile": float(args.tau_exp),
            "beta": float(args.beta),
            "reward_scale": float(args.rscale),
            "support_weight": float(args.support_weight),
            "channel_dropout": float(args.channel_dropout),
            "observation_normalization": "obs_and_next_obs_std_floor_1e-3",
            "actor_supervision_contract": manifest.get("actor_supervision_contract"),
            "frozen_action_dimensions": manifest.get(
                "actor_frozen_action_dimensions", []
            ),
            "action_parameterization": manifest.get("action_parameterization", "absolute"),
            "critic_action_contract": manifest.get(
                "critic_action_contract", "behavior_action_v1"
            ),
            "critic_terminal_contract": manifest.get("critic_terminal_contract"),
            "critic_reward_contract": manifest.get("critic_reward_contract"),
            "critic_timeout_bootstrap_count": manifest.get(
                "critic_timeout_bootstrap_count", 0
            ),
            "critic_wall_clock_reward_adjustment_count": manifest.get(
                "critic_wall_clock_reward_adjustment_count", 0
            ),
            "balanced_cells": bool(args.balanced_cells),
        },
        "action_support_low": torch.as_tensor(support_lo),
        "action_support_high": torch.as_tensor(support_hi),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "full_action_v2" / "*.npz"))
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau-exp", "--tau_exp", dest="tau_exp", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--rscale", type=float, default=0.01)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--support-weight", type=float, default=0.1)
    parser.add_argument("--channel-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-parameterization",
        choices=("absolute", "pstack_residual"),
        default="absolute",
    )
    parser.add_argument("--balanced-cells", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "actor_full_iql.pt"))
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (
        obs, action, reward, next_obs, done, actor_supervision, actor_target,
        sampling_cells, actor_sampling_cells, manifest,
    ) = load_data(args.data, args.action_parameterization)
    obs_dim, action_dim = obs.shape[1], action.shape[1]
    state_support = np.concatenate((obs, next_obs), axis=0)
    obs_mu = state_support.mean(0)
    obs_sd = np.maximum(state_support.std(0), 1.0e-3)
    normalize = lambda value: torch.as_tensor((value - obs_mu) / obs_sd)
    obs_tensor = normalize(obs)
    action_tensor = torch.as_tensor(action)
    actor_target_tensor = torch.as_tensor(actor_target)
    raw_obs_tensor = torch.as_tensor(obs)
    reward_tensor = torch.as_tensor(reward * args.rscale)
    next_tensor = normalize(next_obs)
    done_tensor = torch.as_tensor(done)

    actor, actor_class = _make_actor(obs_dim, action_dim, manifest)
    actor.set_obs_norm(obs_mu, obs_sd)
    critic, critic_target = Critic(obs_dim, action_dim), Critic(obs_dim, action_dim)
    critic_target.load_state_dict(critic.state_dict())
    value_net = mlp(obs_dim, 128, 1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), 3.0e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), 3.0e-4)
    value_optimizer = torch.optim.Adam(value_net.parameters(), 3.0e-4)
    actor_indices = np.flatnonzero(actor_supervision)
    if actor_indices.size == 0:
        raise SystemExit("continuous actor has no eligible supervision transitions")
    actor_action = actor_target[actor_indices]
    if args.action_parameterization == "pstack_residual":
        support_lo = actor_action.min(0).astype(np.float32)
        support_hi = actor_action.max(0).astype(np.float32)
    else:
        support_lo = np.quantile(actor_action, 0.01, axis=0).astype(np.float32)
        support_hi = np.quantile(actor_action, 0.99, axis=0).astype(np.float32)
    support_lo_t = torch.as_tensor(support_lo)
    support_hi_t = torch.as_tensor(support_hi)
    count = obs_tensor.shape[0]
    critic_probabilities = (
        balanced_sampling_probabilities(sampling_cells)
        if args.balanced_cells else None
    )
    actor_probabilities = (
        balanced_sampling_probabilities(
            actor_sampling_cells, actor_supervision,
        )[actor_indices]
        if args.balanced_cells else None
    )
    if actor_probabilities is not None:
        actor_probabilities = actor_probabilities / actor_probabilities.sum()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        index = torch.as_tensor(np.random.choice(
            count,
            min(args.batch, count),
            replace=True,
            p=critic_probabilities,
        ))
        batch_obs = obs_tensor[index]
        batch_action = action_tensor[index]
        batch_reward = reward_tensor[index]
        batch_next = next_tensor[index]
        batch_done = done_tensor[index]

        with torch.no_grad():
            q1_target, q2_target = critic_target(batch_obs, batch_action)
            q_target = torch.min(q1_target, q2_target)
        value = value_net(batch_obs)
        difference = q_target - value
        expectile_weight = torch.where(difference < 0, 1.0 - args.tau_exp, args.tau_exp)
        value_loss = (expectile_weight * difference.pow(2)).mean()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        with torch.no_grad():
            td_target = batch_reward + args.gamma * (1.0 - batch_done) * value_net(batch_next)
        q1, q2 = critic(batch_obs, batch_action)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_index = torch.as_tensor(np.random.choice(
            actor_indices,
            min(args.batch, actor_indices.size),
            replace=True,
            p=actor_probabilities,
        ))
        actor_obs = obs_tensor[actor_index]
        actor_batch_action = action_tensor[actor_index]
        actor_batch_target = actor_target_tensor[actor_index]
        with torch.no_grad():
            q1_policy, q2_policy = critic(actor_obs, actor_batch_action)
            advantage = torch.min(q1_policy, q2_policy) - value_net(actor_obs)
            awr_weight = torch.clamp(torch.exp(args.beta * advantage), max=100.0)
        actor_mean, _ = actor(raw_obs_tensor[actor_index])
        predicted_action = torch.tanh(actor_mean)
        channel_keep = _channel_keep_mask(action_dim, manifest, args.channel_dropout)
        squared_error = (
            (predicted_action - actor_batch_target) * channel_keep
        ).pow(2).mean(-1, keepdim=True)
        support_distance = (
            torch.relu(support_lo_t - predicted_action).pow(2)
            + torch.relu(predicted_action - support_hi_t).pow(2)
        ).mean()
        actor_loss = (awr_weight * squared_error).mean() + args.support_weight * support_distance
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        with torch.no_grad():
            for parameter, target_parameter in zip(critic.parameters(), critic_target.parameters()):
                target_parameter.data.mul_(1.0 - args.polyak).add_(args.polyak * parameter.data)

        if step == 1 or step % 2000 == 0 or step == args.steps:
            violation = float(((predicted_action < support_lo_t) | (predicted_action > support_hi_t)).float().mean())
            print(
                f"step={step:6d} V={value_loss.item():.4f} Q={critic_loss.item():.4f} "
                f"pi={actor_loss.item():.4f} support_out={violation:.3f} "
                f"adv={advantage.min().item():+.3f}/{advantage.mean().item():+.3f}/{advantage.max().item():+.3f}",
                flush=True,
            )
            torch.save(
                _checkpoint(actor, actor_class, manifest, args, step, support_lo, support_hi),
                output,
            )
    print(f"saved actor(IQL) -> {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
