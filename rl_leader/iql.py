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
from src.controllers.coordination import ACTION_SCHEMA_VERSION


def _manifest_from_npz(dataset) -> dict:
    if "manifest_json" not in dataset.files:
        return {}
    value = dataset["manifest_json"]
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return json.loads(str(value))


def load_data(pattern):
    obs_parts, action_parts, reward_parts, next_parts, done_parts = [], [], [], [], []
    manifests = []
    patterns = [value.strip() for value in str(pattern).split(",") if value.strip()]
    files = sorted({path for value in patterns for path in glob.glob(value)})
    if not files:
        raise SystemExit(f"no dataset files matched: {pattern}")
    for path in files:
        try:
            dataset = np.load(path, allow_pickle=False)
            if int(dataset["obs"].shape[0]) == 0:
                continue
            obs_parts.append(dataset["obs"])
            action_parts.append(dataset["act"])
            reward_parts.append(dataset["rew"])
            next_parts.append(dataset["next_obs"])
            done_parts.append(dataset["done"])
            manifest = _manifest_from_npz(dataset)
            if manifest:
                if (
                    "optimizer_anchor" in manifest.get("modes", [])
                    and manifest.get("optimizer_anchor_transition_contract")
                    != "rl_adapter_replay_v1"
                ):
                    raise ValueError(
                        "optimizer_anchor transitions bypassed the deployable RL adapter"
                    )
                manifests.append(manifest)
        except Exception as exc:
            print(f"skip {path}: {exc}", flush=True)
    if not obs_parts:
        raise SystemExit("all matched dataset files were empty or invalid")
    obs = np.concatenate(obs_parts).astype(np.float32)
    action = np.concatenate(action_parts).astype(np.float32)
    reward = np.concatenate(reward_parts).astype(np.float32).reshape(-1, 1)
    next_obs = np.concatenate(next_parts).astype(np.float32)
    done = np.concatenate(done_parts).astype(np.float32).reshape(-1, 1)
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
    dataset_names = sorted({item.get("dataset", "legacy_v1") for item in manifests})
    if dataset_names:
        manifest["dataset"] = "+".join(dataset_names)
        manifest["dataset_components"] = dataset_names
    manifest["transition_count"] = int(obs.shape[0])
    manifest["source_files"] = files
    print(
        f"dataset {len(obs_parts)} files -> {obs.shape[0]} transitions, "
        f"obs_dim={obs.shape[1]}, act_dim={action.shape[1]}",
        flush=True,
    )
    return obs, action, reward, next_obs, done, manifest


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
        "format_version": "rl_coordination_checkpoint_v2",
        "actor_class": actor_class,
        "actor_state_dict": actor.state_dict(),
        "git_commit": manifest.get("git_commit", "unknown"),
        "dataset": {
            "name": manifest.get("dataset", "legacy_v1"),
            "seed": manifest.get("seed"),
            "transition_count": manifest.get("transition_count"),
            "response_contract": manifest.get("response_contract", "legacy"),
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
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "actor_full_iql.pt"))
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs, action, reward, next_obs, done, manifest = load_data(args.data)
    obs_dim, action_dim = obs.shape[1], action.shape[1]
    state_support = np.concatenate((obs, next_obs), axis=0)
    obs_mu = state_support.mean(0)
    obs_sd = np.maximum(state_support.std(0), 1.0e-3)
    normalize = lambda value: torch.as_tensor((value - obs_mu) / obs_sd)
    obs_tensor = normalize(obs)
    action_tensor = torch.as_tensor(action)
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
    support_lo = np.quantile(action, 0.01, axis=0).astype(np.float32)
    support_hi = np.quantile(action, 0.99, axis=0).astype(np.float32)
    support_lo_t = torch.as_tensor(support_lo)
    support_hi_t = torch.as_tensor(support_hi)
    count = obs_tensor.shape[0]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        index = torch.as_tensor(np.random.randint(0, count, min(args.batch, count)))
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

        with torch.no_grad():
            q1_policy, q2_policy = critic(batch_obs, batch_action)
            advantage = torch.min(q1_policy, q2_policy) - value_net(batch_obs)
            awr_weight = torch.clamp(torch.exp(args.beta * advantage), max=100.0)
        actor_mean, _ = actor(raw_obs_tensor[index])
        predicted_action = torch.tanh(actor_mean)
        channel_keep = _channel_keep_mask(action_dim, manifest, args.channel_dropout)
        squared_error = ((predicted_action - batch_action) * channel_keep).pow(2).mean(-1, keepdim=True)
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
