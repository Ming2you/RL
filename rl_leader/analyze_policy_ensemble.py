"""Rank action blocks for active collection using seed disagreement and support."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from rl_leader.nets import UnifiedCoordinationActor


def _load_actor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    action_schema = checkpoint["action_schema"]
    observation_schema = checkpoint["observation_schema"]
    actor = UnifiedCoordinationActor(
        observation_schema["dimension"],
        len(action_schema["signals"]),
        len(action_schema["ramps"]),
        len(action_schema.get("nonmerge_vsl_keys", [])),
        len(action_schema.get("certificate_ramps", [])),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", default="data/full_action_v2/*.npz")
    parser.add_argument("--out", default="results/full_action_ensemble_audit.json")
    args = parser.parse_args(argv)
    checkpoint_paths = []
    for pattern in args.checkpoints:
        checkpoint_paths.extend(sorted(glob.glob(pattern)) or [pattern])
    if len(checkpoint_paths) < 2:
        raise ValueError("at least two checkpoints are required for disagreement")
    observations = []
    for path in sorted(glob.glob(args.data)):
        observations.append(np.load(path, allow_pickle=False)["obs"])
    if not observations:
        raise ValueError("no dataset observations found")
    obs = np.concatenate(observations).astype(np.float32)
    actors = []
    checkpoints = []
    for path in checkpoint_paths:
        actor, checkpoint = _load_actor(path)
        actors.append(actor)
        checkpoints.append(checkpoint)
    with torch.no_grad():
        tensor = torch.as_tensor(obs)
        actions = np.stack([torch.tanh(actor(tensor)[0]).cpu().numpy() for actor in actors])
    support_low_all = np.stack([
        checkpoint["action_support_low"].cpu().numpy() for checkpoint in checkpoints
    ])
    support_high_all = np.stack([
        checkpoint["action_support_high"].cpu().numpy() for checkpoint in checkpoints
    ])
    per_seed_support_out = np.mean(
        (actions < support_low_all[:, None, :]) | (actions > support_high_all[:, None, :]),
        axis=(1, 2),
    )
    per_seed_control_support_out = np.mean(
        (actions[:, :, 2:] < support_low_all[:, None, 2:])
        | (actions[:, :, 2:] > support_high_all[:, None, 2:]),
        axis=(1, 2),
    )
    per_seed_dead_policy_dimensions = np.mean(actions.std(axis=1) < 1.0e-3, axis=1)
    schema = checkpoints[0]["action_schema"]
    blocks = []
    layout = []
    offset = 2
    for family, owners in (("urban", schema["signals"]), ("freeway", schema["ramps"])):
        for owner in owners:
            layout.append((family, owner, slice(offset, offset + 5)))
            offset += 5
    for owner in schema.get("nonmerge_vsl_keys", []):
        layout.append(("vsl", owner, slice(offset, offset + 2)))
        offset += 2
    for owner in schema.get("certificate_ramps", []):
        layout.append(("certificate", owner, slice(offset, offset + 1)))
        offset += 1
    for family, owner, block_slice in layout:
        values = actions[:, :, block_slice]
        disagreement = values.std(axis=0)
        support_low = np.stack([
            checkpoint["action_support_low"].cpu().numpy()[block_slice]
            for checkpoint in checkpoints
        ]).min(axis=0)
        support_high = np.stack([
            checkpoint["action_support_high"].cpu().numpy()[block_slice]
            for checkpoint in checkpoints
        ]).max(axis=0)
        ensemble_mean = values.mean(axis=0)
        support_out = np.mean((ensemble_mean < support_low) | (ensemble_mean > support_high))
        policy_action_std = ensemble_mean.std(axis=0)
        blocks.append({
            "family": family,
            "owner": owner,
            "disagreement_score": float(disagreement.mean()),
            "support_out_fraction": float(support_out),
            "mean_abs_action": float(np.abs(ensemble_mean).mean()),
            "policy_action_std": policy_action_std.tolist(),
            "dead_policy_dimension_fraction": float(np.mean(policy_action_std < 1.0e-3)),
        })
    blocks.sort(key=lambda item: (item["support_out_fraction"], item["disagreement_score"]), reverse=True)
    report = {
        "checkpoints": checkpoint_paths,
        "observations": int(obs.shape[0]),
        "recommended_exploration_blocks": blocks,
        "per_seed_support_out_fraction": {
            path: float(value) for path, value in zip(checkpoint_paths, per_seed_support_out)
        },
        "per_seed_control_support_out_fraction": {
            path: float(value)
            for path, value in zip(checkpoint_paths, per_seed_control_support_out)
        },
        "per_seed_dead_policy_dimension_fraction": {
            path: float(value)
            for path, value in zip(checkpoint_paths, per_seed_dead_policy_dimensions)
        },
        "global_seed_action_disagreement": float(actions.std(axis=0).mean()),
        "global_support_out_fraction": float(np.mean([
            item["support_out_fraction"] for item in blocks
        ])),
        "global_dead_policy_dimension_fraction": float(np.mean([
            item["dead_policy_dimension_fraction"] for item in blocks
        ])),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "top exploration blocks: "
        + ", ".join(
            f"{item['owner']}(out={item['support_out_fraction']:.2f},std={item['disagreement_score']:.3f})"
            for item in blocks[:5]
        ),
        flush=True,
    )
    print(f"saved ensemble audit -> {output}", flush=True)


if __name__ == "__main__":
    main()
