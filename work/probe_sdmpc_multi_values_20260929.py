"""Read-only action-value surface probe; no plant rollouts or optimizer steps."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(ROOT / ".deps-budget"), str(ROOT / "work/sdmpc_rl_multi_20260929")]
import torch
from budget_runtime import DEFAULT_SNAPSHOT, read, save
from run_budget import file_hash, verify_pins
from td3 import TD3, SCENARIOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pilot = args.pilot.resolve()
    completion = read(pilot / "completion.json")
    if completion["status"] != "completed":
        raise ValueError("Only completed frozen pilots may be probed")
    verify_pins(DEFAULT_SNAPSHOT, read(pilot / "plan.json")["source_pins"])
    path = pilot / "train_round1/model_final.pt"
    before = file_hash(path)
    if before != completion["shared_model_sha256"]:
        raise ValueError("Final model changed")
    model = torch.load(path, map_location="cpu", weights_only=False)
    state = model["learner"]
    learner = TD3(state["observation_dim"], state["spec"]["seed"], state["spec"]["hidden"])
    learner.load_state_dict(state)
    grid = torch.cartesian_prod(torch.tensor([-1., 0., 1.]), torch.tensor([-1., 0., 1.]))
    rows = []
    with torch.no_grad():
        for scenario in SCENARIOS:
            replay = state["replay"][scenario]
            obs, next_obs = replay["observations"], replay["next_observations"]
            action, done = replay["actions"], replay["terminated"]
            assert len(obs) == 150 and int(done.sum()) == 2
            policy = learner.actor(obs)
            inputs = torch.cat((obs[:, None, :].expand(-1, 9, -1), grid[None].expand(len(obs), -1, -1)), dim=2)
            next_inputs = torch.cat((next_obs, learner.actor_target(next_obs)), dim=1)
            target_q = torch.minimum(*(critic(next_inputs).squeeze(1) for critic in learner.critic_targets))
            targets = replay["rewards"] + torch.where(done, 0., target_q)
            heads = []
            for critic in learner.critics:
                values = critic(inputs.reshape(-1, inputs.shape[-1])).reshape(len(obs), 9)
                zero_q = critic(torch.cat((obs, torch.zeros_like(action)), dim=1)).squeeze(1)
                torch.testing.assert_close(values[:, 4], zero_q)
                chosen_q = critic(torch.cat((obs, policy), dim=1)).squeeze(1)
                behavior_q = critic(torch.cat((obs, action), dim=1)).squeeze(1)
                preferred = values.argmax(dim=1).tolist()
                heads.append(dict(
                    preferred_grid_action_counts={str(grid[i].tolist()): count for i, count in Counter(preferred).items()},
                    mean_grid_q_range=float((values.max(dim=1).values-values.min(dim=1).values).mean()),
                    actor_minus_zero_q_mean=float((chosen_q-zero_q).mean()),
                    actor_above_zero_fraction=float((chosen_q > zero_q).float().mean()),
                    unsmoothed_target_td_rmse=float((behavior_q-targets).square().mean().sqrt()),
                    terminal_prediction=behavior_q[done].tolist(), terminal_reward=replay["rewards"][done].tolist()))
            rows.append(dict(scenario=scenario, transitions=len(obs), critics=heads))
    if file_hash(path) != before or learner.updates != state["updates"]:
        raise ValueError("Probe mutated its model")
    save(args.output, dict(model_sha256=before, probe_sha256=file_hash(__file__), rows=rows,
        scope="Stored replay states and learned Q surface only; no counterfactual traffic outcomes.",
        td_caveat="Diagnostic target omits TD3 target smoothing noise; not a training loss replay.",
        ranking_caveat="A learned preference is not proof of the true ordering of alternative actions."))
    print("MULTI_VALUE_SURFACE_PROBE_PASS", before)


if __name__ == "__main__":
    main()
