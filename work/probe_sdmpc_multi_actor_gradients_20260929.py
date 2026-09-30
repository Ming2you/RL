"""Compare saved actor saturation and critic action gradients across both rounds."""
import argparse
from pathlib import Path
from probe_sdmpc_multi_values_20260929 import torch, TD3, SCENARIOS, read, save, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if read(args.pilot / "completion.json")["status"] != "completed":
        raise ValueError("Pilot incomplete")
    rows, hashes = [], {}
    for round_index in (0, 1):
        folder = args.pilot / f"train_round{round_index}"
        path = folder / "model_final.pt"
        hashes[str(round_index)] = file_hash(path)
        if hashes[str(round_index)] != read(folder / "completion.json")["model_sha256"]:
            raise ValueError("Model identity mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)["learner"]
        learner = TD3(state["observation_dim"], state["spec"]["seed"], state["spec"]["hidden"])
        learner.load_state_dict(state)
        for scenario in SCENARIOS:
            obs = state["replay"][scenario]["observations"]
            action = learner.actor(obs).detach().requires_grad_(True)
            q = learner.critics[0](torch.cat((obs, action), dim=1))
            gradient = torch.autograd.grad(q.sum(), action)[0]
            slope = 1. - action.detach().square()
            preactivation_gradient = gradient * slope
            assert bool(torch.isfinite(gradient).all()) and bool((slope >= 0).all())
            rows.append(dict(round=round_index, scenario=scenario, states=len(obs),
                actor_mean=action.detach().mean(dim=0).tolist(),
                dq_da_mean=gradient.mean(dim=0).tolist(),
                dq_da_positive_fraction=(gradient > 0).float().mean(dim=0).tolist(),
                dq_da_negative_fraction=(gradient < 0).float().mean(dim=0).tolist(),
                tanh_slope_mean=slope.mean(dim=0).tolist(),
                tanh_slope_min=slope.min(dim=0).values.tolist(),
                tanh_slope_max=slope.max(dim=0).values.tolist(),
                dq_d_preactivation_mean=preactivation_gradient.mean(dim=0).tolist()))
        if file_hash(path) != hashes[str(round_index)]:
            raise ValueError("Model changed during probe")
    save(args.output, dict(model_sha256=hashes, probe_sha256=file_hash(__file__),
        imported_probe_sha256=file_hash(Path(__file__).with_name("probe_sdmpc_multi_values_20260929.py")),
        rows=rows, scope="Read-only derivative of Q1, the actual TD3 actor objective, at stored replay states.",
        caveat="Local tanh attenuation is measured; optimizer history and other network Jacobians also affect recovery. No retraining or causal rescue demonstrated."))
    print("MULTI_ACTOR_GRADIENT_PROBE_PASS")


if __name__ == "__main__":
    main()
