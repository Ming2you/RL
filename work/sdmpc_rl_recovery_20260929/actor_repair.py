"""Three controlled, balanced actor-only fits against one frozen critic."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
from common import (torch, TD3, SCENARIOS, BASE_HASH, load_base, sources, stopped,
                    save, read, runtime_versions, exclusive_run, checkpoint_save)

VARIANTS = ("continue", "reset_optimizer", "reset_head")
SOURCE_NAMES = ("common.py", "actor_repair.py")
FORMAT = "sdmpc-recovery-actor-v1"
SEED = 7100
UPDATES = 375


def state_batch(replay, generator):
    return torch.cat([data["observations"][torch.randint(len(data["observations"]), (8,), generator=generator)]
                      for s in SCENARIOS for data in (replay[s],)])


def prepare(base, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown actor treatment")
    learner = deepcopy(base)
    learner.critics.requires_grad_(False)
    if variant == "reset_head":
        torch.nn.init.zeros_(learner.actor[-2].weight)
        torch.nn.init.zeros_(learner.actor[-2].bias)
    learner.actor.requires_grad_(False)
    learner.actor[-2].requires_grad_(True)
    if variant != "continue":
        learner.actor_optimizer = torch.optim.Adam(learner.actor.parameters(), lr=learner.learning_rate)
    return learner


def actor_step(learner, observations):
    action = learner.actor(observations)
    loss = -learner.critics[0](torch.cat((observations, action), dim=1)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite actor objective")
    learner.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    learner.actor_optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def measure(learner, replay):
    rows = {}
    grid = torch.cartesian_prod(torch.tensor([-1., 0., 1.]), torch.tensor([-1., 0., 1.]))
    for scenario in SCENARIOS:
        obs = replay[scenario]["observations"]
        action = learner.actor(obs)
        q = learner.critics[0](torch.cat((obs, action), dim=1)).ravel()
        inputs = torch.cat((obs[:, None].expand(-1, 9, -1), grid[None].expand(len(obs), -1, -1)), dim=2)
        best = learner.critics[0](inputs.reshape(-1, inputs.shape[-1])).reshape(len(obs), 9).max(dim=1).values
        rows[scenario] = dict(q_mean=float(q.mean()), mean_grid_gap=float((best-q).mean()),
            action_mean=action.mean(dim=0).tolist(),
            saturation=(action.abs() >= .95).float().mean(dim=0).tolist(),
            nuf_positive_fraction=float((action[:, 1] > 0).float().mean()),
            tanh_slope_mean=(1-action.square()).mean(dim=0).tolist())
    return dict(equal_scenario_q=sum(r["q_mean"] for r in rows.values())/5,
                mean_grid_gap=sum(r["mean_grid_gap"] for r in rows.values())/5, scenarios=rows)


def select_variant(initial, completed):
    # Select before seeing any physical evaluation; Q fit is not traffic acceptance.
    winner = max(VARIANTS, key=lambda name: completed[name]["equal_scenario_q"])
    result = completed[winner]
    admitted = (result["equal_scenario_q"] > initial["equal_scenario_q"] + 1e-4 and
                result["mean_grid_gap"] < .75 * initial["mean_grid_gap"])
    return winner if admitted else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with exclusive_run(output):
        if stopped(output):
            save(output / "partial.json", dict(status="paused", completed_update=0))
            return
        model, base = load_base()
        replay = model["learner"]["replay"]
        source = sources(*SOURCE_NAMES)
        settings = dict(format=FORMAT, base_sha256=BASE_HASH, variants=list(VARIANTS),
            source_sha256=source, runtime=runtime_versions(), seed=SEED,
            actor_updates=UPDATES, batch_size=40, samples_per_scenario=8,
            trainable_parameters="actor_output_head_only",
            objective="Q1_only_fixed_final_critic", new_plant_transitions=0,
            scope="actor_repair_ablation_not_joint_TD3_training")
        save(output / "settings.json", settings)
        baseline = measure(base, replay)
        completed, records = {}, {}
        for variant in VARIANTS:
            learner = prepare(base, variant)
            generator = torch.Generator().manual_seed(SEED)
            metrics = [dict(update=0, **measure(learner, replay))]
            started = time.perf_counter()
            for update in range(1, UPDATES + 1):
                if stopped(output):
                    save(output / "partial.json", dict(status="paused", variant=variant,
                        completed_update=update-1, metrics=metrics, completed=records))
                    return
                loss = actor_step(learner, state_batch(replay, generator))
                if update % 25 == 0:
                    metrics.append(dict(update=update, actor_loss=loss, **measure(learner, replay)))
            if sources(*SOURCE_NAMES) != source:
                raise ValueError("Actor experiment source changed")
            for name, value in learner.critics.state_dict().items():
                torch.testing.assert_close(value, base.critics.state_dict()[name], rtol=0, atol=0)
            for name, value in learner.actor.state_dict().items():
                if not name.startswith("4."):
                    torch.testing.assert_close(value, base.actor.state_dict()[name], rtol=0, atol=0)
            completed[variant] = metrics[-1]
            payload = dict(format=FORMAT, settings=settings, variant=variant,
                base_contract=model["contract"], base_training_profiles=model["training_profiles"],
                actor_state=learner.actor.state_dict(), actor_optimizer=learner.actor_optimizer.state_dict(),
                sampled_per_scenario=dict.fromkeys(SCENARIOS, UPDATES*8),
                metrics=metrics, wall_seconds=time.perf_counter()-started)
            checkpoint_save(torch, output / f"{variant}.pt", payload)
            records[variant] = {k: v for k, v in payload.items()
                                if k not in ("actor_state", "actor_optimizer", "base_contract", "base_training_profiles")}
            save(output / f"{variant}.json", records[variant])
        selected = select_variant(baseline, completed)
        from common import file_hash
        save(output / "completion.json", dict(status="completed", settings=settings,
            baseline=baseline, final_metrics=completed, selected=selected,
            exports={name: file_hash(output / f"{name}.pt") for name in VARIANTS},
            selection_rule="greatest_equal_scenario_Q1_then_variant_order; Q_gain>1e-4_and_grid_gap_reduced_25pct",
            traffic_performance_claim=False))
        print("ACTOR_REPAIR_COMPLETE", selected, flush=True)


if __name__ == "__main__":
    main()
