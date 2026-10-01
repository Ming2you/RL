"""Resumable canonical evaluation of a transparent actor-only repair export."""
import argparse
import copy
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from common import (torch, SCENARIOS, BASE_HASH, DEFAULT_SNAPSHOT, load_base,
                    sources, stopped, read, save, plain, file_hash, runtime_versions,
                    exclusive_run, checkpoint_save, verify_pins)
from actor_repair import FORMAT as ACTOR_FORMAT, SOURCE_NAMES, VARIANTS, select_variant
from budget_runtime import boot, digest
from run_budget import observation_schema, policy_contract, scenario_manifest
from compare_runs import validate_episode

FORMAT = "sdmpc-actor-repair-eval-v1"
EVAL_SOURCES = (*SOURCE_NAMES, "evaluate.py")


def load_actor(path, base_model, learner):
    completion = read(path.parent / "completion.json")
    if completion["status"] != "completed":
        raise ValueError("Actor fit incomplete")
    selected = select_variant(completion["baseline"], completion["final_metrics"])
    if selected is None or completion["selected"] != selected or path.name != selected + ".pt":
        raise ValueError("Not the pre-evaluation selected actor")
    if file_hash(path) != completion["exports"][selected]:
        raise ValueError("Actor export hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    settings = payload["settings"]
    if (payload["format"] != ACTOR_FORMAT or payload["variant"] != selected or
            settings != completion["settings"] or settings["base_sha256"] != BASE_HASH or
            settings["source_sha256"] != sources(*SOURCE_NAMES) or
            settings["runtime"] != runtime_versions() or settings["actor_updates"] != 375 or
            settings["seed"] != 7100 or settings["batch_size"] != 40 or settings["samples_per_scenario"] != 8 or
            settings["variants"] != list(VARIANTS) or settings["new_plant_transitions"] != 0 or
            settings["objective"] != "Q1_only_fixed_final_critic" or
            settings["scope"] != "actor_repair_ablation_not_joint_TD3_training" or
            settings["trainable_parameters"] != "actor_output_head_only" or
            payload["base_contract"] != base_model["contract"] or
            payload["base_training_profiles"] != base_model["training_profiles"] or
            payload["sampled_per_scenario"] != dict.fromkeys(SCENARIOS, 3000) or
            payload["metrics"][-1] != completion["final_metrics"][selected]):
        raise ValueError("Actor repair provenance differs")
    if any(not torch.isfinite(tensor).all() for tensor in payload["actor_state"].values()):
        raise ValueError("Nonfinite actor weights")
    learner.actor.load_state_dict(payload["actor_state"], strict=True)
    return payload


def validate_checkpoint(checkpoint, settings):
    if checkpoint["format"] != FORMAT or checkpoint["settings"] != settings:
        raise ValueError("Resume identity changed")
    k = checkpoint["environment"]["k"]
    if (type(k) is not int or not 5 <= k <= 80 or len(checkpoint["trace"]) != k-5 or
            len(checkpoint["observations"]) != k-5 or
            checkpoint["environment"]["profile_hash"] != settings["profile_sha256"][0]):
        raise ValueError("Resume boundaries changed")


def summarize(env, trace, scenario):
    import numpy as np
    times = [row["decision_wall_seconds"] for row in trace]
    return dict(episode=0, scenario=scenario, training_seed=None, profile_sha256=env.profile_hash,
        ttt=env.sim.total_ttt, freeway_ttt=trace[-1]["freeway_ttt"], urban_ttt=trace[-1]["urban_ttt"],
        warmup_ttt=env.warmup_ttt, full_run=True, control_steps=len(trace),
        simulation_seconds=env.sim.state.time_sec, terminal_inventory=trace[-1]["inventory"],
        decision_wall_seconds=sum(times), decision_cpu_seconds=sum(r["decision_cpu_seconds"] for r in trace),
        decision_p50=float(np.median(times)), decision_p95=float(np.percentile(times, 95)),
        decision_max=max(times), fallback_count=sum(r["selection_source"] == "reference_fallback" for r in trace),
        pfo_calls=sum(r["pfo_calls"] for r in trace), pfo_recovery_count=sum(r["reference_recovery"] for r in trace),
        lower_candidate_solves=sum(r["lower_candidate_count"] for r in trace),
        converged_count=sum(r["converged"] for r in trace), controller_acceptance=False,
        evaluation=True, exploration=False, base_learner_updates=750, actor_repair_updates=375)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-mask", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=75)
    args = parser.parse_args()
    if not 1 <= args.stop_after <= 75:
        parser.error("stop-after must be 1..75; partial runs never emit completion")
    output, actor_path = args.output.resolve(), args.actor.resolve()
    with exclusive_run(output):
        if (output / "completion.json").exists():
            raise ValueError("Already completed, refusing duplicate evaluation")
        if (output / "checkpoint.pt").exists() != args.resume:
            raise ValueError("Checkpoint exists iff --resume is supplied")
        if stopped(output):
            save(output / "status.json", dict(status="paused", before_environment_initialization=True))
            return
        rt = boot(DEFAULT_SNAPSHOT, args.cpu_mask)
        from budget_env import BudgetEnv
        model, learner = load_base()
        load_actor(actor_path, model, learner)
        env = BudgetEnv(rt, scenario=args.scenario, guard_mode="physical")
        settings = dict(format=FORMAT,
            run_id=read(output / "settings.json")["run_id"] if (output / "settings.json").exists() else uuid.uuid4().hex,
            mode="rl", scenario=args.scenario, seeds=[None], profile_sha256=[env.profile_hash],
            base_model_sha256=BASE_HASH, model_sha256=file_hash(actor_path), actor_path=str(actor_path),
            actor_fit_completion_sha256=file_hash(actor_path.parent / "completion.json"),
            source_pins=model["contract"]["source_pins"], eval_source_sha256=sources(*EVAL_SOURCES),
            runtime_versions=runtime_versions(), cpu_mask=args.cpu_mask, reward_divisor=100., gamma=1.,
            total_seconds=14400, warmup_steps=5, controlled_steps=75, guard_mode="physical",
            environment_contract=dict(config_sha256=digest(rt["rc"].to_plain_dict(rt["cfg"])),
                options_sha256=digest(rt["rc"].to_plain_dict(rt["options"])), scenarios=scenario_manifest(rt),
                coordinator={k: v for k, v in env.contract().items()
                             if k not in ("cfg", "options", "observation", "source_snapshot")}),
            scope="actor_only_repair_ablation_not_joint_TD3_or_critic_recalibration")
        if (output / "settings.json").exists() and read(output / "settings.json") != settings:
            raise ValueError("Existing evaluation settings differ")
        save(output / "settings.json", settings)
        save(output / "process.json", dict(pid=os.getpid(), command=sys.argv, started=time.time()))
        trace, observations, elapsed = [], [], 0.
        started = time.perf_counter()
        try:
            if args.resume:
                checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
                validate_checkpoint(checkpoint, settings)
                obs = env.restore(checkpoint["environment"])
                trace, observations, elapsed = checkpoint["trace"], checkpoint["observations"], checkpoint["elapsed"]
            else:
                obs = env.reset()
            if policy_contract(env, settings) != model["contract"]:
                raise ValueError("Physical observation/policy contract changed")
            save(output / "observation_schema.json", observation_schema(env))

            def checkpoint_now():
                checkpoint_save(torch, output / "checkpoint.pt", dict(format=FORMAT, settings=settings,
                    environment=env.checkpoint(), trace=trace, observations=observations,
                    elapsed=elapsed+time.perf_counter()-started))

            checkpoint_now()
            while env.k < 80:
                if stopped(output) or len(trace) >= args.stop_after:
                    save(output / "status.json", dict(status="paused", control_steps=len(trace)))
                    return
                verify_pins(DEFAULT_SNAPSHOT, settings["source_pins"])
                if sources(*EVAL_SOURCES) != settings["eval_source_sha256"] or file_hash(actor_path) != settings["model_sha256"]:
                    raise ValueError("Frozen evaluator or actor changed")
                tick, cpu = time.perf_counter(), time.process_time()
                with torch.no_grad():
                    action = learner.act(obs)
                    inputs = torch.cat((torch.from_numpy(obs), torch.from_numpy(action))).unsqueeze(0)
                    q = [float(c(inputs).item()) for c in learner.critics]
                wall, cpu_seconds = time.perf_counter()-tick, time.process_time()-cpu
                observations.append(obs.copy())
                next_obs, reward, terminal, row = env.step(action, "rl", wall, actor_cpu_seconds=cpu_seconds)
                row.update(reward=reward, episode=0, scenario=args.scenario, profile_sha256=env.profile_hash,
                           policy_q=q, learning=[], learning_wall_seconds=0.)
                trace.append(plain(row))
                obs = next_obs
                checkpoint_now()
                save(output / "status.json", dict(status="running", control_steps=len(trace),
                    simulation_seconds=env.sim.state.time_sec, ttt=env.sim.total_ttt))
                print(args.scenario, f"{len(trace)}/75", env.sim.total_ttt, flush=True)
            summary = summarize(env, trace, args.scenario)
            validate_episode(summary, trace, settings, 0)
            if sources(*EVAL_SOURCES) != settings["eval_source_sha256"] or file_hash(actor_path) != settings["model_sha256"]:
                raise ValueError("Evaluator or actor changed at completion")
            verify_pins(DEFAULT_SNAPSHOT, settings["source_pins"])
            save(output / "episode_00_summary.json", summary)
            save(output / "episode_00_trace.json", trace)
            checkpoint_save(torch, output / "evaluation_observations.pt", dict(settings=settings, observations=observations))
            save(output / "completion.json", dict(format=FORMAT, status="completed", settings=settings,
                summary=summary, elapsed_wall_seconds=elapsed+time.perf_counter()-started,
                observations_sha256=file_hash(output / "evaluation_observations.pt"),
                performance_claim=False))
            save(output / "status.json", dict(status="completed"))
        except BaseException:
            save(output / "failure.json", dict(traceback=traceback.format_exc(), step=env.k))
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
