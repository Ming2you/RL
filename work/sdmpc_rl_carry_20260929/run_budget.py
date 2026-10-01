"""Bounded, resumable sequential collection or full-run budget-policy evaluation."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from budget_runtime import boot, DEFAULT_SNAPSHOT, HERE, REPO, read, save, digest, plain

RUN_FORMAT = "sdmpc-carry-run-v1"
POLICY_FORMAT = "sdmpc-carry-policy-v1"
CHECKPOINT_FORMAT = "sdmpc-carry-checkpoint-v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def exclusive_run(output):
    import msvcrt
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def pins(root):
    from freeze_runtime import verify_snapshot
    manifest = verify_snapshot(root.parent)
    sources = {p.name: file_hash(p) for p in HERE.glob("*.py") if not p.name.startswith("test_")}
    return dict(snapshot_manifest_sha256=file_hash(root.parent / "manifest.json"),
                frozen_file_count=manifest["file_count"], implementation=sources)


def verify_pins(root, expected):
    if pins(root) != expected:
        raise RuntimeError("Experiment source changed; new version required")


def checkpoint_save(torch, path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def runtime_versions():
    return dict(python=sys.version, **{name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "torch", "PyYAML")})


def write_run_settings(output, settings, resume):
    if resume:
        if read(output / "settings.json") != settings:
            raise RuntimeError("Resume configuration or runtime differs")
        if read(output / "runtime_versions.json") != settings["runtime_versions"]:
            raise RuntimeError("Resume runtime record differs")
    else:
        save(output / "settings.json", settings)
        save(output / "runtime_versions.json", settings["runtime_versions"])


def observation_schema(env):
    return dict(names=list(env.observer.names), scale="fixed_physical_constants_no_eval_fitting",
                delay_bins=env.observer.delay + 1, normalization=env.observer.contract(),
                Markov_sufficiency="not_proven_recovery_PFO_memory_is_checkpointed_but_not_fully_observed")


def policy_contract(env, settings):
    return dict(source_pins=settings["source_pins"], runtime_versions=settings["runtime_versions"],
                environment_contract=settings["environment_contract"],
                observation_schema=observation_schema(env),
                reward_divisor=settings["reward_divisor"], gamma=settings["gamma"])


def training_profiles(completed):
    return [dict(seed=row["training_seed"], profile_sha256=row["profile_sha256"])
            for row in completed]


def export_policy(learner, env, settings, completed):
    return dict(format=POLICY_FORMAT, learner=learner.state_dict(),
                observation_names=list(env.observer.names), contract=policy_contract(env, settings),
                training_run_id=settings["run_id"], training_profiles=training_profiles(completed))


def load_policy(learner, model, env, settings):
    if (model.get("format") != POLICY_FORMAT or
            model.get("contract") != policy_contract(env, settings) or
            model.get("observation_names") != list(env.observer.names)):
        raise RuntimeError("Policy source, runtime, observation, or reward contract differs")
    if (not model.get("training_run_id") or not model.get("training_profiles") or
            any(type(p.get("seed")) is not int or not p.get("profile_sha256")
                for p in model["training_profiles"])):
        raise RuntimeError("Policy training provenance missing")
    learner.load_state_dict(model["learner"])


def validate_checkpoint(checkpoint, settings):
    if (checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("settings") != settings):
        raise RuntimeError("Checkpoint run contract differs")
    episode = checkpoint["episode"]
    if type(episode) is not int or not 0 <= episode < len(settings["seeds"]):
        raise RuntimeError("Checkpoint episode index differs")
    env = checkpoint["environment"]
    if (len(checkpoint["completed"]) != episode or not 5 <= env["k"] <= 80 or
            len(checkpoint["trace"]) != env["k"] - 5 or
            env["profile_hash"] != settings["profile_sha256"][episode]):
        raise RuntimeError("Checkpoint episode boundary differs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("train", "center", "rl"), required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--cpu-mask", type=int, default=1)
    p.add_argument("--training-seeds", type=int, nargs="+", default=[6201, 6202])
    p.add_argument("--validation-seed", type=int)
    p.add_argument("--policy-seed", type=int, default=6200)
    p.add_argument("--guard-mode", choices=("physical", "h3"), default="physical")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.mode == "rl" and args.model is None:
        p.error("RL evaluation requires --model")
    if args.mode == "train" and args.validation_seed is not None:
        p.error("Validation must not train")
    with exclusive_run(args.output):
        if (args.output / "completion.json").exists():
            raise RuntimeError("Already completed; refusing duplicate work")
        if (args.output / "checkpoint.pt").exists() != args.resume:
            raise RuntimeError("Use --resume only for an existing checkpoint")
        saved_pins = pins(args.root)
        rt = boot(args.root, args.cpu_mask)
        import numpy as np
        import torch
        from budget_env import BudgetEnv
        from td3 import TD3
        torch.set_num_threads(1)
        seeds = args.training_seeds if args.mode == "train" else [args.validation_seed]
        # Construction verifies each profile without executing a plant interval or PFO solve.
        profiles = [BudgetEnv(rt, training_seed=seed, guard_mode=args.guard_mode) for seed in seeds]
        settings = dict(format=RUN_FORMAT,
                        run_id=read(args.output / "settings.json")["run_id"] if args.resume else uuid.uuid4().hex,
                        mode=args.mode, seeds=seeds, policy_seed=args.policy_seed,
                        profile_sha256=[env.profile_hash for env in profiles],
                        environment_contract=dict(config_sha256=digest(rt["rc"].to_plain_dict(rt["cfg"])),
                                                  options_sha256=digest(rt["rc"].to_plain_dict(rt["options"])),
                                                  protocol_sha256=digest(profiles[0].protocol),
                                                  coordinator={key: value for key, value in profiles[0].contract().items()
                                                               if key not in ("cfg", "options", "observation", "source_snapshot")}),
                        runtime_versions=runtime_versions(),
                        model_sha256=None if args.model is None else file_hash(args.model),
                        reward_divisor=100., gamma=1., exploration_std=.3,
                        initial_random_steps=32, batch_size=32, updates_per_interval=20,
                        guard_mode=args.guard_mode,
                        total_seconds=14400, warmup_steps=5, controlled_steps=75,
                        cpu_mask=args.cpu_mask, source_pins=saved_pins)
        write_run_settings(args.output, settings, args.resume)
        save(args.output / "process.json", dict(pid=os.getpid(), command=sys.argv, started=time.time()))
        rng = np.random.default_rng(args.policy_seed)
        learner, completed, trace, episode, obs, env = None, [], [], 0, None, None
        elapsed_before = 0.
        started = time.perf_counter()
        try:
            if args.resume:
                ck = torch.load(args.output / "checkpoint.pt", map_location="cpu", weights_only=False)
                validate_checkpoint(ck, settings)
                episode, completed, trace = ck["episode"], ck["completed"], ck["trace"]
                elapsed_before = ck["elapsed_wall_seconds"]
                rng.bit_generator.state = ck["exploration_rng"]
            while episode < len(seeds):
                if env is not None:
                    env.close()
                env = profiles[episode]
                if args.resume:
                    obs = env.restore(ck["environment"])
                    learner = TD3(len(obs), args.policy_seed)
                    learner.load_state_dict(ck["learner"])
                    args.resume = False
                else:
                    obs = env.reset()
                    trace = []
                    if learner is None:
                        learner = TD3(len(obs), args.policy_seed)
                        if args.model is not None:
                            model = torch.load(args.model, map_location="cpu", weights_only=False)
                            load_policy(learner, model, env, settings)
                schema = observation_schema(env)
                if args.output.joinpath("observation_schema.json").exists():
                    if read(args.output / "observation_schema.json") != schema:
                        raise RuntimeError("Run observation schema differs")
                save(args.output / "observation_schema.json", schema)
                def checkpoint():
                    checkpoint_save(torch, args.output / "checkpoint.pt", dict(
                        format=CHECKPOINT_FORMAT, settings=settings,
                        episode=episode, completed=completed, trace=trace,
                        environment=env.checkpoint(), learner=learner.state_dict(),
                        exploration_rng=rng.bit_generator.state,
                        elapsed_wall_seconds=elapsed_before + time.perf_counter() - started))
                checkpoint()
                while env.k < 80:
                    if (args.output / "STOP").exists() or (args.output.parent / "STOP").exists():
                        checkpoint()
                        save(args.output / "status.json", dict(status="paused", episode=episode, step=env.k))
                        return
                    verify_pins(args.root, saved_pins)
                    tick, actor_cpu = time.perf_counter(), time.process_time()
                    if args.mode == "train" and len(learner.replay) < 32:
                        action = rng.uniform(-1., 1., 2).astype(np.float32)
                    elif args.mode in ("rl", "train"):
                        action = learner.act(obs)
                        if args.mode == "train":
                            action = np.clip(action + rng.normal(0., .3, 2), -1., 1.).astype(np.float32)
                    else:
                        action = np.zeros(2, dtype=np.float32)
                    actor_seconds = time.perf_counter() - tick
                    actor_cpu_seconds = time.process_time() - actor_cpu
                    next_obs, reward, terminal, row = env.step(
                        action, "rl" if args.mode == "train" else args.mode, actor_seconds,
                        actor_cpu_seconds=actor_cpu_seconds)
                    if args.mode == "train":
                        learner.add(obs, action, reward, next_obs, bool(terminal))
                        learning_start = time.perf_counter()
                        row["learning"] = [learner.update(batch_size=32)
                                           for _ in range(settings["updates_per_interval"])]
                        row["learning_wall_seconds"] = time.perf_counter() - learning_start
                    row["reward"] = reward
                    row.update(episode=episode, profile_sha256=env.profile_hash)
                    trace.append(plain(row))
                    obs = next_obs
                    checkpoint()
                    save(args.output / "trace.json", trace)
                    save(args.output / "status.json", dict(status="running", episode=episode,
                         step=env.k, control_steps=env.k - 5, simulation_seconds=env.sim.state.time_sec,
                         ttt=env.sim.total_ttt, learner_updates=learner.updates))
                    print(args.mode, episode, f"{env.k-5}/75", f"TTT={env.sim.total_ttt:.6f}", flush=True)
                verify_pins(args.root, saved_pins)
                if len(trace) != 75 or not trace[-1]["terminated"]:
                    raise RuntimeError("Incomplete full trajectory")
                if abs(env.warmup_ttt + sum(row["interval_ttt"] for row in trace) - env.sim.total_ttt) > 1e-8:
                    raise RuntimeError("Full TTT does not reconcile")
                decision_times = [row["decision_wall_seconds"] for row in trace]
                summary = dict(episode=episode, training_seed=seeds[episode],
                               profile_sha256=env.profile_hash, ttt=env.sim.total_ttt,
                               freeway_ttt=trace[-1]["freeway_ttt"], urban_ttt=trace[-1]["urban_ttt"],
                               warmup_ttt=env.warmup_ttt, full_run=True, control_steps=75,
                               simulation_seconds=14400, terminal_inventory=trace[-1]["inventory"],
                               decision_wall_seconds=sum(decision_times),
                               decision_cpu_seconds=sum(row["decision_cpu_seconds"] for row in trace),
                               decision_p50=float(np.median(decision_times)),
                               decision_p95=float(np.percentile(decision_times, 95)),
                               decision_max=max(decision_times),
                               fallback_count=sum(row["selection_source"] == "reference_fallback" for row in trace),
                               pfo_calls=sum(row["pfo_calls"] for row in trace),
                               pfo_recovery_count=sum(row["reference_recovery"] for row in trace),
                               lower_candidate_solves=sum(row["lower_candidate_count"] for row in trace),
                               converged_count=sum(row["converged"] for row in trace),
                               learner_updates=learner.updates, controller_acceptance=False,
                               evaluation=args.mode != "train", exploration=args.mode == "train")
                from compare_runs import validate_episode
                validate_episode(summary, trace, settings, episode)
                completed.append(summary)
                save(args.output / f"episode_{episode:02d}_summary.json", summary)
                save(args.output / f"episode_{episode:02d}_trace.json", trace)
                if args.mode == "train":
                    checkpoint_save(torch, args.output / f"model_episode_{episode:02d}.pt",
                                    export_policy(learner, env, settings, completed))
                episode += 1
            if args.mode == "train":
                policy = export_policy(learner, env, settings, completed)
                checkpoint_save(torch, args.output / "model_final.pt", policy)
                save(args.output / "model_hash.json", dict(sha256=file_hash(args.output / "model_final.pt"),
                     format=POLICY_FORMAT, contract=policy["contract"],
                     training_run_id=settings["run_id"], training_profiles=policy["training_profiles"],
                     learner=learner.spec(), train_seeds=seeds, frozen=True))
            save(args.output / "completion.json", dict(format=RUN_FORMAT, run_id=settings["run_id"],
                 runtime_versions=settings["runtime_versions"], status="completed", mode=args.mode,
                 episodes=completed, source_pins=saved_pins,
                 elapsed_wall_seconds=elapsed_before + time.perf_counter() - started,
                 goal_claim=False, controller_acceptance=False))
            save(args.output / "status.json", dict(status="completed", episodes=len(completed)))
        except Exception:
            save(args.output / "failure.json", dict(traceback=traceback.format_exc(), episode=episode,
                 step=None if env is None else getattr(env, "k", None), terminated_as_success=False))
            save(args.output / "status.json", dict(status="failed"))
            raise
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
