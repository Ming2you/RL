"""One resumable trajectory under a frozen shared policy; never trains locally."""
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
from budget_runtime import boot, DEFAULT_SNAPSHOT, HERE, read, save, digest, plain, protocol

RUN_FORMAT = "sdmpc-multi-run-v1"
POLICY_FORMAT = "sdmpc-multi-policy-v1"
CHECKPOINT_FORMAT = "sdmpc-multi-episode-checkpoint-v1"
EXPERIENCE_FORMAT = "sdmpc-multi-experience-v1"


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


def scenario_manifest(runtime):
    from td3 import SCENARIOS
    rows = []
    for scenario in SCENARIOS:
        folder, data, _ = protocol(runtime, scenario)
        rows.append(dict(scenario=scenario, protocol_sha256=digest(data),
                         profile_sha256=digest(read(folder / "forecast.json")),
                         initial_state_sha256=digest(read(folder / "initial_state.json"))))
    return rows


def observation_schema(env):
    return dict(names=list(env.observer.names), scale="fixed_physical_constants_no_eval_fitting",
                delay_bins=env.observer.delay + 1, normalization=env.observer.contract(),
                Markov_sufficiency="not_proven_recovery_PFO_memory_is_checkpointed_but_not_fully_observed")


def policy_contract(env, settings):
    return dict(source_pins=settings["source_pins"], runtime_versions=settings["runtime_versions"],
                environment_contract=settings["environment_contract"],
                observation_schema=observation_schema(env), reward_divisor=100., gamma=1.)


def validate_policy(model, expected):
    from td3 import SCENARIOS
    if model.get("format") != POLICY_FORMAT or model.get("contract") != expected:
        raise ValueError("Shared policy contract differs")
    profiles = model.get("training_profiles", [])
    counts = {scenario: 0 for scenario in SCENARIOS}
    identities = set()
    for row in profiles:
        identity = (row.get("scenario"), row.get("seed"))
        if (row.get("scenario") not in counts or type(row.get("seed")) is not int or
                identity in identities or not row.get("run_id") or not row.get("experience_sha256") or
                not row.get("profile_sha256")):
            raise ValueError("Invalid or duplicate policy training provenance")
        identities.add(identity)
        counts[row["scenario"]] += 1
    if not profiles or len(set(counts.values())) != 1 or not model.get("training_run_id"):
        raise ValueError("Policy does not have balanced five-scenario provenance")


def stopped(output):
    return any((folder / "STOP").exists() for folder in (output, output.parent, output.parent.parent))


def validate_checkpoint(ck, settings):
    if ck.get("format") != CHECKPOINT_FORMAT or ck.get("settings") != settings:
        raise ValueError("Episode checkpoint contract differs")
    env, trace, experience = ck["environment"], ck["trace"], ck["experience"]
    if (type(env["k"]) is not int or not 5 <= env["k"] <= 80 or
            len(trace) != env["k"] - 5 or env["profile_hash"] != settings["profile_sha256"][0] or
            len(experience) != (len(trace) if settings["mode"] == "collect" else 0)):
        raise ValueError("Checkpoint trajectory boundary differs")


def write_run_settings(output, settings, resume):
    if resume:
        if (read(output / "settings.json") != settings or
                read(output / "runtime_versions.json") != settings["runtime_versions"]):
            raise ValueError("Resume settings/runtime mismatch")
    else:
        save(output / "settings.json", settings)
        save(output / "runtime_versions.json", settings["runtime_versions"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("collect", "center", "rl"), required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--training-seed", type=int)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--model", type=Path)
    p.add_argument("--cpu-mask", type=int, default=1)
    p.add_argument("--policy-seed", type=int, default=6300)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    with exclusive_run(args.output):
        if (args.output / "completion.json").exists():
            raise RuntimeError("Already completed; refusing duplicate collection")
        if (args.output / "checkpoint.pt").exists() != args.resume:
            raise ValueError("Use --resume only for existing checkpoint")
        saved_pins = pins(args.root)
        rt = boot(args.root, args.cpu_mask)
        import numpy as np
        import torch
        from budget_env import BudgetEnv
        from td3 import TD3, SCENARIOS
        if args.scenario not in SCENARIOS or args.round not in (0, 1):
            raise ValueError("Unknown scenario or collection round")
        if (args.mode == "collect") != (args.training_seed is not None):
            raise ValueError("Only collection uses a training demand seed")
        if (args.mode == "rl" or (args.mode == "collect" and args.round > 0)) and args.model is None:
            raise ValueError("Missing shared policy")
        if (args.mode == "center" or (args.mode == "collect" and args.round == 0)) and args.model is not None:
            raise ValueError("Zero-initialized mode must not load a policy")
        torch.set_num_threads(1)
        env = BudgetEnv(rt, scenario=args.scenario, training_seed=args.training_seed, guard_mode="physical")
        existing_settings = (args.output / "settings.json").exists()
        settings = dict(format=RUN_FORMAT,
            run_id=read(args.output / "settings.json")["run_id"] if existing_settings else uuid.uuid4().hex,
            mode=args.mode, scenario=args.scenario, round=args.round, seeds=[args.training_seed],
            policy_seed=args.policy_seed, profile_sha256=[env.profile_hash],
            environment_contract=dict(config_sha256=digest(rt["rc"].to_plain_dict(rt["cfg"])),
                options_sha256=digest(rt["rc"].to_plain_dict(rt["options"])),
                scenarios=scenario_manifest(rt),
                coordinator={k: v for k, v in env.contract().items()
                             if k not in ("cfg", "options", "observation", "source_snapshot")}),
            runtime_versions=runtime_versions(),
            model_sha256=None if args.model is None else file_hash(args.model),
            reward_divisor=100., gamma=1., exploration_std=.3,
            initial_random_steps=32 if args.mode == "collect" and args.round == 0 else 0,
            batch_size=40, updates_per_interval=0, guard_mode="physical",
            total_seconds=14400, warmup_steps=5, controlled_steps=75,
            cpu_mask=args.cpu_mask, source_pins=saved_pins)
        write_run_settings(args.output, settings, existing_settings)
        save(args.output / "process.json", dict(pid=os.getpid(), command=sys.argv, started=time.time()))
        rng = np.random.default_rng(args.training_seed if args.training_seed is not None else args.policy_seed)
        trace, experience, elapsed_before = [], [], 0.
        started = time.perf_counter()
        try:
            if args.resume:
                ck = torch.load(args.output / "checkpoint.pt", map_location="cpu", weights_only=False)
                validate_checkpoint(ck, settings)
                obs = env.restore(ck["environment"])
                trace, experience = ck["trace"], ck["experience"]
                elapsed_before = ck["elapsed_wall_seconds"]
                rng.bit_generator.state = ck["exploration_rng"]
            else:
                obs = env.reset()
            learner = TD3(len(obs), args.policy_seed)
            if args.model is not None:
                model = torch.load(args.model, map_location="cpu", weights_only=False)
                validate_policy(model, policy_contract(env, settings))
                learner.load_state_dict(model["learner"])
            schema = observation_schema(env)
            if (args.output / "observation_schema.json").exists() and read(args.output / "observation_schema.json") != schema:
                raise ValueError("Observation schema changed")
            save(args.output / "observation_schema.json", schema)
            initial_updates = learner.updates

            def checkpoint():
                checkpoint_save(torch, args.output / "checkpoint.pt", dict(
                    format=CHECKPOINT_FORMAT, settings=settings, environment=env.checkpoint(),
                    trace=trace, experience=experience,
                    exploration_rng=rng.bit_generator.state,
                    elapsed_wall_seconds=elapsed_before + time.perf_counter() - started))

            checkpoint()
            while env.k < 80:
                if stopped(args.output):
                    save(args.output / "status.json", dict(status="paused", step=env.k))
                    return
                verify_pins(args.root, saved_pins)
                tick, cpu = time.perf_counter(), time.process_time()
                if args.mode == "center":
                    action = np.zeros(2, dtype=np.float32)
                elif args.mode == "collect" and len(trace) < settings["initial_random_steps"]:
                    action = rng.uniform(-1., 1., 2).astype(np.float32)
                else:
                    action = learner.act(obs)
                    if args.mode == "collect":
                        action = np.clip(action + rng.normal(0., .3, 2), -1., 1.).astype(np.float32)
                policy_q = None
                if args.mode != "center":
                    with torch.no_grad():
                        inputs = torch.from_numpy(np.concatenate((obs, action))).unsqueeze(0)
                        policy_q = [float(critic(inputs).item()) for critic in learner.critics]
                actor_wall, actor_cpu = time.perf_counter() - tick, time.process_time() - cpu
                next_obs, reward, terminal, row = env.step(action,
                    "center" if args.mode == "center" else "rl", actor_wall, actor_cpu_seconds=actor_cpu)
                if args.mode == "collect":
                    experience.append((obs.copy(), action.copy(), reward, next_obs.copy(), bool(terminal)))
                row.update(reward=reward, episode=0, scenario=args.scenario, profile_sha256=env.profile_hash,
                           policy_q=policy_q, learning=[], learning_wall_seconds=0.)
                trace.append(plain(row))
                obs = next_obs
                checkpoint()
                save(args.output / "status.json", dict(status="running", scenario=args.scenario,
                    control_steps=env.k-5, simulation_seconds=env.sim.state.time_sec,
                    ttt=env.sim.total_ttt, learner_updates=learner.updates))
                print(args.mode, args.scenario, f"{env.k-5}/75", f"TTT={env.sim.total_ttt:.6f}", flush=True)
            verify_pins(args.root, saved_pins)
            if args.model is not None and file_hash(args.model) != settings["model_sha256"]:
                raise ValueError("Frozen policy changed during trajectory")
            if learner.updates != initial_updates:
                raise RuntimeError("Collector/evaluator unexpectedly trained")
            times = [r["decision_wall_seconds"] for r in trace]
            summary = dict(episode=0, scenario=args.scenario, training_seed=args.training_seed,
                profile_sha256=env.profile_hash, ttt=env.sim.total_ttt,
                freeway_ttt=trace[-1]["freeway_ttt"], urban_ttt=trace[-1]["urban_ttt"],
                warmup_ttt=env.warmup_ttt, full_run=True, control_steps=len(trace),
                simulation_seconds=env.sim.state.time_sec, terminal_inventory=trace[-1]["inventory"],
                decision_wall_seconds=sum(times), decision_cpu_seconds=sum(r["decision_cpu_seconds"] for r in trace),
                decision_p50=float(np.median(times)), decision_p95=float(np.percentile(times, 95)),
                decision_max=max(times), fallback_count=sum(r["selection_source"] == "reference_fallback" for r in trace),
                pfo_calls=sum(r["pfo_calls"] for r in trace), pfo_recovery_count=sum(r["reference_recovery"] for r in trace),
                lower_candidate_solves=sum(r["lower_candidate_count"] for r in trace),
                converged_count=sum(r["converged"] for r in trace), learner_updates=learner.updates,
                controller_acceptance=False, evaluation=args.mode != "collect", exploration=args.mode == "collect")
            from compare_runs import validate_episode
            validate_episode(summary, trace, settings, 0)
            save(args.output / "episode_00_summary.json", summary)
            save(args.output / "episode_00_trace.json", trace)
            experience_hash = None
            if args.mode == "collect":
                checkpoint_save(torch, args.output / "experience.pt", dict(format=EXPERIENCE_FORMAT,
                    settings=settings, contract=policy_contract(env, settings), transitions=experience))
                experience_hash = file_hash(args.output / "experience.pt")
            save(args.output / "completion.json", dict(format=RUN_FORMAT, run_id=settings["run_id"],
                runtime_versions=settings["runtime_versions"], status="completed", mode=args.mode,
                episodes=[summary], source_pins=saved_pins, experience_sha256=experience_hash,
                elapsed_wall_seconds=elapsed_before + time.perf_counter() - started,
                goal_claim=False, controller_acceptance=False))
            save(args.output / "status.json", dict(status="completed"))
        except Exception:
            save(args.output / "failure.json", dict(traceback=traceback.format_exc(), step=getattr(env, "k", None)))
            save(args.output / "status.json", dict(status="failed"))
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
