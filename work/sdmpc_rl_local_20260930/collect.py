"""One checkpointed, training-only carry or local-budget trajectory."""
import argparse
import os
from pathlib import Path
import sys
import time
from local_runtime import (np, torch, REPO, FORMAT, BEHAVIORS, MASKS, SCENARIOS,
    DEFAULT_SNAPSHOT, boot, read, save, plain, digest, file_hash, exclusive_run,
    identity, verify_identity, contract, verify_environment, seeds, check_stop,
    Stopped, checkpoint_save, observation_schema)
from local_runtime import (canonical_worker, claim_worker, release_worker, durable_save,
                           session_ids, reserved_sessions, session_timing, process_identity,
                           reservation_run_id, checkpoint_progress)
from budget_env import BudgetEnv
from exploration import LocalBudgetPolicy
from validate import validate_rows, summarize, replay_policy, validate_boundary, validate_observation


def check_output(output, resume):
    if (output / "completion.json").exists():
        raise ValueError("Already completed; refusing duplicate collection")
    if resume:
        if not (output / "checkpoint.pt").exists():
            raise ValueError("Resume requires a durable interval checkpoint")
    else:
        allowed = {"runner.lock", "stdout.log", "stderr.log", "launcher.json"}
        if output.exists() and any(p.name not in allowed for p in output.iterdir()):
            raise ValueError("Nonempty output; resume the durable checkpoint")


def validate_checkpoint(ck, settings):
    if ck["format"] != FORMAT or ck["settings"] != settings:
        raise ValueError("Checkpoint identity changed")
    if ck["schema"] != settings["contract"]["observation_schema"]:
        raise ValueError("Checkpoint schema changed")
    trace, experience, env = ck["trace"], ck["transitions"], ck["environment"]
    n = len(trace)
    if env["k"] != n+5 or env["profile_hash"] != settings["profile_sha256"]:
        raise ValueError("Checkpoint boundary differs")
    total = validate_rows(trace, experience, settings, ck["schema"], settings["capacity"], complete=False)
    if abs(env["sim"].total_ttt-total) > 1e-8 or env["sim"].state.time_sec != (5+n)*180:
        raise ValueError("Checkpoint simulator/trace boundary differs")
    validate_boundary(ck, settings)
    policy = replay_policy(trace, settings)
    if digest(policy.state_dict()) != digest(ck["policy"]):
        raise ValueError("Saved exploration state differs from actual trace")
    if n:
        np.testing.assert_array_equal(ck["observation"], experience[-1][3])
    return policy


def run(args):
    output = args.output.resolve()
    canonical_worker(output, args.scenario, args.behavior, args.cpu_mask)
    check_stop(output, args.abort_file)
    check_output(output, args.resume)
    with exclusive_run(output):
        check_stop(output, args.abort_file)
        check_output(output, args.resume)
        reservation = claim_worker(output, args.scenario, args.behavior, args.cpu_mask,
                                   getattr(args, "reservation", None), resume=args.resume)
        started = time.perf_counter()
        session = reservation
        sessions = sorted(set(session_ids(output)) | set(reserved_sessions(output)))
        start_file = output / "sessions" / f"{session}.start.json"
        env = None
        outcome, result = "failed", None
        try:
            durable_save(start_file, dict(session_id=session, process=process_identity(), started=time.time()))
            save(output / "process.json", dict(**process_identity(), started=time.time()))
            save(output / "status.json", dict(status="initializing", pid=os.getpid()))
            source = identity()
            expected = contract()
            rt = boot(DEFAULT_SNAPSHOT, args.cpu_mask)
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            demand_seed, exploration_seed = seeds(args.scenario)
            env = BudgetEnv(rt, scenario=args.scenario, training_seed=demand_seed, guard_mode="physical")
            verify_environment(env, rt, expected)
            capacity = float(rt["cfg"].network.total_ramp_capacity)
            previous = read(output / "settings.json") if args.resume else None
            settings = dict(format=FORMAT, run_id=reservation_run_id(reservation),
                scenario=args.scenario, behavior=args.behavior, training_seed=demand_seed,
                exploration_seed=exploration_seed, profile_sha256=env.profile_hash,
                evaluation=False, model_sha256=None, identity=source, contract=expected,
                capacity=capacity, cpu_mask=args.cpu_mask,
                physical_config=plain(rt["rc"].to_plain_dict(rt["cfg"])),
                warmup_ttt=previous["warmup_ttt"] if previous else None)
            check_stop(output, args.abort_file)
            if args.resume:
                if previous != settings:
                    raise ValueError("Resume settings changed")
                ck = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
                policy = validate_checkpoint(ck, settings)
                checkpoint_progress(reservation, settings["run_id"], len(ck["trace"]))
                obs = env.restore(ck["environment"])
                np.testing.assert_array_equal(obs, ck["observation"])
                trace, experience = ck["trace"], ck["transitions"]
                sessions = sorted(set(sessions) | set(ck["session_ids"]))
            else:
                obs = env.reset()
                settings["warmup_ttt"] = env.warmup_ttt
                policy = LocalBudgetPolicy(args.behavior, exploration_seed, capacity)
                trace, experience = [], []
            verify_environment(env, rt, expected, schema=True)
            schema = observation_schema(env)
            check_stop(output, args.abort_file)
            save(output / "settings.json", settings)
            save(output / "observation_schema.json", schema)

            def checkpoint():
                checkpoint_save(torch, output / "checkpoint.pt", dict(format=FORMAT,
                    settings=settings, schema=schema, environment=env.checkpoint(),
                    observation=obs.copy(), trace=trace, transitions=experience,
                    policy=policy.state_dict(), session_ids=sessions))
                checkpoint_progress(reservation, settings["run_id"], len(trace))

            checkpoint()
            steps_this_session = 0
            while env.k < 80:
                check_stop(output, args.abort_file)
                verify_identity(source)
                validate_observation(obs, env.k, schema, env.controller.action_anchor,
                                     env.last_requested, env.last_executed)
                tick, cpu = time.perf_counter(), time.process_time()
                action, _ = policy.choose(env.controller.action_anchor.copy())
                actor_wall, actor_cpu = time.perf_counter()-tick, time.process_time()-cpu
                nxt, reward, terminal, row = env.step(action, "center" if args.behavior == "carry" else "rl",
                                                    actor_wall, actor_cpu_seconds=actor_cpu)
                exploration = policy.commit(row)
                row.update(exploration_audit=exploration, reward=reward, scenario=args.scenario,
                           behavior=args.behavior, profile_sha256=env.profile_hash,
                           policy_q=None, learning=[], learning_wall_seconds=0.)
                experience.append((obs.copy(), action.copy(), reward, nxt.copy(), bool(terminal)))
                trace.append(plain(row))
                obs = nxt
                checkpoint()
                steps_this_session += 1
                save(output / "status.json", dict(status="running", pid=os.getpid(),
                    control_steps=len(trace), simulation_seconds=env.sim.state.time_sec,
                    ttt=env.sim.total_ttt))
                print(args.behavior, args.scenario, f"{len(trace)}/75", f"TTT={env.sim.total_ttt:.6f}", flush=True)
                if args.max_new_steps is not None and steps_this_session >= args.max_new_steps and not terminal:
                    save(output / "status.json", dict(status="checkpointed", control_steps=len(trace), pid=os.getpid()))
                    outcome = "checkpointed"
                    return outcome
            check_stop(output, args.abort_file)
            verify_identity(source)
            validate_rows(trace, experience, settings, schema, capacity)
            validate_boundary(dict(environment=env.checkpoint(), trace=trace, observation=obs, schema=schema), settings)
            replay_policy(trace, settings)
            prior_timing = session_timing(output, sessions, exclude=(session,))
            summary = summarize(trace, settings, prior_timing["timing_status"])
            save(output / "trace.json", trace)
            save(output / "summary.json", summary)
            checkpoint_save(torch, output / "experience.pt", dict(format=FORMAT, settings=settings, transitions=experience))
            result = dict(format=FORMAT, status="completed", settings=settings,
                outputs_sha256={name: file_hash(output / name) for name in
                    ("trace.json", "summary.json", "experience.pt", "observation_schema.json")})
            verify_identity(source)
            if read(output / "settings.json") != settings:
                raise ValueError("Persisted settings changed during collection")
            check_stop(output, args.abort_file)
            save(output / "status.json", dict(status="completed", control_steps=75, pid=os.getpid()))
            outcome = "completed"
            return outcome
        except Stopped:
            result, outcome = None, "stopped"
            save(output / "status.json", dict(status="stopped", pid=os.getpid()))
            return "stopped"
        except BaseException as exc:
            result, outcome = None, "failed"
            save(output / "status.json", dict(status="failed", pid=os.getpid(), error=f"{type(exc).__name__}: {exc}"))
            raise
        finally:
            try:
                if env is not None:
                    env.close()
            except BaseException as exc:
                result, outcome = None, "failed"
                save(output / "status.json", dict(status="failed", pid=os.getpid(), error=f"close: {exc}"))
                raise
            finally:
                release_worker(reservation, finished=result is not None)
                if start_file.exists():
                    elapsed = time.perf_counter()-started
                    durable_save(output / "sessions" / f"{session}.end.json", dict(session_id=session,
                        start_sha256=file_hash(start_file), outcome=outcome, elapsed_wall_seconds=elapsed))
                    timing = session_timing(output, sessions)
                    durable_save(output / "timing.json", timing)
            if result is not None:
                check_stop(output, args.abort_file)
                result["outputs_sha256"]["timing.json"] = file_hash(output / "timing.json")
                save(output / "completion.json", result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--behavior", choices=BEHAVIORS, required=True)
    p.add_argument("--cpu-mask", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--abort-file", type=Path)
    p.add_argument("--reservation", help=argparse.SUPPRESS)
    p.add_argument("--max-new-steps", type=int)
    args = p.parse_args(argv)
    if args.cpu_mask != MASKS[SCENARIOS.index(args.scenario)]:
        p.error("Use the pinned scenario CPU mask")
    if args.max_new_steps is not None and not 1 <= args.max_new_steps <= 75:
        p.error("max-new-steps must be 1..75; it checkpoints, never declares completion")
    return 2 if run(args) == "stopped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
