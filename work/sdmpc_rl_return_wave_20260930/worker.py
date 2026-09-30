"""One locked training-wave slot. Parent owns dispatch and the global worker budget."""
import argparse
import time
import uuid
import wave_support as w
from actor import authenticate, verify_live
from checks import helpers, validate_settings, validate_checkpoint, validate_rows, tag, audit_receipt, summarize
from budget_env import BudgetEnv
import restore_accounting


def check_output(output, resume):
    if (output / "completion.json").exists():
        raise FileExistsError("Completed slot cannot be recollected or resumed")
    if resume:
        if not (output / "latest.json").exists() or not (output / "settings.json").exists():
            raise ValueError("Resume requires the retained full environment checkpoint and settings")
    elif output.exists() and any(p.name not in {"runner.lock", "stdout.log", "stderr.log", "launcher.json"}
                                 for p in output.iterdir()):
        raise FileExistsError("Partial/orphan slot exists; preserve it and resume its checkpoint")


def operation(output, step, session, previous):
    name = "reset" if step == 0 else f"step_{step:02d}"
    w.save_once(output / "operations" / (name + ".json"),
        dict(session_id=session, control_steps=step, previous_checkpoint=previous))


def check_operations(output, steps):
    expected = {"reset.json"} | {f"step_{i:02d}.json" for i in range(1, steps+1)}
    actual = {p.name for p in (output / "operations").glob("*.json")}
    if actual != expected:
        raise ValueError("Uncheckpointed reset/step or missing operation evidence; preserve orphans, do not repeat physics")


def run(args):
    if args.scenario not in w.SCENARIOS or (args.max_new_steps is not None and
            (type(args.max_new_steps) is not int or not 1 <= args.max_new_steps <= 75)):
        raise ValueError("Invalid scenario or max-new-steps")
    output = w.WAVE / args.scenario
    w.check_stop(output)
    check_output(output, args.resume)
    with w.exclusive_run(output):
        w.check_stop(output)
        check_output(output, args.resume)
        started = time.perf_counter()
        source = w.sources()
        previous = w.read(output / "settings.json") if args.resume else None
        run_id = previous["run_id"] if previous else uuid.uuid4().hex
        session = uuid.uuid4().hex
        process = w.process_identity()
        start_path = output / "sessions" / f"{session}.start.json"
        w.save_once(start_path, dict(session_id=session, process=process, started_unix=time.time(),
            run_id=run_id, scenario=args.scenario, model_sha256=w.MODEL_SHA, sources=source,
            settings_digest=w.digest(previous) if previous else None, lock_held=True,
            resume_requested=bool(args.resume), scope=w.TIMING_SCOPE))
        w.save(output / "process.json", dict(**process, session_id=session, run_id=run_id,
            scenario=args.scenario, model_sha256=w.MODEL_SHA, sources=source, lock_held=True))
        env, settings, ready = None, None, False
        outcome = "failed"
        steps = 0
        try:
            actor, authentication = authenticate()
            verify_live(authentication, source)
            w.check_stop(output)
            mask = w.MASKS[w.SCENARIOS.index(args.scenario)]
            rt = w.boot(w.DEFAULT_SNAPSHOT, mask)
            w.import_boundary(physical=True)
            env = BudgetEnv(rt, scenario=args.scenario,
                            training_seed=7301+w.SCENARIOS.index(args.scenario), guard_mode="physical")
            helpers().verify_environment(env, rt, authentication["contract"])
            settings = dict(format=w.FORMAT, run_id=run_id, scenario=args.scenario,
                behavior="frozen_shared_actor", mode="training_wave",
                training_seed=7301+w.SCENARIOS.index(args.scenario), profile_sha256=env.profile_hash,
                evaluation=False, exploration=False, policy_improvement_claim=False,
                model_sha256=w.MODEL_SHA, authentication=authentication, sources=source,
                contract=authentication["contract"], capacity=float(rt["cfg"].network.total_ramp_capacity),
                cpu_mask=mask, physical_config=w.plain(rt["rc"].to_plain_dict(rt["cfg"])),
                physical_options=w.plain(rt["rc"].to_plain_dict(rt["options"])),
                reward_divisor=100., gamma=1., warmup_steps=5, controlled_steps=75, total_seconds=14400,
                critic_continuation="carry", updates_per_interval=0, threads=1,
                warmup_ttt=previous["warmup_ttt"] if previous else None)
            if env.profile_hash != authentication["profiles"][args.scenario]:
                raise ValueError("Constructed demand profile differs from declared training seed")
            w.check_stop(output)
            verify_live(authentication, source)
            if args.resume:
                validate_settings(settings, authentication, source, args.scenario)
                if previous != settings:
                    raise ValueError("Resume source/model/settings mismatch")
                record, path = w.checkpoint_record(output)
                check_operations(output, record["control_steps"])
                ck = w.torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint(ck, settings, actor)
                if (record["settings_digest"] != w.digest(settings) or
                        record["control_steps"] != len(ck["trace"]) or record["session_ids"] != ck["session_ids"]):
                    raise ValueError("Checkpoint pointer/progress mismatch")
                w.session_timing(output, settings, ck["session_ids"])
                w.check_stop(output)
                verify_live(authentication, source)
                obs = restore_accounting.restore_environment(env, ck["environment"], output,
                                                              session, settings, record)
                w.np.testing.assert_array_equal(obs, ck["observation"])
                trace, transitions = ck["trace"], ck["transitions"]
            else:
                operation(output, 0, session, None)
                obs = env.reset()
                settings["warmup_ttt"] = env.warmup_ttt
                trace, transitions = [], []
            helpers().verify_environment(env, rt, authentication["contract"], schema=True)
            schema = w.observation_schema(env)
            validate_settings(settings, authentication, source, args.scenario)
            w.save_once(output / "settings.json", settings)
            w.save_once(output / "observation_schema.json", schema)

            def checkpoint():
                ck = dict(format=w.FORMAT, settings=settings, model_sha256=w.MODEL_SHA, schema=schema,
                    environment=env.checkpoint(), observation=obs.copy(), trace=trace, transitions=transitions,
                    session_ids=w.session_ids(output), process=process,
                    session_progress=dict(session_id=session, elapsed_wall_seconds=time.perf_counter()-started,
                                          control_steps=len(trace), simulation_seconds=env.sim.state.time_sec))
                record = w.persist_checkpoint(output, ck, lambda value: validate_checkpoint(value, settings, actor))
                verify_live(authentication, source)
                if w.read(output / "settings.json") != settings:
                    raise ValueError("Persisted settings mutated")
                return record

            if not args.resume:
                record = checkpoint()
            steps = len(trace)
            new_steps = 0
            while env.k < 80:
                w.check_stop(output)
                verify_live(authentication, source)
                if w.read(output / "settings.json") != settings:
                    raise ValueError("Persisted settings mutated")
                helpers().validate_observation(obs, env.k, schema, env.controller.action_anchor,
                                               env.last_requested, env.last_executed)
                tick, cpu = time.perf_counter(), time.process_time()
                action = actor.act(obs)
                actor_wall, actor_cpu = time.perf_counter()-tick, time.process_time()-cpu
                w.check_stop(output)
                operation(output, steps+1, session, record)
                nxt, reward, terminal, row = env.step(action, "rl", actor_wall, actor_cpu_seconds=actor_cpu)
                row.update(reward=reward, scenario=args.scenario, behavior="frozen_shared_actor",
                    mode="training_wave", evaluation=False, exploration=False, model_sha256=w.MODEL_SHA,
                    profile_sha256=env.profile_hash, policy_q=None, learning=[], learning_wall_seconds=0.)
                trace.append(w.plain(row))
                transitions.append((obs.copy(), action.copy(), reward, nxt.copy(), bool(terminal)))
                obs = nxt
                record = checkpoint()
                steps, new_steps = len(trace), new_steps+1
                w.save(output / "status.json", dict(status="checkpointed", control_steps=steps,
                    session_id=session, model_sha256=w.MODEL_SHA, latest=record))
                print(args.scenario, f"{steps}/75", f"TTT={env.sim.total_ttt:.6f}", flush=True)
                if args.max_new_steps is not None and new_steps >= args.max_new_steps and not terminal:
                    outcome = "checkpointed"
                    return outcome
            w.check_stop(output)
            verify_live(authentication, source)
            validate_rows(trace, transitions, settings, schema, actor)
            payload = dict(format=w.FORMAT, settings=settings, transitions=transitions)
            w.save_tensor(output / "experience.pt", payload)
            w.save(output / "trace.json", tag(trace)[0])
            w.save(output / "diagnostic-audit.json", audit_receipt(output, trace, payload))
            ready, outcome = True, "completed"
        except w.Stopped:
            outcome = "stopped"
            return outcome
        except BaseException as exc:
            w.save(output / "failure-" / f"{session}.json", dict(error=f"{type(exc).__name__}: {exc}",
                session_id=session, control_steps=steps, preserve_all_orphans=True))
            raise
        finally:
            try:
                if env is not None:
                    env.close()
            except BaseException:
                outcome, ready = "failed", False
                raise
            finally:
                w.save_once(output / "sessions" / f"{session}.end.json", dict(session_id=session,
                    process=process, start_sha256=w.file_hash(start_path), outcome=outcome,
                    restoration_sha256=restore_accounting.files(output, session),
                    elapsed_wall_seconds=time.perf_counter()-started, control_steps=steps))
                w.save(output / "status.json", dict(status=outcome, session_id=session, control_steps=steps))
        if ready:
            w.check_stop(output)
            verify_live(authentication, source)
            timing = w.session_timing(output, settings, record["session_ids"])
            w.save(output / "timing.json", timing)
            w.save(output / "summary.json", summarize(trace, settings, timing))
            names = ("settings.json", "latest.json", "trace.json", "experience.pt", "observation_schema.json",
                     "diagnostic-audit.json", "timing.json", "summary.json")
            w.check_stop(output)
            w.save_once(output / "completion.json", dict(format=w.FORMAT, status="completed", settings=settings,
                model_sha256=w.MODEL_SHA, checkpoint=record, outputs_sha256={n: w.file_hash(output / n) for n in names}))
        return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=w.SCENARIOS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-steps", type=int)
    args = parser.parse_args(argv)
    try:
        return 2 if run(args) == "stopped" else 0
    except w.Stopped:
        print("stopped_before_environment", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
