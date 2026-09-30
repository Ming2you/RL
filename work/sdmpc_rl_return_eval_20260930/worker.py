"""One parent-admitted canonical scenario; no dispatch, optimization or policy Q."""
import argparse
import time
import uuid
import support as s
import checks
import preflight
from policy import authenticate
from budget_env import BudgetEnv

_ops = dict(w=s)
s.reuse("worker.py", ("check_output", "operation", "check_operations"), _ops)
check_output, operation, check_operations = (_ops[n] for n in ("check_output", "operation", "check_operations"))
OUTPUTS = ("settings.json", "latest.json", "trace.json", "experience.pt", "observation_schema.json",
           "diagnostic-audit.json", "timing.json", "summary.json", "finalization.json")


def tensor_once(path, payload):
    if path.exists():
        retained = s.torch.load(path, map_location="cpu", weights_only=False)
        if s.digest(retained) != s.digest(payload):
            raise FileExistsError("Retained evaluation evidence differs")
    else:
        s.save_tensor(path, payload)


def finalize(output, settings, actor):
    """Publication-only resume from a stable intent; never loads simulator state."""
    marker = s.read(output / "finalization.json")
    record, _ = s.checkpoint_record(output)
    check_operations(output, 75)
    if marker["settings_digest"] != s.digest(settings) or marker["checkpoint"] != record:
        raise ValueError("Finalization identity differs")
    for name, expected in marker["retained_outputs"].items():
        s.check_hash(output / name, expected)
    payload = s.torch.load(output / "experience.pt", map_location="cpu", weights_only=False)
    if payload["settings"] != settings or payload["format"] != s.FORMAT or payload["eval_only"] is not True:
        raise ValueError("Evaluation evidence identity differs")
    trace = checks.raw_trace(s.read(output / "trace.json"))
    schema = s.read(output / "observation_schema.json")
    checks.validate_rows(trace, payload["transitions"], settings, schema, actor)
    if checks.audit_receipt(output, trace, payload) != s.read(output / "diagnostic-audit.json"):
        raise ValueError("Diagnostic binding differs")
    timing = s.session_timing(output, settings, record["session_ids"])
    if timing != marker["timing"]:
        raise ValueError("Finalization sessions changed")
    s.save_once(output / "timing.json", timing)
    s.save_once(output / "summary.json", checks.summarize(trace, settings, timing))
    s.check_stop(output)
    s.save_once(output / "completion.json", dict(format=s.FORMAT, status="completed", settings=settings,
        model_sha256=s.MODEL_SHA, checkpoint=record, outputs_sha256={n: s.file_hash(output / n) for n in OUTPUTS}))


def run(args):
    if args.scenario not in s.SCENARIOS or (args.max_new_steps is not None and
            (type(args.max_new_steps) is not int or not 1 <= args.max_new_steps <= 75)):
        raise ValueError("Invalid scenario or step limit")
    output = s.ROOT / args.scenario
    s.check_stop(output)
    admitted, review = preflight.admission(args.preflight, args.review_receipt)
    check_output(output, args.resume)
    with s.exclusive_run(output):
        started, cpu_started = time.perf_counter(), time.process_time()
        s.check_stop(output)
        check_output(output, args.resume)
        if args.resume:
            retained, _ = s.checkpoint_record(output)
            check_operations(output, retained["control_steps"])
            final_names = ("experience.pt", "trace.json", "diagnostic-audit.json", "finalization.json", "completion.json")
            if retained["control_steps"] < 75 and any((output / name).exists() for name in final_names):
                raise FileExistsError("Final exports contradict partial checkpoint; preserve retained evidence")
        actor, auth = authenticate()
        if auth != admitted["authentication"]:
            raise ValueError("Preflight model authentication differs")
        source = s.sources()
        previous = s.read(output / "settings.json") if args.resume else None
        if previous:
            checks.validate_settings(previous, auth, source, args.scenario)
            if previous["review"] != review or previous["preflight_sha256"] != s.file_hash(args.preflight):
                raise ValueError("Resume admission differs")
        if (output / "finalization.json").exists():
            finalize(output, previous, actor)
            return "completed"
        session, process = uuid.uuid4().hex, s.process_identity()
        run_id = previous["run_id"] if previous else uuid.uuid4().hex
        start_path = output / "sessions" / f"{session}.start.json"
        s.save_once(start_path, dict(session_id=session, process=process, started_unix=time.time(),
            run_id=run_id, scenario=args.scenario, model_sha256=s.MODEL_SHA, sources=source,
            settings_digest=s.digest(previous) if previous else None, lock_held=True,
            resume_requested=bool(args.resume), scope=s.TIMING_SCOPE))
        s.save(output / "process.json", dict(**process, session_id=session, run_id=run_id, lock_held=True))
        env, ready, steps, outcome = None, False, 0, "failed"
        try:
            preflight.live(admitted)
            s.check_stop(output)
            rt = s.boot(s.DEFAULT_SNAPSHOT, s.MASKS[s.SCENARIOS.index(args.scenario)])
            s.import_boundary(physical=True)
            env = BudgetEnv(rt, scenario=args.scenario, training_seed=None, guard_mode="physical")
            checks.verify_environment(env, rt, auth["contract"])
            settings = dict(checks.metadata(args.scenario, auth, source), run_id=run_id,
                review=review, preflight_sha256=s.file_hash(args.preflight),
                physical_config=s.plain(rt["rc"].to_plain_dict(rt["cfg"])),
                physical_options=s.plain(rt["rc"].to_plain_dict(rt["options"])),
                warmup_ttt=previous["warmup_ttt"] if previous else None)
            s.check_stop(output)
            if args.resume:
                if settings != previous:
                    raise ValueError("Resume settings changed")
                record, path = s.checkpoint_record(output)
                check_operations(output, record["control_steps"])
                ck = s.torch.load(path, map_location="cpu", weights_only=False)
                checks.validate_checkpoint(ck, settings, actor)
                if (record["settings_digest"] != s.digest(settings) or record["control_steps"] != len(ck["trace"]) or
                        record["session_ids"] != ck["session_ids"]):
                    raise ValueError("Checkpoint pointer/progress differs")
                s.session_timing(output, settings, ck["session_ids"])
                obs = s.restore_accounting.restore_environment(env, ck["environment"], output, session, settings, record)
                s.np.testing.assert_array_equal(obs, ck["observation"])
                trace, transitions = ck["trace"], ck["transitions"]
            else:
                operation(output, 0, session, None)
                obs = env.reset()
                settings["warmup_ttt"] = env.warmup_ttt
                trace, transitions = [], []
            checks.verify_environment(env, rt, auth["contract"], schema=True)
            schema = s.observation_schema(env)
            checks.validate_settings(settings, auth, source, args.scenario)
            baseline = next(r for r in admitted["centers"]["rows"] if r["scenario"] == args.scenario)
            s.same_cost(settings["warmup_ttt"], baseline["warmup_ttt"], "matched warmup")
            s.save_once(output / "settings.json", settings)
            s.save_once(output / "observation_schema.json", schema)

            def checkpoint():
                ck = dict(format=s.FORMAT, settings=settings, model_sha256=s.MODEL_SHA, schema=schema,
                    environment=env.checkpoint(), observation=obs.copy(), trace=trace, transitions=transitions,
                    session_ids=s.session_ids(output), process=process)
                return s.persist_checkpoint(output, ck, lambda value: checks.validate_checkpoint(value, settings, actor))

            if not args.resume:
                record = checkpoint()
            steps, new_steps = len(trace), 0
            while env.k < 80:
                s.check_stop(output)
                preflight.live(admitted)
                if s.read(output / "settings.json") != settings:
                    raise ValueError("Persisted settings mutated")
                checks.helpers().validate_observation(obs, env.k, schema, env.controller.action_anchor,
                                                      env.last_requested, env.last_executed)
                tick, cpu = time.perf_counter(), time.process_time()
                action = actor.act(obs)
                actor_wall, actor_cpu = time.perf_counter()-tick, time.process_time()-cpu
                s.check_stop(output)
                operation(output, steps+1, session, record)
                nxt, reward, terminal, row = env.step(action, "rl", actor_wall, actor_cpu_seconds=actor_cpu)
                row.update(reward=reward, scenario=args.scenario, behavior="frozen_shared_actor",
                    mode="canonical_evaluation", evaluation=True, eval_only=True, exploration=False,
                    model_sha256=s.MODEL_SHA, profile_sha256=env.profile_hash,
                    policy_q=None, learning=[], learning_wall_seconds=0.)
                trace.append(s.plain(row))
                transitions.append((obs.copy(), action.copy(), reward, nxt.copy(), bool(terminal)))
                obs = nxt
                record = checkpoint()
                steps, new_steps = len(trace), new_steps+1
                s.save(output / "status.json", dict(status="checkpointed", control_steps=steps, latest=record))
                print(args.scenario, f"{steps}/75", f"TTT={env.sim.total_ttt:.6f}", flush=True)
                if args.max_new_steps is not None and new_steps >= args.max_new_steps and not terminal:
                    outcome = "checkpointed"
                    return outcome
            s.check_stop(output)
            preflight.live(admitted)
            checks.validate_rows(trace, transitions, settings, schema, actor)
            payload = dict(format=s.FORMAT, settings=settings, eval_only=True, transitions=transitions)
            tensor_once(output / "experience.pt", payload)
            s.save_once(output / "trace.json", checks.tag(trace)[0])
            s.save_once(output / "diagnostic-audit.json", checks.audit_receipt(output, trace, payload))
            ready, outcome = True, "completed"
        except s.Stopped:
            outcome = "stopped"
            return outcome
        except BaseException as exc:
            s.save_once(output / "failures" / f"{session}.json", dict(error=f"{type(exc).__name__}: {exc}",
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
                s.save_once(output / "sessions" / f"{session}.end.json", dict(session_id=session,
                    process=process, start_sha256=s.file_hash(start_path), outcome=outcome,
                    restoration_sha256=s.restore_accounting.files(output, session), control_steps=steps,
                    elapsed_wall_seconds=time.perf_counter()-started, elapsed_cpu_seconds=time.process_time()-cpu_started))
                s.save(output / "status.json", dict(status=outcome, session_id=session, control_steps=steps))
        if ready:
            timing = s.session_timing(output, settings, record["session_ids"])
            s.save_once(output / "finalization.json", dict(settings_digest=s.digest(settings), checkpoint=record,
                timing=timing, retained_outputs={n: s.file_hash(output / n) for n in
                    ("settings.json", "latest.json", "experience.pt", "trace.json", "diagnostic-audit.json", "observation_schema.json")}))
            s.check_stop(output)
            preflight.live(admitted)
            finalize(output, settings, actor)
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=s.SCENARIOS, required=True)
    parser.add_argument("--preflight", type=s.Path, required=True)
    parser.add_argument("--review-receipt", type=s.Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-steps", type=int)
    args = parser.parse_args()
    try:
        return 2 if run(args) == "stopped" else 0
    except s.Stopped:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
