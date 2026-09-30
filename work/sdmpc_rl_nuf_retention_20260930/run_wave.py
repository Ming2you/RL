"""Explicit pilot170 first, then four local slots only after its unchanged gate passes."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from local_runtime import (HERE, REPO, GOAL, SCENARIOS, BEHAVIORS, MASKS, FORMAT,
    identity, verify_identity, seeds, read, save, file_hash, exclusive_run,
    check_stop, Stopped)
from local_runtime import (canonical_root, ensure_cohort_idle, reserve_worker, bind_launch, release_worker,
                           process_identity, process_dead, probe_process)
from validate import load_completed
from references import authenticate
from exploration import TREATMENT
from probe import PILOT, PILOT_LIMITS, pilot_report, balanced_report


def jobs_for(output):
    return [dict(key=f"{behavior}/{scenario}", behavior=behavior, scenario=scenario,
                 training_seed=seeds(scenario)[0], exploration_seed=seeds(scenario)[1], cpu_mask=MASKS[i])
            for behavior in BEHAVIORS for i, scenario in enumerate(SCENARIOS)]


def command_for(output, job, abort):
    folder = output / job["key"]
    command = [sys.executable, "-B", "-u", str(HERE / "collect.py"), "--output", str(folder),
               "--scenario", job["scenario"], "--behavior", job["behavior"],
               "--cpu-mask", str(job["cpu_mask"]), "--abort-file", str(abort)]
    if (folder / "checkpoint.pt").exists():
        from collect import check_output
        check_output(folder, resume=True)
        command.append("--resume")
    return command


def validate_job(output, job, expected):
    result, settings, _ = load_completed(output / job["key"], expected)
    for key in ("behavior", "scenario", "training_seed", "exploration_seed", "cpu_mask"):
        if settings[key] != job[key]:
            raise ValueError("Child differs from wave: " + key)
    return result


def check_all_stops(output, jobs):
    check_stop(output)
    for job in jobs:
        check_stop(output / job["key"])


def worker_exited(output, token):
    # A redirector's exit is not the actual worker's exit, including after completion.
    record = next(r for r in read(output / ".ownership/reservations.json") if r["token"] == token)
    pending = False
    for name in ("worker", "launcher"):
        process = record[name]
        if process is None:
            if name == "worker":
                raise OSError("Unclaimed actual worker identity UNKNOWN")
            continue
        if process_dead(process):
            continue
        if probe_process(process["pid"]) != ("live", process["created"]):
            raise OSError("Actual worker/launcher identity UNKNOWN")
        pending = True
    return not pending


def wait_worker_exit(output, token):
    while not worker_exited(output, token):
        time.sleep(.1)


def run_stage(output, jobs, expected, abort, all_jobs):
    active = {}
    try:
        for job in jobs:
            check_all_stops(output, all_jobs)
            verify_identity(expected)
            folder = output / job["key"]
            if (folder / "completion.json").exists():
                validate_job(output, job, expected)
                continue
            command = command_for(output, job, abort)
            token = reserve_worker(folder, job["scenario"], job["behavior"], job["cpu_mask"],
                                   launch=True, resume="--resume" in command)
            command.extend(["--reservation", token])
            out = (folder / "stdout.log").open("a", encoding="utf-8")
            err = (folder / "stderr.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            try:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=err,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except BaseException:
                out.close()
                err.close()
                # A failed spawn with no returned handle remains UNKNOWN, fail closed.
                raise
            active[job["key"]] = child, out, err, job, token
            bind_launch(token, child.pid, command)
            save(folder / "launcher.json", dict(**process_identity(child.pid, command), started=time.time(),
                                                attempt=str(abort.parent)))
        while active:
            check_all_stops(output, all_jobs)
            for key, (child, out, err, job, token) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                if code != 0:
                    # Keep this child active so finally publishes ABORT before draining it.
                    raise RuntimeError(f"Child failed/stopped: {key}, exit={code}")
                if not worker_exited(output, token):
                    continue
                out.close()
                err.close()
                release_worker(token)
                del active[key]
                validate_job(output, job, expected)
                print("COLLECTION_COMPLETED", key, flush=True)
            save(output / "status.json", dict(status="running", behavior=jobs[0]["behavior"],
                                              children={k: v[0].pid for k, v in active.items()}))
            if active:
                time.sleep(5)
    finally:
        if active:
            # A per-attempt abort is distinct from persistent user STOP markers.
            save(abort, dict(reason="coordinator_draining", created=time.time()))
            release_error = None
            for child, out, err, _, token in active.values():
                child.wait()
                out.close()
                err.close()
                try:
                    wait_worker_exit(output, token)
                    release_worker(token)
                except OSError as exc:
                    # A dead redirector does not prove its actual worker is dead.
                    release_error = exc
            if release_error is not None:
                raise release_error


def select_stage(jobs, stage, pilot):
    if stage == "pilot":
        return [j for j in jobs if j["scenario"] == PILOT]
    if stage != "remaining" or not pilot or not pilot["integrity_pass"] or not pilot["advance_allowed"]:
        raise ValueError("Remaining four require a completed, unchanged pilot passing every criterion")
    return [j for j in jobs if j["scenario"] != PILOT]


def run(output, resume=False, stage="pilot"):
    output = Path(output).resolve()
    canonical_root(output)
    check_stop(output)
    if output.exists() and not resume:
        raise FileExistsError("Use --resume for existing wave/checkpoint")
    with exclusive_run(output):
        check_stop(output)
        if (output / "completion.json").exists():
            raise ValueError("Wave already completed")
        jobs = jobs_for(output)
        check_all_stops(output, jobs)
        source = identity()
        reference_started = time.perf_counter()
        references = authenticate()
        reference_seconds = time.perf_counter()-reference_started
        plan = dict(format=FORMAT, jobs=jobs, identity=source, treatment=TREATMENT,
                    paired_references=references, maximum_workers_by_stage=dict(pilot=1, remaining=4),
                    global_worker_ceiling=8, threads_per_worker=1, episodes=5,
                    pilot_transitions=75, conditional_additional_transitions=300,
                    transitions=375, pilot_limits=PILOT_LIMITS, training_only=True,
                    future_minibatch_scenario_fractions={s: .2 for s in SCENARIOS}, learner_admitted=False)
        if (output / "plan.json").exists() and read(output / "plan.json") != plan:
            raise ValueError("Wave plan/source changed")
        # Detect orphaned live workers after coordinator loss before dispatching any.
        ensure_cohort_idle(output)
        with ExitStack() as locks:
            for job in jobs:
                locks.enter_context(exclusive_run(output / job["key"]))
        for job in jobs:
            if (output / job["key"] / "completion.json").exists():
                validate_job(output, job, source)
        save(output / "plan.json", plan)
        attempt = output / "attempts" / uuid.uuid4().hex
        abort = attempt / "ABORT.json"
        process = dict(**process_identity(), started=time.time(), attempt=str(attempt), stage=stage)
        save(attempt / "process.json", process)
        save(output / "process.json", process)
        save(attempt / "verification.json", dict(reference_authentication_wall_seconds=reference_seconds,
            reference_root_sha256=references["root_sha256"],
            scope="Read-only original-validator subprocess plus reference hash reconciliation; "
                  "separate from worker sessions and subsequent coordinator attempt"))
        coordinator_started = time.perf_counter()
        try:
            pilot = None
            if stage == "remaining":
                if not (output / "local" / PILOT / "completion.json").exists():
                    raise ValueError("Completed pilot required before remaining stage")
                pilot = pilot_report(output, source, references)
                save(attempt / "pilot.json", pilot)
                save(output / "pilot.json", pilot)
            selected = select_stage(jobs, stage, pilot)
            run_stage(output, selected, source, abort, jobs)
            check_all_stops(output, jobs)
            verify_identity(source)
            if stage == "pilot":
                pilot = pilot_report(output, source, references)
                save(attempt / "pilot.json", pilot)
                save(output / "pilot.json", pilot)
                status = "pilot_passed" if pilot["advance_allowed"] else "pilot_blocked"
                save(output / "status.json", dict(status=status))
                return status
            comparison = balanced_report(output, source, references)
            save(output / "comparison.json", comparison)
            check_all_stops(output, jobs)
            save(output / "status.json", dict(status="completed"))
            save(output / "completion.json", dict(format=FORMAT, status="completed", plan=plan,
                comparison_sha256=file_hash(output / "comparison.json"),
                child_completion_sha256={j["key"]: file_hash(output / j["key"] / "completion.json") for j in jobs},
                all_integrity_and_screens_pass=comparison["all_integrity_and_screens_pass"],
                learner_admitted=False, goal_claim=False))
            print("NUF_RETENTION_COLLECTION_COMPLETE", comparison["all_integrity_and_screens_pass"], flush=True)
            return "completed"
        except Stopped:
            save(output / "status.json", dict(status="stopped"))
            return "stopped"
        except BaseException as exc:
            save(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
            save(attempt / "failure.json", dict(error=f"{type(exc).__name__}: {exc}"))
            if isinstance(exc, (ValueError, AssertionError)):
                save(attempt / "integrity-failure.json", dict(error=f"{type(exc).__name__}: {exc}",
                    identity=source, reference_root_sha256=references["root_sha256"],
                    attribution_valid=False, performance_claim=False, remaining_dispatch_allowed=False))
            raise
        finally:
            save(attempt / "timing.json", dict(elapsed_wall_seconds=time.perf_counter()-coordinator_started,
                scope="Coordinator attempt from process publication through dispatch, child waits and readout; "
                      "overlaps worker sessions, excludes reference preflight and interpreter startup; not additive"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=GOAL / "nuf_retention_v1")
    p.add_argument("--stage", choices=("pilot", "remaining"), required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    return 2 if run(args.output, args.resume, args.stage) in ("stopped", "pilot_blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
