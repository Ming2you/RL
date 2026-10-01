"""Two finite five-worker training waves; no actor training or evaluation."""
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
from local_runtime import canonical_root, ensure_cohort_idle, reserve_worker, bind_launch, release_worker
from validate import load_completed, compare_pairs


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
            save(folder / "launcher.json", dict(pid=child.pid, command=command, started=time.time(),
                                                attempt=str(abort.parent)))
        while active:
            check_all_stops(output, all_jobs)
            for key, (child, out, err, job, token) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                out.close()
                err.close()
                release_worker(token)
                del active[key]
                if code != 0:
                    raise RuntimeError(f"Child failed/stopped: {key}, exit={code}")
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
            for child, out, err, _, token in active.values():
                child.wait()
                out.close()
                err.close()
                release_worker(token)


def run(output, resume=False):
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
        plan = dict(format=FORMAT, jobs=jobs, identity=source, maximum_workers=5,
                    threads_per_worker=1, episodes=10, transitions=750, training_only=True)
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
        process = dict(pid=os.getpid(), command=sys.argv, started=time.time(), attempt=str(attempt))
        save(attempt / "process.json", process)
        save(output / "process.json", process)
        try:
            for behavior in BEHAVIORS:
                run_stage(output, [j for j in jobs if j["behavior"] == behavior], source, abort, jobs)
            check_all_stops(output, jobs)
            verify_identity(source)
            comparison = compare_pairs(output, source)
            save(output / "comparison.json", comparison)
            check_all_stops(output, jobs)
            save(output / "status.json", dict(status="completed"))
            save(output / "completion.json", dict(format=FORMAT, status="completed", plan=plan,
                comparison_sha256=file_hash(output / "comparison.json"),
                child_completion_sha256={j["key"]: file_hash(output / j["key"] / "completion.json") for j in jobs},
                coverage_screen_pass=comparison["coverage_screen_pass"], goal_claim=False))
            print("LOCAL_TRAINING_WAVE_COMPLETE", comparison["coverage_screen_pass"], flush=True)
            return "completed"
        except Stopped:
            save(output / "status.json", dict(status="stopped"))
            return "stopped"
        except BaseException as exc:
            save(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
            save(attempt / "failure.json", dict(error=f"{type(exc).__name__}: {exc}"))
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=GOAL / "local_budget_v1")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    return 2 if run(args.output, args.resume) == "stopped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
