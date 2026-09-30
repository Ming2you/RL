"""Finite five-worker collection waves and one centralized balanced learner."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
from budget_runtime import DEFAULT_SNAPSHOT, HERE, REPO, read, save
from run_budget import exclusive_run, file_hash, pins, runtime_versions, verify_pins


def jobs_for(output, scenarios):
    jobs = []
    masks = (1, 4, 16, 64, 256)
    for round_index in (0, 1):
        model = None if round_index == 0 else output / "train_round0/model_final.pt"
        for i, scenario in enumerate(scenarios):
            jobs.append(dict(key=f"collect_round{round_index}/{scenario}", kind="collect",
                stage=round_index*2, scenario=scenario, round=round_index,
                seed=(6301 if round_index == 0 else 6401)+i, cpu_mask=masks[i],
                model=None if model is None else str(model)))
        jobs.append(dict(key=f"train_round{round_index}", kind="train", stage=round_index*2+1,
                         round=round_index, model=None if model is None else str(model)))
    for kind, stage in (("center", 4), ("rl", 5)):
        for i, scenario in enumerate(scenarios):
            jobs.append(dict(key=f"{kind}/{scenario}", kind=kind, stage=stage,
                scenario=scenario, round=1 if kind == "rl" else 0, seed=None,
                cpu_mask=masks[i], model=str(output / "train_round1/model_final.pt") if kind == "rl" else None))
    return jobs


def validate_training(folder, job, plan, output):
    from td3 import SCENARIOS
    from train_round import validate_training_output
    return validate_training_output(folder, job["round"], plan["source_pins"], plan["runtime_versions"],
        job["model"], [output / f"collect_round{job['round']}" / s for s in SCENARIOS])


def validate_job(output, job, plan):
    folder = output / job["key"]
    if job["kind"] == "train":
        return validate_training(folder, job, plan, output)
    from compare_runs import load_completed_run
    model_hash = None if job["model"] is None else file_hash(job["model"])
    result, settings, _, _ = load_completed_run(folder, job["kind"], [job["seed"]],
        plan["source_pins"], plan["runtime_versions"], model_hash, job["scenario"])
    if (settings["round"] != job["round"] or settings["model_sha256"] != model_hash or
            settings["cpu_mask"] != job["cpu_mask"] or settings["policy_seed"] != 6300):
        raise ValueError("Completed worker differs from pilot plan")
    return result


def command_for(output, job, scenarios):
    folder = output / job["key"]
    if job["kind"] == "train":
        command = [sys.executable, "-B", "-u", str(HERE / "train_round.py"),
            "--output", str(folder), "--round", str(job["round"]), "--collections",
            *[str(output / f"collect_round{job['round']}" / s) for s in scenarios]]
    else:
        command = [sys.executable, "-B", "-u", str(HERE / "run_budget.py"),
            "--output", str(folder), "--mode", job["kind"], "--scenario", job["scenario"],
            "--cpu-mask", str(job["cpu_mask"]), "--round", str(job["round"])]
        if job["seed"] is not None:
            command += ["--training-seed", str(job["seed"])]
    if job["model"] is not None:
        command += ["--model", job["model"]]
    if (folder / "checkpoint.pt").exists():
        command.append("--resume")
    return command


def request_stop(output, reason):
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output / "STOP").open("x", encoding="utf-8") as stream:
            stream.write("pilot_coordinator: " + reason + "\n")
    except FileExistsError:
        pass


def propagate_stop(output, jobs):
    if any((p / "STOP").exists() for job in jobs for p in
           (output, output / job["key"], (output / job["key"]).parent)):
        request_stop(output, "child_or_parent_stop")
        return True
    return False


def ensure_idle(folders):
    # Check all output locks before launching any sibling, including after parent loss.
    with ExitStack() as stack:
        for folder in folders:
            stack.enter_context(exclusive_run(folder))


def run_stage(output, stage, jobs, plan, scenarios):
    active = {}
    stop_jobs = plan.get("jobs", jobs)
    try:
        if propagate_stop(output, stop_jobs):
            return False
        for job in jobs:
            folder = output / job["key"]
            if (folder / "completion.json").exists():
                validate_job(output, job, plan)
        ensure_idle([output / job["key"] for job in jobs])
        for job in jobs:
            if propagate_stop(output, stop_jobs):
                break
            folder = output / job["key"]
            if (folder / "completion.json").exists():
                continue
            command = command_for(output, job, scenarios)
            out = (folder / "stdout.log").open("a", encoding="utf-8")
            err = (folder / "stderr.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            try:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            except BaseException:
                out.close()
                err.close()
                raise
            active[job["key"]] = (child, out, err, job)
            save(folder / "launcher.json", dict(pid=child.pid, command=command, started=time.time()))
        while active:
            for key, (child, out, err, job) in list(active.items()):
                propagate_stop(output, stop_jobs)
                code = child.poll()
                propagate_stop(output, stop_jobs)
                if code is None:
                    continue
                out.close()
                err.close()
                del active[key]
                folder = output / key
                if code != 0:
                    save(output / "failure.json", dict(stage=stage, child=key, returncode=code))
                    request_stop(output, "worker_failure")
                elif not (folder / "completion.json").exists():
                    request_stop(output, "worker_paused_or_incomplete")
                else:
                    validate_job(output, job, plan)
                print("CHILD_FINISHED", key, code, flush=True)
            save(output / "status.json", dict(status="draining" if (output / "STOP").exists() else "running",
                stage=stage, children={k: v[0].pid for k, v in active.items()}))
            if active:
                time.sleep(5)
        return not (output / "STOP").exists()
    except BaseException as exc:
        save(output / "failure.json", dict(stage=stage, error=str(exc)))
        request_stop(output, "coordinator_failure")
        save(output / "status.json", dict(status="failed", stage=stage))
        raise
    finally:
        if active:
            request_stop(output, "draining_active_workers")
            for child, out, err, _ in active.values():
                child.wait()
                out.close()
                err.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(REPO / ".deps-budget"))
    from td3 import SCENARIOS
    from build_preflight import verify_gate
    gate = verify_gate(args.gate)
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    with exclusive_run(output):
        if (output / "completion.json").exists():
            raise ValueError("Pilot already complete")
        jobs = jobs_for(output, SCENARIOS)
        plan = dict(format="sdmpc-multi-pilot-v1", scenarios=list(SCENARIOS), jobs=jobs,
            source_pins=gate["source_pins"], runtime_versions=runtime_versions(),
            gate_sha256=file_hash(args.gate), maximum_numerical_workers=5, threads_per_worker=1,
            batch_size=40, samples_per_scenario_per_update=8, updates_per_new_transition=1,
            scheduled_automation=False, evaluation_scope="five_training_scenarios_not_generalization")
        if args.resume and read(output / "plan.json") != plan:
            raise ValueError("Pilot plan changed")
        if propagate_stop(output, jobs):
            save(output / "status.json", dict(status="paused"))
            return
        ensure_idle([output / job["key"] for job in jobs])
        # Reject stale completed work before starting any new experiment.
        for job in jobs:
            if (output / job["key"] / "completion.json").exists():
                validate_job(output, job, plan)
        save(output / "plan.json", plan)
        save(output / "process.json", dict(pid=os.getpid(), command=sys.argv, started=time.time()))
        for stage in range(6):
            verify_pins(DEFAULT_SNAPSHOT, plan["source_pins"])
            if not run_stage(output, stage, [j for j in jobs if j["stage"] == stage], plan, SCENARIOS):
                save(output / "status.json", dict(status="paused_or_failed", stage=stage))
                return
        from compare_runs import compare
        comparisons = [dict(scenario=s, **compare({kind: output / kind / s for kind in ("center", "rl")}))
                       for s in SCENARIOS]
        model_hash = file_hash(output / "train_round1/model_final.pt")
        if any(row["model_sha256"] != model_hash for c in comparisons for row in c["rows"] if row["mode"] == "rl"):
            raise ValueError("Evaluations do not use the same frozen model")
        verify_pins(DEFAULT_SNAPSHOT, plan["source_pins"])
        save(output / "comparison.json", dict(status="reconciled", comparisons=comparisons,
            shared_model_sha256=model_hash, generalization_claim=False, improvement_claim=False))
        save(output / "completion.json", dict(status="completed", finite_pilot_complete=True,
            training_episodes=10, evaluation_episodes=10, shared_model_sha256=model_hash,
            goal_claim=False, controller_acceptance=False))
        save(output / "status.json", dict(status="completed"))
        print("MULTI_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
