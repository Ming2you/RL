"""One finite pilot: two training episodes, validation and three full evaluations."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from budget_runtime import DEFAULT_SNAPSHOT, HERE, REPO, read, save
from compare_runs import compare, load_completed_run
from run_budget import file_hash


def validate_child(output, name, plan):
    mode = "rl" if name == "validation" else name
    seeds = (plan["training_seeds"] if name == "train" else
             [plan["validation_seed"]] if name == "validation" else [None])
    model_hash = file_hash(output / "train/model_final.pt") if mode == "rl" else None
    return load_completed_run(output / name, mode, seeds, plan["source_pins"],
                              plan["runtime_versions"], model_hash)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    from run_budget import exclusive_run, pins, runtime_versions, verify_pins
    gate = read(args.gate)
    if not gate.get("ready_for_bounded_pilot") or gate["source_pins"] != pins(DEFAULT_SNAPSHOT):
        raise RuntimeError("Preflight gate missing or current source differs")
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    # Match the dependency search order used by child bootstrap without booting a simulator.
    sys.path.insert(0, str(REPO / ".deps-budget"))
    with exclusive_run(output):
        if (output / "completion.json").exists():
            raise RuntimeError("Pilot already complete")
        plan = dict(training_episodes=2, training_seeds=[6101, 6102], validation_seed=6103,
                    normal_reset_evaluations=["native", "center", "rl"],
                    control_steps=75, simulation_seconds=14400,
                    maximum_child_processes=3, numerical_threads_per_child=1,
                    scheduled_automation=False, source_pins=gate["source_pins"], runtime_versions=runtime_versions(),
                    timing_limitation="Separate_single_CPU_affinity_with_concurrent_jobs_not_isolated_speedup_benchmark")
        if args.resume and read(output / "plan.json") != plan:
            raise RuntimeError("Pilot plan changed")
        save(output / "plan.json", plan)
        save(output / "process.json", dict(pid=os.getpid(), command=sys.argv, started=time.time()))
        model = output / "train/model_final.pt"
        stages = [
            [("train", "train", 1, ["--training-seeds", "6101", "6102"]),
             ("native", "native", 4, []), ("center", "center", 16, [])],
            [("validation", "rl", 1, ["--model", str(model), "--validation-seed", "6103"]),
             ("rl", "rl", 4, ["--model", str(model)])],
        ]
        for index, stage in enumerate(stages):
            active = {}
            try:
                verify_pins(DEFAULT_SNAPSHOT, plan["source_pins"])
                for name, mode, cpu, extra in stage:
                    if (output / "STOP").exists():
                        save(output / "status.json", dict(status="paused", stage=index))
                        return
                    folder = output / name
                    if (folder / "completion.json").exists():
                        validate_child(output, name, plan)
                        continue
                    if (folder / "STOP").exists():
                        (output / "STOP").touch(exist_ok=True)
                        save(output / "status.json", dict(status="paused", stage=index, child=name))
                        return
                    command = [sys.executable, "-B", "-u", str(HERE / "run_budget.py"),
                               "--output", str(folder), "--mode", mode, "--cpu-mask", str(cpu), *extra]
                    if (folder / "checkpoint.pt").exists():
                        command.append("--resume")
                    stdout = (output / f"{name}.stdout.log").open("a", encoding="utf-8")
                    stderr = (output / f"{name}.stderr.log").open("a", encoding="utf-8")
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    try:
                        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr,
                                                 creationflags=subprocess.CREATE_NO_WINDOW)
                    except BaseException:
                        stdout.close()
                        stderr.close()
                        raise
                    active[name] = (child, stdout, stderr)
                    save(output / f"{name}_process.json", dict(pid=child.pid, command=command))
                while active:
                    statuses = {}
                    for name, (child, stdout, stderr) in list(active.items()):
                        code = child.poll()
                        statuses[name] = dict(pid=child.pid, returncode=code)
                        if code is None:
                            continue
                        stdout.close()
                        stderr.close()
                        del active[name]
                        if code != 0:
                            save(output / "failure.json", dict(child=name, returncode=code, stage=index))
                            # Stop siblings at the next checkpoint; never discard collected data.
                            (output / "STOP").touch(exist_ok=True)
                        else:
                            folder = output / name
                            if (not (folder / "completion.json").exists() and
                                    (folder / "status.json").exists() and
                                    read(folder / "status.json").get("status") == "paused"):
                                (output / "STOP").touch(exist_ok=True)
                            else:
                                try:
                                    validate_child(output, name, plan)
                                except (ValueError, OSError) as exc:
                                    save(output / "failure.json", dict(child=name, returncode=code,
                                         stage=index, error=str(exc)))
                                    (output / "STOP").touch(exist_ok=True)
                        print("CHILD_FINISHED", name, code, flush=True)
                    save(output / "status.json", dict(status="running", stage=index, children=statuses))
                    if active:
                        time.sleep(5)
                if (output / "STOP").exists():
                    save(output / "status.json", dict(status="paused_or_failed", stage=index))
                    return
            except Exception as exc:
                save(output / "failure.json", dict(stage=index, error=str(exc)))
                (output / "STOP").touch(exist_ok=True)
                save(output / "status.json", dict(status="failed", stage=index))
                raise
            finally:
                # Do not leave orphaned workers after an orchestrator exception.
                if active:
                    (output / "STOP").touch(exist_ok=True)
                    for child, stdout, stderr in active.values():
                        child.wait()
                        stdout.close()
                        stderr.close()
        verify_pins(DEFAULT_SNAPSHOT, plan["source_pins"])
        for name in ("train", "native", "center", "validation", "rl"):
            validate_child(output, name, plan)
        comparison = compare({name: output / name for name in ("native", "center", "rl")})
        save(output / "comparison.json", comparison)
        save(output / "completion.json", dict(status="completed", finite_pilot_complete=True,
             goal_claim=False, training_episodes=2, evaluation_runs=4))
        save(output / "status.json", dict(status="completed"))
        print("PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
