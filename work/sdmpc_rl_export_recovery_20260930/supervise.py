"""Bounded companion: export known terminal failures, resume immutable v2."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import export_recovery as recovery
import coordinator_bootstrap as bootstrap

MAX_REPAIRS = 10
MAX_WAVES = 11


def inspect(lr, validator, wave, root):
    """No arbitrary retry: reconcile the exact primary error and drained siblings."""
    records, attempt = recovery.prove_dead(lr, wave, root)
    source = recovery.authenticate(lr, wave, root)
    jobs = wave.jobs_for(root)
    receipts = {}
    for job in jobs:
        folder = root / job["key"]
        if (folder / "completion.json").exists():
            result = wave.validate_job(root, job, source)
            if "export_recovery" in result:
                receipts[job["key"]] = recovery.verify_complete(lr, validator, folder, source)
    status = lr.read(root / "status.json")
    if (root / "completion.json").exists():
        done = lr.read(root / "completion.json")
        if (status != {"status": "completed"} or done["status"] != "completed" or
                done["plan"] != lr.read(root / "plan.json") or
                done["child_completion_sha256"] != {j["key"]: lr.file_hash(root / j["key"] / "completion.json") for j in jobs} or
                done["comparison_sha256"] != lr.file_hash(root / "comparison.json") or
                lr.read(root / "comparison.json") != validator.compare_pairs(root, source)):
            raise ValueError("Wave completion differs")
        return True, []
    primary = next((j["key"] for j in jobs if status == dict(status="failed",
        error=f"RuntimeError: Child failed/stopped: {j['key']}, exit=1")), None)
    if primary is None or lr.read(attempt / "failure.json") != {"error": status["error"]}:
        raise ValueError("Coordinator failure is not the known recoverable child failure")
    repairs = []
    for job in jobs:
        folder = root / job["key"]
        history = [r for r in records if r["key"] == job["key"]]
        if (folder / "completion.json").exists():
            if job["key"] in receipts:
                receipt = receipts[job["key"]]
                if job["key"] == primary and receipt["coordinator_attempt"] != str(attempt):
                    raise ValueError("Primary repaired failure belongs to another attempt")
            elif job["key"] == primary:
                raise ValueError("Primary failure lacks an export receipt")
            continue
        if (folder / recovery.STAGE).exists():
            raise FileExistsError("Interrupted export; preserve staging for review")
        if not history:
            # Unstarted canonical slots are left to the original runner.
            if folder.exists() and any(p.name != "runner.lock" for p in folder.iterdir()):
                raise ValueError("Unowned slot artifacts")
            continue
        record = history[-1]
        if not (folder / "checkpoint.pt").is_file():
            raise ValueError("Missing retained checkpoint")
        if lr.read(folder / "settings.json")["run_id"] != record["run_id"]:
            raise ValueError("Retained checkpoint run identity differs")
        child = lr.read(folder / "status.json")
        if child.get("status") == "failed":
            recovery.known_failure(lr, folder, record)
            if record["control_steps"] != 75:
                raise ValueError("Failure is not terminal")
            repairs.append(job["key"])
        elif child.get("status") == "stopped":
            abort = attempt / "ABORT.json"
            command = record["worker"]["command"]
            if (not abort.exists() or lr.read(abort).get("reason") != "coordinator_draining" or
                    "--abort-file" not in command or
                    Path(command[command.index("--abort-file") + 1]).resolve() != abort):
                raise ValueError("Stopped worker is not a drained sibling of this attempt")
        else:
            raise ValueError("Unexpected incomplete worker status")
        if job["key"] == primary and job["key"] not in repairs:
            raise ValueError("Primary child failure is not recoverable")
    if primary not in repairs and not (root / primary / "completion.json").exists():
        raise ValueError("Primary checkpoint missing")
    return False, repairs


def launched(lr, wave, root, folder, state, command, kind):
    """Persist launch intent first. An interrupted/ambiguous launch is never retried."""
    recovery.check_stops(lr, wave, root)
    if state["helper_sha256"] != recovery.helper_hashes(lr):
        raise ValueError("Supervisor helper source changed")
    attempt = dict(id=uuid.uuid4().hex, kind=kind, command=command, phase="launching", started=time.time())
    path = folder / "attempts" / (attempt["id"] + ".json")
    if kind == "wave":
        attempt.update(runner_command=command, owner=lr.process_identity(),
                       helper_sha256=state["helper_sha256"], source=state["source"])
        command = [sys.executable, "-B", "-u", str(recovery.HERE / "coordinator_bootstrap.py"), "--attempt", str(path)]
        attempt["command"] = command
    state["attempts"].append(attempt)
    lr.durable_save(folder / "state.json", state)
    lr.durable_save(path, attempt)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (folder / "stdout.log").open("a", encoding="utf-8") as out, (folder / "stderr.log").open("a", encoding="utf-8") as err:
        out.write(f"\n{attempt['id']} {kind} {command!r}\n")
        out.flush()
        child = subprocess.Popen(command, cwd=lr.REPO, env=env, stdout=out, stderr=err,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            identity = lr.process_identity(child.pid, command)
            attempt.update(phase="running", process=identity)
            lr.durable_save(path, attempt)
            lr.durable_save(folder / "state.json", state)
            if kind == "wave":
                binding_file = path.with_suffix(".coordinator.json")
                for _ in range(201):
                    recovery.check_stops(lr, wave, root)
                    if binding_file.exists():
                        binding = lr.read(binding_file)
                        bootstrap.check_binding(lr, attempt, binding, live=True)
                        attempt.update(coordinator=binding["actual"], binding_sha256=lr.file_hash(binding_file))
                        lr.durable_save(path, attempt)
                        lr.durable_save(folder / "state.json", state)
                        break
                    if child.poll() is not None:
                        raise OSError("Coordinator exited without identity binding")
                    if not binding_file.exists():
                        time.sleep(.05)
                else:
                    raise OSError("Coordinator creation identity UNKNOWN")
        finally:
            # Even a bookkeeping failure after spawn must wait for the launched handle.
            code = child.wait()
    recovery.require_dead(lr, identity)
    if kind == "wave":
        process = lr.read(root / "process.json")
        if (process["pid"] != attempt["coordinator"]["pid"] or
                process["command"] != attempt["runner_command"][3:] or process["started"] < attempt["started"]):
            raise OSError("Coordinator attempt identity UNKNOWN")
        attempt["coordinator_process"] = process
        lr.durable_save(path, attempt)
        lr.durable_save(folder / "state.json", state)
        while True:
            try:
                # Wait for known actual identities even if their launcher already exited.
                # STOP prevents continuation, but must not bypass this draining proof.
                recovery.prove_dead(lr, wave, root, honor_stop=False)
                with lr.exclusive_run(root):
                    recovery.prove_dead(lr, wave, root, honor_stop=False)
                break
            except recovery.LiveProcess:
                time.sleep(1)
            # UNKNOWN and inaccessible locks fail closed, never dispatch a duplicate.
    attempt.update(phase="exited", exit_code=code, ended=time.time())
    lr.durable_save(path, attempt)
    lr.durable_save(folder / "state.json", state)
    recovery.check_stops(lr, wave, root)
    return code


def supervise():
    lr, _, validator, wave = recovery.modules()
    root = lr.canonical_root(lr.COHORT_ROOT)
    recovery.check_stops(lr, wave, root)
    folder = root / ".export-supervisor"
    with lr.exclusive_run(folder):
        state_path = folder / "state.json"
        source = recovery.authenticate(lr, wave, root)
        state = (lr.read(state_path) if state_path.exists() else
                 dict(source=source, helper_sha256=recovery.helper_hashes(lr), repairs=[], attempts=[], status="ready"))
        if (state["source"] != source or state["helper_sha256"] != recovery.helper_hashes(lr) or
                state["status"] not in ("ready", "completed")):
            raise ValueError("Interrupted/changed supervisor; preserve state for review")
        if (len(state["repairs"]) > MAX_REPAIRS or len(set(state["repairs"])) != len(state["repairs"]) or
                not set(state["repairs"]) <= {j["key"] for j in wave.jobs_for(root)}):
            raise ValueError("Invalid durable repair bound")
        process = lr.process_identity()
        if type(process.get("created")) is not int or lr.probe_process(process["pid"]) != ("live", process["created"]):
            raise OSError("Supervisor creation identity UNKNOWN")
        state.update(status="running", process=process)
        lr.durable_save(state_path, state)
        try:
            while True:
                with ExitStack() as locks:
                    locks.enter_context(lr.exclusive_run(root))
                    completed, repairs = inspect(lr, validator, wave, root)
                if completed:
                    state["status"] = "completed"
                    lr.durable_save(state_path, state)
                    return "completed"
                for slot in repairs:
                    if slot in state["repairs"] or len(state["repairs"]) >= MAX_REPAIRS:
                        raise ValueError("Terminal repair bound/once-per-slot exceeded")
                    state["repairs"].append(slot)
                    lr.durable_save(state_path, state)
                    code = launched(lr, wave, root, folder, state,
                        [sys.executable, "-B", "-u", str(recovery.HERE / "export_recovery.py"),
                         "--slot", slot, "--execute"], "export")
                    if code != 0:
                        raise RuntimeError("Export failed; no automatic retry")
                    recovery.verify_complete(lr, validator, root / slot, source)
                if sum(a["kind"] == "wave" for a in state["attempts"]) >= MAX_WAVES:
                    raise ValueError("Bounded coordinator continuation exhausted")
                recovery.check_stops(lr, wave, root)
                lr.verify_identity(source)
                previous = lr.file_hash(root / "process.json")
                code = launched(lr, wave, root, folder, state,
                    bootstrap.runner_command(root), "wave")
                if lr.file_hash(root / "process.json") == previous:
                    raise ValueError("Launched coordinator did not record a new attempt")
                if code == 0 and not (root / "completion.json").exists():
                    raise ValueError("Coordinator exited without completion")
                if code not in (0, 1):
                    raise RuntimeError("Coordinator stopped or exited unexpectedly")
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            lr.durable_save(state_path, state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Continue only after parent review")
    if not parser.parse_args(argv).execute:
        parser.error("No continuation performed; explicit --execute is required")
    supervise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
