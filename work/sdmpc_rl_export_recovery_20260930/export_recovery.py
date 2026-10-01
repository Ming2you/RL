"""Terminal JSON export adapter for the frozen local-budget v2 cohort."""
import argparse
from contextlib import ExitStack
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2 = REPO / "work/sdmpc_rl_local_20260930_v2"
ERROR = "ValueError: Out of range float values are not JSON compliant: inf"
OUTPUTS = ("trace.json", "summary.json", "experience.pt", "observation_schema.json", "timing.json")
STAGE = ".export-recovery"
RECEIPT = "export-receipt.json"
EXPORT_SCOPE = (
    "This successful invocation: from entry before cohort/slot locks through identity and "
    "checkpoint validation, class-registration boot, serialization, original completed-loader "
    "validation, output publication and retained-cohort finish bookkeeping. Excludes receipt "
    "and completion publication, lock release, interpreter startup/teardown and supervisor "
    "time. Separate additional cost; original worker-session timing is unchanged."
)


class LiveProcess(OSError):
    pass


def modules():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(V2))
    loaded = tuple(importlib.import_module(n) for n in ("local_runtime", "collect", "validate", "run_wave"))
    for module in loaded:
        if Path(module.__file__).resolve().parent != V2:
            raise ValueError("Mixed collector modules")
    return loaded


def tag_diagnostics(trace):
    """Copy JSON structure; only typed list-index candidate paths admit +inf."""
    changes = []

    def walk(value, path):
        if type(value) is dict:
            if "nonfinite_float" in value or any(type(k) is not str for k in value):
                raise ValueError("Pre-tagged or non-JSON trace object")
            return {k: walk(v, path + [k]) for k, v in value.items()}
        if type(value) is list:
            return [walk(v, path + [i]) for i, v in enumerate(value)]
        if type(value) is float and not math.isfinite(value):
            candidate = (len(path) in (4, 6) and type(path[0]) is int and
                         path[1] == "candidates" and type(path[2]) is int)
            allowed = candidate and (
                len(path) == 4 and path[3] == "stationarity" or
                len(path) == 6 and path[3] == "rows" and type(path[4]) is int and
                path[5] == "primal_stationarity")
            if not allowed or value != math.inf:
                raise ValueError(f"Forbidden nonfinite trace value at {path}")
            tag = {"nonfinite_float": "+inf"}
            changes.append(dict(path=path, original_value="+inf", original_type="float", exported=tag))
            return tag.copy()
        if value is not None and type(value) not in (str, bool, int, float):
            raise TypeError(f"Non-JSON trace value at {path}")
        return value

    if type(trace) is not list:
        raise TypeError("Trace must be a list")
    return walk(trace, []), changes


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def publish_bytes(path, data):
    """Exclusive publication; an interrupted pending file is evidence, never reused."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)  # Atomic, refuses an existing destination on Windows and POSIX.
    pending.unlink()


def check_stops(lr, wave, root):
    lr.check_stop(root)
    for job in wave.jobs_for(root):
        lr.check_stop(root / job["key"])


def require_dead(lr, record):
    if (not isinstance(record, dict) or type(record.get("pid")) is not int or not 0 < record["pid"] <= 0xffffffff or
            type(record.get("created")) is not int or record["created"] <= 0):
        raise OSError("Missing creation identity; process state UNKNOWN")
    state, created = lr.probe_process(record["pid"])
    if state == "dead" or (state == "live" and type(created) is int and created != record["created"]):
        return
    if state == "live" and created == record["created"]:
        raise LiveProcess("Process live")
    raise OSError("Process live or UNKNOWN")


def prove_dead(lr, wave, root, honor_stop=True):
    """Called under the cohort lock; do not trust an exited label alone."""
    if honor_stop:
        check_stops(lr, wave, root)
    records = lr.read(root / ".ownership/reservations.json")
    keys = {j["key"] for j in wave.jobs_for(root)}
    if not records:
        raise ValueError("Missing retained cohort identities")
    for record in records:
        if record["key"] not in keys or record["state"] not in ("reserved", "active", "exited", "completed"):
            raise ValueError("Unknown cohort reservation")
        require_dead(lr, record["owner"])
        require_dead(lr, record["worker"])
        if record["launcher"] is not None:
            require_dead(lr, record["launcher"])
        elif record["state"] == "reserved":
            raise OSError("Unbound launch UNKNOWN")
    process = lr.read(root / "process.json")
    attempt = Path(process["attempt"]).resolve()
    if attempt.parent != root / "attempts" or lr.read(attempt / "process.json") != process:
        raise ValueError("Coordinator attempt identity differs")
    owners = [r["owner"] for r in records if r["owner"]["pid"] == process["pid"] and
              r["owner"]["command"] == process["command"]]
    from coordinator_bootstrap import prove_bound_dead
    bound = prove_bound_dead(lr, root, process)
    if not bound and (not owners or len({o["created"] for o in owners}) != 1):
        raise OSError("Coordinator creation identity UNKNOWN")
    for key in {r["key"] for r in records}:
        current = [r for r in records if r["key"] == key][-1]
        folder = root / key
        worker = lr.read(folder / "process.json")
        if any(worker.get(k) != current["worker"].get(k) for k in ("pid", "created", "command")):
            raise OSError("Actual worker identity UNKNOWN")
        if current["launcher"] is not None:
            launcher = lr.read(folder / "launcher.json")
            if any(launcher.get(k) != current["launcher"].get(k) for k in ("pid", "command")):
                raise OSError("Launcher identity UNKNOWN")
    return records, attempt


def known_failure(lr, folder, record):
    status = lr.read(folder / "status.json")
    if status != dict(status="failed", pid=record["worker"]["pid"], error=ERROR):
        raise ValueError("Not the exact terminal diagnostic JSON failure")
    log = (folder / "stderr.log").read_text(encoding="utf-8-sig")
    last = log.rsplit("Traceback (most recent call last):", 1)[-1]
    fragments = (f'File "{V2 / "collect.py"}", line 155, in run',
                 'save(output / "trace.json", trace)',
                 f'File "{V2 / "local_runtime.py"}", line 42, in save',
                 "json.dumps(plain(value), allow_nan=False)")
    if "Traceback (most recent call last):" not in log or not all(s in last for s in fragments) or last.strip().splitlines()[-1] != ERROR:
        raise ValueError("Failure traceback is not the frozen trace serialization boundary")
    return status


def authenticate(lr, wave, root):
    source = lr.identity()
    plan = dict(format=lr.FORMAT, jobs=wave.jobs_for(root), identity=source, maximum_workers=5,
                threads_per_worker=1, episodes=10, transitions=750, training_only=True)
    if lr.read(root / "plan.json") != plan:
        raise ValueError("Wave plan/source changed")
    return source


def hashes(lr, folder, names):
    return {name: lr.file_hash(folder / name) for name in names}


def verify_complete(lr, validator, folder, source):
    completion = lr.read(folder / "completion.json")
    if completion.get("export_recovery", {}).get("receipt") != RECEIPT:
        raise ValueError("Completion is not this adapter's export")
    receipt = lr.read(folder / RECEIPT)
    if completion["export_recovery"]["sha256"] != lr.file_hash(folder / RECEIPT):
        raise ValueError("Export receipt changed")
    if receipt["helper_sha256"] != helper_hashes(lr):
        raise ValueError("Export helper source changed")
    if lr.read(folder / "status.json") != dict(status="export-completed", control_steps=75,
            export_receipt_sha256=completion["export_recovery"]["sha256"]):
        raise ValueError("Export current status changed")
    inputs = dict(receipt["input_sha256"])
    original = inputs.pop("status.json")
    if (hashes(lr, folder, inputs) != inputs or
            lr.file_hash(folder / STAGE / "original-status.json") != original or
            hashes(lr, folder, OUTPUTS) != receipt["output_sha256"] or
            completion["outputs_sha256"] != receipt["output_sha256"]):
        raise ValueError("Export inputs/outputs changed")
    _, settings, _ = validator.load_completed(folder, source)
    slot = f"{settings['behavior']}/{settings['scenario']}"
    records = lr.read(lr.COHORT_ROOT / ".ownership/reservations.json")
    retained = [r for r in records if r["token"] == receipt["token"]]
    history = [r for r in records if r["key"] == slot]
    if (Path(folder).resolve() != lr.COHORT_ROOT.resolve() / slot or receipt["slot"] != slot or
            receipt["run_id"] != settings["run_id"] or len(retained) != 1 or not history or
            retained[0] != history[-1] or retained[0]["key"] != slot or
            retained[0]["run_id"] != settings["run_id"] or
            retained[0]["control_steps"] != 75 or retained[0]["cohort_phase"] != "finished"):
        raise ValueError("Completed export retained identity differs")
    return receipt


def helper_hashes(lr):
    return hashes(lr, HERE, ("export_recovery.py", "supervise.py", "coordinator_bootstrap.py"))


def export(folder):
    started = time.perf_counter()
    lr, collector, validator, wave = modules()
    root, folder = lr.COHORT_ROOT.resolve(), Path(folder).resolve()
    lr.canonical_root(root)
    job = next((j for j in wave.jobs_for(root) if root / j["key"] == folder), None)
    if job is None:
        raise ValueError("Export requires a canonical v2 slot")
    check_stops(lr, wave, root)
    with ExitStack() as locks:
        locks.enter_context(lr.exclusive_run(root))
        locks.enter_context(lr.exclusive_run(folder))
        records, attempt = prove_dead(lr, wave, root)
        source = authenticate(lr, wave, root)
        if (folder / "completion.json").exists():
            return verify_complete(lr, validator, folder, source)
        stage = folder / STAGE
        if stage.exists() or (folder / RECEIPT).exists():
            raise FileExistsError("Interrupted export; preserve staging for review")
        for name in ("trace.json", "summary.json", "experience.pt"):
            if (folder / name).exists():
                raise FileExistsError("Conflicting preexisting output: " + name)
        retained = [r for r in records if r["key"] == job["key"]][-1]
        failure = known_failure(lr, folder, retained)
        names = ["checkpoint.pt", "settings.json", "status.json", "timing.json", "observation_schema.json",
                 "process.json", "stderr.log", "stdout.log"]
        names += [p.relative_to(folder).as_posix() for p in sorted((folder / "sessions").glob("*.json"))]
        if (folder / "launcher.json").exists():
            names.append("launcher.json")
        inputs = hashes(lr, folder, names)
        helpers = helper_hashes(lr)
        settings = lr.read(folder / "settings.json")
        if (settings["identity"] != source or settings["contract"] != lr.contract() or
                any(settings[k] != job[k] for k in ("behavior", "scenario", "cpu_mask", "training_seed", "exploration_seed")) or
                settings["run_id"] != retained["run_id"] or retained["control_steps"] != 75 or
                settings["evaluation"] is not False or settings["model_sha256"] is not None):
            raise ValueError("Terminal slot/settings/cohort identity differs")
        # Boot registers frozen pickle classes. Never construct, restore or step an environment.
        lr.boot(lr.DEFAULT_SNAPSHOT, job["cpu_mask"])
        ck = lr.torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        if len(ck["trace"]) != 75 or len(ck["transitions"]) != 75 or ck["environment"]["k"] != 80:
            raise ValueError("Need the saved 75-step terminal checkpoint")
        transformed, changes = tag_diagnostics(ck["trace"])
        if not changes:
            raise ValueError("No allowlisted diagnostic infinity to repair")
        env = ck["environment"]
        physical = {k: env[k] for k in ("previous", "last_budget", "last_requested", "last_executed", "warmup_ttt")}
        physical.update(state=env["sim"].state, total_ttt=env["sim"].total_ttt,
                        freeway_ttt=env["sim"].freeway_ttt, urban_ttt=env["sim"].urban_ttt)
        json_bytes(lr.plain(dict(physical=physical, transitions=ck["transitions"],
                               observation=ck["observation"], settings=settings, policy=ck["policy"])))
        collector.validate_checkpoint(ck, settings)
        validator.validate_rows(ck["trace"], ck["transitions"], settings, ck["schema"], settings["capacity"])
        timing = lr.read(folder / "timing.json")
        required = sorted(set(ck["session_ids"]) | {r["token"] for r in records if r["key"] == job["key"]})
        if not required or timing != lr.session_timing(folder, required):
            raise ValueError("Original timing/session reconciliation differs")
        if lr.read(folder / "observation_schema.json") != ck["schema"]:
            raise ValueError("Original schema differs")
        summary = lr.plain(validator.summarize(ck["trace"], settings, timing["timing_status"]))
        exported_ck = dict(ck, trace=transformed)
        collector.validate_checkpoint(exported_ck, settings)
        validator.validate_rows(transformed, ck["transitions"], settings, ck["schema"], settings["capacity"])
        if lr.plain(validator.summarize(transformed, settings, timing["timing_status"])) != summary:
            raise ValueError("Export summary differs")
        payload = dict(format=lr.FORMAT, settings=settings, transitions=ck["transitions"])
        experience_digest = lr.digest(payload)
        stage.mkdir()
        publish_bytes(stage / "original-status.json", (folder / "status.json").read_bytes())
        publish_bytes(stage / "intent.json", json_bytes(dict(input_sha256=inputs, helper_sha256=helpers,
                      changed_paths=changes, started=time.time(), interruption_policy="preserve_and_fail_closed")))
        for name in ("settings.json", "observation_schema.json", "timing.json", *[n for n in names if n.startswith("sessions/")]):
            publish_bytes(stage / name, (folder / name).read_bytes())
        publish_bytes(stage / "trace.json", json_bytes(transformed))
        publish_bytes(stage / "summary.json", json_bytes(summary))
        lr.checkpoint_save(lr.torch, stage / "experience.pt", payload)
        with (stage / "experience.pt").open("r+b") as stream:
            os.fsync(stream.fileno())
        if lr.digest(lr.torch.load(stage / "experience.pt", map_location="cpu", weights_only=False)) != experience_digest:
            raise ValueError("Export changed training experience")
        output_hashes = hashes(lr, stage, OUTPUTS)
        completion = dict(format=lr.FORMAT, status="completed", settings=settings, outputs_sha256=output_hashes)
        publish_bytes(stage / "completion.json", json_bytes(completion))
        validator.load_completed(stage, source)
        if hashes(lr, folder, inputs) != inputs or helper_hashes(lr) != helpers:
            raise ValueError("Export inputs/helper changed during validation")
        lr.verify_identity(source)
        prove_dead(lr, wave, root)
        for name in ("trace.json", "summary.json", "experience.pt"):
            publish_bytes(folder / name, (stage / name).read_bytes())
        # Existing locked helper retains the actual token, run_id and 75-step count.
        lr.release_worker(retained["token"], finished=True)
        check_stops(lr, wave, root)
        if hashes(lr, folder, inputs) != inputs or hashes(lr, folder, OUTPUTS) != output_hashes:
            raise ValueError("Export inputs/outputs changed before completion")
        lr.verify_identity(source)
        receipt = dict(format="sdmpc-terminal-export-receipt-v1", slot=job["key"],
            run_id=settings["run_id"], token=retained["token"], helper_sha256=helpers,
            input_sha256=inputs, source_identity=source, known_failure=failure,
            original_status=f"{STAGE}/original-status.json", changed_paths=changes,
            output_sha256=output_hashes, experience_digest=experience_digest,
            summary_digest=lr.digest(summary), control_steps=75, true_terminals=1,
            worker_timing_scope=lr.TIMING_SCOPE, export_elapsed_wall_seconds=time.perf_counter()-started,
            export_timing_scope=EXPORT_SCOPE, coordinator_attempt=str(attempt),
            cohort_before_sha256=lr.digest(records), checkpoint_authoritative=True,
            convergence_claim=False, physical_or_training_change=False)
        publish_bytes(folder / RECEIPT, json_bytes(receipt))
        completion["export_recovery"] = dict(receipt=RECEIPT, sha256=lr.file_hash(folder / RECEIPT))
        lr.durable_save(folder / "status.json", dict(status="export-completed", control_steps=75,
                                                   export_receipt_sha256=completion["export_recovery"]["sha256"]))
        check_stops(lr, wave, root)
        publish_bytes(folder / "completion.json", json_bytes(completion))
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True, help="Canonical behavior/scenario")
    parser.add_argument("--execute", action="store_true", help="Publish only after parent review")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("No export performed; explicit --execute is required")
    lr, _, _, wave = modules()
    if args.slot not in {j["key"] for j in wave.jobs_for(lr.COHORT_ROOT)}:
        parser.error("Unknown canonical v2 slot")
    export(lr.COHORT_ROOT / args.slot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
