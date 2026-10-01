"""Versioned collection helpers; frozen physical runtime stays read-only."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO / "work/sdmpc_rl_multi_20260929"
GOAL = REPO / "results/sdmpc_rl_balanced_goal_20260930"
COHORT_ROOT = GOAL / "local_budget_v2"
REQUIRE_MIGRATION = True
GATE = GOAL / "value_audit_v1/projection/completion.json"
GATE_HASH = "b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO / ".deps-budget"), str(OLD)]
from budget_runtime import boot, DEFAULT_SNAPSHOT, read, plain, digest, save as atomic_save
from run_budget import exclusive_run, file_hash, runtime_versions, verify_pins, checkpoint_save, observation_schema
import numpy as np
import torch
from td3 import SCENARIOS
from launch_identity import probe_process, verify_claim

BEHAVIORS = ("carry", "local")
MASKS = (1, 4, 16, 64, 256)
FORMAT = "sdmpc-local-training-v1"
COHORT_PHASES = ("started", "checkpointed", "finished")
TIMING_SCOPE = ("Aggregate locked worker sessions: from before session-start publication through boot, "
    "reset/restore, steps, all checkpoint and data-output serialization, environment close, status and "
    "ownership exit bookkeeping. Excludes final session/timing-ledger and completion-marker publication, "
    "lock release, interpreter startup/teardown, coordinator time and offline analysis. "
    "UNKNOWN means missing session elapsed time; known_elapsed_wall_seconds is then only a lower bound.")


def save(path, value):
    json.dumps(plain(value), allow_nan=False)
    atomic_save(path, value)


def durable_save(path, value):
    save(path, value)
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


def canonical_root(output):
    if Path(output).resolve() != COHORT_ROOT.resolve():
        raise ValueError("Use the declared canonical cohort root")
    migration_guard()
    return COHORT_ROOT.resolve()


def canonical_worker(output, scenario, behavior, cpu_mask):
    if scenario not in SCENARIOS or behavior not in BEHAVIORS or cpu_mask != MASKS[SCENARIOS.index(scenario)]:
        raise ValueError("Worker differs from declared cohort slot")
    expected = COHORT_ROOT.resolve() / behavior / scenario
    if Path(output).resolve() != expected:
        raise ValueError("Use the declared canonical cohort output")
    migration_guard()
    return f"{behavior}/{scenario}"


def migration_guard(allow_absent=False):
    folder = COHORT_ROOT / ".migration"
    if not folder.exists() and REQUIRE_MIGRATION and not allow_absent:
        raise ValueError("v2 cohort requires committed prefix migration before collection")
    if folder.exists():
        if not (folder / "committed.json").exists():
            raise ValueError("Migration incomplete; preserve staging for diagnosis")
        receipt = read(folder / "committed.json")
        pending = read(folder / "pending.json")
        if (receipt.get("status") != "committed" or receipt.get("new_root") != str(COHORT_ROOT.resolve()) or
                pending != dict(manifest_sha256=receipt.get("manifest_sha256"), root=str(COHORT_ROOT.resolve())) or
                receipt.get("reused_transitions") != 2 or receipt.get("additional_transitions") != 748):
            raise ValueError("Migration committed root/identity differs")


def process_identity(pid=None, command=None):
    pid = os.getpid() if pid is None else pid
    _, created = probe_process(pid)
    return dict(pid=pid, created=created, command=list(sys.argv if command is None else command))


def process_dead(record):
    if record is None:
        return False
    status, created = probe_process(record["pid"])
    return status == "dead" or (status == "live" and record["created"] is not None and
                                 created is not None and created != record["created"])


@contextmanager
def ownership():
    # Five children can claim together; retry only the small ownership-file critical section.
    for attempt in range(101):
        lock = exclusive_run(COHORT_ROOT / ".ownership")
        try:
            lock.__enter__()
            break
        except OSError:
            if attempt == 100:
                raise
            time.sleep(.05)
    try:
        migration_guard()
        path = COHORT_ROOT / ".ownership/reservations.json"
        records = read(path) if path.exists() else []
        for record in records:
            if record["state"] not in ("reserved", "active", "exited", "completed"):
                raise ValueError("Unknown cohort reservation state")
            if (COHORT_ROOT / record["key"] / "completion.json").exists():
                record["state"] = "completed"
                record["cohort_phase"] = "finished"
            elif record["state"] == "reserved" and process_dead(record["launcher"]):
                # A claim requires a live bound launcher, so no future worker can start.
                record["state"] = "exited"
            elif (record["state"] == "active" and process_dead(record["worker"]) and
                  (record["launcher"] is None or process_dead(record["launcher"]))):
                record["state"] = "exited"
        yield records
        durable_save(path, records)
    finally:
        lock.__exit__(None, None, None)


def ensure_cohort_idle(output):
    canonical_root(output)
    with ownership() as records:
        if any(r["state"] in ("reserved", "active") for r in records):
            raise OSError("Cohort has a live or UNKNOWN startup/worker reservation")


def reserve_worker(output, scenario, behavior, cpu_mask, launch=False, resume=False):
    key = canonical_worker(output, scenario, behavior, cpu_mask)
    with ownership() as records:
        if (Path(output) / "completion.json").exists() or any(r["key"] == key and r["state"] == "completed" for r in records):
            raise ValueError("Already completed cohort slot")
        occupied = [r for r in records if r["state"] in ("reserved", "active")]
        if any(r["key"] == key for r in occupied):
            raise OSError("Cohort slot has a live or UNKNOWN reservation")
        if len(occupied) >= 5:
            raise OSError("Maximum five numerical workers/reservations")
        history = [r for r in records if r["key"] == key]
        if history and not resume:
            phase = "completed" if any(r["cohort_phase"] == "finished" for r in history) else "started"
            raise ValueError(f"Already {phase} cohort slot; resume its existing checkpoint")
        if resume:
            if not history or not (Path(output) / "checkpoint.pt").exists():
                raise ValueError("Cohort resume requires retained identity and checkpoint")
            if read(Path(output) / "settings.json")["run_id"] != history[-1]["run_id"]:
                raise ValueError("Cohort checkpoint run identity differs")
        if launch:
            with exclusive_run(Path(output)):
                pass
        token = uuid.uuid4().hex
        records.append(dict(key=key, token=token, state="reserved" if launch else "active",
            owner=process_identity(), launcher=None, worker=None if launch else process_identity(),
            training_seed=seeds(scenario)[0], exploration_seed=seeds(scenario)[1], cpu_mask=cpu_mask,
            resume=resume, run_id=history[-1]["run_id"] if history else uuid.uuid4().hex,
            cohort_phase=max((r["cohort_phase"] for r in history), key=COHORT_PHASES.index, default="started"),
            control_steps=max((r["control_steps"] for r in history), default=-1)))
    return token


def bind_launch(token, pid, command):
    with ownership() as records:
        record = next(r for r in records if r["token"] == token)
        current = process_identity()
        if (record["state"] != "reserved" or record["launcher"] is not None or
                any(current[k] != record["owner"][k] for k in ("pid", "created"))):
            raise ValueError("Launch binding is owner-only and once-only")
        record["launcher"] = process_identity(pid, command)


def claim_worker(output, scenario, behavior, cpu_mask, token=None, resume=False):
    key = canonical_worker(output, scenario, behavior, cpu_mask)
    if token is None:
        return reserve_worker(output, scenario, behavior, cpu_mask, resume=resume)
    for attempt in range(201):
        with ownership() as records:
            record = next((r for r in records if r["token"] == token), None)
            if record is None or record["key"] != key or record["state"] != "reserved":
                raise ValueError("Invalid or already claimed launch reservation")
            if record["resume"] != resume:
                raise ValueError("Launch reservation resume mode differs")
            if record["launcher"] is not None:
                current = process_identity()
                current["parent_pid"] = os.getppid()
                mode = verify_claim(record["launcher"], current, record["owner"],
                                    current["parent_pid"], probe=probe_process)
                record.update(state="active", worker=current, launch_mode=mode)
                return token
            if process_dead(record["owner"]):
                raise OSError("Unbound launch owner exited; startup remains UNKNOWN")
        if attempt == 200:
            raise OSError("Launch binding timeout; startup remains UNKNOWN")
        check_stop(output)
        time.sleep(.05)


def reservation_run_id(token):
    return next(r["run_id"] for r in read(COHORT_ROOT / ".ownership/reservations.json") if r["token"] == token)


def checkpoint_progress(token, run_id, steps):
    with ownership() as records:
        record = next(r for r in records if r["token"] == token)
        if record["run_id"] != run_id or not record["control_steps"] <= steps <= 75:
            raise ValueError("Cohort checkpoint identity/progress differs")
        if record["cohort_phase"] == "finished" and steps != 75:
            raise ValueError("Finished cohort requires its terminal checkpoint")
        record["control_steps"] = steps
        if record["cohort_phase"] == "started":
            record["cohort_phase"] = "checkpointed"


def release_worker(token, finished=False):
    with ownership() as records:
        record = next(r for r in records if r["token"] == token)
        current = process_identity()
        worker = record["worker"]
        is_worker = worker is not None and all(current[k] == worker[k] for k in ("pid", "created"))
        if (record["state"] == "active" and not is_worker and
                (not process_dead(worker) or
                 record["launcher"] is not None and not process_dead(record["launcher"]))):
            raise OSError("Actual worker/launcher is live or UNKNOWN; cannot release")
        if record["state"] == "reserved" and not process_dead(record["launcher"]):
            raise OSError("Unclaimed launcher is live or UNKNOWN; cannot release")
        if finished:
            if record["control_steps"] != 75:
                raise ValueError("Cannot finish a cohort without its terminal checkpoint")
            record["cohort_phase"] = "finished"
        if record["state"] != "completed":
            record["state"] = "completed" if (COHORT_ROOT / record["key"] / "completion.json").exists() else "exited"


def session_ids(output):
    return sorted(p.name[:-11] for p in (Path(output) / "sessions").glob("*.start.json"))


def reserved_sessions(output):
    key = Path(output).resolve().relative_to(COHORT_ROOT.resolve()).as_posix()
    return [r["token"] for r in read(COHORT_ROOT / ".ownership/reservations.json") if r["key"] == key]


def session_timing(output, required=(), exclude=()):
    output = Path(output)
    ids = sorted((set(required) | set(session_ids(output))) - set(exclude))
    known, unknown, hashes = 0., [], {}
    for session in ids:
        if len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
            raise ValueError("Invalid timing session ID")
        start, end = [output / "sessions" / f"{session}.{kind}.json" for kind in ("start", "end")]
        for path in (start, end):
            if path.exists():
                hashes[str(path.relative_to(output)).replace("\\", "/")] = file_hash(path)
        if not start.exists() or not end.exists():
            unknown.append(session)
            continue
        record = read(end)
        if (read(start)["session_id"] != session or record["session_id"] != session or
                record["start_sha256"] != file_hash(start)):
            raise ValueError("Session timing provenance differs")
        elapsed = record["elapsed_wall_seconds"]
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid session elapsed time")
        known += elapsed
    return dict(timing_status="UNKNOWN" if unknown else "KNOWN", elapsed_wall_seconds=None if unknown else known,
                known_elapsed_wall_seconds=known, unknown_sessions=unknown, session_ids=ids,
                session_sha256=hashes, timing_scope=TIMING_SCOPE)


class Stopped(RuntimeError):
    pass


def stop_paths(output, abort=None):
    output = Path(output)
    paths = {p / "STOP" for p in (REPO, GOAL, output, output.parent, output.parent.parent)}
    if abort is not None:
        paths.add(Path(abort))
    return paths


def check_stop(output, abort=None):
    if any(p.exists() for p in stop_paths(output, abort)):
        raise Stopped("STOP or current coordinator abort requested")


def contract():
    if file_hash(GATE) != GATE_HASH:
        raise ValueError("Authenticated physical/schema contract gate changed")
    gate = read(GATE)
    if gate["status"] != "completed":
        raise ValueError("Gate not complete")
    return gate["settings"]["contract"]


def identity():
    expected = contract()
    for name, sha in read(GATE)["settings"]["source_sha256"].items():
        if file_hash(REPO / name) != sha:
            raise ValueError("Admitted validator source changed")
    verify_pins(DEFAULT_SNAPSHOT, expected["source_pins"])
    versions = runtime_versions()
    if versions != expected["runtime_versions"]:
        raise ValueError("Runtime version drift")
    extras = [REPO / "work/sdmpc_rl_value_audit_20260930/projection_audit.py",
              REPO / "work/sdmpc_rl_recovery_20260929/common.py"]
    paths = [p for p in HERE.glob("*.py") if not p.name.startswith("test_")] + extras
    return dict(physical=expected["source_pins"], runtime=versions,
                gate_sha256=GATE_HASH, contract_sha256=digest(expected),
                local_sources={str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)})


def verify_identity(expected):
    if identity() != expected:
        raise ValueError("Collection source/runtime/contract changed; preserve this version")


def seeds(scenario):
    index = SCENARIOS.index(scenario)
    return 6801 + index, 6901 + index


def verify_environment(env, rt, expected, schema=False):
    physical = expected["environment_contract"]
    if (digest(rt["rc"].to_plain_dict(rt["cfg"])) != physical["config_sha256"] or
            digest(rt["rc"].to_plain_dict(rt["options"])) != physical["options_sha256"]):
        raise ValueError("Physical configuration/options differ")
    actual = {k: v for k, v in env.contract().items()
              if k not in ("cfg", "options", "observation", "source_snapshot")}
    if actual != physical["coordinator"]:
        raise ValueError("Coordinator/reward/guard changed")
    canonical = next(r for r in physical["scenarios"] if r["scenario"] == env.protocol["scenario"])
    if env.profile_hash == canonical["profile_sha256"]:
        raise ValueError("Training profile equals canonical evaluation profile")
    if schema and observation_schema(env) != expected["observation_schema"]:
        raise ValueError("Observation schema drift")


def sequence_validator():
    path = REPO / "work/sdmpc_rl_value_audit_20260930/projection_audit.py"
    spec = importlib.util.spec_from_file_location("local_sequence_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_sequence
