"""Frozen imports, provenance and durable single-slot bookkeeping; no dispatcher."""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOAL = REPO / "results/sdmpc_rl_balanced_goal_20260930"
WAVE = GOAL / "return_policy_wave_v1"
PREDECESSOR = GOAL / "return_init_v1"
OLD = REPO / "work/sdmpc_rl_multi_20260929"
NUF = REPO / "work/sdmpc_rl_nuf_retention_20260930"
SCENARIOS = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")
MASKS = (1, 4, 16, 64, 256)
FORMAT = "sdmpc-return-policy-wave-v1"
MODEL_SHA = "7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904"
COMPLETION_SHA = "34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757"
SPEC_SHA = "933e29c1b316530ceff1605d527532e482a658c12fa6c13fd24a93074ef6593e"
GATE = GOAL / "value_audit_v1/projection/completion.json"
GATE_SHA = "b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3"
TAG_PATH = REPO / "work/sdmpc_rl_export_recovery_20260930/export_recovery.py"
TAG_SHA = "f79513942cd892e346dd68a188ecd2221e862b55558170c3a35d90ed8f80a4b3"
TIMING_SCOPE = (
    "Locked worker sessions from before start publication through authentication, boot, "
    "reset/restore, inference, physical steps, validation, checkpoints, data export and close. "
    "Excludes final end/timing/summary/completion publication, lock release, interpreter startup/teardown, "
    "parent launch/budget work and offline readout. Missing interrupted ends are UNKNOWN; "
    "known session seconds are then a lower bound. Worker sums are not wave elapsed wall time.")

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(REPO / ".deps-budget"), str(OLD)]
import numpy as np
import torch
from budget_runtime import boot, DEFAULT_SNAPSHOT, read as raw_read, plain, digest
from run_budget import exclusive_run, file_hash, runtime_versions, verify_pins, observation_schema
from budget_controller import residual_budget
from compare_runs import finite, same_cost, queue_exposure

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def check_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError("Frozen source/artifact changed: " + str(path))


def module(name, path, expected):
    check_hash(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def definitions(path, names, namespace, expected):
    """Reuse unchanged pure definitions without importing unrelated learner modules."""
    check_hash(path, expected)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError("Missing/ambiguous frozen helper")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor")
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite array")
    elif isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite_tree(v)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError("Nonfinite scalar")


def read(path):
    value = raw_read(path)
    finite_tree(value)
    return value


def save(path, value):
    path = Path(path)
    data = json.dumps(plain(value), indent=2, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def save_once(path, value):
    if path.exists():
        if read(path) != plain(value):
            raise FileExistsError("Retained output differs: " + str(path))
    else:
        save(path, value)


def save_tensor(path, payload):
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def persist_checkpoint(output, payload, validate=None):
    path = output / "checkpoints" / (uuid.uuid4().hex + ".pt")
    save_tensor(path, payload)
    if validate is not None:
        validate(payload)  # Failed validation leaves the raw orphan as evidence.
    record = dict(path=path.relative_to(output).as_posix(), sha256=file_hash(path),
                  control_steps=len(payload["trace"]), settings_digest=digest(payload["settings"]),
                  session_ids=payload["session_ids"])
    save(output / "latest.json", record)
    return record


def checkpoint_record(output):
    record = read(output / "latest.json")
    path = (output / record["path"]).resolve()
    if path.parent != (output / "checkpoints").resolve() or path.suffix != ".pt":
        raise ValueError("Checkpoint pointer escaped slot")
    check_hash(path, record["sha256"])
    return record, path


class Stopped(RuntimeError):
    pass


def check_stop(output):
    if any((p / "STOP").exists() for p in (REPO, GOAL, WAVE, output)):
        raise Stopped("STOP requested; marker retained")


def sources():
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(HERE.glob("*.py"))}


def verify_files(files):
    for name, expected in files.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()):
            raise ValueError("Manifest path outside repository")
        check_hash(path, expected)


def expected_profile(scenario):
    rows = read(DEFAULT_SNAPSHOT / "outputs/sdmpc_budget_exception_all_20260922/protocols_0" /
                scenario / "forecast.json")
    rng = np.random.default_rng(7301 + SCENARIOS.index(scenario))
    multipliers = {field: rng.uniform(.98, 1.02) for field in
                   ("freeway_mainline", "urban_boundary", "ramp_arrival")}
    for row in rows:
        for field, multiplier in multipliers.items():
            row[field] = {k: v * multiplier for k, v in row[field].items()}
    return digest(rows)


def import_boundary(physical=False):
    forbidden = {"learner", "data", "td3", "train_round", "local_runtime"}
    if forbidden.intersection(sys.modules):
        raise ValueError("Learner module leaked into inference process")
    runtime = sys.modules.get("runtime")
    if runtime is not None and Path(runtime.__file__).resolve() != (
            DEFAULT_SNAPSHOT / "work/sdmpc_slide_alignment_20260922/runtime.py").resolve():
        raise ValueError("Generic runtime import collision")
    if physical and runtime is None:
        raise ValueError("Physical runtime not loaded")


def process_identity():
    helper = module("wave_process_identity", NUF / "launch_identity.py",
        "5cc1a94037b349072ff7ff37e9ceb756c3205b84cb8c1940b4305a5f1514e160")
    state, created = helper.probe_process(os.getpid())
    if state != "live" or type(created) is not int or created <= 0:
        raise ValueError("Actual worker creation identity UNKNOWN")
    return dict(pid=os.getpid(), created=created, parent_pid=os.getppid(),
                executable=sys.executable, command=list(sys.orig_argv))


def session_ids(output):
    return sorted(p.name[:-11] for p in (output / "sessions").glob("*.start.json"))


def session_timing(output, settings, required=()):
    import restore_accounting
    ids = sorted(set(required) | set(session_ids(output)))
    known, unknown, hashes, restores = 0., [], {}, []
    if any(p.name.split(".")[0] not in ids for p in (output / "sessions").glob("*.restore-*.json")):
        raise ValueError("Restoration without retained session")
    for sid in ids:
        if len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
            raise ValueError("Malformed worker session")
        start, end = [output / "sessions" / f"{sid}.{kind}.json" for kind in ("start", "end")]
        if not start.exists():
            raise ValueError("Missing retained session start")
        record = read(start)
        proc = record["process"]
        if (record["session_id"] != sid or record["run_id"] != settings["run_id"] or
                record["model_sha256"] != MODEL_SHA or record["sources"] != settings["sources"] or
                record["scenario"] != settings["scenario"] or record["lock_held"] is not True or
                type(proc["pid"]) is not int or proc["pid"] <= 0 or
                type(proc["created"]) is not int or proc["created"] <= 0 or not proc["command"]):
            raise ValueError("Session worker/source identity differs")
        hashes[start.relative_to(output).as_posix()] = file_hash(start)
        restores.append(restore_accounting.session_record(output, sid, settings, record,
                                                           read(end) if end.exists() else None))
        if not end.exists():
            unknown.append(sid)
            continue
        finish = read(end)
        if (finish["session_id"] != sid or finish["start_sha256"] != file_hash(start) or
                finish["process"] != proc):
            raise ValueError("Session end binding differs")
        known += finite(finish["elapsed_wall_seconds"], "session elapsed")
        hashes[end.relative_to(output).as_posix()] = file_hash(end)
    return dict(timing_status="UNKNOWN" if unknown else "KNOWN", session_ids=ids,
        elapsed_wall_seconds=None if unknown else known, known_elapsed_wall_seconds=known,
        unknown_sessions=unknown, session_sha256=hashes, timing_scope=TIMING_SCOPE,
        restoration=restore_accounting.summarize(restores))
