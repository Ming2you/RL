"""One-time pinned carry155 recovery. Default is dry-run; never reset or step."""
import argparse
import copy
import hashlib
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import pickle
import shutil

import local_runtime as lr
from collect import validate_checkpoint
from run_wave import jobs_for

OLD_ROOT = lr.GOAL / "local_budget_v1"
MANIFEST = lr.REPO / ".superpowers/sdd/rl_budget_local_collection_plan_20260930/v1-frozen-inputs.json"
MANIFEST_SHA = "cdbbc81d260887ccd2e74221d9ea44befd88e5f00013988aeaf09be927994262"
CHECKPOINT_SHA = "659f936702be59880760677d76a8ccc35957a03aa881a3fe85a9fb4ea1923816"
SETTINGS_SHA = "933269729875452e776be948fc8a66b8af15423f6b63e732c6fcfb6328ee5571"
RUN_ID = "2962784e595d4ee8b1272dda7762dbc6"
PRIOR_SECONDS = 66.75636630004738
ATTEMPT = "bae7788116f247999100f4ad6a2d3d19"
KEY = "carry/sweet_155_w"
FAILURE = "RuntimeError: Child failed/stopped: carry/sweet_155_w, exit=1"


def wave_plan(root, identity):
    return dict(format=lr.FORMAT, jobs=jobs_for(root), identity=identity, maximum_workers=5,
                threads_per_worker=1, episodes=10, transitions=750, training_only=True)


def check_stops():
    for root in (OLD_ROOT, lr.COHORT_ROOT):
        lr.check_stop(root)
        for job in jobs_for(root):
            lr.check_stop(root / job["key"])


@contextmanager
def existing_lock(path):
    """Lock existing v1 bytes via a read-only handle; never create/rewrite them."""
    import msvcrt
    with path.open("rb") as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield stream
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def input_bytes(path, locked):
    stream = locked.get(path.resolve())
    if stream is None:
        return path.read_bytes()
    # Windows byte locks also deny reads through another handle in this process.
    stream.seek(0)
    return stream.read()


def authenticate_files(locked):
    if lr.file_hash(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Pinned v1 preservation manifest changed")
    entries = lr.read(MANIFEST)
    old_files = {}
    for entry in entries:
        path = lr.REPO / entry["path"]
        if not path.resolve().is_relative_to(lr.REPO.resolve()):
            raise ValueError("Pinned input moved outside repository")
        if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents) if p != p.parent):
            raise ValueError("Pinned input uses a relocated link")
        if path.stat().st_size != entry["length"] or hashlib.sha256(input_bytes(path, locked)).hexdigest() != entry["sha256"]:
            raise ValueError("Pinned v1 input changed: " + str(path))
        if path.resolve().is_relative_to(OLD_ROOT.resolve()):
            old_files[path.relative_to(OLD_ROOT).as_posix()] = entry["sha256"]
    actual = {p.relative_to(OLD_ROOT).as_posix() for p in OLD_ROOT.rglob("*") if p.is_file()}
    if set(old_files) != actual or not old_files:
        raise ValueError("Pinned v1 input inventory moved or changed")
    folder = OLD_ROOT / KEY
    if (lr.file_hash(folder / "checkpoint.pt") != CHECKPOINT_SHA or
            lr.file_hash(folder / "settings.json") != SETTINGS_SHA):
        raise ValueError("Exact two-interval checkpoint/settings pin differs")
    return old_files


def authenticate_evidence(new_identity):
    settings = lr.read(OLD_ROOT / KEY / "settings.json")
    old_identity = settings["identity"]
    for key in ("physical", "runtime", "gate_sha256", "contract_sha256"):
        if old_identity[key] != new_identity[key]:
            raise ValueError("Old/new physical source/runtime/contract differs: " + key)
    for name, sha in old_identity["local_sources"].items():
        path = (lr.REPO / name).resolve()
        if not path.is_relative_to(lr.REPO.resolve()) or lr.file_hash(path) != sha:
            raise ValueError("Old pinned local source changed")
    if (settings["contract"] != lr.contract() or settings["run_id"] != RUN_ID or
            settings["scenario"] != lr.SCENARIOS[0] or settings["behavior"] != "carry" or
            settings["evaluation"] is not False or settings["model_sha256"] is not None or
            settings["cpu_mask"] != 1 or settings["capacity"] != 6000. or
            (settings["training_seed"], settings["exploration_seed"]) != lr.seeds(lr.SCENARIOS[0])):
        raise ValueError("Pinned training settings differ")
    if lr.read(OLD_ROOT / "plan.json") != wave_plan(OLD_ROOT, old_identity):
        raise ValueError("Pinned failed wave plan differs")
    attempt = OLD_ROOT / "attempts" / ATTEMPT
    process = lr.read(attempt / "process.json")
    if (Path(process["attempt"]).resolve() != attempt.resolve() or
            lr.read(OLD_ROOT / "process.json") != process or
            lr.read(attempt / "failure.json") != {"error": FAILURE} or
            lr.read(OLD_ROOT / "status.json") != {"status": "failed", "error": FAILURE} or
            lr.read(attempt / "ABORT.json")["reason"] != "coordinator_draining"):
        raise ValueError("Failed attempt evidence differs")
    records = lr.read(OLD_ROOT / ".ownership/reservations.json")
    if len(records) != 7 or len({r["token"] for r in records}) != 7:
        raise ValueError("Expected two sessions and five failed launch intents")
    for r in records:
        if r["state"] != "exited" or not all(lr.process_dead(r[k]) for k in ("owner", "worker")):
            raise OSError("Old numerical owner/worker is live or UNKNOWN")
    prior, failed = records[:2], records[2:]
    for i, r in enumerate(prior):
        if (r["key"] != KEY or r["run_id"] != RUN_ID or r["control_steps"] != i+1 or
                r["cohort_phase"] != "checkpointed" or r["resume"] is not bool(i)):
            raise ValueError("Prior session history differs")
        start = lr.read(OLD_ROOT / KEY / "sessions" / (r["token"] + ".start.json"))
        end = lr.read(OLD_ROOT / KEY / "sessions" / (r["token"] + ".end.json"))
        if start["process"] != r["worker"] or end["outcome"] != "checkpointed":
            raise ValueError("Prior measured worker session differs")
    for i, (r, job) in enumerate(zip(failed, jobs_for(OLD_ROOT)[:5])):
        folder = OLD_ROOT / job["key"]
        if (any(r[k] != job[k] for k in ("key", "cpu_mask", "training_seed", "exploration_seed")) or
                r["owner"]["pid"] != process["pid"] or r["owner"]["command"] != process["command"] or
                r["control_steps"] != (2 if i == 0 else -1) or r["resume"] is not (i == 0) or
                r["cohort_phase"] != ("checkpointed" if i == 0 else "started")):
            raise ValueError("Failed launch intent differs")
        launcher = lr.read(folder / "launcher.json")
        command = r["worker"]["command"]
        if (launcher["pid"] != r["worker"]["pid"] or launcher["command"] != command or
                Path(launcher["attempt"]).resolve() != attempt.resolve()):
            raise ValueError("Failed launcher provenance differs")
        for option, value in (("--reservation", r["token"]), ("--scenario", job["scenario"]),
                              ("--behavior", "carry"), ("--cpu-mask", str(job["cpu_mask"]))):
            if command.count(option) != 1 or command[command.index(option)+1] != value:
                raise ValueError("Failed launcher command differs")
        for option, expected in (("--output", folder), ("--abort-file", attempt / "ABORT.json")):
            if command.count(option) != 1 or Path(command[command.index(option)+1]).resolve() != expected.resolve():
                raise ValueError("Failed launch path moved")
        error = (folder / "stderr.log").read_text(encoding="utf-8")
        if ("claim_worker" not in error or not error.rstrip().endswith(
                "ValueError: Launch reservation process identity differs") or
                (folder / "stdout.log").stat().st_size != 0 or
                any((folder / "sessions").glob(r["token"] + ".*"))):
            raise ValueError("Missing pre-session claim failure evidence")
        if i and {p.name for p in folder.iterdir()} != {"runner.lock", "launcher.json", "stdout.log", "stderr.log"}:
            raise ValueError("Unclaimed slot has unexpected physical/session outputs")
    for job in jobs_for(OLD_ROOT)[5:]:
        if {p.name for p in (OLD_ROOT / job["key"]).iterdir()} != {"runner.lock"}:
            raise ValueError("Local behavior has unexpected prior work")
    ids = sorted(r["token"] for r in prior)
    timing = lr.session_timing(OLD_ROOT / KEY, ids)
    if (lr.session_ids(OLD_ROOT / KEY) != ids or timing != lr.read(OLD_ROOT / KEY / "timing.json") or
            timing["timing_status"] != "KNOWN" or abs(timing["elapsed_wall_seconds"]-PRIOR_SECONDS) > 1e-7):
        raise ValueError("Two prior measured sessions differ")
    return settings, prior, failed, timing


def load_physical(path, settings):
    # Called only by the parent's explicit migration command, or mocked in synthetic tests.
    from budget_env import BudgetEnv
    rt = lr.boot(lr.DEFAULT_SNAPSHOT, settings["cpu_mask"])
    env = BudgetEnv(rt, scenario=settings["scenario"], training_seed=settings["training_seed"], guard_mode="physical")
    try:
        lr.verify_environment(env, rt, settings["contract"])
        ck = lr.torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(ck, settings)
        observation = env.restore(ck["environment"])
        lr.np.testing.assert_array_equal(observation, ck["observation"])
        lr.verify_environment(env, rt, settings["contract"], schema=True)
        return ck
    finally:
        env.close()


def exact_equal(a, b, path="payload", seen=None):
    """Compare all saved physical attributes, including non-dataclass cache fields."""
    seen = set() if seen is None else seen
    if type(a) is not type(b):
        raise ValueError("Migration equality type differs: " + path)
    if (id(a), id(b)) in seen:
        return
    seen.add((id(a), id(b)))
    if isinstance(a, lr.np.ndarray):
        equal = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Migration equality keys differ: " + path)
        for k in a:
            exact_equal(a[k], b[k], path + "/" + str(k), seen)
        return
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Migration equality length differs: " + path)
        for i, (x, y) in enumerate(zip(a, b)):
            exact_equal(x, y, path + "/" + str(i), seen)
        return
    elif hasattr(a, "__dict__") and not callable(a):
        exact_equal(vars(a), vars(b), path, seen)
        return
    else:
        equal = pickle.dumps(a, protocol=5) == pickle.dumps(b, protocol=5)
    if not equal:
        raise ValueError("Migration equality differs: " + path)


def assert_equal_payload(old, new):
    exact_equal({k: v for k, v in old.items() if k != "settings"},
                {k: v for k, v in new.items() if k != "settings"})
    exact_equal({k: v for k, v in old["settings"].items() if k != "identity"},
                {k: v for k, v in new["settings"].items() if k != "identity"}, "settings")


def publish_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def verify_committed(root, identity):
    lr.migration_guard()
    receipt = lr.read(root / ".migration/committed.json")
    if receipt["new_identity"] != identity or receipt["old_root"] != str(OLD_ROOT.resolve()):
        raise ValueError("Committed migration identity/root differs")
    stage = root / ".migration/stage"
    for name, sha in receipt["stage_sha256"].items():
        if lr.file_hash(stage / name) != sha:
            raise ValueError("Committed migration staging/provenance changed")
    for name, sha in receipt["published_sha256"].items():
        if lr.file_hash(root / name) != sha:
            raise ValueError("Committed migration published output changed; never overwrite")
    return receipt


def migrate(dry_run=True):
    root = lr.COHORT_ROOT.resolve()
    check_stops()
    # Pending/partly published migrations require review; never clean or overwrite them.
    lr.migration_guard(allow_absent=True)
    new_identity = lr.identity()
    with ExitStack() as locks:
        locked = {}
        for folder in [OLD_ROOT, OLD_ROOT / ".ownership", *[OLD_ROOT / j["key"] for j in jobs_for(OLD_ROOT)]]:
            path = folder / "runner.lock"
            locked[path.resolve()] = locks.enter_context(existing_lock(path))
        old_files = authenticate_files(locked)
        settings, prior, failed, timing = authenticate_evidence(new_identity)
        if (root / ".migration/committed.json").exists():
            # No boot/deserialization or writes on an idempotent invocation.
            locks.enter_context(existing_lock(root / "runner.lock"))
            locks.enter_context(existing_lock(root / ".ownership/runner.lock"))
            for job in jobs_for(root):
                locks.enter_context(existing_lock(root / job["key"] / "runner.lock"))
            records = lr.read(root / ".ownership/reservations.json")
            if any(not lr.process_dead(r["worker"]) or
                   r["launcher"] is not None and not lr.process_dead(r["launcher"]) for r in records):
                raise OSError("New numerical worker is live or UNKNOWN")
            return verify_committed(root, new_identity)
        if root.exists():
            raise ValueError("New migration root is nonempty or incomplete; preserve for diagnosis")
        ck = load_physical(OLD_ROOT / KEY / "checkpoint.pt", settings)
        validate_checkpoint(ck, settings)
        if len(ck["transitions"]) != 2 or sorted(ck["session_ids"]) != sorted(r["token"] for r in prior):
            raise ValueError("Expected exactly two nonterminal prior transitions/sessions")
        updated = copy.deepcopy(ck)
        updated["settings"]["identity"] = new_identity
        validate_checkpoint(updated, updated["settings"])
        assert_equal_payload(ck, updated)
        receipt = dict(status="dry-run" if dry_run else "committed", old_root=str(OLD_ROOT.resolve()),
            new_root=str(root), old_identity=settings["identity"], new_identity=new_identity,
            manifest_sha256=MANIFEST_SHA, run_id=RUN_ID, reused_transitions=2, additional_transitions=748,
            total_transitions=750, transitions_per_scenario=150,
            prior_session_ids=ck["session_ids"], prior_locked_session_seconds=timing["elapsed_wall_seconds"],
            timing_scope=lr.TIMING_SCOPE, excluded_costs="Failed coordinator and five unclaimed launcher startup costs are outside locked worker-session timing; elapsed cost is not measured here, not zero.",
            failed_unclaimed_intents=[dict(old_intent=r, new_slot=str(root / r["key"]),
                disposition="resume two-interval prefix" if r["key"] == KEY else "first physical episode allowed") for r in failed],
            equality=dict(non_settings_payload="exact recursive types, array dtype/shape/bytes and every saved object attribute", settings="only identity changes", physical_contract="exact", transitions=2,
                          boundary_k=7, simulation_seconds=1260, policy_and_rng="exact", sessions="byte-exact"))
        if dry_run:
            authenticate_files(locked)
            check_stops()
            return receipt
        # Block admission first, then own every destination lock before publication.
        locks.enter_context(lr.exclusive_run(root))
        if set(p.name for p in root.iterdir()) != {"runner.lock"}:
            raise ValueError("New migration root changed during preparation")
        lr.durable_save(root / ".migration/pending.json", dict(manifest_sha256=MANIFEST_SHA, root=str(root)))
        locks.enter_context(lr.exclusive_run(root / ".ownership"))
        for job in jobs_for(root):
            locks.enter_context(lr.exclusive_run(root / job["key"]))
        check_stops()
        stage = root / ".migration/stage"
        outputs = stage / "outputs"
        provenance = stage / "provenance/original"
        for name in old_files:
            target = provenance / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(input_bytes(OLD_ROOT / name, locked))
                stream.flush()
                os.fsync(stream.fileno())
            if lr.file_hash(target) != old_files[name]:
                raise ValueError("Original provenance copy differs")
        for entry in lr.read(MANIFEST):
            source = lr.REPO / entry["path"]
            if not source.resolve().is_relative_to(OLD_ROOT.resolve()):
                target = stage / "provenance/sources" / source.relative_to(lr.REPO)
                publish_file(source, target)
                if lr.file_hash(target) != entry["sha256"]:
                    raise ValueError("Original source provenance copy differs")
        publish_file(MANIFEST, stage / "provenance/v1-frozen-inputs.json")
        slot = outputs / KEY
        slot.mkdir(parents=True)
        lr.durable_save(slot / "settings.json", updated["settings"])
        lr.checkpoint_save(lr.torch, slot / "checkpoint.pt", updated)
        with (slot / "checkpoint.pt").open("r+b") as stream:
            os.fsync(stream.fileno())
        # Compare a fresh deserialization, not just the in-memory copy.
        loaded = lr.torch.load(slot / "checkpoint.pt", map_location="cpu", weights_only=False)
        validate_checkpoint(loaded, updated["settings"])
        assert_equal_payload(ck, loaded)
        for name in ("observation_schema.json", "timing.json"):
            publish_file(OLD_ROOT / KEY / name, slot / name)
        for path in (OLD_ROOT / KEY / "sessions").iterdir():
            publish_file(path, slot / "sessions" / path.name)
        history = [dict(r, launcher=None) for r in prior]
        lr.durable_save(outputs / ".ownership/reservations.json", history)
        lr.durable_save(outputs / "plan.json", wave_plan(root, new_identity))
        receipt["published_sha256"] = {p.relative_to(outputs).as_posix(): lr.file_hash(p)
                                       for p in sorted(outputs.rglob("*")) if p.is_file()}
        receipt["stage_sha256"] = {p.relative_to(stage).as_posix(): lr.file_hash(p)
                                   for p in sorted(stage.rglob("*")) if p.is_file()}
        lr.verify_identity(new_identity)
        authenticate_files(locked)
        check_stops()
        for name in receipt["published_sha256"]:
            publish_file(outputs / name, root / name)
        for name, sha in receipt["published_sha256"].items():
            if lr.file_hash(root / name) != sha:
                raise ValueError("Migration publication verification failed")
        check_stops()
        lr.verify_identity(new_identity)
        authenticate_files(locked)
        lr.durable_save(root / ".migration/committed.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Commit the one-time migration; default only validates")
    args = parser.parse_args()
    print(json.dumps(migrate(dry_run=not args.execute), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
