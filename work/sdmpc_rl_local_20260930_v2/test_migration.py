"""Synthetic one-time migration fixtures; never read or load actual checkpoints."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_runtime as lr
import collect
import run_wave
import migrate_prefix as migration
from test_collection import synthetic, FakeEnv, args, CONTRACT, CONFIG

LOAD_PHYSICAL = migration.load_physical


def hashes(root):
    return {p.relative_to(root).as_posix(): lr.file_hash(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def old_prefix(tmp_path, synthetic):
    old, new = tmp_path / "old", tmp_path / "new"
    source = tmp_path / "old_code.py"
    source.write_text("# synthetic pinned source\n", encoding="ascii")
    identity = dict(physical={}, runtime={}, gate_sha256="fixture-gate",
                    contract_sha256=lr.digest(CONTRACT), local_sources={"old_code.py": lr.file_hash(source)})
    synthetic.setattr(lr, "REPO", tmp_path)
    synthetic.setattr(lr, "COHORT_ROOT", old)
    synthetic.setattr(collect, "identity", lambda: copy.deepcopy(identity))
    folder = old / migration.KEY
    collect.run(args(folder, "carry", limit=1))
    collect.run(args(folder, "carry", resume=True, limit=1))
    settings = lr.read(folder / "settings.json")
    records = lr.read(old / ".ownership/reservations.json")
    for r in records:
        r.pop("launcher")
    owner = dict(pid=900, created=900, command=["old/run_wave.py", "--resume"])
    attempt = old / "attempts" / migration.ATTEMPT
    process = dict(pid=900, command=owner["command"], started=1., attempt=str(attempt))
    lr.save(attempt / "process.json", process)
    lr.save(old / "process.json", process)
    lr.save(attempt / "failure.json", {"error": migration.FAILURE})
    lr.save(attempt / "ABORT.json", {"reason": "coordinator_draining", "created": 2.})
    lr.save(old / "status.json", {"status": "failed", "error": migration.FAILURE})
    for i, job in enumerate(run_wave.jobs_for(old)):
        slot = old / job["key"]
        with lr.exclusive_run(slot):
            pass
        if job["behavior"] != "carry":
            continue
        token = f"{100+i:032x}"
        command = [sys.executable, "-B", "collect.py", "--output", str(slot), "--scenario", job["scenario"],
                   "--behavior", "carry", "--cpu-mask", str(job["cpu_mask"]), "--abort-file", str(attempt / "ABORT.json")]
        if i == 0:
            command.append("--resume")
        command.extend(["--reservation", token])
        records.append(dict(**{k: job[k] for k in ("key", "training_seed", "exploration_seed", "cpu_mask")},
            token=token, owner=owner, worker=dict(pid=1000+i, created=1000+i, command=command),
            state="exited", resume=i == 0, run_id=settings["run_id"] if i == 0 else f"{200+i:032x}",
            cohort_phase="checkpointed" if i == 0 else "started", control_steps=2 if i == 0 else -1))
        lr.save(slot / "launcher.json", dict(pid=1000+i, command=command, attempt=str(attempt), started=1.))
        (slot / "stderr.log").write_text('reservation = claim_worker(...)\nValueError: Launch reservation process identity differs\n', encoding="ascii")
        (slot / "stdout.log").write_bytes(b"")
    lr.save(old / ".ownership/reservations.json", records)
    lr.save(old / "plan.json", migration.wave_plan(old, identity))
    with lr.exclusive_run(old):
        pass
    manifest = tmp_path / "pins.json"
    entries = [dict(path=p.relative_to(tmp_path).as_posix(), length=p.stat().st_size, sha256=lr.file_hash(p))
               for p in sorted(old.rglob("*")) if p.is_file()]
    entries.append(dict(path="old_code.py", length=source.stat().st_size, sha256=lr.file_hash(source)))
    lr.save(manifest, entries)
    synthetic.setattr(migration, "OLD_ROOT", old)
    synthetic.setattr(migration, "MANIFEST", manifest)
    for name, value in dict(MANIFEST_SHA=lr.file_hash(manifest),
                            CHECKPOINT_SHA=lr.file_hash(folder / "checkpoint.pt"),
                            SETTINGS_SHA=lr.file_hash(folder / "settings.json"), RUN_ID=settings["run_id"],
                            PRIOR_SECONDS=lr.read(folder / "timing.json")["elapsed_wall_seconds"]).items():
        synthetic.setattr(migration, name, value)
    new_identity = dict(identity, local_sources={"new": "fixture"})
    synthetic.setattr(lr, "COHORT_ROOT", new)
    assert migration.OLD_ROOT.resolve().is_relative_to(tmp_path.resolve())
    assert lr.COHORT_ROOT.resolve().is_relative_to(tmp_path.resolve())
    assert migration.MANIFEST.resolve().is_relative_to(tmp_path.resolve())
    synthetic.setattr(lr, "identity", lambda: copy.deepcopy(new_identity))
    synthetic.setattr(lr, "contract", lambda: copy.deepcopy(CONTRACT))
    synthetic.setattr(lr, "verify_identity", lambda expected: None)
    synthetic.setattr(lr, "probe_process", lambda pid: ("dead", None))
    synthetic.setattr(migration, "load_physical", lambda path, settings: lr.torch.load(path, weights_only=False))
    return old, new, synthetic, new_identity


def test_dry_run_has_no_writes_and_commit_preserves_two_sessions(old_prefix):
    old, new, patch, identity = old_prefix
    before = hashes(old)
    result = migration.migrate(dry_run=True)
    assert result["reused_transitions"] == 2 and result["additional_transitions"] == 748
    assert not new.exists() and hashes(old) == before
    result = migration.migrate(dry_run=False)
    assert result["status"] == "committed" and hashes(old) == before
    assert migration.migrate(dry_run=False) == result
    ck = lr.torch.load(new / migration.KEY / "checkpoint.pt", weights_only=False)
    original = lr.torch.load(old / migration.KEY / "checkpoint.pt", weights_only=False)
    migration.assert_equal_payload(original, ck)
    assert len(ck["session_ids"]) == 2
    history = lr.read(new / ".ownership/reservations.json")
    assert len(history) == 2 and all(r["key"] == migration.KEY for r in history)
    assert len(result["failed_unclaimed_intents"]) == 5
    assert result["prior_locked_session_seconds"] == migration.PRIOR_SECONDS
    assert (new / ".migration/stage/provenance/original" / migration.KEY / "checkpoint.pt").read_bytes() == (old / migration.KEY / "checkpoint.pt").read_bytes()
    patch.setattr(collect, "identity", lambda: copy.deepcopy(identity))
    patch.setattr(lr, "probe_process", lambda pid: ("live", 777))
    original_step = FakeEnv.step
    seen = []
    def continuation(self, *a, **kw):
        seen.append(self.k)
        return original_step(self, *a, **kw)
    patch.setattr(FakeEnv, "step", continuation)
    assert collect.run(args(new / migration.KEY, "carry", resume=True, limit=1)) == "checkpointed"
    assert seen == [7]
    resumed = lr.torch.load(new / migration.KEY / "checkpoint.pt", weights_only=False)
    assert len(resumed["transitions"]) == 3 and resumed["settings"]["run_id"] == migration.RUN_ID


@pytest.mark.parametrize("fault", ["checkpoint", "settings", "source", "moved", "extra_session", "old_stop", "new_stop", "active", "unknown"])
def test_rejects_changed_or_nonidle_inputs(old_prefix, fault):
    old, new, patch, _ = old_prefix
    if fault in ("checkpoint", "settings"):
        with (old / migration.KEY / ("checkpoint.pt" if fault == "checkpoint" else "settings.json")).open("ab") as f:
            f.write(b"bad")
    elif fault == "source":
        (old.parent / "old_code.py").write_text("changed", encoding="ascii")
    elif fault == "moved":
        (old / migration.KEY).rename(old / "relocated")
    elif fault == "extra_session":
        lr.save(old / migration.KEY / "sessions/extra.start.json", {})
    elif fault == "old_stop":
        (old / "STOP").touch()
    elif fault == "new_stop":
        new.mkdir()
        (new / "STOP").touch()
    else:
        patch.setattr(lr, "probe_process", lambda pid: ("live", None) if fault == "active" else ("unknown", None))
    with pytest.raises((ValueError, OSError, lr.Stopped, FileNotFoundError)):
        migration.migrate(dry_run=False)
    assert not (new / ".migration/committed.json").exists()


@pytest.mark.parametrize("after", [0, 1, 4])
def test_partial_publication_is_preserved_and_blocks_collection(old_prefix, after):
    _, new, patch, _ = old_prefix
    original = migration.publish_file
    calls = []
    def interrupt(source, destination):
        if destination.relative_to(new).parts[0] != ".migration":
            if len(calls) == after:
                raise RuntimeError("synthetic publication interruption")
            calls.append(destination)
        original(source, destination)
    patch.setattr(migration, "publish_file", interrupt)
    with pytest.raises(RuntimeError, match="publication interruption"):
        migration.migrate(dry_run=False)
    before = hashes(new)
    with pytest.raises(ValueError, match="incomplete"):
        migration.migrate(dry_run=False)
    assert hashes(new) == before
    with pytest.raises(ValueError, match="incomplete"):
        lr.canonical_root(new)
    with pytest.raises(ValueError, match="incomplete"):
        lr.claim_worker(new / migration.KEY, lr.SCENARIOS[0], "carry", lr.MASKS[0], resume=True)


def repin_fixture(patch):
    entries = lr.read(migration.MANIFEST)
    for entry in entries:
        path = lr.REPO / entry["path"]
        entry.update(length=path.stat().st_size, sha256=lr.file_hash(path))
    lr.save(migration.MANIFEST, entries)
    patch.setattr(migration, "MANIFEST_SHA", lr.file_hash(migration.MANIFEST))
    patch.setattr(migration, "CHECKPOINT_SHA", lr.file_hash(migration.OLD_ROOT / migration.KEY / "checkpoint.pt"))


@pytest.mark.parametrize("fault", ["transition", "physical", "policy", "terminal", "sessions"])
def test_semantic_prefix_corruption_rejected_before_publication(old_prefix, fault):
    old, new, patch, _ = old_prefix
    path = old / migration.KEY / "checkpoint.pt"
    ck = lr.torch.load(path, weights_only=False)
    if fault == "transition":
        ck["transitions"][0][0][0] += .1
    elif fault == "physical":
        ck["environment"]["sim"].state.ramp_queue["r"] += 1
    elif fault == "policy":
        ck["policy"]["rng"]["state"]["state"] += 1
    elif fault == "terminal":
        ck["transitions"][1] = (*ck["transitions"][1][:4], True)
    else:
        ck["session_ids"].append("f"*32)
    lr.checkpoint_save(lr.torch, path, ck)
    repin_fixture(patch)
    with pytest.raises((ValueError, AssertionError)):
        migration.migrate(dry_run=False)
    assert not new.exists()


def test_physical_loader_restore_only(old_prefix):
    import budget_env
    old, _, patch, _ = old_prefix
    patch.setattr(budget_env, "BudgetEnv", FakeEnv)
    patch.setattr(lr, "boot", collect.boot)
    patch.setattr(lr, "verify_environment", lambda *a, **k: None)
    patch.setattr(FakeEnv, "reset", lambda *a: pytest.fail("migration reset"))
    patch.setattr(FakeEnv, "step", lambda *a: pytest.fail("migration step/PFO"))
    restored = LOAD_PHYSICAL(old / migration.KEY / "checkpoint.pt", lr.read(old / migration.KEY / "settings.json"))
    assert restored["environment"]["k"] == 7


def test_existing_new_worker_lock_blocks_idempotent_migration(old_prefix):
    _, new, _, _ = old_prefix
    migration.migrate(dry_run=False)
    with lr.exclusive_run(new / migration.KEY):
        with pytest.raises(OSError):
            migration.migrate(dry_run=False)


def test_production_admission_requires_committed_migration(old_prefix):
    _, new, patch, _ = old_prefix
    patch.setattr(lr, "REQUIRE_MIGRATION", True)
    with pytest.raises(ValueError, match="requires committed"):
        lr.canonical_root(new)
    with pytest.raises(ValueError, match="requires committed"):
        collect.run(args(new / migration.KEY, "carry", limit=1))
    assert not new.exists()
    migration.migrate(dry_run=False)
    assert lr.canonical_root(new) == new.resolve()


def test_committed_corruption_never_overwrites(old_prefix):
    _, new, _, _ = old_prefix
    migration.migrate(dry_run=False)
    path = new / migration.KEY / "checkpoint.pt"
    path.write_bytes(b"corrupt")
    before = hashes(new)
    with pytest.raises(ValueError, match="published"):
        migration.migrate(dry_run=False)
    assert hashes(new) == before
