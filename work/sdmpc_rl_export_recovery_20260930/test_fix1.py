"""Accepted F1-F3 regressions; static fixtures and fake processes only."""
import copy
from pathlib import Path

import pytest

import export_recovery as recovery
import supervise as supervisor
from conftest import lr, validator, wave, IDENTITY, TOKEN, static_checkpoint
from test_supervisor import ready_state


def complete_other_slots(cohort):
    for job in wave.jobs_for(cohort.root)[1:]:
        folder = cohort.root / job["key"]
        settings = dict(cohort.settings, run_id=job["key"],
                        **{k: v for k, v in job.items() if k != "key"})
        ck = static_checkpoint(settings)
        for row in ck["trace"]:
            row["candidates"] = []
        for name, value in (("settings.json", settings), ("observation_schema.json", ck["schema"]),
                            ("trace.json", ck["trace"]), ("status.json", {"status": "completed"})):
            lr.save(folder / name, value)
        start = folder / "sessions" / f"{TOKEN}.start.json"
        lr.save(start, dict(session_id=TOKEN, started=1.))
        lr.save(folder / "sessions" / f"{TOKEN}.end.json", dict(session_id=TOKEN,
            start_sha256=lr.file_hash(start), outcome="completed", elapsed_wall_seconds=1.))
        lr.save(folder / "timing.json", lr.session_timing(folder, [TOKEN]))
        lr.save(folder / "summary.json", validator.summarize(ck["trace"], settings, "KNOWN"))
        lr.checkpoint_save(lr.torch, folder / "experience.pt", dict(format=lr.FORMAT,
            settings=settings, transitions=ck["transitions"]))
        lr.save(folder / "completion.json", dict(format=lr.FORMAT, status="completed", settings=settings,
            outputs_sha256=recovery.hashes(lr, folder, recovery.OUTPUTS)))


def finish_root(cohort, process=None):
    if process is not None:
        lr.save(cohort.root / "process.json", process)
        lr.save(Path(process["attempt"]) / "process.json", process)
    lr.save(cohort.root / "status.json", dict(status="completed"))
    lr.save(cohort.root / "comparison.json", validator.compare_pairs(cohort.root, IDENTITY))
    lr.save(cohort.root / "completion.json", dict(status="completed", plan=lr.read(cohort.root / "plan.json"),
        child_completion_sha256={j["key"]: lr.file_hash(cohort.root / j["key"] / "completion.json")
                                 for j in wave.jobs_for(cohort.root)},
        comparison_sha256=lr.file_hash(cohort.root / "comparison.json")))


@pytest.mark.parametrize("redirector", [False, True])
def test_fix1_f1_new_coordinator_no_reservations(cohort, monkeypatch, redirector):
    launch_finalization(cohort, monkeypatch, redirector)


def launch_finalization(cohort, monkeypatch, redirector=False):
    recovery.export(cohort.folder)
    complete_other_slots(cohort)
    original_records = (cohort.root / ".ownership/reservations.json").read_bytes()
    folder = cohort.root / ".export-supervisor"
    folder.mkdir()
    state = ready_state()
    state["process"] = lr.process_identity()
    actual_pid = 600 if redirector else 500
    runner = [supervisor.sys.executable, "-B", "-u", str(recovery.V2 / "run_wave.py"),
              "--output", str(cohort.root), "--resume"]

    class Child:
        pid = 500

        def __init__(self, command, **kwargs):
            self.command = command
            cohort.statuses[500] = ("live", 10000)
            cohort.statuses[actual_pid] = ("live", 10001 if redirector else 10000)

        def poll(self):
            # The new bootstrap's registration is exercised once it exists.
            import coordinator_bootstrap as bootstrap
            attempt_path = folder / "attempts" / (state["attempts"][-1]["id"] + ".json")
            binding_path = attempt_path.with_suffix(".coordinator.json")
            if not binding_path.exists():
                bootstrap.register(lr, wave, cohort.root, attempt_path,
                    dict(pid=actual_pid, created=cohort.statuses[actual_pid][1], command=self.command[3:]),
                    500 if redirector else 999)
            return None

        def wait(self):
            attempt = state["attempts"][-1]
            process = dict(pid=actual_pid, command=runner[3:], started=attempt["started"] + 1,
                           attempt=str(cohort.root / "attempts" / ("d" * 32)))
            finish_root(cohort, process)
            cohort.statuses[500] = ("dead", 10000)
            cohort.statuses[actual_pid] = ("dead", 10001 if redirector else 10000)
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    assert supervisor.launched(lr, wave, cohort.root, folder, state, runner, "wave") == 0
    assert supervisor.inspect(lr, validator, wave, cohort.root) == (True, [])
    assert (cohort.root / ".ownership/reservations.json").read_bytes() == original_records
    assert state["attempts"][-1]["phase"] == "exited"
    return state


@pytest.mark.parametrize("fault", ["missing", "changed", "missing_ack", "process", "live", "unknown", "STOP"])
def test_fix1_f1_invalid_binding_never_proves_completion(cohort, monkeypatch, fault):
    state = launch_finalization(cohort, monkeypatch, redirector=True)
    attempt = state["attempts"][-1]
    path = cohort.root / ".export-supervisor/attempts" / (attempt["id"] + ".json")
    binding = path.with_suffix(".coordinator.json")
    if fault == "missing":
        binding.unlink()
    elif fault == "changed":
        binding.write_bytes(binding.read_bytes() + b" ")
    elif fault == "missing_ack":
        attempt.pop("binding_sha256")
        lr.save(path, attempt)
        lr.save(cohort.root / ".export-supervisor/state.json", state)
    elif fault == "process":
        process = lr.read(cohort.root / "process.json")
        process["pid"] = 777
        lr.save(cohort.root / "process.json", process)
        lr.save(Path(process["attempt"]) / "process.json", process)
    elif fault in ("live", "unknown"):
        cohort.statuses[600] = (fault, 10001 if fault == "live" else None)
    else:
        (cohort.root / "STOP").touch()
    with pytest.raises((ValueError, OSError, KeyError, lr.Stopped)):
        supervisor.inspect(lr, validator, wave, cohort.root)


@pytest.mark.parametrize("field", ["token", "run_id"])
@pytest.mark.parametrize("acceptance", ["export", "verify"])
def test_fix1_f2_replaced_retained_identity(cohort, field, acceptance):
    recovery.export(cohort.folder)
    records = lr.read(cohort.root / ".ownership/reservations.json")
    records[-1][field] = "f"*32 if field == "token" else "replacement-run"
    lr.save(cohort.root / ".ownership/reservations.json", records)
    with pytest.raises(ValueError, match="identity|retained"):
        if acceptance == "export":
            recovery.export(cohort.folder)
        else:
            recovery.verify_complete(lr, validator, cohort.folder, IDENTITY)


def test_fix1_f2_legitimate_completed_bookkeeping(cohort):
    receipt = recovery.export(cohort.folder)
    lr.release_worker(TOKEN)
    assert lr.read(cohort.root / ".ownership/reservations.json")[-1]["state"] == "completed"
    assert recovery.export(cohort.folder) == receipt


@pytest.mark.parametrize("fault", ["slot", "receipt_run_id", "appended_replacement"])
def test_fix1_f2_receipt_and_record_must_identify_same_slot(cohort, fault):
    receipt = recovery.export(cohort.folder)
    if fault == "appended_replacement":
        records = lr.read(cohort.root / ".ownership/reservations.json")
        records.append(dict(records[-1], token="f"*32))
        lr.save(cohort.root / ".ownership/reservations.json", records)
    else:
        receipt["slot" if fault == "slot" else "run_id"] = "replacement"
        lr.save(cohort.folder / recovery.RECEIPT, receipt)
        completion = lr.read(cohort.folder / "completion.json")
        completion["export_recovery"]["sha256"] = lr.file_hash(cohort.folder / recovery.RECEIPT)
        lr.save(cohort.folder / "completion.json", completion)
        lr.save(cohort.folder / "status.json", dict(status="export-completed", control_steps=75,
            export_receipt_sha256=completion["export_recovery"]["sha256"]))
    with pytest.raises(ValueError, match="retained identity"):
        recovery.verify_complete(lr, validator, cohort.folder, IDENTITY)


def failed_other_slot(cohort):
    job = wave.jobs_for(cohort.root)[1]
    folder = cohort.root / job["key"]
    folder.mkdir(parents=True)
    settings = dict(cohort.settings, run_id="other-run",
                    **{k: v for k, v in job.items() if k != "key"})
    record = dict(copy.deepcopy(cohort.record), key=job["key"], token="e"*32, run_id="other-run")
    records = lr.read(cohort.root / ".ownership/reservations.json")
    lr.save(cohort.root / ".ownership/reservations.json", records + [record])
    lr.save(folder / "settings.json", settings)
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", static_checkpoint(settings))
    for name in ("process.json", "launcher.json", "status.json", "stderr.log"):
        source = cohort.folder / (recovery.STAGE + "/original-status.json" if name == "status.json" else name)
        (folder / name).write_bytes(source.read_bytes())
    error = f"RuntimeError: Child failed/stopped: {job['key']}, exit=1"
    lr.save(cohort.root / "status.json", dict(status="failed", error=error))
    lr.save(cohort.attempt / "failure.json", dict(error=error))


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("fault", ["missing", "changed", "input", "token", "run_id"])
def test_fix1_f3_linked_receipt_required_on_every_inspect(cohort, completed, fault):
    recovery.export(cohort.folder)
    if completed:
        complete_other_slots(cohort)
        finish_root(cohort)
    else:
        failed_other_slot(cohort)
    receipt = cohort.folder / recovery.RECEIPT
    if fault == "missing":
        receipt.unlink()
    elif fault == "changed":
        receipt.write_bytes(receipt.read_bytes() + b" ")
    elif fault == "input":
        path = cohort.folder / "checkpoint.pt"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        records = lr.read(cohort.root / ".ownership/reservations.json")
        records[0][fault] = "replacement"
        lr.save(cohort.root / ".ownership/reservations.json", records)
    with pytest.raises((ValueError, FileNotFoundError)):
        supervisor.inspect(lr, validator, wave, cohort.root)


def bootstrap_intent(cohort):
    import coordinator_bootstrap as bootstrap
    folder = cohort.root / ".export-supervisor"
    path = folder / "attempts" / ("9"*32 + ".json")
    command = [supervisor.sys.executable, "-B", "-u", str(recovery.HERE / "coordinator_bootstrap.py"), "--attempt", str(path)]
    actual = dict(pid=500, created=10000, command=command[3:])
    cohort.statuses[500] = ("live", 10000)
    attempt = dict(id="9"*32, kind="wave", phase="running", started=1., command=command,
        runner_command=bootstrap.runner_command(cohort.root), owner=lr.process_identity(),
        process=dict(pid=500, created=10000, command=command), helper_sha256=recovery.helper_hashes(lr), source=IDENTITY)
    lr.save(path, attempt)
    lr.save(folder / "state.json", dict(ready_state(), attempts=[attempt]))
    return path, actual


@pytest.mark.parametrize("fault", ["parent", "creation", "unknown", "command", "STOP"])
def test_fix1_bootstrap_rejects_invalid_registration(cohort, fault):
    import coordinator_bootstrap as bootstrap
    path, actual = bootstrap_intent(cohort)
    parent = 999
    if fault == "parent":
        parent = 123
    elif fault == "creation":
        actual["created"] += 1
    elif fault == "unknown":
        cohort.statuses[500] = ("unknown", None)
    elif fault == "command":
        actual["command"] = ["wrong"]
    else:
        (cohort.root / "STOP").touch()
    with pytest.raises((ValueError, lr.Stopped)):
        bootstrap.register(lr, wave, cohort.root, path, actual, parent)
    assert not path.with_suffix(".coordinator.json").exists()


def test_fix1_bootstrap_delegates_only_after_durable_ack(cohort, monkeypatch):
    import coordinator_bootstrap as bootstrap
    path, actual = bootstrap_intent(cohort)
    original_ack = bootstrap.acknowledged
    delegated = []

    def ack(lr, root, path, binding_hash):
        assert not original_ack(lr, root, path, binding_hash)
        attempt = lr.read(path)
        attempt.update(binding_sha256=binding_hash, coordinator=actual)
        lr.save(path, attempt)
        assert not original_ack(lr, root, path, binding_hash)
        lr.save(root / ".export-supervisor/state.json", dict(ready_state(), attempts=[attempt]))
        return original_ack(lr, root, path, binding_hash)

    def delegate(args):
        delegated.append((args, list(bootstrap.sys.argv)))
        return 7

    monkeypatch.setattr(bootstrap, "acknowledged", ack)
    monkeypatch.setattr(bootstrap.os, "getppid", lambda: 999)
    monkeypatch.setattr(lr, "process_identity", lambda **k: actual)
    monkeypatch.setattr(wave, "main", delegate)
    assert bootstrap.run(path) == 7
    assert delegated == [(bootstrap.runner_command(cohort.root)[4:], bootstrap.runner_command(cohort.root)[3:])]
