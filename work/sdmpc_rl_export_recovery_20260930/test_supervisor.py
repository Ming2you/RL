import copy
from pathlib import Path

import pytest

import export_recovery as recovery
import supervise as supervisor
from conftest import lr, validator, wave, IDENTITY, register_fake_coordinator, finish_fake_coordinator


def test_inspect_only_exact_failure_and_skip_original_runner(cohort, monkeypatch):
    assert supervisor.inspect(lr, validator, wave, cohort.root) == (False, [cohort.job["key"]])
    abort = (cohort.attempt / "ABORT.json").read_bytes()
    recovery.export(cohort.folder)
    assert supervisor.inspect(lr, validator, wave, cohort.root) == (False, [])
    monkeypatch.setattr(wave.subprocess, "Popen", lambda *a, **k: pytest.fail("completed slot relaunched"))
    wave.run_stage(cohort.root, [cohort.job], IDENTITY, cohort.attempt / "ABORT.json", [cohort.job])
    assert (cohort.attempt / "ABORT.json").read_bytes() == abort


@pytest.mark.parametrize("fault", ["wrong_error", "missing_checkpoint", "stopped", "UNKNOWN", "STOP", "wrong_traceback"])
def test_unrecoverable_does_not_launch(cohort, monkeypatch, fault):
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected launch"))
    if fault == "wrong_error":
        lr.save(cohort.root / "status.json", dict(status="failed", error="arbitrary failure"))
    elif fault == "missing_checkpoint":
        (cohort.folder / "checkpoint.pt").unlink()
    elif fault == "stopped":
        lr.save(cohort.root / "status.json", dict(status="stopped"))
    elif fault == "UNKNOWN":
        cohort.statuses[300] = ("unknown", None)
    elif fault == "STOP":
        (cohort.root / "STOP").touch()
    else:
        (cohort.folder / "stderr.log").write_text(recovery.ERROR)
    with pytest.raises((ValueError, OSError, lr.Stopped)):
        supervisor.supervise()


def test_duplicate_supervisor_is_excluded(cohort, monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected launch"))
    with lr.exclusive_run(cohort.root / ".export-supervisor"):
        with pytest.raises(OSError):
            supervisor.supervise()


def ready_state():
    return dict(source=IDENTITY, helper_sha256=recovery.helper_hashes(lr), repairs=[], attempts=[], status="ready")


@pytest.mark.parametrize("status", ["running", "failed", "launching"])
def test_interrupted_supervisor_not_restarted(cohort, status):
    state = dict(ready_state(), status=status)
    lr.save(cohort.root / ".export-supervisor/state.json", state)
    with pytest.raises(ValueError, match="Interrupted"):
        supervisor.supervise()
    assert lr.read(cohort.root / ".export-supervisor/state.json") == state


def test_bounded_supervisor_exports_waits_and_resumes(cohort, monkeypatch):
    launched, waited = [], []
    done = [False]
    original_inspect = supervisor.inspect

    def inspect(*args):
        if done[0]:
            return True, []  # Numerical completion is deliberately outside this test.
        return original_inspect(*args)

    class Child:
        pid = 500

        def __init__(self, command, **kwargs):
            assert len(launched) == len(waited)
            assert kwargs["env"]["OMP_NUM_THREADS"] == kwargs["env"]["MKL_NUM_THREADS"] == kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
            cohort.statuses[500] = ("live", 10000)
            self.command = command
            launched.append(command)

        def poll(self):
            register_fake_coordinator(cohort, self.command)
            return None

        def wait(self):
            if Path(self.command[3]).name == "export_recovery.py":
                recovery.export(cohort.folder)
            else:
                assert Path(self.command[3]) == recovery.HERE / "coordinator_bootstrap.py"
                attempt = lr.read(Path(self.command[-1]))
                assert attempt["runner_command"] == supervisor.bootstrap.runner_command(cohort.root)
                # The original run_stage's completed-slot skip is exercised separately.
                finish_fake_coordinator(cohort, self.command)
                lr.save(cohort.root / "completion.json", {"synthetic": True})
                done[0] = True
            cohort.statuses[500] = ("dead", 10000)
            waited.append(self.command)
            return 0

    monkeypatch.setattr(supervisor, "inspect", inspect)
    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    before_abort = (cohort.attempt / "ABORT.json").read_bytes()
    assert supervisor.supervise() == "completed"
    assert len(launched) == len(waited) == 2
    state = lr.read(cohort.root / ".export-supervisor/state.json")
    assert state["repairs"] == [cohort.job["key"]] and state["status"] == "completed"
    assert [a["kind"] for a in state["attempts"]] == ["export", "wave"]
    assert all(a["phase"] == "exited" and a["exit_code"] == 0 and a["process"]["created"] == 10000 for a in state["attempts"])
    assert (cohort.root / ".export-supervisor/stdout.log").stat().st_size > 0
    assert (cohort.attempt / "ABORT.json").read_bytes() == before_abort


@pytest.mark.parametrize("prior", ["same_slot", "ten_repairs", "eleven_waves"])
def test_durable_bounds_block_new_launch(cohort, monkeypatch, prior):
    state = ready_state()
    if prior == "same_slot":
        state["repairs"] = [cohort.job["key"]]
    elif prior == "ten_repairs":
        state["repairs"] = [j["key"] for j in wave.jobs_for(cohort.root)]
    else:
        state["attempts"] = [dict(kind="wave", phase="exited", exit_code=1) for _ in range(11)]
        monkeypatch.setattr(supervisor, "inspect", lambda *a: (False, []))
    lr.save(cohort.root / ".export-supervisor/state.json", state)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("bound allowed a launch"))
    with pytest.raises(ValueError, match="bound|Bounded"):
        supervisor.supervise()


def test_ten_repairs_once_each_then_resume(cohort, monkeypatch):
    slots = [j["key"] for j in wave.jobs_for(cohort.root)]
    calls, finished = [], [False]
    monkeypatch.setattr(supervisor, "inspect", lambda *a: (True, []) if finished[0] else (False, slots))
    monkeypatch.setattr(recovery, "verify_complete", lambda *a: None)

    def launch(lr, wave, root, folder, state, command, kind):
        calls.append((kind, command))
        if kind == "wave":
            finished[0] = True
            lr.save(root / "process.json", {"new": True})
            lr.save(root / "completion.json", {"synthetic": True})
        return 0

    monkeypatch.setattr(supervisor, "launched", launch)
    assert supervisor.supervise() == "completed"
    assert len(calls) == 11 and [c[0] for c in calls] == ["export"] * 10 + ["wave"]
    assert lr.read(cohort.root / ".export-supervisor/state.json")["repairs"] == slots


def test_export_failure_never_continues(cohort, monkeypatch):
    calls = []

    def launch(*args):
        calls.append(args[-1])
        return 1

    monkeypatch.setattr(supervisor, "launched", launch)
    with pytest.raises(RuntimeError, match="Export failed"):
        supervisor.supervise()
    assert calls == ["export"]


@pytest.mark.parametrize("code", [1, 2, 17])
def test_wrong_coordinator_failures_not_retried(cohort, monkeypatch, code):
    recovery.export(cohort.folder)
    calls = []

    def launch(lr, wave, root, folder, state, command, kind):
        calls.append(kind)
        lr.save(root / "process.json", {"new": True})
        lr.save(root / "status.json", {"status": "failed", "error": "arbitrary failure"})
        return code

    monkeypatch.setattr(supervisor, "launched", launch)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        supervisor.supervise()
    assert calls == ["wave"]


def test_stop_after_export_prevents_resume(cohort, monkeypatch):
    calls = []

    def launch(lr, wave, root, folder, state, command, kind):
        calls.append(kind)
        recovery.export(cohort.folder)
        (root / "STOP").touch()
        return 0

    monkeypatch.setattr(supervisor, "launched", launch)
    with pytest.raises(lr.Stopped):
        supervisor.supervise()
    assert calls == ["export"]


def test_launched_waits_before_stop_exception(cohort, monkeypatch):
    waited = []
    folder = cohort.root / ".export-supervisor"
    folder.mkdir()

    class Child:
        pid = 500

        def __init__(self, command, **k):
            self.command = command
            cohort.statuses[500] = ("live", 10000)

        def poll(self):
            register_fake_coordinator(cohort, self.command)
            return None

        def wait(self):
            waited.append(True)
            finish_fake_coordinator(cohort, self.command)
            cohort.statuses[500] = ("dead", 10000)
            (cohort.root / "STOP").touch()
            return 2

    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    with pytest.raises(lr.Stopped):
        supervisor.launched(lr, wave, cohort.root, folder, ready_state(), supervisor.bootstrap.runner_command(cohort.root), "wave")
    assert waited == [True]


@pytest.mark.parametrize("after_launcher", ["live", "unknown"])
def test_actual_children_drained_or_unknown_blocks(cohort, monkeypatch, after_launcher):
    folder = cohort.root / ".export-supervisor"
    folder.mkdir()
    sleeps = []

    class Child:
        pid = 500

        def __init__(self, command, **k):
            self.command = command
            cohort.statuses[500] = ("live", 10000)

        def poll(self):
            register_fake_coordinator(cohort, self.command)
            return None

        def wait(self):
            finish_fake_coordinator(cohort, self.command)
            cohort.statuses[500] = ("dead", 10000)
            cohort.statuses[300] = (after_launcher, 3000 if after_launcher == "live" else None)
            (cohort.root / "STOP").touch()
            return 2

    def drained(seconds):
        sleeps.append(seconds)
        cohort.statuses[300] = ("dead", 3000)

    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    monkeypatch.setattr(supervisor.time, "sleep", drained)
    with pytest.raises(lr.Stopped if after_launcher == "live" else OSError):
        supervisor.launched(lr, wave, cohort.root, folder, ready_state(), supervisor.bootstrap.runner_command(cohort.root), "wave")
    assert sleeps == ([1] if after_launcher == "live" else [])


def test_spawn_bookkeeping_failure_still_waits(cohort, monkeypatch):
    folder = cohort.root / ".export-supervisor"
    folder.mkdir()
    waits = []
    save = lr.durable_save

    class Child:
        pid = 500

        def __init__(self, *a, **k):
            cohort.statuses[500] = ("live", 5000)

        def wait(self):
            waits.append(True)
            cohort.statuses[500] = ("dead", 5000)
            return 1

    def fail(path, value):
        if value.get("phase") == "running":
            raise OSError("synthetic durable write failed")
        save(path, value)

    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    monkeypatch.setattr(lr, "durable_save", fail)
    with pytest.raises(OSError, match="durable write"):
        supervisor.launched(lr, wave, cohort.root, folder, ready_state(), ["synthetic"], "export")
    assert waits == [True]


def test_supervisor_unknown_creation_does_not_launch(cohort, monkeypatch):
    cohort.statuses[999] = ("unknown", None)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected launch"))
    with pytest.raises(OSError, match="UNKNOWN"):
        supervisor.supervise()
