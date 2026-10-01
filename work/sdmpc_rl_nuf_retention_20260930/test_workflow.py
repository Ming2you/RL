"""Five-slot ownership and explicit conditional stages without numerical children."""
import copy
from types import SimpleNamespace as NS
import pytest
import local_runtime as lr
import run_wave
import collect
import probe
import launch_identity
from conftest import IDENTITY


def test_exact_five_local_slots_and_independent_resume(tmp_path, synthetic):
    jobs = run_wave.jobs_for(tmp_path)
    assert len(jobs) == 5 and {j["behavior"] for j in jobs} == {"local"}
    tokens = [lr.reserve_worker(tmp_path / j["key"], j["scenario"], j["behavior"], j["cpu_mask"]) for j in jobs]
    records = lr.read(tmp_path / ".ownership/reservations.json")
    assert len({r["run_id"] for r in records}) == 5
    for i, (job, token) in enumerate(zip(jobs, tokens)):
        folder = tmp_path / job["key"]
        with pytest.raises(OSError, match="reservation"):
            lr.reserve_worker(folder, job["scenario"], "local", job["cpu_mask"])
        lr.checkpoint_progress(token, records[i]["run_id"], i+1)
        lr.save(folder / "settings.json", dict(run_id=records[i]["run_id"]))
        lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", {"synthetic": True})
        lr.release_worker(token)
        with pytest.raises(ValueError, match="resume"):
            lr.reserve_worker(folder, job["scenario"], "local", job["cpu_mask"])
        resumed = lr.reserve_worker(folder, job["scenario"], "local", job["cpu_mask"], resume=True)
        assert lr.reservation_run_id(resumed) == records[i]["run_id"]
        with pytest.raises(ValueError, match="progress"):
            lr.checkpoint_progress(resumed, records[i]["run_id"], i)
        lr.checkpoint_progress(resumed, records[i]["run_id"], 75)
        lr.release_worker(resumed, finished=True)
        lr.save(folder / "completion.json", {"synthetic": True})
        with pytest.raises(ValueError, match="completed"):
            lr.reserve_worker(folder, job["scenario"], "local", job["cpu_mask"], resume=True)
    with pytest.raises(ValueError, match="slot"):
        lr.canonical_worker(tmp_path / "carry" / lr.SCENARIOS[0], lr.SCENARIOS[0], "carry", lr.MASKS[0])


@pytest.mark.parametrize("stage,passed,expected", [("pilot", False, 1), ("pilot", True, 1),
                                                ("remaining", True, 4)])
def test_stage_selection(stage, passed, expected):
    jobs = run_wave.jobs_for(None)
    selected = run_wave.select_stage(jobs, stage, dict(integrity_pass=True, advance_allowed=passed))
    assert len(selected) == expected
    assert all((j["scenario"] == probe.PILOT) == (stage == "pilot") for j in selected)


@pytest.mark.parametrize("pilot", [None, dict(integrity_pass=False, advance_allowed=True),
                                dict(integrity_pass=True, advance_allowed=False)])
def test_pilot_failure_blocks_remaining(pilot):
    with pytest.raises(ValueError, match="Remaining four"):
        run_wave.select_stage(run_wave.jobs_for(None), "remaining", pilot)


def mock_coordinator(monkeypatch, root):
    dispatched = []
    monkeypatch.setattr(run_wave, "ensure_cohort_idle", lambda output: None)
    monkeypatch.setattr(run_wave, "validate_job", lambda *args: None)
    def staged(output, jobs, *args):
        for job in jobs:
            path = output / job["key"] / "completion.json"
            if not path.exists():
                dispatched.append(job["scenario"])
                lr.save(path, {"synthetic": True})
    monkeypatch.setattr(run_wave, "run_stage", staged)
    monkeypatch.setattr(run_wave, "balanced_report", lambda *args: {"all_integrity_and_screens_pass": True})
    return dispatched


@pytest.mark.parametrize("passed", [True, False])
def test_full_staging_never_repeats_pilot(tmp_path, synthetic, passed):
    root = tmp_path / "new-root"
    synthetic.patch.setattr(lr, "COHORT_ROOT", root)
    dispatched = mock_coordinator(synthetic.patch, root)
    synthetic.patch.setattr(run_wave, "pilot_report", lambda *args:
        dict(integrity_pass=True, advance_allowed=passed, identity=IDENTITY))
    assert run_wave.run(root, stage="pilot") == ("pilot_passed" if passed else "pilot_blocked")
    plan = lr.read(root / "plan.json")
    assert plan["episodes"] == 5 and plan["transitions"] == 375
    assert dispatched == [probe.PILOT]
    run_wave.run(root, resume=True, stage="pilot")
    assert dispatched == [probe.PILOT]
    if passed:
        assert run_wave.run(root, resume=True, stage="remaining") == "completed"
        assert sorted(dispatched) == sorted(lr.SCENARIOS) and dispatched.count(probe.PILOT) == 1
    else:
        with pytest.raises(ValueError, match="Remaining four"):
            run_wave.run(root, resume=True, stage="remaining")
        assert dispatched == [probe.PILOT]
    process = lr.read(root / "process.json")
    assert type(process["created"]) is int and process["created"] > 0
    assert lr.read(root / "plan.json") == plan


def test_changed_candidate_blocks_remaining_before_dispatch(tmp_path, synthetic):
    root = tmp_path / "new-root"
    synthetic.patch.setattr(lr, "COHORT_ROOT", root)
    dispatched = mock_coordinator(synthetic.patch, root)
    synthetic.patch.setattr(run_wave, "pilot_report", lambda *args: dict(integrity_pass=True, advance_allowed=True))
    run_wave.run(root, stage="pilot")
    synthetic.patch.setattr(run_wave, "identity", lambda: dict(IDENTITY, local_sources={"changed": True}))
    with pytest.raises(ValueError, match="plan/source"):
        run_wave.run(root, resume=True, stage="remaining")
    assert dispatched == [probe.PILOT]


@pytest.mark.parametrize("status", ["failed", "unknown", "export-completed"])
def test_arbitrary_failed_work_is_not_retried(tmp_path, synthetic, status):
    job = run_wave.jobs_for(tmp_path)[0]
    folder = tmp_path / job["key"]
    lr.save(folder / "status.json", dict(status=status))
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", {"synthetic": True})
    with pytest.raises(ValueError, match="Failed/unknown"):
        run_wave.command_for(tmp_path, job, tmp_path / "abort")


def test_completed_stage_skips_before_spawn(tmp_path, synthetic):
    job = run_wave.jobs_for(tmp_path)[1]
    lr.save(tmp_path / job["key"] / "completion.json", {"synthetic": True})
    synthetic.patch.setattr(run_wave, "validate_job", lambda *a: None)
    synthetic.patch.setattr(run_wave.subprocess, "Popen", lambda *a, **k: pytest.fail("completed pilot spawned"))
    run_wave.run_stage(tmp_path, [job], IDENTITY, tmp_path / "ABORT.json", [job])


def test_unknown_creation_and_orphan_worker_fail_closed(tmp_path, synthetic):
    synthetic.patch.setattr(lr, "probe_process", lambda pid: ("unknown", None))
    with pytest.raises(OSError, match="creation"):
        lr.process_identity()
    job = run_wave.jobs_for(tmp_path)[0]
    rec = dict(key=job["key"], state="active", worker=dict(pid=30, created=300), launcher=dict(pid=20, created=200),
               owner=dict(pid=10, created=100), cohort_phase="checkpointed")
    lr.save(tmp_path / ".ownership/reservations.json", [rec])
    with pytest.raises(OSError, match="UNKNOWN"):
        lr.ensure_cohort_idle(tmp_path)


def test_redirector_claim_and_actual_creation_identity(monkeypatch):
    owner, launcher, worker = [dict(pid=p, created=p*10) for p in (10, 20, 30)]
    monkeypatch.setattr(launch_identity.sys, "executable", "venv/python.exe")
    monkeypatch.setattr(launch_identity.sys, "_base_executable", "base/python.exe")
    live = lambda pid: ("live", pid*10)
    assert launch_identity.verify_claim(launcher, worker, owner, 20, probe=live) == "windows-venv-redirector"
    with pytest.raises(ValueError, match="parent"):
        launch_identity.verify_claim(launcher, worker, owner, 10, probe=live)
    with pytest.raises(ValueError, match="UNKNOWN"):
        launch_identity.verify_claim(launcher, dict(worker, created=999), owner, 20, probe=live)


def test_drain_waits_actual_worker_after_redirector_exit(tmp_path, synthetic):
    record = dict(token="token", worker=dict(pid=30, created=300), launcher=dict(pid=20, created=200))
    lr.save(tmp_path / ".ownership/reservations.json", [record])
    state = {20: ("dead", 200), 30: ("live", 300)}
    synthetic.patch.setattr(run_wave, "process_dead", lambda p: state[p["pid"]][0] == "dead")
    synthetic.patch.setattr(run_wave, "probe_process", lambda pid: state[pid])
    sleeps = []
    def slept(delay):
        sleeps.append(delay)
        state[30] = ("dead", 300)
    synthetic.patch.setattr(run_wave.time, "sleep", slept)
    run_wave.wait_worker_exit(tmp_path, "token")
    assert len(sleeps) == 1
    state[30] = ("unknown", None)
    with pytest.raises(OSError, match="UNKNOWN"):
        run_wave.wait_worker_exit(tmp_path, "token")


def test_all_launchers_drained_even_when_actual_identity_unknown(tmp_path, synthetic):
    children = []
    class Child:
        def __init__(self):
            self.pid, self.waited = 100+len(children), False
        def poll(self):
            return 1
        def wait(self):
            self.waited = True
            return 1
    def spawn(*a, **k):
        child = Child()
        children.append(child)
        return child
    def unknown(*a):
        raise OSError("Actual worker UNKNOWN")
    synthetic.patch.setattr(run_wave.subprocess, "Popen", spawn)
    synthetic.patch.setattr(run_wave, "bind_launch", lambda *a: None)
    synthetic.patch.setattr(run_wave, "process_identity", lambda pid, command: dict(pid=pid, created=pid*10, command=command))
    synthetic.patch.setattr(run_wave, "wait_worker_exit", unknown)
    jobs = run_wave.jobs_for(tmp_path)[:2]
    with pytest.raises(OSError, match="UNKNOWN"):
        run_wave.run_stage(tmp_path, jobs, IDENTITY, tmp_path / "attempt/ABORT.json", jobs)
    assert len(children) == 2 and all(c.waited for c in children)
    assert lr.read(tmp_path / "attempt/ABORT.json")["reason"] == "coordinator_draining"


def test_spawn_intent_is_retained_and_unknown_never_relaunched(tmp_path, synthetic):
    job = run_wave.jobs_for(tmp_path)[1]
    def interrupted(command, **kwargs):
        token = command[command.index("--reservation")+1]
        records = lr.read(tmp_path / ".ownership/reservations.json")
        assert records[-1]["token"] == token and records[-1]["state"] == "reserved"
        assert records[-1]["launcher"] is None
        raise RuntimeError("synthetic interrupted spawn")
    synthetic.patch.setattr(run_wave.subprocess, "Popen", interrupted)
    with pytest.raises(RuntimeError, match="interrupted spawn"):
        run_wave.run_stage(tmp_path, [job], IDENTITY, tmp_path / "attempt/ABORT.json", [job])
    with pytest.raises(OSError, match="UNKNOWN"):
        lr.ensure_cohort_idle(tmp_path)


def test_stop_any_slot_blocks_pilot_before_spawn(tmp_path, synthetic):
    jobs = run_wave.jobs_for(tmp_path)
    stopped = tmp_path / jobs[-1]["key"] / "STOP"
    stopped.parent.mkdir(parents=True)
    stopped.touch()
    synthetic.patch.setattr(run_wave.subprocess, "Popen", lambda *a, **k: pytest.fail("STOP ignored"))
    with pytest.raises(lr.Stopped):
        run_wave.run_stage(tmp_path, [jobs[1]], IDENTITY, tmp_path / "attempt/ABORT.json", jobs)


@pytest.mark.parametrize("trigger", ["nonzero_exit", "sibling_stop", "sibling_failure"])
def test_fix1_abort_precedes_actual_worker_death(tmp_path, synthetic, trigger, record_property):
    jobs = run_wave.jobs_for(tmp_path)
    abort = tmp_path / "attempt/ABORT.json"
    stop = tmp_path / jobs[-1]["key"] / "STOP"
    children, released, validated, aborts = [], [], [], []
    processes, ticks = {}, [0]
    original_probe = lr.probe_process
    original_bind, original_release, original_save = run_wave.bind_launch, run_wave.release_worker, run_wave.save

    def process_probe(pid):
        return processes[pid] if pid in processes else original_probe(pid)

    class Child:
        def __init__(self, command):
            self.index, self.pid = len(children), 1000001+len(children)
            self.worker_pid, self.waited = self.pid+100, False
            self.token = command[command.index("--reservation")+1]
            processes[self.pid] = ("live", self.pid*10)
            processes[self.worker_pid] = ("live", self.worker_pid*10)
        def poll(self):
            if self.index == 0 or trigger == "sibling_failure" and ticks[0] >= 1:
                processes[self.pid] = ("dead", self.pid*10)
                return int(trigger == "nonzero_exit" if self.index == 0 else True)
            return None
        def wait(self):
            assert abort.exists(), "launcher drain must follow shared ABORT"
            self.waited = True
            processes[self.pid] = ("dead", self.pid*10)
            return 0

    def spawn(command, **kwargs):
        child = Child(command)
        children.append(child)
        return child

    def bind(token, pid, command):
        original_bind(token, pid, command)
        child = next(c for c in children if c.pid == pid)
        with lr.ownership() as records:
            record = next(r for r in records if r["token"] == token)
            record.update(state="active", worker=dict(pid=child.worker_pid,
                created=child.worker_pid*10, command=command))

    def save(path, value):
        if path == abort:
            aborts.append(dict(tick=ticks[0], first_worker=processes[children[0].worker_pid][0],
                               released=list(released)))
        original_save(path, value)

    def release(token):
        child = next(c for c in children if c.token == token)
        assert processes[child.pid][0] == processes[child.worker_pid][0] == "dead"
        original_release(token)
        released.append(child.index)

    def tick(delay):
        ticks[0] += 1
        assert ticks[0] <= 5, "bounded synthetic drain must terminate"
        if trigger == "sibling_stop" and ticks[0] == 1:
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.touch()
        if abort.exists():
            for child in children:
                processes[child.worker_pid] = ("dead", child.worker_pid*10)
        elif ticks[0] == 3:
            # Bound the reviewed bug: let its blocking wait finish so evidence is saved.
            for child in children:
                processes[child.worker_pid] = ("dead", child.worker_pid*10)

    synthetic.patch.setattr(lr, "probe_process", process_probe)
    synthetic.patch.setattr(run_wave, "probe_process", process_probe)
    synthetic.patch.setattr(run_wave.subprocess, "Popen", spawn)
    synthetic.patch.setattr(run_wave, "bind_launch", bind)
    synthetic.patch.setattr(run_wave, "release_worker", release)
    synthetic.patch.setattr(run_wave, "save", save)
    synthetic.patch.setattr(run_wave, "validate_job", lambda *args: validated.append(args[1]["key"]))
    synthetic.patch.setattr(run_wave.time, "sleep", tick)
    error = lr.Stopped if trigger == "sibling_stop" else RuntimeError
    with pytest.raises(error):
        run_wave.run_stage(tmp_path, jobs[:2], IDENTITY, abort, jobs)
    import json
    record_property("abort_observations", json.dumps(aborts, sort_keys=True))
    record_property("launchers_drained", json.dumps([c.waited for c in children]))
    assert aborts == [dict(tick=0 if trigger == "nonzero_exit" else 1,
                          first_worker="live", released=[])]
    assert len(children) == 2 and all(c.waited for c in children)
    assert released == [0, 1] and not validated
    assert all(r["state"] == "exited" for r in lr.read(tmp_path / ".ownership/reservations.json"))
    if trigger == "sibling_stop":
        assert stop.exists()
