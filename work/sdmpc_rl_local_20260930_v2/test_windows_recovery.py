"""Ownership regressions, using fake identities and process-only subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_runtime as lr
import run_wave
from test_collection import synthetic


@pytest.fixture
def launch(monkeypatch, tmp_path):
    monkeypatch.setattr(lr, "COHORT_ROOT", tmp_path)
    monkeypatch.setattr(lr, "REQUIRE_MIGRATION", False)
    identities = {10: ("live", 100), 20: ("live", 200), 30: ("live", 300)}
    current = [10]
    monkeypatch.setattr(lr, "probe_process", lambda pid: identities.get(pid, ("unknown", None)))
    monkeypatch.setattr(lr.os, "getpid", lambda: current[0])
    monkeypatch.setattr(lr.os, "getppid", lambda: 20)
    job = run_wave.jobs_for(tmp_path)[0]
    call = (tmp_path / job["key"], job["scenario"], job["behavior"], job["cpu_mask"])
    token = lr.reserve_worker(*call, launch=True)
    lr.bind_launch(token, 20, [sys.executable, "-B", "collect.py"])
    current[0] = 30
    return call, token, identities, current


def test_redirector_claim_records_both_identities(launch):
    call, token, _, _ = launch
    assert lr.claim_worker(*call, token) == token
    record = lr.read(lr.COHORT_ROOT / ".ownership/reservations.json")[-1]
    assert record["launcher"]["pid"] == 20 and record["worker"]["pid"] == 30
    assert record["worker"]["parent_pid"] == 20
    with pytest.raises(ValueError, match="claimed"):
        lr.claim_worker(*call, token)


@pytest.mark.parametrize("fault", ["parent", "launcher_reuse", "launcher_unknown", "worker_unknown", "older_worker"])
def test_claim_rejects_unproven_identity(launch, monkeypatch, fault):
    call, token, identities, _ = launch
    if fault == "parent":
        monkeypatch.setattr(lr.os, "getppid", lambda: 99)
    elif fault == "launcher_reuse":
        identities[20] = ("live", 201)
    elif fault == "launcher_unknown":
        identities[20] = ("unknown", None)
    elif fault == "worker_unknown":
        identities[30] = ("unknown", None)
    else:
        identities[30] = ("live", 199)
    with pytest.raises((ValueError, OSError)):
        lr.claim_worker(*call, token)
    assert lr.read(lr.COHORT_ROOT / ".ownership/reservations.json")[-1]["state"] != "active"


def test_actual_worker_blocks_recovery_after_launcher_exit(launch):
    call, token, identities, current = launch
    lr.claim_worker(*call, token)
    identities[20] = ("dead", 200)
    current[0] = 10
    with pytest.raises(OSError, match="live|UNKNOWN"):
        lr.ensure_cohort_idle(lr.COHORT_ROOT)
    with pytest.raises(OSError, match="live|UNKNOWN"):
        lr.release_worker(token)
    identities[30] = ("unknown", None)
    with pytest.raises(OSError):
        lr.ensure_cohort_idle(lr.COHORT_ROOT)
    identities[30] = ("live", 301)  # Old worker exited and PID was reused.
    lr.ensure_cohort_idle(lr.COHORT_ROOT)


def test_direct_child_claim_and_wrong_parent(launch, monkeypatch):
    call, token, _, current = launch
    current[0] = 20
    with pytest.raises(ValueError, match="parent"):
        lr.claim_worker(*call, token)
    monkeypatch.setattr(lr.os, "getppid", lambda: 10)
    assert lr.claim_worker(*call, token) == token
    assert lr.read(lr.COHORT_ROOT / ".ownership/reservations.json")[-1]["launch_mode"] == "direct"


def test_bind_is_owner_only_and_once_only(launch):
    _, token, _, current = launch
    with pytest.raises(ValueError, match="once-only"):
        lr.bind_launch(token, 30, [])
    current[0] = 10
    with pytest.raises(ValueError, match="once-only"):
        lr.bind_launch(token, 20, [])


def test_child_waits_for_binding_without_claiming_early(launch, monkeypatch):
    call, token, _, current = launch
    path = lr.COHORT_ROOT / ".ownership/reservations.json"
    records = lr.read(path)
    records[-1]["launcher"] = None
    lr.save(path, records)
    sleeps = []
    def delayed_bind(_):
        record = lr.read(path)[-1]
        assert record["worker"] is None and record["state"] == "reserved"
        sleeps.append(True)
        current[0] = 10
        lr.bind_launch(token, 20, [])
        current[0] = 30
    monkeypatch.setattr(lr.time, "sleep", delayed_bind)
    lr.claim_worker(*call, token)
    assert sleeps == [True]


def test_unbound_dead_owner_stays_unknown(launch):
    call, token, identities, _ = launch
    path = lr.COHORT_ROOT / ".ownership/reservations.json"
    records = lr.read(path)
    records[-1]["launcher"] = None
    lr.save(path, records)
    identities[10] = ("dead", 100)
    with pytest.raises(OSError, match="UNKNOWN"):
        lr.claim_worker(*call, token)
    with pytest.raises(OSError, match="UNKNOWN"):
        lr.ensure_cohort_idle(lr.COHORT_ROOT)


def test_five_redirected_workers_still_consume_five_slots(launch, monkeypatch):
    call, token, identities, current = launch
    lr.claim_worker(*call, token)
    jobs = run_wave.jobs_for(lr.COHORT_ROOT)
    for i, job in enumerate(jobs[1:5], 1):
        current[0] = 10
        request = (lr.COHORT_ROOT / job["key"], job["scenario"], job["behavior"], job["cpu_mask"])
        launcher, worker = 20+i, 30+i
        identities[launcher], identities[worker] = ("live", 200+i), ("live", 300+i)
        reserved = lr.reserve_worker(*request, launch=True)
        lr.bind_launch(reserved, launcher, [])
        current[0] = worker
        monkeypatch.setattr(lr.os, "getppid", lambda pid=launcher: pid)
        lr.claim_worker(*request, reserved)
    job = jobs[5]
    with pytest.raises(OSError, match="five"):
        lr.reserve_worker(lr.COHORT_ROOT / job["key"], job["scenario"], job["behavior"], job["cpu_mask"])
    records = lr.read(lr.COHORT_ROOT / ".ownership/reservations.json")
    assert len(records) == 5 and all(r["state"] == "active" for r in records)


def test_real_venv_redirector_process_only():
    # This child imports only the stdlib ownership verifier, never local_runtime.
    code = """import json,os,sys
from launch_identity import probe_process, verify_claim
launch = json.loads(sys.stdin.readline())
worker = dict(pid=os.getpid(), created=probe_process(os.getpid())[1])
verify_claim(launch['launcher'], worker, launch['owner'], os.getppid())
print(json.dumps(dict(worker=worker,parent=os.getppid(),numerical_imports=any(x in sys.modules for x in ('numpy','torch')))),flush=True)
sys.stdin.readline()
"""
    command = [sys.executable, "-B", "-c", code]
    child = subprocess.Popen(command, cwd=Path(__file__).parent, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        from launch_identity import probe_process
        request = dict(launcher=dict(pid=child.pid, created=probe_process(child.pid)[1]),
                       owner=dict(pid=os.getpid(), created=probe_process(os.getpid())[1]))
        out, err = child.communicate(json.dumps(request)+"\nexit\n", timeout=15)
        assert child.returncode == 0, err
        result = json.loads(out)
        assert result["worker"]["pid"] != child.pid
        assert result["parent"] == child.pid and not result["numerical_imports"]
        print(json.dumps(dict(launcher=child.pid, **result)))
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=15)


def test_drain_waits_all_launchers_even_if_actual_worker_is_unknown(tmp_path, synthetic):
    children = []
    class Child:
        def __init__(self):
            self.pid = 100 + len(children)
            self.waited = False
        def poll(self):
            return 1
        def wait(self):
            self.waited = True
            return 1
    def spawn(*a, **kw):
        child = Child()
        children.append(child)
        return child
    def blocked_release(token):
        raise OSError("Actual worker is UNKNOWN")
    synthetic.setattr(run_wave.subprocess, "Popen", spawn)
    synthetic.setattr(run_wave, "release_worker", blocked_release)
    jobs = run_wave.jobs_for(tmp_path)[:2]
    with pytest.raises(OSError, match="UNKNOWN"):
        run_wave.run_stage(tmp_path, jobs, {}, tmp_path / "attempt/ABORT.json", jobs)
    assert len(children) == 2 and all(child.waited for child in children)
