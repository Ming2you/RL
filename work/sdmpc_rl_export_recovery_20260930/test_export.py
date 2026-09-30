import copy
import json
import math

import pytest

import export_recovery as recovery
from conftest import lr, collector, validator, wave, IDENTITY, TOKEN


def contents(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def save_checkpoint(cohort):
    lr.checkpoint_save(lr.torch, cohort.folder / "checkpoint.pt", cohort.ck)


def test_export_original_validators_retained_identity_and_complete_retry(cohort, monkeypatch):
    before = contents(cohort.folder)
    original_summary = validator.summarize(cohort.ck["trace"], cohort.settings, "KNOWN")
    calls = []
    original_loader = validator.load_completed

    def load(*a, **k):
        calls.append(a[0])
        return original_loader(*a, **k)

    monkeypatch.setattr(validator, "load_completed", load)
    receipt = recovery.export(cohort.folder)
    result, settings, summary = original_loader(cohort.folder, IDENTITY)
    assert len(cohort.boot_calls) == 1
    assert calls == [cohort.folder / recovery.STAGE]
    assert settings == cohort.settings and summary["ttt"] == 80.
    assert lr.read(cohort.folder / "summary.json") == original_summary
    assert result["outputs_sha256"].keys() == set(recovery.OUTPUTS)
    assert result["export_recovery"]["sha256"] == lr.file_hash(cohort.folder / recovery.RECEIPT)
    assert len(receipt["changed_paths"]) == 4 and receipt["export_elapsed_wall_seconds"] > 0.
    assert receipt["convergence_claim"] is False and receipt["true_terminals"] == 1
    exported = lr.read(cohort.folder / "trace.json")
    assert exported[14]["selection_source"] == "reference_fallback"
    assert exported[14]["candidates"][0]["converged"] is False
    assert exported[14]["candidates"][0]["stationarity"] == {"nonfinite_float": "+inf"}
    raw = lr.torch.load(cohort.folder / "checkpoint.pt", weights_only=False)
    assert math.isinf(raw["trace"][14]["candidates"][0]["stationarity"])
    payload = lr.torch.load(cohort.folder / "experience.pt", weights_only=False)
    assert lr.digest(payload["transitions"]) == lr.digest(raw["transitions"])
    for name, data in before.items():
        if name != "status.json":
            assert (cohort.folder / name).read_bytes() == data
    assert (cohort.folder / recovery.STAGE / "original-status.json").read_bytes() == before["status.json"]
    retained = lr.read(cohort.root / ".ownership/reservations.json")[-1]
    assert retained["cohort_phase"] == "finished" and retained["control_steps"] == 75
    assert retained["token"] == TOKEN and retained["run_id"] == cohort.record["run_id"]
    after = contents(cohort.root)
    assert recovery.export(cohort.folder) == receipt
    assert contents(cohort.root) == after and len(cohort.boot_calls) == 1


@pytest.mark.parametrize("fault", ["length", "terminal", "state", "reward", "action", "physical_nan", "extra_inf", "policy"])
def test_reject_invalid_checkpoint(cohort, fault):
    ck = cohort.ck
    if fault == "length":
        ck["trace"].pop()
    elif fault == "terminal":
        ck["transitions"][-1] = (*ck["transitions"][-1][:-1], False)
    elif fault == "state":
        ck["environment"]["sim"].total_ttt += 1.
    elif fault == "reward":
        ck["transitions"][0] = (ck["transitions"][0][0], ck["transitions"][0][1], float("inf"), *ck["transitions"][0][3:])
    elif fault == "action":
        ck["transitions"][0][1][0] = float("nan")
    elif fault == "physical_nan":
        ck["environment"]["sim"].state.freeway_speed["a"][0] = float("nan")
    elif fault == "extra_inf":
        ck["trace"][0]["execution_check"]["stationarity"] = float("inf")
    else:
        ck["policy"]["count"] -= 1
    save_checkpoint(cohort)
    with pytest.raises((ValueError, AssertionError)):
        recovery.export(cohort.folder)
    assert not (cohort.folder / "completion.json").exists()


@pytest.mark.parametrize("artifact", ["checkpoint.pt", "settings.json", "observation_schema.json", "timing.json", "stderr.log", "process.json"])
def test_missing_artifacts_fail_before_publication(cohort, artifact):
    (cohort.folder / artifact).unlink()
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        recovery.export(cohort.folder)
    assert not (cohort.folder / "completion.json").exists()


@pytest.mark.parametrize("artifact", ["settings.json", "timing.json", "observation_schema.json", "sessions/" + TOKEN + ".end.json"])
def test_changed_artifacts_rejected(cohort, artifact):
    value = lr.read(cohort.folder / artifact)
    value["unadmitted"] = True
    lr.save(cohort.folder / artifact, value)
    with pytest.raises((ValueError, AssertionError)):
        recovery.export(cohort.folder)


def test_source_drift_rejected(cohort, monkeypatch):
    monkeypatch.setattr(lr, "identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="source"):
        recovery.export(cohort.folder)
    assert not cohort.boot_calls


@pytest.mark.parametrize("fault", ["missing_source", "changed_source", "runtime"])
def test_source_and_runtime_authentication_fail_closed(cohort, monkeypatch, fault):
    source = cohort.root / "synthetic-source.py"
    source.write_text("immutable fixture")
    expected = lr.file_hash(source)

    def authenticated():
        if lr.file_hash(source) != expected or fault == "runtime":
            raise ValueError("Source/runtime changed")
        return IDENTITY

    monkeypatch.setattr(lr, "identity", authenticated)
    if fault == "missing_source":
        source.unlink()
    elif fault == "changed_source":
        source.write_text("changed fixture")
    with pytest.raises((FileNotFoundError, ValueError)):
        recovery.export(cohort.folder)
    assert not cohort.boot_calls


@pytest.mark.parametrize("who", [100, 200, 300])
@pytest.mark.parametrize("state", ["live", "unknown"])
def test_live_or_unknown_blocks_before_boot(cohort, who, state):
    cohort.statuses[who] = (state, who*10 if state == "live" else None)
    with pytest.raises(OSError, match="live|UNKNOWN"):
        recovery.export(cohort.folder)
    assert not cohort.boot_calls


def test_pid_reuse_proves_original_exit(cohort):
    cohort.statuses[300] = ("live", 3001)
    recovery.export(cohort.folder)


@pytest.mark.parametrize("fault", ["error", "traceback", "creation", "cohort_steps", "run_id", "STOP", "sibling_STOP", "relocated"])
def test_fail_closed_identity_failure_stop(cohort, fault):
    if fault == "error":
        lr.save(cohort.folder / "status.json", dict(status="failed", pid=300, error="ValueError: physical inf"))
    elif fault == "traceback":
        (cohort.folder / "stderr.log").write_text(recovery.ERROR)
    elif fault in ("creation", "cohort_steps", "run_id"):
        record = copy.deepcopy(cohort.record)
        if fault == "creation":
            record["owner"]["created"] = None
        else:
            record["control_steps" if fault == "cohort_steps" else "run_id"] = 74 if fault == "cohort_steps" else "wrong"
        lr.save(cohort.root / ".ownership/reservations.json", [record])
    elif fault in ("STOP", "sibling_STOP"):
        folder = cohort.folder if fault == "STOP" else cohort.root / wave.jobs_for(cohort.root)[1]["key"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "STOP").touch()
    target = cohort.folder if fault != "relocated" else cohort.root / "wrong" / cohort.job["scenario"]
    with pytest.raises((ValueError, OSError, lr.Stopped)):
        recovery.export(target)
    assert not cohort.boot_calls


@pytest.mark.parametrize("name", ["trace.json", "summary.json", "experience.pt"])
def test_conflicting_output_never_overwritten(cohort, name):
    (cohort.folder / name).write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        recovery.export(cohort.folder)
    assert (cohort.folder / name).read_bytes() == b"keep me"


@pytest.mark.parametrize("point", ["stage", "output", "receipt", "completion"])
def test_interruption_preserves_stage_and_blocks_retry(cohort, monkeypatch, point):
    publish = recovery.publish_bytes

    def interrupt(path, data):
        selected = {"stage": cohort.folder / recovery.STAGE / "trace.json",
                    "output": cohort.folder / "summary.json", "receipt": cohort.folder / recovery.RECEIPT,
                    "completion": cohort.folder / "completion.json"}[point]
        if path == selected:
            raise RuntimeError("synthetic interruption")
        publish(path, data)

    monkeypatch.setattr(recovery, "publish_bytes", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        recovery.export(cohort.folder)
    assert not (cohort.folder / "completion.json").exists()
    assert (cohort.folder / recovery.STAGE / "original-status.json").exists()
    before = contents(cohort.root)
    with pytest.raises(FileExistsError, match="Interrupted"):
        recovery.export(cohort.folder)
    assert contents(cohort.root) == before


@pytest.mark.parametrize("name", ["trace.json", "checkpoint.pt", "status.json", "export-receipt.json", "sessions/" + TOKEN + ".start.json"])
def test_complete_retry_detects_tampering(cohort, name):
    recovery.export(cohort.folder)
    path = cohort.folder / name
    if name == "status.json":
        lr.save(path, {"status": "wrong"})
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        recovery.export(cohort.folder)


def test_completion_is_last(cohort, monkeypatch):
    events, publish = [], recovery.publish_bytes

    def record(path, data):
        events.append(path)
        if path == cohort.folder / "completion.json":
            retained = lr.read(cohort.root / ".ownership/reservations.json")[-1]
            assert retained["cohort_phase"] == "finished"
            assert (cohort.folder / recovery.RECEIPT).exists()
        publish(path, data)

    monkeypatch.setattr(recovery, "publish_bytes", record)
    recovery.export(cohort.folder)
    assert events[-1] == cohort.folder / "completion.json"


def test_input_change_during_validation_preserved_and_rejected(cohort, monkeypatch):
    original = validator.load_completed

    def changed(*a, **k):
        result = original(*a, **k)
        (cohort.folder / "stdout.log").write_text("changed")
        return result

    monkeypatch.setattr(validator, "load_completed", changed)
    with pytest.raises(ValueError, match="changed during"):
        recovery.export(cohort.folder)
    assert not (cohort.folder / "completion.json").exists()


@pytest.mark.parametrize("point", ["STOP", "output_tamper"])
def test_final_publication_guard(cohort, monkeypatch, point):
    original = lr.release_worker

    def release(*a, **k):
        original(*a, **k)
        if point == "STOP":
            (cohort.root / "STOP").touch()
        else:
            (cohort.folder / "trace.json").write_text("[]")

    monkeypatch.setattr(lr, "release_worker", release)
    with pytest.raises((lr.Stopped, ValueError)):
        recovery.export(cohort.folder)
    assert not (cohort.folder / "completion.json").exists()
    assert (cohort.folder / recovery.STAGE / "original-status.json").exists()


@pytest.mark.parametrize("lock", ["cohort", "slot"])
def test_original_locks_exclude_export(cohort, lock):
    with lr.exclusive_run(cohort.root if lock == "cohort" else cohort.folder):
        with pytest.raises(OSError):
            recovery.export(cohort.folder)
    assert not cohort.boot_calls


def test_retains_previous_sessions_and_75_step_record(cohort):
    older = copy.deepcopy(cohort.record)
    older.update(token="c"*32, control_steps=2)
    lr.save(cohort.root / ".ownership/reservations.json", [older, cohort.record])
    start = cohort.folder / "sessions" / (older["token"] + ".start.json")
    lr.save(start, dict(session_id=older["token"], process=older["worker"], started=1.))
    lr.save(start.with_name(older["token"] + ".end.json"), dict(session_id=older["token"],
        start_sha256=lr.file_hash(start), outcome="checkpointed", elapsed_wall_seconds=3.))
    cohort.ck["session_ids"].append(older["token"])
    save_checkpoint(cohort)
    lr.save(cohort.folder / "timing.json", lr.session_timing(cohort.folder, cohort.ck["session_ids"]))
    before = contents(cohort.folder / "sessions")
    recovery.export(cohort.folder)
    records = lr.read(cohort.root / ".ownership/reservations.json")
    assert records[0] == older and records[1]["cohort_phase"] == "finished"
    assert records[1]["control_steps"] == 75 and records[1]["token"] == cohort.record["token"]
    assert contents(cohort.folder / "sessions") == before
    assert validator.load_completed(cohort.folder, IDENTITY)[2]["elapsed_wall_seconds"] == 15.


@pytest.mark.parametrize("job_index,inf_count", [(1, 7), (4, 5), (5, 4)])
def test_later_slots_use_same_allowlist(cohort, job_index, inf_count):
    from conftest import static_checkpoint
    job = wave.jobs_for(cohort.root)[job_index]
    folder = cohort.root / job["key"]
    folder.parent.mkdir(parents=True, exist_ok=True)
    cohort.folder.rename(folder)
    settings = dict(cohort.settings, **{k: v for k, v in job.items() if k != "key"})
    record = dict(cohort.record, key=job["key"], cpu_mask=job["cpu_mask"],
                  training_seed=job["training_seed"], exploration_seed=job["exploration_seed"])
    ck = static_checkpoint(settings)
    for index in range(inf_count - 4):
        ck["trace"][index]["candidates"][0]["stationarity"] = float("inf")
    lr.save(folder / "settings.json", settings)
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", ck)
    lr.save(cohort.root / ".ownership/reservations.json", [record])
    receipt = recovery.export(folder)
    assert receipt["slot"] == job["key"] and len(receipt["changed_paths"]) == inf_count
    validator.load_completed(folder, IDENTITY)
