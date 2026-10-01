"""Scoped end-to-end serialization/resume tests using only FakeEnv."""
import copy
import math
import pytest
import collect
import validate
import diagnostics
import references
import local_runtime as lr
from conftest import request, IDENTITY, FakeEnv


def test_checkpoint_resume_terminal_diagnostic_export(tmp_path, synthetic):
    args = request(tmp_path, limit=10)
    assert collect.run(args) == "checkpointed"
    before = lr.torch.load(args.output / "checkpoint.pt", weights_only=False)
    assert len(before["trace"]) == 10
    run_id = before["settings"]["run_id"]
    assert collect.run(request(tmp_path, resume=True)) == "completed"
    result, settings, summary = validate.load_completed(args.output, IDENTITY)
    ck = lr.torch.load(args.output / "checkpoint.pt", weights_only=False)
    payload = lr.torch.load(args.output / "experience.pt", weights_only=False)
    assert lr.digest(ck["transitions"]) == lr.digest(payload["transitions"])
    assert len(ck["trace"]) == 75 and sum(t[-1] for t in payload["transitions"]) == 1
    assert settings["run_id"] == run_id and summary["simulation_seconds"] == 14400
    assert math.isinf(ck["trace"][9]["candidates"][0]["stationarity"])
    audit = lr.read(args.output / diagnostics.RECEIPT)
    assert len(audit["changed_paths"]) == 150 and audit["core_experience_unchanged"]
    assert summary["timing_status"] == "KNOWN" and len(summary["session_ids"]) == 2
    assert result["outputs_sha256"][diagnostics.RECEIPT] == lr.file_hash(args.output / diagnostics.RECEIPT)
    assert lr.read(tmp_path / ".ownership/reservations.json")[-1]["control_steps"] == 75
    assert [r["exploration_audit"] for r in ck["trace"][:10]] == [r["exploration_audit"] for r in before["trace"]]
    with pytest.raises(ValueError, match="completed"):
        request(tmp_path, resume=True)


@pytest.mark.parametrize("field", ["profile_sha256", "source_identity_sha256", "physical", "run_id", "warmup_ttt", "training_seed"])
def test_paired_binding_rejects_source_profile_or_run_change(tmp_path, synthetic, field):
    args = request(tmp_path, limit=1)
    collect.run(args)
    settings = lr.read(args.output / "settings.json")
    if field == "source_identity_sha256":
        settings["paired_references"]["local"][field] = "altered"
    elif field == "physical":
        settings["identity"][field] = {"changed": True}
    elif field == "run_id":
        settings[field] = settings["paired_references"]["local"]["run_id"]
    else:
        settings[field] = "wrong" if field == "profile_sha256" else 99999.
    with pytest.raises(ValueError):
        references.validate_binding(settings, synthetic.references)


@pytest.mark.parametrize("location,value", [("stationarity", float("nan")), ("stationarity", -float("inf")),
                                         ("reward", float("inf")), ("physical", float("inf"))])
def test_unknown_nonfinite_rejected(location, value):
    trace = [dict(candidates=[dict(stationarity=1., rows=[dict(primal_stationarity=1.)])], reward=0., physical=0.)]
    if location == "stationarity":
        trace[0]["candidates"][0][location] = value
    else:
        trace[0][location] = value
    with pytest.raises(ValueError, match="Forbidden"):
        diagnostics.tagger()(trace)


@pytest.mark.parametrize("fault", ["path", "experience", "checkpoint", "tag"])
def test_diagnostic_receipt_reconciles_paths_and_core(tmp_path, synthetic, fault):
    args = request(tmp_path)
    collect.run(args)
    trace = lr.read(args.output / "trace.json")
    payload = lr.torch.load(args.output / "experience.pt", weights_only=False)
    if fault == "path":
        audit = lr.read(args.output / diagnostics.RECEIPT)
        audit["changed_paths"][0]["path"][-1] = "physical"
        lr.save(args.output / diagnostics.RECEIPT, audit)
    elif fault == "experience":
        payload["transitions"][0][1][0] += .1
    elif fault == "checkpoint":
        lr.checkpoint_save(lr.torch, args.output / "checkpoint.pt", {"changed": True})
    else:
        trace[0]["physical"] = {"nonfinite_float": "+inf"}
    with pytest.raises(ValueError):
        diagnostics.validate_audit(args.output, trace, payload)


def test_stop_boundary_and_monotonic_resume(tmp_path, synthetic):
    args = request(tmp_path, limit=2)
    collect.run(args)
    ck = lr.torch.load(args.output / "checkpoint.pt", weights_only=False)
    synthetic.patch.setattr(FakeEnv, "step", lambda *a, **k: pytest.fail("rewound checkpoint stepped"))
    ck["trace"].pop()
    ck["transitions"].pop()
    lr.checkpoint_save(lr.torch, args.output / "checkpoint.pt", ck)
    with pytest.raises(ValueError):
        collect.run(request(tmp_path, resume=True))
    assert not (args.output / "completion.json").exists()
    with pytest.raises(ValueError, match="Failed/unknown"):
        collect.check_output(args.output, resume=True)


def test_stop_after_interval_keeps_checkpoint(tmp_path, synthetic):
    args = request(tmp_path)
    step = FakeEnv.step
    def stopped_step(*a, **k):
        result = step(*a, **k)
        (args.output / "STOP").touch()
        return result
    synthetic.patch.setattr(FakeEnv, "step", stopped_step)
    assert collect.run(args) == "stopped"
    assert len(lr.torch.load(args.output / "checkpoint.pt", weights_only=False)["trace"]) == 1
    assert not (args.output / "completion.json").exists()
    with pytest.raises(lr.Stopped):
        lr.check_stop(args.output)


def test_terminal_resume_no_step_and_serialization_in_session_scope(tmp_path, synthetic):
    args = request(tmp_path, limit=74)
    collect.run(args)
    real_save = collect.save
    def interrupt(path, value):
        if path.name == "completion.json":
            raise RuntimeError("synthetic publication interruption")
        real_save(path, value)
    synthetic.patch.setattr(collect, "save", interrupt)
    with pytest.raises(RuntimeError, match="publication"):
        collect.run(request(tmp_path, resume=True))
    assert lr.read(tmp_path / ".ownership/reservations.json")[-1]["cohort_phase"] == "finished"
    synthetic.patch.setattr(collect, "save", real_save)
    synthetic.patch.setattr(FakeEnv, "reset", lambda *a: pytest.fail("terminal resume reset"))
    synthetic.patch.setattr(FakeEnv, "step", lambda *a, **k: pytest.fail("terminal resume stepped"))
    clock = [0.]
    synthetic.patch.setattr(collect.time, "perf_counter", lambda: clock[0])
    export = collect.export_trace
    def measured_export(*a):
        export(*a)
        clock[0] += 7.
    synthetic.patch.setattr(collect, "export_trace", measured_export)
    assert collect.run(request(tmp_path, resume=True)) == "completed"
    timing = validate.load_completed(args.output, IDENTITY)[2]
    assert timing["known_elapsed_wall_seconds"] >= 7.


def test_checkpoint_physical_and_policy_memory_stay_bound(tmp_path, synthetic):
    args = request(tmp_path, limit=10)
    collect.run(args)
    ck = lr.torch.load(args.output / "checkpoint.pt", weights_only=False)
    for key in ("policy", "anchor", "configuration"):
        broken = copy.deepcopy(ck)
        if key == "policy":
            broken["policy"]["base"][1] -= 1.
        elif key == "anchor":
            broken["environment"]["prepared"]["action_anchor"][1] -= 1.
        else:
            broken["environment"]["sim"].cfg["network"]["total_ramp_capacity"] -= 1.
        with pytest.raises((ValueError, AssertionError)):
            collect.validate_checkpoint(broken, broken["settings"])


def test_completed_output_reconciles_retained_run_id(tmp_path, synthetic):
    args = request(tmp_path)
    collect.run(args)
    records = lr.read(tmp_path / ".ownership/reservations.json")
    records[-1]["run_id"] = "relabelled-fixture"
    lr.save(tmp_path / ".ownership/reservations.json", records)
    with pytest.raises(ValueError, match="retained cohort"):
        validate.load_completed(args.output, IDENTITY)


def test_serialization_failure_preserves_terminal_checkpoint_without_retry(tmp_path, synthetic):
    args = request(tmp_path)
    step = FakeEnv.step
    def unknown_diagnostic(*a, **k):
        nxt, reward, terminal, row = step(*a, **k)
        if terminal:
            row["not_allowlisted"] = float("inf")
        return nxt, reward, terminal, row
    synthetic.patch.setattr(FakeEnv, "step", unknown_diagnostic)
    with pytest.raises(ValueError, match="Forbidden"):
        collect.run(args)
    assert not (args.output / "completion.json").exists()
    assert len(lr.torch.load(args.output / "checkpoint.pt", weights_only=False)["trace"]) == 75
    with pytest.raises(ValueError, match="Failed/unknown"):
        collect.check_output(args.output, resume=True)
