"""Focused new-contract tests, with physical execution guarded by conftest."""
import copy
import json
from types import SimpleNamespace as NS
import pytest
import support as s
import policy
import checks
import worker
import preflight
import readout
import baselines
from conftest import ACTOR, AUTH, SCHEMA, CONFIG, OPTIONS


def test_actor_inference_boundary_and_no_rng():
    state = s.torch.random.get_rng_state().clone()
    actor = policy.FrozenActor(ACTOR.state_dict())
    assert s.torch.equal(state, s.torch.random.get_rng_state())
    assert not actor.training and all(not p.requires_grad for p in actor.parameters())
    assert not hasattr(actor, "critics") and not hasattr(actor, "optimizer")
    assert actor.act(s.np.zeros(2367, dtype=s.np.float32)).dtype == s.np.float32
    with pytest.raises(ValueError):
        actor.act(s.np.zeros(2367, dtype=s.np.float64))
    s.import_boundary()


@pytest.mark.parametrize("change", ("format", "phase", "counts", "ancestor_counts", "continuation", "actor", "settings"))
def test_model_mismatch(change):
    payload = s.torch.load(s.MODEL / "model_final.pt", map_location="cpu", weights_only=False)
    settings, done = copy.deepcopy(payload["settings"]), copy.deepcopy(s.MODEL_DONE)
    if change == "format":
        payload["format"] = "generic"
    elif change == "phase":
        payload["learner"]["phase"] = "mc_fit"
    elif change == "counts":
        payload["learner"]["counts"]["mc_critic"] = 249
    elif change == "ancestor_counts":
        payload["learner"]["ancestor_counts"]["actor"] = 11
    elif change == "continuation":
        payload["learner"]["critic_continuation"] = "carry"
    elif change == "actor":
        payload["learner"]["models"]["actor"]["net.0.bias"][0] += 1
    else:
        settings["spec"]["gamma"] = .99
    with pytest.raises(ValueError):
        policy.validate_model(payload, settings, done)


def test_canonical_admitted_by_new_rejected_by_old(synthetic):
    f = synthetic
    assert worker.run(f.request(limit=1)) == "checkpointed"
    settings = s.read(s.ROOT / s.SCENARIOS[0] / "settings.json")
    checks.validate_settings(settings, f.auth, f.source, s.SCENARIOS[0])
    with pytest.raises(ValueError, match="Wave settings"):
        checks.legacy.validate_settings(settings, f.auth, f.source, s.SCENARIOS[0])
    env = worker.BudgetEnv(None, s.SCENARIOS[0], None, "physical")
    cfg = NS()
    rt = dict(cfg=cfg, options=OPTIONS, rc=NS(to_plain_dict=lambda v: CONFIG if v is cfg else OPTIONS))
    checks.verify_environment(env, rt, f.auth["contract"])
    with pytest.raises(ValueError, match="Training profile equals canonical"):
        checks.helpers().verify_environment(env, rt, f.auth["contract"])


@pytest.mark.parametrize("field,value", (("evaluation", False), ("eval_only", False), ("training_seed", 7301),
    ("policy_q", [0., 0.]), ("updates_per_interval", 1), ("profile_sha256", "other"), ("model_sha256", "other")))
def test_settings_reject_wrong_contract(synthetic, field, value):
    f = synthetic
    worker.run(f.request(limit=1))
    settings = s.read(s.ROOT / s.SCENARIOS[0] / "settings.json")
    settings[field] = value
    with pytest.raises(ValueError):
        checks.validate_settings(settings, f.auth, f.source, s.SCENARIOS[0])


@pytest.mark.parametrize("field,value", (("evaluation", False), ("eval_only", False), ("policy_q", [1, 2]),
    ("learning", ["update"]), ("lower_candidate_count", 2), ("guard_mode", "h3"), ("total_ttt", 999.)))
def test_trace_rejects_metadata_accounting_and_q(synthetic, field, value):
    f = synthetic
    worker.run(f.request(limit=1))
    _, path = s.checkpoint_record(s.ROOT / s.SCENARIOS[0])
    ck = s.torch.load(path, map_location="cpu", weights_only=False)
    ck["trace"][0][field] = value
    with pytest.raises((ValueError, AssertionError)):
        checks.validate_checkpoint(ck, ck["settings"], f.actor)


def test_parent_gate_precedes_physical_start(synthetic):
    f = synthetic
    receipt = s.read(f.receipt)
    receipt["source_sha256"] = "wrong"
    s.save(f.receipt, receipt)
    with pytest.raises(ValueError, match="parent review"):
        worker.run(f.request())
    assert f.env.steps == f.env.resets == 0 and not s.ROOT.exists()


@pytest.mark.parametrize("scope", ("REPO", "GOAL", "ROOT", "scenario"))
def test_stop_scopes(synthetic, scope):
    folder = s.ROOT / s.SCENARIOS[0] if scope == "scenario" else getattr(s, scope)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "STOP").touch()
    with pytest.raises(s.Stopped):
        worker.run(synthetic.request())
    assert synthetic.env.resets == synthetic.env.steps == 0 and (folder / "STOP").exists()


def test_kernel_lock(tmp_path):
    with s.exclusive_run(tmp_path):
        with pytest.raises(OSError):
            with s.exclusive_run(tmp_path):
                pytest.fail("Duplicate writer acquired lock")
    with s.exclusive_run(tmp_path):
        pass


def test_resume_and_completed_duplicate(synthetic):
    f = synthetic
    worker.run(f.request(limit=2))
    folder = s.ROOT / s.SCENARIOS[0]
    _, path = s.checkpoint_record(folder)
    before_hash = s.file_hash(path)
    assert worker.run(f.request(resume=True)) == "completed"
    assert (f.env.resets, f.env.steps, f.env.restores) == (1, 75, 1)
    assert before_hash == s.file_hash(path)
    summary = readout.load_completed(folder, s.SCENARIOS[0], f.actor, f.auth, f.source)
    restore = summary["worker_sessions"]["restoration"]
    assert restore["returned_branch_reference_reconstructions"] == 1
    assert restore["elapsed_wall_seconds"] <= summary["worker_sessions"]["elapsed_wall_seconds"]
    assert summary["policy_q"] is None and summary["eval_only"] is True
    with pytest.raises(FileExistsError):
        worker.run(f.request(resume=True))


def test_orphan_operation_refused(synthetic):
    f = synthetic
    worker.run(f.request(limit=1))
    folder = s.ROOT / s.SCENARIOS[0]
    worker.operation(folder, 2, "orphan", None)
    before = s.file_hash(folder / "operations/step_02.json")
    with pytest.raises(ValueError, match="Uncheckpointed"):
        worker.run(f.request(resume=True))
    assert f.env.steps == 1 and before == s.file_hash(folder / "operations/step_02.json")


def test_finalization_stop_resume_no_more_environment(synthetic):
    f = synthetic
    original = worker.finalize
    def stop(output, settings, actor):
        (output / "STOP").touch()
        original(output, settings, actor)
    f.patch.setattr(worker, "finalize", stop)
    with pytest.raises(s.Stopped):
        worker.run(f.request())
    folder = s.ROOT / s.SCENARIOS[0]
    stable = {n: s.file_hash(folder / n) for n in ("finalization.json", "experience.pt", "trace.json", "timing.json", "summary.json")}
    (folder / "STOP").unlink()
    f.patch.setattr(worker, "finalize", original)
    assert worker.run(f.request(resume=True)) == "completed"
    assert (f.env.resets, f.env.steps, f.env.restores) == (1, 75, 0)
    assert stable == {n: s.file_hash(folder / n) for n in stable}


def test_stop_at_step75_before_exports_does_not_repeat_physics(synthetic):
    f = synthetic
    original = s.persist_checkpoint
    def stop(output, payload, validate=None):
        record = original(output, payload, validate)
        if len(payload["trace"]) == 75:
            (output / "STOP").touch()
        return record
    f.patch.setattr(s, "persist_checkpoint", stop)
    assert worker.run(f.request()) == "stopped"
    folder = s.ROOT / s.SCENARIOS[0]
    (folder / "STOP").unlink()
    assert worker.run(f.request(resume=True)) == "completed"
    assert (f.env.resets, f.env.steps, f.env.restores) == (1, 75, 1)
    assert s.read(folder / "timing.json")["restoration"]["returned_branch_reference_reconstructions"] == 0


def test_retained_export_mismatch_is_not_overwritten(synthetic):
    f = synthetic
    worker.run(f.request(limit=1))
    folder = s.ROOT / s.SCENARIOS[0]
    s.save(folder / "trace.json", ["wrong retained evidence"])
    before = s.file_hash(folder / "trace.json")
    with pytest.raises(FileExistsError):
        worker.run(f.request(resume=True))
    assert before == s.file_hash(folder / "trace.json")
    assert f.env.steps == 1


def test_partial_full_and_all_five(synthetic):
    f = synthetic
    empty = readout.evaluate(f.preflight, partial=True)
    assert len(empty["scenarios"]) == 5 and not empty["all_five_improved"]
    assert all(r["status"] == "absent" for r in empty["scenarios"])
    for scenario in s.SCENARIOS:
        worker.run(f.request(scenario))
    result = readout.evaluate(f.preflight)
    assert result["status"] == "eligible_for_separate_reproduction"
    assert result["integrity_pass"] and not result["goal_achieved"]
    assert not result["automatic_reproduction_dispatch"]
    rows = copy.deepcopy(result["scenarios"])
    rows[-1]["improved"] = False
    assert readout.acceptance(rows)["status"] == "failed_acceptance"
    assert not readout.acceptance(rows[:-1])["integrity_pass"]
    assert not readout.acceptance(result["scenarios"], ["extra"])["integrity_pass"]
    rows[-1]["candidate"]["run_id"] = rows[0]["candidate"]["run_id"]
    assert not readout.acceptance(rows)["integrity_pass"]


@pytest.mark.parametrize("direction,passes", ((float("inf"), False), (-float("inf"), True)))
def test_strict_threshold(direction, passes):
    scenario = s.SCENARIOS[0]
    base = s.BASE_TTT[scenario]
    value = float(s.np.nextafter(base-max(1e-6, 1e-8*base), direction))
    delta = base-value
    baseline = dict(scenario=scenario, profile_sha256="test", warmup_ttt=5., ttt=base,
                    interval_ttt=[(base-5.)/75]*75)
    candidate = dict(baseline, model_sha256=s.MODEL_SHA, control_steps=75, simulation_seconds=14400,
                     ttt=value, interval_ttt=[(base-5.-delta)/75]*75)
    result = readout.matched(candidate, baseline)
    assert result["improved"] is passes
    assert set(result["windows"]) == {"1..25", "26..50", "51..75"}


def test_physical_import_collision(monkeypatch):
    monkeypatch.setitem(s.sys.modules, "learner", NS())
    with pytest.raises(ValueError, match="Learner module"):
        s.import_boundary()


def test_preflight_source_and_profile_mismatch(monkeypatch):
    malformed = dict(format="wrong")
    with pytest.raises(ValueError):
        preflight.live(malformed)
    cfg = NS()
    env = NS(training_seed=None, profile_hash="wrong", protocol=dict(scenario=s.SCENARIOS[0]),
             contract=lambda: AUTH["contract"]["environment_contract"]["coordinator"])
    rt = dict(cfg=cfg, options=OPTIONS, rc=NS(to_plain_dict=lambda v: CONFIG if v is cfg else OPTIONS))
    auth = copy.deepcopy(AUTH)
    auth["contract"]["environment_contract"].update(config_sha256=s.digest(CONFIG), options_sha256=s.digest(OPTIONS))
    with pytest.raises(ValueError, match="Canonical environment"):
        checks.verify_environment(env, rt, auth["contract"])


@pytest.mark.parametrize("field", ("source_pins", "runtime_versions", "environment_contract", "profile", "ttt", "seed", "schema"))
def test_center_mismatch_rejected_before_physical_helpers(field, monkeypatch):
    scenario, contract = s.SCENARIOS[0], copy.deepcopy(AUTH["contract"])
    settings = dict(scenario=scenario, mode="center", seeds=[None], model_sha256=None,
        source_pins=contract["source_pins"], runtime_versions=contract["runtime_versions"],
        environment_contract=contract["environment_contract"])
    settings = copy.deepcopy(settings)
    episode = dict(ttt=s.BASE_TTT[scenario], profile_sha256=AUTH["profiles"][scenario], warmup_ttt=5.)
    schema = copy.deepcopy(SCHEMA)
    if field in settings:
        settings[field] = {}
    elif field == "profile":
        episode["profile_sha256"] = "wrong"
    elif field == "ttt":
        episode["ttt"] += 1.
    elif field == "seed":
        settings["seeds"] = [7301]
    else:
        schema = {}
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible baseline reached physical validation")
    monkeypatch.setattr(checks, "physical_rows", forbidden)
    with pytest.raises(ValueError):
        baselines.validate_center(dict(episodes=[episode]), settings, [[]], schema, scenario, contract, {})


def test_partial_readout_hashes_without_loading_checkpoint(synthetic):
    f = synthetic
    worker.run(f.request(limit=2))
    original = s.torch.load
    def guarded(path, *args, **kwargs):
        if "checkpoints" in s.Path(path).parts:
            pytest.fail("Readout unpickled physical state")
        return original(path, *args, **kwargs)
    f.patch.setattr(s.torch, "load", guarded)
    result = readout.evaluate(f.preflight, partial=True)
    assert result["scenarios"][0]["checkpoint_progress"]["control_steps"] == 2
    assert len(result["scenarios"]) == 5 and not result["all_five_improved"]


def test_atomic_pointer_does_not_publish_failed_checkpoint(tmp_path):
    payload = dict(trace=[], settings={}, session_ids=[])
    first = s.persist_checkpoint(tmp_path, payload)
    def rejected(value):
        raise ValueError("synthetic validator failure")
    with pytest.raises(ValueError):
        s.persist_checkpoint(tmp_path, payload, rejected)
    assert s.read(tmp_path / "latest.json") == first
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2


def test_source_receipt_and_input_hash_changes(synthetic):
    f = synthetic
    receipt = s.read(f.receipt)
    receipt["spec_sha256"] = "different"
    s.save(f.receipt, receipt)
    with pytest.raises(ValueError):
        preflight.admission(f.preflight, f.receipt)
    before = s.file_hash(f.preflight)
    s.save(f.preflight, dict(f.admitted, changed=True))
    with pytest.raises(ValueError):
        s.check_hash(f.preflight, before)


