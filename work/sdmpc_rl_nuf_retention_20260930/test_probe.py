"""Read-only real references and pure prefix/screen regression; no plant execution."""
import copy
import sys
from dataclasses import is_dataclass, fields
import pytest
import local_runtime as lr
import references
import probe
import frozen_inventory


def test_original_validator_authenticates_completed_references(record_property):
    before = {name: references.sha(references.ROOT / name) for name in references.ROOT_PINS}
    authenticated = references.authenticate()
    assert authenticated["root_sha256"] == references.ROOT_PINS
    assert len(authenticated["entries"]) == 10
    assert sum(bool(r["export_recovery"]) for r in authenticated["entries"].values()) == 8
    assert before == {name: references.sha(references.ROOT / name) for name in references.ROOT_PINS}
    record_property("root_completion_sha256", before["completion.json"])
    record_property("root_comparison_sha256", before["comparison.json"])
    record_property("reference_runs", 10)
    record_property("verified_repaired_receipts", 8)


@pytest.fixture(scope="module")
def saved_pilot():
    folder = references.ROOT / "local" / probe.PILOT
    done = references.read(folder / "completion.json")
    for name in ("experience.pt", "trace.json"):
        references.check_hash(folder / name, done["outputs_sha256"][name])
    return (lr.read(folder / "trace.json"),
            lr.torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)["transitions"])


def test_prefix_compares_physical_control_native_observation_reward_and_all_rng(saved_pilot):
    old, experience = saved_pilot
    candidate, new_experience = copy.deepcopy(old), copy.deepcopy(experience)
    action = new_experience[10][1]
    action[1] += .05
    candidate[0]["lower_wall_seconds"] += 10.
    candidate[9]["exploration_audit"]["base_after"][1] = 6000.
    result = probe.causal_integrity(candidate, new_experience, old, experience, pilot=True)
    assert result["first_action_difference"] == 10 and result["all_75_noise_vectors_exact"]


@pytest.mark.parametrize("fault", ["control", "observation", "reward", "ttt", "noise74", "late_action", "early_action"])
def test_prefix_rng_integrity_failures_cannot_get_performance_label(saved_pilot, fault):
    old, experience = saved_pilot
    candidate, new_experience = copy.deepcopy(old), copy.deepcopy(experience)
    new_experience[10][1][1] += .05
    if fault == "control":
        candidate[9]["control"]["ramp_metering"]["R_D_W"] += 1e-6
    elif fault == "observation":
        new_experience[9][3][0] += .1
    elif fault == "reward":
        row = list(new_experience[9])
        row[2] += 1e-6
        new_experience[9] = tuple(row)
    elif fault == "ttt":
        candidate[9]["total_ttt"] += 1e-6
    elif fault == "noise74":
        candidate[74]["exploration_audit"]["noise"][1] += 1e-10
    elif fault == "late_action":
        new_experience[10][1][:] = experience[10][1]
        new_experience[15][1][1] += .05
    else:
        new_experience[9][1][1] += .05
    with pytest.raises((ValueError, AssertionError)):
        probe.causal_integrity(candidate, new_experience, old, experience, pilot=True)


@pytest.mark.parametrize("zero,inventory,ttt,passed", [
    (5, 517.470147382039, 11945., True),
    (6, 517., 11945., False), (5, 517.470147392039, 11945., False),
    (5, 500., 11945.939292655728, False), (5, 500., 11946., False)])
def test_pilot_thresholds_are_predeclared_and_strict(monkeypatch, tmp_path, zero, inventory, ttt, passed):
    settings = dict(run_id="new-fixture")
    summary = dict(zero_nuf_requests=zero, terminal_inventory=inventory, ttt=ttt, interval_ttt=[1.]*75)
    carry = dict(zero_nuf_requests=0, terminal_inventory=413.97611790563116,
                 ttt=4602.4114708736415, interval_ttt=[1.]*75)
    old = dict(ttt=11945.939292655728, interval_ttt=[1.]*75)
    bound = dict(local=dict(path=str(tmp_path / "old")), carry=dict(path=str(tmp_path / "carry")))
    monkeypatch.setattr(probe, "load_completed", lambda *a: ({}, settings, summary))
    monkeypatch.setattr(probe, "validate_binding", lambda *a: bound)
    monkeypatch.setattr(probe, "read", lambda path: carry if path.parent.name == "carry" else
                        old if path.name == "summary.json" else [])
    monkeypatch.setattr(probe.torch, "load", lambda *a, **k: {"transitions": []})
    monkeypatch.setattr(probe, "causal_integrity", lambda *a: {"fixture": True})
    monkeypatch.setattr(probe, "file_hash", lambda *a: "fixture-completion")
    result = probe.scenario_report(tmp_path, probe.PILOT, {}, {})
    assert result["advance_allowed"] is passed
    assert result["integrity_pass"] and not result["learner_admitted"]
    assert result["carry_ttt"] == carry["ttt"] and not result["shared_policy_improvement_claim"]


def test_integrity_failure_readout_is_archivable_and_blocks_expansion(monkeypatch, tmp_path):
    def invalid(*args):
        raise ValueError("physical prefix differs")
    monkeypatch.setattr(probe, "scenario_report", invalid)
    report = probe.pilot_report(tmp_path, {"candidate": "fixture"}, {"root_sha256": references.ROOT_PINS})
    lr.save(tmp_path / "invalid-pilot.json", report)
    assert not report["integrity_pass"] and not report["advance_allowed"]
    assert report["conclusion"] == "invalid_attribution_requires_diagnosis"


def test_real_frozen_physical_config_extra_serializes_without_boot():
    settings = references.read(references.ROOT / "local" / probe.PILOT / "settings.json")
    frozen_inventory.accounting()
    config_type = sys.modules["local_frozen_state"].ExperimentConfig
    def reconstruct(cls, values):
        # Rehydrate the saved real dataclasses without their topology-building initializer.
        target = cls.__new__(cls)
        factories = {f.name: f.default_factory for f in fields(cls)}
        for key, value in values.items():
            factory = factories.get(key)
            setattr(target, key, reconstruct(factory, value) if is_dataclass(factory) else copy.deepcopy(value))
        return target
    cfg = reconstruct(config_type, settings["physical_config"])
    assert cfg.network.terminal_zero_gradient is True
    assert frozen_inventory.complete_config(cfg) == settings["physical_config"]
    assert lr.digest(frozen_inventory.complete_config(cfg)) == settings["contract"]["environment_contract"]["config_sha256"]
    cfg.network.terminal_zero_gradient = False
    assert lr.digest(frozen_inventory.complete_config(cfg)) != settings["contract"]["environment_contract"]["config_sha256"]


def test_new_identity_uses_frozen_physical_runtime_and_separate_collection_format():
    identity = lr.identity()
    old = references.read(references.ROOT / "plan.json")["identity"]
    assert all(identity[k] == old[k] for k in ("physical", "runtime", "gate_sha256", "contract_sha256"))
    assert identity["local_sources"] != old["local_sources"]
    assert lr.FORMAT != "sdmpc-local-training-v1"
    assert lr.COHORT_ROOT.name == "nuf_retention_v1"
