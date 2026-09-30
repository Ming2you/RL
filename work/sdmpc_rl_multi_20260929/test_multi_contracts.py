"""Synthetic multi-scenario artifacts and admission regressions; no runtime boot."""
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import build_preflight
import compare_runs
import run_budget
import train_round
from budget_runtime import digest, read, save
from td3 import SCENARIOS, TD3


PINS = {"snapshot": "synthetic-frozen-source"}
VERSIONS = {name: "synthetic" for name in ("python", "numpy", "scipy", "torch", "PyYAML")}
SCHEMA = dict(names=["state/time", "state/memory"],
              scale="fixed_physical_constants_no_eval_fitting", delay_bins=3,
              normalization={"scale": 80.},
              Markov_sufficiency="not_proven_recovery_PFO_memory_is_checkpointed_but_not_fully_observed")
MANIFEST = [dict(scenario=s, protocol_sha256=digest([s, "protocol"]),
                 profile_sha256=digest({"synthetic_forecast": s}),
                 initial_state_sha256=digest([s, "initial"])) for s in SCENARIOS]
COORDINATOR = dict(version=3, guard_mode="physical", action_anchor="previous_executed_budget")
ENVIRONMENT = dict(config_sha256=digest({}), options_sha256=digest({}),
                   scenarios=MANIFEST, coordinator=COORDINATOR)
MODEL_HASH = "a" * 64


def settings_for(name, scenario=SCENARIOS[0], mode="collect", round_index=0,
                 model_hash=None, cpu_mask=1):
    index = SCENARIOS.index(scenario)
    seed = (6301 if round_index == 0 else 6401) + index if mode == "collect" else None
    profile = (digest([scenario, seed]) if seed is not None else MANIFEST[index]["profile_sha256"])
    return dict(format=run_budget.RUN_FORMAT, run_id=name, mode=mode, scenario=scenario,
                round=round_index, seeds=[seed], policy_seed=6300, profile_sha256=[profile],
                environment_contract=copy.deepcopy(ENVIRONMENT), runtime_versions=VERSIONS,
                model_sha256=model_hash, reward_divisor=100., gamma=1., exploration_std=.3,
                initial_random_steps=32 if mode == "collect" and round_index == 0 else 0,
                batch_size=40, updates_per_interval=0, guard_mode="physical", total_seconds=14400,
                warmup_steps=5, controlled_steps=75, cpu_mask=cpu_mask, source_pins=PINS)


def contract_for(settings, schema=SCHEMA):
    return dict(source_pins=settings["source_pins"], runtime_versions=settings["runtime_versions"],
                environment_contract=settings["environment_contract"], observation_schema=schema,
                reward_divisor=100., gamma=1.)


def trajectory(settings):
    rows, transitions = [], []
    seed = settings["seeds"][0]
    rng = np.random.default_rng(seed if seed is not None else 6300)
    marker = SCENARIOS.index(settings["scenario"]) + settings["round"] * 10
    for i in range(75):
        source = "pfo_initial" if i == 0 else "pfo_recovery" if i == 30 else "previous"
        pfo = int(source != "previous")
        action = (rng.uniform(-1., 1., 2).astype(np.float32) if seed is not None
                  else np.zeros(2, dtype=np.float32))
        row = dict(episode=0, scenario=settings["scenario"], profile_sha256=settings["profile_sha256"][0],
                   step=i+5, control_step=i, time_sec=(i+6)*180, terminated=i == 74, truncated=False,
                   interval_ttt=1., total_ttt=i+6., reward=-.01,
                   freeway_ttt=(i+6.)*.6, urban_ttt=(i+6.)*.4,
                   selected_TTT=4., reference_TTT=3., h3_guard_enabled=False,
                   reference_source=source, pfo_calls=pfo, reference_recovery=source == "pfo_recovery",
                   selection_source="reference_fallback" if i == 10 else "lower_solution",
                   lower_candidate_count=1, converged=i != 10, learning=[], learning_wall_seconds=0.,
                   policy_q=None if settings["mode"] == "center" else [-.8, -.9],
                   action_requested=action.tolist(), B_requested=[[-50., 500.]], B_executed=[-20., 1500.],
                   execution_check=dict(physical_control_valid=True, budget_feasible=True),
                   inventory={"vehicles": 1.}, queue_state={"ramp_queue": {"r": 9.}, "boundary_queue": {}},
                   queue_near_capacity_estimate=dict(method="interval_endpoint_sampled",
                       exact_substep_exposure=False, threshold_fraction=.9, interval_seconds=180.,
                       duration_units="s", capacity_units="veh", boundary_queue={},
                       ramp_queue={"r": dict(queue_veh=9., capacity_veh=10., near_capacity_seconds=180.)}))
        for prefix, duration in (("forecast", 0.), ("observation", 0.), ("pfo", float(pfo)),
                                 ("reference", 1.), ("lower", 1.), ("guard", 1.), ("actor", 1.)):
            row[prefix+"_wall_seconds"] = row[prefix+"_cpu_seconds"] = duration
        row["decision_wall_seconds"] = row["decision_cpu_seconds"] = 4. + pfo
        rows.append(row)
        obs = np.array([(i+5)/80., marker], dtype=np.float32)
        next_obs = np.array([0., 0.] if i == 74 else [(i+6)/80., marker], dtype=np.float32)
        transitions.append((obs, action, -.01, next_obs, i == 74))
    summary = dict(episode=0, scenario=settings["scenario"], training_seed=seed,
                   profile_sha256=settings["profile_sha256"][0], ttt=80., warmup_ttt=5.,
                   freeway_ttt=48., urban_ttt=32., full_run=True, control_steps=75,
                   simulation_seconds=14400, terminal_inventory=rows[-1]["inventory"],
                   evaluation=seed is None, exploration=seed is not None,
                   decision_wall_seconds=302., decision_cpu_seconds=302.,
                   decision_p50=4., decision_p95=4., decision_max=5., fallback_count=1,
                   pfo_calls=2, pfo_recovery_count=1, lower_candidate_solves=75, converged_count=74,
                   learner_updates=0, controller_acceptance=False)
    return summary, rows, transitions


def completed_run(folder, scenario=SCENARIOS[0], mode="collect", round_index=0,
                  model_hash=None, cpu_mask=1):
    folder = Path(folder)
    settings = settings_for(str(folder), scenario, mode, round_index, model_hash, cpu_mask)
    summary, rows, transitions = trajectory(settings)
    save(folder / "settings.json", settings)
    save(folder / "runtime_versions.json", VERSIONS)
    save(folder / "observation_schema.json", SCHEMA)
    save(folder / "episode_00_summary.json", summary)
    save(folder / "episode_00_trace.json", rows)
    experience_hash = None
    if mode == "collect":
        torch.save(dict(format=run_budget.EXPERIENCE_FORMAT, settings=settings,
                        contract=contract_for(settings), transitions=transitions), folder / "experience.pt")
        experience_hash = run_budget.file_hash(folder / "experience.pt")
    save(folder / "completion.json", dict(format=run_budget.RUN_FORMAT, run_id=settings["run_id"],
        runtime_versions=VERSIONS, status="completed", mode=mode, episodes=[summary],
        source_pins=PINS, experience_sha256=experience_hash, elapsed_wall_seconds=400.,
        goal_claim=False, controller_acceptance=False))
    return folder


def collection_group(root, round_index=0, model_hash=None):
    return [completed_run(root / s, s, round_index=round_index, model_hash=model_hash) for s in SCENARIOS]


def change_json(path, mutate):
    data = read(path)
    mutate(data)
    save(path, data)


def change_experience(folder, mutate):
    payload = torch.load(folder / "experience.pt", weights_only=False)
    mutate(payload)
    torch.save(payload, folder / "experience.pt")
    change_json(folder / "completion.json", lambda r: r.update(
        experience_sha256=run_budget.file_hash(folder / "experience.pt")))


def synthetic_training(output, job, plan):
    """Create a valid trainer artifact in-process from synthetic experiences only."""
    folder = output / job["key"]
    index = job["round"]
    previous_hash = None if job["model"] is None else run_budget.file_hash(job["model"])
    contract, runs = train_round.load_collections(
        [output / f"collect_round{index}" / s for s in SCENARIOS], index, PINS, VERSIONS, previous_hash)
    learner, provenance = TD3(len(SCHEMA["names"]), 6300), []
    if job["model"]:
        previous = torch.load(job["model"], weights_only=False)
        learner.load_state_dict(previous["learner"])
        provenance = previous["training_profiles"]
    for s in SCENARIOS:
        for transition in runs[s]["transitions"]:
            learner.add(*transition, scenario=s)
        provenance.append(runs[s]["provenance"])
    metrics = [learner.update(40) for _ in range(375)]
    settings = dict(format=train_round.TRAIN_FORMAT, round=index, policy_seed=6300,
        run_id=str(folder), source_pins=PINS, runtime_versions=VERSIONS, contract=contract,
        model_sha256=previous_hash, collections={s: runs[s]["identity"] for s in SCENARIOS},
        updates=375, batch_size=40, updates_per_new_transition=1)
    save(folder / "settings.json", settings)
    torch.save(dict(format=run_budget.POLICY_FORMAT, learner=learner.state_dict(), contract=contract,
        training_run_id=settings["run_id"], training_profiles=provenance, training_round=index),
        folder / "model_final.pt")
    save(folder / "metrics.json", metrics)
    save(folder / "completion.json", dict(format=train_round.TRAIN_FORMAT, status="completed",
        settings=settings, model_sha256=run_budget.file_hash(folder / "model_final.pt"),
        replay_counts=learner.replay_counts(), learner_updates=learner.updates,
        sampled_per_scenario=dict.fromkeys(SCENARIOS, 3000), elapsed_wall_seconds=1.))


def test_only_new_experiment_modules_are_loaded():
    here = Path(__file__).resolve().parent
    for module in (run_budget, train_round, compare_runs, build_preflight, sys.modules["td3"],
                   sys.modules["budget_runtime"]):
        assert Path(module.__file__).resolve().parent == here
    assert not any(name == "src" or name.startswith("src.") for name in sys.modules)


@pytest.mark.parametrize("round_index,model_hash", [(0, None), (1, MODEL_HASH)])
def test_five_collectors_admit_exact_transitions_and_true_terminals(tmp_path, round_index, model_hash):
    folders = collection_group(tmp_path, round_index, model_hash)
    contract, runs = train_round.load_collections(folders, round_index, PINS, VERSIONS, model_hash)
    assert set(runs) == set(SCENARIOS)
    assert contract == contract_for(read(folders[0] / "settings.json"))
    assert [runs[s]["provenance"]["seed"] for s in SCENARIOS] == list(
        range(6301 if round_index == 0 else 6401, 6306 if round_index == 0 else 6406))
    for folder, s in zip(folders, SCENARIOS):
        transitions = runs[s]["transitions"]
        assert len(transitions) == 75
        assert [i for i, t in enumerate(transitions) if t[4]] == [74]
        for previous, following in zip(transitions, transitions[1:]):
            np.testing.assert_array_equal(previous[3], following[0])
        assert runs[s]["identity"]["experience_sha256"] == run_budget.file_hash(folder / "experience.pt")


@pytest.mark.parametrize("fault", ["missing", "duplicate_folder", "duplicate_scenario", "duplicate_run",
    "seed", "round", "policy_seed", "model", "source", "runtime", "schema", "partial",
    "missing_trace", "experience_hash", "experience_settings", "experience_contract", "continuity",
    "requested_action", "reward", "terminal", "nonfinite_observation", "length", "evaluation"])
def test_collection_admission_rejects_corrupt_or_noncollection_inputs(tmp_path, fault):
    folders = collection_group(tmp_path)
    folder = folders[-1]
    if fault == "missing":
        folders.pop()
    elif fault == "duplicate_folder":
        folders[-1] = folders[0]
    elif fault == "duplicate_scenario":
        completed_run(folder, SCENARIOS[0])
    elif fault == "duplicate_run":
        identity = read(folders[0] / "settings.json")["run_id"]
        for filename in ("settings.json", "completion.json"):
            change_json(folder / filename, lambda r: r.update(run_id=identity))
    elif fault in ("seed", "round", "policy_seed", "model", "source", "runtime"):
        key, value = {"seed": ("seeds", [6405]), "round": ("round", 1),
            "policy_seed": ("policy_seed", 7), "model": ("model_sha256", MODEL_HASH),
            "source": ("source_pins", {}), "runtime": ("runtime_versions", {})}[fault]
        change_json(folder / "settings.json", lambda r: r.update({key: value}))
    elif fault == "schema":
        schema = dict(SCHEMA, names=["other/time", "state/memory"])
        save(folder / "observation_schema.json", schema)
        change_experience(folder, lambda r: r["contract"].update(observation_schema=schema))
    elif fault == "partial":
        change_json(folder / "completion.json", lambda r: r.update(status="paused"))
    elif fault == "missing_trace":
        (folder / "episode_00_trace.json").unlink()
    elif fault == "experience_hash":
        with (folder / "experience.pt").open("ab") as stream:
            stream.write(b"tamper")
    elif fault in ("experience_settings", "experience_contract"):
        key = "settings" if fault == "experience_settings" else "contract"
        change_experience(folder, lambda r: r.update({key: {}}))
    elif fault == "evaluation":
        completed_run(folder, SCENARIOS[-1], mode="center")
    else:
        def mutate(payload):
            transitions = payload["transitions"]
            if fault == "length":
                transitions.pop()
                return
            transition = list(transitions[30])
            if fault == "continuity":
                transition[0] = transition[0] + 1
            elif fault == "requested_action":
                transition[1] = np.zeros(2, dtype=np.float32)
            elif fault == "reward":
                transition[2] = -.02
            elif fault == "terminal":
                transition[4] = True
            elif fault == "nonfinite_observation":
                transition[3][0] = np.nan
            transitions[30] = tuple(transition)
        change_experience(folder, mutate)
    with pytest.raises((ValueError, AssertionError)):
        train_round.load_collections(folders, 0, PINS, VERSIONS, None)


@pytest.mark.parametrize("field,value", [("time_sec", 14399), ("terminated", True), ("truncated", True),
    ("interval_ttt", 2.), ("total_ttt", 999.), ("freeway_ttt", 999.), ("reward", 0.),
    ("pfo_calls", 1), ("reference_source", "pfo_initial"), ("reference_recovery", True),
    ("lower_candidate_count", 2), ("decision_wall_seconds", 0.), ("decision_cpu_seconds", 0.),
    ("h3_guard_enabled", True), ("learning", [{"updates": 1}]), ("learning_wall_seconds", .1),
    ("execution_check", {"physical_control_valid": True, "budget_feasible": False}),
    ("execution_check", {"physical_control_valid": False, "budget_feasible": True})])
def test_completed_trace_rejects_timeline_accounting_guard_and_local_learning(field, value):
    settings = settings_for("synthetic")
    summary, rows, _ = trajectory(settings)
    compare_runs.validate_episode(summary, rows, settings, 0)
    rows[4][field] = value
    with pytest.raises(ValueError):
        compare_runs.validate_episode(summary, rows, settings, 0)


@pytest.mark.parametrize("field", ["interval_ttt", "total_ttt", "reward", "selected_TTT",
                                     "reference_TTT", "pfo_wall_seconds", "actor_cpu_seconds"])
def test_completed_trace_rejects_nonfinite_values(field):
    settings = settings_for("synthetic")
    summary, rows, _ = trajectory(settings)
    rows[3][field] = float("nan")
    with pytest.raises(ValueError):
        compare_runs.validate_episode(summary, rows, settings, 0)


@pytest.mark.parametrize("mode", ["collect", "rl"])
@pytest.mark.parametrize("value", [None, [], [-1.], [-1., -2., -3.], [float("nan"), -1.],
                                  [-1., float("inf")], [True, -1.]])
def test_policy_q_requires_two_finite_critic_estimates(mode, value):
    settings = settings_for("synthetic", mode=mode, model_hash=MODEL_HASH if mode == "rl" else None)
    summary, rows, _ = trajectory(settings)
    rows[3]["policy_q"] = value
    with pytest.raises(ValueError):
        compare_runs.validate_episode(summary, rows, settings, 0)


def test_center_rejects_policy_q_audit():
    settings = settings_for("synthetic", mode="center")
    summary, rows, _ = trajectory(settings)
    rows[3]["policy_q"] = [-1., -1.]
    with pytest.raises(ValueError, match="policy values"):
        compare_runs.validate_episode(summary, rows, settings, 0)


@pytest.mark.parametrize("mode", ["center", "rl"])
def test_evaluation_uses_canonical_unseeded_profile_and_never_experience(tmp_path, mode):
    folder = completed_run(tmp_path / mode, mode=mode, model_hash=MODEL_HASH if mode == "rl" else None)
    result, settings, _, _ = compare_runs.load_completed_run(folder, mode, [None])
    assert not result["episodes"][0]["exploration"]
    assert result["episodes"][0]["evaluation"]
    assert not (folder / "experience.pt").exists()
    settings["profile_sha256"] = ["noncanonical"]
    save(folder / "settings.json", settings)
    with pytest.raises(ValueError, match="canonical"):
        compare_runs.load_completed_run(folder, mode, [None])


def smoke_rows():
    return [dict(scenario=s, passed=True, serialized_resume=True, source_pins=PINS,
        runtime_versions=VERSIONS,
        step=7, observation_schema=SCHEMA, observation_dim=len(SCHEMA["names"]),
        profile_sha256=MANIFEST[i]["profile_sha256"],
        scope="two_actual_intervals_with_serialized_resume_not_full_run",
        replay_counts={other: int(other == s) for other in SCENARIOS},
        audit=dict(execution_check=dict(physical_control_valid=True, budget_feasible=True),
                   h3_guard_enabled=False, pfo_calls=1, reference_source="pfo_initial"),
        carried_step_audit=dict(execution_check=dict(physical_control_valid=True, budget_feasible=True),
                   h3_guard_enabled=False, pfo_calls=0, reference_source="previous"))
        for i, s in enumerate(SCENARIOS)]


def mock_canonical_reads(monkeypatch):
    def fake_read(path):
        path = Path(path)
        if path.name == "forecast.json":
            return {"synthetic_forecast": path.parent.name}
        return read(path)
    monkeypatch.setattr(build_preflight, "read", fake_read)


@pytest.mark.parametrize("fault", [None, "duplicate", "missing", "source", "schema", "profile",
    "serialized", "scope", "physical", "budget", "h3", "carry", "replay"])
def test_gate_requires_five_matching_canonical_physical_smokes(monkeypatch, fault):
    mock_canonical_reads(monkeypatch)
    rows = copy.deepcopy(smoke_rows())
    if fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "missing":
        rows.pop()
    elif fault in ("source", "schema", "profile", "serialized", "scope", "replay"):
        key, value = {"source": ("source_pins", {}), "schema": ("observation_schema", {"names": ["other"]}),
            "profile": ("profile_sha256", "not-canonical"), "serialized": ("serialized_resume", False),
            "scope": ("scope", "declaration_only"), "replay": ("replay_counts", {})}[fault]
        rows[1][key] = value
    elif fault in ("physical", "budget"):
        rows[1]["carried_step_audit"]["execution_check"][
            "physical_control_valid" if fault == "physical" else "budget_feasible"] = False
    elif fault == "h3":
        rows[1]["audit"]["h3_guard_enabled"] = True
    elif fault == "carry":
        rows[1]["carried_step_audit"]["reference_source"] = "pfo_recovery"
    if fault:
        with pytest.raises(ValueError):
            build_preflight.validate_smokes(rows, PINS, SCENARIOS)
    else:
        assert build_preflight.validate_smokes(rows, PINS, SCENARIOS) == SCHEMA


def admission_fixture(root, monkeypatch):
    """Complete, explicitly synthetic evidence; exercise the real builder and consumer."""
    from types import SimpleNamespace
    evidence, sources = root / "evidence", root / "sources"
    evidence.mkdir(parents=True)
    sources.mkdir()
    for name in build_preflight.REQUIRED_TESTS:
        (sources / name).write_text("# Synthetic evidence fixture only\n", encoding="utf-8")
    monkeypatch.setattr(build_preflight, "HERE", sources)
    monkeypatch.setattr(build_preflight, "pins", lambda _: PINS)
    monkeypatch.setattr(build_preflight, "runtime_versions", lambda: VERSIONS)
    monkeypatch.setattr(sys, "path", list(sys.path))
    mock_canonical_reads(monkeypatch)
    xml, record_path = evidence / "tests.xml", evidence / "test_source_pins.json"
    suite, nodeids = ET.Element("testsuite"), []
    for source in sorted(sources.glob("test_*.py")):
        for index in range(10):
            nodeid = str(source.resolve()) + f"::test_synthetic_{index}"
            nodeids.append(nodeid)
            case = ET.SubElement(suite, "testcase", name=f"test_synthetic_{index}", classname=source.stem)
            ET.SubElement(ET.SubElement(case, "properties"), "property", name="nodeid", value=nodeid)
    ET.ElementTree(suite).write(xml, encoding="utf-8")
    record = dict(format=build_preflight.TEST_FORMAT, exit_code=0, source_pins=PINS,
        runtime_versions=VERSIONS, xml_sha256=run_budget.file_hash(xml),
        test_sha256=build_preflight.current_test_hashes(), collected_nodeids=nodeids, passed_nodeids=nodeids)
    save(record_path, record)
    review, attestation_path = root / "review.md", root / "attestation.json"
    review.write_text("Synthetic fixture review only; not physical admission evidence.", encoding="utf-8")
    refs = dict(test_xml=build_preflight.evidence_ref(xml),
                test_record=build_preflight.evidence_ref(record_path),
                review_report=build_preflight.evidence_ref(review))
    for row in smoke_rows():
        path = evidence / f"smoke_{row['scenario']}.json"
        save(path, row)
        refs["smoke:" + row["scenario"]] = build_preflight.evidence_ref(path)
    save(attestation_path, dict(format=build_preflight.REVIEW_FORMAT, review_kind="final_admission",
        status="final", decision="approved", reviewer="synthetic-test-fixture",
        source_pins=PINS, runtime_versions=VERSIONS, test_sha256=record["test_sha256"],
        evidence_sha256={role: ref["sha256"] for role, ref in refs.items()},
        smoke_runtime_versions={s: VERSIONS for s in SCENARIOS}))
    gate = root / "gate.json"

    def build():
        monkeypatch.setattr(sys, "argv", ["build_preflight.py", "--evidence", str(evidence),
            "--review", str(review), "--attestation", str(attestation_path), "--output", str(gate)])
        build_preflight.main()

    return SimpleNamespace(gate=gate, evidence=evidence, sources=sources, xml=xml,
        record=record_path, review=review, attestation=attestation_path, build=build)


@pytest.mark.parametrize("fault", [None, "xml_failure", "xml_skipped", "xml_short", "source",
    "test_hash", "xml_hash", "exit_code", "review", "runtime", "suite_missing", "identity_missing",
    "duplicate_identity", "unexecuted_case"])
def test_gate_binds_passing_test_review_and_smoke_evidence(tmp_path, monkeypatch, fault):
    fixture = admission_fixture(tmp_path, monkeypatch)
    if fault in ("xml_failure", "xml_skipped", "xml_short", "identity_missing", "duplicate_identity"):
        tree = ET.parse(fixture.xml)
        cases = tree.getroot().findall("testcase")
        if fault == "xml_short":
            tree.getroot().remove(cases[-1])
        elif fault == "identity_missing":
            cases[0].remove(cases[0].find("properties"))
        elif fault == "duplicate_identity":
            cases[0].find("./properties/property").set("value", cases[1].find("./properties/property").get("value"))
        else:
            ET.SubElement(cases[0], "failure" if fault == "xml_failure" else "skipped")
        tree.write(fixture.xml, encoding="utf-8")
        change_json(fixture.record, lambda r: r.update(xml_sha256=run_budget.file_hash(fixture.xml)))
    elif fault in ("source", "test_hash", "xml_hash", "exit_code", "runtime"):
        key, value = {"source": ("source_pins", {}), "test_hash": ("test_sha256", {}),
            "xml_hash": ("xml_sha256", "drift"), "exit_code": ("exit_code", 1),
            "runtime": ("runtime_versions", {})}[fault]
        change_json(fixture.record, lambda r: r.update({key: value}))
    elif fault == "review":
        # A quoted PASS line cannot substitute for the independent final decision.
        fixture.review.write_text("Quoted MULTI_REVIEW: PASS; superseded by rejection.", encoding="utf-8")
        change_json(fixture.attestation, lambda r: r.update(decision="changes_required"))
    elif fault == "suite_missing":
        (fixture.sources / "test_multi_runner.py").unlink()
    elif fault == "unexecuted_case":
        change_json(fixture.record, lambda r: r["passed_nodeids"].pop())
    if fault:
        with pytest.raises(ValueError):
            fixture.build()
        assert not fixture.gate.exists()
    else:
        fixture.build()
        assert len(build_preflight.verify_gate(fixture.gate)["evidence"]) == 9
        fixture.review.write_text("changed after admission", encoding="utf-8")
        with pytest.raises(ValueError, match="evidence changed"):
            build_preflight.verify_gate(fixture.gate)


@pytest.mark.parametrize("fault", ["empty", "missing", "arbitrary", "legacy", "test_edit", "source",
    "runtime", "smoke_role", "smoke_profile", "xml_failure", "suite_omitted"])
def test_gate_consumption_revalidates_current_required_evidence(tmp_path, monkeypatch, fault):
    fixture = admission_fixture(tmp_path, monkeypatch)
    fixture.build()
    gate = read(fixture.gate)
    if fault in ("empty", "missing", "arbitrary"):
        if fault == "empty":
            gate["evidence"] = {}
        elif fault == "missing":
            gate["evidence"].pop("test_record")
        else:
            gate["evidence"]["unrelated"] = gate["evidence"].pop("test_record")
    elif fault == "legacy":
        gate = dict(ready_for_bounded_pilot=True, source_pins=PINS, evidence_sha256={})
    elif fault == "test_edit":
        (fixture.sources / "test_multi_runner.py").write_text("# changed test\n", encoding="utf-8")
    elif fault == "source":
        monkeypatch.setattr(build_preflight, "pins", lambda _: {"snapshot": "new-source"})
    elif fault == "runtime":
        monkeypatch.setattr(build_preflight, "runtime_versions", lambda: dict(VERSIONS, torch="new"))
    elif fault.startswith("smoke_"):
        role = "smoke:" + SCENARIOS[0]
        path = Path(gate["evidence"][role]["path"])
        key, value = ("scenario", SCENARIOS[1]) if fault == "smoke_role" else ("profile_sha256", "new")
        change_json(path, lambda r: r.update({key: value}))
        gate["evidence"][role] = build_preflight.evidence_ref(path)
    else:
        tree = ET.parse(fixture.xml)
        if fault == "xml_failure":
            ET.SubElement(tree.getroot().find("testcase"), "failure")
        else:
            # Keep >=60 cases and matching execution records, but omit one suite.
            for case in tree.getroot().findall("testcase"):
                prop = case.find("./properties/property")
                if "test_multi_runner.py" in prop.get("value"):
                    prop.set("value", prop.get("value").replace("test_multi_runner.py", "test_multi_pilot.py") + "_extra")
            ids = [c.find("./properties/property").get("value") for c in tree.getroot().findall("testcase")]
            change_json(fixture.record, lambda r: r.update(collected_nodeids=ids, passed_nodeids=ids))
        tree.write(fixture.xml, encoding="utf-8")
        change_json(fixture.record, lambda r: r.update(xml_sha256=run_budget.file_hash(fixture.xml)))
        gate["evidence"]["test_xml"] = build_preflight.evidence_ref(fixture.xml)
        gate["evidence"]["test_record"] = build_preflight.evidence_ref(fixture.record)
    save(fixture.gate, gate)
    with pytest.raises(ValueError):
        build_preflight.verify_gate(fixture.gate)


@pytest.mark.parametrize("fault", ["stale_source", "stale_test", "stale_smoke", "stale_report",
    "stale_xml", "code_review", "superseded", "rejected", "conflicting", "smoke_runtime"])
def test_final_review_attestation_binds_exact_candidate_inputs(tmp_path, monkeypatch, fault):
    fixture = admission_fixture(tmp_path, monkeypatch)
    attestation = read(fixture.attestation)
    if fault in ("stale_source", "stale_test", "code_review", "superseded", "rejected", "smoke_runtime"):
        key, value = {"stale_source": ("source_pins", {}), "stale_test": ("test_sha256", {}),
            "code_review": ("review_kind", "code_review"), "superseded": ("status", "superseded"),
            "rejected": ("decision", "changes_required"), "smoke_runtime": ("smoke_runtime_versions", {})}[fault]
        attestation[key] = value
    elif fault == "conflicting":
        attestation["verdict"] = "FAIL"
    else:
        role = {"stale_smoke": "smoke:" + SCENARIOS[-1],
                "stale_report": "review_report", "stale_xml": "test_xml"}[fault]
        attestation["evidence_sha256"][role] = "0" * 64
    save(fixture.attestation, attestation)
    with pytest.raises(ValueError, match="attestation"):
        fixture.build()
    assert not fixture.gate.exists()


@pytest.mark.parametrize("boundary", ["builder", "consumer"])
@pytest.mark.parametrize("fault", ["missing", "wrong", "attestation_mismatch"])
def test_smoke_recorded_runtime_matches_admission_and_attestation(tmp_path, monkeypatch, boundary, fault):
    fixture = admission_fixture(tmp_path, monkeypatch)
    if boundary == "consumer":
        fixture.build()
    scenario = SCENARIOS[2]
    role = "smoke:" + scenario
    path = fixture.evidence / f"smoke_{scenario}.json"
    attestation = read(fixture.attestation)
    if fault == "missing":
        change_json(path, lambda row: row.pop("runtime_versions"))
    elif fault == "wrong":
        changed = dict(VERSIONS, torch="different-smoke-runtime")
        change_json(path, lambda row: row.update(runtime_versions=changed))
        # Even a reviewer agreeing with the smoke cannot override the admitted runtime.
        attestation["smoke_runtime_versions"][scenario] = changed
    else:
        attestation["smoke_runtime_versions"][scenario] = dict(VERSIONS, torch="different-review-runtime")
    attestation["evidence_sha256"][role] = run_budget.file_hash(path)
    save(fixture.attestation, attestation)
    expected = "attestation" if fault == "attestation_mismatch" else "Recorded smoke runtime"
    if boundary == "consumer":
        gate = read(fixture.gate)
        gate["evidence"][role] = build_preflight.evidence_ref(path)
        gate["evidence"]["review_attestation"] = build_preflight.evidence_ref(fixture.attestation)
        save(fixture.gate, gate)
        with pytest.raises(ValueError, match=expected):
            build_preflight.verify_gate(fixture.gate)
    else:
        with pytest.raises(ValueError, match=expected):
            fixture.build()
        assert not fixture.gate.exists()


def test_test_evidence_records_absolute_identities_and_all_three_phases(tmp_path):
    from types import SimpleNamespace
    from run_tests import ExecutionEvidence
    plugin = ExecutionEvidence()
    item = SimpleNamespace(path=tmp_path / "test_one.py", nodeid="test_one.py::test_x", user_properties=[])
    plugin.pytest_collection_modifyitems([item])
    plugin.pytest_collection_finish(SimpleNamespace(items=[item]))
    expected = str(item.path.resolve()) + "::test_x"
    assert item.user_properties == [("nodeid", expected)]
    assert plugin.collected == [expected]
    for phase in ("setup", "call"):
        plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid, when=phase, outcome="passed"))
    assert plugin.passed() == []
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid, when="teardown", outcome="failed"))
    assert plugin.passed() == []
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid, when="teardown", outcome="passed"))
    assert plugin.passed() == [expected]
