import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import projection_audit as audit


NAMES = ["memory/remaining/0", "state/time_sec", "memory/action_anchor/0",
         "memory/action_anchor/1", "memory/previous_requested/0", "memory/previous_requested/1",
         "memory/previous_executed/0", "memory/previous_executed/1"]


@pytest.fixture(autouse=True)
def isolate_production(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr(audit, "REPO", repo)
    monkeypatch.setattr(audit, "BASE", repo / "pilot")
    monkeypatch.setattr(audit, "GOAL_ROOT", repo / "goal")
    monkeypatch.setattr(audit, "DEFAULT_SNAPSHOT", repo / "snapshot")

    def forbidden(*args, **kwargs):
        pytest.fail("Production loader/runtime called by a synthetic test")

    for name in ("load_base", "load_collections", "boot", "verify_pins", "runtime_versions"):
        monkeypatch.setattr(audit, name, forbidden)


@pytest.fixture
def synthetic_audit(tmp_path, monkeypatch):
    cfg = SimpleNamespace(network=SimpleNamespace(total_ramp_capacity=6000.))
    cfg_plain, options = {"capacity": 6000.}, {"horizon": 3}
    contract = dict(source_pins={"fixture": "source"}, runtime_versions={"fixture": "runtime"},
        observation_schema={"names": NAMES}, environment_contract=dict(
            config_sha256=audit.digest(cfg_plain), options_sha256=audit.digest(options)))
    ctx = SimpleNamespace(output=tmp_path / "output", runs={}, inputs=[], hash_reads=[],
                          loaded_rounds=[], q_calls=0, hashed_before_load=set())
    source = audit.REPO / "work/audit/projection_audit.py"
    for path in (source, audit.REPO / "work/sdmpc_rl_recovery_20260929/common.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic diagnostic source\n", encoding="ascii")
    monkeypatch.setattr(audit, "__file__", str(source))
    for name in ("completion.json", "train_round0/model_final.pt", "train_round1/model_final.pt"):
        path = audit.BASE / name
        audit.save(path, {"fixture": name})
        ctx.inputs.append(path)
    profiles = []
    for round_index in (0, 1):
        ctx.runs[round_index] = {}
        for index, scenario in enumerate(audit.SCENARIOS):
            folder = audit.BASE / f"collect_round{round_index}" / scenario
            transitions, rows = episode()
            for name in ("completion.json", "settings.json", "experience.pt",
                         "episode_00_trace.json", "episode_00_summary.json",
                         "observation_schema.json", "runtime_versions.json"):
                path = folder / name
                audit.save(path, rows if name == "episode_00_trace.json" else {"fixture": str(path)})
                ctx.inputs.append(path)
            provenance = dict(scenario=scenario, seed=6301 + round_index * 100 + index,
                profile_sha256=audit.file_hash(folder / "settings.json"),
                run_id=f"round{round_index}-{scenario}",
                experience_sha256=audit.file_hash(folder / "experience.pt"))
            ctx.runs[round_index][scenario] = dict(transitions=transitions, provenance=provenance,
                identity=dict(folder=str(folder), completion_sha256=audit.file_hash(folder / "completion.json"),
                    settings_sha256=audit.file_hash(folder / "settings.json"),
                    experience_sha256=provenance["experience_sha256"]))
            profiles.append(copy.deepcopy(provenance))
    replay = {}
    for scenario in audit.SCENARIOS:
        transitions = ctx.runs[0][scenario]["transitions"] + ctx.runs[1][scenario]["transitions"]
        replay[scenario] = {
            key: torch.tensor(np.asarray(values), dtype=torch.bool if key == "terminated" else torch.float32)
            for key, values in zip(("observations", "actions", "rewards", "next_observations", "terminated"),
                                   zip(*transitions))}
    ctx.model = dict(contract=contract, learner={"replay": replay}, training_profiles=profiles)
    ctx.before = {str(path.relative_to(audit.REPO)): audit.file_hash(path) for path in ctx.inputs}
    monkeypatch.setattr(audit, "BASE_HASH", audit.file_hash(audit.BASE / "train_round1/model_final.pt"))
    original_hash = audit.file_hash

    def fixture_hash(path):
        path = Path(path)
        assert path.is_relative_to(tmp_path), "Hashing must stay inside the synthetic fixture"
        ctx.hash_reads.append(path)
        return original_hash(path)

    def load_base():
        ctx.hashed_before_load = set(ctx.hash_reads)
        return ctx.model, None

    def load_collections(folders, round_index, source_pins, runtime, model_hash):
        assert folders == [audit.BASE / f"collect_round{round_index}" / s for s in audit.SCENARIOS]
        assert source_pins == contract["source_pins"] and runtime == contract["runtime_versions"]
        assert model_hash == (None if round_index == 0 else
                              original_hash(audit.BASE / "train_round0/model_final.pt"))
        ctx.loaded_rounds.append(round_index)
        return contract, ctx.runs[round_index]

    def q_aliases(*args):
        ctx.q_calls += 1
        return []

    @contextmanager
    def lock(output):
        output.mkdir(parents=True, exist_ok=True)
        (output / "runner.lock").write_bytes(b"0")
        yield

    monkeypatch.setattr(audit, "file_hash", fixture_hash)
    monkeypatch.setattr(audit, "load_base", load_base)
    monkeypatch.setattr(audit, "load_collections", load_collections)
    monkeypatch.setattr(audit, "boot", lambda *args: dict(cfg=cfg, options=options,
        rc=SimpleNamespace(to_plain_dict=lambda value: cfg_plain if value is cfg else value)))
    monkeypatch.setattr(audit, "runtime_versions", lambda: contract["runtime_versions"])
    monkeypatch.setattr(audit, "verify_pins", lambda *args: None)
    monkeypatch.setattr(audit, "q_aliases", q_aliases)
    monkeypatch.setattr(audit, "exclusive_run", lock)
    monkeypatch.setattr(audit.sys, "argv", [str(source), "--output", str(ctx.output)])
    return ctx


def episode():
    observations, rows = [], []
    for i in range(75):
        previous = [0., 0.] if i == 0 else [0., .6]
        observations.append(np.array([(75-i)/75., (900+180*i)/14400., 0., .6,
                                      *previous, *previous], dtype=np.float32))
        rows.append(dict(terminated=i == 74, control_step=i, time_sec=1080.+180*i,
            action_anchor=[0.,6000.], B_requested=[[0.,6000.]], B_executed=[0.,6000.],
            B_raw=[0.,6500.], interval_ttt=100., action_requested=[0.,.5]))
    transitions = [(obs,np.array([0.,.5], dtype=np.float32),-1.,
                    np.zeros(len(NAMES), dtype=np.float32) if i == 74 else observations[i+1],i == 74)
                   for i,obs in enumerate(observations)]
    return transitions, rows


def test_sequence_accepts_clock_memory_reward_and_projection():
    transitions, rows = episode()
    result = audit.validate_sequence(transitions, rows, NAMES, 6000.)
    assert result["transitions"] == 75 and result["true_terminals"] == 1


@pytest.mark.parametrize("fault", ["clock", "remaining", "terminal", "reward", "anchor", "memory", "projection", "next"])
def test_sequence_rejects_contract_drift(fault):
    transitions, rows = episode()
    if fault == "clock": transitions[5][0][1] += .01
    elif fault == "remaining": transitions[74][0][0] = 0
    elif fault == "terminal": rows[0]["terminated"] = True
    elif fault == "reward": rows[20]["interval_ttt"] = 200.
    elif fault == "anchor": rows[15]["action_anchor"][0] = 3.
    elif fault == "memory": transitions[7][0][6] = 1.
    elif fault == "projection": rows[7]["B_requested"] = [[0.,6500.]]
    elif fault == "next": transitions[74][3][0] = 1.
    with pytest.raises((AssertionError, ValueError)):
        audit.validate_sequence(transitions, rows, NAMES, 6000.)


def test_projection_aliases_cap_without_inventing_interior_aliases():
    groups = audit.alias_actions([12.,6000.],6000.,0.)
    assert len(groups) == 1 and len(groups[0]["actions"]) == 3
    assert np.array_equal(groups[0]["request"], [12.,6000.])
    assert audit.alias_actions([12.,4000.],6000.,0.) == []
    assert audit.alias_actions([12.,5000.],6000.,0.) == []


def test_alias_q_detects_raw_action_dependence_without_mutating_network():
    from types import SimpleNamespace
    critics = [torch.nn.Linear(len(NAMES)+2,1) for _ in range(2)]
    with torch.no_grad():
        for critic in critics:
            critic.weight.zero_(); critic.bias.zero_(); critic.weight[0,-1] = 2.
    state = copy.deepcopy([c.state_dict() for c in critics])
    rows = audit.q_aliases(SimpleNamespace(critics=critics), np.zeros(len(NAMES),np.float32), [0.,6000.],6000.)
    assert len(rows) == 3
    for row in rows:
        assert min(row["twin_q_spread"]) > 1.99
    for saved,critic in zip(state,critics):
        for key,value in saved.items(): assert torch.equal(value,critic.state_dict()[key])


def test_stop_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,"REPO",tmp_path/"repo")
    monkeypatch.setattr(audit,"GOAL_ROOT",tmp_path/"goal")
    output=tmp_path/"output"
    for folder in (audit.REPO,audit.GOAL_ROOT,output):
        folder.mkdir(); (folder/"STOP").touch()
        assert audit.stopped(output)
        (folder/"STOP").unlink()
    assert not audit.stopped(output)


def test_unique_schema_column():
    with pytest.raises(ValueError): audit.column(["x","x"],"x")
    with pytest.raises(ValueError): audit.column(["y"],"x")


def test_main_records_manifest_before_loading_and_covers_all_replay(synthetic_audit):
    ctx = synthetic_audit
    audit.main()
    result = audit.read(ctx.output / "completion.json")
    assert result["settings"]["input_sha256"] == ctx.before
    assert audit.read(ctx.output / "settings.json")["input_sha256"] == ctx.before
    assert set(ctx.inputs) <= ctx.hashed_before_load
    assert all(ctx.hash_reads.count(path) >= 2 for path in ctx.inputs)
    assert ctx.loaded_rounds == [0, 1] and ctx.q_calls == 750
    assert {(row["round"], row["scenario"]) for row in result["episodes"]} == {
        (r, s) for r in (0, 1) for s in audit.SCENARIOS}
    assert audit.read(ctx.output / "status.json")["status"] == "completed"
    assert {str(path.relative_to(audit.REPO)): audit.file_hash(path) for path in ctx.inputs} == ctx.before


@pytest.mark.parametrize("relative", [
    "completion.json", "train_round0/model_final.pt", "train_round1/model_final.pt",
    *[f"collect_round{r}/sweet_155_w/{name}" for r in (0, 1) for name in (
        "completion.json", "settings.json", "experience.pt", "episode_00_trace.json",
        "episode_00_summary.json", "observation_schema.json", "runtime_versions.json")]])
def test_main_rejects_input_drift_before_completion(synthetic_audit, monkeypatch, relative):
    ctx = synthetic_audit
    target = audit.BASE / relative
    original_q = audit.q_aliases

    def mutate_after_loading(*args):
        result = original_q(*args)
        if ctx.q_calls == 750:
            target.write_bytes(target.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(audit, "q_aliases", mutate_after_loading)
    with pytest.raises(ValueError, match="Input.*changed"):
        audit.main()
    assert not (ctx.output / "completion.json").exists()
    assert audit.read(ctx.output / "status.json")["status"] != "completed"


@pytest.mark.parametrize("round_index", [0, 1])
@pytest.mark.parametrize("field", ["scenario", "seed", "profile_sha256", "run_id", "experience_sha256"])
def test_main_rejects_changed_provenance_before_comparison(synthetic_audit, round_index, field):
    ctx = synthetic_audit
    run = ctx.runs[round_index][audit.SCENARIOS[0]]
    run["provenance"][field] = 9999 if field == "seed" else "different"
    with pytest.raises(ValueError, match="Collection provenance"):
        audit.main()
    assert ctx.q_calls == round_index * 375
    assert not (ctx.output / "completion.json").exists()


def test_main_rejects_repackaged_collection_with_identical_transitions(synthetic_audit):
    ctx = synthetic_audit
    run = ctx.runs[1][audit.SCENARIOS[0]]
    folder = Path(run["identity"]["folder"])
    audit.save(folder / "experience.pt", {"fixture": "repackaged, identical transition values"})
    run["provenance"].update(run_id="replacement-run", experience_sha256=audit.file_hash(folder / "experience.pt"))
    run["identity"]["experience_sha256"] = run["provenance"]["experience_sha256"]
    with pytest.raises(ValueError, match="Collection provenance"):
        audit.main()
    assert ctx.q_calls == 375
    assert not (ctx.output / "completion.json").exists()


@pytest.mark.parametrize("entry", ["completion.json", "STOP", "goal-stop", "repo-stop"])
def test_main_rechecks_guards_under_lock(synthetic_audit, monkeypatch, entry):
    ctx = synthetic_audit

    def no_start(*args):
        pytest.fail("Audit started despite an entry created while acquiring its lock")

    @contextmanager
    def racing_lock(output):
        output.mkdir()
        (output / "runner.lock").write_bytes(b"0")
        target = (audit.GOAL_ROOT / "STOP" if entry == "goal-stop" else
                  audit.REPO / "STOP" if entry == "repo-stop" else output / entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preserve this entry")
        yield
        assert target.read_bytes() == b"preserve this entry"

    monkeypatch.setattr(audit, "exclusive_run", racing_lock)
    monkeypatch.setattr(audit, "load_base", no_start)
    monkeypatch.setattr(audit, "boot", no_start)
    if entry == "completion.json":
        with pytest.raises(FileExistsError):
            audit.main()
        assert (ctx.output / entry).read_bytes() == b"preserve this entry"
    else:
        audit.main()
    assert not (ctx.output / "process.json").exists()
    assert not (ctx.output / "settings.json").exists()


@pytest.mark.parametrize("anchor_metering, expected_groups", [(5000.00012, 3), (5000.00001, 0)])
def test_alias_q_uses_distinct_float32_boundary_actions(anchor_metering, expected_groups):
    critics = [torch.nn.Linear(len(NAMES) + 2, 1) for _ in range(2)]
    with torch.no_grad():
        for critic in critics:
            critic.weight.zero_()
            critic.bias.zero_()
            critic.weight[0, -1] = 2.
    anchor = [0., anchor_metering]
    records = audit.q_aliases(SimpleNamespace(critics=critics), np.zeros(len(NAMES), np.float32), anchor, 6000.)
    assert len(records) == expected_groups
    for row in records:
        actions = np.asarray(row["nominal_actions"], dtype=np.float32)
        assert len(np.unique(actions, axis=0)) == len(actions) >= 2
        for action in actions:
            np.testing.assert_array_equal(audit.residual_budget(action, anchor, 6000.)[1], row["projected_request"])
        assert min(row["twin_q_spread"]) > 0.
