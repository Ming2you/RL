import copy
import json
import sys
from types import SimpleNamespace as NS
import pytest
import wave_support as w
import actor as adapter
import checks
import worker
import readout
from conftest import ACTOR, AUTH, FakeEnv, request


def retained():
    output = w.WAVE / w.SCENARIOS[0]
    record, path = w.checkpoint_record(output)
    return output, record, w.torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture
def candidate():
    return w.torch.load(w.PREDECESSOR / "model_final.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "dtype", "bounds", "nan", "inf"])
def test_actor_strict_complete_weights(candidate, bad):
    state = candidate["learner"]["models"]["actor"]
    if bad == "missing":
        del state["net.4.bias"]
    elif bad == "extra":
        state["extra"] = w.torch.zeros(1)
    elif bad == "shape":
        state["net.0.weight"] = state["net.0.weight"][:, :-1]
    elif bad == "dtype":
        state["bounds"] = state["bounds"].double()
    elif bad == "bounds":
        state["bounds"][0] = .3
    else:
        state["net.0.bias"][0] = float(bad)
    with pytest.raises(ValueError):
        adapter.FrozenActor(state)


@pytest.mark.parametrize("bad", ["phase", "phi", "critic", "actor", "polyak", "spec", "continuation", "gate", "candidate"])
def test_candidate_admission_failures(candidate, bad):
    settings = copy.deepcopy(candidate["settings"])
    done = w.read(w.PREDECESSOR / "completion.json")
    learner = candidate["learner"]
    if bad == "phase":
        learner["phase"] = "actor"
    elif bad in adapter.COUNTS:
        learner["counts"][bad] -= 1
    elif bad == "spec":
        learner["spec"]["gamma"] = .99
    elif bad == "continuation":
        learner["critic_continuation"] = "new_actor"
    elif bad == "gate":
        learner["metrics"]["gate"]["final"]["pooled"] = 1e10
    else:
        candidate["candidate"] = False
    with pytest.raises(ValueError):
        adapter.validate_candidate(candidate, settings, done)


def test_native_actor_no_rng_or_grad(candidate):
    before = w.torch.get_rng_state().clone()
    actor = adapter.FrozenActor(candidate["learner"]["models"]["actor"])
    assert w.torch.equal(before, w.torch.get_rng_state())
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())
    out = actor.act(w.np.zeros(2367, dtype=w.np.float32))
    assert out.dtype == w.np.float32 and out.shape == (2,)
    with pytest.raises(ValueError):
        actor.act(w.np.zeros(2367, dtype=w.np.float64))


def test_import_boundary(monkeypatch):
    checks.helpers()
    w.import_boundary()
    for name in ("learner", "data", "td3", "train_round", "local_runtime"):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, name, NS())
            with pytest.raises(ValueError, match="Learner"):
                w.import_boundary()
    monkeypatch.setitem(sys.modules, "runtime", NS(__file__="wrong/runtime.py"))
    with pytest.raises(ValueError, match="collision"):
        w.import_boundary()


@pytest.mark.parametrize("kind", ["model", "source", "settings", "completion"])
def test_auth_hash_fail_closed(tmp_path, monkeypatch, kind):
    path = tmp_path / (kind+".json")
    w.save(path, {"value": 1})
    original = w.file_hash(path)
    w.save(path, {"value": 2})
    with pytest.raises(ValueError, match="changed"):
        w.check_hash(path, original)
    if kind == "completion":
        monkeypatch.setattr(w, "PREDECESSOR", tmp_path)
        with pytest.raises(ValueError):
            adapter.authenticate()


@pytest.mark.parametrize("level", ["REPO", "GOAL", "WAVE", "slot"])
def test_stop_all_levels(synthetic, level):
    root = w.WAVE / w.SCENARIOS[0] if level == "slot" else getattr(w, level)
    w.save(root / "STOP", {"keep": True})
    before = w.file_hash(root / "STOP")
    with pytest.raises(w.Stopped):
        worker.run(request())
    assert FakeEnv.resets == FakeEnv.steps == 0
    assert w.file_hash(root / "STOP") == before


def test_kernel_duplicate_and_prefix_resume(synthetic):
    output = w.WAVE / w.SCENARIOS[0]
    with w.exclusive_run(output):
        with pytest.raises(OSError):
            worker.run(request())
    assert worker.run(request(limit=1)) == "checkpointed"
    folder, record, ck = retained()
    first = ck["transitions"][0]
    first_hash = record["sha256"]
    with pytest.raises(FileExistsError):
        worker.run(request())
    assert worker.run(request(resume=True, limit=1)) == "checkpointed"
    _, _, resumed = retained()
    assert FakeEnv.resets == 1 and FakeEnv.restores == 1 and FakeEnv.steps == 2
    assert w.file_hash(folder / record["path"]) == first_hash
    for actual, expected in zip(resumed["transitions"][0], first):
        w.np.testing.assert_array_equal(actual, expected)
    assert len(resumed["session_ids"]) == 2
    assert resumed["trace"][0] == ck["trace"][0]
    timing = w.session_timing(folder, resumed["settings"])
    restore = timing["restoration"]
    assert restore["returned_branch_reference_reconstructions"] == 1
    assert restore["elapsed_wall_seconds"] > 0 and restore["elapsed_cpu_seconds"] >= 0
    assert timing["elapsed_wall_seconds"] == sum(w.read(p)["elapsed_wall_seconds"]
        for p in (folder / "sessions").glob("*.end.json"))
    checks.validate_checkpoint(resumed, resumed["settings"], ACTOR)


@pytest.mark.parametrize("kind", ["source", "model", "settings"])
def test_midrun_mutation_stops_next_interval(synthetic, kind):
    original = FakeEnv.step
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kind == "source":
            synthetic.source["synthetic"] = "changed"
        elif kind == "model":
            synthetic.auth["model_sha256"] = "changed"
        else:
            output = w.WAVE / w.SCENARIOS[0]
            settings = w.read(output / "settings.json")
            settings["warmup_ttt"] += 1
            w.save(output / "settings.json", settings)
        return result
    synthetic.patch.setattr(FakeEnv, "step", step)
    with pytest.raises(ValueError):
        worker.run(request())
    assert FakeEnv.steps == 1
    assert not (w.WAVE / w.SCENARIOS[0] / "completion.json").exists()


def test_stop_after_interval_preserves_checkpoint(synthetic):
    original = FakeEnv.step
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        w.save(w.WAVE / "STOP", {"keep": True})
        return result
    synthetic.patch.setattr(FakeEnv, "step", step)
    assert worker.run(request()) == "stopped"
    assert retained()[1]["control_steps"] == 1 and FakeEnv.steps == 1


@pytest.mark.parametrize("field", ["action", "obs_chain", "reward", "inventory", "clock", "terminal", "area", "guard", "solve", "pfo", "timing", "control", "budget", "physical_inf", "q", "meter", "vsl", "green", "offset", "allocation"])
def test_retained_sequence_rejects_tamper(synthetic, field):
    worker.run(request(limit=2))
    _, _, ck = retained()
    rows, transitions = ck["trace"], ck["transitions"]
    if field == "action":
        transitions[0][1][0] += .01
    elif field == "obs_chain":
        transitions[1][0][0] += .01
    elif field == "reward":
        obs, act, _, nxt, done = transitions[0]
        transitions[0] = (obs, act, -.1, nxt, done)
    elif field == "inventory":
        rows[0]["plant_state"]["urban_link_storage"]["urban"] += 1
    elif field == "clock":
        rows[0]["time_sec"] += 180
    elif field == "terminal":
        rows[0]["terminated"] = True
    elif field == "area":
        rows[0]["urban_ttt"] += 1
    elif field == "guard":
        rows[0]["h3_guard_enabled"] = True
    elif field == "solve":
        rows[0]["lower_candidate_count"] = 2
    elif field == "pfo":
        rows[1]["pfo_calls"] = 1
    elif field == "timing":
        rows[0]["decision_cpu_seconds"] += 1
    elif field == "control":
        rows[0]["control"]["N_UF_star"] += 1
    elif field == "budget":
        rows[0]["execution_check"]["achieved"][1] += 1
    elif field == "physical_inf":
        rows[0]["plant_state"]["freeway_speed"]["a"][0] = float("inf")
    elif field == "meter":
        rows[0]["control"]["ramp_metering"]["r"] = 6001.
    elif field == "vsl":
        rows[0]["control"]["vsl"]["a__seg0"] = 90.
    elif field == "green":
        rows[0]["control"]["green_times"]["A_p1"] += 1.
    elif field == "offset":
        rows[0]["control"]["offsets"]["A"] = 16.
    elif field == "allocation":
        rows[0]["control"]["inflow_outflow_allocation"]["unexpected"] = 1.
    else:
        rows[0]["policy_q"] = [0., 0.]
    with pytest.raises((ValueError, AssertionError)):
        checks.validate_checkpoint(ck, ck["settings"], ACTOR)


@pytest.mark.parametrize("bad", [float("nan"), -float("inf"), {"nonfinite_float": "+inf"}])
def test_diagnostic_infinity_boundary(bad):
    trace = [dict(candidates=[dict(stationarity=bad)])]
    with pytest.raises(ValueError):
        checks.tag(trace)
    tagged, changes = checks.tag([dict(candidates=[dict(stationarity=float("inf"))])])
    assert len(changes) == 1
    assert checks.raw_trace(tagged)[0]["candidates"][0]["stationarity"] == float("inf")
    with pytest.raises(ValueError):
        checks.tag([dict(reward=float("inf"))])


def test_interrupt_unknown_and_exact_restore(synthetic):
    worker.run(request(limit=1))
    output, record, ck = retained()
    # A retained start without an end models a killed worker, not a zero-duration session.
    sid = "f"*32
    start = w.read(output / "sessions" / f"{ck['session_ids'][0]}.start.json")
    start["session_id"] = sid
    w.save(output / "sessions" / f"{sid}.start.json", start)
    worker.run(request(resume=True, limit=1))
    ck = retained()[2]
    timing = w.session_timing(output, ck["settings"], ck["session_ids"])
    assert timing["timing_status"] == "UNKNOWN" and timing["elapsed_wall_seconds"] is None
    assert timing["unknown_sessions"] == [sid] and timing["known_elapsed_wall_seconds"] > 0
    assert FakeEnv.resets == 1 and FakeEnv.restores == 1 and FakeEnv.steps == 2


def test_uncertain_inflight_refused_and_orphan_preserved(synthetic):
    worker.run(request(limit=1))
    output, record, ck = retained()
    worker.operation(output, 2, ck["session_ids"][-1], record)
    orphan = output / "checkpoints/orphan.tmp"
    w.save(orphan, {"retained": True})
    with pytest.raises(ValueError, match="Uncheckpointed"):
        worker.run(request(resume=True, limit=1))
    assert FakeEnv.resets == FakeEnv.steps == 1 and FakeEnv.restores == 0
    assert w.read(orphan) == {"retained": True}


@pytest.mark.parametrize("field", ["k", "profile_hash", "last_requested", "physical_state", "dual", "observation", "options", "normalization"])
def test_checkpoint_boundary_rejected_before_restore(synthetic, field):
    worker.run(request(limit=1))
    output, record, ck = retained()
    env = ck["environment"]
    if field == "k":
        env["k"] += 1
    elif field == "profile_hash":
        env["profile_hash"] = "wrong"
    elif field == "last_requested":
        env["last_requested"][0] += 1.
    elif field == "physical_state":
        env["sim"].state.ramp_queue["r"] += 1.
    elif field == "dual":
        env["dual"][0, 0] = float("nan")
    elif field == "options":
        env["contract"]["options"]["fd_green_sec"] += 1
    elif field == "normalization":
        env["contract"]["observation"]["delay"] += 1
    else:
        ck["observation"][0] += 1.
    w.persist_checkpoint(output, ck)
    with pytest.raises((ValueError, AssertionError)):
        worker.run(request(resume=True))
    assert FakeEnv.restores == 0 and FakeEnv.steps == 1


@pytest.mark.parametrize("zero,inventory,passed", [(5, 550., True), (6, 550., False), (5, 550.01, False)])
def test_health_exact_boundaries(synthetic, zero, inventory, passed):
    worker.run(request(limit=1))
    ck = retained()[2]
    trace = [copy.deepcopy(ck["trace"][0]) for _ in range(75)]
    for row in trace[:zero]:
        row["B_requested"][0][1] = 0.
    trace[-1]["inventory"] = inventory
    summary = checks.summarize(trace, ck["settings"], {})
    assert all(summary["health"].values()) is passed


def test_restored_observation_must_be_exact(synthetic):
    worker.run(request(limit=1))
    original = FakeEnv.restore
    def changed(self, ck):
        obs = original(self, ck)
        obs[0] += .01
        return obs
    synthetic.patch.setattr(FakeEnv, "restore", changed)
    with pytest.raises(AssertionError):
        worker.run(request(resume=True))
    assert FakeEnv.steps == 1


def test_invalid_checkpoint_retains_raw_orphan(synthetic):
    original = FakeEnv.step
    def changed(self, *args, **kwargs):
        obs, reward, done, row = original(self, *args, **kwargs)
        row["reward"] = float("nan")
        return obs, float("nan"), done, row
    synthetic.patch.setattr(FakeEnv, "step", changed)
    with pytest.raises(ValueError):
        worker.run(request())
    output, record, ck = retained()
    assert record["control_steps"] == 0 and len(list((output / "checkpoints").glob("*.pt"))) == 2


def test_five_completed_readonly_and_rejections(synthetic):
    worker.run(request(limit=1))
    first_record = retained()[1]
    for scenario in w.SCENARIOS:
        assert worker.run(request(scenario, resume=scenario == w.SCENARIOS[0])) == "completed"
    assert FakeEnv.resets == 5 and FakeEnv.steps == 375 and FakeEnv.restores == 1
    first = w.WAVE / w.SCENARIOS[0]
    assert w.file_hash(first / first_record["path"]) == first_record["sha256"]
    original_load = w.torch.load
    def no_raw_checkpoint(path, **kwargs):
        assert "checkpoints" not in str(path), "Completed validator must not deserialize physical checkpoints"
        return original_load(path, **kwargs)
    synthetic.patch.setattr(w.torch, "load", no_raw_checkpoint)
    result = readout.validate_wave()
    assert result["integrity_pass"] and result["wave_elapsed_wall_seconds"] is None
    assert [s["scenario"] for s in result["scenarios"]] == list(w.SCENARIOS)
    for row in result["scenarios"]:
        assert row["control_steps"] == 75 and row["simulation_seconds"] == 14400
        assert row["ttt"] == 80. and row["terminal_inventory"] == 63.
        assert row["pfo_calls"] == 1 and row["lower_candidate_solves"] == 75
        assert row["timing"]["decision_wall"]["sum_seconds"] > row["timing"]["actor_wall"]["sum_seconds"]
        restore = row["worker_sessions"]["restoration"]
        assert restore["timing_status"] == "KNOWN"
        assert restore["returned_branch_reference_reconstructions"] == int(row["scenario"] == w.SCENARIOS[0])
        with pytest.raises(FileExistsError):
            worker.run(request(row["scenario"], resume=True))
    done = w.read(first / "completion.json")
    restore_path = next((first / "sessions").glob("*.restore-end.json"))
    restore_end = w.read(restore_path)
    w.save(restore_path, dict(restore_end, elapsed_cpu_seconds=restore_end["elapsed_cpu_seconds"]+1.))
    with pytest.raises(ValueError, match="restoration binding"):
        readout.validate_wave()
    w.save(restore_path, restore_end)
    settings = w.read(first / "settings.json")
    for field, bad in (("model_sha256", "different"), ("profile_sha256", "wrong-profile"),
                       ("training_seed", 42), ("evaluation", True)):
        changed = dict(settings, **{field: bad})
        w.save(first / "settings.json", changed)
        with pytest.raises(ValueError):
            readout.validate_wave()
        w.save(first / "settings.json", settings)
    extra = w.WAVE / "extra/completion.json"
    w.save(extra, done)
    with pytest.raises(ValueError, match="exactly five"):
        readout.validate_wave()
    extra.rename(extra.with_suffix(".preserved"))
    missing = first / "completion.json"
    missing.rename(first / "completion.preserved")
    with pytest.raises(ValueError, match="exactly five"):
        readout.validate_wave()
    (first / "completion.preserved").rename(missing)
    loader = readout.load_completed
    # Every scenario independently participates in both health screens.
    for scenario in w.SCENARIOS:
        for gate in ("zero_nuf_requests_le_5", "terminal_inventory_le_550"):
            def health_failure(folder, name, *args):
                row = copy.deepcopy(next(r for r in result["scenarios"] if r["scenario"] == name))
                if name == scenario:
                    row["health"][gate] = False
                return row
            synthetic.patch.setattr(readout, "load_completed", health_failure)
            assert readout.validate_wave()["all_health_gates_pass"] is False
    synthetic.patch.setattr(readout, "load_completed", lambda folder, name, *args:
        dict(next(r for r in result["scenarios"] if r["scenario"] == name), run_id="duplicate"))
    with pytest.raises(ValueError, match="Duplicate"):
        readout.validate_wave()
    synthetic.patch.setattr(readout, "load_completed", loader)
    assert len(list(w.WAVE.glob("*/completion.json"))) == 5
