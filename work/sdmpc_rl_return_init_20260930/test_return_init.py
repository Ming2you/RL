"""All optimization here is on small synthetic arrays; real data is read-only."""
import copy
import json
import sys
from pathlib import Path
import pytest
import runtime as rt
import data as dataset
import learner as learning
import run as runner
from runtime import np, torch, SPEC, SCENARIOS


def synthetic():
    rows = []
    for scenario in range(5):
        for local in (False, True):
            for step in range(5):
                obs = np.array([1., (5-step)/5., scenario/5., local*.1, .2, .6], dtype=np.float32)
                rows.append(dict(obs=obs, next_obs=obs*.9 if step != 4 else np.zeros(6, np.float32),
                    anchor=[-19.1234567890123, 6000. if step % 2 else 5900.123456789],
                    next_anchor=[-20.9876543210987, 5900.] if step != 4 else [0., 0.],
                    action=[.1, .1] if local else [0., 0.], reward=-.2,
                    returns=-(5-step)*.2, scenario=scenario, local=local,
                    terminal=step == 4, horizon=5-step))
    values = {key: torch.from_numpy(np.asarray([row[key] for row in rows])) for key in rows[0]}
    values["action"] = values["action"].float()
    return values


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setitem(SPEC, "phi_updates", 6)
    monkeypatch.setitem(SPEC, "critic_updates", 4)
    monkeypatch.setitem(SPEC, "actor_updates", 3)
    monkeypatch.setattr(learning, "numerical_gate", lambda initial, final: dict(passed=True))
    return synthetic()


def finish(model):
    while model.phase not in ("done", "gate_failed"):
        model.update()
    return model


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_frozen_specification_and_architecture():
    assert (SPEC["phi_updates"], SPEC["critic_updates"], SPEC["actor_updates"]) == (1000, 250, 10)
    assert SPEC["seed"] == 7200 and SPEC["batch_size"] == 40
    actor = learning.Actor()
    assert [(m.in_features, m.out_features) for m in actor.net if isinstance(m, torch.nn.Linear)] == [(2367, 64), (64, 64), (64, 2)]
    assert sum(isinstance(m, torch.nn.ReLU) for m in actor.net) == 2
    assert torch.equal(actor(torch.ones(4, 2367)), torch.zeros(4, 2))
    with torch.no_grad():
        actor.net[-1].bias[:] = torch.tensor([100., -100.])
    assert torch.equal(actor(torch.ones(1, 2367)), torch.tensor([[.2, -.1]]))


def test_projection_float64_signed_and_zero_after_fallback():
    anchor = torch.tensor([[-12.1234567890123, 5999.123456789012], [-300., 7.]], dtype=torch.float64)
    action = torch.tensor([[.123456789, .1], [-.2, -.1]], dtype=torch.float32, requires_grad=True)
    request = dataset.projected(anchor, action)
    expected = anchor.numpy() + action.detach().numpy().astype(np.float64)*[50., 1000.]
    expected[:, 1] = expected[:, 1].clip(0., 6000.)
    np.testing.assert_array_equal(request.detach().numpy(), expected)
    assert request.dtype == torch.float64 and request[1, 0] < 0
    assert torch.equal(dataset.projected(anchor, torch.zeros_like(action)), anchor)
    assert not torch.equal(dataset.projected(anchor.float().double(), action), request)
    normalized = dataset.network_budget(anchor, action)
    torch.testing.assert_close(normalized, (request/torch.tensor([1000., 10000.])).float(), rtol=0, atol=0)
    gradient = torch.autograd.grad(normalized.sum(), action)[0]
    torch.testing.assert_close(gradient, torch.tensor([[.05, 0.], [.05, 0.]]))
    with pytest.raises(ValueError, match="float64"):
        dataset.projected(anchor.float(), action)


@pytest.mark.parametrize("phi", [True, False])
def test_every_minibatch_exact_balance_and_terminal_quota(phi):
    data, rng = synthetic(), np.random.default_rng(7200)
    for _ in range(50):
        batch = dataset.sample(data, rng, phi)
        assert len(batch["obs"]) == 40
        for scenario in range(5):
            mask = batch["scenario"] == scenario
            assert int(mask.sum()) == 8
            for local in ([False] if phi else [False, True]):
                group = mask & (batch["local"] == local)
                assert int(group.sum()) == (8 if phi else 4)
                assert int(batch["terminal"][group].sum()) >= 1
        if phi:
            assert not batch["local"].any()


def test_sampling_rejects_missing_or_duplicate_terminal():
    data = synthetic()
    data["terminal"][:] = False
    with pytest.raises(ValueError, match="exactly one"):
        dataset.sample(data, np.random.default_rng(1))
    data["terminal"][:] = True
    with pytest.raises(ValueError, match="exactly one"):
        dataset.sample(data, np.random.default_rng(1))


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, obs):
        assert torch.isfinite(obs).all()
        return torch.full((*obs.shape[:-1], 1), self.value, dtype=torch.float32)


def test_residual_algebra_and_true_terminal_masking():
    data = synthetic()
    model = learning.Learner(data)
    model.phi = Constant(7.)
    model.targets = torch.nn.ModuleList([Constant(2.), Constant(3.)])
    terminal = data["terminal"]
    data["next_obs"][terminal] = float("nan")
    data["next_anchor"][terminal] = float("nan")
    # Carry rows use MC only; even nonterminal carry next states must not be evaluated.
    data["next_obs"][~data["local"]] = float("nan")
    target = model.target(data)
    assert not target.requires_grad
    torch.testing.assert_close(target[~data["local"]], data["returns"][~data["local"]] - 7.)
    torch.testing.assert_close(target[data["local"] & terminal], data["reward"][data["local"] & terminal] - 7.)
    live = data["local"] & ~terminal
    torch.testing.assert_close(target[live], data["reward"][live] + 2.)


def test_terminal_only_never_evaluates_next_networks():
    data = synthetic()
    data = {k: v[data["terminal"]] for k, v in data.items()}
    model = learning.Learner(data)
    data["next_obs"][:] = float("nan")
    data["next_anchor"][:] = float("nan")
    def forbidden(*args):
        raise AssertionError("Terminal next head was evaluated")
    model.targets.forward = forbidden
    for target in model.targets:
        target.forward = forbidden
    model.target(data)


def test_phase_freezing_targets_and_no_changing_actor_td(small):
    model = learning.Learner(small)
    zero_actor = rt.tensor_hash(model.actor.state_dict())
    while model.phase == "phi":
        assert model.phi[0].weight.requires_grad
        assert not model.actor.net[0].weight.requires_grad
        model.update()
    frozen_phi = rt.tensor_hash(model.phi.state_dict())
    equal(model.critics.state_dict(), model.targets.state_dict())
    for critic in model.critics:
        assert torch.count_nonzero(critic[-1].weight) == torch.count_nonzero(critic[-1].bias) == 0
    initial_target = copy.deepcopy(model.targets.state_dict())
    model.update()
    equal(initial_target, model.targets.state_dict())
    model.update()
    for name, value in model.targets.state_dict().items():
        expected = initial_target[name]*(1.-.005) + model.critics.state_dict()[name]*.005
        torch.testing.assert_close(value, expected, rtol=1e-7, atol=1e-9)
    while model.phase == "critic":
        model.update()
    assert rt.tensor_hash(model.actor.state_dict()) == zero_actor
    model.update()
    critics, targets = copy.deepcopy(model.critics.state_dict()), copy.deepcopy(model.targets.state_dict())
    assert model.phase == "actor"
    model.target = lambda _: (_ for _ in ()).throw(AssertionError("TD during actor proposal"))
    finish(model)
    equal(critics, model.critics.state_dict())
    equal(targets, model.targets.state_dict())
    assert rt.tensor_hash(model.phi.state_dict()) == frozen_phi
    assert model.counts == dict(phi=6, critic=4, actor=3, polyak=2)
    with pytest.raises(ValueError, match="No updates"):
        model.update()


@pytest.mark.parametrize("split", [1, 6, 7, 10, 11, 12, 14])
def test_deterministic_resume_all_phases(small, tmp_path, split):
    uninterrupted = finish(learning.Learner(small))
    interrupted = learning.Learner(small)
    for _ in range(split):
        interrupted.update()
    record = rt.checkpoint(tmp_path, dict(learner=interrupted.state()))
    assert len(record["sha256"]) == 64
    resumed = learning.Learner(small)
    resumed.load(rt.restore(tmp_path)["learner"])
    finish(resumed)
    equal(uninterrupted.state(), resumed.state())


def test_independent_rng_streams():
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    np.random.seed(23)
    legacy = np.random.get_state()
    learning.Learner(synthetic()).update()
    assert torch.equal(before, torch.get_rng_state())
    equal(legacy, np.random.get_state())


def test_numerical_gate_exact_threshold_and_per_scenario():
    initial = dict(pooled=10., per_scenario={s: 10. for s in SCENARIOS})
    final = dict(pooled=5., per_scenario={s: 5. for s in SCENARIOS})
    assert learning.numerical_gate(initial, final)["passed"]
    final["pooled"] = 5.000001
    assert not learning.numerical_gate(initial, final)["passed"]
    final["pooled"] = 4.
    final["per_scenario"][SCENARIOS[0]] = 10.
    assert not learning.numerical_gate(initial, final)["passed"]


def test_gate_failure_no_actor_steps_or_auto_extension(small, monkeypatch):
    monkeypatch.setattr(learning, "numerical_gate", lambda *args: dict(passed=False))
    model = finish(learning.Learner(small))
    assert model.phase == "gate_failed" and model.counts == dict(phi=6, critic=4, actor=0, polyak=2)
    assert rt.tensor_hash(model.actor.state_dict()) == model.metrics["zero_actor_sha256"]
    with pytest.raises(ValueError):
        model.update()


def test_diagnostics_scope_finite_and_aliases(small):
    model = learning.Learner(small)
    while model.phase != "diagnostics":
        model.update()
    model.update()
    metrics = model.metrics["before_actor"]
    json.dumps(metrics, allow_nan=False)
    assert len(metrics["rows"]) == 50
    assert len([r for r in metrics["rows"] if r["terminal_exact_error"] is not None]) == 10
    assert any(a["exact_projected_aliases"] for a in metrics["projection_aliases"])
    assert "not a calibrated Q error" in metrics["labels"]
    for row in metrics["rows"]:
        assert len(row["q1_q2_min"]) == len(row["own_target_residual"]) == 3


def trajectory():
    obs = [np.full(2367, k/75., dtype=np.float32) for k in range(75)] + [np.zeros(2367, np.float32)]
    transitions, trace = [], []
    for k in range(75):
        anchor = [-12.1234567890123-k/10., 5990.123456789012-k/10.]
        execution = [-12.1234567890123-(k+1)/10., 5990.123456789012-(k+1)/10.]
        action = np.zeros(2, np.float32)
        transitions.append((obs[k], action, -1., obs[k+1], k == 74))
        trace.append(dict(scenario=SCENARIOS[0], behavior="carry", control_step=k, step=k+5,
            profile_sha256="training", terminated=k == 74, truncated=False, interval_ttt=100.,
            action_anchor=anchor, B_requested=[anchor], B_executed=execution,
            action_requested=action.tolist()))
    return transitions, trace, dict(evaluation=False, profile_sha256="training", warmup_ttt=99999.)


def test_trajectory_returns_exclude_warmup_exact_trace_anchors():
    transitions, trace, settings = trajectory()
    rows = dataset.episode_arrays(transitions, trace, settings, SCENARIOS[0], "carry")
    assert rows[0]["returns"] == -75. and rows[-1]["returns"] == -1.
    np.testing.assert_array_equal(rows[0]["anchor"], trace[0]["action_anchor"])
    np.testing.assert_array_equal(rows[0]["next_anchor"], trace[1]["action_anchor"])
    np.testing.assert_array_equal(rows[-1]["next_anchor"], [0., 0.])
    assert rows[0]["anchor"].dtype == np.float64
    assert not np.array_equal(rows[0]["anchor"], rows[0]["anchor"].astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("corruption", ["evaluation", "reward", "chain", "terminal", "carry_action", "request", "anchor_chain"])
def test_data_failures(corruption):
    transitions, trace, settings = trajectory()
    if corruption == "evaluation":
        settings["evaluation"] = True
    elif corruption == "reward":
        trace[0]["interval_ttt"] = 1.
    elif corruption == "chain":
        transitions[0][3][0] = -100.
        transitions[1] = (np.ones(2367, np.float32), *transitions[1][1:])
    elif corruption == "terminal":
        trace[0]["terminated"] = True
    elif corruption == "carry_action":
        transitions[0][1][0] = .1
    elif corruption == "request":
        trace[0]["B_requested"] = [[0., 0.]]
    else:
        trace[0]["B_executed"] = [0., 0.]
    with pytest.raises((ValueError, AssertionError)):
        dataset.episode_arrays(transitions, trace, settings, SCENARIOS[0], "carry")


def test_hash_failure_before_deserialization(monkeypatch):
    monkeypatch.setitem(dataset.ROOT_PINS, "completion.json", "0"*64)
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("deserialized")))
    with pytest.raises(ValueError, match="root hash"):
        dataset.manifest()


@pytest.mark.parametrize("leak", ["flag", "profile"])
def test_manifest_rejects_evaluation_leakage(monkeypatch, leak):
    original = dataset.read
    def altered(path):
        value = original(path)
        def corrupt(settings):
            if leak == "flag":
                settings["evaluation"] = True
            else:
                settings["profile_sha256"] = settings["contract"]["environment_contract"]["scenarios"][0]["profile_sha256"]
        if Path(path).name == "settings.json":
            corrupt(value)
        elif Path(path).name == "completion.json" and "settings" in value:
            corrupt(value["settings"])
        return value
    monkeypatch.setattr(dataset, "read", altered)
    with pytest.raises(ValueError, match="Training-only|evaluation leakage"):
        dataset.manifest()


def test_atomic_strict_json_hash_and_orphan_resume(tmp_path, small):
    model = learning.Learner(small)
    payload = dict(learner=model.state())
    rt.checkpoint(tmp_path, payload)
    rt.save(tmp_path / "finite.json", {"good": 1.})
    before = rt.sha(tmp_path / "finite.json")
    with pytest.raises(ValueError):
        rt.save(tmp_path / "finite.json", {"bad": float("nan")})
    assert rt.sha(tmp_path / "finite.json") == before
    (tmp_path / "checkpoints/orphan.tmp").write_bytes(b"partial")
    equal(rt.restore(tmp_path), payload)
    with pytest.raises(FileExistsError):
        rt.save_once(tmp_path / "finite.json", {"other": 2.})
    latest = rt.read(tmp_path / "latest.json")
    (tmp_path / latest["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        rt.restore(tmp_path)


def test_checkpoint_phase_and_frozen_weight_tampering(small):
    model = finish(learning.Learner(small))
    state = model.state()
    state["counts"]["actor"] += 1
    with pytest.raises(ValueError, match="limit"):
        learning.Learner(small).load(state)
    state = model.state()
    state["models"]["phi"]["0.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="Frozen Phi"):
        learning.Learner(small).load(state)


def settings():
    return dict(spec=copy.deepcopy(SPEC), sources=rt.sources(), data=dict(files={}, synthetic=True))


def test_runner_stop_resume_no_overwrite(small, tmp_path, monkeypatch):
    config = settings()
    original = learning.Learner.update
    stop = tmp_path / "STOP"
    def stopping(model):
        original(model)
        if model.counts["phi"] == 2:
            stop.touch()
    monkeypatch.setattr(learning.Learner, "update", stopping)
    assert runner.execute(tmp_path, small, config) == "stopped"
    assert rt.restore(tmp_path)["learner"]["counts"]["phi"] == 2
    assert not (tmp_path / "completion.json").exists()
    with pytest.raises(rt.Stopped):
        runner.execute(tmp_path, small, config, True)
    stop.unlink()
    monkeypatch.setattr(learning.Learner, "update", original)
    assert runner.execute(tmp_path, small, config, True) == "completed"
    final_hash = rt.sha(tmp_path / "model_final.pt")
    final = torch.load(tmp_path / "model_final.pt", weights_only=False)
    equal(final["learner"], finish(learning.Learner(small)).state())
    for resume in (False, True):
        with pytest.raises(FileExistsError):
            runner.execute(tmp_path, small, config, resume)
    assert rt.sha(tmp_path / "model_final.pt") == final_hash
    assert final["critic_continuation"] == "carry"


def test_final_publication_stop_resume_changed_process(small, tmp_path, monkeypatch):
    config = settings()
    completed = finish(learning.Learner(small)).state()
    identities = [dict(pid=pid, created=pid*100, parent_pid=1, executable="synthetic-test",
                       command=["synthetic-publication-resume"], session_id=f"publication-{pid}")
                  for pid in (101, 202, 303)]
    rt.save(tmp_path / "settings.json", config)
    original = rt.checkpoint(tmp_path, dict(settings=config, learner=completed,
        process=identities[0], critic_continuation="carry", candidate=True))
    pointer_hash = rt.sha(tmp_path / "latest.json")
    resumed_identities = iter(identities[1:])
    monkeypatch.setattr(runner, "process_identity", lambda: next(resumed_identities))
    updates = []

    def forbidden_update(model):
        updates.append(model.phase)
        raise AssertionError("Completed phases must not retrain during publication")

    monkeypatch.setattr(learning.Learner, "update", forbidden_update)
    stop = tmp_path / "STOP"
    link = runner.os.link

    def stop_after_link(source, candidate):
        link(source, candidate)
        stop.touch()

    monkeypatch.setattr(runner.os, "link", stop_after_link)
    assert runner.execute(tmp_path, small, config, True) == "stopped"
    assert not (tmp_path / "completion.json").exists()
    candidate_hash = rt.sha(tmp_path / "model_final.pt")
    evidence = dict(original_checkpoint=original,
        latest_after_stop=rt.read(tmp_path / "latest.json"), candidate_sha256=candidate_hash,
        process_identities=identities, counters=completed["counts"], resumed_optimizer_updates=len(updates))
    stop.unlink()
    try:
        outcome = runner.execute(tmp_path, small, config, True)
    except BaseException as exc:
        rt.save(tmp_path / "r1-publication.json", dict(evidence, second_resume=f"{type(exc).__name__}: {exc}"))
        raise
    rt.save(tmp_path / "r1-publication.json", dict(evidence, second_resume=outcome))
    assert outcome == "completed" and updates == []
    assert rt.sha(tmp_path / "latest.json") == pointer_hash
    assert rt.read(tmp_path / "latest.json") == original
    assert rt.sha(tmp_path / "model_final.pt") == candidate_hash == original["sha256"]
    assert list((tmp_path / "checkpoints").glob("*.pt")) == [tmp_path / original["path"]]
    restored = rt.restore(tmp_path)
    assert restored["process"] == identities[0]
    equal(restored["learner"], completed)
    done = rt.read(tmp_path / "completion.json")
    assert done["counters"] == completed["counts"]
    for name, expected in done["outputs_sha256"].items():
        assert rt.sha(tmp_path / name) == expected
    for identity, event in zip(identities[1:], ("stopped", "completed")):
        session = rt.read(tmp_path / "sessions" / (identity["session_id"] + ".end.json"))
        assert session["pid"] == identity["pid"] and session["created"] == identity["created"]
        assert session["event"] == event


def test_runner_gate_failed_no_candidate(small, tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "numerical_gate", lambda *a: dict(passed=False))
    assert runner.execute(tmp_path, small, settings()) == "numerical_fit_failed"
    assert not (tmp_path / "model_final.pt").exists()
    assert rt.read(tmp_path / "completion.json")["candidate_admitted"] is False


def test_stop_before_output_and_review_identity(tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    with pytest.raises(rt.Stopped):
        runner.execute(tmp_path, synthetic(), settings())
    assert {p.name for p in tmp_path.iterdir()} == {"STOP"}
    review = tmp_path / "review.json"
    source = rt.sources()
    rt.save(review, dict(status="approved", reviewer="independent-test-reviewer", source_sha256=source, spec_sha256=rt.digest(SPEC)))
    assert runner.review_receipt(review, source)["sha256"] == rt.sha(review)
    with pytest.raises(ValueError, match="Independent review"):
        runner.review_receipt(review, {})


def test_partial_run_and_source_mutation_refused(tmp_path, small):
    rt.save(tmp_path / "partial.json", {})
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, small, settings())
    with pytest.raises(ValueError, match="No frozen settings"):
        runner.execute(tmp_path, small, settings(), True)
    config = settings()
    config["sources"][next(iter(config["sources"]))] = "0"*64
    with pytest.raises(ValueError, match="changed"):
        runner.execute(tmp_path / "new", small, config)


def test_no_environment_imports_in_learner_process():
    assert "budget_env" not in sys.modules
    assert "collect" not in sys.modules
    assert "local_runtime" not in sys.modules
    assert not any(k == "src" or k.startswith("src.") for k in sys.modules)


def test_single_numerical_thread():
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
