"""Task 3 only: all optimizer tests use independently generated synthetic arrays."""
import copy
from pathlib import Path
import random
import sys
import uuid
import pytest
import mc_common as c
import mc_data as d
import mc_learner as m
import run


@pytest.fixture
def folder():
    path = c.HERE / "synthetic-fixtures" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture(scope="session")
def synthetic():
    rng = c.np.random.Generator(c.np.random.PCG64(9901))
    obs = c.torch.from_numpy(rng.normal(size=(375, 6)).astype(c.np.float32))
    with c.torch.random.fork_rng(devices=[]):
        c.torch.manual_seed(9902)
        actor, phi = c.Actor(6), c.mlp(6, 1)
        critics = c.torch.nn.ModuleList([c.mlp(8, 1) for _ in range(2)])
    with c.torch.no_grad():
        actor.net[-1].bias.copy_(c.torch.tensor([.05, .02]))
    optimizer = c.torch.optim.Adam(critics.parameters(), lr=3e-4)
    sum(p.square().sum() for p in critics.parameters()).backward()
    optimizer.step()
    # Synthetic ancestor fixture has nonzero matching moments and a declared 250-step counter.
    for state in optimizer.state.values():
        state["step"].fill_(250)
    targets = copy.deepcopy(critics)
    with c.torch.no_grad():
        action = actor(obs)
    anchor = c.torch.from_numpy(rng.normal(size=(375, 2))).double()
    anchor[:, 0] -= 100.
    anchor[:, 1] = 6000.
    reward = -c.torch.from_numpy(rng.uniform(.1, .8, 375))
    returns = reward.clone().reshape(5, 75)
    returns = returns.flip(1).cumsum(1).flip(1).flatten()
    data = dict(obs=obs, action=action, reward=reward, returns=returns, anchor=anchor,
        request=c.projected(anchor, action), scenario=c.torch.arange(5).repeat_interleave(75),
        terminal=c.torch.tensor([False]*74+[True]).repeat(5), horizon=c.torch.arange(75, 0, -1).repeat(5))
    parent = dict(spec=c.rt.SPEC, phase="done", counts=c.ANCESTOR_COUNTS, critic_continuation="carry",
        models={k: v.state_dict() for k, v in dict(actor=actor, phi=phi, critics=critics, targets=targets).items()},
        optimizers=dict(critics=optimizer.state_dict()))
    return data, copy.deepcopy(parent)


@pytest.fixture
def learner(synthetic):
    return m.Learner(*synthetic)


@pytest.fixture(scope="session")
def completed(synthetic):
    learner = m.Learner(*synthetic)
    for _ in range(250):
        learner.update()
    return learner.state()


def settings():
    return dict(format=c.SPEC["format"], spec=c.SPEC, sources={}, data=dict(files={}), synthetic=True)


def seed_checkpoint(folder, learner, state=None):
    if state is not None:
        learner.load(state)
    c.save(folder / "settings.json", settings())
    return c.checkpoint(folder, dict(format=c.SPEC["format"], settings=settings(), learner=learner.state(),
        process=dict(synthetic_old_process=1), target_policy_parent_model_sha256=c.MODEL_SHA))


def test_pure_import_boundary():
    assert not {"runtime", "learner", "data", "wave_support", "worker", "budget_env", "src"}.intersection(sys.modules)
    assert c.torch.get_num_threads() == c.torch.get_num_interop_threads() == 1


def test_exact_parent_and_continued_adam(learner, synthetic):
    parent = synthetic[1]
    for key, model in learner.models().items():
        assert c.tensor_hash(model.state_dict()) == c.tensor_hash(parent["models"][key])
    assert m.state_hash(learner.optimizer.state_dict()) == m.state_hash(parent["optimizers"]["critics"])
    assert learner.optimizer_provenance["continued"]
    assert learner.optimizer.param_groups[0]["lr"] == 3e-4
    learner.update()
    assert {float(x["step"]) for x in learner.optimizer.state.values()} == {251.}
    assert "carry" in learner.before["interpretation"] and "not independent" in learner.before["interpretation"]


def test_only_critics_receive_gradients(learner):
    before = learner.initial_hashes.copy()
    learner.update()
    assert all(p.grad is not None for p in learner.critics.parameters())
    assert all(p.grad is None and not p.requires_grad for model in
        (learner.actor, learner.phi, learner.targets) for p in model.parameters())
    assert c.tensor_hash(learner.critics.state_dict()) != before["critics"]
    for name in ("actor", "phi", "targets"):
        assert c.tensor_hash(learner.models()[name].state_dict()) == before[name]


def test_does_not_consume_global_rng(synthetic):
    torch_before, numpy_before, python_before = c.torch.get_rng_state().clone(), c.np.random.get_state(), random.getstate()
    learner = m.Learner(*synthetic)
    learner.update()
    assert c.torch.equal(torch_before, c.torch.get_rng_state())
    assert m.state_hash(numpy_before) == m.state_hash(c.np.random.get_state())
    assert python_before == random.getstate()


def test_mc_targets_and_sum_of_head_mse_no_bootstrap(learner, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No target network/bootstrap call is allowed")
    monkeypatch.setattr(learner.targets, "forward", forbidden)
    for head in learner.targets:
        monkeypatch.setattr(head, "forward", forbidden)
    assert c.torch.equal(learner.mc_targets, learner.data["returns"]-learner.phi_values)
    rng = c.np.random.Generator(c.np.random.PCG64(7201))
    indices, _ = d.sample_indices(learner.data, rng)
    errors = c.residuals(learner.critics, learner.data["obs"][indices], learner.budgets[indices]).double() - learner.mc_targets[indices, None]
    expected = float(errors.square().mean(dim=0).sum().detach())
    learner.update()
    assert learner.losses[0]["loss"] == expected


def test_fixed_quotas_and_final_copy(completed, learner):
    learner.load(completed)
    assert learner.counts == dict(mc_critic=250)
    assert learner.sample_counts == [2000]*5
    assert learner.forced_terminal_counts == [250]*5
    assert all(n > 250 for n in learner.terminal_counts)
    assert c.tensor_hash(learner.targets.state_dict()) == c.tensor_hash(learner.critics.state_dict())
    assert all(x["samples_per_scenario"] == [8]*5 and x["forced_terminals_per_scenario"] == [1]*5 for x in learner.losses)
    assert all({idx for row in learner.losses for idx in row["indices"] if idx//75 == i} == set(range(i*75, (i+1)*75)) for i in range(5))
    with pytest.raises(ValueError, match="250"):
        learner.update()
    metrics = m.final_metrics(learner)
    assert len(metrics["after"]["rows"]) == 375
    assert len(metrics["after"]["per_scenario_horizon"]) == 25
    assert "TRAINING" in metrics["after"]["interpretation"]
    assert metrics["after"]["coverage"]["nuf_cap_count"] == 375
    assert metrics["after"]["coverage"]["distinct_nuf_requests"] == 1
    assert metrics["traffic_improvement_claim"] is False


def test_deterministic_resume_including_optimizer_rng_and_metrics(synthetic, completed):
    first = m.Learner(*synthetic)
    for _ in range(73):
        first.update()
    resumed = m.Learner(*synthetic)
    resumed.load(first.state())
    for _ in range(177):
        resumed.update()
    assert m.state_hash(resumed.state()) == m.state_hash(completed)


def test_float64_projection_signed_np_and_caps():
    anchor = c.torch.tensor([[-123.123456789012, 5999.9999999], [-1.e8+.123456789, -4.]], dtype=c.torch.float64)
    action = c.torch.tensor([[.123456789, .1], [-.2, -.1]], dtype=c.torch.float32)
    expected = anchor.numpy() + action.numpy().astype(c.np.float64)*[50., 1000.]
    expected[:, 1] = c.np.clip(expected[:, 1], 0., 6000.)
    c.np.testing.assert_array_equal(c.projected(anchor, action).numpy(), expected)
    c.np.testing.assert_array_equal(c.network_budget(anchor, action).numpy(), (expected/[1000., 10000.]).astype(c.np.float32))
    assert bool((c.projected(anchor, action)[:, 0] < 0).all())
    with pytest.raises(ValueError, match="float64"):
        c.projected(anchor.float(), action)


def episode_fixture():
    settings = dict(evaluation=False, profile_sha256="synthetic", warmup_ttt=123456.)
    rows, transitions = [], []
    for k in range(75):
        obs = c.np.full(2367, k, dtype=c.np.float32)
        nxt = c.np.full(2367, k+1, dtype=c.np.float32) if k < 74 else c.np.zeros(2367, dtype=c.np.float32)
        action = c.np.zeros(2, dtype=c.np.float32)
        reward = -(k+1)/100.
        transitions.append((obs, action, reward, nxt, k == 74))
        rows.append(dict(terminated=k == 74, truncated=False, control_step=k, step=k+5,
            scenario=c.SCENARIOS[0], behavior="frozen_shared_actor", profile_sha256="synthetic",
            action_anchor=[-123.123456789, 6000.], B_executed=[-123.123456789, 6000.],
            B_requested=[[-123.123456789, 6000.]], action_requested=[0., 0.], interval_ttt=k+1))
    return transitions, rows, settings


def test_returns_terminal_and_warmup_exclusion():
    transitions, trace, settings_ = episode_fixture()
    rows = c.episode_arrays(transitions, trace, settings_, c.SCENARIOS[0], "frozen_shared_actor")
    assert abs(rows[0]["returns"] + sum(range(1, 76))/100.) < 1e-12
    assert rows[-1]["returns"] == transitions[-1][2]
    assert rows[-1]["horizon"] == 1 and rows[0]["horizon"] == 75
    settings_["warmup_ttt"] *= 99
    again = c.episode_arrays(transitions, trace, settings_, c.SCENARIOS[0], "frozen_shared_actor")
    assert [x["returns"] for x in rows] == [x["returns"] for x in again]


@pytest.mark.parametrize("mutation", ["early_terminal", "truncated", "warmup_step", "reward", "next_state", "request", "evaluation"])
def test_invalid_episode_rejected(mutation):
    transitions, trace, settings_ = episode_fixture()
    if mutation == "early_terminal":
        transitions[3] = (*transitions[3][:4], True)
    elif mutation == "truncated":
        trace[74]["truncated"] = True
    elif mutation == "warmup_step":
        trace[0]["step"] = 0
    elif mutation == "reward":
        trace[3]["interval_ttt"] += 5
    elif mutation == "next_state":
        transitions[2][3][0] += 1
    elif mutation == "request":
        trace[3]["B_requested"][0][0] += 1
    else:
        settings_["evaluation"] = True
    with pytest.raises((ValueError, AssertionError)):
        c.episode_arrays(transitions, trace, settings_, c.SCENARIOS[0], "frozen_shared_actor")


@pytest.mark.parametrize("mutation", ["continuation", "count", "lr", "moment", "step"])
def test_parent_provenance_rejected(synthetic, mutation):
    data, parent = synthetic[0], copy.deepcopy(synthetic[1])
    opt = parent["optimizers"]["critics"]
    if mutation == "continuation":
        parent["critic_continuation"] = "pi1"
    elif mutation == "count":
        parent["counts"]["actor"] = 11
    elif mutation == "lr":
        opt["param_groups"][0]["lr"] = 1e-3
    elif mutation == "moment":
        opt["state"][0]["exp_avg"] = c.torch.zeros(1)
    else:
        opt["state"][0]["step"].fill_(0)
    with pytest.raises(ValueError):
        m.Learner(data, parent)


@pytest.mark.parametrize("mutation", ["rng", "data", "actor", "phi", "targets", "quota", "continuation", "optimizer", "nonfinite"])
def test_resume_provenance_rejected(learner, mutation):
    learner.update()
    state = learner.state()
    if mutation == "rng":
        state["numpy_rng"]["state"]["state"] += 1
    elif mutation == "data":
        state["data_arrays_sha256"] = "bad"
    elif mutation in ("actor", "phi", "targets"):
        next(iter(state["models"][mutation].values())).add_(1.)
    elif mutation == "quota":
        state["losses"][0]["indices"][0] = 0
    elif mutation == "continuation":
        state["critic_continuation"] = "carry"
    elif mutation == "optimizer":
        state["critic_optimizer"]["state"][0]["step"].fill_(250)
    else:
        state["models"]["critics"]["0.0.weight"].fill_(float("nan"))
    with pytest.raises(ValueError):
        learner.load(state)


@pytest.mark.parametrize("label", ["source.py", "model.pt", "experience.pt"])
def test_source_model_data_hash_failures(folder, label):
    path = folder / label
    path.write_bytes(b"synthetic original")
    manifest = {path.relative_to(c.REPO).as_posix(): c.sha(path)}
    c.verify(manifest)
    path.write_bytes(b"synthetic changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        c.verify(manifest)


def test_authentication_reconciles_readout_not_flags(monkeypatch):
    baseline = dict(integrity_pass=True, all_health_gates_pass=True, scenarios=[dict(ttt=10)])
    monkeypatch.setattr(d, "read", lambda path: baseline)
    receipt = dict(status="authenticated", manifest={}, readout=copy.deepcopy(baseline),
        raw_environment_loads=0, environment_calls=0)
    d.validate_receipt(receipt, {})
    receipt["readout"]["scenarios"][0]["ttt"] = 20
    with pytest.raises(ValueError, match="readout"):
        d.validate_receipt(receipt, {})


def test_review_receipt_and_fixed_output(folder):
    path = folder / "review.json"
    c.save(path, dict(status="approved", reviewer="synthetic", source_sha256={}, spec_sha256=c.digest(c.SPEC)))
    run.review_receipt(path, {})
    with pytest.raises(ValueError, match="review"):
        run.review_receipt(path, {"new": "source"})
    with pytest.raises(ValueError, match="fixed"):
        run.main(["--review", str(path), "--output", str(folder)])


@pytest.mark.parametrize("location", ["repo", "goal", "output"])
def test_stop_locations_without_real_markers(folder, monkeypatch, location):
    repo, goal, output = [folder / name for name in ("repo", "goal", "output")]
    for path in (repo, goal, output):
        path.mkdir()
    monkeypatch.setattr(c.rt, "REPO", repo)
    monkeypatch.setattr(c.rt, "GOAL", goal)
    dict(repo=repo, goal=goal, output=output)[location].joinpath("STOP").write_text("synthetic")
    with pytest.raises(c.Stopped):
        c.check_stop(output)


def test_kernel_lock_exclusion(folder):
    with c.locked(folder):
        with pytest.raises(OSError):
            with c.locked(folder):
                pass


def test_nonfinite_json_rejected(folder):
    with pytest.raises(ValueError, match="Nonfinite"):
        c.save(folder / "bad.json", dict(value=float("inf")))
    assert not (folder / "bad.json").exists()


def test_atomic_json_keeps_prior_orphan(folder, learner):
    orphan = folder / "latest.json.tmp"
    orphan.write_bytes(b"interrupted previous write")
    seed_checkpoint(folder, learner)
    assert orphan.read_bytes() == b"interrupted previous write"
    c.restore(folder)


@pytest.mark.parametrize("boundary", ["metrics", "final_model"])
def test_final_publication_stop_resume_changed_process(folder, synthetic, learner, completed, monkeypatch, boundary):
    record = seed_checkpoint(folder, learner, completed)
    original_stop = run.check_stop
    intercepted = False

    def stop_once(output):
        nonlocal intercepted
        reached = (output / ("metrics.json" if boundary == "metrics" else "model_final.pt")).exists()
        if reached and not intercepted:
            intercepted = True
            raise c.Stopped("synthetic finalization STOP")
        original_stop(output)

    monkeypatch.setattr(run, "check_stop", stop_once)
    if boundary == "metrics":
        original_metrics = run.final_metrics
        def stop_after_metrics(value):
            result = original_metrics(value)
            c.save_once(folder / "metrics.json", result)
            stop_once(folder)
            return result
        monkeypatch.setattr(run, "final_metrics", stop_after_metrics)
    assert run.execute(folder, *synthetic, settings(), resume=True) == "stopped"
    assert c.read(folder / "latest.json") == record
    assert not (folder / "completion.json").exists()
    monkeypatch.setattr(run, "check_stop", original_stop)
    if boundary == "metrics":
        monkeypatch.setattr(run, "final_metrics", original_metrics)
    assert run.execute(folder, *synthetic, settings(), resume=True) == "completed"
    assert c.read(folder / "latest.json") == record
    done = c.read(folder / "completion.json")
    for name, expected in done["outputs_sha256"].items():
        assert c.sha(folder / name) == expected
    assert c.sha(folder / "model_final.pt") == record["sha256"]
    assert len(list((folder / "sessions").glob("*.start.json"))) == 2
    with pytest.raises(FileExistsError, match="Completed"):
        run.execute(folder, *synthetic, settings(), resume=True)


def test_runner_stop_midfit_durable_resume_and_orphan(folder, synthetic, completed, monkeypatch):
    original = run.checkpoint
    orphan = folder / "orphan.tmp"
    orphan.write_bytes(b"preserve synthetic evidence")
    c.save(folder / "settings.json", settings())
    def interrupted(output, payload):
        record = original(output, payload)
        if payload["learner"]["counts"]["mc_critic"] == 7:
            raise c.Stopped("after durable update 7")
        return record
    monkeypatch.setattr(run, "checkpoint", interrupted)
    assert run.execute(folder, *synthetic, settings(), resume=True) == "stopped"
    assert c.restore(folder)["learner"]["counts"] == dict(mc_critic=7)
    monkeypatch.setattr(run, "checkpoint", original)
    assert run.execute(folder, *synthetic, settings(), resume=True) == "completed"
    assert m.state_hash(c.restore(folder)["learner"]) == m.state_hash(completed)
    assert orphan.read_bytes() == b"preserve synthetic evidence"


def test_checkpoint_hash_and_settings_mismatch(folder, learner, synthetic):
    record = seed_checkpoint(folder, learner)
    mismatched = dict(settings(), changed=True)
    with pytest.raises(ValueError, match="settings"):
        run.execute(folder, *synthetic, mismatched, resume=True)
    path = folder / record["path"]
    path.write_bytes(b"synthetic corrupt checkpoint")
    with pytest.raises(ValueError, match="hash"):
        c.restore(folder)


def test_partial_and_mismatched_final_refusal(folder, learner, synthetic, completed):
    seed_checkpoint(folder, learner, completed)
    with pytest.raises(FileExistsError, match="Partial"):
        run.execute(folder, *synthetic, settings())
    path = folder / "model_final.pt"
    path.write_bytes(b"synthetic unmatched orphan")
    with pytest.raises(FileExistsError, match="final model"):
        run.execute(folder, *synthetic, settings(), resume=True)
    assert path.read_bytes() == b"synthetic unmatched orphan"
    assert not (folder / "completion.json").exists()
