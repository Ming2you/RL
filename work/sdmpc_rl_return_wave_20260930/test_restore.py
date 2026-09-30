"""Pinned real restore/reference bodies, with strictly in-memory lower spies."""
import builtins
import copy
from dataclasses import dataclass, field
from types import SimpleNamespace as NS
import pytest
import wave_support as w
import restore_accounting as accounting
from budget_env import Observation


@dataclass
class SpyState:
    time_sec: float
    urban_arrival_buffer: dict = field(default_factory=dict)
    urban_storage_release_buffer: dict = field(default_factory=dict)
    ramp_queue: dict = field(default_factory=lambda: {"r": 3.})


def plain_namespace(value):
    if isinstance(value, NS):
        return {key: plain_namespace(v) for key, v in vars(value).items()}
    return w.plain(value)


def frozen_case(k, source="previous", raise_at=None):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("FORBIDDEN reset/step/PFO/solve")
        raise AssertionError(calls[-1])
    def imported(name, *args, **kwargs):
        if name == "fixed_policy":
            return NS(mask=lambda: w.np.ones((2, 2), dtype=bool))
        return builtins.__import__(name, *args, **kwargs)
    # Extract unchanged class definitions only; imports and lower physics never run.
    namespace = dict(np=w.np, copy=copy, plain=w.plain, Observation=Observation,
        SimpleNamespace=NS, __builtins__=dict(vars(builtins), __import__=imported))
    w.definitions(w.OLD / "budget_controller.py", {"BudgetController", "InvalidReference"},
                  namespace, accounting.PINS["budget_controller.py"])
    w.definitions(w.OLD / "budget_env.py", {"BudgetEnv"},
                  namespace, accounting.PINS["budget_env.py"])
    cfg = NS(network=NS(urban_link_storage_veh={"u": 50.}, urban_movements={},
        boundary_queue_max_veh=100., urban_avg_vehicle_length_m=5., urban_avg_speed_km_h=30.,
        freeway_lanes=2, freeway_links=["a"], freeway_segments_per_link=1,
        signals=["A"], cycle_length=120.), simulation=NS(T_u_h=5./3600., T_u_sec=5.))
    previous = NS(point=w.np.array([.25, .75]), offsets={"A": 10.})
    control = NS(point=w.np.array([.5, .75]), offsets={"A": 10.})
    budget = w.np.array([-5., 4000.])
    last_budget = None if k == 5 else w.np.array([-10., 3500.])
    dual = w.np.array([[1., 2.], [3., 4.]])
    class Coordinates:
        lower, upper = w.np.zeros(2), w.np.ones(2)
        def __init__(self, *args):
            pass
        def encode(self, value):
            return value.point.copy()
        def decode(self, point):
            return NS(point=point.copy(), offsets={"A": 10.})
        def quantize(self, point):
            return point.copy()
    class Lower:
        def __init__(self, cfg, options):
            self.cfg, self.options = cfg, options
            self.dual, self.last_budget = w.np.zeros((2, 2)), None
        def begin(self, state, forecast, prior):
            calls.append("lower.begin")
            assert state.time_sec == k*180.
            w.np.testing.assert_array_equal(prior.point, previous.point)
            self.coords = Coordinates()
        def evaluate(self, seed):
            calls.append("lower.evaluate")
            if raise_at == "evaluate":
                raise RuntimeError("spy interrupted")
            w.np.testing.assert_array_equal(seed, control.point)
            return NS(budget_vector=budget.copy(), total_ttt=12.)
        def execution_check(self, seed, value):
            calls.append("lower.execution_check")
            w.np.testing.assert_array_equal(value, budget)
            return dict(physical_control_valid=True, budget_feasible=True)
        solve = staticmethod(forbidden)
    rt = dict(cfg=cfg, options={}, baseline=Lower, coordinates=Coordinates,
              rc=NS(to_plain_dict=plain_namespace), snapshot_identity={"synthetic": True})
    env = namespace["BudgetEnv"].__new__(namespace["BudgetEnv"])
    env.rt, env.cfg, env.observer = rt, cfg, Observation(cfg)
    env.reward_scale, env.guard_mode, env.profile_hash, env.warm = 100., "physical", "spy-profile", None
    env.reset = env.step = forbidden
    reference = dict(budget=budget.copy(), point=control.point.copy(), control=copy.deepcopy(control),
        evaluation=NS(total_ttt=12.), source=source,
        action_anchor=budget.copy() if last_budget is None else last_budget.copy())
    ck = dict(k=k, profile_hash=env.profile_hash, sim=NS(cfg=cfg, state=SpyState(k*180.),
        total_ttt=22., step=forbidden), warm=NS(memory={"retained": [7, 9]}, solve=forbidden),
        previous=previous, forecast=[NS(freeway_mainline={"a": 1000.}, urban_boundary={"u": 200.},
        ramp_arrival={"r": 100.}, incident_capacity_factor=1., freeway_lane_loss={})],
        last_requested=w.np.array([-10., 3500.]), last_executed=w.np.array([-10., 3500.]),
        last_slack=w.np.array([0., 1.]), last_fallback=False, warmup_ttt=5.,
        dual=dual, last_budget=last_budget, prepared=reference if k < 80 else None,
        reference_timing=dict(reference_wall_seconds=7., reference_cpu_seconds=6., pfo_calls=int(k == 5)),
        observation_timing=dict(observation_wall_seconds=.5, observation_cpu_seconds=.25))
    probe = NS(**ck, controller=NS(incoming=dual, lower=NS(coords=Coordinates()),
        reference=reference, action_anchor=reference["action_anchor"]))
    expected = env.observer.encode(probe)
    ck["contract"], ck["observer"] = env.contract(), copy.deepcopy(env.observer)
    env.observer = Observation(cfg)
    return NS(env=env, checkpoint=ck, calls=calls,
              observation=expected if k < 80 else w.np.zeros_like(expected))


def ledger(tmp_path, k, resume=True):
    output, sid = tmp_path / "slot", "a"*32
    settings = dict(run_id="spy-run", sources=w.sources(), scenario=w.SCENARIOS[0])
    start = dict(session_id=sid, run_id=settings["run_id"], sources=settings["sources"],
        model_sha256=w.MODEL_SHA, scenario=settings["scenario"], lock_held=True,
        process=dict(pid=1, created=1, command=["spy"]), resume_requested=resume,
        settings_digest=w.digest(settings) if resume else None)
    w.save(output / "sessions" / f"{sid}.start.json", start)
    checkpoint_path = output / "checkpoints/spy.pt"
    w.save(checkpoint_path, {"synthetic_only": True, "k": k})
    record = dict(path="checkpoints/spy.pt", sha256=w.file_hash(checkpoint_path),
        settings_digest=w.digest(settings), control_steps=k-5, session_ids=[])
    return NS(output=output, sid=sid, settings=settings, record=record, start=start)


def finish_session(data):
    w.save(data.output / "sessions" / f"{data.sid}.end.json", dict(session_id=data.sid,
        start_sha256=w.file_hash(data.output / "sessions" / f"{data.sid}.start.json"),
        process=data.start["process"], elapsed_wall_seconds=100.,
        restoration_sha256=accounting.files(data.output, data.sid)))


def measured(case, data, monkeypatch):
    wall, cpu = iter([10., 13.]), iter([20., 22.])
    monkeypatch.setattr(accounting, "time", NS(perf_counter=lambda: next(wall), process_time=lambda: next(cpu)))
    return accounting.restore_environment(case.env, case.checkpoint, data.output, data.sid,
                                           data.settings, data.record)


@pytest.mark.parametrize("k,source", [(5, "pfo_initial"), (6, "previous"),
    (6, "pfo_recovery"), (79, "previous"), (80, "previous")])
def test_real_frozen_restore_boundary_and_cost(tmp_path, monkeypatch, k, source):
    case, data = frozen_case(k, source), ledger(tmp_path, k)
    ck, env = case.checkpoint, case.env
    before = copy.deepcopy(ck)
    obs = measured(case, data, monkeypatch)
    assert case.calls == (["lower.begin", "lower.evaluate", "lower.execution_check"] if k < 80 else [])
    w.np.testing.assert_array_equal(obs, case.observation)
    assert obs.dtype == w.np.float32 and env.observer.contract() == ck["observer"].contract()
    assert env.observer is not ck["observer"] and env.sim is not ck["sim"]
    assert env.k == k and env.sim.state == ck["sim"].state and env.sim.total_ttt == 22.
    assert env.warm.memory == {"retained": [7, 9]} and env.warm is not ck["warm"]
    w.np.testing.assert_array_equal(env.controller.lower.dual, ck["dual"])
    w.np.testing.assert_array_equal(env.controller.lower.last_budget, ck["last_budget"])
    w.np.testing.assert_array_equal(env.previous.point, before["previous"].point)
    for name in ("last_requested", "last_executed", "last_slack"):
        w.np.testing.assert_array_equal(getattr(env, name), before[name])
    assert env.controller.prepared is (k < 80)
    if k < 80:
        ref = env.controller.reference
        assert ref["source"] == source and ref["evaluation"].total_ttt == 12.
        for name in ("budget", "point", "action_anchor"):
            w.np.testing.assert_array_equal(ref[name], before["prepared"][name])
        w.np.testing.assert_array_equal(ref["control"].point, ck["prepared"]["control"].point)
        w.np.testing.assert_array_equal(env.controller.incoming, ck["dual"])
    else:
        assert env.controller.action_anchor is None and ck["prepared"] is None
    for name in accounting.TIMINGS:
        assert getattr(env, name) == ck[name] == before[name]
        assert getattr(env, name) is not ck[name]
    finish_session(data)
    timing = w.session_timing(data.output, data.settings)
    restore = timing["restoration"]
    assert timing["elapsed_wall_seconds"] == 100.  # Not 103: restore is already enclosed.
    assert restore["elapsed_wall_seconds"] == 3. and restore["elapsed_cpu_seconds"] == 2.
    assert restore["returned_branch_reference_reconstructions"] == int(k < 80)
    row = restore["records"][0]
    assert row["expected_reference_reconstructions"] == int(k < 80)
    assert row["provenance"]["checkpoint"] == data.record
    assert row["provenance"]["inherited_decision_timing"] == row["inherited_decision_timing_after"]
    assert row["provenance"]["frozen_source_sha256"] == accounting.PINS
    assert len(row["records_sha256"]) == 2
    assert "Not an isolated reference-only duration" in restore["scope"]


def test_real_frozen_raised_restore_measured_but_count_unknown(tmp_path, monkeypatch):
    case, data = frozen_case(6, raise_at="evaluate"), ledger(tmp_path, 6)
    with pytest.raises(RuntimeError, match="spy interrupted"):
        measured(case, data, monkeypatch)
    assert case.calls == ["lower.begin", "lower.evaluate"]
    finish_session(data)
    restore = w.session_timing(data.output, data.settings)["restoration"]
    assert restore["elapsed_wall_seconds"] == 3. and restore["elapsed_cpu_seconds"] == 2.
    assert restore["reference_reconstruction_status"] == "UNKNOWN"
    assert restore["returned_branch_reference_reconstructions"] is None
    assert restore["records"][0]["outcome"] == "raised"
    assert restore["records"][0]["inherited_decision_timing_after"] is None


@pytest.mark.parametrize("k", [6, 80])
def test_interrupted_restore_publication_retains_unknown(tmp_path, monkeypatch, k):
    case, data = frozen_case(k), ledger(tmp_path, k)
    original = w.save_once
    def interrupted(path, value):
        if path.name.endswith(".restore-end.json"):
            raise KeyboardInterrupt("synthetic lost end")
        original(path, value)
    monkeypatch.setattr(w, "save_once", interrupted)
    with pytest.raises(KeyboardInterrupt):
        measured(case, data, monkeypatch)
    timing = w.session_timing(data.output, data.settings)
    assert timing["timing_status"] == "UNKNOWN" and timing["elapsed_wall_seconds"] is None
    restore = timing["restoration"]
    assert restore["timing_status"] == restore["reference_reconstruction_status"] == "UNKNOWN"
    assert restore["elapsed_wall_seconds"] is restore["elapsed_cpu_seconds"] is None
    assert restore["known_elapsed_wall_seconds"] == 0.
    assert restore["returned_branch_reference_reconstructions"] is None
    assert restore["records"][0]["expected_reference_reconstructions"] == int(k < 80)
    # Even a retained enclosing end cannot invent a lost restore measurement.
    finish_session(data)
    assert w.session_timing(data.output, data.settings)["restoration"]["timing_status"] == "UNKNOWN"


@pytest.mark.parametrize("resume,ended,status", [(False, False, "KNOWN"),
    (False, True, "KNOWN"), (True, False, "UNKNOWN"), (True, True, "KNOWN")])
def test_restore_not_started_or_uncertain(tmp_path, resume, ended, status):
    data = ledger(tmp_path, 6, resume)
    if ended:
        finish_session(data)
    timing = w.session_timing(data.output, data.settings)
    restore = timing["restoration"]
    assert restore["timing_status"] == status
    assert restore["elapsed_wall_seconds"] == (None if status == "UNKNOWN" else 0.)
    assert timing["elapsed_wall_seconds"] == (100. if ended else None)


@pytest.mark.parametrize("bad", ["session_hash", "checkpoint_bytes", "checkpoint_escape", "branch", "source",
    "settings", "session", "restore_start_hash", "inherited", "negative", "nonfinite", "enclosing"])
def test_restore_provenance_rejects_tamper(tmp_path, monkeypatch, bad):
    case, data = frozen_case(6), ledger(tmp_path, 6)
    measured(case, data, monkeypatch)
    finish_session(data)
    start = data.output / "sessions" / f"{data.sid}.restore-start.json"
    end = data.output / "sessions" / f"{data.sid}.restore-end.json"
    saved, finish = w.read(start), w.read(end)
    if bad == "checkpoint_bytes":
        w.save(data.output / data.record["path"], {"changed": True})
    elif bad == "checkpoint_escape":
        saved["checkpoint"]["path"] = "../escaped.pt"
    elif bad == "branch":
        saved["expected_reference_reconstructions"] = 0
    elif bad == "source":
        saved["frozen_source_sha256"]["budget_env.py"] = "wrong"
    elif bad == "settings":
        saved["settings_digest"] = "wrong"
    elif bad == "session":
        saved["session_id"] = "b"*32
    elif bad == "inherited":
        finish["inherited_decision_timing_after"]["reference_timing"]["reference_wall_seconds"] += 3.
    elif bad in ("negative", "nonfinite", "enclosing"):
        finish["elapsed_wall_seconds"] = {"negative": -1., "nonfinite": "NaN", "enclosing": 101.}[bad]
    else:
        finish["elapsed_cpu_seconds"] += 1.
    w.save(start, saved)
    finish["restore_start_sha256"] = "wrong" if bad == "restore_start_hash" else w.file_hash(start)
    w.save(end, finish)
    if bad != "session_hash":
        finish_session(data)  # Exercise inner semantic validation beyond the outer hash binding.
    with pytest.raises(ValueError):
        w.session_timing(data.output, data.settings)


def test_changed_inherited_timing_fails_without_rewriting_checkpoint(tmp_path, monkeypatch):
    case, data = frozen_case(6), ledger(tmp_path, 6)
    original = case.env.restore
    def changed(ck):
        obs = original(ck)
        case.env.reference_timing["reference_wall_seconds"] += 3.
        return obs
    case.env.restore = changed
    with pytest.raises(ValueError, match="inherited decision timing"):
        measured(case, data, monkeypatch)
    assert case.checkpoint["reference_timing"]["reference_wall_seconds"] == 7.
    finish_session(data)
    with pytest.raises(ValueError, match="inherited decision timing"):
        w.session_timing(data.output, data.settings)
