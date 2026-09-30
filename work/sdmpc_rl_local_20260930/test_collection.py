"""Synthetic orchestration tests; no physical simulator or historical data loaded."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
from dataclasses import dataclass, field, asdict, make_dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_runtime as lr
import collect
import validate
import run_wave
from exploration import LocalBudgetPolicy

SCHEMA = {"names": ["state/time_sec", "memory/remaining/0"] +
    [f"memory/{name}/{i}" for name in ("action_anchor", "previous_requested", "previous_executed") for i in range(2)]}
CONFIG = dict(network=dict(total_ramp_capacity=6000., freeway_links=["a"],
    freeway_segments_per_link=1, freeway_segment_length_km=.5, freeway_lanes=2,
    freeway_buffer_segments=1, off_ramp_storage_link={"r": "off"},
    urban_link_storage_veh={"urban": 50., "off": 30.}))
CONTRACT = {"observation_schema": SCHEMA, "environment_contract": {"config_sha256": lr.digest(CONFIG)}}
IDENTITY = {"fixture": "immutable"}


@dataclass
class FakeState:
    time_sec: float = 900.
    freeway_density: dict = field(default_factory=lambda: {"a": [20.]})
    freeway_speed: dict = field(default_factory=lambda: {"a": [50.]})
    freeway_flow: dict = field(default_factory=lambda: {"a": [2000.]})
    freeway_effective_lanes: dict = field(default_factory=lambda: {"a": [2.]})
    ramp_queue: dict = field(default_factory=lambda: {"r": 3.})
    mainline_origin_queue: dict = field(default_factory=lambda: {"a": 4.})
    urban_queue: dict = field(default_factory=lambda: {"legacy": 999.})
    boundary_queue: dict = field(default_factory=lambda: {"legacy": 999.})
    urban_movement_queue: dict = field(default_factory=lambda: {"m": 5.})
    urban_link_storage: dict = field(default_factory=lambda: {"urban": 40., "off": 23.})
    freeway_buffer_up_density: dict = field(default_factory=lambda: {"a": [6.]})
    freeway_buffer_down_density: dict = field(default_factory=lambda: {"a": [8.]})
    urban_arrival_buffer: dict = field(default_factory=lambda: {"urban": {1: 999.}})
    urban_storage_release_buffer: dict = field(default_factory=lambda: {"urban": {1: 999.}})


class FakeEnv:
    def __init__(self, rt, **kwargs):
        self.profile_hash = "training-profile"
        self.observer = NS(names=SCHEMA["names"])
        self.close_count = 0

    def observe(self):
        if self.k == 80:
            return lr.np.zeros(len(SCHEMA["names"]), dtype=lr.np.float32)
        scale = lr.np.array([1000., 10000.])
        return lr.np.array([self.k*180/14400., (80-self.k)/75.,
            *(self.controller.action_anchor/scale), *(self.last_requested/scale),
            *(self.last_executed/scale)], dtype=lr.np.float32)

    def reset(self):
        self.k, self.warmup_ttt = 5, 5.
        self.sim = NS(total_ttt=5., freeway_ttt=5., urban_ttt=0., state=FakeState(), cfg=copy.deepcopy(CONFIG))
        self.controller = NS(action_anchor=lr.np.array([-10., 6000.]))
        self.last_requested = lr.np.zeros(2)
        self.last_executed = lr.np.zeros(2)
        self.previous = {k: {"a": 1.} for k in validate.CONTROL_FIELDS}
        return self.observe()

    def step(self, action, mode, actor_wall, actor_cpu_seconds):
        anchor = self.controller.action_anchor.copy()
        raw = anchor + lr.np.array([50., 1000.])*action
        request = raw.copy()
        request[1] = lr.np.clip(request[1], 0., 6000.)
        i = self.k-5
        self.k += 1
        self.sim.total_ttt += 1.
        self.sim.freeway_ttt += 1.
        self.sim.state.time_sec += 180.
        self.controller.action_anchor = request.copy()
        self.last_requested, self.last_executed = request.copy(), request.copy()
        row = dict(action_anchor=anchor, action_requested=action, B_requested=[request], B_executed=request,
                   B_raw=raw, selection_source="lower_solution", reference_source="pfo_initial" if i == 0 else "previous",
                   reference_recovery=False, guard_mode="physical", h3_guard_enabled=False,
                   execution_check=dict(physical_control_valid=True, budget_feasible=True),
                   lower_candidate_count=1, pfo_calls=int(i == 0),
                   step=i+5, control_step=i, time_sec=self.sim.state.time_sec, terminated=self.k == 80,
                   truncated=False, interval_ttt=1., total_ttt=self.sim.total_ttt,
                   freeway_ttt=self.sim.total_ttt, urban_ttt=0., inventory=63.,
                   plant_state=asdict(self.sim.state), control=copy.deepcopy(self.previous))
        for p in ("forecast", "observation", "pfo", "reference", "lower", "guard"):
            row[p+"_wall_seconds"] = row[p+"_cpu_seconds"] = .1
        row.update(actor_wall_seconds=actor_wall, actor_cpu_seconds=actor_cpu_seconds,
                   decision_wall_seconds=.6+actor_wall, decision_cpu_seconds=.6+actor_cpu_seconds)
        return self.observe(), -.01, self.k == 80, row

    def checkpoint(self):
        return dict(k=self.k, profile_hash=self.profile_hash, sim=copy.deepcopy(self.sim),
                    anchor=self.controller.action_anchor.copy(), warmup_ttt=self.warmup_ttt,
                    previous=copy.deepcopy(self.previous), last_requested=self.last_requested.copy(),
                    last_executed=self.last_executed.copy(),
                    last_budget=self.last_executed.copy() if self.k > 5 else None,
                    prepared=dict(budget=self.controller.action_anchor.copy(),
                                  action_anchor=self.controller.action_anchor.copy()) if self.k < 80 else None)

    def restore(self, ck):
        self.k, self.sim, self.warmup_ttt = ck["k"], copy.deepcopy(ck["sim"]), ck["warmup_ttt"]
        self.controller = NS(action_anchor=ck["anchor"].copy())
        self.previous = copy.deepcopy(ck["previous"])
        self.last_requested, self.last_executed = ck["last_requested"].copy(), ck["last_executed"].copy()
        return self.observe()

    def close(self):
        self.close_count += 1


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(lr, "COHORT_ROOT", tmp_path, raising=False)
    lr.torch.set_num_threads(1)
    for module in (collect, validate, run_wave):
        monkeypatch.setattr(module, "verify_identity", lambda x: None)
    monkeypatch.setattr(collect, "identity", lambda: copy.deepcopy(IDENTITY))
    monkeypatch.setattr(run_wave, "identity", lambda: copy.deepcopy(IDENTITY))
    monkeypatch.setattr(collect, "contract", lambda: copy.deepcopy(CONTRACT))
    monkeypatch.setattr(validate, "contract", lambda: copy.deepcopy(CONTRACT))
    monkeypatch.setattr(collect, "boot", lambda *x: dict(cfg=NS(network=NS(total_ramp_capacity=6000.)),
        rc=NS(to_plain_dict=lambda cfg: copy.deepcopy(CONFIG))))
    monkeypatch.setattr(collect, "verify_environment", lambda *a, **k: None)
    monkeypatch.setattr(collect, "BudgetEnv", FakeEnv)
    monkeypatch.setattr(collect, "observation_schema", lambda env: copy.deepcopy(SCHEMA))
    monkeypatch.setattr(validate, "queue_exposure", lambda x: None)
    return monkeypatch


def args(folder, behavior="local", resume=False, limit=None, scenario=None):
    scenario = scenario or lr.SCENARIOS[0]
    return NS(output=folder, behavior=behavior, scenario=scenario,
              cpu_mask=lr.MASKS[lr.SCENARIOS.index(scenario)], resume=resume, abort_file=None,
              max_new_steps=limit)


@pytest.mark.parametrize("behavior", lr.BEHAVIORS)
def test_actual_collector_checkpoint_resume_and_completed_loader(tmp_path, synthetic, behavior):
    folder = tmp_path / behavior / lr.SCENARIOS[0]
    assert collect.run(args(folder, behavior, limit=2)) == "checkpointed"
    assert not (folder / "completion.json").exists()
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    assert len(ck["transitions"]) == 2
    assert collect.run(args(folder, behavior, resume=True)) == "completed"
    result, settings, summary = validate.load_completed(folder, IDENTITY)
    assert summary["ttt"] == 80. and summary["control_steps"] == 75
    assert settings["evaluation"] is False and settings["model_sha256"] is None
    assert summary["pfo_calls"] == 1 and summary["lower_candidate_solves"] == 75
    assert summary["physical_control_unique"] == 1
    with pytest.raises(ValueError, match="Already completed"):
        collect.run(args(folder, behavior, resume=True))
    synthetic.setattr(lr, "COHORT_ROOT", tmp_path / "independent")
    uninterrupted = lr.COHORT_ROOT / behavior / lr.SCENARIOS[0]
    collect.run(args(uninterrupted, behavior))
    saved = lr.read(folder / "trace.json")
    direct = lr.read(uninterrupted / "trace.json")
    assert [r["exploration_audit"] for r in saved] == [r["exploration_audit"] for r in direct]


def test_resume_rejects_behavior_source_and_state_changes(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    with pytest.raises(ValueError, match="cohort"):
        collect.run(args(folder, behavior="carry", resume=True))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    ck["environment"]["sim"].total_ttt += 1
    with pytest.raises(ValueError, match="boundary"):
        collect.validate_checkpoint(ck, ck["settings"])


def test_stop_before_boot_and_checkpoint_boundary(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    folder.mkdir(parents=True)
    (folder / "STOP").touch()
    with pytest.raises(lr.Stopped):
        collect.run(args(folder))
    assert set(p.name for p in folder.iterdir()) == {"STOP"}
    (folder / "STOP").unlink()
    original = FakeEnv.step
    def stop_after_step(self, *a, **k):
        result = original(self, *a, **k)
        (folder / "STOP").touch()
        return result
    synthetic.setattr(FakeEnv, "step", stop_after_step)
    assert collect.run(args(folder)) == "stopped"
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    assert len(ck["trace"]) == 1
    assert not (folder / "completion.json").exists()


def test_aborted_attempt_and_source_drift_never_complete(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    def reject(_):
        raise ValueError("source drift")
    synthetic.setattr(collect, "verify_identity", reject)
    with pytest.raises(ValueError, match="source drift"):
        collect.run(args(folder))
    assert lr.read(folder / "status.json")["status"] == "failed"
    assert not (folder / "completion.json").exists()
    assert (folder / "checkpoint.pt").exists()


@pytest.mark.parametrize("field,value", [("terminated", True), ("lower_candidate_count", 2),
    ("h3_guard_enabled", True), ("pfo_calls", 2), ("reward", -4.), ("total_ttt", 6.1),
    ("profile_sha256", "wrong"), ("policy_q", [1., 2.]), ("decision_wall_seconds", 100.)])
def test_trace_contract_rejection(tmp_path, synthetic, field, value):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    ck["trace"][0][field] = value
    with pytest.raises((ValueError, AssertionError)):
        validate.validate_rows(ck["trace"], ck["transitions"], ck["settings"], SCHEMA, 6000., False)


def test_summary_excludes_nonphysical_budget_and_metadata_from_diversity(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=2))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    for i, row in enumerate(ck["trace"]):
        row["control"].update(N_P_star=i, diagnostics={"iteration": i}, infeasibility={"gap": i})
    assert validate.summarize(ck["trace"], ck["settings"])["physical_control_unique"] == 1


def test_exact_exploration_replay_mismatch(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    ck["trace"][0]["exploration_audit"]["fabricated"] = True
    with pytest.raises(ValueError, match="audit"):
        validate.replay_policy(ck["trace"], ck["settings"])


def test_manifest_detects_changed_output(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder))
    lr.save(folder / "summary.json", {"tampered": True})
    with pytest.raises(ValueError, match="Output changed"):
        validate.load_completed(folder, IDENTITY)


def test_paired_comparison_and_admission_screen(tmp_path, synthetic):
    for behavior in lr.BEHAVIORS:
        for scenario in lr.SCENARIOS:
            collect.run(args(tmp_path / behavior / scenario, behavior, scenario=scenario))
    comparison = validate.compare_pairs(tmp_path, IDENTITY)
    assert comparison["coverage_screen_pass"] and not comparison["goal_claim"]
    assert len(comparison["rows"]) == 5 and comparison["transitions_per_scenario"] == 150
    assert all(r["training_ttt_delta"] == 0 for r in comparison["rows"])


def test_wave_jobs_and_resume_command(tmp_path):
    jobs = run_wave.jobs_for(tmp_path)
    assert len(jobs) == 10
    assert [j["behavior"] for j in jobs] == ["carry"]*5+["local"]*5
    for i, (a, b) in enumerate(zip(jobs[:5], jobs[5:])):
        assert a["training_seed"] == b["training_seed"] == 6801+i
        assert a["exploration_seed"] == b["exploration_seed"] == 6901+i
        assert a["cpu_mask"] == b["cpu_mask"] == lr.MASKS[i]
    job = jobs[0]
    folder = tmp_path / job["key"]
    folder.mkdir(parents=True)
    assert "--resume" not in run_wave.command_for(tmp_path, job, tmp_path / "ABORT.json")
    (folder / "checkpoint.pt").touch()
    assert "--resume" in run_wave.command_for(tmp_path, job, tmp_path / "ABORT.json")


def test_existing_live_lock_prevents_any_dispatch(tmp_path, synthetic):
    output = tmp_path
    child = output / run_wave.jobs_for(output)[0]["key"]
    synthetic.setattr(run_wave.subprocess, "Popen", lambda *a, **k: pytest.fail("must not dispatch"))
    with lr.exclusive_run(child):
        with pytest.raises(OSError):
            run_wave.run(output, resume=True)


def test_wave_worker_failure_drains_other_children(tmp_path, synthetic):
    jobs = run_wave.jobs_for(tmp_path)[:2]
    for job in jobs:
        (tmp_path / job["key"]).mkdir(parents=True)
    children = []
    class Child:
        def __init__(self):
            self.pid = len(children)+100
            self.waited = False
        def poll(self):
            return 1 if self.pid == 100 else None
        def wait(self):
            self.waited = True
            return 2
    def launch(*a, **k):
        value = Child()
        children.append(value)
        return value
    synthetic.setattr(run_wave.subprocess, "Popen", launch)
    abort = tmp_path / "attempt/ABORT.json"
    with pytest.raises(RuntimeError, match="Child failed"):
        run_wave.run_stage(tmp_path, jobs, IDENTITY, abort, jobs)
    assert abort.exists() and children[1].waited
    assert not (tmp_path / "STOP").exists()


def test_wave_all_completed_skip_and_plan_changes(tmp_path, synthetic):
    output = tmp_path
    for j in run_wave.jobs_for(output):
        folder = output / j["key"]
        folder.mkdir(parents=True)
        lr.save(folder / "completion.json", {"done": True})
    synthetic.setattr(run_wave, "validate_job", lambda *x: {})
    synthetic.setattr(run_wave, "compare_pairs", lambda *x: {"coverage_screen_pass": False})
    synthetic.setattr(run_wave.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate"))
    assert run_wave.run(output, resume=True) == "completed"
    assert lr.read(output / "completion.json")["coverage_screen_pass"] is False
    with pytest.raises(ValueError, match="already completed"):
        run_wave.run(output, resume=True)


def test_fix1_duplicate_destination_rejected(tmp_path, synthetic):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    collect.run(args(folder, "carry"))
    with pytest.raises(ValueError, match="canonical|declared|cohort"):
        collect.run(args(tmp_path / "duplicate" / "carry" / lr.SCENARIOS[0], "carry"))


@pytest.mark.parametrize("field", SCHEMA["names"] + ["B_raw"])
def test_fix1_prefix_fault_before_step(tmp_path, synthetic, field):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    if field == "B_raw":
        ck["trace"][0][field][0] += 1.
    else:
        ck["transitions"][0][0][SCHEMA["names"].index(field)] += .125
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", ck)
    synthetic.setattr(FakeEnv, "step", lambda *a, **k: pytest.fail("corrupt prefix reached step"))
    with pytest.raises((ValueError, AssertionError)):
        collect.run(args(folder, resume=True))


@pytest.mark.parametrize("fault", ["inventory", "state", "control", "area_ttt"])
def test_fix1_terminal_physical_boundary(tmp_path, synthetic, fault):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    original = collect.validate_rows
    def fail_publication(*a, **k):
        if k.get("complete", True):
            raise RuntimeError("synthetic terminal interruption")
        return original(*a, **k)
    synthetic.setattr(collect, "validate_rows", fail_publication)
    with pytest.raises(RuntimeError, match="terminal interruption"):
        collect.run(args(folder, "carry"))
    synthetic.setattr(collect, "validate_rows", original)
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    if fault == "inventory":
        ck["trace"][-1]["inventory"] = 0.
    elif fault == "state":
        ck["environment"]["sim"].state.ramp_queue["r"] += 1.
    elif fault == "control":
        ck["environment"]["previous"]["ramp_metering"]["a"] += 1.
    else:
        ck["environment"]["sim"].freeway_ttt -= 1.
        ck["environment"]["sim"].urban_ttt += 1.
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", ck)
    with pytest.raises((ValueError, AssertionError)):
        collect.run(args(folder, "carry", resume=True))
    assert not (folder / "completion.json").exists()


@pytest.mark.parametrize("swap", ["behavior", "scenario"])
def test_fix1_direct_pairs_reject_intact_swaps(tmp_path, synthetic, swap):
    for behavior in lr.BEHAVIORS:
        for scenario in lr.SCENARIOS:
            collect.run(args(tmp_path / behavior / scenario, behavior, scenario=scenario))
    if swap == "behavior":
        a, b = tmp_path / "carry", tmp_path / "local"
        a.rename(tmp_path / "swap")
        b.rename(a)
        (tmp_path / "swap").rename(b)
    else:
        for behavior in lr.BEHAVIORS:
            a, b = [tmp_path / behavior / s for s in lr.SCENARIOS[:2]]
            a.rename(tmp_path / "swap")
            b.rename(a)
            (tmp_path / "swap").rename(b)
    with pytest.raises(ValueError, match="slot"):
        validate.compare_pairs(tmp_path, IDENTITY)


@pytest.mark.parametrize("prefix", [None, 2])
def test_fix1_session_time_includes_writes_and_close(tmp_path, synthetic, prefix):
    clock = [0.]
    synthetic.setattr(collect.time, "perf_counter", lambda: clock[0])
    original_checkpoint, original_save = collect.checkpoint_save, collect.save
    def costly_checkpoint(*a, **k):
        original_checkpoint(*a, **k)
        clock[0] += 3.
    def costly_save(path, value):
        original_save(path, value)
        if path.name == "status.json":
            clock[0] += 2.
    def costly_close(self):
        clock[0] += 5.
    synthetic.setattr(collect, "checkpoint_save", costly_checkpoint)
    synthetic.setattr(collect, "save", costly_save)
    synthetic.setattr(FakeEnv, "close", costly_close)
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=prefix))
    if prefix is not None:
        collect.run(args(folder, resume=True))
    _, _, summary = validate.load_completed(folder, IDENTITY)
    assert summary["elapsed_wall_seconds"] == clock[0]


def test_fix1_alternate_coordinator_root_rejected(tmp_path, synthetic):
    synthetic.setattr(run_wave, "identity", lambda: pytest.fail("must reject before preflight"))
    with pytest.raises(ValueError, match="canonical"):
        run_wave.run(tmp_path / "duplicate")


def test_fix1_startup_intent_survives_coordinator_loss(tmp_path, synthetic):
    job = run_wave.jobs_for(tmp_path)[0]
    folder = tmp_path / job["key"]
    token = lr.reserve_worker(folder, job["scenario"], job["behavior"], job["cpu_mask"], launch=True)
    # The launching coordinator is gone; absence of child PID is not proof of no child.
    synthetic.setattr(lr, "probe_process", lambda pid: ("dead", None))
    synthetic.setattr(run_wave.subprocess, "Popen", lambda *a, **k: pytest.fail("startup overlap"))
    with pytest.raises(OSError, match="UNKNOWN"):
        run_wave.run(tmp_path, resume=True)
    with pytest.raises(OSError, match="UNKNOWN"):
        collect.run(args(folder, "carry"))
    assert lr.read(tmp_path / ".ownership/reservations.json")[0]["token"] == token


@pytest.mark.parametrize("status,created,blocked", [("live", 101, True), ("unknown", None, True),
                                                  ("dead", None, False), ("live", 102, False)])
def test_fix1_registered_orphan_identity(tmp_path, synthetic, status, created, blocked):
    job = run_wave.jobs_for(tmp_path)[0]
    synthetic.setattr(lr, "probe_process", lambda pid: ("live", 101))
    token = lr.reserve_worker(tmp_path / job["key"], job["scenario"], job["behavior"], job["cpu_mask"], launch=True)
    lr.bind_launch(token, 999, ["python", "collect.py"])
    synthetic.setattr(lr, "probe_process", lambda pid: (status, created))
    if blocked:
        with pytest.raises(OSError, match="UNKNOWN"):
            lr.ensure_cohort_idle(tmp_path)
    else:
        lr.ensure_cohort_idle(tmp_path)
        assert lr.read(tmp_path / ".ownership/reservations.json")[0]["state"] == "exited"


def test_fix1_shared_five_worker_cap_and_claim_once(tmp_path, synthetic):
    jobs = run_wave.jobs_for(tmp_path)
    first = jobs[0]
    token = lr.reserve_worker(tmp_path / first["key"], first["scenario"], first["behavior"], first["cpu_mask"])
    for job in jobs[1:5]:
        lr.reserve_worker(tmp_path / job["key"], job["scenario"], job["behavior"], job["cpu_mask"], launch=True)
    synthetic.setattr(collect, "boot", lambda *a: pytest.fail("sixth worker booted"))
    with pytest.raises(OSError, match="five"):
        collect.run(args(tmp_path / jobs[5]["key"], "local"))
    with pytest.raises(OSError, match="reservation"):
        lr.reserve_worker(tmp_path / first["key"], first["scenario"], first["behavior"], first["cpu_mask"], launch=True)
    lr.release_worker(token)
    first = jobs[5]
    token = lr.reserve_worker(tmp_path / first["key"], first["scenario"], first["behavior"], first["cpu_mask"], launch=True)
    assert lr.claim_worker(tmp_path / first["key"], first["scenario"], first["behavior"], first["cpu_mask"], token) == token
    with pytest.raises(ValueError, match="claimed"):
        lr.claim_worker(tmp_path / first["key"], first["scenario"], first["behavior"], first["cpu_mask"], token)


def test_fix1_reservation_is_persisted_before_spawn(tmp_path, synthetic):
    job = run_wave.jobs_for(tmp_path)[0]
    def launch(command, **kwargs):
        token = command[command.index("--reservation")+1]
        records = lr.read(tmp_path / ".ownership/reservations.json")
        assert records[-1]["token"] == token and records[-1]["state"] == "reserved"
        assert records[-1]["worker"] is None
        raise RuntimeError("synthetic spawn interruption")
    synthetic.setattr(run_wave.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="spawn interruption"):
        run_wave.run_stage(tmp_path, [job], IDENTITY, tmp_path / "attempt/ABORT.json", [job])
    with pytest.raises(OSError, match="UNKNOWN"):
        lr.ensure_cohort_idle(tmp_path)


@pytest.mark.parametrize("steps", [0, 1, 75])
def test_fix1_saved_observation_boundary(tmp_path, synthetic, steps):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    if steps == 0:
        synthetic.setattr(collect, "verify_identity", lambda _: (_ for _ in ()).throw(RuntimeError("zero boundary")))
        with pytest.raises(RuntimeError, match="zero boundary"):
            collect.run(args(folder, "carry"))
    else:
        collect.run(args(folder, "carry", limit=steps))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    collect.validate_checkpoint(ck, ck["settings"])
    ck["observation"][0] += .125
    if steps:
        ck["transitions"][-1][3][0] += .125
    with pytest.raises((ValueError, AssertionError)):
        collect.validate_checkpoint(ck, ck["settings"])


@pytest.mark.parametrize("field", ["last_requested", "last_executed", "last_budget", "prepared"])
def test_fix1_checkpoint_budget_binding(tmp_path, synthetic, field):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    value = ck["environment"][field]
    if field == "prepared":
        value["action_anchor"][0] += 1.
    else:
        value[0] += 1.
    with pytest.raises((ValueError, AssertionError)):
        collect.validate_checkpoint(ck, ck["settings"])


def test_fix1_exact_frozen_inventory_components():
    state = asdict(FakeState())
    # 20 core + 3 ramp + 4 origin + 5 movement + 10 urban storage + 7 off-ramp + 14 buffers.
    assert validate.inventory(state, CONFIG) == 63.
    state["urban_arrival_buffer"]["urban"][1] += 10000.
    state["urban_storage_release_buffer"]["urban"][1] += 10000.
    state["urban_queue"]["legacy"] += 10000.
    state["boundary_queue"]["legacy"] += 10000.
    assert validate.inventory(state, CONFIG) == 63.
    state["freeway_effective_lanes"]["a"] = [1.]
    state["freeway_buffer_down_density"]["a"] = [9.]
    assert validate.inventory(state, CONFIG) == 54.
    del state["urban_link_storage"]
    with pytest.raises(ValueError, match="Missing physical"):
        validate.inventory(state, CONFIG)


@pytest.mark.parametrize("missing", ["end", "both"])
def test_fix1_missing_elapsed_survives_resume_as_unknown(tmp_path, synthetic, missing):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    session = lr.session_ids(folder)[0]
    start = folder / "sessions" / f"{session}.start.json"
    preserved = start.read_bytes()
    (folder / "sessions" / f"{session}.end.json").unlink()
    if missing == "both":
        start.unlink()
    collect.run(args(folder, resume=True))
    _, _, summary = validate.load_completed(folder, IDENTITY)
    assert summary["timing_status"] == "UNKNOWN" and summary["elapsed_wall_seconds"] is None
    assert summary["unknown_sessions"] == [session] and summary["known_elapsed_wall_seconds"] > 0
    assert lr.read(folder / "summary.json")["timing_status"] == "UNKNOWN"
    if missing == "end":
        assert start.read_bytes() == preserved
    assert not (folder / "sessions" / f"{session}.end.json").exists()


def test_fix1_failed_session_retains_elapsed_and_completion_last(tmp_path, synthetic):
    clock = [0.]
    synthetic.setattr(collect.time, "perf_counter", lambda: clock[0])
    original_step = FakeEnv.step
    def fail_step(*a, **k):
        clock[0] += 11.
        raise RuntimeError("synthetic failed work")
    synthetic.setattr(FakeEnv, "step", fail_step)
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    with pytest.raises(RuntimeError, match="failed work"):
        collect.run(args(folder))
    timing = lr.read(folder / "timing.json")
    assert timing["elapsed_wall_seconds"] == 11.
    session = timing["session_ids"][0]
    end_file = folder / "sessions" / f"{session}.end.json"
    assert lr.read(end_file)["outcome"] == "failed"
    original_bytes = end_file.read_bytes()
    writes, original_save = [], lr.save
    def record_write(path, value):
        writes.append(Path(path))
        original_save(path, value)
    synthetic.setattr(lr, "save", record_write)
    synthetic.setattr(collect, "save", record_write)
    synthetic.setattr(FakeEnv, "step", original_step)
    collect.run(args(folder, resume=True))
    assert writes[-1] == folder / "completion.json"
    assert validate.load_completed(folder, IDENTITY)[2]["elapsed_wall_seconds"] == 11.
    assert end_file.read_bytes() == original_bytes
    before = {str(p.relative_to(folder)): lr.file_hash(p) for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Already completed"):
        collect.run(args(folder, resume=True))
    assert before == {str(p.relative_to(folder)): lr.file_hash(p) for p in folder.rglob("*") if p.is_file()}


def test_fix1_loss_before_session_start_is_unknown(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    # A resume acquires ownership, then disappears before publishing its timing start.
    token = lr.reserve_worker(folder, lr.SCENARIOS[0], "local", lr.MASKS[0], resume=True)
    lr.release_worker(token)  # Synthetic proof of process exit; no process was launched.
    collect.run(args(folder, resume=True))
    summary = validate.load_completed(folder, IDENTITY)[2]
    assert summary["timing_status"] == "UNKNOWN" and summary["unknown_sessions"] == [token]


def test_fix1_completed_identity_retained_in_cohort(tmp_path, synthetic):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    collect.run(args(folder, "carry"))
    folder.rename(tmp_path / "preserved")
    with pytest.raises(ValueError, match="completed cohort"):
        collect.run(args(folder, "carry"))
    with pytest.raises(ValueError, match="canonical"):
        collect.run(args(tmp_path / "another" / "carry" / lr.SCENARIOS[0], "carry"))


def test_fix1_initial_live_observation_rejected_before_step(tmp_path, synthetic):
    original = FakeEnv.reset
    def invalid_reset(self):
        obs = original(self)
        obs[0] += .125
        return obs
    synthetic.setattr(FakeEnv, "reset", invalid_reset)
    synthetic.setattr(FakeEnv, "step", lambda *a, **k: pytest.fail("invalid initial observation reached step"))
    with pytest.raises(AssertionError):
        collect.run(args(tmp_path / "carry" / lr.SCENARIOS[0], "carry"))


def test_fix1_missing_prefix_schema_rejected(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    broken = copy.deepcopy(SCHEMA)
    broken["names"][0] = "unrecognized_clock"
    with pytest.raises(ValueError, match="Missing/duplicate"):
        validate.validate_rows(ck["trace"], ck["transitions"], ck["settings"], broken, 6000., False)


def test_fix1_prefix_projection_rejected(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    ck["trace"][0]["B_requested"][0][1] += 1.
    with pytest.raises(AssertionError):
        validate.validate_rows(ck["trace"], ck["transitions"], ck["settings"], SCHEMA, 6000., False)


def test_fix1_session_file_change_rejected(tmp_path, synthetic):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    collect.run(args(folder, "carry"))
    session = lr.session_ids(folder)[0]
    path = folder / "sessions" / f"{session}.end.json"
    end = lr.read(path)
    end["elapsed_wall_seconds"] += 1.
    lr.save(path, end)
    with pytest.raises(ValueError, match="Timing reconciliation"):
        validate.load_completed(folder, IDENTITY)


@pytest.mark.parametrize("field", ["total_ttt", "freeway_ttt", "urban_ttt"])
def test_fix1_zero_boundary_nonfinite_accounting(tmp_path, synthetic, field):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    def interrupt(_):
        raise RuntimeError("zero boundary")
    synthetic.setattr(collect, "verify_identity", interrupt)
    with pytest.raises(RuntimeError, match="zero boundary"):
        collect.run(args(folder, "carry"))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    setattr(ck["environment"]["sim"], field, float("nan"))
    with pytest.raises(ValueError, match="finite|accounting"):
        collect.validate_checkpoint(ck, ck["settings"])


def test_fix2_complete_nested_config_preserves_runtime_extra():
    network_type = make_dataclass("SyntheticNetwork", [(key, object) for key in CONFIG["network"]])
    config_type = make_dataclass("SyntheticConfig", [("network", object)])
    cfg = config_type(network_type(**copy.deepcopy(CONFIG["network"])))
    cfg.network.terminal_zero_gradient = True
    expected = copy.deepcopy(CONFIG)
    expected["network"]["terminal_zero_gradient"] = True
    settings = dict(physical_config=expected, warmup_ttt=5., contract={
        "environment_contract": {"config_sha256": lr.digest(expected)}})
    env = FakeEnv(None)
    obs = env.reset()
    env.sim.cfg = cfg
    ck = dict(environment=env.checkpoint(), trace=[], observation=obs, schema=SCHEMA)
    validate.validate_boundary(ck, settings)
    ck["environment"]["sim"].cfg.network.terminal_zero_gradient = False
    with pytest.raises(ValueError, match="configuration"):
        validate.validate_boundary(ck, settings)
    settings["physical_config"]["network"]["terminal_zero_gradient"] = False
    with pytest.raises(ValueError, match="configuration"):
        validate.validate_boundary(ck, settings)


@pytest.mark.parametrize("phase", ["started", "checkpointed", "finished", "coordinator_finished"])
def test_fix2_immediate_relocation_cannot_restart_slot(tmp_path, synthetic, phase):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    request = args(folder, "carry", limit=1 if phase in ("started", "checkpointed") else None)
    if phase == "coordinator_finished":
        request.reservation = lr.reserve_worker(folder, request.scenario, "carry", request.cpu_mask, launch=True)
        # Emulate the child claiming a dispatched intent. No parent post-exit release follows.
    if phase == "started":
        original_identity = collect.identity
        def fail_initialization():
            raise RuntimeError("synthetic initialization loss")
        synthetic.setattr(collect, "identity", fail_initialization)
        with pytest.raises(RuntimeError, match="initialization loss"):
            collect.run(request)
        synthetic.setattr(collect, "identity", original_identity)
    else:
        collect.run(request)
    preserved = tmp_path / "preserved"
    folder.rename(preserved)
    hashes = {str(p.relative_to(preserved)): lr.file_hash(p) for p in preserved.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="cohort"):
        collect.run(args(folder, "carry", limit=1))
    assert hashes == {str(p.relative_to(preserved)): lr.file_hash(p) for p in preserved.rglob("*") if p.is_file()}


@pytest.mark.parametrize("interruption", ["session_end", "completion_marker"])
def test_fix2_interrupted_finalization_resumes_terminal_checkpoint(tmp_path, synthetic, interruption):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    original_save, original_durable = collect.save, collect.durable_save
    def interrupt(path, value):
        if (interruption == "session_end" and path.name.endswith(".end.json") or
                interruption == "completion_marker" and path.name == "completion.json"):
            raise RuntimeError("synthetic publication interruption")
        (original_durable if interruption == "session_end" else original_save)(path, value)
    synthetic.setattr(collect, "durable_save" if interruption == "session_end" else "save", interrupt)
    with pytest.raises(RuntimeError, match="publication interruption"):
        collect.run(args(folder, "carry"))
    record = lr.read(tmp_path / ".ownership/reservations.json")[-1]
    assert record["cohort_phase"] == "finished" and record["control_steps"] == 75
    assert record["state"] == "exited" and not (folder / "completion.json").exists()
    run_id = lr.read(folder / "settings.json")["run_id"]
    original_sessions = {p.name: p.read_bytes() for p in (folder / "sessions").iterdir()}
    synthetic.setattr(collect, "save", original_save)
    synthetic.setattr(collect, "durable_save", original_durable)
    synthetic.setattr(FakeEnv, "step", lambda *a, **k: pytest.fail("terminal resume must not step"))
    assert collect.run(args(folder, "carry", resume=True)) == "completed"
    _, settings, summary = validate.load_completed(folder, IDENTITY)
    assert settings["run_id"] == run_id
    assert summary["timing_status"] == ("UNKNOWN" if interruption == "session_end" else "KNOWN")
    assert all((folder / "sessions" / name).read_bytes() == value for name, value in original_sessions.items())
    assert all(r["run_id"] == run_id for r in lr.read(tmp_path / ".ownership/reservations.json"))


def test_fix2_relocated_checkpoint_can_resume_when_restored(tmp_path, synthetic):
    folder = tmp_path / "local" / lr.SCENARIOS[0]
    collect.run(args(folder, limit=2))
    run_id = lr.read(folder / "settings.json")["run_id"]
    record = lr.read(tmp_path / ".ownership/reservations.json")[-1]
    assert record["cohort_phase"] == "checkpointed" and record["control_steps"] == 2
    preserved = tmp_path / "preserved"
    folder.rename(preserved)
    with pytest.raises(ValueError, match="cohort"):
        collect.run(args(folder, limit=1))
    folder.rename(tmp_path / "rejected_empty_output")
    preserved.rename(folder)
    assert collect.run(args(folder, resume=True, limit=1)) == "checkpointed"
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    assert ck["settings"]["run_id"] == run_id and len(ck["trace"]) == 3
    assert lr.read(tmp_path / ".ownership/reservations.json")[-1]["control_steps"] == 3


@pytest.mark.parametrize("fault", ["rewind", "run_id"])
def test_fix2_resume_cannot_replace_cohort_history(tmp_path, synthetic, fault):
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    collect.run(args(folder, "carry", limit=1))
    ck = lr.torch.load(folder / "checkpoint.pt", weights_only=False)
    if fault == "rewind":
        collect.run(args(folder, "carry", resume=True, limit=1))
    else:
        ck["settings"]["run_id"] = "unrelated-run"
        lr.save(folder / "settings.json", ck["settings"])
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", ck)
    before = lr.file_hash(folder / "checkpoint.pt")
    synthetic.setattr(FakeEnv, "step", lambda *a, **k: pytest.fail("invalid cohort resume reached step"))
    with pytest.raises(ValueError, match="Cohort"):
        collect.run(args(folder, "carry", resume=True))
    assert lr.file_hash(folder / "checkpoint.pt") == before


def test_fix2_finished_ledger_write_is_measured_and_completion_last(tmp_path, synthetic):
    clock, writes = [0.], []
    synthetic.setattr(collect.time, "perf_counter", lambda: clock[0])
    original_durable, original_save = lr.durable_save, collect.save
    def costly_finished_ledger(path, value):
        original_durable(path, value)
        writes.append(path)
        if path.name == "reservations.json" and any(r["cohort_phase"] == "finished" for r in value):
            clock[0] += 7.
    def record_save(path, value):
        original_save(path, value)
        writes.append(path)
    synthetic.setattr(lr, "durable_save", costly_finished_ledger)
    synthetic.setattr(collect, "save", record_save)
    folder = tmp_path / "carry" / lr.SCENARIOS[0]
    collect.run(args(folder, "carry"))
    assert validate.load_completed(folder, IDENTITY)[2]["elapsed_wall_seconds"] == 7.
    assert writes[-1] == folder / "completion.json"
    assert lr.read(tmp_path / ".ownership/reservations.json")[-1]["cohort_phase"] == "finished"
