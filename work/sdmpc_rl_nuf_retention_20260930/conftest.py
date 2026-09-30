"""Small synthetic collector fixtures; real boot and BudgetEnv are forbidden."""
import copy
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace as NS
import pytest
import local_runtime as lr
import collect
import validate
import run_wave
import references
from exploration import TREATMENT

SEQUENCE_VALIDATOR = lr.sequence_validator()

SCHEMA = {"names": ["state/time_sec", "memory/remaining/0"] +
    [f"memory/{name}/{i}" for name in ("action_anchor", "previous_requested", "previous_executed") for i in range(2)]}
CONFIG = dict(network=dict(total_ramp_capacity=6000., freeway_links=["a"],
    freeway_segments_per_link=1, freeway_segment_length_km=.5, freeway_lanes=2,
    freeway_buffer_segments=1, off_ramp_storage_link={"r": "off"},
    urban_link_storage_veh={"urban": 50., "off": 30.}))
CONTRACT = {"observation_schema": SCHEMA, "environment_contract": {"config_sha256": lr.digest(CONFIG)}}
IDENTITY = dict(physical={"fixture": "physical"}, runtime={"fixture": "runtime"},
                gate_sha256="fixture", contract_sha256=lr.digest(CONTRACT), local_sources={"fixture": "candidate"})


@dataclass
class StaticState:
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


class FakeEnv:
    def __init__(self, rt, scenario, **kwargs):
        self.profile_hash = "fixture-" + scenario
        self.observer = NS(names=SCHEMA["names"])

    def observe(self):
        if self.k == 80:
            return lr.np.zeros(8, dtype=lr.np.float32)
        scale = lr.np.array([1000., 10000.])
        return lr.np.array([self.k/80., (80-self.k)/75., *(self.controller.action_anchor/scale),
            *(self.last_requested/scale), *(self.last_executed/scale)], dtype=lr.np.float32)

    def reset(self):
        self.k, self.warmup_ttt = 5, 5.
        self.sim = NS(total_ttt=5., freeway_ttt=5., urban_ttt=0., state=StaticState(), cfg=copy.deepcopy(CONFIG))
        self.controller = NS(action_anchor=lr.np.array([-10., 6000.]))
        self.last_requested = self.last_executed = lr.np.zeros(2)
        self.previous = {k: {"a": 1.} for k in validate.CONTROL_FIELDS}
        return self.observe()

    def step(self, action, mode, actor_wall, actor_cpu_seconds):
        from budget_controller import residual_budget
        anchor = self.controller.action_anchor.copy()
        raw, request = residual_budget(action, anchor, 6000.)
        i = self.k-5
        self.k += 1
        self.sim.total_ttt += 1.
        self.sim.freeway_ttt += 1.
        self.sim.state.time_sec += 180.
        self.controller.action_anchor = request.copy()
        self.last_requested, self.last_executed = request.copy(), request.copy()
        row = dict(action_anchor=anchor, action_requested=action, B_requested=[request], B_executed=request,
            B_raw=raw, G_achieved=request, selection_source="reference_fallback" if i == 9 else "lower_solution",
            reference_source="pfo_initial" if i == 0 else "previous", reference_recovery=False,
            guard_mode="physical", h3_guard_enabled=False,
            execution_check=dict(physical_control_valid=True, budget_feasible=True),
            lower_candidate_count=1, pfo_calls=int(i == 0),
            step=i+5, control_step=i, time_sec=self.sim.state.time_sec, terminated=self.k == 80,
            truncated=False, interval_ttt=1., total_ttt=self.sim.total_ttt,
            freeway_ttt=self.sim.total_ttt, urban_ttt=0., inventory=63.,
            plant_state=asdict(self.sim.state), control=copy.deepcopy(self.previous),
            candidates=[dict(stationarity=float("inf"), rows=[dict(primal_stationarity=float("inf"))])])
        for component in ("forecast", "observation", "pfo", "reference", "lower", "guard"):
            row[component+"_wall_seconds"] = row[component+"_cpu_seconds"] = .1
        row.update(actor_wall_seconds=actor_wall, actor_cpu_seconds=actor_cpu_seconds,
                   decision_wall_seconds=.6+actor_wall, decision_cpu_seconds=.6+actor_cpu_seconds)
        return self.observe(), -.01, self.k == 80, row

    def checkpoint(self):
        return dict(k=self.k, profile_hash=self.profile_hash, sim=copy.deepcopy(self.sim),
            anchor=self.controller.action_anchor.copy(), warmup_ttt=self.warmup_ttt,
            previous=copy.deepcopy(self.previous), last_requested=self.last_requested.copy(),
            last_executed=self.last_executed.copy(), last_budget=self.last_executed.copy() if self.k > 5 else None,
            prepared=dict(budget=self.controller.action_anchor.copy(),
                          action_anchor=self.controller.action_anchor.copy()) if self.k < 80 else None)

    def restore(self, ck):
        self.k, self.sim, self.warmup_ttt = ck["k"], copy.deepcopy(ck["sim"]), ck["warmup_ttt"]
        self.controller = NS(action_anchor=ck["anchor"].copy())
        self.previous = copy.deepcopy(ck["previous"])
        self.last_requested, self.last_executed = ck["last_requested"].copy(), ck["last_executed"].copy()
        return self.observe()

    def close(self):
        pass


def fixture_references():
    entries = {}
    for job in run_wave.jobs_for(None):
        for behavior in ("carry", "local"):
            key = f"{behavior}/{job['scenario']}"
            entries[key] = dict(**{k: v for k, v in job.items() if k not in ("key", "behavior")},
                behavior=behavior, run_id="old-"+key, path="synthetic/"+key,
                profile_sha256="fixture-"+job["scenario"], warmup_ttt=5., capacity=6000.,
                contract_sha256=lr.digest(CONTRACT), physical_config_sha256=lr.digest(CONFIG),
                completion_sha256="fixture", outputs_sha256={}, export_recovery=None)
    return dict(entries=entries, source_identity=dict(IDENTITY, local_sources={"fixture": "old"}),
                root_sha256={"completion.json": "fixture"})


@pytest.fixture(autouse=True)
def forbid_real_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real simulator/runtime forbidden in scoped tests")
    monkeypatch.setattr(collect, "boot", forbidden)
    monkeypatch.setattr(lr, "boot", forbidden)
    monkeypatch.setattr(collect, "BudgetEnv", forbidden)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(lr, "COHORT_ROOT", tmp_path)
    monkeypatch.setattr(lr, "REPO", tmp_path)
    monkeypatch.setattr(lr, "GOAL", tmp_path)
    refs = fixture_references()
    monkeypatch.setattr(references, "manifest", lambda: copy.deepcopy(refs))
    monkeypatch.setattr(collect, "manifest", lambda: copy.deepcopy(refs))
    monkeypatch.setattr(run_wave, "authenticate", lambda: copy.deepcopy(refs))
    for module in (lr, collect, validate, run_wave):
        if hasattr(module, "identity"):
            monkeypatch.setattr(module, "identity", lambda: copy.deepcopy(IDENTITY))
        def verify(value):
            if value != IDENTITY:
                raise ValueError("Synthetic source identity changed")
        monkeypatch.setattr(module, "verify_identity", verify)
        if hasattr(module, "contract"):
            monkeypatch.setattr(module, "contract", lambda: copy.deepcopy(CONTRACT))
    monkeypatch.setattr(collect, "boot", lambda *args: dict(cfg=NS(network=NS(total_ramp_capacity=6000.)),
        rc=NS(to_plain_dict=lambda cfg: copy.deepcopy(CONFIG))))
    monkeypatch.setattr(collect, "BudgetEnv", FakeEnv)
    monkeypatch.setattr(collect, "verify_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(collect, "observation_schema", lambda env: copy.deepcopy(SCHEMA))
    monkeypatch.setattr(validate, "queue_exposure", lambda trace: None)
    monkeypatch.setattr(validate, "sequence_validator", lambda: SEQUENCE_VALIDATOR)
    monkeypatch.setattr(lr.os, "getppid", lr.os.getpid)
    return NS(patch=monkeypatch, references=refs)


def request(root, scenario=None, resume=False, limit=None):
    scenario = scenario or lr.SCENARIOS[0]
    folder = root / "local" / scenario
    mask = lr.MASKS[lr.SCENARIOS.index(scenario)]
    token = lr.reserve_worker(folder, scenario, "local", mask, launch=True, resume=resume)
    lr.bind_launch(token, lr.os.getpid(), ["synthetic-direct-child"])
    return NS(output=folder, scenario=scenario, behavior="local", cpu_mask=mask, reservation=token,
              resume=resume, max_new_steps=limit, abort_file=None)
