"""Synthetic ledger environment only; actual boot/reset/step is forbidden here."""
import copy
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace as NS
import pytest
import wave_support as w
import worker
import readout
import checks
from actor import authenticate

HELPERS = checks.helpers()
ACTOR, AUTH = authenticate()
SCHEMA = AUTH["contract"]["observation_schema"]
CONFIG = dict(network=dict(total_ramp_capacity=6000., freeway_links=["a"],
    freeway_segments_per_link=8, freeway_segment_length_km=.5, freeway_lanes=2,
    freeway_buffer_segments=1, off_ramp_storage_link={"r": "off"},
    urban_link_storage_veh={"urban": 50., "off": 30.}, ramps=["r"],
    ramp_to_freeway={"r": "a"}, ramp_merge_segment_index={"r": 0}, ramp_capacity_veh_h={"r": 6000.},
    signals=["A"], cycle_length=120., lost_time=8., green_min=20., green_max=92.),
    freeway_follower=dict(vsl_set=[60., 80., 100.], max_vsl_step=20.),
    urban_follower=dict(max_offset_step=15.))
OPTIONS = dict(fd_metering_veh_h=10., fd_green_sec=.5, fd_offset_sec=.5)


@dataclass
class State:
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
    resets = 0
    restores = 0
    steps = 0
    contract_value = None

    def __init__(self, rt, scenario, training_seed, guard_mode):
        assert guard_mode == "physical" and training_seed == 7301+w.SCENARIOS.index(scenario)
        self.profile_hash = AUTH["profiles"][scenario]

    def observe(self):
        obs = w.np.zeros(2367, dtype=w.np.float32)
        if self.k == 80:
            return obs
        values = {"state/time_sec": self.k/80., "memory/remaining/0": (80-self.k)/75.}
        for key, budget in (("action_anchor", self.controller.action_anchor),
                            ("previous_requested", self.last_requested), ("previous_executed", self.last_executed)):
            values.update({f"memory/{key}/{i}": v for i, v in enumerate(budget/[1000., 10000.])})
        for name, value in values.items():
            obs[SCHEMA["names"].index(name)] = value
        return obs

    def reset(self):
        type(self).resets += 1
        self.k, self.warmup_ttt = 5, 5.
        self.sim = NS(total_ttt=5., freeway_ttt=5., urban_ttt=0., state=State(), cfg=copy.deepcopy(CONFIG))
        self.controller = NS(action_anchor=w.np.array([-10., 5000.]))
        self.last_requested = w.np.zeros(2)
        self.last_executed = w.np.zeros(2)
        self.previous = dict(ramp_metering={"r": 100.}, vsl={"a": 100., **{f"a__seg{i}": 100. for i in range(8)}},
            green_times={"A_p1": 56., "A_p2": 56.}, offsets={"A": 0.}, inflow_outflow_allocation={},
            N_P_star=-10., N_UF_star=5000.)
        return self.observe()

    def step(self, action, mode, actor_wall, actor_cpu_seconds):
        assert mode == "rl"
        type(self).steps += 1
        anchor = self.controller.action_anchor.copy()
        raw, request = w.residual_budget(action, anchor, 6000.)
        i = self.k-5
        self.k += 1
        self.sim.total_ttt += 1.
        self.sim.freeway_ttt += 1.
        self.sim.state.time_sec += 180.
        self.controller.action_anchor = request.copy()
        self.last_requested, self.last_executed = request.copy(), request.copy()
        self.previous.update(N_P_star=request[0], N_UF_star=request[1])
        state = asdict(self.sim.state)
        queues = {k: state[k] for k in ("ramp_queue", "mainline_origin_queue", "boundary_queue", "urban_movement_queue")}
        exposure = dict(method="interval_endpoint_sampled", exact_substep_exposure=False,
            threshold_fraction=.9, interval_seconds=180., duration_units="s", capacity_units="veh")
        for kind in ("ramp_queue", "boundary_queue"):
            exposure[kind] = {k: dict(queue_veh=v, capacity_veh=180., near_capacity_seconds=180.*(v>=162.))
                              for k, v in queues[kind].items()}
        row = dict(action_anchor=anchor, action_requested=action, B_requested=[request], B_executed=request,
            B_raw=raw, G_achieved=request, selection_source="lower_solution", fallback_reasons=[],
            reference_source="pfo_initial" if i == 0 else "previous", reference_recovery=False,
            guard_mode="physical", h3_guard_enabled=False,
            execution_check=dict(physical_control_valid=True, budget_feasible=True, budget=request,
                                 achieved=request, point=[1.], ttt=3., band_excess=[0., 0.]),
            selected_identity=dict(point=[1.]), selected_TTT=3., reference_TTT=4.,
            lower_candidate_count=1, pfo_calls=int(i == 0), step=i+5, control_step=i,
            time_sec=self.sim.state.time_sec, terminated=self.k == 80, truncated=False,
            interval_ttt=1., total_ttt=self.sim.total_ttt, freeway_ttt=self.sim.total_ttt, urban_ttt=0.,
            inventory=63., plant_state=state, control=copy.deepcopy(self.previous), queue_state=queues,
            queue_near_capacity_estimate=exposure,
            candidates=[dict(stationarity=float("inf"), rows=[dict(primal_stationarity=float("inf"))])])
        for part in ("forecast", "observation", "pfo", "reference", "lower", "guard", "plant"):
            row[part+"_wall_seconds"] = row[part+"_cpu_seconds"] = .1
        row.update(actor_wall_seconds=actor_wall, actor_cpu_seconds=actor_cpu_seconds,
                   decision_wall_seconds=.6+actor_wall, decision_cpu_seconds=.6+actor_cpu_seconds)
        return self.observe(), -.01, self.k == 80, row

    def checkpoint(self):
        return dict(k=self.k, profile_hash=self.profile_hash, sim=copy.deepcopy(self.sim),
            contract=copy.deepcopy(self.contract_value),
            anchor=self.controller.action_anchor.copy(), warmup_ttt=self.warmup_ttt,
            previous=copy.deepcopy(self.previous), last_requested=self.last_requested.copy(),
            last_executed=self.last_executed.copy(), last_budget=self.last_executed.copy() if self.k>5 else None,
            dual=w.np.zeros((2, 2)), last_slack=w.np.zeros(2), reference_timing={}, observation_timing={},
            prepared=dict(budget=self.controller.action_anchor.copy(),
                          action_anchor=self.controller.action_anchor.copy()) if self.k<80 else None)

    def restore(self, ck):
        type(self).restores += 1
        self.k, self.sim, self.warmup_ttt = ck["k"], copy.deepcopy(ck["sim"]), ck["warmup_ttt"]
        self.controller = NS(action_anchor=ck["anchor"].copy())
        self.previous = copy.deepcopy(ck["previous"])
        self.reference_timing = copy.deepcopy(ck["reference_timing"])
        self.observation_timing = copy.deepcopy(ck["observation_timing"])
        self.last_requested, self.last_executed = ck["last_requested"].copy(), ck["last_executed"].copy()
        return self.observe()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def forbid_real(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real environment execution forbidden in synthetic tests")
    monkeypatch.setattr(w, "boot", forbidden)
    monkeypatch.setattr(worker, "BudgetEnv", forbidden)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    auth = copy.deepcopy(AUTH)
    auth["contract"]["environment_contract"]["config_sha256"] = w.digest(CONFIG)
    auth["contract"]["environment_contract"]["options_sha256"] = w.digest(OPTIONS)
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(w, "REPO", root)
    monkeypatch.setattr(w, "GOAL", root / "goal")
    monkeypatch.setattr(w, "WAVE", root / "goal/wave")
    source = {"synthetic": "unchanged"}
    monkeypatch.setattr(w, "sources", lambda: source.copy())
    monkeypatch.setattr(w, "import_boundary", lambda **kwargs: None)
    monkeypatch.setattr(w, "process_identity", lambda: dict(pid=123, created=456, parent_pid=12,
                                                           command=["synthetic"], executable="synthetic"))
    def verify(authentication, sources):
        if authentication != auth or sources != source:
            raise ValueError("Synthetic source/model mutation")
    for mod in (worker, readout):
        monkeypatch.setattr(mod, "authenticate", lambda: (ACTOR, copy.deepcopy(auth)))
        monkeypatch.setattr(mod, "verify_live", verify)
    cfg = NS(network=NS(total_ramp_capacity=6000.))
    monkeypatch.setattr(w, "boot", lambda *args: dict(cfg=cfg, options=OPTIONS,
        rc=NS(to_plain_dict=lambda value: copy.deepcopy(CONFIG if value is cfg else OPTIONS))))
    monkeypatch.setattr(w, "observation_schema", lambda env: copy.deepcopy(SCHEMA))
    monkeypatch.setattr(HELPERS, "verify_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "BudgetEnv", FakeEnv)
    FakeEnv.contract_value = dict(auth["contract"]["environment_contract"]["coordinator"],
        cfg=CONFIG, options=OPTIONS, observation=SCHEMA["normalization"], source_snapshot=auth["physical_snapshot"])
    FakeEnv.resets = FakeEnv.steps = FakeEnv.restores = 0
    return NS(auth=auth, actor=ACTOR, source=source, patch=monkeypatch)


def request(scenario=None, resume=False, limit=None):
    return NS(scenario=scenario or w.SCENARIOS[0], resume=resume, max_new_steps=limit)
