"""Static synthetic checkpoint data. No environment construction/reset/step."""
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import export_recovery as recovery

lr, collector, validator, wave = recovery.modules()
SEQUENCE_VALIDATOR = lr.sequence_validator()
from exploration import LocalBudgetPolicy

SCHEMA = {"names": ["state/time_sec", "memory/remaining/0"] +
    [f"memory/{name}/{i}" for name in ("action_anchor", "previous_requested", "previous_executed") for i in range(2)]}
CONFIG = dict(network=dict(total_ramp_capacity=6000., freeway_links=["a"],
    freeway_segments_per_link=1, freeway_segment_length_km=.5, freeway_lanes=2,
    freeway_buffer_segments=1, off_ramp_storage_link={"r": "off"},
    urban_link_storage_veh={"urban": 50., "off": 30.}))
CONTRACT = {"observation_schema": SCHEMA, "environment_contract": {"config_sha256": lr.digest(CONFIG)}}
IDENTITY = {"fixture": "isolated-static-data"}
TOKEN = "a" * 32


def register_fake_coordinator(cohort, command, pid=500, parent_pid=999):
    import coordinator_bootstrap as bootstrap
    path = Path(command[-1])
    assert cohort.root in path.parents
    if not path.with_suffix(".coordinator.json").exists():
        bootstrap.register(lr, wave, cohort.root, path,
            dict(pid=pid, created=cohort.statuses[pid][1], command=command[3:]), parent_pid)


def finish_fake_coordinator(cohort, command, pid=500):
    attempt = lr.read(Path(command[-1]))
    process = dict(pid=pid, command=attempt["runner_command"][3:], started=attempt["started"] + 1,
                   attempt=str(cohort.root / "attempts" / attempt["id"]))
    lr.save(cohort.root / "process.json", process)
    lr.save(Path(process["attempt"]) / "process.json", process)


@dataclass
class StaticState:
    time_sec: float = 14400.
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
    urban_arrival_buffer: dict = field(default_factory=dict)
    urban_storage_release_buffer: dict = field(default_factory=dict)


def static_checkpoint(settings):
    np = lr.np
    anchor, previous = np.array([-10., 6000.]), np.zeros(2)
    policy = LocalBudgetPolicy(settings["behavior"], settings["exploration_seed"], 6000.)
    control = {k: {"a": 1.} for k in validator.CONTROL_FIELDS}

    def observation(k, budget, memory):
        if k == 80:
            return np.zeros(8, dtype=np.float32)
        return np.array([k*180/14400., (80-k)/75., *(budget/[1000., 10000.]),
                         *(memory/[1000., 10000.]), *(memory/[1000., 10000.])], dtype=np.float32)

    trace, experience = [], []
    for i in range(75):
        obs = observation(i+5, anchor, previous)
        action, audit = policy.choose(anchor)
        request = np.array(audit["projected_request"])
        nxt = observation(i+6, request, request)
        row = dict(action_anchor=anchor.tolist(), action_requested=action.tolist(),
            B_requested=[request.tolist()], B_executed=request.tolist(), B_raw=audit["raw_request"],
            selection_source="reference_fallback" if i == 14 else "lower_solution",
            reference_source="pfo_initial" if i == 0 else "previous", reference_recovery=False,
            guard_mode="physical", h3_guard_enabled=False,
            execution_check=dict(physical_control_valid=True, budget_feasible=True),
            lower_candidate_count=1, pfo_calls=int(i == 0), step=i+5, control_step=i,
            time_sec=(i+6)*180, terminated=i == 74, truncated=False,
            interval_ttt=1., total_ttt=6.+i, freeway_ttt=6.+i, urban_ttt=0., inventory=63.,
            plant_state=asdict(StaticState(time_sec=(i+6)*180)), control=copy.deepcopy(control),
            reward=-.01, scenario=settings["scenario"], behavior=settings["behavior"],
            profile_sha256=settings["profile_sha256"], policy_q=None, learning=[], learning_wall_seconds=0.,
            candidates=[dict(stationarity=1., converged=False,
                             rows=[dict(primal_stationarity=1.) for _ in range(6)])],
            queue_state=dict(ramp_queue={"r": 3.}, boundary_queue={"legacy": 999.}),
            queue_near_capacity_estimate=dict(method="interval_endpoint_sampled", exact_substep_exposure=False,
                threshold_fraction=.9, interval_seconds=180, duration_units="s", capacity_units="veh",
                ramp_queue={"r": dict(queue_veh=3., capacity_veh=100., near_capacity_seconds=0.)},
                boundary_queue={"legacy": dict(queue_veh=999., capacity_veh=2000., near_capacity_seconds=0.)}))
        for component in ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor"):
            for kind in ("wall", "cpu"):
                row[f"{component}_{kind}_seconds"] = .1
        row.update(decision_wall_seconds=.7, decision_cpu_seconds=.7)
        row["exploration_audit"] = policy.commit(row)
        trace.append(row)
        experience.append((obs, action, -.01, nxt, i == 74))
        anchor, previous = request, request
    candidate = trace[14]["candidates"][0]
    candidate["stationarity"] = float("inf")
    for index in (2, 3, 5):
        candidate["rows"][index]["primal_stationarity"] = float("inf")
    return dict(format=lr.FORMAT, settings=settings, schema=SCHEMA, trace=trace, transitions=experience,
        environment=dict(k=80, profile_hash=settings["profile_sha256"], warmup_ttt=5.,
            sim=NS(total_ttt=80., freeway_ttt=80., urban_ttt=0., state=StaticState(), cfg=copy.deepcopy(CONFIG)),
            previous=control, last_budget=anchor, last_requested=previous, last_executed=previous, prepared=None),
        observation=experience[-1][3], policy=policy.state_dict(), session_ids=[TOKEN])


@pytest.fixture
def cohort(monkeypatch, tmp_path):
    # All writable roots and process probes are explicitly isolated before helper calls.
    root = (tmp_path / "cohort").resolve()
    assert recovery.REPO not in root.parents or recovery.HERE in root.parents
    monkeypatch.setattr(lr, "COHORT_ROOT", root)
    monkeypatch.setattr(lr, "REPO", tmp_path)
    monkeypatch.setattr(lr, "GOAL", tmp_path)
    monkeypatch.setattr(lr, "REQUIRE_MIGRATION", False)
    monkeypatch.setattr(lr, "identity", lambda: copy.deepcopy(IDENTITY))
    monkeypatch.setattr(lr, "contract", lambda: copy.deepcopy(CONTRACT))
    monkeypatch.setattr(validator, "contract", lambda: copy.deepcopy(CONTRACT))
    monkeypatch.setattr(validator, "sequence_validator", lambda: SEQUENCE_VALIDATOR)

    def verify(value):
        if value != IDENTITY:
            raise ValueError("Synthetic source drift")

    monkeypatch.setattr(lr, "verify_identity", verify)
    monkeypatch.setattr(validator, "verify_identity", verify)
    monkeypatch.setattr(wave, "verify_identity", verify)
    boot_calls = []
    monkeypatch.setattr(lr, "boot", lambda *a: boot_calls.append(a))
    monkeypatch.setattr(collector, "boot", lambda *a: pytest.fail("collector boot forbidden"))
    monkeypatch.setattr(collector, "run", lambda *a: pytest.fail("collector run forbidden"))
    monkeypatch.setattr(collector, "BudgetEnv", lambda *a, **k: pytest.fail("environment forbidden"))
    monkeypatch.setattr(wave, "run", lambda *a: pytest.fail("actual wave forbidden"))
    statuses = {100: ("dead", 1000), 200: ("dead", 2000), 300: ("dead", 3000), 999: ("live", 9999)}
    monkeypatch.setattr(lr, "probe_process", lambda pid: statuses.get(pid, ("unknown", None)))
    monkeypatch.setattr(lr, "process_identity", lambda pid=None, command=None:
        dict(pid=999 if pid is None else pid, created=9999 if pid is None else statuses[pid][1], command=command or []))
    job = wave.jobs_for(root)[0]
    folder = root / job["key"]
    attempt = root / "attempts" / ("b" * 32)
    owner = dict(pid=100, created=1000, command=[str(recovery.V2 / "run_wave.py"), "--resume"])
    command = [str(recovery.V2 / "collect.py"), "--abort-file", str(attempt / "ABORT.json")]
    record = dict(key=job["key"], token=TOKEN, state="exited", owner=owner,
                  launcher=dict(pid=200, created=2000, command=command),
                  worker=dict(pid=300, created=3000, command=command),
                  run_id="synthetic-run", cohort_phase="checkpointed", control_steps=75,
                  training_seed=job["training_seed"], exploration_seed=job["exploration_seed"], cpu_mask=job["cpu_mask"])
    settings = dict(format=lr.FORMAT, run_id=record["run_id"], **{k: v for k, v in job.items() if k != "key"},
        identity=IDENTITY, contract=CONTRACT, capacity=6000., physical_config=CONFIG, warmup_ttt=5.,
        profile_sha256="synthetic-training-profile", evaluation=False, model_sha256=None)
    ck = static_checkpoint(settings)
    process = dict(pid=100, command=owner["command"], started=123., attempt=str(attempt))
    error = f"RuntimeError: Child failed/stopped: {job['key']}, exit=1"
    for path, value in {
        root / ".ownership/reservations.json": [record], root / "process.json": process,
        attempt / "process.json": process, attempt / "failure.json": dict(error=error),
        attempt / "ABORT.json": dict(reason="coordinator_draining", created=124.),
        root / "status.json": dict(status="failed", error=error),
        root / "plan.json": dict(format=lr.FORMAT, jobs=wave.jobs_for(root), identity=IDENTITY,
            maximum_workers=5, threads_per_worker=1, episodes=10, transitions=750, training_only=True),
        folder / "settings.json": settings, folder / "observation_schema.json": SCHEMA,
        folder / "status.json": dict(status="failed", pid=300, error=recovery.ERROR),
        folder / "process.json": record["worker"],
        folder / "launcher.json": dict(pid=200, command=command, attempt=str(attempt)),
        folder / "sessions" / f"{TOKEN}.start.json": dict(session_id=TOKEN, process=record["worker"], started=123.),
    }.items():
        lr.save(path, value)
    lr.save(folder / "sessions" / f"{TOKEN}.end.json", dict(session_id=TOKEN,
        start_sha256=lr.file_hash(folder / "sessions" / f"{TOKEN}.start.json"), outcome="failed", elapsed_wall_seconds=12.))
    lr.save(folder / "timing.json", lr.session_timing(folder, [TOKEN]))
    lr.checkpoint_save(lr.torch, folder / "checkpoint.pt", ck)
    (folder / "stdout.log").write_text("synthetic 75/75\n", encoding="utf-8")
    (folder / "stderr.log").write_text(
        f'Traceback (most recent call last):\n  File "{recovery.V2 / "collect.py"}", line 155, in run\n'
        '    save(output / "trace.json", trace)\n'
        f'  File "{recovery.V2 / "local_runtime.py"}", line 42, in save\n'
        '    json.dumps(plain(value), allow_nan=False)\n' + recovery.ERROR + "\n", encoding="utf-8")
    return NS(root=root, folder=folder, ck=ck, settings=settings, record=record,
              attempt=attempt, statuses=statuses, boot_calls=boot_calls, job=job)
