"""Pure frozen accounting reuse, actor replay and diagnostic-only trace export."""
import hashlib
import json
from functools import lru_cache
from types import SimpleNamespace
import wave_support as w

CONTROL_FIELDS = ("ramp_metering", "vsl", "green_times", "offsets", "inflow_outflow_allocation")


@lru_cache(maxsize=1)
def helpers():
    inv = w.module("wave_frozen_inventory", w.NUF / "frozen_inventory.py",
        "9d8204d860df1a673f0e6f59941cb4a0ace35aa04fedcd6f79fa36de8201feaa")
    ns = dict(np=w.np, plain=w.plain, digest=w.digest, finite=w.finite, same_cost=w.same_cost,
        residual_budget=w.residual_budget, inventory=inv.inventory, complete_config=inv.complete_config,
        queue_exposure=w.queue_exposure, observation_schema=w.observation_schema)
    inv.accounting()
    state = w.sys.modules["local_frozen_state"]
    ns.update(ControlAction=state.ControlAction, segment_vsl=state.segment_vsl,
              NetworkConfig=state.NetworkConfig)
    w.definitions(w.DEFAULT_SNAPSHOT / "work/sdmpc_externality_ablation_20260923/prox_controller.py",
        ("Coordinates",), ns, "89a57ae4311a7ec5e27aa97d4c2b7ac78375fddbef8926ce4609cf59c1a9161f")
    w.definitions(w.REPO / "work/sdmpc_rl_value_audit_20260930/projection_audit.py",
        ("column", "validate_sequence"), ns,
        "ab3787b94c8bbc4a1e76a9d04b942159a01c6753b292b6d6f5d9c21a256b86db")
    ns["sequence_validator"] = lambda: ns["validate_sequence"]
    w.definitions(w.NUF / "validate.py", ("validate_observation", "physical_config", "validate_boundary", "validate_rows"),
        ns, "7110993c8dff56ff399f0f843d23d64854a6800f099d8ccc7c4716c34086567f")
    w.definitions(w.NUF / "local_runtime.py", ("verify_environment",), ns,
        "1cf91506f5f3d4ec209854b46d5318a06ba27f1983259f859d1d658622bbf4e1")
    w.import_boundary()
    return SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


def control_context(settings):
    h = helpers()
    config = settings["physical_config"]
    values = config["network"]
    network = SimpleNamespace(**values)
    network.effective_green_total = h.NetworkConfig.effective_green_total.fget(network)
    cfg = SimpleNamespace(network=network,
        freeway_follower=SimpleNamespace(**config["freeway_follower"]),
        urban_follower=SimpleNamespace(**config["urban_follower"]))
    options = settings["physical_options"]
    if w.digest(options) != settings["contract"]["environment_contract"]["options_sha256"]:
        raise ValueError("Physical options differ")
    return cfg, SimpleNamespace(**options), h.ControlAction.uncontrolled(cfg)


def validate_controls(trace, settings):
    h = helpers()
    cfg, options, previous = control_context(settings)
    for row in trace:
        control = h.ControlAction(**row["control"])
        checked = h.Coordinates(cfg, options, previous).validate(control, discrete=True)
        if checked["valid"] is not True:
            raise ValueError("Frozen physical control constraints failed: " + str(checked["violations"]))
        previous = control


def tag(trace):
    helper = w.module("wave_diagnostic_tagger", w.TAG_PATH, w.TAG_SHA)
    return helper.tag_diagnostics(trace)


def raw_trace(tagged):
    if isinstance(tagged, dict):
        if "nonfinite_float" in tagged:
            if tagged != {"nonfinite_float": "+inf"}:
                raise ValueError("Unknown diagnostic tag")
            return float("inf")
        return {k: raw_trace(v) for k, v in tagged.items()}
    if isinstance(tagged, list):
        return [raw_trace(v) for v in tagged]
    return tagged


def validate_settings(settings, authentication, sources, scenario):
    expected = dict(format=w.FORMAT, scenario=scenario, behavior="frozen_shared_actor",
        mode="training_wave", training_seed=7301+w.SCENARIOS.index(scenario),
        profile_sha256=authentication["profiles"][scenario], evaluation=False, exploration=False,
        policy_improvement_claim=False, model_sha256=w.MODEL_SHA, authentication=authentication,
        sources=sources, contract=authentication["contract"], capacity=6000.,
        cpu_mask=w.MASKS[w.SCENARIOS.index(scenario)], reward_divisor=100., gamma=1.,
        warmup_steps=5, controlled_steps=75, total_seconds=14400, critic_continuation="carry",
        updates_per_interval=0, threads=1)
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Wave settings/profile/model/contract mismatch")
    rid = settings["run_id"]
    if len(rid) != 32 or any(c not in "0123456789abcdef" for c in rid):
        raise ValueError("Invalid run identity")
    w.finite(settings["warmup_ttt"], "warmup TTT")
    helpers().physical_config(settings)
    control_context(settings)
    canonical = next(r for r in settings["contract"]["environment_contract"]["scenarios"]
                     if r["scenario"] == scenario)
    if settings["profile_sha256"] == canonical["profile_sha256"]:
        raise ValueError("Canonical evaluation data forbidden")


def validate_rows(trace, transitions, settings, schema, actor, complete=True):
    tag(trace)  # Only the pinned diagnostic paths may contain +Inf.
    w.finite_tree(transitions)
    total = helpers().validate_rows(trace, transitions, settings, schema, settings["capacity"], complete)
    validate_controls(trace, settings)
    for row, (obs, action, reward, nxt, done) in zip(trace, transitions):
        if any(not isinstance(a, w.np.ndarray) or a.dtype != w.np.float32 for a in (obs, action, nxt)):
            raise ValueError("Experience must preserve native float32 arrays")
        w.np.testing.assert_array_equal(actor.act(obs), action)
        if (row["model_sha256"] != settings["model_sha256"] or row["mode"] != "training_wave" or
                row["evaluation"] is not False or row["exploration"] is not False or
                row["selection_source"] not in ("lower_solution", "reference_fallback") or
                len(row["B_requested"]) != 1 or len(row["candidates"]) != 1 or
                "H3_TTT_guard" in row["fallback_reasons"]):
            raise ValueError("Frozen actor/physical guard/one-candidate contract differs")
        for key in ("interval_ttt", "total_ttt", "freeway_ttt", "urban_ttt", "inventory",
                    "selected_TTT", "reference_TTT", "plant_wall_seconds", "plant_cpu_seconds"):
            w.finite(row[key], key)
        check = row["execution_check"]
        for left, right in ((row["B_executed"], check["budget"]),
                            (row["G_achieved"], check["achieved"]),
                            (row["selected_identity"]["point"], check["point"])):
            w.np.testing.assert_array_equal(left, right)
        w.same_cost(row["selected_TTT"], check["ttt"], "selected prediction")
        if w.np.any(w.np.asarray(check["achieved"]) > w.np.asarray(check["budget"])):
            raise ValueError("Executed upper budget violated")
        w.np.testing.assert_array_equal(check["band_excess"], [0., 0.])
        if not 0 <= row["B_executed"][1] <= settings["capacity"]:
            raise ValueError("Executed NUF outside capacity")
        control = row["control"]
        for key in CONTROL_FIELDS:
            w.finite_tree(control[key])
        w.np.testing.assert_array_equal([control["N_P_star"], control["N_UF_star"]], row["B_executed"])
        for key, values in row["queue_state"].items():
            if values != row["plant_state"][key]:
                raise ValueError("Queue state/physical state mismatch")
    if trace:
        w.queue_exposure(trace)
    return total


def validate_checkpoint(ck, settings, actor):
    if (ck["format"] != w.FORMAT or ck["settings"] != settings or ck["model_sha256"] != w.MODEL_SHA or
            ck["schema"] != settings["contract"]["observation_schema"]):
        raise ValueError("Checkpoint identity/schema differs")
    env = ck["environment"]
    expected_contract = dict(settings["contract"]["environment_contract"]["coordinator"],
        cfg=settings["physical_config"], options=settings["physical_options"],
        observation=settings["contract"]["observation_schema"]["normalization"],
        source_snapshot=settings["authentication"]["physical_snapshot"])
    if env["contract"] != expected_contract or ck["observation"].dtype != w.np.float32:
        raise ValueError("Checkpoint environment/normalization/options contract differs")
    n = len(ck["trace"])
    if (env["k"] != n+5 or env["profile_hash"] != settings["profile_sha256"] or
            env["sim"].state.time_sec != (n+5)*180):
        raise ValueError("Checkpoint physical clock/profile boundary differs")
    w.finite_tree(w.plain(env["sim"].state))
    w.finite_tree(w.plain(env["previous"]))
    for key in ("dual", "last_requested", "last_executed", "last_slack", "last_budget",
                "reference_timing", "observation_timing"):
        w.finite_tree(env[key])
    total = validate_rows(ck["trace"], ck["transitions"], settings, ck["schema"], actor, complete=False)
    w.same_cost(env["sim"].total_ttt, total, "checkpoint cumulative TTT")
    helpers().validate_boundary(ck, settings)
    if n:
        w.np.testing.assert_array_equal(ck["observation"], ck["transitions"][-1][3])
    return ck


def audit_receipt(output, trace, payload):
    tagged, changes = tag(trace)
    w.finite_tree(payload)
    record, _ = w.checkpoint_record(output)
    return dict(format=w.FORMAT+"-diagnostic-audit", run_id=payload["settings"]["run_id"],
        helper_sha256=w.TAG_SHA, changed_paths=changes, checkpoint=record,
        trace_sha256=w.file_hash(output / "trace.json"),
        raw_trace_digest=hashlib.sha256(json.dumps(trace, sort_keys=True, allow_nan=True).encode()).hexdigest(),
        experience_sha256=w.file_hash(output / "experience.pt"), experience_digest=w.digest(payload),
        settings_digest=w.digest(payload["settings"]), core_experience_unchanged=True,
        convergence_claim=False, physical_or_training_change=False)


def summarize(trace, settings, timing):
    requests = w.np.array([r["B_requested"][0] for r in trace])
    executed = w.np.array([r["B_executed"] for r in trace])
    actions = w.np.array([r["action_requested"] for r in trace], dtype=w.np.float32)
    anchor0 = w.np.array(trace[0]["action_anchor"])
    inventories = [r["inventory"] for r in trace]
    fallback = [i for i, r in enumerate(trace) if r["selection_source"] == "reference_fallback"]
    closures = [i for i, r in enumerate(trace) if all(k in r["control"]["ramp_metering"] and
        r["control"]["ramp_metering"][k] <= 1e-6 for k in ("R_D_W", "R_D_E"))]
    times = {}
    for part in ("actor", "decision", "plant"):
        for kind in ("wall", "cpu"):
            values = w.np.array([r[f"{part}_{kind}_seconds"] for r in trace])
            times[f"{part}_{kind}"] = dict(sum_seconds=float(values.sum()),
                p50_seconds=float(w.np.median(values)), p95_seconds=float(w.np.quantile(values, .95)),
                max_seconds=float(values.max()))
    zero = int((requests[:, 1] == 0).sum())
    return dict(format=w.FORMAT, run_id=settings["run_id"], scenario=settings["scenario"],
        mode="training_wave", training_seed=settings["training_seed"], profile_sha256=settings["profile_sha256"],
        model_sha256=settings["model_sha256"], evaluation=False, exploration=False, policy_improvement_claim=False,
        critic_continuation="carry", q_present=False, training_performed=False, control_steps=len(trace),
        simulation_seconds=trace[-1]["time_sec"], warmup_ttt=settings["warmup_ttt"], ttt=trace[-1]["total_ttt"],
        freeway_ttt=trace[-1]["freeway_ttt"], urban_ttt=trace[-1]["urban_ttt"],
        interval_ttt=[r["interval_ttt"] for r in trace], inventories=inventories,
        terminal_inventory=inventories[-1], peak_inventory=max(inventories), zero_nuf_requests=zero,
        nuf_cap_requests=int((requests[:, 1] == settings["capacity"]).sum()),
        action_exact_saturation_count=(abs(actions) == w.np.array([.2, .1], dtype=w.np.float32)).sum(axis=0).tolist(),
        action_99pct_saturation_count=(abs(actions) >= .99*w.np.array([.2, .1], dtype=w.np.float32)).sum(axis=0).tolist(),
        projected_request_unique=len({tuple(r) for r in requests}),
        executed_budget_unique=len({tuple(r) for r in executed}),
        physical_control_unique=len({w.digest({k: r["control"][k] for k in CONTROL_FIELDS}) for r in trace}),
        requested_drift_max_abs=abs(requests-anchor0).max(axis=0).tolist(),
        executed_drift_max_abs=abs(executed-anchor0).max(axis=0).tolist(),
        requested_min=requests.min(axis=0).tolist(), requested_max=requests.max(axis=0).tolist(),
        fallback_count=len(fallback), fallback_steps=fallback, d_ramp_closure_steps=closures,
        pfo_calls=sum(r["pfo_calls"] for r in trace), pfo_recovery_count=sum(r["reference_recovery"] for r in trace),
        lower_candidate_solves=sum(r["lower_candidate_count"] for r in trace),
        queue_exposure=w.queue_exposure(trace), timing=times, worker_sessions=timing,
        health=dict(zero_nuf_requests_le_5=zero <= 5, terminal_inventory_le_550=inventories[-1] <= 550),
        timing_note=("Actor inference is one component of full decision latency. Saved decision timings "
            "are inherited unchanged on restore. Additional whole-restore cost is separately identified "
            "in worker_sessions.restoration and already included once in worker session wall time; "
            "it is not charged to any decision. No wave elapsed wall estimate."),
        comparison_note="New training profiles have no paired carry; no traffic improvement comparison.")
