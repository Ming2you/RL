"""Canonical metadata adapters around the unchanged physical validator primitives."""
import hashlib
import json
import support as s

legacy = s.old.module("canonical_legacy_checks", s.WAVE_SOURCE / "checks.py",
    s.DEPENDENCIES[(s.WAVE_SOURCE / "checks.py").relative_to(s.REPO).as_posix()])
helpers, tag, raw_trace = legacy.helpers, legacy.tag, legacy.raw_trace
CONTROL_FIELDS = legacy.CONTROL_FIELDS
validate_controls, control_context = legacy.validate_controls, legacy.control_context


def verify_environment(env, rt, expected, schema=False):
    physical = expected["environment_contract"]
    if (s.digest(rt["rc"].to_plain_dict(rt["cfg"])) != physical["config_sha256"] or
            s.digest(rt["rc"].to_plain_dict(rt["options"])) != physical["options_sha256"]):
        raise ValueError("Physical configuration/options differ")
    actual = {k: v for k, v in env.contract().items()
              if k not in ("cfg", "options", "observation", "source_snapshot")}
    canonical = next(r for r in physical["scenarios"] if r["scenario"] == env.protocol["scenario"])
    if actual != physical["coordinator"] or env.training_seed is not None or env.profile_hash != canonical["profile_sha256"]:
        raise ValueError("Canonical environment/profile/seed differs")
    if schema and s.observation_schema(env) != expected["observation_schema"]:
        raise ValueError("Observation schema drift")


def metadata(scenario, auth, source):
    return dict(format=s.FORMAT, scenario=scenario, behavior="frozen_shared_actor",
        mode="canonical_evaluation", training_seed=None, profile_sha256=auth["profiles"][scenario],
        evaluation=True, eval_only=True, exploration=False, policy_q=None, policy_improvement_claim=False,
        model_sha256=s.MODEL_SHA, authentication=auth, sources=source, contract=auth["contract"],
        spec=s.SPEC, spec_sha256=s.SPEC_SHA, capacity=6000., cpu_mask=s.MASKS[s.SCENARIOS.index(scenario)],
        reward_divisor=100., gamma=1., warmup_steps=5, controlled_steps=75, total_seconds=14400,
        critic_continuation=s.CONTINUATION, updates_per_interval=0, threads=1)


def validate_settings(settings, auth, source, scenario):
    expected = metadata(scenario, auth, source)
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Canonical settings/profile/model/contract mismatch")
    rid = settings["run_id"]
    if len(rid) != 32 or any(c not in "0123456789abcdef" for c in rid):
        raise ValueError("Invalid run identity")
    s.finite(settings["warmup_ttt"], "warmup TTT")
    helpers().physical_config(settings)
    control_context(settings)
    canonical = next(r for r in settings["contract"]["environment_contract"]["scenarios"] if r["scenario"] == scenario)
    if settings["profile_sha256"] != canonical["profile_sha256"]:
        raise ValueError("Evaluation requires canonical profile")


def physical_rows(trace, settings):
    tag(trace)
    validate_controls(trace, settings)
    cfg = helpers().physical_config(settings)
    for row in trace:
        if (row["selection_source"] not in ("lower_solution", "reference_fallback") or
                len(row["B_requested"]) != 1 or len(row["candidates"]) != 1 or
                "H3_TTT_guard" in row["fallback_reasons"] or row["guard_mode"] != "physical" or
                row["h3_guard_enabled"] is not False or row.get("policy_q") is not None or
                row["learning"] != [] or row["learning_wall_seconds"] != 0):
            raise ValueError("Physical one-candidate/no-Q/no-learning contract differs")
        check = row["execution_check"]
        if check["physical_control_valid"] is not True or check["budget_feasible"] is not True:
            raise ValueError("Physical execution/fallback invalid")
        for left, right in ((row["B_executed"], check["budget"]), (row["G_achieved"], check["achieved"]),
                            (row["selected_identity"]["point"], check["point"])):
            s.np.testing.assert_array_equal(left, right)
        s.same_cost(row["selected_TTT"], check["ttt"], "selected prediction")
        if s.np.any(s.np.asarray(check["achieved"]) > s.np.asarray(check["budget"])):
            raise ValueError("Executed upper budget violated")
        s.np.testing.assert_array_equal(check["band_excess"], [0., 0.])
        if not 0 <= row["B_executed"][1] <= settings["capacity"]:
            raise ValueError("Executed NUF outside capacity")
        control = row["control"]
        for key in CONTROL_FIELDS:
            s.finite_tree(control[key])
        s.np.testing.assert_array_equal([control["N_P_star"], control["N_UF_star"]], row["B_executed"])
        for key, values in row["queue_state"].items():
            if values != row["plant_state"][key]:
                raise ValueError("Queue/physical state differs")
        s.same_cost(helpers().inventory(row["plant_state"], cfg), row["inventory"], "physical inventory")
        for part in ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor", "decision", "plant"):
            for kind in ("wall", "cpu"):
                s.finite(row[f"{part}_{kind}_seconds"], "timing")
    if trace:
        s.old.queue_exposure(trace)


def validate_rows(trace, transitions, settings, schema, actor, complete=True):
    s.finite_tree(transitions)
    total = helpers().validate_rows(trace, transitions, settings, schema, settings["capacity"], complete)
    physical_rows(trace, settings)
    for row, (obs, action, _, nxt, _) in zip(trace, transitions):
        if any(not isinstance(a, s.np.ndarray) or a.dtype != s.np.float32 for a in (obs, action, nxt)):
            raise ValueError("Native float32 arrays required")
        if actor.act(obs).tobytes() != action.tobytes():
            raise ValueError("Actor action not bitexact")
        if (row["model_sha256"] != s.MODEL_SHA or row["mode"] != "canonical_evaluation" or
                row["evaluation"] is not True or row["eval_only"] is not True or row["exploration"] is not False):
            raise ValueError("Canonical trace metadata differs")
    return total


# Reuse raw checkpoint and diagnostic verification without changing their bodies.
_ck = dict(w=s, helpers=helpers, validate_rows=validate_rows)
s.reuse("checks.py", ("validate_checkpoint",), _ck)
validate_checkpoint = _ck["validate_checkpoint"]
s.TAG_SHA = s.old.TAG_SHA
s.reuse("checks.py", ("audit_receipt",), _audit := dict(w=s, tag=tag, hashlib=hashlib, json=json))
audit_receipt = _audit["audit_receipt"]


def summarize(trace, settings, timing):
    requests = s.np.array([r["B_requested"][0] for r in trace])
    executed = s.np.array([r["B_executed"] for r in trace])
    actions = s.np.array([r["action_requested"] for r in trace], dtype=s.np.float32)
    inventories = [r["inventory"] for r in trace]
    times = {}
    for part in ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor", "decision", "plant"):
        for kind in ("wall", "cpu"):
            values = s.np.array([r[f"{part}_{kind}_seconds"] for r in trace])
            times[f"{part}_{kind}"] = dict(sum_seconds=float(values.sum()), p50_seconds=float(s.np.median(values)),
                p95_seconds=float(s.np.quantile(values, .95)), max_seconds=float(values.max()))
    return dict(format=s.FORMAT, run_id=settings["run_id"], scenario=settings["scenario"],
        model_sha256=s.MODEL_SHA, profile_sha256=settings["profile_sha256"], evaluation=True, eval_only=True,
        training_seed=None, exploration=False, policy_q=None, training_performed=False,
        control_steps=len(trace), simulation_seconds=trace[-1]["time_sec"], warmup_ttt=settings["warmup_ttt"],
        ttt=trace[-1]["total_ttt"], freeway_ttt=trace[-1]["freeway_ttt"], urban_ttt=trace[-1]["urban_ttt"],
        interval_ttt=[r["interval_ttt"] for r in trace], inventories=inventories,
        terminal_inventory=inventories[-1], peak_inventory=max(inventories),
        requested_min=requests.min(axis=0).tolist(), requested_max=requests.max(axis=0).tolist(),
        executed_min=executed.min(axis=0).tolist(), executed_max=executed.max(axis=0).tolist(),
        projected_request_unique=len({tuple(r) for r in requests}), executed_budget_unique=len({tuple(r) for r in executed}),
        physical_control_unique=len({s.digest({k: r["control"][k] for k in CONTROL_FIELDS}) for r in trace}),
        zero_nuf_requests=int((requests[:, 1] == 0).sum()), nuf_cap_requests=int((requests[:, 1] == 6000.).sum()),
        action_exact_saturation_count=(abs(actions) == s.np.array([.2, .1], dtype=s.np.float32)).sum(axis=0).tolist(),
        fallback_steps=[i+1 for i, r in enumerate(trace) if r["selection_source"] == "reference_fallback"],
        fallback_count=sum(r["selection_source"] == "reference_fallback" for r in trace),
        pfo_calls=sum(r["pfo_calls"] for r in trace), pfo_recovery_count=sum(r["reference_recovery"] for r in trace),
        lower_candidate_solves=sum(r["lower_candidate_count"] for r in trace),
        queue_exposure=s.old.queue_exposure(trace), timing=times, worker_sessions=timing,
        timing_note="Actor time is only a decision component; restore is already inside session time once. Parallel sums are not wall elapsed.")

