"""Authenticate training-only collections and report paired coverage, not wins."""
from pathlib import Path
from local_runtime import (np, torch, read, file_hash, digest, SCENARIOS, BEHAVIORS,
                           FORMAT, MASKS, seeds, sequence_validator, verify_identity, contract, plain,
                           session_timing, TIMING_SCOPE)
from compare_runs import finite, same_cost, queue_exposure
from budget_controller import residual_budget
from frozen_inventory import inventory, complete_config
from references import validate_binding
from diagnostics import validate_audit, RECEIPT

CONTROL_FIELDS = ("ramp_metering", "vsl", "green_times", "offsets", "inflow_outflow_allocation")


def validate_observation(obs, k, schema, anchor, requested, executed):
    names = schema["names"]
    obs = np.asarray(obs)
    if obs.shape != (len(names),) or not np.isfinite(obs).all():
        raise ValueError("Observation shape/finite mismatch")
    expected = {"state/time_sec": k*180/14400., "memory/remaining/0": (80-k)/75.}
    for key, budget in (("action_anchor", anchor), ("previous_requested", requested),
                        ("previous_executed", executed)):
        budget = np.asarray(budget)
        if budget.shape != (2,) or not np.isfinite(budget).all():
            raise ValueError("Invalid boundary budget: " + key)
        expected.update({f"memory/{key}/{i}": value for i, value in enumerate(budget/[1000., 10000.])})
    for name, value in expected.items():
        if names.count(name) != 1:
            raise ValueError("Missing/duplicate observation field: " + name)
        np.testing.assert_equal(obs[names.index(name)], np.float32(0. if k == 80 else value))
    if k == 80:
        np.testing.assert_array_equal(obs, np.zeros(len(names), dtype=np.float32))


def physical_config(settings):
    cfg = settings["physical_config"]
    if digest(cfg) != settings["contract"]["environment_contract"]["config_sha256"]:
        raise ValueError("Physical accounting configuration differs")
    return cfg


def validate_boundary(ck, settings):
    env, trace = ck["environment"], ck["trace"]
    n = len(trace)
    cfg = physical_config(settings)
    if complete_config(env["sim"].cfg) != cfg or env["warmup_ttt"] != settings["warmup_ttt"]:
        raise ValueError("Checkpoint physical configuration/accounting differs")
    state = plain(env["sim"].state)
    finite(inventory(state, cfg), "checkpoint inventory")
    totals = {key: finite(getattr(env["sim"], key), "checkpoint accounting " + key)
              for key in ("total_ttt", "freeway_ttt", "urban_ttt")}
    same_cost(totals["freeway_ttt"]+totals["urban_ttt"], totals["total_ttt"], "checkpoint area accounting")
    if n:
        row = trace[-1]
        if state != row["plant_state"] or plain(env["previous"]) != row["control"]:
            raise ValueError("Checkpoint physical state/control boundary differs")
        for key in ("total_ttt", "freeway_ttt", "urban_ttt"):
            same_cost(getattr(env["sim"], key), row[key], "checkpoint " + key)
        same_cost(inventory(state, cfg), row["inventory"], "checkpoint inventory")
        requested, executed = row["B_requested"][0], row["B_executed"]
        np.testing.assert_array_equal(env["last_budget"], executed)
        anchor = executed
    else:
        same_cost(totals["total_ttt"], settings["warmup_ttt"], "checkpoint warmup accounting")
        requested = executed = np.zeros(2)
        if env["last_budget"] is not None or env["prepared"] is None:
            raise ValueError("Initial checkpoint budget boundary differs")
        anchor = env["prepared"]["budget"]
    np.testing.assert_array_equal(env["last_requested"], requested)
    np.testing.assert_array_equal(env["last_executed"], executed)
    if env["k"] < 80:
        if env["prepared"] is None:
            raise ValueError("Missing prepared boundary reference")
        np.testing.assert_array_equal(env["prepared"]["action_anchor"], anchor)
    validate_observation(ck["observation"], env["k"], ck["schema"], anchor, requested, executed)


def validate_rows(trace, experience, settings, schema, capacity, complete=True):
    n = len(trace)
    if n != len(experience) or not 0 <= n <= 75 or (complete and n != 75):
        raise ValueError("Collection length differs")
    total = settings["warmup_ttt"]
    cfg = physical_config(settings)
    for i, (row, transition) in enumerate(zip(trace, experience)):
        obs, action, reward, nxt, done = transition
        if (row["scenario"] != settings["scenario"] or row["behavior"] != settings["behavior"] or
                row["profile_sha256"] != settings["profile_sha256"] or row["control_step"] != i or
                row["step"] != i+5 or row["time_sec"] != (i+6)*180 or
                done is not (i == 74) or row["terminated"] is not done or row["truncated"] is not False):
            raise ValueError("Sequential identity/terminal mismatch")
        for array in (obs, nxt):
            if np.asarray(array).shape != (len(schema["names"]),) or not np.isfinite(array).all():
                raise ValueError("Observation shape/finite mismatch")
        if i and not np.array_equal(obs, experience[i-1][3]):
            raise ValueError("Observation chain mismatch")
        previous_request = trace[i-1]["B_requested"][0] if i else np.zeros(2)
        previous_execution = trace[i-1]["B_executed"] if i else np.zeros(2)
        if i:
            np.testing.assert_array_equal(row["action_anchor"], previous_execution)
        validate_observation(obs, i+5, schema, row["action_anchor"], previous_request, previous_execution)
        validate_observation(nxt, i+6, schema, row["B_executed"], row["B_requested"][0], row["B_executed"])
        raw, request = residual_budget(action, row["action_anchor"], capacity)
        np.testing.assert_allclose(raw, row["B_raw"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(request, row["B_requested"][0], rtol=0, atol=1e-10)
        if row["plant_state"]["time_sec"] != row["time_sec"]:
            raise ValueError("Physical state clock differs")
        same_cost(inventory(row["plant_state"], cfg), row["inventory"], "physical inventory")
        np.testing.assert_array_equal(action, np.asarray(row["action_requested"], dtype=np.float32))
        same_cost(reward, -row["interval_ttt"]/100., "interval reward")
        same_cost(row["reward"], reward, "trace reward")
        total += finite(row["interval_ttt"], "interval TTT")
        same_cost(row["total_ttt"], total, "cumulative TTT")
        same_cost(row["freeway_ttt"]+row["urban_ttt"], total, "area TTT")
        if (row["guard_mode"] != "physical" or row["h3_guard_enabled"] is not False or
                row["execution_check"]["physical_control_valid"] is not True or
                row["execution_check"]["budget_feasible"] is not True or
                row["lower_candidate_count"] != 1 or row["policy_q"] is not None or
                row["learning"] != [] or row["learning_wall_seconds"] != 0):
            raise ValueError("Execution/one-solve/no-learning contract mismatch")
        source = row["reference_source"]
        if (source not in ("previous", "pfo_initial", "pfo_recovery") or
                row["pfo_calls"] != int(source != "previous") or
                row["reference_recovery"] is not (source == "pfo_recovery") or
                (i == 0) != (source == "pfo_initial")):
            raise ValueError("PFO reference contract mismatch")
        for kind in ("wall", "cpu"):
            same_cost(row[f"decision_{kind}_seconds"], sum(finite(row[f"{p}_{kind}_seconds"], p)
                for p in ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor")),
                "decision timing components")
    if complete:
        sequence_validator()(experience, trace, schema["names"], capacity)
        queue_exposure(trace)
    return total


def summarize(trace, settings, timing_status="UNKNOWN"):
    requests = np.array([r["B_requested"][0] for r in trace])
    executed = np.array([r["B_executed"] for r in trace])
    anchor0 = np.array(trace[0]["action_anchor"])
    times = np.array([r["decision_wall_seconds"] for r in trace])
    return dict(scenario=settings["scenario"], behavior=settings["behavior"],
        training_seed=settings["training_seed"], profile_sha256=settings["profile_sha256"],
        evaluation=False, policy_improvement_claim=False, control_steps=len(trace),
        simulation_seconds=trace[-1]["time_sec"], warmup_ttt=settings["warmup_ttt"],
        ttt=trace[-1]["total_ttt"], terminal_inventory=trace[-1]["inventory"],
        interval_ttt=[r["interval_ttt"] for r in trace],
        inventories=[r["inventory"] for r in trace],
        nuf_headroom=[settings["capacity"]-r["B_requested"][0][1] for r in trace],
        nuf_base=[r["exploration_audit"]["base_after"][1] for r in trace],
        fallback_steps=[i for i, r in enumerate(trace) if r["selection_source"] == "reference_fallback"],
        d_ramp_closure_steps=[i for i, r in enumerate(trace)
            if all(k in r["control"]["ramp_metering"] and r["control"]["ramp_metering"][k] <= 1e-6
                   for k in ("R_D_W", "R_D_E"))],
        q_present=False, training_performed=False,
        zero_nuf_requests=int((requests[:, 1] == 0).sum()),
        requested_drift_max_abs=np.abs(requests-anchor0).max(axis=0).tolist(),
        executed_drift_max_abs=np.abs(executed-anchor0).max(axis=0).tolist(),
        projected_request_unique=len({tuple(r) for r in requests}),
        executed_budget_unique=len({tuple(r) for r in executed}),
        physical_control_unique=len({digest({k: r["control"][k] for k in CONTROL_FIELDS}) for r in trace}),
        fallback_count=sum(r["selection_source"] == "reference_fallback" for r in trace),
        pfo_calls=sum(r["pfo_calls"] for r in trace),
        pfo_recovery_count=sum(r["reference_recovery"] for r in trace),
        lower_candidate_solves=sum(r["lower_candidate_count"] for r in trace),
        decision_wall_seconds=float(times.sum()), decision_p50=float(np.median(times)),
        decision_p95=float(np.quantile(times, .95)),
        decision_cpu_seconds=sum(r["decision_cpu_seconds"] for r in trace),
        timing_status=timing_status, timing_ledger="timing.json", timing_scope=TIMING_SCOPE)


def load_completed(folder, expected_identity=None):
    folder = Path(folder)
    result = read(folder / "completion.json")
    settings = read(folder / "settings.json")
    if (result["status"] != "completed" or result["format"] != FORMAT or
            result["settings"] != settings or settings["format"] != FORMAT or
            settings["behavior"] not in BEHAVIORS or settings["scenario"] not in SCENARIOS or
            (settings["training_seed"], settings["exploration_seed"]) != seeds(settings["scenario"]) or
            settings["evaluation"] is not False or settings["model_sha256"] is not None or
            settings["capacity"] != 6000. or settings["contract"] != contract()):
        raise ValueError("Training run identity mismatch")
    if expected_identity is not None and settings["identity"] != expected_identity:
        raise ValueError("Wave/child source identity mismatch")
    verify_identity(settings["identity"])
    validate_binding(settings)
    import local_runtime as runtime
    slot = "local/" + settings["scenario"]
    if folder.resolve() != runtime.COHORT_ROOT.resolve() / slot:
        raise ValueError("Completed output slot differs")
    records = read(runtime.COHORT_ROOT / ".ownership/reservations.json")
    history = [r for r in records if r["key"] == slot]
    if (not history or any(r["run_id"] != settings["run_id"] for r in history) or
            history[-1]["control_steps"] != 75 or history[-1]["cohort_phase"] != "finished" or
            history[-1]["state"] not in ("exited", "completed") or
            any(history[i]["control_steps"] > history[i+1]["control_steps"] for i in range(len(history)-1))):
        raise ValueError("Completed output/retained cohort identity differs")
    for name, expected in result["outputs_sha256"].items():
        if name not in {"experience.pt", "trace.json", "summary.json", "observation_schema.json", "timing.json", RECEIPT}:
            raise ValueError("Unknown output manifest entry")
        if file_hash(folder / name) != expected:
            raise ValueError("Output changed: " + name)
    required = {"experience.pt", "trace.json", "summary.json", "observation_schema.json", "timing.json", RECEIPT}
    if set(result["outputs_sha256"]) != required:
        raise ValueError("Incomplete output manifest")
    schema, trace = read(folder / "observation_schema.json"), read(folder / "trace.json")
    if schema != settings["contract"]["observation_schema"]:
        raise ValueError("Schema mismatch")
    payload = torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)
    if payload["format"] != FORMAT or payload["settings"] != settings:
        raise ValueError("Experience provenance mismatch")
    experience = payload["transitions"]
    validate_audit(folder, trace, payload)
    validate_rows(trace, experience, settings, schema, settings["capacity"])
    replay_policy(trace, settings)
    summary = read(folder / "summary.json")
    finite(summary["terminal_inventory"], "terminal inventory")
    timing = read(folder / "timing.json")
    if not timing["session_ids"] or timing != session_timing(folder, timing["session_ids"]):
        raise ValueError("Timing reconciliation mismatch")
    if summary != summarize(trace, settings, timing["timing_status"]):
        raise ValueError("Summary reconciliation mismatch")
    summary.update(timing)
    return result, settings, summary


def replay_policy(trace, settings):
    # Replaying the small generator authenticates behavior without a plant solve.
    from exploration import LocalBudgetPolicy
    policy = LocalBudgetPolicy(settings["behavior"], settings["exploration_seed"], settings["capacity"])
    for row in trace:
        action, _ = policy.choose(row["action_anchor"])
        np.testing.assert_array_equal(action, np.asarray(row["action_requested"], dtype=np.float32))
        if policy.commit(row) != row["exploration_audit"]:
            raise ValueError("Exploration audit/RNG trajectory differs")
    return policy


