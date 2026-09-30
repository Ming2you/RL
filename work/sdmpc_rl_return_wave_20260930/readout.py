"""Read-only acceptance of exactly five completed slots; no physical checkpoint load."""
import json
import wave_support as w
from actor import authenticate, verify_live
from checks import validate_settings, validate_rows, raw_trace, tag, audit_receipt, summarize
from worker import check_operations


def load_completed(folder, scenario, actor, authentication, sources):
    if folder.resolve() != (w.WAVE / scenario).resolve():
        raise ValueError("Wrong completed slot")
    done = w.read(folder / "completion.json")
    settings = w.read(folder / "settings.json")
    validate_settings(settings, authentication, sources, scenario)
    if (done["format"] != w.FORMAT or done["status"] != "completed" or done["settings"] != settings or
            done["model_sha256"] != w.MODEL_SHA):
        raise ValueError("Completion identity differs")
    names = {"settings.json", "latest.json", "trace.json", "experience.pt", "observation_schema.json",
             "diagnostic-audit.json", "timing.json", "summary.json"}
    if set(done["outputs_sha256"]) != names:
        raise ValueError("Incomplete output manifest")
    for name, expected in done["outputs_sha256"].items():
        w.check_hash(folder / name, expected)
    record, _ = w.checkpoint_record(folder)  # Hash bytes only; no simulator unpickle/boot.
    if (record != done["checkpoint"] or record["control_steps"] != 75 or
            record["settings_digest"] != w.digest(settings)):
        raise ValueError("Completed checkpoint binding differs")
    check_operations(folder, 75)
    schema = w.read(folder / "observation_schema.json")
    if schema != settings["contract"]["observation_schema"]:
        raise ValueError("Observation schema differs")
    payload = w.torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)
    if payload["format"] != w.FORMAT or payload["settings"] != settings:
        raise ValueError("Experience identity differs")
    tagged = w.read(folder / "trace.json")
    trace = raw_trace(tagged)
    if tag(trace)[0] != tagged or audit_receipt(folder, trace, payload) != w.read(folder / "diagnostic-audit.json"):
        raise ValueError("Diagnostic receipt/experience binding differs")
    validate_rows(trace, payload["transitions"], settings, schema, actor)
    timing = w.session_timing(folder, settings, record["session_ids"])
    if timing != w.read(folder / "timing.json") or not timing["session_ids"]:
        raise ValueError("Worker timing differs")
    summary = summarize(trace, settings, timing)
    if summary != w.read(folder / "summary.json"):
        raise ValueError("Summary reconciliation differs")
    return summary


def validate_wave():
    actor, authentication = authenticate()
    sources = w.sources()
    completed = {p.parent.resolve() for p in w.WAVE.rglob("completion.json")}
    expected = {(w.WAVE / s).resolve() for s in w.SCENARIOS}
    if completed != expected:
        raise ValueError("Need exactly five unique completed scenario slots, no missing/extra/duplicate trajectories")
    rows = [load_completed(w.WAVE / s, s, actor, authentication, sources) for s in w.SCENARIOS]
    if len({r["run_id"] for r in rows}) != 5 or {r["model_sha256"] for r in rows} != {w.MODEL_SHA}:
        raise ValueError("Duplicate runs or mixed policy")
    verify_live(authentication, sources)
    return dict(format=w.FORMAT, status="completed", integrity_pass=True,
        all_health_gates_pass=all(all(r["health"].values()) for r in rows), scenarios=rows,
        model_sha256=w.MODEL_SHA, evaluation=False, exploration=False, policy_improvement_claim=False,
        critic_continuation="carry", wave_elapsed_wall_seconds=None,
        timing_note=("Parent owns dispatch timestamps and actual worker drain; worker seconds are not wave wall time. "
            "Per-scenario worker_sessions.restoration is a component of session time, not an additional charge "
            "or an isolated reference-only duration. Saved decision timings remain unchanged."),
        health_note="Each zero-NUF requests <=5 and terminal inventory <=550. Failure is diagnostic; preserve results.",
        next_action="Parent reviews actual readout and decides Task3; no automatic fit, retry or next wave.")


def main():
    print(json.dumps(validate_wave(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
