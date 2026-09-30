"""Whole-call restore accounting; no change to the frozen restoration path."""
import copy
import time
import wave_support as w

PINS = {
    "budget_env.py": "6ccb4ef400106e29282797ec5d69b7a8dbeee5a1310bd7d6f97b7c7e0c7ee2c4",
    "budget_controller.py": "0402511ae8e965e315b4b9808e23bbdc9c2fd274666a983b2a3303e6b916f9c2",
}
SCOPE = (
    "Wall and process CPU around the entire unchanged env.restore call only, including "
    "validation, copies, observer work and reference reconstruction. Not an isolated "
    "reference-only duration. This extra restore cost is already inside worker session "
    "wall time; never add it to that total or to saved decision timings. Reconstruction "
    "counts are source-derived from the pinned k<80 branch, not measured preview counts. "
    "An interrupted call has UNKNOWN duration/completion; a raised call has measured "
    "duration but UNKNOWN reconstruction completion.")
TIMINGS = ("reference_timing", "observation_timing")


def files(output, sid):
    return {p.relative_to(output).as_posix(): w.file_hash(p)
            for kind in ("start", "end")
            if (p := output / "sessions" / f"{sid}.restore-{kind}.json").exists()}


def restore_environment(env, checkpoint, output, sid, settings, record):
    for name, sha in PINS.items():
        w.check_hash(w.OLD / name, sha)
    start = output / "sessions" / f"{sid}.restore-start.json"
    end = output / "sessions" / f"{sid}.restore-end.json"
    if start.exists() or end.exists():
        raise FileExistsError("Restore already attempted in this session")
    inherited = copy.deepcopy({key: w.plain(checkpoint[key]) for key in TIMINGS})
    w.save_once(start, dict(session_id=sid, run_id=settings["run_id"],
        session_start_sha256=w.file_hash(output / "sessions" / f"{sid}.start.json"),
        settings_digest=w.digest(settings), checkpoint=record, checkpoint_k=checkpoint["k"],
        expected_reference_reconstructions=int(checkpoint["k"] < 80),
        frozen_source_sha256=PINS, inherited_decision_timing=inherited, scope=SCOPE))
    outcome = "raised"
    tick, cpu = time.perf_counter(), time.process_time()
    try:
        observation = env.restore(checkpoint)
        outcome = "returned"
    finally:
        elapsed, cpu_elapsed = time.perf_counter()-tick, time.process_time()-cpu
        after = {key: w.plain(getattr(env, key, None)) for key in TIMINGS} if outcome == "returned" else None
        w.save_once(end, dict(session_id=sid, restore_start_sha256=w.file_hash(start),
            outcome=outcome, elapsed_wall_seconds=elapsed, elapsed_cpu_seconds=cpu_elapsed,
            inherited_decision_timing_after=after))
    if after != inherited:
        raise ValueError("Restore changed inherited decision timing")
    return observation


def session_record(output, sid, settings, session_start, session_end):
    """Read only JSON and checkpoint hashes, never deserialize physical state."""
    hashes = files(output, sid)
    if session_end is not None and session_end.get("restoration_sha256") != hashes:
        raise ValueError("Session restoration binding differs")
    start = output / "sessions" / f"{sid}.restore-start.json"
    end = output / "sessions" / f"{sid}.restore-end.json"
    resume = session_start.get("resume_requested")
    if type(resume) is not bool:
        raise ValueError("Session restore intent missing")
    result = dict(session_id=sid, records_sha256=hashes, timing_status="KNOWN",
                  elapsed_wall_seconds=0., elapsed_cpu_seconds=0.,
                  expected_reference_reconstructions=None,
                  returned_branch_reference_reconstructions=0, outcome="not_started")
    if not start.exists():
        if end.exists():
            raise ValueError("Restore end without start")
        if resume and session_end is None:
            result.update(outcome="UNKNOWN", timing_status="UNKNOWN", elapsed_wall_seconds=None,
                          elapsed_cpu_seconds=None, returned_branch_reference_reconstructions=None)
        return result
    saved = w.read(start)
    k, record = saved["checkpoint_k"], saved["checkpoint"]
    if (not resume or type(k) is not int or not 5 <= k <= 80 or
            saved["session_id"] != sid or saved["run_id"] != settings["run_id"] or
            saved["session_start_sha256"] != w.file_hash(output / "sessions" / f"{sid}.start.json") or
            saved["settings_digest"] != w.digest(settings) or
            session_start["settings_digest"] != w.digest(settings) or
            record["settings_digest"] != w.digest(settings) or record["control_steps"] != k-5 or
            saved["expected_reference_reconstructions"] != int(k < 80) or
            saved["frozen_source_sha256"] != PINS or saved["scope"] != SCOPE or
            set(saved["inherited_decision_timing"]) != set(TIMINGS)):
        raise ValueError("Restore checkpoint/session/source binding differs")
    path = (output / record["path"]).resolve()
    if path.parent != (output / "checkpoints").resolve() or path.suffix != ".pt":
        raise ValueError("Restore checkpoint escaped slot")
    w.check_hash(path, record["sha256"])
    result.update(provenance=saved, outcome="UNKNOWN", timing_status="UNKNOWN",
                  elapsed_wall_seconds=None, elapsed_cpu_seconds=None,
                  expected_reference_reconstructions=int(k < 80),
                  returned_branch_reference_reconstructions=None)
    if not end.exists():
        return result
    finish = w.read(end)
    if (finish["session_id"] != sid or finish["restore_start_sha256"] != w.file_hash(start) or
            finish["outcome"] not in ("returned", "raised")):
        raise ValueError("Restore end binding differs")
    for key in ("elapsed_wall_seconds", "elapsed_cpu_seconds"):
        value = w.finite(finish[key], "restore elapsed")
        if value < 0:
            raise ValueError("Negative restore duration")
        result[key] = value
    if session_end is not None and result["elapsed_wall_seconds"] > session_end["elapsed_wall_seconds"]:
        raise ValueError("Restore duration exceeds enclosing session")
    after = finish["inherited_decision_timing_after"]
    if finish["outcome"] == "returned":
        if after != saved["inherited_decision_timing"]:
            raise ValueError("Restore changed inherited decision timing")
        result["returned_branch_reference_reconstructions"] = int(k < 80)
    elif after is not None:
        raise ValueError("Raised restore cannot attest inherited decision timing")
    result.update(outcome=finish["outcome"], timing_status="KNOWN",
                  inherited_decision_timing_after=after)
    return result


def summarize(records):
    unknown = [r["session_id"] for r in records if r["timing_status"] == "UNKNOWN"]
    incomplete = [r["session_id"] for r in records if r["returned_branch_reference_reconstructions"] is None]
    result = dict(scope=SCOPE, records=records, timing_status="UNKNOWN" if unknown else "KNOWN",
        unknown_timing_sessions=unknown, unknown_reconstruction_sessions=incomplete,
        reference_reconstruction_status="UNKNOWN" if incomplete else "SOURCE_DERIVED",
        returned_branch_reference_reconstructions=None if incomplete else
            sum(r["returned_branch_reference_reconstructions"] for r in records))
    for key in ("elapsed_wall_seconds", "elapsed_cpu_seconds"):
        known = sum(r[key] for r in records if r[key] is not None)
        result["known_"+key] = known
        result[key] = None if unknown else known
    return result
