"""Read-only partial/full all-five acceptance; never dispatches reproduction."""
import argparse
import json
import support as s
import checks
import preflight
from policy import authenticate
from worker import check_operations, OUTPUTS


def load_completed(folder, scenario, actor, auth, source):
    if folder.resolve() != (s.ROOT / scenario).resolve():
        raise ValueError("Wrong canonical slot")
    done, settings = s.read(folder / "completion.json"), s.read(folder / "settings.json")
    checks.validate_settings(settings, auth, source, scenario)
    if (done["format"] != s.FORMAT or done["status"] != "completed" or done["settings"] != settings or
            done["model_sha256"] != s.MODEL_SHA or set(done["outputs_sha256"]) != set(OUTPUTS)):
        raise ValueError("Completion identity/output manifest differs")
    for name, expected in done["outputs_sha256"].items():
        s.check_hash(folder / name, expected)
    record, _ = s.checkpoint_record(folder)
    if (record != done["checkpoint"] or record["control_steps"] != 75 or record["settings_digest"] != s.digest(settings)):
        raise ValueError("Completed checkpoint binding differs")
    check_operations(folder, 75)
    schema = s.read(folder / "observation_schema.json")
    if schema != settings["contract"]["observation_schema"]:
        raise ValueError("Observation schema differs")
    payload = s.torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)
    if payload["format"] != s.FORMAT or payload["settings"] != settings or payload["eval_only"] is not True:
        raise ValueError("Evaluation-only evidence identity differs")
    tagged = s.read(folder / "trace.json")
    trace = checks.raw_trace(tagged)
    if checks.tag(trace)[0] != tagged or checks.audit_receipt(folder, trace, payload) != s.read(folder / "diagnostic-audit.json"):
        raise ValueError("Diagnostic/experience binding differs")
    checks.validate_rows(trace, payload["transitions"], settings, schema, actor)
    timing = s.session_timing(folder, settings, record["session_ids"])
    marker = s.read(folder / "finalization.json")
    if (not timing["session_ids"] or timing != s.read(folder / "timing.json") or marker["timing"] != timing or
            marker["settings_digest"] != s.digest(settings) or marker["checkpoint"] != record):
        raise ValueError("Finalization/timing identity differs")
    for name, expected in marker["retained_outputs"].items():
        if name not in OUTPUTS:
            raise ValueError("Finalization output path differs")
        s.check_hash(folder / name, expected)
    summary = checks.summarize(trace, settings, timing)
    if summary != s.read(folder / "summary.json"):
        raise ValueError("Summary reconciliation differs")
    return summary


def matched(candidate, baseline):
    if (candidate["scenario"] != baseline["scenario"] or candidate["profile_sha256"] != baseline["profile_sha256"] or
            candidate["model_sha256"] != s.MODEL_SHA or candidate["control_steps"] != 75 or
            candidate["simulation_seconds"] != 14400 or len(candidate["interval_ttt"]) != 75):
        raise ValueError("Matched canonical comparison identity differs")
    s.same_cost(candidate["warmup_ttt"], baseline["warmup_ttt"], "matched warmup")
    base = s.finite(baseline["ttt"], "baseline TTT")
    value = s.finite(candidate["ttt"], "candidate TTT")
    s.same_cost(base, s.BASE_TTT[candidate["scenario"]], "baseline TTT")
    delta = base-value
    differences = [a-b for a, b in zip(baseline["interval_ttt"], candidate["interval_ttt"])]
    s.same_cost(sum(differences), delta, "matched interval delta")
    threshold = max(1e-6, 1e-8*base)
    return dict(scenario=candidate["scenario"], status="completed", baseline=baseline, candidate=candidate,
        improvement_ttt=delta, improvement_percent=100*delta/base, required_improvement_ttt=threshold,
        improved=delta > threshold, interval_improvement=differences,
        windows={f"{lo+1}..{hi}": dict(baseline_ttt=sum(baseline["interval_ttt"][lo:hi]),
            candidate_ttt=sum(candidate["interval_ttt"][lo:hi]), improvement_ttt=sum(differences[lo:hi]))
            for lo, hi in ((0, 25), (25, 50), (50, 75))})


def acceptance(rows, extras=()):
    complete = len(rows) == 5 and [r["scenario"] for r in rows] == list(s.SCENARIOS) and not extras
    complete = complete and all(r["status"] == "completed" for r in rows)
    if complete:
        ids = [r["candidate"]["run_id"] for r in rows]
        complete = len(set(ids)) == 5 and all(r["candidate"]["model_sha256"] == s.MODEL_SHA for r in rows)
    passed = complete and all(r["improved"] for r in rows)
    return dict(format=s.FORMAT, status="eligible_for_separate_reproduction" if passed else
        "failed_acceptance" if complete else "incomplete_or_invalid", integrity_pass=bool(complete),
        all_five_improved=bool(passed), goal_achieved=False, scenarios=rows, extra_completions=list(extras),
        evaluation=True, eval_only=True, model_sha256=s.MODEL_SHA, automatic_reproduction_dispatch=False,
        parallel_elapsed_wall_seconds=None,
        limitations="Queue exposure is an endpoint estimate. Actor time is not total computation; restore is inside session time. Parallel sums are not wall elapsed. No preview-free or nonlinear-price causation claim. Deterministic repeats alone do not establish stochastic generalization.")


def evaluate(path, partial=False):
    admitted = s.read(path)
    preflight.live(admitted)
    actor, auth = authenticate()
    if auth != admitted["authentication"]:
        raise ValueError("Readout authentication differs")
    expected = {(s.ROOT / scenario / "completion.json").resolve() for scenario in s.SCENARIOS}
    extras = sorted(str(p) for p in s.ROOT.rglob("completion.json") if p.resolve() not in expected)
    rows = []
    for scenario, baseline in zip(s.SCENARIOS, admitted["centers"]["rows"]):
        folder = s.ROOT / scenario
        try:
            if not (folder / "completion.json").exists():
                progress = None
                if (folder / "latest.json").exists():
                    progress, _ = s.checkpoint_record(folder)
                    check_operations(folder, progress["control_steps"])
                rows.append(dict(scenario=scenario, status="partial" if folder.exists() else "absent",
                                 checkpoint_progress=progress, baseline=baseline))
                continue
            candidate = load_completed(folder, scenario, actor, auth, s.sources())
            if s.read(folder / "settings.json")["preflight_sha256"] != s.file_hash(path):
                raise ValueError("Readout preflight differs")
            rows.append(matched(candidate, baseline))
        except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
            rows.append(dict(scenario=scenario, status="invalid", error=f"{type(exc).__name__}: {exc}", baseline=baseline))
    result = acceptance(rows, extras)
    result["partial_requested"] = partial
    preflight.live(admitted)
    s.import_boundary()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=s.Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.preflight, args.partial)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if args.partial or result["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
