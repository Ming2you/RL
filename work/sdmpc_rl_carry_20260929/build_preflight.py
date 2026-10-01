"""Freeze inspectable evidence before admitting the bounded traffic pilot."""
import argparse
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from budget_runtime import DEFAULT_SNAPSHOT, read, save
from run_budget import file_hash, pins


PROBE_MODES = ["v1_pfo_h3", "carry_physical"]
PROBE_ACTIONS = {"zero": [0., 0.], "np_plus": [.25, 0.], "np_minus": [-.25, 0.],
                 "nuf_plus": [0., .25], "nuf_minus": [0., -.25]}


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Nonfinite or nonnumeric probe value")
    return value


def vector(value, size=None):
    if not isinstance(value, list) or not value or (size is not None and len(value) != size):
        raise ValueError("Invalid probe vector shape")
    return [finite(item) for item in value]


def same_vector(actual, expected):
    actual, expected = vector(actual), vector(expected)
    return len(actual) == len(expected) and all(abs(a - b) <= 1e-9 for a, b in zip(actual, expected))


def validate_probe(folder, step, current_pins, read_record):
    row = read_record(folder / "completion.json")
    settings = row["settings"]
    if (row.get("status") != "completed" or row.get("step") != step or
            settings.get("step") != step or settings.get("source_pins") != current_pins or
            settings.get("modes") != PROBE_MODES or settings.get("actions") != PROBE_ACTIONS or
            read_record(folder / "settings.json") != settings):
        raise ValueError("Probe state, modes, actions or source contract differs")
    for label, action in settings["actions"].items():
        if vector(action, 2) != PROBE_ACTIONS[label]:
            raise ValueError("Probe action values differ")
    expected = {(mode, label) for mode in PROBE_MODES for label in PROBE_ACTIONS}
    summaries = {}
    for summary in row["records"]:
        key = summary["mode"], summary["candidate"]
        if key not in expected or key in summaries or vector(summary["action"], 2) != PROBE_ACTIONS[key[1]]:
            raise ValueError("Duplicate or mislabeled probe summary")
        summaries[key] = summary
    if set(summaries) != expected:
        raise ValueError("Incomplete probe summary records")
    detail_names = {f"{mode}_{label}.json" for mode, label in expected}
    if {path.name for path in folder.glob("*.json")} - {"completion.json", "settings.json", "status.json"} != detail_names:
        raise ValueError("Missing or unexpected detailed probe records")
    details = {}
    for mode, label in sorted(expected):
        detail = read_record(folder / f"{mode}_{label}.json")
        if (detail["settings"] != settings or detail["mode"] != mode or detail["candidate"] != label or
                vector(detail["action"], 2) != PROBE_ACTIONS[label] or detail["full_run"] is not False or
                finite(detail["original_state_seconds"]) != step * 180 or
                finite(detail["reached_state"]["time_sec"]) != (step + 1) * 180):
            raise ValueError("Detailed probe identity or current-state timeline differs")
        audit, point = detail["audit"], vector(detail["physical_point"])
        check = audit["execution_check"]
        budget, achieved = vector(audit["B_executed"], 2), vector(audit["G_achieved"], 2)
        if (check["physical_control_valid"] is not True or check["budget_feasible"] is not True or
                not same_vector(check["point"], point) or
                not same_vector(audit["selected_identity"]["point"], point) or
                not same_vector(check["budget"], budget) or not same_vector(check["achieved"], achieved) or
                any(g > b for g, b in zip(achieved, budget))):
            raise ValueError("Detailed probe execution is invalid or disagrees with executed control")
        if len(audit["B_requested"]) != 1:
            raise ValueError("Probe requires exactly one requested budget")
        vector(audit["B_requested"][0], 2)
        selected_ttt, reference_ttt = finite(audit["selected_TTT"]), finite(audit["reference_TTT"])
        if min(selected_ttt, reference_ttt, finite(detail["interval_ttt"]),
               finite(detail["pfo_seconds"]), finite(detail["decision_seconds"])) < 0:
            raise ValueError("Negative probe cost or timing")
        if abs(finite(check["ttt"]) - selected_ttt) > 1e-8:
            raise ValueError("Probe execution TTT differs")
        source = "pfo_current" if mode == "v1_pfo_h3" else "pfo_initial" if step == 5 else "previous"
        if (detail["source"] != source or type(detail["pfo_calls"]) is not int or
                detail["pfo_calls"] != int(source != "previous") or
                (source == "previous" and detail["pfo_seconds"] != 0)):
            raise ValueError("Detailed probe PFO calls or reference source differs")
        if mode == "carry_physical":
            if (audit["reference_source"] != source or audit["guard_mode"] != "physical" or
                    audit["h3_guard_enabled"] is not False or
                    audit["selection_source"] not in ("lower_solution", "reference_fallback")):
                raise ValueError("Detailed carry reference or guard differs")
        elif selected_ttt > reference_ttt or audit["selection_source"] not in ("lower_solution", "PFO_reference"):
            raise ValueError("Detailed v1 H3 guard or selection differs")
        details[mode, label] = detail
    influenced = False
    for (mode, label), detail in details.items():
        zero, audit = details[mode, "zero"], detail["audit"]
        if len(detail["physical_point"]) != len(zero["physical_point"]):
            raise ValueError("Probe control coordinate dimensions differ")
        changed = not same_vector(detail["physical_point"], zero["physical_point"])
        derived = dict(mode=mode, candidate=label, action=detail["action"],
            changed_physical_control=changed, interval_ttt=detail["interval_ttt"],
            delta_vs_zero=detail["interval_ttt"] - zero["interval_ttt"],
            requested=audit["B_requested"][0], executed=audit["B_executed"], source=detail["source"],
            selection_source=audit["selection_source"], fallback_reasons=audit["fallback_reasons"],
            h3_would_reject=audit.get("h3_guard_would_reject"), pfo_calls=detail["pfo_calls"],
            decision_seconds=detail["decision_seconds"], counts=audit["counts"])
        if summaries[mode, label] != derived:
            raise ValueError("Probe summary disagrees with detailed audit/control records")
        influenced |= mode == "carry_physical" and label != "zero" and changed
    if (row.get("all_executions_valid") is not True or row.get("carry_influence") is not influenced or
            row.get("no_ordinary_pfo") is not (True if step > 5 else None)):
        raise ValueError("Probe completion claims disagree with detailed evidence")
    return influenced


def build(evidence, reviews, output):
    if output.exists():
        raise FileExistsError(output)
    records = {}

    def record(path):
        records[str(path.resolve())] = file_hash(path)

    def read_record(path):
        before = file_hash(path)
        data = read(path)
        if file_hash(path) != before:
            raise ValueError("Admission evidence changed while reading: " + str(path))
        records[str(path.resolve())] = before
        return data

    current_pins = pins(DEFAULT_SNAPSHOT)
    for name in ("smoke_resume.json",):
        path = evidence / name
        diagnostic = read_record(path)
        if diagnostic.get("passed") is not True or diagnostic.get("source_pins") != current_pins:
            raise ValueError("Missing or failed diagnostic: " + name)
    influence_states = set()
    for step in (5, 30, 50):
        try:
            if validate_probe(evidence / f"probe_step{step}", step, current_pins, read_record):
                influence_states.add(step)
        except (KeyError, TypeError, OSError) as exc:
            raise ValueError("Missing or malformed probe evidence at step " + str(step)) from exc
    if len(influence_states) < 2:
        raise ValueError("Insufficient executable action influence")
    tests = evidence / "preflight_tests.xml"
    root = ET.parse(tests).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    count = sum(int(s.get("tests", 0)) for s in suites)
    if count == 0 or any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped")):
        raise ValueError("Incomplete or failed unit regression suite")
    record(tests)
    if not reviews:
        raise ValueError("Independent final review required")
    for path in reviews:
        if "PILOT_REVIEW: PASS" not in path.read_text(encoding="utf-8"):
            raise ValueError("Final pilot review did not pass")
        record(path)
    save(output, dict(ready_for_bounded_pilot=True, created_unix=time.time(),
                     source_pins=current_pins, evidence_sha256=records,
                     unit_tests=count, scope="two_training_episodes_and_center_RL_frozen_evaluations",
                     carry_influence_states=len(influence_states),
                     legacy_ddqn_resumed=False, performance_acceptance=False,
                     influence_probe_role="diagnostic_only_not_training_targets"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--review", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.evidence, args.review, args.output)


if __name__ == "__main__":
    main()
