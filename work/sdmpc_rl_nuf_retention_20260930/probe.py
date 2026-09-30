"""Offline paired causal integrity, predeclared screens and descriptive readout."""
import argparse
import time
from pathlib import Path
from local_runtime import np, torch, read, save, file_hash, identity, COHORT_ROOT, SCENARIOS
from references import authenticate, pair, validate_binding
from validate import load_completed, CONTROL_FIELDS

PILOT = "sweet_170_w"
PILOT_LIMITS = dict(zero_nuf_requests=5, terminal_inventory=517.470147382039,
                    old_local_ttt=11945.939292655728, carry_ttt=4602.4114708736415)
PREFIX_FIELDS = ("action_anchor", "action_requested", "B_raw", "B_requested", "B_executed",
    "G_achieved", "plant_state", "interval_ttt", "reward", "total_ttt", "freeway_ttt",
    "urban_ttt", "inventory", "selection_source", "reference_source", "reference_recovery",
    "execution_check", "incoming_dual", "committed_dual", "reference_budget", "reference_TTT",
    "pfo_calls", "lower_candidate_count", "terminated", "truncated")


def equal_numeric(a, b, label):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError("Prefix keys differ: " + label)
        for k in a:
            equal_numeric(a[k], b[k], label + "/" + k)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Prefix length differs: " + label)
        for i, (x, y) in enumerate(zip(a, b)):
            equal_numeric(x, y, label + f"/{i}")
    elif isinstance(a, (str, bool)) or a is None or isinstance(b, (str, bool)) or b is None:
        if type(a) is not type(b) or a != b:
            raise ValueError("Prefix value differs: " + label)
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-8, equal_nan=False, err_msg=label)


def causal_integrity(trace, experience, old_trace, old_experience, pilot=False):
    if any(len(value) != 75 for value in (trace, experience, old_trace, old_experience)):
        raise ValueError("Causal comparison requires all 75 actual intervals")
    for new, old in zip(trace, old_trace):
        np.testing.assert_array_equal(new["exploration_audit"]["noise"], old["exploration_audit"]["noise"])
    first = next((i for i, (a, b) in enumerate(zip(experience, old_experience))
                  if not np.allclose(a[1], b[1], rtol=0, atol=1e-8, equal_nan=False)), None)
    if pilot:
        for i in range(10):
            for key in PREFIX_FIELDS:
                equal_numeric(trace[i][key], old_trace[i][key], f"step {i}/{key}")
            for key in CONTROL_FIELDS:
                equal_numeric(trace[i]["control"][key], old_trace[i]["control"][key], f"step {i}/control/{key}")
            for j, (a, b) in enumerate(zip(experience[i], old_experience[i])):
                equal_numeric(a, b, f"step {i}/transition/{j}")
        if first != 10:
            raise ValueError("Pilot first actual action divergence must be step 10")
    return dict(all_75_noise_vectors_exact=True, prefix_through_step9=pilot,
                first_action_difference=first, accounting_atol=1e-8,
                post_intervention_old_actions_required=False)


def scenario_report(output, scenario, source, references):
    folder = Path(output) / "local" / scenario
    result, settings, summary = load_completed(folder, source)
    bound = validate_binding(settings, references)
    old_folder = Path(bound["local"]["path"])
    carry = read(Path(bound["carry"]["path"]) / "summary.json")
    old = read(old_folder / "summary.json")
    trace = read(folder / "trace.json")
    experience = torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)["transitions"]
    old_trace = read(old_folder / "trace.json")
    old_experience = torch.load(old_folder / "experience.pt", map_location="cpu", weights_only=False)["transitions"]
    integrity = causal_integrity(trace, experience, old_trace, old_experience, scenario == PILOT)
    checks = dict(zero_request_extra_le5=summary["zero_nuf_requests"] <= carry["zero_nuf_requests"]+5,
                  inventory_ratio_le1_25=summary["terminal_inventory"] <= 1.25*max(1., carry["terminal_inventory"]))
    if scenario == PILOT:
        equal_numeric(old["ttt"], PILOT_LIMITS["old_local_ttt"], "pinned old pilot TTT")
        equal_numeric(carry["ttt"], PILOT_LIMITS["carry_ttt"], "pinned pilot carry TTT")
        equal_numeric(1.25*max(1., carry["terminal_inventory"]), PILOT_LIMITS["terminal_inventory"],
                      "pinned pilot inventory threshold")
        # Use the plan's literal decimal at the boundary (one ULP above its product).
        checks["inventory_ratio_le1_25"] = summary["terminal_inventory"] <= PILOT_LIMITS["terminal_inventory"]
        checks.update(zero_nuf_le5=summary["zero_nuf_requests"] <= PILOT_LIMITS["zero_nuf_requests"],
            pilot_inventory=summary["terminal_inventory"] <= PILOT_LIMITS["terminal_inventory"],
            lower_ttt_than_old_local=summary["ttt"] < PILOT_LIMITS["old_local_ttt"])
    passed = all(checks.values())
    return dict(scenario=scenario, run_id=settings["run_id"], completion_sha256=file_hash(folder / "completion.json"),
        paired_references=bound, integrity=integrity, integrity_pass=True, checks=checks,
        coverage_screen_pass=checks["zero_request_extra_le5"] and checks["inventory_ratio_le1_25"],
        advance_allowed=passed, candidate=summary,
        old_local_ttt=old["ttt"], carry_ttt=carry["ttt"],
        ttt_delta_old_local=summary["ttt"]-old["ttt"], ttt_delta_carry=summary["ttt"]-carry["ttt"],
        interval_ttt_delta_old_local=(np.array(summary["interval_ttt"])-old["interval_ttt"]).tolist(),
        interval_ttt_delta_carry=(np.array(summary["interval_ttt"])-carry["interval_ttt"]).tolist(),
        conclusion="screen_pass_only" if passed else "valid_treatment_insufficient",
        canonical_evaluation=False, shared_policy_improvement_claim=False, learner_admitted=False)


def pilot_report(output, source, references):
    started = time.perf_counter()
    try:
        report = scenario_report(output, PILOT, source, references)
    except (ValueError, AssertionError, KeyError, OSError) as exc:
        report = dict(scenario=PILOT, integrity_pass=False, advance_allowed=False,
            conclusion="invalid_attribution_requires_diagnosis", error=f"{type(exc).__name__}: {exc}",
            canonical_evaluation=False, shared_policy_improvement_claim=False, learner_admitted=False)
    folder = Path(output) / "local" / PILOT
    inputs = {name: file_hash(folder / name) for name in
              ("completion.json", "settings.json", "trace.json", "experience.pt", "diagnostic-audit.json")
              if (folder / name).is_file()}
    return dict(report, identity=source, input_sha256=inputs, reference_root_sha256=references["root_sha256"],
                verification_wall_seconds=time.perf_counter()-started,
                verification_scope="This offline pilot readout only; excludes worker and coordinator sessions")


def balanced_report(output, source, references):
    rows = [scenario_report(output, scenario, source, references) for scenario in SCENARIOS]
    if len({row["run_id"] for row in rows}) != 5:
        raise ValueError("Duplicate new run IDs")
    return dict(rows=rows, new_transitions=375, transitions_per_scenario=75,
        future_minibatch_scenario_fractions={s: .2 for s in SCENARIOS},
        all_integrity_and_screens_pass=all(r["advance_allowed"] for r in rows),
        reference_transitions_recollected=0, learner_admitted=False,
        policy_q_available=False, canonical_evaluation=False, shared_policy_improvement_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=COHORT_ROOT)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    from local_runtime import canonical_root
    canonical_root(args.output)
    source, references = identity(), authenticate()
    report = balanced_report(args.output, source, references) if args.all else pilot_report(args.output, source, references)
    import json
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("advance_allowed", report.get("all_integrity_and_screens_pass", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
