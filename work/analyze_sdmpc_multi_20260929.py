"""Read-only reconciliation and closed-loop diagnostics of the completed pilot."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(ROOT / ".deps-budget"), str(ROOT / "work/sdmpc_rl_multi_20260929")]
import numpy as np
from budget_runtime import DEFAULT_SNAPSHOT, digest, read, save
from run_budget import file_hash, verify_pins
from run_pilot import jobs_for, validate_job
from compare_runs import compare
from td3 import SCENARIOS


def returns(rewards, done):
    if not rewards or len(rewards) != len(done) or done[-1] is not True:
        raise ValueError("Need complete finite trajectory")
    out, value = np.empty(len(rewards)), 0.
    for i in range(len(rewards)-1, -1, -1):
        value = rewards[i] + (0. if done[i] else value)
        out[i] = value
    return out


def trajectory_diagnostics(trace, mode):
    actions = np.asarray([r["action_requested"] for r in trace])
    zero = [all(abs(v) <= 1e-9 for v in r["control"]["ramp_metering"].values()) for r in trace]
    streak = longest = 0
    for closed in zero:
        streak = streak + 1 if closed else 0
        longest = max(longest, streak)
    result = dict(action_mean=actions.mean(axis=0).tolist(), action_min=actions.min(axis=0).tolist(),
        action_max=actions.max(axis=0).tolist(), action_abs_ge_095_fraction=(abs(actions) >= .95).mean(axis=0).tolist(),
        fallback_reasons=dict(Counter(reason for r in trace for reason in r["fallback_reasons"])),
        reference_sources=dict(Counter(r["reference_source"] for r in trace)),
        nuf_zero_count=sum(abs(r["B_executed"][1]) <= 1e-9 for r in trace),
        all_ramps_zero_count=sum(zero), all_ramps_zero_longest_streak=longest,
        nuf_clipped_count=sum(abs(r["B_raw"][1]-r["B_requested"][0][1]) > 1e-9 for r in trace),
        distinct_exact_physical_controls=len({digest({k: r["control"][k] for k in
            ("ramp_metering", "vsl", "green_times", "offsets")}) for r in trace}),
        response_count_caveat="Exact physical dictionaries across different states, not state-matched equivalence classes.",
        terminal_inventory=trace[-1]["inventory"],
        h3_would_reject_count=sum(r["h3_guard_would_reject"] for r in trace),
        mean_phase_seconds={p: float(np.mean([r[p+"_wall_seconds"] for r in trace]))
                            for p in ("pfo", "reference", "actor", "lower", "guard")},
        actor_timing_scope="Actor plus read-only twin-Q diagnostic inference; no response previews beyond the lower solve.",
        interval_ttt_blocks=[dict(start_control=i, end_control=i+14,
            ttt=sum(r["interval_ttt"] for r in trace[i:i+15])) for i in range(0, 75, 15)])
    if mode == "rl":
        actual = returns([r["reward"] for r in trace], [r["terminated"] for r in trace])
        q = np.min([r["policy_q"] for r in trace], axis=1)
        result["value_diagnostics"] = dict(q_mean=float(q.mean()), return_mean=float(actual.mean()),
            q_minus_return_mean=float(np.mean(q-actual)), absolute_error_mean=float(np.mean(abs(q-actual))),
            terminal_absolute_error=float(abs(q[-1]-actual[-1])),
            caveat="Realized return under this same frozen deterministic policy and canonical profile, not out-of-sample generalization. TD3 training targets use action smoothing.")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pilot = args.pilot.resolve()
    plan = read(pilot / "plan.json")
    if read(pilot / "completion.json")["status"] != "completed":
        raise ValueError("Pilot is incomplete")
    verify_pins(DEFAULT_SNAPSHOT, plan["source_pins"])
    before = file_hash(pilot / "train_round1/model_final.pt")
    for job in jobs_for(pilot, SCENARIOS):
        validate_job(pilot, job, plan)
    comparisons, diagnostics, relative = [], [], []
    for scenario in SCENARIOS:
        result = compare({kind: pilot / kind / scenario for kind in ("center", "rl")})
        comparisons.append(dict(scenario=scenario, **result))
        relative.append(next(row["improvement_vs_center_pct"] for row in result["rows"] if row["mode"] == "rl"))
        for mode in ("center", "rl"):
            trace = read(pilot / mode / scenario / "episode_00_trace.json")
            diagnostics.append(dict(scenario=scenario, mode=mode, **trajectory_diagnostics(trace, mode)))
    saved = read(pilot / "comparison.json")
    if saved["comparisons"] != comparisons or saved["shared_model_sha256"] != before:
        raise ValueError("Saved comparison does not reproduce")
    training = [read(pilot / f"train_round{i}/completion.json") for i in (0, 1)]
    if file_hash(pilot / "train_round1/model_final.pt") != before:
        raise ValueError("Frozen checkpoint changed during analysis")
    save(args.output, dict(shared_model_sha256=before, analysis_source_sha256=file_hash(__file__),
        scenarios=list(SCENARIOS), per_scenario_replay_counts=training[-1]["replay_counts"],
        updates=training[-1]["learner_updates"],
        samples_per_scenario={s: sum(t["sampled_per_scenario"][s] for t in training) for s in SCENARIOS},
        macro_relative_improvement_pct=float(np.mean(relative)), worst_relative_improvement_pct=float(min(relative)),
        every_scenario_improves=all(x > 0 for x in relative), diagnostics=diagnostics,
        scope="balanced_shared_policy_development_results_not_generalization",
        aggregate_caveat="Equal-weight relative improvements; individual scenario losses remain authoritative."))
    print("MULTI_DIAGNOSTICS_PASS", before, relative)


if __name__ == "__main__":
    main()
