"""Real matched-state parity check; does not advance or relabel any old run."""
import argparse
import copy
from pathlib import Path
import time
from budget_runtime import boot, read, protocol, save, plain


def load_fixture(rt, step):
    import numpy as np
    folder = rt["root"] / "outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper"
    row = read(folder / f"plant_{step-1:03d}.json")
    return (rt["restore_state"](rt["rc"], row["state"]), rt["rc"].ControlAction(**row["control"]),
            np.array(row["persistent_dual"]), None if row["last_budget"] is None else np.array(row["last_budget"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--cpu-mask", type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rt = boot(args.root, args.cpu_mask)
    import numpy as np
    from budget_controller import BudgetController, residual_budget
    _, _, profile = protocol(rt)
    state, previous, dual, budget = load_fixture(rt, args.step)
    ff = profile.horizon(state.time_sec, 3)
    warm = rt["rc"].make_controller("WU-FAITHFUL-FOLLOWER", rt["cfg"])
    try:
        warm_control = warm.solve(state.copy(), None, ff, previous).control
        baseline = rt["baseline"](rt["cfg"], rt["options"])
        baseline.dual, baseline.last_budget = dual.copy(), copy.deepcopy(budget)
        tick = time.perf_counter()
        old, old_results = baseline.decide(state.copy(), ff, previous.copy(), warm_control.copy())
        old_seconds = time.perf_counter() - tick
        adapter = BudgetController(rt)
        adapter.lower.dual, adapter.lower.last_budget = dual.copy(), copy.deepcopy(budget)
        tick = time.perf_counter()
        ref = adapter.prepare_reference(state.copy(), ff, previous.copy(), warm_control.copy())
        raw, zero = residual_budget([0., 0.], ref["budget"], rt["cfg"].network.total_ramp_capacity)
        np.testing.assert_array_equal(zero, ref["budget"])
        new, audit = adapter.evaluate_and_commit(mode="native")
        new_seconds = time.perf_counter() - tick
        if old is None:
            raise AssertionError("Baseline has no executable response")
        assert len(old_results) == len(adapter.results)
        for a, b in zip(old_results, adapter.results):
            for key in ("budget", "point", "dual", "original_residual"):
                np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-10, err_msg=key)
            assert len(a["rows"]) == len(b["rows"])
            for x, y in zip(a["rows"], b["rows"]):
                for key in ("anchor", "own_gradient", "externality", "dual_before", "dual_after"):
                    np.testing.assert_allclose(x[key], y[key], rtol=0, atol=1e-10, err_msg=key)
        for key in ("point", "budget", "original_residual"):
            np.testing.assert_allclose(old[key], new[key], rtol=0, atol=1e-10)
        np.testing.assert_allclose(old["evaluation"].total_ttt, new["evaluation"].total_ttt, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(baseline.dual, adapter.lower.dual)
        assert plain(old["control"]) == plain(new["control"])
        assert old["selection_source"] == new["selection_source"]
        assert not new["converged"]
        result = dict(passed=True, step=args.step, source="matched_saved_state_not_full_run",
                      baseline_wall_seconds=old_seconds, wrapper_wall_seconds=new_seconds,
                      candidate_count=len(old_results), audit=audit)
        save(args.output, result)
        print("PARITY_PASS", args.step, old_seconds, new_seconds, flush=True)
    finally:
        if hasattr(warm, "close"):
            warm.close()
        rt["verify"]()


if __name__ == "__main__":
    main()
