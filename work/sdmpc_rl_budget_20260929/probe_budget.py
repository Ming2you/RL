"""Matched-state actual follower responses and short sequential branch diagnostics."""
import argparse
import copy
from pathlib import Path
import time
from budget_runtime import boot, save, plain


def from_fixture(rt, step):
    from check_wrapper import load_fixture
    from budget_env import BudgetEnv
    from budget_controller import BudgetController
    import numpy as np
    env = BudgetEnv(rt)
    state, previous, dual, budget = load_fixture(rt, step)
    env.sim = rt["rc"].MixedTrafficSimulator(rt["cfg"])
    env.sim.state = state.copy()
    env.warm = rt["rc"].make_controller("WU-FAITHFUL-FOLLOWER", rt["cfg"])
    env.controller = BudgetController(rt)
    env.controller.lower.dual = dual.copy()
    env.controller.lower.last_budget = copy.deepcopy(budget)
    env.previous, env.k = previous.copy(), step
    env.last_requested = np.zeros(2) if budget is None else budget.copy()
    env.last_executed, env.last_slack = env.last_requested.copy(), np.zeros(2)
    env.last_fallback, env.warmup_ttt = False, 0.
    env.prepare()
    env._observe()
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--cpu-mask", type=int, default=1)
    p.add_argument("--tail", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rt = boot(args.root, args.cpu_mask)
    import numpy as np
    env = from_fixture(rt, args.step)
    checkpoint = env.checkpoint()
    candidates = {"center": [0., 0.], "np_plus": [1., 0.], "np_minus": [-1., 0.],
                  "nuf_plus": [0., 1.], "nuf_minus": [0., -1.]}
    records = []
    started = time.perf_counter()
    try:
        for label, action in candidates.items():
            if (args.output / "STOP").exists():
                save(args.output / "status.json", dict(status="paused", completed=len(records)))
                return
            env.restore(checkpoint)
            trace = []
            for j in range(min(args.tail, 80 - args.step)):
                _, reward, done, row = env.step(action if j == 0 else [0., 0.])
                trace.append(row)
                if done:
                    break
            first = trace[0]
            record = dict(candidate=label, action=action, tail_ttt=env.sim.total_ttt,
                          initial_step=args.step, interval_count=len(trace),
                          full_run=False, continuation="center_budget_policy_not_optimal_Q_label",
                          requested=first["B_requested"][0], executed=first["B_executed"],
                          achieved=first["G_achieved"], fallback=first["fallback_reasons"],
                          selection_source=first["selection_source"],
                          trace=trace)
            save(args.output / (label + ".json"), record)
            records.append(record)
            save(args.output / "status.json", dict(status="running", completed=len(records)))
            print("PROBE", args.step, label, record["tail_ttt"], record["fallback"], flush=True)
        center = records[0]
        coordinates = rt["coordinates"](rt["cfg"], rt["options"], checkpoint["previous"])
        center_control = coordinates.encode(center["trace"][0]["control"])
        for row in records:
            control = coordinates.encode(row["trace"][0]["control"])
            row["changed_physical_control"] = bool(np.max(abs(control - center_control)) > 1e-9)
            row["tail_delta_vs_center"] = row["tail_ttt"] - center["tail_ttt"]
        save(args.output / "summary.json", dict(
            status="completed", step=args.step, wall_seconds=time.perf_counter() - started,
            action_influence_observed=any(r["changed_physical_control"] for r in records[1:]),
            records=[{k: v for k, v in r.items() if k != "trace"} for r in records],
            scope="matched_state_diagnostic_not_full_run_or_training_label"))
        save(args.output / "status.json", dict(status="completed", completed=len(records)))
    finally:
        env.close()
        rt["verify"]()


if __name__ == "__main__":
    main()
