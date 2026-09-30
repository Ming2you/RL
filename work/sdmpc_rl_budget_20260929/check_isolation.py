"""Check actual lower solves are independent of evaluation order and commit state."""
import argparse
from pathlib import Path
from budget_runtime import boot, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-mask", type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rt = boot(args.root, args.cpu_mask)
    from probe_budget import from_fixture
    from budget_controller import residual_budget
    import numpy as np
    records = []
    for order in (("center", "np_plus"), ("np_plus", "center")):
        env = from_fixture(rt, 30)
        try:
            lower, reference = env.controller.lower, env.controller.reference
            dual, prior = lower.dual.copy(), lower.last_budget.copy()
            result = {}
            for label in order:
                _, budget = residual_budget([0., 0.] if label == "center" else [1., 0.],
                                             reference["budget"], rt["cfg"].network.total_ramp_capacity)
                row = env.controller.evaluate_budget(budget)
                np.testing.assert_array_equal(lower.dual, dual)
                np.testing.assert_array_equal(lower.last_budget, prior)
                result[label] = dict(point=row["point"].copy(), dual=row["dual"].copy(),
                                     ttt=row["evaluation"].total_ttt, feasible=row["feasible"])
            records.append(result)
        finally:
            env.close()
            rt["verify"]()
    for label in ("center", "np_plus"):
        for key in ("point", "dual", "ttt", "feasible"):
            np.testing.assert_allclose(records[0][label][key], records[1][label][key], rtol=0, atol=1e-10)
    save(args.output, dict(passed=True, step=30, records=records,
                          scope="candidate_order_and_persistent_price_isolation"))
    print("ISOLATION_PASS", flush=True)


if __name__ == "__main__":
    main()
