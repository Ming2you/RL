"""Isolated read-only invocation of the frozen NUF and carry validators."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.dont_write_bytecode = True


def main():
    # Do not import this module into the learner: old collectors share module names.
    from data import manifest
    before = manifest()
    sys.path.insert(0, str(REPO / "work/sdmpc_rl_nuf_retention_20260930"))
    import local_runtime as lr
    import collect
    import run_wave
    import budget_runtime
    import budget_env
    import references
    import probe

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline authentication must not boot/reset/step/collect")

    lr.boot = budget_runtime.boot = collect.boot = collect.run = run_wave.run = forbidden
    budget_env.BudgetEnv.reset = budget_env.BudgetEnv.step = forbidden
    collect.BudgetEnv = forbidden
    original_load = lr.torch.load

    def experience_only(path, *args, **kwargs):
        if Path(path).name != "experience.pt":
            raise AssertionError("Raw environment checkpoints are forbidden")
        return original_load(path, *args, **kwargs)

    lr.torch.load = experience_only
    refs = references.authenticate()
    done = lr.read(lr.COHORT_ROOT / "completion.json")
    source = lr.identity()
    if done["plan"]["identity"] != source:
        raise ValueError("NUF source identity mismatch")
    report = probe.balanced_report(lr.COHORT_ROOT, source, refs)
    if report != lr.read(lr.COHORT_ROOT / "comparison.json") or not report["all_integrity_and_screens_pass"]:
        raise ValueError("NUF completion/comparison reconciliation failed")
    if before != manifest():
        raise ValueError("Inputs changed during authentication")
    print(json.dumps(dict(status="authenticated", manifest=before, carry_references=5,
        local_episodes=5, new_environment_runs=0, raw_environment_loads=0), allow_nan=False))


if __name__ == "__main__":
    main()
