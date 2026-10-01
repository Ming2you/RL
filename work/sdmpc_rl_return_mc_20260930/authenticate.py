"""Fresh read-only wave validation with physical entry points and unsafe loads blocked."""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mc_common import REPO, WAVE_SOURCE, WAVE, PARENT, SCENARIOS, read, READOUT
from mc_data import manifest


def main():
    before = manifest()
    sys.path.insert(0, str(WAVE_SOURCE))
    import readout
    import wave_support as w
    import worker
    import budget_runtime
    import budget_env

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only authentication forbids physical calls and optimization")

    w.boot = budget_runtime.boot = worker.run = forbidden
    for name in ("__init__", "reset", "step", "restore"):
        if hasattr(budget_env.BudgetEnv, name):
            setattr(budget_env.BudgetEnv, name, forbidden)
    w.torch.optim.Adam.step = forbidden
    original_load = w.torch.load
    permitted = {(PARENT / "model_final.pt").resolve()} | {
        (WAVE / scenario / "experience.pt").resolve() for scenario in SCENARIOS}

    def safe_load(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved not in permitted:
            raise AssertionError("Raw physical checkpoint unpickle forbidden")
        w.check_hash(resolved, before["files"][resolved.relative_to(REPO).as_posix()])
        return original_load(path, *args, **kwargs)

    w.torch.load = safe_load
    checked = readout.validate_wave()
    if checked != read(READOUT) or not checked["all_health_gates_pass"] or not checked["integrity_pass"]:
        raise ValueError("Recomputed source health/readout differs from retained Task 2 evidence")
    if manifest() != before:
        raise ValueError("Authenticated inputs changed")
    print(json.dumps(dict(status="authenticated", manifest=before, readout=checked,
        raw_environment_loads=0, environment_calls=0, optimizer_updates=0), allow_nan=False))


if __name__ == "__main__":
    main()
