"""Read-only original-validator entrypoint. Never boot or load raw environments."""
import json
import sys
from pathlib import Path
import references


def main():
    references.verify_helper_sources()
    before = references.manifest()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(references.RECOVERY))
    import export_recovery as recovery
    import supervise
    lr, collector, validator, wave = recovery.modules()
    def forbidden(*args, **kwargs):
        raise AssertionError("Reference authentication cannot boot, collect or simulate")
    lr.boot = collector.boot = collector.run = collector.BudgetEnv = wave.run = forbidden
    original_load = lr.torch.load
    def experience_only(path, *args, **kwargs):
        if Path(path).name != "experience.pt":
            raise AssertionError("Only ordinary saved experience arrays may be deserialized")
        return original_load(path, *args, **kwargs)
    lr.torch.load = experience_only
    complete, repairs = supervise.inspect(lr, validator, wave, lr.COHORT_ROOT.resolve())
    if not complete or repairs or before != references.manifest():
        raise ValueError("Original v2 authentication failed or references changed")
    print(json.dumps(before, sort_keys=True, allow_nan=False, ensure_ascii=True))


if __name__ == "__main__":
    main()
