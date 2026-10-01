"""Scoped synthetic tests and actual read-only authentication; preserve all inputs."""
import os
import sys
import uuid
import xml.etree.ElementTree as ET
import support as s
import preflight


def main():
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    import budget_env
    import budget_runtime
    import run_budget

    def forbidden(*args, **kwargs):
        raise AssertionError("Actual physical execution and optimizer construction forbidden")

    s.boot = s.old.boot = budget_runtime.boot = run_budget.boot = forbidden
    for name in ("__init__", "reset", "step", "restore"):
        setattr(budget_env.BudgetEnv, name, forbidden)
    s.torch.optim.Adam.__init__ = s.torch.optim.Adam.step = forbidden
    original_load = s.torch.load
    permitted = {(s.MODEL / "model_final.pt").resolve()} | {
        (s.WAVE / scenario / "experience.pt").resolve() for scenario in s.SCENARIOS}

    def guarded_load(path, *args, **kwargs):
        resolved = s.Path(path).resolve()
        if resolved not in permitted and not resolved.is_relative_to(s.HERE):
            raise AssertionError("Actual physical checkpoint deserialization forbidden")
        return original_load(path, *args, **kwargs)

    s.torch.load = guarded_load
    before, source = s.preservation(), s.sources()
    output = s.HERE / "evidence" / uuid.uuid4().hex[:8]
    output.mkdir(parents=True)
    admitted = preflight.build()
    s.save_once(output / "preflight.json", admitted)
    import pytest
    code = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", "--confcutdir="+str(s.HERE),
        "--basetemp="+str(output / "synthetic"), "--junitxml="+str(output / "tests.xml"),
        str(s.HERE / "test_eval.py"), *sys.argv[1:]])
    after = s.preservation()
    if before != after or s.sources() != source:
        raise ValueError("Actual preserved input or tested source changed")
    suite = ET.parse(output / "tests.xml").getroot().find("testsuite")
    s.save_once(output / "evidence.json", dict(exit_code=code, source_sha256=source,
        source_digest=s.digest(source), spec_sha256=s.SPEC_SHA, spec=s.SPEC,
        command=[sys.executable, "-B", str(s.HERE / "run_tests.py"), *sys.argv[1:]],
        test_summary=suite.attrib, tests_sha256=s.file_hash(output / "tests.xml"),
        preflight_sha256=s.file_hash(output / "preflight.json"),
        preserved_before_sha256=before, preserved_after_sha256=after, preserved_file_count=len(before),
        actual_physical_steps=0, actual_optimizer_constructions=0, actual_optimizer_updates=0,
        actual_evaluation_outputs_absent=not s.ROOT.exists(), runtime_versions=s.old.runtime_versions()))
    print("EVIDENCE", output, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
