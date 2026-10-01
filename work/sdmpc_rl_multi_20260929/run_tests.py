"""Run local synthetic tests and bind their evidence to exact source hashes."""
import argparse
import os
from pathlib import Path
import sys
from budget_runtime import DEFAULT_SNAPSHOT, HERE, REPO, save
from run_budget import file_hash, pins, verify_pins, runtime_versions
from build_preflight import TEST_FORMAT, current_test_hashes


class ExecutionEvidence:
    def __init__(self):
        self.collected = []
        self.phases = {}
        self.identities = {}

    def pytest_collection_modifyitems(self, items):
        for item in items:
            identity = str(item.path.resolve()) + "::" + item.nodeid.split("::", 1)[1]
            self.identities[item.nodeid] = identity
            item.user_properties.append(("nodeid", identity))

    def pytest_collection_finish(self, session):
        self.collected = [self.identities[item.nodeid] for item in session.items]

    def pytest_runtest_logreport(self, report):
        self.phases.setdefault(self.identities[report.nodeid], {})[report.when] = report.outcome

    def passed(self):
        return [nodeid for nodeid, phases in self.phases.items()
                if phases == dict(setup="passed", call="passed", teardown="passed")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xml, record = args.output / "tests.xml", args.output / "test_source_pins.json"
    if xml.exists() or record.exists():
        raise FileExistsError("Preserve previous test evidence; use a new output directory")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / ".deps-budget"))
    import pytest
    source_pins = pins(DEFAULT_SNAPSHOT)
    tests, versions = current_test_hashes(), runtime_versions()
    execution = ExecutionEvidence()
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--confcutdir="+str(HERE),
                        "--junitxml="+str(xml.resolve()), str(HERE)], plugins=[execution])
    verify_pins(DEFAULT_SNAPSHOT, source_pins)
    if tests != current_test_hashes() or versions != runtime_versions():
        raise ValueError("Tests changed during execution")
    save(record, dict(format=TEST_FORMAT, exit_code=int(code), source_pins=source_pins,
                      runtime_versions=versions, test_sha256=tests, xml_sha256=file_hash(xml),
                      collected_nodeids=execution.collected, passed_nodeids=execution.passed()))
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
