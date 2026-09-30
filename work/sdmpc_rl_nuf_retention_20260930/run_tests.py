"""Run only new isolated covering tests using already installed dependencies."""
import os
import hashlib
import json
from pathlib import Path
import sys
import uuid
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(HERE.parents[1] / ".deps-budget"))
import pytest
import references


def preservation_hashes():
    root = references.ROOT
    done = references.read(root / "completion.json")
    paths = {root / name for name in ("completion.json", "comparison.json", "plan.json", "status.json",
                                     "process.json", ".ownership/reservations.json")}
    for key in done["child_completion_sha256"]:
        folder = root / key
        completion = references.read(folder / "completion.json")
        paths.update(folder / name for name in ("completion.json", "settings.json", "checkpoint.pt"))
        paths.update(folder / name for name in completion["outputs_sha256"])
        repair = completion.get("export_recovery")
        if repair:
            paths.add(folder / repair["receipt"])
            receipt = references.read(folder / repair["receipt"])
            paths.update(folder / (".export-recovery/original-status.json" if name == "status.json" else name)
                         for name in receipt["input_sha256"])
    for directory in (references.REPO / "work/sdmpc_rl_local_20260930_v2", references.RECOVERY):
        paths.update(directory.glob("*.py"))
    return {p.relative_to(references.REPO).as_posix(): references.sha(p) for p in sorted(paths)}

if __name__ == "__main__":
    run_id = uuid.uuid4().hex[:8]
    run = HERE / "test-evidence" / run_id
    run.mkdir(parents=True)
    (HERE / "_t").mkdir(exist_ok=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}
    preserved = preservation_hashes()
    code = pytest.main(["-q", "--tb=short", "--show-capture=no", "-o", "junit_family=xunit1",
        "-p", "no:cacheprovider", "--confcutdir=" + str(HERE),
        "--basetemp=" + str(HERE / "_t" / run_id), "--junitxml=" + str(run / "tests.xml")]
        + [str(p) for p in sorted(HERE.glob("test_*.py"))] + sys.argv[1:])
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}
    if sources != after:
        raise ValueError("Tested source changed during tests")
    if preserved != preservation_hashes():
        raise ValueError("Immutable reference/source artifacts changed during tests")
    from local_runtime import identity
    suite = ET.parse(run / "tests.xml").getroot().find("testsuite")
    (run / "evidence.json").write_text(json.dumps(dict(exit_code=code, source_sha256=sources,
        xml_sha256=hashlib.sha256((run / "tests.xml").read_bytes()).hexdigest(),
        python=sys.version, isolated_fixtures=str(HERE / "_t" / run_id), test_summary=suite.attrib,
        candidate_identity=identity(), preserved_before_after_sha256=preserved,
        command=[sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        real_simulations=0, training_runs=0, old_suites_run=False), indent=2), encoding="utf-8")
    raise SystemExit(code)
