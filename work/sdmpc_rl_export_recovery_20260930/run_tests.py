"""Run only new isolated covering tests using already installed dependencies."""
import os
import hashlib
import json
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(HERE.parents[1] / ".deps-budget"))
import pytest

if __name__ == "__main__":
    run_id = uuid.uuid4().hex[:8]
    run = HERE / "test-evidence" / run_id
    run.mkdir(parents=True)
    (HERE / "_t").mkdir(exist_ok=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}
    code = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", "--confcutdir=" + str(HERE),
        "--basetemp=" + str(HERE / "_t" / run_id), "--junitxml=" + str(run / "tests.xml")]
        + [str(p) for p in sorted(HERE.glob("test_*.py"))] + sys.argv[1:])
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}
    if sources != after:
        raise ValueError("Tested source changed during tests")
    (run / "evidence.json").write_text(json.dumps(dict(exit_code=code, source_sha256=sources,
        xml_sha256=hashlib.sha256((run / "tests.xml").read_bytes()).hexdigest(),
        python=sys.version, isolated_fixtures=str(HERE / "_t" / run_id)), indent=2), encoding="utf-8")
    raise SystemExit(code)
