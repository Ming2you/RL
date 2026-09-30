"""Synthetic updates plus real read-only authentication, with preservation evidence."""
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from runtime import HERE, REPO, SPEC, np, torch, sources, read, save, sha, digest
from data import manifest, authenticate, load_data, ROOT, CARRY_ROOT

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest


def preservation():
    identity = manifest()
    paths = {REPO / p for p in identity["files"]}
    for root in (ROOT, CARRY_ROOT):
        for name in ("completion.json", "comparison.json", "plan.json", "status.json", "process.json", ".ownership/reservations.json"):
            if (root / name).is_file():
                paths.add(root / name)
        done = read(root / "completion.json")
        for key in done["child_completion_sha256"]:
            folder = root / key
            child = read(folder / "completion.json")
            paths.update(folder / name for name in child["outputs_sha256"])
            for name in ("completion.json", "settings.json", "checkpoint.pt"):
                if (folder / name).is_file():
                    paths.add(folder / name)
    snapshot = REPO / "artifacts/sdmpc_budget_baseline_20260929"
    paths.add(snapshot / "manifest.json")
    paths.update(snapshot / row["snapshot_relative"] for row in read(snapshot / "manifest.json")["files"])
    for directory in ("sdmpc_rl_multi_20260929", "sdmpc_rl_nuf_retention_20260930", "sdmpc_rl_local_20260930_v2",
                      "sdmpc_rl_export_recovery_20260930", "sdmpc_rl_value_audit_20260930"):
        paths.update((REPO / "work" / directory).glob("*.py"))
    return {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(paths)}


def main():
    run_id = uuid.uuid4().hex[:8]
    evidence = HERE / "test-evidence" / run_id
    evidence.mkdir(parents=True)
    (HERE / "_t").mkdir(exist_ok=True)
    source, before = sources(), preservation()
    checked = authenticate()
    # Shape/provenance checks only: never pass these arrays to a learner in tests.
    data = load_data(checked["manifest"])
    assert len(data["obs"]) == 750
    assert data["anchor"].dtype == data["next_anchor"].dtype == torch.float64
    assert int(data["local"].sum()) == 375
    assert [int((data["scenario"] == i).sum()) for i in range(5)] == [150]*5
    save(evidence / "authentication.json", checked)
    del data
    code = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", "--confcutdir="+str(HERE),
        "--basetemp="+str(HERE / "_t" / run_id), "--junitxml="+str(evidence / "tests.xml"),
        str(HERE / "test_return_init.py"), *sys.argv[1:]])
    after = preservation()
    if before != after or source != sources():
        raise ValueError("Preserved artifacts or tested source changed")
    suite = ET.parse(evidence / "tests.xml").getroot().find("testsuite")
    save(evidence / "evidence.json", dict(exit_code=code, command=[sys.executable, "-B", str(HERE / "run_tests.py"), *sys.argv[1:]],
        source_sha256=source, spec_sha256=digest(SPEC), spec=SPEC, test_summary=suite.attrib,
        xml_sha256=sha(evidence / "tests.xml"), authentication_sha256=sha(evidence / "authentication.json"),
        preserved_before_after_sha256=before, preserved_file_count=len(before),
        python=sys.version, torch=torch.__version__, numpy=np.__version__,
        actual_data_training_updates=0, environment_runs=0, actual_rows_validated=750,
        canonical_rows_loaded_for_training=0))
    print("EVIDENCE", evidence, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
