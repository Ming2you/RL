"""Scoped synthetic tests plus read-only parity/authentication and preservation hashes."""
import os
import sys
import uuid
import xml.etree.ElementTree as ET
import wave_support as w
import parity


def preservation():
    settings = w.read(w.PREDECESSOR / "settings.json")
    files = {w.REPO / p for p in settings["sources"]} | {w.REPO / p for p in settings["data"]["files"]}
    files.update(p for p in w.PREDECESSOR.rglob("*") if p.is_file())
    for directory in (w.OLD, w.NUF, w.REPO / "work/sdmpc_rl_return_init_20260930"):
        files.update(directory.glob("*.py"))
    manifest = w.DEFAULT_SNAPSHOT.parent / "manifest.json"
    files.add(manifest)
    files.update(w.DEFAULT_SNAPSHOT.parent / r["snapshot_relative"] for r in w.read(manifest)["files"])
    files.add(w.GATE)
    return {p.relative_to(w.REPO).as_posix(): w.file_hash(p) for p in sorted(files)}


def main():
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    import pytest
    run_id = uuid.uuid4().hex[:8]
    output = w.HERE / "test-evidence" / run_id
    (w.HERE / "_t").mkdir(exist_ok=True)
    before, source = preservation(), w.sources()
    checked = parity.verify()
    w.save(output / "parity-authentication.json", checked)
    code = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", "--confcutdir="+str(w.HERE),
        "--basetemp="+str(w.HERE / "_t" / run_id), "--junitxml="+str(output / "tests.xml"),
        str(w.HERE / "test_wave.py"), str(w.HERE / "test_restore.py"), *sys.argv[1:]])
    if preservation() != before or w.sources() != source:
        raise ValueError("Preserved inputs or tested source changed")
    suite = ET.parse(output / "tests.xml").getroot().find("testsuite")
    w.save(output / "evidence.json", dict(exit_code=code, source_sha256=source,
        command=[sys.executable, "-B", str(w.HERE / "run_tests.py"), *sys.argv[1:]],
        test_summary=suite.attrib, tests_sha256=w.file_hash(output / "tests.xml"),
        parity_sha256=w.file_hash(output / "parity-authentication.json"),
        preserved_before_after_sha256=before, preserved_file_count=len(before),
        actual_physical_steps=0, actual_optimization_updates=0,
        actual_wave_dispatched=False, runtime_versions=w.runtime_versions()))
    print("EVIDENCE", output, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
