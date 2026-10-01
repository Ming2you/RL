"""Bind tests, five real serialized smokes and final review to frozen source."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from budget_runtime import DEFAULT_SNAPSHOT, HERE, REPO, digest, read, save
from run_budget import file_hash, pins, runtime_versions

GATE_FORMAT = "sdmpc-multi-admission-v2"
TEST_FORMAT = "sdmpc-multi-test-evidence-v2"
REVIEW_FORMAT = "sdmpc-multi-final-review-v1"
REQUIRED_TESTS = {"test_td3.py", "test_carry_controller.py", "test_carry_environment.py",
                  "test_multi_contracts.py", "test_multi_pilot.py", "test_multi_runner.py"}


def current_test_hashes():
    tests = {str(p.resolve()): file_hash(p) for p in HERE.glob("test_*.py")}
    if not REQUIRED_TESTS <= {Path(p).name for p in tests}:
        raise ValueError("Required test suites are missing")
    return tests


def evidence_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def evidence_roles(scenarios):
    return {"test_xml", "test_record", "review_report", "review_attestation",
            *("smoke:" + s for s in scenarios)}


def validate_smokes(smokes, source_pins, scenarios):
    if len(smokes) != 5 or {r["scenario"] for r in smokes} != set(scenarios):
        raise ValueError("Need five distinct scenario smokes")
    common_schema = smokes[0]["observation_schema"]
    for row in smokes:
        profile_path = DEFAULT_SNAPSHOT / "outputs/sdmpc_budget_exception_all_20260922/protocols_0" / row["scenario"] / "forecast.json"
        if row["profile_sha256"] != digest(read(profile_path)):
            raise ValueError("Smoke does not use the frozen canonical scenario")
        if (row["passed"] is not True or row["serialized_resume"] is not True or
                row["source_pins"] != source_pins or row["step"] != 7 or
                row["observation_schema"] != common_schema or row["observation_dim"] != len(common_schema["names"]) or
                row["scope"] != "two_actual_intervals_with_serialized_resume_not_full_run" or
                row["replay_counts"] != {s: int(s == row["scenario"]) for s in scenarios}):
            raise ValueError("Smoke source/schema/serialized result differs")
        for index, audit in enumerate((row["audit"], row["carried_step_audit"])):
            if (audit["execution_check"]["physical_control_valid"] is not True or
                    audit["execution_check"]["budget_feasible"] is not True or
                    audit["h3_guard_enabled"] is not False or audit["pfo_calls"] != int(index == 0) or
                    audit["reference_source"] != ("pfo_initial" if index == 0 else "previous")):
                raise ValueError("Smoke physical/carry reference evidence failed")
    if not common_schema["names"] or len(set(common_schema["names"])) != len(common_schema["names"]):
        raise ValueError("Invalid common feature schema")
    return common_schema


def validate_evidence(evidence, source_pins, versions, scenarios):
    if not isinstance(evidence, dict) or set(evidence) != evidence_roles(scenarios):
        raise ValueError("Admission requires the exact typed evidence roles")
    paths = {}
    for role, ref in evidence.items():
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or
                not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute() or
                not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64 or
                any(c not in "0123456789abcdef" for c in ref["sha256"])):
            raise ValueError("Invalid admission evidence reference: " + role)
        paths[role] = Path(ref["path"]).resolve()
        if file_hash(paths[role]) != ref["sha256"]:
            raise ValueError("Admission evidence changed: " + role)
    if len(set(paths.values())) != len(paths):
        raise ValueError("Admission evidence roles must use distinct files")
    tests = current_test_hashes()
    record = read(paths["test_record"])
    record_fields = {"format", "exit_code", "source_pins", "runtime_versions", "test_sha256",
                     "xml_sha256", "collected_nodeids", "passed_nodeids"}
    if (not isinstance(record, dict) or set(record) != record_fields or record["format"] != TEST_FORMAT or
            type(record["exit_code"]) is not int or record["exit_code"] != 0 or
            record["source_pins"] != source_pins or record["runtime_versions"] != versions or
            record["test_sha256"] != tests or record["xml_sha256"] != evidence["test_xml"]["sha256"]):
        raise ValueError("Passing test evidence does not cover current sources/tests/runtime")
    root = ET.parse(paths["test_xml"]).getroot()
    cases = root.findall(".//testcase")
    if len(cases) < 60 or any(case.find(name) is not None for case in cases for name in ("failure", "error", "skipped")):
        raise ValueError("Unit/regression evidence is missing or not passing")
    nodeids = []
    for case in cases:
        identities = case.findall("./properties/property[@name='nodeid']")
        if len(identities) != 1 or not identities[0].get("value"):
            raise ValueError("Test XML lacks unique execution identities")
        nodeids.append(identities[0].get("value"))
    collected, passed = record["collected_nodeids"], record["passed_nodeids"]
    if (not isinstance(collected, list) or not isinstance(passed, list) or
            any(not isinstance(n, str) for n in collected + passed) or
            len(set(nodeids)) != len(nodeids) or sorted(nodeids) != sorted(collected) or
            sorted(nodeids) != sorted(passed) or
            {Path(n.split("::")[0]).resolve() for n in nodeids} != {Path(p).resolve() for p in tests}):
        raise ValueError("Required test suites were not completely executed")
    smokes = [read(paths["smoke:" + s]) for s in scenarios]
    for scenario, smoke in zip(scenarios, smokes):
        if smoke.get("scenario") != scenario:
            raise ValueError("Smoke role/scenario identity differs")
        if smoke.get("runtime_versions") != versions:
            raise ValueError("Recorded smoke runtime differs from admitted runtime")
    smoke_versions = {s: smoke["runtime_versions"] for s, smoke in zip(scenarios, smokes)}
    schema = validate_smokes(smokes, source_pins, scenarios)
    attestation = read(paths["review_attestation"])
    review_fields = {"format", "review_kind", "status", "decision", "reviewer",
                     "source_pins", "runtime_versions", "test_sha256", "evidence_sha256",
                     "smoke_runtime_versions"}
    reviewed_hashes = {role: ref["sha256"] for role, ref in evidence.items() if role != "review_attestation"}
    if (not isinstance(attestation, dict) or set(attestation) != review_fields or
            attestation["format"] != REVIEW_FORMAT or attestation["review_kind"] != "final_admission" or
            attestation["status"] != "final" or attestation["decision"] != "approved" or
            not isinstance(attestation["reviewer"], str) or not attestation["reviewer"].strip() or
            attestation["source_pins"] != source_pins or attestation["runtime_versions"] != versions or
            attestation["test_sha256"] != tests or attestation["evidence_sha256"] != reviewed_hashes or
            attestation["smoke_runtime_versions"] != smoke_versions):
        raise ValueError("Final independent reviewer attestation differs")
    if not paths["review_report"].read_text(encoding="utf-8").strip():
        raise ValueError("Independent human review report is empty")
    return dict(tests=len(cases), observation_dim=len(schema["names"]))


def verify_gate(path):
    from td3 import SCENARIOS
    try:
        gate = read(path)
        fields = {"format", "ready_for_bounded_pilot", "source_pins", "runtime_versions",
                  "scenarios", "evidence", "tests", "observation_dim"}
        if (not isinstance(gate, dict) or set(gate) != fields or gate["format"] != GATE_FORMAT or
                gate["ready_for_bounded_pilot"] is not True or gate["source_pins"] != pins(DEFAULT_SNAPSHOT) or
                gate["runtime_versions"] != runtime_versions() or gate["scenarios"] != list(SCENARIOS)):
            raise ValueError("Admission gate/source/runtime schema mismatch")
        facts = validate_evidence(gate["evidence"], gate["source_pins"], gate["runtime_versions"], SCENARIOS)
        if any(gate[key] != value or type(gate[key]) is not int for key, value in facts.items()):
            raise ValueError("Admission gate summary differs")
        return gate
    except (KeyError, TypeError, OSError, ET.ParseError) as exc:
        raise ValueError("Missing or malformed admission evidence") from exc


def main():
    import sys
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(REPO / ".deps-budget"))
    from td3 import SCENARIOS
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--attestation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_pins, versions = pins(DEFAULT_SNAPSHOT), runtime_versions()
    evidence = {"test_xml": evidence_ref(args.evidence / "tests.xml"),
                "test_record": evidence_ref(args.evidence / "test_source_pins.json"),
                "review_report": evidence_ref(args.review),
                "review_attestation": evidence_ref(args.attestation)}
    evidence.update({"smoke:" + s: evidence_ref(args.evidence / f"smoke_{s}.json") for s in SCENARIOS})
    facts = validate_evidence(evidence, source_pins, versions, SCENARIOS)
    save(args.output, dict(format=GATE_FORMAT, ready_for_bounded_pilot=True, source_pins=source_pins,
        runtime_versions=versions, scenarios=list(SCENARIOS), evidence=evidence, **facts))
    verify_gate(args.output)
    print("MULTI_PREFLIGHT_PASS", facts["tests"], len(SCENARIOS))


if __name__ == "__main__":
    main()
