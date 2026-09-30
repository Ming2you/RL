"""Freeze inspectable evidence before admitting the bounded traffic pilot."""
import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from budget_runtime import DEFAULT_SNAPSHOT, read, save
from run_budget import file_hash, pins


def build(evidence, reviews, output):
    if output.exists():
        raise FileExistsError(output)
    records = {}

    def record(path):
        records[str(path.resolve())] = file_hash(path)

    for name in ("parity_step30_repaired.json", "isolation_step30.json", "smoke_resume_config_contract.json"):
        path = evidence / name
        if read(path).get("passed") is not True:
            raise ValueError("Missing or failed diagnostic: " + name)
        record(path)
    for step in (5, 10, 30, 50):
        path = evidence / f"probe_step{step}" / "summary.json"
        row = read(path)
        if row.get("status") != "completed" or row.get("action_influence_observed") is not True:
            raise ValueError("Action influence gate failed")
        record(path)
    tests = evidence / "preflight_tests.xml"
    root = ET.parse(tests).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    count = sum(int(s.get("tests", 0)) for s in suites)
    if count == 0 or any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped")):
        raise ValueError("Incomplete or failed unit regression suite")
    record(tests)
    if not reviews:
        raise ValueError("Independent final review required")
    for path in reviews:
        if "PILOT_REVIEW: PASS" not in path.read_text(encoding="utf-8"):
            raise ValueError("Final pilot review did not pass")
        record(path)
    save(output, dict(ready_for_bounded_pilot=True, created_unix=time.time(),
                     source_pins=pins(DEFAULT_SNAPSHOT), evidence_sha256=records,
                     unit_tests=count, scope="two_training_episodes_and_four_frozen_evaluations",
                     legacy_ddqn_resumed=False, performance_acceptance=False,
                     influence_probe_role="diagnostic_only_not_training_targets"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--review", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.evidence, args.review, args.output)


if __name__ == "__main__":
    main()
