"""Fail closed when a coordination-v4 dataset is not ready for offline training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    RL_RESPONSE_CONTRACT_VERSION,
)


REQUIRED_TARGETS = {
    "sweet_155_w60",
    "sweet_170_w60",
    "sweet_170_incident_w60",
    "sweet_170_skew15_w60",
    "sweet_190_w60",
}


def validate(report: dict, minimum_transitions: int) -> list[str]:
    failures = []
    if int(report.get("transitions", 0)) < int(minimum_transitions):
        failures.append(f"transition count is below {minimum_transitions}")
    if float(report.get("validity_pass_fraction", 0.0)) < 0.999:
        failures.append("validity pass fraction is below 0.999")
    if report.get("response_contract") != RL_RESPONSE_CONTRACT_VERSION:
        failures.append("response contract does not match the deployable v4 controller")
    if report.get("action_schema_version") != ACTION_SCHEMA_VERSION:
        failures.append("action schema version does not match the deployable v4 controller")
    if report.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        failures.append("observation schema version does not match the deployable v4 controller")
    if report.get("optimizer_anchor_transition_contracts") != ["rl_adapter_replay_v1"]:
        failures.append("optimizer anchors did not exclusively use RL adapter replay")
    targets = set(report.get("target_scenarios", {}))
    missing_targets = sorted(REQUIRED_TARGETS - targets)
    if missing_targets:
        failures.append(f"missing target scenarios: {missing_targets}")
    if int(report.get("phase_transitions", {}).get("recovery", 0)) <= 0:
        failures.append("dataset contains no recovery transitions")
    if report.get("action_support", {}).get("dead_dimension_indices"):
        failures.append(
            "dead action dimensions: "
            f"{report['action_support']['dead_dimension_indices']}"
        )
    certificate_blocks = [
        block for block in report.get("blocks", []) if block.get("family") == "certificate"
    ]
    if not certificate_blocks:
        failures.append("release certificate action blocks are missing")
    for block in certificate_blocks:
        fraction = float(block.get("release_true_fraction", -1.0))
        if not 0.01 <= fraction <= 0.99:
            failures.append(
                f"certificate {block.get('owner')} lacks both outcomes: true_fraction={fraction}"
            )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--minimum-transitions", type=int, default=10000)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    failures = validate(report, args.minimum_transitions)
    result = {
        "format_version": "coordination_v4_dataset_gate_v1",
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "response_contract": RL_RESPONSE_CONTRACT_VERSION,
        "passed": not failures,
        "failures": failures,
        "audit": str(Path(args.audit)),
    }
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
