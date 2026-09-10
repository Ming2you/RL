"""Fail closed when a coordination dataset is not ready for offline training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    RL_RESPONSE_CONTRACT_VERSION,
)
from rl_leader.env import (
    OPTIMIZER_ANCHOR_CONTRACT,
    OPTIMIZER_ANCHOR_TRANSITION_CONTRACT,
    WARMUP_CONTROL_CONTRACT,
)
from rl_leader.data_contract import PSTACK_RESIDUAL_DATA_CONTRACT
from rl_leader.experiment_contract import EXPERIMENT_PROFILE_ID


REQUIRED_TARGETS = {
    "sweet_155_w60",
    "sweet_170_w60",
    "sweet_170_incident_w60",
    "sweet_170_skew15_w60",
    "sweet_190_w60",
}


def validate(
    report: dict,
    minimum_transitions: int,
    *,
    allow_partial_residual_support: bool = False,
) -> list[str]:
    failures = []
    if not bool(report.get("experiment_contract_valid", False)):
        failures.append("experiment contracts are missing, invalid, or unresolved")
    if report.get("experiment_profiles") != [EXPERIMENT_PROFILE_ID]:
        failures.append(
            "dataset experiment profile does not match the canonical Phase 0 profile"
        )
    if not report.get("experiment_contract_sha256"):
        failures.append("dataset has no verified experiment contract fingerprints")
    if int(report.get("transitions", 0)) < int(minimum_transitions):
        failures.append(f"transition count is below {minimum_transitions}")
    if float(report.get("validity_pass_fraction", 0.0)) < 0.999:
        failures.append("validity pass fraction is below 0.999")
    if report.get("response_contract") != RL_RESPONSE_CONTRACT_VERSION:
        failures.append("response contract does not match the deployable v6 controller")
    if report.get("action_schema_version") != ACTION_SCHEMA_VERSION:
        failures.append("action schema version does not match the deployable v6 controller")
    if report.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        failures.append("observation schema version does not match the deployable v6 controller")
    if report.get("optimizer_anchor_transition_contracts") != [
        OPTIMIZER_ANCHOR_TRANSITION_CONTRACT
    ]:
        failures.append("optimizer anchors did not use the native P-Stack anchor gate")
    if report.get("warmup_control_contracts") != [WARMUP_CONTROL_CONTRACT]:
        failures.append("dataset warm-up did not use the native uncontrolled contract")
    if report.get("optimizer_anchor_contracts") != [
        OPTIMIZER_ANCHOR_CONTRACT
    ]:
        failures.append("optimizer anchors did not use native production P-Stack")
    if report.get("pfo_supervisor_flags") != [False]:
        failures.append("external PFO supervisor must be disabled for every dataset component")
    if report.get("pstack_anchor_flags") != [True]:
        failures.append("P-Stack anchor gate must be enabled for every dataset component")
    if report.get("action_parameterization_supports") != [
        PSTACK_RESIDUAL_DATA_CONTRACT
    ]:
        failures.append("dataset does not use the anchored residual action contract")
    teacher = report.get("optimizer_teacher_labels", {})
    if int(teacher.get("labeled_transitions", 0)) <= 0:
        failures.append("optimizer teacher branch labels are missing")
    if int(teacher.get("outer_pfo_transitions", 0)) != 0:
        failures.append("optimizer teacher contains external PFO supervisor transitions")
    anchor_replay = teacher.get("pstack_selected_anchor_replay", {})
    if int(anchor_replay.get("exact_transitions", 0)) <= 0:
        failures.append("dataset contains no exact P-Stack-selected zero residuals")
    if float(anchor_replay.get("exact_fraction", 0.0)) < 0.999:
        failures.append("P-Stack-selected zero residuals do not replay exactly")
    residual = report.get("pstack_residual_actor_supervision", {})
    if int(residual.get("eligible_transitions", 0)) <= 0:
        failures.append("P-Stack residual actor has no eligible supervision transitions")
    if int(residual.get("anchor_gate_rl_pick_transitions", 0)) <= 0:
        failures.append("P-Stack anchor gate never selected an exploratory RL response")
    residual_abs_max = residual.get("residual_abs_max")
    if residual_abs_max is None or float(residual_abs_max) > 1.0 + 1.0e-6:
        failures.append("P-Stack residual targets exceed the actor action range")
    if residual.get("dead_dimension_indices") and not allow_partial_residual_support:
        failures.append(
            "dead P-Stack residual dimensions: "
            f"{residual['dead_dimension_indices']}"
        )
    if int(residual.get("nonzero_peak_transitions", 0)) <= 0:
        failures.append("dataset contains no nonzero peak residual exploration")
    if int(residual.get("nonzero_recovery_transitions", 0)) <= 0:
        failures.append("dataset contains no nonzero recovery residual exploration")
    targets = set(report.get("target_scenarios", {}))
    required_targets = (
        {"sweet_170_w60", "sweet_190_w60"}
        if allow_partial_residual_support
        else REQUIRED_TARGETS
    )
    missing_targets = sorted(required_targets - targets)
    if missing_targets:
        failures.append(f"missing target scenarios: {missing_targets}")
    if int(report.get("phase_transitions", {}).get("recovery", 0)) <= 0:
        failures.append("dataset contains no recovery transitions")
    frozen = set(residual.get("frozen_dimension_indices", []))
    dead_trainable = sorted(
        set(report.get("action_support", {}).get("dead_dimension_indices", []))
        - frozen
    )
    if dead_trainable and not allow_partial_residual_support:
        failures.append(
            f"dead trainable action dimensions: {dead_trainable}"
        )
    certificate_blocks = [
        block for block in report.get("blocks", []) if block.get("family") == "certificate"
    ]
    if not certificate_blocks:
        failures.append("release certificate action blocks are missing")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--minimum-transitions", type=int, default=10000)
    parser.add_argument("--allow-partial-residual-support", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    failures = validate(
        report,
        args.minimum_transitions,
        allow_partial_residual_support=args.allow_partial_residual_support,
    )
    result = {
        "format_version": "coordination_v6_dataset_gate_v1",
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "response_contract": RL_RESPONSE_CONTRACT_VERSION,
        "allow_partial_residual_support": bool(args.allow_partial_residual_support),
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
