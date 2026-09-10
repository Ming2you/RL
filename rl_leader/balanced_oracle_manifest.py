"""Freeze and validate balanced owner-block oracle states."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rl_leader.diagnose_candidate_ablation import (
    ROOT,
    _implementation_fingerprints,
)
from rl_leader.diagnose_pcent_guided_oracle import _runtime_fingerprint
from rl_leader.diagnose_phase0_parity import (
    _digest,
    _normalize,
    _physical_snapshot,
    controller_behavior_fingerprint,
)
from rl_leader.env import RLLeaderEnv
from rl_leader.oracle_candidate_ablation import AblationConfig


FROZEN_MANIFEST_FORMAT = "balanced_oracle_frozen_states_v2"
FROZEN_MANIFEST_CONTRACT = "pstack_stratified_anchor_snapshot_v2"
FROZEN_MANIFEST_FORMAT_V3 = "balanced_oracle_frozen_states_v3"
FROZEN_MANIFEST_CONTRACT_V3 = "pstack_dense_anchor_snapshot_v3"
FROZEN_MANIFEST_FORMAT_V3_EXTRA = "balanced_oracle_frozen_states_v3_extra"
FROZEN_MANIFEST_CONTRACT_V3_EXTRA = "pstack_dense_anchor_supplement_v1"
FROZEN_MANIFEST_FORMAT_V4 = "balanced_oracle_frozen_states_v4_extended"
FROZEN_MANIFEST_CONTRACT_V4 = "pstack_dense_anchor_snapshot_v4_extended"
DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS = 50
BALANCED_SCENARIOS = (
    "sweet_155_w60",
    "sweet_170_w60",
    "sweet_170_incident_w60",
    "sweet_170_skew15_w60",
    "sweet_190_w60",
)
COMMON_FROZEN_EVENTS = (
    ("ramp_up", 1),
    ("plateau", 9),
    ("recovery_boundary", 24),
)
INCIDENT_FROZEN_EVENTS = (
    ("incident_onset", 5),
    ("incident_end", 15),
)
DENSE_COMMON_FROZEN_EVENTS = (
    ("ramp_up", 1),
    ("ramp_up", 3),
    ("growth", 5),
    ("growth", 7),
    ("plateau", 9),
    ("plateau", 11),
    ("plateau", 13),
    ("plateau", 15),
    ("plateau", 17),
    ("plateau", 19),
    ("late_plateau", 21),
    ("recovery_boundary", 24),
    ("early_recovery", 27),
    ("early_recovery", 30),
    ("recovery", 33),
    ("recovery", 36),
    ("recovery", 40),
    ("late_recovery", 45),
    ("late_recovery", 50),
    ("tail_monitor", 55),
)
DENSE_INCIDENT_LABEL_OVERRIDES = {
    5: "incident_onset",
    15: "incident_end",
}
DENSE_SUPPLEMENT_EVENT_PLAN_POLICY = "dense_supplement_steps_v1"
DENSE_EXTENDED_EVENT_PLAN_POLICY = "dense_common_plus_supplement_v1"
MANIFEST_DEPENDENCY_SOURCES = tuple(sorted(set(
    tuple(_implementation_fingerprints()) + (
        "rl_leader/experiment_contract.py",
        "src/controllers/pstack_factory.py",
        "src/models/demand.py",
        "src/models/state.py",
        "src/models/metanet.py",
        "src/models/urban_queue_model.py",
        "src/config/default.yaml",
        "src/config/scenarios.yaml",
    )
)))


def frozen_events_for_scenario(scenario: str) -> tuple[tuple[str, int], ...]:
    events = list(COMMON_FROZEN_EVENTS)
    if scenario == "sweet_170_incident_w60":
        events.extend(INCIDENT_FROZEN_EVENTS)
    return tuple(sorted(events, key=lambda item: item[1]))


def dense_frozen_events_for_scenario(scenario: str) -> tuple[tuple[str, int], ...]:
    if scenario not in BALANCED_SCENARIOS:
        raise ValueError(f"scenario is not in balanced pilot: {scenario}")
    events = []
    for stratum, step in DENSE_COMMON_FROZEN_EVENTS:
        if scenario == "sweet_170_incident_w60":
            stratum = DENSE_INCIDENT_LABEL_OVERRIDES.get(step, stratum)
        events.append((stratum, step))
    return tuple(sorted(events, key=lambda item: item[1]))


def dense_stratum_for_policy_step(scenario: str, policy_step: int) -> str:
    if scenario not in BALANCED_SCENARIOS:
        raise ValueError(f"scenario is not in balanced pilot: {scenario}")
    step = int(policy_step)
    if scenario == "sweet_170_incident_w60" and step in DENSE_INCIDENT_LABEL_OVERRIDES:
        return DENSE_INCIDENT_LABEL_OVERRIDES[step]
    if step <= 3:
        return "ramp_up"
    if step <= 7:
        return "growth"
    if step <= 20:
        return "plateau"
    if step <= 23:
        return "late_plateau"
    if step <= 26:
        return "recovery_boundary"
    if step <= 30:
        return "early_recovery"
    if step <= 40:
        return "recovery"
    if step <= 50:
        return "late_recovery"
    return "tail_monitor"


def dense_extra_events_for_scenario(
    scenario: str,
    policy_steps: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    if scenario not in BALANCED_SCENARIOS:
        raise ValueError(f"scenario is not in balanced pilot: {scenario}")
    steps = [int(step) for step in policy_steps]
    if not steps:
        raise ValueError("dense supplement requires policy steps")
    if len(steps) != len(set(steps)):
        raise ValueError("dense supplement policy steps must be unique")
    base_steps = {step for _, step in dense_frozen_events_for_scenario(scenario)}
    duplicated = sorted(set(steps) & base_steps)
    if duplicated:
        raise ValueError(
            f"dense supplement duplicates dense base steps: {duplicated}"
        )
    return tuple(
        sorted(
            (
                (dense_stratum_for_policy_step(scenario, step), step)
                for step in steps
            ),
            key=lambda item: item[1],
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_manifest() -> dict:
    return {
        "format_version": FROZEN_MANIFEST_FORMAT,
        "contract": FROZEN_MANIFEST_CONTRACT,
        "implementation_sha256": _implementation_fingerprints(),
        "scenario_order": list(BALANCED_SCENARIOS),
        "dependency_sha256": {
            relative: _sha256_file(ROOT / relative)
            for relative in MANIFEST_DEPENDENCY_SOURCES
        },
        "event_plan": {
            "common": [
                {"stratum": stratum, "policy_step": step}
                for stratum, step in COMMON_FROZEN_EVENTS
            ],
            "incident_extra": [
                {"stratum": stratum, "policy_step": step}
                for stratum, step in INCIDENT_FROZEN_EVENTS
            ],
        },
        "candidate_contract": {
            "candidate_mode": "owner_block_v2_urban",
            "master_seed": 20260828,
            "magnitude_levels": [0.25, 0.5],
            "templates": ["signed_axis", "signed_corner"],
            "screening_horizon": 1,
        },
        "scenarios": [],
        "passed": False,
    }


def _new_dense_manifest() -> dict:
    return {
        "format_version": FROZEN_MANIFEST_FORMAT_V3,
        "contract": FROZEN_MANIFEST_CONTRACT_V3,
        "implementation_sha256": _implementation_fingerprints(),
        "scenario_order": list(BALANCED_SCENARIOS),
        "dependency_sha256": {
            relative: _sha256_file(ROOT / relative)
            for relative in MANIFEST_DEPENDENCY_SOURCES
        },
        "event_plan": {
            "policy": "dense_common_steps_with_incident_label_overrides_v1",
            "minimum_coordination_eligible_events": (
                DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS
            ),
            "common": [
                {"stratum": stratum, "policy_step": step}
                for stratum, step in DENSE_COMMON_FROZEN_EVENTS
            ],
            "incident_label_overrides": [
                {"stratum": stratum, "policy_step": step}
                for step, stratum in sorted(DENSE_INCIDENT_LABEL_OVERRIDES.items())
            ],
        },
        "candidate_contract": {
            "candidate_mode": "owner_block_v2_all",
            "master_seed": 20260830,
            "magnitude_levels": [0.25, 0.5],
            "templates": ["signed_axis", "signed_corner"],
            "screening_horizon": 1,
        },
        "scenarios": [],
        "passed": False,
    }


def _new_dense_extra_manifest(
    scenario: str,
    policy_steps: tuple[int, ...],
) -> dict:
    events = dense_extra_events_for_scenario(scenario, policy_steps)
    result = _new_dense_manifest()
    result["format_version"] = FROZEN_MANIFEST_FORMAT_V3_EXTRA
    result["contract"] = FROZEN_MANIFEST_CONTRACT_V3_EXTRA
    result["event_plan"] = {
        "policy": DENSE_SUPPLEMENT_EVENT_PLAN_POLICY,
        "base_policy": result["event_plan"]["policy"],
        "minimum_coordination_eligible_events": (
            DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS
        ),
        "scenarios": [{
            "scenario": scenario,
            "events": [
                {"stratum": stratum, "policy_step": step}
                for stratum, step in events
            ],
        }],
    }
    result["passed"] = False
    return result


def _new_dense_extended_manifest(scenarios: list[dict]) -> dict:
    result = _new_dense_manifest()
    result["format_version"] = FROZEN_MANIFEST_FORMAT_V4
    result["contract"] = FROZEN_MANIFEST_CONTRACT_V4
    result["event_plan"] = {
        "policy": DENSE_EXTENDED_EVENT_PLAN_POLICY,
        "base_policy": result["event_plan"]["policy"],
        "supplement_policy": DENSE_SUPPLEMENT_EVENT_PLAN_POLICY,
        "minimum_coordination_eligible_events": (
            DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS
        ),
        "scenarios": [
            {
                "scenario": scenario["scenario"],
                "events": [
                    {"stratum": event["stratum"], "policy_step": int(event["policy_step"])}
                    for event in sorted(
                        scenario.get("events", []),
                        key=lambda item: int(item["policy_step"]),
                    )
                ],
            }
            for scenario in scenarios
        ],
    }
    result["passed"] = False
    return result


def _freeze_event(env: RLLeaderEnv, stratum: str, policy_step: int, context) -> dict:
    observation = env._observe().astype(float).tolist()
    anchor_envelope = env.action_schema.serialize_anchor(
        context.coordination
    ).astype(float).tolist()
    physical = _physical_snapshot(env)
    current_demand = env.profile.at(float(env.sim.state.time_sec))
    optimizer = getattr(env, "optimizer_controller", None)
    branch = str(context.coordination.selected_branch)
    return {
        "stratum": stratum,
        "stratum_predicate": {
            "simulation_time_sec": float(env.sim.state.time_sec),
            "pulse_fraction": env.profile._pulse_fraction(
                float(env.sim.state.time_sec)
            ),
            "incident_active": bool(current_demand.freeway_lane_loss),
        },
        "policy_step": int(policy_step),
        "simulation_step": int(env.step_idx),
        "simulation_time_sec": float(env.sim.state.time_sec),
        "native_anchor_branch": branch,
        "coordination_eligible": branch in AblationConfig().allowed_anchor_branches,
        "anchor_fingerprint": str(context.anchor_fingerprint),
        "pre_runtime_sha256": _runtime_fingerprint(env),
        "forecast_sha256": _digest(context.forecast),
        "forecast": _normalize(context.forecast),
        "current_demand": _normalize(current_demand),
        "observation_sha256": _digest(observation),
        "observation": observation,
        "anchor_envelope_sha256": _digest(anchor_envelope),
        "anchor_envelope": anchor_envelope,
        "physical_snapshot_sha256": _digest(physical),
        "physical_snapshot": physical,
        "rl_controller_sha256": controller_behavior_fingerprint(env.controller),
        "optimizer_controller_sha256": (
            None if optimizer is None
            else controller_behavior_fingerprint(optimizer)
        ),
    }


def _event_differences(actual: dict, expected: dict) -> list[str]:
    return sorted(
        key for key in set(actual) | set(expected)
        if _normalize(actual.get(key)) != _normalize(expected.get(key))
    )


def freeze_scenario(scenario: str, output_path: Path) -> dict:
    if scenario not in BALANCED_SCENARIOS:
        raise ValueError(f"scenario is not in balanced pilot: {scenario}")
    result = _new_manifest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    started = time.monotonic()
    env = RLLeaderEnv(
        scenario_name=scenario,
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    events = {step: stratum for stratum, step in frozen_events_for_scenario(scenario)}
    max_event = max(events)
    frozen = {
        "scenario": scenario,
        **env.experiment_contract.artifact_fields(),
        "action_schema": env.action_schema.metadata(),
        "observation_schema": env.observation_schema.metadata(),
        "control_interval_sec": float(env.dt),
        "warmup_steps": int(env.warmup),
        "events": [],
    }
    while int(env.step_idx - env.warmup) <= max_event:
        step = int(env.step_idx - env.warmup)
        if step in events:
            context = env.prepare_pstack_anchor_context()
            event = _freeze_event(env, events[step], step, context)
            frozen["events"].append(event)
            result["pending_event"] = event
            result["scenarios"] = [frozen]
            result["elapsed_sec"] = float(time.monotonic() - started)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"freeze scenario={scenario} phase={events[step]} step={step} "
                f"branch={event['native_anchor_branch']}",
                flush=True,
            )
            if step == max_event:
                break
            env.step_prepared_optimizer_anchor(context)
        else:
            env.step_optimizer_anchor(sync_follower_state=True)
    result.pop("pending_event", None)
    result["scenarios"] = [frozen]
    result["elapsed_sec"] = float(time.monotonic() - started)
    result["passed"] = True
    validate_frozen_manifest(result, require_all_scenarios=False)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def freeze_dense_scenario(scenario: str, output_path: Path) -> dict:
    return _freeze_dense_scenario(scenario, output_path, resume=False)


def _partial_dense_events(output_path: Path, scenario: str) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FROZEN_MANIFEST_FORMAT_V3:
        raise ValueError("dense resume found an incompatible manifest format")
    scenarios = payload.get("scenarios", [])
    if len(scenarios) != 1 or scenarios[0].get("scenario") != scenario:
        raise ValueError("dense resume found an incompatible scenario")
    events = list(scenarios[0].get("events", []))
    expected_steps = [step for _, step in dense_frozen_events_for_scenario(scenario)]
    actual_steps = [int(event.get("policy_step", -1)) for event in events]
    if len(actual_steps) != len(set(actual_steps)):
        raise ValueError("dense resume found duplicate policy steps")
    if actual_steps != expected_steps[:len(actual_steps)]:
        raise ValueError("dense resume requires a prefix of the dense event plan")
    return events


def _freeze_dense_scenario(
    scenario: str,
    output_path: Path,
    *,
    resume: bool,
) -> dict:
    if scenario not in BALANCED_SCENARIOS:
        raise ValueError(f"scenario is not in balanced pilot: {scenario}")
    existing_events = _partial_dense_events(output_path, scenario) if resume else []
    existing_by_step = {
        int(event["policy_step"]): event for event in existing_events
    }
    result = _new_dense_manifest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not existing_events:
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    started = time.monotonic()
    env = RLLeaderEnv(
        scenario_name=scenario,
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    events = {step: stratum for stratum, step in dense_frozen_events_for_scenario(scenario)}
    max_event = max(events)
    frozen = {
        "scenario": scenario,
        **env.experiment_contract.artifact_fields(),
        "action_schema": env.action_schema.metadata(),
        "observation_schema": env.observation_schema.metadata(),
        "control_interval_sec": float(env.dt),
        "warmup_steps": int(env.warmup),
        "events": [],
    }
    while int(env.step_idx - env.warmup) <= max_event:
        step = int(env.step_idx - env.warmup)
        if step in events:
            context = env.prepare_pstack_anchor_context()
            actual = _freeze_event(env, events[step], step, context)
            resumed_event = step in existing_by_step
            event = existing_by_step.get(step, actual)
            if resumed_event:
                differing = _event_differences(actual, event)
                if differing:
                    raise ValueError(
                        f"dense resume frozen state drift at {scenario}:{step}: "
                        f"{differing}"
                    )
            frozen["events"].append(event)
            if not resumed_event:
                result["pending_event"] = event
                result["scenarios"] = [frozen]
                result["elapsed_sec"] = float(time.monotonic() - started)
                result["resume_from_events"] = len(existing_events)
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"freeze-dense scenario={scenario} phase={events[step]} step={step} "
                f"branch={event['native_anchor_branch']} "
                f"resumed={resumed_event}",
                flush=True,
            )
            if step == max_event:
                break
            env.step_prepared_optimizer_anchor(context)
        else:
            env.step_optimizer_anchor(sync_follower_state=True)
    result.pop("pending_event", None)
    result["scenarios"] = [frozen]
    result["elapsed_sec"] = float(time.monotonic() - started)
    result["resume_from_events"] = len(existing_events)
    result["passed"] = True
    validate_frozen_manifest(result, require_all_scenarios=False)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def resume_dense_scenario(scenario: str, output_path: Path) -> dict:
    return _freeze_dense_scenario(scenario, output_path, resume=True)


def _partial_dense_extra_events(
    output_path: Path,
    scenario: str,
    policy_steps: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FROZEN_MANIFEST_FORMAT_V3_EXTRA:
        raise ValueError("dense supplement resume found an incompatible manifest format")
    scenarios = payload.get("scenarios", [])
    if len(scenarios) != 1 or scenarios[0].get("scenario") != scenario:
        raise ValueError("dense supplement resume found an incompatible scenario")
    if payload.get("event_plan") != _new_dense_extra_manifest(
        scenario, policy_steps
    )["event_plan"]:
        raise ValueError("dense supplement resume found an incompatible event plan")
    events = list(scenarios[0].get("events", []))
    expected_steps = [
        step for _, step in dense_extra_events_for_scenario(scenario, policy_steps)
    ]
    actual_steps = [int(event.get("policy_step", -1)) for event in events]
    if len(actual_steps) != len(set(actual_steps)):
        raise ValueError("dense supplement resume found duplicate policy steps")
    if actual_steps != expected_steps[:len(actual_steps)]:
        raise ValueError(
            "dense supplement resume requires a prefix of the event plan"
        )
    return events


def _freeze_dense_extra_scenario(
    scenario: str,
    policy_steps: tuple[int, ...],
    output_path: Path,
    *,
    resume: bool,
) -> dict:
    if scenario not in BALANCED_SCENARIOS:
        raise ValueError(f"scenario is not in balanced pilot: {scenario}")
    planned_events = dense_extra_events_for_scenario(scenario, policy_steps)
    existing_events = (
        _partial_dense_extra_events(output_path, scenario, policy_steps)
        if resume else []
    )
    existing_by_step = {
        int(event["policy_step"]): event for event in existing_events
    }
    result = _new_dense_extra_manifest(scenario, policy_steps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not existing_events:
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    started = time.monotonic()
    env = RLLeaderEnv(
        scenario_name=scenario,
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    events = {step: stratum for stratum, step in planned_events}
    max_event = max(events)
    frozen = {
        "scenario": scenario,
        **env.experiment_contract.artifact_fields(),
        "action_schema": env.action_schema.metadata(),
        "observation_schema": env.observation_schema.metadata(),
        "control_interval_sec": float(env.dt),
        "warmup_steps": int(env.warmup),
        "events": [],
    }
    while int(env.step_idx - env.warmup) <= max_event:
        step = int(env.step_idx - env.warmup)
        if step in events:
            context = env.prepare_pstack_anchor_context()
            actual = _freeze_event(env, events[step], step, context)
            resumed_event = step in existing_by_step
            event = existing_by_step.get(step, actual)
            if resumed_event:
                differing = _event_differences(actual, event)
                if differing:
                    raise ValueError(
                        f"dense supplement resume frozen state drift at "
                        f"{scenario}:{step}: {differing}"
                    )
            frozen["events"].append(event)
            if not resumed_event:
                result["pending_event"] = event
                result["scenarios"] = [frozen]
                result["elapsed_sec"] = float(time.monotonic() - started)
                result["resume_from_events"] = len(existing_events)
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"freeze-dense-extra scenario={scenario} "
                f"phase={events[step]} step={step} "
                f"branch={event['native_anchor_branch']} "
                f"resumed={resumed_event}",
                flush=True,
            )
            if step == max_event:
                break
            env.step_prepared_optimizer_anchor(context)
        else:
            env.step_optimizer_anchor(sync_follower_state=True)
    result.pop("pending_event", None)
    result["scenarios"] = [frozen]
    result["elapsed_sec"] = float(time.monotonic() - started)
    result["resume_from_events"] = len(existing_events)
    result["passed"] = True
    validate_frozen_manifest(result, require_all_scenarios=False)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def freeze_dense_extra_scenario(
    scenario: str,
    policy_steps: tuple[int, ...],
    output_path: Path,
) -> dict:
    return _freeze_dense_extra_scenario(
        scenario, policy_steps, output_path, resume=False
    )


def resume_dense_extra_scenario(
    scenario: str,
    policy_steps: tuple[int, ...],
    output_path: Path,
) -> dict:
    return _freeze_dense_extra_scenario(
        scenario, policy_steps, output_path, resume=True
    )


def _dynamic_event_plan_events(
    manifest: dict,
    scenario: str,
) -> tuple[tuple[str, int], ...]:
    for row in manifest.get("event_plan", {}).get("scenarios", []):
        if row.get("scenario") == scenario:
            return tuple(
                (str(event["stratum"]), int(event["policy_step"]))
                for event in row.get("events", [])
            )
    raise ValueError("dynamic frozen manifest event plan is missing a scenario")


def _validate_dynamic_event_plan(manifest: dict) -> None:
    plan = manifest.get("event_plan", {})
    if not isinstance(plan, dict):
        raise ValueError("dynamic frozen manifest event plan is invalid")
    format_version = manifest.get("format_version")
    rows = plan.get("scenarios", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("dynamic frozen manifest has no scenario event plan")
    names = [row.get("scenario") for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("dynamic frozen manifest has duplicate scenario plans")
    if format_version == FROZEN_MANIFEST_FORMAT_V3_EXTRA:
        if plan.get("policy") != DENSE_SUPPLEMENT_EVENT_PLAN_POLICY:
            raise ValueError("unexpected dense supplement event-plan policy")
        if plan.get("base_policy") != _new_dense_manifest()["event_plan"]["policy"]:
            raise ValueError("unexpected dense supplement base policy")
        for row in rows:
            scenario = row.get("scenario")
            if scenario not in BALANCED_SCENARIOS:
                raise ValueError("dense supplement contains an unknown scenario")
            steps = tuple(int(event["policy_step"]) for event in row["events"])
            expected = [
                {"stratum": stratum, "policy_step": step}
                for stratum, step in dense_extra_events_for_scenario(
                    str(scenario), steps
                )
            ]
            actual = [
                {"stratum": event.get("stratum"), "policy_step": int(event["policy_step"])}
                for event in row["events"]
            ]
            if actual != expected:
                raise ValueError("dense supplement event plan drift")
        return
    if format_version != FROZEN_MANIFEST_FORMAT_V4:
        raise ValueError("unexpected dynamic frozen manifest format")
    if plan.get("policy") != DENSE_EXTENDED_EVENT_PLAN_POLICY:
        raise ValueError("unexpected dense extended event-plan policy")
    if plan.get("base_policy") != _new_dense_manifest()["event_plan"]["policy"]:
        raise ValueError("unexpected dense extended base policy")
    if plan.get("supplement_policy") != DENSE_SUPPLEMENT_EVENT_PLAN_POLICY:
        raise ValueError("unexpected dense extended supplement policy")
    if int(plan.get("minimum_coordination_eligible_events", 0)) != (
        DENSE_MIN_COORDINATION_ELIGIBLE_EVENTS
    ):
        raise ValueError("unexpected dense extended eligibility threshold")
    if names != list(BALANCED_SCENARIOS):
        raise ValueError("dense extended scenario plan order drift")
    for row in rows:
        scenario = str(row["scenario"])
        actual = [
            {"stratum": event.get("stratum"), "policy_step": int(event["policy_step"])}
            for event in row.get("events", [])
        ]
        sorted_actual = sorted(actual, key=lambda item: item["policy_step"])
        if actual != sorted_actual:
            raise ValueError("dense extended event plan order drift")
        steps = [event["policy_step"] for event in actual]
        if len(steps) != len(set(steps)):
            raise ValueError("dense extended event plan has duplicate policy steps")
        base_plan = [
            {"stratum": stratum, "policy_step": step}
            for stratum, step in dense_frozen_events_for_scenario(scenario)
        ]
        missing_base = [event for event in base_plan if event not in actual]
        if missing_base:
            raise ValueError("dense extended event plan is missing base events")


def _manifest_template(manifest: dict) -> dict:
    if manifest.get("format_version") == FROZEN_MANIFEST_FORMAT:
        return _new_manifest()
    if manifest.get("format_version") == FROZEN_MANIFEST_FORMAT_V3:
        return _new_dense_manifest()
    raise ValueError("unexpected frozen manifest format")


def _expected_events(manifest: dict, scenario: str) -> tuple[tuple[str, int], ...]:
    if manifest.get("format_version") == FROZEN_MANIFEST_FORMAT:
        return frozen_events_for_scenario(scenario)
    if manifest.get("format_version") == FROZEN_MANIFEST_FORMAT_V3:
        return dense_frozen_events_for_scenario(scenario)
    if manifest.get("format_version") in (
        FROZEN_MANIFEST_FORMAT_V3_EXTRA,
        FROZEN_MANIFEST_FORMAT_V4,
    ):
        return _dynamic_event_plan_events(manifest, scenario)
    raise ValueError("unexpected frozen manifest format")


def validate_frozen_manifest(
    manifest: dict,
    *,
    require_all_scenarios: bool,
    allow_implementation_drift: bool = False,
    allow_dependency_drift: bool = False,
) -> None:
    contracts = {
        FROZEN_MANIFEST_FORMAT: FROZEN_MANIFEST_CONTRACT,
        FROZEN_MANIFEST_FORMAT_V3: FROZEN_MANIFEST_CONTRACT_V3,
        FROZEN_MANIFEST_FORMAT_V3_EXTRA: FROZEN_MANIFEST_CONTRACT_V3_EXTRA,
        FROZEN_MANIFEST_FORMAT_V4: FROZEN_MANIFEST_CONTRACT_V4,
    }
    if manifest.get("format_version") not in contracts:
        raise ValueError("unexpected frozen manifest format")
    if manifest.get("contract") != contracts[manifest["format_version"]]:
        raise ValueError("unexpected frozen manifest contract")
    if (
        not allow_implementation_drift
        and manifest.get("implementation_sha256") != _implementation_fingerprints()
    ):
        raise ValueError("frozen manifest implementation drift")
    if manifest["format_version"] in (
        FROZEN_MANIFEST_FORMAT,
        FROZEN_MANIFEST_FORMAT_V3,
    ):
        template = _manifest_template(manifest)
        if manifest.get("event_plan") != template["event_plan"]:
            raise ValueError("frozen manifest event plan drift")
        expected_dependency = template["dependency_sha256"]
    else:
        _validate_dynamic_event_plan(manifest)
        expected_dependency = _new_dense_manifest()["dependency_sha256"]
        if manifest.get("candidate_contract") != _new_dense_manifest()["candidate_contract"]:
            raise ValueError("dynamic frozen manifest candidate contract drift")
    if (
        not allow_dependency_drift
        and manifest.get("dependency_sha256") != expected_dependency
    ):
        raise ValueError("frozen manifest dependency drift")
    scenarios = manifest.get("scenarios", [])
    names = [row.get("scenario") for row in scenarios]
    expected_names = list(BALANCED_SCENARIOS) if require_all_scenarios else names
    if names != expected_names or len(names) != len(set(names)):
        raise ValueError("frozen manifest scenario order or uniqueness drift")
    for scenario in scenarios:
        if scenario["scenario"] not in BALANCED_SCENARIOS:
            raise ValueError("frozen manifest contains an unknown scenario")
        events = scenario.get("events", [])
        expected_scenario_plan = [
            {"stratum": stratum, "policy_step": step}
            for stratum, step in _expected_events(manifest, scenario["scenario"])
        ]
        actual_plan = [
            {"stratum": row.get("stratum"), "policy_step": row.get("policy_step")}
            for row in events
        ]
        if actual_plan != expected_scenario_plan:
            raise ValueError(f"incomplete frozen events for {scenario['scenario']}")
        if scenario.get("experiment_contract_sha256") != _digest(
            scenario.get("experiment_contract")
        ):
            raise ValueError("frozen experiment contract payload drift")
        dt = float(scenario["control_interval_sec"])
        warmup = int(scenario["warmup_steps"])
        for event in events:
            expected_simulation_step = warmup + int(event["policy_step"])
            if int(event["simulation_step"]) != expected_simulation_step:
                raise ValueError("frozen event simulation step drift")
            if float(event["simulation_time_sec"]) != expected_simulation_step * dt:
                raise ValueError("frozen event simulation time drift")
            for field in (
                "native_anchor_branch",
                "anchor_fingerprint",
                "pre_runtime_sha256",
                "forecast_sha256",
                "observation_sha256",
                "anchor_envelope_sha256",
                "physical_snapshot_sha256",
                "rl_controller_sha256",
            ):
                if not event.get(field):
                    raise ValueError(f"frozen event is missing {field}")
            if _digest(event.get("observation")) != event["observation_sha256"]:
                raise ValueError("frozen observation payload drift")
            if _digest(event.get("anchor_envelope")) != event["anchor_envelope_sha256"]:
                raise ValueError("frozen anchor payload drift")
            if _digest(event.get("physical_snapshot")) != event["physical_snapshot_sha256"]:
                raise ValueError("frozen physical payload drift")
    if manifest["format_version"] in (
        FROZEN_MANIFEST_FORMAT_V3,
        FROZEN_MANIFEST_FORMAT_V4,
    ) and require_all_scenarios:
        eligible = sum(
            1
            for scenario in scenarios
            for event in scenario.get("events", [])
            if event.get("coordination_eligible") is True
        )
        required = int(
            manifest["event_plan"]["minimum_coordination_eligible_events"]
        )
        if eligible < required:
            raise ValueError(
                "dense frozen manifest has too few coordination-eligible events"
            )
    if manifest.get("passed") is not True:
        raise ValueError("frozen manifest is not complete")


def merge_manifests(paths: list[Path], output_path: Path) -> dict:
    by_scenario = {}
    elapsed = 0.0
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_manifest(source, require_all_scenarios=False)
        elapsed += float(source.get("elapsed_sec", 0.0))
        for scenario in source["scenarios"]:
            name = scenario["scenario"]
            if name in by_scenario and by_scenario[name] != scenario:
                raise ValueError(f"conflicting frozen scenario: {name}")
            by_scenario[name] = scenario
    result = _new_manifest()
    result["scenarios"] = [by_scenario[name] for name in BALANCED_SCENARIOS]
    result["source_manifest_sha256"] = {
        str(path): _sha256_file(path) for path in paths
    }
    result["source_elapsed_sec_sum"] = elapsed
    result["passed"] = True
    validate_frozen_manifest(result, require_all_scenarios=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def merge_dense_manifests(paths: list[Path], output_path: Path) -> dict:
    by_scenario = {}
    elapsed = 0.0
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_manifest(source, require_all_scenarios=False)
        if source.get("format_version") != FROZEN_MANIFEST_FORMAT_V3:
            raise ValueError("dense merge requires dense frozen manifests")
        elapsed += float(source.get("elapsed_sec", 0.0))
        for scenario in source["scenarios"]:
            name = scenario["scenario"]
            if name in by_scenario and by_scenario[name] != scenario:
                raise ValueError(f"conflicting frozen scenario: {name}")
            by_scenario[name] = scenario
    result = _new_dense_manifest()
    result["scenarios"] = [by_scenario[name] for name in BALANCED_SCENARIOS]
    result["source_manifest_sha256"] = {
        str(path): _sha256_file(path) for path in paths
    }
    result["source_elapsed_sec_sum"] = elapsed
    result["passed"] = True
    validate_frozen_manifest(result, require_all_scenarios=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _assert_scenario_metadata_compatible(base: dict, extra: dict) -> None:
    for key in (
        "experiment_contract_version",
        "experiment_contract_sha256",
        "experiment_contract",
        "action_schema",
        "observation_schema",
        "control_interval_sec",
        "warmup_steps",
    ):
        if base.get(key) != extra.get(key):
            raise ValueError("dense supplement scenario metadata drift")


def merge_dense_extended_manifests(
    base_paths: list[Path],
    extra_paths: list[Path],
    output_path: Path,
) -> dict:
    by_scenario = {}
    elapsed = 0.0
    source_paths = list(base_paths) + list(extra_paths)
    for path in base_paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_manifest(source, require_all_scenarios=False)
        if source.get("format_version") != FROZEN_MANIFEST_FORMAT_V3:
            raise ValueError("dense extended merge requires dense base manifests")
        elapsed += float(source.get("elapsed_sec", 0.0))
        for scenario in source["scenarios"]:
            name = scenario["scenario"]
            if name in by_scenario and by_scenario[name] != scenario:
                raise ValueError(f"conflicting frozen scenario: {name}")
            by_scenario[name] = copy.deepcopy(scenario)
    missing = sorted(set(BALANCED_SCENARIOS) - set(by_scenario))
    if missing:
        raise ValueError(f"dense extended merge is missing base scenarios: {missing}")
    for path in extra_paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_manifest(source, require_all_scenarios=False)
        if source.get("format_version") != FROZEN_MANIFEST_FORMAT_V3_EXTRA:
            raise ValueError("dense extended merge requires supplement manifests")
        elapsed += float(source.get("elapsed_sec", 0.0))
        for extra in source["scenarios"]:
            name = extra["scenario"]
            if name not in by_scenario:
                raise ValueError(f"supplement scenario has no dense base: {name}")
            target = by_scenario[name]
            _assert_scenario_metadata_compatible(target, extra)
            existing_steps = {
                int(event["policy_step"]) for event in target.get("events", [])
            }
            for event in extra.get("events", []):
                step = int(event["policy_step"])
                if step in existing_steps:
                    raise ValueError(
                        f"dense supplement duplicates base event: {name}:{step}"
                    )
                target["events"].append(copy.deepcopy(event))
                existing_steps.add(step)
            target["events"] = sorted(
                target["events"], key=lambda item: int(item["policy_step"])
            )
    scenarios = [by_scenario[name] for name in BALANCED_SCENARIOS]
    result = _new_dense_extended_manifest(scenarios)
    result["scenarios"] = scenarios
    result["source_manifest_sha256"] = {
        str(path): _sha256_file(path) for path in source_paths
    }
    result["source_elapsed_sec_sum"] = elapsed
    result["passed"] = True
    validate_frozen_manifest(result, require_all_scenarios=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate_oracle_against_frozen(
    artifact: dict,
    frozen_scenario: dict,
    *,
    strata: set[str] | None = None,
    policy_steps: set[int] | None = None,
    eligible_only: bool = True,
) -> dict:
    expected = {
        int(event["policy_step"]): event
        for event in frozen_scenario["events"]
        if (strata is None or event["stratum"] in strata)
        and (policy_steps is None or int(event["policy_step"]) in policy_steps)
        and (not eligible_only or event["coordination_eligible"])
    }
    if not expected:
        raise ValueError("oracle validation selected no frozen events")
    pools = artifact.get("candidate_pools", [])
    checks = []
    for pool in pools:
        step = int(pool["policy_step"])
        event = expected.get(step)
        if event is None:
            raise ValueError(f"oracle artifact contains an unfrozen event: {step}")
        checks.append({
            "policy_step": step,
            "stratum": event["stratum"],
            "simulation_step_exact": int(pool["simulation_step"]) == int(event["simulation_step"]),
            "simulation_time_exact": float(pool["simulation_time_sec"]) == float(event["simulation_time_sec"]),
            "native_branch_exact": pool["native_anchor_branch"] == event["native_anchor_branch"],
            "anchor_fingerprint_exact": pool["anchor_fingerprint"] == event["anchor_fingerprint"],
            "pre_runtime_exact": pool["pre_runtime_sha256"] == event["pre_runtime_sha256"],
            "forecast_exact": pool["forecast_sha256"] == event["forecast_sha256"],
        })
    expected_steps = set(expected)
    actual_steps = {int(pool["policy_step"]) for pool in pools}
    coverage_exact = actual_steps == expected_steps
    passed = coverage_exact and all(
        all(value for key, value in row.items() if key not in ("policy_step", "stratum"))
        for row in checks
    )
    return {"coverage_exact": coverage_exact, "checks": checks, "passed": passed}


def _parse_policy_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps:
        raise ValueError("policy step selection is empty")
    return steps


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--scenario", required=True, choices=BALANCED_SCENARIOS)
    freeze.add_argument("--output", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    freeze_dense = subparsers.add_parser("freeze-dense")
    freeze_dense.add_argument("--scenario", required=True, choices=BALANCED_SCENARIOS)
    freeze_dense.add_argument("--output", required=True)
    freeze_dense.add_argument("--resume", action="store_true")
    merge_dense = subparsers.add_parser("merge-dense")
    merge_dense.add_argument("--inputs", nargs="+", required=True)
    merge_dense.add_argument("--output", required=True)
    freeze_dense_extra = subparsers.add_parser("freeze-dense-extra")
    freeze_dense_extra.add_argument("--scenario", required=True, choices=BALANCED_SCENARIOS)
    freeze_dense_extra.add_argument("--policy-steps", required=True)
    freeze_dense_extra.add_argument("--output", required=True)
    freeze_dense_extra.add_argument("--resume", action="store_true")
    merge_dense_extended = subparsers.add_parser("merge-dense-extended")
    merge_dense_extended.add_argument("--base-inputs", nargs="+", required=True)
    merge_dense_extended.add_argument("--extra-inputs", nargs="+", required=True)
    merge_dense_extended.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_scenario(args.scenario, Path(args.output))
    elif args.command == "merge":
        result = merge_manifests(
            [Path(path) for path in args.inputs], Path(args.output)
        )
    elif args.command == "freeze-dense":
        if args.resume:
            result = resume_dense_scenario(args.scenario, Path(args.output))
        else:
            result = freeze_dense_scenario(args.scenario, Path(args.output))
    elif args.command == "merge-dense":
        result = merge_dense_manifests(
            [Path(path) for path in args.inputs], Path(args.output)
        )
    elif args.command == "freeze-dense-extra":
        policy_steps = _parse_policy_steps(args.policy_steps)
        if args.resume:
            result = resume_dense_extra_scenario(
                args.scenario, policy_steps, Path(args.output)
            )
        else:
            result = freeze_dense_extra_scenario(
                args.scenario, policy_steps, Path(args.output)
            )
    else:
        result = merge_dense_extended_manifests(
            [Path(path) for path in args.base_inputs],
            [Path(path) for path in args.extra_inputs],
            Path(args.output),
        )
    print(json.dumps({
        "passed": result["passed"],
        "scenarios": [row["scenario"] for row in result["scenarios"]],
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
