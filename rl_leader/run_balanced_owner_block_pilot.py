"""Run owner-block oracle probes against a frozen balanced-state manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from rl_leader.balanced_oracle_manifest import (
    BALANCED_SCENARIOS,
    _freeze_event,
    validate_frozen_manifest,
    validate_oracle_against_frozen,
)
from rl_leader.diagnose_candidate_ablation import (
    _implementation_fingerprints,
    run_scenario,
)
from rl_leader.diagnose_phase0_parity import _normalize
from rl_leader.oracle_candidate_ablation import AblationConfig
from rl_leader.env import RLLeaderEnv


PILOT_INDEX_FORMAT = "balanced_owner_block_pilot_index_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _selected_strata(manifest: dict, requested: tuple[str, ...]) -> set[str]:
    known = {
        str(event["stratum"])
        for scenario in manifest["scenarios"]
        for event in scenario.get("events", [])
    }
    if requested == ("all",):
        return known
    selected = set(requested)
    if not selected or not selected <= known:
        raise ValueError("unknown or empty balanced-pilot stratum selection")
    return selected


def _selected_scenarios(manifest: dict, requested: tuple[str, ...]) -> list[dict]:
    by_name = {row["scenario"]: row for row in manifest["scenarios"]}
    names = list(requested) if requested else list(BALANCED_SCENARIOS)
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"scenarios are not frozen: {unknown}")
    return [by_name[name] for name in names]


def _frozen_event_differences(actual: dict, expected: dict) -> list[str]:
    return sorted(
        key for key in set(actual) | set(expected)
        if _normalize(actual.get(key)) != _normalize(expected.get(key))
    )


def _preflight_frozen_scenario(
    frozen: dict,
    selected_strata: set[str],
    policy_steps: set[int] | None = None,
) -> dict:
    env = RLLeaderEnv(
        scenario_name=frozen["scenario"],
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    if env.experiment_contract.artifact_fields() != {
        key: frozen[key]
        for key in (
            "experiment_contract_version",
            "experiment_contract_sha256",
            "experiment_contract",
        )
    }:
        raise ValueError("preflight experiment contract drift")
    selected = {
        int(event["policy_step"]): event
        for event in frozen["events"]
        if event["stratum"] in selected_strata
        and (policy_steps is None or int(event["policy_step"]) in policy_steps)
    }
    max_event = max(selected)
    checks = []
    while int(env.step_idx - env.warmup) <= max_event:
        step = int(env.step_idx - env.warmup)
        if step in selected:
            expected = selected[step]
            context = env.prepare_pstack_anchor_context()
            actual = _freeze_event(
                env, expected["stratum"], step, context
            )
            differing = _frozen_event_differences(actual, expected)
            exact = not differing
            checks.append({
                "stratum": expected["stratum"],
                "policy_step": step,
                "exact": exact,
            })
            if not exact:
                raise ValueError(
                    f"preflight frozen state drift at {frozen['scenario']}:{step}: "
                    f"{differing}"
                )
            if step == max_event:
                break
            env.step_prepared_optimizer_anchor(context)
        else:
            env.step_optimizer_anchor(sync_follower_state=True)
    return {"checks": checks, "passed": all(row["exact"] for row in checks)}


def run_balanced_pilot(
    manifest_path: Path,
    output_dir: Path,
    *,
    scenarios: tuple[str, ...] = (),
    strata: tuple[str, ...] = ("ramp_up", "plateau", "recovery_boundary"),
    candidate_mode: str = "owner_block_v2_urban",
    run_h12: bool = False,
    policy_steps: tuple[int, ...] = (),
    replay_cache_payloads: dict[int, Path] | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_manifest(
        manifest,
        require_all_scenarios=True,
        allow_implementation_drift=True,
        allow_dependency_drift=True,
    )
    selected_strata = _selected_strata(manifest, strata)
    selected_policy_steps = set(policy_steps) if policy_steps else None
    if candidate_mode not in ("owner_block_v2_urban", "owner_block_v2_all"):
        raise ValueError("balanced pilot requires an owner-block v2 candidate mode")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "pilot_index.json"
    result = {
        "format_version": PILOT_INDEX_FORMAT,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "implementation_sha256": _implementation_fingerprints(),
        "candidate_mode": candidate_mode,
        "run_h12": bool(run_h12),
        "strata": sorted(selected_strata),
        "policy_steps": None if selected_policy_steps is None else sorted(
            selected_policy_steps
        ),
        "screening_horizon": 12 if run_h12 else 1,
        "selection_valid": bool(run_h12),
        "h1_interpretation": (
            None if run_h12 else
            "response diversity and H1 screening only; H12 winner is undefined"
        ),
        "scenarios": [],
        "passed": False,
    }
    index_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cfg = AblationConfig(radius=2.0 ** -0.5, master_seed=20260828)
    started = time.monotonic()
    for frozen in _selected_scenarios(manifest, scenarios):
        scenario = frozen["scenario"]
        preflight = _preflight_frozen_scenario(
            frozen, selected_strata, selected_policy_steps
        )
        selected_events = [
            event for event in frozen["events"]
            if event["stratum"] in selected_strata
            and (
                selected_policy_steps is None
                or int(event["policy_step"]) in selected_policy_steps
            )
        ]
        eligible_events = [
            event for event in selected_events if event["coordination_eligible"]
        ]
        event_steps = tuple(
            int(event["policy_step"])
            for event in eligible_events
        )
        if not event_steps:
            result["scenarios"].append({
                "scenario": scenario,
                "artifact": None,
                "included_events": 0,
                "excluded_events": len(selected_events),
                "excluded": [{
                    "stratum": event["stratum"],
                    "policy_step": event["policy_step"],
                    "native_anchor_branch": event["native_anchor_branch"],
                } for event in selected_events],
                "preflight": preflight,
                "passed": True,
            })
            continue
        artifact_path = output_dir / f"{scenario}.json"
        artifact = run_scenario(
            scenario,
            event_steps,
            cfg=cfg,
            run_h12=run_h12,
            output_path=artifact_path,
            candidate_mode=candidate_mode,
            replay_cache_payloads=replay_cache_payloads,
        )
        frozen_check = validate_oracle_against_frozen(
            artifact,
            frozen,
            strata=selected_strata,
            policy_steps=selected_policy_steps,
            eligible_only=True,
        )
        if not frozen_check["passed"]:
            raise ValueError(f"oracle artifact drifted from frozen state: {scenario}")
        result["scenarios"].append({
            "scenario": scenario,
            "artifact": str(artifact_path),
            "artifact_sha256": _sha256_file(artifact_path),
            "candidate_pools": len(artifact["candidate_pools"]),
            "included_events": len(eligible_events),
            "excluded_events": len(selected_events) - len(eligible_events),
            "excluded": [{
                "stratum": event["stratum"],
                "policy_step": event["policy_step"],
                "native_anchor_branch": event["native_anchor_branch"],
            } for event in selected_events if not event["coordination_eligible"]],
            "unique_physical_residuals": sum(
                int(pool["unique_physical_residual_count"])
                for pool in artifact["candidate_pools"]
            ),
            "unique_realized_outcomes": sum(
                len({
                    row["response_memory_outcome_sha256"]
                    for row in pool["rows"]
                    if row.get("execution_branch") == "coordination"
                })
                for pool in artifact["candidate_pools"]
            ),
            "frozen_state_check": frozen_check,
            "preflight": preflight,
            "passed": True,
        })
        result["elapsed_sec"] = float(time.monotonic() - started)
        index_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["elapsed_sec"] = float(time.monotonic() - started)
    result["passed"] = True
    index_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", default="")
    parser.add_argument(
        "--strata", default="ramp_up,plateau,recovery_boundary"
    )
    parser.add_argument("--policy-steps", default="")
    parser.add_argument(
        "--candidate-mode",
        choices=("owner_block_v2_urban", "owner_block_v2_all"),
        default="owner_block_v2_urban",
    )
    parser.add_argument("--run-h12", action="store_true")
    args = parser.parse_args(argv)
    result = run_balanced_pilot(
        Path(args.manifest),
        Path(args.output_dir),
        scenarios=_parse_csv(args.scenarios),
        strata=_parse_csv(args.strata),
        candidate_mode=args.candidate_mode,
        run_h12=bool(args.run_h12),
        policy_steps=_parse_int_csv(args.policy_steps),
    )
    print(json.dumps({
        "passed": result["passed"],
        "scenarios": len(result["scenarios"]),
        "elapsed_sec": result["elapsed_sec"],
        "output": str(Path(args.output_dir) / "pilot_index.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
