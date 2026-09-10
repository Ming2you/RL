"""Verify exact native-PStack versus zero-residual anchor-context execution."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rl_leader.diagnose_phase0_parity import (
    _control_payload,
    _digest,
    _physical_snapshot,
    follower_behavior_payload,
    follower_behavior_fingerprint,
)
from rl_leader.env import RLLeaderEnv
from src.controllers.coordination import CoordinationPotentialAdapter


_POTENTIAL_FIELDS = (
    "signal_marginal_price",
    "offset_marginal_price",
    "metering_marginal_price",
    "vsl_marginal_price",
    "green_offset_cross_price",
    "vsl_meter_cross_price",
    "signal_quadratic_price",
    "offset_quadratic_price",
    "metering_quadratic_price",
    "vsl_quadratic_price",
    "signal_marginal_price_ref",
    "offset_marginal_price_ref",
    "green_offset_cross_ref",
    "metering_marginal_price_ref",
    "vsl_marginal_price_ref",
    "vsl_meter_cross_ref",
    "signal_marginal_price_trust_sec",
    "offset_marginal_price_trust_sec",
    "metering_marginal_price_trust_frac",
    "vsl_marginal_price_trust_kmh",
    "metering_release_certified",
    "joint_green_offset_enabled",
    "ramp_offset_enabled",
    "priced_vsl_segment_candidates_enabled",
)


def _potential_payload(follower) -> dict:
    return {
        name: copy.deepcopy(getattr(follower, name, None))
        for name in _POTENTIAL_FIELDS
    }


def _dynamic_follower_payload(follower) -> dict:
    payload = follower_behavior_payload(follower)
    for name in _POTENTIAL_FIELDS:
        payload.pop(name, None)
    return payload


def _budget_dual_diagnostics(control) -> dict:
    tokens = ("n_p", "n_uf", "lambda", "target", "provider", "bounded", "raw")
    return {
        key: value
        for key, value in control.diagnostics.items()
        if any(token in key.lower() for token in tokens)
    }


def _segment_model_differences(native_follower, candidate_follower) -> dict:
    native_models = getattr(native_follower, "_segment_agent_models", {})
    candidate_models = getattr(candidate_follower, "_segment_agent_models", {})
    result = {}
    for key in sorted(set(native_models) | set(candidate_models)):
        native_model = native_models.get(key)
        candidate_model = candidate_models.get(key)
        native_payload = getattr(native_model, "__dict__", {})
        candidate_payload = getattr(candidate_model, "__dict__", {})
        differing = [
            field
            for field in sorted(set(native_payload) | set(candidate_payload))
            if field != "cfg"
            and _digest(native_payload.get(field)) != _digest(candidate_payload.get(field))
        ]
        if differing:
            result[str(key)] = differing
    return result


class _ExactPotentialProbeAdapter:
    def __init__(self, payload: dict):
        self.base = CoordinationPotentialAdapter()
        self.payload = copy.deepcopy(payload)

    def clear(self, follower) -> None:
        self.base.clear(follower)

    def apply(self, action, follower) -> dict:
        metadata = self.base.apply(action, follower)
        for name, value in self.payload.items():
            setattr(follower, name, copy.deepcopy(value))
        return metadata


def run_anchor_context_parity(
    scenario: str,
    *,
    t_total: float = 14400.0,
    warmup: int = 5,
    force_native_potential: bool = False,
    output: str | Path | None = None,
) -> dict:
    source = RLLeaderEnv(
        scenario_name=scenario,
        T_total=t_total,
        warmup_nc_steps=warmup,
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    context = source.prepare_pstack_anchor_context()
    adapter_follower = copy.deepcopy(source.controller.nash_solver)
    CoordinationPotentialAdapter().apply(context.coordination, adapter_follower)
    native_potential = _potential_payload(context.optimizer_controller.nash_solver)
    adapter_potential = _potential_payload(adapter_follower)
    potential_field_equal = {
        name: _digest(native_potential[name]) == _digest(adapter_potential[name])
        for name in _POTENTIAL_FIELDS
    }
    native = copy.deepcopy(source)
    candidate = copy.deepcopy(source)
    if force_native_potential:
        candidate.controller.potential_adapter = _ExactPotentialProbeAdapter(
            native_potential
        )
    native_obs, native_reward, native_done, _, _ = (
        native.step_prepared_optimizer_anchor(copy.deepcopy(context))
    )
    candidate_obs, candidate_reward, candidate_done, _ = (
        candidate.step_anchored_candidate(
            np.zeros(candidate.action_dim, dtype=np.float32),
            context,
        )
    )
    observation_linf = float(np.max(np.abs(native_obs - candidate_obs)))
    native_follower = native.optimizer_controller.nash_solver
    candidate_follower = candidate.controller.nash_solver
    native_dynamic = _dynamic_follower_payload(native_follower)
    candidate_dynamic = _dynamic_follower_payload(candidate_follower)
    dynamic_field_equal = {
        name: _digest(native_dynamic.get(name)) == _digest(candidate_dynamic.get(name))
        for name in sorted(set(native_dynamic) | set(candidate_dynamic))
    }
    native_response = native.response_vector(native.previous)
    candidate_response = candidate.response_vector(candidate.previous)
    observation_differences = [
        {
            "name": name,
            "native": float(native_value),
            "candidate": float(candidate_value),
            "abs_error": float(abs(native_value - candidate_value)),
        }
        for name, native_value, candidate_value in zip(
            native.observation_schema.names,
            native_obs,
            candidate_obs,
        )
        if native_value != candidate_value
    ]
    observation_differences.sort(key=lambda row: row["abs_error"], reverse=True)
    checks = {
        "coordination_exact": (
            _digest(candidate.last_requested_coordination)
            == _digest(context.coordination)
        ),
        "control_exact": (
            _control_payload(native.previous)
            == _control_payload(candidate.previous)
        ),
        "physical_state_exact": (
            _physical_snapshot(native) == _physical_snapshot(candidate)
        ),
        "reward_exact": native_reward == candidate_reward,
        "done_exact": native_done == candidate_done,
        "observation_exact": observation_linf == 0.0,
        "follower_post_memory_exact": (
            _digest(native_dynamic) == _digest(candidate_dynamic)
        ),
        "zero_residual_deployed": bool(
            np.count_nonzero(candidate.last_deployed_residual) == 0
        ),
    }
    result = {
        "format_version": "anchor_context_zero_roundtrip_v2_follower_seed",
        "scenario": scenario,
        **source.experiment_contract.artifact_fields(),
        "anchor_context_contract": context.contract_version,
        "anchor_fingerprint": context.anchor_fingerprint,
        "anchor_selected_branch": context.coordination.selected_branch,
        "force_native_potential": bool(force_native_potential),
        "anchor_coordination_budget": {
            "N_P_star": float(context.coordination.N_P_star),
            "N_UF_star": float(context.coordination.N_UF_star),
            "raw_budget": list(context.coordination.raw_budget or ()),
            "native_budget_exact": bool(context.coordination.native_budget_exact),
        },
        "anchor_result_metadata": {
            key: value
            for key, value in context.result.metadata.items()
            if any(
                token in key.lower()
                for token in ("n_p", "n_uf", "lambda", "target", "selected")
            )
        },
        "observation_linf": observation_linf,
        "native_step_ttt": float(-native_reward),
        "candidate_step_ttt": float(-candidate_reward),
        "native_response": native_response.astype(float).tolist(),
        "candidate_response": candidate_response.astype(float).tolist(),
        "response_linf": float(np.max(np.abs(native_response - candidate_response))),
        "observation_differences": observation_differences,
        "dynamic_follower_field_equal": dynamic_field_equal,
        "dynamic_follower_differing_fields": [
            name for name, equal in dynamic_field_equal.items() if not equal
        ],
        "segment_agent_model_differing_fields": _segment_model_differences(
            native_follower,
            candidate_follower,
        ),
        "native_follower_fingerprint": follower_behavior_fingerprint(native_follower),
        "candidate_follower_fingerprint": follower_behavior_fingerprint(candidate_follower),
        "potential_field_equal": potential_field_equal,
        "native_potential": native_potential,
        "adapter_potential": adapter_potential,
        "native_control": _control_payload(native.previous),
        "candidate_control": _control_payload(candidate.previous),
        "native_budget_dual_diagnostics": _budget_dual_diagnostics(native.previous),
        "candidate_budget_dual_diagnostics": _budget_dual_diagnostics(
            candidate.previous
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{scenario} anchor-context parity failed: {failed}")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="sweet_170_w60,sweet_190_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--force-native-potential", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/rl_phase0_implementation_20260828/anchor_context_parity",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.out_dir)
    results = []
    for scenario in (
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ):
        result = run_anchor_context_parity(
            scenario,
            t_total=args.t_total,
            warmup=args.warmup,
            force_native_potential=args.force_native_potential,
            output=output_dir / f"{scenario}.json",
        )
        results.append(result)
        print(
            f"scenario={scenario} branch={result['anchor_selected_branch']} "
            f"ttt={result['native_step_ttt']:.6f} parity=pass",
            flush=True,
        )
    print(json.dumps({
        "passed": all(result["passed"] for result in results),
        "scenarios": [result["scenario"] for result in results],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
