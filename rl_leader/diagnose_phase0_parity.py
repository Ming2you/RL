"""Exact direct-native versus forced-anchor Phase 0 parity gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

from rl_leader.env import RLLeaderEnv


def _normalize(value):
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "diagnostics"
        }
    if isinstance(value, (list, tuple, deque)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize(item) for item in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.__dict__.items())
            if key not in {
                "cfg", "_specs", "_phase_movements", "_local_models",
                "_local_freeway_models", "_segment_agent_models",
            }
        }
    return repr(value)


def _digest(value) -> str:
    canonical = json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _control_payload(control) -> dict:
    payload = asdict(control)
    payload.pop("diagnostics", None)
    return payload


def _anchor_info(info: dict) -> dict:
    return {
        key: value
        for key, value in info.items()
        if (
            key.startswith("leader_rl_pstack_anchor_")
            or key.startswith("leader_pfo_")
            or key.startswith("sup_")
        )
    }


def _physical_snapshot(env: RLLeaderEnv) -> dict:
    return {
        "step_idx": int(env.step_idx),
        "state": asdict(env.sim.state),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "urban_ttt": float(env.sim.urban_ttt),
        "previous_control": _control_payload(env.previous),
    }


def controller_behavior_fingerprint(controller) -> str:
    excluded = {
        "cfg", "leader", "nash_solver", "last_decision",
        "_pfo_incumbent_eval", "_pfo_incumbent_center", "_link_share_ctx",
        "_candidate_common_solver", "_nuf_solve_cache", "_dedupe_hits",
        "_signal_price_meta", "_leader_process_pool",
    }
    payload = {
        key: value
        for key, value in controller.__dict__.items()
        if key not in excluded
    }
    payload["cfg"] = controller.cfg.to_dict()
    payload["follower_runtime"] = follower_behavior_payload(
        controller.nash_solver
    )
    return _digest(payload)


def follower_behavior_payload(follower) -> dict:
    excluded = {
        "cfg", "_wu", "_specs", "_phase_movements", "_local_models",
        "_local_freeway_models", "_segment_agent_models", "last_candidate_trace",
    }
    payload = {
        key: value
        for key, value in follower.__dict__.items()
        if key not in excluded
    }
    segment_models = getattr(follower, "_segment_agent_models", {})
    payload["segment_agent_model_keys"] = sorted(segment_models)
    wu = follower._wu
    wu_excluded = {
        "cfg", "_specs", "_phase_movements", "_local_models",
        "_local_freeway_models",
    }
    payload["wu_runtime"] = {
        key: value
        for key, value in wu.__dict__.items()
        if key not in wu_excluded
    }
    return payload


def follower_behavior_fingerprint(follower) -> str:
    return _digest(follower_behavior_payload(follower))


def _alias_invariants(env: RLLeaderEnv) -> dict[str, bool]:
    optimizer = env._ensure_optimizer_controller()
    return {
        "rl_controller_cfg": env.controller.cfg is env.cfg,
        "rl_leader_cfg": env.controller.leader.cfg is env.cfg,
        "rl_follower_cfg": env.controller.nash_solver.cfg is env.cfg,
        "rl_wu_cfg": env.controller.nash_solver._wu.cfg is env.cfg,
        "optimizer_controller_cfg": optimizer.cfg is env.optimizer_cfg,
        "optimizer_leader_cfg": optimizer.leader.cfg is env.optimizer_cfg,
        "optimizer_follower_cfg": optimizer.nash_solver.cfg is env.optimizer_cfg,
        "optimizer_wu_cfg": optimizer.nash_solver._wu.cfg is env.optimizer_cfg,
        "rl_specs_alias": (
            env.controller.nash_solver._specs
            is env.controller.nash_solver._wu._specs
        ),
        "optimizer_specs_alias": (
            optimizer.nash_solver._specs is optimizer.nash_solver._wu._specs
        ),
    }


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_parity_scenario(
    scenario: str,
    *,
    steps: int = 75,
    t_total: float = 14400.0,
    warmup: int = 5,
    output: str | Path | None = None,
) -> dict:
    kwargs = {
        "scenario_name": scenario,
        "T_total": float(t_total),
        "warmup_nc_steps": int(warmup),
        "pstack_anchor": True,
        "action_parameterization": "pstack_residual",
    }
    direct = RLLeaderEnv(**kwargs)
    forced = RLLeaderEnv(
        pstack_anchor=True,
        action_parameterization="pstack_residual",
        experiment_contract=direct.experiment_contract,
    )
    result = {
        "format_version": "phase0_direct_forced_parity_v1",
        "scenario": scenario,
        **direct.experiment_contract.artifact_fields(),
        "requested_steps": int(steps),
        "passed": True,
        "failure": None,
        "rows": [],
    }
    path = Path(output) if output is not None else None
    if direct.experiment_contract_fingerprint != forced.experiment_contract_fingerprint:
        raise RuntimeError("direct and forced arms resolved different contracts")

    try:
        for local_step in range(int(steps)):
            direct_forecast = direct._optimizer_forecast()
            forced_forecast = forced._optimizer_forecast()
            forecast_equal = _digest(direct_forecast) == _digest(forced_forecast)
            start = time.monotonic()
            direct_obs, direct_reward, direct_done, direct_info, direct_raw = (
                direct.step_optimizer_anchor(sync_follower_state=True)
            )
            direct_elapsed = time.monotonic() - start
            start = time.monotonic()
            forced_obs, forced_reward, forced_done, forced_info = forced.step(
                np.zeros(forced.action_dim, dtype=np.float32)
            )
            forced_elapsed = time.monotonic() - start

            control_equal = (
                _control_payload(direct.previous)
                == _control_payload(forced.previous)
            )
            physical_equal = _physical_snapshot(direct) == _physical_snapshot(forced)
            optimizer_equal = (
                controller_behavior_fingerprint(direct.optimizer_controller)
                == controller_behavior_fingerprint(forced.optimizer_controller)
            )
            optimizer_follower_equal = (
                follower_behavior_fingerprint(direct.optimizer_controller.nash_solver)
                == follower_behavior_fingerprint(forced.optimizer_controller.nash_solver)
            )
            rl_equal = (
                controller_behavior_fingerprint(direct.controller)
                == controller_behavior_fingerprint(forced.controller)
            )
            rl_follower_equal = (
                follower_behavior_fingerprint(direct.controller.nash_solver)
                == follower_behavior_fingerprint(forced.controller.nash_solver)
            )
            raw_equal = np.array_equal(
                np.asarray(direct_raw), np.asarray(forced.last_applied_raw_action)
            )
            raw_linf = float(np.max(np.abs(
                np.asarray(direct_raw) - np.asarray(forced.last_applied_raw_action)
            )))
            observation_error = float(np.max(np.abs(direct_obs - forced_obs)))
            coordination_equal = (
                _digest(forced.last_requested_coordination)
                == _digest(forced.last_optimizer_anchor_coordination)
            )
            aliases = {
                "direct": _alias_invariants(direct),
                "forced": _alias_invariants(forced),
            }
            aliases_ok = all(
                value for arm in aliases.values() for value in arm.values()
            )
            checks = {
                "forecast_equal": forecast_equal,
                "pstack_selected": bool(
                    forced_info.get("leader_rl_pstack_anchor_pick_pstack", 0.0) > 0.5
                ),
                "identity_not_forced": bool(
                    forced_info.get(
                        "leader_rl_pstack_anchor_identity_forced", 0.0
                    ) == 0.0
                ),
                "requested_coordination_exact": coordination_equal,
                "control_equal": control_equal,
                "physical_state_equal": physical_equal,
                "reward_equal": direct_reward == forced_reward,
                "done_equal": direct_done == forced_done,
                "observation_equal": observation_error == 0.0,
                "selected_raw_equal": raw_equal,
                "optimizer_controller_equal": optimizer_equal,
                "optimizer_follower_equal": optimizer_follower_equal,
                "rl_controller_equal": rl_equal,
                "rl_follower_equal": rl_follower_equal,
                "aliases_ok": aliases_ok,
                "log_count_equal": len(direct.sim.logs) == len(forced.sim.logs),
            }
            row = {
                "policy_step": int(local_step),
                "simulation_step": int(direct.step_idx - 1),
                "time_sec": float(direct.sim.state.time_sec),
                "step_ttt": float(-direct_reward),
                "cumulative_ttt": float(direct.sim.total_ttt),
                "observation_linf": observation_error,
                "selected_raw_linf": raw_linf,
                "direct_selected_raw": np.asarray(direct_raw, dtype=float).tolist(),
                "forced_selected_raw": np.asarray(
                    forced.last_applied_raw_action, dtype=float
                ).tolist(),
                "direct_elapsed_sec": float(direct_elapsed),
                "forced_elapsed_sec": float(forced_elapsed),
                "direct_control_sha256": _digest(_control_payload(direct.previous)),
                "forced_control_sha256": _digest(_control_payload(forced.previous)),
                "direct_physical_sha256": _digest(_physical_snapshot(direct)),
                "forced_physical_sha256": _digest(_physical_snapshot(forced)),
                "requested_coordination_sha256": _digest(
                    forced.last_requested_coordination
                ),
                "native_anchor_coordination_sha256": _digest(
                    forced.last_optimizer_anchor_coordination
                ),
                "aliases": aliases,
                "checks": checks,
            }
            if not all(checks.values()):
                row.update({
                    "direct_anchor_info": _anchor_info(direct_info),
                    "forced_anchor_info": _anchor_info(forced_info),
                    "direct_teacher_metadata": dict(direct.last_optimizer_anchor_metadata),
                    "forced_teacher_metadata": dict(forced.last_optimizer_anchor_metadata),
                    "direct_control": _control_payload(direct.previous),
                    "forced_control": _control_payload(forced.previous),
                })
            result["rows"].append(row)
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(
                    f"{scenario} policy step {local_step} parity failed: {failed}"
                )
            print(
                f"scenario={scenario} step={local_step + 1}/{steps} "
                f"ttt={direct.sim.total_ttt:.6f} "
                f"direct_sec={direct_elapsed:.2f} forced_sec={forced_elapsed:.2f}",
                flush=True,
            )
            if path is not None:
                _write_result(path, result)
            if direct_done:
                break
    except Exception as exc:
        result["passed"] = False
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if path is not None:
            _write_result(path, result)
        raise

    result["completed_steps"] = len(result["rows"])
    if path is not None:
        _write_result(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="sweet_170_w60,sweet_190_w60")
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--out-dir", default="results/rl_phase0_implementation_20260827/parity"
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.out_dir)
    summaries = []
    for scenario in (
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ):
        output = output_dir / f"{scenario}.json"
        summaries.append(run_parity_scenario(
            scenario,
            steps=args.steps,
            t_total=args.t_total,
            warmup=args.warmup,
            output=output,
        ))
    print(json.dumps({
        "passed": all(item["passed"] for item in summaries),
        "scenarios": [item["scenario"] for item in summaries],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
