"""Permutation and commit diagnostics for native PStack follower candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_leader.diagnose_phase0_parity import (
    _control_payload,
    _digest,
    follower_behavior_fingerprint,
)
from rl_leader.env import RLLeaderEnv
from src.controllers.leader import LeaderAction
from src.models.state import ControlAction


def _rank(evaluations) -> list[int]:
    return [
        int(item.index)
        for item in sorted(evaluations, key=lambda item: (item.objective, item.index))
    ]


def _evaluation_payload(evaluation) -> dict:
    follower = evaluation.follower_solver_snapshot
    return {
        "index": int(evaluation.index),
        "objective": float(evaluation.objective),
        "control_sha256": _digest(_control_payload(evaluation.nash.control)),
        "follower_sha256": follower_behavior_fingerprint(follower),
        "has_postsolve_coupling": follower._prev_coupling is not None,
        "controller_cfg_alias": follower.cfg is evaluation.follower_solver_snapshot.cfg,
    }


def _warm_fixture(scenario: str, uncontrolled_steps: int):
    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=max(14400.0, 180.0 * (uncontrolled_steps + 1)),
        warmup_nc_steps=0,
    )
    uncontrolled = ControlAction.uncontrolled(env.cfg)
    for _ in range(int(uncontrolled_steps)):
        env._advance(uncontrolled)
    controller = env._ensure_optimizer_controller()
    controller.cfg.mpc.horizon_steps = 1
    controller.cfg.mpc.leader_value_depth = 0
    controller.cfg.mpc.leader_rollout_early_stop = False
    controller.cfg.mpc.max_nash_iter = 1
    controller.cfg.freeway_follower.vsl_sequence_horizon_steps = 1
    controller.candidate_dedupe_enabled = False
    forecast = env.profile.horizon(env.step_idx * env.dt, 1)
    return env, controller, forecast


def run_candidate_order_diagnostic(
    *,
    scenario: str = "sweet_170_skew15_w60",
    uncontrolled_steps: int = 10,
    permutations: int = 20,
    seed: int = 20260827,
    output: str | Path | None = None,
) -> dict:
    env, controller, forecast = _warm_fixture(scenario, uncontrolled_steps)
    candidates = [
        LeaderAction(200.0, nuf)
        for nuf in (100.0, 700.0, 1300.0, 1600.0, 1900.0, 2500.0, 3100.0)
    ]
    live_before = follower_behavior_fingerprint(controller.nash_solver)
    rng = np.random.default_rng(seed)
    orders = [list(range(len(candidates)))]
    while len(orders) < max(1, int(permutations)):
        order = rng.permutation(len(candidates)).astype(int).tolist()
        if order not in orders:
            orders.append(order)

    reference = None
    rows = []
    for permutation_index, order in enumerate(orders):
        evaluations = controller._evaluate_candidate_set(
            candidates,
            order,
            env.sim.state.copy(),
            forecast,
            env.previous,
        )
        mapped = {
            int(item.index): _evaluation_payload(item) for item in evaluations
        }
        row = {
            "permutation": int(permutation_index),
            "order": order,
            "rank": _rank(evaluations),
            "evaluations": mapped,
            "live_unchanged": (
                follower_behavior_fingerprint(controller.nash_solver) == live_before
            ),
        }
        if reference is None:
            reference = row
        else:
            row["matches_reference"] = bool(
                row["rank"] == reference["rank"]
                and row["evaluations"] == reference["evaluations"]
                and row["live_unchanged"]
            )
            if not row["matches_reference"]:
                raise RuntimeError(
                    f"candidate order mismatch at permutation {permutation_index}"
                )
        rows.append(row)
        print(
            f"candidate permutation={permutation_index + 1}/{len(orders)} "
            f"rank={row['rank']}",
            flush=True,
        )

    controller.candidate_dedupe_enabled = True
    controller._nuf_solve_cache = {}
    dedupe_candidates = [
        LeaderAction(100.0, 1600.0),
        LeaderAction(300.0, 1600.0),
    ]
    dedupe_evaluations = controller._evaluate_candidate_set(
        dedupe_candidates,
        [0, 1],
        env.sim.state.copy(),
        forecast,
        env.previous,
    )
    dedupe_rows = [_evaluation_payload(item) for item in dedupe_evaluations]
    dedupe_passed = bool(
        all(item["has_postsolve_coupling"] for item in dedupe_rows)
        and dedupe_rows[0]["follower_sha256"] == dedupe_rows[1]["follower_sha256"]
    )
    if not dedupe_passed:
        raise RuntimeError("dedupe candidate did not carry the representative post-state")

    controller._pfo_incumbent_eval = None
    controller._regret_force_this_step = False
    selected, _ = controller._select_with_fallback_guard(dedupe_evaluations, [])
    aliases = {
        "controller_follower_cfg": controller.nash_solver.cfg is controller.cfg,
        "wu_cfg": controller.nash_solver._wu.cfg is controller.cfg,
        "specs": controller.nash_solver._specs is controller.nash_solver._wu._specs,
        "phase_movements": (
            controller.nash_solver._phase_movements
            is controller.nash_solver._wu._phase_movements
        ),
    }
    if not all(aliases.values()):
        raise RuntimeError(f"selected candidate broke aliases: {aliases}")

    result = {
        "format_version": "pstack_candidate_order_v1",
        "scenario": scenario,
        "simulation_time_sec": float(env.sim.state.time_sec),
        "permutations": rows,
        "dedupe": {
            "passed": dedupe_passed,
            "evaluations": dedupe_rows,
        },
        "selected_index": int(selected.index),
        "selected_aliases": aliases,
        "passed": True,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="sweet_170_skew15_w60")
    parser.add_argument("--uncontrolled-steps", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--out",
        default="results/rl_phase0_implementation_20260827/candidate_order.json",
    )
    args = parser.parse_args(argv)
    run_candidate_order_diagnostic(
        scenario=args.scenario,
        uncontrolled_steps=args.uncontrolled_steps,
        permutations=args.permutations,
        seed=args.seed,
        output=args.out,
    )
    print(f"saved candidate-order diagnostic -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
