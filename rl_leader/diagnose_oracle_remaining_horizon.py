"""Extend positive oracle outcomes from H12 to the simulation horizon."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.diagnose_candidate_ablation import (
    LABEL_HORIZONS,
    ROOT,
    _implementation_fingerprints,
)
from rl_leader.diagnose_pcent_guided_oracle import (
    OracleGateError,
    _require_rollout_coverage,
)
from rl_leader.diagnose_reachable_candidate_attribution import (
    _rollout_price_candidate,
    _rollout_pstack,
)
from rl_leader.env import RLLeaderEnv
from rl_leader.generate_long_horizon_labels import counterfactual_verdict
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


REMAINING_HORIZON_CONTRACT = "oracle_positive_remaining_horizon_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_minimal_execution_alias(pool: dict, representative: dict) -> dict:
    """Choose the sparsest, smallest, target-independent alias of one outcome."""
    rows = {
        str(row["candidate_id"]): row
        for row in pool["rows"]
        if row.get("execution_branch") == "coordination"
    }
    aliases = [rows[candidate_id] for candidate_id in representative["candidate_aliases"]]
    outcome = (
        representative["response_memory_outcome_sha256"],
        representative["post_physical_sha256"],
    )
    if any((
        row["response_memory_outcome_sha256"], row["post_physical_sha256"]
    ) != outcome for row in aliases):
        raise ValueError("candidate aliases do not share one realized outcome")

    def priority(row: dict) -> tuple:
        residual = np.asarray(row["continuous_residual"], dtype=float)
        return (
            bool(row.get("uses_pcent_target", False)),
            int(np.count_nonzero(residual)),
            float(np.linalg.norm(residual)),
            str(row["candidate_id"]),
        )

    return min(aliases, key=priority)


def _exact_h12_replay(
    candidate: dict,
    native: dict,
    representative: dict,
) -> dict:
    expected = representative["horizon_labels"]["12"]
    candidate_checkpoint = candidate["checkpoints"]["12"]
    native_checkpoint = native["checkpoints"]["12"]
    checks = {
        "candidate_ttt_exact": (
            float(candidate_checkpoint["ttt"]) == float(expected["candidate_ttt"])
        ),
        "native_ttt_exact": (
            float(native_checkpoint["ttt"]) == float(expected["native_ttt"])
        ),
        "candidate_inventory_exact": (
            float(candidate_checkpoint["terminal_inventory"])
            == float(expected["candidate_terminal_inventory"])
        ),
        "native_inventory_exact": (
            float(native_checkpoint["terminal_inventory"])
            == float(expected["native_terminal_inventory"])
        ),
        "candidate_follower_exact": (
            candidate_checkpoint["follower_memory_sha256"]
            == expected["candidate_follower_memory_sha256"]
        ),
        "native_follower_exact": (
            native_checkpoint["follower_memory_sha256"]
            == expected["native_follower_memory_sha256"]
        ),
        "candidate_physical_exact": (
            candidate_checkpoint["physical_state_sha256"]
            == expected["candidate_physical_state_sha256"]
        ),
        "native_physical_exact": (
            native_checkpoint["physical_state_sha256"]
            == expected["native_physical_state_sha256"]
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def _replay_source(pool: dict) -> tuple[RLLeaderEnv, object]:
    env = RLLeaderEnv(
        scenario_name=str(pool["scenario"]),
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    env.reset()
    policy_step = int(pool["policy_step"])
    while int(env.step_idx - env.warmup) < policy_step:
        env.step_optimizer_anchor(sync_follower_state=True)
    context = env.prepare_pstack_anchor_context()
    if context.anchor_fingerprint != pool["anchor_fingerprint"]:
        raise OracleGateError(
            "remaining-horizon replay reached a different anchor",
            {
                "expected": pool["anchor_fingerprint"],
                "actual": context.anchor_fingerprint,
            },
        )
    return env, context


def run_remaining_horizon(source_path: Path, output_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(source, require_decision_horizon=True)
    current_implementation = _implementation_fingerprints()
    if source["implementation_sha256"] != current_implementation:
        raise ValueError("source oracle artifact does not match the current implementation")
    if len(source["candidate_pools"]) != 1:
        raise ValueError("remaining-horizon v1 requires exactly one candidate pool")
    pool = source["candidate_pools"][0]
    positives = [
        row for row in pool["rows"]
        if row.get("execution_branch") == "coordination"
        and row["horizon_labels"]["12"].get("positive") is True
    ]
    if not positives:
        raise ValueError("source oracle artifact contains no positive H12 outcome")

    result = {
        "format_version": "oracle_remaining_horizon_results_v1",
        "contract": REMAINING_HORIZON_CONTRACT,
        "source_artifact": str(source_path),
        "source_artifact_sha256": _sha256_file(source_path),
        "source_implementation_sha256": current_implementation,
        "implementation_sha256": {
            "rl_leader/diagnose_oracle_remaining_horizon.py": _sha256_file(
                ROOT / "rl_leader/diagnose_oracle_remaining_horizon.py"
            ),
            **current_implementation,
        },
        "scenario": pool["scenario"],
        "policy_step": int(pool["policy_step"]),
        "anchor_fingerprint": pool["anchor_fingerprint"],
        "oracle_semantics": (
            "one price intervention followed by native P-Stack to simulation end"
        ),
        "outcomes": [],
        "passed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    started = time.monotonic()
    env, context = _replay_source(pool)
    remaining = int(env.n_steps - env.step_idx)
    horizons = tuple(sorted(set(LABEL_HORIZONS + (remaining,))))
    native = _rollout_pstack(
        env, context, rollout_steps=remaining, horizons=horizons
    )
    _require_rollout_coverage(native, horizons, label="remaining-horizon P-Stack")
    for representative in positives:
        execution = select_minimal_execution_alias(pool, representative)
        candidate = _rollout_price_candidate(
            env,
            np.asarray(execution["continuous_residual"], dtype=np.float32),
            context,
            rollout_steps=remaining,
            horizons=horizons,
        )
        _require_rollout_coverage(
            candidate, horizons, label=f"remaining-horizon {execution['candidate_id']}"
        )
        h12_replay = _exact_h12_replay(candidate, native, representative)
        if not h12_replay["passed"]:
            raise OracleGateError(
                "remaining-horizon rollout did not replay the source H12 label",
                {"candidate_id": execution["candidate_id"], **h12_replay},
            )
        terminal = counterfactual_verdict(
            candidate_ttt=float(candidate["ttt"]),
            pstack_ttt=float(native["ttt"]),
            candidate_terminal_inventory=float(candidate["terminal_inventory"]),
            pstack_terminal_inventory=float(native["terminal_inventory"]),
        )
        result["outcomes"].append({
            "representative_candidate_id": representative["candidate_id"],
            "execution_candidate_id": execution["candidate_id"],
            "execution_residual": execution["continuous_residual"],
            "execution_residual_l2": execution["residual_l2"],
            "candidate_aliases": representative["candidate_aliases"],
            "rollout_steps": remaining,
            "final_simulation_time_sec": candidate["final_simulation_time_sec"],
            "candidate_ttt": candidate["ttt"],
            "native_ttt": native["ttt"],
            "candidate_terminal_inventory": candidate["terminal_inventory"],
            "native_terminal_inventory": native["terminal_inventory"],
            "h12_replay": h12_replay,
            **terminal,
        })
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["remaining_steps"] = remaining
    result["elapsed_sec"] = float(time.monotonic() - started)
    result["passed"] = True
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_remaining_horizon(Path(args.source), Path(args.output))
    print(json.dumps({
        "passed": result["passed"],
        "scenario": result["scenario"],
        "remaining_steps": result["remaining_steps"],
        "outcomes": len(result["outcomes"]),
        "elapsed_sec": result["elapsed_sec"],
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
