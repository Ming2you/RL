"""Extend positive balanced-pilot H12 outcomes to the simulation horizon."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
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
from rl_leader.evaluate_balanced_horizons import (
    BALANCED_HORIZON_FORMAT,
    _replay_event,
)
from rl_leader.generate_long_horizon_labels import counterfactual_verdict
from rl_leader.oracle_label_contract import validate_oracle_label_artifact


BALANCED_REMAINING_FORMAT = "balanced_positive_remaining_horizon_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def exact_balanced_h12_replay(
    candidate: dict,
    native: dict,
    expected: dict,
) -> dict:
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
        "candidate_validity_exact": (
            bool(candidate_checkpoint["validity_gate_pass"])
            == bool(expected["validity_gate_pass"])
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def run_balanced_remaining_horizon(source_path: Path, output_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format_version") != BALANCED_HORIZON_FORMAT:
        raise ValueError("source is not a balanced H3/H12 artifact")
    if source.get("passed") is not True:
        raise ValueError("source balanced artifact did not pass")
    current_implementation = _implementation_fingerprints()
    if source.get("implementation_sha256") != current_implementation:
        raise ValueError("source balanced artifact implementation drift")

    h1_path = _recorded_path(source["source_h1_artifact"])
    manifest_path = _recorded_path(source["source_manifest"])
    if _sha256_file(h1_path) != source["source_h1_sha256"]:
        raise ValueError("source H1 artifact SHA mismatch")
    if _sha256_file(manifest_path) != source["source_manifest_sha256"]:
        raise ValueError("source manifest SHA mismatch")
    h1 = json.loads(h1_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_oracle_label_artifact(h1, require_decision_horizon=False)
    validate_frozen_manifest(manifest, require_all_scenarios=True)
    if len(h1["candidate_pools"]) != 1:
        raise ValueError("balanced remaining-horizon gate requires one H1 pool")
    pool = h1["candidate_pools"][0]
    if (
        pool["scenario"] != source["scenario"]
        or int(pool["policy_step"]) != int(source["policy_step"])
        or pool["anchor_fingerprint"] != source["anchor_fingerprint"]
    ):
        raise ValueError("balanced sidecar does not match its H1 source pool")

    positives = [
        row for row in source["outcomes"]
        if row.get("h12_label") is not None
        and row["h12_label"].get("positive") is True
    ]
    if not positives:
        raise ValueError("source balanced artifact contains no positive H12 outcome")

    frozen = next(
        row for row in manifest["scenarios"] if row["scenario"] == source["scenario"]
    )
    event = next(
        row for row in frozen["events"]
        if int(row["policy_step"]) == int(source["policy_step"])
    )
    if event["stratum"] != source["stratum"] or not event["coordination_eligible"]:
        raise ValueError("balanced sidecar frozen event mismatch")

    result = {
        "format_version": BALANCED_REMAINING_FORMAT,
        "source_artifact": str(source_path),
        "source_artifact_sha256": _sha256_file(source_path),
        "source_h1_artifact": str(h1_path),
        "source_manifest": str(manifest_path),
        "source_implementation_sha256": current_implementation,
        "implementation_sha256": {
            "rl_leader/diagnose_balanced_remaining_horizon.py": _sha256_file(
                ROOT / "rl_leader/diagnose_balanced_remaining_horizon.py"
            ),
            **current_implementation,
        },
        "scenario": source["scenario"],
        "stratum": source["stratum"],
        "policy_step": int(source["policy_step"]),
        "anchor_fingerprint": source["anchor_fingerprint"],
        "oracle_semantics": (
            "one balanced owner-block price intervention followed by native P-Stack "
            "to simulation end"
        ),
        "outcomes": [],
        "passed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    started = time.monotonic()
    env, context = _replay_event(frozen, event)
    remaining = int(env.n_steps - env.step_idx)
    horizons = tuple(sorted(set(LABEL_HORIZONS + (remaining,))))
    native = _rollout_pstack(env, context, rollout_steps=remaining, horizons=horizons)
    _require_rollout_coverage(native, horizons, label="balanced remaining P-Stack")
    for row in positives:
        candidate = _rollout_price_candidate(
            env,
            np.asarray(row["continuous_residual"], dtype=np.float32),
            context,
            rollout_steps=remaining,
            horizons=horizons,
        )
        _require_rollout_coverage(
            candidate, horizons, label=f"balanced remaining {row['candidate_id']}"
        )
        replay = exact_balanced_h12_replay(candidate, native, row["h12_label"])
        if not replay["passed"]:
            raise OracleGateError(
                "remaining-horizon rollout did not replay balanced H12 label",
                {"candidate_id": row["candidate_id"], **replay},
            )
        terminal = counterfactual_verdict(
            candidate_ttt=float(candidate["ttt"]),
            pstack_ttt=float(native["ttt"]),
            candidate_terminal_inventory=float(candidate["terminal_inventory"]),
            pstack_terminal_inventory=float(native["terminal_inventory"]),
        )
        result["outcomes"].append({
            "candidate_id": row["candidate_id"],
            "representative_candidate_id": row["representative_candidate_id"],
            "candidate_aliases": row["candidate_aliases"],
            "continuous_residual": row["continuous_residual"],
            "rollout_steps": remaining,
            "final_simulation_time_sec": candidate["final_simulation_time_sec"],
            "candidate_ttt": candidate["ttt"],
            "native_ttt": native["ttt"],
            "candidate_terminal_inventory": candidate["terminal_inventory"],
            "native_terminal_inventory": native["terminal_inventory"],
            "h12_replay": replay,
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
    result = run_balanced_remaining_horizon(Path(args.source), Path(args.output))
    print(json.dumps({
        "passed": result["passed"],
        "scenario": result["scenario"],
        "stratum": result["stratum"],
        "remaining_steps": result["remaining_steps"],
        "outcomes": len(result["outcomes"]),
        "elapsed_sec": result["elapsed_sec"],
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
