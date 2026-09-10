"""Complete selective balanced H12 artifacts to exhaustive outcome coverage."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rl_leader.balanced_oracle_manifest import validate_frozen_manifest
from rl_leader.diagnose_candidate_ablation import (
    _horizon_label,
    _implementation_fingerprints,
)
from rl_leader.diagnose_pcent_guided_oracle import _require_rollout_coverage
from rl_leader.diagnose_reachable_candidate_attribution import (
    _rollout_price_candidate,
    _rollout_pstack,
)
from rl_leader.evaluate_balanced_horizons import (
    BALANCED_HORIZON_FORMAT,
    _replay_event,
)


EXHAUSTIVE_H12_FORMAT = "balanced_owner_block_exhaustive_h12_v1"


def _checkpoint_matches_label(checkpoint: dict, native: dict, label: dict) -> bool:
    return all((
        float(checkpoint["ttt"]) == float(label["candidate_ttt"]),
        float(native["ttt"]) == float(label["native_ttt"]),
        float(checkpoint["terminal_inventory"])
        == float(label["candidate_terminal_inventory"]),
        float(native["terminal_inventory"])
        == float(label["native_terminal_inventory"]),
        checkpoint["follower_memory_sha256"]
        == label["candidate_follower_memory_sha256"],
        native["follower_memory_sha256"] == label["native_follower_memory_sha256"],
        checkpoint["physical_state_sha256"]
        == label["candidate_physical_state_sha256"],
        native["physical_state_sha256"] == label["native_physical_state_sha256"],
    ))


def complete_h12(source_path: Path, manifest_path: Path, output_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format_version") != BALANCED_HORIZON_FORMAT:
        raise ValueError("unexpected selective H12 source format")
    if source.get("implementation_sha256") != _implementation_fingerprints():
        raise ValueError("selective H12 implementation drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_manifest(manifest, require_all_scenarios=True)
    frozen = next(
        row for row in manifest["scenarios"] if row["scenario"] == source["scenario"]
    )
    event = next(
        row for row in frozen["events"] if row["policy_step"] == source["policy_step"]
    )
    env, context = _replay_event(frozen, event)
    native = _rollout_pstack(env, context, rollout_steps=12, horizons=(3, 12))
    _require_rollout_coverage(native, (3, 12), label="exhaustive H12 native")
    outcomes = []
    for source_row in source["outcomes"]:
        row = {key: value for key, value in source_row.items()}
        existing = row.get("h12_label")
        if existing is not None:
            if not _checkpoint_matches_label(
                native["checkpoints"]["12"],
                native["checkpoints"]["12"],
                {
                    **existing,
                    "candidate_ttt": existing["native_ttt"],
                    "candidate_terminal_inventory": existing["native_terminal_inventory"],
                    "candidate_follower_memory_sha256": existing["native_follower_memory_sha256"],
                    "candidate_physical_state_sha256": existing["native_physical_state_sha256"],
                },
            ):
                raise ValueError("existing H12 native replay drift")
            row["h12_source"] = "reused_selective_v1"
            outcomes.append(row)
            continue
        rollout = _rollout_price_candidate(
            env,
            np.asarray(row["continuous_residual"], dtype=np.float32),
            context,
            rollout_steps=12,
            horizons=(3, 12),
        )
        _require_rollout_coverage(rollout, (3, 12), label=row["candidate_id"])
        if not _checkpoint_matches_label(
            rollout["checkpoints"]["3"], native["checkpoints"]["3"], row["h3_label"]
        ):
            raise ValueError(f"H12 rollout failed stored H3 replay: {row['candidate_id']}")
        row["h12_label"] = _horizon_label(
            rollout["checkpoints"]["12"], native["checkpoints"]["12"], horizon=12
        )
        row["h12_source"] = "new_exhaustive_v1"
        outcomes.append(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "format_version": EXHAUSTIVE_H12_FORMAT,
            "source_selective_artifact": str(source_path),
            "scenario": source["scenario"],
            "stratum": source["stratum"],
            "policy_step": source["policy_step"],
            "outcomes": outcomes,
            "passed": False,
        }, indent=2), encoding="utf-8")
    result = {
        "format_version": EXHAUSTIVE_H12_FORMAT,
        "source_selective_artifact": str(source_path),
        "source_manifest": str(manifest_path),
        "implementation_sha256": _implementation_fingerprints(),
        "scenario": source["scenario"],
        "stratum": source["stratum"],
        "policy_step": source["policy_step"],
        "anchor_fingerprint": source["anchor_fingerprint"],
        "outcomes": outcomes,
        "passed": len(outcomes) == len(source["outcomes"]),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = complete_h12(Path(args.source), Path(args.manifest), Path(args.output))
    print(json.dumps({
        "passed": result["passed"],
        "outcomes": len(result["outcomes"]),
        "positive": sum(row["h12_label"]["positive"] for row in result["outcomes"]),
        "elapsed_sec": time.monotonic() - started,
        "output": args.output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
