"""Summarize lossy P-Stack action-schema round trips from reachability artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_leader.diagnose_pcent_reachability import (
    _response_scales_and_families,
    response_distance,
)
from rl_leader.env import RLLeaderEnv
from rl_leader.experiment_contract import ExperimentContract


def summarize_file(path: Path, exact_tolerance: float) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    trace_path = Path(source["source_trace"])
    meta = json.loads(trace_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    contract = ExperimentContract.from_artifact(
        meta["experiment_contract"], meta["experiment_contract_sha256"]
    )
    env = RLLeaderEnv(
        experiment_contract=contract,
        mask=str(source["mask"]),
        pstack_anchor=True,
    )
    scales, families = _response_scales_and_families(env)
    events = []
    for event in source["events"]:
        anchor = next(
            row for row in event["all_unique_candidates"]
            if "anchor" in row["aliases"]
        )
        frozen_actor = next(
            row for row in event["all_unique_candidates"]
            if "frozen_actor" in row["aliases"]
        )
        adapter_response = np.asarray(anchor["response"], dtype=float)
        frozen_response = np.asarray(frozen_actor["response"], dtype=float)
        pstack_response = np.asarray(event["pstack_response"], dtype=float)
        raw_linf = float(np.max(np.abs(adapter_response - pstack_response)))
        distance = response_distance(
            adapter_response, pstack_response, scales, families
        )
        events.append({
            "step": int(event["step"]),
            "simulation_time_sec": float(event["simulation_time_sec"]),
            "anchor_alias_count": len(anchor["aliases"]),
            "frozen_actor_same_as_adapter_anchor": bool(
                np.array_equal(frozen_response, adapter_response)
            ),
            "raw_response_linf": raw_linf,
            "normalized_distance": distance,
            "exact_roundtrip": bool(raw_linf <= exact_tolerance),
        })
    return {
        "source_reachability": str(path),
        "source_trace": str(trace_path),
        "scenario": str(source["scenario"]),
        "experiment_contract_sha256": str(source["experiment_contract_sha256"]),
        "event_count": len(events),
        "exact_count": sum(int(row["exact_roundtrip"]) for row in events),
        "frozen_actor_anchor_alias_count": sum(
            int(row["frozen_actor_same_as_adapter_anchor"]) for row in events
        ),
        "events": events,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reachability", nargs="+")
    parser.add_argument("--exact-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = [
        summarize_file(Path(value), float(args.exact_tolerance))
        for value in args.reachability
    ]
    result = {
        "format_version": "pstack_anchor_roundtrip_summary_v1",
        "exact_tolerance": float(args.exact_tolerance),
        "scenario_count": len(rows),
        "event_count": sum(row["event_count"] for row in rows),
        "exact_count": sum(row["exact_count"] for row in rows),
        "frozen_actor_anchor_alias_count": sum(
            row["frozen_actor_anchor_alias_count"] for row in rows
        ),
        "all_exact": all(row["exact_count"] == row["event_count"] for row in rows),
        "scenarios": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"anchor roundtrip exact={result['exact_count']}/{result['event_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
