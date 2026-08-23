"""Freeze the inputs and contracts used to diagnose the RL versus P-Stack gap."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl_leader.env import OPTIMIZER_ANCHOR_CONTRACT, RLLeaderEnv
from src.controllers.coordination import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    RL_RESPONSE_CONTRACT_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "sweet_155_w60",
    "sweet_170_w60",
    "sweet_170_incident_w60",
    "sweet_170_skew15_w60",
    "sweet_190_w60",
)


def _run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _collect_files(patterns: tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def _dataset_record(path: Path) -> dict:
    record = _file_record(path)
    with np.load(path, allow_pickle=False) as payload:
        record["arrays"] = {
            name: {"shape": list(payload[name].shape), "dtype": str(payload[name].dtype)}
            for name in payload.files
            if name != "manifest_json"
        }
        if "manifest_json" in payload.files:
            record["embedded_manifest"] = json.loads(str(payload["manifest_json"].item()))
    return record


def build_manifest() -> dict:
    env = RLLeaderEnv(scenario_name=SCENARIOS[0])
    datasets = _collect_files((
        "data/full_action_v3_response_fixed/*.npz",
        "data/targeted_24h_v1/*.npz",
    ))
    checkpoints = _collect_files(("checkpoints/actor_full_iql_targeted24h_s*.pt",))
    baselines = _collect_files(("results/five_cell_baselines/**/*",))
    current_results = _collect_files(("results/five_cell_targeted_v1/*",))
    source_contract = _collect_files((
        "rl_leader/env.py",
        "rl_leader/iql.py",
        "rl_leader/nets.py",
        "src/controllers/coordination.py",
        "src/controllers/rl_stackelberg.py",
        "src/controllers/f1_wu_faithful_follower.py",
        "src/controllers/stackelberg_mpc.py",
        "src/config/default.yaml",
        "src/config/scenarios.yaml",
        "work/run_claude_style_five_controller.py",
        "work/run_targeted_24h.ps1",
        "work/run_targeted_postprocess.ps1",
    ))
    return {
        "format_version": "pstack_gap_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(ROOT),
            "git_commit": _run_git("rev-parse", "HEAD"),
            "git_branch": _run_git("branch", "--show-current"),
            "git_status_porcelain": _run_git("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "evaluation_contract": {
            "scenarios": list(SCENARIOS),
            "simulation_total_sec": 14400.0,
            "warmup_sec": 900.0,
            "peak_sec": [900.0, 5220.0],
            "recovery_sec": [5220.0, 14400.0],
            "metric": "warmup-excluded total travel time in vehicle-hours",
            "required_seeds": [0, 1, 2],
        },
        "controller_contract": {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "response_contract_version": RL_RESPONSE_CONTRACT_VERSION,
            "optimizer_anchor_contract": OPTIMIZER_ANCHOR_CONTRACT,
            "action_schema": env.action_schema.metadata(),
            "observation_schema": env.observation_schema.metadata(),
        },
        "artifacts": {
            "datasets": [_dataset_record(path) for path in datasets],
            "checkpoints": [_file_record(path) for path in checkpoints],
            "baselines": [_file_record(path) for path in baselines],
            "current_results": [_file_record(path) for path in current_results],
            "source_contract": [_file_record(path) for path in source_contract],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/pstack_gap_diagnosis_v1/manifest.json",
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "datasets": len(manifest["artifacts"]["datasets"]),
        "checkpoints": len(manifest["artifacts"]["checkpoints"]),
        "baselines": len(manifest["artifacts"]["baselines"]),
        "current_results": len(manifest["artifacts"]["current_results"]),
    }, indent=2))


if __name__ == "__main__":
    main()
