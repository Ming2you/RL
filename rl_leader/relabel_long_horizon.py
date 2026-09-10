"""Attach exact long-horizon labels to P-Stack residual datasets."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from rl_leader.data_contract import (
    LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT,
    validate_pstack_residual_rows,
)
from rl_leader.experiment_contract import verify_contract_map


def build_label_arrays(
    dataset,
    labels: list[dict],
    episode_contracts: dict[int, str] | None = None,
) -> dict[str, np.ndarray]:
    count = int(dataset["obs"].shape[0])
    valid = np.zeros(count, dtype=np.float32)
    positive = np.zeros(count, dtype=np.float32)
    gain = np.full(count, np.nan, dtype=np.float32)
    required_gain = np.full(count, np.nan, dtype=np.float32)
    inventory_delta = np.full(count, np.nan, dtype=np.float32)
    rollout_steps = np.zeros(count, dtype=np.int32)
    modes = np.asarray(dataset["behavior_mode"]).astype(str)
    picked = np.asarray(dataset["pstack_anchor_pick_rl"], dtype=float) > 0.5
    for label in labels:
        index = int(label["source_row_index"])
        if not 0 <= index < count:
            raise ValueError(f"long-horizon row index is out of range: {index}")
        if valid[index] > 0.5:
            raise ValueError(f"duplicate long-horizon label for row {index}")
        if modes[index] != "optimizer_local" or not picked[index]:
            raise ValueError(f"row {index} is not a gate-accepted optimizer_local action")
        if int(dataset["episode"][index]) != int(label["episode"]):
            raise ValueError(f"episode mismatch for long-horizon row {index}")
        if int(dataset["step"][index]) != int(label["step"]):
            raise ValueError(f"step mismatch for long-horizon row {index}")
        if episode_contracts is not None:
            expected = episode_contracts[int(dataset["episode"][index])]
            actual = str(label.get("experiment_contract_sha256", ""))
            if actual != expected:
                raise ValueError(
                    f"experiment contract mismatch for long-horizon row {index}: "
                    f"expected {expected}, got {actual}"
                )
        valid[index] = 1.0
        positive[index] = float(bool(label["long_horizon_positive"]))
        gain[index] = float(label["ttt_gain"])
        required_gain[index] = float(label["required_gain"])
        inventory_delta[index] = float(label["terminal_inventory_delta"])
        rollout_steps[index] = int(label["rollout_steps"])
    return {
        "long_horizon_label_valid": valid,
        "long_horizon_positive": positive,
        "long_horizon_ttt_gain": gain,
        "long_horizon_required_gain": required_gain,
        "long_horizon_terminal_inventory_delta": inventory_delta,
        "long_horizon_rollout_steps": rollout_steps,
    }


def _manifest(dataset) -> dict:
    value = dataset["manifest_json"]
    value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return json.loads(str(value))


def _canonical(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def quarantined_missing_mask(
    dataset,
    label_arrays: dict[str, np.ndarray],
    episodes: set[int],
) -> np.ndarray:
    modes = np.asarray(dataset["behavior_mode"]).astype(str)
    picked = np.asarray(dataset["pstack_anchor_pick_rl"], dtype=float) > 0.5
    episode = np.asarray(dataset["episode"], dtype=int)
    accepted = (modes == "optimizer_local") & picked
    valid = label_arrays["long_horizon_label_valid"] > 0.5
    return accepted & ~valid & np.isin(episode, tuple(sorted(episodes)))


def relabel_file(
    source: str | Path,
    output: str | Path,
    labels: list[dict],
    *,
    label_contract: str,
    label_source: str,
    dataset_name: str,
    quarantined_episodes: set[int] | None = None,
    label_contracts: dict | None = None,
) -> dict:
    source = Path(source)
    output = Path(output)
    with np.load(source, allow_pickle=False) as dataset:
        arrays = {name: np.asarray(dataset[name]) for name in dataset.files}
        manifest = _manifest(dataset)
        validate_pstack_residual_rows(dataset, manifest)
        source_sha256 = _file_sha256(source)
        source_contracts = verify_contract_map(
            manifest.get("experiment_contracts", {})
        )
        if not source_contracts:
            raise ValueError(f"{source} is missing verified experiment contracts")
        episode_contracts = {
            int(summary["episode"]): str(
                summary.get("experiment_contract_sha256", "")
            )
            for summary in manifest.get("episode_summaries", [])
        }
        unknown = set(episode_contracts.values()) - set(source_contracts)
        if unknown:
            raise ValueError(
                f"{source} references unknown experiment contracts: {sorted(unknown)}"
            )
        verified_labels = verify_contract_map(label_contracts or {})
        for sha256, contract in verified_labels.items():
            if sha256 in source_contracts and contract.payload != source_contracts[sha256].payload:
                raise ValueError(f"label/source contract collision for {sha256}")
        label_source_hashes = {
            str(label.get("source_file_sha256", "")) for label in labels
        }
        if label_source_hashes != {source_sha256}:
            raise ValueError(
                "long-horizon labels do not match the source dataset content"
            )
        label_arrays = build_label_arrays(dataset, labels, episode_contracts)
        modes = np.asarray(dataset["behavior_mode"]).astype(str)
        picked = np.asarray(dataset["pstack_anchor_pick_rl"], dtype=float) > 0.5
        accepted = (modes == "optimizer_local") & picked
        quarantined = quarantined_missing_mask(
            dataset, label_arrays, quarantined_episodes or set()
        )
    arrays.update(label_arrays)
    source_dataset = str(manifest.get("dataset", source.parent.name))
    manifest.update({
        "dataset": dataset_name,
        "source_dataset": source_dataset,
        "actor_supervision_contract": LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT,
        "long_horizon_label_contract": str(label_contract),
        "long_horizon_label_source": str(label_source),
        "long_horizon_source_file_sha256": source_sha256,
        "long_horizon_label_valid_count": int(label_arrays["long_horizon_label_valid"].sum()),
        "long_horizon_positive_count": int(label_arrays["long_horizon_positive"].sum()),
        "long_horizon_quarantined_count": int(np.count_nonzero(quarantined)),
    })
    arrays["manifest_json"] = np.asarray(json.dumps(manifest))
    _write_npz(output, arrays)
    return {
        "source_file": str(source),
        "output_file": str(output),
        "transitions": int(arrays["obs"].shape[0]),
        "accepted_local": int(np.count_nonzero(accepted)),
        "valid_labels": int(label_arrays["long_horizon_label_valid"].sum()),
        "positive_labels": int(label_arrays["long_horizon_positive"].sum()),
        "quarantined_labels": int(np.count_nonzero(quarantined)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--data", default="",
        help="optional comma-separated raw dataset globs; defaults to generator inputs",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minimum-positive", type=int, default=0)
    parser.add_argument("--required-positive-scenarios", default="")
    parser.add_argument("--required-labeled-scenarios", default="")
    parser.add_argument("--require-complete-accepted-labels", action="store_true")
    parser.add_argument("--fail-on-label-errors", action="store_true")
    parser.add_argument("--quarantine-label-errors", action="store_true")
    args = parser.parse_args(argv)

    label_path = Path(args.labels)
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    label_contracts = {
        sha256: contract.payload
        for sha256, contract in verify_contract_map(
            payload.get("experiment_contracts", {})
        ).items()
    }
    if not label_contracts:
        raise SystemExit("long-horizon labels are missing verified experiment contracts")
    if args.fail_on_label_errors and payload.get("errors"):
        raise SystemExit(f"long-horizon generator reported {len(payload['errors'])} errors")
    quarantined_by_file: dict[str, set[int]] = {}
    if args.quarantine_label_errors:
        for error in payload.get("errors", []):
            if "source_file" not in error or "episode" not in error:
                raise SystemExit(
                    "cannot quarantine a generator error without source_file and episode"
                )
            quarantined_by_file.setdefault(
                _canonical(error["source_file"]), set()
            ).add(int(error["episode"]))
    labels_by_file: dict[str, list[dict]] = {}
    for label in payload.get("labels", []):
        labels_by_file.setdefault(_canonical(label["source_file"]), []).append(label)
    output_dir = Path(args.out_dir)
    dataset_name = output_dir.name
    if args.data:
        patterns = [value.strip() for value in args.data.split(",") if value.strip()]
        sources = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    else:
        sources = list(payload.get("inputs", []))
    if not sources:
        raise SystemExit("no raw datasets were provided for long-horizon relabeling")
    rows = []
    for source in sources:
        source_path = Path(source)
        rows.append(relabel_file(
            source_path,
            output_dir / source_path.name,
            labels_by_file.get(_canonical(source_path), []),
            label_contract=str(payload.get("label_contract", "unknown")),
            label_source=str(label_path),
            dataset_name=dataset_name,
            quarantined_episodes=quarantined_by_file.get(
                _canonical(source_path), set()
            ),
            label_contracts=label_contracts,
        ))
    accepted = sum(row["accepted_local"] for row in rows)
    valid = sum(row["valid_labels"] for row in rows)
    positive = sum(row["positive_labels"] for row in rows)
    quarantined = sum(row["quarantined_labels"] for row in rows)
    positive_scenarios = Counter(
        str(label.get("target_scenario", "unknown"))
        for label in payload.get("labels", [])
        if label.get("long_horizon_positive")
    )
    labeled_scenarios = Counter(
        str(label.get("target_scenario", "unknown"))
        for label in payload.get("labels", [])
    )
    summary = {
        "format_version": "long_horizon_relabel_summary_v1",
        "actor_supervision_contract": LONG_HORIZON_ACTOR_SUPERVISION_CONTRACT,
        "label_contract": payload.get("label_contract"),
        "accepted_local": int(accepted),
        "valid_labels": int(valid),
        "positive_labels": int(positive),
        "quarantined_labels": int(quarantined),
        "labeled_scenarios": dict(labeled_scenarios),
        "positive_scenarios": dict(positive_scenarios),
        "generator_errors": payload.get("errors", []),
        "files": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "relabel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if args.require_complete_accepted_labels and valid + quarantined != accepted:
        raise SystemExit(
            "long-horizon labels are incomplete after quarantine: "
            f"valid={valid}, quarantined={quarantined}, accepted={accepted}"
        )
    if positive < args.minimum_positive:
        raise SystemExit(
            f"long-horizon positives {positive} are below minimum {args.minimum_positive}"
        )
    required_labeled = {
        value.strip() for value in args.required_labeled_scenarios.split(",")
        if value.strip()
    }
    missing_labeled = sorted(required_labeled - set(labeled_scenarios))
    if missing_labeled:
        raise SystemExit(f"no long-horizon labels for scenarios: {missing_labeled}")
    required = {
        value.strip() for value in args.required_positive_scenarios.split(",")
        if value.strip()
    }
    missing = sorted(required - set(positive_scenarios))
    if missing:
        raise SystemExit(f"no long-horizon positive labels for scenarios: {missing}")
    print(f"saved relabeled datasets -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
