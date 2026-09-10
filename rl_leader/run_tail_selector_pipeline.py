"""Run the fail-closed tail selector pipeline over completed drain artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl_leader.build_tail_pairwise_dataset import build_tail_pairwise_dataset
from rl_leader.select_long_horizon_candidates import select_long_horizon_candidates
from rl_leader.train_tail_selector import train_tail_selector


DRAIN_OUT_FORMAT = "balanced_positive_zero_demand_drain_out_v1"
TAIL_SELECTOR_PIPELINE_FORMAT = "pstack_anchored_tail_selector_pipeline_v1"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_complete_drain_artifact(path: Path) -> bool:
    payload = _load_json(path)
    if payload is None:
        return False
    return bool(
        payload.get("format_version") == DRAIN_OUT_FORMAT
        and payload.get("status") == "complete"
        and payload.get("passed") is True
        and isinstance(payload.get("outcomes"), list)
        and len(payload["outcomes"]) > 0
    )


def discover_complete_drain_artifacts(roots: list[Path]) -> list[Path]:
    """Find replayable drain artifacts and ignore in-progress or empty files."""
    discovered: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else root.rglob("*.json")
        for path in files:
            if _is_complete_drain_artifact(path):
                discovered[str(path.resolve())] = path
    return [discovered[key] for key in sorted(discovered)]


def _write_pipeline_summary(
    output_path: Path,
    *,
    drain_paths: list[Path],
    pairwise_path: Path,
    selector_path: Path,
    model_path: Path,
    pairwise: dict,
    selector: dict,
    model: dict,
) -> dict:
    result = {
        "format_version": TAIL_SELECTOR_PIPELINE_FORMAT,
        "inputs": [str(path) for path in drain_paths],
        "outputs": {
            "pairwise_dataset": str(pairwise_path),
            "oracle_selector": str(selector_path),
            "learned_selector_model": str(model_path),
        },
        "summary": {
            "complete_drain_artifacts": len(drain_paths),
            "pairwise": pairwise.get("summary", {}),
            "selector": selector.get("summary", {}),
            "model": model.get("summary", {}),
        },
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_tail_selector_pipeline(
    roots: list[Path],
    result_dir: Path,
    *,
    name: str = "current_complete",
    min_margin_ratio: float = 0.0,
    min_gain_veh_h: float = 0.0,
    max_terminal_inventory_delta_veh: float | None = None,
    ensemble_size: int = 8,
    seed: int = 0,
    logistic_steps: int = 2000,
    logistic_lr: float = 0.1,
    classifier_l2: float = 1.0e-3,
    regressor_l2: float = 1.0e-2,
    lcb_z: float = 1.0,
    min_prob_lcb: float = 0.55,
    min_gain_lcb_veh_h: float = 0.0,
    min_margin_lcb: float = 0.0,
    min_event_groups: int = 50,
    min_positive_rows: int = 20,
) -> dict:
    drain_paths = discover_complete_drain_artifacts(roots)
    if not drain_paths:
        raise ValueError("tail selector pipeline found no complete drain artifacts")

    pairwise_path = result_dir / "tail_pairwise_v1" / f"{name}.json"
    selector_path = result_dir / "tail_selector_v1" / f"{name}.json"
    model_path = result_dir / "tail_selector_model_v1" / f"{name}.json"
    summary_path = result_dir / "tail_selector_pipeline_v1" / f"{name}.json"

    pairwise = build_tail_pairwise_dataset(drain_paths, pairwise_path)
    selector = select_long_horizon_candidates(
        pairwise_path,
        selector_path,
        min_margin_ratio=min_margin_ratio,
        min_gain_veh_h=min_gain_veh_h,
        max_terminal_inventory_delta_veh=max_terminal_inventory_delta_veh,
    )
    model = train_tail_selector(
        pairwise_path,
        model_path,
        ensemble_size=ensemble_size,
        seed=seed,
        logistic_steps=logistic_steps,
        logistic_lr=logistic_lr,
        classifier_l2=classifier_l2,
        regressor_l2=regressor_l2,
        lcb_z=lcb_z,
        min_prob_lcb=min_prob_lcb,
        min_gain_lcb_veh_h=min_gain_lcb_veh_h,
        min_margin_lcb=min_margin_lcb,
        min_event_groups=min_event_groups,
        min_positive_rows=min_positive_rows,
    )
    return _write_pipeline_summary(
        summary_path,
        drain_paths=drain_paths,
        pairwise_path=pairwise_path,
        selector_path=selector_path,
        model_path=model_path,
        pairwise=pairwise,
        selector=selector,
        model=model,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "results/rl_phase0_implementation_20260829",
            "results/rl_phase0_implementation_20260830",
        ],
    )
    parser.add_argument(
        "--result-dir",
        default="results/rl_phase0_implementation_20260830",
    )
    parser.add_argument("--name", default="current_complete")
    parser.add_argument("--min-margin-ratio", type=float, default=0.0)
    parser.add_argument("--min-gain-veh-h", type=float, default=0.0)
    parser.add_argument("--max-terminal-inventory-delta-veh", type=float)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logistic-steps", type=int, default=2000)
    parser.add_argument("--logistic-lr", type=float, default=0.1)
    parser.add_argument("--classifier-l2", type=float, default=1.0e-3)
    parser.add_argument("--regressor-l2", type=float, default=1.0e-2)
    parser.add_argument("--lcb-z", type=float, default=1.0)
    parser.add_argument("--min-prob-lcb", type=float, default=0.55)
    parser.add_argument("--min-gain-lcb-veh-h", type=float, default=0.0)
    parser.add_argument("--min-margin-lcb", type=float, default=0.0)
    parser.add_argument("--min-event-groups", type=int, default=50)
    parser.add_argument("--min-positive-rows", type=int, default=20)
    args = parser.parse_args(argv)

    result = run_tail_selector_pipeline(
        [Path(value) for value in args.roots],
        Path(args.result_dir),
        name=args.name,
        min_margin_ratio=args.min_margin_ratio,
        min_gain_veh_h=args.min_gain_veh_h,
        max_terminal_inventory_delta_veh=args.max_terminal_inventory_delta_veh,
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        logistic_steps=args.logistic_steps,
        logistic_lr=args.logistic_lr,
        classifier_l2=args.classifier_l2,
        regressor_l2=args.regressor_l2,
        lcb_z=args.lcb_z,
        min_prob_lcb=args.min_prob_lcb,
        min_gain_lcb_veh_h=args.min_gain_lcb_veh_h,
        min_margin_lcb=args.min_margin_lcb,
        min_event_groups=args.min_event_groups,
        min_positive_rows=args.min_positive_rows,
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
