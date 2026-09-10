"""Train a frozen-batch response-aware Double DQN ensemble."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch

from rl_leader.response_dqn import (
    MODEL_FORMAT,
    VALUE_PARAMETERIZATIONS,
    ResponseDQNConfig,
    _check_stop_files,
    train_response_dqn_member,
)
from rl_leader.response_dqn_catalog import StructuredActionCatalog
from rl_leader.response_dqn_data import load_frozen_response_replay


_TORCH_INTEROP_THREADS_CONFIGURED = False


def validate_sequential_td_replay(replay, *, gamma: float) -> None:
    """Fail before training if a sequential experiment receives fixed labels."""
    if replay.manifest.get("reward_semantics") != "interval_negative_ttt":
        raise ValueError("sequential TD requires interval_negative_ttt rewards")
    if replay.manifest.get("done_semantics") != "environment_terminal":
        raise ValueError("sequential TD requires environment_terminal done flags")
    if not (replay.done == 0).any():
        raise ValueError("sequential TD needs nonterminal transitions")
    if not (replay.option_steps == 1).all():
        raise ValueError("sequential TD requires one-step transitions")
    if float(gamma) != 1.0:
        raise ValueError("undiscounted finite-horizon total TTT requires gamma=1")


def _configure_torch_threads(member_threads: int) -> None:
    global _TORCH_INTEROP_THREADS_CONFIGURED
    torch.set_num_threads(int(member_threads))
    if _TORCH_INTEROP_THREADS_CONFIGURED:
        return
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        message = str(exc)
        if "cannot set number of interop threads" not in message:
            raise
    _TORCH_INTEROP_THREADS_CONFIGURED = True


def _parse_hidden(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("hidden must contain at least one width")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--gradient-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--backup-horizon", type=int, default=1,
        help="capped greedy-consistent DDQN backup; >1 requires raw intervals, gamma=1 and --no-group-bootstrap",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--soft-target-tau", type=float, default=0.0)
    parser.add_argument("--hidden", type=_parse_hidden, default=(256, 256))
    parser.add_argument("--min-action-support", type=int, default=1)
    parser.add_argument(
        "--value-parameterization", choices=VALUE_PARAMETERIZATIONS, default="free_q",
        help="optional raw-phase finite-horizon cost head; free_q preserves legacy Q",
    )
    parser.add_argument(
        "--conservative-alpha", type=float, default=0.0,
        help="optional masked CQL penalty weight; zero preserves vanilla DDQN",
    )
    parser.add_argument(
        "--terminal-batch-fraction", type=float, default=0.0,
        help="optional terminal replay stratum; zero preserves uniform batch sampling",
    )
    parser.add_argument(
        "--mask-constant-features", action="store_true",
        help="mask exactly constant fitted observation and response columns",
    )
    parser.add_argument(
        "--no-bootstrap",
        "--no-group-bootstrap",
        action="store_true",
        help="disable event-group resampling; Bellman bootstrapping is unchanged",
    )
    parser.add_argument(
        "--require-sequential-td",
        action="store_true",
        help="reject terminal-label regression data and require undiscounted interval TTT",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--member-threads", type=int, default=1)
    parser.add_argument(
        "--stop-file", dest="stop_files", type=Path, action="append", default=[],
        help="interrupt training when this path exists; may be repeated",
    )
    return parser.parse_args()


def _train_member_worker(payload: dict) -> dict:
    stop_files = payload.get("stop_files", ())
    _check_stop_files(stop_files)
    _configure_torch_threads(int(payload["member_threads"]))
    replay = load_frozen_response_replay(payload["data"])
    catalog = StructuredActionCatalog.from_manifest(replay.manifest["catalog"])
    config = ResponseDQNConfig(**payload["config"])
    member = int(payload["member"])
    model = train_response_dqn_member(
        replay,
        catalog.feature_matrix,
        catalog_fingerprint=catalog.fingerprint,
        config=config,
        seed=int(payload["seed"]) + member,
        device=payload["device"],
        bootstrap=bool(payload["bootstrap"]),
        stop_files=stop_files,
    )
    path = Path(payload["output_dir"]) / f"response_dqn_member_{member:02d}.pt"
    model.save(path)
    losses = model.training_losses
    return {
        "member": member,
        "checkpoint": str(path),
        "seed": model.seed,
        "gradient_steps": len(losses),
        "initial_loss": float(losses[0]) if losses else None,
        "final_loss": float(losses[-1]) if losses else None,
        "tail_mean_loss": (
            float(sum(losses[-100:]) / min(100, len(losses))) if losses else None
        ),
    }


def main() -> None:
    args = _parse_args()
    _check_stop_files(args.stop_files)
    if args.workers < 1 or args.member_threads < 1:
        raise SystemExit("workers and member-threads must be positive")
    if args.workers > 1 and args.device != "cpu":
        raise SystemExit("parallel ensemble training currently requires --device cpu")
    replay = load_frozen_response_replay(args.data)
    if args.require_sequential_td:
        validate_sequential_td_replay(replay, gamma=args.gamma)
    catalog = StructuredActionCatalog.from_manifest(replay.manifest["catalog"])
    config = ResponseDQNConfig(
        gamma=args.gamma,
        backup_horizon=args.backup_horizon,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        soft_target_tau=args.soft_target_tau,
        reward_scale=args.reward_scale,
        hidden=args.hidden,
        ensemble_size=args.ensemble_size,
        min_action_support=args.min_action_support,
        conservative_alpha=args.conservative_alpha,
        terminal_batch_fraction=args.terminal_batch_fraction,
        mask_constant_features=args.mask_constant_features,
        value_parameterization=args.value_parameterization,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [
        args.output_dir / f"response_dqn_member_{member:02d}.pt"
        for member in range(args.ensemble_size)
    ]
    collisions = [path for path in expected_paths if path.exists()]
    manifest_path = args.output_dir / "ensemble_manifest.json"
    if manifest_path.exists():
        collisions.append(manifest_path)
    if collisions:
        raise SystemExit(
            "refusing to overwrite existing training outputs: "
            + ", ".join(map(str, collisions))
        )

    payloads = [{
        "member": member,
        "data": str(args.data),
        "output_dir": str(args.output_dir),
        "device": args.device,
        "seed": args.seed,
        "member_threads": args.member_threads,
        "bootstrap": not args.no_bootstrap,
        "stop_files": tuple(map(str, args.stop_files)),
        "config": asdict(config),
    } for member in range(args.ensemble_size)]
    member_results = []
    if args.workers == 1:
        for payload in payloads:
            result = _train_member_worker(payload)
            member_results.append(result)
            print(json.dumps({"event": "member_completed", **result}), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, args.ensemble_size)
        ) as executor:
            futures = {
                executor.submit(_train_member_worker, payload): payload["member"]
                for payload in payloads
            }
            for future in as_completed(futures):
                result = future.result()
                member_results.append(result)
                print(json.dumps({"event": "member_completed", **result}), flush=True)
    member_results.sort(key=lambda row: row["member"])
    checkpoints = [row["checkpoint"] for row in member_results]
    summary = {
        "format_version": f"{MODEL_FORMAT}_ensemble_manifest_v1",
        "training_phase": "frozen_batch_no_simulator_interaction",
        "training_objective": {
            "bellman_loss": "smooth_l1_double_dqn",
            "conservative_alpha": args.conservative_alpha,
            "conservative_penalty": "masked_logsumexp_q_minus_behavior_q",
            "temperature": 1.0,
        },
        "target_semantics": {
            "reward": replay.manifest.get("reward_semantics", "legacy_unspecified"),
            "done": replay.manifest.get("done_semantics", "legacy_unspecified"),
            "bellman_nonterminal_rows": int((replay.done == 0).sum()),
            "group_resampling": not args.no_bootstrap,
            "gamma": args.gamma,
            "backup_horizon": args.backup_horizon,
        },
        "source_dataset": str(args.data),
        "response_equivalence_mode": replay.manifest.get("response_equivalence_mode", "legacy_follower_runtime_v1"),
        "catalog_fingerprint": catalog.fingerprint,
        "ensemble_size": len(member_results),
        "checkpoints": checkpoints,
        "member_results": member_results,
        "action_support_counts": replay.action_support_counts().tolist(),
        "event_groups": replay.unique_event_groups().tolist(),
        "config": {key: value for key, value in vars(args).items() if key != "stop_files"}
        | {"data": str(args.data), "output_dir": str(args.output_dir)},
    }
    summary_path = manifest_path
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
