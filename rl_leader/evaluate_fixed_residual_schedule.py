"""Evaluate an explicit continuous residual schedule in closed loop."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl_leader.env import RLLeaderEnv


EVAL_FORMAT = "fixed_response_residual_schedule_eval_v1"


@dataclass(frozen=True)
class ScheduledResidual:
    step: int
    label: str
    residual: tuple[float, ...]
    source: str

    def residual_array(self) -> np.ndarray:
        return np.asarray(self.residual, dtype=np.float32)


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _residual_from_sparse(
    action_names: tuple[str, ...],
    sparse: dict[str, Any],
) -> tuple[float, ...]:
    by_name = {name: index for index, name in enumerate(action_names)}
    residual = np.zeros(len(action_names), dtype=np.float32)
    missing = sorted(set(map(str, sparse)) - set(by_name))
    if missing:
        raise ValueError(f"residual references unknown action channels: {missing}")
    for name, value in sparse.items():
        residual[by_name[str(name)]] = float(value)
    if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1.0 + 1.0e-7:
        raise ValueError("residual values must stay finite and inside [-1, 1]")
    return tuple(map(float, residual))


def _residual_from_vector(
    action_names: tuple[str, ...],
    values: Sequence[float],
) -> tuple[float, ...]:
    residual = np.asarray(values, dtype=np.float32).reshape(-1)
    if residual.shape != (len(action_names),):
        raise ValueError(
            f"residual vector has dimension {residual.size}, expected {len(action_names)}"
        )
    if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1.0 + 1.0e-7:
        raise ValueError("residual values must stay finite and inside [-1, 1]")
    return tuple(map(float, residual))


def parse_sparse_schedule_items(
    items: list[str] | tuple[str, ...],
    action_names: tuple[str, ...],
) -> dict[int, ScheduledResidual]:
    """Parse ``STEP=name:value,name:value`` and explicit ``coordination_zero``."""
    schedule: dict[int, ScheduledResidual] = {}
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"schedule item must be STEP=SPEC: {item}")
        raw_step, raw_spec = str(item).split("=", 1)
        step = int(raw_step)
        if step < 0:
            raise ValueError(f"schedule step must be nonnegative: {step}")
        spec = raw_spec.strip()
        if not spec:
            raise ValueError("schedule spec must not be empty")
        if spec == "coordination_zero":
            residual = tuple([0.0] * len(action_names))
            label = spec
        else:
            sparse: dict[str, float] = {}
            for part in spec.split(","):
                if not part.strip():
                    continue
                if ":" not in part:
                    raise ValueError(f"residual entry must be name:value: {part}")
                name, raw_value = part.split(":", 1)
                sparse[name.strip()] = float(raw_value)
            if not sparse:
                raise ValueError(f"residual schedule has no nonzero entries: {item}")
            residual = _residual_from_sparse(action_names, sparse)
            label = "manual_sparse"
        if step in schedule:
            raise ValueError(f"duplicate schedule step: {step}")
        schedule[step] = ScheduledResidual(
            step=step,
            label=label,
            residual=residual,
            source="manual",
        )
    return dict(sorted(schedule.items()))


def _sampler_horizon_key(artifact: dict) -> str:
    return str(int(artifact.get("horizon_steps", 12)))


def _sampler_gain(record: dict, horizon_key: str) -> float:
    return float(
        record.get("horizon_labels", {})
        .get(horizon_key, {})
        .get("ttt_gain", 0.0)
    )


def _sampler_positive(record: dict, horizon_key: str) -> bool:
    label = record.get("horizon_labels", {}).get(horizon_key, {})
    return bool(label.get("positive", False) and label.get("validity_gate_pass", False))


def load_sampled_residual_schedule(
    artifact_path: Path,
    action_names: tuple[str, ...],
    *,
    selection: str = "best_h12",
) -> dict[int, ScheduledResidual]:
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if artifact.get("format_version") != "cached_tail_residual_sampler_v1":
        raise ValueError("sample artifact has the wrong format")
    step = int(artifact["control_step"])
    horizon_key = _sampler_horizon_key(artifact)
    records = list(artifact.get("h12_records", []))
    if not records:
        raise ValueError("sample artifact has no H12 records")
    if selection == "best_h12":
        selected = max(
            records,
            key=lambda row: (
                _sampler_gain(row, horizon_key),
                _sampler_positive(row, horizon_key),
                str(row.get("candidate_id", "")),
            ),
        )
    elif selection == "best_positive_h12":
        positives = [row for row in records if _sampler_positive(row, horizon_key)]
        if not positives:
            raise ValueError("sample artifact has no positive H12 records")
        selected = max(
            positives,
            key=lambda row: (
                _sampler_gain(row, horizon_key),
                str(row.get("candidate_id", "")),
            ),
        )
    else:
        selected = next(
            (
                row for row in records
                if selection in {
                    str(row.get("candidate_id", "")),
                    str(row.get("label", "")),
                }
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"sample artifact has no candidate {selection!r}")
    return {
        step: ScheduledResidual(
            step=step,
            label=str(selected.get("candidate_id", selection)),
            residual=_residual_from_vector(
                action_names,
                selected["continuous_residual"],
            ),
            source=str(artifact_path),
        )
    }


def _event_candidate_by_label(event: dict, label: str) -> dict | None:
    candidates = list(event.get("candidates_by_ttt", []))
    candidates.extend(event.get("candidate_generation", {}).get("all_unique_probes", []))
    for row in candidates:
        labels = [str(row.get("representative_label", ""))]
        labels.extend(map(str, row.get("aliases", [])))
        if label in labels:
            return row
    return None


def load_oracle_choice_schedule(
    artifact_path: Path,
    action_names: tuple[str, ...],
    *,
    selection: str = "oracle_choice",
) -> dict[int, ScheduledResidual]:
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    schedule: dict[int, ScheduledResidual] = {}
    for event in artifact.get("events", []):
        step = int(event["step"])
        if selection == "oracle_choice":
            label = str(event.get("oracle_choice", {}).get("selected", "anchor"))
        else:
            label = str(selection)
        if label == "anchor":
            continue
        if label == "coordination_zero":
            sparse = {}
        else:
            candidate = _event_candidate_by_label(event, label)
            if candidate is None:
                raise ValueError(
                    f"artifact {artifact_path} event {step} has no candidate label {label!r}"
                )
            sparse = dict(candidate.get("residual_nonzero", {}))
        if step in schedule:
            raise ValueError(f"duplicate oracle schedule step: {step}")
        schedule[step] = ScheduledResidual(
            step=step,
            label=label,
            residual=_residual_from_sparse(action_names, sparse),
            source=str(artifact_path),
        )
    return dict(sorted(schedule.items()))


def merge_schedules(*schedules: dict[int, ScheduledResidual]) -> dict[int, ScheduledResidual]:
    merged: dict[int, ScheduledResidual] = {}
    for schedule in schedules:
        overlap = sorted(set(merged) & set(schedule))
        if overlap:
            raise ValueError(f"duplicate schedule step(s): {overlap}")
        merged.update(schedule)
    return dict(sorted(merged.items()))


def run_fixed_residual_schedule(
    *,
    scenario: str,
    t_total: float,
    schedule: dict[int, ScheduledResidual],
    output_dir: Path,
    pstack_summary: dict | None = None,
    force_without_h3_gate: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "trace.jsonl"
    collisions = [path for path in (summary_path, trace_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing schedule outputs: "
            + ", ".join(map(str, collisions))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    env = RLLeaderEnv(
        scenario_name=scenario,
        T_total=t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    observation = np.asarray(env.reset(), dtype=np.float32)
    rewards: list[float] = []
    forced_steps: list[int] = []
    applied_nonzero_steps: list[int] = []
    coordination_zero_steps: list[int] = []
    labels: list[str] = []

    with trace_path.open("w", encoding="utf-8") as trace:
        step = 0
        done = False
        while not done:
            anchor_context = env.prepare_pstack_anchor_context()
            scheduled = schedule.get(step)
            if scheduled is None:
                next_observation, reward, done, info, _ = env.step_prepared_optimizer_anchor(
                    anchor_context,
                )
                label = "anchor"
                residual = np.zeros(env.action_dim, dtype=np.float32)
            else:
                forced_steps.append(step)
                residual = scheduled.residual_array()
                if np.linalg.norm(residual, ord=2) <= 1.0e-9:
                    coordination_zero_steps.append(step)
                if force_without_h3_gate:
                    next_observation, reward, done, info = env.step_anchored_candidate(
                        residual,
                        anchor_context,
                    )
                else:
                    next_observation, reward, done, info = env._step(
                        residual,
                        anchor_context=anchor_context,
                        apply_anchor_gate=True,
                    )
                deployed = np.asarray(env.last_deployed_residual, dtype=np.float32)
                if np.linalg.norm(deployed, ord=2) > 1.0e-9:
                    applied_nonzero_steps.append(step)
                label = scheduled.label

            rewards.append(float(reward))
            labels.append(label)
            payload = {
                "format_version": EVAL_FORMAT,
                "control_step": int(step),
                "scheduled": scheduled is not None,
                "label": label,
                "source": None if scheduled is None else scheduled.source,
                "step_ttt": float(-reward),
                "total_ttt_so_far": float(env.sim.total_ttt),
                "terminal_inventory": float(info.get("inventory_after", env._inventory())),
                "validity_gate_pass": bool(info.get("validity_gate_pass", False)),
                "force_without_h3_gate": bool(force_without_h3_gate),
                "residual_l2": float(np.linalg.norm(residual)),
                "residual_nonzero_count": int(np.count_nonzero(np.abs(residual) > 1.0e-9)),
                "observation_norm": float(np.linalg.norm(observation)),
                "next_observation_norm": float(np.linalg.norm(next_observation)),
            }
            for key in (
                "leader_rl_pstack_anchor_pick_rl",
                "leader_rl_pstack_anchor_pick_pstack",
                "leader_rl_pstack_anchor_gain",
                "leader_rl_pstack_anchor_required_gain",
            ):
                if key in info:
                    payload[key] = float(info[key])
            trace.write(json.dumps(payload, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps({
                "control_step": int(step),
                "done": bool(done),
                "label": label,
                "scheduled": scheduled is not None,
                "step_ttt": round(float(-reward), 6),
            }, sort_keys=True), flush=True)
            observation = np.asarray(next_observation, dtype=np.float32)
            step += 1

    rewards_array = np.asarray(rewards, dtype=np.float64)
    total_ttt = float(env.sim.total_ttt)
    summary = {
        "format_version": EVAL_FORMAT,
        "scenario": scenario,
        "t_total_sec": float(t_total),
        "control_steps": int(len(labels)),
        "total_ttt": total_ttt,
        "control_ttt": float(np.sum(-rewards_array)),
        "urban_ttt": float(env.sim.urban_ttt),
        "freeway_ttt": float(env.sim.freeway_ttt),
        "terminal_inventory": float(env._inventory()),
        "forced_steps": list(map(int, forced_steps)),
        "applied_nonzero_steps": list(map(int, applied_nonzero_steps)),
        "coordination_zero_steps": list(map(int, coordination_zero_steps)),
        "scheduled_labels": {
            str(step): item.label for step, item in sorted(schedule.items())
        },
        "force_without_h3_gate": bool(force_without_h3_gate),
        "trace_path": str(trace_path),
        "wall_seconds": float(time.perf_counter() - started),
    }
    if pstack_summary is not None:
        pstack_total = float(pstack_summary["total_ttt"])
        gap = total_ttt - pstack_total
        summary.update({
            "pstack_total_ttt": pstack_total,
            "vs_pstack_ttt_gap": float(gap),
            "vs_pstack_percent": float(
                100.0 * gap / max(abs(pstack_total), 1.0e-9)
            ),
            "beats_pstack": bool(gap < 0.0),
            "target_5pct_total_ttt": float(0.95 * pstack_total),
            "meets_5pct_target": bool(total_ttt <= 0.95 * pstack_total),
        })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="sweet_170_incident_w60")
    parser.add_argument("--t-total", type=float, default=14400.0)
    parser.add_argument(
        "--schedule",
        action="append",
        default=[],
        help=(
            "repeatable STEP=coordination_zero or "
            "STEP=channel:value,channel:value entry"
        ),
    )
    parser.add_argument(
        "--oracle-artifact",
        action="append",
        default=[],
        type=Path,
        help="P-CENT-guided oracle artifact; uses each event's oracle_choice",
    )
    parser.add_argument(
        "--oracle-selection",
        default="oracle_choice",
        help="oracle_choice or a fixed representative_label present in each event",
    )
    parser.add_argument(
        "--sample-artifact",
        action="append",
        default=[],
        type=Path,
        help="cached-tail residual sampler summary; selects one event candidate",
    )
    parser.add_argument(
        "--sample-selection",
        default="best_h12",
        help="best_h12, best_positive_h12, or a sampled candidate_id/label",
    )
    parser.add_argument(
        "--use-h3-anchor-gate",
        action="store_true",
        help="apply the existing H3 P-Stack anchor gate instead of forcing residuals",
    )
    parser.add_argument("--pstack-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = RLLeaderEnv(
        scenario_name=args.scenario,
        T_total=args.t_total,
        action_mode="full",
        mask="RL-FULL",
        pstack_anchor=True,
        action_parameterization="pstack_residual",
    )
    action_names = tuple(env.action_schema.names)
    schedules = [
        parse_sparse_schedule_items(args.schedule, action_names),
        *[
            load_oracle_choice_schedule(
                path,
                action_names,
                selection=args.oracle_selection,
            )
            for path in args.oracle_artifact
        ],
        *[
            load_sampled_residual_schedule(
                path,
                action_names,
                selection=args.sample_selection,
            )
            for path in args.sample_artifact
        ],
    ]
    schedule = merge_schedules(*schedules)
    if not schedule:
        raise SystemExit(
            "at least one --schedule, --oracle-artifact, or --sample-artifact entry is required"
        )
    run_fixed_residual_schedule(
        scenario=args.scenario,
        t_total=args.t_total,
        schedule=schedule,
        output_dir=args.output_dir,
        pstack_summary=_read_json(args.pstack_summary),
        force_without_h3_gate=not bool(args.use_h3_anchor_gate),
    )


if __name__ == "__main__":
    main()
