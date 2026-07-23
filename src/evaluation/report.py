from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.models.state import DiagnosticResult, EvaluationResult, ExperimentConfig


def _yes(value: bool) -> str:
    return "Yes" if value else "No"


def render_report(
    cfg: ExperimentConfig,
    scenario_name: str,
    baseline_mode: str,
    controller_mode: str,
    result: EvaluationResult,
    diagnostic: DiagnosticResult,
    attempts: Iterable[Mapping[str, Any]],
    ablation_results: Mapping[str, Any] | None = None,
) -> str:
    m = result.metrics
    lines = [
        f"# Experiment Report: {scenario_name}",
        "",
        "## Metadata",
        f"- Scenario: {scenario_name}",
        f"- Baseline mode: {baseline_mode}",
        f"- Controller mode: {controller_mode}",
        f"- Seed: {cfg.simulation.random_seed}",
        f"- Simulation horizon: {cfg.simulation.T_total} s",
        "",
        "## Final Result",
        f"**{'PASS' if result.passed else 'FAIL'}**",
        "",
        "| Metric | Baseline | Proposed | Improvement | Pass |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Total TTT | {m.get('baseline_total_ttt', 0.0):.3f} | "
            f"{m.get('proposed_total_ttt', 0.0):.3f} | "
            f"{result.improvement_pct:.2f}% | "
            f"{_yes(result.improvement_pct >= cfg.evaluation.min_improvement_pct)} |"
        ),
        (
            f"| Freeway TTT | {m.get('baseline_freeway_ttt', 0.0):.3f} | "
            f"{m.get('proposed_freeway_ttt', 0.0):.3f} |  |  |"
        ),
        (
            f"| Urban TTT | {m.get('baseline_urban_ttt', 0.0):.3f} | "
            f"{m.get('proposed_urban_ttt', 0.0):.3f} |  |  |"
        ),
        (
            f"| Boundary B_in | {m.get('baseline_B_in', 0.0):.4f} | "
            f"{m.get('proposed_B_in', 0.0):.4f} |  | "
            f"{_yes(result.control_validation.get('boundary_balance', {}).get('pass', False))} |"
        ),
        (
            f"| Boundary B_out | {m.get('baseline_B_out', 0.0):.4f} | "
            f"{m.get('proposed_B_out', 0.0):.4f} |  | "
            f"{_yes(result.control_validation.get('boundary_balance', {}).get('pass', False))} |"
        ),
        (
            f"| Boundary CV (descriptive) | {m.get('baseline_CV_boundary', 0.0):.3f} | "
            f"{m.get('proposed_CV_boundary', 0.0):.3f} |  |  |"
        ),
        "",
        "## Control Validation",
    ]
    for name, values in result.control_validation.items():
        status = _yes(bool(values.get("pass", False)))
        detail = ", ".join(f"{k}={v:.4g}" for k, v in values.items() if isinstance(v, (int, float)) and k != "pass")
        lines.append(f"- {name}: {status}. {detail}")
    lines.extend([
        "",
        "## Diagnostics",
        f"- Dominant failure mode: {diagnostic.dominant_failure_mode}",
    ])
    for suggestion in diagnostic.suggestions:
        lines.append(f"- {suggestion}")
    lines.extend(["", "## Attempt History"])
    for attempt in attempts:
        lines.append(
            f"- attempt_{attempt.get('attempt_id')}: "
            f"improvement={attempt.get('improvement_pct', 0.0):.2f}%, "
            f"passed={attempt.get('passed', False)}, "
            f"changes={'; '.join(attempt.get('changes', []))}"
        )
    lines.extend(["", "## Ablation Results"])
    if ablation_results:
        for name, values in ablation_results.items():
            lines.append(f"- {name}: total_ttt={values.get('total_ttt', 0.0):.3f}")
    else:
        lines.append("- Not run for this attempt.")
    lines.extend([
        "",
        "## Caveats",
        "- This implementation uses deterministic grid/heuristic follower solvers when nonlinear optimization dependencies are unavailable.",
        "- `N_P_star` is an urban accumulation target in vehicles; `N_UF_star` is a freeway on-ramp flow target in vehicles per hour.",
        "- Boundary queue balancing is evaluated separately from Total TTT.",
    ])
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, *args: Any, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(*args, **kwargs), encoding="utf-8")
    return path
