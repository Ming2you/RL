from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.controllers.auto_tuner import AutoTuner
from src.evaluation.diagnostics import diagnose
from src.evaluation.metrics import evaluate
from src.evaluation.plots import write_placeholder_plots
from src.evaluation.report import write_report
from src.models.demand import ScenarioConfig, load_scenarios
from src.models.state import DiagnosticResult, EvaluationResult, ExperimentConfig
from src.simulation.closed_loop_runner import run_closed_loop


def _default_scenarios_path(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent / "scenarios.yaml"


def _save_attempt(
    out: Path,
    cfg: ExperimentConfig,
    evaluation: EvaluationResult,
    diagnostic: DiagnosticResult,
    changes: List[str],
    attempt_id: int,
) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out / "metrics_summary.json").write_text(json.dumps({
        "attempt_id": attempt_id,
        "passed": evaluation.passed,
        "improvement_pct": evaluation.improvement_pct,
        "metrics": evaluation.metrics,
        "control_validation": evaluation.control_validation,
        "changes": changes,
    }, indent=2), encoding="utf-8")
    (out / "diagnostics.json").write_text(json.dumps({
        "failure_modes": diagnostic.failure_modes,
        "dominant_failure_mode": diagnostic.dominant_failure_mode,
        "suggestions": diagnostic.suggestions,
    }, indent=2), encoding="utf-8")
    write_placeholder_plots(out, evaluation.metrics)
    return {
        "attempt_id": attempt_id,
        "passed": evaluation.passed,
        "improvement_pct": evaluation.improvement_pct,
        "changes": changes,
        "path": str(out),
    }


def run_attempt(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
    baseline_mode: str,
    controller_mode: str,
    out: Path,
) -> tuple[EvaluationResult, DiagnosticResult]:
    baseline = run_closed_loop(
        cfg,
        scenario,
        mode="baseline",
        output_dir=out / "baseline",
        baseline_mode=baseline_mode,
    )
    proposed = run_closed_loop(
        cfg,
        scenario,
        mode=controller_mode,
        output_dir=out / "proposed",
        baseline_mode=baseline_mode,
    )
    evaluation = evaluate(baseline, proposed, cfg)
    diagnostic = diagnose(evaluation, cfg)
    return evaluation, diagnostic


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--scenarios-config", default=None)
    parser.add_argument("--scenario", default="peak_demand")
    parser.add_argument("--baseline", default="fixed_signal_fixed_speed")
    parser.add_argument("--controller", default="stackelberg_mpc")
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--min-improvement-pct", type=float, default=None)
    parser.add_argument("--output", default="outputs/peak_demand_stackelberg")
    parser.add_argument("--T-total", type=float, default=None, help="Optional short-run override for smoke tests.")
    parser.add_argument("--mpc-horizon-steps", type=int, default=None, help="Optional MPC horizon override.")
    parser.add_argument("--leader-candidate-count", type=int, default=None, help="Optional leader candidate override.")
    parser.add_argument(
        "--follower-solver-mode",
        choices=["two_block", "distributed"],
        default=None,
        help="Optional follower solver override.",
    )
    parser.add_argument("--max-nash-iter", type=int, default=None, help="Optional Nash iteration override.")
    parser.add_argument("--freeway-horizon-beam-width", type=int, default=None, help="Optional freeway beam width override.")
    parser.add_argument("--freeway-ramp-candidate-limit", type=int, default=None, help="Optional ramp candidate override.")
    parser.add_argument("--freeway-vsl-candidate-limit", type=int, default=None, help="Optional VSL candidate override.")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.min_improvement_pct is not None:
        overrides.setdefault("evaluation", {})["min_improvement_pct"] = args.min_improvement_pct
    if args.T_total is not None:
        overrides.setdefault("simulation", {})["T_total"] = args.T_total
    if args.mpc_horizon_steps is not None:
        overrides.setdefault("mpc", {})["horizon_steps"] = args.mpc_horizon_steps
    if args.leader_candidate_count is not None:
        overrides.setdefault("mpc", {})["leader_candidate_count"] = args.leader_candidate_count
    if args.follower_solver_mode is not None:
        overrides.setdefault("mpc", {})["follower_solver_mode"] = args.follower_solver_mode
    if args.max_nash_iter is not None:
        overrides.setdefault("mpc", {})["max_nash_iter"] = args.max_nash_iter
    if args.freeway_horizon_beam_width is not None:
        overrides.setdefault("freeway_follower", {})["horizon_beam_width"] = args.freeway_horizon_beam_width
    if args.freeway_ramp_candidate_limit is not None:
        overrides.setdefault("freeway_follower", {})["horizon_ramp_candidate_limit"] = args.freeway_ramp_candidate_limit
    if args.freeway_vsl_candidate_limit is not None:
        overrides.setdefault("freeway_follower", {})["horizon_vsl_candidate_limit_per_link"] = args.freeway_vsl_candidate_limit
    cfg = ExperimentConfig.from_file(args.config, overrides)
    scenarios_path = Path(args.scenarios_config) if args.scenarios_config else _default_scenarios_path(args.config)
    scenarios = load_scenarios(scenarios_path)
    if args.scenario not in scenarios:
        raise SystemExit(f"Unknown scenario {args.scenario}. Available: {', '.join(sorted(scenarios))}")
    scenario = scenarios[args.scenario]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tuner = AutoTuner(cfg)
    max_attempt = cfg.auto_tuning.max_iterations if args.auto_tune else 0
    attempts = []
    best_eval: EvaluationResult | None = None
    best_diag: DiagnosticResult | None = None
    best_attempt_dir: Path | None = None
    dominant = "none"

    for attempt_id in range(max_attempt + 1):
        attempt = tuner.attempt_config(attempt_id, dominant)
        attempt_dir = out / f"attempt_{attempt_id}"
        evaluation, diagnostic = run_attempt(
            attempt.config,
            scenario,
            args.baseline,
            args.controller,
            attempt_dir,
        )
        attempts.append(_save_attempt(
            attempt_dir,
            attempt.config,
            evaluation,
            diagnostic,
            attempt.changes,
            attempt_id,
        ))
        dominant = diagnostic.dominant_failure_mode
        if best_eval is None or evaluation.improvement_pct > best_eval.improvement_pct:
            best_eval = evaluation
            best_diag = diagnostic
            best_attempt_dir = attempt_dir
        if evaluation.passed:
            best_eval = evaluation
            best_diag = diagnostic
            best_attempt_dir = attempt_dir
            break

    assert best_eval is not None and best_diag is not None and best_attempt_dir is not None
    write_report(
        out / "report.md",
        cfg,
        scenario.name,
        args.baseline,
        args.controller,
        best_eval,
        best_diag,
        attempts,
    )
    (out / "attempt_history.json").write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")
    print(
        f"{'PASS' if best_eval.passed else 'FAIL'} "
        f"improvement={best_eval.improvement_pct:.2f}% "
        f"best_attempt={best_attempt_dir.name} report={out / 'report.md'}"
    )


if __name__ == "__main__":
    main()
