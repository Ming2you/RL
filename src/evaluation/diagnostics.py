from __future__ import annotations

from typing import List

from src.models.state import DiagnosticResult, EvaluationResult, ExperimentConfig


def diagnose(result: EvaluationResult, cfg: ExperimentConfig) -> DiagnosticResult:
    modes: List[str] = []
    suggestions: List[str] = []
    if result.improvement_pct < cfg.evaluation.min_improvement_pct:
        modes.append("main_improvement_below_target")
        suggestions.extend([
            "expand leader candidate ranges",
            "reduce smoothness weights",
            "increase prediction horizon or Nash iterations",
        ])
    validation = result.control_validation
    if not validation.get("vsl", {}).get("pass", True):
        modes.append("freeway_density_or_vsl_validation_failed")
        suggestions.append("increase density penalty or lower N_UF_star upper range")
    if not validation.get("ramp_metering", {}).get("pass", True):
        modes.append("ramp_metering_or_queue_validation_failed")
        suggestions.append("increase ramp queue penalty or relax infeasible N_UF tracking")
    if not validation.get("boundary_balance", {}).get("pass", True):
        modes.append("boundary_queue_balance_failed")
        if validation.get("boundary_balance", {}).get("boundary_balance_degenerate", 0.0) > 0.5:
            suggestions.append("boundary balance is degenerate; inspect saturation/empty ratios before tuning")
        else:
            suggestions.append("increase movement-level boundary balance weight or switch receiving-space rule")
    if not validation.get("offset", {}).get("pass", True):
        modes.append("offset_validation_failed")
        suggestions.append("reduce max offset step or increase offset smoothness")
    if not modes:
        modes.append("none")
        suggestions.append("all configured acceptance criteria passed")
    return DiagnosticResult(
        failure_modes=modes,
        suggestions=suggestions,
        dominant_failure_mode=modes[0],
    )
