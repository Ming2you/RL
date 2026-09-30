"""고정 검증의 입력·수치 옵션을 실행 설정과 연결하는 읽기 전용 검사.

source/scenario/seed gate와 derivative 합격 판정은 기존 runner가 담당한다.
여기는 검증한 계산 조건이 무엇인지를 검사하며 실패를 PASS로 바꾸지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


# 계산량 변경은 수렴 보장이 아니라 의도된 실행 예산 차이로 반드시 반환한다.
COMPUTE_BUDGET_OPTIONS = frozenset({
    "max_iterations", "local_max_iterations", "local_max_evaluations", "max_candidates",
})


class ValidationProvenanceError(ValueError):
    """검증 조건이 맞지 않을 때 파일에 보존할 수 있는 상세 report를 제공한다."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("Validation provenance mismatch: " + "; ".join(report["errors"]))


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    # JSON 저장과 동일하게 key/tuple 표현을 정규화하되 NaN/Inf는 허용하지 않는다.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_differences(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}.length"]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in _differences(a, b, f"{path}[{index}]")]
    return [] if left == right else [path]


def verify_validation_provenance(
    fixed_validation_path: Path | str,
    sensitivity_validation_path: Path | str,
    current_config: Any,
    current_options: Any,
) -> dict[str, Any]:
    """필수 입력을 비교하고 JSON report를 반환한다. 불일치면 상세 예외를 낸다.

    config는 to_dict 지원 객체/JSON dict, options는 dataclass/JSON dict를 받는다.
    실행 구간 수 T_total도 동일해야 한다. 고정 snapshot 시점과 closed-loop 시작
    시점은 서로 다른 용도이므로 같다고 주장하지 않고 report에 범위를 기록한다.
    """
    fixed_path, sensitivity_path = Path(fixed_validation_path), Path(sensitivity_validation_path)
    fixed_dir, evidence_dir = fixed_path.parent, sensitivity_path.parent
    report: dict[str, Any] = {"pass": False, "errors": [], "input_files": {},
        "comparisons": {}, "compute_budget_differences": {},
        "derivative_pass_is_not_overridden": True}

    def read(path: Path, label: str, *, csv_rows: bool = False):
        raw = path.read_bytes()
        report["input_files"][label] = {"path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest()}
        text = raw.decode("utf-8-sig")
        return list(csv.DictReader(text.splitlines())) if csv_rows else _plain(json.loads(text))

    def compare(label: str, left: Any, right: Any):
        changed = _differences(left, right)
        report["comparisons"][label] = {"equal": not changed, "different_fields": changed}
        if changed:
            report["errors"].append(f"{label}: {', '.join(changed[:8])}")

    try:
        fixed = read(fixed_path, "fixed_gate")
        evidence = read(sensitivity_path, "sensitivity_summary")
        fixed_config = read(fixed_dir / "config_used.yaml", "fixed_config")
        evidence_config = read(evidence_dir / "config.json", "sensitivity_config")
        fixed_options = read(fixed_dir / "solver_options.json", "fixed_options")
        cfg, options = _plain(current_config), _plain(current_options)
        compare("fixed_vs_sensitivity_config", fixed_config, evidence_config)
        compare("fixed_vs_current_config", fixed_config, cfg)

        # 상태와 horizon 수요가 같은지 실제 배열을 비교한다. 파일명만 신뢰하지 않는다.
        saved = {}
        for label, fixed_name, evidence_name in (
            ("state", "snapshot_state.json", "state.json"),
            ("forecast", "forecast.json", "forecast.json"),
            ("previous_control", "previous_control.json", "previous.json"),
            ("seed_control", "seed_control.json", "seed.json"),
        ):
            left = read(fixed_dir / fixed_name, f"fixed_{label}")
            right = read(evidence_dir / evidence_name, f"sensitivity_{label}")
            compare(f"fixed_vs_sensitivity_{label}", left, right)
            saved[label] = left

        # 계산 예산 4항목 외의 모든 옵션 차이를 기본 거부한다. FD·허용오차·안정화
        # 옵션을 바꾼 후 예전 PASS를 재사용하는 것을 막는다.
        for key in sorted(fixed_options.keys() | options.keys()):
            if key in fixed_options and key in options and fixed_options[key] == options[key]:
                continue
            if key in COMPUTE_BUDGET_OPTIONS and key in fixed_options and key in options:
                report["compute_budget_differences"][key] = {
                    "fixed": fixed_options[key], "current": options[key]}
            else:
                report["errors"].append(f"solver_option_changed: {key}")

        dt = float(cfg["simulation"]["control_interval"])
        horizon = int(options["horizon_steps"])
        snapshot_time = float(saved["state"]["time_sec"])
        compare("forecast_length_vs_current_horizon", len(saved["forecast"]), horizon)
        compare("snapshot_time_vs_sensitivity_summary", snapshot_time, evidence["snapshot_time_sec"])
        compare("forecast_seconds_vs_sensitivity_summary", horizon * dt, evidence["horizon_seconds"])
        report["time_windows"] = {"fixed_snapshot_time_sec": snapshot_time,
            "prediction_horizon_sec": horizon * dt,
            "fixed_simulation_duration_sec": fixed_config["simulation"]["T_total"],
            "sensitivity_simulation_duration_sec": evidence_config["simulation"]["T_total"],
            "current_simulation_duration_sec": cfg["simulation"]["T_total"],
            "fixed_demand_generation_horizon_sec": fixed["demand_horizon_sec"],
            "independent_full_demand_profile_verified": False,
            "scope": "Same saved snapshot and forecast; the independent summary does not record a complete demand profile or its generation horizon."}

        # 실제 physical stencil 폭을 읽는다. refined summary의 배율 이름만으로는
        # 0.01 s / 0.3 veh/h를 검증했다고 간주할 수 없다.
        basis = read(evidence_dir / "basis_sensitivities.csv", "sensitivity_basis", csv_rows=True)
        kind_option = {"green": "fd_green_sec", "offset": "fd_offset_sec", "meter": "fd_metering_veh_h"}
        base_widths = {kind: {float(row["physical_stencil_half_width"])
            for row in basis if row["kind"] == kind} for kind in kind_option}
        if any(len(widths) != 1 or not all(math.isfinite(w) and w > 0 for w in widths)
               for widths in base_widths.values()):
            raise ValueError("Missing or inconsistent physical widths in basis_sensitivities.csv")
        widths = {kind: next(iter(values)) for kind, values in base_widths.items()}
        ratios = [float(options[key]) / widths[kind] for kind, key in kind_option.items()]
        factor = ratios[0]
        if not all(math.isclose(value, factor, rel_tol=1e-10, abs_tol=1e-12) for value in ratios):
            raise ValueError("Current physical FD widths do not share a recorded stencil factor")
        if math.isclose(factor, 1., rel_tol=1e-10, abs_tol=1e-12):
            trials = read(evidence_dir / "independent_directional_derivatives.csv",
                          "sensitivity_directional_trials", csv_rows=True)
            expected_count = int(evidence["directional_trials"])
        else:
            sweep = read(evidence_dir / "basis_stencil_sweep.csv", "sensitivity_stencil_sweep", csv_rows=True)
            trials = [row for row in sweep if math.isclose(float(row["basis_stencil_factor"]), factor,
                      rel_tol=1e-10, abs_tol=1e-12)]
            summaries = [value for key, value in evidence["basis_stencil_sweep"].items()
                         if math.isclose(float(key), factor, rel_tol=1e-10, abs_tol=1e-12)]
            if len(summaries) != 1:
                raise ValueError("Current FD stencil has no unique summary entry")
            expected_count = int(summaries[0]["all_epsilon_tests"])
            if any(not math.isclose(float(row["signal_stencil_sec"]), float(options["fd_green_sec"]), rel_tol=1e-10)
                   or not math.isclose(float(row["meter_stencil_veh_h"]), float(options["fd_metering_veh_h"]), rel_tol=1e-10)
                   for row in trials):
                raise ValueError("Recorded sweep physical widths differ from current FD options")
            compare("stencil_agreements_vs_summary", sum(row["agreement"] == "True" for row in trials),
                    summaries[0]["all_epsilon_agreements"])
        if not trials or expected_count <= 0:
            raise ValueError("No independent trials exist for the current FD stencil")
        compare("stencil_trial_count_vs_summary", len(trials), expected_count)
        report["stencil_evidence"] = {"factor_relative_to_recorded_basis": factor,
            "physical_widths": {key: options[key] for key in kind_option.values()},
            "matching_trial_count": len(trials),
            "matching_agreement_count": sum(row["agreement"] == "True" for row in trials),
            "all_matching_trials_agree": all(row["agreement"] == "True" for row in trials),
            "original_summary_all_independent_directional_agreement": evidence["all_independent_directional_agreement"],
            "scope": "Recorded stencil coverage only; this provenance PASS is not derivative or optimization convergence PASS."}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        report["errors"].append(f"validation_input_error: {exc}")
    report["pass"] = not report["errors"]
    if not report["pass"]:
        raise ValidationProvenanceError(report)
    return report
