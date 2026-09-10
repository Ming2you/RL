"""Train a conservative selector over P-Stack-relative tail labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rl_leader.build_tail_pairwise_dataset import TAIL_PAIRWISE_FORMAT
from rl_leader.select_long_horizon_candidates import LONG_HORIZON_SELECTOR_FORMAT


TAIL_SELECTOR_MODEL_FORMAT = "pstack_anchored_tail_selector_model_v1"


@dataclass(frozen=True)
class FeatureSpec:
    native_anchor_branches: tuple[str, ...]
    candidate_execution_branches: tuple[str, ...]
    numeric_dimension: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != TAIL_PAIRWISE_FORMAT:
        raise ValueError("tail selector training requires a tail-pairwise dataset")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("tail selector training dataset has no rows")
    return data


def _valid_rows(data: dict) -> list[dict]:
    rows = [row for row in data["rows"] if row.get("target_valid") is True]
    if not rows:
        raise ValueError("tail selector training dataset has no valid target rows")
    targets = {int(row["target"]) for row in rows}
    if targets != {0, 1}:
        raise ValueError("tail selector training needs both positive and negative rows")
    return rows


def _numeric_payload(row: dict) -> np.ndarray:
    values = (
        list(row.get("observation", []))
        + list(row.get("anchor_envelope", []))
        + list(row.get("candidate_residual", []))
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("tail selector row has invalid numeric features")
    return vector


def _feature_spec(rows: list[dict]) -> FeatureSpec:
    dimensions = {_numeric_payload(row).size for row in rows}
    if len(dimensions) != 1:
        raise ValueError("tail selector rows have inconsistent feature dimensions")
    return FeatureSpec(
        native_anchor_branches=tuple(sorted({
            str(row.get("native_anchor_branch", "")) for row in rows
        })),
        candidate_execution_branches=tuple(sorted({
            str(row.get("candidate_execution_branch", "")) for row in rows
        })),
        numeric_dimension=dimensions.pop(),
    )


def _raw_features(rows: list[dict], spec: FeatureSpec) -> np.ndarray:
    branch_width = (
        len(spec.native_anchor_branches)
        + len(spec.candidate_execution_branches)
    )
    matrix = np.zeros((len(rows), spec.numeric_dimension + branch_width), dtype=np.float64)
    native_index = {
        value: index for index, value in enumerate(spec.native_anchor_branches)
    }
    candidate_index = {
        value: index for index, value in enumerate(spec.candidate_execution_branches)
    }
    offset = spec.numeric_dimension
    for row_index, row in enumerate(rows):
        numeric = _numeric_payload(row)
        if numeric.size != spec.numeric_dimension:
            raise ValueError("tail selector feature dimension drift")
        matrix[row_index, :spec.numeric_dimension] = numeric
        native_branch = str(row.get("native_anchor_branch", ""))
        candidate_branch = str(row.get("candidate_execution_branch", ""))
        if native_branch not in native_index or candidate_branch not in candidate_index:
            raise ValueError("tail selector branch is outside the feature spec")
        matrix[row_index, offset + native_index[native_branch]] = 1.0
        matrix[
            row_index,
            offset + len(spec.native_anchor_branches) + candidate_index[candidate_branch],
        ] = 1.0
    return matrix


def _standardize(
    matrix: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = matrix.mean(axis=0)
    if scale is None:
        scale = matrix.std(axis=0)
        scale = np.where(scale < 1.0e-6, 1.0, scale)
    return (matrix - mean) / scale, mean, scale


def _with_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((matrix.shape[0], 1)), matrix], axis=1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _fit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    l2: float,
    lr: float,
    steps: int,
) -> np.ndarray:
    x = _with_intercept(matrix)
    y = target.astype(np.float64)
    w = weight.astype(np.float64)
    w = w / max(float(w.sum()), 1.0e-12)
    coef = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.ones_like(coef)
    penalty[0] = 0.0
    for _ in range(int(steps)):
        prob = _sigmoid(x @ coef)
        grad = x.T @ (w * (prob - y)) + float(l2) * penalty * coef
        coef -= float(lr) * grad
    return coef


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    l2: float,
) -> np.ndarray:
    x = _with_intercept(matrix)
    sqrt_w = np.sqrt(weight.astype(np.float64)).reshape(-1, 1)
    xw = x * sqrt_w
    yw = target.astype(np.float64).reshape(-1, 1) * sqrt_w
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    normal = xw.T @ xw + penalty
    rhs = xw.T @ yw
    try:
        coef = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(normal) @ rhs
    return coef.reshape(-1)


def _predict(matrix: np.ndarray, model: dict) -> dict[str, np.ndarray]:
    x = _with_intercept(matrix)
    class_logits = []
    gain_predictions = []
    margin_predictions = []
    for member in model["ensemble"]:
        class_logits.append(x @ np.asarray(member["classifier_coef"], dtype=np.float64))
        gain_predictions.append(x @ np.asarray(member["gain_coef"], dtype=np.float64))
        margin_predictions.append(x @ np.asarray(member["margin_coef"], dtype=np.float64))
    class_prob = _sigmoid(np.vstack(class_logits))
    gain = np.vstack(gain_predictions)
    margin = np.vstack(margin_predictions)
    return {
        "prob_mean": class_prob.mean(axis=0),
        "prob_std": class_prob.std(axis=0),
        "gain_mean": gain.mean(axis=0),
        "gain_std": gain.std(axis=0),
        "margin_mean": margin.mean(axis=0),
        "margin_std": margin.std(axis=0),
    }


def _row_weights(rows: list[dict]) -> np.ndarray:
    weights = np.asarray([
        float(row.get("sample_weight", 1.0)) for row in rows
    ], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("tail selector sample weights must be finite and nonnegative")
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(rows), dtype=np.float64)
    return weights


def _train_model(
    rows: list[dict],
    *,
    spec: FeatureSpec,
    ensemble_size: int,
    seed: int,
    logistic_steps: int,
    logistic_lr: float,
    classifier_l2: float,
    regressor_l2: float,
) -> dict:
    raw = _raw_features(rows, spec)
    standardized, mean, scale = _standardize(raw)
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.float64)
    if set(map(int, targets.tolist())) != {0, 1}:
        raise ValueError("tail selector model needs both positive and negative rows")
    gains = np.asarray([float(row["gain_veh_h"]) for row in rows], dtype=np.float64)
    margins = np.asarray([float(row["margin_ratio"]) for row in rows], dtype=np.float64)
    weights = _row_weights(rows)
    rng = np.random.default_rng(int(seed))
    ensemble = []
    for member in range(int(ensemble_size)):
        if int(ensemble_size) == 1:
            indices = np.arange(len(rows))
        else:
            indices = rng.choice(len(rows), size=len(rows), replace=True, p=weights / weights.sum())
            if len({int(targets[index]) for index in indices}) < 2:
                indices = np.arange(len(rows))
        ensemble.append({
            "member": member,
            "classifier_coef": _fit_logistic(
                standardized[indices],
                targets[indices],
                weights[indices],
                l2=classifier_l2,
                lr=logistic_lr,
                steps=logistic_steps,
            ).tolist(),
            "gain_coef": _fit_ridge(
                standardized[indices],
                gains[indices],
                weights[indices],
                l2=regressor_l2,
            ).tolist(),
            "margin_coef": _fit_ridge(
                standardized[indices],
                margins[indices],
                weights[indices],
                l2=regressor_l2,
            ).tolist(),
        })
    return {
        "feature_spec": {
            "numeric_dimension": int(spec.numeric_dimension),
            "native_anchor_branches": list(spec.native_anchor_branches),
            "candidate_execution_branches": list(spec.candidate_execution_branches),
        },
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "ensemble": ensemble,
    }


def _score_rows(rows: list[dict], model: dict, *, lcb_z: float) -> list[dict]:
    spec = FeatureSpec(
        native_anchor_branches=tuple(model["feature_spec"]["native_anchor_branches"]),
        candidate_execution_branches=tuple(
            model["feature_spec"]["candidate_execution_branches"]
        ),
        numeric_dimension=int(model["feature_spec"]["numeric_dimension"]),
    )
    raw = _raw_features(rows, spec)
    standardized, _, _ = _standardize(
        raw,
        mean=np.asarray(model["feature_mean"], dtype=np.float64),
        scale=np.asarray(model["feature_scale"], dtype=np.float64),
    )
    predictions = _predict(standardized, model)
    scored = []
    for index, row in enumerate(rows):
        scored.append({
            "candidate_id": row.get("candidate_id"),
            "target": int(row.get("target", 0)),
            "target_valid": bool(row.get("target_valid")),
            "gain_veh_h": row.get("gain_veh_h"),
            "required_gain_veh_h": row.get("required_gain_veh_h"),
            "margin_ratio": row.get("margin_ratio"),
            "prob_positive_mean": float(predictions["prob_mean"][index]),
            "prob_positive_lcb": float(
                predictions["prob_mean"][index]
                - float(lcb_z) * predictions["prob_std"][index]
            ),
            "gain_lcb_veh_h": float(
                predictions["gain_mean"][index]
                - float(lcb_z) * predictions["gain_std"][index]
            ),
            "margin_lcb": float(
                predictions["margin_mean"][index]
                - float(lcb_z) * predictions["margin_std"][index]
            ),
        })
    return scored


def _select_scored_group(
    scored: list[dict],
    *,
    min_prob_lcb: float,
    min_gain_lcb_veh_h: float,
    min_margin_lcb: float,
) -> dict:
    eligible = [
        row for row in scored
        if row["prob_positive_lcb"] >= float(min_prob_lcb)
        and row["gain_lcb_veh_h"] > float(min_gain_lcb_veh_h)
        and row["margin_lcb"] >= float(min_margin_lcb)
    ]
    if not eligible:
        return {
            "decision": "pstack_fallback",
            "selected_candidate_id": None,
            "selected_is_true_positive": None,
            "scored_candidates": scored,
        }
    best = max(
        eligible,
        key=lambda row: (
            row["margin_lcb"],
            row["gain_lcb_veh_h"],
            row["prob_positive_lcb"],
            str(row["candidate_id"]),
        ),
    )
    return {
        "decision": "select_candidate",
        "selected_candidate_id": best["candidate_id"],
        "selected_is_true_positive": bool(best["target"] == 1),
        "scored_candidates": scored,
    }


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["event_group_id"]), []).append(row)
    return grouped


def _evaluate_predictions(
    rows: list[dict],
    model: dict,
    *,
    lcb_z: float,
    min_prob_lcb: float,
    min_gain_lcb_veh_h: float,
    min_margin_lcb: float,
) -> dict:
    selections = []
    for group_id, group_rows in sorted(_groups(rows).items()):
        scored = _score_rows(group_rows, model, lcb_z=lcb_z)
        selected = _select_scored_group(
            scored,
            min_prob_lcb=min_prob_lcb,
            min_gain_lcb_veh_h=min_gain_lcb_veh_h,
            min_margin_lcb=min_margin_lcb,
        )
        selected.update({
            "event_group_id": group_id,
            "scenario": group_rows[0].get("scenario"),
            "stratum": group_rows[0].get("stratum"),
            "has_true_positive": any(int(row["target"]) == 1 for row in group_rows),
        })
        selections.append(selected)
    selected = [row for row in selections if row["decision"] == "select_candidate"]
    false_positive = [
        row for row in selected if row["selected_is_true_positive"] is not True
    ]
    missed_positive = [
        row for row in selections
        if row["has_true_positive"] and row["decision"] != "select_candidate"
    ]
    return {
        "event_groups": len(selections),
        "selected_event_groups": len(selected),
        "false_positive_event_groups": len(false_positive),
        "missed_positive_event_groups": len(missed_positive),
        "selections": selections,
    }


def _folds(data: dict, rows: list[dict]) -> list[dict]:
    row_groups = _groups(rows)
    declared = data.get("folds") or []
    folds = []
    if declared:
        for fold in declared:
            test_groups = [
                group for group in fold.get("test_event_group_ids", [])
                if group in row_groups
            ]
            train_groups = [
                group for group in fold.get("train_event_group_ids", [])
                if group in row_groups
            ]
            if test_groups:
                folds.append({
                    "held_out_scenario": fold.get("held_out_scenario"),
                    "train_groups": train_groups,
                    "test_groups": test_groups,
                })
    if folds:
        return folds
    scenarios = sorted({str(row.get("scenario")) for row in rows})
    for scenario in scenarios:
        folds.append({
            "held_out_scenario": scenario,
            "train_groups": sorted({
                row["event_group_id"] for row in rows
                if str(row.get("scenario")) != scenario
            }),
            "test_groups": sorted({
                row["event_group_id"] for row in rows
                if str(row.get("scenario")) == scenario
            }),
        })
    return folds


def train_tail_selector(
    dataset_path: Path,
    output_path: Path,
    *,
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
    data = _load_dataset(dataset_path)
    rows = _valid_rows(data)
    spec = _feature_spec(rows)
    full_model = _train_model(
        rows,
        spec=spec,
        ensemble_size=ensemble_size,
        seed=seed,
        logistic_steps=logistic_steps,
        logistic_lr=logistic_lr,
        classifier_l2=classifier_l2,
        regressor_l2=regressor_l2,
    )
    train_eval = _evaluate_predictions(
        rows,
        full_model,
        lcb_z=lcb_z,
        min_prob_lcb=min_prob_lcb,
        min_gain_lcb_veh_h=min_gain_lcb_veh_h,
        min_margin_lcb=min_margin_lcb,
    )

    fold_results = []
    row_groups = _groups(rows)
    for fold_index, fold in enumerate(_folds(data, rows)):
        train_rows = [
            row for group in fold["train_groups"] for row in row_groups[group]
        ]
        test_rows = [
            row for group in fold["test_groups"] for row in row_groups[group]
        ]
        status = "evaluated"
        try:
            fold_model = _train_model(
                train_rows,
                spec=spec,
                ensemble_size=ensemble_size,
                seed=seed + fold_index + 1,
                logistic_steps=logistic_steps,
                logistic_lr=logistic_lr,
                classifier_l2=classifier_l2,
                regressor_l2=regressor_l2,
            )
            evaluation = _evaluate_predictions(
                test_rows,
                fold_model,
                lcb_z=lcb_z,
                min_prob_lcb=min_prob_lcb,
                min_gain_lcb_veh_h=min_gain_lcb_veh_h,
                min_margin_lcb=min_margin_lcb,
            )
        except Exception as exc:
            status = "skipped"
            evaluation = {
                "event_groups": len(set(fold["test_groups"])),
                "selected_event_groups": 0,
                "false_positive_event_groups": 0,
                "missed_positive_event_groups": 0,
                "selections": [],
                "skip_reason": f"{type(exc).__name__}: {exc}",
            }
        fold_results.append({
            "held_out_scenario": fold["held_out_scenario"],
            "status": status,
            **evaluation,
        })

    event_groups = len(_groups(rows))
    positive_rows = sum(int(row["target"]) == 1 for row in rows)
    fold_false_positives = sum(
        fold["false_positive_event_groups"] for fold in fold_results
        if fold["status"] == "evaluated"
    )
    fold_missed_positives = sum(
        fold["missed_positive_event_groups"] for fold in fold_results
        if fold["status"] == "evaluated"
    )
    skipped_folds = sum(fold["status"] == "skipped" for fold in fold_results)
    deployable_gate_pass = bool(
        event_groups >= int(min_event_groups)
        and positive_rows >= int(min_positive_rows)
        and skipped_folds == 0
        and fold_false_positives == 0
        and fold_missed_positives == 0
    )
    result = {
        "format_version": TAIL_SELECTOR_MODEL_FORMAT,
        "input_dataset": str(dataset_path),
        "input_dataset_sha256": _sha256_file(dataset_path),
        "input_dataset_format": data.get("format_version"),
        "teacher_selector_format": LONG_HORIZON_SELECTOR_FORMAT,
        "hyperparameters": {
            "ensemble_size": int(ensemble_size),
            "seed": int(seed),
            "logistic_steps": int(logistic_steps),
            "logistic_lr": float(logistic_lr),
            "classifier_l2": float(classifier_l2),
            "regressor_l2": float(regressor_l2),
            "lcb_z": float(lcb_z),
            "min_prob_lcb": float(min_prob_lcb),
            "min_gain_lcb_veh_h": float(min_gain_lcb_veh_h),
            "min_margin_lcb": float(min_margin_lcb),
            "min_event_groups": int(min_event_groups),
            "min_positive_rows": int(min_positive_rows),
        },
        "summary": {
            "rows": len(rows),
            "positive_rows": positive_rows,
            "negative_rows": len(rows) - positive_rows,
            "event_groups": event_groups,
            "train_selected_event_groups": train_eval["selected_event_groups"],
            "train_false_positive_event_groups": train_eval[
                "false_positive_event_groups"
            ],
            "fold_false_positive_event_groups": int(fold_false_positives),
            "fold_missed_positive_event_groups": int(fold_missed_positives),
            "skipped_folds": int(skipped_folds),
            "deployable_gate_pass": deployable_gate_pass,
        },
        "train_evaluation": train_eval,
        "folds": fold_results,
        "model": full_model,
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
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
    result = train_tail_selector(
        Path(args.dataset),
        Path(args.output),
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
