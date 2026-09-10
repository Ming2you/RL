"""Support-aware priorities for the next simulator data-construction batch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpansionWeights:
    disagreement: float = 1.0
    decision_boundary: float = 1.0
    low_support: float = 1.0
    novel_response: float = 1.0
    boundary_scale: float = 1.0


def expansion_priority(
    ensemble_delta_q: np.ndarray,
    action_support_counts: np.ndarray,
    valid_action_mask: np.ndarray,
    *,
    novel_response_mask: np.ndarray | None = None,
    weights: ExpansionWeights | None = None,
) -> dict[str, np.ndarray]:
    """Rank only executable structured actions and explain every score."""
    weights = weights or ExpansionWeights()
    values = np.asarray(ensemble_delta_q, dtype=np.float64)
    support = np.asarray(action_support_counts, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[1] != valid.size or support.shape != valid.shape:
        raise ValueError("expansion inputs must share the action dimension")
    mean = values.mean(axis=0)
    disagreement = values.std(axis=0)
    scale = max(float(weights.boundary_scale), 1.0e-9)
    boundary = np.exp(-np.abs(mean) / scale)
    low_support = 1.0 / np.sqrt(support + 1.0)
    novel = (
        np.zeros_like(mean)
        if novel_response_mask is None
        else np.asarray(novel_response_mask, dtype=bool).astype(float)
    )
    if novel.shape != valid.shape:
        raise ValueError("novel response mask shape mismatch")
    score = (
        weights.disagreement * disagreement
        + weights.decision_boundary * boundary
        + weights.low_support * low_support
        + weights.novel_response * novel
    )
    score = np.where(valid, score, -np.inf)
    return {
        "score": score,
        "mean_delta_q": mean,
        "ensemble_disagreement": disagreement,
        "decision_boundary": boundary,
        "low_support": low_support,
        "novel_response": novel,
    }
