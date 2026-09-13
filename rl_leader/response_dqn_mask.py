"""State-dependent executable-response masking for a fixed action catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


RESPONSE_MASK_FORMAT = "response_equivalence_mask_v1"
LEGACY_EQUIVALENCE = "legacy_follower_runtime_v1"
CONTINUATION_EQUIVALENCE = "post_commit_continuation_v1"
FIVE_CELL_CONTINUATION_EQUIVALENCE = "post_commit_continuation_five_cell_v1"
CONTINUATION_EQUIVALENCE_MODES = frozenset({
    CONTINUATION_EQUIVALENCE, FIVE_CELL_CONTINUATION_EQUIVALENCE,
})


@dataclass(frozen=True)
class CandidateResponse:
    action_id: int
    response: tuple[float, ...]
    follower_memory_fingerprint: str
    valid: bool = True
    invalid_reason: str = ""

    def response_array(self) -> np.ndarray:
        return np.asarray(self.response, dtype=np.float32)


@dataclass(frozen=True)
class ResponseMask:
    valid_action_mask: np.ndarray
    representative_of: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    invalid_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        self.valid_action_mask.setflags(write=False)
        self.representative_of.setflags(write=False)

    @property
    def valid_action_ids(self) -> np.ndarray:
        return np.flatnonzero(self.valid_action_mask)

    def as_dict(self) -> dict:
        return {
            "format_version": RESPONSE_MASK_FORMAT,
            "valid_action_mask": self.valid_action_mask.astype(int).tolist(),
            "representative_of": self.representative_of.astype(int).tolist(),
            "groups": [list(map(int, group)) for group in self.groups],
            "invalid_reasons": list(self.invalid_reasons),
        }


def build_response_mask(
    responses: Sequence[CandidateResponse],
    *,
    catalog_size: int,
    response_atol: float = 1.0e-6,
    anchor_action_id: int = 0,
) -> ResponseMask:
    """Keep one representative per physical-control and memory outcome."""
    catalog_size = int(catalog_size)
    anchor_action_id = int(anchor_action_id)
    if catalog_size <= 0 or not 0 <= anchor_action_id < catalog_size:
        raise ValueError("invalid catalog or anchor size")
    if response_atol < 0.0:
        raise ValueError("response_atol must be nonnegative")
    by_id: dict[int, CandidateResponse] = {}
    response_dim: int | None = None
    for item in responses:
        action_id = int(item.action_id)
        if not 0 <= action_id < catalog_size:
            raise ValueError(f"response action_id out of range: {action_id}")
        if action_id in by_id:
            raise ValueError(f"duplicate response action_id: {action_id}")
        values = item.response_array()
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid response vector for action {action_id}")
        response_dim = values.size if response_dim is None else response_dim
        if values.size != response_dim:
            raise ValueError("candidate response dimensions do not match")
        by_id[action_id] = item
    if len(by_id) != catalog_size:
        missing = sorted(set(range(catalog_size)) - set(by_id))
        raise ValueError(f"response mask requires every action ID; missing {missing[:8]}")
    if not by_id[anchor_action_id].valid:
        raise ValueError("P-Stack anchor must always be valid")

    groups: list[list[int]] = []
    invalid_reasons = ["" for _ in range(catalog_size)]
    for action_id in range(catalog_size):
        item = by_id[action_id]
        if not item.valid:
            invalid_reasons[action_id] = item.invalid_reason or "physical_invalid"
            continue
        values = item.response_array()
        matching_group = None
        for group in groups:
            representative = by_id[group[0]]
            if (
                item.follower_memory_fingerprint
                == representative.follower_memory_fingerprint
                and np.allclose(
                    values,
                    representative.response_array(),
                    atol=response_atol,
                    rtol=0.0,
                )
            ):
                matching_group = group
                break
        if matching_group is None:
            groups.append([action_id])
        else:
            matching_group.append(action_id)

    representatives = np.full(catalog_size, -1, dtype=np.int64)
    mask = np.zeros(catalog_size, dtype=bool)
    normalized_groups = []
    for group in groups:
        group = sorted(group)
        representative = (
            anchor_action_id if anchor_action_id in group else group[0]
        )
        mask[representative] = True
        for action_id in group:
            representatives[action_id] = representative
            if action_id != representative:
                invalid_reasons[action_id] = f"response_duplicate_of:{representative}"
        normalized_groups.append(tuple(group))
    if not mask[anchor_action_id]:
        raise RuntimeError("response deduplication removed the P-Stack anchor")
    return ResponseMask(
        valid_action_mask=mask,
        representative_of=representatives,
        groups=tuple(normalized_groups),
        invalid_reasons=tuple(invalid_reasons),
    )


def response_feature_matrix(
    responses: Sequence[CandidateResponse],
    *,
    catalog_size: int,
    anchor_action_id: int = 0,
) -> np.ndarray:
    """Return raw response and anchor-relative response for every action ID."""
    ordered: list[CandidateResponse | None] = [None] * int(catalog_size)
    for item in responses:
        if not 0 <= int(item.action_id) < len(ordered):
            raise ValueError(f"response action_id out of range: {item.action_id}")
        if ordered[int(item.action_id)] is not None:
            raise ValueError(f"duplicate response action_id: {item.action_id}")
        ordered[int(item.action_id)] = item
    if any(item is None for item in ordered):
        raise ValueError("response features require every action ID")
    raw = np.stack([item.response_array() for item in ordered if item is not None])
    anchor = raw[int(anchor_action_id)]
    return np.concatenate((raw, raw - anchor[None, :]), axis=1).astype(np.float32)
