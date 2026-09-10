"""Deterministic equal-budget candidates for same-state oracle ablations."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


ABLATION_CONTRACT = "linear_price_equal_l2_ablation_v1"
OWNER_BLOCK_ABLATION_CONTRACT = "sparse_owner_block_exhaustive_h12_v2"
ABLATION_GENERATORS = (
    "pcent_sign",
    "structured",
    "pcent_jacobian",
    "orthogonal_random",
)
LINEAR_PRICE_GROUPS = ("green", "offset", "meter", "vsl")
URBAN_OWNERS = ("A", "B", "C", "D", "F")
RAMP_OWNERS = ("R_D_W", "R_F_W", "R_D_E", "R_F_E")
KNOWN_POSITIVE_CANARY_COORDINATE = 0.5
KNOWN_POSITIVE_CANARY_RADIUS = math.sqrt(2.0) * KNOWN_POSITIVE_CANARY_COORDINATE


@dataclass(frozen=True)
class AblationConfig:
    radius: float = 0.5
    h1_candidates_per_generator: int = 8
    h12_slots_per_generator: int = 1
    master_seed: int = 20260828
    allowed_anchor_branches: tuple[str, ...] = ("coarse", "refined")

    def validate(self) -> None:
        if not 0.0 < float(self.radius) <= 1.0:
            raise ValueError("ablation radius must be in (0, 1]")
        if int(self.h1_candidates_per_generator) != 8:
            raise ValueError("linear-price v1 requires exactly 8 H1 candidates per generator")
        if int(self.h12_slots_per_generator) != 1:
            raise ValueError("linear-price v1 requires exactly one H12 slot per generator")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    generator: str
    phase: str
    label: str
    residual: tuple[float, ...]
    uses_pcent_target: bool
    teacher_signal_missing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def residual_array(self) -> np.ndarray:
        return np.asarray(self.residual, dtype=np.float32)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def derive_seed(master_seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(master_seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def residual_sha256(residual) -> str:
    values = np.asarray(residual, dtype="<f4").reshape(-1).copy()
    values[values == 0.0] = 0.0
    return hashlib.sha256(values.tobytes()).hexdigest()


def linear_price_group_indices(action_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    groups = {
        "green": [i for i, name in enumerate(action_names) if name.endswith(".g_green")],
        "offset": [i for i, name in enumerate(action_names) if name.endswith(".g_offset")],
        "meter": [i for i, name in enumerate(action_names) if name.endswith(".g_meter")],
        "vsl": [i for i, name in enumerate(action_names) if name.endswith(".g_vsl")],
    }
    expected = {"green": 5, "offset": 5, "meter": 4, "vsl": 16}
    actual = {name: len(indices) for name, indices in groups.items()}
    if actual != expected:
        raise ValueError(
            f"linear-price v1 expects the canonical 30-D layout: {actual} != {expected}"
        )
    flattened = [index for indices in groups.values() for index in indices]
    if len(set(flattened)) != 30:
        raise ValueError("linear-price groups overlap")
    return {
        name: np.asarray(indices, dtype=int) for name, indices in groups.items()
    }


def linear_price_bundle_basis(action_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    groups = linear_price_group_indices(action_names)
    basis = {}
    for name, indices in groups.items():
        direction = np.zeros(len(action_names), dtype=np.float64)
        direction[indices] = 1.0 / math.sqrt(float(indices.size))
        basis[name] = direction
    return basis


def _scaled_direction(direction, radius: float) -> np.ndarray:
    values = np.asarray(direction, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        raise ValueError("candidate direction has zero norm")
    result = values * (float(radius) / norm)
    if np.max(np.abs(result)) > 1.0 + 1.0e-12:
        raise ValueError("equal-L2 candidate exceeds the normalized action box")
    return result.astype(np.float32)


def _candidate_spec(
    *,
    generator: str,
    phase: str,
    label: str,
    residual: np.ndarray,
    uses_pcent_target: bool,
    teacher_signal_missing: bool = False,
    metadata: dict[str, Any] | None = None,
) -> CandidateSpec:
    values = np.asarray(residual, dtype=np.float32).copy()
    values[values == 0.0] = 0.0
    digest = residual_sha256(values)
    return CandidateSpec(
        candidate_id=f"{generator}:{label}:{digest[:12]}",
        generator=generator,
        phase=phase,
        label=label,
        residual=tuple(map(float, values)),
        uses_pcent_target=bool(uses_pcent_target),
        teacher_signal_missing=bool(teacher_signal_missing),
        metadata=dict(metadata or {}),
    )


def generate_pcent_sign_candidates(
    action_names: tuple[str, ...],
    directed_groups: dict[str, np.ndarray],
    cfg: AblationConfig,
) -> list[CandidateSpec]:
    """Use P-CENT only to orient four physical linear-price bundles."""
    cfg.validate()
    basis = linear_price_bundle_basis(action_names)
    candidates = []
    for group in LINEAR_PRICE_GROUPS:
        supplied = directed_groups.get(group)
        missing = supplied is None or not np.any(np.asarray(supplied) != 0.0)
        direction = basis[group] if missing else np.asarray(supplied, dtype=float)
        toward = _scaled_direction(direction, cfg.radius)
        for sign, suffix in ((1.0, "toward"), (-1.0, "away")):
            candidates.append(_candidate_spec(
                generator="pcent_sign",
                phase="candidate",
                label=f"{group}:{suffix}",
                residual=sign * toward,
                uses_pcent_target=not missing,
                teacher_signal_missing=missing,
                metadata={"bundle": group},
            ))
    return candidates


def generate_structured_candidates(
    action_names: tuple[str, ...], cfg: AblationConfig
) -> list[CandidateSpec]:
    """Target-independent signed bundle basis; no teacher argument by design."""
    cfg.validate()
    basis = linear_price_bundle_basis(action_names)
    candidates = []
    for group in LINEAR_PRICE_GROUPS:
        direction = _scaled_direction(basis[group], cfg.radius)
        for sign, suffix in ((1.0, "positive"), (-1.0, "negative")):
            candidates.append(_candidate_spec(
                generator="structured",
                phase="candidate",
                label=f"{group}:{suffix}",
                residual=sign * direction,
                uses_pcent_target=False,
                metadata={"bundle": group},
            ))
    return candidates


def generate_known_positive_canary(
    action_names: tuple[str, ...],
) -> CandidateSpec:
    """Reproduce the historical step-9 C green/offset positive H12 label."""
    residual = np.zeros(len(action_names), dtype=np.float32)
    indices = {name: index for index, name in enumerate(action_names)}
    required = ("urban.C.g_green", "urban.C.g_offset")
    missing = [name for name in required if name not in indices]
    if missing:
        raise ValueError(f"known-positive canary channels are missing: {missing}")
    residual[indices[required[0]]] = -KNOWN_POSITIVE_CANARY_COORDINATE
    residual[indices[required[1]]] = KNOWN_POSITIVE_CANARY_COORDINATE
    return _candidate_spec(
        generator="structured",
        phase="regression_canary",
        label="owner:C:green-negative_offset-positive",
        residual=residual,
        uses_pcent_target=False,
        metadata={
            "owner": "C",
            "channels": ["green", "offset"],
            "historical_scenario": "sweet_170_w60",
            "historical_policy_step": 9,
        },
    )


def _owner_block_channels(
    action_names: tuple[str, ...],
    *,
    include_urban: bool,
    include_freeway: bool,
) -> list[tuple[str, str, str, str]]:
    available = set(action_names)
    blocks: list[tuple[str, str, str, str]] = []
    if include_urban:
        blocks.extend((
            "urban",
            owner,
            f"urban.{owner}.g_green",
            f"urban.{owner}.g_offset",
        ) for owner in URBAN_OWNERS)
    if include_freeway:
        blocks.extend((
            "freeway",
            owner,
            f"freeway.{owner}.g_meter",
            f"freeway.{owner}.g_vsl",
        ) for owner in RAMP_OWNERS)
    missing = sorted({channel for block in blocks for channel in block[2:] if channel not in available})
    if missing:
        raise ValueError(f"owner-block channels are missing: {missing}")
    return blocks


def generate_owner_block_structured_candidates(
    action_names: tuple[str, ...],
    *,
    include_urban: bool = True,
    include_freeway: bool = True,
    magnitudes: tuple[float, ...] = (0.25, 0.5),
) -> list[CandidateSpec]:
    """Enumerate target-independent signed axes and corners of owner blocks."""
    indices = {name: index for index, name in enumerate(action_names)}
    candidates = []
    signs = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    for domain, owner, first, second in _owner_block_channels(
        action_names,
        include_urban=include_urban,
        include_freeway=include_freeway,
    ):
        channel_labels = (
            ("green", "offset") if domain == "urban" else ("meter", "vsl")
        )
        templates = [
            (first_sign, second_sign, "corner")
            for first_sign, second_sign in signs
        ] + [
            (-1.0, 0.0, "axis"),
            (1.0, 0.0, "axis"),
            (0.0, -1.0, "axis"),
            (0.0, 1.0, "axis"),
        ]
        for magnitude in magnitudes:
            if not 0.0 < float(magnitude) <= 0.5:
                raise ValueError("owner-block magnitudes must be in (0, 0.5]")
            for first_sign, second_sign, template in templates:
                residual = np.zeros(len(action_names), dtype=np.float32)
                residual[indices[first]] = first_sign * float(magnitude)
                residual[indices[second]] = second_sign * float(magnitude)
                active_labels = []
                if first_sign != 0.0:
                    active_labels.append(
                        f"{channel_labels[0]}-"
                        f"{'positive' if first_sign > 0 else 'negative'}"
                    )
                if second_sign != 0.0:
                    active_labels.append(
                        f"{channel_labels[1]}-"
                        f"{'positive' if second_sign > 0 else 'negative'}"
                    )
                sign_label = "_".join(active_labels)
                candidates.append(_candidate_spec(
                    generator="structured",
                    phase="owner_block_v2",
                    label=(
                        f"{domain}:{owner}:{template}:{sign_label}:m{magnitude:g}"
                    ),
                    residual=residual,
                    uses_pcent_target=False,
                    metadata={
                        "domain": domain,
                        "owner": owner,
                        "channels": list(channel_labels),
                        "signs": [first_sign, second_sign],
                        "template": template,
                        "magnitude": float(magnitude),
                    },
                ))
    return candidates


def generate_owner_block_pcent_candidates(
    action_names: tuple[str, ...],
    directed_groups: dict[str, np.ndarray],
    *,
    include_urban: bool = True,
    include_freeway: bool = True,
    magnitudes: tuple[float, ...] = (0.25, 0.5),
) -> list[CandidateSpec]:
    """Orient sparse owner blocks with P-CENT while preserving separate provenance."""
    indices = {name: index for index, name in enumerate(action_names)}
    candidates = []
    for domain, owner, first, second in _owner_block_channels(
        action_names,
        include_urban=include_urban,
        include_freeway=include_freeway,
    ):
        group_names = (
            ("green", "offset") if domain == "urban" else ("meter", "vsl")
        )
        raw = []
        missing = False
        for group, channel in zip(group_names, (first, second)):
            direction = directed_groups.get(group)
            if direction is None:
                value = 0.0
            else:
                values = np.asarray(direction, dtype=float)
                value = float(values[indices[channel]])
            missing = missing or abs(value) <= 1.0e-12
            raw.append(value)
        teacher_guided_channels = [
            group for group, value in zip(group_names, raw)
            if abs(value) > 1.0e-12
        ]
        fallback_channels = [
            group for group, value in zip(group_names, raw)
            if abs(value) <= 1.0e-12
        ]
        uses_pcent_target = bool(teacher_guided_channels)
        signs = [1.0 if value >= 0.0 else -1.0 for value in raw]
        for magnitude in magnitudes:
            if not 0.0 < float(magnitude) <= 0.5:
                raise ValueError("owner-block magnitudes must be in (0, 0.5]")
            toward = np.zeros(len(action_names), dtype=np.float32)
            toward[indices[first]] = signs[0] * float(magnitude)
            toward[indices[second]] = signs[1] * float(magnitude)
            for multiplier, suffix in ((1.0, "toward"), (-1.0, "away")):
                candidates.append(_candidate_spec(
                    generator="pcent_sign",
                    phase="owner_block_v2",
                    label=f"{domain}:{owner}:{suffix}:m{magnitude:g}",
                    residual=multiplier * toward,
                    uses_pcent_target=uses_pcent_target,
                    teacher_signal_missing=missing,
                    metadata={
                        "domain": domain,
                        "owner": owner,
                        "channels": list(group_names),
                        "raw_direction": raw,
                        "teacher_guided_channels": teacher_guided_channels,
                        "fallback_channels": fallback_channels,
                        "magnitude": float(magnitude),
                    },
                ))
    return candidates


def select_owner_block_h12_candidates(
    records: list[dict],
    *,
    anchor_response: np.ndarray,
    known_canary_label: str | None = None,
) -> list[dict]:
    """Choose H12 representatives from H3 outcomes without using P-CENT targets."""
    eligible = [row for row in records if bool(row.get("validity_gate_pass", False))]
    if not eligible:
        return []
    anchor = np.asarray(anchor_response, dtype=float)

    def response_distance_from_anchor(row: dict) -> float:
        response = np.asarray(row["response"], dtype=float)
        return float(np.linalg.norm(response - anchor))

    chosen = [
        min(eligible, key=lambda row: (
            float(row["h3_ttt"]), str(row["residual_sha256"]), str(row["candidate_id"])
        )),
        min(eligible, key=lambda row: (
            float(row["h3_terminal_inventory"]),
            str(row["residual_sha256"]),
            str(row["candidate_id"]),
        )),
        max(eligible, key=lambda row: (
            response_distance_from_anchor(row),
            str(row["residual_sha256"]),
            str(row["candidate_id"]),
        )),
    ]
    if known_canary_label is not None:
        canaries = [
            row for row in eligible
            if row.get("candidate_label") == known_canary_label
        ]
        if canaries:
            chosen.append(min(canaries, key=lambda row: str(row["candidate_id"])))
    unique = {}
    for row in chosen:
        unique.setdefault(str(row["candidate_id"]), row)
    return list(unique.values())


def generate_jacobian_secant_candidates(
    action_names: tuple[str, ...], cfg: AblationConfig
) -> list[CandidateSpec]:
    """Four target-independent positive probes for a local secant response model."""
    cfg.validate()
    basis = linear_price_bundle_basis(action_names)
    return [
        _candidate_spec(
            generator="pcent_jacobian",
            phase="secant_probe",
            label=f"{group}:secant",
            residual=_scaled_direction(basis[group], cfg.radius),
            uses_pcent_target=False,
            metadata={"bundle": group},
        )
        for group in LINEAR_PRICE_GROUPS
    ]


def generate_orthogonal_random_candidates(
    action_names: tuple[str, ...],
    cfg: AblationConfig,
    *,
    pre_runtime_sha256: str,
) -> list[CandidateSpec]:
    """Four deterministic orthogonal directions and their negatives."""
    cfg.validate()
    groups = linear_price_group_indices(action_names)
    active = np.asarray([
        index for group in LINEAR_PRICE_GROUPS for index in groups[group]
    ], dtype=int)
    rng = np.random.default_rng(derive_seed(
        cfg.master_seed, pre_runtime_sha256, "orthogonal_random", "basis"
    ))
    matrix = rng.standard_normal((active.size, 4))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    candidates = []
    for column in range(4):
        local = q[:, column]
        first = int(np.flatnonzero(np.abs(local) > 1.0e-12)[0])
        if local[first] < 0.0:
            local = -local
        direction = np.zeros(len(action_names), dtype=np.float64)
        direction[active] = local
        residual = _scaled_direction(direction, cfg.radius)
        for sign, suffix in ((1.0, "positive"), (-1.0, "negative")):
            candidates.append(_candidate_spec(
                generator="orthogonal_random",
                phase="candidate",
                label=f"q{column}:{suffix}",
                residual=sign * residual,
                uses_pcent_target=False,
                metadata={"basis_index": column},
            ))
    return candidates


def synthesize_jacobian_inverse_candidates(
    secant_specs: list[CandidateSpec],
    probe_records: dict[str, dict],
    *,
    anchor_response: np.ndarray,
    target_response: np.ndarray,
    response_scales: np.ndarray,
    cfg: AblationConfig,
) -> tuple[list[CandidateSpec], dict[str, Any]]:
    """Fit a four-bundle local secant model and generate four ridge inverses."""
    if len(secant_specs) != 4:
        raise ValueError("Jacobian v1 requires four secant probes")
    ordered = sorted(secant_specs, key=lambda spec: spec.label)
    missing = [spec.candidate_id for spec in ordered if spec.candidate_id not in probe_records]
    if missing:
        raise ValueError(f"Jacobian secant probe records are missing: {missing}")
    x = np.stack([spec.residual_array() for spec in ordered]).astype(np.float64)
    anchor = np.asarray(anchor_response, dtype=np.float64)
    target = np.asarray(target_response, dtype=np.float64)
    scales = np.asarray(response_scales, dtype=np.float64)
    if anchor.shape != target.shape or scales.shape != anchor.shape:
        raise ValueError("Jacobian response vectors or scales are incompatible")
    y = np.stack([
        (np.asarray(probe_records[spec.candidate_id]["response"], dtype=np.float64)
         - anchor) / scales
        for spec in ordered
    ])
    target_delta = (target - anchor) / scales
    singular_values = np.linalg.svd(y, compute_uv=False)
    tolerance = (
        max(y.shape) * np.finfo(np.float64).eps * singular_values[0]
        if singular_values.size and singular_values[0] > 0.0 else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    nonzero = singular_values[singular_values > tolerance]
    condition = (
        float(nonzero[0] / nonzero[-1]) if nonzero.size > 1
        else (1.0 if nonzero.size == 1 else None)
    )
    spectral_scale = float(
        max(singular_values[0] ** 2, 1.0e-12)
        if singular_values.size else 1.0e-12
    )
    gram = y @ y.T
    rhs = y @ target_delta
    ridge_factors = (1.0e-4, 1.0e-2, 1.0, 100.0)
    candidates = []
    for index, factor in enumerate(ridge_factors):
        regularization = float(factor * spectral_scale)
        weights = np.linalg.solve(
            gram + regularization * np.eye(gram.shape[0]), rhs
        )
        direction = weights @ x
        fallback = float(np.linalg.norm(direction)) <= 1.0e-12
        if fallback:
            direction = x[index]
        residual = _scaled_direction(direction, cfg.radius)
        candidates.append(_candidate_spec(
            generator="pcent_jacobian",
            phase="inverse_candidate",
            label=f"ridge:{factor:g}",
            residual=residual,
            uses_pcent_target=True,
            teacher_signal_missing=False,
            metadata={
                "regularization": regularization,
                "ridge_factor": factor,
                "secant_rank": rank,
                "secant_condition": condition,
                "degenerate_fallback": fallback,
            },
        ))
    diagnostics = {
        "rank": rank,
        "singular_values": singular_values.astype(float).tolist(),
        "condition_number": condition,
        "response_dimension": int(y.shape[1]),
        "secant_count": int(y.shape[0]),
        "ridge_factors": list(ridge_factors),
    }
    return candidates, diagnostics


def validate_candidate_specs(
    specs: list[CandidateSpec],
    action_names: tuple[str, ...],
    cfg: AblationConfig,
    *,
    require_all_generators: bool = True,
    allowed_radii: tuple[float, ...] | None = None,
) -> dict[str, int]:
    cfg.validate()
    groups = linear_price_group_indices(action_names)
    allowed = np.zeros(len(action_names), dtype=bool)
    for indices in groups.values():
        allowed[indices] = True
    ids = set()
    counts = {generator: 0 for generator in ABLATION_GENERATORS}
    for spec in specs:
        if spec.candidate_id in ids:
            raise ValueError(f"duplicate ablation candidate id: {spec.candidate_id}")
        ids.add(spec.candidate_id)
        if spec.generator not in counts:
            raise ValueError(f"unknown ablation generator: {spec.generator}")
        residual = spec.residual_array()
        if residual.shape != (len(action_names),) or not np.all(np.isfinite(residual)):
            raise ValueError(f"candidate {spec.candidate_id} residual is invalid")
        if np.any(residual[~allowed] != 0.0):
            raise ValueError(
                f"candidate {spec.candidate_id} leaves the 30-D linear-price subspace"
            )
        radii = (float(cfg.radius),) if allowed_radii is None else allowed_radii
        if not any(math.isclose(
            float(np.linalg.norm(residual)), float(radius),
            rel_tol=0.0, abs_tol=1.0e-6,
        ) for radius in radii):
            raise ValueError(f"candidate {spec.candidate_id} violates the L2 budget")
        if float(np.max(np.abs(residual))) > 1.0 + 1.0e-7:
            raise ValueError(f"candidate {spec.candidate_id} exceeds the action box")
        counts[spec.generator] += 1
    if require_all_generators and any(
        count != cfg.h1_candidates_per_generator for count in counts.values()
    ):
        raise ValueError(f"ablation candidate budgets are unequal: {counts}")
    return counts


def candidate_manifest_sha256(specs: list[CandidateSpec]) -> str:
    rows = sorted((
        {
            "candidate_id": spec.candidate_id,
            "generator": spec.generator,
            "phase": spec.phase,
            "label": spec.label,
            "residual_sha256": residual_sha256(spec.residual),
            "uses_pcent_target": spec.uses_pcent_target,
            "teacher_signal_missing": spec.teacher_signal_missing,
            "metadata": spec.metadata,
        }
        for spec in specs
    ), key=lambda row: row["candidate_id"])
    return _sha256(rows)


def select_h12_candidates_without_teacher(
    records: list[dict],
    *,
    anchor_response: np.ndarray,
    anchor_follower_memory_sha256: str,
    cfg: AblationConfig,
) -> list[dict]:
    """Select one H12 slot per generator using only executable H1 outcomes."""
    cfg.validate()
    anchor = np.asarray(anchor_response, dtype=float)
    selected = []
    for generator in ABLATION_GENERATORS:
        eligible = []
        for record in records:
            if record.get("generator") != generator:
                continue
            if not bool(record.get("validity_gate_pass", False)):
                continue
            response = np.asarray(record.get("response"), dtype=float)
            same_anchor = (
                np.array_equal(response, anchor)
                and str(record.get("post_follower_sha256"))
                == str(anchor_follower_memory_sha256)
            )
            if same_anchor:
                continue
            eligible.append(record)
        if not eligible:
            continue
        selected.append(min(eligible, key=lambda row: (
            float(row["step_ttt"]),
            float(row["terminal_inventory"]),
            str(row["residual_sha256"]),
            str(row["candidate_id"]),
        )))
    return selected
