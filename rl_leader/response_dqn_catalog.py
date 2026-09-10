"""Stable structured actions for the response-aware discrete RL leader."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CATALOG_FORMAT = "response_dqn_action_catalog_v1"
ACTION_FAMILIES = ("anchor", "linear", "quadratic", "cross", "combo", "hybrid")
ACTION_DOMAINS = ("anchor", "urban", "freeway")


@dataclass(frozen=True)
class DiscreteLeaderAction:
    action_id: int
    key: str
    domain: str
    owner: str
    template: str
    magnitude: float
    family: str
    residual: tuple[float, ...]

    def residual_array(self) -> np.ndarray:
        return np.asarray(self.residual, dtype=np.float32)

    def as_dict(self) -> dict:
        return {
            "action_id": int(self.action_id),
            "key": self.key,
            "domain": self.domain,
            "owner": self.owner,
            "template": self.template,
            "magnitude": float(self.magnitude),
            "family": self.family,
            "residual": list(map(float, self.residual)),
        }


class StructuredActionCatalog:
    """State-independent IDs for state-dependent anchor-relative operators."""

    def __init__(
        self,
        action_names: Sequence[str],
        actions: Sequence[DiscreteLeaderAction],
    ) -> None:
        self.action_names = tuple(map(str, action_names))
        self.actions = tuple(actions)
        self._validate()
        self._features = self._build_feature_matrix()

    @property
    def size(self) -> int:
        return len(self.actions)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def feature_matrix(self) -> np.ndarray:
        return self._features.copy()

    @property
    def fingerprint(self) -> str:
        payload = {
            "format_version": CATALOG_FORMAT,
            "action_names": self.action_names,
            "actions": [action.as_dict() for action in self.actions],
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()

    def action(self, action_id: int) -> DiscreteLeaderAction:
        action_id = int(action_id)
        if action_id < 0 or action_id >= self.size:
            raise IndexError(f"action_id out of range: {action_id}")
        return self.actions[action_id]

    def residual(self, action_id: int) -> np.ndarray:
        return self.action(action_id).residual_array()

    def as_manifest(self) -> dict:
        return {
            "format_version": CATALOG_FORMAT,
            "fingerprint": self.fingerprint,
            "action_names": list(self.action_names),
            "feature_dimension": int(self._features.shape[1]),
            "actions": [action.as_dict() for action in self.actions],
        }

    @classmethod
    def from_manifest(cls, manifest: dict) -> "StructuredActionCatalog":
        if manifest.get("format_version") != CATALOG_FORMAT:
            raise ValueError("action catalog manifest has the wrong format")
        actions = [DiscreteLeaderAction(
            action_id=int(row["action_id"]),
            key=str(row["key"]),
            domain=str(row["domain"]),
            owner=str(row["owner"]),
            template=str(row["template"]),
            magnitude=float(row["magnitude"]),
            family=str(row["family"]),
            residual=tuple(map(float, row["residual"])),
        ) for row in manifest["actions"]]
        catalog = cls(manifest["action_names"], actions)
        if catalog.fingerprint != manifest.get("fingerprint"):
            raise ValueError("action catalog manifest fingerprint mismatch")
        return catalog

    def _validate(self) -> None:
        if not self.actions:
            raise ValueError("action catalog must not be empty")
        if [action.action_id for action in self.actions] != list(range(len(self.actions))):
            raise ValueError("action IDs must be contiguous and ordered from zero")
        if len({action.key for action in self.actions}) != len(self.actions):
            raise ValueError("action keys must be unique")
        anchor = self.actions[0]
        if anchor.family != "anchor" or anchor.key != "anchor" or anchor.template != "identity":
            raise ValueError("action 0 must be the P-Stack identity action")
        for action in self.actions:
            if action.family not in ACTION_FAMILIES:
                raise ValueError(f"unknown action family: {action.family}")
            if action.domain not in ACTION_DOMAINS:
                raise ValueError(f"unknown action domain: {action.domain}")
            residual = action.residual_array()
            if residual.shape != (len(self.action_names),):
                raise ValueError(f"action {action.action_id} residual has the wrong shape")
            if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1.0:
                raise ValueError(f"action {action.action_id} residual is outside [-1, 1]")
        if np.any(self.actions[0].residual_array() != 0.0):
            raise ValueError("anchor residual must be exactly zero")

    def _build_feature_matrix(self) -> np.ndarray:
        features = []
        family_index = {name: index for index, name in enumerate(ACTION_FAMILIES)}
        domain_index = {name: index for index, name in enumerate(ACTION_DOMAINS)}
        for action in self.actions:
            family = np.zeros(len(ACTION_FAMILIES), dtype=np.float32)
            family[family_index[action.family]] = 1.0
            domain = np.zeros(len(ACTION_DOMAINS), dtype=np.float32)
            domain[domain_index[action.domain]] = 1.0
            features.append(np.concatenate((
                action.residual_array(),
                family,
                domain,
                np.asarray([action.magnitude], dtype=np.float32),
            )))
        return np.stack(features).astype(np.float32)


def _owner_blocks(action_names: Sequence[str]) -> list[tuple[str, str, tuple[str, ...]]]:
    names = tuple(map(str, action_names))
    available = set(names)
    owners: list[tuple[str, str, tuple[str, ...]]] = []
    for domain, first_term, second_term in (
        ("urban", "g_green", "g_offset"),
        ("freeway", "g_meter", "g_vsl"),
    ):
        prefix = f"{domain}."
        suffix = f".{first_term}"
        domain_owners = [
            name[len(prefix):-len(suffix)]
            for name in names
            if name.startswith(prefix) and name.endswith(suffix)
        ]
        for owner in domain_owners:
            block = tuple(
                f"{domain}.{owner}.{term}"
                for term in (first_term, second_term, "l11", "l21", "l22")
            )
            missing = [name for name in block if name not in available]
            if missing:
                raise ValueError(f"owner block is incomplete for {domain}:{owner}: {missing}")
            owners.append((domain, owner, block))
    if not owners:
        raise ValueError("action schema contains no supported owner blocks")
    return owners


def _templates(families: set[str]) -> list[tuple[str, str, tuple[float, ...]]]:
    templates: list[tuple[str, str, tuple[float, ...]]] = []
    if "linear" in families:
        templates.extend((
            ("linear_first_negative", "linear", (-1, 0, 0, 0, 0)),
            ("linear_first_positive", "linear", (1, 0, 0, 0, 0)),
            ("linear_second_negative", "linear", (0, -1, 0, 0, 0)),
            ("linear_second_positive", "linear", (0, 1, 0, 0, 0)),
            ("linear_corner_nn", "linear", (-1, -1, 0, 0, 0)),
            ("linear_corner_np", "linear", (-1, 1, 0, 0, 0)),
            ("linear_corner_pn", "linear", (1, -1, 0, 0, 0)),
            ("linear_corner_pp", "linear", (1, 1, 0, 0, 0)),
        ))
    if "quadratic" in families:
        templates.extend((
            ("quadratic_first_decrease", "quadratic", (0, 0, -1, 0, 0)),
            ("quadratic_first_increase", "quadratic", (0, 0, 1, 0, 0)),
            ("quadratic_second_decrease", "quadratic", (0, 0, 0, 0, -1)),
            ("quadratic_second_increase", "quadratic", (0, 0, 0, 0, 1)),
        ))
    if "cross" in families:
        # Positive diagonals make the signed l21 perturbation a valid PSD
        # cross-potential even when the native anchor has zero curvature.
        templates.extend((
            ("cross_negative", "cross", (0, 0, 1, -1, 1)),
            ("cross_positive", "cross", (0, 0, 1, 1, 1)),
        ))
    if "combo" in families:
        # Nonlinear prices need a directional gradient and curvature together:
        # pure curvature around the anchor reference often leaves the follower
        # response unchanged.
        templates.extend((
            (
                "combo_first_pos_quad_first_inc",
                "combo",
                (1, 0, 1, 0, 0),
            ),
            (
                "combo_first_pos_quad_first_dec",
                "combo",
                (1, 0, -1, 0, 0),
            ),
            (
                "combo_second_pos_quad_second_inc",
                "combo",
                (0, 1, 0, 0, 1),
            ),
            (
                "combo_second_pos_quad_second_dec",
                "combo",
                (0, 1, 0, 0, -1),
            ),
            (
                "combo_corner_pp_quad_both_inc",
                "combo",
                (1, 1, 1, 0, 1),
            ),
            (
                "combo_corner_pp_cross_pos",
                "combo",
                (1, 1, 1, 1, 1),
            ),
            (
                "combo_corner_pn_cross_neg",
                "combo",
                (1, -1, 1, -1, 1),
            ),
            (
                "combo_corner_np_cross_neg",
                "combo",
                (-1, 1, 1, -1, 1),
            ),
        ))
    if "hybrid" in families:
        # Keep most of the proven linear direction and add small curvature.
        # These actions test whether nonlinear price terms help beyond the
        # linear push instead of replacing it.
        templates.extend((
            (
                "hybrid_first_pos_quad_first_inc",
                "hybrid",
                (1, 0, 0.25, 0, 0),
            ),
            (
                "hybrid_corner_pp_quad_first_inc",
                "hybrid",
                (1, 1, 0.25, 0, 0),
            ),
            (
                "hybrid_corner_pp_quad_second_inc",
                "hybrid",
                (1, 1, 0, 0, 0.25),
            ),
            (
                "hybrid_corner_pp_quad_both_inc",
                "hybrid",
                (1, 1, 0.25, 0, 0.25),
            ),
            (
                "hybrid_corner_pp_cross_pos",
                "hybrid",
                (1, 1, 0.25, 0.25, 0.25),
            ),
            (
                "hybrid_corner_pp_cross_neg",
                "hybrid",
                (1, 1, 0.25, -0.25, 0.25),
            ),
            (
                "hybrid_corner_pn_cross_neg",
                "hybrid",
                (1, -1, 0.25, -0.25, 0.25),
            ),
            (
                "hybrid_corner_np_cross_neg",
                "hybrid",
                (-1, 1, 0.25, -0.25, 0.25),
            ),
        ))
    return templates


def _residual_from_sparse(
    action_names: Sequence[str],
    sparse: dict[str, Any],
) -> np.ndarray:
    index = {name: offset for offset, name in enumerate(map(str, action_names))}
    unknown = sorted(set(map(str, sparse)) - set(index))
    if unknown:
        raise ValueError(f"extra action references unknown channel(s): {unknown}")
    residual = np.zeros(len(index), dtype=np.float32)
    for name, value in sparse.items():
        residual[index[str(name)]] = float(value)
    return residual


def _residual_from_extra_record(
    action_names: Sequence[str],
    record: dict[str, Any],
) -> np.ndarray:
    for key in ("residual_nonzero", "sparse_residual", "sparse"):
        value = record.get(key)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a channel:value mapping")
            return _residual_from_sparse(action_names, value)
    for key in ("residual", "continuous_residual"):
        value = record.get(key)
        if value is not None:
            if isinstance(value, dict):
                return _residual_from_sparse(action_names, value)
            residual = np.asarray(value, dtype=np.float32).reshape(-1)
            if residual.shape != (len(action_names),):
                raise ValueError(
                    f"{key} has dimension {residual.size}, expected {len(action_names)}"
                )
            return residual
    raise ValueError("extra action record must contain residual or residual_nonzero")


def _normalize_extra_family(value: Any) -> str:
    family = str(value or "hybrid")
    if family in ACTION_FAMILIES and family != "anchor":
        return family
    return "hybrid"


def _infer_extra_domain(
    action_names: Sequence[str],
    residual: np.ndarray,
    requested: Any = None,
) -> str:
    requested_domain = str(requested or "")
    if requested_domain in {"urban", "freeway"}:
        return requested_domain
    active = [
        str(name)
        for name, value in zip(action_names, residual)
        if abs(float(value)) > 1.0e-9
    ]
    has_freeway = any(name.startswith("freeway.") for name in active)
    has_urban = any(name.startswith("urban.") for name in active)
    if has_freeway:
        return "freeway"
    if has_urban:
        return "urban"
    # Budget-only residuals affect the network-level allocation around the
    # freeway/urban boundary, so keep them in the broader freeway bucket.
    return "freeway"


def _extra_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if not isinstance(payload, dict):
        raise ValueError("extra action file must contain an object or list")
    if payload.get("format_version") == "cached_tail_residual_sampler_v1":
        records = []
        horizon = str(payload.get("horizon_steps", ""))
        source_records = payload.get("h12_records") or payload.get("h1_records") or []
        if not isinstance(source_records, list):
            raise ValueError("sampler artifact records must be a list")
        for record in source_records:
            record = dict(record)
            label = str(record.get("label") or record.get("candidate_id") or "residual")
            candidate_id = str(record.get("candidate_id") or label)
            horizon_label = {}
            labels = record.get("horizon_labels")
            if isinstance(labels, dict):
                horizon_label = dict(labels.get(horizon) or {})
            metadata = dict(record.get("metadata") or {})
            metadata.update({
                "candidate_id": candidate_id,
                "control_step": payload.get("control_step"),
                "horizon_steps": payload.get("horizon_steps"),
                "horizon_ttt_gain": horizon_label.get("ttt_gain"),
                "horizon_positive": horizon_label.get("positive"),
            })
            records.append({
                "key": f"artifact:{candidate_id}",
                "owner": str(record.get("generator") or "artifact"),
                "template": label,
                "family": record.get("family", "hybrid"),
                "domain": record.get("domain"),
                "magnitude": None,
                "metadata": metadata,
                "residual": record.get("continuous_residual", record.get("residual")),
            })
        return records
    if payload.get("format_version") == "residual_neighborhood_search_v1":
        records: list[dict[str, Any]] = []
        for candidate in payload.get("candidates", []):
            metadata = dict(candidate.get("metadata") or {})
            label = str(candidate.get("label") or candidate.get("candidate_id") or "")
            for item in candidate.get("schedule", []):
                item = dict(item)
                item_label = str(item.get("label") or label or "residual")
                records.append({
                    "key": f"artifact:{item_label}",
                    "owner": str(item.get("owner") or metadata.get("owner") or "artifact"),
                    "template": item_label,
                    "family": metadata.get("family", "hybrid"),
                    "domain": item.get("domain") or metadata.get("domain"),
                    "magnitude": metadata.get("scale"),
                    "residual": item.get("residual"),
                })
        return records
    for key in ("actions", "extra_actions", "residual_actions"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"{key} must be a list")
            return [dict(row) for row in value]
    raise ValueError("extra action file has no actions, extra_actions, or residual_actions")


def load_extra_action_specs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load manual or artifact-derived residual action records."""
    specs: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        for record in _extra_records_from_payload(payload):
            row = dict(record)
            row.setdefault("source_path", str(source))
            specs.append(row)
    return specs


def build_structured_action_catalog(
    action_names: Sequence[str],
    *,
    magnitudes: Iterable[float] = (0.25, 0.5),
    families: Iterable[str] = ("linear", "quadratic", "cross"),
    domains: Iterable[str] = ("urban", "freeway"),
    owners: Iterable[str] | None = None,
    extra_actions: Iterable[dict[str, Any]] | None = None,
) -> StructuredActionCatalog:
    """Build a deterministic catalog of anchor-relative owner operators."""
    names = tuple(map(str, action_names))
    magnitude_values = tuple(float(value) for value in magnitudes)
    if not magnitude_values or any(value <= 0.0 or value > 1.0 for value in magnitude_values):
        raise ValueError("magnitudes must be non-empty and in (0, 1.0]")
    if len(set(magnitude_values)) != len(magnitude_values):
        raise ValueError("magnitudes must be unique")
    family_set = set(map(str, families))
    unsupported = family_set - set(ACTION_FAMILIES[1:])
    if unsupported:
        raise ValueError(f"unsupported action families: {sorted(unsupported)}")
    domain_set = set(map(str, domains))
    if not domain_set or domain_set - {"urban", "freeway"}:
        raise ValueError(f"unsupported action domains: {sorted(domain_set)}")
    owner_set = None if owners is None else set(map(str, owners))

    actions = [DiscreteLeaderAction(
        action_id=0,
        key="anchor",
        domain="anchor",
        owner="P-Stack",
        template="identity",
        magnitude=0.0,
        family="anchor",
        residual=tuple(0.0 for _ in names),
    )]
    index = {name: i for i, name in enumerate(names)}
    templates = _templates(family_set)
    for domain, owner, block in _owner_blocks(names):
        if domain not in domain_set or (owner_set is not None and owner not in owner_set):
            continue
        block_indices = [index[name] for name in block]
        for magnitude in magnitude_values:
            for template, family, direction in templates:
                residual = np.zeros(len(names), dtype=np.float32)
                unit_direction = np.asarray(direction, dtype=np.float32)
                unit_direction /= np.linalg.norm(unit_direction)
                residual[block_indices] = unit_direction * magnitude
                key = f"{domain}:{owner}:{template}:m{magnitude:g}"
                actions.append(DiscreteLeaderAction(
                    action_id=len(actions),
                    key=key,
                    domain=domain,
                    owner=owner,
                    template=template,
                    magnitude=magnitude,
                    family=family,
                    residual=tuple(map(float, residual)),
                ))
    seen_keys = {action.key for action in actions}
    seen_residuals_by_key = {
        action.key: np.asarray(action.residual, dtype=np.float32)
        for action in actions
    }
    for index_value, record in enumerate(extra_actions or ()):
        row = dict(record)
        residual = _residual_from_extra_record(names, row)
        key = str(
            row.get("key")
            or row.get("label")
            or row.get("candidate_id")
            or f"extra_action_{index_value}"
        )
        if not key:
            raise ValueError("extra action key must not be empty")
        if key in seen_keys:
            previous = seen_residuals_by_key.get(key)
            if previous is not None and np.allclose(previous, residual, atol=1.0e-7):
                continue
            raise ValueError(f"extra action key duplicates an existing action: {key}")
        magnitude_value = row.get("magnitude")
        magnitude = (
            float(magnitude_value)
            if magnitude_value is not None
            else float(np.linalg.norm(residual))
        )
        actions.append(DiscreteLeaderAction(
            action_id=len(actions),
            key=key,
            domain=_infer_extra_domain(names, residual, row.get("domain")),
            owner=str(row.get("owner") or "artifact"),
            template=str(row.get("template") or row.get("label") or key),
            magnitude=magnitude,
            family=_normalize_extra_family(row.get("family")),
            residual=tuple(map(float, residual)),
        ))
        seen_keys.add(key)
        seen_residuals_by_key[key] = np.asarray(residual, dtype=np.float32)
    if len(actions) == 1:
        raise ValueError("catalog filters removed every non-anchor action")
    return StructuredActionCatalog(names, actions)
